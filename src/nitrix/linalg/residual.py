# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Residualisation via linear least squares.

The canonical fMRI-confound use case: regress out a set of
nuisance regressors ``X`` from a data tensor ``Y``, returning the
residual (signal of interest) or the projection (the part
explained by confounds).

Two solver paths, both differentiable:

- ``method="cholesky"`` (**default**): normal-equations OLS via a
  **cuSOLVER-free** rolled Cholesky (:func:`spd_chol`) + cuBLAS triangular
  solves, so it runs on the cuSOLVER-affected GPU stacks (where
  ``jnp.linalg.cholesky`` / ``svd`` fail to create a handle).  Tall-and-skinny
  optimised -- for ``X`` of shape :math:`(\\mathrm{obs}, k)` with
  :math:`\\mathrm{obs} \\gg k` (typical fMRI: 400 TRs :math:`\\times`
  24 confounds), the heavy step is :math:`X^{\\top} X`
  (:math:`O(\\mathrm{obs} \\cdot k^2)`, bandwidth-limited) followed by a
  :math:`k \\times k` Cholesky and two triangular solves (:math:`O(k^3)`).
  :math:`X^{\\top} X` is numerically squared in condition number, so this
  path wants a well-conditioned ``X`` or an ``l2 > 0`` ridge; a rank-deficient
  Gram is regularised by the modified-Cholesky pivot floor (a finite result,
  not ``NaN`` -- but prefer ``l2 > 0`` for a controlled treatment).
- ``method="svd"``: SVD-based lstsq.  Stable even for rank-deficient ``X``;
  slower (~2× on tall systems) but works without an L2 ridge on collinear
  columns.  **cuSOLVER-backed** (``gesvd``), so on the broken stacks use the
  ``"cholesky"`` default (or run on a working backend).

QR was considered as a third path but is currently broken on
the test runner's GPU (cuSolver ABI mismatch); ``method="qr"``
is not exposed.  We can add it back when the cuSolver issue
is fixed -- the wrapper machinery doesn't change.

Per-observation weights (heteroscedastic regression) are
supported via ``weights``: a length-``obs`` vector defining the
diagonal of the weight matrix.  Implementation note: we
*pre-scale* rows by ``sqrt(weights)`` rather than form
:math:`X^{\\top} \\operatorname{diag}(w) X` explicitly -- preserves the
QR / SVD structure and is numerically equivalent.

Differentiability: both solver paths are reverse-mode
differentiable (JAX ships VJPs for ``qr``, ``lstsq``, and
triangular solve).  Gradients have been verified by
finite-difference parity in the test suite.

What the legacy ``hypercoil.functional.resid.residualise`` had
that we drop:

- ``proj_atol`` / ``proj_rtol``: the "if proj is numerically
  close to Y, return Y" trick.  Composable post-hoc by the user
  if needed; not a primitive concern.

What the legacy did NOT have that we add:

- ``weights`` for heteroscedastic / GLS-like regression.
- ``method`` selection for the speed / stability trade-off.
"""

from __future__ import annotations

from typing import Any, Callable, Literal, Optional, cast

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsla
from jaxtyping import Array, Float, Num

from ._smalllinalg import spd_chol

__all__ = ['residualise', 'partial_residualise']


ReturnMode = Literal['residual', 'projection']
Method = Literal['cholesky', 'svd']
Shrinkage = Literal['none', 'james-stein']

# Floor on the projection energy in the James-Stein denominator (a channel with
# a numerically-zero projection has nothing to shrink -> factor clamps to 0).
_JS_EPS = 1e-30


def _james_stein_shrink(
    resid: Float[Array, 'n m'],
    proj: Float[Array, 'n m'],
    k: int,
) -> Float[Array, 'n m']:
    """Positive-part James-Stein shrinkage of the per-channel nuisance projection.

    Scales each target channel's projection by the factor
    :math:`c = \\left(1 - (k - 2)\\,\\sigma^2 / \\lVert \\mathrm{proj} \\rVert^2
    \\right)_+` -- the soft / shrunk confound removal.  Here :math:`\\sigma^2`
    is the per-channel residual-variance estimate (from ``resid``, with
    :math:`n - k` degrees of freedom) and :math:`k` is the number of nuisance
    regressors.  When the confounds explain a channel only weakly (small
    :math:`\\lVert \\mathrm{proj} \\rVert` relative to the noise) the removal is
    shrunk toward zero, so a high-dimensional / collinear confound set does not
    strip real signal variance it merely captured by chance.

    This is the positive-part James-Stein *heuristic*: the strict dominance
    theorem (lower total MSE for :math:`k \\geq 3`) holds for the canonical
    known-variance Gaussian-mean problem; here it is a sound plug-in analogue
    (estimated per-channel :math:`\\sigma^2`, applied to the projection), not
    that exact result.  For :math:`k \\leq 2` the factor is :math:`\\geq 1` and
    clamps to :math:`1` (no shrinkage); the per-channel :math:`c` is likewise
    clamped to :math:`[0, 1]`.

    Parameters
    ----------
    resid
        Per-channel regression residual, shape ``(n, m)`` (observations by
        target channels), used to estimate the per-channel noise variance
        :math:`\\sigma^2`.
    proj
        Per-channel nuisance projection to shrink, shape ``(n, m)``.
    k
        Number of nuisance regressors, entering both the degrees-of-freedom
        correction and the shrinkage numerator.

    Returns
    -------
    Float[Array, 'n m']
        The projection ``proj`` scaled per target channel by the clamped
        James-Stein factor :math:`c`, same shape ``(n, m)``.
    """
    n = resid.shape[-2]
    dof = max(n - k, 1)
    sigma2 = jnp.sum(resid * resid, axis=-2) / dof  # (m,)
    proj_ss = jnp.sum(proj * proj, axis=-2)  # (m,)
    c = 1.0 - (k - 2.0) * sigma2 / jnp.clip(proj_ss, _JS_EPS, None)
    c = jnp.clip(c, 0.0, 1.0)
    return c[..., None, :] * proj


def _solve_cholesky(
    X: Float[Array, 'n k'],
    Y: Float[Array, 'n m'],
    l2: float,
) -> Float[Array, 'k m']:
    """Normal-equations OLS via Cholesky of :math:`X^{\\top} X + l_2 I`.

    Heavy step: :math:`X^{\\top} X` of shape ``(k, k)``.  Cheap when
    :math:`n \\gg k`; the Cholesky and two triangular solves are :math:`O(k^3)`.

    The L2 ridge is added *to the normal-equation matrix*, not to
    the data; this is equivalent to the augmented-system view but
    avoids materialising the ``(n + k, k)`` augmented matrix.
    Numerically more stable than the augmented form when ``l2``
    is very small.

    Parameters
    ----------
    X
        Regressor / design matrix, shape ``(n, k)`` (observations by
        regressors).
    Y
        Target matrix, shape ``(n, m)`` (observations by target channels).
    l2
        Ridge regularisation strength.  When ``> 0``, ``l2 * I`` is added to
        the Gram matrix :math:`X^{\\top} X`; ``0`` gives plain OLS.

    Returns
    -------
    Float[Array, 'k m']
        The regression coefficients :math:`\\beta`, shape ``(k, m)``, solving
        the (ridge) normal equations.
    """
    k = X.shape[-1]
    XtX = X.T @ X
    if l2 > 0.0:
        XtX = XtX + l2 * jnp.eye(k, dtype=X.dtype)
    XtY = X.T @ Y
    L = spd_chol(XtX, k)
    z = jsla.solve_triangular(L, XtY, lower=True)
    return jsla.solve_triangular(L.T, z, lower=False)


def _solve_svd(
    X: Float[Array, 'n k'],
    Y: Float[Array, 'n m'],
    l2: float,
) -> Float[Array, 'k m']:
    """SVD-based lstsq.  Stable even for rank-deficient ``X``.

    When ``l2 > 0``, we use the augmented-system view (stack
    ``sqrt(l2) I`` below ``X`` and zeros below ``Y``) so the
    same ``lstsq`` call handles the ridge.

    Parameters
    ----------
    X
        Regressor / design matrix, shape ``(n, k)`` (observations by
        regressors).
    Y
        Target matrix, shape ``(n, m)`` (observations by target channels).
    l2
        Ridge regularisation strength.  When ``> 0``, the system is augmented
        with ``sqrt(l2) * I`` (and matching zero rows in ``Y``) before the
        least-squares solve; ``0`` gives plain least squares.

    Returns
    -------
    Float[Array, 'k m']
        The least-squares regression coefficients :math:`\\beta`, shape
        ``(k, m)``.  Under rank deficiency with ``l2 == 0`` this is the
        minimum-norm solution.
    """
    if l2 > 0.0:
        k = X.shape[-1]
        m = Y.shape[-1]
        sqrt_l2 = jnp.sqrt(jnp.asarray(l2, dtype=X.dtype))
        ridge_X = sqrt_l2 * jnp.eye(k, dtype=X.dtype)
        ridge_Y = jnp.zeros((k, m), dtype=Y.dtype)
        X = jnp.concatenate([X, ridge_X], axis=-2)
        Y = jnp.concatenate([Y, ridge_Y], axis=-2)
    betas, _, _, _ = jnp.linalg.lstsq(X, Y, rcond=None)
    return betas


def _residualise_core(
    Y: Float[Array, 'n m'],
    X: Float[Array, 'n k'],
    *,
    weights: Optional[Float[Array, 'n']],
    l2: float,
    return_mode: ReturnMode,
    method: Method,
    shrinkage: Shrinkage,
) -> Float[Array, 'n m']:
    """Unbatched residualisation core; callers compose batch via ``jax.vmap``.

    Solve the (optionally weighted, optionally ridged) least-squares fit of
    ``Y`` on ``X``, form the projection ``X @ betas``, optionally shrink it,
    and return either that projection or the residual ``Y - proj``.

    Parameters
    ----------
    Y
        Target matrix, shape ``(n, m)`` (observations by target channels).
    X
        Regressor / design matrix, shape ``(n, k)`` (observations by
        regressors).
    weights
        Optional per-observation weights, shape ``(n,)``.  When given, rows of
        ``X`` and ``Y`` are pre-scaled by ``sqrt(weights)`` (weighted least
        squares); ``None`` is unweighted.  Weights affect the fitted
        coefficients only, not the projection geometry (the projection is
        reconstructed on the unweighted ``X``).
    l2
        Ridge regularisation strength passed to the chosen solver; ``0`` is
        ordinary least squares.
    return_mode
        ``"residual"`` returns ``Y - proj``; ``"projection"`` returns the
        (possibly shrunk) fitted projection.
    method
        ``"cholesky"`` or ``"svd"`` -- selects the least-squares solver.
    shrinkage
        ``"none"`` for the raw projection, or ``"james-stein"`` for
        positive-part James-Stein shrinkage of the projection.

    Returns
    -------
    Float[Array, 'n m']
        The residual or projection, shape ``(n, m)``, per ``return_mode``.

    Raises
    ------
    ValueError
        If ``method``, ``shrinkage``, or ``return_mode`` is not a recognised
        value.
    """
    if weights is not None:
        sqrt_w = jnp.sqrt(weights)[:, None]
        X_w = X * sqrt_w
        Y_w = Y * sqrt_w
    else:
        X_w = X
        Y_w = Y

    if method == 'cholesky':
        betas = _solve_cholesky(X_w, Y_w, l2)
    elif method == 'svd':
        betas = _solve_svd(X_w, Y_w, l2)
    else:
        raise ValueError(f'method={method!r}; expected "cholesky" or "svd".')

    # Reconstruct projection on the *unweighted* X; weights only
    # affect the betas, not the projection geometry.
    proj = X @ betas
    if shrinkage == 'james-stein':
        proj = _james_stein_shrink(Y - proj, proj, X.shape[-1])
    elif shrinkage != 'none':
        raise ValueError(
            f'shrinkage={shrinkage!r}; expected "none" or "james-stein".'
        )
    if return_mode == 'projection':
        return proj
    if return_mode == 'residual':
        return Y - proj
    raise ValueError(
        f'return_mode={return_mode!r}; expected "residual" or "projection".'
    )


def residualise(
    Y: Num[Array, '... C_Y obs'],
    X: Num[Array, '... C_X obs'],
    *,
    weights: Optional[Float[Array, '... obs']] = None,
    l2: float = 0.0,
    rowvar: bool = True,
    return_mode: ReturnMode = 'residual',
    method: Method = 'cholesky',
    shrinkage: Shrinkage = 'none',
) -> Num[Array, '... C_Y obs']:
    """Project ``Y`` orthogonally to the span of ``X`` (OLS).

    For each observation channel of ``Y``, regress against the
    columns of ``X`` and return the residual (or projection, per
    ``return_mode``).

    Parameters
    ----------
    Y
        Target tensor.  Default layout ``(..., C_Y, obs)`` --
        last axis is the observation axis.  Set ``rowvar=False``
        to flip.
    X
        Regressor tensor.  Same layout convention as ``Y``.
        Number of observations along the obs axis must match.
    weights
        Optional per-observation weights, ``(..., obs)``.  When
        given, performs weighted least squares with the named
        weights as the diagonal of the weight matrix W.
        ``None`` (default) is unweighted (W = I).
    l2
        Ridge regularisation strength.  ``0`` (default) is OLS;
        ``l2 > 0`` adds an ``l2 * I`` to the normal-equation
        matrix.  Bounds the condition number of the augmented
        system at :math:`1 / \\sqrt{l_2}`; recommended whenever ``X``
        might be near-collinear.  This *is* the **ridge-shrunk
        ("soft") residualisation**: a shrunk nuisance fit for an
        ill-conditioned confound set.
    rowvar
        ``True`` (default): observation axis is the *last* axis
        of ``Y`` / ``X``.  ``False``: observation axis is the
        *penultimate* axis (variables are last).
    return_mode
        ``"residual"`` (default) -- ``Y - proj``.
        ``"projection"`` -- the OLS prediction ``X @ betas``.
    method
        ``"cholesky"`` (default) -- fast normal-equations path via a
        rolled, cuSOLVER-free Cholesky of :math:`X^{\\top} X + l_2 I`
        (~2x faster
        on tall systems; runs on the broken-cuSOLVER GPU).  Recommended
        for well-conditioned ``X`` or whenever ``l2 > 0``.  For
        rank-deficient ``X`` -- more regressors than observations
        (``p > obs``) or exactly collinear columns -- the modified-
        Cholesky pivot floor keeps the factor **finite** (a regularised
        solution) rather than NaN, but the coefficients are then not
        uniquely meaningful; regularise with ``l2 > 0`` or switch to
        ``"svd"`` for the principled min-norm projection.
        ``"svd"`` -- SVD-based least squares (``jnp.linalg.lstsq``),
        robust for rank-deficient / ill-conditioned ``X``.  When the
        OLS solution is non-unique (rank-deficient ``X`` with
        ``l2 == 0``) it returns the **minimum-norm** least-squares
        solution: the coefficient vector of smallest ``||beta||_2``
        among all minimisers.  The *coefficients* are thus not
        uniquely interpretable under rank deficiency, but the
        **projection** ``X @ beta`` -- and hence the residual -- *is*
        unique: it is the orthogonal projection of ``Y`` onto
        ``col(X)``, which is why ``residual + projection == Y`` stays
        exact as ``p -> obs`` and beyond.  Use when ``X`` may be
        near-collinear and ``l2 == 0``.

        Pitfall: ``lstsq`` (``rcond=None``) zeros singular values
        below a relative cutoff (``~ max(obs, p) * eps * sigma_max``),
        so for *near*-collinear ``X`` (small but non-zero singular
        values) the effective column space -- exactly which directions
        get residualised out -- depends on that threshold and can flip
        for borderline directions.  If you need a deterministic,
        smoothly-shrinking treatment of weak directions, prefer
        ``l2 > 0`` (which makes the system full-rank, so the solve is
        the unique ridge estimate rather than a min-norm truncation)
        over relying on the SVD cutoff.
    shrinkage
        ``"none"`` (default) -- the raw projection is removed.
        ``"james-stein"`` -- positive-part James-Stein shrinkage
        of the per-channel nuisance projection toward zero (the
        other "soft" variant): the projection is scaled by
        :math:`c = \\left(1 - (k-2)\\,\\sigma^2 /
        \\lVert \\mathrm{proj} \\rVert^2 \\right)_+` before being
        subtracted, so confounds that explain a channel only
        weakly remove correspondingly less -- protecting real
        signal variance a high-dimensional / collinear confound
        set would otherwise strip.  A James-Stein shrinkage
        heuristic (a plug-in analogue of the :math:`k \\geq 3`
        dominance result, not the strict theorem); a no-op for
        :math:`k \\leq 2`.  Composes with ``l2`` (shrinks whatever
        ridge / OLS projection was fit).  Note the residual is then
        no longer exactly orthogonal to ``X`` -- the intended
        bias-variance trade of a soft removal.

    Returns
    -------
    Num[Array, '... C_Y obs']
        Residual or projection (per ``return_mode``), same shape as ``Y``.

    Notes
    -----
    The legacy ``hypercoil.functional.resid.residualise`` always
    used SVD via ``lstsq``; the Cholesky default here is ~2× faster
    on tall systems.  Behaviour matches SVD to within
    machine-precision when the system is well-conditioned (see
    the ``test_residualise_cholesky_vs_svd_parity_at_fp64`` test
    for the verified bound).
    """
    if rowvar:
        X_in = jnp.swapaxes(X, -1, -2)
        Y_in = jnp.swapaxes(Y, -1, -2)
    else:
        X_in, Y_in = X, Y

    if X_in.shape[-2] != Y_in.shape[-2]:
        raise ValueError(
            f'residualise: number of observations mismatch -- '
            f'X has {X_in.shape[-2]}, Y has {Y_in.shape[-2]} '
            f'(under rowvar={rowvar}).'
        )

    # Broadcast leading axes.
    batch_shape = jnp.broadcast_shapes(X_in.shape[:-2], Y_in.shape[:-2])
    X_in = jnp.broadcast_to(X_in, batch_shape + X_in.shape[-2:])
    Y_in = jnp.broadcast_to(Y_in, batch_shape + Y_in.shape[-2:])
    if weights is not None:
        weights = jnp.broadcast_to(weights, batch_shape + (X_in.shape[-2],))

    fn: Callable[..., Any]
    if weights is None:

        def core(
            X_: Float[Array, '...'], Y_: Float[Array, '...']
        ) -> Float[Array, '...']:
            return _residualise_core(
                Y_,
                X_,
                weights=None,
                l2=l2,
                return_mode=return_mode,
                method=method,
                shrinkage=shrinkage,
            )

        fn = core
        for _ in range(len(batch_shape)):
            fn = jax.vmap(fn, in_axes=(0, 0))
        out = fn(X_in, Y_in)
    else:

        def core_w(
            X_: Float[Array, '...'],
            Y_: Float[Array, '...'],
            w_: Float[Array, '...'],
        ) -> Float[Array, '...']:
            return _residualise_core(
                Y_,
                X_,
                weights=w_,
                l2=l2,
                return_mode=return_mode,
                method=method,
                shrinkage=shrinkage,
            )

        fn = core_w
        for _ in range(len(batch_shape)):
            fn = jax.vmap(fn, in_axes=(0, 0, 0))
        out = fn(X_in, Y_in, weights)

    if rowvar:
        out = jnp.swapaxes(out, -1, -2)
    return out


def _partial_core(
    Y: Float[Array, 'n m'],
    signal: Float[Array, 'n ks'],
    noise: Float[Array, 'n kn'],
    *,
    l2: float,
    method: Method,
    shrinkage: Shrinkage,
) -> Float[Array, 'n m']:
    """Unbatched non-aggressive residualisation core.

    Joint OLS fit ``Y ~ [signal | noise]``, then subtract **only** the noise
    contribution ``noise @ beta_noise``.  Because the betas come from the joint
    fit, variance *shared* between signal and noise stays with the (retained)
    signal -- the ICA-AROMA "non-aggressive" denoising, in contrast to the
    aggressive :func:`residualise` (``residualise(Y, noise)``) which removes
    the marginal noise projection (shared variance included).

    Parameters
    ----------
    Y
        Target matrix, shape ``(n, m)`` (observations by target channels).
    signal
        Regressors to retain, shape ``(n, ks)`` (observations by signal
        regressors).
    noise
        Regressors to remove, shape ``(n, kn)`` (observations by noise
        regressors).
    l2
        Ridge regularisation strength added to the joint Gram; ``0`` is OLS.
    method
        ``"cholesky"`` or ``"svd"`` -- selects the least-squares solver.
    shrinkage
        ``"none"`` for the raw noise projection, or ``"james-stein"`` for
        positive-part James-Stein shrinkage of the (joint-fit) noise
        projection before removal.

    Returns
    -------
    Float[Array, 'n m']
        The denoised target ``Y - noise_proj``, shape ``(n, m)``, with only
        the (possibly shrunk) noise-unique projection removed.

    Raises
    ------
    ValueError
        If ``method`` or ``shrinkage`` is not a recognised value.
    """
    X = jnp.concatenate([signal, noise], axis=-1)  # (n, ks + kn)
    if method == 'cholesky':
        betas = _solve_cholesky(X, Y, l2)
    elif method == 'svd':
        betas = _solve_svd(X, Y, l2)
    else:
        raise ValueError(f'method={method!r}; expected "cholesky" or "svd".')
    ks = signal.shape[-1]
    noise_proj = noise @ betas[ks:]
    if shrinkage == 'james-stein':
        # sigma^2 from the *full*-fit residual (signal removed too -> the actual
        # noise); shrink only the noise projection (k = #noise regressors).
        noise_proj = _james_stein_shrink(
            Y - X @ betas, noise_proj, noise.shape[-1]
        )
    elif shrinkage != 'none':
        raise ValueError(
            f'shrinkage={shrinkage!r}; expected "none" or "james-stein".'
        )
    return Y - noise_proj


def partial_residualise(
    Y: Num[Array, '... C_Y obs'],
    *,
    signal: Num[Array, '... C_S obs'],
    noise: Num[Array, '... C_N obs'],
    l2: float = 0.0,
    rowvar: bool = True,
    method: Method = 'cholesky',
    shrinkage: Shrinkage = 'none',
) -> Num[Array, '... C_Y obs']:
    """Non-aggressive (ICA-AROMA) partial residualisation of ``Y``.

    Fit ``Y`` jointly against ``[signal | noise]`` and remove **only** the
    fitted noise component ``noise @ beta_noise``, leaving the signal component
    *and the variance shared between signal and noise* intact.  This is the
    ICA-AROMA "non-aggressive" denoising: the standard aggressive residualisation
    (``residualise(Y, noise)``) regresses noise out marginally and so also strips
    any signal that happens to correlate with the noise regressors; the joint fit
    here attributes shared variance to ``signal`` and removes only the
    noise-unique part.

    Parameters
    ----------
    Y
        Target tensor, default layout ``(..., C_Y, obs)`` (obs on the last
        axis; set ``rowvar=False`` to flip).
    signal
        Regressors to **retain** (e.g. the AROMA signal components), same layout
        as ``Y``; the observation axis must match.
    noise
        Regressors to **remove** (e.g. the AROMA motion components), same layout.
    l2
        Ridge added to the joint Gram (``0`` = OLS).  Recommended if
        ``[signal | noise]`` may be near-collinear -- the ridge-shrunk ("soft")
        partial residualisation.
    rowvar
        ``True`` (default): observation axis is the last axis; ``False``: the
        penultimate axis.
    method
        ``"cholesky"`` (default, cuSOLVER-free) or ``"svd"`` -- as in
        :func:`residualise`.
    shrinkage
        ``"none"`` (default) or ``"james-stein"`` -- positive-part James-Stein
        shrinkage of the (joint-fit) **noise** projection toward zero before it
        is removed, as in :func:`residualise`; the soft variant for a large /
        collinear noise set (:math:`\\sigma^2` taken from the full-fit residual,
        :math:`k` = number of noise regressors).

    Returns
    -------
    Num[Array, '... C_Y obs']
        Denoised tensor, same shape as ``Y``.

    Notes
    -----
    With an empty ``signal`` (:math:`C_S = 0`) this reduces exactly to
    ``residualise(Y, noise)`` (aggressive); with an empty ``noise`` it returns
    ``Y`` unchanged.
    """
    if rowvar:
        S_in = jnp.swapaxes(signal, -1, -2)
        N_in = jnp.swapaxes(noise, -1, -2)
        Y_in = jnp.swapaxes(Y, -1, -2)
    else:
        S_in, N_in, Y_in = signal, noise, Y

    obs = Y_in.shape[-2]
    if S_in.shape[-2] != obs or N_in.shape[-2] != obs:
        raise ValueError(
            f'partial_residualise: observation-count mismatch -- Y has {obs}, '
            f'signal has {S_in.shape[-2]}, noise has {N_in.shape[-2]} '
            f'(under rowvar={rowvar}).'
        )

    batch_shape = jnp.broadcast_shapes(
        S_in.shape[:-2], N_in.shape[:-2], Y_in.shape[:-2]
    )
    S_in = jnp.broadcast_to(S_in, batch_shape + S_in.shape[-2:])
    N_in = jnp.broadcast_to(N_in, batch_shape + N_in.shape[-2:])
    Y_in = jnp.broadcast_to(Y_in, batch_shape + Y_in.shape[-2:])

    def core(
        Y_: Float[Array, '...'],
        S_: Float[Array, '...'],
        N_: Float[Array, '...'],
    ) -> Float[Array, '...']:
        return _partial_core(
            Y_, S_, N_, l2=l2, method=method, shrinkage=shrinkage
        )

    fn: Callable[..., Any] = core
    for _ in range(len(batch_shape)):
        fn = jax.vmap(fn, in_axes=(0, 0, 0))
    out = fn(Y_in, S_in, N_in)

    if rowvar:
        out = jnp.swapaxes(out, -1, -2)
    return cast(Num[Array, '... C_Y obs'], out)
