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
  **cuSOLVER-free** rolled Cholesky (``_chol_lower``) + cuBLAS triangular
  solves, so it runs on the cuSOLVER-affected GPU stacks (where
  ``jnp.linalg.cholesky`` / ``svd`` fail to create a handle).  Tall-and-skinny
  optimised -- for ``X: (obs, k)`` with ``obs >> k`` (typical fMRI: 400 TRs ×
  24 confounds), the heavy step is ``X^T X`` (``O(obs * k^2)``,
  bandwidth-limited) followed by a ``k × k`` Cholesky and two triangular solves
  (``O(k^3)``).  ``X^T X`` is numerically squared in condition number, so this
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
``X^T diag(w) X`` explicitly -- preserves the QR / SVD structure
and is numerically equivalent.

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
from jax import lax
from jaxtyping import Array, Float, Num

__all__ = ['residualise', 'partial_residualise']


ReturnMode = Literal['residual', 'projection']
Method = Literal['cholesky', 'svd']

# Modified-Cholesky pivot floor (relative to the Gram's diagonal scale): a
# well-conditioned solve is unperturbed, while a degenerate pivot stays positive
# (finite, regularised) instead of producing sqrt(negative) = NaN.
_PIVOT_REL_FLOOR = 1e-12


def _chol_lower(A: Float[Array, 'k k']) -> Float[Array, 'k k']:
    """Lower-triangular Cholesky factor ``L`` (``A = L L^T``) of an SPD ``A``
    via a rolled, cuSOLVER-free factorization.

    Mirrors ``linalg._smalllinalg.spd_inv_logdet_chol``: ``jnp.linalg.cholesky``
    issues a ``potrf`` custom-call that fails on the broken-cuSOLVER L4, so we
    factor column-by-column with a ``lax.fori_loop`` (rolled, ``O(k^2)`` graph)
    using only elementwise algebra -- no cuSOLVER routine, fully differentiable.
    The modified-Cholesky pivot floor keeps a rank-deficient Gram finite (a
    regularised factor) rather than NaN.
    """
    k = A.shape[-1]
    idx = jnp.arange(k)
    floor = (
        _PIVOT_REL_FLOOR * jnp.max(jnp.diagonal(A)) + jnp.finfo(A.dtype).tiny
    )

    def body(j: Array, L: Float[Array, 'k k']) -> Float[Array, 'k k']:
        Lj = lax.dynamic_index_in_dim(L, j, axis=0, keepdims=False)  # (k,)
        s = L @ Lj  # s_i = sum_{l<j} L[i,l] L[j,l]
        diag = jnp.sqrt(jnp.maximum(A[j, j] - s[j], floor))
        col = (A[:, j] - s) / diag
        new_col = jnp.where(idx == j, diag, jnp.where(idx > j, col, 0.0))
        return cast(
            Float[Array, 'k k'],
            lax.dynamic_update_index_in_dim(L, new_col, j, axis=1),
        )

    return cast(
        Float[Array, 'k k'],
        lax.fori_loop(0, k, body, jnp.zeros((k, k), A.dtype)),
    )


def _solve_cholesky(
    X: Float[Array, 'n k'],
    Y: Float[Array, 'n m'],
    l2: float,
) -> Float[Array, 'k m']:
    """Normal-equations OLS via Cholesky of ``X^T X + l2 I``.

    Heavy step: ``X^T X`` of shape ``(k, k)``.  Cheap when ``n
    >> k``; the Cholesky and two triangular solves are ``O(k^3)``.

    The L2 ridge is added *to the normal-equation matrix*, not to
    the data; this is equivalent to the augmented-system view but
    avoids materialising the ``(n + k, k)`` augmented matrix.
    Numerically more stable than the augmented form when ``l2``
    is very small.
    """
    k = X.shape[-1]
    XtX = X.T @ X
    if l2 > 0.0:
        XtX = XtX + l2 * jnp.eye(k, dtype=X.dtype)
    XtY = X.T @ Y
    L = _chol_lower(XtX)
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
) -> Float[Array, 'n m']:
    """Unbatched core; callers compose batch via ``jax.vmap``."""
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
        system at ``1 / sqrt(l2)``; recommended whenever ``X``
        might be near-collinear.
    rowvar
        ``True`` (default): observation axis is the *last* axis
        of ``Y`` / ``X``.  ``False``: observation axis is the
        *penultimate* axis (variables are last).
    return_mode
        ``"residual"`` (default) -- ``Y - proj``.
        ``"projection"`` -- the OLS prediction ``X @ betas``.
    method
        ``"cholesky"`` (default) -- fast normal-equations path via a
        rolled, cuSOLVER-free Cholesky of ``X^T X + l2 I`` (~2x faster
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

    Returns
    -------
    Residual or projection, same shape as ``Y``.

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
) -> Float[Array, 'n m']:
    """Unbatched non-aggressive residualisation core.

    Joint OLS fit ``Y ~ [signal | noise]``, then subtract **only** the noise
    contribution ``noise @ beta_noise``.  Because the betas come from the joint
    fit, variance *shared* between signal and noise stays with the (retained)
    signal -- the ICA-AROMA "non-aggressive" denoising, in contrast to the
    aggressive ``residualise(Y, noise)`` which removes the marginal noise
    projection (shared variance included).
    """
    X = jnp.concatenate([signal, noise], axis=-1)  # (n, ks + kn)
    if method == 'cholesky':
        betas = _solve_cholesky(X, Y, l2)
    elif method == 'svd':
        betas = _solve_svd(X, Y, l2)
    else:
        raise ValueError(f'method={method!r}; expected "cholesky" or "svd".')
    ks = signal.shape[-1]
    return Y - noise @ betas[ks:]


def partial_residualise(
    Y: Num[Array, '... C_Y obs'],
    *,
    signal: Num[Array, '... C_S obs'],
    noise: Num[Array, '... C_N obs'],
    l2: float = 0.0,
    rowvar: bool = True,
    method: Method = 'cholesky',
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
        ``[signal | noise]`` may be near-collinear.
    rowvar
        ``True`` (default): observation axis is the last axis; ``False``: the
        penultimate axis.
    method
        ``"cholesky"`` (default, cuSOLVER-free) or ``"svd"`` -- as in
        :func:`residualise`.

    Returns
    -------
    Denoised tensor, same shape as ``Y``.

    Notes
    -----
    With an empty ``signal`` (``C_S = 0``) this reduces exactly to
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
        return _partial_core(Y_, S_, N_, l2=l2, method=method)

    fn: Callable[..., Any] = core
    for _ in range(len(batch_shape)):
        fn = jax.vmap(fn, in_axes=(0, 0, 0))
    out = fn(Y_in, S_in, N_in)

    if rowvar:
        out = jnp.swapaxes(out, -1, -2)
    return cast(Num[Array, '... C_Y obs'], out)
