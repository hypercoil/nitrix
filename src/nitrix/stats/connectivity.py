# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Regularised covariance and sparse-precision estimators for connectivity.

The raw empirical covariance is noisy and may be singular when the number of
variables approaches the number of observations (:math:`p \\sim n` -- the
small-sample regime of resting-state connectomes).  This module provides the
regularised estimators that the connectome literature commonly defaults to.

Analytic-shrinkage estimators -- the Ledoit-Wolf and oracle-approximating
shrinkage (OAS) methods -- form a convex blend of the sample covariance
:math:`S` toward a scaled identity,

.. math::

    \\hat{\\Sigma} = (1 - \\alpha)\\, S + \\alpha\\, \\mu\\, I,
    \\qquad \\mu = \\operatorname{tr}(S) / p,

with the shrinkage intensity :math:`\\alpha` given in closed form (no
cross-validation).  Ledoit-Wolf is a widely used default connectome covariance
estimator, so this supplies a covariance whose inverse gives an equivalent
precision / partial-correlation path (inverting :math:`\\hat{\\Sigma}` with the
cuSOLVER-free small-matrix solve).

The graphical LASSO estimates the sparse inverse covariance (precision), whose
support is the conditional-independence graph long used in the functional-MRI
literature,

.. math::

    \\hat{\\Theta} = \\operatorname*{argmin}_{\\Theta}\\;
    \\langle S, \\Theta \\rangle - \\log\\det\\Theta
    + \\lambda \\lVert\\Theta\\rVert_{1,\\mathrm{off}},

solved by Friedman, Hastie and Tibshirani (2008) coordinate descent on the
working covariance :math:`W = \\Theta^{-1}` directly (no per-iteration
factorisation).  :func:`glasso_path` performs a warm-started :math:`\\lambda`
sweep and :func:`ebic_score` provides extended-BIC model selection (Foygel and
Drton, 2010).

The implementation is pure JAX.  The shrinkage path is a handful of trace and
Frobenius-norm reductions plus one scalar :math:`\\alpha`; the graphical-LASSO
path is rolled coordinate descent (``lax.fori_loop`` / ``lax.scan``), so the
computation graph stays :math:`O(p^2)` and compilation is flat in :math:`p`.
Everything is differentiable and GPU-resident, and cuSOLVER-free (the
:math:`\\log\\det` used by the extended BIC goes through a rolled Cholesky).
Batch over subjects or edges with ``vmap``.  The data matrix ``X`` is
``(n_samples, n_features)`` (the ``sklearn.covariance`` convention), whereas
:func:`glasso` consumes a ``(p, p)`` sample covariance ``S`` directly.

References
----------
.. [LW2004] Ledoit, O. and Wolf, M. (2004). A well-conditioned estimator for
   large-dimensional covariance matrices. *Journal of Multivariate Analysis*,
   88(2), 365-411. :doi:`10.1016/S0047-259X(03)00096-4`
.. [Chen2010] Chen, Y., Wiesel, A., Eldar, Y. C. and Hero, A. O. (2010).
   Shrinkage algorithms for MMSE covariance estimation. *IEEE Transactions on
   Signal Processing*, 58(10), 5016-5029. :doi:`10.1109/TSP.2010.2053029`
.. [FHT2008] Friedman, J., Hastie, T. and Tibshirani, R. (2008). Sparse inverse
   covariance estimation with the graphical lasso. *Biostatistics*, 9(3),
   432-441. :doi:`10.1093/biostatistics/kxm045`
.. [FD2010] Foygel, R. and Drton, M. (2010). Extended Bayesian information
   criteria for Gaussian graphical models. *Advances in Neural Information
   Processing Systems*, 23, 604-612.
"""

from __future__ import annotations

from typing import Literal, Tuple, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jaxtyping import Array, Float

from ..linalg._smalllinalg import small_inv_logdet

__all__ = [
    'ebic_score',
    'glasso',
    'glasso_path',
    'ledoit_wolf',
    'oas',
    'shrunk_covariance',
]

ShrinkageMethod = Literal['ledoit_wolf', 'oas']


def _empirical(
    X: Float[Array, 'n p'], assume_centered: bool
) -> Tuple[Float[Array, 'n p'], Float[Array, 'p p'], int, int]:
    """Centre the data and form the biased empirical covariance.

    Parameters
    ----------
    X : Float[Array, 'n p']
        Data matrix of ``n`` observations by ``p`` features.
    assume_centered : bool
        If ``True``, treat ``X`` as already mean-centred and skip subtracting
        the per-feature mean; otherwise centre each column first.

    Returns
    -------
    Xc : Float[Array, 'n p']
        The (possibly) centred data matrix.
    s : Float[Array, 'p p']
        The empirical covariance :math:`X_c^{\\top} X_c / n` (the biased,
        divide-by-``n`` estimate).
    n : int
        Number of observations.
    p : int
        Number of features.
    """
    n, p = X.shape
    Xc = X if assume_centered else X - jnp.mean(X, axis=0, keepdims=True)
    s = (Xc.T @ Xc) / n
    return Xc, s, n, p


def _blend(
    s: Float[Array, 'p p'], alpha: Float[Array, ''], p: int
) -> Float[Array, 'p p']:
    mu = jnp.trace(s) / p
    return (1.0 - alpha) * s + alpha * mu * jnp.eye(p, dtype=s.dtype)


def ledoit_wolf(
    X: Float[Array, 'n p'],
    *,
    assume_centered: bool = False,
) -> Tuple[Float[Array, 'p p'], Float[Array, '']]:
    """Ledoit-Wolf analytic-shrinkage covariance estimator.

    Forms the convex blend :math:`(1 - \\alpha) S + \\alpha \\mu I` of the
    sample covariance :math:`S` toward the scaled identity
    :math:`\\mu I` (with :math:`\\mu = \\operatorname{tr}(S) / p`), using the
    closed-form Ledoit-Wolf shrinkage intensity

    .. math::

        \\alpha = \\beta^2 / \\delta^2,
        \\qquad \\delta^2 = \\lVert S - \\mu I \\rVert_F^2,
        \\qquad \\beta^2 = \\frac{1}{n^2} \\sum_k \\lVert x_k \\rVert^4
        - \\frac{1}{n} \\lVert S \\rVert_F^2,

    with :math:`\\beta^2` clipped to :math:`[0, \\delta^2]`.  This matches
    ``sklearn.covariance.ledoit_wolf``.

    Parameters
    ----------
    X : Float[Array, 'n p']
        Data matrix of ``n`` observations by ``p`` features.
    assume_centered : bool, optional
        If ``True``, treat ``X`` as already mean-centred; otherwise centre each
        feature before estimating the covariance.  Default ``False``.

    Returns
    -------
    cov : Float[Array, 'p p']
        The shrunk covariance matrix :math:`\\hat{\\Sigma}`.
    shrinkage : Float[Array, '']
        The scalar shrinkage intensity :math:`\\alpha \\in [0, 1]`.
    """
    Xc, s, n, p = _empirical(X, assume_centered)
    mu = jnp.trace(s) / p
    s_norm2 = jnp.sum(s * s)
    delta2 = s_norm2 - p * mu * mu
    sq_norms = jnp.sum(Xc * Xc, axis=1)  # ||x_k||^2
    beta2 = jnp.sum(sq_norms * sq_norms) / (n * n) - s_norm2 / n
    beta2 = jnp.clip(beta2, 0.0, delta2)
    alpha = jnp.where(delta2 > 0, beta2 / delta2, 0.0)
    return _blend(s, alpha, p), alpha


def oas(
    X: Float[Array, 'n p'],
    *,
    assume_centered: bool = False,
) -> Tuple[Float[Array, 'p p'], Float[Array, '']]:
    """Oracle-approximating-shrinkage (OAS) covariance estimator.

    Uses the same convex blend of the sample covariance toward a scaled
    identity as :func:`ledoit_wolf`, but with the different closed-form
    shrinkage intensity of Chen, Wiesel, Eldar and Hero (2010).  This matches
    ``sklearn.covariance.oas``.

    Parameters
    ----------
    X : Float[Array, 'n p']
        Data matrix of ``n`` observations by ``p`` features.
    assume_centered : bool, optional
        If ``True``, treat ``X`` as already mean-centred; otherwise centre each
        feature before estimating the covariance.  Default ``False``.

    Returns
    -------
    cov : Float[Array, 'p p']
        The shrunk covariance matrix :math:`\\hat{\\Sigma}`.
    shrinkage : Float[Array, '']
        The scalar shrinkage intensity :math:`\\alpha \\in [0, 1]`.
    """
    _, s, n, p = _empirical(X, assume_centered)
    mu = jnp.trace(s) / p
    s_sq_mean = jnp.mean(s * s)
    num = s_sq_mean + mu * mu
    den = (n + 1) * (s_sq_mean - mu * mu / p)
    alpha = jnp.where(den > 0, jnp.clip(num / den, 0.0, 1.0), 1.0)
    return _blend(s, alpha, p), alpha


def shrunk_covariance(
    X: Float[Array, 'n p'],
    *,
    method: ShrinkageMethod = 'ledoit_wolf',
    assume_centered: bool = False,
) -> Float[Array, 'p p']:
    """Analytic-shrinkage covariance, returning the covariance only.

    Thin wrapper over :func:`ledoit_wolf` and :func:`oas` that returns just the
    shrunk covariance, discarding the shrinkage intensity.

    Parameters
    ----------
    X : Float[Array, 'n p']
        Data matrix of ``n`` observations by ``p`` features.
    method : {'ledoit_wolf', 'oas'}, optional
        Which analytic-shrinkage estimator to use.  ``'ledoit_wolf'`` (the
        default) is a common connectome default; ``'oas'`` selects the
        oracle-approximating-shrinkage intensity.
    assume_centered : bool, optional
        If ``True``, treat ``X`` as already mean-centred; otherwise centre each
        feature before estimating the covariance.  Default ``False``.

    Returns
    -------
    Float[Array, 'p p']
        The shrunk covariance matrix.

    Raises
    ------
    ValueError
        If ``method`` is neither ``'ledoit_wolf'`` nor ``'oas'``.
    """
    if method == 'ledoit_wolf':
        return ledoit_wolf(X, assume_centered=assume_centered)[0]
    if method == 'oas':
        return oas(X, assume_centered=assume_centered)[0]
    raise ValueError(f"method={method!r}; expected 'ledoit_wolf' or 'oas'.")


# ---------------------------------------------------------------------------
# Graphical LASSO -- sparse precision via FHT (2008) coordinate descent.
# ---------------------------------------------------------------------------


def _soft(x: Float[Array, '...'], t: Float[Array, '']) -> Float[Array, '...']:
    """Soft-thresholding operator, the proximal map of the lasso penalty.

    Computes :math:`\\operatorname{sign}(x) \\cdot \\max(\\lvert x \\rvert - t,
    0)` elementwise.

    Parameters
    ----------
    x : Float[Array, '...']
        Input array; thresholding is applied elementwise.
    t : Float[Array, '']
        Non-negative threshold.

    Returns
    -------
    Float[Array, '...']
        The soft-thresholded array, of the same shape as ``x``.
    """
    return jnp.sign(x) * jnp.clip(jnp.abs(x) - t, 0.0, None)


def _glasso_wb(
    S: Float[Array, 'p p'],
    lam: Float[Array, ''],
    W: Float[Array, 'p p'],
    B: Float[Array, 'p p'],
    n_outer: int,
    n_inner: int,
) -> Tuple[Float[Array, 'p p'], Float[Array, 'p p']]:
    """Coordinate-descent core of the graphical LASSO.

    Runs the block-coordinate-descent solver, returning the converged working
    covariance ``W`` and the per-column lasso coefficients ``B``, where column
    ``j`` holds the regression of node ``j`` on the remaining nodes.  Every loop
    is rolled (``lax.fori_loop``) so the computation graph stays :math:`O(p^2)`
    and compilation is flat in :math:`p`.  The diagonal of ``W`` is held fixed
    (the off-diagonal-only penalty pins :math:`W_{jj} = S_{jj}`), so seeding
    ``W = S`` enforces the diagonal optimality condition.

    Parameters
    ----------
    S : Float[Array, 'p p']
        Sample covariance matrix.
    lam : Float[Array, '']
        Non-negative L1 penalty on the off-diagonal entries.
    W : Float[Array, 'p p']
        Initial working covariance (warm start); typically ``S`` or the ``W``
        carried from a previous penalty.
    B : Float[Array, 'p p']
        Initial per-column lasso coefficients (warm start).
    n_outer : int
        Number of outer sweeps over columns.
    n_inner : int
        Number of inner lasso sweeps per column.

    Returns
    -------
    W : Float[Array, 'p p']
        The converged working covariance.
    B : Float[Array, 'p p']
        The converged per-column lasso coefficients.
    """
    p = S.shape[0]
    idx = jnp.arange(p)

    def per_var(
        j: Array, carry: Tuple[Float[Array, 'p p'], Float[Array, 'p p']]
    ) -> Tuple[Float[Array, 'p p'], Float[Array, 'p p']]:
        W, B = carry
        beta0 = B[:, j]

        def coord(k: Array, beta: Float[Array, 'p']) -> Float[Array, 'p']:
            # Lasso coordinate update for the j-th column regression; the j-th
            # coordinate is pinned to 0 (no self-edge in the off-diagonal block).
            r = S[k, j] - W[k] @ beta + W[k, k] * beta[k]
            return beta.at[k].set(
                jnp.where(k == j, 0.0, _soft(r, lam) / W[k, k])
            )

        def sweep(_: Array, beta: Float[Array, 'p']) -> Float[Array, 'p']:
            return cast(Float[Array, 'p'], lax.fori_loop(0, p, coord, beta))

        beta = lax.fori_loop(0, n_inner, sweep, beta0)
        # w_12 <- W_11 beta; the diagonal w_jj stays at S_jj (held fixed).
        newcol = jnp.where(idx == j, W[j, j], W @ beta)
        W = W.at[:, j].set(newcol).at[j, :].set(newcol)
        B = B.at[:, j].set(beta)
        return W, B

    def outer(
        _: Array, carry: Tuple[Float[Array, 'p p'], Float[Array, 'p p']]
    ) -> Tuple[Float[Array, 'p p'], Float[Array, 'p p']]:
        return cast(
            Tuple[Float[Array, 'p p'], Float[Array, 'p p']],
            lax.fori_loop(0, p, per_var, carry),
        )

    return cast(
        Tuple[Float[Array, 'p p'], Float[Array, 'p p']],
        lax.fori_loop(0, n_outer, outer, (W, B)),
    )


def _theta_from_wb(
    W: Float[Array, 'p p'], B: Float[Array, 'p p']
) -> Float[Array, 'p p']:
    """Recover the sparse precision from the working covariance.

    Reconstructs the precision matrix :math:`\\Theta` from the converged
    working covariance and the per-column lasso coefficients via the
    partitioned-inverse identities of Friedman, Hastie and Tibshirani (2008):
    for column ``j``,

    .. math::

        \\theta_{jj} = \\frac{1}{w_{jj} - w_{12}^{\\top} \\beta_j},
        \\qquad \\theta_{-j,\\,j} = -\\beta_j\\, \\theta_{jj}.

    Sparsity is inherited from ``B`` (the soft-thresholding zeros entries
    exactly), so :math:`\\Theta` carries the conditional-independence support
    rather than a dense :math:`W^{-1}`.

    Parameters
    ----------
    W : Float[Array, 'p p']
        Converged working covariance.
    B : Float[Array, 'p p']
        Converged per-column lasso coefficients.

    Returns
    -------
    Float[Array, 'p p']
        The recovered sparse precision matrix :math:`\\Theta`.
    """
    p = W.shape[0]
    idx = jnp.arange(p)

    def rec(j: Array) -> Float[Array, 'p']:
        beta = B[:, j]
        t22 = 1.0 / (W[j, j] - W[:, j] @ beta)
        return (-beta * t22).at[j].set(t22)

    return jax.vmap(rec)(idx).T


def glasso(
    S: Float[Array, 'p p'],
    lam: float | Float[Array, ''],
    *,
    n_outer: int = 100,
    n_inner: int = 50,
) -> Float[Array, 'p p']:
    """Graphical-LASSO sparse precision (inverse covariance).

    Solves

    .. math::

        \\operatorname*{argmin}_{\\Theta}\\;
        \\langle S, \\Theta \\rangle - \\log\\det\\Theta
        + \\lambda \\lVert\\Theta\\rVert_{1,\\mathrm{off}}

    (an off-diagonal-only L1 penalty -- the standard convention) by the
    block-coordinate descent of Friedman, Hastie and Tibshirani (2008): an
    outer sweep over columns, each an inner lasso on the working covariance
    ``W``, with the precision recovered from the converged per-column
    coefficients.  There is no factorisation and no cuSOLVER; all loops are
    rolled (an :math:`O(p^2)` graph), so the solve is differentiable through the
    fixed iteration budget and compiles flat in :math:`p` (well suited to
    connectome sizes of :math:`p = 100`-:math:`400`).

    Parameters
    ----------
    S
        ``(p, p)`` sample covariance (e.g. ``stats.covariance.cov`` or a
        shrinkage estimate).  Off-diagonal-only penalty fixes ``W_jj = S_jj``.
    lam
        Non-negative L1 penalty.  Larger ``lam`` -> sparser ``Theta``.
    n_outer, n_inner
        Outer column sweeps and inner lasso sweeps per column (fixed budget; the
        defaults reach KKT stationarity to machine precision on well-scaled
        connectome covariances and can be reduced for speed).

    Returns
    -------
    Theta
        ``(p, p)`` symmetric sparse precision; the inactive off-diagonals are
        exactly zero (the conditional-independence graph).
    """
    p = S.shape[0]
    # Round 4: the off-diagonal-only solver fixes W_jj = S_jj and divides by it
    # (and seeds 1 / S_jj), so a non-positive diagonal yields inf / nan. Reject it
    # host-side (skip under jit, where S is a tracer).
    if not isinstance(S, jax.core.Tracer):
        diag = np.diag(np.asarray(S))
        if np.any(diag <= 0.0):
            raise ValueError(
                'glasso: the input covariance S has a non-positive diagonal '
                f'entry (min={float(diag.min()):.3g}); the solver fixes '
                'W_jj = S_jj and divides by it. Pass a valid covariance with a '
                'strictly positive diagonal (e.g. a regularised / shrunk S).'
            )
    lam = jnp.asarray(lam, dtype=S.dtype)
    B0 = jnp.zeros((p, p), S.dtype)
    W, B = _glasso_wb(S, lam, S, B0, n_outer, n_inner)
    return _theta_from_wb(W, B)


def glasso_path(
    S: Float[Array, 'p p'],
    lambdas: Float[Array, 'L'],
    *,
    n_outer: int = 100,
    n_inner: int = 50,
) -> Float[Array, 'L p p']:
    """Warm-started graphical-LASSO regularisation path over ``lambdas``.

    Sweeps the penalties in the given order, carrying the working covariance
    ``W`` and the lasso coefficients ``B`` from one penalty to the next so that
    each solve starts warm.  This is far cheaper than a cold restart at every
    penalty and is the usual way to trace a graphical-LASSO path; either sweep
    direction (descending, dense to sparse, or ascending) warm-starts.  Pair the
    returned precisions with :func:`ebic_score` for model selection.

    Parameters
    ----------
    S : Float[Array, 'p p']
        Sample covariance matrix shared across all penalties.
    lambdas : Float[Array, 'L']
        The ``L`` non-negative L1 penalties, swept in the given order.
    n_outer : int, optional
        Number of outer column sweeps per solve.  Default ``100``.
    n_inner : int, optional
        Number of inner lasso sweeps per column.  Default ``50``.

    Returns
    -------
    Float[Array, 'L p p']
        The stacked precision matrices, one ``(p, p)`` estimate per penalty and
        aligned with ``lambdas``.
    """
    p = S.shape[0]
    lambdas = jnp.asarray(lambdas, dtype=S.dtype)
    B0 = jnp.zeros((p, p), S.dtype)

    def step(
        carry: Tuple[Float[Array, 'p p'], Float[Array, 'p p']],
        lam: Float[Array, ''],
    ) -> Tuple[
        Tuple[Float[Array, 'p p'], Float[Array, 'p p']], Float[Array, 'p p']
    ]:
        W, B = carry
        W, B = _glasso_wb(S, lam, W, B, n_outer, n_inner)
        return (W, B), _theta_from_wb(W, B)

    _, thetas = lax.scan(step, (S, B0), lambdas)
    return thetas


def ebic_score(
    theta: Float[Array, 'p p'],
    S: Float[Array, 'p p'],
    n: int | Float[Array, ''],
    *,
    gamma: float = 0.5,
    edge_tol: float = 1e-8,
) -> Float[Array, '']:
    """Extended Bayesian information criterion of a precision estimate.

    Evaluates the extended BIC of Foygel and Drton (2010),

    .. math::

        \\mathrm{EBIC}_\\gamma = n\\,(\\operatorname{tr}(S \\Theta)
        - \\log\\det\\Theta) + E \\log n + 4 \\gamma E \\log p,

    where :math:`E` is the number of off-diagonal edges (upper-triangle entries
    with :math:`\\lvert\\theta\\rvert >` ``edge_tol``, counted once).  The
    Gaussian-graphical-model term
    :math:`-2\\,\\ell = n\\,(\\operatorname{tr}(S \\Theta) - \\log\\det\\Theta)`
    uses the cuSOLVER-free rolled-Cholesky :math:`\\log\\det`.  The parameter
    :math:`\\gamma \\in [0, 1]` tunes the extra sparsity prior
    (:math:`\\gamma = 0` recovers the plain BIC); select the path point with the
    smallest score.

    Parameters
    ----------
    theta : Float[Array, 'p p']
        Precision (inverse covariance) estimate to score.
    S : Float[Array, 'p p']
        Sample covariance matrix that produced ``theta``.
    n : int or Float[Array, '']
        Number of observations underlying ``S``.
    gamma : float, optional
        Extended-BIC sparsity-prior weight :math:`\\gamma \\in [0, 1]`;
        ``gamma = 0`` recovers the plain BIC.  Default ``0.5``.
    edge_tol : float, optional
        Magnitude threshold above which an off-diagonal entry counts as an
        edge.  Default ``1e-8``.

    Returns
    -------
    Float[Array, '']
        The scalar extended-BIC score; lower is better.
    """
    p = theta.shape[0]
    _, logdet = small_inv_logdet(theta, p)
    tr_s_theta = jnp.sum(S * theta)
    neg2ll = n * (tr_s_theta - logdet)
    offdiag = jnp.abs(theta) > edge_tol
    n_edges = jnp.sum(jnp.triu(offdiag, k=1))
    return neg2ll + n_edges * jnp.log(n) + 4.0 * gamma * n_edges * jnp.log(p)
