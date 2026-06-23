# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Mass-univariate Gaussian-process regression with lengthscale estimation.

``gp_fit`` fits, per element (voxel / vertex / fixel), a Gaussian-process
smooth of a single covariate ``x`` with its kernel **lengthscale ``rho``
estimated by REML** -- the piece the fixed-``rho`` :func:`~nitrix.stats.basis.hsgp_basis`
(which rides ``gam_fit`` unchanged) leaves open::

    y_v = T alpha_v + f_v(x) + e_v,   f_v ~ GP(0, sigma_f^2 K_rho),
    e_v ~ N(0, sigma_e^2)

The engine is the Hilbert-space approximation (HSGP; Solin-Sarkka 2020,
Riutort-Mayol/Burkner 2023): ``f`` is expanded in a **fixed** Laplace-Dirichlet
eigenbasis ``Phi = [phi_j(x)]`` on the bounded domain ``[c - L, c + L]``, and the
kernel enters *only* through the spectral-density weights
``s_j(rho) = S_rho(sqrt(lambda_j))`` (:func:`~nitrix.linalg.kernel.spectral_density`).
This is what makes lengthscale estimation tractable: because ``Phi`` does **not**
depend on ``rho``, the GP is a penalised regression with a fixed design and a
**diagonal, ``rho``-dependent penalty** ``S(rho) = diag(1 / s_j(rho))`` -- the
REML criterion is smooth in ``rho`` with **no eigendecomposition in the loop**
(contrast the kriging :func:`~nitrix.stats.basis.gp_basis`, whose eigenbasis
moves with ``rho``).

Two profiled parameters, one shared search
------------------------------------------

For a candidate ``rho`` the model is a Gaussian penalised regression with the
single smoothing parameter ``lambda = sigma_e^2 / sigma_f^2`` (the GAM identity:
the Fellner-Schall smoothing parameter *is* the inverse GP amplitude).  So:

- **Inner** (fixed ``rho``): the generalized Fellner-Schall step (Wood & Fasiolo
  2017) selects ``lambda`` -- exactly the ``gam_fit`` machinery, here with one
  diagonal penalty ``diag(1 / s(rho))`` on the smooth block.
- **Outer** (select ``rho``): a 1-D search over ``log rho`` of the **pooled REML
  marginal likelihood** ``sum_v V_r(rho, lambda_v)``, on a fixed log-spaced grid
  with a parabolic refinement.  ``rho`` is **shared across elements** (one
  eigenbasis, one smoothness), while the amplitude ``sigma_f^2`` and noise
  ``sigma_e^2`` are **per element** -- the natural mass-univariate factoring.

Every solve is cuSOLVER-free (``linalg._smalllinalg``) and ``jit`` / ``vmap``
clean; the only ``N``-sized objects are the one-off cross-products ``X^T Y`` and
``diag(Y Y^T)``, so peak memory is ``O(V (m + q)^2)`` -- no per-element ``(N, N)``
GP covariance is ever materialised.

REML criterion
--------------

With the scale ``sigma_e^2`` profiled out, the per-element restricted negative
log-likelihood (up to an additive constant in ``n`` and ``M_0`` only -- the same
for every ``rho`` and every competing GP model on a given ``y``) is

    -2 l_R = (n - M_0) log(D_p) + log|H| - log|S_lambda|_+,

with ``H = X^T X + S_lambda`` the penalised Hessian, ``D_p = y^T y - beta^T X^T y``
the penalised residual sum of squares, ``S_lambda = lambda diag(0, 1/s(rho))``,
``log|S_lambda|_+ = m log lambda - sum_j log s_j(rho)`` its log-pseudo-determinant,
and ``M_0`` the number of unpenalised (fixed-effect) columns.  ``GPResult.log_mlik``
reports ``l_R`` (so larger is better; the dropped constant cancels in any
model comparison on the same data).

References
----------
- Solin, A. & Sarkka, S. (2020). Hilbert space methods for reduced-rank Gaussian
  process regression.  Statistics and Computing 30, 419-446.
- Riutort-Mayol, G. et al. (2023). Practical Hilbert space approximate Bayesian
  Gaussian processes for probabilistic programming.  Stat. Comput. 33, 17.
- Wood, S. N. & Fasiolo, M. (2017). A generalized Fellner-Schall method for
  smoothing parameter optimization.  Biometrics 73, 1071-1081.
- Wood, S. N. (2011). Fast stable restricted maximum likelihood ... J. R.
  Statist. Soc. B 73, 3-36.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Literal,
    Optional,
    Protocol,
    Tuple,
    Union,
    cast,
)

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jaxtyping import Array, Float

from ._batching import blocked_vmap
from ._family import Family, resolve_family
from ._hsgp import (
    _hsgp_domain,
    _hsgp_eigen,
    _hsgp_eigen_nd,
    _hsgp_features,
    _hsgp_features_nd,
    _penalty_diag,
    _penalty_diag_nd_ard,
    _penalty_diag_nd_iso,
)
from ._irls import safe_dmu
from ._penreml import mb_fs, mb_quantities, mb_reml_nll
from ._result import register_result

__all__ = [
    'GPResult',
    'gp_fit',
    'gp_predict',
    'gp_aic',
    'gp_bic',
]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@register_result(
    children=(
        'coef',
        'cov_unscaled',
        'theta',
        'log_mlik',
        'edf',
        'dispersion',
        'corr_rho',
    ),
    aux=(
        'kernel', 'engine', 'corr', 'n_obs', 'rank', 'n_fixed',
        'lo', 'hi', 'boundary', 'nd_meta', 'family',
    ),
)
@dataclass(frozen=True)
class GPResult:
    """Per-element Gaussian-process fit output (``gp_fit``).

    Attributes
    ----------
    coef
        ``(V, p)`` coefficients over ``[fixed | smooth]`` (``p = n_fixed + rank``):
        the unpenalised fixed effects (intercept then any ``parametric`` columns)
        followed by the ``rank`` smooth-basis coefficients ``gamma`` (HSGP
        eigenfunctions, or kernel eigenfeatures ``U Lambda^{1/2}`` for
        ``engine='exact'``).
    cov_unscaled
        ``(V, p, p)`` Bayesian covariance ``(X^T X + S_lambda)^{-1}``; multiply by
        ``dispersion`` for the scaled posterior covariance.
    theta
        ``(V, 3)`` per-element hyperparameters ``[log sigma_f^2, log sigma_e^2,
        log rho]``.  The lengthscale ``rho`` is shared, so its column is constant
        across elements.
    log_mlik
        ``(V,)`` REML log marginal likelihood at the selected ``(rho, lambda_v)``
        (up to an additive constant in ``n`` / ``n_fixed`` only).
    edf
        ``(V,)`` total effective degrees of freedom ``tr((X^T X + S_lambda)^{-1}
        X^T X)`` (fixed effects plus the smooth).
    dispersion
        ``(V,)`` residual-scale estimate ``sigma_e^2`` (the residual is
        ``sigma_e^2 R(corr_rho)`` when a ``corr`` structure is fitted).
    corr_rho
        ``(V,)`` natural residual-correlation parameter of the ``corr`` structure
        (lag-1 AR / decay / exchangeable correlation); ``0`` when ``corr='iid'``.
    kernel
        Stationary kernel name (``'matern52'`` / ``'matern32'`` / ``'matern12'`` /
        ``'rbf'``).
    engine
        Reduced-rank engine: ``'hsgp'`` (Hilbert-space eigenfunctions) or
        ``'exact'`` (kernel eigenfeatures -- full-rank when ``rank == N``, else the
        eigen-truncated Karhunen-Loeve / Nystrom approximation).
    corr
        Within-group residual-correlation structure name (``'ar1'`` / ``'car1'`` /
        ``'cs'``), or ``'iid'`` (no correlation).
    n_obs
        Number of observations ``N``.
    rank
        Smooth-basis rank ``m`` (HSGP eigenfunctions, or retained kernel
        eigenfeatures for ``engine='exact'``).
    n_fixed
        Number of unpenalised fixed-effect columns ``M_0`` (intercept + parametric).
    lo, hi, boundary
        Domain descriptors recorded for ``engine='hsgp'`` re-evaluation: the data
        range ``[lo, hi]`` and the boundary factor ``L = boundary * (hi - lo) / 2``
        (unused by ``engine='exact'``, which rebuilds the kernel from ``x_train``).
    nd_meta
        ``None`` for a 1-D fit; for a multi-dimensional fit (``X`` is ``(N, D)``)
        a hashable tuple ``(m_per, bounds, ard_rho)`` -- the per-axis ranks, the
        per-axis ``(lo, hi)``, and (for ARD) the per-axis lengthscales -- used to
        rebuild the tensor-product eigenbasis in :func:`gp_predict`.  ``theta[:, 2]``
        then carries ``log rho`` (the shared isotropic lengthscale, or the
        geometric mean of the ARD lengthscales).
    family
        Response family name: ``'gaussian'`` (default; exact REML), or
        ``'binomial'`` / ``'poisson'`` for a non-Gaussian GP whose lengthscale is
        estimated by PQL-REML (``theta``'s ``log sigma_e^2`` column is then the
        quasi-likelihood dispersion, ``1`` nominally for those families, and
        ``coef`` is on the *link* scale -- see :func:`gp_predict` ``type=``).
    """

    coef: Float[Array, 'V p']
    cov_unscaled: Float[Array, 'V p p']
    theta: Float[Array, 'V 3']
    log_mlik: Float[Array, 'V']
    edf: Float[Array, 'V']
    dispersion: Float[Array, 'V']
    corr_rho: Float[Array, 'V']
    kernel: str
    engine: str
    corr: str
    n_obs: int
    rank: int
    n_fixed: int
    lo: float
    hi: float
    boundary: float
    nd_meta: Optional[Any]
    family: str = 'gaussian'



# ---------------------------------------------------------------------------
# Exact-engine kernel eigenfeatures (rho-dependent design; host eigendecomp)
# ---------------------------------------------------------------------------


def _normalise_kernel(kernel: str) -> str:
    k = kernel.lower().replace('/', '').replace('-', '').replace('_', '')
    if k in ('rbf', 'se', 'sqexp', 'squaredexponential', 'gaussian'):
        return 'rbf'
    if k in ('matern12', 'exp', 'exponential'):
        return 'matern12'
    if k in ('matern32',):
        return 'matern32'
    if k in ('matern52',):
        return 'matern52'
    raise ValueError(
        f'gp_fit: unknown kernel {kernel!r}; expected one of '
        "'rbf'/'se', 'matern12'/'exp', 'matern32', 'matern52'."
    )


def _kernel_gram(
    r: np.ndarray, kernel: str, rho: float
) -> np.ndarray:
    """Stationary covariance ``k(r)`` on a distance array ``r`` (host numpy),
    matched to scikit-learn ``Matern(length_scale=rho)`` / ``RBF(length_scale=rho)``
    -- the same parameterisation as :func:`~nitrix.linalg.kernel.spectral_density`,
    so the exact engine and the HSGP engine share a lengthscale convention."""
    k = _normalise_kernel(kernel)
    rho = max(float(rho), 1e-12)
    if k == 'rbf':
        return np.asarray(np.exp(-0.5 * (r / rho) ** 2))
    if k == 'matern12':
        return np.asarray(np.exp(-r / rho))
    if k == 'matern32':
        a = np.sqrt(3.0) * r / rho
        return np.asarray((1.0 + a) * np.exp(-a))
    a = np.sqrt(5.0) * r / rho  # matern52
    return np.asarray((1.0 + a + a * a / 3.0) * np.exp(-a))


def _exact_eigen(
    x_np: np.ndarray, kernel: str, rho: float, rank: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Leading ``rank`` Karhunen-Loeve eigenpairs of the kernel Gram ``K_rho`` on
    ``x`` (host ``eigh`` -- data-independent of ``Y``, hence cuSOLVER-free).

    Returns ``(U_k, sqrt_lam_k)`` with ``U_k`` ``(N, rank)`` and ``sqrt_lam_k``
    ``(rank,)``: the training design is ``Phi = U_k diag(sqrt_lam_k)`` (so
    ``Phi Phi^T`` is the rank-``rank`` truncation of ``K_rho``, exact when
    ``rank == N``), and the out-of-sample feature map is ``Phi(x*) = K(x*, x)
    U_k diag(1/sqrt_lam_k)`` (Nystrom)."""
    r = np.abs(x_np[:, None] - x_np[None, :])
    K = _kernel_gram(r, kernel, rho)
    w, U = np.linalg.eigh(0.5 * (K + K.T))
    idx = np.argsort(w)[::-1][:rank]
    sqrt_lam = np.sqrt(np.clip(w[idx], 1e-12, None))
    return U[:, idx], sqrt_lam


def _exact_features_train(
    x_np: np.ndarray, kernel: str, rho: float, rank: int, dtype: Any
) -> Float[Array, 'N rank']:
    """The exact-engine training design ``Phi = U_k diag(sqrt_lam_k)`` (host)."""
    U_k, sqrt_lam = _exact_eigen(x_np, kernel, rho, rank)
    return jnp.asarray(U_k * sqrt_lam[None, :], dtype=dtype)


def _exact_features_predict(
    x_new_np: np.ndarray,
    x_train_np: np.ndarray,
    kernel: str,
    rho: float,
    rank: int,
    dtype: Any,
) -> Float[Array, 'g rank']:
    """The exact-engine out-of-sample design ``Phi(x*) = K(x*, x) U_k / sqrt_lam``
    (Nystrom), consistent with :func:`_exact_features_train`."""
    U_k, sqrt_lam = _exact_eigen(x_train_np, kernel, rho, rank)
    r = np.abs(x_new_np[:, None] - x_train_np[None, :])
    k_cross = _kernel_gram(r, kernel, rho)  # (g, N)
    phi = k_cross @ (U_k / sqrt_lam[None, :])
    return jnp.asarray(phi, dtype=dtype)


# ---------------------------------------------------------------------------
# Inner fit at a fixed (rho, lambda): the penalised-regression quantities
# ---------------------------------------------------------------------------


def _quantities(
    lam: Float[Array, ''],
    c: Float[Array, ' p'],
    g: Float[Array, ''],
    xtx: Float[Array, 'p p'],
    d: Float[Array, ' p'],
    p: int,
    ridge: float,
) -> Tuple[
    Float[Array, 'p p'],
    Float[Array, ''],
    Float[Array, ' p'],
    Float[Array, ''],
    Float[Array, ''],
    Float[Array, ''],
]:
    """At smoothing parameter ``lambda`` (penalty ``S = lambda diag(d)``), return
    ``(V, log|H|, beta, edf, rss, D_p)`` from the cross-products ``c = X^T y`` and
    ``g = y^T y``.

    ``H = X^T X + lambda diag(d) + ridge I``; ``V = H^{-1}``; ``beta = V c``;
    ``edf = tr(V X^T X)``; ``rss = ||y - X beta||^2``; ``D_p = y^T y - beta^T c``
    is the penalised residual sum of squares (``= rss + beta^T S beta``).  This is
    the ``K = 1`` case of :func:`nitrix.stats._penreml.mb_quantities`.
    """
    return mb_quantities(jnp.atleast_1d(lam), c, g, xtx, d[None, :], p, ridge)


def _fs_lambda(
    c: Float[Array, ' p'],
    g: Float[Array, ''],
    xtx: Float[Array, 'p p'],
    d: Float[Array, ' p'],
    n: int,
    p: int,
    m: int,
    n_outer: int,
    ridge: float,
    lam_floor: float,
    lam_ceil: float,
) -> Float[Array, '']:
    """Generalized Fellner-Schall selection of the single smoothing parameter
    ``lambda`` for the diagonal penalty ``diag(d)``, from cross-products.

    ``tr(S_lambda^+ S) = m / lambda`` for a disjoint diagonal penalty, so the
    update is ``lambda <- lambda (m / lambda - tr(V diag(d))) / (energy / phi)``
    with ``energy = beta^T diag(d) beta`` and ``phi = rss / (n - edf)``.  This is
    the ``K = 1`` case of :func:`nitrix.stats._penreml.mb_fs` (rank ``m``).
    """
    ranks = jnp.asarray([m], dtype=xtx.dtype)
    return mb_fs(
        c, g, xtx, d[None, :], ranks, n, p, n_outer, ridge, lam_floor, lam_ceil
    )[0]


def _reml_nll(
    d_p: Float[Array, ''],
    logdet_h: Float[Array, ''],
    lam: Float[Array, ''],
    log_pdet_pen: Float[Array, ''],
    n: int,
    m: int,
    n_fixed: int,
) -> Float[Array, '']:
    """Per-element restricted negative log-likelihood ``-2 l_R`` (full, incl. the
    ``(n, M_0)`` constant):  ``(n - M_0) log D_p + log|H| - log|S_lambda|_+ +
    (n - M_0)(log 2pi + 1 - log(n - M_0))``, with ``log|S_lambda|_+ = m log lambda
    + sum_j log(1/s_j)``.  The ``K = 1`` case of
    :func:`nitrix.stats._penreml.mb_reml_nll` (rank ``m``)."""
    return mb_reml_nll(
        d_p,
        logdet_h,
        jnp.atleast_1d(lam),
        jnp.asarray([m], dtype=jnp.asarray(lam).dtype),
        jnp.atleast_1d(log_pdet_pen),
        n,
        n_fixed,
    )


# ---------------------------------------------------------------------------
# Per-element full fit at a fixed rho (the final pass)
# ---------------------------------------------------------------------------


def _gp_fit_one(
    c: Float[Array, ' p'],
    g: Float[Array, ''],
    xtx: Float[Array, 'p p'],
    d: Float[Array, ' p'],
    log_pdet_pen: Float[Array, ''],
    n: int,
    p: int,
    m: int,
    n_fixed: int,
    n_outer: int,
    ridge: float,
    lam_floor: float,
    lam_ceil: float,
) -> Tuple[
    Float[Array, ' p'],
    Float[Array, 'p p'],
    Float[Array, ''],
    Float[Array, ''],
    Float[Array, ''],
    Float[Array, ''],
]:
    """Single-element fit at fixed ``rho`` (penalty diagonal ``d``).  Returns
    ``(beta, V, lambda, edf, dispersion, log_mlik)``."""
    lam = _fs_lambda(
        c, g, xtx, d, n, p, m, n_outer, ridge, lam_floor, lam_ceil
    )
    v, logdet_h, beta, edf, rss, d_p = _quantities(lam, c, g, xtx, d, p, ridge)
    phi = rss / jnp.clip(n - edf, 1e-3, None)
    nll = _reml_nll(d_p, logdet_h, lam, log_pdet_pen, n, m, n_fixed)
    return beta, v, lam, edf, phi, -0.5 * nll


def _pooled_nll_one(
    c: Float[Array, ' p'],
    g: Float[Array, ''],
    xtx: Float[Array, 'p p'],
    d: Float[Array, ' p'],
    log_pdet_pen: Float[Array, ''],
    n: int,
    p: int,
    m: int,
    n_fixed: int,
    n_outer: int,
    ridge: float,
    lam_floor: float,
    lam_ceil: float,
) -> Float[Array, '']:
    """The per-element ``-2 l_R`` at fixed ``rho`` (its own ``lambda_v``); summed
    over elements this is the pooled REML objective minimised over ``rho``."""
    lam = _fs_lambda(
        c, g, xtx, d, n, p, m, n_outer, ridge, lam_floor, lam_ceil
    )
    _, logdet_h, _, _, _, d_p = _quantities(lam, c, g, xtx, d, p, ridge)
    return _reml_nll(d_p, logdet_h, lam, log_pdet_pen, n, m, n_fixed)


def _pooled_nll_from_design(
    Y: Float[Array, 'V N'],
    X: Float[Array, 'N p'],
    d: Float[Array, ' p'],
    log_pdet_pen: Float[Array, ''],
    n: int,
    m: int,
    n_fixed: int,
    n_outer: int,
    ridge: float,
    lam_floor: float,
    lam_ceil: float,
    block: Optional[int],
) -> Float[Array, '']:
    """Pooled ``-2 l_R`` for a design ``X`` whose smooth columns move with ``rho``
    (the ``engine='exact'`` case: the cross-products are rebuilt per ``rho``).

    The per-element reduction goes through ``blocked_vmap`` so ``block`` bounds
    the rho-search's peak memory (``O(block * p**2)``) exactly as it bounds the
    final fit -- for ``engine='exact'`` with ``rank=N`` the per-element Hessian
    is ``(N, N)``, so an un-chunked search over all ``V`` is the OOM cliff."""
    p = X.shape[1]
    xtx = X.T @ X
    c_all = Y @ X
    g_all = jnp.sum(Y * Y, axis=1)
    per = blocked_vmap(
        lambda c_v, g_v: _pooled_nll_one(
            c_v, g_v, xtx, d, log_pdet_pen, n, p, m, n_fixed,
            n_outer, ridge, lam_floor, lam_ceil,
        ),
        (c_all, g_all),
        block=block,
    )
    return jnp.sum(per)


def _final_fit_from_design(
    Y: Float[Array, 'V N'],
    X: Float[Array, 'N p'],
    d: Float[Array, ' p'],
    log_pdet_pen: Float[Array, ''],
    n: int,
    m: int,
    n_fixed: int,
    n_outer: int,
    ridge: float,
    lam_floor: float,
    lam_ceil: float,
    block: Optional[int],
) -> Tuple[Array, Array, Array, Array, Array, Array]:
    """Per-element final fit for a design ``X``: ``(beta, V, lambda, edf,
    dispersion, log_mlik)`` via the shared penalised-REML core."""
    p = X.shape[1]
    xtx = X.T @ X
    c_all = Y @ X
    g_all = jnp.sum(Y * Y, axis=1)

    def _one(c_v: Array, g_v: Array) -> Tuple[Array, ...]:
        return _gp_fit_one(
            c_v, g_v, xtx, d, log_pdet_pen, n, p, m, n_fixed,
            n_outer, ridge, lam_floor, lam_ceil,
        )

    return cast(
        Tuple[Array, Array, Array, Array, Array, Array],
        blocked_vmap(_one, (c_all, g_all), block=block),
    )


# ---------------------------------------------------------------------------
# Per-engine (design, penalty) closures over rho -- shared by the corr= path
# ---------------------------------------------------------------------------

_RhoArg = Union[float, Float[Array, '']]  # rho as a host float or a traced scalar
_DesignFn = Callable[[float], Float[Array, 'N p']]
_PenFn = Callable[[_RhoArg], Tuple[Float[Array, ' p'], Float[Array, '']]]


def _hsgp_design_pen(
    x: Float[Array, ' N'],
    T: Float[Array, 'N q0'],
    kernel: str,
    m: int,
    n_fixed: int,
    lo: float,
    hi: float,
    boundary: float,
    dtype: Any,
) -> Tuple[_DesignFn, _PenFn]:
    """HSGP ``(design, penalty)`` closures: a fixed design, a ``rho``-dependent
    diagonal penalty ``diag(1/s(rho))``."""
    c_mid, big_l = _hsgp_domain(lo, hi, boundary)
    sqrt_lambda, phase, inv_sqrt_L = _hsgp_eigen(m, c_mid, big_l, dtype)
    X = jnp.concatenate(
        [T, _hsgp_features(x, sqrt_lambda, phase, inv_sqrt_L)], axis=1
    )

    def design(_rho: float) -> Float[Array, 'N p']:
        return X

    def pen(rho: _RhoArg) -> Tuple[Array, Array]:
        return _penalty_diag(
            sqrt_lambda, kernel, jnp.asarray(rho, dtype=dtype), n_fixed
        )

    return design, pen


def _exact_design_pen(
    x_np: np.ndarray,
    T: Float[Array, 'N q0'],
    kernel: str,
    m: int,
    n_fixed: int,
    dtype: Any,
) -> Tuple[_DesignFn, _PenFn]:
    """Exact ``(design, penalty)`` closures: a ``rho``-dependent kernel-
    eigenfeature design, a fixed identity penalty (unit spectral weights)."""
    d_unit = jnp.concatenate(
        [jnp.zeros((n_fixed,), dtype), jnp.ones((m,), dtype)]
    )
    log_pdet_unit = jnp.asarray(0.0, dtype)

    def design(rho: float) -> Float[Array, 'N p']:
        return jnp.concatenate(
            [T, _exact_features_train(x_np, kernel, rho, m, dtype)], axis=1
        )

    def pen(_rho: _RhoArg) -> Tuple[Array, Array]:
        return d_unit, log_pdet_unit

    return design, pen


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def gp_fit(
    Y: Float[Array, 'V N'],
    x: Union[Float[Array, ' N'], Float[Array, 'N D']],
    *,
    parametric: Optional[Float[Array, 'N q']] = None,
    kernel: str = 'matern52',
    rank: Optional[Any] = None,
    boundary: float = 1.5,
    bounds: Optional[Tuple[float, float]] = None,
    engine: Literal['hsgp', 'exact'] = 'hsgp',
    select: Literal['shared-rho'] = 'shared-rho',
    ard: bool = False,
    rho_bounds: Optional[Tuple[float, float]] = None,
    n_rho: int = 24,
    map_rho: Optional[Callable[[Float[Array, '']], Float[Array, '']]] = None,
    corr: Optional[Any] = None,
    group: Optional[Any] = None,
    time: Optional[Float[Array, ' N']] = None,
    n_corr: int = 9,
    corr_raw_bounds: Tuple[float, float] = (-4.0, 4.0),
    family: Union[str, Family] = 'gaussian',
    prior_weights: Optional[Float[Array, ' N']] = None,
    n_pql: int = 8,
    n_outer: int = 30,
    n_search: int = 15,
    ridge: float = 1e-8,
    lam_floor: float = 1e-6,
    lam_ceil: float = 1e8,
    block: Optional[int] = None,
) -> GPResult:
    """Fit a mass-univariate Gaussian-process regression with REML-estimated
    lengthscale (Gaussian responses).

    Parameters
    ----------
    Y
        ``(V, N)`` responses (one row per element).
    x
        ``(N,)`` covariate, or ``(N, D)`` for a **multi-dimensional** GP (a spatial
        smooth / smooth interaction; HSGP tensor-product engine).  This is a single
        covariate the GP smooths *over* -- **not** the full design matrix ``X`` of
        ``glm_fit`` / ``gam_fit``; linear covariates go to ``parametric=``.
    parametric
        Optional ``(N, q)`` unpenalised linear design (covariates entering
        linearly alongside the intercept).
    kernel
        Stationary kernel: ``'matern52'`` (default) / ``'matern32'`` /
        ``'matern12'`` / ``'rbf'``.
    rank
        Smooth-basis rank ``m``.  ``None`` (default) uses an engine-appropriate
        value: ``20`` for 1-D ``'hsgp'`` (small ``rho`` needs a larger ``rank``),
        ``N`` for ``'exact'`` (full-rank), and ``8`` **per axis** for a
        multi-dimensional fit.  An explicit ``rank < N`` with ``engine='exact'``
        gives the eigen-truncated Karhunen-Loeve / Nystrom approximation; for
        multi-D, ``rank`` may be a per-axis sequence ``[m_1, ..., m_D]``.
    ard
        Multi-D only: ``False`` (default) estimates one **isotropic** lengthscale
        (a radial kernel of ``||x - x'||``); ``True`` estimates a **per-axis**
        lengthscale (ARD / separable kernel) by coordinate descent over the axes.
    boundary
        Domain-extension factor ``L / half-range`` (``>= 1``; default ``1.5``;
        ``'hsgp'`` only).
    bounds
        ``(lo, hi)`` data-range override (defaults to the data min/max).
    engine
        Reduced-rank engine.  ``'hsgp'`` (default; Hilbert-space eigenfunctions, a
        fixed design with an ``eigh``-free ``rho``-dependent diagonal penalty) or
        ``'exact'`` (kernel eigenfeatures ``U Lambda^{1/2}`` -- the full-rank GP
        when ``rank == N``; a one-off host ``eigh`` of the shared kernel Gram per
        ``rho``, cuSOLVER-free, equivalent to ``lme.reml_fit`` by the
        penalty<->variance-component identity).
    select
        Lengthscale-selection mode.  ``'shared-rho'``: one ``rho`` across all
        elements, per-element amplitude and noise.
    rho_bounds
        ``(rho_lo, rho_hi)`` search range.  Defaults to ``(0.05, 2.0) * (hi - lo)``.
    n_rho
        Number of log-spaced grid points for the ``rho`` search.
    map_rho
        Optional callable ``rho -> -log p(rho)`` adding a lengthscale prior to the
        pooled objective (a MAP / prior-regularised ``rho``); ``None`` is pure
        REML.  Use a builder from :mod:`nitrix.stats.priors`
        (``halfnormal_prior`` / ``invgamma_prior`` / ``lognormal_prior``) or any
        pure-JAX callable.
    corr
        Within-group residual-correlation structure: ``'ar1'`` (discrete AR(1)),
        ``'car1'`` (continuous-time AR(1); pass ``time``), ``'cs'`` (compound
        symmetry), or a ``lme.CorrSpec``.  ``None`` (default) is the i.i.d.
        residual.  When set, ``group`` is required; the residual is
        ``sigma_e^2 R(rho_c)`` block-diagonal across ``group``, with ``rho_c``
        estimated jointly with the lengthscale.
    group
        ``(N,)`` integer grouping factor for ``corr`` (the residual is correlated
        *within* groups, independent across them).  Required when ``corr`` is set.
    time
        ``(N,)`` observation times for ``corr='car1'`` (and to order ``ar1`` when
        rows are not in within-group time order).
    n_corr
        Number of grid points for the residual-correlation parameter search.
    corr_raw_bounds
        ``(lo, hi)`` range of the structure's *unconstrained* grid parameter
        (MC5).  ``ar1`` maps it by ``tanh`` -- the default ``(-4, 4)`` spans
        ``rho_c in (-0.999, 0.999)``.  ``car1`` / ``cs`` map it by ``sigmoid``,
        a **one-sided** window ``rho_c in (0.018, 0.982)`` (``cs`` is positive by
        construction, the common exchangeable regime; ``car1`` is a positive
        decay).  The raw grid is not parabolically refined, so widen the bounds /
        raise ``n_corr`` for a finer estimate; an estimate at the grid edge warns.
    n_outer, n_search
        Fellner-Schall iterations for the final fit and for each ``rho``-search
        evaluation, respectively.
    ridge
        Small stabiliser on the penalised normal equations.
    lam_floor, lam_ceil
        Clamp on the smoothing parameter ``lambda = sigma_e^2 / sigma_f^2``.
    block
        Optional element-block size bounding peak memory on brain-scale ``V``.

    Returns
    -------
    ``GPResult`` (coefficients, Bayesian covariance, per-element ``[log sigma_f^2,
    log sigma_e^2, log rho]``, REML log marginal likelihood, EDF, dispersion).
    """
    if engine not in ('hsgp', 'exact'):
        raise NotImplementedError(
            f"gp_fit: engine={engine!r} -- expected 'hsgp' or 'exact'."
        )
    if select != 'shared-rho':
        raise NotImplementedError(
            f"gp_fit: select={select!r} -- only 'shared-rho' is implemented "
            '(per-element lengthscales are a later add).'
        )
    if not boundary >= 1.0:
        raise ValueError(f'gp_fit: boundary={boundary} must be >= 1.0.')

    fam = resolve_family(family)
    non_gaussian = fam.name != 'gaussian'
    if non_gaussian:
        if fam.name not in ('binomial', 'poisson'):
            raise NotImplementedError(
                f'gp_fit: family={fam.name!r} -- non-Gaussian lengthscale '
                "estimation currently supports 'binomial' and 'poisson' "
                '(Phase 1, PQL-REML); use a fixed rho via hsgp_basis(...) + '
                'gam_fit for other families.'
            )
        if engine != 'hsgp':
            raise NotImplementedError(
                "gp_fit: a non-Gaussian family needs engine='hsgp' (Phase 1)."
            )
        if corr is not None:
            raise NotImplementedError(
                'gp_fit: corr= is not supported with a non-Gaussian family '
                '(Phase 1).'
            )

    Y = jnp.asarray(Y)
    if not jnp.issubdtype(Y.dtype, jnp.floating):
        Y = Y.astype(float)  # ER6: integer Y would coerce the whole fit to int
    x = jnp.asarray(x, dtype=Y.dtype)
    n = Y.shape[-1]
    if x.shape[0] != n:
        raise ValueError(
            f'gp_fit: x has {x.shape[0]} points; expected N={n} to match Y.'
        )
    if x.ndim == 2 and x.shape[1] == 1:
        x = x[:, 0]  # an (N, 1) covariate is the 1-D case

    # --- multi-dimensional (tensor-product) HSGP: X is (N, D), D >= 2 --------
    if x.ndim == 2:
        if non_gaussian:
            raise NotImplementedError(
                'gp_fit: a non-Gaussian family is 1-D only (Phase 1); '
                'multi-dimensional non-Gaussian GP is a Phase-2 add.'
            )
        if engine != 'hsgp':
            raise NotImplementedError(
                "gp_fit: multi-dimensional X needs engine='hsgp' (the exact "
                'engine is 1-D only).'
            )
        if corr is not None:
            raise NotImplementedError(
                'gp_fit: corr= is not supported with multi-dimensional X.'
            )
        d_in = x.shape[1]
        if rank is None:
            m_per = (8,) * d_in
        elif isinstance(rank, (int, np.integer)):
            m_per = (int(rank),) * d_in
        else:
            m_per = tuple(int(v) for v in rank)
            if len(m_per) != d_in:
                raise ValueError(
                    f'gp_fit: rank has {len(m_per)} entries; expected D={d_in}.'
                )
        if any(mm < 1 for mm in m_per):
            raise ValueError('gp_fit: every per-axis rank must be >= 1.')
        fixed_nd = [jnp.ones((n, 1), dtype=Y.dtype)]
        if parametric is not None:
            fixed_nd.append(jnp.asarray(parametric, dtype=Y.dtype))
        n_fixed_nd = sum(b.shape[1] for b in fixed_nd)
        return _gp_fit_nd(
            Y, np.asarray(x, dtype=np.float64),
            jnp.concatenate(fixed_nd, axis=1), n_fixed_nd, kernel, m_per,
            boundary, ard, rho_bounds, n_rho, map_rho,
            n_outer, n_search, ridge, lam_floor, lam_ceil, block,
        )

    # Engine-appropriate default rank: 20 (hsgp) or N (exact, full-rank).
    m = (20 if engine == 'hsgp' else n) if rank is None else int(rank)
    if m < 1:
        raise ValueError(f'gp_fit: rank={rank} must be >= 1.')
    if engine == 'exact' and m > n:
        m = n  # at most N kernel eigenfeatures

    x_np = np.asarray(x, dtype=np.float64)
    lo = float(np.min(x_np)) if bounds is None else float(bounds[0])
    hi = float(np.max(x_np)) if bounds is None else float(bounds[1])

    # Shared unpenalised design T = [intercept | parametric].
    fixed_blocks = [jnp.ones((n, 1), dtype=Y.dtype)]
    if parametric is not None:
        fixed_blocks.append(jnp.asarray(parametric, dtype=Y.dtype))
    n_fixed = sum(b.shape[1] for b in fixed_blocks)
    T = jnp.concatenate(fixed_blocks, axis=1)  # (N, n_fixed)

    # --- rho search grid (shared by both engines) ---------------------------
    span = max(hi - lo, 1e-6)
    if rho_bounds is None:
        rho_lo, rho_hi = 0.05 * span, 2.0 * span
    else:
        rho_lo, rho_hi = float(rho_bounds[0]), float(rho_bounds[1])
    log_rho_grid_np = np.linspace(
        np.log(rho_lo), np.log(rho_hi), int(n_rho)
    )

    # --- corr= composition: GP smooth + structured within-group residual -----
    if corr is not None:
        if group is None:
            raise ValueError("gp_fit: corr= requires a `group` factor.")
        from .lme._corr import resolve_corr

        corr_spec = resolve_corr(corr)
        if time is None and corr_spec.name in ('ar1', 'car1'):
            # MC6: ar1/car1 pair consecutive within-group rows; without `time`
            # that assumes the rows are already in within-group time order.
            warnings.warn(
                f"gp_fit: corr='{corr_spec.name}' with time=None assumes each "
                "group's rows are already in within-group time order; pass "
                'time= if they are not.',
                stacklevel=2,
            )
        raw_grid_np = np.linspace(
            float(corr_raw_bounds[0]), float(corr_raw_bounds[1]), int(n_corr)
        )
        if engine == 'hsgp':
            _design, _pen = _hsgp_design_pen(
                x, T, kernel, m, n_fixed, lo, hi, boundary, Y.dtype
            )
        else:  # exact
            _design, _pen = _exact_design_pen(
                x_np, T, kernel, m, n_fixed, Y.dtype
            )

        return _gp_fit_corr(
            Y, _design, _pen, group, time, corr_spec, n, m, n_fixed,
            log_rho_grid_np, raw_grid_np, map_rho, kernel, engine,
            n_outer, n_search, ridge, lam_floor, lam_ceil, block,
            lo, hi, boundary,
        )

    if engine == 'exact':
        return _gp_fit_exact(
            Y, x_np, T, n_fixed, kernel, m, log_rho_grid_np, map_rho,
            n_outer, n_search, ridge, lam_floor, lam_ceil, block,
            lo, hi, boundary,
        )

    # ==================== engine == 'hsgp' ==================================
    if non_gaussian:
        return _gp_fit_glm_hsgp(
            Y, x, T, n_fixed, fam, kernel, m, log_rho_grid_np, map_rho,
            n_pql, n_outer, n_search, ridge, lam_floor, lam_ceil, block,
            lo, hi, boundary, prior_weights,
        )

    # Fixed design + rho-dependent diagonal-penalty closures (DS2: the shared
    # construction, also used by the corr / exact paths). The HSGP design is
    # rho-independent, so it is evaluated once and the cross-products reused.
    design_fn, pen_fn = _hsgp_design_pen(
        x, T, kernel, m, n_fixed, lo, hi, boundary, Y.dtype
    )
    X = design_fn(0.0)  # (N, p) -- fixed; rho enters only the penalty
    p = X.shape[1]

    xtx = X.T @ X
    c_all = Y @ X  # (V, p) -- row v is X^T y_v
    g_all = jnp.sum(Y * Y, axis=1)  # (V,)

    log_rho_grid = jnp.asarray(log_rho_grid_np, dtype=Y.dtype)

    def _pooled_nll(log_rho: Float[Array, '']) -> Float[Array, '']:
        rho = jnp.exp(log_rho)
        d, log_pdet_pen = pen_fn(rho)
        per = blocked_vmap(
            lambda c_v, g_v: _pooled_nll_one(
                c_v, g_v, xtx, d, log_pdet_pen, n, p, m, n_fixed,
                n_search, ridge, lam_floor, lam_ceil,
            ),
            (c_all, g_all),
            block=block,
        )
        nll = jnp.sum(per)
        if map_rho is not None:
            nll = nll + 2.0 * map_rho(rho)
        return nll

    nll_grid = lax.map(_pooled_nll, log_rho_grid)  # (n_rho,)
    log_rho_hat = _parabolic_argmin(
        np.asarray(log_rho_grid), np.asarray(nll_grid)
    )
    rho_hat = float(np.exp(log_rho_hat))

    # --- final per-element fit at rho_hat -----------------------------------
    d_hat, log_pdet_hat = pen_fn(rho_hat)

    def _final_one(
        c_v: Float[Array, ' p'], g_v: Float[Array, '']
    ) -> Tuple[
        Float[Array, ' p'],
        Float[Array, 'p p'],
        Float[Array, ''],
        Float[Array, ''],
        Float[Array, ''],
        Float[Array, ''],
    ]:
        return _gp_fit_one(
            c_v, g_v, xtx, d_hat, log_pdet_hat, n, p, m, n_fixed,
            n_outer, ridge, lam_floor, lam_ceil,
        )

    beta, v, lam, edf, phi, log_mlik = blocked_vmap(
        _final_one, (c_all, g_all), block=block
    )
    return _assemble_gp_result(
        beta, v, lam, edf, phi, log_mlik, rho_hat, kernel, 'hsgp',
        n, m, n_fixed, lo, hi, boundary,
    )


def _assemble_gp_result(
    beta: Array,
    v: Array,
    lam: Array,
    edf: Array,
    phi: Array,
    log_mlik: Array,
    rho_hat: float,
    kernel: str,
    engine: str,
    n: int,
    m: int,
    n_fixed: int,
    lo: float,
    hi: float,
    boundary: float,
    corr_name: str = 'iid',
    corr_rho: Optional[Array] = None,
    nd_meta: Optional[Any] = None,
    family: str = 'gaussian',
) -> GPResult:
    """Pack the per-element fit arrays into a :class:`GPResult` (shared by both
    engines): ``theta = [log sigma_f^2, log sigma_e^2, log rho]`` with
    ``sigma_f^2 = sigma_e^2 / lambda`` and the shared ``rho`` broadcast."""
    sigma_e2 = phi
    sigma_f2 = phi / jnp.clip(lam, 1e-30, None)
    log_rho_col = jnp.full_like(sigma_e2, np.log(rho_hat))
    theta = jnp.stack(
        [jnp.log(jnp.clip(sigma_f2, 1e-30, None)),
         jnp.log(jnp.clip(sigma_e2, 1e-30, None)),
         log_rho_col],
        axis=-1,
    )
    if corr_rho is None:
        corr_rho = jnp.zeros_like(sigma_e2)
    return GPResult(
        coef=beta,
        cov_unscaled=v,
        theta=theta,
        log_mlik=log_mlik,
        edf=edf,
        dispersion=sigma_e2,
        corr_rho=corr_rho,
        kernel=kernel,
        engine=engine,
        corr=corr_name,
        n_obs=int(n),
        rank=int(m),
        n_fixed=int(n_fixed),
        lo=lo,
        hi=hi,
        boundary=float(boundary),
        nd_meta=nd_meta,
        family=family,
    )


def _gp_fit_glm_hsgp(
    Y: Float[Array, 'V N'],
    x: Float[Array, ' N'],
    T: Float[Array, 'N q0'],
    n_fixed: int,
    family: Family,
    kernel: str,
    m: int,
    log_rho_grid_np: np.ndarray,
    map_rho: Optional[Callable[[Float[Array, '']], Float[Array, '']]],
    n_pql: int,
    n_outer: int,
    n_search: int,
    ridge: float,
    lam_floor: float,
    lam_ceil: float,
    block: Optional[int],
    lo: float,
    hi: float,
    boundary: float,
    prior_weights: Optional[Float[Array, ' N']],
) -> GPResult:
    """Non-Gaussian GP (HSGP engine) by PQL-REML lengthscale estimation (CV2).

    Each PQL iteration relinearises the GLM to a per-element *weighted* Gaussian
    working problem ``(z_v, W_v)`` (the IRLS working response / weights), then
    profiles the shared ``rho`` and selects ``lambda`` with the **same** pooled-REML
    core as the Gaussian path -- only the cross-products become ``X^T W_v X`` /
    ``X^T W_v z_v``, computed per element *inside* ``blocked_vmap`` so ``block``
    bounds peak memory (no full ``(V, p, p)`` working Gram).  The working weights
    are held fixed within each inner ``rho`` search (their Jacobian is constant in
    ``rho``), so the search is exactly the Gaussian profile REML; the outer loop
    relinearises until the linear predictor settles.

    This is **PQL** (Breslow-Clayton): biased for binary / low-count data -- the
    documented Phase-1 caveat (cf. ``glmm_fit``); the proper Laplace REML / LAML is
    the Phase-2 upgrade.  ``coef`` is on the link scale; the reported ``dispersion``
    is the working quasi-likelihood scale (``~1`` for a calibrated binomial /
    poisson fit)."""
    dtype = Y.dtype
    n = Y.shape[-1]
    c_mid, big_l = _hsgp_domain(lo, hi, boundary)
    sqrt_lambda, phase, inv_sqrt_L = _hsgp_eigen(m, c_mid, big_l, dtype)
    phi_design = _hsgp_features(x, sqrt_lambda, phase, inv_sqrt_L)  # (N, m)
    X = jnp.concatenate([T, phi_design], axis=1)  # (N, p)
    p = X.shape[1]
    pw = (
        jnp.ones((n,), dtype) if prior_weights is None
        else jnp.asarray(prior_weights, dtype)
    )

    def _working_xprod(
        eta_v: Float[Array, ' N'], y_v: Float[Array, ' N']
    ) -> Tuple[Array, Array, Array]:
        """One element's working cross-products ``(X^T W X, X^T W z, z^T W z)``
        (IRLS working weights ``W`` / response ``z``, as in ``_irls._working``)."""
        eta_c = family.clip_eta(eta_v)
        mu = family.linkinv(eta_c)
        dmu = family.mu_eta(eta_c)
        wts = dmu * dmu / jnp.clip(family.variance(mu), 1e-10, None) * pw
        z = eta_c + (y_v - mu) / safe_dmu(dmu)
        xw = X * wts[:, None]
        return xw.T @ X, xw.T @ z, jnp.sum(wts * z * z)

    @jax.jit
    def _pooled(eta_all: Array, d: Array, log_pdet: Array) -> Array:
        def one(eta_v: Array, y_v: Array) -> Array:
            xtwx, c, g = _working_xprod(eta_v, y_v)
            return _pooled_nll_one(
                c, g, xtwx, d, log_pdet, n, p, m, n_fixed,
                n_search, ridge, lam_floor, lam_ceil,
            )

        return jnp.sum(blocked_vmap(one, (eta_all, Y), block=block))

    @jax.jit
    def _final(
        eta_all: Array, d: Array, log_pdet: Array
    ) -> Tuple[Array, Array, Array, Array, Array, Array]:
        def one(eta_v: Array, y_v: Array) -> Tuple[Array, ...]:
            xtwx, c, g = _working_xprod(eta_v, y_v)
            return _gp_fit_one(
                c, g, xtwx, d, log_pdet, n, p, m, n_fixed,
                n_outer, ridge, lam_floor, lam_ceil,
            )

        return cast(
            Tuple[Array, Array, Array, Array, Array, Array],
            blocked_vmap(one, (eta_all, Y), block=block),
        )

    # PQL outer loop: relinearise until the linear predictor settles.
    eta = family.clip_eta(family.link(family.init_mu(Y)))  # (V, N)
    rho_hat = float(np.exp(float(np.mean(log_rho_grid_np))))
    fit: Optional[Tuple[Array, Array, Array, Array, Array, Array]] = None
    for _pql in range(max(int(n_pql), 1)):
        nll_grid = []
        for log_rho in log_rho_grid_np:
            rho = float(np.exp(log_rho))
            d, log_pdet = _penalty_diag(
                sqrt_lambda, kernel, jnp.asarray(rho, dtype), n_fixed
            )
            val = float(_pooled(eta, d, log_pdet))
            if map_rho is not None:
                val = val + 2.0 * float(map_rho(jnp.asarray(rho, dtype)))
            nll_grid.append(val)
        log_rho_hat = _parabolic_argmin(log_rho_grid_np, np.asarray(nll_grid))
        rho_hat = float(np.exp(log_rho_hat))
        d_hat, log_pdet_hat = _penalty_diag(
            sqrt_lambda, kernel, jnp.asarray(rho_hat, dtype), n_fixed
        )
        fit = _final(eta, d_hat, log_pdet_hat)
        eta = family.clip_eta(fit[0] @ X.T)  # (V, N) updated linear predictor

    assert fit is not None
    beta, v, lam, edf, phi, log_mlik = fit
    return _assemble_gp_result(
        beta, v, lam, edf, phi, log_mlik, rho_hat, kernel, 'hsgp',
        n, m, n_fixed, lo, hi, boundary, family=family.name,
    )


def _gp_fit_exact(
    Y: Float[Array, 'V N'],
    x_np: np.ndarray,
    T: Float[Array, 'N q0'],
    n_fixed: int,
    kernel: str,
    m: int,
    log_rho_grid_np: np.ndarray,
    map_rho: Optional[Callable[[Float[Array, '']], Float[Array, '']]],
    n_outer: int,
    n_search: int,
    ridge: float,
    lam_floor: float,
    lam_ceil: float,
    block: Optional[int],
    lo: float,
    hi: float,
    boundary: float,
) -> GPResult:
    """``engine='exact'``: full-rank (or KL-truncated) GP via the kernel
    eigenfeature design ``Phi(rho) = U Lambda^{1/2}``.

    The kernel Gram eigendecomposition is data-independent of ``Y``, so it is done
    on the host (numpy ``eigh``) once per grid ``rho`` -- cuSOLVER-free, shared
    across all elements.  ``Phi(rho)`` *moves* with ``rho`` (unlike HSGP's fixed
    eigenbasis), so the penalty is the plain identity (unit spectral weights) and
    the cross-products are rebuilt per ``rho``.  Everything else (the
    Fellner-Schall ``lambda``, the REML criterion, the pooled ``rho`` search) is
    the shared PR2 penalised-REML core -- equivalent to ``lme.reml_fit`` on
    ``Z = chol(K_rho)`` by the penalty<->variance-component identity."""
    dtype = Y.dtype
    n = Y.shape[-1]
    # DS2: shared exact (design, penalty) closures (also used by the corr path) --
    # a rho-dependent kernel-eigenfeature design and a fixed unit-weight penalty.
    _design, _pen = _exact_design_pen(x_np, T, kernel, m, n_fixed, dtype)
    d, log_pdet_pen = _pen(0.0)

    # Compile the pooled REML once; the moving design enters as a traced arg
    # (the host kernel eigh per rho stays outside the compiled region).
    @jax.jit
    def _nll_jit(X: Array) -> Array:
        return _pooled_nll_from_design(
            Y, X, d, log_pdet_pen, n, m, n_fixed,
            n_search, ridge, lam_floor, lam_ceil, block,
        )

    # --- rho search: host loop (each rho needs a host kernel eigh) ----------
    nll_grid = []
    for log_rho in log_rho_grid_np:
        rho = float(np.exp(log_rho))
        nll = float(_nll_jit(_design(rho)))
        if map_rho is not None:
            nll = nll + 2.0 * float(map_rho(jnp.asarray(rho, dtype=dtype)))
        nll_grid.append(nll)

    log_rho_hat = _parabolic_argmin(
        log_rho_grid_np, np.asarray(nll_grid)
    )
    rho_hat = float(np.exp(log_rho_hat))

    # --- final per-element fit at rho_hat -----------------------------------
    beta, v, lam, edf, phi, log_mlik = _final_fit_from_design(
        Y, _design(rho_hat), d, log_pdet_pen, n, m, n_fixed,
        n_outer, ridge, lam_floor, lam_ceil, block,
    )
    return _assemble_gp_result(
        beta, v, lam, edf, phi, log_mlik, rho_hat, kernel, 'exact',
        n, m, n_fixed, lo, hi, boundary,
    )


def _gp_fit_nd(
    Y: Float[Array, 'V N'],
    x_np: np.ndarray,
    T: Float[Array, 'N q0'],
    n_fixed: int,
    kernel: str,
    m_per: Tuple[int, ...],
    boundary: float,
    ard: bool,
    rho_bounds: Optional[Tuple[float, float]],
    n_rho: int,
    map_rho: Optional[Callable[[Float[Array, '']], Float[Array, '']]],
    n_outer: int,
    n_search: int,
    ridge: float,
    lam_floor: float,
    lam_ceil: float,
    block: Optional[int],
) -> GPResult:
    """Multi-dimensional HSGP fit (tensor-product eigenbasis).  Isotropic (one
    shared ``rho``, a 1-D grid) or ARD (per-axis ``rho``, coordinate descent over
    the axes); the eigenbasis is ``rho``-independent so the pooled-REML core is
    reused with the diagonal penalty as the only moving part."""
    dtype = Y.dtype
    d_in = x_np.shape[1]
    n = Y.shape[-1]
    freqs, phase, inv_sqrt_L, omega_norm, bounds = _hsgp_eigen_nd(
        x_np, m_per, boundary, dtype
    )
    phi = _hsgp_features_nd(jnp.asarray(x_np, dtype=dtype), freqs, phase, inv_sqrt_L)
    X = jnp.concatenate([T, phi], axis=1)
    p = X.shape[1]
    m = phi.shape[1]
    xtx = X.T @ X
    c_all = Y @ X
    g_all = jnp.sum(Y * Y, axis=1)

    @jax.jit
    def _pooled(d: Array, log_pdet: Array) -> Array:
        per = blocked_vmap(
            lambda c_v, g_v: _pooled_nll_one(
                c_v, g_v, xtx, d, log_pdet, n, p, m, n_fixed,
                n_search, ridge, lam_floor, lam_ceil,
            ),
            (c_all, g_all),
            block=block,
        )
        return jnp.sum(per)

    half = [0.5 * (bd[1] - bd[0]) for bd in bounds]

    def _grid(base: float) -> np.ndarray:
        lo_b, hi_b = (
            (0.05 * base, 2.0 * base) if rho_bounds is None
            else (float(rho_bounds[0]), float(rho_bounds[1]))
        )
        return np.linspace(np.log(lo_b), np.log(hi_b), int(n_rho))

    if not ard:
        # --- isotropic: 1-D grid over the shared rho -----------------------
        grid = _grid(float(np.mean(half)))
        nll_grid = []
        for lr in grid:
            rho = float(np.exp(lr))
            d, lpp = _penalty_diag_nd_iso(omega_norm, kernel, rho, d_in, n_fixed)
            nll = float(_pooled(d, lpp))
            if map_rho is not None:
                nll = nll + 2.0 * float(map_rho(jnp.asarray(rho, dtype=dtype)))
            nll_grid.append(nll)
        rho_hat = float(np.exp(_parabolic_argmin(grid, np.asarray(nll_grid))))
        d_hat, lpp_hat = _penalty_diag_nd_iso(
            omega_norm, kernel, rho_hat, d_in, n_fixed
        )
        ard_rho: Optional[Tuple[float, ...]] = None
        rho_report = rho_hat
    else:
        # --- ARD: coordinate descent over the per-axis lengthscales --------
        rho_vec = list(half)
        for _cycle in range(3):
            for axis in range(d_in):
                grid = _grid(half[axis])
                nll_axis = []
                for lr in grid:
                    trial = list(rho_vec)
                    trial[axis] = float(np.exp(lr))
                    d, lpp = _penalty_diag_nd_ard(
                        freqs, kernel, tuple(trial), n_fixed
                    )
                    nll = float(_pooled(d, lpp))
                    if map_rho is not None:
                        nll = nll + 2.0 * float(
                            map_rho(jnp.asarray(trial[axis], dtype=dtype))
                        )
                    nll_axis.append(nll)
                rho_vec[axis] = float(
                    np.exp(_parabolic_argmin(grid, np.asarray(nll_axis)))
                )
        ard_rho = tuple(rho_vec)
        d_hat, lpp_hat = _penalty_diag_nd_ard(freqs, kernel, ard_rho, n_fixed)
        rho_report = float(np.exp(np.mean(np.log(rho_vec))))  # geometric mean

    def _one(c_v: Array, g_v: Array) -> Tuple[Array, ...]:
        return _gp_fit_one(
            c_v, g_v, xtx, d_hat, lpp_hat, n, p, m, n_fixed,
            n_outer, ridge, lam_floor, lam_ceil,
        )

    beta, v, lam, edf, disp, log_mlik = cast(
        Tuple[Array, Array, Array, Array, Array, Array],
        blocked_vmap(_one, (c_all, g_all), block=block),
    )
    nd_meta = (tuple(m_per), bounds, ard_rho)
    return _assemble_gp_result(
        beta, v, lam, edf, disp, log_mlik, rho_report, kernel, 'hsgp',
        n, m, n_fixed, bounds[0][0], bounds[0][1], boundary,
        nd_meta=nd_meta,
    )


def _gp_fit_corr(
    Y: Float[Array, 'V N'],
    design_fn: Callable[[float], Float[Array, 'N p']],
    pen_fn: Callable[[float], Tuple[Float[Array, ' p'], Float[Array, '']]],
    group: Any,
    time: Any,
    corr_spec: Any,
    n: int,
    m: int,
    n_fixed: int,
    log_rho_grid_np: np.ndarray,
    raw_grid_np: np.ndarray,
    map_rho: Optional[Callable[[Float[Array, '']], Float[Array, '']]],
    kernel: str,
    engine: str,
    n_outer: int,
    n_search: int,
    ridge: float,
    lam_floor: float,
    lam_ceil: float,
    block: Optional[int],
    lo: float,
    hi: float,
    boundary: float,
) -> GPResult:
    """GP smooth + a structured within-group residual ``Cov(eps) = sigma_e^2
    R(rho_c)`` (the ``corr=`` composition).

    Whitening by ``R(rho_c)`` (``W R W^T = I`` per group; reused verbatim from
    ``lme._corr``) turns the model into the PR2 penalised regression on whitened
    data, so the criterion is the shared profiled REML **plus** the whitening
    Jacobian ``log|R(rho_c)|``.  ``rho_c`` joins the lengthscale in a joint
    ``(rho_GP, raw_c)`` grid (``raw_c`` the structure's unconstrained parameter);
    the final fit's posterior is in whitened space, so ``gp_predict`` (the latent
    GP mean/variance) is unchanged."""
    from .lme._corrfit import build_group_layout

    dtype = Y.dtype
    v_count = Y.shape[0]
    layout = build_group_layout(jnp.asarray(group), time)
    idx, gaps, nsize, mask = (
        layout.idx, layout.gaps, layout.nsize, layout.mask
    )
    mask_f = mask.astype(dtype)
    # Y padded as channels (G, T, V) so one whitener whitens all elements.
    y_pad = jnp.transpose(Y[:, idx], (1, 2, 0)) * mask_f[..., None]

    def _whiten(stack: Array, raw: Array) -> Tuple[Array, Array]:
        return cast(
            Tuple[Array, Array],
            corr_spec.whiten(stack, gaps, nsize, mask, raw),
        )

    def _cross(X: Array, raw: Array) -> Tuple[Array, Array, Array, Array]:
        x_pad = X[idx] * mask_f[..., None]  # (G, T, p)
        x_tilde, half_logdet = _whiten(x_pad, raw)
        y_tilde, _ = _whiten(y_pad, raw)
        xtx = jnp.einsum('gtp,gtq->pq', x_tilde, x_tilde)
        c_all = jnp.einsum('gtp,gtv->vp', x_tilde, y_tilde)
        g_all = jnp.einsum('gtv,gtv->v', y_tilde, y_tilde)
        return xtx, c_all, g_all, half_logdet

    # The whitened cross-products and the pooled REML are compiled **once** and
    # reused across every grid cell (the moving design / penalty / corr parameter
    # enter as traced array arguments) -- a Python loop that re-traced per cell
    # would recompile O(n_rho * n_corr) programs and exhaust the compiler.
    @jax.jit
    def _nll_jit(X: Array, raw: Array, d: Array, log_pdet: Array) -> Array:
        xtx, c_all, g_all, half_logdet = _cross(X, raw)
        p = xtx.shape[0]
        per = blocked_vmap(
            lambda c_v, g_v: _pooled_nll_one(
                c_v, g_v, xtx, d, log_pdet, n, p, m, n_fixed,
                n_search, ridge, lam_floor, lam_ceil,
            ),
            (c_all, g_all),
            block=block,
        )
        return jnp.sum(per) + v_count * 2.0 * half_logdet

    cross_jit = jax.jit(_cross)

    # --- joint (raw_c, rho_GP) grid search ----------------------------------
    grid = np.empty((len(raw_grid_np), len(log_rho_grid_np)))
    for i, raw_v in enumerate(raw_grid_np):
        raw = jnp.asarray([raw_v], dtype=dtype)
        for j, log_rho in enumerate(log_rho_grid_np):
            rho = float(np.exp(log_rho))
            d, log_pdet_pen = pen_fn(rho)
            nll = float(_nll_jit(design_fn(rho), raw, d, log_pdet_pen))
            if map_rho is not None:
                nll = nll + 2.0 * float(map_rho(jnp.asarray(rho, dtype=dtype)))
            grid[i, j] = nll

    i_star = int(np.argmin(grid.min(axis=1)))
    raw_hat = float(raw_grid_np[i_star])
    if i_star in (0, len(raw_grid_np) - 1):
        # MC5: the corr parameter is grid-quantised; a boundary argmin means the
        # estimate is clamped at the search edge (the true value may be stronger).
        rho_c_edge = float(corr_spec.to_natural(jnp.asarray([raw_hat], dtype=dtype)))
        warnings.warn(
            f"gp_fit: the corr='{corr_spec.name}' parameter hit the edge of the "
            f'search grid (rho_c={rho_c_edge:.3f}); widen corr_raw_bounds if the '
            'true within-group correlation is stronger (note cs / car1 are '
            'positive by construction).',
            stacklevel=2,
        )
    log_rho_hat = _parabolic_argmin(log_rho_grid_np, grid[i_star])
    rho_hat = float(np.exp(log_rho_hat))

    # --- final per-element fit at (rho_hat, raw_hat) ------------------------
    raw = jnp.asarray([raw_hat], dtype=dtype)
    xtx, c_all, g_all, half_logdet = cross_jit(design_fn(rho_hat), raw)
    d_hat, log_pdet_hat = pen_fn(rho_hat)
    p = xtx.shape[0]

    def _one(c_v: Array, g_v: Array) -> Tuple[Array, ...]:
        return _gp_fit_one(
            c_v, g_v, xtx, d_hat, log_pdet_hat, n, p, m, n_fixed,
            n_outer, ridge, lam_floor, lam_ceil,
        )

    beta, v, lam, edf, phi, log_mlik = cast(
        Tuple[Array, Array, Array, Array, Array, Array],
        blocked_vmap(_one, (c_all, g_all), block=block),
    )
    # Add the whitening Jacobian to the reported marginal likelihood.
    log_mlik = log_mlik - half_logdet
    corr_rho = jnp.full_like(
        phi, float(corr_spec.to_natural(jnp.asarray([raw_hat], dtype=dtype)))
    )
    return _assemble_gp_result(
        beta, v, lam, edf, phi, log_mlik, rho_hat, kernel, engine,
        n, m, n_fixed, lo, hi, boundary,
        corr_name=corr_spec.name, corr_rho=corr_rho,
    )


def _parabolic_argmin(
    log_rho: np.ndarray, nll: np.ndarray
) -> float:
    """Sub-grid minimiser of ``nll(log_rho)`` by a 3-point parabolic fit around
    the grid argmin (clamped to the bracketing interval); returns the grid point
    itself at a boundary minimum."""
    i = int(np.argmin(nll))
    if i == 0 or i == len(nll) - 1:
        return float(log_rho[i])
    x0, x1, x2 = log_rho[i - 1], log_rho[i], log_rho[i + 1]
    y0, y1, y2 = nll[i - 1], nll[i], nll[i + 1]
    denom = (x0 - x1) * (x0 - x2) * (x1 - x2)
    if abs(denom) < 1e-30:
        return float(x1)
    a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / denom
    b = (
        x2 * x2 * (y0 - y1)
        + x1 * x1 * (y2 - y0)
        + x0 * x0 * (y1 - y2)
    ) / denom
    if a <= 0.0:  # not convex -- fall back to the grid point
        return float(x1)
    vertex = -0.5 * b / a
    return float(np.clip(vertex, x0, x2))


def gp_predict(
    result: GPResult,
    x_new: Float[Array, ' g'],
    *,
    parametric: Optional[Float[Array, 'g q']] = None,
    x_train: Optional[Float[Array, ' N']] = None,
    type: Literal['link', 'response'] = 'link',
) -> Tuple[Float[Array, 'V g'], Float[Array, 'V g']]:
    """Per-element posterior mean and standard deviation of the GP at ``x_new``.

    Evaluates ``mean = X_new beta`` and the predictive variance
    ``sigma_e^2 diag(X_new V X_new^T)``, where the smooth columns of ``X_new`` are
    the basis re-evaluated at ``x_new``: for ``engine='hsgp'`` the (``rho``-
    independent) eigenfunctions, reconstructed from the recorded domain
    ``(lo, hi, boundary)`` and ``rank`` (self-contained); for ``engine='exact'``
    the Nystrom kernel features ``K(x_new, x_train) U Lambda^{-1/2}``, which need
    the original ``x_train`` (the result does not store the shared training grid
    among its per-element fields).

    Parameters
    ----------
    result
        A :class:`GPResult` from :func:`gp_fit`.
    x_new
        ``(g,)`` covariate grid to predict on.
    parametric
        ``(g, q)`` parametric covariates matching ``gp_fit``'s ``parametric`` (must
        be supplied when the fit used parametric columns).
    x_train
        ``(N,)`` original training covariate -- **required for ``engine='exact'``**
        (ignored for ``'hsgp'``).
    type
        ``'link'`` (default) returns the latent linear-predictor ``eta``; for a
        non-Gaussian fit, ``'response'`` maps the mean through ``family.linkinv``
        (the mean response, e.g. a probability / rate) and scales ``std`` by the
        delta-method factor ``|d mu / d eta|`` (a point-estimate response-scale
        SD).  A no-op for the Gaussian (identity-link) fit.

    Returns
    -------
    ``(mean, std)`` -- each ``(V, g)``.  ``std`` is the **posterior** standard
    deviation of the latent mean (observation noise is ``sqrt(dispersion)`` on
    top).
    """
    dtype = result.coef.dtype
    x_new = jnp.asarray(x_new, dtype=dtype)
    if x_new.ndim == 2 and x_new.shape[1] == 1 and result.nd_meta is None:
        x_new = x_new[:, 0]
    if result.nd_meta is not None:
        # Multi-dimensional fit: rebuild the tensor-product eigenbasis from the
        # recorded per-axis ranks and bounds (rho-independent -- self-contained).
        m_per, bounds, _ard_rho = result.nd_meta
        x_np = np.asarray(x_new, dtype=np.float64)
        lo_arr = np.asarray([b[0] for b in bounds])
        hi_arr = np.asarray([b[1] for b in bounds])
        # _hsgp_eigen_nd derives the domain from the data min/max -- feed the
        # recorded bounds as two rows so it reconstructs the training domain.
        freqs, phase, inv_sqrt_L, _omega, _b = _hsgp_eigen_nd(
            np.stack([lo_arr, hi_arr]), tuple(m_per), result.boundary, dtype
        )
        phi_new = _hsgp_features_nd(x_new, freqs, phase, inv_sqrt_L)
        _ = x_np
    elif result.engine == 'exact':
        if x_train is None:
            raise ValueError(
                "gp_predict: engine='exact' needs the original `x_train` to "
                'rebuild the kernel features at x_new.'
            )
        rho_hat = float(jnp.exp(result.theta[0, 2]))
        phi_new = _exact_features_predict(
            np.asarray(x_new, dtype=np.float64),
            np.asarray(x_train, dtype=np.float64),
            result.kernel, rho_hat, result.rank, dtype,
        )  # (g, m)
    else:
        c_mid, big_l = _hsgp_domain(result.lo, result.hi, result.boundary)
        sqrt_lambda, phase, inv_sqrt_L = _hsgp_eigen(
            result.rank, c_mid, big_l, dtype
        )
        phi_new = _hsgp_features(x_new, sqrt_lambda, phase, inv_sqrt_L)

    g = x_new.shape[0]
    fixed_blocks = [jnp.ones((g, 1), dtype=result.coef.dtype)]
    n_param = result.n_fixed - 1
    if n_param > 0:
        if parametric is None:
            raise ValueError(
                f'gp_predict: the fit used {n_param} parametric column(s); pass '
                'a matching `parametric` for x_new.'
            )
        fixed_blocks.append(jnp.asarray(parametric, dtype=result.coef.dtype))
    elif parametric is not None:
        raise ValueError(
            'gp_predict: the fit had no parametric columns; pass parametric=None.'
        )
    x_design = jnp.concatenate(fixed_blocks + [phi_new], axis=1)  # (g, p)

    if type not in ('link', 'response'):
        raise ValueError(
            f"gp_predict: type={type!r}; expected 'link' or 'response'."
        )
    eta = result.coef @ x_design.T  # (V, g) -- latent (link-scale) mean
    var = jnp.einsum(
        'gi,vij,gj->vg', x_design, result.cov_unscaled, x_design
    )
    std = jnp.sqrt(jnp.clip(result.dispersion[:, None] * var, 1e-30, None))
    if type == 'response' and result.family != 'gaussian':
        fam = resolve_family(result.family)
        # delta method: sd(mu) ~ |d mu / d eta| sd(eta), a point estimate.
        std = jnp.abs(fam.mu_eta(eta)) * std
        return fam.linkinv(eta), std
    return eta, std


# ---------------------------------------------------------------------------
# Model selection -- AIC / BIC from the REML marginal likelihood
# ---------------------------------------------------------------------------


class _ICResult(Protocol):
    """The fields ``gp_aic`` / ``gp_bic`` read -- :class:`GPResult` and
    ``HGPResult`` both conform structurally."""

    log_mlik: Float[Array, 'V']
    edf: Any
    n_obs: int
    n_fixed: int


def _total_edf(result: _ICResult) -> Float[Array, 'V']:
    """Total effective degrees of freedom (the AIC/BIC complexity ``k``).

    ``GPResult.edf`` is already the total (fixed effects + smooth); the
    hierarchical ``HGPResult.edf`` is ``(V, 2)`` over the smooth blocks only, so
    the unpenalised ``n_fixed`` is added back."""
    edf = jnp.asarray(result.edf)
    if edf.ndim == 1:
        return edf
    return jnp.sum(edf, axis=-1) + result.n_fixed


def gp_aic(result: _ICResult) -> Float[Array, 'V']:
    """Per-element Akaike information criterion ``-2 l_R + 2 k`` for a GP / HGP
    fit (lower is better).

    Uses the **REML log marginal likelihood** ``log_mlik`` (so it is comparable
    across kernels / ranks / amplitudes on the same data) and the effective
    degrees of freedom ``k`` (:func:`_total_edf`) as the model complexity -- the
    mgcv-style marginal AIC.  Like any REML criterion it is valid for models with
    the **same fixed-effect structure** (a different ``parametric`` design changes
    the restriction); within that, it is immediate for GP-vs-GP and GP-vs-spline
    (an ``hsgp_basis`` / kernel choice) selection.
    """
    return -2.0 * result.log_mlik + 2.0 * _total_edf(result)


def gp_bic(result: _ICResult) -> Float[Array, 'V']:
    """Per-element Bayesian information criterion ``-2 l_R + k log N`` for a GP /
    HGP fit (lower is better); see :func:`gp_aic` for the comparability caveat."""
    k = _total_edf(result)
    return -2.0 * result.log_mlik + k * jnp.log(jnp.asarray(float(result.n_obs)))
