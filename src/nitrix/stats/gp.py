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

from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional, Tuple, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jaxtyping import Array, Float

from ..linalg._smalllinalg import small_inv_logdet
from ..linalg.kernel import spectral_density
from ._batching import blocked_vmap
from ._result import register_result

__all__ = [
    'GPResult',
    'gp_fit',
    'gp_predict',
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
    ),
    aux=(
        'kernel', 'engine', 'n_obs', 'rank', 'n_fixed', 'lo', 'hi', 'boundary'
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
        ``(V,)`` residual-scale estimate ``sigma_e^2``.
    kernel
        Stationary kernel name (``'matern52'`` / ``'matern32'`` / ``'matern12'`` /
        ``'rbf'``).
    engine
        Reduced-rank engine: ``'hsgp'`` (Hilbert-space eigenfunctions) or
        ``'exact'`` (kernel eigenfeatures -- full-rank when ``rank == N``, else the
        eigen-truncated Karhunen-Loeve / Nystrom approximation).
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
    """

    coef: Float[Array, 'V p']
    cov_unscaled: Float[Array, 'V p p']
    theta: Float[Array, 'V 3']
    log_mlik: Float[Array, 'V']
    edf: Float[Array, 'V']
    dispersion: Float[Array, 'V']
    kernel: str
    engine: str
    n_obs: int
    rank: int
    n_fixed: int
    lo: float
    hi: float
    boundary: float


# ---------------------------------------------------------------------------
# Fixed eigenstructure (rho-independent): the Hilbert-space design
# ---------------------------------------------------------------------------


def _hsgp_domain(
    lo: float, hi: float, boundary: float
) -> Tuple[float, float]:
    """``(c, L)``: the domain midrange and the half-width of ``[c - L, c + L]``."""
    c = 0.5 * (lo + hi)
    half = max(0.5 * (hi - lo), 1e-6)
    return c, float(boundary) * half


def _hsgp_eigen(
    rank: int, c: float, L: float, dtype: Any
) -> Tuple[Float[Array, ' m'], Float[Array, ' m'], float]:
    """The fixed Laplace-Dirichlet eigen-frequencies, per-mode phase, and the
    ``sqrt(1/L)`` amplitude -- everything the eigenfunctions need, independent of
    the kernel and ``rho``.

    ``phi_j(x) = sqrt(1/L) sin(sqrt(lambda_j) (x - c + L)) = sqrt(1/L)
    sin(sqrt(lambda_j) x + phase_j)`` with ``phase_j = sqrt(lambda_j) (L - c)``
    (folding the centring constant ``c`` into the stored phase).
    """
    j = np.arange(1, rank + 1, dtype=np.float64)
    sqrt_lambda = j * np.pi / (2.0 * L)
    phase = sqrt_lambda * (L - c)
    return (
        jnp.asarray(sqrt_lambda, dtype=dtype),
        jnp.asarray(phase, dtype=dtype),
        float(np.sqrt(1.0 / L)),
    )


def _hsgp_features(
    x: Float[Array, ' g'],
    sqrt_lambda: Float[Array, ' m'],
    phase: Float[Array, ' m'],
    inv_sqrt_L: float,
) -> Float[Array, 'g m']:
    """The (rho-independent) eigenfunction design ``Phi`` at covariate ``x``."""
    return inv_sqrt_L * jnp.sin(
        sqrt_lambda[None, :] * x[:, None] + phase[None, :]
    )


def _penalty_diag(
    sqrt_lambda: Float[Array, ' m'],
    kernel: str,
    rho: Float[Array, ''],
    n_fixed: int,
) -> Tuple[Float[Array, ' p'], Float[Array, '']]:
    """The diagonal penalty core ``d`` over the full ``p = n_fixed + m`` columns
    and the smooth-block log-pseudo-determinant contribution ``sum_j log d_j``.

    The penalty is ``S_lambda = lambda diag(d)`` with ``d = [0, ..., 0, 1/s_1,
    ..., 1/s_m]`` (zeros on the unpenalised fixed-effect columns); ``s_j =
    S_rho(sqrt(lambda_j))`` is the spectral density (amplitude folded into
    ``lambda``).  Returns ``(d, sum_j log(1/s_j))``.
    """
    s = spectral_density(sqrt_lambda, kernel=kernel, rho=rho, amplitude=1.0)
    inv_s = 1.0 / jnp.clip(s, 1e-30, None)
    d = jnp.concatenate(
        [jnp.zeros((n_fixed,), dtype=inv_s.dtype), inv_s]
    )
    return d, jnp.sum(jnp.log(inv_s))


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
    is the penalised residual sum of squares (``= rss + beta^T S beta``).
    """
    s_diag = lam * d + ridge
    h = xtx + jnp.diag(s_diag)
    v, logdet_h = small_inv_logdet(h, p)
    beta = v @ c
    edf = jnp.sum(v * xtx)  # tr(V X^T X)
    rss = g - 2.0 * (beta @ c) + beta @ (xtx @ beta)
    d_p = g - beta @ c
    return v, logdet_h, beta, edf, rss, d_p


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
    with ``energy = beta^T diag(d) beta`` and ``phi = rss / (n - edf)``.
    """

    def outer(lam: Float[Array, ''], _: Array) -> Tuple[Float[Array, ''], None]:
        v, _, beta, edf, rss, _ = _quantities(lam, c, g, xtx, d, p, ridge)
        phi = rss / jnp.clip(n - edf, 1e-3, None)
        tr_vd = jnp.sum(jnp.diagonal(v) * d)  # tr(V diag(d))
        energy = (d * beta) @ beta
        num = jnp.clip(m - lam * tr_vd, 1e-8, None)
        den = jnp.clip(energy / phi, 1e-12, None)
        return jnp.clip(num / den, lam_floor, lam_ceil), None

    lam0 = jnp.asarray(1.0, dtype=xtx.dtype)
    lam, _ = lax.scan(outer, lam0, xs=None, length=n_outer)
    return lam


def _reml_nll(
    d_p: Float[Array, ''],
    logdet_h: Float[Array, ''],
    lam: Float[Array, ''],
    log_pdet_pen: Float[Array, ''],
    n: int,
    m: int,
    n_fixed: int,
) -> Float[Array, '']:
    """Per-element restricted negative log-likelihood (x2, up to an additive
    constant in ``n`` / ``n_fixed``):  ``(n - M_0) log D_p + log|H| -
    log|S_lambda|_+`` with ``log|S_lambda|_+ = m log lambda + sum_j log(1/s_j)``.
    """
    log_pdet_s = m * jnp.log(lam) + log_pdet_pen
    return (n - n_fixed) * jnp.log(jnp.clip(d_p, 1e-30, None)) + logdet_h - log_pdet_s


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
) -> Float[Array, '']:
    """Pooled ``-2 l_R`` for a design ``X`` whose smooth columns move with ``rho``
    (the ``engine='exact'`` case: the cross-products are rebuilt per ``rho``)."""
    p = X.shape[1]
    xtx = X.T @ X
    c_all = Y @ X
    g_all = jnp.sum(Y * Y, axis=1)
    per = jax.vmap(
        lambda c_v, g_v: _pooled_nll_one(
            c_v, g_v, xtx, d, log_pdet_pen, n, p, m, n_fixed,
            n_outer, ridge, lam_floor, lam_ceil,
        )
    )(c_all, g_all)
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
# Public API
# ---------------------------------------------------------------------------


def gp_fit(
    Y: Float[Array, 'V N'],
    x: Float[Array, ' N'],
    *,
    parametric: Optional[Float[Array, 'N q']] = None,
    kernel: str = 'matern52',
    rank: Optional[int] = None,
    boundary: float = 1.5,
    bounds: Optional[Tuple[float, float]] = None,
    engine: Literal['hsgp', 'exact'] = 'hsgp',
    select: Literal['shared-rho'] = 'shared-rho',
    rho_bounds: Optional[Tuple[float, float]] = None,
    n_rho: int = 24,
    map_rho: Optional[Callable[[Float[Array, '']], Float[Array, '']]] = None,
    corr: Optional[Any] = None,
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
        ``(N,)`` covariate.
    parametric
        Optional ``(N, q)`` unpenalised linear design (covariates entering
        linearly alongside the intercept).
    kernel
        Stationary kernel: ``'matern52'`` (default) / ``'matern32'`` /
        ``'matern12'`` / ``'rbf'``.
    rank
        Smooth-basis rank ``m``.  ``None`` (default) uses an engine-appropriate
        value: ``20`` for ``'hsgp'`` (small ``rho`` needs a larger ``rank``), and
        ``N`` for ``'exact'`` (full-rank).  An explicit ``rank < N`` with
        ``engine='exact'`` gives the eigen-truncated Karhunen-Loeve / Nystrom
        approximation.
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
        Optional callable ``rho -> penalty`` adding ``-log p(rho)`` to the pooled
        objective (a MAP / prior-regularised lengthscale); ``None`` is pure REML.
    corr
        Reserved for correlated-residual composition (PR3b); must be ``None``.
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
    if corr is not None:
        raise NotImplementedError(
            'gp_fit: corr= (correlated-residual composition) is PR3b; pass None.'
        )
    if not boundary >= 1.0:
        raise ValueError(f'gp_fit: boundary={boundary} must be >= 1.0.')

    Y = jnp.asarray(Y)
    x = jnp.asarray(x, dtype=Y.dtype)
    n = Y.shape[-1]
    if x.shape[0] != n:
        raise ValueError(
            f'gp_fit: x has {x.shape[0]} points; expected N={n} to match Y.'
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

    if engine == 'exact':
        return _gp_fit_exact(
            Y, x_np, T, n_fixed, kernel, m, log_rho_grid_np, map_rho,
            n_outer, n_search, ridge, lam_floor, lam_ceil, block,
            lo, hi, boundary,
        )

    # ==================== engine == 'hsgp' ==================================
    c_mid, big_l = _hsgp_domain(lo, hi, boundary)
    sqrt_lambda, phase, inv_sqrt_L = _hsgp_eigen(m, c_mid, big_l, Y.dtype)

    # Fixed full design X = [intercept | parametric | Phi]; rho enters only the
    # diagonal penalty.
    phi_design = _hsgp_features(x, sqrt_lambda, phase, inv_sqrt_L)  # (N, m)
    X = jnp.concatenate([T, phi_design], axis=1)  # (N, p)
    p = X.shape[1]

    xtx = X.T @ X
    c_all = Y @ X  # (V, p) -- row v is X^T y_v
    g_all = jnp.sum(Y * Y, axis=1)  # (V,)

    log_rho_grid = jnp.asarray(log_rho_grid_np, dtype=Y.dtype)

    def _pooled_nll(log_rho: Float[Array, '']) -> Float[Array, '']:
        rho = jnp.exp(log_rho)
        d, log_pdet_pen = _penalty_diag(sqrt_lambda, kernel, rho, n_fixed)
        per = jax.vmap(
            lambda c_v, g_v: _pooled_nll_one(
                c_v, g_v, xtx, d, log_pdet_pen, n, p, m, n_fixed,
                n_search, ridge, lam_floor, lam_ceil,
            )
        )(c_all, g_all)
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
    d_hat, log_pdet_hat = _penalty_diag(
        sqrt_lambda, kernel, jnp.asarray(rho_hat, dtype=Y.dtype), n_fixed
    )

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
    return GPResult(
        coef=beta,
        cov_unscaled=v,
        theta=theta,
        log_mlik=log_mlik,
        edf=edf,
        dispersion=sigma_e2,
        kernel=kernel,
        engine=engine,
        n_obs=int(n),
        rank=int(m),
        n_fixed=int(n_fixed),
        lo=lo,
        hi=hi,
        boundary=float(boundary),
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
    # Unit spectral weights: penalty diag = [0...0, 1...1]; log-pdet 0.
    d = jnp.concatenate(
        [jnp.zeros((n_fixed,), dtype=dtype), jnp.ones((m,), dtype=dtype)]
    )
    log_pdet_pen = jnp.asarray(0.0, dtype=dtype)
    n = Y.shape[-1]

    def _design(rho: float) -> Float[Array, 'N p']:
        phi = _exact_features_train(x_np, kernel, rho, m, dtype)  # (N, m)
        return jnp.concatenate([T, phi], axis=1)

    # --- rho search: host loop (each rho needs a host kernel eigh) ----------
    nll_grid = []
    for log_rho in log_rho_grid_np:
        rho = float(np.exp(log_rho))
        nll = _pooled_nll_from_design(
            Y, _design(rho), d, log_pdet_pen, n, m, n_fixed,
            n_search, ridge, lam_floor, lam_ceil,
        )
        if map_rho is not None:
            nll = nll + 2.0 * map_rho(jnp.asarray(rho, dtype=dtype))
        nll_grid.append(float(nll))

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

    Returns
    -------
    ``(mean, std)`` -- each ``(V, g)``.  ``std`` is the **posterior** standard
    deviation of the latent mean (observation noise is ``sqrt(dispersion)`` on
    top).
    """
    dtype = result.coef.dtype
    x_new = jnp.asarray(x_new, dtype=dtype)
    if result.engine == 'exact':
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

    mean = result.coef @ x_design.T  # (V, g)
    var = jnp.einsum(
        'gi,vij,gj->vg', x_design, result.cov_unscaled, x_design
    )
    std = jnp.sqrt(jnp.clip(result.dispersion[:, None] * var, 1e-30, None))
    return mean, std
