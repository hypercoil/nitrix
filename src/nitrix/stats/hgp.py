# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Mass-univariate **hierarchical** Gaussian-process regression (multi-level GP).

``hgp_fit`` fits, per element, a *hierarchical* GP smooth of a covariate ``x``
over a grouping factor -- the GP analogue of a random-slope mixed model, and the
"GS" hierarchical GAM of Pedersen et al. (2019): a **global** smooth plus
**group-level** smooth deviations that share a kernel::

    y_ij = beta0 + f(x_ij) + f_{g(i)}(x_ij) + e_ij
    f      ~ GP(0, sigma_pop^2 K_rho)              (the population trend)
    f_g    ~ GP(0, sigma_grp^2 K_rho)  iid over g  (group deviations)
    e_ij   ~ N(0, sigma_e^2)

The group curves are *random deviations* around the population curve, sharing one
amplitude ``sigma_grp^2`` and the kernel lengthscale ``rho`` (so a group with few
observations is shrunk toward the population trend -- partial pooling).  This is
the multi-level / "(a)" scope of the GP feature: a hierarchical GP in the mixed-
model sense.

Construction (HSGP, the penalty<->variance-component identity)
--------------------------------------------------------------

Both the population and the group smooths use the **fixed** Hilbert-space
eigenbasis ``Phi`` (:mod:`nitrix.stats.gp`); the kernel enters only as the
diagonal spectral reweighting ``s_j(rho)``.  Stacking the population columns and
the group factor-smooth columns (``Phi`` masked to each group's rows) gives a
single penalised design with a **block-diagonal, fully diagonal** penalty -- one
smoothing-parameter block per variance component::

    X = [ 1 | Phi(x) | Phi(x) (x) onehot(g) ]
    penalty = blkdiag( lam_pop diag(1/s) ,  lam_grp diag(1/s) (x) I_L )

``lam_pop = sigma_e^2 / sigma_pop^2`` and ``lam_grp = sigma_e^2 / sigma_grp^2``
are the two smoothing parameters (the inverse GP amplitudes), selected by the
generalized Fellner-Schall step exactly as a multi-smooth GAM; the shared ``rho``
is profiled by the same pooled-REML grid as :func:`~nitrix.stats.gp.gp_fit`.
Because every penalty block is diagonal with **disjoint** columns, the FS penalty
trace is the closed form ``rank_k / lam_k`` and the REML log-pseudo-determinant is
a per-block sum -- no eigendecomposition.

The fit reuses the :mod:`nitrix.stats.gp` HSGP eigenstructure and is cuSOLVER-free
and ``vmap`` clean; the working size is ``O(V (M_0 + (1 + L) m)^2)`` -- the
factor-smooth interaction is ``L`` times wider than a plain GP, the inherent cost
of per-group curves (bound it with ``block`` on brain-scale ``V``).

References
----------
- Pedersen, E. J., Miller, D. L., Simpson, G. L. & Ross, N. (2019). Hierarchical
  generalized additive models in ecology.  PeerJ 7, e6876.
- Wood, S. N. & Fasiolo, M. (2017). A generalized Fellner-Schall method ...
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jaxtyping import Array, Float, Int

from ..linalg._smalllinalg import small_inv_logdet
from ..linalg.kernel import spectral_density
from ._batching import blocked_vmap
from ._result import register_result
from .gp import _hsgp_domain, _hsgp_eigen, _hsgp_features, _parabolic_argmin

__all__ = [
    'HGPResult',
    'hgp_fit',
    'hgp_predict',
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
        'kernel', 'engine', 'model', 'n_levels', 'n_obs', 'rank', 'n_fixed',
        'lo', 'hi', 'boundary',
    ),
)
@dataclass(frozen=True)
class HGPResult:
    """Per-element hierarchical-GP fit output (HSGP engine, ``'GS'`` model).

    Attributes
    ----------
    coef
        ``(V, p)`` coefficients over ``[fixed | pop-smooth | group-smooths]`` with
        ``p = n_fixed + m + n_levels * m``: the unpenalised fixed effects, the
        population eigenfunction coefficients, then each group's deviation
        coefficients (group ``g`` occupies ``[n_fixed + (1+g) m : n_fixed +
        (2+g) m]``).
    cov_unscaled
        ``(V, p, p)`` Bayesian covariance ``(X^T X + S_lambda)^{-1}``.
    theta
        ``(V, 4)`` ``[log sigma_pop^2, log sigma_grp^2, log sigma_e^2, log rho]``
        (the ``rho`` column is constant -- one shared lengthscale).
    log_mlik
        ``(V,)`` REML log marginal likelihood at the fit (up to an additive
        constant in ``n`` / ``n_fixed`` only).
    edf
        ``(V, 2)`` effective degrees of freedom of the population and the (pooled)
        group smooths.
    dispersion
        ``(V,)`` residual variance ``sigma_e^2``.
    kernel, engine, model
        Kernel name, reduced-rank engine (``'hsgp'``), and hierarchical model
        (``'GS'`` -- global + group smoothers, shared group wiggliness).
    n_levels
        Number of factor levels ``L``.
    n_obs, rank, n_fixed
        ``N``, the per-smooth rank ``m``, and the number of fixed columns ``M_0``.
    lo, hi, boundary
        HSGP domain descriptors (for re-evaluation in :func:`hgp_predict`).
    """

    coef: Float[Array, 'V p']
    cov_unscaled: Float[Array, 'V p p']
    theta: Float[Array, 'V 4']
    log_mlik: Float[Array, 'V']
    edf: Float[Array, 'V 2']
    dispersion: Float[Array, 'V']
    kernel: str
    engine: str
    model: str
    n_levels: int
    n_obs: int
    rank: int
    n_fixed: int
    lo: float
    hi: float
    boundary: float


# ---------------------------------------------------------------------------
# Multi-block diagonal penalised-REML core (K smoothing parameters)
# ---------------------------------------------------------------------------
#
# Generalises the single-block core of nitrix.stats.gp to a penalty that is the
# sum of K disjoint diagonal blocks ``S_lambda = sum_k lam_k diag(d_blocks[k])``.
# Every quantity stays diagonal, so there is no eigendecomposition and the FS
# penalty trace keeps the disjoint closed form ``tr(S_lambda^+ S_k) = rank_k /
# lam_k``.


def _mb_quantities(
    lam: Float[Array, ' K'],
    c: Float[Array, ' p'],
    g: Float[Array, ''],
    xtx: Float[Array, 'p p'],
    d_blocks: Float[Array, 'K p'],
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
    """``(V, log|H|, beta, edf, rss, D_p)`` at smoothing parameters ``lam`` for the
    diagonal penalty ``sum_k lam_k diag(d_blocks[k])`` (cf. ``gp._quantities``)."""
    s_diag = lam @ d_blocks + ridge  # (p,)
    h = xtx + jnp.diag(s_diag)
    v, logdet_h = small_inv_logdet(h, p)
    beta = v @ c
    edf = jnp.sum(v * xtx)
    rss = g - 2.0 * (beta @ c) + beta @ (xtx @ beta)
    d_p = g - beta @ c
    return v, logdet_h, beta, edf, rss, d_p


def _mb_fs(
    c: Float[Array, ' p'],
    g: Float[Array, ''],
    xtx: Float[Array, 'p p'],
    d_blocks: Float[Array, 'K p'],
    ranks: Float[Array, ' K'],
    n: int,
    p: int,
    n_outer: int,
    ridge: float,
    lam_floor: float,
    lam_ceil: float,
) -> Float[Array, ' K']:
    """Generalized Fellner-Schall selection of the ``K`` smoothing parameters
    (one per diagonal penalty block), from cross-products."""

    def outer(lam: Float[Array, ' K'], _: Array) -> Tuple[Float[Array, ' K'], None]:
        v, _, beta, edf, rss, _ = _mb_quantities(lam, c, g, xtx, d_blocks, p, ridge)
        phi = rss / jnp.clip(n - edf, 1e-3, None)
        vdiag = jnp.diagonal(v)  # (p,)

        def fs(k: Array) -> Float[Array, '']:
            dk = d_blocks[k]
            tr_vd = jnp.sum(vdiag * dk)
            energy = jnp.sum(dk * beta * beta)
            num = jnp.clip(ranks[k] - lam[k] * tr_vd, 1e-8, None)
            den = jnp.clip(energy / phi, 1e-12, None)
            return jnp.clip(num / den, lam_floor, lam_ceil)

        return jax.vmap(fs)(jnp.arange(d_blocks.shape[0])), None

    lam0 = jnp.ones((d_blocks.shape[0],), dtype=xtx.dtype)
    lam, _ = lax.scan(outer, lam0, xs=None, length=n_outer)
    return lam


def _mb_reml_nll(
    d_p: Float[Array, ''],
    logdet_h: Float[Array, ''],
    lam: Float[Array, ' K'],
    ranks: Float[Array, ' K'],
    log_pdets: Float[Array, ' K'],
    n: int,
    n_fixed: int,
) -> Float[Array, '']:
    """Per-element restricted negative log-likelihood (x2, up to a constant in
    ``n`` / ``n_fixed``):  ``(n - M_0) log D_p + log|H| - sum_k (rank_k log lam_k
    + log_pdet_k)`` with ``log_pdet_k = sum_{j in block k} log d_j``."""
    log_pdet_s = jnp.sum(ranks * jnp.log(lam) + log_pdets)
    return (n - n_fixed) * jnp.log(jnp.clip(d_p, 1e-30, None)) + logdet_h - log_pdet_s


def _hgp_fit_one(
    c: Float[Array, ' p'],
    g: Float[Array, ''],
    xtx: Float[Array, 'p p'],
    d_blocks: Float[Array, 'K p'],
    ranks: Float[Array, ' K'],
    log_pdets: Float[Array, ' K'],
    block_cols: Tuple[Tuple[int, int], ...],
    n: int,
    p: int,
    n_fixed: int,
    n_outer: int,
    ridge: float,
    lam_floor: float,
    lam_ceil: float,
) -> Tuple[
    Float[Array, ' p'],
    Float[Array, 'p p'],
    Float[Array, ' K'],
    Float[Array, ' 2'],
    Float[Array, ''],
    Float[Array, ''],
]:
    """Single-element hierarchical fit at fixed ``rho``.  Returns ``(beta, V, lam,
    edf2, dispersion, log_mlik)`` with ``edf2 = [edf_pop, edf_grp]``."""
    lam = _mb_fs(
        c, g, xtx, d_blocks, ranks, n, p, n_outer, ridge, lam_floor, lam_ceil
    )
    v, logdet_h, beta, edf, rss, d_p = _mb_quantities(
        lam, c, g, xtx, d_blocks, p, ridge
    )
    phi = rss / jnp.clip(n - edf, 1e-3, None)
    nll = _mb_reml_nll(d_p, logdet_h, lam, ranks, log_pdets, n, n_fixed)
    # Per-smooth EDF: tr over each smooth's columns of the influence V X^T X.
    influence_diag = jnp.sum(v * xtx, axis=0)  # (p,) diag(V X^T X) columnwise
    (p_lo, p_hi), (g_lo, g_hi) = block_cols
    edf_pop = jnp.sum(influence_diag[p_lo:p_hi])
    edf_grp = jnp.sum(influence_diag[g_lo:g_hi])
    return beta, v, lam, jnp.stack([edf_pop, edf_grp]), phi, -0.5 * nll


def _hgp_pooled_nll_one(
    c: Float[Array, ' p'],
    g: Float[Array, ''],
    xtx: Float[Array, 'p p'],
    d_blocks: Float[Array, 'K p'],
    ranks: Float[Array, ' K'],
    log_pdets: Float[Array, ' K'],
    n: int,
    p: int,
    n_fixed: int,
    n_outer: int,
    ridge: float,
    lam_floor: float,
    lam_ceil: float,
) -> Float[Array, '']:
    """The per-element ``-2 l_R`` at fixed ``rho`` (its own ``lam``)."""
    lam = _mb_fs(
        c, g, xtx, d_blocks, ranks, n, p, n_outer, ridge, lam_floor, lam_ceil
    )
    _, logdet_h, _, _, _, d_p = _mb_quantities(
        lam, c, g, xtx, d_blocks, p, ridge
    )
    return _mb_reml_nll(d_p, logdet_h, lam, ranks, log_pdets, n, n_fixed)


# ---------------------------------------------------------------------------
# Design assembly
# ---------------------------------------------------------------------------


def _factor_smooth_design(
    phi: Float[Array, 'N m'], group: Int[Array, ' N'], n_levels: int
) -> Float[Array, 'N Lm']:
    """The factor-smooth interaction ``Phi(x) (x) onehot(group)`` (group ``g``'s
    columns are ``Phi`` on its own rows, zero elsewhere), as ``(N, L*m)``."""
    onehot = jax.nn.one_hot(group, n_levels, dtype=phi.dtype)  # (N, L)
    # (N, L, m) -> (N, L*m): block g holds phi where group==g.
    inter = onehot[:, :, None] * phi[:, None, :]
    return inter.reshape(phi.shape[0], n_levels * phi.shape[1])


def _block_diag_weights(
    inv_s: Float[Array, ' m'], n_fixed: int, n_levels: int
) -> Tuple[Float[Array, '2 p'], Float[Array, ' 2'], Tuple[Tuple[int, int], ...]]:
    """The two diagonal penalty blocks (population, group) over the full ``p``
    columns, their ranks, and the smooth column slices.

    ``d_blocks[0]`` carries ``1/s`` on the population columns, ``d_blocks[1]``
    carries ``1/s`` (tiled over the ``L`` groups) on the factor-smooth columns;
    both are zero on the unpenalised fixed columns."""
    m = inv_s.shape[0]
    p = n_fixed + m + n_levels * m
    zeros_fixed = jnp.zeros((n_fixed,), dtype=inv_s.dtype)
    zeros_pop = jnp.zeros((m,), dtype=inv_s.dtype)
    zeros_grp = jnp.zeros((n_levels * m,), dtype=inv_s.dtype)
    d_pop = jnp.concatenate([zeros_fixed, inv_s, zeros_grp])
    d_grp = jnp.concatenate(
        [zeros_fixed, zeros_pop, jnp.tile(inv_s, n_levels)]
    )
    d_blocks = jnp.stack([d_pop, d_grp])  # (2, p)
    ranks = jnp.asarray([m, n_levels * m], dtype=inv_s.dtype)
    block_cols = (
        (n_fixed, n_fixed + m),
        (n_fixed + m, p),
    )
    return d_blocks, ranks, block_cols


def _inv_s(sqrt_lambda: Array, kernel: str, rho: float, dtype: Any) -> Array:
    s = spectral_density(
        sqrt_lambda, kernel=kernel, rho=jnp.asarray(rho, dtype=dtype),
        amplitude=1.0,
    )
    return 1.0 / jnp.clip(s, 1e-30, None)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def hgp_fit(
    Y: Float[Array, 'V N'],
    x: Float[Array, ' N'],
    group: Int[Array, ' N'],
    *,
    parametric: Optional[Float[Array, 'N q']] = None,
    kernel: str = 'matern52',
    rank: int = 12,
    model: str = 'GS',
    boundary: float = 1.5,
    bounds: Optional[Tuple[float, float]] = None,
    rho_bounds: Optional[Tuple[float, float]] = None,
    n_rho: int = 24,
    map_rho: Optional[Callable[[Float[Array, '']], Float[Array, '']]] = None,
    n_levels: Optional[int] = None,
    n_outer: int = 30,
    n_search: int = 15,
    ridge: float = 1e-8,
    lam_floor: float = 1e-6,
    lam_ceil: float = 1e8,
    block: Optional[int] = None,
) -> HGPResult:
    """Fit a mass-univariate hierarchical GP: a population smooth plus group-level
    smooth deviations sharing a kernel (the "GS" hierarchical GAM; HSGP engine).

    Parameters
    ----------
    Y
        ``(V, N)`` responses.
    x
        ``(N,)`` covariate.
    group
        ``(N,)`` integer factor labels ``0 .. L-1`` (the grouping the smooth
        varies over -- subjects, sites, ...).
    parametric
        Optional ``(N, q)`` unpenalised linear design (with the intercept).
    kernel, rank, boundary, bounds
        HSGP basis parameters (see :func:`nitrix.stats.gp.gp_fit`); ``rank`` is the
        per-smooth eigenfunction count (default ``12`` -- the design is ``1 + L``
        smooths wide, so a smaller rank than ``gp_fit`` is usual).
    model
        Hierarchical structure.  ``'GS'`` (only): a global smoother **and**
        group-level smoothers with a single shared group wiggliness
        (``sigma_grp^2``).
    rho_bounds, n_rho, map_rho
        Shared-lengthscale search (as :func:`~nitrix.stats.gp.gp_fit`).
    n_levels
        Number of factor levels ``L`` (defaults to ``int(group.max()) + 1``).
    n_outer, n_search, ridge, lam_floor, lam_ceil, block
        Fellner-Schall / solver controls (as :func:`~nitrix.stats.gp.gp_fit`).

    Returns
    -------
    ``HGPResult`` with per-element coefficients, ``theta = [log sigma_pop^2,
    log sigma_grp^2, log sigma_e^2, log rho]``, REML marginal likelihood, the
    population / group EDF, and dispersion.
    """
    if model != 'GS':
        raise NotImplementedError(
            f"hgp_fit: model={model!r} -- only 'GS' (global + group smoothers) "
            'is implemented.'
        )
    if rank < 1:
        raise ValueError(f'hgp_fit: rank={rank} must be >= 1.')
    if not boundary >= 1.0:
        raise ValueError(f'hgp_fit: boundary={boundary} must be >= 1.0.')

    Y = jnp.asarray(Y)
    x = jnp.asarray(x, dtype=Y.dtype)
    group = jnp.asarray(group)
    n = Y.shape[-1]
    if x.shape[0] != n or group.shape[0] != n:
        raise ValueError(
            f'hgp_fit: x ({x.shape[0]}) and group ({group.shape[0]}) must have '
            f'length N={n} to match Y.'
        )
    L = int(n_levels) if n_levels is not None else int(jnp.max(group)) + 1
    m = int(rank)

    x_np = np.asarray(x, dtype=np.float64)
    lo = float(np.min(x_np)) if bounds is None else float(bounds[0])
    hi = float(np.max(x_np)) if bounds is None else float(bounds[1])
    c_mid, big_l = _hsgp_domain(lo, hi, boundary)
    sqrt_lambda, phase, inv_sqrt_L = _hsgp_eigen(m, c_mid, big_l, Y.dtype)

    # Fixed design: [1 | parametric | Phi_pop | Phi_factor]; rho enters the
    # diagonal penalty weights only.
    phi = _hsgp_features(x, sqrt_lambda, phase, inv_sqrt_L)  # (N, m)
    fixed_blocks = [jnp.ones((n, 1), dtype=Y.dtype)]
    if parametric is not None:
        fixed_blocks.append(jnp.asarray(parametric, dtype=Y.dtype))
    n_fixed = sum(b.shape[1] for b in fixed_blocks)
    phi_factor = _factor_smooth_design(phi, group, L)  # (N, L*m)
    X = jnp.concatenate(fixed_blocks + [phi, phi_factor], axis=1)  # (N, p)
    p = X.shape[1]

    xtx = X.T @ X
    c_all = Y @ X
    g_all = jnp.sum(Y * Y, axis=1)

    # --- shared-rho search --------------------------------------------------
    span = max(hi - lo, 1e-6)
    if rho_bounds is None:
        rho_lo, rho_hi = 0.05 * span, 2.0 * span
    else:
        rho_lo, rho_hi = float(rho_bounds[0]), float(rho_bounds[1])
    log_rho_grid = np.linspace(np.log(rho_lo), np.log(rho_hi), int(n_rho))

    def _blocks(
        rho: float,
    ) -> Tuple[Array, Array, Array, Tuple[Tuple[int, int], ...]]:
        inv_s = _inv_s(sqrt_lambda, kernel, rho, Y.dtype)
        d_blocks, ranks, block_cols = _block_diag_weights(inv_s, n_fixed, L)
        log_pdets = jnp.asarray(
            [jnp.sum(jnp.log(inv_s)), L * jnp.sum(jnp.log(inv_s))],
            dtype=Y.dtype,
        )
        return d_blocks, ranks, log_pdets, block_cols

    @jax.jit
    def _pooled(d_blocks: Array, ranks: Array, log_pdets: Array) -> Array:
        per = jax.vmap(
            lambda c_v, g_v: _hgp_pooled_nll_one(
                c_v, g_v, xtx, d_blocks, ranks, log_pdets, n, p, n_fixed,
                n_search, ridge, lam_floor, lam_ceil,
            )
        )(c_all, g_all)
        return jnp.sum(per)

    nll_grid = []
    for log_rho in log_rho_grid:
        rho = float(np.exp(log_rho))
        d_blocks, ranks, log_pdets, _ = _blocks(rho)
        nll = float(_pooled(d_blocks, ranks, log_pdets))
        if map_rho is not None:
            nll = nll + 2.0 * float(map_rho(jnp.asarray(rho, dtype=Y.dtype)))
        nll_grid.append(nll)

    log_rho_hat = _parabolic_argmin(log_rho_grid, np.asarray(nll_grid))
    rho_hat = float(np.exp(log_rho_hat))

    # --- final per-element fit at rho_hat -----------------------------------
    d_blocks, ranks, log_pdets, block_cols = _blocks(rho_hat)

    def _final(c_v: Array, g_v: Array) -> Tuple[Array, ...]:
        return _hgp_fit_one(
            c_v, g_v, xtx, d_blocks, ranks, log_pdets, block_cols, n, p,
            n_fixed, n_outer, ridge, lam_floor, lam_ceil,
        )

    beta, v, lam, edf, phi_disp, log_mlik = cast(
        Tuple[Array, Array, Array, Array, Array, Array],
        blocked_vmap(_final, (c_all, g_all), block=block),
    )

    sigma_e2 = phi_disp
    sigma_pop2 = sigma_e2 / jnp.clip(lam[:, 0], 1e-30, None)
    sigma_grp2 = sigma_e2 / jnp.clip(lam[:, 1], 1e-30, None)
    log_rho_col = jnp.full_like(sigma_e2, np.log(rho_hat))
    theta = jnp.stack(
        [jnp.log(jnp.clip(sigma_pop2, 1e-30, None)),
         jnp.log(jnp.clip(sigma_grp2, 1e-30, None)),
         jnp.log(jnp.clip(sigma_e2, 1e-30, None)),
         log_rho_col],
        axis=-1,
    )
    return HGPResult(
        coef=beta,
        cov_unscaled=v,
        theta=theta,
        log_mlik=log_mlik,
        edf=edf,
        dispersion=sigma_e2,
        kernel=kernel,
        engine='hsgp',
        model='GS',
        n_levels=int(L),
        n_obs=int(n),
        rank=int(m),
        n_fixed=int(n_fixed),
        lo=lo,
        hi=hi,
        boundary=float(boundary),
    )


def hgp_predict(
    result: HGPResult,
    x_new: Float[Array, ' g'],
    *,
    levels: Optional[Int[Array, ' L']] = None,
    parametric: Optional[Float[Array, 'g q']] = None,
) -> Tuple[Float[Array, '...'], Float[Array, '...']]:
    """Posterior population (or per-group) curves at ``x_new``.

    With ``levels=None`` (default) returns the **population** smooth
    ``(mean, std)`` -- each ``(V, g)`` -- the fixed effects plus ``Phi(x_new)
    beta_pop``.  With ``levels`` a sequence of group indices, returns the
    **group** curves (population + that group's deviation) shaped
    ``(V, len(levels), g)``.

    Parameters
    ----------
    result
        An :class:`HGPResult` from :func:`hgp_fit`.
    x_new
        ``(g,)`` covariate grid.
    levels
        Optional group indices to render; ``None`` gives the population curve.
    parametric
        ``(g, q)`` parametric covariates (required if the fit used them).
    """
    dtype = result.coef.dtype
    x_new = jnp.asarray(x_new, dtype=dtype)
    c_mid, big_l = _hsgp_domain(result.lo, result.hi, result.boundary)
    sqrt_lambda, phase, inv_sqrt_L = _hsgp_eigen(
        result.rank, c_mid, big_l, dtype
    )
    phi_new = _hsgp_features(x_new, sqrt_lambda, phase, inv_sqrt_L)  # (g, m)
    gsz = x_new.shape[0]
    m, mf = result.rank, result.n_fixed

    fixed_blocks = [jnp.ones((gsz, 1), dtype=dtype)]
    n_param = mf - 1
    if n_param > 0:
        if parametric is None:
            raise ValueError(
                f'hgp_predict: the fit used {n_param} parametric column(s); pass '
                'a matching `parametric` for x_new.'
            )
        fixed_blocks.append(jnp.asarray(parametric, dtype=dtype))
    elif parametric is not None:
        raise ValueError(
            'hgp_predict: the fit had no parametric columns; pass parametric=None.'
        )
    t_new = jnp.concatenate(fixed_blocks, axis=1)  # (g, mf)

    # Population design: [T | Phi | 0 ... 0] over the group columns.
    fixed_part = result.coef[:, :mf] @ t_new.T  # (V, g)
    beta_pop = result.coef[:, mf:mf + m]  # (V, m)
    pop_smooth = beta_pop @ phi_new.T  # (V, g)
    pop_mean = fixed_part + pop_smooth

    if levels is None:
        x_design = jnp.concatenate(
            [t_new, phi_new, jnp.zeros((gsz, result.n_levels * m), dtype=dtype)],
            axis=1,
        )
        var = jnp.einsum('gi,vij,gj->vg', x_design, result.cov_unscaled, x_design)
        std = jnp.sqrt(jnp.clip(result.dispersion[:, None] * var, 1e-30, None))
        return pop_mean, std

    levels = jnp.asarray(levels).astype(jnp.int32)

    def _one_level(ell: Array) -> Tuple[Array, Array]:
        start = mf + m + ell * m
        beta_g = jax.lax.dynamic_slice_in_dim(result.coef, start, m, axis=1)
        mean = pop_mean + beta_g @ phi_new.T  # (V, g)
        # Curve design: T + population Phi + this group's Phi block (the other
        # group blocks zero), as (g, L*m).
        onehot = jax.nn.one_hot(ell, result.n_levels, dtype=dtype)  # (L,)
        grp_cols = (
            phi_new[:, None, :] * onehot[None, :, None]
        ).reshape(gsz, result.n_levels * m)  # (g, L*m)
        x_design = jnp.concatenate([t_new, phi_new, grp_cols], axis=1)
        var = jnp.einsum('gi,vij,gj->vg', x_design, result.cov_unscaled, x_design)
        std = jnp.sqrt(jnp.clip(result.dispersion[:, None] * var, 1e-30, None))
        return mean, std

    means, stds = jax.vmap(_one_level)(levels)  # (L_sel, V, g)
    return jnp.transpose(means, (1, 0, 2)), jnp.transpose(stds, (1, 0, 2))
