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
from jaxtyping import Array, Float, Int

from ..linalg.kernel import spectral_density
from ._batching import blocked_vmap
from ._penreml import mb_fs, mb_quantities, mb_reml_nll
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
        ``(V, p)`` coefficients over ``[fixed | pop-smooth | group-smooths…]``:
        the unpenalised fixed effects, the population eigenfunction coefficients,
        then each grouping level's per-group deviation coefficients.  ``'GS'``:
        ``p = n_fixed + m + L m``; ``'nested'``: ``p = n_fixed + m + (L1 + L2) m``.
    cov_unscaled
        ``(V, p, p)`` Bayesian covariance ``(X^T X + S_lambda)^{-1}``.
    theta
        ``(V, K + 2)`` ``[log sigma_pop^2, (log sigma_grp_i^2)…, log sigma_e^2,
        log rho]`` -- one GP variance per component (``K = 2`` for ``'GS'``,
        ``3`` for ``'nested'``); the ``rho`` column is constant (one shared
        lengthscale).
    log_mlik
        ``(V,)`` REML log marginal likelihood at the fit.
    edf
        ``(V, K)`` effective degrees of freedom of each GP component (population,
        then each grouping level).
    dispersion
        ``(V,)`` residual variance ``sigma_e^2``.
    kernel, engine, model
        Kernel name, reduced-rank engine (``'hsgp'``), and hierarchical model
        (``'GS'`` -- global + group smoothers; ``'nested'`` -- two-level g1/g2).
    n_levels
        Number of factor levels: ``L`` (``'GS'``) or the tuple ``(L1, L2)``
        (``'nested'``).
    n_obs, rank, n_fixed
        ``N``, the per-smooth rank ``m``, and the number of fixed columns ``M_0``.
    lo, hi, boundary
        HSGP domain descriptors (for re-evaluation in :func:`hgp_predict`).
    """

    coef: Float[Array, 'V p']
    cov_unscaled: Float[Array, 'V p p']
    theta: Float[Array, 'V K2']
    log_mlik: Float[Array, 'V']
    edf: Float[Array, 'V K']
    dispersion: Float[Array, 'V']
    kernel: str
    engine: str
    model: str
    n_levels: Any
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
# The K-block core (``mb_quantities`` / ``mb_fs`` / ``mb_reml_nll``) lives in
# ``nitrix.stats._penreml`` -- the single source of truth shared with the
# single-block (``K = 1``) ``nitrix.stats.gp``.  Re-exported here under the
# historical ``_mb_*`` names used by the entry points below and the tests; ``hgp``
# builds the ``(K,)`` penalty layout (``_block_weights``) on top of it.
_mb_quantities = mb_quantities
_mb_fs = mb_fs
_mb_reml_nll = mb_reml_nll


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
    Float[Array, ' K'],
    Float[Array, ''],
    Float[Array, ''],
]:
    """Single-element hierarchical fit at fixed ``rho``.  Returns ``(beta, V, lam,
    edf_blocks, dispersion, log_mlik)`` with ``edf_blocks`` the per-GP-component
    effective dof (population, then each grouping level)."""
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
    edf_blocks = jnp.stack(
        [jnp.sum(influence_diag[lo:hi]) for (lo, hi) in block_cols]
    )
    return beta, v, lam, edf_blocks, phi, -0.5 * nll


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


def _block_weights(
    inv_s: Float[Array, ' m'], n_fixed: int, level_counts: Tuple[int, ...]
) -> Tuple[Float[Array, 'K p'], Float[Array, ' K'], Tuple[Tuple[int, int], ...]]:
    """The ``K = 1 + len(level_counts)`` diagonal penalty blocks over the full
    ``p`` columns, their ranks, and the smooth column slices.

    Block 0 is the **population** smooth (``1/s`` on its ``m`` columns); block
    ``i+1`` is the ``i``-th **factor-smooth** (``1/s`` tiled over its
    ``level_counts[i]`` groups). ``level_counts = (L,)`` is the GS model;
    ``(L1, L2)`` is the nested two-level model. Every block is diagonal with
    disjoint columns."""
    m = inv_s.shape[0]
    reps = (1,) + tuple(level_counts)  # population is one copy
    widths = tuple(rep * m for rep in reps)
    p = n_fixed + sum(widths)
    d_blocks = []
    ranks = []
    block_cols = []
    col = n_fixed
    for rep, w in zip(reps, widths):
        pre = jnp.zeros((col,), dtype=inv_s.dtype)
        post = jnp.zeros((p - col - w,), dtype=inv_s.dtype)
        d_blocks.append(jnp.concatenate([pre, jnp.tile(inv_s, rep), post]))
        ranks.append(float(w))
        block_cols.append((col, col + w))
        col += w
    return (
        jnp.stack(d_blocks),
        jnp.asarray(ranks, dtype=inv_s.dtype),
        tuple(block_cols),
    )


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
    group_inner: Optional[Int[Array, ' N']] = None,
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
    n_levels_inner: Optional[int] = None,
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
        ``(N,)`` covariate the GP smooths *over* (a single covariate, **not** the
        full design matrix ``X``; linear covariates go to ``parametric=``).
    group
        ``(N,)`` integer factor labels ``0 .. L-1`` (the grouping the smooth
        varies over -- subjects, sites, ...).  For ``model='nested'`` this is the
        **outer** factor.
    group_inner
        ``(N,)`` integer labels of the **inner** factor (nested within ``group``),
        ``0 .. L2-1`` globally.  Required for ``model='nested'``.
    parametric
        Optional ``(N, q)`` unpenalised linear design (with the intercept).
    kernel, rank, boundary, bounds
        HSGP basis parameters (see :func:`nitrix.stats.gp.gp_fit`); ``rank`` is the
        per-smooth eigenfunction count (default ``12`` -- the design is ``1 + L``
        smooths wide, so a smaller rank than ``gp_fit`` is usual).
    model
        Hierarchical structure.  ``'GS'`` (default): a global smoother plus
        group-level smoothers with one shared group wiggliness ``sigma_grp^2``.
        ``'nested'``: a two-level hierarchy ``(gp | g1/g2)`` -- population +
        outer-group + inner-group(nested) GP deviations, three variance components
        (``sigma_pop^2``, ``sigma_outer^2``, ``sigma_inner^2``) sharing ``rho``.
    n_levels, n_levels_inner
        Outer / inner factor-level counts (default to ``max + 1``); pass when a
        level is absent so the block widths are stable.
    rho_bounds, n_rho, map_rho
        Shared-lengthscale search (as :func:`~nitrix.stats.gp.gp_fit`); ``map_rho``
        is an optional ``rho -> -log p(rho)`` lengthscale prior (e.g. a builder
        from :mod:`nitrix.stats.priors`).
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
    if model not in ('GS', 'nested'):
        raise NotImplementedError(
            f"hgp_fit: model={model!r} -- expected 'GS' (global + group "
            "smoothers) or 'nested' (two-level g1/g2)."
        )
    if rank < 1:
        raise ValueError(f'hgp_fit: rank={rank} must be >= 1.')
    if not boundary >= 1.0:
        raise ValueError(f'hgp_fit: boundary={boundary} must be >= 1.0.')

    Y = jnp.asarray(Y)
    if not jnp.issubdtype(Y.dtype, jnp.floating):
        Y = Y.astype(float)  # ER6: integer Y would coerce the whole fit to int
    x = jnp.asarray(x, dtype=Y.dtype)
    group = jnp.asarray(group)
    n = Y.shape[-1]
    if x.shape[0] != n or group.shape[0] != n:
        raise ValueError(
            f'hgp_fit: x ({x.shape[0]}) and group ({group.shape[0]}) must have '
            f'length N={n} to match Y.'
        )
    nested = model == 'nested' or group_inner is not None
    # Validate the label range host-side: jax.nn.one_hot maps any label >= L or
    # < 0 to an all-zero row, so an out-of-range/negative label would silently
    # drop that observation out of the group structure (a wrong-but-finite fit).
    g_min, g_max = int(jnp.min(group)), int(jnp.max(group))
    L = int(n_levels) if n_levels is not None else g_max + 1
    if g_min < 0 or g_max >= L:
        raise ValueError(
            f'hgp_fit: group labels must lie in [0, n_levels={L}); got '
            f'min={g_min}, max={g_max}. Use contiguous 0-based labels (and, if '
            'passing n_levels explicitly, n_levels >= max(group)+1) so no '
            'observation silently drops out of the group structure.'
        )
    groupings = [group]
    level_counts = [L]
    if nested:
        if group_inner is None:
            raise ValueError(
                "hgp_fit: model='nested' requires `group_inner` (the inner "
                'factor nested within `group`).'
            )
        group_inner = jnp.asarray(group_inner)
        if group_inner.shape[0] != n:
            raise ValueError(
                f'hgp_fit: group_inner ({group_inner.shape[0]}) must have '
                f'length N={n}.'
            )
        gi_min, gi_max = int(jnp.min(group_inner)), int(jnp.max(group_inner))
        L2 = int(n_levels_inner) if n_levels_inner is not None else gi_max + 1
        if gi_min < 0 or gi_max >= L2:
            raise ValueError(
                f'hgp_fit: group_inner labels must lie in [0, '
                f'n_levels_inner={L2}); got min={gi_min}, max={gi_max}. Inner '
                'labels must be globally numbered 0..L2-1 (not per-outer).'
            )
        # ER2: the inner factor must be GLOBALLY numbered -- each inner level
        # nested within exactly one outer level. Per-outer numbering (each outer
        # group's inner labels restarting at 0) passes the range check above but
        # silently aliases distinct subjects across outer groups -> mis-pooling.
        # A properly nested factor has each inner label paired with one outer.
        go_np = np.asarray(group)
        gi_np = np.asarray(group_inner)
        pairs = np.unique(np.stack([gi_np, go_np], axis=1), axis=0)
        if pairs.shape[0] != len(np.unique(gi_np)):
            raise ValueError(
                'hgp_fit: model=\'nested\' requires globally-numbered inner '
                'labels (each inner level nested within a single `group` level); '
                'an inner label appears under multiple outer levels. Renumber '
                'the inner factor globally (e.g. encode the (outer, inner) pair), '
                'not per-outer -- or use a crossed model if the factors are '
                'genuinely crossed.'
            )
        groupings.append(group_inner)
        level_counts.append(L2)
    m = int(rank)
    n_blocks = 1 + len(level_counts)

    x_np = np.asarray(x, dtype=np.float64)
    lo = float(np.min(x_np)) if bounds is None else float(bounds[0])
    hi = float(np.max(x_np)) if bounds is None else float(bounds[1])
    c_mid, big_l = _hsgp_domain(lo, hi, boundary)
    sqrt_lambda, phase, inv_sqrt_L = _hsgp_eigen(m, c_mid, big_l, Y.dtype)

    # Fixed design: [1 | parametric | Phi_pop | Phi_factor(g1) | Phi_factor(g2)…];
    # rho enters the diagonal penalty weights only.
    phi = _hsgp_features(x, sqrt_lambda, phase, inv_sqrt_L)  # (N, m)
    fixed_blocks = [jnp.ones((n, 1), dtype=Y.dtype)]
    if parametric is not None:
        fixed_blocks.append(jnp.asarray(parametric, dtype=Y.dtype))
    n_fixed = sum(b.shape[1] for b in fixed_blocks)
    factor_designs = [
        _factor_smooth_design(phi, grp, lc)
        for grp, lc in zip(groupings, level_counts)
    ]
    X = jnp.concatenate(
        fixed_blocks + [phi] + factor_designs, axis=1
    )  # (N, p)
    p = X.shape[1]
    level_counts_t = tuple(level_counts)

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
        d_blocks, ranks, block_cols = _block_weights(
            inv_s, n_fixed, level_counts_t
        )
        reps = (1,) + level_counts_t
        log_pdets = jnp.asarray(
            [rep * jnp.sum(jnp.log(inv_s)) for rep in reps], dtype=Y.dtype
        )
        return d_blocks, ranks, log_pdets, block_cols

    @jax.jit
    def _pooled(d_blocks: Array, ranks: Array, log_pdets: Array) -> Array:
        # Chunk the pooled-NLL reduction by `block` too: the hierarchical design
        # is (1 + sum(level_counts)) * m wide, so an un-chunked search over all V
        # is the acuter OOM cliff. blocked_vmap(...).sum() is a drop-in.
        per = blocked_vmap(
            lambda c_v, g_v: _hgp_pooled_nll_one(
                c_v, g_v, xtx, d_blocks, ranks, log_pdets, n, p, n_fixed,
                n_search, ridge, lam_floor, lam_ceil,
            ),
            (c_all, g_all),
            block=block,
        )
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

    # theta = [log sigma_k^2 (one per GP component), log sigma_e^2, log rho].
    sigma_e2 = phi_disp
    sigma_gp2 = [
        sigma_e2 / jnp.clip(lam[:, k], 1e-30, None) for k in range(n_blocks)
    ]
    log_rho_col = jnp.full_like(sigma_e2, np.log(rho_hat))
    theta = jnp.stack(
        [jnp.log(jnp.clip(s, 1e-30, None)) for s in sigma_gp2]
        + [jnp.log(jnp.clip(sigma_e2, 1e-30, None)), log_rho_col],
        axis=-1,
    )
    n_levels_aux: Any = (
        tuple(level_counts) if nested else int(level_counts[0])
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
        model='nested' if nested else 'GS',
        n_levels=n_levels_aux,
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

    # Population design: [T | Phi | 0 ... 0] over all group columns (works for
    # GS and nested -- the group-column count is read off the coefficient width).
    n_group_cols = result.coef.shape[1] - mf - m
    fixed_part = result.coef[:, :mf] @ t_new.T  # (V, g)
    beta_pop = result.coef[:, mf:mf + m]  # (V, m)
    pop_smooth = beta_pop @ phi_new.T  # (V, g)
    pop_mean = fixed_part + pop_smooth

    if levels is None:
        x_design = jnp.concatenate(
            [t_new, phi_new, jnp.zeros((gsz, n_group_cols), dtype=dtype)],
            axis=1,
        )
        var = jnp.einsum('gi,vij,gj->vg', x_design, result.cov_unscaled, x_design)
        std = jnp.sqrt(jnp.clip(result.dispersion[:, None] * var, 1e-30, None))
        return pop_mean, std

    if result.model == 'nested':
        raise NotImplementedError(
            "hgp_predict: per-group curves for model='nested' are not yet "
            'supported; use levels=None for the population curve.'
        )
    n_lev = int(result.n_levels)
    levels = jnp.asarray(levels).astype(jnp.int32)

    def _one_level(ell: Array) -> Tuple[Array, Array]:
        start = mf + m + ell * m
        beta_g = jax.lax.dynamic_slice_in_dim(result.coef, start, m, axis=1)
        mean = pop_mean + beta_g @ phi_new.T  # (V, g)
        # Curve design: T + population Phi + this group's Phi block (the other
        # group blocks zero), as (g, L*m).
        onehot = jax.nn.one_hot(ell, n_lev, dtype=dtype)  # (L,)
        grp_cols = (
            phi_new[:, None, :] * onehot[None, :, None]
        ).reshape(gsz, n_lev * m)  # (g, L*m)
        x_design = jnp.concatenate([t_new, phi_new, grp_cols], axis=1)
        var = jnp.einsum('gi,vij,gj->vg', x_design, result.cov_unscaled, x_design)
        std = jnp.sqrt(jnp.clip(result.dispersion[:, None] * var, 1e-30, None))
        return mean, std

    means, stds = jax.vmap(_one_level)(levels)  # (L_sel, V, g)
    return jnp.transpose(means, (1, 0, 2)), jnp.transpose(stds, (1, 0, 2))
