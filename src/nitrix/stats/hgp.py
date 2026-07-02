# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Mass-univariate **hierarchical** Gaussian-process regression (multi-level GP).

:func:`hgp_fit` fits, per element, a *hierarchical* GP smooth of a covariate
``x`` over a grouping factor -- the GP analogue of a random-slope mixed model,
and the "GS" hierarchical GAM of Pedersen et al. (2019): a **global** smooth plus
**group-level** smooth deviations that share a kernel:

.. math::

    y_{ij} &= \\beta_0 + f(x_{ij}) + f_{g(i)}(x_{ij}) + e_{ij} \\\\
    f &\\sim \\mathrm{GP}(0, \\sigma_{\\mathrm{pop}}^2 K_\\rho)
       \\quad \\text{(the population trend)} \\\\
    f_g &\\sim \\mathrm{GP}(0, \\sigma_{\\mathrm{grp}}^2 K_\\rho)
       \\quad \\text{iid over } g \\quad \\text{(group deviations)} \\\\
    e_{ij} &\\sim \\mathcal{N}(0, \\sigma_e^2)

The group curves are *random deviations* around the population curve, sharing one
amplitude :math:`\\sigma_{\\mathrm{grp}}^2` and the kernel lengthscale
:math:`\\rho` (so a group with few observations is shrunk toward the population
trend -- partial pooling).  This is the multi-level scope of the GP feature: a
hierarchical GP in the mixed-model sense.

Construction (HSGP, the penalty-variance-component identity)
------------------------------------------------------------

Both the population and the group smooths use the **fixed** Hilbert-space
eigenbasis :math:`\\Phi` (see :func:`nitrix.stats.gp.gp_fit`); the kernel enters
only as the diagonal spectral reweighting :math:`s_j(\\rho)`.  Stacking the
population columns and the group factor-smooth columns (:math:`\\Phi` masked to
each group's rows) gives a single penalised design with a **block-diagonal, fully
diagonal** penalty -- one smoothing-parameter block per variance component:

.. math::

    X &= [\\, \\mathbf{1} \\mid \\Phi(x) \\mid \\Phi(x) \\otimes
        \\operatorname{onehot}(g) \\,] \\\\
    \\text{penalty} &= \\operatorname{blkdiag}\\!\\big(
        \\lambda_{\\mathrm{pop}} \\operatorname{diag}(1/s),\\;
        \\lambda_{\\mathrm{grp}} \\operatorname{diag}(1/s) \\otimes I_L \\big)

:math:`\\lambda_{\\mathrm{pop}} = \\sigma_e^2 / \\sigma_{\\mathrm{pop}}^2` and
:math:`\\lambda_{\\mathrm{grp}} = \\sigma_e^2 / \\sigma_{\\mathrm{grp}}^2` are the
two smoothing parameters (the inverse GP amplitudes), selected by the generalised
Fellner-Schall step exactly as a multi-smooth GAM; the shared :math:`\\rho` is
profiled by the same pooled-REML grid as :func:`nitrix.stats.gp.gp_fit`.
Because every penalty block is diagonal with **disjoint** columns, the
Fellner-Schall penalty trace is the closed form
:math:`\\operatorname{rank}_k / \\lambda_k` and the REML log-pseudo-determinant is
a per-block sum -- no eigendecomposition.

The fit reuses the :func:`nitrix.stats.gp.gp_fit` HSGP eigenstructure and is
cuSOLVER-free and ``vmap`` clean; the working size is
:math:`O(V (M_0 + (1 + L) m)^2)` -- the factor-smooth interaction is :math:`L`
times wider than a plain GP, the inherent cost of per-group curves (bound it with
``block`` on brain-scale :math:`V`).

References
----------
- Pedersen, E. J., Miller, D. L., Simpson, G. L. & Ross, N. (2019). Hierarchical
  generalized additive models in ecology: an introduction with mgcv. PeerJ 7,
  e6876. https://doi.org/10.7717/peerj.6876
- Wood, S. N. & Fasiolo, M. (2017). A generalized Fellner-Schall method for
  smoothing parameter optimization with application to Tweedie location, scale
  and shape models. Biometrics 73(4), 1071-1081.
  https://doi.org/10.1111/biom.12666
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
from ._hsgp import _hsgp_domain, _hsgp_eigen, _hsgp_features
from ._penreml import mb_fs, mb_quantities, mb_reml_nll
from ._result import register_result
from .gp import _parabolic_argmin_jax

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
        'kernel',
        'engine',
        'model',
        'n_levels',
        'n_obs',
        'rank',
        'n_fixed',
        'lo',
        'hi',
        'boundary',
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
        :math:`p = M_0 + m + L m`; ``'nested'``:
        :math:`p = M_0 + m + (L_1 + L_2) m`.
    cov_unscaled
        ``(V, p, p)`` Bayesian covariance
        :math:`(X^{\\top} X + S_\\lambda)^{-1}`.
    theta
        ``(V, K + 2)`` array
        :math:`[\\log \\sigma_{\\mathrm{pop}}^2, (\\log
        \\sigma_{\\mathrm{grp},i}^2)\\ldots, \\log \\sigma_e^2, \\log \\rho]`
        -- one GP variance per component (:math:`K = 2` for ``'GS'``, ``3`` for
        ``'nested'``); the :math:`\\rho` column is constant (one shared
        lengthscale).
    log_mlik
        ``(V,)`` REML log marginal likelihood at the fit.
    edf
        ``(V, K)`` effective degrees of freedom of each GP component (population,
        then each grouping level).
    dispersion
        ``(V,)`` residual variance :math:`\\sigma_e^2`.
    kernel, engine, model
        Kernel name, reduced-rank engine (``'hsgp'``), and hierarchical model
        (``'GS'`` -- global + group smoothers; ``'nested'`` -- two-level g1/g2).
    n_levels
        Number of factor levels: :math:`L` (``'GS'``) or the tuple
        :math:`(L_1, L_2)` (``'nested'``).
    n_obs, rank, n_fixed
        :math:`N`, the per-smooth rank :math:`m`, and the number of fixed columns
        :math:`M_0`.
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
    """Single-element hierarchical fit at a fixed lengthscale.

    Runs the multi-block Fellner-Schall smoothing-parameter update, forms the
    penalised-least-squares quantities, and computes the per-component effective
    degrees of freedom for one response vector.

    Parameters
    ----------
    c
        ``(p,)`` cross-product :math:`X^{\\top} y` for this element.
    g
        Scalar sum of squares :math:`y^{\\top} y` for this element.
    xtx
        ``(p, p)`` Gram matrix :math:`X^{\\top} X` (shared across elements).
    d_blocks
        ``(K, p)`` diagonal penalty weights, one row per variance component.
    ranks
        ``(K,)`` rank (column count) of each penalty block.
    log_pdets
        ``(K,)`` log pseudo-determinant of each penalty block.
    block_cols
        Tuple of ``(lo, hi)`` column slices, one per smooth, over which each
        component's effective degrees of freedom are accumulated.
    n
        Number of observations :math:`N`.
    p
        Total number of design columns.
    n_fixed
        Number of unpenalised fixed columns :math:`M_0`.
    n_outer
        Number of Fellner-Schall outer iterations.
    ridge
        Ridge added to the penalised Gram matrix for numerical stability.
    lam_floor, lam_ceil
        Lower and upper clamps on each smoothing parameter.

    Returns
    -------
    beta : Float[Array, ' p']
        Penalised coefficient estimate.
    v : Float[Array, 'p p']
        Bayesian covariance :math:`(X^{\\top} X + S_\\lambda)^{-1}`.
    lam : Float[Array, ' K']
        Fitted smoothing parameter per variance component.
    edf_blocks : Float[Array, ' K']
        Per-GP-component effective degrees of freedom (population, then each
        grouping level).
    phi : Float[Array, '']
        Residual variance (dispersion) :math:`\\sigma_e^2`.
    log_mlik : Float[Array, '']
        REML log marginal likelihood at the fit.
    """
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
    """Per-element REML deviance :math:`-2 l_R` at a fixed lengthscale.

    Fits this element's smoothing parameters by Fellner-Schall and returns its
    REML deviance; summed across elements this is the pooled objective minimised
    over the shared lengthscale.

    Parameters
    ----------
    c
        ``(p,)`` cross-product :math:`X^{\\top} y` for this element.
    g
        Scalar sum of squares :math:`y^{\\top} y` for this element.
    xtx
        ``(p, p)`` Gram matrix :math:`X^{\\top} X` (shared across elements).
    d_blocks
        ``(K, p)`` diagonal penalty weights, one row per variance component.
    ranks
        ``(K,)`` rank (column count) of each penalty block.
    log_pdets
        ``(K,)`` log pseudo-determinant of each penalty block.
    n
        Number of observations :math:`N`.
    p
        Total number of design columns.
    n_fixed
        Number of unpenalised fixed columns :math:`M_0`.
    n_outer
        Number of Fellner-Schall outer iterations.
    ridge
        Ridge added to the penalised Gram matrix for numerical stability.
    lam_floor, lam_ceil
        Lower and upper clamps on each smoothing parameter.

    Returns
    -------
    Float[Array, '']
        The REML deviance :math:`-2 l_R` for this element.
    """
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
    """Build the factor-smooth interaction design.

    Forms the interaction :math:`\\Phi(x) \\otimes \\operatorname{onehot}(group)`:
    each group's :math:`m`-column block holds the basis :math:`\\Phi` on that
    group's own rows and zero elsewhere, laid out contiguously as
    :math:`(N, L m)`.

    Parameters
    ----------
    phi
        ``(N, m)`` Hilbert-space eigenbasis evaluated at the covariate.
    group
        ``(N,)`` integer factor labels ``0 .. n_levels - 1``.
    n_levels
        Number of factor levels :math:`L`.

    Returns
    -------
    Float[Array, 'N Lm']
        The ``(N, L * m)`` block-sparse factor-smooth design.
    """
    onehot = jax.nn.one_hot(group, n_levels, dtype=phi.dtype)  # (N, L)
    # (N, L, m) -> (N, L*m): block g holds phi where group==g.
    inter = onehot[:, :, None] * phi[:, None, :]
    return inter.reshape(phi.shape[0], n_levels * phi.shape[1])


def _block_weights(
    inv_s: Float[Array, ' m'], n_fixed: int, level_counts: Tuple[int, ...]
) -> Tuple[
    Float[Array, 'K p'], Float[Array, ' K'], Tuple[Tuple[int, int], ...]
]:
    """Assemble the diagonal penalty blocks for the hierarchical design.

    Returns the :math:`K = 1 + \\mathrm{len}(level\\_counts)` diagonal penalty
    blocks laid out over the full :math:`p` columns, together with each block's
    rank and its column slice. Block 0 is the **population** smooth
    (:math:`1/s` on its :math:`m` columns); block :math:`i + 1` is the
    :math:`i`-th **factor-smooth** (:math:`1/s` tiled over its
    ``level_counts[i]`` groups). ``level_counts = (L,)`` is the GS model;
    :math:`(L_1, L_2)` is the nested two-level model. Every block is diagonal
    with disjoint columns.

    Parameters
    ----------
    inv_s
        ``(m,)`` inverse spectral weights :math:`1/s_j(\\rho)` for one smooth.
    n_fixed
        Number of leading unpenalised fixed columns :math:`M_0`.
    level_counts
        Factor-level counts per grouping factor: ``(L,)`` for the GS model or
        :math:`(L_1, L_2)` for the nested model.

    Returns
    -------
    d_blocks : Float[Array, 'K p']
        Diagonal penalty weights, one row per block, zero outside the block's
        own columns.
    ranks : Float[Array, ' K']
        Column count (rank) of each block.
    block_cols : tuple of (int, int)
        The ``(lo, hi)`` column slice of each block.
    """
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
        sqrt_lambda,
        kernel=kernel,
        rho=jnp.asarray(rho, dtype=dtype),
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
    """Fit a mass-univariate hierarchical Gaussian process.

    Fits, per element, a population smooth of the covariate ``x`` plus
    group-level smooth deviations that share a kernel -- the "GS" hierarchical
    GAM, using the reduced-rank Hilbert-space (HSGP) engine. The shared
    lengthscale is profiled over a pooled-REML grid and each variance component's
    smoothing parameter is fitted by the generalised Fellner-Schall step.

    Parameters
    ----------
    Y
        ``(V, N)`` responses (one row per element).
    x
        ``(N,)`` covariate the GP smooths *over* (a single covariate, **not** the
        full design matrix; linear covariates go to ``parametric``).
    group
        ``(N,)`` integer factor labels :math:`0 \\ldots L-1` (the grouping the
        smooth varies over -- subjects, sites, ...).  For ``model='nested'`` this
        is the **outer** factor.
    group_inner
        ``(N,)`` integer labels of the **inner** factor (nested within
        ``group``), :math:`0 \\ldots L_2-1` globally.  Required for
        ``model='nested'``.
    parametric
        Optional ``(N, q)`` unpenalised linear design (with the intercept).
    kernel
        Covariance kernel name (as :func:`nitrix.stats.gp.gp_fit`); enters only
        through the diagonal spectral reweighting.
    rank
        Per-smooth eigenfunction count :math:`m` (default ``12`` -- the design is
        :math:`1 + L` smooths wide, so a smaller rank than :func:`gp_fit
        <nitrix.stats.gp.gp_fit>` is usual).
    model
        Hierarchical structure.  ``'GS'`` (default): a global smoother plus
        group-level smoothers with one shared group wiggliness
        :math:`\\sigma_{\\mathrm{grp}}^2`.  ``'nested'``: a two-level hierarchy
        (outer ``g1`` / inner ``g2``) -- population + outer-group + inner-group
        (nested) GP deviations, three variance components
        (:math:`\\sigma_{\\mathrm{pop}}^2`, :math:`\\sigma_{\\mathrm{outer}}^2`,
        :math:`\\sigma_{\\mathrm{inner}}^2`) sharing :math:`\\rho`.
    boundary
        HSGP domain-extension factor (as :func:`nitrix.stats.gp.gp_fit`); must be
        ``>= 1.0``.
    bounds
        Optional ``(lo, hi)`` covariate domain; defaults to the range of ``x``.
    rho_bounds
        Optional ``(lo, hi)`` bounds of the shared-lengthscale search; defaults
        to a span-relative range.
    n_rho
        Number of grid points in the shared-lengthscale search.
    map_rho
        Optional lengthscale prior :math:`\\rho \\mapsto -\\log p(\\rho)` added to
        the pooled objective (e.g. a builder from :mod:`nitrix.stats.priors`).
    n_levels, n_levels_inner
        Outer / inner factor-level counts (default to ``max + 1``); pass when a
        level is absent so the block widths are stable.
    n_outer
        Number of Fellner-Schall iterations for the final per-element fit.
    n_search
        Number of Fellner-Schall iterations during the lengthscale search.
    ridge
        Ridge added to the penalised Gram matrix for numerical stability.
    lam_floor, lam_ceil
        Lower and upper clamps on each smoothing parameter.
    block
        Optional element-block size for the chunked ``vmap`` reduction (bounds
        peak memory on brain-scale :math:`V`).

    Returns
    -------
    HGPResult
        Per-element fit with coefficients, the log-variance parameters
        ``theta`` (per-component GP variances, residual variance and shared
        lengthscale), the REML marginal likelihood, the population / group
        effective degrees of freedom, and the dispersion.
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
    # NumPy (not jnp.min) so the concrete closed-over `group` does not concretise
    # a tracer under jax.jit -- the grouping is static, like the covariate domain.
    group_np = np.asarray(group)
    g_min, g_max = int(group_np.min()), int(group_np.max())
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
        gi_np = np.asarray(group_inner)
        gi_min, gi_max = int(gi_np.min()), int(gi_np.max())
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
                "hgp_fit: model='nested' requires globally-numbered inner "
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

    reps = (1,) + level_counts_t

    def _blocks(
        rho: Any,
    ) -> Tuple[Array, Array, Array, Tuple[Tuple[int, int], ...]]:
        """Penalty layout at ``rho``: ``(d_blocks, ranks, log_pdets, block_cols)``
        (``rho`` may be a Python float or a traced scalar)."""
        inv_s = _inv_s(sqrt_lambda, kernel, rho, Y.dtype)
        d_blocks, ranks, block_cols = _block_weights(
            inv_s, n_fixed, level_counts_t
        )
        log_pdets = jnp.asarray(
            [rep * jnp.sum(jnp.log(inv_s)) for rep in reps], dtype=Y.dtype
        )
        return d_blocks, ranks, log_pdets, block_cols

    # PF3: profile rho on-device with lax.map (mirrors gp_fit's HSGP search) --
    # the prior host Python loop forced n_rho separate device->host syncs.  The
    # pooled-NLL reduction is chunked by `block` (blocked_vmap(...).sum()): the
    # hierarchical design is (1 + sum(level_counts)) * m wide, so an un-chunked
    # search over all V is the acuter OOM cliff.
    def _pooled_nll(log_rho: Float[Array, '']) -> Float[Array, '']:
        d_blocks, ranks, log_pdets, _ = _blocks(jnp.exp(log_rho))
        per = blocked_vmap(
            lambda c_v, g_v: _hgp_pooled_nll_one(
                c_v,
                g_v,
                xtx,
                d_blocks,
                ranks,
                log_pdets,
                n,
                p,
                n_fixed,
                n_search,
                ridge,
                lam_floor,
                lam_ceil,
            ),
            (c_all, g_all),
            block=block,
        )
        nll = jnp.sum(per)
        if map_rho is not None:
            nll = nll + 2.0 * map_rho(jnp.exp(log_rho))
        return nll

    log_rho_grid_j = jnp.asarray(log_rho_grid, dtype=Y.dtype)
    nll_grid = jax.lax.map(_pooled_nll, log_rho_grid_j)  # (n_rho,) on-device
    # Traceable rho refinement (mirrors gp_fit's HSGP epilogue): the JAX-native
    # parabolic argmin keeps rho_hat a traced scalar, so the shared Gaussian-HSGP
    # search runs under jax.jit / jax.vmap with the covariate domain and grouping
    # closed over (e.g. vmap-fit over datasets sharing a covariate / factor).
    log_rho_hat = _parabolic_argmin_jax(log_rho_grid_j, nll_grid)
    rho_hat = jnp.exp(log_rho_hat)

    # --- final per-element fit at rho_hat -----------------------------------
    d_blocks, ranks, log_pdets, block_cols = _blocks(rho_hat)

    def _final(c_v: Array, g_v: Array) -> Tuple[Array, ...]:
        return _hgp_fit_one(
            c_v,
            g_v,
            xtx,
            d_blocks,
            ranks,
            log_pdets,
            block_cols,
            n,
            p,
            n_fixed,
            n_outer,
            ridge,
            lam_floor,
            lam_ceil,
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
    log_rho_col = jnp.full_like(sigma_e2, jnp.log(rho_hat))
    theta = jnp.stack(
        [jnp.log(jnp.clip(s, 1e-30, None)) for s in sigma_gp2]
        + [jnp.log(jnp.clip(sigma_e2, 1e-30, None)), log_rho_col],
        axis=-1,
    )
    n_levels_aux: Any = tuple(level_counts) if nested else int(level_counts[0])
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

    Returns
    -------
    mean : Float[Array, '...']
        Posterior mean curve. ``(V, g)`` for the population smooth
        (``levels=None``), or ``(V, len(levels), g)`` for the requested group
        curves.
    std : Float[Array, '...']
        Posterior standard deviation of the curve, matching the shape of
        ``mean``.
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
    beta_pop = result.coef[:, mf : mf + m]  # (V, m)
    pop_smooth = beta_pop @ phi_new.T  # (V, g)
    pop_mean = fixed_part + pop_smooth

    if levels is None:
        x_design = jnp.concatenate(
            [t_new, phi_new, jnp.zeros((gsz, n_group_cols), dtype=dtype)],
            axis=1,
        )
        var = jnp.einsum(
            'gi,vij,gj->vg', x_design, result.cov_unscaled, x_design
        )
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
        grp_cols = (phi_new[:, None, :] * onehot[None, :, None]).reshape(
            gsz, n_lev * m
        )  # (g, L*m)
        x_design = jnp.concatenate([t_new, phi_new, grp_cols], axis=1)
        var = jnp.einsum(
            'gi,vij,gj->vg', x_design, result.cov_unscaled, x_design
        )
        std = jnp.sqrt(jnp.clip(result.dispersion[:, None] * var, 1e-30, None))
        return mean, std

    means, stds = jax.vmap(_one_level)(levels)  # (L_sel, V, g)
    return jnp.transpose(means, (1, 0, 2)), jnp.transpose(stds, (1, 0, 2))
