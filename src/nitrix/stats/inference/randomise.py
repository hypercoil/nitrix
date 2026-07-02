# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Permutation inference for the general linear model.

:func:`permutation_test` is a non-parametric, family-wise-error-controlling
test of a general linear model (GLM) contrast over a statistic image, built
from the parts in this subpackage.  It is an on-device analogue of the FSL
``randomise`` tool.

The procedure
-------------

For data :math:`Y` (one response per observation per voxel), design :math:`X`,
and a contrast -- a ``(p,)`` t-contrast :math:`c` or a ``(m, p)`` F-contrast
:math:`C`:

1. Fit ordinary least squares and form the observed statistic image (t / F, or
   a variance-smoothed pseudo-t), then **enhance** it (TFCE / cluster / raw
   voxel).
2. For each of ``n_perm`` relabellings (sign flips or permutations, honouring
   exchangeability blocks), refit and re-enhance.  Nuisance regressors are
   handled by **Freedman-Lane**: permute the residuals of the reduced
   (nuisance-only) model and add the nuisance fit back, so only the tested
   effect is exchanged.
3. Accumulate, without storing every permutation: the **uncorrected** p-map
   (fraction of permutations whose voxel statistic :math:`\\geq` the observed)
   and the **FWE** p-map (fraction whose *spatial maximum* enhanced statistic
   :math:`\\geq` the observed enhanced value).  The first permutation is the
   identity, so the observed is included and :math:`p \\geq 1 / n_{perm}`.
   Optionally (``pvalue_method='gpd'``) the FWE p-value is read instead from a
   generalised-Pareto fit to the null-max tail, resolving below the
   :math:`1 / n_{perm}` floor at fewer permutations (see :func:`gpd_pvalue`).

The only linear solve is a shared ``(p, p)`` inverse, so no dense factorisation
library is required.  The enhancement is a non-differentiable inference kernel
(it forms discrete clusters); the returned maps are plain arrays.

References
----------
.. [1] Winkler AM, Ridgway GR, Douaud G, Nichols TE, Smith SM (2016). Faster
   permutation inference in brain imaging. *NeuroImage*, 141, 502-516.
   :doi:`10.1016/j.neuroimage.2016.05.068`
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Any, Literal, Optional, Tuple, cast

import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Bool, Float, Int

from ...linalg._smalllinalg import small_inv_logdet
from ...morphology import connected_components
from .._result import register_result
from .cluster import cluster_mass_map, cluster_size_map
from .permutation import permutations, sign_flips
from .tfce import tfce

__all__ = ['PermResult', 'gpd_pvalue', 'permutation_test']

Enhancement = Literal['tfce', 'voxel', 'cluster_extent', 'cluster_mass']
Exchange = Literal['sign', 'perm']
PValueMethod = Literal['empirical', 'gpd']

# A voxel whose observed residual variance is below this fraction of the
# typical in-mask variance is treated as degenerate (constant / no information)
# and excluded -- its SE-floored statistic would otherwise be a spurious +inf.
_VAR_REL_FLOOR = 1e-10


@register_result(
    children=('stat', 'enhanced', 'p_fwe', 'p_uncorrected', 'null_max'),
)
@dataclass(frozen=True)
class PermResult:
    """Permutation-test output (all maps in the input spatial shape).

    Attributes
    ----------
    stat
        The observed statistic image (t / pseudo-t).
    enhanced
        The observed enhanced image (TFCE / cluster / |t|).
    p_fwe
        Family-wise-error-corrected p-map (max-statistic null).
    p_uncorrected
        Uncorrected voxelwise permutation p-map.
    null_max
        ``(n_perm,)`` null distribution of the spatial-maximum enhanced
        statistic (the FWE reference distribution).
    """

    stat: Float[Array, '*spatial']
    enhanced: Float[Array, '*spatial']
    p_fwe: Float[Array, '*spatial']
    p_uncorrected: Float[Array, '*spatial']
    null_max: Float[Array, 'n_perm']


def _ols_pre(
    X: Float[Array, 'N p'],
) -> Tuple[Float[Array, 'p N'], Float[Array, '']]:
    """Shared ordinary-least-squares pieces of the design.

    Computes the coefficient pre-multiplier
    :math:`M = (X^{\\top} X)^{-1} X^{\\top}` (so that the fitted coefficients
    for a response :math:`y` are :math:`M y`) together with the inverse Gram
    matrix :math:`(X^{\\top} X)^{-1}`, which the caller combines with the
    contrast to form the contrast-independent variance factor.

    Parameters
    ----------
    X : Float[Array, 'N p']
        The ``(N, p)`` design matrix (``N`` observations, ``p`` regressors).

    Returns
    -------
    Float[Array, 'p N']
        The ``(p, N)`` coefficient pre-multiplier
        :math:`(X^{\\top} X)^{-1} X^{\\top}`.
    Float[Array, '']
        The ``(p, p)`` inverse Gram matrix :math:`(X^{\\top} X)^{-1}`.
    """
    p = X.shape[-1]
    xtx_inv, _ = small_inv_logdet(X.T @ X, p)
    return xtx_inv @ X.T, xtx_inv


def _residual_former(
    Z: Optional[Float[Array, 'N q']], n: int, dtype: Any
) -> Tuple[Float[Array, 'N N'], Float[Array, 'N N']]:
    """Reduced-model fit and residual-forming matrices for Freedman-Lane.

    Builds the two projection matrices used to strip the nuisance space out of
    the data before permuting.  The nuisance fit (hat) matrix is
    :math:`\\mathrm{fit}_z = Z (Z^{\\top} Z)^{-1} Z^{\\top}` and the
    residual-forming matrix is its complement
    :math:`\\mathrm{res}_z = I - \\mathrm{fit}_z`.  When ``Z`` is ``None`` there
    are no nuisance regressors, so :math:`\\mathrm{fit}_z = 0` and
    :math:`\\mathrm{res}_z = I` -- the data are permuted directly.

    Parameters
    ----------
    Z : Float[Array, 'N q'] or None
        The ``(N, q)`` nuisance design matrix, or ``None`` for the
        no-nuisance case.
    n : int
        The number of observations :math:`N`, fixing the ``(N, N)`` matrix size.
    dtype
        The floating dtype of the returned matrices.

    Returns
    -------
    Float[Array, 'N N']
        The ``(N, N)`` nuisance fit matrix
        :math:`\\mathrm{fit}_z = Z (Z^{\\top} Z)^{-1} Z^{\\top}` (all-zero when
        ``Z`` is ``None``).
    Float[Array, 'N N']
        The ``(N, N)`` residual-forming matrix
        :math:`\\mathrm{res}_z = I - \\mathrm{fit}_z` (the identity when ``Z``
        is ``None``).
    """
    eye = jnp.eye(n, dtype=dtype)
    if Z is None:
        return jnp.zeros((n, n), dtype=dtype), eye
    q = Z.shape[-1]
    ztz_inv, _ = small_inv_logdet(Z.T @ Z, q)
    fit_z = Z @ (ztz_inv @ Z.T)
    return fit_z, eye - fit_z


def permutation_test(
    data: Float[Array, '*spatial N'],
    design: Float[Array, 'N p'],
    contrast: Float[Array, 'p'] | Float[Array, 'm p'],
    *,
    key: Array,
    n_perm: int = 500,
    enhancement: Enhancement = 'tfce',
    exchange: Exchange = 'sign',
    nuisance: Optional[Float[Array, 'N q']] = None,
    blocks: Optional[Int[Array, 'N']] = None,
    mask: Optional[Bool[Array, '*spatial']] = None,
    two_sided: bool = True,
    var_smooth: Optional[float] = None,
    connectivity: int = 1,
    cluster_thresh: Optional[float] = None,
    tfce_E: float = 0.5,
    tfce_H: float = 2.0,
    tfce_steps: int = 100,
    pvalue_method: PValueMethod = 'empirical',
    gpd_n_exceedances: int = 250,
) -> PermResult:
    """Permutation test of a GLM t- or F-contrast with FWE control.

    Parameters
    ----------
    data
        ``(*spatial, N)`` responses (the observation axis is last).
    design
        ``(N, p)`` GLM design (includes the effect and any nuisance columns).
    contrast
        The tested contrast.  A ``(p,)`` vector ``c`` is a **t-contrast**
        (signed, two-sided by default); a ``(m, p)`` matrix ``C`` (``m > 1``)
        is an **F-contrast** -- the joint test ``C beta = 0`` reported as the
        non-negative F-statistic (``two_sided`` is then irrelevant and ignored,
        the F already captures both directions).
    key
        ``jax.random`` key (RNG policy is the caller's).
    n_perm
        Number of permutations incl. the identity (default ``500``).
    enhancement
        ``'tfce'`` (default), ``'voxel'`` (raw statistic, voxelwise FWE), or
        ``'cluster_extent'`` / ``'cluster_mass'`` (cluster-forming at
        ``cluster_thresh``).
    exchange
        ``'sign'`` (sign-flipping; symmetric / one-sample) or ``'perm'``
        (row permutation; two-sample / regression).
    nuisance
        Optional ``(N, q)`` nuisance design for Freedman-Lane.  ``None`` permutes
        the data directly (exact for designs with no nuisance).
    blocks
        Optional per-observation exchangeability-block labels.
    mask
        Optional spatial mask; out-of-mask voxels do not contribute to clusters
        or the maximum.
    two_sided
        Two-sided test (default ``True``).  Ignored for an F-contrast.
    var_smooth
        Optional Gaussian sigma for variance smoothing (pseudo-t).
    connectivity
        Spatial connectivity order used when forming clusters (for TFCE and the
        cluster-based enhancements).  Default ``1``.
    cluster_thresh
        Cluster-forming statistic threshold; **required** for
        ``enhancement='cluster_extent'`` / ``'cluster_mass'`` (the classic
        cluster-extent / cluster-mass FWE -- one threshold, far cheaper per
        permutation than TFCE).
    tfce_E, tfce_H, tfce_steps
        Threshold-free cluster enhancement parameters: the extent exponent
        :math:`E` (default ``0.5``), the height exponent :math:`H`
        (default ``2.0``), and the number of integration steps (default
        ``100``).  Only used when ``enhancement='tfce'``.
    pvalue_method
        ``'empirical'`` (default) -- the FWE p-map is the discrete fraction of
        permutations whose null max :math:`\\geq` the observed (floored at
        :math:`1 / n_{perm}`).  ``'gpd'`` -- fit a generalised Pareto
        distribution to the upper tail of the null-max distribution so FWE
        p-values resolve **below** the :math:`1 / n_{perm}` floor, letting
        ``n_perm`` drop.  Only the FWE map is GPD-smoothed; the uncorrected map
        stays empirical.
    gpd_n_exceedances
        Number of upper-tail null-max exceedances used for the GPD fit when
        ``pvalue_method='gpd'`` (default ``250``; clamped to ``n_perm - 1``).

    Returns
    -------
    PermResult
        A :class:`PermResult` holding the observed statistic and enhanced maps,
        the FWE and uncorrected p-maps, and the null-max distribution.
    """
    spatial = data.shape[:-1]
    n = data.shape[-1]
    p = design.shape[-1]
    v = prod(spatial)
    if design.shape[0] != n:
        raise ValueError(
            f'permutation_test: design has {design.shape[0]} rows; N={n}.'
        )
    if (
        enhancement in ('cluster_extent', 'cluster_mass')
        and cluster_thresh is None
    ):
        raise ValueError(
            f'permutation_test: enhancement={enhancement!r} needs a '
            'cluster_thresh (the cluster-forming statistic threshold).'
        )
    Y = data.reshape(v, n)
    c = jnp.asarray(contrast)
    is_f = c.ndim == 2
    if is_f and c.shape[-1] != p:
        raise ValueError(
            f'permutation_test: F-contrast has {c.shape[-1]} columns; p={p}.'
        )
    # An F-statistic is non-negative and already joint over the contrast rows,
    # so enhancement / the uncorrected comparison are one-sided regardless of
    # the (t-only) ``two_sided`` flag.
    enh_two_sided = two_sided and not is_f
    mask_flat = jnp.ones((v,), dtype=bool) if mask is None else mask.reshape(v)

    M, xtx_inv = _ols_pre(design)
    dof = float(n - p)
    if dof <= 0:
        raise ValueError(
            f'permutation_test: saturated design (n={n} <= p={p}); residual '
            f'dof={dof:.0f}, so the t/F statistics and their permutation null '
            'are undefined. Reduce the design rank or add observations.'
        )
    if is_f:
        m_rank = c.shape[0]
        # The (m, m) middle matrix C (X^T X)^{-1} C^T depends only on the design
        # and the contrast -- constant across permutations, so its inverse is
        # hoisted out of the per-permutation statistic.
        fmat_inv, _ = small_inv_logdet(c @ xtx_inv @ c.T, m_rank)
    else:
        c_var = c @ (xtx_inv @ c)  # scalar
    fit_z, res_z = _residual_former(nuisance, n, Y.dtype)
    e0 = Y @ res_z.T  # reduced-model residuals (V, N)
    base = Y @ fit_z.T  # nuisance fit to add back (V, N); 0 if no nuisance

    # Fold zero-variance (constant / degenerate) voxels into the mask: their
    # statistic is undefined and the SE floor would otherwise inflate it into a
    # spurious maximum that corrupts the max-statistic null.  Identified once
    # from the observed data (relative to the typical in-mask variance) and
    # excluded everywhere -- clustering, the max statistic, and the p-maps.
    beta_obs = Y @ M.T
    resid_obs = Y - beta_obs @ design.T
    sigma2_obs = jnp.sum(resid_obs * resid_obs, axis=-1) / dof
    var_scale = jnp.max(jnp.where(mask_flat, sigma2_obs, 0.0))
    mask_flat = mask_flat & (sigma2_obs > _VAR_REL_FLOOR * var_scale)

    def statistic(Yp: Float[Array, 'V N']) -> Float[Array, 'V']:
        beta = Yp @ M.T  # (V, p)
        resid = Yp - beta @ design.T
        sigma2 = jnp.sum(resid * resid, axis=-1) / dof
        if var_smooth is not None:
            sigma2 = jnp.reshape(
                _smooth(jnp.reshape(sigma2, spatial), var_smooth), (v,)
            )
        if is_f:
            cb = beta @ c.T  # (V, m)
            quad = jnp.einsum('vi,ij,vj->v', cb, fmat_inv, cb)
            return quad / (m_rank * jnp.clip(sigma2, 1e-30, None))
        effect = beta @ c
        se = jnp.sqrt(jnp.clip(sigma2 * c_var, 1e-30, None))
        return effect / se

    def _cluster_side(s: Float[Array, '*spatial']) -> Float[Array, '*spatial']:
        thr = cast(float, cluster_thresh)  # validated non-None above
        labels = connected_components(s > thr, connectivity=connectivity)
        if enhancement == 'cluster_extent':
            return cluster_size_map(labels)
        return cluster_mass_map(labels, s, thr)

    def enhance(stat_v: Float[Array, 'V']) -> Float[Array, 'V']:
        stat_v = jnp.where(mask_flat, stat_v, 0.0)
        if enhancement == 'voxel':
            out = (
                jnp.abs(stat_v)
                if enh_two_sided
                else jnp.clip(stat_v, 0.0, None)
            )
            return out
        spatial_stat = jnp.reshape(stat_v, spatial)
        if enhancement == 'tfce':
            enhanced = tfce(
                spatial_stat,
                E=tfce_E,
                H=tfce_H,
                n_steps=tfce_steps,
                connectivity=connectivity,
                two_sided=enh_two_sided,
                mask=None if mask is None else mask,
            )
        else:  # cluster_extent / cluster_mass at a fixed forming threshold
            enhanced = _cluster_side(spatial_stat)
            if enh_two_sided:
                enhanced = enhanced + _cluster_side(-spatial_stat)
        return jnp.reshape(enhanced, (v,))

    # Relabellings (row 0 = identity).
    if exchange == 'sign':
        ops: Array = sign_flips(n, n_perm, key, blocks=blocks)
    else:
        ops = permutations(n, n_perm, key, blocks=blocks)

    def per_perm(
        op: Array,
    ) -> Tuple[Float[Array, 'V'], Float[Array, 'V'], Float[Array, '']]:
        if exchange == 'sign':
            Yp = base + e0 * op[None, :]
        else:
            Yp = base + e0[:, op]
        stat_p = statistic(Yp)
        enhanced_p = enhance(stat_p)
        return (
            stat_p,
            enhanced_p,
            jnp.max(jnp.where(mask_flat, enhanced_p, -jnp.inf)),
        )

    # The observed result IS permutation 0 (the identity).  We capture its
    # statistic / enhanced map from *inside* the scan (iteration 0), so the
    # reference used for the comparisons is bit-identical to the identity
    # permutation's own enhanced map -- guaranteeing the identity contributes a
    # ``+1`` everywhere and ``p_fwe >= 1 / n_perm``.  (TFCE is discretely
    # sensitive to float reassociation, so a separately-compiled observed path
    # would not match the scan body.)
    Carry = Tuple[
        Float[Array, 'V'],
        Float[Array, 'V'],
        Float[Array, 'V'],
        Float[Array, 'V'],
    ]

    def scan_body(
        carry: Carry, step: Tuple[Array, Array]
    ) -> Tuple[Carry, Float[Array, '']]:
        i, op = step
        fwe_count, unc_count, stat_obs, enhanced_obs = carry
        stat_p, enhanced_p, m = per_perm(op)
        is_first = i == 0
        stat_obs = jnp.where(is_first, stat_p, stat_obs)
        enhanced_obs = jnp.where(is_first, enhanced_p, enhanced_obs)
        # FWE: the spatial-max enhanced statistic vs the observed enhanced map.
        fwe_count = fwe_count + (m >= enhanced_obs)
        # Uncorrected: the *raw* per-voxel statistic vs the observed (FSL
        # convention) -- two-sided compares magnitudes.  Not the enhanced value.
        # (An F-statistic is already one-sided, so ``enh_two_sided`` is False.)
        cstat_p = jnp.abs(stat_p) if enh_two_sided else stat_p
        cstat_obs = jnp.abs(stat_obs) if enh_two_sided else stat_obs
        unc_count = unc_count + (cstat_p >= cstat_obs)
        return (fwe_count, unc_count, stat_obs, enhanced_obs), m

    zeros = jnp.zeros((v,), Y.dtype)
    init: Carry = (zeros, zeros, zeros, zeros)
    (fwe_count, unc_count, stat_obs, enhanced_obs), null_max = lax.scan(
        scan_body, init, (jnp.arange(n_perm), ops)
    )

    if pvalue_method == 'gpd':
        # GPD-smooth only the FWE map (the max-statistic tail); the uncorrected
        # voxelwise map stays empirical.
        p_fwe_flat = gpd_pvalue(
            enhanced_obs, null_max, n_exceedances=gpd_n_exceedances
        )
    else:
        p_fwe_flat = fwe_count / n_perm
    p_fwe = jnp.where(mask_flat, p_fwe_flat, 1.0)
    p_unc = jnp.where(mask_flat, unc_count / n_perm, 1.0)
    # Zero the observed statistic at excluded voxels (out-of-mask or degenerate)
    # so the returned map carries no SE-floor artifact.
    stat_obs = jnp.where(mask_flat, stat_obs, 0.0)

    return PermResult(
        stat=jnp.reshape(stat_obs, spatial),
        enhanced=jnp.reshape(enhanced_obs, spatial),
        p_fwe=jnp.reshape(p_fwe, spatial),
        p_uncorrected=jnp.reshape(p_unc, spatial),
        null_max=null_max,
    )


def gpd_pvalue(
    stat: Float[Array, '...'],
    null_dist: Float[Array, 'n'],
    *,
    n_exceedances: int = 250,
) -> Float[Array, '...']:
    """Tail-accelerated exceedance p-value via a generalised Pareto fit.

    The permutation FWE p-value of an observed statistic :math:`T` is the
    survival fraction :math:`P(\\max_{\\mathrm{null}} \\geq T)`.  Estimated
    empirically it is discrete and floored at :math:`1 / n` -- so resolving a
    small p-value needs a large :math:`n`.  Following Winkler et al. (2016;
    see the module references), the upper tail of ``null_dist`` above a
    threshold :math:`u` (the :math:`(k+1)`-th largest of :math:`n`, with
    :math:`k =` ``n_exceedances``) is modelled by a **generalised Pareto
    distribution** fitted by the method of moments, giving a smooth p-value
    that can fall below :math:`1 / n` at a fraction of the permutations:

    .. list-table::
       :widths: auto

       * - :math:`P(T) = (k / n)\\, S_{\\mathrm{GPD}}(T - u)`
         - :math:`T > u` (tail, GPD survival)
       * - :math:`P(T) = \\#\\{\\mathrm{null} \\geq T\\} / n`
         - :math:`T \\leq u` (body, empirical)

    The GPD shape :math:`\\xi` and scale :math:`\\sigma` are recovered from the
    exceedance mean and variance,
    :math:`\\xi = (1 - \\mathrm{mean}^2 / \\mathrm{var}) / 2` and
    :math:`\\sigma = \\mathrm{mean}\\,(1 - \\xi)`.  The routine is pure and
    differentiable, and requires no dense factorisation library.

    Parameters
    ----------
    stat : Float[Array, '...']
        The observed statistic(s) whose FWE p-value(s) are wanted.  A scalar or
        an array of arbitrary shape (e.g. an enhanced FWE statistic map).
    null_dist : Float[Array, 'n']
        The ``(n,)`` null distribution of the spatial-maximum statistic (one
        value per permutation).
    n_exceedances : int, optional
        The number of upper-tail exceedances :math:`k` used to fit the GPD
        (default ``250``).  Clamped to at most ``n - 1``.

    Returns
    -------
    Float[Array, '...']
        The FWE p-value(s), the same shape as ``stat``: the GPD-based tail
        estimate where ``stat`` exceeds the threshold :math:`u`, and the
        empirical fraction otherwise.
    """
    s = jnp.asarray(stat)
    null = jnp.asarray(null_dist)
    n = null.shape[0]
    k = min(int(n_exceedances), n - 1)
    srt = jnp.sort(null)  # ascending
    u = srt[n - k - 1]  # threshold: the (k+1)-th largest
    exc = srt[n - k :] - u  # the top-k exceedances (>= 0)
    ybar = jnp.mean(exc)
    s2 = jnp.var(exc)
    xi = 0.5 * (1.0 - ybar * ybar / jnp.clip(s2, 1e-30, None))
    sigma = jnp.clip(ybar * (1.0 - xi), 1e-30, None)
    t = s - u
    z = 1.0 + xi * t / sigma
    surv = jnp.where(
        jnp.abs(xi) < 1e-6,
        jnp.exp(-t / sigma),
        jnp.clip(z, 0.0, None) ** (-1.0 / xi),
    )
    p_tail = jnp.clip((k / n) * surv, 0.0, 1.0)
    # Empirical body: #{null >= s} / n via searchsorted (memory-light, no
    # (V, n) broadcast).  side='left' counts strictly-less, so n - idx = #{>=}.
    idx = jnp.searchsorted(srt, s, side='left')
    p_emp = (n - idx) / n
    return jnp.where(s > u, p_tail, p_emp)


def _smooth(
    x: Float[Array, '*spatial'], sigma: float
) -> Float[Array, '*spatial']:
    from ...smoothing import gaussian

    return gaussian(x, sigma=sigma)
