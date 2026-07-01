# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
r"""
Information-theoretic and variance-ratio image-similarity metrics.

The cross-modal workhorses (T1 <-> T2, EPI <-> T1, ...), where intensities
are related by an unknown -- possibly non-monotonic -- mapping:

- :func:`joint_histogram` -- a differentiable (Parzen / linear soft-binned)
  joint intensity histogram.
- :func:`mutual_information` -- mutual information (and normalised mutual
  information) from the joint histogram.
- :func:`mi_grad` -- the closed-form Mattes gradient
  :math:`\partial\mathrm{MI}/\partial\mathrm{moving}`, an analytic gradient
  that replaces the ``jax.grad(mutual_information)`` autodiff tape.
- :func:`correlation_ratio` -- Roche's :math:`\eta^2` between-group variance
  ratio, which assumes only a *functional* (not affine) intensity
  relationship.

Differentiability is via the smooth soft-binning weights (the bin *index* is
piecewise-constant, but its gradient is zero almost everywhere; the gradient
flows through the interpolation weights).  All return a scalar.  By
convention, mutual information, normalised mutual information, and the
correlation ratio are *similarities* -- maximise them, or minimise the
negative (or the :math:`1 - \eta^2` complement of the correlation ratio).
"""

from __future__ import annotations

from typing import Optional

import jax.numpy as jnp
from jaxtyping import Array, Float

from .._internal.backend import default_backend_is_gpu
from .._internal.config import resolve_driver
from ._common import _resolve_range, _soft_bin

__all__ = [
    'joint_histogram',
    'mutual_information',
    'mi_grad',
    'nmi_grad',
    'correlation_ratio',
]

# Span floor for the moving-axis bin scale ``s_m = (bins-1)/span`` in
# ``mi_grad``.  Bounds the force on a *degenerate* (near-uniform) unpinned
# range -- a real or pinned range is orders of magnitude above it, where
# ``s_m`` stays exactly consistent with ``_soft_bin``'s own binning; the
# recommended pinned-range path makes it moot.
_MI_SPAN_FLOOR = 1e-6

# Voxel-count gate (E2) for the deterministic one-hot-matmul joint histogram,
# used **on GPU only** (see the branch).  Below this the soft bins are
# materialised as two ``(N, bins)`` soft one-hot matrices whose product is the
# histogram -- a matmul reduction, which is **deterministic on GPU**, unlike the
# ``.at[].add`` scatter (a non-associative atomic add whose float result depends
# on the run -- the affine-MI GPU non-determinism source).  Measured trade
# (L4): the matmul is ~2-6x *slower* than the scatter per isolated histogram,
# but the histogram is a small fraction of an MI evaluation, so end-to-end
# affine-MI is perf-neutral -- and it makes ``restarts=1`` deterministic,
# retiring the 4-6x ``restarts`` multiplier that was the only prior GPU
# determinism mitigation (the real win).  Above the gate the ``(N, bins)``
# one-hots would dominate memory, so the scatter takes over (the finest
# full-resolution level only; the coarse affine levels -- where the 12-DOF basin
# is decided -- stay deterministic).  ``bins=32`` => ~50 MB of fp32 one-hots at
# the cap.  **CPU keeps the scatter unconditionally**: it is already
# deterministic there (no atomic contention) and 5-20x faster than the one-hot.
_ONEHOT_HIST_MAX_VOXELS = 200_000


def _joint_hist_from_softbins(
    lower_m: Array,
    frac_m: Array,
    lower_f: Array,
    frac_f: Array,
    bins: int,
    dtype: jnp.dtype,
    weight: Optional[Array] = None,
    driver: str = 'auto',
) -> Float[Array, 'bins bins']:
    r"""Accumulate pre-computed soft bins into the normalised joint histogram.

    Factored out of :func:`joint_histogram` so that the cost
    (:func:`mutual_information`) and the closed-form gradient (:func:`mi_grad`)
    share one accumulation and one normalisation -- the gradient is then
    exactly the derivative of the table the cost sees.

    Two equivalent accumulations are available, selected by the ``driver``
    axis. ``'onehot'`` turns the soft bins into two ``(N, bins)`` soft one-hot
    matrices and forms the histogram as their product
    :math:`\mathrm{oh}_m^{\top} \, \mathrm{oh}_f` -- a matmul, which is
    deterministic on GPU. ``'scatter'`` is the ``.at[].add`` accumulation, a
    non-associative atomic add whose float result varies run-to-run on GPU
    (the source of affine mutual-information non-determinism there) but is
    deterministic on CPU. ``driver='auto'`` picks ``'onehot'`` on GPU below
    ``_ONEHOT_HIST_MAX_VOXELS`` voxels (above which the one-hots would dominate
    memory, so the scatter takes over) and ``'scatter'`` on CPU (already
    deterministic and faster there). :func:`nitrix.reproducible` forces the
    canonical ``'onehot'`` for determinism at any size -- note this costs
    :math:`O(N \cdot \mathrm{bins})` memory. Both the platform and ``N`` are
    static under ``jit``, so the branch is resolved at trace time.

    Parameters
    ----------
    lower_m, frac_m
        Lower bin index and fractional weight for each moving-image sample,
        as returned by the soft-binning routine. ``lower_m`` has shape
        ``(N,)`` (integer bin) and ``frac_m`` has shape ``(N,)`` (weight in
        ``[0, 1]`` toward ``lower_m + 1``).
    lower_f, frac_f
        The corresponding lower bin index and fractional weight for the fixed
        image, each of shape ``(N,)``.
    bins
        Number of bins per axis; the returned histogram is ``(bins, bins)``.
    dtype
        Floating dtype of the accumulated histogram.
    weight
        Optional per-voxel non-negative weight (the domain mask), same length
        ``(N,)`` as the soft-bin index vectors, gating the accumulation: a
        masked-out voxel contributes nothing to the histogram. ``None``
        (default) is the unweighted form.
    driver
        Accumulation recipe: ``'onehot'`` (deterministic matmul),
        ``'scatter'`` (atomic add), or ``'auto'`` (default; resolved from the
        platform and voxel count as described above).

    Returns
    -------
    Float[Array, 'bins bins']
        Joint probability table, normalised to sum to 1, with rows indexing
        the moving image and columns the fixed image.
    """
    n = lower_m.shape[0]
    resolved = resolve_driver(
        driver,
        op='metrics.joint_histogram',
        fast=lambda: (
            'onehot'
            if default_backend_is_gpu() and n <= _ONEHOT_HIST_MAX_VOXELS
            else 'scatter'
        ),
    )
    if resolved == 'onehot':
        idx = jnp.arange(bins)
        oh_m = (
            (1.0 - frac_m)[:, None] * (lower_m[:, None] == idx)
            + frac_m[:, None] * ((lower_m + 1)[:, None] == idx)
        ).astype(dtype)
        oh_f = (
            (1.0 - frac_f)[:, None] * (lower_f[:, None] == idx)
            + frac_f[:, None] * ((lower_f + 1)[:, None] == idx)
        ).astype(dtype)
        if weight is not None:
            oh_m = oh_m * weight[:, None]
        hist = oh_m.T @ oh_f
    else:
        hist = jnp.zeros((bins, bins), dtype=dtype)
        for idx_m, w_m in ((lower_m, 1.0 - frac_m), (lower_m + 1, frac_m)):
            for idx_f, w_f in ((lower_f, 1.0 - frac_f), (lower_f + 1, frac_f)):
                contrib = w_m * w_f if weight is None else weight * w_m * w_f
                hist = hist.at[idx_m, idx_f].add(contrib)
    return hist / jnp.maximum(hist.sum(), 1e-12)


def joint_histogram(
    moving: Float[Array, '...'],
    fixed: Float[Array, '...'],
    *,
    bins: int = 32,
    range_moving: Optional[tuple[float, float]] = None,
    range_fixed: Optional[tuple[float, float]] = None,
    mask: Optional[Float[Array, '...']] = None,
) -> Float[Array, 'bins bins']:
    """Differentiable joint intensity histogram (Parzen soft-binning).

    Each voxel contributes its bilinear soft weight to the four
    surrounding ``(moving, fixed)`` bin pairs; the result is normalised
    to a joint probability ``P`` summing to 1.

    Parameters
    ----------
    moving, fixed
        Images of identical shape.
    bins
        Number of bins per axis (default 32).
    range_moving, range_fixed
        Optional ``(lo, hi)`` intensity range per image.  ``None``
        (default) uses the per-image data min / max -- convenient, but
        the bounds are then piecewise-constant in the inputs; pin an
        explicit range for a binning that is stable across optimisation
        steps.
    mask
        Optional non-negative per-voxel weight (same shape as the images)
        gating the scatter: an out-of-mask voxel contributes nothing to the
        joint distribution, so a downstream MI / CR ignores it.

    Returns
    -------
    Joint probability table, ``(bins, bins)``, rows indexing ``moving``
    and columns ``fixed``.

    Notes
    -----
    The soft weight is **linear** (order-1 B-spline / triangular
    Parzen, support fixed at two bins): there is no temperature /
    bandwidth knob, so it does **not** reduce to hard (nearest-bin)
    binning at a fixed ``bins`` under any parameter limit.  It
    coincides with hard binning only when every sample lands on a bin
    centre; as ``bins`` grows both converge to the same continuous
    distribution.
    """
    m = moving.reshape(-1)
    f = fixed.reshape(-1)
    lo_m, hi_m = _resolve_range(m, range_moving)
    lo_f, hi_f = _resolve_range(f, range_fixed)
    lower_m, frac_m = _soft_bin(m, bins, lo_m, hi_m)
    lower_f, frac_f = _soft_bin(f, bins, lo_f, hi_f)
    weight = None if mask is None else mask.reshape(-1)
    return _joint_hist_from_softbins(
        lower_m, frac_m, lower_f, frac_f, bins, m.dtype, weight
    )


def mutual_information(
    moving: Float[Array, '...'],
    fixed: Float[Array, '...'],
    *,
    bins: int = 32,
    range_moving: Optional[tuple[float, float]] = None,
    range_fixed: Optional[tuple[float, float]] = None,
    normalized: bool = False,
    eps: float = 1e-10,
    mask: Optional[Float[Array, '...']] = None,
) -> Float[Array, '']:
    r"""Mutual information (or normalised mutual information) of two images.

    Computes
    :math:`\mathrm{MI} = \sum_{i,j} P(i,j) \log\!\left(P(i,j) / (P_m(i)\,P_f(j))\right)`
    over the soft joint histogram. With ``normalized=True`` returns
    Studholme's normalised mutual information
    :math:`(H_m + H_f) / H_{mf}` (in ``[1, 2]``; invariant to overlap size).
    Higher is more similar; use the negation as a cost.

    Parameters
    ----------
    moving, fixed
        Images of identical shape.
    bins
        Number of bins per axis (default 32), forwarded to
        :func:`joint_histogram`.
    range_moving, range_fixed
        Optional ``(lo, hi)`` intensity range per image, forwarded to
        :func:`joint_histogram`. ``None`` (default) uses the per-image data
        min / max.
    normalized
        If ``True``, return Studholme's normalised mutual information
        :math:`(H_m + H_f) / H_{mf}` instead of the raw mutual information.
    eps
        Small constant guarding the logarithms and quotient (default
        ``1e-10``).
    mask
        Optional non-negative per-voxel weight (same shape as the images)
        gating the joint-histogram scatter: an out-of-mask voxel is excluded
        from the distribution.

    Returns
    -------
    Float[Array, '']
        Scalar mutual information (or normalised mutual information).

    Notes
    -----
    Built on the linear (order-1 B-spline) Parzen histogram, so it is
    differentiable and in the same B-spline-Parzen family as ITK
    ``MattesMutualInformation`` (order-3 cubic) -- distinct from the hard
    (order-0) binning of ``sklearn.mutual_info_score``. At a fixed ``bins``
    the three give different numbers (Parzen smoothing biases mutual
    information downward at coarse bins); all converge to the true continuous
    mutual information in the fine-bin limit. The domain tools are divergent
    references, not bit-exact oracles, here.
    """
    hist = joint_histogram(
        moving,
        fixed,
        bins=bins,
        range_moving=range_moving,
        range_fixed=range_fixed,
        mask=mask,
    )
    p_m = hist.sum(axis=1)
    p_f = hist.sum(axis=0)
    if normalized:
        h_m = -jnp.sum(jnp.where(p_m > 0, p_m * jnp.log(p_m + eps), 0.0))
        h_f = -jnp.sum(jnp.where(p_f > 0, p_f * jnp.log(p_f + eps), 0.0))
        h_mf = -jnp.sum(jnp.where(hist > 0, hist * jnp.log(hist + eps), 0.0))
        return (h_m + h_f) / (h_mf + eps)
    outer = p_m[:, None] * p_f[None, :]
    ratio = jnp.where(hist > 0, hist / (outer + eps), 1.0)
    return jnp.sum(jnp.where(hist > 0, hist * jnp.log(ratio), 0.0))


def mi_grad(
    moving: Float[Array, '...'],
    fixed: Float[Array, '...'],
    *,
    bins: int = 32,
    range_moving: Optional[tuple[float, float]] = None,
    range_fixed: Optional[tuple[float, float]] = None,
    sample_stride: int = 1,
) -> Float[Array, '...']:
    r"""Closed-form Mattes mutual-information gradient.

    The analytic per-voxel derivative of :func:`mutual_information`
    (unnormalised) with respect to the moving intensities,
    :math:`\partial\mathrm{MI}/\partial\mathrm{moving}`, replacing the autodiff
    tape ``jax.grad(mutual_information)``. Differentiating
    :math:`\mathrm{MI} = \sum P \log(P / (P_m P_f))` through only the moving
    soft-bin weights, the :math:`+1` and :math:`\log P_f` terms cancel (the
    soft weights sum to one, so :math:`\sum_a \partial\beta_m/\partial m = 0`),
    leaving the Mattes (2003) form

    .. math::

        \frac{\partial\mathrm{MI}}{\partial m}(x)
        = \frac{s_m}{N}\left[(1 - t_f)\, D[k,l] + t_f\, D[k,l+1]\right]

    where :math:`s_m = (\mathrm{bins} - 1) / \mathrm{span}_m`, :math:`k` and
    :math:`l` are the moving and fixed lower bins, :math:`t_f` the fixed
    fractional weight, and :math:`D[a,b] = W[a+1,b] - W[a,b]` is the forward
    difference (along the moving axis) of the reduced table
    :math:`W[a,b] = \log P[a,b] - \log P_m[a]`. This is a two-value gather into
    the tiny :math:`(\mathrm{bins} - 1, \mathrm{bins})` table :math:`D` per
    voxel -- no autodiff scatter tape.

    Parameters
    ----------
    moving, fixed
        Images of identical shape.
    bins
        Bins per axis (must match the cost; default 32).
    range_moving, range_fixed
        Optional ``(lo, hi)`` intensity ranges. **Pin** them (the recipe does
        so once from the full-resolution images): a data ``min/max`` range
        drifts as the moving image deforms (a non-stationary objective) and
        truncates the force at the clip boundary. ``None`` (default) uses the
        per-image data min / max.
    sample_stride
        Estimate the joint PDF from every ``sample_stride``-th voxel (ITK
        "Regular" sampling); the gradient is still applied densely. ``1``
        (default) is the exact full histogram; ``> 1`` trades a noisier PDF for
        a cheaper scatter -- about 0.98 cosine-aligned at stride 4 (~25 %) on
        real cross-modal data. Jit-static (a Python int).

    Returns
    -------
    Float[Array, '...']
        The gradient :math:`\partial\mathrm{MI}/\partial\mathrm{moving}`, same
        shape as ``moving``.

    Notes
    -----
    The empty-bin masking (``where(P > 0, ...)``) is deliberately identical to
    that of :func:`mutual_information`, so :func:`mi_grad` matches
    ``jax.grad(mutual_information)`` to tolerance on populated bins (the parity
    oracle); the derivative at an exactly-empty bin is a documented divergence
    (the mask is non-smooth there -- the analogue of :func:`lncc_grad`'s
    boundary divergence). Voxels outside ``[lo_m, hi_m]`` get zero force (the
    soft bin is clipped, so its derivative vanishes). This is the
    **unnormalised** Mattes mutual information only -- the normalised
    (Studholme) gradient is :func:`nmi_grad` (the quotient-rule form).

    References
    ----------
    Mattes, D., Haynor, D. R., Vesselle, H., Lewellen, T. K., & Eubank, W.
    (2003). PET-CT image registration in the chest using free-form
    deformations. *IEEE Transactions on Medical Imaging*, 22(1), 120-128.
    https://doi.org/10.1109/TMI.2003.809072
    """
    m = moving.reshape(-1)
    f = fixed.reshape(-1)
    lo_m, hi_m = _resolve_range(m, range_moving)
    lo_f, hi_f = _resolve_range(f, range_fixed)
    s_m = (bins - 1) / jnp.maximum(hi_m - lo_m, _MI_SPAN_FLOOR)
    k, frac_m = _soft_bin(m, bins, lo_m, hi_m)
    ll, frac_f = _soft_bin(f, bins, lo_f, hi_f)
    if sample_stride > 1:
        # ITK "Regular" sampling: estimate the joint PDF from a strided voxel
        # subset (the scatter is the cost; the PDF is a smooth global statistic
        # well estimated from a sample), then apply the gradient DENSELY.  At
        # 25% (stride 4) the force stays ~0.98 cos-aligned with the full one on
        # real cross-modal data, for a ~3x cheaper histogram.  ``sample_stride``
        # is jit-static (a Python int); a non-strided ``P`` (default) is exact.
        sub = slice(None, None, sample_stride)
        p = _joint_hist_from_softbins(
            k[sub], frac_m[sub], ll[sub], frac_f[sub], bins, m.dtype
        )
    else:
        p = _joint_hist_from_softbins(k, frac_m, ll, frac_f, bins, m.dtype)
    p_m = p.sum(axis=1)
    log_p = jnp.where(p > 0, jnp.log(p), 0.0)
    log_pm = jnp.where(p_m > 0, jnp.log(p_m), 0.0)
    w = jnp.where(p > 0, log_p - log_pm[:, None], 0.0)  # (bins, bins)
    d = w[1:, :] - w[:-1, :]  # (bins-1, bins): forward diff along moving axis
    g = (s_m / m.size) * ((1.0 - frac_f) * d[k, ll] + frac_f * d[k, ll + 1])
    g = jnp.where((m >= lo_m) & (m <= hi_m), g, 0.0)  # clip -> zero derivative
    return g.reshape(moving.shape)


def nmi_grad(
    moving: Float[Array, '...'],
    fixed: Float[Array, '...'],
    *,
    bins: int = 32,
    range_moving: Optional[tuple[float, float]] = None,
    range_fixed: Optional[tuple[float, float]] = None,
    sample_stride: int = 1,
    eps: float = 1e-10,
) -> Float[Array, '...']:
    r"""Closed-form normalised-mutual-information (Studholme) gradient.

    The analytic per-voxel derivative of :func:`mutual_information` with
    ``normalized=True`` (:math:`\mathrm{NMI} = (H_m + H_f) / H_{mf}`) with
    respect to the moving intensities,
    :math:`\partial\mathrm{NMI}/\partial\mathrm{moving}`, the normalised
    analogue of :func:`mi_grad` (the quotient-rule form). As in the Mattes
    derivation, the gradient flows through only the moving soft-bin weights, so
    the fixed marginal entropy :math:`H_f` is invariant
    (:math:`\sum_a \partial\beta_m/\partial m = 0`). The quotient rule over
    :math:`H_m` (which varies) and :math:`H_{mf}` then gives

    .. math::

        \frac{\partial\mathrm{NMI}}{\partial m}(x)
        = -\frac{s_m}{N\,H_{mf}}\left[D_m[k]
          - \mathrm{NMI}\left((1 - t_f)\, D_{mf}[k,l]
          + t_f\, D_{mf}[k,l+1]\right)\right]

    with :math:`s_m = (\mathrm{bins} - 1) / \mathrm{span}_m`; :math:`k` and
    :math:`l` the moving and fixed lower bins; :math:`t_f` the fixed fractional
    weight; :math:`D_m[a] = \log P_m[a+1] - \log P_m[a]` the moving-marginal
    forward difference; and :math:`D_{mf}[a,b] = \log P[a+1,b] - \log P[a,b]`
    the joint forward difference (along the moving axis). Setting
    :math:`\mathrm{NMI} \to 1` and the prefactor
    :math:`-1/(N\,H_{mf}) \to 1/N` recovers the same :math:`D_m` / :math:`D_{mf}`
    structure to which :func:`mi_grad` reduces
    :math:`\mathrm{MI} = H_m + H_f - H_{mf}`.

    Parameters
    ----------
    moving, fixed
        Images of identical shape.
    bins
        Bins per axis (must match the cost; default 32).
    range_moving, range_fixed
        Optional ``(lo, hi)`` intensity ranges. **Pin** them for a stationary
        objective, as for :func:`mi_grad`. ``None`` (default) uses the
        per-image data min / max.
    sample_stride
        Estimate the joint PDF from every ``sample_stride``-th voxel (ITK
        "Regular" sampling); the gradient is still applied densely. ``1``
        (default) is the exact full histogram. Jit-static (a Python int).
    eps
        Small constant guarding the logarithms and the :math:`H_{mf}`
        denominators (default ``1e-10``).

    Returns
    -------
    Float[Array, '...']
        The gradient :math:`\partial\mathrm{NMI}/\partial\mathrm{moving}` (an
        ascent direction, as normalised mutual information is a similarity),
        same shape as ``moving``.

    Notes
    -----
    The empty-bin ``where(... > 0, ...)`` masking matches the
    ``normalized=True`` branch of :func:`mutual_information`, so
    :func:`nmi_grad` matches ``jax.grad`` of ``mutual_information`` with
    ``normalized=True`` to tolerance on populated bins (the parity oracle); the
    boundary divergence at an exactly-empty bin is the same documented
    non-smoothness as :func:`mi_grad`. Voxels outside ``[lo_m, hi_m]`` get zero
    force.
    """
    m = moving.reshape(-1)
    f = fixed.reshape(-1)
    lo_m, hi_m = _resolve_range(m, range_moving)
    lo_f, hi_f = _resolve_range(f, range_fixed)
    s_m = (bins - 1) / jnp.maximum(hi_m - lo_m, _MI_SPAN_FLOOR)
    k, frac_m = _soft_bin(m, bins, lo_m, hi_m)
    ll, frac_f = _soft_bin(f, bins, lo_f, hi_f)
    if sample_stride > 1:
        sub = slice(None, None, sample_stride)
        p = _joint_hist_from_softbins(
            k[sub], frac_m[sub], ll[sub], frac_f[sub], bins, m.dtype
        )
    else:
        p = _joint_hist_from_softbins(k, frac_m, ll, frac_f, bins, m.dtype)
    p_m = p.sum(axis=1)
    p_f = p.sum(axis=0)
    # Entropies exactly as mutual_information(normalized=True) (same NMI value).
    h_m = -jnp.sum(jnp.where(p_m > 0, p_m * jnp.log(p_m + eps), 0.0))
    h_f = -jnp.sum(jnp.where(p_f > 0, p_f * jnp.log(p_f + eps), 0.0))
    h_mf = -jnp.sum(jnp.where(p > 0, p * jnp.log(p + eps), 0.0))
    nmi = (h_m + h_f) / (h_mf + eps)
    # Forward differences along the moving axis (empty-bin masked, as the cost).
    log_p = jnp.where(p > 0, jnp.log(p), 0.0)
    log_pm = jnp.where(p_m > 0, jnp.log(p_m), 0.0)
    d_mf = log_p[1:, :] - log_p[:-1, :]  # (bins-1, bins)
    d_m = log_pm[1:] - log_pm[:-1]  # (bins-1,)
    g_mf = (1.0 - frac_f) * d_mf[k, ll] + frac_f * d_mf[k, ll + 1]
    g = -(s_m / (m.size * (h_mf + eps))) * (d_m[k] - nmi * g_mf)
    g = jnp.where((m >= lo_m) & (m <= hi_m), g, 0.0)  # clip -> zero derivative
    return g.reshape(moving.shape)


def correlation_ratio(
    moving: Float[Array, '...'],
    fixed: Float[Array, '...'],
    *,
    bins: int = 32,
    range_fixed: Optional[tuple[float, float]] = None,
    eps: float = 1e-10,
    mask: Optional[Float[Array, '...']] = None,
) -> Float[Array, '']:
    r"""Roche's correlation ratio :math:`\eta^2(\mathrm{moving} \mid \mathrm{fixed})`.

    Soft-bins ``fixed`` into ``bins`` groups and measures how much of the
    variance of ``moving`` is explained by the (soft) group means:

    .. math::

        \eta^2 = \frac{\sum_k n_k (\mu_k - \mu)^2}{\sum (m - \mu)^2}.

    Lies in ``[0, 1]``: it is ``1`` when ``moving`` is a deterministic function
    of ``fixed`` (any functional relationship, not just affine), and ``0`` when
    the two are unrelated. The FSL ``mcflirt`` cost is :math:`1 - \eta^2`. It is
    asymmetric by construction (``fixed`` is the explanatory variable).

    Parameters
    ----------
    moving, fixed
        Images of identical shape. ``moving`` is the response and ``fixed`` the
        explanatory variable partitioned into groups.
    bins
        Number of groups into which ``fixed`` is soft-binned (default 32).
    range_fixed
        Optional ``(lo, hi)`` intensity range for ``fixed``, controlling its
        binning. ``None`` (default) uses the data min / max.
    eps
        Small constant guarding the group-count and total-variance
        denominators (default ``1e-10``).
    mask
        Optional non-negative per-voxel weight (same shape as the images)
        gating the per-group scatter **and** the global mean / total variance,
        so an out-of-mask voxel is excluded from :math:`\eta^2`.

    Returns
    -------
    Float[Array, '']
        Scalar correlation ratio :math:`\eta^2` in ``[0, 1]``.

    Notes
    -----
    FSL / Roche lineage; SimpleITK's registration framework ships no
    correlation-ratio metric, so there is no domain co-oracle (a NumPy fp64
    reimplementation is the only reference). Uses the same linear soft binning
    as :func:`joint_histogram` (differentiable).
    """
    m = moving.reshape(-1)
    f = fixed.reshape(-1)
    w = None if mask is None else mask.reshape(-1)
    lo_f, hi_f = _resolve_range(f, range_fixed)
    lower_f, frac_f = _soft_bin(f, bins, lo_f, hi_f)

    n_k = jnp.zeros((bins,), dtype=m.dtype)
    s_k = jnp.zeros((bins,), dtype=m.dtype)
    for idx_f, w_f in ((lower_f, 1.0 - frac_f), (lower_f + 1, frac_f)):
        wf = w_f if w is None else w * w_f
        n_k = n_k.at[idx_f].add(wf)
        s_k = s_k.at[idx_f].add(wf * m)

    if w is None:
        mu = m.mean()
        total = jnp.sum((m - mu) ** 2)
    else:
        sw = jnp.maximum(w.sum(), eps)
        mu = jnp.sum(w * m) / sw
        total = jnp.sum(w * (m - mu) ** 2)
    mu_k = s_k / (n_k + eps)
    between = jnp.sum(n_k * (mu_k - mu) ** 2)
    return between / (total + eps)
