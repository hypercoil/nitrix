# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Information-theoretic & variance-ratio image-similarity metrics.

The cross-modal workhorses (T1<->T2, EPI<->T1, ...), where intensities
are related by an unknown -- possibly non-monotonic -- mapping:

- ``joint_histogram`` -- a differentiable (Parzen / linear soft-binned)
  joint intensity histogram.
- ``mutual_information`` -- MI (and normalised MI) from the joint
  histogram.
- ``mi_grad`` -- the closed-form Mattes ``∂MI/∂moving`` (the analytic
  gradient that backs a Tier-1 ``register.MIForce``, replacing the
  ``jax.grad(mutual_information)`` autodiff tape).
- ``correlation_ratio`` -- Roche's η² between-group variance ratio
  (the FSL ``mcflirt`` default), which assumes only a *functional*
  (not affine) intensity relationship.

Differentiability is via the smooth soft-binning weights (the bin
*index* is piecewise-constant, but its gradient is zero almost
everywhere; the gradient flows through the interpolation weights).  All
return a scalar.  Convention: MI / NMI / CR are *similarities*
(maximise, or minimise the negative / the ``1 - CR`` complement).
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
    """Accumulate pre-computed soft bins into the normalised joint histogram.

    Factored out of :func:`joint_histogram` so the cost (``mutual_information``)
    and the closed-form gradient (:func:`mi_grad`) share one accumulation and
    one normalisation -- the gradient is then exactly the derivative of the table
    the cost sees.

    Two equivalent accumulations, selected by the ``driver`` axis (the divergent
    op ``metrics.joint_histogram``): ``'onehot'`` turns the soft bins into two
    ``(N, bins)`` soft one-hot matrices and forms the histogram as their product
    ``ohₘᵀ @ ohf`` -- a matmul (**deterministic**); ``'scatter'`` is the
    ``.at[].add`` accumulation (a non-associative atomic add on GPU -- the
    **run-to-run** MI non-determinism source that ``affine_register(restarts=)``
    otherwise papers over; deterministic on CPU).  ``driver='auto'`` picks
    ``'onehot'`` on GPU below ``_ONEHOT_HIST_MAX_VOXELS`` voxels (above which the
    one-hots would dominate memory, so the scatter takes over) and ``'scatter'``
    on CPU (already deterministic and faster there).  ``nitrix.reproducible()``
    forces the canonical ``'onehot'`` for determinism on **any** size -- note
    this is ``O(N·bins)`` memory, the cost of reproducibility at scale.  Both the
    platform and ``N`` are static under ``jit``, so the branch is resolved at
    trace time.

    ``weight`` (optional, the per-voxel domain mask, same length as the soft-bin
    index vectors) gates the accumulation: a masked-out voxel contributes nothing
    to the histogram (A3).  ``None`` is the unweighted form.
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
    """Mutual information (or normalised MI) of two images.

    ``MI = Σ P(i,j) log( P(i,j) / (P_m(i) P_f(j)) )`` over the soft
    joint histogram.  With ``normalized=True`` returns Studholme's NMI
    ``(H_m + H_f) / H_mf`` (in ``[1, 2]``; invariant to overlap size).
    Higher is more similar; use ``-mutual_information`` as a cost.

    ``bins`` / ``range_*`` are forwarded to ``joint_histogram``;
    ``eps`` guards the logs.  ``mask`` (optional per-voxel weight) gates the
    joint-histogram scatter (an out-of-mask voxel is excluded from the
    distribution -- the correct masking point for a histogram statistic).

    Notes
    -----
    Built on the linear (order-1 B-spline) Parzen histogram, so it is
    differentiable and in the **same B-spline-Parzen family** as ITK
    ``MattesMutualInformation`` (order-3 cubic) -- distinct from the
    hard (order-0) binning of ``sklearn.mutual_info_score``.  At a
    fixed ``bins`` the three give different numbers (Parzen smoothing
    biases MI downward at coarse bins); all converge to the true
    continuous MI in the fine-bin limit.  The domain tools are
    *divergent references*, not bit-oracles, here.
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
    """Closed-form ``∂MI/∂moving`` -- the Mattes mutual-information gradient.

    The analytic per-voxel derivative of :func:`mutual_information`
    (unnormalised) w.r.t. the moving intensities, replacing the autodiff tape
    (``jax.grad(mutual_information)``).  Differentiating ``MI = Σ P log(P /
    (P_m P_f))`` through **only** the moving soft-bin weights, the ``+1`` and
    ``log P_f`` terms cancel (the soft weights sum to one, so ``Σ_a ∂β_m/∂m =
    0``), leaving the Mattes (2003) form

    ``∂MI/∂m(x) = (s_m / N) · [(1−t_f)·D[k,l] + t_f·D[k,l+1]]``

    where ``s_m = (bins−1)/span_m``, ``k``/``l`` are the moving/fixed lower
    bins, ``t_f`` the fixed fractional weight, and ``D[a,b] = W[a+1,b] − W[a,b]``
    is the forward difference (along the moving axis) of the reduced table
    ``W[a,b] = log(P[a,b]) − log(P_m[a])``.  A two-value gather into the tiny
    ``(bins−1, bins)`` ``D`` table per voxel -- no autodiff scatter tape.

    Parameters
    ----------
    moving, fixed
        Images of identical shape.
    bins
        Bins per axis (must match the cost).
    range_moving, range_fixed
        Intensity ranges.  **Pin** them (the recipe does so once from the
        full-resolution images): a data ``min/max`` range drifts as the moving
        image deforms (a non-stationary objective) and truncates the force at
        the clip boundary.
    sample_stride
        Estimate the joint PDF from every ``sample_stride``-th voxel (ITK
        "Regular" sampling); the gradient is still applied densely.  ``1``
        (default) is the exact full histogram; ``> 1`` trades a noisier PDF for
        a cheaper scatter (the histogram is the cost) -- ~0.98 cos-aligned at
        stride 4 (~25 %) on real cross-modal data.  Jit-static (a Python int).

    Returns
    -------
    ``∂MI/∂moving``, same shape as ``moving``.

    Notes
    -----
    The empty-bin masking (``where(P > 0, …)``) is deliberately identical to
    :func:`mutual_information`'s, so ``mi_grad == jax.grad(mutual_information)``
    to tolerance on populated bins (the parity oracle); the derivative at an
    exactly-empty bin is a documented divergence (the mask is non-smooth there
    -- the analogue of ``lncc_grad``'s boundary divergence).  Voxels outside
    ``[lo_m, hi_m]`` get zero force (the soft bin is clipped, so its derivative
    vanishes).  **Unnormalised** Mattes MI only -- the normalised (Studholme) NMI
    gradient is :func:`nmi_grad` (the quotient-rule form).
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
    """Closed-form ``∂NMI/∂moving`` -- the normalised-MI (Studholme) gradient.

    The analytic per-voxel derivative of :func:`mutual_information` with
    ``normalized=True`` (``NMI = (H_m + H_f) / H_mf``) w.r.t. the moving
    intensities, the NMI analogue of :func:`mi_grad` (the deferred quotient-rule
    form, now shipped).  As in the Mattes derivation the gradient flows through
    **only** the moving soft-bin weights, so the fixed marginal entropy ``H_f`` is
    invariant (``Σ_a ∂β_m/∂m = 0``).  The quotient rule over ``H_m`` (varies) and
    ``H_mf`` then gives

    ``∂NMI/∂m(x) = −(s_m / (N·H_mf)) · [ Dₘ[k] − NMI·((1−t_f)·D_mf[k,l]
                  + t_f·D_mf[k,l+1]) ]``

    with ``s_m = (bins−1)/span_m``; ``k``/``l`` the moving/fixed lower bins; ``t_f``
    the fixed fractional weight; ``Dₘ[a] = log P_m[a+1] − log P_m[a]`` the
    moving-marginal forward difference; and ``D_mf[a,b] = log P[a+1,b] − log P[a,b]``
    the joint forward difference (along the moving axis).  (Setting ``NMI -> 1`` and
    the prefactor ``-1/(N·H_mf) -> 1/N`` recovers the same ``Dₘ`` / ``D_mf``
    structure :func:`mi_grad` reduces ``MI = H_m + H_f − H_mf`` to.)

    Parameters mirror :func:`mi_grad` (``bins`` / ``range_*`` / ``sample_stride``;
    pin the ranges).  Returns ``∂NMI/∂moving`` (an **ascent** direction -- NMI is
    a similarity), same shape as ``moving``.

    Notes
    -----
    The empty-bin ``where(· > 0, …)`` masking matches
    :func:`mutual_information`'s ``normalized=True`` branch, so
    ``nmi_grad == jax.grad(mutual_information(normalized=True))`` to tolerance on
    populated bins (the parity oracle); the boundary divergence at an exactly-empty
    bin is the same documented non-smoothness as :func:`mi_grad`.  Voxels outside
    ``[lo_m, hi_m]`` get zero force.
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
    """Roche's correlation ratio ``η²(moving | fixed)``.

    Soft-bins ``fixed`` into ``bins`` groups and measures how much of
    ``moving``'s variance is explained by the (soft) group means:

    ``η² = Σ_k n_k (μ_k − μ)² / Σ (m − μ)²``.

    Lies in ``[0, 1]``: ``1`` when ``moving`` is a deterministic
    function of ``fixed`` (any functional relationship, not just
    affine), ``0`` when unrelated.  The FSL ``mcflirt`` cost is
    ``1 − η²``.  Asymmetric by construction (``fixed`` is the
    explanatory variable); ``bins`` / ``range_fixed`` control its
    binning.

    Notes
    -----
    FSL / Roche lineage; SimpleITK's registration framework ships no
    correlation-ratio metric, so there is no domain co-oracle (a numpy
    fp64 reimplementation is the only reference).  Uses the same linear
    soft binning as :func:`joint_histogram` (differentiable).  ``mask``
    (optional per-voxel weight) gates the per-group scatter **and** the global
    mean / total variance, so an out-of-mask voxel is excluded from η².
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
