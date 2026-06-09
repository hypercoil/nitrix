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

from ._common import _resolve_range, _soft_bin

__all__ = [
    'joint_histogram',
    'mutual_information',
    'correlation_ratio',
]


def joint_histogram(
    moving: Float[Array, '...'],
    fixed: Float[Array, '...'],
    *,
    bins: int = 32,
    range_moving: Optional[tuple[float, float]] = None,
    range_fixed: Optional[tuple[float, float]] = None,
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

    hist = jnp.zeros((bins, bins), dtype=m.dtype)
    for idx_m, w_m in ((lower_m, 1.0 - frac_m), (lower_m + 1, frac_m)):
        for idx_f, w_f in ((lower_f, 1.0 - frac_f), (lower_f + 1, frac_f)):
            hist = hist.at[idx_m, idx_f].add(w_m * w_f)
    return hist / jnp.maximum(hist.sum(), 1e-12)


def mutual_information(
    moving: Float[Array, '...'],
    fixed: Float[Array, '...'],
    *,
    bins: int = 32,
    range_moving: Optional[tuple[float, float]] = None,
    range_fixed: Optional[tuple[float, float]] = None,
    normalized: bool = False,
    eps: float = 1e-10,
) -> Float[Array, '']:
    """Mutual information (or normalised MI) of two images.

    ``MI = Σ P(i,j) log( P(i,j) / (P_m(i) P_f(j)) )`` over the soft
    joint histogram.  With ``normalized=True`` returns Studholme's NMI
    ``(H_m + H_f) / H_mf`` (in ``[1, 2]``; invariant to overlap size).
    Higher is more similar; use ``-mutual_information`` as a cost.

    ``bins`` / ``range_*`` are forwarded to ``joint_histogram``;
    ``eps`` guards the logs.

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


def correlation_ratio(
    moving: Float[Array, '...'],
    fixed: Float[Array, '...'],
    *,
    bins: int = 32,
    range_fixed: Optional[tuple[float, float]] = None,
    eps: float = 1e-10,
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
    soft binning as :func:`joint_histogram` (differentiable).
    """
    m = moving.reshape(-1)
    f = fixed.reshape(-1)
    lo_f, hi_f = _resolve_range(f, range_fixed)
    lower_f, frac_f = _soft_bin(f, bins, lo_f, hi_f)

    n_k = jnp.zeros((bins,), dtype=m.dtype)
    s_k = jnp.zeros((bins,), dtype=m.dtype)
    for idx_f, w_f in ((lower_f, 1.0 - frac_f), (lower_f + 1, frac_f)):
        n_k = n_k.at[idx_f].add(w_f)
        s_k = s_k.at[idx_f].add(w_f * m)

    mu = m.mean()
    mu_k = s_k / (n_k + eps)
    between = jnp.sum(n_k * (mu_k - mu) ** 2)
    total = jnp.sum((m - mu) ** 2)
    return between / (total + eps)
