# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Shared helpers for ``nitrix.metrics``: separable box sums, masked
reductions, intensity-range resolution, and differentiable soft binning.

These are private; the public surface is in ``intensity`` and
``information``.
"""

from __future__ import annotations

from typing import Literal, Optional, Sequence

import jax.numpy as jnp
from jaxtyping import Array

from .._internal.separable import SeparableBoundaryMode, correlate1d

Reduction = Literal['mean', 'sum', 'none']
# The windowed box sum and ``geometry.spatial_gradient`` share one
# cross-correlation engine and one boundary-mode vocabulary; both live
# in ``nitrix._internal.separable``.
BoundaryMode = SeparableBoundaryMode


def _box_sum(
    x: Array,
    sizes: Sequence[int],
    spatial_axes: Sequence[int],
    mode: BoundaryMode,
) -> Array:
    """Separable windowed sum (uniform box filter, un-normalised)."""
    out = x
    for ax, sz in zip(spatial_axes, sizes):
        kernel = jnp.ones((int(sz),), dtype=x.dtype)
        out = correlate1d(out, kernel, ax, mode)
    return out


def _reduce(
    values: Array,
    mask: Optional[Array],
    reduction: Reduction,
) -> Array:
    """Apply an optional 0/1 (or soft) ``mask`` and reduce."""
    if mask is not None:
        values = values * mask
    if reduction == 'none':
        return values
    if reduction == 'sum':
        return values.sum()
    if reduction == 'mean':
        if mask is not None:
            return values.sum() / jnp.maximum(mask.sum(), 1e-12)
        return values.mean()
    raise ValueError(
        f'reduction={reduction!r}; expected "mean", "sum", or "none".'
    )


def _resolve_range(
    x: Array,
    value_range: Optional[tuple[float, float]],
) -> tuple[Array, Array]:
    """Resolve an intensity range to ``(lo, hi)`` arrays.

    ``None`` uses the data min / max.  Note: data-derived bounds are
    piecewise-constant in the inputs (so their gradient is zero almost
    everywhere); pin an explicit range when a stable binning across
    optimisation steps is wanted.
    """
    if value_range is None:
        return jnp.min(x), jnp.max(x)
    lo, hi = value_range
    return jnp.asarray(lo, dtype=x.dtype), jnp.asarray(hi, dtype=x.dtype)


def _soft_bin(
    values: Array,
    bins: int,
    lo: Array,
    hi: Array,
    *,
    eps: float = 1e-12,
) -> tuple[Array, Array]:
    """Linear (Parzen) soft binning into ``bins`` bins.

    Returns ``(lower, frac)``: the lower bin index (clipped so
    ``lower + 1`` is valid) and the fractional weight ``frac`` toward
    ``lower + 1``.  Differentiable through ``frac`` (the index is
    piecewise-constant; the gradient flows via the weights).
    """
    span = jnp.maximum(hi - lo, eps)
    scaled = (values - lo) / span * (bins - 1)
    scaled = jnp.clip(scaled, 0.0, float(bins - 1))
    lower = jnp.floor(scaled).astype(jnp.int32)
    lower = jnp.clip(lower, 0, bins - 2)
    frac = scaled - lower
    return lower, frac
