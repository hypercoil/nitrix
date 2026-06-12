# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Shared helpers for ``nitrix.metrics``: separable box sums, masked
reductions, intensity-range resolution, and differentiable soft binning.

These are private; the public surface is in ``intensity`` and
``information``.
"""

from __future__ import annotations

from typing import Optional, Sequence

import jax.numpy as jnp
from jax import lax
from jaxtyping import Array

from .._internal.reductions import Reduction, reduce
from .._internal.separable import _PAD_MODE, SeparableBoundaryMode

# ``Reduction`` is re-exported from the shared reduction surface
# (``_internal.reductions``) so the metrics modules keep one import site.
__all__ = ['Reduction']
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
    """Separable windowed sum (uniform box filter, un-normalised).

    Computed as a per-axis **integral image** (prefix-sum difference): pad the
    axis by the window's half-widths under ``mode`` (the *same* padding the
    ones-kernel ``correlate1d`` uses), prefix-sum, and difference the window
    endpoints ``P[i+size] - P[i]``.  O(N) per axis, **independent of the window
    size** (the equivalent cross-correlation is O(N·size)).  In exact arithmetic
    it is identical to that correlation -- the same windowed sum, so the
    ``lncc`` / ``lncc_grad`` contracts (self-adjoint box filter, interior-exact)
    are unchanged.  In **fp32** the prefix sum's magnitude (~axis_length·max)
    introduces a cancellation in the difference: safe at the default LNCC radii /
    intensity ranges (the fp64-vs-fp32 gate in the tests), but use fp64 -- or
    winsorise -- for a large grid at a wide intensity range.
    """
    out = x
    pad_mode = _PAD_MODE[mode]
    for ax, sz in zip(spatial_axes, sizes):
        sz = int(sz)
        half_l, half_r = (sz - 1) // 2, sz // 2
        pad = [(0, 0)] * out.ndim
        pad[ax] = (half_l, half_r)
        if pad_mode == 'constant':
            padded = jnp.pad(out, pad, mode='constant', constant_values=0.0)
        else:
            padded = jnp.pad(out, pad, mode=pad_mode)
        zero_shape = list(padded.shape)
        zero_shape[ax] = 1
        prefix = jnp.concatenate(
            [
                jnp.zeros(zero_shape, dtype=padded.dtype),
                jnp.cumsum(padded, axis=ax),
            ],
            axis=ax,
        )
        n = out.shape[ax]
        out = lax.slice_in_dim(prefix, sz, sz + n, axis=ax) - lax.slice_in_dim(
            prefix, 0, n, axis=ax
        )
    return out


def _reduce(
    values: Array,
    mask: Optional[Array],
    reduction: Reduction,
) -> Array:
    """Masked reduce -- a thin adapter over the shared ``reduce`` leaf.

    ``mask`` is the per-element domain mask (``SPEC_UPDATE_v0.5 §1.2``); it
    maps to the shared helper's ``weight`` so ``reduction='mean'`` is the
    domain-mask weighted mean ``Σ(w·x)/Σw``.
    """
    return reduce(values, weight=mask, reduction=reduction)


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
