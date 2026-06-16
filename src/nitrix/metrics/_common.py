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


# Above this window size the integral image (O(N), radius-free) overtakes the
# shifted-slice sum (O(N·size)); below it the slice sum is faster -- on *both*
# CPU and GPU (the crossover is ~30 on each, so this is a window-size, not a
# hardware, dispatch) -- and avoids the prefix sum's fp32 cancellation.  The
# LNCC radii (window 5-9) sit deep in the slice-sum regime (~2.5-3.6x).
_BOX_SHIFT_MAX_WINDOW = 29


def _pad_axis(x: Array, ax: int, half_l: int, half_r: int, pad_mode: str) -> Array:
    pad = [(0, 0)] * x.ndim
    pad[ax] = (half_l, half_r)
    if pad_mode == 'constant':
        return jnp.pad(x, pad, mode='constant', constant_values=0.0)
    return jnp.pad(x, pad, mode=pad_mode)


def _box_sum_axis_shift(x: Array, sz: int, ax: int, pad_mode: str) -> Array:
    """Windowed sum along one axis as a sum of ``sz`` window-shifted slices.

    O(N·size) but a handful of vectorised slice-adds -- faster than the integral
    image for small windows and free of its prefix-sum fp32 cancellation.
    """
    half_l, half_r = (sz - 1) // 2, sz // 2
    p = _pad_axis(x, ax, half_l, half_r, pad_mode)
    n = x.shape[ax]
    acc = lax.slice_in_dim(p, 0, n, axis=ax)
    for j in range(1, sz):
        acc = acc + lax.slice_in_dim(p, j, j + n, axis=ax)
    return acc


def _box_sum_axis_integral(x: Array, sz: int, ax: int, pad_mode: str) -> Array:
    """Windowed sum along one axis as an integral image (prefix-sum difference).

    O(N), window-size independent -- the choice for large windows.
    """
    half_l, half_r = (sz - 1) // 2, sz // 2
    padded = _pad_axis(x, ax, half_l, half_r, pad_mode)
    zero_shape = list(padded.shape)
    zero_shape[ax] = 1
    prefix = jnp.concatenate(
        [jnp.zeros(zero_shape, dtype=padded.dtype), jnp.cumsum(padded, axis=ax)],
        axis=ax,
    )
    n = x.shape[ax]
    return lax.slice_in_dim(prefix, sz, sz + n, axis=ax) - lax.slice_in_dim(
        prefix, 0, n, axis=ax
    )


def _box_sum(
    x: Array,
    sizes: Sequence[int],
    spatial_axes: Sequence[int],
    mode: BoundaryMode,
) -> Array:
    """Separable windowed sum (uniform box filter, un-normalised).

    Per axis, dispatched on the window size (L2c): a **shifted-slice sum** for
    small windows (the LNCC radii) and the **integral image** (prefix-sum
    difference) for large ones.  Both compute the *same* windowed sum -- exact-
    equal in exact arithmetic, so the ``lncc`` / ``lncc_grad`` contracts
    (self-adjoint box filter, interior-exact, ``== autodiff``) are unchanged;
    they differ only in fp32 rounding (the slice sum, lacking the prefix sum's
    ``~axis_length·max`` cancellation, is the more accurate of the two at the
    LNCC radii).  The shift sum is also ~2.5-3.6x faster there on **both** CPU
    and GPU (:data:`_BOX_SHIFT_MAX_WINDOW`); the integral image's O(N),
    radius-free advantage only pays off above the crossover.
    """
    out = x
    pad_mode = _PAD_MODE[mode]
    for ax, sz in zip(spatial_axes, sizes):
        sz = int(sz)
        if sz <= _BOX_SHIFT_MAX_WINDOW:
            out = _box_sum_axis_shift(out, sz, ax, pad_mode)
        else:
            out = _box_sum_axis_integral(out, sz, ax, pad_mode)
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
