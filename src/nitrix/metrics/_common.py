# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Shared helpers for the metrics subpackage.

Collects the low-level building blocks reused across the metric kernels:
separable box (uniform-window) sums, masked reductions, intensity-range
resolution, and differentiable soft binning. These helpers are private; the
public surface lives in the ``intensity`` and ``information`` modules.
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


def _pad_axis(
    x: Array, ax: int, half_l: int, half_r: int, pad_mode: str
) -> Array:
    pad = [(0, 0)] * x.ndim
    pad[ax] = (half_l, half_r)
    if pad_mode == 'constant':
        return jnp.pad(x, pad, mode='constant', constant_values=0.0)
    return jnp.pad(x, pad, mode=pad_mode)


def _box_sum_axis_shift(x: Array, sz: int, ax: int, pad_mode: str) -> Array:
    """Windowed sum along one axis as a sum of ``sz`` window-shifted slices.

    The axis is padded by the (asymmetric) half-window on each side and the
    ``sz`` shifted views are summed. This costs :math:`O(N \\cdot \\mathrm{sz})`
    work but reduces to a handful of vectorised slice-adds; for small windows it
    is faster than the integral image and free of the prefix sum's fp32
    cancellation.

    Parameters
    ----------
    x : Array
        Input array of arbitrary rank.
    sz : int
        Window size along ``ax``.
    ax : int
        Axis along which to compute the windowed sum.
    pad_mode : str
        Boundary padding mode passed to :func:`jax.numpy.pad` (a ``'constant'``
        mode pads with zeros).

    Returns
    -------
    Array
        Array with the same shape as ``x``, each element replaced by the sum
        over the length-``sz`` window centred on it.
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

    The axis is padded by the half-window, a zero-prefixed cumulative sum is
    formed, and the windowed sum is recovered as a difference of two shifted
    slices of that prefix. This costs :math:`O(N)` work independent of the
    window size, making it the preferred variant for large windows.

    Parameters
    ----------
    x : Array
        Input array of arbitrary rank.
    sz : int
        Window size along ``ax``.
    ax : int
        Axis along which to compute the windowed sum.
    pad_mode : str
        Boundary padding mode passed to :func:`jax.numpy.pad` (a ``'constant'``
        mode pads with zeros).

    Returns
    -------
    Array
        Array with the same shape as ``x``, each element replaced by the sum
        over the length-``sz`` window centred on it.
    """
    half_l, half_r = (sz - 1) // 2, sz // 2
    padded = _pad_axis(x, ax, half_l, half_r, pad_mode)
    zero_shape = list(padded.shape)
    zero_shape[ax] = 1
    prefix = jnp.concatenate(
        [
            jnp.zeros(zero_shape, dtype=padded.dtype),
            jnp.cumsum(padded, axis=ax),
        ],
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

    Applies a uniform box (moving-sum) window along each requested spatial
    axis in turn. Per axis, the implementation is dispatched on the window
    size: a shifted-slice sum (:func:`_box_sum_axis_shift`) for small windows,
    such as the local normalised cross-correlation radii, and an integral image
    (:func:`_box_sum_axis_integral`) for large ones.

    Both variants compute the same windowed sum and are equal in exact
    arithmetic, so the :func:`~nitrix.metrics.intensity.lncc` and
    :func:`~nitrix.metrics.intensity.lncc_grad` contracts (self-adjoint box
    filter, interior-exact, agreement with autodiff) are unchanged; they differ
    only in fp32 rounding. The slice sum lacks the prefix sum's
    :math:`\\sim\\!\\mathrm{axis\\_length} \\cdot \\max` cancellation, making it
    the more accurate of the two at the relevant radii, and it is also markedly
    faster there on both CPU and GPU (see :data:`_BOX_SHIFT_MAX_WINDOW`); the
    integral image's window-size-independent :math:`O(N)` cost only pays off
    above the crossover.

    Parameters
    ----------
    x : Array
        Input array of arbitrary rank.
    sizes : Sequence[int]
        Window size for each axis in ``spatial_axes``, in the same order.
    spatial_axes : Sequence[int]
        Axes along which to apply the box window.
    mode : BoundaryMode
        Boundary handling mode; resolved to the underlying padding mode used by
        each per-axis pass.

    Returns
    -------
    Array
        Array with the same shape as ``x`` holding the separable windowed sum.
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
    """Masked reduction, a thin adapter over the shared reduction leaf.

    Forwards to the shared :func:`~nitrix._internal.reductions.reduce` helper,
    mapping the per-element domain mask onto that helper's ``weight`` argument.
    With ``reduction='mean'`` this yields the domain-mask weighted mean
    :math:`\\sum(w \\cdot x) / \\sum w`.

    Parameters
    ----------
    values : Array
        Values to reduce.
    mask : Array or None
        Per-element domain mask (weights), broadcastable to ``values``. If
        ``None``, all elements are weighted equally.
    reduction : Reduction
        Reduction mode: one of ``'none'``, ``'sum'`` or ``'mean'``.

    Returns
    -------
    Array
        The reduced result: the unchanged ``values`` for ``'none'``, otherwise
        a scalar (or reduced-shape) array.
    """
    return reduce(values, weight=mask, reduction=reduction)


def _resolve_range(
    x: Array,
    value_range: Optional[tuple[float, float]],
) -> tuple[Array, Array]:
    """Resolve an intensity range to ``(lo, hi)`` arrays.

    Passing ``None`` derives the bounds from the data minimum and maximum;
    otherwise the supplied bounds are returned as arrays matching the dtype of
    ``x``.

    Notes
    -----
    Data-derived bounds are piecewise-constant in the inputs, so their gradient
    is zero almost everywhere; pin an explicit range when a stable binning
    across optimisation steps is wanted.

    Parameters
    ----------
    x : Array
        Input intensities used to derive the range when ``value_range`` is
        ``None``.
    value_range : tuple of float or None
        Explicit ``(lo, hi)`` bounds, or ``None`` to use the data min / max.

    Returns
    -------
    lo : Array
        Lower bound of the intensity range, as a scalar array in the dtype of
        ``x``.
    hi : Array
        Upper bound of the intensity range, as a scalar array in the dtype of
        ``x``.
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

    Rescales each value into ``[0, bins - 1]`` given the range ``(lo, hi)`` and
    splits it linearly between two adjacent bins. The scheme is differentiable
    through the fractional weight: the integer bin index is piecewise-constant,
    so the gradient flows via ``frac`` rather than the index.

    Parameters
    ----------
    values : Array
        Values to bin.
    bins : int
        Number of bins.
    lo : Array
        Lower edge of the binning range.
    hi : Array
        Upper edge of the binning range.
    eps : float, optional
        Floor on the range span ``hi - lo`` to avoid division by zero.

    Returns
    -------
    lower : Array
        Lower bin index for each value, of integer dtype and clipped so that
        ``lower + 1`` is a valid bin.
    frac : Array
        Fractional weight toward ``lower + 1`` (the weight on ``lower`` is
        ``1 - frac``), same shape as ``values``.
    """
    span = jnp.maximum(hi - lo, eps)
    scaled = (values - lo) / span * (bins - 1)
    scaled = jnp.clip(scaled, 0.0, float(bins - 1))
    lower = jnp.floor(scaled).astype(jnp.int32)
    lower = jnp.clip(lower, 0, bins - 2)
    frac = scaled - lower
    return lower, frac
