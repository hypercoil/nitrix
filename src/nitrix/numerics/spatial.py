# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Volumetric shape and windowing utilities.

The mundane-but-ubiquitous spatial pre- and post-processing that every
convolutional volume pipeline needs:

- :func:`pad_to_multiple` and :func:`crop_to_multiple` -- size each
  spatial axis up (or down) to a multiple of the network's pooling
  factor, returning the per-axis widths so the crop-back (and any
  affine-origin update, which is the container layer's job) is
  recoverable.
- :func:`nonzero_bounding_box` -- the bounding box of the above-threshold
  region, as ``(lo, hi)`` index arrays.  Returning indices (not a crop)
  keeps the operation JIT-clean: the crop *shape* is data-dependent, the
  index arithmetic is not.
- :func:`gaussian_window` and :func:`overlap_add` -- the two numeric
  pieces of sliding-window patch inference: a separable Gaussian patch
  weight and the weighted overlap-add normalisation
  :math:`\\sum(w \\cdot \\mathrm{patch}) / \\sum w`.  The tiling and
  scheduling around them belong to the orchestration layer.

The spatial axes are the leading ``spatial_rank`` axes (channels-last
``(*spatial, c)`` convention).
"""

from __future__ import annotations

from math import ceil
from typing import Sequence, Tuple, Union

import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, Num

from .._internal.gaussian import gaussian_profile_1d

__all__ = [
    'pad_to_multiple',
    'crop_to_multiple',
    'nonzero_bounding_box',
    'gaussian_window',
    'overlap_add',
]

_IntArg = Union[int, Sequence[int]]


def _per_axis(multiple: _IntArg, spatial_rank: int) -> Tuple[int, ...]:
    if isinstance(multiple, int):
        return (multiple,) * spatial_rank
    out = tuple(int(m) for m in multiple)
    if len(out) != spatial_rank:
        raise ValueError(
            f'multiple must be an int or length-{spatial_rank} sequence; '
            f'got {multiple!r}.'
        )
    return out


def pad_to_multiple(
    x: Num[Array, '...'],
    multiple: _IntArg,
    *,
    spatial_rank: int,
    mode: str = 'constant',
    cval: float = 0.0,
) -> Tuple[Num[Array, '...'], Tuple[Tuple[int, int], ...]]:
    """Pad the leading ``spatial_rank`` axes up to a multiple.

    Each spatial axis is symmetrically padded so that its length becomes
    the next multiple of ``multiple`` (a per-axis sequence is allowed).
    When the required padding is odd, the extra element is placed on the
    high side of the axis. Trailing axes (e.g. channels) are left
    untouched.

    Parameters
    ----------
    x : Num[Array, '...']
        Input array whose first ``spatial_rank`` axes are spatial.
    multiple : int or sequence of int
        Target multiple for each spatial axis. A single integer is
        broadcast to every spatial axis; a sequence must have length
        ``spatial_rank``.
    spatial_rank : int
        Number of leading axes treated as spatial.
    mode : str, optional
        Padding mode forwarded to :func:`jax.numpy.pad`. Defaults to
        ``'constant'``.
    cval : float, optional
        Constant value used when ``mode == 'constant'``. Defaults to
        ``0.0``.

    Returns
    -------
    padded : Num[Array, '...']
        The padded array.
    pad_widths : tuple of tuple of int
        The per-spatial-axis ``(lo, hi)`` padding applied -- enough to
        crop back and, for the container layer, to shift the affine
        origin.
    """
    mult = _per_axis(multiple, spatial_rank)
    pad_width = []
    for axis in range(x.ndim):
        if axis < spatial_rank:
            size = x.shape[axis]
            m = mult[axis]
            target = ceil(size / m) * m
            total = target - size
            lo = total // 2
            pad_width.append((lo, total - lo))
        else:
            pad_width.append((0, 0))
    if mode == 'constant':
        padded = jnp.pad(x, pad_width, mode='constant', constant_values=cval)
    else:
        padded = jnp.pad(x, pad_width, mode=mode)
    return padded, tuple(pad_width[:spatial_rank])


def crop_to_multiple(
    x: Num[Array, '...'],
    multiple: _IntArg,
    *,
    spatial_rank: int,
) -> Tuple[Num[Array, '...'], Tuple[Tuple[int, int], ...]]:
    """Crop the leading ``spatial_rank`` axes down to a multiple.

    Each spatial axis is symmetrically trimmed to the largest multiple of
    ``multiple`` not exceeding its length. When the excess is odd, the
    extra element is removed from the high side of the axis. Trailing
    axes (e.g. channels) are left untouched.

    Parameters
    ----------
    x : Num[Array, '...']
        Input array whose first ``spatial_rank`` axes are spatial.
    multiple : int or sequence of int
        Target multiple for each spatial axis. A single integer is
        broadcast to every spatial axis; a sequence must have length
        ``spatial_rank``.
    spatial_rank : int
        Number of leading axes treated as spatial.

    Returns
    -------
    cropped : Num[Array, '...']
        The cropped array.
    crop_widths : tuple of tuple of int
        The per-spatial-axis ``(lo, hi)`` amounts removed from each
        spatial axis.
    """
    mult = _per_axis(multiple, spatial_rank)
    slices = []
    widths = []
    for axis in range(x.ndim):
        if axis < spatial_rank:
            size = x.shape[axis]
            m = mult[axis]
            keep = (size // m) * m
            excess = size - keep
            lo = excess // 2
            slices.append(slice(lo, lo + keep))
            widths.append((lo, excess - lo))
        else:
            slices.append(slice(None))
    return x[tuple(slices)], tuple(widths)


def nonzero_bounding_box(
    x: Union[Num[Array, '*spatial'], Bool[Array, '*spatial']],
    *,
    threshold: float = 0.0,
) -> Tuple[Int[Array, 'ndim'], Int[Array, 'ndim']]:
    """Bounding box of the ``x > threshold`` region.

    Computes the tightest axis-aligned box enclosing every element that
    exceeds ``threshold``. Only the index arithmetic is returned -- the
    crop itself has a data-dependent shape and belongs to the caller or
    container layer.

    Parameters
    ----------
    x : Num[Array, '*spatial'] or Bool[Array, '*spatial']
        Array to threshold. For a boolean array, ``threshold`` of
        ``0.0`` selects the ``True`` elements.
    threshold : float, optional
        Elements strictly greater than this value are treated as
        foreground. Defaults to ``0.0``.

    Returns
    -------
    lo : Int[Array, 'ndim']
        Per-axis lower index of the half-open box (inclusive), of length
        ``x.ndim``.
    hi : Int[Array, 'ndim']
        Per-axis upper index of the half-open box (exclusive), of length
        ``x.ndim``, so that the box spans ``lo <= idx < hi`` on each
        axis. An empty region yields ``lo == hi == 0`` on every axis.
    """
    mask = x > threshold
    ndim = x.ndim
    los = []
    his = []
    for axis in range(ndim):
        other = tuple(i for i in range(ndim) if i != axis)
        proj = jnp.any(mask, axis=other) if other else mask
        any_fg = jnp.any(proj)
        first = jnp.argmax(proj)
        last = x.shape[axis] - 1 - jnp.argmax(proj[::-1])
        los.append(jnp.where(any_fg, first, 0))
        his.append(jnp.where(any_fg, last + 1, 0))
    return jnp.stack(los).astype(jnp.int32), jnp.stack(his).astype(jnp.int32)


def gaussian_window(
    shape: Sequence[int],
    *,
    sigma_scale: float = 0.125,
    dtype: jnp.dtype = jnp.float32,
) -> Float[Array, '*shape']:
    """Separable Gaussian patch-weight window, peak 1 at the centre.

    Builds an ``len(shape)``-dimensional window as the outer product of
    per-axis unnormalised Gaussian profiles, each centred on its axis and
    peaking at 1. Used to down-weight patch borders when blending
    overlapping sliding-window predictions, so seams between tiles are
    smooth.

    Parameters
    ----------
    shape : sequence of int
        Per-axis window size; the returned array has this shape.
    sigma_scale : float, optional
        Per-axis standard deviation as a fraction of the axis size: the
        standard deviation for an axis of length ``size`` is
        :math:`\\sigma_{\\mathrm{scale}} \\cdot \\mathrm{size}` (floored
        at a small positive value). Defaults to ``0.125``.
    dtype : jax.numpy.dtype, optional
        Floating-point dtype of the returned window. Defaults to
        ``jnp.float32``.

    Returns
    -------
    Float[Array, '*shape']
        The separable Gaussian window with the requested ``shape``,
        valued 1 at the centre and decaying towards the borders.
    """
    window = jnp.ones((1,) * len(shape), dtype=dtype)
    for axis, size in enumerate(shape):
        centre = (size - 1) / 2.0
        sigma = max(sigma_scale * size, 1e-8)
        coord = jnp.arange(size, dtype=dtype)
        line = gaussian_profile_1d(coord, sigma, center=centre)
        reshape = [1] * len(shape)
        reshape[axis] = size
        window = window * line.reshape(reshape)
    return window


def overlap_add(
    weighted_sum: Float[Array, '...'],
    weight: Float[Array, '...'],
    *,
    eps: float = 1e-8,
) -> Float[Array, '...']:
    """Normalise an accumulated overlap-add.

    Computes the weighted mean
    :math:`\\mathrm{weighted\\_sum} / \\max(\\mathrm{weight}, \\epsilon)`,
    the finalisation step of weighted patch stitching. The orchestrator
    accumulates ``weighted_sum += w * patch`` and ``weight += w`` over the
    tiles, and ``eps`` guards voxels that no patch covered.

    Parameters
    ----------
    weighted_sum : Float[Array, '...']
        Accumulated sum of ``weight * patch`` contributions over all
        tiles.
    weight : Float[Array, '...']
        Accumulated sum of the patch weights over all tiles, broadcast-
        compatible with ``weighted_sum``.
    eps : float, optional
        Lower bound on the denominator, guarding voxels covered by no
        patch (zero accumulated weight). Defaults to ``1e-8``.

    Returns
    -------
    Float[Array, '...']
        The normalised overlap-add, with the shape of the broadcast of
        ``weighted_sum`` and ``weight``.
    """
    return weighted_sum / jnp.maximum(weight, eps)
