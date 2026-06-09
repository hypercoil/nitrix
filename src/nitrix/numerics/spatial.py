# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Volumetric shape / windowing utilities.

The mundane-but-ubiquitous spatial pre / post-processing every
convolutional volume pipeline needs:

- ``pad_to_multiple`` / ``crop_to_multiple`` -- size each spatial axis up
  (or down) to a multiple of the net's pooling factor, returning the
  per-axis widths so the crop-back (and any affine-origin update, which
  is the container layer's job) is recoverable.
- ``nonzero_bounding_box`` -- the bounding box of the above-threshold
  region, as ``(lo, hi)`` index arrays.  Returning indices (not a crop)
  keeps the op jit-clean: the crop *shape* is data-dependent, the index
  math is not.
- ``gaussian_window`` + ``overlap_add`` -- the two numeric pieces of
  sliding-window patch inference: a separable Gaussian patch weight and
  the weighted overlap-add normalisation ``Σ(w·patch) / Σw``.  The tiling
  / scheduling around them belongs to the orchestration layer.

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

    Each spatial axis is symmetrically padded so its length becomes the
    next multiple of ``multiple`` (a per-axis sequence is allowed).

    Returns
    -------
    ``(padded, pad_widths)`` where ``pad_widths`` is the per-spatial-axis
    ``(lo, hi)`` padding -- enough to crop back (``unpad``) and, for the
    container layer, to shift the affine origin.
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
    ``multiple`` not exceeding its length.

    Returns
    -------
    ``(cropped, crop_widths)`` with the per-spatial-axis ``(lo, hi)``
    amounts removed.
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

    Returns ``(lo, hi)`` index arrays of length ``ndim`` (a half-open box
    ``lo <= idx < hi`` per axis).  An empty region yields ``lo == hi == 0``
    on every axis.  Only the index math is returned -- the crop itself has
    a data-dependent shape and belongs to the caller / container layer.
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

    The per-axis standard deviation is ``sigma_scale * size``.  Used to
    down-weight patch borders when blending overlapping sliding-window
    predictions, so seams between tiles are smooth.
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
    """Normalise an accumulated overlap-add: ``weighted_sum / max(weight, eps)``.

    The finalisation step of weighted patch stitching, where the
    orchestrator has accumulated ``weighted_sum += w * patch`` and
    ``weight += w`` over the tiles; ``eps`` guards voxels no patch
    covered.
    """
    return weighted_sum / jnp.maximum(weight, eps)
