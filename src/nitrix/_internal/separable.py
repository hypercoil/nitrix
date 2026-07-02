# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Separable 1-D cross-correlation along a single axis.

The shared engine behind several separable spatial operators:
:func:`~nitrix.geometry.spatial_gradient` (derivative plus Sobel/Scharr
smoothing taps) and the local normalised cross-correlation of
:mod:`nitrix.metrics` (windowed box sums) both lower onto
:func:`correlate1d`.  It is factored here so that the boundary-mode
mapping and the pad -> moveaxis -> VALID ``conv_general_dilated`` ->
moveaxis-back pattern live in exactly one place rather than being copied
per operator.

The near-sibling Gaussian smoothing convolution helper is specialised
for the *even-kernel half-pixel-shift* case: it uses an asymmetric
``(K // 2 - 1, K // 2)`` pad to realise the documented half-pixel output
shift for even Gaussian kernels.  This engine instead uses the symmetric
``((K - 1) // 2, K // 2)`` pad, which is identical for odd kernels (the
only kind that :func:`~nitrix.geometry.spatial_gradient` and the local
cross-correlation box sums use).  The two are deliberately kept separate
so that the Gaussian even-kernel semantics are not perturbed.
"""

from __future__ import annotations

from typing import Literal

import jax.numpy as jnp
from jax import lax
from jaxtyping import Array

__all__ = ['SeparableBoundaryMode', 'correlate1d']

# Boundary-mode vocabulary shared by every separable operator built on
# this engine.  Maps onto ``jnp.pad`` modes; the mapping matches the
# ``map_coordinates`` / ``smoothing.gaussian`` conventions (scipy
# ``reflect`` -- include the boundary sample -- is numpy ``symmetric``;
# scipy ``mirror`` -- exclude it -- is numpy ``reflect``).
SeparableBoundaryMode = Literal[
    'nearest', 'reflect', 'mirror', 'wrap', 'constant'
]

_PAD_MODE: dict[str, str] = {
    'nearest': 'edge',
    'reflect': 'symmetric',
    'mirror': 'reflect',
    'wrap': 'wrap',
    'constant': 'constant',
}


def correlate1d(
    x: Array,
    kernel: Array,
    axis: int,
    mode: SeparableBoundaryMode,
) -> Array:
    """Cross-correlate ``x`` with a 1-D ``kernel`` along ``axis``.

    Pads ``x`` along ``axis`` according to ``mode`` and runs a VALID,
    same-size cross-correlation.  ``lax.conv_general_dilated`` does
    **not** flip the kernel, so tap :math:`k` maps to offset
    :math:`k - (K - 1) // 2`; that is,
    :math:`\\mathrm{out}[i] = \\sum_k \\mathrm{kernel}[k] \\,
    x[i + k - (K - 1) // 2]`, where :math:`K` is the kernel length.
    The kernel is cast to the
    dtype of ``x``, and all axes other than ``axis`` are carried as
    batch dimensions.

    Parameters
    ----------
    x : Array
        Input array of arbitrary rank.  The correlation is applied along
        ``axis`` and every other axis is treated as an independent batch
        dimension.
    kernel : Array
        One-dimensional correlation kernel of length :math:`K`.  It is
        cast to the dtype of ``x`` before use.
    axis : int
        Axis of ``x`` along which to correlate.
    mode : SeparableBoundaryMode
        Boundary-handling convention used to pad ``x`` before the VALID
        correlation, one of ``'nearest'``, ``'reflect'``, ``'mirror'``,
        ``'wrap'`` or ``'constant'`` (the last pads with zeros).

    Returns
    -------
    Array
        Array of the same shape and dtype as ``x`` holding the
        cross-correlation of ``x`` with ``kernel`` along ``axis``.
    """
    k_size = int(kernel.shape[0])
    half_l = (k_size - 1) // 2
    half_r = k_size // 2
    pad = [(0, 0)] * x.ndim
    pad[axis] = (half_l, half_r)
    pad_mode = _PAD_MODE[mode]
    if pad_mode == 'constant':
        x_padded = jnp.pad(x, pad, mode='constant', constant_values=0.0)
    else:
        x_padded = jnp.pad(x, pad, mode=pad_mode)
    x_moved = jnp.moveaxis(x_padded, axis, -1)
    batch_shape = x_moved.shape[:-1]
    length = x_moved.shape[-1]
    x_flat = x_moved.reshape(-1, 1, length)
    k = kernel.astype(x.dtype).reshape(1, 1, k_size)
    out = lax.conv_general_dilated(
        x_flat, k, window_strides=(1,), padding='VALID'
    )
    out = out.reshape(*batch_shape, out.shape[-1])
    return jnp.moveaxis(out, -1, axis)
