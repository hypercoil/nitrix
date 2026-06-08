# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Separable 1-D cross-correlation along a single axis.

The shared engine behind several separable spatial operators:
``geometry.spatial_gradient`` (derivative + Sobel/Scharr smoothing taps)
and ``nitrix.metrics`` LNCC (windowed box sums) both lower onto
``correlate1d``.  Factored here so the boundary-mode mapping and the
pad -> moveaxis -> VALID ``conv_general_dilated`` -> moveaxis-back
pattern live in exactly one place rather than being copied per operator.

Relationship to ``smoothing.gaussian._conv_1d_along_axis``: that helper
is a near-sibling specialised for the *even-kernel half-pixel-shift*
case -- it uses an asymmetric ``(K // 2 - 1, K // 2)`` pad to realise
the documented half-pixel output shift for even Gaussian kernels.  This
engine uses the symmetric ``((K - 1) // 2, K // 2)`` pad, which is
identical for odd kernels (the only kind ``spatial_gradient`` and the
LNCC box sums use).  The two are deliberately kept separate so the
Gaussian even-kernel semantics are not perturbed; converging them (this
engine plus an explicit-pad override) is a possible future cleanup, not
a correctness need.
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

    Pads ``x`` along ``axis`` per ``mode`` and runs a VALID, same-size
    cross-correlation.  ``lax.conv_general_dilated`` does **not** flip
    the kernel, so tap ``k`` maps to offset ``k - (K - 1) // 2`` -- i.e.
    ``out[i] = Σ_k kernel[k] * x[i + k - (K - 1) // 2]``.  The kernel is
    cast to ``x``'s dtype; all other axes are carried as batch.
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
