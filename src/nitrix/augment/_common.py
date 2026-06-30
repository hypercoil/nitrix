# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Shared internals for ``nitrix.augment``.

- ``_default_float`` -- the x64-aware default floating dtype (``float64``
  when ``jax_enable_x64`` is set, else ``float32``), so generators do not
  silently downcast a float64 pipeline.
- ``_coarse_random_field`` -- the "low-resolution Gaussian field, linearly
  upsampled" idiom shared by ``simulate_bias_field`` and
  ``random_svf_displacement``.
"""

from __future__ import annotations

from typing import Optional, Sequence, cast

import jax
import jax.numpy as jnp
from jax.typing import DTypeLike
from jaxtyping import Array, Float


def _default_float() -> DTypeLike:
    """The canonical float dtype under the active x64 setting."""
    return jnp.result_type(float)


def _coarse_random_field(
    spatial_shape: Sequence[int],
    key: Array,
    *,
    std: Float[Array, ''],
    grid_fraction: float,
    channels: Optional[int] = None,
    dtype: DTypeLike = jnp.float32,
) -> Float[Array, '...']:
    """A smooth random field: low-res ``N(0, std**2)`` upsampled linearly.

    Draws an i.i.d. Gaussian field on a coarse grid (``round(shape *
    grid_fraction)`` per axis, ≥ 2) scaled by ``std``, then linearly
    upsamples to ``spatial_shape``.  With ``channels`` it generates a
    vector field with that trailing channel count (channels-last);
    otherwise a scalar field.

    Note: This is an upsampling of i.i.d. Gaussian noise, not a true Gaussian
    random field.
    """
    coarse = tuple(
        max(2, int(round(s * grid_fraction))) for s in spatial_shape
    )
    full = tuple(spatial_shape)
    if channels is not None:
        coarse = (*coarse, channels)
        full = (*full, channels)
    field = jax.random.normal(key, coarse, dtype=dtype) * std
    return cast(
        Float[Array, '...'], jax.image.resize(field, full, method='linear')
    )
