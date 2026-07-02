# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Shared internals for :mod:`nitrix.augment`.

- :func:`_default_float` -- the x64-aware default floating dtype
  (``float64`` when ``jax_enable_x64`` is set, else ``float32``), so
  generators do not silently downcast a float64 pipeline.
- :func:`_coarse_random_field` -- the "low-resolution Gaussian field,
  linearly upsampled" idiom shared by :func:`simulate_bias_field` and
  :func:`random_svf_displacement`.
"""

from __future__ import annotations

from typing import Optional, Sequence, cast

import jax
import jax.numpy as jnp
from jax.typing import DTypeLike
from jaxtyping import Array, Float


def _default_float() -> DTypeLike:
    """The canonical float dtype under the active x64 setting.

    Returns
    -------
    DTypeLike
        ``float64`` when JAX's ``jax_enable_x64`` flag is set, otherwise
        ``float32``. This mirrors the dtype JAX assigns to Python floats,
        so downstream generators do not silently downcast a float64
        pipeline.
    """
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
    """A smooth random field: low-resolution :math:`N(0, \\sigma^2)` noise
    upsampled linearly.

    Draws an i.i.d. Gaussian field on a coarse grid -- with per-axis size
    ``round(spatial_shape * grid_fraction)``, clamped to at least 2 -- and
    scales it by ``std``, then linearly upsamples the result to
    ``spatial_shape``. When ``channels`` is given, a vector field is
    generated with that trailing channel count (channels-last); otherwise
    the field is scalar.

    Parameters
    ----------
    spatial_shape : sequence of int
        Spatial extent of the output field along each axis.
    key : Array
        A :func:`jax.random` key seeding the Gaussian draw on the coarse
        grid.
    std : Float[Array, '']
        Scalar standard deviation :math:`\\sigma` applied to the coarse
        Gaussian samples before upsampling.
    grid_fraction : float
        Fraction of each spatial extent used for the coarse grid; the
        coarse per-axis size is ``round(s * grid_fraction)``, clamped to a
        minimum of 2. Smaller values yield a smoother field.
    channels : int, optional
        If given, a channels-last vector field with this many trailing
        channels is generated; if ``None``, a scalar field is returned.
    dtype : DTypeLike, default: ``jnp.float32``
        Floating dtype of the drawn field.

    Returns
    -------
    Float[Array, '...']
        The upsampled random field of shape ``spatial_shape`` (with a
        trailing ``channels`` axis when ``channels`` is given).

    Notes
    -----
    This is an upsampling of i.i.d. Gaussian noise, not a true Gaussian
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
