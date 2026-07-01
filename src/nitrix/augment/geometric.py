# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Spatial augmentation transforms (N-D, channels-last).

- :func:`random_flip` -- independent per-axis Bernoulli reflection.
- :func:`random_crop` -- random-offset, fixed-size crop (a pure index
  slice).
- :func:`random_resized_crop` -- a random sub-window resampled to a fixed
  output shape (the "zoom" augmentation), built on the
  :func:`~nitrix.geometry.spatial_transform` resampler.
- :func:`random_affine_matrix` -- a random affine, drawn in geometric
  parameters (rotation / scale / shear / translation) and assembled with
  :func:`~nitrix.geometry.params_to_affine_matrix`.
- :func:`random_svf_displacement` -- a random smooth diffeomorphic
  displacement field, from a low-resolution stationary velocity field
  integrated by :func:`~nitrix.geometry.integrate_velocity_field`.

The matrix / field generators return the *transform*; apply it with the
geometry warp primitives (an affine grid plus
:func:`~nitrix.geometry.spatial_transform`).  Coordinates and fields
follow the nitrix channels-last convention (``(*spatial, ndim)``).
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
from jax.typing import DTypeLike
from jaxtyping import Array, Float

from ..geometry import (
    Interpolator,
    Linear,
    integrate_velocity_field,
    params_to_affine_matrix,
    spatial_transform,
)
from ._common import _coarse_random_field, _default_float

__all__ = [
    'random_flip',
    'random_crop',
    'random_resized_crop',
    'random_affine_matrix',
    'random_svf_displacement',
]


def random_flip(
    x: Float[Array, '...'],
    key: Array,
    *,
    axes: Optional[Sequence[int]] = None,
    p: float = 0.5,
) -> Float[Array, '...']:
    """Independently reflect along each eligible axis with probability ``p``.

    Parameters
    ----------
    x
        Input tensor.
    key
        PRNG key.
    axes
        Axes eligible for flipping.  ``None`` (default) makes every axis
        eligible; for a channels-last volume pass the spatial axes so the
        channel axis is never reflected.
    p
        Per-axis reflection probability.

    Returns
    -------
    Float[Array, '...']
        A tensor with the same shape and dtype as ``x``, reflected along
        each eligible axis for which the per-axis coin came up heads.
    """
    flip_axes = tuple(range(x.ndim)) if axes is None else tuple(axes)
    if not flip_axes:
        return x
    keys = jax.random.split(key, len(flip_axes))
    for axis, k in zip(flip_axes, keys):
        coin = jax.random.uniform(k) < p
        x = jnp.where(coin, jnp.flip(x, axis=axis), x)
    return x


def random_crop(
    x: Float[Array, '...'],
    key: Array,
    *,
    size: Sequence[int],
) -> Float[Array, '...']:
    """Random-offset fixed-size crop (a pure index slice).

    ``size`` gives the output extent on **every** axis of ``x`` (pass an
    axis's full length to leave it uncropped).  A uniform random offset
    is drawn per axis and the window extracted with ``lax.dynamic_slice``
    (static size, traced offset -> jit-clean).

    Parameters
    ----------
    x
        Input tensor of arbitrary rank.
    key
        PRNG key.
    size
        Per-axis output extent, one entry for each axis of ``x``.  Each
        entry must not exceed the corresponding axis length; use an
        axis's full length to leave it uncropped.

    Returns
    -------
    Float[Array, '...']
        The cropped window, of shape ``tuple(size)`` and the same dtype
        as ``x``.

    Raises
    ------
    ValueError
        If ``len(size) != x.ndim`` or any element exceeds ``x``'s shape.
    """
    size = tuple(int(s) for s in size)
    if len(size) != x.ndim:
        raise ValueError(
            f'size has {len(size)} entries but x has {x.ndim} axes.'
        )
    if any(s > dim for s, dim in zip(size, x.shape)):
        raise ValueError(f'size {size} exceeds shape {x.shape} on some axis.')
    keys = jax.random.split(key, x.ndim)
    offsets = tuple(
        jax.random.randint(k, (), 0, dim - s + 1)
        for k, dim, s in zip(keys, x.shape, size)
    )
    return jax.lax.dynamic_slice(x, offsets, size)


def random_resized_crop(
    x: Float[Array, '*spatial c'],
    key: Array,
    *,
    size: Sequence[int],
    scale_range: Tuple[float, float] = (0.5, 1.0),
    method: Interpolator = Linear(),
) -> Float[Array, '*size c']:
    """Random sub-window resampled to a fixed output shape (zoom crop).

    Samples a per-axis crop extent :math:`\\sim \\mathcal{U}(\\text{scale\\_range})
    \\cdot \\text{input\\_extent}` and a uniform offset within the remaining
    slack, builds the output grid of coordinates mapped into that window,
    and resamples with :func:`~nitrix.geometry.spatial_transform`.  The
    output is always ``(*size, c)`` regardless of the sampled window, so
    it composes into a fixed-shape batch.

    Parameters
    ----------
    x
        Channels-last input ``(*spatial, c)``; the spatial rank is
        ``len(size)``.
    key
        PRNG key.
    size
        Output spatial shape, one entry per spatial axis.
    scale_range
        ``(low, high)`` bounds on the per-axis window extent, expressed
        as a fraction of the input axis length; the extent is drawn
        uniformly from this range independently for each axis.
    method
        Interpolation kernel (a :class:`~nitrix.geometry.Interpolator`;
        :class:`~nitrix.geometry.Linear` by default,
        :class:`~nitrix.geometry.NearestNeighbour` for label maps).

    Returns
    -------
    Float[Array, '*size c']
        The resampled window, of shape ``(*size, c)`` with the same
        channel count as ``x``.  Coordinates falling outside the input
        are filled with zero.
    """
    ndim = len(size)
    spatial = x.shape[:ndim]
    # Coordinate maths in (at least) the input's float precision, so a
    # float64 volume is not sampled through float32 coordinates.
    coord_dtype = jnp.result_type(x.dtype, jnp.float32)
    in_shape = jnp.asarray(spatial, dtype=coord_dtype)
    k_scale, k_offset = jax.random.split(key, 2)
    scales = jax.random.uniform(
        k_scale, (ndim,), minval=scale_range[0], maxval=scale_range[1]
    )
    extents = scales * in_shape
    max_offsets = jnp.maximum(in_shape - extents, 0.0)
    offsets = jax.random.uniform(k_offset, (ndim,)) * max_offsets
    coord_axes = [
        offsets[i]
        + ((jnp.arange(size[i], dtype=coord_dtype) + 0.5) / size[i])
        * extents[i]
        - 0.5
        for i in range(ndim)
    ]
    mesh = jnp.meshgrid(*coord_axes, indexing='ij')
    coords = jnp.stack(mesh, axis=-1)  # (*size, ndim)
    return spatial_transform(
        x, coords, method=method, mode='constant', cval=0.0
    )


def random_affine_matrix(
    key: Array,
    *,
    ndim: int = 3,
    max_rotation: float = 15.0,
    max_scale: float = 0.15,
    max_shear: float = 0.012,
    max_translation: float = 0.0,
) -> Float[Array, 'd d1']:
    """Sample a random affine ``(ndim, ndim+1)`` in geometric parameters.

    Supports ``ndim`` in ``{2, 3}`` (returns ``(2, 3)`` or ``(3, 4)``).
    Rotation :math:`\\sim \\mathcal{U}(-\\text{max\\_rotation},
    \\text{max\\_rotation})` degrees (a single angle in 2-D, three in 3-D),
    scale :math:`\\sim \\mathcal{U}(1 - \\text{max\\_scale}, 1 +
    \\text{max\\_scale})`, and shear / translation drawn from their
    symmetric ranges, then assembled as a translation, rotation, scale
    and shear composition via
    :func:`~nitrix.geometry.params_to_affine_matrix`.  All-zero bounds
    give the identity.

    Parameters
    ----------
    key
        PRNG key.
    ndim
        Spatial dimensionality; either ``2`` or ``3``.
    max_rotation
        Half-width, in degrees, of the symmetric uniform range from
        which each rotation angle is drawn.  One angle is sampled in 2-D
        and three in 3-D.
    max_scale
        Half-width of the per-axis scale range; the scale of each axis is
        drawn from ``(1 - max_scale, 1 + max_scale)``.
    max_shear
        Half-width of the symmetric uniform range from which each shear
        parameter is drawn.
    max_translation
        Half-width of the symmetric uniform range from which each
        per-axis translation is drawn.

    Returns
    -------
    Float[Array, 'd d1']
        The sampled affine matrix, of shape ``(ndim, ndim + 1)`` (i.e.
        ``(2, 3)`` or ``(3, 4)``), mapping homogeneous coordinates.
    """
    rot_count = ndim * (ndim - 1) // 2
    k_r, k_s, k_sh, k_t = jax.random.split(key, 4)
    rot = jax.random.uniform(
        k_r, (rot_count,), minval=-max_rotation, maxval=max_rotation
    )
    scale = jax.random.uniform(
        k_s, (ndim,), minval=1.0 - max_scale, maxval=1.0 + max_scale
    )
    shear = jax.random.uniform(
        k_sh, (rot_count,), minval=-max_shear, maxval=max_shear
    )
    trans = jax.random.uniform(
        k_t, (ndim,), minval=-max_translation, maxval=max_translation
    )
    par = jnp.concatenate([trans, rot, scale, shear])
    return params_to_affine_matrix(par, ndim=ndim)


def random_svf_displacement(
    spatial_shape: Sequence[int],
    key: Array,
    *,
    max_std: float = 3.0,
    grid_fraction: float = 0.0625,
    n_steps: int = 5,
    dtype: Optional[DTypeLike] = None,
) -> Float[Array, '*spatial ndim']:
    """Sample a smooth diffeomorphic displacement field (channels-last).

    Draws a low-resolution stationary velocity field with standard
    deviation :math:`\\sim \\mathcal{U}(0, \\text{max\\_std})`, linearly
    upsamples it to ``spatial_shape``, and integrates it by
    scaling-and-squaring (:func:`~nitrix.geometry.integrate_velocity_field`)
    into a smooth, invertible displacement.  ``max_std == 0`` gives a zero
    field.

    Parameters
    ----------
    spatial_shape
        Spatial extent of the output field along each axis; the spatial
        rank ``ndim`` is ``len(spatial_shape)``.
    key
        PRNG key.
    max_std
        Upper bound of the uniform range from which the velocity field's
        standard deviation is drawn.
    grid_fraction
        Fraction of each spatial extent used for the coarse velocity
        grid; smaller values yield a smoother, lower-frequency field.
    n_steps
        Number of scaling-and-squaring steps used to integrate the
        velocity field into a displacement.
    dtype
        Floating dtype of the field.  ``None`` (default) selects the
        x64-aware default float, avoiding a silent float32 downcast.

    Returns
    -------
    Float[Array, '*spatial ndim']
        The channels-last displacement field, of shape
        ``(*spatial_shape, ndim)`` and dtype ``dtype``.
    """
    ndim = len(spatial_shape)
    dt = _default_float() if dtype is None else dtype
    k_std, k_field = jax.random.split(key, 2)
    std = jax.random.uniform(k_std, (), dtype=dt, minval=0.0, maxval=max_std)
    vel = _coarse_random_field(
        spatial_shape,
        k_field,
        std=std,
        grid_fraction=grid_fraction,
        channels=ndim,
        dtype=dt,
    )
    return integrate_velocity_field(vel, n_steps=n_steps)
