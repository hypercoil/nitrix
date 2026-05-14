# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Regular-grid deformable-registration primitives.

The voxelmorph-style operations needed to express dense deformation
fields and integrate stationary velocity fields:

- ``identity_grid``       -- the "no-op" deformation, useful as a
  starting point for learned displacements.
- ``spatial_transform``   -- warp an image by a displacement field
  via (multi-)linear interpolation.
- ``integrate_velocity_field`` -- scaling-and-squaring SVF
  integration (the diffeomorphic exponential map).
- ``resample``            -- linear-interpolation resize to a target
  spatial shape.

Plus ``center_of_mass_grid`` for weighted centre-of-mass on a
regular grid (replaces the legacy ``cmass_regular_grid``).

Channel-last layout for images: ``(..., *spatial, C)``.
Displacement fields are channel-last with ``ndim`` channels (one
per spatial axis): ``(..., *spatial, ndim)``.

Renamings from legacy code, for clarity:

- ``rescale`` -> ``resample``    (avoids confusion with intensity rescaling).
- ``vec_int`` -> ``integrate_velocity_field``.
- ``cmass_regular_grid`` -> ``center_of_mass_grid``.
"""
from __future__ import annotations

from functools import reduce
from operator import mul
from typing import Optional, Sequence, Union

import jax
import jax.numpy as jnp
import jax.scipy.ndimage as jsp_ndi
from jaxtyping import Array, Float


__all__ = [
    'identity_grid',
    'spatial_transform',
    'integrate_velocity_field',
    'resample',
    'center_of_mass_grid',
    # legacy alias kept for now -- remove at v0.1 cleanup
    'cmass_regular_grid',
    'vec_int',
    'rescale',
]


# ---------------------------------------------------------------------------
# Identity grid
# ---------------------------------------------------------------------------


def identity_grid(
    spatial_shape: Sequence[int],
    *,
    dtype=jnp.float32,
) -> Float[Array, '*spatial ndim']:
    '''The identity deformation: per-pixel its own coordinate.

    For a 2D image of shape ``(H, W)`` returns a ``(H, W, 2)`` grid
    where ``grid[i, j] == (i, j)``.  For 3D ``(D, H, W)`` returns
    ``(D, H, W, 3)`` with ``grid[k, i, j] == (k, i, j)``.

    Adding a displacement ``delta`` to this grid yields a
    deformation field consumable by ``spatial_transform``.

    Parameters
    ----------
    spatial_shape
        Per-axis sizes of the grid; the result has rank
        ``len(spatial_shape) + 1`` with the trailing dim of size
        ``len(spatial_shape)``.
    dtype
        Floating-point dtype of the returned coordinates.

    Returns
    -------
    Coordinate grid, ``(*spatial_shape, ndim)``.
    '''
    spatial_shape = tuple(int(s) for s in spatial_shape)
    ndim = len(spatial_shape)
    grids = jnp.meshgrid(
        *[jnp.arange(s, dtype=dtype) for s in spatial_shape],
        indexing='ij',
    )
    return jnp.stack(grids, axis=-1)


# ---------------------------------------------------------------------------
# Spatial transform
# ---------------------------------------------------------------------------


def _gather_coords_linear(
    image: Float[Array, '... *spatial c'],
    coords: Float[Array, '... *spatial ndim'],
    *,
    cval: float,
) -> Float[Array, '... *spatial c']:
    '''Sample ``image`` at the given coordinates using linear interpolation.

    ``coords[..., d]`` is the floating-point coordinate along axis
    ``d`` to sample.  Out-of-bounds positions are filled with
    ``cval``.  Channel dim ``c`` is gathered identically across all
    channels.
    '''
    ndim = coords.shape[-1]
    # ``map_coordinates`` wants coords in shape ``(ndim, n_samples)``,
    # one row per spatial axis of the input.  We have ``coords:
    # (..., *spatial, ndim)`` -- move the ndim axis to the front,
    # flatten the spatial dims.
    # For multi-channel images, we vmap over the trailing channel axis.
    coords_t = jnp.moveaxis(coords, -1, 0)          # (ndim, ..., *spatial)
    coords_flat = coords_t.reshape(ndim, -1)        # (ndim, N)

    def sample_one_channel(img_ch):
        '''Sample one channel: img_ch.shape == (*spatial,)'''
        return jsp_ndi.map_coordinates(
            img_ch, coords_flat, order=1, mode='constant', cval=cval,
        )

    # Iterate channels via vmap on the trailing channel axis.
    sample_v = jax.vmap(sample_one_channel, in_axes=-1, out_axes=-1)
    flat_out = sample_v(image)                       # (N, c)
    out_spatial = coords.shape[:-1]
    return flat_out.reshape(out_spatial + (image.shape[-1],))


def spatial_transform(
    image: Float[Array, '*spatial c'],
    deformation: Float[Array, '*spatial ndim'],
    *,
    cval: float = 0.0,
) -> Float[Array, '*spatial c']:
    '''Warp an image by a per-pixel deformation field.

    For each output pixel at coordinate ``p``, the result is
    ``image`` sampled at ``deformation[p]`` via (multi-)linear
    interpolation.  ``deformation`` is therefore an *absolute*
    coordinate map, not a relative displacement.  To use a
    displacement field ``delta``, add ``identity_grid`` first::

        warped = spatial_transform(image, identity_grid(spatial) + delta)

    Parameters
    ----------
    image
        Channel-last image, ``(*spatial, c)``.  Currently single-
        batch (no leading dims); ``vmap`` for batched warping.
    deformation
        Absolute sample coordinates, ``(*spatial, ndim)`` with
        ``ndim`` matching the spatial rank of ``image``.
    cval
        Constant fill value for coordinates outside the input
        bounds.  Default ``0``.

    Returns
    -------
    Warped image of the same shape as ``image``.
    '''
    spatial_rank = deformation.shape[-1]
    if image.ndim != spatial_rank + 1:
        raise ValueError(
            f'image.ndim={image.ndim} must equal '
            f'deformation.shape[-1] + 1 = {spatial_rank + 1} '
            '(channel-last layout: trailing dim of image is the '
            'channel; trailing dim of deformation indexes spatial '
            'axes).'
        )
    if deformation.shape[:-1] != image.shape[:-1]:
        raise ValueError(
            f'deformation.shape[:-1]={deformation.shape[:-1]} must '
            f'equal image.shape[:-1]={image.shape[:-1]}.'
        )
    return _gather_coords_linear(image, deformation, cval=cval)


# ---------------------------------------------------------------------------
# Integrate stationary velocity field
# ---------------------------------------------------------------------------


def integrate_velocity_field(
    velocity: Float[Array, '*spatial ndim'],
    *,
    n_steps: int = 7,
) -> Float[Array, '*spatial ndim']:
    '''Diffeomorphic exponential map via scaling-and-squaring.

    Integrates a *stationary* velocity field ``v`` to obtain a
    displacement field ``phi`` such that ``phi ~= exp(v)`` (in the
    Lie-group sense, with composition rather than addition).  The
    standard voxelmorph trick:

    1. Scale ``v`` by ``1 / 2**n_steps`` (small enough that the
       first-order approximation ``id + v`` is itself approximately
       diffeomorphic).
    2. Repeatedly compose the field with itself ``n_steps`` times::

           phi_0 = v / 2**n_steps
           phi_{i+1} = phi_i o (id + phi_i)

       (where ``o`` denotes function composition via spatial_transform).

    Parameters
    ----------
    velocity
        Stationary velocity field, channel-last,
        ``(*spatial, ndim)``.
    n_steps
        Number of doublings.  Default ``7`` (i.e., 128× scaling).
        Larger ``n_steps`` -> more accurate but more compute.

    Returns
    -------
    Displacement field of the same shape as ``velocity``; adding
    ``identity_grid`` yields the absolute deformation suitable for
    ``spatial_transform``.
    '''
    spatial_rank = velocity.shape[-1]
    spatial_shape = velocity.shape[:-1]
    if len(spatial_shape) != spatial_rank:
        raise ValueError(
            f'velocity rank mismatch: spatial_shape={spatial_shape} '
            f'has {len(spatial_shape)} dims but trailing ndim is '
            f'{spatial_rank}.'
        )
    id_grid = identity_grid(spatial_shape, dtype=velocity.dtype)
    # Initial small step.
    phi = velocity / float(2 ** n_steps)
    # ``phi`` is treated as a displacement (relative); we compose
    # by warping it through the deformation ``id + phi`` each step.
    for _ in range(n_steps):
        phi = phi + spatial_transform(phi, id_grid + phi)
    return phi


# ---------------------------------------------------------------------------
# Resample
# ---------------------------------------------------------------------------


def resample(
    image: Float[Array, '*spatial c'],
    target_shape: Sequence[int],
    *,
    cval: float = 0.0,
) -> Float[Array, '*target c']:
    '''Resize a channel-last image to ``target_shape`` via linear interpolation.

    Sample positions are evenly distributed: output pixel ``i`` along
    each axis samples the input at coordinate
    ``i * (in_size - 1) / (out_size - 1)``.  This matches scipy's
    ``align_corners=True`` convention and preserves the boundary
    pixels exactly.

    Parameters
    ----------
    image
        Channel-last image, ``(*spatial, c)``.
    target_shape
        Per-axis target sizes; must match the spatial rank of
        ``image``.
    cval
        Constant fill value for the very rare out-of-bounds sample
        induced by floating-point rounding; default ``0``.

    Returns
    -------
    Resampled image, ``(*target_shape, c)``.
    '''
    spatial_shape = image.shape[:-1]
    if len(target_shape) != len(spatial_shape):
        raise ValueError(
            f'target_shape={target_shape} must match spatial rank '
            f'of image (got image spatial shape {spatial_shape}).'
        )
    # Build a coordinate grid for the target.
    axes = []
    for in_size, out_size in zip(spatial_shape, target_shape):
        if out_size < 2:
            ax = jnp.array([0.0], dtype=image.dtype)
        else:
            ax = jnp.linspace(
                0.0, float(in_size - 1), int(out_size), dtype=image.dtype,
            )
        axes.append(ax)
    grids = jnp.meshgrid(*axes, indexing='ij')
    coords = jnp.stack(grids, axis=-1)  # (*target_shape, ndim)
    return _gather_coords_linear(image, coords, cval=cval)


# ---------------------------------------------------------------------------
# Centre of mass on a regular grid
# ---------------------------------------------------------------------------


def center_of_mass_grid(
    weight: Float[Array, '...'],
    *,
    axes: Optional[Sequence[int]] = None,
    na_value: Optional[float] = None,
) -> Float[Array, '... ndim']:
    '''Centre of mass over a regular grid of unit-spaced coordinates.

    For a tensor ``weight``, computes the centre of mass treating
    each cell's coordinate along axis ``d`` as its index ``i_d``
    (zero-based).  ``axes`` selects which axes to reduce over; the
    reduction is over those axes per slice spanned by the
    *remaining* axes.

    Parameters
    ----------
    weight
        Non-negative weight tensor.
    axes
        Axes to reduce over.  ``None`` (default) means all axes.
        With ``axes=[-3, -2, -1]`` on a ``(B, D, H, W)`` tensor, a
        separate centre-of-mass vector is produced per batch element
        over the trailing three axes.
    na_value
        Value to substitute when a slice has zero total weight (the
        centre-of-mass is undefined there; default behaviour
        produces ``NaN``).  Set to a float to fill with that value.

    Returns
    -------
    Per-slice centre-of-mass coordinates, with the reduced axes
    replaced by a single trailing axis of length ``len(axes)``
    (in the order they were reduced).
    '''
    ndim = weight.ndim
    if axes is None:
        axes = tuple(range(ndim))
    axes = tuple(ax % ndim for ax in axes)
    out_shape = tuple(s for ax, s in enumerate(weight.shape) if ax not in axes)
    out_shape = out_shape + (len(axes),)
    out = jnp.zeros(out_shape, dtype=weight.dtype)
    for i, ax in enumerate(axes):
        coor = jnp.arange(weight.shape[ax], dtype=weight.dtype)
        # Reshape coor so it broadcasts only along axis ``ax``.
        shape = [1] * ndim
        shape[ax] = weight.shape[ax]
        coor = coor.reshape(shape)
        num = (coor * weight).sum(axes)
        denom = weight.sum(axes)
        cm = num / denom
        if na_value is not None:
            cm = jnp.where(denom == 0, na_value, cm)
        out = out.at[..., i].set(cm)
    return out


# ---------------------------------------------------------------------------
# Legacy-name aliases (will be removed at v0.1 cleanup)
# ---------------------------------------------------------------------------


cmass_regular_grid = center_of_mass_grid
vec_int = integrate_velocity_field
rescale = resample
