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
from typing import Literal, Optional, Sequence, Union

import jax
import jax.numpy as jnp
import jax.scipy.ndimage as jsp_ndi
from jaxtyping import Array, Float


__all__ = [
    'identity_grid',
    'spatial_transform',
    'integrate_velocity_field',
    'jacobian_displacement',
    'jacobian_det_displacement',
    'resample',
    'center_of_mass_grid',
    # legacy alias kept for now -- remove at v0.1 cleanup
    'cmass_regular_grid',
    'vec_int',
    'rescale',
]


# Boundary modes accepted by ``spatial_transform``, ``resample``, and the
# central-difference paths.  Names match ``jax.scipy.ndimage.map_coordinates``.
BoundaryMode = Literal['constant', 'nearest', 'wrap', 'mirror', 'reflect']


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
    image: Float[Array, '*spatial c'],
    coords: Float[Array, '*out_spatial ndim'],
    *,
    mode: BoundaryMode,
    cval: float,
) -> Float[Array, '*out_spatial c']:
    '''Sample ``image`` at the given coordinates using linear interpolation.

    Operates on the *unbatched* core shapes -- batch dims are
    composed by callers via ``jax.vmap``.

    ``coords[..., d]`` is the floating-point coordinate along axis
    ``d`` to sample.  Out-of-bounds positions are handled according
    to ``mode``:

    - ``"constant"``: fill with ``cval``.
    - ``"nearest"``: clamp to the input extent (edge replication --
      the voxelmorph default for SVF integration).
    - ``"wrap"`` / ``"mirror"`` / ``"reflect"``: periodic / mirror
      / mirror-with-edge per ``jax.scipy.ndimage.map_coordinates``.
    '''
    ndim = coords.shape[-1]
    # ``map_coordinates`` wants coords in shape ``(ndim, n_samples)``,
    # one row per spatial axis of the input.  We have ``coords:
    # (*out_spatial, ndim)`` -- move the ndim axis to the front,
    # flatten the out_spatial dims.
    # For multi-channel images, we vmap over the trailing channel axis.
    coords_t = jnp.moveaxis(coords, -1, 0)          # (ndim, *out_spatial)
    coords_flat = coords_t.reshape(ndim, -1)        # (ndim, N)

    def sample_one_channel(img_ch):
        return jsp_ndi.map_coordinates(
            img_ch, coords_flat, order=1, mode=mode, cval=cval,
        )

    sample_v = jax.vmap(sample_one_channel, in_axes=-1, out_axes=-1)
    flat_out = sample_v(image)                       # (N, c)
    out_spatial = coords.shape[:-1]
    return flat_out.reshape(out_spatial + (image.shape[-1],))


def spatial_transform(
    image: Float[Array, '*leading *spatial c'],
    deformation: Float[Array, '*leading *spatial ndim'],
    *,
    mode: BoundaryMode = 'constant',
    cval: float = 0.0,
) -> Float[Array, '*leading *spatial c']:
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
        Channel-last image, ``(*leading, *spatial, c)``.  Leading
        batch dims are broadcast-compatible with ``deformation``.
    deformation
        Absolute sample coordinates, ``(*leading, *spatial, ndim)``
        with ``ndim`` matching the spatial rank of ``image``.  The
        spatial shape need not match ``image``'s spatial shape --
        for resampling-like ops, ``deformation`` indexes anywhere
        in ``image``.
    mode
        Out-of-bounds handling.  Default ``"constant"`` (preserves
        the prior contract for image-sampling).  Set
        ``mode="nearest"`` for edge-replicate (voxelmorph
        convention for flow-warp); see ``integrate_velocity_field``
        which defaults to ``"nearest"`` for the same reason.  Other
        valid values: ``"wrap"``, ``"mirror"``, ``"reflect"``.
    cval
        Constant fill value when ``mode="constant"``.  Ignored
        otherwise.  Default ``0``.

    Returns
    -------
    Warped image of the same shape as ``deformation`` with the
    trailing channel axis from ``image``.

    Notes
    -----
    Leading batch dims are composed via ``jax.vmap`` internally;
    the per-sample core takes ``(*spatial, c)`` / ``(*spatial,
    ndim)`` and returns ``(*spatial, c)``.  All leading axes that
    appear in both inputs must match exactly -- broadcast of
    leading axes is *not* attempted (the asymmetry between image
    and deformation shape semantics makes silent broadcast
    error-prone).
    '''
    ndim = deformation.shape[-1]
    # The image has one trailing channel dim plus ``ndim`` spatial
    # dims; the deformation has one trailing ``ndim`` axis plus
    # ``ndim`` spatial dims.  Everything before that is batch.
    n_image_core = ndim + 1
    n_deform_core = ndim + 1
    if image.ndim < n_image_core or deformation.ndim < n_deform_core:
        raise ValueError(
            f'spatial_transform: image.ndim={image.ndim} or '
            f'deformation.ndim={deformation.ndim} too small for '
            f'ndim={ndim} spatial axes plus '
            '(image: channel, deformation: ndim) trailing axis.'
        )
    image_batch = image.shape[:-(n_image_core)]
    deform_batch = deformation.shape[:-(n_deform_core)]
    if image_batch != deform_batch:
        raise ValueError(
            f'spatial_transform: leading batch dims must match; got '
            f'image batch {image_batch} vs deformation batch '
            f'{deform_batch}.  Broadcast is not attempted -- vmap '
            'manually if you need it.'
        )

    def core(image_, deformation_):
        return _gather_coords_linear(
            image_, deformation_, mode=mode, cval=cval,
        )

    fn = core
    for _ in range(len(image_batch)):
        fn = jax.vmap(fn, in_axes=(0, 0))
    return fn(image, deformation)


# ---------------------------------------------------------------------------
# Integrate stationary velocity field
# ---------------------------------------------------------------------------


def integrate_velocity_field(
    velocity: Float[Array, '*spatial ndim'],
    *,
    n_steps: int = 7,
    mode: BoundaryMode = 'nearest',
    cval: float = 0.0,
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
           phi_{i+1} = phi_i + phi_i o (id + phi_i)

       (where ``o`` denotes function composition via
       ``spatial_transform``).

    Parameters
    ----------
    velocity
        Stationary velocity field, channel-last,
        ``(*spatial, ndim)``.
    n_steps
        Number of doublings.  Default ``7`` (i.e., 128× scaling).
        Larger ``n_steps`` -> more accurate but more compute.
    mode
        Boundary behaviour for the per-step
        ``spatial_transform(phi, id + phi)`` call.  Default
        ``"nearest"`` (edge-replicate) -- this is the
        voxelmorph convention and the *only* mode that yields
        a numerically sensible integrated flow at the boundary.
        ``"constant"`` (with ``cval=0``) silently miscomputes the
        flow within ~`n_steps` voxels of every edge; do not use
        unless you understand exactly what your boundary should
        be.
    cval
        Constant fill value when ``mode="constant"``.  Ignored
        otherwise.

    Returns
    -------
    Displacement field of the same shape as ``velocity``; adding
    ``identity_grid`` yields the absolute deformation suitable for
    ``spatial_transform``.

    Notes
    -----
    The default ``mode`` was flipped from ``"constant"`` (the
    naive choice) to ``"nearest"`` to match voxelmorph's
    ``VecInt(method='ss', int_steps=...)``.  This is the *only*
    semantics-changing behaviour change in the JOSA-feedback
    sprint; downstream code that relied on the old default needs
    to pass ``mode="constant"`` explicitly.  See
    ``docs/design/geometry.md`` for the rationale.
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
        phi = phi + spatial_transform(
            phi, id_grid + phi, mode=mode, cval=cval,
        )
    return phi


# ---------------------------------------------------------------------------
# Resample
# ---------------------------------------------------------------------------


def resample(
    image: Float[Array, '*spatial c'],
    target_shape: Sequence[int],
    *,
    mode: BoundaryMode = 'constant',
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
    mode
        Out-of-bounds handling.  Sample positions are nominally
        within the input extent (rounded up to the endpoint), but
        floating-point drift can push them out by an ULP; ``mode``
        selects the fill.  Default ``"constant"``; ``"nearest"`` is
        the safer default for general downsampling but ``"constant"``
        + ``cval=0`` matches the prior contract.
    cval
        Constant fill value when ``mode="constant"``.  Default ``0``.

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
    return _gather_coords_linear(image, coords, mode=mode, cval=cval)


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


# ---------------------------------------------------------------------------
# Jacobian field of a displacement / deformation
# ---------------------------------------------------------------------------


def _central_diff_along_axis(
    field: Float[Array, '... s c'],
    spatial_axis: int,
    *,
    mode: BoundaryMode,
    spacing: float,
) -> Float[Array, '... s c']:
    '''Per-axis central difference of a channel-last field.

    ``spatial_axis`` indexes the axis to differentiate along
    (negative-indexable).  Boundary handling: ``mode='nearest'``
    duplicates the edge (forward / backward diff at the very edge),
    ``'wrap'`` is periodic, ``'mirror'`` is mirror without edge
    repetition.

    The denominator is ``2 * spacing`` (the standard central
    difference) for interior points.  At ``'nearest'`` boundary
    cells the formula degenerates to a forward / backward
    difference; the denominator is still ``2 * spacing`` (matching
    scipy / voxelmorph convention which treats the boundary cell
    as if neighbouring itself).
    '''
    n = field.ndim
    ax = spatial_axis % n
    # Build the "next" and "prev" along ax by rolling, then patch
    # the boundary rows according to ``mode``.
    nxt = jnp.roll(field, -1, axis=ax)
    prv = jnp.roll(field, +1, axis=ax)
    if mode == 'nearest':
        # Replace the wrapped boundary values with the edge cell.
        L = field.shape[ax]
        sl_last = [slice(None)] * n
        sl_last[ax] = L - 1
        sl_first = [slice(None)] * n
        sl_first[ax] = 0
        # nxt at index L-1 should be field[L-1] (no neighbour beyond).
        nxt = nxt.at[tuple(sl_last)].set(field[tuple(sl_last)])
        # prv at index 0 should be field[0].
        prv = prv.at[tuple(sl_first)].set(field[tuple(sl_first)])
    elif mode == 'wrap':
        pass  # roll already gives periodic behaviour
    elif mode == 'mirror':
        # Mirror without repeating the edge cell: field[L-1] reflects
        # to field[L-2], field[0] reflects to field[1].
        L = field.shape[ax]
        sl_last = [slice(None)] * n
        sl_last[ax] = L - 1
        sl_first = [slice(None)] * n
        sl_first[ax] = 0
        # nxt at L-1 -> field[L-2]
        sl_lm1 = [slice(None)] * n
        sl_lm1[ax] = L - 2
        nxt = nxt.at[tuple(sl_last)].set(field[tuple(sl_lm1)])
        # prv at 0 -> field[1]
        sl_one = [slice(None)] * n
        sl_one[ax] = 1
        prv = prv.at[tuple(sl_first)].set(field[tuple(sl_one)])
    else:
        raise ValueError(
            f'boundary_mode={mode!r}; expected "nearest", "wrap", or '
            '"mirror".'
        )
    return (nxt - prv) / (2.0 * spacing)


def _negate_at_index(x: Array, axis: int, index: int) -> Array:
    '''Negate ``x`` at ``index`` along ``axis``, leave the rest unchanged.'''
    n = x.ndim
    axis_norm = axis % n
    sl = [slice(None)] * n
    sl[axis_norm] = slice(index, index + 1)
    return x.at[tuple(sl)].set(-x[tuple(sl)])


def jacobian_displacement(
    u: Float[Array, '... *spatial d'],
    *,
    boundary_mode: Literal['nearest', 'wrap', 'mirror'] = 'nearest',
    spacing: Union[float, Sequence[float]] = 1.0,
) -> Float[Array, '... *spatial d d']:
    '''Per-point Jacobian of the deformation φ = id + u.

    For a channel-last displacement field ``u`` with ``d`` spatial
    axes, returns ``J[..., i, j] = δ_{i,j} + ∂ u_i / ∂ x_j``,
    computed via central differences along each spatial axis.

    Parameters
    ----------
    u
        Displacement field, channel-last,
        ``(*leading, *spatial, d)`` with ``len(spatial) == d``.
    boundary_mode
        Boundary handling for the central difference.  ``"nearest"``
        (default) clamps to the edge cell -- this is what
        voxelmorph uses for QA on learned warps.  ``"wrap"`` is
        periodic.  ``"mirror"`` is non-repeating mirror.
    spacing
        Voxel spacing along each spatial axis.  ``float`` ->
        isotropic; sequence -> per-axis (length ``d``).  For
        non-isotropic fMRI / MRI grids the spacing should be in
        the same physical units as ``u``; the central-difference
        denominator is ``2 * spacing[axis]``.

    Returns
    -------
    Jacobian field, ``(*leading, *spatial, d, d)``.  The trailing
    two axes index ``(component, partial-direction)`` of the
    Jacobian matrix at each spatial location.

    Notes
    -----
    For folding detection use ``jacobian_det_displacement`` which
    avoids materialising the full Jacobian when only the
    determinant is needed.
    '''
    d = u.shape[-1]
    spatial_shape = u.shape[-(d + 1):-1]
    if len(spatial_shape) != d:
        raise ValueError(
            f'jacobian_displacement: trailing displacement-channel '
            f'axis has length {d}, but only {len(spatial_shape)} '
            'preceding spatial axes were detected.'
        )
    if isinstance(spacing, (int, float)):
        spacing_per_axis = (float(spacing),) * d
    else:
        spacing_per_axis = tuple(float(s) for s in spacing)
        if len(spacing_per_axis) != d:
            raise ValueError(
                f'spacing length {len(spacing_per_axis)} != d={d}.'
            )

    # Differentiate u (which is (..., *spatial, d)) along each spatial
    # axis.  Spatial axes are positions [-(d+1), -d).  axis -1 is the
    # displacement-component axis (not differentiated).
    cols = []
    for j in range(d):
        spatial_axis_j = -(d + 1) + j  # 0 <= j < d
        cols.append(_central_diff_along_axis(
            u, spatial_axis_j,
            mode=boundary_mode, spacing=spacing_per_axis[j],
        ))
    # Stack columns into the Jacobian: each cols[j] has shape
    # (..., *spatial, d) representing ∂u/∂x_j (one displacement
    # component per index along the trailing axis).  J[..., i, j] is
    # the i-th displacement's j-th partial.  Stack along a new axis
    # at position -1: cols[j][..., i] -> J[..., i, j].
    J_off = jnp.stack(cols, axis=-1)  # (..., *spatial, d, d)
    # Add the identity (δ_{ij}) for J = I + ∇u.
    eye = jnp.eye(d, dtype=u.dtype)
    return J_off + eye


def jacobian_det_displacement(
    u: Float[Array, '... *spatial d'],
    *,
    boundary_mode: Literal['nearest', 'wrap', 'mirror'] = 'nearest',
    spacing: Union[float, Sequence[float]] = 1.0,
) -> Float[Array, '... *spatial']:
    '''Per-point determinant of the deformation Jacobian.

    Computes ``det(I + ∇u)`` at each spatial location.  For
    ``d <= 3`` uses the explicit closed-form determinant (avoiding
    LU factorisation noise and ``O(d^3)`` cost); for ``d > 3``
    falls back to ``jnp.linalg.det``.

    Parameters and ``boundary_mode`` / ``spacing`` semantics match
    ``jacobian_displacement``.

    Folding interpretation: ``det(J) > 0`` means the deformation
    is locally orientation-preserving; ``det(J) <= 0`` indicates
    the warp folds at this voxel (and is not a diffeomorphism
    there).  The standard QA threshold is "no voxels with
    ``det <= 0``"; the soft regulariser is
    ``mean(max(0, ε - det)^2)`` for some ``ε > 0``.
    '''
    J = jacobian_displacement(
        u, boundary_mode=boundary_mode, spacing=spacing,
    )
    d = u.shape[-1]
    if d == 1:
        return J[..., 0, 0]
    if d == 2:
        # J = [[a, b], [c, d]]; det = a*d - b*c
        a = J[..., 0, 0]
        b = J[..., 0, 1]
        c = J[..., 1, 0]
        d_ = J[..., 1, 1]
        return a * d_ - b * c
    if d == 3:
        # Rule of Sarrus.
        a = J[..., 0, 0]; b = J[..., 0, 1]; c = J[..., 0, 2]
        d_ = J[..., 1, 0]; e = J[..., 1, 1]; f = J[..., 1, 2]
        g = J[..., 2, 0]; h = J[..., 2, 1]; i = J[..., 2, 2]
        return (
            a * (e * i - f * h)
            - b * (d_ * i - f * g)
            + c * (d_ * h - e * g)
        )
    return jnp.linalg.det(J)


# ---------------------------------------------------------------------------
# Legacy aliases (kept for migration; removed at v0.1)
# ---------------------------------------------------------------------------


cmass_regular_grid = center_of_mass_grid
vec_int = integrate_velocity_field
rescale = resample
