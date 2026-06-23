# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Regular-grid deformable-registration primitives.

The voxelmorph-style operations needed to express dense deformation
fields and integrate stationary velocity fields:

- ``identity_grid``       -- the "no-op" deformation, useful as a
  starting point for learned displacements.
- ``spatial_transform``   -- warp an image by a displacement field;
  the sampling kernel is selectable via ``method=``.
- ``integrate_velocity_field`` -- scaling-and-squaring SVF
  integration (the diffeomorphic exponential map).
- ``resample``            -- resize to a target spatial shape; the
  common dispatcher over the available interpolation methods.

``spatial_transform`` and ``resample`` share one sampler: they differ
only in *where* the sample coordinates come from (a deformation field
vs an align-corners resize grid).  The interpolation *kernel* is an
orthogonal axis selected by ``method=`` -- an immutable
``Interpolator`` record (``Linear`` (default), ``NearestNeighbour``,
``Lanczos``, ``CubicBSpline``, ``MultiLabel``).  See
``geometry._interpolate``.

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

import math
from typing import Any, Callable, Literal, Optional, Sequence, Union, cast

import jax
import jax.numpy as jnp
from jax.typing import DTypeLike
from jaxtyping import Array, Float

from ._interpolate import (
    BoundaryMode,
    CubicBSpline,
    Interpolator,
    Lanczos,
    Linear,
    MultiLabel,
    NearestNeighbour,
    _resample_on_grid,
    _sample_at_coords,
)

__all__ = [
    'identity_grid',
    'spatial_transform',
    'spatial_transform_batched',
    'sample_at_points',
    'integrate_velocity_field',
    'jacobian_displacement',
    'jacobian_det_displacement',
    'resample',
    'center_of_mass_grid',
    # interpolation-method ADT (re-exported from ``._interpolate``)
    'Interpolator',
    'Linear',
    'NearestNeighbour',
    'Lanczos',
    'CubicBSpline',
    'MultiLabel',
    # legacy alias kept for now -- remove at v0.1 cleanup
    'cmass_regular_grid',
    'vec_int',
    'rescale',
]


# ``BoundaryMode`` (the ``map_coordinates`` boundary-mode Literal) and the
# ``Interpolator`` records are defined in ``._interpolate`` and imported
# above; ``grid`` re-exports the records as part of its public surface.


# ---------------------------------------------------------------------------
# Identity grid
# ---------------------------------------------------------------------------


def identity_grid(
    spatial_shape: Sequence[int],
    *,
    dtype: DTypeLike = jnp.float32,
) -> Float[Array, '*spatial ndim']:
    """The identity deformation: per-pixel its own coordinate.

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
    """
    spatial_shape = tuple(int(s) for s in spatial_shape)
    grids = jnp.meshgrid(
        *[jnp.arange(s, dtype=dtype) for s in spatial_shape],
        indexing='ij',
    )
    return jnp.stack(grids, axis=-1)


# ---------------------------------------------------------------------------
# Spatial transform
# ---------------------------------------------------------------------------


def spatial_transform(
    image: Float[Array, '*leading *spatial c'],
    deformation: Float[Array, '*leading *spatial ndim'],
    *,
    mode: BoundaryMode = 'constant',
    cval: float = 0.0,
    method: Interpolator = Linear(),
) -> Float[Array, '*leading *spatial c']:
    """Warp an image by a per-pixel deformation field.

    For each output pixel at coordinate ``p``, the result is
    ``image`` sampled at ``deformation[p]`` via the interpolation
    ``method`` ((multi-)linear by default).  ``deformation`` is
    therefore an *absolute* coordinate map, not a relative
    displacement.  To use a displacement field ``delta``, add
    ``identity_grid`` first::

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
    method
        Interpolation kernel -- an ``Interpolator`` record.  Default
        ``Linear()`` ((multi-)linear, the prior behaviour).
        ``NearestNeighbour()`` is the label-preserving choice but is
        not differentiable w.r.t. the deformation (so it cannot drive
        a registration loss); see ``geometry._interpolate`` for the
        full method set and their differentiability contracts.

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
    error-prone).  For a leading-batch convenience that broadcasts
    a shared image or deformation, see ``spatial_transform_batched``.

    No cubic (``order=3``) B-spline path exists yet (tracked in
    ``docs/feature-requests/cubic-resample.md``).  See ``resample``
    Notes for the parity implication.
    """
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

    def core(
        image_: Float[Array, '*spatial c'],
        deformation_: Float[Array, '*spatial ndim'],
    ) -> Float[Array, '*spatial c']:
        return _sample_at_coords(
            image_,
            deformation_,
            method=method,
            mode=mode,
            cval=cval,
        )

    fn: Callable[..., Any] = core
    for _ in range(len(image_batch)):
        fn = jax.vmap(fn, in_axes=(0, 0))
    # ``jax.vmap`` erases the return type to Any; restore it.  (cast is
    # runtime-evaluated, so use a single-variadic shape -- jaxtyping
    # rejects two ``*`` specifiers when the form is constructed.)
    return cast(Float[Array, '...'], fn(image, deformation))


def spatial_transform_batched(
    image: Float[Array, '...'],
    deformation: Float[Array, '...'],
    *,
    mode: BoundaryMode = 'constant',
    cval: float = 0.0,
    method: Interpolator = Linear(),
) -> Float[Array, '...']:
    """Map ``spatial_transform`` over a single leading batch axis.

    Convenience wrapper for the common case where the batch axis is
    carried by only one operand: a batch of images warped by a
    single shared deformation (or a single image under a batch of
    deformations).  ``spatial_transform`` itself requires the
    leading batch dims of ``image`` and ``deformation`` to match
    exactly -- it deliberately does not broadcast -- so this case
    otherwise needs an explicit ``jax.vmap`` with the right
    ``in_axes``.  This picks ``in_axes`` from whichever operand
    carries the leading axis and saves the per-model wrapper line.

    The batch axis is identified as a leading axis beyond the
    ``ndim + 1`` "core" rank (``*spatial`` plus the trailing
    channel / ``ndim`` axis).  If both operands carry it, this maps
    one axis and ``spatial_transform`` composes any further leading
    dims natively.

    Parameters
    ----------
    image
        ``(*spatial, c)`` or ``(B, *spatial, c)``.
    deformation
        ``(*spatial, ndim)`` or ``(B, *spatial, ndim)``.  At least
        one of ``image`` / ``deformation`` must carry the leading
        batch axis.
    mode, cval, method
        Forwarded to ``spatial_transform``.

    Returns
    -------
    ``(B, *spatial, c)`` warped batch.

    Raises
    ------
    ValueError
        If neither operand carries a leading batch axis (call
        ``spatial_transform`` directly for the unbatched case).

    Notes
    -----
    Maps over exactly **one** leading axis.  To broadcast a shared
    operand across *several* leading batch dims, broadcast it to the
    full batch shape and call ``spatial_transform`` directly (which
    handles multiple matching leading dims natively).
    """
    ndim = deformation.shape[-1]
    core_rank = ndim + 1
    image_batched = image.ndim > core_rank
    deform_batched = deformation.ndim > core_rank
    if not image_batched and not deform_batched:
        raise ValueError(
            'spatial_transform_batched: neither image '
            f'(ndim={image.ndim}) nor deformation '
            f'(ndim={deformation.ndim}) carries a leading batch axis '
            f'beyond the core rank {core_rank}.  Call '
            'spatial_transform directly for the unbatched case.'
        )
    in_axes: tuple[Optional[int], Optional[int]] = (
        0 if image_batched else None,
        0 if deform_batched else None,
    )

    def core(
        image_: Float[Array, '*spatial c'],
        deformation_: Float[Array, '*spatial ndim'],
    ) -> Float[Array, '*spatial c']:
        return spatial_transform(
            image_,
            deformation_,
            mode=mode,
            cval=cval,
            method=method,
        )

    # ``jax.vmap`` erases the return type to Any (matching the
    # ``spatial_transform`` pattern above); restore it via cast.
    vfn: Callable[..., Any] = jax.vmap(core, in_axes=in_axes)
    return cast(Float[Array, '...'], vfn(image, deformation))


# ---------------------------------------------------------------------------
# Integrate stationary velocity field
# ---------------------------------------------------------------------------


# ``n_steps='auto'`` bounds (F2): the floor keeps at least one squaring (so the
# result is the composed flow, not the bare first-order step); the cap bounds
# the scan length for a pathologically large velocity -- a warp beyond ~0.5*2^cap
# voxels cannot stay diffeomorphic on the grid regardless of squaring count (a
# resolution limit, not an integration-step one), so more steps are wasted.
_AUTO_SS_FLOOR = 1
_AUTO_SS_CAP = 12


def _resolve_n_steps(
    velocity: Float[Array, '*spatial ndim'],
    n_steps: Union[int, Literal['auto']],
) -> int:
    """Resolve ``n_steps`` (F2): pass an ``int`` through; derive ``'auto'``.

    Scaling-and-squaring is diffeomorphic only when the first sub-step ``v /
    2**n_steps`` is small enough (~<= 0.5 vox) that ``id + v/2**n`` is itself
    invertible; below that the integrated flow can fold (measured: a moderate
    velocity folds at ``n_steps=0``).  ``'auto'`` sets the standard
    ``n_steps = ceil(log2(2 * max|v|))`` so the first sub-step clears the regime,
    clamped to ``[_AUTO_SS_FLOOR, _AUTO_SS_CAP]``.  It reads ``max|v|`` as a
    **concrete** value, so it is unavailable under ``jax.jit`` (the scan length
    must be static, which a traced reduction cannot provide) -- pass an explicit
    ``int`` there (the recipes do; their pre-affine-aligned residual velocities
    sit well inside the default ``n_steps``'s regime).
    """
    if isinstance(n_steps, int):
        return n_steps
    if n_steps != 'auto':
        raise ValueError(f"n_steps must be an int or 'auto'; got {n_steps!r}.")
    try:
        max_v = float(jnp.max(jnp.sqrt(jnp.sum(velocity * velocity, axis=-1))))
    except (
        jax.errors.TracerArrayConversionError,
        jax.errors.ConcretizationTypeError,
    ) as exc:
        raise ValueError(
            "integrate_velocity_field(n_steps='auto') needs a concrete velocity "
            '-- it reads max|v| to size the static scaling-and-squaring scan, '
            'which jax.jit cannot do from a traced array.  Pass an explicit int '
            'n_steps under jit.'
        ) from exc
    rule = math.ceil(math.log2(max(2.0 * max_v, 1.0)))
    return int(min(_AUTO_SS_CAP, max(_AUTO_SS_FLOOR, rule)))


def integrate_velocity_field(
    velocity: Float[Array, '*spatial ndim'],
    *,
    n_steps: Union[int, Literal['auto']] = 7,
    mode: BoundaryMode = 'nearest',
    cval: float = 0.0,
) -> Float[Array, '*spatial ndim']:
    """Diffeomorphic exponential map via scaling-and-squaring.

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
        Number of doublings.  Default ``7`` (i.e., 128× scaling).  Larger
        ``n_steps`` -> more accurate but more compute (the per-step accuracy
        roughly halves with each doubling).  Scaling-and-squaring is
        diffeomorphic only while the first sub-step ``max|v| / 2**n_steps`` is
        within the ``~0.5``-voxel regime where ``id + v/2**n`` is itself
        invertible -- below that the flow can **fold**, so ``n_steps`` should
        cover the velocity magnitude (``>= ceil(log2(2*max|v|))``).  ``'auto'``
        picks exactly that (F2) from the velocity, but reads ``max|v|`` as a
        concrete value, so it is **eager-only** (unavailable under ``jax.jit`` --
        the scan length must be static; pass an explicit ``int`` there).  Note a
        warp larger than the grid can resolve folds at *any* ``n_steps`` (a
        resolution limit, not an integration one).
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
    """
    spatial_rank = velocity.shape[-1]
    spatial_shape = velocity.shape[:-1]
    if len(spatial_shape) != spatial_rank:
        raise ValueError(
            f'velocity rank mismatch: spatial_shape={spatial_shape} '
            f'has {len(spatial_shape)} dims but trailing ndim is '
            f'{spatial_rank}.'
        )
    n_steps = _resolve_n_steps(velocity, n_steps)  # F2: 'auto' -> int (eager)
    id_grid = identity_grid(spatial_shape, dtype=velocity.dtype)
    # Initial small step.
    phi0 = velocity / float(2**n_steps)

    # ``phi`` is treated as a displacement (relative); we compose by
    # warping it through the deformation ``id + phi`` each step.  The
    # doubling loop is rolled with ``lax.scan`` (static ``n_steps``), so
    # it compiles once and stays differentiable -- the diffeomorphic
    # recipe nests this inside its iteration loop, so the unroll would
    # otherwise multiply the compiled graph.
    def _double(phi: Array, _: Any) -> tuple[Array, None]:
        composed = phi + spatial_transform(
            phi,
            id_grid + phi,
            mode=mode,
            cval=cval,
        )
        return composed, None

    phi, _ = jax.lax.scan(_double, phi0, xs=None, length=n_steps)
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
    method: Interpolator = Linear(),
) -> Float[Array, '*target c']:
    """Resize a channel-last image to ``target_shape``.

    The common dispatcher over the available interpolation methods.
    Sample positions are evenly distributed: output pixel ``i`` along
    each axis samples the input at coordinate
    ``i * (in_size - 1) / (out_size - 1)``.  This matches scipy's
    ``align_corners=True`` convention and preserves the boundary
    pixels exactly.  The interpolation *kernel* is selected by
    ``method``; the align-corners grid above is the same regardless.

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
    method
        Interpolation kernel -- an ``Interpolator`` record.  Default
        ``Linear()`` ((multi-)linear, the prior behaviour).
        ``NearestNeighbour()`` preserves label values exactly (the
        choice for integer segmentations when anti-aliasing is *not*
        wanted).  See ``geometry._interpolate`` for the method set and
        their differentiability contracts.

    Returns
    -------
    Resampled image, ``(*target_shape, c)``.

    Notes
    -----
    No cubic (``order=3``) B-spline path exists yet, so bit-parity
    with resamplers that default to order-3 splines (e.g. nnUNet /
    ``hd_bet`` preprocessing, ``scipy.ndimage.zoom(order=3)``) is
    **not** achieved; ``Linear`` is adequate for most consumers.  A
    separable B-spline prefilter + cubic sampling is tracked as a
    future enhancement (``docs/feature-requests/cubic-resample.md``).
    """
    spatial_shape = image.shape[:-1]
    if len(target_shape) != len(spatial_shape):
        raise ValueError(
            f'target_shape={target_shape} must match spatial rank '
            f'of image (got image spatial shape {spatial_shape}).'
        )
    # Build the per-axis 1-D sample-coordinate vectors of the resize
    # grid.  The grid is their outer product, so a separable kernel
    # resamples it axis-by-axis (cheap 1-D passes) rather than over the
    # dense meshgrid -- ``_resample_on_grid`` makes that choice.
    axes = []
    for in_size, out_size in zip(spatial_shape, target_shape):
        if out_size < 2:
            ax = jnp.array([0.0], dtype=image.dtype)
        else:
            ax = jnp.linspace(
                0.0,
                float(in_size - 1),
                int(out_size),
                dtype=image.dtype,
            )
        axes.append(ax)
    return _resample_on_grid(
        image,
        axes,
        method=method,
        mode=mode,
        cval=cval,
    )


# ---------------------------------------------------------------------------
# Centre of mass on a regular grid
# ---------------------------------------------------------------------------


def center_of_mass_grid(
    weight: Float[Array, '...'],
    *,
    axes: Optional[Sequence[int]] = None,
    na_value: Optional[float] = None,
) -> Float[Array, '... ndim']:
    """Centre of mass over a regular grid of unit-spaced coordinates.

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
    """
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
    """Per-axis central difference of a channel-last field.

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
    """
    n = field.ndim
    ax = spatial_axis % n
    # Build the "next" and "prev" along ax by rolling, then patch
    # the boundary rows according to ``mode``.
    nxt = jnp.roll(field, -1, axis=ax)
    prv = jnp.roll(field, +1, axis=ax)
    if mode == 'nearest':
        # Replace the wrapped boundary values with the edge cell.
        L = field.shape[ax]
        sl_last: list[Any] = [slice(None)] * n
        sl_last[ax] = L - 1
        sl_first: list[Any] = [slice(None)] * n
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
        sl_lm1: list[Any] = [slice(None)] * n
        sl_lm1[ax] = L - 2
        nxt = nxt.at[tuple(sl_last)].set(field[tuple(sl_lm1)])
        # prv at 0 -> field[1]
        sl_one: list[Any] = [slice(None)] * n
        sl_one[ax] = 1
        prv = prv.at[tuple(sl_first)].set(field[tuple(sl_one)])
    else:
        raise ValueError(
            f'boundary_mode={mode!r}; expected "nearest", "wrap", or "mirror".'
        )
    return (nxt - prv) / (2.0 * spacing)


def _negate_at_index(x: Array, axis: int, index: int) -> Array:
    """Negate ``x`` at ``index`` along ``axis``, leave the rest unchanged."""
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
    """Per-point Jacobian of the deformation φ = id + u.

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
    """
    d = u.shape[-1]
    spatial_shape = u.shape[-(d + 1) : -1]
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
        cols.append(
            _central_diff_along_axis(
                u,
                spatial_axis_j,
                mode=boundary_mode,
                spacing=spacing_per_axis[j],
            )
        )
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
    """Per-point determinant of the deformation Jacobian.

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
    """
    J = jacobian_displacement(
        u,
        boundary_mode=boundary_mode,
        spacing=spacing,
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
        a = J[..., 0, 0]
        b = J[..., 0, 1]
        c = J[..., 0, 2]
        d_ = J[..., 1, 0]
        e = J[..., 1, 1]
        f = J[..., 1, 2]
        g = J[..., 2, 0]
        h = J[..., 2, 1]
        i = J[..., 2, 2]
        return (
            a * (e * i - f * h) - b * (d_ * i - f * g) + c * (d_ * h - e * g)
        )
    # ``jnp.linalg.det`` is typed as returning Any; restore the array type.
    return cast(Float[Array, '...'], jnp.linalg.det(J))


# ---------------------------------------------------------------------------
# Legacy aliases (kept for migration; removed at v0.1)
# ---------------------------------------------------------------------------


cmass_regular_grid = center_of_mass_grid
vec_int = integrate_velocity_field
rescale = resample


def sample_at_points(
    volume: Float[Array, '...'],
    points: Float[Array, '*n ndim'],
    *,
    method: Interpolator = Linear(),
    mode: BoundaryMode = 'constant',
    cval: float = 0.0,
) -> Float[Array, '...']:
    """Sample a volume at an arbitrary list of continuous points.

    The scattered-point complement to ``spatial_transform`` (which samples
    on a coordinate *grid*): given ``points`` of shape ``(*n, ndim)`` in
    **index/voxel coordinates** (coordinate ``c`` maps to array index
    ``c`` -- the ``align_corners`` convention, matching ``identity_grid``),
    interpolate ``volume`` at each point.

    Parameters
    ----------
    volume
        Either channel-free ``(*spatial)`` or channels-last
        ``(*spatial, c)``; the spatial rank is ``ndim = points.shape[-1]``.
    points
        ``(*n, ndim)`` sample coordinates in index space.
    method
        Interpolation kernel (``Linear`` default; ``NearestNeighbour`` for
        label volumes; any ``geometry`` ``Interpolator``).
    mode
        Out-of-bounds boundary mode: ``"constant"`` (zero-/``cval``-fill,
        the default) or ``"nearest"`` (edge-clamp / border), etc.
    cval
        Fill value for ``mode="constant"``.

    Returns
    -------
    ``(*n)`` for a channel-free volume, or ``(*n, c)`` for a channels-last
    volume.
    """
    ndim = points.shape[-1]
    if volume.ndim == ndim:
        sampled = _sample_at_coords(
            volume[..., None], points, method=method, mode=mode, cval=cval
        )
        return sampled[..., 0]
    if volume.ndim == ndim + 1:
        return _sample_at_coords(
            volume, points, method=method, mode=mode, cval=cval
        )
    raise ValueError(
        f'volume.ndim={volume.ndim} is neither ndim={ndim} (channel-free) '
        f'nor ndim+1 (channels-last) for points with {ndim} coordinates.'
    )
