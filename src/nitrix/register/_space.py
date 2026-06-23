# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Coordinate space a matrix-transform recipe optimises in.

A registration recipe optimises a transform that maps the **fixed** image
onto the **moving** image; that transform has to be turned into a
per-voxel sampling grid (``fixed-voxel -> moving-voxel``) the warp
consumes.  *How* the parameter transform relates the two images' voxel
grids is the one axis that separates a within-grid, index-space
registration from a physically correct one -- and it is the **only** axis:
the pyramid, the coarse-to-fine loop, the per-level optimise, and the
result assembly are identical either way.  So that axis is factored behind
a ``CoordinateSpace`` (the ``Interpolator`` / ``Metric`` / ``TransformModel``
Protocol-plus-frozen-record precedent), and the driver
(:func:`._core.multi_resolution_register`) is written once against it.

Two implementers:

- :class:`IndexSpace` (default) -- the transform is read **directly** in
  voxel/index coordinates, about the fixed-image centre; both images are
  assumed to share a voxel grid (the within-modality / motion-correction
  norm).  Its sampling matrix is just ``model.exp(params)`` -- **no general
  affine inverse**, so it stays fully on-device (no cuSolver-fallback host
  sync), and the parameter vector is in voxel units (the inter-level
  warm-start rescales translations with the grid).  This is the leaner
  path, and the natural frame for the closed-form steepest-descent /
  inverse-compositional kernel (the transform acts on the fixed grid with
  no conjugating change-of-basis).
- :class:`WorldSpace` -- the transform is read in **physical/world**
  coordinates (mm).  Each image carries a voxel->world affine; the sampling
  map is the composition ``A_moving⁻¹ · T_world · A_fixed`` (with the
  rotation centred at the fixed image's world centre).  This is the path
  that is correct under **anisotropic voxels** (a rigid transform in index
  space shears in physical space) and that relates two images on
  **different grids**.  It pays one ``safe_inv`` of the moving affine per
  registration (a 4x4, once -- not in the hot loop).

The space's three responsibilities are the three things that differ:
``index_sampling`` (params transform -> sampling matrix + grid centre),
``requires_grid_rescale`` (whether the warm-start rescales voxel-unit
params between levels), and ``result_transform`` (what "the recovered
matrix" means -- a voxel-space index map for :class:`IndexSpace`, a
world->world transform for :class:`WorldSpace`).  **Both spaces return a
self-contained matrix**: the centre the warp applies the transform about (the
fixed grid centre for index space, the fixed world centre for world space) is
baked into ``result.matrix`` via :func:`_conjugate_about`, so ``apply_affine``
reproduces the warp and the two compose -- the raw about-origin parameters live
on ``result.params``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Optional, Protocol

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from ..geometry import apply_affine
from ..linalg._solver import safe_inv

__all__ = [
    'CoordinateSpace',
    'IndexSpace',
    'WorldSpace',
]


# ---------------------------------------------------------------------------
# Small homogeneous-matrix builders (index space, GPU-native)
# ---------------------------------------------------------------------------


def _translation_homog(
    t: Float[Array, ' d'], ndim: int, dtype: jnp.dtype
) -> Float[Array, ' d1 d1']:
    """Homogeneous ``(ndim+1, ndim+1)`` pure-translation matrix."""
    eye = jnp.eye(ndim + 1, dtype=dtype)
    return eye.at[:ndim, ndim].set(t.astype(dtype))


def _scale_homog(
    s: Any, ndim: int, dtype: jnp.dtype
) -> Float[Array, ' d1 d1']:
    """Homogeneous ``(ndim+1, ndim+1)`` per-axis diagonal scaling matrix.

    ``s`` is array-like (a host NumPy ``_grid_scale`` constant or an ``Array``);
    it is cast to ``dtype`` here."""
    eye = jnp.eye(ndim + 1, dtype=dtype)
    idx = jnp.arange(ndim)
    return eye.at[idx, idx].set(jnp.asarray(s, dtype=dtype))


def _conjugate_about(
    transform: Float[Array, ' d1 d1'],
    center: Float[Array, ' d'],
    ndim: int,
    dtype: jnp.dtype,
) -> Float[Array, ' d1 d1']:
    """``Tr(center) @ transform @ Tr(-center)`` -- re-express an about-origin
    homogeneous transform as the equivalent map applied about ``center``.

    The shared centering used to make a returned matrix **self-contained**: the
    optimiser works with ``model.exp(params)`` about the origin, but the warp
    applies it about the grid (or world) centre, so the returned matrix bakes
    that centre in -- ``apply_affine(coords, matrix)`` then reproduces the warp
    directly, and two centred matrices compose without re-deriving the centre.
    """
    t_pos = _translation_homog(center, ndim, dtype)
    t_neg = _translation_homog(-center, ndim, dtype)
    return t_pos @ transform @ t_neg


def _grid_scale(
    full_shape: tuple[int, ...], shape: tuple[int, ...]
) -> np.ndarray:
    """Per-axis ``align-corners`` index scale from a pyramid level to full.

    ``downsample`` resamples with the align-corners grid (output voxel
    ``i`` samples input ``i * (in-1)/(out-1)``), so a level-``l`` voxel
    index maps to the full-resolution index by the pure per-axis scale
    ``(full - 1) / (level - 1)`` (no offset; corner 0 -> 0).

    Computed on the **static** shape tuples with NumPy (a host constant), so no
    ``float64`` literal enters the traced graph to be silently down-cast under
    x64-off (float32); the consumer (``_scale_homog``) casts to the run dtype.
    """
    full = np.asarray(full_shape, dtype=np.float64)
    lvl = np.asarray(shape, dtype=np.float64)
    # Guard the degenerate single-voxel axis (no scale is well-defined).
    denom = np.maximum(lvl - 1.0, 1.0)
    return (full - 1.0) / denom


# ---------------------------------------------------------------------------
# Resolved per-registration samplers (the driver calls these per level)
# ---------------------------------------------------------------------------


class _Sampler(Protocol):
    """Per-registration sampling map (already bound to the full shapes)."""

    def index_sampling(
        self,
        transform: Float[Array, ' d1 d1'],
        *,
        fixed_shape: tuple[int, ...],
        moving_shape: tuple[int, ...],
    ) -> tuple[Float[Array, ' d1 d1'], Float[Array, ' d']]:
        """``(fixed-voxel -> moving-voxel)`` matrix + ``affine_grid`` centre."""
        ...

    def result_transform(
        self, transform: Float[Array, ' d1 d1']
    ) -> Float[Array, ' d1 d1']:
        """Map the optimum's ``model.exp`` transform to the returned matrix."""
        ...


@dataclass(frozen=True)
class _IndexSampler:
    """Sampling in voxel/index space (the transform acts directly)."""

    ndim: int
    dtype: jnp.dtype
    full_fixed_shape: tuple[int, ...]

    def index_sampling(
        self,
        transform: Float[Array, ' d1 d1'],
        *,
        fixed_shape: tuple[int, ...],
        moving_shape: tuple[int, ...],
    ) -> tuple[Float[Array, ' d1 d1'], Float[Array, ' d']]:
        center = (jnp.asarray(fixed_shape, dtype=self.dtype) - 1.0) / 2.0
        return transform, center

    def result_transform(
        self, transform: Float[Array, ' d1 d1']
    ) -> Float[Array, ' d1 d1']:
        # Bake in the full-res grid centre (the warp applies the transform about
        # that centre via ``affine_grid``), so the returned matrix is the
        # self-contained ``fixed-voxel -> moving-voxel`` map -- directly usable
        # by ``apply_affine`` and composable with a WorldSpace result (which
        # centres the same way at line ~204).  ``params`` keeps the raw
        # about-origin Lie coordinates.
        center = (
            jnp.asarray(self.full_fixed_shape, dtype=self.dtype) - 1.0
        ) / 2.0
        return _conjugate_about(transform, center, self.ndim, self.dtype)


@dataclass(frozen=True)
class _WorldSampler:
    """Sampling in physical/world space (``A_m⁻¹ · T_world · A_f``)."""

    ndim: int
    dtype: jnp.dtype
    fixed_affine: Float[Array, ' d1 d1']
    moving_affine_inv: Float[Array, ' d1 d1']
    t_pos: Float[Array, ' d1 d1']
    t_neg: Float[Array, ' d1 d1']
    full_fixed_shape: tuple[int, ...]
    full_moving_shape: tuple[int, ...]

    def index_sampling(
        self,
        transform: Float[Array, ' d1 d1'],
        *,
        fixed_shape: tuple[int, ...],
        moving_shape: tuple[int, ...],
    ) -> tuple[Float[Array, ' d1 d1'], Float[Array, ' d']]:
        s_f = _grid_scale(self.full_fixed_shape, fixed_shape)
        s_m = _grid_scale(self.full_moving_shape, moving_shape)
        af_l = self.fixed_affine @ _scale_homog(s_f, self.ndim, self.dtype)
        am_l_inv = (
            _scale_homog(1.0 / s_m, self.ndim, self.dtype)
            @ self.moving_affine_inv
        )
        world = self.result_transform(transform)
        matrix = am_l_inv @ world @ af_l
        center = jnp.zeros(self.ndim, dtype=self.dtype)
        return matrix, center

    def result_transform(
        self, transform: Float[Array, ' d1 d1']
    ) -> Float[Array, ' d1 d1']:
        # Centre the world transform at the fixed image's world centre, so
        # the translation parameter is the displacement of that centre.
        return self.t_pos @ transform @ self.t_neg


# ---------------------------------------------------------------------------
# Public coordinate spaces
# ---------------------------------------------------------------------------


class CoordinateSpace(Protocol):
    """Relation between the parameter transform and the two voxel grids.

    Attributes
    ----------
    requires_grid_rescale
        Whether the parameter vector is in **voxel units** (so the
        coarse-to-fine driver must rescale the translation block with the
        grid when warm-starting a finer level).  ``True`` for
        :class:`IndexSpace`; ``False`` for :class:`WorldSpace`, whose
        parameters are grid-independent physical quantities.
    """

    requires_grid_rescale: ClassVar[bool]

    def sampler(
        self,
        *,
        ndim: int,
        full_fixed_shape: tuple[int, ...],
        full_moving_shape: tuple[int, ...],
        dtype: jnp.dtype,
    ) -> _Sampler:
        """Bind the space to one registration's full-resolution shapes."""
        ...


@dataclass(frozen=True)
class IndexSpace:
    """Register in voxel/index space (shared-grid, on-device fast path).

    The parameter transform is read directly in voxel coordinates, about
    the fixed-image centre; the sampling matrix is ``model.exp(params)``
    with no change-of-basis.  Assumes ``moving`` and ``fixed`` share a
    voxel grid (the within-modality / motion-correction norm); under
    **anisotropic** voxels a rigid parametrisation is not physically rigid
    -- use :class:`WorldSpace` there.

    No general matrix inverse is taken, so the path stays fully on-device
    (no cuSolver fallback).  ``result.matrix`` is the **self-contained**
    ``fixed-voxel -> moving-voxel`` homogeneous index map -- ``model.exp(params)``
    re-centred at the fixed grid centre (``_conjugate_about``), so
    ``apply_affine(coords, result.matrix)`` reproduces the warp directly and it
    composes with a ``WorldSpace`` result.  ``result.params`` holds the raw
    about-origin Lie coordinates.
    """

    requires_grid_rescale: ClassVar[bool] = True

    def sampler(
        self,
        *,
        ndim: int,
        full_fixed_shape: tuple[int, ...],
        full_moving_shape: tuple[int, ...],
        dtype: jnp.dtype,
    ) -> _IndexSampler:
        return _IndexSampler(
            ndim=ndim, dtype=dtype, full_fixed_shape=full_fixed_shape
        )


@dataclass(frozen=True)
class WorldSpace:
    """Register in physical/world space (anisotropy- and grid-correct).

    Each image carries a voxel->world affine (``(ndim+1, ndim+1)``
    homogeneous, e.g. a NIfTI ``sform``); ``None`` defaults to the
    identity (voxel == world).  The parameter transform is the world->world
    map, and the per-voxel sampling matrix is the composition
    ``A_moving⁻¹ · T_world · A_fixed`` (with ``T_world`` centred at the
    fixed image's world centre), built per pyramid level from the
    align-corners level scale.

    This is the correct path when voxels are anisotropic (a rotation in
    index space shears in physical space) or when the two images live on
    different grids.  ``result`` is the ``fixed-world -> moving-world``
    homogeneous transform (mm).  One ``safe_inv`` of the moving affine is
    taken per registration (a 4x4, once -- not in the hot loop).

    Attributes
    ----------
    fixed_affine, moving_affine
        Voxel->world homogeneous affines ``(ndim+1, ndim+1)``.  ``None``
        -> identity.  For pure anisotropic spacing pass
        ``diag([*spacing, 1])``.
    """

    fixed_affine: Optional[Float[Array, ' d1 d1']] = None
    moving_affine: Optional[Float[Array, ' d1 d1']] = None

    requires_grid_rescale: ClassVar[bool] = False

    def sampler(
        self,
        *,
        ndim: int,
        full_fixed_shape: tuple[int, ...],
        full_moving_shape: tuple[int, ...],
        dtype: jnp.dtype,
    ) -> _WorldSampler:
        eye = jnp.eye(ndim + 1, dtype=dtype)
        a_f = (
            eye
            if self.fixed_affine is None
            else jnp.asarray(self.fixed_affine, dtype=dtype)
        )
        a_m = (
            eye
            if self.moving_affine is None
            else jnp.asarray(self.moving_affine, dtype=dtype)
        )
        a_m_inv = safe_inv(a_m)
        # World centre = the fixed image's full-res voxel centre, mapped to
        # world.  Level-independent, so the translation parameter means the
        # same physical displacement at every resolution.
        c_full = (jnp.asarray(full_fixed_shape, dtype=dtype) - 1.0) / 2.0
        c_world = apply_affine(c_full, a_f)
        return _WorldSampler(
            ndim=ndim,
            dtype=dtype,
            fixed_affine=a_f,
            moving_affine_inv=a_m_inv,
            t_pos=_translation_homog(c_world, ndim, dtype),
            t_neg=_translation_homog(-c_world, ndim, dtype),
            full_fixed_shape=full_fixed_shape,
            full_moving_shape=full_moving_shape,
        )
