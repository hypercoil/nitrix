# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
r"""
Coordinate space a matrix-transform recipe optimises in.

A registration recipe optimises a transform that maps the **fixed** image
onto the **moving** image; that transform has to be turned into a
per-voxel sampling grid (``fixed-voxel -> moving-voxel``) the warp
consumes.  *How* the parameter transform relates the two images' voxel
grids is the one axis that separates a within-grid, index-space
registration from a physically correct one -- and it is the **only** axis:
the pyramid, the coarse-to-fine loop, the per-level optimise, and the
result assembly are identical either way.  That axis is factored behind a
:class:`CoordinateSpace` (following the same protocol-plus-frozen-record
precedent as the interpolator, metric, and transform-model abstractions),
and the coarse-to-fine driver is written once against it.

Two implementers:

- :class:`IndexSpace` (default) -- the transform is read **directly** in
  voxel/index coordinates, about the fixed-image centre; both images are
  assumed to share a voxel grid (the within-modality / motion-correction
  norm).  Its sampling matrix is just the exponentiated parameter transform
  -- **no general affine inverse**, so it stays fully on-device (no
  cuSolver-fallback host sync), and the parameter vector is in voxel units
  (the inter-level warm-start rescales translations with the grid).  This is
  the leaner path, and the natural frame for the closed-form
  steepest-descent / inverse-compositional kernel (the transform acts on the
  fixed grid with no conjugating change-of-basis).
- :class:`WorldSpace` -- the transform is read in **physical/world**
  coordinates (mm).  Each image carries a voxel->world affine; the sampling
  map is the composition :math:`A_{moving}^{-1} \cdot T_{world} \cdot
  A_{fixed}` (with the rotation centred at the fixed image's world centre).
  This is the path that is correct under **anisotropic voxels** (a rigid
  transform in index space shears in physical space) and that relates two
  images on **different grids**.  It pays one inverse of the moving affine
  per registration (a 4x4, once -- not in the hot loop).

The space's three responsibilities are the three things that differ:
mapping the parameter transform to a sampling matrix plus grid centre,
declaring whether the warm-start rescales voxel-unit params between levels,
and defining what "the recovered matrix" means -- a voxel-space index map
for :class:`IndexSpace`, a world->world transform for :class:`WorldSpace`.
**Both spaces return a self-contained matrix**: the centre the warp applies
the transform about (the fixed grid centre for index space, the fixed world
centre for world space) is baked into the result matrix via
:func:`_conjugate_about`, so :func:`~nitrix.geometry.apply_affine`
reproduces the warp and the two compose -- the raw about-origin parameters
live on the result's ``params``.
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
    """Build a homogeneous pure-translation matrix.

    Parameters
    ----------
    t : Float[Array, ' d']
        Translation vector, one entry per spatial axis.
    ndim : int
        Number of spatial dimensions ``d``.
    dtype : jnp.dtype
        Precision of the returned matrix; ``t`` is cast to it.

    Returns
    -------
    Float[Array, ' d1 d1']
        The ``(ndim+1, ndim+1)`` homogeneous matrix whose leading
        ``ndim`` rows of the final column hold ``t`` and whose linear block
        is the identity.
    """
    eye = jnp.eye(ndim + 1, dtype=dtype)
    return eye.at[:ndim, ndim].set(t.astype(dtype))


def _scale_homog(
    s: Any, ndim: int, dtype: jnp.dtype
) -> Float[Array, ' d1 d1']:
    """Build a homogeneous per-axis diagonal scaling matrix.

    Parameters
    ----------
    s : array-like
        Per-axis scale factors, one entry per spatial axis.  Typically a
        host NumPy constant from :func:`_grid_scale` or an ``Array``; it is
        cast to ``dtype`` here.
    ndim : int
        Number of spatial dimensions ``d``.
    dtype : jnp.dtype
        Precision of the returned matrix.

    Returns
    -------
    Float[Array, ' d1 d1']
        The ``(ndim+1, ndim+1)`` homogeneous matrix whose leading
        ``ndim`` diagonal entries are ``s`` and whose final diagonal entry
        is one.
    """
    eye = jnp.eye(ndim + 1, dtype=dtype)
    idx = jnp.arange(ndim)
    return eye.at[idx, idx].set(jnp.asarray(s, dtype=dtype))


def _conjugate_about(
    transform: Float[Array, ' d1 d1'],
    center: Float[Array, ' d'],
    ndim: int,
    dtype: jnp.dtype,
) -> Float[Array, ' d1 d1']:
    r"""Re-express an about-origin transform as a map applied about a centre.

    Computes :math:`Tr(\text{center}) \cdot \text{transform} \cdot
    Tr(-\text{center})`, converting a homogeneous transform defined about the
    origin into the equivalent map applied about ``center``.

    This is the shared centring that makes a returned matrix
    **self-contained**: the optimiser works with the exponentiated parameter
    transform about the origin, but the warp applies it about the grid (or
    world) centre, so the returned matrix bakes that centre in.  Applying the
    baked-in matrix to coordinates then reproduces the warp directly, and two
    centred matrices compose without re-deriving the centre.

    Parameters
    ----------
    transform : Float[Array, ' d1 d1']
        The ``(ndim+1, ndim+1)`` homogeneous transform defined about the
        origin.
    center : Float[Array, ' d']
        The point about which the equivalent transform should act.
    ndim : int
        Number of spatial dimensions ``d``.
    dtype : jnp.dtype
        Precision of the intermediate translation matrices and result.

    Returns
    -------
    Float[Array, ' d1 d1']
        The ``(ndim+1, ndim+1)`` homogeneous transform applied about
        ``center``.
    """
    t_pos = _translation_homog(center, ndim, dtype)
    t_neg = _translation_homog(-center, ndim, dtype)
    return t_pos @ transform @ t_neg


def _grid_scale(
    full_shape: tuple[int, ...], shape: tuple[int, ...]
) -> np.ndarray:
    r"""Per-axis align-corners index scale from a pyramid level to full.

    :func:`~nitrix.geometry.downsample` resamples with the align-corners grid
    (output voxel :math:`i` samples input :math:`i \cdot (in-1)/(out-1)`), so a
    level-:math:`l` voxel index maps to the full-resolution index by the pure
    per-axis scale :math:`(\text{full} - 1) / (\text{level} - 1)` (no offset;
    corner 0 maps to 0).

    Computed on the **static** shape tuples with NumPy (a host constant), so no
    ``float64`` literal enters the traced graph to be silently down-cast under
    x64-off (float32); the consumer, :func:`_scale_homog`, casts to the run
    dtype.

    Parameters
    ----------
    full_shape : tuple of int
        The full-resolution spatial shape, one entry per axis.
    shape : tuple of int
        The pyramid-level spatial shape, one entry per axis.

    Returns
    -------
    numpy.ndarray
        Per-axis scale factors ``(full - 1) / max(shape - 1, 1)`` as a host
        ``float64`` array, one entry per axis.  A degenerate single-voxel
        axis is guarded so its denominator is one.
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
        """Map a parameter transform to a per-voxel sampling matrix and centre.

        Parameters
        ----------
        transform : Float[Array, ' d1 d1']
            The ``(ndim+1, ndim+1)`` homogeneous parameter transform for one
            optimiser step.
        fixed_shape : tuple of int
            Spatial shape of the fixed image at the current pyramid level.
        moving_shape : tuple of int
            Spatial shape of the moving image at the current pyramid level.

        Returns
        -------
        matrix : Float[Array, ' d1 d1']
            The ``(ndim+1, ndim+1)`` homogeneous ``fixed-voxel ->
            moving-voxel`` sampling matrix.
        center : Float[Array, ' d']
            The centre about which :func:`~nitrix.geometry.affine_grid`
            applies the sampling matrix.
        """
        ...

    def result_transform(
        self, transform: Float[Array, ' d1 d1']
    ) -> Float[Array, ' d1 d1']:
        """Map the optimum's parameter transform to the returned matrix.

        Parameters
        ----------
        transform : Float[Array, ' d1 d1']
            The ``(ndim+1, ndim+1)`` homogeneous exponentiated parameter
            transform at the optimum.

        Returns
        -------
        Float[Array, ' d1 d1']
            The self-contained ``(ndim+1, ndim+1)`` homogeneous matrix that
            represents the recovered transform for this space (a voxel-space
            index map or a world->world transform, depending on the
            implementer).
        """
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
    r"""Sampling in physical/world space.

    Composes the per-voxel sampling matrix as :math:`A_m^{-1} \cdot T_{world}
    \cdot A_f`, where :math:`A_f` and :math:`A_m` are the fixed and moving
    voxel->world affines and :math:`T_{world}` is the world->world parameter
    transform centred at the fixed image's world centre.
    """

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
        """Bind the space to one registration's full-resolution shapes.

        Parameters
        ----------
        ndim : int
            Number of spatial dimensions.
        full_fixed_shape : tuple of int
            Full-resolution spatial shape of the fixed image.
        full_moving_shape : tuple of int
            Full-resolution spatial shape of the moving image.
        dtype : jnp.dtype
            Precision of the matrices the resolved sampler builds.

        Returns
        -------
        _Sampler
            A per-registration sampler bound to these shapes, ready to be
            called at each pyramid level.
        """
        ...


@dataclass(frozen=True)
class IndexSpace:
    """Register in voxel/index space (shared-grid, on-device fast path).

    The parameter transform is read directly in voxel coordinates, about
    the fixed-image centre; the sampling matrix is the exponentiated
    parameter transform with no change-of-basis.  Assumes ``moving`` and
    ``fixed`` share a voxel grid (the within-modality / motion-correction
    norm); under **anisotropic** voxels a rigid parametrisation is not
    physically rigid -- use :class:`WorldSpace` there.

    No general matrix inverse is taken, so the path stays fully on-device
    (no cuSolver fallback).  The result matrix is the **self-contained**
    ``fixed-voxel -> moving-voxel`` homogeneous index map -- the
    exponentiated parameter transform re-centred at the fixed grid centre
    via :func:`_conjugate_about`, so applying it to coordinates reproduces
    the warp directly and it composes with a :class:`WorldSpace` result.
    The result's ``params`` hold the raw about-origin Lie coordinates.
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
        """Bind the index space to one registration's full-resolution shapes.

        Parameters
        ----------
        ndim : int
            Number of spatial dimensions.
        full_fixed_shape : tuple of int
            Full-resolution spatial shape of the fixed image.
        full_moving_shape : tuple of int
            Full-resolution spatial shape of the moving image.  Unused in
            index space (both images share a grid), accepted for a uniform
            interface.
        dtype : jnp.dtype
            Precision of the matrices the resolved sampler builds.

        Returns
        -------
        _IndexSampler
            A voxel/index-space sampler bound to the fixed shape and dtype.
        """
        return _IndexSampler(
            ndim=ndim, dtype=dtype, full_fixed_shape=full_fixed_shape
        )


@dataclass(frozen=True)
class WorldSpace:
    r"""Register in physical/world space (anisotropy- and grid-correct).

    Each image carries a voxel->world affine (``(ndim+1, ndim+1)``
    homogeneous, e.g. a NIfTI ``sform``); ``None`` defaults to the
    identity (voxel == world).  The parameter transform is the world->world
    map, and the per-voxel sampling matrix is the composition
    :math:`A_{moving}^{-1} \cdot T_{world} \cdot A_{fixed}` (with
    :math:`T_{world}` centred at the fixed image's world centre), built per
    pyramid level from the align-corners level scale.

    This is the correct path when voxels are anisotropic (a rotation in
    index space shears in physical space) or when the two images live on
    different grids.  The result is the ``fixed-world -> moving-world``
    homogeneous transform (mm).  One numerically safe inverse of the moving
    affine is taken per registration (a 4x4, once -- not in the hot loop).

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
        """Bind the world space to one registration's full-resolution shapes.

        Resolves the fixed and moving voxel->world affines (defaulting each
        to the identity when unset), inverts the moving affine once, and
        precomputes the translation matrices that centre the world transform
        at the fixed image's full-resolution world centre.

        Parameters
        ----------
        ndim : int
            Number of spatial dimensions.
        full_fixed_shape : tuple of int
            Full-resolution spatial shape of the fixed image; sets the world
            centre.
        full_moving_shape : tuple of int
            Full-resolution spatial shape of the moving image; used for the
            per-level align-corners scale.
        dtype : jnp.dtype
            Precision of the resolved affines and matrices.

        Returns
        -------
        _WorldSampler
            A physical/world-space sampler bound to the resolved affines and
            world centre.
        """
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
