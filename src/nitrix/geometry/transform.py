# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Lie-group parametrisation of rigid and affine transforms.

Registration is optimisation on a transform group; this module supplies
the chart between a flat parameter vector (what the Gauss-Newton /
Levenberg-Marquardt optimiser moves) and a homogeneous transform matrix
(what ``affine_grid`` + ``spatial_transform`` apply).

- **Rigid** (``rigid_exp`` / ``rigid_log``) -- rotation from the
  axis-angle (``so(n)``) exponential in **closed form** (Rodrigues in 3-D,
  the planar rotation in 2-D) plus a **direct** translation.  Closed-form
  rotation is GPU-native (no cuSolver), so the rigid optimiser's hot loop
  stays on-device.  Parameter order: 3-D ``(ω_x, ω_y, ω_z, t_x, t_y,
  t_z)`` (6); 2-D ``(θ, t_x, t_y)`` (3).
- **Affine** (``affine_exp``) -- the linear block is ``matrix_exp(A)`` of
  a general ``gl(n)`` generator (guaranteeing an invertible,
  orientation-preserving ``det > 0`` map) plus a direct translation.
  Parameter order: ``n²`` linear entries (row-major ``A``) then ``n``
  translation.  Uses ``linalg.matrix_exp`` (the §12.2 graduation).

``apply_affine`` / ``affine_grid`` turn a homogeneous matrix into the
absolute sample coordinates ``spatial_transform`` consumes, with an
optional rotation/scaling ``center`` (registration rotates about the
image centre, not the voxel origin).

Coordinates are in index space, matching ``identity_grid`` (``grid[i] ==
i``).  Supported spatial ranks: 2 and 3.
"""

from __future__ import annotations

from typing import Optional, Sequence

import jax.numpy as jnp
from jaxtyping import Array, Float

from ..linalg.matrix_function import matrix_exp
from .grid import identity_grid

__all__ = [
    'rigid_exp',
    'rigid_log',
    'affine_exp',
    'apply_affine',
    'affine_grid',
]

_SMALL = 1e-6


def _skew3(omega: Float[Array, '... 3']) -> Float[Array, '... 3 3']:
    """Skew-symmetric matrix ``[ω]_×`` of a 3-vector (batched)."""
    ox = omega[..., 0]
    oy = omega[..., 1]
    oz = omega[..., 2]
    zero = jnp.zeros_like(ox)
    row0 = jnp.stack([zero, -oz, oy], axis=-1)
    row1 = jnp.stack([oz, zero, -ox], axis=-1)
    row2 = jnp.stack([-oy, ox, zero], axis=-1)
    return jnp.stack([row0, row1, row2], axis=-2)


def _so3_exp(omega: Float[Array, '... 3']) -> Float[Array, '... 3 3']:
    """Rodrigues' rotation: ``exp([ω]_×)``, small-angle safe."""
    theta2 = jnp.sum(omega * omega, axis=-1)
    theta = jnp.sqrt(theta2)
    small = theta < _SMALL
    theta_safe = jnp.where(small, 1.0, theta)
    a = jnp.where(small, 1.0 - theta2 / 6.0, jnp.sin(theta_safe) / theta_safe)
    b = jnp.where(
        small,
        0.5 - theta2 / 24.0,
        (1.0 - jnp.cos(theta_safe)) / (theta_safe * theta_safe),
    )
    k = _skew3(omega)
    kk = k @ k
    eye = jnp.eye(3, dtype=omega.dtype)
    return eye + a[..., None, None] * k + b[..., None, None] * kk


def _so3_log(r: Float[Array, '... 3 3']) -> Float[Array, '... 3']:
    """Axis-angle ``ω`` with ``exp([ω]_×) = R``, small-angle safe.

    Valid for rotations away from ``θ = π`` (the registration regime,
    where rotations are small); the antipodal case is not handled.
    """
    trace = r[..., 0, 0] + r[..., 1, 1] + r[..., 2, 2]
    cos_theta = jnp.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    theta = jnp.arccos(cos_theta)
    small = theta < _SMALL
    sin_theta = jnp.where(small, 1.0, jnp.sin(theta))
    vec = jnp.stack(
        [
            r[..., 2, 1] - r[..., 1, 2],
            r[..., 0, 2] - r[..., 2, 0],
            r[..., 1, 0] - r[..., 0, 1],
        ],
        axis=-1,
    )
    factor = jnp.where(
        small, 0.5 + theta * theta / 12.0, theta / (2.0 * sin_theta)
    )
    return factor[..., None] * vec


def _so2_exp(angle: Float[Array, '...']) -> Float[Array, '... 2 2']:
    c = jnp.cos(angle)
    s = jnp.sin(angle)
    row0 = jnp.stack([c, -s], axis=-1)
    row1 = jnp.stack([s, c], axis=-1)
    return jnp.stack([row0, row1], axis=-2)


def _so2_log(r: Float[Array, '... 2 2']) -> Float[Array, '...']:
    return jnp.arctan2(r[..., 1, 0], r[..., 0, 0])


def _homogeneous(
    linear: Float[Array, '... d d'],
    translation: Float[Array, '... d'],
) -> Float[Array, '... d1 d1']:
    """Assemble a homogeneous ``(d+1, d+1)`` matrix from a linear block
    and a translation (batched over leading dims)."""
    d = linear.shape[-1]
    batch = linear.shape[:-2]
    top = jnp.concatenate([linear, translation[..., None]], axis=-1)
    bottom = jnp.zeros(batch + (1, d + 1), dtype=linear.dtype)
    bottom = bottom.at[..., 0, d].set(1.0)
    return jnp.concatenate([top, bottom], axis=-2)


def rigid_exp(
    params: Float[Array, '... p'],
    *,
    ndim: int,
) -> Float[Array, '... d1 d1']:
    """Homogeneous rigid transform from its Lie parameters.

    Parameters
    ----------
    params
        ``(..., p)`` with ``p = 3`` for ``ndim == 2`` (``θ, t_x, t_y``)
        or ``p = 6`` for ``ndim == 3`` (``ω_x, ω_y, ω_z, t_x, t_y,
        t_z``).  ``ω`` is the axis-angle rotation vector; the
        translation is applied directly.
    ndim
        Spatial dimensionality (2 or 3).

    Returns
    -------
    Homogeneous matrix, ``(..., ndim + 1, ndim + 1)``.
    """
    if ndim == 3:
        rot = _so3_exp(params[..., :3])
        trans = params[..., 3:6]
    elif ndim == 2:
        rot = _so2_exp(params[..., 0])
        trans = params[..., 1:3]
    else:
        raise ValueError(f'ndim must be 2 or 3; got {ndim}.')
    return _homogeneous(rot, trans)


def rigid_log(
    matrix: Float[Array, '... d1 d1'],
    *,
    ndim: int,
) -> Float[Array, '... p']:
    """Inverse of ``rigid_exp``: recover the Lie parameters of a rigid
    homogeneous matrix (the rotation block must be a proper rotation)."""
    if ndim == 3:
        omega = _so3_log(matrix[..., :3, :3])
        trans = matrix[..., :3, 3]
        return jnp.concatenate([omega, trans], axis=-1)
    if ndim == 2:
        angle = _so2_log(matrix[..., :2, :2])
        trans = matrix[..., :2, 2]
        return jnp.concatenate([angle[..., None], trans], axis=-1)
    raise ValueError(f'ndim must be 2 or 3; got {ndim}.')


def affine_exp(
    params: Float[Array, '... p'],
    *,
    ndim: int,
) -> Float[Array, '... d1 d1']:
    """Homogeneous affine transform from its Lie parameters.

    The linear block is ``matrix_exp(A)`` of the ``ndim x ndim``
    generator ``A`` (row-major in the first ``ndim²`` parameters),
    guaranteeing an invertible, orientation-preserving map; the
    remaining ``ndim`` parameters are the translation, applied
    directly.  ``params`` has length ``ndim² + ndim`` (12 in 3-D, 6 in
    2-D).
    """
    if ndim not in (2, 3):
        raise ValueError(f'ndim must be 2 or 3; got {ndim}.')
    n2 = ndim * ndim
    batch = params.shape[:-1]
    generator = params[..., :n2].reshape(batch + (ndim, ndim))
    linear = matrix_exp(generator)
    trans = params[..., n2 : n2 + ndim]
    return _homogeneous(linear, trans)


def apply_affine(
    coords: Float[Array, '... d'],
    matrix: Float[Array, 'd1 d1'],
    *,
    center: Optional[Float[Array, 'd']] = None,
) -> Float[Array, '... d']:
    """Apply a homogeneous transform to a field of coordinates.

    ``out = M (p - c) + t + c`` for the linear block ``M`` and
    translation ``t`` of ``matrix``, with optional rotation/scaling
    ``center`` ``c`` (default origin).  ``coords`` is ``(..., ndim)``.
    """
    ndim = matrix.shape[-1] - 1
    linear = matrix[:ndim, :ndim]
    trans = matrix[:ndim, ndim]
    p = coords if center is None else coords - center
    out = p @ linear.T + trans
    if center is not None:
        out = out + center
    return out


def affine_grid(
    matrix: Float[Array, 'd1 d1'],
    spatial_shape: Sequence[int],
    *,
    center: Optional[Float[Array, 'd']] = None,
) -> Float[Array, '*spatial d']:
    """Absolute sample coordinates of a homogeneous transform on a grid.

    Maps each output voxel ``i`` to ``M (i - c) + t + c`` -- the
    coordinate ``spatial_transform`` samples the moving image at.

    Parameters
    ----------
    matrix
        Homogeneous transform, ``(ndim + 1, ndim + 1)``.
    spatial_shape
        Output grid shape (length ``ndim``).
    center
        Rotation/scaling centre.  ``None`` (default) uses the grid
        centre ``(shape - 1) / 2`` -- the registration convention
        (rotate about the image centre, not the voxel origin).

    Returns
    -------
    Absolute coordinate grid, ``(*spatial_shape, ndim)``.
    """
    grid = identity_grid(spatial_shape, dtype=matrix.dtype)
    if center is None:
        center = (jnp.asarray(spatial_shape, dtype=matrix.dtype) - 1.0) / 2.0
    return apply_affine(grid, matrix, center=center)
