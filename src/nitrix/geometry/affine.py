# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Geometric-parameter affine algebra (batched, differentiable).

The complement to the Lie-group chart in ``transform`` (``rigid_exp`` /
``affine_exp``): here an affine is parametrised by its **geometric
factors** -- translation, Euler rotation, anisotropic scale, and shear --
and composed as ``T @ R @ S @ E``.  This is the parametrisation used when
those factors are individually meaningful (e.g. reading off the scales of
a fitted transform, or constraining a fit to be rigid by dropping scale
and shear), as opposed to the exponential chart, which is the natural one
for unconstrained gradient-based optimisation.

Also here: ``fit_affine``, the closed-form (weighted) least-squares
affine between two corresponding point sets.

Conventions:

- Matrices are ``(..., N, N+1)`` (the implied last row is
  ``(*[0]*N, 1)``) unless made square; everything is batched over leading
  dimensions and differentiable.  A transform maps coordinates as
  ``y = mat[..., :, :-1] @ x + mat[..., :, -1]``.
- Rotations are right-handed **intrinsic** ``R = X @ Y @ Z`` (rotate about
  x, then y, then z), angles in degrees by default.
- 3-D only for the rotation / parameter decomposition (the Euler
  convention is dimension-specific); the shape helpers and ``fit_affine``
  are N-D.

GPU note: the factorisations (``inv`` / ``cholesky`` / ``det``) are routed
through ``linalg._solver.safe_*`` (or computed analytically for the 3x3 /
diagonal cases) so they work on the cuSolver-affected GPU stacks, where
the dense-solver handle pool is broken.
"""

from __future__ import annotations

from typing import Optional, Sequence

import jax.numpy as jnp
from jaxtyping import Array, Float

from ..linalg._solver import safe_cho_solve, safe_cholesky, safe_inv

__all__ = [
    'make_square_affine',
    'invert_affine',
    'compose_affine',
    'fit_affine',
    'angles_to_rotation_matrix',
    'rotation_matrix_to_angles',
    'params_to_affine_matrix',
    'affine_matrix_to_params',
]


def _diag(v: Float[Array, '... n']) -> Float[Array, '... n n']:
    """Batched ``diag``: ``(..., N)`` -> ``(..., N, N)``."""
    return v[..., None] * jnp.eye(v.shape[-1], dtype=v.dtype)


def _det3x3(m: Float[Array, '... 3 3']) -> Float[Array, '...']:
    """Analytic batched determinant of a ``(..., 3, 3)`` matrix.

    Closed form (cofactor expansion) -- avoids the LU factorisation
    ``jnp.linalg.det`` would use, which is unavailable on the
    cuSolver-affected GPU stacks.
    """
    return (
        m[..., 0, 0]
        * (m[..., 1, 1] * m[..., 2, 2] - m[..., 1, 2] * m[..., 2, 1])
        - m[..., 0, 1]
        * (m[..., 1, 0] * m[..., 2, 2] - m[..., 1, 2] * m[..., 2, 0])
        + m[..., 0, 2]
        * (m[..., 1, 0] * m[..., 2, 1] - m[..., 1, 1] * m[..., 2, 0])
    )


def _clip11(x: Float[Array, '...']) -> Float[Array, '...']:
    return jnp.clip(x, -1.0, 1.0)


def _divide_no_nan(
    a: Float[Array, '...'], b: Float[Array, '...']
) -> Float[Array, '...']:
    return jnp.where(b == 0, jnp.zeros_like(a), a / jnp.where(b == 0, 1.0, b))


# ---------------------------------------------------------------------
# Shape helpers


def make_square_affine(
    mat: Float[Array, '... n np1'],
) -> Float[Array, '... np1 np1']:
    """``(..., N, N+1)`` -> ``(..., N+1, N+1)`` by appending ``(*[0]*N, 1)``.

    A no-op if ``mat`` is already square.
    """
    *batch, rows, cols = mat.shape
    if rows == cols:
        return mat
    row = jnp.zeros((*batch, 1, cols), dtype=mat.dtype)
    row = row.at[..., 0, -1].set(1.0)
    return jnp.concatenate([mat, row], axis=-2)


def invert_affine(
    mat: Float[Array, '... n np1'],
) -> Float[Array, '... n np1']:
    """Multiplicative inverse of an affine ``(..., N, N+1)`` (or square)."""
    rows = mat.shape[-2]
    return safe_inv(make_square_affine(mat))[..., :rows, :]


def compose_affine(
    mats: Sequence[Float[Array, '... n np1']],
) -> Float[Array, '... n np1']:
    """Compose affines left-to-right: ``compose((A, B, C))`` applies C first.

    Returns ``A @ B @ C`` (as square matrices), dropping the implied last
    row back to ``(..., N, N+1)``.
    """
    if not mats:
        raise ValueError('compose_affine: empty sequence')
    rows = mats[0].shape[-2]
    out = make_square_affine(mats[0])
    for m in mats[1:]:
        out = out @ make_square_affine(m)
    return out[..., :rows, :]


# ---------------------------------------------------------------------
# Least-squares affine fit between corresponding point sets


def fit_affine(
    source: Float[Array, '... m n'],
    target: Float[Array, '... m n'],
    weights: Optional[Float[Array, '... m']] = None,
) -> Float[Array, '... n np1']:
    """Closed-form (weighted) least-squares affine between point sets.

    Fits ``mat`` ``(..., N, N+1)`` so that ``mat`` maps ``target`` onto
    ``source``: ``source ≈ mat[..., :, :-1] @ target + mat[..., :, -1]``,
    minimising the (weighted) squared residual over the ``M``
    corresponding points of ``source`` / ``target`` ``(..., M, N)``.

    Solves the normal equations ``(Xᵀ W X) β = Xᵀ W y`` (with ``X`` the
    homogeneous ``target`` and ``y = source``) via an SPD Cholesky solve;
    ``weights`` is ``(..., M)`` or ``(..., M, 1)``.
    """
    ones = jnp.ones((*target.shape[:-1], 1), dtype=target.dtype)
    x = jnp.concatenate([target, ones], axis=-1)  # (..., M, N+1)
    x_transp = jnp.swapaxes(x, -1, -2)  # (..., N+1, M)
    y = source  # (..., M, N)
    if weights is not None:
        if weights.ndim == x.ndim:
            weights = weights[..., 0]
        x_transp = x_transp * weights[..., None, :]
    gram = x_transp @ x  # (..., N+1, N+1), SPD
    rhs = x_transp @ y  # (..., N+1, N)
    beta = safe_cho_solve(gram, rhs)  # (..., N+1, N)
    return jnp.swapaxes(beta, -1, -2)  # (..., N, N+1)


# ---------------------------------------------------------------------
# Rotation <-> Euler angles (3D, right-handed intrinsic R = X @ Y @ Z)


def angles_to_rotation_matrix(
    ang: Float[Array, '... 3'], *, deg: bool = True
) -> Float[Array, '... 3 3']:
    """3D Euler angles ``(..., 3)`` -> rotation matrix ``(..., 3, 3)``.

    Right-handed intrinsic rotations ``R = X @ Y @ Z`` (rotate about x,
    then y, then z).  Angles in degrees by default (``deg=False`` for
    radians).
    """
    ang = jnp.asarray(ang)
    if ang.shape[-1] != 3:
        raise ValueError(f'expected 3 angles, got shape {ang.shape}')
    if deg:
        ang = ang * (jnp.pi / 180.0)
    a1, a2, a3 = ang[..., 0], ang[..., 1], ang[..., 2]
    c1, s1 = jnp.cos(a1), jnp.sin(a1)
    c2, s2 = jnp.cos(a2), jnp.sin(a2)
    c3, s3 = jnp.cos(a3), jnp.sin(a3)
    r00 = c2 * c3
    r01 = -c2 * s3
    r02 = s2
    r10 = s1 * s2 * c3 + c1 * s3
    r11 = -s1 * s2 * s3 + c1 * c3
    r12 = -s1 * c2
    r20 = -c1 * s2 * c3 + s1 * s3
    r21 = c1 * s2 * s3 + s1 * c3
    r22 = c1 * c2
    rows = [
        jnp.stack([r00, r01, r02], axis=-1),
        jnp.stack([r10, r11, r12], axis=-1),
        jnp.stack([r20, r21, r22], axis=-1),
    ]
    return jnp.stack(rows, axis=-2)


def rotation_matrix_to_angles(
    mat: Float[Array, '... 3 3'], *, deg: bool = True
) -> Float[Array, '... 3']:
    """3D rotation matrix ``(..., 3, 3[+1])`` -> Euler angles ``(..., 3)``.

    Inverse of :func:`angles_to_rotation_matrix` (``R = X @ Y @ Z``), with
    the gimbal-lock convention ``ang[0] = 0`` when ``|ang[1]| = 90 deg``.
    """
    ang2 = jnp.arcsin(_clip11(mat[..., 0, 2]))

    # Gimbal-lock branch (|ang2| == 90 deg): ang1 := 0, solve ang3.
    ang1_a = jnp.zeros_like(ang2)
    ang3_a = jnp.arctan2(_clip11(mat[..., 1, 0]), _clip11(mat[..., 1, 1]))

    c2 = jnp.cos(ang2)
    ang1_b = jnp.arctan2(
        _clip11(_divide_no_nan(-mat[..., 1, 2], c2)),
        _clip11(_divide_no_nan(mat[..., 2, 2], c2)),
    )
    ang3_b = jnp.arctan2(
        _clip11(_divide_no_nan(-mat[..., 0, 1], c2)),
        _clip11(_divide_no_nan(mat[..., 0, 0], c2)),
    )

    is_lock = jnp.abs(jnp.abs(ang2) - 0.5 * jnp.pi) < 1e-6
    ang1 = jnp.where(is_lock, ang1_a, ang1_b)
    ang3 = jnp.where(is_lock, ang3_a, ang3_b)
    ang = jnp.stack([ang1, ang2, ang3], axis=-1)
    if deg:
        ang = ang * (180.0 / jnp.pi)
    return ang


# ---------------------------------------------------------------------
# Affine <-> geometric parameters (translation, rotation, scale, shear)


def params_to_affine_matrix(
    par: Float[Array, '... p'],
    *,
    deg: bool = True,
    shift_scale: bool = False,
) -> Float[Array, '... 3 4']:
    """3D affine params ``(..., M<=12)`` -> matrix ``(..., 3, 4)``.

    Param order along the last axis: translation (3), rotation (3), scale
    (3), shear (3); missing trailing params default to identity (scale ->
    1, others -> 0), so a 6-vector gives a rigid transform.  Builds
    ``T @ R @ S @ E``.  With ``shift_scale=True``, 1 is added to the scale
    params (so a zero vector is the identity -- useful when the params are
    a perturbation predicted by a network).
    """
    par = jnp.asarray(par)
    ndims = 3
    num_par = ndims * (ndims + 1)  # 12
    if par.shape[-1] > num_par:
        raise ValueError(f'too many params: {par.shape[-1]} > {num_par}')

    # Pad to full 12, with scale defaulting to 1 (or 0 if shift_scale).
    cuts = (3, 6, 9, 12)
    for i in (1, 2, 3):
        need = max(cuts[i] - par.shape[-1], 0)
        if need:
            default = 1.0 if (i == 2 and not shift_scale) else 0.0
            pad = [(0, 0)] * (par.ndim - 1) + [(0, need)]
            par = jnp.pad(par, pad, constant_values=default)
    shift = par[..., 0:3]
    rot = par[..., 3:6]
    scale = par[..., 6:9]
    shear = par[..., 9:12]

    one = jnp.ones(shift[..., :1].shape, dtype=par.dtype)
    zero = jnp.zeros_like(one)
    s0 = shear[..., 0:1]
    s1 = shear[..., 1:2]
    s2 = shear[..., 2:3]
    mat_shear = jnp.stack(
        [
            jnp.concatenate([one, s0, s1], axis=-1),
            jnp.concatenate([zero, one, s2], axis=-1),
            jnp.concatenate([zero, zero, one], axis=-1),
        ],
        axis=-2,
    )

    mat_scale = _diag(scale + 1.0 if shift_scale else scale)
    mat_rot = angles_to_rotation_matrix(rot, deg=deg)
    out = mat_rot @ (mat_scale @ mat_shear)  # (..., 3, 3)
    return jnp.concatenate([out, shift[..., None]], axis=-1)  # (..., 3, 4)


def affine_matrix_to_params(
    mat: Float[Array, '... 3 4'], *, deg: bool = True
) -> Float[Array, '... 12']:
    """3D affine ``(..., 3, 4[+1])`` -> params ``(..., 12)``.

    Inverse of :func:`params_to_affine_matrix`: returns translation (3),
    rotation (3), scale (3), shear (3).  The linear block is factored as
    ``R @ S @ E`` via the Cholesky of ``mᵀ m`` (an RQ-style
    decomposition); the scale sign is fixed up with the determinant so a
    negative-determinant (reflection) matrix is representable.
    """
    shift = mat[..., :3, -1]
    m = mat[..., :3, :3]
    lower = safe_cholesky(jnp.swapaxes(m, -1, -2) @ m)
    scale = jnp.diagonal(lower, axis1=-2, axis2=-1)
    sign = jnp.sign(_det3x3(m))
    scale = jnp.concatenate(
        [(scale[..., 0] * sign)[..., None], scale[..., 1:]], axis=-1
    )

    # upper = inv(diag(scale)) @ lowerᵀ; inv of a diagonal is elementwise.
    upper = (1.0 / scale)[..., :, None] * jnp.swapaxes(lower, -1, -2)
    flat = upper.reshape((*upper.shape[:-2], 9))
    shear = flat[..., jnp.asarray([1, 2, 5])]  # (0,1), (0,2), (1,2)

    # Rotation, after stripping scale + shear.
    zero = jnp.zeros((*scale.shape[:-1], 6), dtype=mat.dtype)
    par = jnp.concatenate([zero, scale, shear], axis=-1)
    strip3 = params_to_affine_matrix(par, deg=deg)[..., :3]  # (..., 3, 3)
    rot_mat = m @ safe_inv(strip3)
    rot = rotation_matrix_to_angles(rot_mat, deg=deg)

    return jnp.concatenate([shift, rot, scale, shear], axis=-1)
