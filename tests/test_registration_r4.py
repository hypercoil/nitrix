# -*- coding: utf-8 -*-
"""R4 gate: physical-space (anisotropic-voxel) registration.

The correctness oracle for ``WorldSpace``: plant a *known physical* rigid
transform on an **anisotropic** voxel grid and recover it.  Under
anisotropic spacing a rotation parametrised in index space is not
physically rigid (it shears), so ``IndexSpace`` cannot represent the true
map and recovers it poorly; ``WorldSpace`` optimises the world transform
and recovers it.  The planted ground truth is built here from raw
homogeneous matrix algebra -- independent of the ``WorldSpace``
implementation it validates.
"""

from __future__ import annotations

import jax

jax.config.update('jax_enable_x64', True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from nitrix.geometry import (  # noqa: E402
    affine_grid,
    rigid_exp,
    spatial_transform,
)
from nitrix.metrics import ncc  # noqa: E402
from nitrix.register import (  # noqa: E402
    IndexSpace,
    RegistrationSpec,
    WorldSpace,
    rigid_register,
)


def _blobs_2d(n=64):
    yy, xx = np.mgrid[0:n, 0:n].astype('float64')

    def blob(cy, cx, s, a):
        return a * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s * s))

    img = (
        blob(0.30 * n, 0.38 * n, 0.11 * n, 1.0)
        + blob(0.62 * n, 0.69 * n, 0.14 * n, 0.7)
        + blob(0.75 * n, 0.28 * n, 0.09 * n, 0.6)
        + blob(0.47 * n, 0.81 * n, 0.12 * n, 0.5)
    )
    return jnp.asarray(img)


def _blobs_3d(n=24):
    zz, yy, xx = np.mgrid[0:n, 0:n, 0:n].astype('float64')

    def blob(c, s, a):
        return a * np.exp(
            -((xx - c[2]) ** 2 + (yy - c[1]) ** 2 + (zz - c[0]) ** 2)
            / (2 * s * s)
        )

    img = (
        blob((0.4 * n, 0.4 * n, 0.5 * n), 0.16 * n, 1.0)
        + blob((0.6 * n, 0.65 * n, 0.4 * n), 0.2 * n, 0.7)
        + blob((0.5 * n, 0.3 * n, 0.65 * n), 0.13 * n, 0.6)
    )
    return jnp.asarray(img)


def _diag_affine(spacing):
    """Voxel->world affine ``diag([*spacing, 1])`` (anisotropic spacing)."""
    ndim = len(spacing)
    a = np.eye(ndim + 1)
    a[np.arange(ndim), np.arange(ndim)] = spacing
    return a


def _trans(t):
    ndim = len(t)
    m = np.eye(ndim + 1)
    m[:ndim, ndim] = t
    return m


def _plant_world(fixed, true_params, ndim, affine):
    """Make ``moving`` from a known *world* rigid transform.

    ``moving[i] = fixed`` sampled at the index map
    ``A⁻¹ · T_c · A`` (with ``T_c`` the world transform centred at the
    fixed image's world centre).  Built from raw matrices -- an oracle
    independent of ``WorldSpace``.  Returns ``(moving, T_c)``.
    """
    shape = fixed.shape
    a = affine
    a_inv = np.linalg.inv(a)
    c_vox = (np.asarray(shape, dtype='float64') - 1.0) / 2.0
    c_world = (a @ np.concatenate([c_vox, [1.0]]))[:ndim]
    t_world = np.asarray(rigid_exp(jnp.asarray(true_params), ndim=ndim))
    t_c = _trans(c_world) @ t_world @ _trans(-c_world)
    m_index = a_inv @ t_c @ a
    grid = affine_grid(
        jnp.asarray(m_index), shape, center=jnp.zeros(ndim, dtype=fixed.dtype)
    )
    moving = spatial_transform(fixed[..., None], grid, mode='constant')[..., 0]
    return moving, t_c


def test_anisotropic_rigid_world_beats_index_2d():
    # Strong anisotropy (axis-0 voxels 4x axis-1) + a moderate rotation:
    # a physically rigid map that index-space rigid cannot represent.
    fixed = _blobs_2d(64)
    affine = _diag_affine([4.0, 1.0])
    true = [0.15, 2.0, -1.5]  # world: rot 0.15 rad, translation (mm)
    moving, t_c = _plant_world(fixed, true, 2, affine)

    spec = RegistrationSpec(levels=3, iterations=40)
    world = WorldSpace(
        fixed_affine=jnp.asarray(affine), moving_affine=jnp.asarray(affine)
    )
    res_w = rigid_register(moving, fixed, spec=spec, space=world)
    res_i = rigid_register(moving, fixed, spec=spec, space=IndexSpace())

    ncc_w = float(ncc(res_w.warped, fixed))
    ncc_i = float(ncc(res_i.warped, fixed))
    # World space recovers the physical transform; index space cannot.
    assert ncc_w > 0.95
    assert ncc_w > ncc_i + 0.02
    # Recovered world rotation is the inverse of the planted one.
    assert np.isclose(float(res_w.params[0]), -true[0], atol=0.02)
    # res.matrix is the world->world transform inverting the planted T_c:
    # the rotation block (the anisotropy-sensitive quantity) is recovered
    # tightly; the mm-scale translation to sub-voxel (axis-0 voxel = 4 mm).
    inv_tc = np.linalg.inv(t_c)
    assert np.allclose(
        np.asarray(res_w.matrix)[:2, :2], inv_tc[:2, :2], atol=0.01
    )
    assert np.allclose(
        np.asarray(res_w.matrix)[:2, 2], inv_tc[:2, 2], atol=1.0
    )


def test_anisotropic_rigid_world_recovery_3d():
    fixed = _blobs_3d(24)
    affine = _diag_affine([1.0, 1.0, 3.0])  # thick slices along axis-2
    true = [0.05, 0.08, 0.04, 1.0, -0.8, 1.5]
    moving, _ = _plant_world(fixed, true, 3, affine)

    spec = RegistrationSpec(levels=2, iterations=30)
    world = WorldSpace(
        fixed_affine=jnp.asarray(affine), moving_affine=jnp.asarray(affine)
    )
    res = rigid_register(moving, fixed, spec=spec, space=world)
    assert float(ncc(res.warped, fixed)) > 0.95
    assert np.allclose(
        np.asarray(res.params[:3]), -np.asarray(true[:3]), atol=0.03
    )


def test_world_warped_reproducible_from_matrix():
    # The world result is self-contained: composing res.matrix with the
    # two affines reproduces ``warped`` (the WorldSpace analogue of the
    # index-space explicit-warp invariant).
    fixed = _blobs_2d(48)
    affine = _diag_affine([2.0, 1.0])
    moving, _ = _plant_world(fixed, [0.1, 1.0, -1.0], 2, affine)
    world = WorldSpace(
        fixed_affine=jnp.asarray(affine), moving_affine=jnp.asarray(affine)
    )
    res = rigid_register(
        moving,
        fixed,
        spec=RegistrationSpec(levels=2, iterations=25),
        space=world,
    )
    a = affine
    m_index = np.linalg.inv(a) @ np.asarray(res.matrix) @ a
    grid = affine_grid(
        jnp.asarray(m_index), fixed.shape, center=jnp.zeros(2, fixed.dtype)
    )
    explicit = spatial_transform(moving[..., None], grid, mode='constant')[
        ..., 0
    ]
    assert np.allclose(np.asarray(res.warped), np.asarray(explicit), atol=1e-6)


def test_world_identity_affine_recovers_like_index():
    # WorldSpace with identity affines is a valid (if not the leanest) path
    # on isotropic data: it still recovers a planted transform.
    fixed = _blobs_2d(64)
    affine = _diag_affine([1.0, 1.0])
    moving, _ = _plant_world(fixed, [0.12, 3.0, -2.0], 2, affine)
    res = rigid_register(
        moving,
        fixed,
        spec=RegistrationSpec(levels=3, iterations=30),
        space=WorldSpace(),
    )
    assert float(ncc(res.warped, fixed)) > 0.97
