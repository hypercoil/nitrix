# -*- coding: utf-8 -*-
"""Synthetic-recovery tests for the registration recipes (R1d).

The unambiguous correctness oracle: warp a structured image by a *known*
transform to make ``moving``, register it back to the original
``fixed``, and assert the recovered warp reproduces ``fixed`` (high
global NCC, large cost reduction) with the expected rotation.  Covers
2-D / 3-D rigid (SSD), 2-D affine (SSD), and the LNCC metric path;
identity and input-validation cases.
"""

from __future__ import annotations

import jax

jax.config.update('jax_enable_x64', True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from nitrix.geometry import (  # noqa: E402
    affine_exp,
    affine_grid,
    identity_grid,  # noqa: E402
    rigid_exp,
    spatial_transform,
)
from nitrix.metrics import mutual_information, ncc  # noqa: E402
from nitrix.register import (  # noqa: E402
    LNCC,
    MI,
    SSD,
    CorrelationRatio,
    RegistrationSpec,
    WorldSpace,
    affine_register,
    rigid_register,
)
from nitrix.register._metric import pin_metric_ranges  # noqa: E402
from nitrix.register.recipes import _moment_init_matrix  # noqa: E402


def _blobs_2d(n=64):
    yy, xx = np.mgrid[0:n, 0:n].astype('float64')

    def blob(cy, cx, s, a):
        return a * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s * s))

    img = (
        blob(0.3 * n, 0.38 * n, 0.11 * n, 1.0)
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


def _warp_known(fixed, matrix):
    shape = fixed.shape
    center = (jnp.asarray(shape, dtype=fixed.dtype) - 1.0) / 2.0
    grid = affine_grid(matrix, shape, center=center)
    return spatial_transform(fixed[..., None], grid, mode='constant')[..., 0]


def test_rigid_2d_ssd_recovery():
    fixed = _blobs_2d(64)
    true = jnp.asarray([0.13, 4.0, 3.0])  # rot 0.13 rad, trans (4, 3)
    moving = _warp_known(fixed, rigid_exp(true, ndim=2))
    res = rigid_register(
        moving, fixed, spec=RegistrationSpec(levels=3, iterations=30)
    )
    assert float(ncc(res.warped, fixed)) > 0.98
    init = float(ncc(moving, fixed))
    assert float(ncc(res.warped, fixed)) > init + 0.05
    # recovered rotation is the inverse rotation (centre-independent).
    assert np.isclose(float(res.params[0]), -0.13, atol=0.02)
    assert res.matrix.shape == (3, 3)


def test_rigid_3d_ssd_recovery():
    fixed = _blobs_3d(24)
    true = jnp.asarray([0.06, -0.05, 0.08, 1.5, -1.0, 1.2])
    moving = _warp_known(fixed, rigid_exp(true, ndim=3))
    res = rigid_register(
        moving, fixed, spec=RegistrationSpec(levels=2, iterations=25)
    )
    assert float(ncc(res.warped, fixed)) > 0.97
    assert res.matrix.shape == (4, 4)
    # the recovered axis-angle is ~ the negated truth.
    assert np.allclose(
        np.asarray(res.params[:3]), -np.asarray(true[:3]), atol=0.03
    )


def test_affine_2d_ssd_recovery():
    fixed = _blobs_2d(64)
    # small affine: anisotropic scale + shear + translation.
    gen = np.array([[0.08, 0.05], [-0.04, -0.06]])
    true = jnp.asarray(np.concatenate([gen.reshape(-1), [3.0, -2.0]]))
    moving = _warp_known(fixed, affine_exp(true, ndim=2))
    res = affine_register(
        moving, fixed, spec=RegistrationSpec(levels=3, iterations=40)
    )
    assert float(ncc(res.warped, fixed)) > 0.98
    assert res.params.shape == (6,)


def test_rigid_2d_lncc_recovery():
    fixed = _blobs_2d(64)
    true = jnp.asarray([0.1, 3.0, -2.0])
    moving = _warp_known(fixed, rigid_exp(true, ndim=2))
    res = rigid_register(
        moving,
        fixed,
        spec=RegistrationSpec(levels=3, iterations=40, metric=LNCC()),
    )
    assert float(ncc(res.warped, fixed)) > 0.97


def test_pin_metric_ranges_resolves_histogram_metrics():
    # 4d (A6, matrix half): the driver pins a histogram metric's ranges once
    # from the full-res images; a least-squares / already-pinned metric is a
    # no-op (same object).
    moving = _blobs_2d(48)
    fixed = _blobs_2d(48) * 1.4 + 0.2  # different support
    pinned = pin_metric_ranges(MI(bins=16), moving, fixed)
    assert isinstance(pinned, MI)
    assert pinned.range_moving == (float(moving.min()), float(moving.max()))
    assert pinned.range_fixed == (float(fixed.min()), float(fixed.max()))
    cr = pin_metric_ranges(CorrelationRatio(bins=16), moving, fixed)
    assert isinstance(cr, CorrelationRatio)
    assert cr.range_fixed == (float(fixed.min()), float(fixed.max()))
    ssd = SSD()
    assert pin_metric_ranges(ssd, moving, fixed) is ssd
    already = MI(bins=16, range_moving=(0.0, 1.0), range_fixed=(0.0, 2.0))
    assert pin_metric_ranges(already, moving, fixed) is already


def _small_cluster(n=96):
    yy, xx = np.mgrid[0:n, 0:n].astype('float64')
    img = np.zeros((n, n))
    for cy, cx, s, a in [
        (0.46, 0.48, 0.07, 1.0),
        (0.54, 0.55, 0.08, 0.7),
        (0.50, 0.40, 0.06, 0.6),
    ]:
        img += a * np.exp(
            -((xx - cx * n) ** 2 + (yy - cy * n) ** 2) / (2 * (s * n) ** 2)
        )
    return jnp.asarray(img)


def test_moment_init_aligns_centroids():
    # 3c (A8): the moment-init matrix maps the fixed centroid onto the moving
    # centroid (the affine_grid centring convention) -- the warped moving lands
    # its intensity centroid on the fixed centroid.
    fixed = _small_cluster(96)
    true = affine_exp(jnp.asarray([0.0, 0.0, 0.0, 0.0, 30.0, -22.0]), ndim=2)
    moving = _warp_known(fixed, true)
    matrix = _moment_init_matrix(moving, fixed, ndim=2, scale=True)
    center = (jnp.asarray((96, 96), dtype=fixed.dtype) - 1.0) / 2.0
    from nitrix.geometry import affine_grid, spatial_transform

    warped = spatial_transform(
        moving[..., None], affine_grid(matrix, (96, 96), center=center)
    )[..., 0]

    def centroid(img):
        grid = identity_grid(img.shape, dtype=img.dtype)
        w = jnp.clip(img, 0.0, None)
        return jnp.sum(w[..., None] * grid, axis=(0, 1)) / (jnp.sum(w) + 1e-12)

    assert np.allclose(
        np.asarray(centroid(warped)), np.asarray(centroid(fixed)), atol=0.5
    )


def test_moment_init_recovers_where_identity_fails():
    # 3c (A8): a translation large enough that the structure falls out of overlap
    # -- the optimiser cannot find it from a zero start (ncc ~ 0), but the
    # centre-of-mass init lands inside the basin and it recovers.
    fixed = _small_cluster(96)
    true = affine_exp(jnp.asarray([0.0, 0.0, 0.0, 0.0, 38.0, -38.0]), ndim=2)
    moving = _warp_known(fixed, true)
    spec = RegistrationSpec(levels=3, iterations=50)
    identity = affine_register(moving, fixed, spec=spec, init='identity')
    moment = affine_register(moving, fixed, spec=spec, init='moment')
    assert float(ncc(identity.warped, fixed)) < 0.5
    assert float(ncc(moment.warped, fixed)) > 0.9


def test_moment_init_validation():
    fixed, moving = _small_cluster(64), _small_cluster(64)
    with pytest.raises(ValueError, match='IndexSpace'):
        affine_register(
            moving,
            fixed,
            init='moment',
            space=WorldSpace(
                fixed_affine=jnp.eye(3), moving_affine=jnp.eye(3)
            ),
        )
    with pytest.raises(ValueError, match='identity'):
        rigid_register(moving, fixed, init='nonsense')


def test_rigid_2d_mi_recovery_cross_modal():
    # The matrix MI path (forward BFGS) recovers a known rotation on a CROSS-
    # MODAL pair (intensity-remapped moving), with the histogram range pinned
    # once from the full-res images (A6) -- a stationary objective.
    fixed = _blobs_2d(64)
    true = jnp.asarray([0.12, 3.0, -2.5])
    warped = _warp_known(fixed, rigid_exp(true, ndim=2))
    moving = jnp.sqrt(warped - warped.min() + 0.05)  # "different modality"
    res = rigid_register(
        moving,
        fixed,
        spec=RegistrationSpec(levels=3, iterations=40, metric=MI(bins=32)),
    )
    mi0 = float(mutual_information(moving, fixed, bins=32))
    mi1 = float(mutual_information(res.warped, fixed, bins=32))
    assert mi1 > mi0  # the optimised objective improves
    # recovered rotation is the inverse rotation (centre-independent).
    assert np.isclose(float(res.params[0]), -0.12, atol=0.03)


def test_identity_registration_is_near_zero():
    fixed = _blobs_2d(48)
    res = rigid_register(
        fixed, fixed, spec=RegistrationSpec(levels=2, iterations=15)
    )
    assert np.allclose(np.asarray(res.params), 0.0, atol=1e-3)
    assert float(ncc(res.warped, fixed)) > 0.999


def test_register_different_shapes_allowed():
    # The warp is built on the fixed grid, so moving / fixed need not share
    # a shape (a shared voxel grid with a different field of view); the
    # result lives on the fixed grid.
    moving = _blobs_2d(48)
    fixed = _blobs_2d(40)
    res = rigid_register(
        moving, fixed, spec=RegistrationSpec(levels=2, iterations=10)
    )
    assert res.warped.shape == fixed.shape


def test_register_rank_validation():
    # Unsupported spatial rank (not 2-D / 3-D) still raises.
    with pytest.raises(ValueError):
        rigid_register(jnp.zeros((4, 4, 4, 4)), jnp.zeros((4, 4, 4, 4)))
    # Mismatched rank (2-D vs 3-D) raises.
    with pytest.raises(ValueError):
        rigid_register(_blobs_2d(16), jnp.zeros((16, 16, 16)))


def test_result_warped_matches_explicit_warp():
    fixed = _blobs_2d(48)
    moving = _warp_known(
        fixed, rigid_exp(jnp.asarray([0.08, 2.0, 1.0]), ndim=2)
    )
    res = rigid_register(
        moving, fixed, spec=RegistrationSpec(levels=2, iterations=20)
    )
    # res.warped is moving sampled by res.matrix about the image centre.
    explicit = _warp_known(moving, res.matrix)
    assert np.allclose(np.asarray(res.warped), np.asarray(explicit), atol=1e-6)
