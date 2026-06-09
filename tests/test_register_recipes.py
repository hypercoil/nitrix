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
    rigid_exp,
    spatial_transform,
)
from nitrix.metrics import ncc  # noqa: E402
from nitrix.register import (  # noqa: E402
    LNCC,
    RegistrationSpec,
    affine_register,
    rigid_register,
)


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


def test_identity_registration_is_near_zero():
    fixed = _blobs_2d(48)
    res = rigid_register(
        fixed, fixed, spec=RegistrationSpec(levels=2, iterations=15)
    )
    assert np.allclose(np.asarray(res.params), 0.0, atol=1e-3)
    assert float(ncc(res.warped, fixed)) > 0.999


def test_register_shape_validation():
    a = _blobs_2d(32)
    b = _blobs_2d(48)
    with pytest.raises(ValueError):
        rigid_register(a, b)
    with pytest.raises(ValueError):
        rigid_register(jnp.zeros((4, 4, 4, 4)), jnp.zeros((4, 4, 4, 4)))


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
