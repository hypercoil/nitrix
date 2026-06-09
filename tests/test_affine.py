# -*- coding: utf-8 -*-
"""Tests for ``nitrix.geometry.affine`` -- geometric-parameter affine
algebra (Euler/scale/shear compose-decompose) and least-squares fit.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update('jax_enable_x64', True)

from nitrix.geometry import (
    affine_matrix_to_params,
    angles_to_rotation_matrix,
    compose_affine,
    fit_affine,
    invert_affine,
    make_square_affine,
    params_to_affine_matrix,
    rotation_matrix_to_angles,
)


def _apply(mat, pts):
    """mat (3,4) applied to points (M,3) -> (M,3)."""
    return pts @ mat[:, :3].T + mat[:, 3]


# ---------------------------------------------------------------------------
# rotation <-> Euler angles
# ---------------------------------------------------------------------------


def test_angles_to_rotation_is_orthonormal_det_one():
    rng = np.random.default_rng(0)
    ang = jnp.asarray(rng.uniform(-40, 40, size=(5, 3)))
    R = angles_to_rotation_matrix(ang)
    eye = jnp.einsum('...ij,...kj->...ik', R, R)
    np.testing.assert_allclose(
        np.asarray(eye), np.broadcast_to(np.eye(3), (5, 3, 3)), atol=1e-10
    )
    dets = jnp.linalg.det(np.asarray(R))  # host (numpy) det, fine
    np.testing.assert_allclose(np.asarray(dets), 1.0, atol=1e-10)


def test_rotation_angles_roundtrip():
    rng = np.random.default_rng(1)
    ang = jnp.asarray(rng.uniform(-30, 30, size=(4, 3)))
    R = angles_to_rotation_matrix(ang)
    ang_rec = rotation_matrix_to_angles(R)
    np.testing.assert_allclose(np.asarray(ang_rec), np.asarray(ang), atol=1e-6)


# ---------------------------------------------------------------------------
# params <-> matrix
# ---------------------------------------------------------------------------


def test_params_roundtrip_matrix_to_params():
    rng = np.random.default_rng(2)
    shift = rng.uniform(-5, 5, size=3)
    rot = rng.uniform(-25, 25, size=3)
    scale = rng.uniform(0.7, 1.3, size=3)
    shear = rng.uniform(-0.1, 0.1, size=3)
    par = jnp.asarray(np.concatenate([shift, rot, scale, shear]))
    mat = params_to_affine_matrix(par)
    par_rec = affine_matrix_to_params(mat)
    np.testing.assert_allclose(np.asarray(par_rec), np.asarray(par), atol=1e-6)


def test_params_roundtrip_params_to_matrix():
    rng = np.random.default_rng(3)
    par = jnp.asarray(
        np.concatenate(
            [
                rng.uniform(-5, 5, size=3),
                rng.uniform(-25, 25, size=3),
                rng.uniform(0.7, 1.3, size=3),
                rng.uniform(-0.1, 0.1, size=3),
            ]
        )
    )
    mat = params_to_affine_matrix(par)
    mat2 = params_to_affine_matrix(affine_matrix_to_params(mat))
    np.testing.assert_allclose(np.asarray(mat2), np.asarray(mat), atol=1e-6)


def test_params_six_vector_is_rigid():
    # translation (3) + rotation (3); scale defaults to 1, shear to 0.
    par = jnp.asarray([1.0, -2.0, 3.0, 10.0, -5.0, 20.0])
    mat = params_to_affine_matrix(par)
    R = mat[:3, :3]
    np.testing.assert_allclose(np.asarray(R @ R.T), np.eye(3), atol=1e-10)
    np.testing.assert_allclose(np.asarray(mat[:3, 3]), [1.0, -2.0, 3.0])


def test_params_shift_scale_zero_is_identity():
    par = jnp.zeros(12)
    mat = params_to_affine_matrix(par, shift_scale=True)
    expected = jnp.concatenate([jnp.eye(3), jnp.zeros((3, 1))], axis=-1)
    np.testing.assert_allclose(np.asarray(mat), np.asarray(expected), atol=1e-12)


def test_params_to_affine_matrix_differentiable():
    par = jnp.asarray(np.random.default_rng(4).standard_normal(12) * 0.1)
    g = jax.grad(lambda p: jnp.sum(params_to_affine_matrix(p) ** 2))(par)
    assert g.shape == (12,)
    assert bool(jnp.all(jnp.isfinite(g)))


def test_params_batched():
    rng = np.random.default_rng(5)
    par = jnp.asarray(rng.standard_normal((4, 6)) * 0.1)
    mat = params_to_affine_matrix(par)
    assert mat.shape == (4, 3, 4)


# ---------------------------------------------------------------------------
# shape helpers / invert / compose
# ---------------------------------------------------------------------------


def test_make_square_affine_appends_homogeneous_row():
    mat = jnp.arange(12.0).reshape(3, 4)
    sq = make_square_affine(mat)
    assert sq.shape == (4, 4)
    np.testing.assert_allclose(np.asarray(sq[3]), [0.0, 0.0, 0.0, 1.0])
    # Idempotent on square input.
    np.testing.assert_array_equal(np.asarray(make_square_affine(sq)), np.asarray(sq))


def test_invert_affine_roundtrip():
    rng = np.random.default_rng(6)
    par = jnp.asarray(
        np.concatenate(
            [
                rng.uniform(-3, 3, size=3),
                rng.uniform(-20, 20, size=3),
                rng.uniform(0.8, 1.2, size=3),
                rng.uniform(-0.05, 0.05, size=3),
            ]
        )
    )
    mat = params_to_affine_matrix(par)
    inv = invert_affine(mat)
    composed = compose_affine([mat, inv])
    expected = jnp.concatenate([jnp.eye(3), jnp.zeros((3, 1))], axis=-1)
    np.testing.assert_allclose(np.asarray(composed), np.asarray(expected), atol=1e-8)


def test_compose_affine_applies_right_to_left():
    rng = np.random.default_rng(7)
    a = params_to_affine_matrix(jnp.asarray(rng.standard_normal(6) * 0.1))
    b = params_to_affine_matrix(jnp.asarray(rng.standard_normal(6) * 0.1))
    pts = jnp.asarray(rng.standard_normal((10, 3)))
    composed = compose_affine([a, b])
    # compose((A, B)) applies B first, then A.
    direct = _apply(a, _apply(b, pts))
    via = _apply(composed, pts)
    np.testing.assert_allclose(np.asarray(via), np.asarray(direct), atol=1e-9)


# ---------------------------------------------------------------------------
# fit_affine
# ---------------------------------------------------------------------------


def test_fit_affine_recovers_known_transform():
    rng = np.random.default_rng(8)
    par = jnp.asarray(
        np.concatenate(
            [
                rng.uniform(-4, 4, size=3),
                rng.uniform(-20, 20, size=3),
                rng.uniform(0.8, 1.2, size=3),
                rng.uniform(-0.05, 0.05, size=3),
            ]
        )
    )
    A = params_to_affine_matrix(par)
    target = jnp.asarray(rng.standard_normal((30, 3)))
    source = _apply(A, target)
    fit = fit_affine(source, target)
    np.testing.assert_allclose(np.asarray(fit), np.asarray(A), atol=1e-7)


def test_fit_affine_uniform_weights_match_unweighted():
    rng = np.random.default_rng(9)
    target = jnp.asarray(rng.standard_normal((20, 3)))
    A = params_to_affine_matrix(jnp.asarray(rng.standard_normal(12) * 0.1))
    source = _apply(A, target)
    w = jnp.ones(20)
    np.testing.assert_allclose(
        np.asarray(fit_affine(source, target, weights=w)),
        np.asarray(fit_affine(source, target)),
        atol=1e-8,
    )


# ---------------------------------------------------------------------------
# 2-D affine chart (ndim=2)
# ---------------------------------------------------------------------------


def _apply2d(mat, pts):
    """mat (2,3) applied to points (M,2) -> (M,2)."""
    return pts @ mat[:, :2].T + mat[:, 2]


def test_angles_2d_roundtrip():
    rng = np.random.default_rng(20)
    ang = jnp.asarray(rng.uniform(-170, 170, size=(5, 1)))
    R = angles_to_rotation_matrix(ang)
    assert R.shape == (5, 2, 2)
    # Orthonormal, det +1.
    eye = jnp.einsum('...ij,...kj->...ik', R, R)
    np.testing.assert_allclose(
        np.asarray(eye), np.broadcast_to(np.eye(2), (5, 2, 2)), atol=1e-10
    )
    np.testing.assert_allclose(
        np.asarray(rotation_matrix_to_angles(R)), np.asarray(ang), atol=1e-6
    )


def test_params_2d_roundtrip_matrix_to_params():
    rng = np.random.default_rng(21)
    par = jnp.asarray(
        np.concatenate(
            [
                rng.uniform(-4, 4, size=2),  # translation
                rng.uniform(-60, 60, size=1),  # rotation
                rng.uniform(0.7, 1.3, size=2),  # scale
                rng.uniform(-0.1, 0.1, size=1),  # shear
            ]
        )
    )
    mat = params_to_affine_matrix(par, ndim=2)
    assert mat.shape == (2, 3)
    np.testing.assert_allclose(
        np.asarray(affine_matrix_to_params(mat)), np.asarray(par), atol=1e-6
    )


def test_params_2d_roundtrip_params_to_matrix():
    rng = np.random.default_rng(22)
    par = jnp.asarray(
        np.concatenate(
            [
                rng.uniform(-4, 4, size=2),
                rng.uniform(-60, 60, size=1),
                rng.uniform(0.7, 1.3, size=2),
                rng.uniform(-0.1, 0.1, size=1),
            ]
        )
    )
    mat = params_to_affine_matrix(par, ndim=2)
    mat2 = params_to_affine_matrix(affine_matrix_to_params(mat), ndim=2)
    np.testing.assert_allclose(np.asarray(mat2), np.asarray(mat), atol=1e-6)


def test_params_2d_rigid_is_orthonormal():
    par = jnp.asarray([1.0, -2.0, 30.0])  # trans(2) + rotation(1); scale->1
    mat = params_to_affine_matrix(par, ndim=2)
    R = mat[:2, :2]
    np.testing.assert_allclose(np.asarray(R @ R.T), np.eye(2), atol=1e-10)
    np.testing.assert_allclose(np.asarray(mat[:2, 2]), [1.0, -2.0])


def test_fit_affine_2d_recovers_known_transform():
    rng = np.random.default_rng(23)
    par = jnp.asarray(
        np.concatenate(
            [
                rng.uniform(-3, 3, size=2),
                rng.uniform(-45, 45, size=1),
                rng.uniform(0.8, 1.2, size=2),
                rng.uniform(-0.05, 0.05, size=1),
            ]
        )
    )
    a = params_to_affine_matrix(par, ndim=2)
    target = jnp.asarray(rng.standard_normal((20, 2)))
    source = _apply2d(a, target)
    np.testing.assert_allclose(
        np.asarray(fit_affine(source, target)), np.asarray(a), atol=1e-7
    )
