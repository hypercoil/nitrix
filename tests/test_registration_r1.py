# -*- coding: utf-8 -*-
"""Tests for the R1 registration substrate (rigid + affine).

Coverage grows with the R1 sub-phases:

- **linalg.matrix_exp** (R1a): ``e^0 = I``; diagonal / skew-symmetric
  anchors; parity with ``scipy.linalg.expm``; ``det(e^A) = e^{tr A}``;
  batching; differentiability.
- **linalg.cg** (R1a): SPD solve parity with a direct solve; matrix-free
  (matvec) form; the ``l2`` ridge; runs on the GPU through the cuSolver
  wedge; differentiability.
"""

from __future__ import annotations

import jax

jax.config.update('jax_enable_x64', True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import scipy.linalg as sla  # noqa: E402

from nitrix.geometry import (  # noqa: E402
    affine_exp,
    affine_grid,
    apply_affine,
    identity_grid,
    rigid_exp,
    rigid_log,
)
from nitrix.linalg import (  # noqa: E402
    cg,
    gauss_newton,
    levenberg_marquardt,
    matrix_exp,
)


def _skew(omega):
    ox, oy, oz = omega
    return np.array([[0, -oz, oy], [oz, 0, -ox], [-oy, ox, 0]])


def _fd_grad_scalar(f, x, *, eps=1e-6):
    g = float(jax.grad(f)(x))
    fp = float(f(x + eps))
    fm = float(f(x - eps))
    assert np.isclose(g, (fp - fm) / (2 * eps), rtol=1e-4, atol=1e-6)


# ---------------------------------------------------------------------------
# matrix_exp
# ---------------------------------------------------------------------------


def test_matrix_exp_identity_and_diagonal():
    assert np.allclose(np.asarray(matrix_exp(jnp.zeros((4, 4)))), np.eye(4))
    d = np.array([0.3, -1.2, 2.0])
    out = np.asarray(matrix_exp(jnp.diag(jnp.asarray(d))))
    assert np.allclose(out, np.diag(np.exp(d)), atol=1e-10)


def test_matrix_exp_matches_scipy():
    rng = np.random.RandomState(0)
    a = rng.standard_normal((5, 5))
    assert np.allclose(np.asarray(matrix_exp(jnp.asarray(a))), sla.expm(a), atol=1e-8)


def test_matrix_exp_det_equals_exp_trace():
    rng = np.random.RandomState(1)
    a = rng.standard_normal((4, 4))
    out = np.asarray(matrix_exp(jnp.asarray(a)))
    assert np.isclose(np.linalg.det(out), np.exp(np.trace(a)), rtol=1e-7)


def test_matrix_exp_skew_is_rotation():
    # A skew-symmetric generator exponentiates to a rotation (orthogonal,
    # det +1) -- the rigid-transform anchor.
    g = np.array([[0.0, -0.7, 0.2], [0.7, 0.0, -0.4], [-0.2, 0.4, 0.0]])
    r = np.asarray(matrix_exp(jnp.asarray(g)))
    assert np.allclose(r @ r.T, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(r), 1.0, atol=1e-9)


def test_matrix_exp_batched():
    rng = np.random.RandomState(2)
    a = rng.standard_normal((3, 4, 4))
    out = np.asarray(matrix_exp(jnp.asarray(a)))
    assert out.shape == (3, 4, 4)
    for i in range(3):
        assert np.allclose(out[i], sla.expm(a[i]), atol=1e-8)


def test_matrix_exp_differentiable():
    rng = np.random.RandomState(3)
    a = jnp.asarray(rng.standard_normal((3, 3)))

    def f(t):
        return matrix_exp(t * a).sum()

    _fd_grad_scalar(f, 0.5)


# ---------------------------------------------------------------------------
# cg
# ---------------------------------------------------------------------------


def test_cg_matches_direct_solve():
    a_np = np.array([[4.0, 1.0, 0.0], [1.0, 3.0, 0.5], [0.0, 0.5, 2.0]])
    b_np = np.array([1.0, 2.0, -1.0])
    x = cg(jnp.asarray(a_np), jnp.asarray(b_np), tol=1e-12, maxiter=100)
    assert np.allclose(np.asarray(x), np.linalg.solve(a_np, b_np), atol=1e-8)


def test_cg_matrix_free_and_ridge():
    a_np = np.array([[2.0, 0.3], [0.3, 1.5]])
    b_np = np.array([1.0, -2.0])
    a = jnp.asarray(a_np)
    # matvec callable form
    x = cg(lambda v: a @ v, jnp.asarray(b_np), tol=1e-12, maxiter=50)
    assert np.allclose(np.asarray(x), np.linalg.solve(a_np, b_np), atol=1e-8)
    # ridge
    x_r = cg(a, jnp.asarray(b_np), l2=0.5, tol=1e-12, maxiter=50)
    ref = np.linalg.solve(a_np + 0.5 * np.eye(2), b_np)
    assert np.allclose(np.asarray(x_r), ref, atol=1e-8)


def test_cg_stays_on_gpu_when_available():
    # The whole point of the SPD path: matvec-only CG survives the
    # cuSolver wedge and runs on whatever the default device is.
    a = jnp.asarray(np.array([[4.0, 1.0], [1.0, 3.0]]))
    b = jnp.asarray(np.array([1.0, 2.0]))
    x = cg(a, b, tol=1e-12, maxiter=50)
    assert list(x.devices()) == list(b.devices())


def test_cg_differentiable():
    a = jnp.asarray(np.array([[4.0, 1.0], [1.0, 3.0]]))
    b = jnp.asarray(np.array([1.0, 2.0]))

    def f(d):
        return cg(a + d * jnp.eye(2), b, tol=1e-12, maxiter=100).sum()

    _fd_grad_scalar(f, 0.3, eps=1e-5)


# ---------------------------------------------------------------------------
# geometry.transform — rigid
# ---------------------------------------------------------------------------


def test_rigid_exp_identity():
    assert np.allclose(np.asarray(rigid_exp(jnp.zeros(6), ndim=3)), np.eye(4))
    assert np.allclose(np.asarray(rigid_exp(jnp.zeros(3), ndim=2)), np.eye(3))


def test_rigid_exp_rotation_matches_rodrigues():
    omega = np.array([0.3, -0.5, 0.7])
    p = jnp.asarray(np.concatenate([omega, np.zeros(3)]))
    t = np.asarray(rigid_exp(p, ndim=3))
    r = t[:3, :3]
    assert np.allclose(r, sla.expm(_skew(omega)), atol=1e-9)
    assert np.allclose(r @ r.T, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(r), 1.0, atol=1e-9)
    assert np.allclose(t[:3, 3], 0.0)


def test_rigid_exp_translation_is_direct():
    p = jnp.asarray(np.array([0.0, 0.0, 0.0, 1.5, -2.0, 3.0]))
    t = np.asarray(rigid_exp(p, ndim=3))
    assert np.allclose(t[:3, :3], np.eye(3), atol=1e-12)
    assert np.allclose(t[:3, 3], [1.5, -2.0, 3.0])


def test_rigid_log_roundtrip_3d():
    rng = np.random.RandomState(0)
    p = np.concatenate([0.4 * rng.standard_normal(3), rng.standard_normal(3)])
    p_rt = np.asarray(rigid_log(rigid_exp(jnp.asarray(p), ndim=3), ndim=3))
    assert np.allclose(p_rt, p, atol=1e-7)


def test_rigid_2d_rotation_and_log():
    theta = 0.6
    p = jnp.asarray(np.array([theta, 2.0, -1.0]))
    t = np.asarray(rigid_exp(p, ndim=2))
    c, s = np.cos(theta), np.sin(theta)
    assert np.allclose(t[:2, :2], [[c, -s], [s, c]], atol=1e-9)
    assert np.allclose(t[:2, 2], [2.0, -1.0])
    assert np.allclose(np.asarray(rigid_log(jnp.asarray(t), ndim=2)), [theta, 2.0, -1.0], atol=1e-7)


def test_rigid_exp_differentiable():
    rng = np.random.RandomState(1)
    p = jnp.asarray(0.3 * rng.standard_normal(6))

    def f(s):
        return rigid_exp(s * p, ndim=3).sum()

    _fd_grad_scalar(f, 0.7)


# ---------------------------------------------------------------------------
# geometry.transform — affine
# ---------------------------------------------------------------------------


def test_affine_exp_identity_and_linear():
    assert np.allclose(np.asarray(affine_exp(jnp.zeros(12), ndim=3)), np.eye(4))
    rng = np.random.RandomState(2)
    a = 0.2 * rng.standard_normal((3, 3))
    params = np.concatenate([a.reshape(-1), [1.0, 2.0, 3.0]])
    t = np.asarray(affine_exp(jnp.asarray(params), ndim=3))
    assert np.allclose(t[:3, :3], sla.expm(a), atol=1e-7)
    assert np.allclose(t[:3, 3], [1.0, 2.0, 3.0])
    assert np.linalg.det(t[:3, :3]) > 0.0  # orientation-preserving


def test_affine_exp_differentiable():
    rng = np.random.RandomState(3)
    p = jnp.asarray(0.1 * rng.standard_normal(12))

    def f(s):
        return affine_exp(s * p, ndim=3).sum()

    _fd_grad_scalar(f, 0.5)


# ---------------------------------------------------------------------------
# apply_affine / affine_grid
# ---------------------------------------------------------------------------


def test_apply_affine_translation_and_center():
    # Pure translation shifts every coordinate.
    m = jnp.asarray(np.block([[np.eye(3), np.array([[1.0], [2.0], [3.0]])], [np.zeros((1, 3)), 1.0]]))
    coords = jnp.asarray(np.random.RandomState(0).rand(5, 3))
    out = np.asarray(apply_affine(coords, m))
    assert np.allclose(out, np.asarray(coords) + [1.0, 2.0, 3.0])
    # Rotation about a center fixes the center (with zero translation).
    rot = rigid_exp(jnp.asarray([0.0, 0.0, 0.7, 0.0, 0.0, 0.0]), ndim=3)
    c = jnp.asarray([4.0, 5.0, 6.0])
    fixed = np.asarray(apply_affine(c[None], rot, center=c))
    assert np.allclose(fixed[0], np.asarray(c), atol=1e-9)


def test_affine_grid_identity_and_translation():
    shape = (6, 7, 8)
    eye = jnp.eye(4)
    g = affine_grid(eye, shape)
    assert g.shape == (6, 7, 8, 3)
    assert np.allclose(np.asarray(g), np.asarray(identity_grid(shape)), atol=1e-9)
    # Translation: grid shifts by t regardless of center.
    m = jnp.asarray(np.block([[np.eye(3), np.array([[1.0], [-2.0], [0.5]])], [np.zeros((1, 3)), 1.0]]))
    gt = affine_grid(m, shape)
    assert np.allclose(np.asarray(gt), np.asarray(identity_grid(shape)) + [1.0, -2.0, 0.5], atol=1e-6)


# ---------------------------------------------------------------------------
# linalg.optimize — Gauss-Newton / Levenberg-Marquardt
# ---------------------------------------------------------------------------


def test_gauss_newton_solves_linear_least_squares():
    rng = np.random.RandomState(0)
    a = rng.standard_normal((10, 3))
    b = rng.standard_normal(10)
    a_j = jnp.asarray(a)
    b_j = jnp.asarray(b)

    def residual(x):
        return a_j @ x - b_j

    res = gauss_newton(
        residual, jnp.zeros(3), n_iters=2, cg_tol=1e-12, cg_maxiter=100
    )
    x_star, *_ = np.linalg.lstsq(a, b, rcond=None)
    assert np.allclose(np.asarray(res.params), x_star, atol=1e-6)


def test_levenberg_marquardt_nonlinear_recovers_parameter():
    t = np.linspace(0.0, 2.0, 60)
    k_true = 2.3
    data = jnp.asarray(np.exp(-k_true * t))
    t_j = jnp.asarray(t)

    def residual(k):
        return jnp.exp(-k[0] * t_j) - data

    res = levenberg_marquardt(
        residual, jnp.asarray([0.5]), n_iters=30, cg_tol=1e-10
    )
    assert np.isclose(float(res.params[0]), k_true, atol=1e-4)
    # accept/reject -> monotone non-increasing cost.
    ch = np.asarray(res.cost_history)
    assert np.all(np.diff(ch) <= 1e-9)
    assert float(res.cost) < 1e-8


def test_optimize_matrix_free_large_residual():
    # M >> P: the Jacobian is never materialised (jvp/vjp + cg).
    rng = np.random.RandomState(1)
    a = jnp.asarray(rng.standard_normal((2000, 4)))
    b = jnp.asarray(rng.standard_normal(2000))
    res = gauss_newton(
        lambda x: a @ x - b, jnp.zeros(4), n_iters=2, cg_tol=1e-10
    )
    x_star, *_ = np.linalg.lstsq(np.asarray(a), np.asarray(b), rcond=None)
    assert np.allclose(np.asarray(res.params), x_star, atol=1e-5)


def test_optimize_differentiable_through_solve():
    rng = np.random.RandomState(2)
    a = jnp.asarray(rng.standard_normal((8, 2)))

    def solve_sum(b):
        res = gauss_newton(
            lambda x: a @ x - b, jnp.zeros(2), n_iters=1, cg_tol=1e-12
        )
        return res.params.sum()

    b0 = jnp.asarray(rng.standard_normal(8))
    g = np.asarray(jax.grad(solve_sum)(b0))
    # GN(1 step) from 0 gives x = pinv(a) b, so d(sum x)/db = sum of
    # pinv(a) columns; compare to the analytic pseudo-inverse.
    expected = np.asarray(np.linalg.pinv(np.asarray(a))).sum(axis=0)
    assert np.allclose(g, expected, atol=1e-5)
