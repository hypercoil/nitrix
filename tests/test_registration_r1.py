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

from nitrix.linalg import cg, matrix_exp  # noqa: E402


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
