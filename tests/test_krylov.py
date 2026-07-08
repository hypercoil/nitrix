# -*- coding: utf-8 -*-
"""Tests for the matrix-free Krylov solvers (linalg.krylov).

MINRES is pinned on symmetric *indefinite* systems (where CG fails) against
scipy and by its implicit-VJP gradient matching the analytic one; bicgstab /
gmres solve non-symmetric systems; the shipped CG stays correct after the
operator-resolution refactor. Matvec-callable operators, the l2 ridge, jit and
batching are exercised.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update('jax_enable_x64', True)

from nitrix.linalg import bicgstab, cg, gmres, minres  # noqa: E402


def _symmetric_indefinite(n=40, seed=0):
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    eig = rng.standard_normal(n) * 2.0  # mixed signs -> indefinite
    a = q @ np.diag(eig) @ q.T
    a = (a + a.T) / 2
    b = rng.standard_normal(n)
    return jnp.asarray(a), jnp.asarray(b)


def _residual(a, x, b):
    return float(jnp.linalg.norm(a @ x - b))


def test_minres_solves_symmetric_indefinite():
    a, b = _symmetric_indefinite()
    assert _residual(a, minres(a, b, tol=1e-10), b) < 1e-8


def test_minres_matches_scipy():
    sp = pytest.importorskip('scipy.sparse.linalg')
    a, b = _symmetric_indefinite(seed=1)
    x = minres(a, b, tol=1e-10)
    xs, _ = sp.minres(np.asarray(a), np.asarray(b), rtol=1e-10)
    np.testing.assert_allclose(np.asarray(x), xs, atol=1e-7)


def test_minres_gradient_is_the_implicit_vjp():
    # d/db ||A^{-1} b||^2 = 2 A^{-2} b for symmetric A -- the implicit rule.
    a, b = _symmetric_indefinite(seed=2)
    g = jax.grad(lambda b: jnp.sum(minres(a, b, tol=1e-10) ** 2))(b)
    a_inv = np.linalg.inv(np.asarray(a))
    analytic = 2 * a_inv @ a_inv @ np.asarray(b)
    np.testing.assert_allclose(np.asarray(g), analytic, atol=1e-7)


def test_minres_matvec_callable_and_jit():
    a, b = _symmetric_indefinite(seed=3)
    x_cb = minres(lambda v: a @ v, b, tol=1e-10)
    assert _residual(a, x_cb, b) < 1e-8
    x_jit = jax.jit(lambda a, b: minres(a, b, tol=1e-10))(a, b)
    assert _residual(a, x_jit, b) < 1e-8


def test_minres_x0_and_l2():
    a, b = _symmetric_indefinite(seed=4)
    x0 = jnp.ones_like(b)
    assert _residual(a, minres(a, b, x0=x0, tol=1e-10), b) < 1e-8
    # l2 ridge solves (A + l2 I) x = b
    x = minres(a, b, l2=0.5, tol=1e-10)
    assert _residual(a + 0.5 * jnp.eye(a.shape[0]), x, b) < 1e-8


def _nonsymmetric(n=40, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, n)) + n * np.eye(n)  # diagonally dominant
    b = rng.standard_normal(n)
    return jnp.asarray(a), jnp.asarray(b)


def test_bicgstab_and_gmres_solve_nonsymmetric():
    a, b = _nonsymmetric()
    assert _residual(a, bicgstab(a, b, tol=1e-9), b) < 1e-6
    assert _residual(a, gmres(a, b, tol=1e-9, restart=a.shape[0]), b) < 1e-6


def test_bicgstab_grad_finite():
    a, b = _nonsymmetric(seed=1)
    g = jax.grad(lambda b: jnp.sum(bicgstab(a, b, tol=1e-9) ** 2))(b)
    assert bool(jnp.all(jnp.isfinite(g)))


def test_cg_still_solves_spd():
    # regression after the _as_op refactor: CG on an SPD system is unchanged.
    a, b = _symmetric_indefinite(seed=5)
    spd = a @ a.T + jnp.eye(a.shape[0])
    assert _residual(spd, cg(spd, b, tol=1e-9), b) < 1e-6


def test_minres_batched():
    a, b = _symmetric_indefinite(seed=6)
    bb = jnp.stack([b, 2.0 * b, -b])  # (3, n)
    xs = jax.vmap(lambda bi: minres(a, bi, tol=1e-10))(bb)
    assert xs.shape == bb.shape
    for i in range(3):
        assert _residual(a, xs[i], bb[i]) < 1e-8
