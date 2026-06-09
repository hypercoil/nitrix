# -*- coding: utf-8 -*-
"""Compile-invariance guards for the rolled registration loops.

The registration optimiser / Demons / velocity-integration loops are
rolled with ``lax.scan`` (``linalg/optimize.py``,
``register/diffeomorphic.py``, ``geometry/grid.py``).  A ``scan``
compiles its body once, so the traced jaxpr -- and hence the XLA cold
compile -- is *constant* in the iteration count.  These guards trace at
two very different iteration counts and assert the top-level equation
count does not grow, locking out a regression to a Python ``for`` unroll
(which made the cold compile linear in the iteration count: the ~145 s
pathology these rolls fixed).

A second test pins the ``'assembled'`` small-``P`` normal-equation path
to the ``'matrix_free'`` one (same optimum) -- the steady-state fix for
the rigid/affine recipes.
"""

from __future__ import annotations

import jax

jax.config.update('jax_enable_x64', True)

import jax.numpy as jnp

from nitrix.geometry import integrate_velocity_field
from nitrix.linalg import gauss_newton, levenberg_marquardt


def _eqn_count(f, *args) -> int:
    """Top-level equation count of the traced jaxpr of ``f``."""
    return len(jax.make_jaxpr(f)(*args).jaxpr.eqns)


# ---------------------------------------------------------------------------
# Compile cost is constant in the iteration count (the roll did not unroll).
# ---------------------------------------------------------------------------


def test_gauss_newton_compile_constant_in_iters() -> None:
    a = jax.random.normal(jax.random.PRNGKey(0), (40, 6))
    b = jax.random.normal(jax.random.PRNGKey(1), (40,))

    def residual(x):
        return a @ x - b

    x0 = jnp.zeros(6)
    few = _eqn_count(lambda x: gauss_newton(residual, x, n_iters=5).params, x0)
    many = _eqn_count(
        lambda x: gauss_newton(residual, x, n_iters=50).params, x0
    )
    assert few == many


def test_levenberg_marquardt_compile_constant_in_iters() -> None:
    a = jax.random.normal(jax.random.PRNGKey(0), (40, 6))
    b = jax.random.normal(jax.random.PRNGKey(1), (40,))

    def residual(x):
        return a @ x - b

    x0 = jnp.zeros(6)
    few = _eqn_count(
        lambda x: levenberg_marquardt(residual, x, n_iters=5).params, x0
    )
    many = _eqn_count(
        lambda x: levenberg_marquardt(residual, x, n_iters=50).params, x0
    )
    assert few == many


def test_integrate_velocity_compile_constant_in_steps() -> None:
    v = jnp.zeros((8, 8, 2))
    few = _eqn_count(lambda v: integrate_velocity_field(v, n_steps=2), v)
    many = _eqn_count(lambda v: integrate_velocity_field(v, n_steps=16), v)
    assert few == many


# ---------------------------------------------------------------------------
# Assembled small-P normal equations == matrix-free (and recover truth).
# ---------------------------------------------------------------------------


def test_assembled_matches_matrix_free() -> None:
    a = 0.1 * jax.random.normal(jax.random.PRNGKey(2), (30, 4))
    x_true = jnp.array([0.2, -0.1, 0.3, 0.05])
    y = jnp.exp(a @ x_true)

    def residual(x):
        return jnp.exp(a @ x) - y

    x0 = jnp.zeros(4)
    free = levenberg_marquardt(
        residual, x0, n_iters=30, jacobian='matrix_free'
    )
    assembled = levenberg_marquardt(
        residual, x0, n_iters=30, jacobian='assembled'
    )
    auto = levenberg_marquardt(residual, x0, n_iters=30, jacobian='auto')
    # auto resolves to assembled for this P (=4), so all three agree.
    assert jnp.allclose(free.params, assembled.params, atol=1e-5)
    assert jnp.allclose(auto.params, assembled.params, atol=1e-12)
    assert jnp.allclose(assembled.params, x_true, atol=1e-3)
