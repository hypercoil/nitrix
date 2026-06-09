# -*- coding: utf-8 -*-
"""Tests for ``nitrix.numerics.ode`` -- fixed-step ODE integrators."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update('jax_enable_x64', True)

from nitrix.numerics import euler, odeint, rk4


def test_rk4_matches_exponential_decay():
    # dy/dt = -y, y(0) = 1 -> y(t) = exp(-t).
    t = jnp.linspace(0.0, 2.0, 21)
    ys = rk4(lambda _t, y: -y, jnp.array([1.0]), t)
    ref = np.exp(-np.asarray(t))[:, None]
    np.testing.assert_allclose(np.asarray(ys), ref, atol=1e-6)


def test_rk4_more_accurate_than_euler():
    t = jnp.linspace(0.0, 2.0, 11)
    f = lambda _t, y: -y  # noqa: E731
    exact = np.exp(-2.0)
    err_rk4 = abs(float(rk4(f, jnp.array([1.0]), t)[-1, 0]) - exact)
    err_euler = abs(float(euler(f, jnp.array([1.0]), t)[-1, 0]) - exact)
    assert err_rk4 < err_euler
    assert err_rk4 < 1e-4


def test_constant_derivative_is_exact():
    # dy/dt = 2 -> y(t) = y0 + 2 t, exact for both integrators.
    t = jnp.linspace(0.0, 3.0, 7)
    for integ in (euler, rk4):
        ys = integ(lambda _t, y: jnp.array([2.0]), jnp.array([0.0]), t)
        np.testing.assert_allclose(
            np.asarray(ys)[:, 0], 2.0 * np.asarray(t), atol=1e-9
        )


def test_shapes_and_initial_value():
    t = jnp.linspace(0.0, 1.0, 5)
    y0 = jnp.ones((3, 2))
    ys = rk4(lambda _t, y: -y, y0, t)
    assert ys.shape == (5, 3, 2)
    np.testing.assert_array_equal(np.asarray(ys[0]), np.asarray(y0))


def test_harmonic_oscillator_conserves_energy():
    # dy/dt = [v, -x]; energy x^2 + v^2 stays ~constant under rk4.
    f = lambda _t, y: jnp.array([y[1], -y[0]])  # noqa: E731
    t = jnp.linspace(0.0, 2.0 * np.pi, 200)
    ys = rk4(f, jnp.array([1.0, 0.0]), t)
    energy = np.asarray(ys[:, 0] ** 2 + ys[:, 1] ** 2)
    np.testing.assert_allclose(energy, 1.0, atol=1e-4)


def test_odeint_dispatch_and_invalid_method():
    t = jnp.linspace(0.0, 1.0, 6)
    f = lambda _t, y: -y  # noqa: E731
    np.testing.assert_array_equal(
        np.asarray(odeint(f, jnp.array([1.0]), t, method='rk4')),
        np.asarray(rk4(f, jnp.array([1.0]), t)),
    )
    with pytest.raises(ValueError, match='expected'):
        odeint(f, jnp.array([1.0]), t, method='bogus')


def test_ode_differentiable_through_solver():
    t = jnp.linspace(0.0, 1.0, 11)
    f = lambda _t, y: -0.5 * y  # noqa: E731
    g = jax.grad(lambda y0: jnp.sum(rk4(f, y0, t)[-1]))(jnp.array([2.0, 3.0]))
    assert g.shape == (2,)
    assert bool(jnp.all(jnp.isfinite(g)))
    # d/dy0 of y(T) = exp(-0.5 T) for this linear system.
    np.testing.assert_allclose(np.asarray(g), np.exp(-0.5), rtol=1e-4)
