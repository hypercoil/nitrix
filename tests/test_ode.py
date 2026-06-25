# -*- coding: utf-8 -*-
"""Tests for ``nitrix.numerics.ode`` -- fixed-step ODE integrators."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update('jax_enable_x64', True)

from nitrix.numerics import (
    euler,
    local_linearization,
    midpoint,
    odeint,
    rk4,
)

# --- hand-written reference steppers (the textbook explicit-RK tableaux) ----
# These unroll the step expression in a plain (eager) Python loop, with the
# same ``jnp`` arithmetic the integrators use.  They are the oracle for the
# bring-your-own-weights parity gate: the integrator must reproduce this
# tableau to floating-point round-off -- the same arithmetic a library such
# as ``diffrax`` takes for these fixed steps.  Note this is round-off-exact,
# *not* bit-for-bit against the eager loop: XLA may contract a ``y + dt*f``
# multiply-add into a single FMA inside the compiled ``scan``, a last-ULP
# difference.  (Two computations compiled the *same* way -- nitrix ``odeint``
# vs a diffrax solve, both under XLA -- can still agree bit-for-bit; that is
# the ``test_parity_vs_diffrax_when_available`` oracle.)


def _ref(stepper, f, y0, t):
    ys = [y0]
    y = y0
    for i in range(len(t) - 1):
        dt = t[i + 1] - t[i]
        y = stepper(f, t[i], dt, y)
        ys.append(y)
    return jnp.stack(ys)


def _euler_ref(f, t, dt, y):
    return y + dt * f(t, y)


def _midpoint_ref(f, t, dt, y):
    k1 = f(t, y)
    return y + dt * f(t + 0.5 * dt, y + 0.5 * dt * k1)


def _rk4_ref(f, t, dt, y):
    k1 = f(t, y)
    k2 = f(t + 0.5 * dt, y + 0.5 * dt * k1)
    k3 = f(t + 0.5 * dt, y + 0.5 * dt * k2)
    k4 = f(t + dt, y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


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


# --- midpoint (RK2) -------------------------------------------------------


def test_midpoint_matches_exponential_decay():
    t = jnp.linspace(0.0, 2.0, 81)  # h=0.025; RK2 global error ~ h^2
    ys = midpoint(lambda _t, y: -y, jnp.array([1.0]), t)
    ref = np.exp(-np.asarray(t))[:, None]
    np.testing.assert_allclose(np.asarray(ys), ref, atol=1e-4)


def test_midpoint_second_order_between_euler_and_rk4():
    # On a smooth field, accuracy ordering is euler < midpoint < rk4.
    t = jnp.linspace(0.0, 2.0, 11)
    f = lambda _t, y: -y  # noqa: E731
    exact = np.exp(-2.0)
    err_euler = abs(float(euler(f, jnp.array([1.0]), t)[-1, 0]) - exact)
    err_mid = abs(float(midpoint(f, jnp.array([1.0]), t)[-1, 0]) - exact)
    err_rk4 = abs(float(rk4(f, jnp.array([1.0]), t)[-1, 0]) - exact)
    assert err_rk4 < err_mid < err_euler


def test_midpoint_dispatch_via_odeint():
    t = jnp.linspace(0.0, 1.0, 6)
    f = lambda _t, y: -y  # noqa: E731
    np.testing.assert_array_equal(
        np.asarray(odeint(f, jnp.array([1.0]), t, method='midpoint')),
        np.asarray(midpoint(f, jnp.array([1.0]), t)),
    )


# --- round-off parity gate (bring-your-own-weights safety net) -------------


def _vertex_field(t, x):
    # A representative time-independent per-vertex field: a small linear map
    # plus a constant drift, the shape a surface neural-ODE integrates.
    a = jnp.asarray([[-0.3, 0.1, 0.0], [0.05, -0.2, 0.15], [0.1, 0.0, -0.25]])
    return x @ a.T + 0.02


@pytest.mark.parametrize(
    ('name', 'integ', 'ref_step'),
    [
        ('euler', euler, _euler_ref),
        ('midpoint', midpoint, _midpoint_ref),
        ('rk4', rk4, _rk4_ref),
    ],
)
def test_matches_textbook_tableau_to_roundoff(name, integ, ref_step):
    # The load-bearing gate: each method reproduces the explicit tableau
    # unrolled by hand to floating-point round-off -- the same arithmetic a
    # brought checkpoint was trained/evaluated against.  Tolerance is ~1 ULP
    # (FMA contraction in the compiled scan), thousands of times tighter than
    # any algorithmic step error, so this pins the arithmetic, not accuracy.
    t = jnp.linspace(0.0, 1.0, 6)  # CortexODE/SurfNet step ~0.1-0.2 over [0,1]
    x0 = jnp.asarray(
        [
            [0.0, 1.0, -1.0],
            [0.5, -0.5, 0.25],
            [-1.0, 0.2, 0.8],
            [0.3, 0.3, 0.3],
            [-0.7, 0.9, -0.4],
        ]
    )
    got = integ(_vertex_field, x0, t)
    ref = _ref(ref_step, _vertex_field, x0, t)
    np.testing.assert_allclose(
        np.asarray(got), np.asarray(ref), rtol=1e-12, atol=1e-14
    )


def test_parity_vs_diffrax_when_available():
    # The literal diffrax oracle for the BYOW gate, active once diffrax is a
    # test dep (target: bit-exact, since the tableau arithmetic is identical).
    diffrax = pytest.importorskip('diffrax')
    t = jnp.linspace(0.0, 1.0, 6)
    x0 = jnp.asarray([[0.0, 1.0, -1.0], [0.5, -0.5, 0.25]])
    dt0 = float(t[1] - t[0])
    term = diffrax.ODETerm(lambda tt, y, _a: _vertex_field(tt, y))
    saveat = diffrax.SaveAt(ts=t)
    for name, solver in (
        ('euler', diffrax.Euler()),
        ('midpoint', diffrax.Midpoint()),
    ):
        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0=float(t[0]),
            t1=float(t[-1]),
            dt0=dt0,
            y0=x0,
            saveat=saveat,
        )
        ours = odeint(_vertex_field, x0, t, method=name)
        np.testing.assert_allclose(
            np.asarray(ours),
            np.asarray(sol.ys),
            rtol=0.0,
            atol=1e-12,
            err_msg=f'diffrax parity failed for {name}',
        )


# --- the args contract (closed over, not a kwarg) -------------------------


def test_args_are_closed_over_and_differentiable():
    # There is no ``args=`` kwarg; the consumer closes extra parameters over
    # the field.  Closed-over arrays still differentiate end-to-end.
    t = jnp.linspace(0.0, 1.0, 11)

    def integrate_with(rate):
        f = lambda _t, y: rate * y  # noqa: E731  -- ``rate`` closed over
        return odeint(f, jnp.array([1.0]), t, method='rk4')[-1, 0]

    val = integrate_with(jnp.asarray(-0.5))
    np.testing.assert_allclose(float(val), np.exp(-0.5), rtol=1e-4)
    # d/drate of y(1) = exp(rate) is exp(rate) * 1 (T=1); finite + correct.
    g = jax.grad(integrate_with)(jnp.asarray(-0.5))
    assert bool(jnp.isfinite(g))
    np.testing.assert_allclose(float(g), np.exp(-0.5), rtol=1e-3)


# --- adaptive steppers are rejected with a clear message ------------------


@pytest.mark.parametrize('name', ['dopri5', 'dopri8', 'tsit5'])
def test_adaptive_methods_rejected(name):
    t = jnp.linspace(0.0, 1.0, 6)
    f = lambda _t, y: -y  # noqa: E731
    with pytest.raises(ValueError, match='adaptive'):
        odeint(f, jnp.array([1.0]), t, method=name)


# --- local linearization (exponential integrator, DS-3) -------------------
# The A-stable exponential stepper behind SPM's ``spm_int``: exact for an
# affine-autonomous field at any step size, and stable on stiff systems where
# the explicit steppers diverge.  The DCM-Balloon proof-of-feasibility path.


def _linear_exact(a, b, y0, t):
    # Closed form of dy/dt = A y + b (autonomous):
    #   y(t) = expm(tA) (y0 + A^{-1} b) - A^{-1} b.
    from scipy.linalg import expm

    a_np, b_np = np.asarray(a), np.asarray(b)
    ainv_b = np.linalg.solve(a_np, b_np)
    shifted = np.asarray(y0) + ainv_b
    return np.stack(
        [expm(tk * a_np) @ shifted - ainv_b for tk in np.asarray(t)]
    )


def test_ll_exact_for_affine_linear_system():
    # The defining property: LL integrates an affine-autonomous field exactly
    # (to matrix_exp round-off) at ANY step size -- here a deliberately coarse
    # 4-point grid, where rk4 would still carry visible truncation error.
    pytest.importorskip('scipy')
    a = jnp.asarray([[-1.0, 2.0], [0.0, -3.0]])
    b = jnp.asarray([0.5, -0.2])
    y0 = jnp.asarray([1.0, -0.5])
    t = jnp.linspace(0.0, 2.0, 4)
    f = lambda _t, y: a @ y + b  # noqa: E731
    ys = local_linearization(f, y0, t)
    ref = _linear_exact(a, b, y0, t)
    np.testing.assert_allclose(np.asarray(ys), ref, rtol=1e-6, atol=1e-8)


def test_ll_stable_on_stiff_system_where_euler_diverges():
    # Stiff: dy/dt = -100 y.  Explicit Euler at dt=0.1 has amplification
    # |1 - 100*0.1| = 9 -> blows up; LL gives exp(-100 t) and stays bounded.
    f = lambda _t, y: -100.0 * y  # noqa: E731
    t = jnp.linspace(
        0.0, 1.0, 11
    )  # dt=0.1, far past the explicit stability limit
    y0 = jnp.asarray([1.0])
    ll = local_linearization(f, y0, t)
    eu = euler(f, y0, t)
    ref = np.exp(-100.0 * np.asarray(t))[:, None]
    np.testing.assert_allclose(np.asarray(ll), ref, atol=1e-6)
    assert abs(float(eu[-1, 0])) > 1e3  # euler diverged
    assert abs(float(ll[-1, 0])) < 1e-3  # LL bounded (~0)


def test_ll_dispatch_via_odeint():
    t = jnp.linspace(0.0, 1.0, 6)
    f = lambda _t, y: -y  # noqa: E731
    y0 = jnp.asarray([1.0])
    np.testing.assert_array_equal(
        np.asarray(odeint(f, y0, t, method='local_linearization')),
        np.asarray(local_linearization(f, y0, t)),
    )


def test_ll_differentiable_through_solver():
    # grad of a closed-over parameter, straight through the matrix-exp scan.
    t = jnp.linspace(0.0, 1.0, 6)

    def integrate_with(rate):
        f = lambda _t, y: rate * y  # noqa: E731  -- ``rate`` closed over
        return local_linearization(f, jnp.array([1.0]), t)[-1, 0]

    val = integrate_with(jnp.asarray(-0.5))
    np.testing.assert_allclose(float(val), np.exp(-0.5), rtol=1e-5)
    g = jax.grad(integrate_with)(jnp.asarray(-0.5))
    assert bool(jnp.isfinite(g))
    np.testing.assert_allclose(float(g), np.exp(-0.5), rtol=1e-4)


def test_ll_vmap_over_batch():
    # LL requires a flat 1-D state; vmap it for a batch of independent systems
    # (the multi-region pattern the Balloon model uses).
    t = jnp.linspace(0.0, 1.0, 6)
    rates = jnp.asarray([-0.5, -1.0, -2.0])
    y0 = jnp.ones((3, 1))

    def integ(rate, init):
        return local_linearization(lambda _t, y: rate * y, init, t)

    ys = jax.vmap(integ)(rates, y0)
    assert ys.shape == (3, len(t), 1)
    ref = np.exp(np.asarray(rates)[:, None] * np.asarray(t)[None, :])[
        ..., None
    ]
    np.testing.assert_allclose(np.asarray(ys), ref, rtol=1e-5)
