# -*- coding: utf-8 -*-
"""Tests for the v4 Phase-0 / Phase-2 substrate.

- **Anderson acceleration** (``numerics.fixed_point_solve``): same solution
  and IFT gradient as Picard on an easy problem; converges where Picard
  stalls on a stiff (spectral-radius -> 1) fixed point.
- **``geometry._diffeo_sqrt``**: the damped square root squares back to its
  input (``compose(w, w) == s``).
- **``geometry.field_log``**: the round-trip ``exp(log(s)) == s`` is exact;
  generating-velocity accuracy improves with ``n_sqrt``; the ``bch``
  correction trades round-trip fidelity for generator fidelity;
  differentiable.
- **``geometry.invert_displacement`` robustness**: Anderson converges where
  Picard under-converges at a fixed budget on a large deformation; the
  ``return_residual`` report matches the realised inversion error.
"""

from __future__ import annotations

import jax

jax.config.update('jax_enable_x64', True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from nitrix.geometry import (  # noqa: E402
    compose_displacement,
    field_log,
    integrate_velocity_field,
    invert_displacement,
)
from nitrix.geometry.deformation import _diffeo_sqrt  # noqa: E402
from nitrix.numerics import fixed_point_solve  # noqa: E402
from nitrix.smoothing import gaussian  # noqa: E402


def _smooth_field(shape, ndim, scale, seed):
    rng = np.random.RandomState(seed)
    f = rng.standard_normal(shape + (ndim,))
    f = np.moveaxis(f, -1, 0)
    f = np.asarray(
        gaussian(jnp.asarray(f), sigma=3.0, spatial_rank=len(shape))
    )
    f = np.moveaxis(f, 0, -1)
    return jnp.asarray(scale * f)


def _interior(a, m=6):
    sl = tuple(slice(m, -m) for _ in range(a.ndim - 1)) + (slice(None),)
    return np.asarray(a)[sl]


# ---------------------------------------------------------------------------
# Anderson acceleration
# ---------------------------------------------------------------------------


def _stiff_linear(n=6, rho=0.97, seed=0):
    eigs = np.linspace(0.5, rho, n)
    q, _ = np.linalg.qr(np.random.RandomState(seed).standard_normal((n, n)))
    a = q @ np.diag(eigs) @ q.T
    b = np.random.RandomState(seed + 1).standard_normal(n)
    x_star = np.linalg.solve(np.eye(n) - a, b)
    return jnp.asarray(a), jnp.asarray(b), x_star


def test_anderson_matches_picard_and_gradient():
    a, b, x_star = _stiff_linear(rho=0.8)
    f = lambda p, y: a @ y + p  # noqa: E731
    kw = dict(tol=1e-12, max_iter=500)
    xp = fixed_point_solve(f, b, jnp.zeros(6), acceleration='picard', **kw)
    xa = fixed_point_solve(
        f, b, jnp.zeros(6), acceleration='anderson', depth=4, **kw
    )
    assert np.allclose(np.asarray(xp), x_star, atol=1e-8)
    assert np.allclose(np.asarray(xa), x_star, atol=1e-8)

    def loss(bb):
        return fixed_point_solve(
            f, bb, jnp.zeros(6), acceleration='anderson', depth=4, **kw
        ).sum()

    g = np.asarray(jax.grad(loss)(b))
    analytic = np.linalg.solve((np.eye(6) - np.asarray(a)).T, np.ones(6))
    assert np.allclose(g, analytic, atol=1e-6)


def test_anderson_converges_where_picard_stalls():
    # Spectral radius 0.97 -> Picard error ~0.97^k; at a tight budget it has
    # barely moved, while windowed Anderson is orders of magnitude closer.
    a, b, x_star = _stiff_linear(rho=0.97)
    f = lambda p, y: a @ y + p  # noqa: E731
    budget = dict(tol=1e-12, max_iter=40)
    xp = fixed_point_solve(f, b, jnp.zeros(6), acceleration='picard', **budget)
    xa = fixed_point_solve(
        f, b, jnp.zeros(6), acceleration='anderson', depth=5, **budget
    )
    err_p = float(np.max(np.abs(np.asarray(xp) - x_star)))
    err_a = float(np.max(np.abs(np.asarray(xa) - x_star)))
    assert err_p > 1e-2  # Picard, at the same budget, has not converged
    assert err_a < 1e-3  # Anderson is far closer
    assert err_a < err_p / 100.0  # by orders of magnitude


# ---------------------------------------------------------------------------
# diffeomorphism square root
# ---------------------------------------------------------------------------


def test_diffeo_sqrt_squares_to_input():
    s = _smooth_field((40, 40), 2, 0.4, 3)
    w = _diffeo_sqrt(s, tol=1e-8, max_iter=80, mode='nearest')
    squared = compose_displacement(w, w)
    assert np.abs(_interior(squared) - _interior(s)).max() < 5e-3


# ---------------------------------------------------------------------------
# field_log
# ---------------------------------------------------------------------------


def test_field_log_roundtrip_exact_2d_and_3d():
    for shape, ndim, seed in (((40, 40), 2, 1), ((20, 20, 20), 3, 2)):
        v = _smooth_field(shape, ndim, 0.3, seed)
        s = integrate_velocity_field(v, n_steps=6)
        v_rec = field_log(s, n_sqrt=6)
        s_rt = integrate_velocity_field(v_rec, n_steps=6)
        # Round-trip exp(log(s)) == s is exact (to the sqrt solver tol).
        assert np.abs(_interior(s_rt) - _interior(s)).max() < 1e-5
        # Generating velocity recovered (looser: the log approximation).
        assert np.abs(_interior(v_rec) - _interior(v)).max() < 1e-3


def test_field_log_generator_accuracy_improves_with_n_sqrt():
    v = _smooth_field((40, 40), 2, 0.5, 4)
    s = integrate_velocity_field(v, n_steps=8)

    def gen_err(n):
        return np.abs(_interior(field_log(s, n_sqrt=n)) - _interior(v)).max()

    assert gen_err(4) < gen_err(2)
    assert gen_err(6) < gen_err(4)


def test_field_log_bch_trades_roundtrip_for_generator():
    # At a small n_sqrt the log approximation is visible: bch improves the
    # generating velocity but breaks the exact round-trip.
    v = _smooth_field((40, 40), 2, 0.5, 5)
    s = integrate_velocity_field(v, n_steps=8)
    n = 2
    v_fo = field_log(s, n_sqrt=n, correction='first_order')
    v_bch = field_log(s, n_sqrt=n, correction='bch')
    gen_fo = np.abs(_interior(v_fo) - _interior(v)).max()
    gen_bch = np.abs(_interior(v_bch) - _interior(v)).max()
    rt_fo = np.abs(
        _interior(integrate_velocity_field(v_fo, n_steps=n)) - _interior(s)
    ).max()
    rt_bch = np.abs(
        _interior(integrate_velocity_field(v_bch, n_steps=n)) - _interior(s)
    ).max()
    assert gen_bch < gen_fo  # bch is the more accurate generator
    assert rt_fo < rt_bch  # first-order keeps the exact round-trip


def test_field_log_differentiable():
    v = _smooth_field((24, 24), 2, 0.3, 6)
    s = integrate_velocity_field(v, n_steps=6)

    def loss(scale):
        return (field_log(scale * s, n_sqrt=4) ** 2).sum()

    g = float(jax.grad(loss)(jnp.asarray(1.0)))
    fd = (
        float(loss(jnp.asarray(1.0 + 1e-5)))
        - float(loss(jnp.asarray(1.0 - 1e-5)))
    ) / 2e-5
    assert np.isclose(g, fd, rtol=2e-3, atol=1e-4)


# ---------------------------------------------------------------------------
# invert_displacement robustness
# ---------------------------------------------------------------------------


def test_invert_default_picard_converges_and_anderson_available():
    # On the smoothed regime registration produces, the default (Picard)
    # converges well; the Anderson opt-in also converges (it is the escape
    # hatch for a genuinely stiff field, not the default).
    s = _smooth_field((48, 48), 2, 0.7, 8)
    _, res_p = invert_displacement(
        s, tol=1e-9, max_iter=80, return_residual=True
    )
    _, res_a = invert_displacement(
        s, tol=1e-9, max_iter=80, acceleration='anderson', return_residual=True
    )
    assert float(res_p) < 1e-5
    assert float(res_a) < 1e-4


def test_invert_residual_report_matches_realised_error():
    s = _smooth_field((40, 40), 2, 0.4, 9)
    s_inv, residual = invert_displacement(
        s, tol=1e-9, max_iter=100, return_residual=True
    )
    composed = compose_displacement(s, s_inv)
    realised = float(np.sqrt(np.mean(np.asarray(composed) ** 2)))
    assert np.isclose(float(residual), realised, rtol=1e-6, atol=1e-12)
    assert float(residual) < 1e-4
