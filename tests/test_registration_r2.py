# -*- coding: utf-8 -*-
"""Tests for the R2 diffeomorphic substrate.

- **numerics.fixed_point_solve**: forward Picard convergence; the
  implicit-VJP gradient matching the implicit-function-theorem analytic
  (linear) and finite differences (nonlinear); ``x0``-independence.
- **geometry deformation algebra**: ``compose_displacement`` identities
  and warp-by-then-warp equivalence; ``invert_displacement`` (composition
  with the inverse is the identity; differentiable); ``compose_velocity``
  additive / BCH properties.
"""

from __future__ import annotations

import jax

jax.config.update('jax_enable_x64', True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from nitrix.geometry import (  # noqa: E402
    compose_displacement,
    compose_velocity,
    identity_grid,
    invert_displacement,
    spatial_transform,
)
from nitrix.numerics import fixed_point_solve  # noqa: E402
from nitrix.smoothing import gaussian  # noqa: E402


def _smooth_field(shape, ndim, scale, seed):
    rng = np.random.RandomState(seed)
    f = rng.standard_normal(shape + (ndim,))
    f = np.moveaxis(f, -1, 0)
    f = np.asarray(gaussian(jnp.asarray(f), sigma=3.0, spatial_rank=len(shape)))
    f = np.moveaxis(f, 0, -1)
    return jnp.asarray(scale * f)


def _interior(a, m=6):
    sl = tuple(slice(m, -m) for _ in range(a.ndim - 1)) + (slice(None),)
    return np.asarray(a)[sl]


# ---------------------------------------------------------------------------
# fixed_point_solve
# ---------------------------------------------------------------------------


def test_fixed_point_linear_forward_and_grad():
    a = jnp.asarray(
        np.array([[0.3, 0.1, 0.0], [0.05, 0.2, 0.1], [0.0, 0.1, 0.25]])
    )
    b = jnp.asarray(np.array([1.0, -2.0, 0.5]))
    x = fixed_point_solve(
        lambda p, y: a @ y + p, b, jnp.zeros(3), tol=1e-12, max_iter=500
    )
    ref = np.linalg.solve(np.eye(3) - np.asarray(a), np.asarray(b))
    assert np.allclose(np.asarray(x), ref, atol=1e-8)

    def loss(bb):
        return fixed_point_solve(
            lambda p, y: a @ y + p, bb, jnp.zeros(3), tol=1e-12, max_iter=500
        ).sum()

    g = np.asarray(jax.grad(loss)(b))
    analytic = np.linalg.solve((np.eye(3) - np.asarray(a)).T, np.ones(3))
    assert np.allclose(g, analytic, atol=1e-6)


def test_fixed_point_nonlinear_grad_and_x0_independence():
    def fp(a, x0):
        return fixed_point_solve(
            lambda p, x: jnp.tanh(p * x) + 0.3, a, x0, tol=1e-12, max_iter=500
        )

    ga = float(jax.grad(lambda a: fp(a, jnp.zeros(())))(jnp.asarray(0.4)))
    fd = (
        float(fp(jnp.asarray(0.4 + 1e-6), jnp.zeros(())))
        - float(fp(jnp.asarray(0.4 - 1e-6), jnp.zeros(())))
    ) / 2e-6
    assert np.isclose(ga, fd, rtol=1e-4)
    # x0 must not change the solution.
    assert np.isclose(
        float(fp(jnp.asarray(0.4), jnp.zeros(()))),
        float(fp(jnp.asarray(0.4), 3.0 + jnp.zeros(()))),
        atol=1e-8,
    )


# ---------------------------------------------------------------------------
# compose_displacement
# ---------------------------------------------------------------------------


def test_compose_displacement_identities():
    shape = (32, 32)
    v = _smooth_field(shape, 2, 0.5, 0)
    zero = jnp.zeros_like(v)
    assert np.allclose(np.asarray(compose_displacement(v, zero)), np.asarray(v), atol=1e-10)
    assert np.allclose(np.asarray(compose_displacement(zero, v)), np.asarray(v), atol=1e-10)


def test_compose_displacement_warp_then_warp():
    shape = (40, 40)
    img = jnp.asarray(gaussian(jnp.asarray(np.random.RandomState(3).rand(*shape)), sigma=2.0))
    outer = _smooth_field(shape, 2, 0.6, 1)
    inner = _smooth_field(shape, 2, 0.6, 2)
    grid = identity_grid(shape, dtype=img.dtype)
    composed = compose_displacement(outer, inner)
    direct = spatial_transform(img[..., None], grid + composed, mode='nearest')[..., 0]
    seq = spatial_transform(
        spatial_transform(img[..., None], grid + outer, mode='nearest'),
        grid + inner,
        mode='nearest',
    )[..., 0]
    assert np.allclose(_interior(direct[..., None]), _interior(seq[..., None]), atol=2e-3)


# ---------------------------------------------------------------------------
# invert_displacement
# ---------------------------------------------------------------------------


def test_invert_displacement_roundtrip():
    shape = (40, 40)
    s = _smooth_field(shape, 2, 0.4, 5)  # small, diffeomorphic
    s_inv = invert_displacement(s, tol=1e-8, max_iter=80)
    # (id + s) ∘ (id + s_inv) ≈ id  -> composed displacement ≈ 0 (interior).
    composed = compose_displacement(s, s_inv)
    assert np.abs(_interior(composed)).max() < 5e-2
    # other order too.
    composed2 = compose_displacement(s_inv, s)
    assert np.abs(_interior(composed2)).max() < 5e-2


def test_invert_displacement_differentiable():
    shape = (24, 24)
    s = _smooth_field(shape, 2, 0.3, 7)

    def loss(scale):
        return (invert_displacement(scale * s, tol=1e-8, max_iter=80) ** 2).sum()

    g = float(jax.grad(loss)(jnp.asarray(1.0)))
    fd = (float(loss(jnp.asarray(1.0 + 1e-5))) - float(loss(jnp.asarray(1.0 - 1e-5)))) / 2e-5
    assert np.isclose(g, fd, rtol=1e-3, atol=1e-4)


# ---------------------------------------------------------------------------
# compose_velocity
# ---------------------------------------------------------------------------


def test_compose_velocity_additive_and_bch():
    shape = (24, 24)
    v = _smooth_field(shape, 2, 0.3, 8)
    u = _smooth_field(shape, 2, 0.3, 9)
    # order 1 is exact addition.
    assert np.allclose(np.asarray(compose_velocity(v, u, order=1)), np.asarray(v + u), atol=1e-12)
    # order 2 with a zero field is identity.
    zero = jnp.zeros_like(v)
    assert np.allclose(np.asarray(compose_velocity(v, zero, order=2)), np.asarray(v), atol=1e-10)
    # the BCH correction is antisymmetric: bracket(v,u) = -bracket(u,v).
    bvu = np.asarray(compose_velocity(v, u, order=2) - (v + u))
    buv = np.asarray(compose_velocity(u, v, order=2) - (u + v))
    assert np.allclose(bvu, -buv, atol=1e-10)
