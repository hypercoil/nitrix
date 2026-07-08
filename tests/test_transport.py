# -*- coding: utf-8 -*-
"""Tests for entropic optimal transport (transport.sinkhorn).

The Sinkhorn plan is pinned by the transport constraints (its marginals recover
``a`` and ``b``, it is non-negative), by the entropic OT cost approaching the
exact linear-assignment optimum as the regularisation shrinks, and by the
degenerate self-transport having ~zero cost. Differentiability and jit are
exercised, as is the LOG-semiring composition (implicitly, via correctness).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update('jax_enable_x64', True)

from nitrix.transport import (  # noqa: E402
    barycentric_map,
    sinkhorn,
    wasserstein_distance,
)


def _problem(n=6, m=8, seed=0):
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0, 1, n))
    y = np.sort(rng.uniform(0, 1, m))
    cost = jnp.asarray((x[:, None] - y[None, :]) ** 2)
    a = jnp.asarray(np.full(n, 1.0 / n))
    b = jnp.asarray(np.full(m, 1.0 / m))
    return cost, a, b, x, y


def test_plan_recovers_marginals():
    cost, a, b, *_ = _problem()
    res = sinkhorn(cost, a, b, epsilon=0.05, n_iter=500)
    np.testing.assert_allclose(np.asarray(res.plan.sum(1)), np.asarray(a), atol=1e-3)
    np.testing.assert_allclose(np.asarray(res.plan.sum(0)), np.asarray(b), atol=1e-3)


def test_plan_is_nonnegative():
    cost, a, b, *_ = _problem(seed=1)
    res = sinkhorn(cost, a, b, epsilon=0.1, n_iter=200)
    assert bool((res.plan >= 0).all())


def test_self_transport_has_near_zero_cost():
    rng = np.random.default_rng(2)
    z = np.sort(rng.uniform(0, 1, 7))
    cost = jnp.asarray((z[:, None] - z[None, :]) ** 2)
    u = jnp.asarray(np.full(7, 1.0 / 7))
    w = float(wasserstein_distance(cost, u, u, epsilon=0.005, n_iter=800))
    assert w < 5e-3  # only the small entropic bias remains


def test_approaches_exact_optimal_transport():
    opt = pytest.importorskip('scipy.optimize')
    rng = np.random.default_rng(3)
    xx = np.sort(rng.uniform(0, 1, 8))
    yy = np.sort(rng.uniform(0, 1, 8))
    cost_np = (xx[:, None] - yy[None, :]) ** 2
    ri, ci = opt.linear_sum_assignment(cost_np)
    exact = cost_np[ri, ci].sum() / 8  # equal masses -> assignment / n
    u = jnp.asarray(np.full(8, 1.0 / 8))
    approx = float(
        wasserstein_distance(
            jnp.asarray(cost_np), u, u, epsilon=0.002, n_iter=2000
        )
    )
    assert abs(approx - exact) / exact < 0.05


def test_wasserstein_composes_sinkhorn():
    cost, a, b, *_ = _problem(seed=4)
    res = sinkhorn(cost, a, b, epsilon=0.05, n_iter=200)
    manual = float(jnp.sum(res.plan * cost))
    w = float(wasserstein_distance(cost, a, b, epsilon=0.05, n_iter=200))
    np.testing.assert_allclose(w, manual, atol=1e-12)


def test_barycentric_map_pushes_into_target_hull():
    cost, a, b, x, y = _problem(seed=5)
    res = sinkhorn(cost, a, b, epsilon=0.02, n_iter=800)
    mapped = barycentric_map(res.plan, a, jnp.asarray(y)[:, None])
    assert mapped.shape == (x.shape[0], 1)
    # the image of each source point lies within the target range
    mp = np.asarray(mapped)[:, 0]
    assert mp.min() >= y.min() - 1e-6 and mp.max() <= y.max() + 1e-6
    # 1D monotone transport: the map is order-preserving
    assert bool(np.all(np.diff(mp) >= -1e-6))


def test_jit_and_grad():
    cost, a, b, *_ = _problem(seed=6)
    w = jax.jit(
        lambda c: wasserstein_distance(c, a, b, epsilon=0.05, n_iter=100)
    )(cost)
    assert bool(jnp.isfinite(w))
    g = jax.grad(
        lambda c: wasserstein_distance(c, a, b, epsilon=0.05, n_iter=100)
    )(cost)
    assert g.shape == cost.shape
    assert bool(jnp.all(jnp.isfinite(g)))
