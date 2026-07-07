# -*- coding: utf-8 -*-
"""Tests for compensated / precision-aware reductions (numerics.precision).

Each compensated reducer is checked in the regime where it wins over the naive
left fold, against an ``fp64`` ground truth: Kahan and pairwise on many-same-
magnitude accumulation drift, Neumaier on wildly-varying magnitudes (the
classic ``[1, 1e16, 1, -1e16]`` counterexample where Kahan itself fails).
``compensated_dot`` recovers an ill-conditioned inner product, and
``stochastic_round`` is verified unbiased with correct grid membership.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update('jax_enable_x64', True)

from nitrix.numerics.precision import (  # noqa: E402
    compensated_dot,
    kahan_sum,
    neumaier_sum,
    pairwise_sum,
    stochastic_round,
)

_SUMMERS = (kahan_sum, neumaier_sum, pairwise_sum)


def _drift_case():
    """1.0 plus many increments below ulp(1) in fp32 -> naive loses them."""
    x = jnp.concatenate(
        [jnp.ones(1, jnp.float32), jnp.full(200000, 1e-8, jnp.float32)]
    )
    truth = 1.0 + 200000 * 1e-8
    return x, truth


def test_well_conditioned_matches_jnp_sum():
    y = jnp.asarray(np.random.default_rng(1).standard_normal((4, 6, 3)))
    for summer in _SUMMERS:
        for axis in (0, 1, 2, -1):
            np.testing.assert_allclose(
                np.asarray(summer(y, axis=axis)),
                np.asarray(jnp.sum(y, axis=axis)),
                atol=1e-12,
            )


def test_kahan_recovers_accumulation_drift():
    x, truth = _drift_case()
    naive = abs(float(jnp.sum(x)) - truth) / truth
    kahan = abs(float(kahan_sum(x)) - truth) / truth
    assert kahan < 0.1 * naive  # ~29x better in practice


def test_pairwise_recovers_accumulation_drift():
    x, truth = _drift_case()
    naive = abs(float(jnp.sum(x)) - truth) / truth
    pw = abs(float(pairwise_sum(x)) - truth) / truth
    assert pw < naive


def test_neumaier_beats_kahan_on_wild_magnitudes():
    # Classic counterexample: Kahan gives 0, Neumaier gives the exact 2.0.
    x = jnp.asarray([1.0, 1e16, 1.0, -1e16], jnp.float32)
    neu = float(neumaier_sum(x))
    kah = float(kahan_sum(x))
    np.testing.assert_allclose(neu, 2.0, atol=1e-3)
    assert abs(neu - 2.0) < abs(kah - 2.0)


def test_keepdims_shapes():
    y = jnp.asarray(np.random.default_rng(2).standard_normal((4, 6, 3)))
    for summer in _SUMMERS:
        assert summer(y, axis=1, keepdims=True).shape == (4, 1, 3)
        assert summer(y, axis=-1, keepdims=True).shape == (4, 6, 1)
        assert summer(y, axis=0).shape == (6, 3)


def test_pairwise_block_size_invariance():
    """The tree grouping must not change the well-conditioned result."""
    y = jnp.asarray(np.random.default_rng(3).standard_normal(5000))
    ref = float(jnp.sum(y))
    for bs in (1, 7, 64, 1024, 8192):
        np.testing.assert_allclose(
            float(pairwise_sum(y, block_size=bs)), ref, atol=1e-11
        )


def test_compensated_dot_recovers_illconditioned():
    # Exact answer is 1.0; a naive left-fold in fp32 gives 0 (the +1 is lost
    # between +/-1e8). compensated_dot recovers it via the two-sum error term,
    # deterministically across backends (it is a sequential scan, not an
    # order-dependent tree reduction).
    a = jnp.asarray([1e8, 1.0, -1e8], jnp.float32)
    b = jnp.ones(3, jnp.float32)
    np.testing.assert_allclose(float(compensated_dot(a, b)), 1.0, atol=1e-5)


def test_compensated_dot_matches_on_well_conditioned():
    rng = np.random.default_rng(4)
    a = jnp.asarray(rng.standard_normal((3, 50)))
    b = jnp.asarray(rng.standard_normal((3, 50)))
    np.testing.assert_allclose(
        np.asarray(compensated_dot(a, b, axis=-1)),
        np.asarray(jnp.sum(a * b, axis=-1)),
        atol=1e-11,
    )


def test_compensated_dot_broadcasts():
    a = jnp.asarray(np.random.default_rng(5).standard_normal(5))
    b = jnp.asarray(np.random.default_rng(6).standard_normal((4, 5)))
    out = compensated_dot(a, b, axis=-1)
    assert out.shape == (4,)
    np.testing.assert_allclose(
        np.asarray(out), np.asarray(jnp.sum(a * b, axis=-1)), atol=1e-11
    )


def test_stochastic_round_unbiased():
    x = jnp.asarray([0.3, 1.7, -2.4], jnp.float32)
    keys = jax.random.split(jax.random.key(0), 20000)
    samples = jax.vmap(
        lambda k: stochastic_round(x, jnp.float16, key=k).astype(jnp.float32)
    )(keys)
    np.testing.assert_allclose(
        np.asarray(samples.mean(0)), np.asarray(x), atol=1e-3
    )


def test_stochastic_round_grid_membership():
    x = jnp.asarray([0.3, 1.7, -2.4], jnp.float32)
    keys = jax.random.split(jax.random.key(1), 4000)
    samples = jax.vmap(lambda k: stochastic_round(x, jnp.float16, key=k))(keys)
    assert samples.dtype == jnp.float16
    s = np.asarray(samples.astype(jnp.float32))
    for j in range(x.shape[0]):
        uniq = np.unique(s[:, j])
        assert uniq.size <= 2  # only the two bracketing grid points
        assert uniq.min() <= float(x[j]) <= uniq.max()


def test_stochastic_round_exact_for_representable():
    x = jnp.asarray([0.5, 1.0, -2.0, 0.25], jnp.float32)  # exact in fp16
    for i in range(8):
        out = stochastic_round(x, jnp.float16, key=jax.random.key(i))
        np.testing.assert_array_equal(
            np.asarray(out.astype(jnp.float32)), np.asarray(x)
        )


def test_gradients():
    # d(sum)/dx = 1 for every element (compensation is exact-in-value).
    for summer in _SUMMERS:
        g = jax.grad(lambda z: summer(z))(jnp.arange(5.0))
        np.testing.assert_allclose(np.asarray(g), np.ones(5), atol=1e-12)
    # d(a . b)/da = b.
    b = jnp.asarray([2.0, 3.0, 4.0])
    ga = jax.grad(lambda a: compensated_dot(a, b))(jnp.ones(3))
    np.testing.assert_allclose(np.asarray(ga), np.asarray(b), atol=1e-12)


def test_jit_clean():
    y = jnp.asarray(np.random.default_rng(7).standard_normal(256))
    for summer in _SUMMERS:
        v = jax.jit(summer)(y)
        np.testing.assert_allclose(float(v), float(jnp.sum(y)), atol=1e-11)
    d = jax.jit(compensated_dot)(y, y)
    np.testing.assert_allclose(float(d), float(jnp.sum(y * y)), atol=1e-10)
    r = jax.jit(lambda z, k: stochastic_round(z, jnp.float16, key=k))(
        y.astype(jnp.float32), jax.random.key(0)
    )
    assert bool(jnp.all(jnp.isfinite(r.astype(jnp.float32))))
