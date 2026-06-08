# -*- coding: utf-8 -*-
"""Tests for ``nitrix.metrics`` overlap / loss numerics."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update('jax_enable_x64', True)

from nitrix.metrics import dice, jaccard

# ---------------------------------------------------------------------------
# dice / jaccard
# ---------------------------------------------------------------------------


def test_dice_perfect_overlap_is_one():
    x = jnp.asarray([1.0, 0.0, 1.0, 1.0, 0.0])
    assert abs(float(dice(x, x)) - 1.0) < 1e-6


def test_dice_disjoint_is_zero():
    p = jnp.asarray([1.0, 1.0, 0.0, 0.0])
    t = jnp.asarray([0.0, 0.0, 1.0, 1.0])
    assert float(dice(p, t)) < 1e-5


def test_dice_empty_vs_empty_is_one():
    z = jnp.zeros(8)
    # 0/0 -> smooth/smooth == 1 (vacuously perfect).
    assert abs(float(dice(z, z)) - 1.0) < 1e-6


def test_dice_half_overlap_known_value():
    p = jnp.asarray([1.0, 1.0, 0.0, 0.0])
    t = jnp.asarray([1.0, 0.0, 1.0, 0.0])
    # 2*1 / (2 + 2) = 0.5
    np.testing.assert_allclose(float(dice(p, t, smooth=0.0)), 0.5, atol=1e-7)


def test_jaccard_perfect_and_disjoint():
    x = jnp.asarray([1.0, 0.0, 1.0, 1.0])
    assert abs(float(jaccard(x, x)) - 1.0) < 1e-6
    p = jnp.asarray([1.0, 1.0, 0.0, 0.0])
    t = jnp.asarray([0.0, 0.0, 1.0, 1.0])
    assert float(jaccard(p, t)) < 1e-5


def test_jaccard_half_overlap_known_value():
    p = jnp.asarray([1.0, 1.0, 0.0, 0.0])
    t = jnp.asarray([1.0, 0.0, 1.0, 0.0])
    # intersection 1, union 3 -> 1/3
    np.testing.assert_allclose(
        float(jaccard(p, t, smooth=0.0)), 1.0 / 3.0, atol=1e-7
    )


def test_jaccard_dice_relationship():
    rng = np.random.default_rng(0)
    p = jnp.asarray(rng.random((4, 3, 16)))
    t = jnp.asarray((rng.random((4, 3, 16)) > 0.5).astype(float))
    d = dice(p, t, axis=-1, smooth=0.0, reduction='none')
    j = jaccard(p, t, axis=-1, smooth=0.0, reduction='none')
    # jaccard == dice / (2 - dice), elementwise.
    np.testing.assert_allclose(
        np.asarray(j), np.asarray(d / (2.0 - d)), atol=1e-9
    )


def test_dice_per_region_shape_and_reduction():
    rng = np.random.default_rng(1)
    p = jnp.asarray(rng.random((2, 4, 8, 8)))
    t = jnp.asarray((rng.random((2, 4, 8, 8)) > 0.5).astype(float))
    per = dice(p, t, axis=(-2, -1), reduction='none')
    assert per.shape == (2, 4)
    m = dice(p, t, axis=(-2, -1), reduction='mean')
    np.testing.assert_allclose(float(m), float(per.mean()), atol=1e-9)
    s = dice(p, t, axis=(-2, -1), reduction='sum')
    np.testing.assert_allclose(float(s), float(per.sum()), atol=1e-9)


def test_dice_multiclass_one_hot():
    # Argmax-consistent prediction onehot vs target onehot, per class.
    target = jnp.asarray([0, 1, 2, 1, 0])
    n_cls = 3
    t_oh = jax.nn.one_hot(target, n_cls).T  # (class, n)
    p_oh = t_oh  # perfect
    per = dice(p_oh, t_oh, axis=-1, reduction='none')
    np.testing.assert_allclose(np.asarray(per), 1.0, atol=1e-6)


def test_dice_jaccard_differentiable():
    rng = np.random.default_rng(2)
    logits = jnp.asarray(rng.standard_normal((3, 10)))
    t = jnp.asarray((rng.random((3, 10)) > 0.5).astype(float))

    def loss(z):
        p = jax.nn.sigmoid(z)
        return 1.0 - dice(p, t, axis=-1) + 1.0 - jaccard(p, t, axis=-1)

    g = jax.grad(loss)(logits)
    assert g.shape == logits.shape
    assert bool(jnp.all(jnp.isfinite(g)))
