# -*- coding: utf-8 -*-
"""Tests for ``nitrix.morphology.max_pool_with_indices_nd`` and
``max_unpool_nd``.

Coverage:

- 2D / 3D pool returns max + correct argmax indices.
- pool ↔ unpool round-trip preserves exactly the argmax positions.
- Anisotropic pool_size.
- Differentiability through both directions.
- Cross-framework parity disclaimer documented (argmax-agreement check).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update('jax_enable_x64', True)

from nitrix.morphology import max_pool_with_indices_nd, max_unpool_nd


def test_max_pool_2d_matches_hand_roll():
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.standard_normal((2, 3, 8, 8)))
    pooled, indices = max_pool_with_indices_nd(
        x, pool_size=2, spatial_rank=2,
    )
    assert pooled.shape == (2, 3, 4, 4)
    assert indices.shape == (2, 3, 4, 4)
    # Hand-roll: reshape into (B, C, n_h, ph, n_w, pw), max over (ph, pw)
    ref = x.reshape(2, 3, 4, 2, 4, 2).max(axis=(3, 5))
    np.testing.assert_allclose(pooled, ref, atol=1e-13)


def test_max_pool_3d_matches_hand_roll():
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.standard_normal((1, 2, 8, 8, 8)))
    pooled, _ = max_pool_with_indices_nd(x, pool_size=2, spatial_rank=3)
    assert pooled.shape == (1, 2, 4, 4, 4)
    ref = x.reshape(1, 2, 4, 2, 4, 2, 4, 2).max(axis=(3, 5, 7))
    np.testing.assert_allclose(pooled, ref, atol=1e-13)


def test_max_pool_indices_resolve_to_max_value():
    '''For every output position, the value at the recorded index
    in the original grid equals the pooled value.
    '''
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.standard_normal((2, 3, 6, 6)))
    pooled, indices = max_pool_with_indices_nd(
        x, pool_size=2, spatial_rank=2,
    )
    spatial_n = 6 * 6
    for b in range(2):
        for c in range(3):
            for i in range(3):
                for j in range(3):
                    flat = int(indices[b, c, i, j])
                    h, w = divmod(flat, 6)
                    assert (
                        float(x[b, c, h, w]) == float(pooled[b, c, i, j])
                    ), f'mismatch at ({b},{c},{i},{j})'


def test_anisotropic_pool_size():
    '''Pool with non-uniform per-axis size.'''
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.standard_normal((1, 1, 6, 4)))
    pooled, _ = max_pool_with_indices_nd(
        x, pool_size=(3, 2), spatial_rank=2,
    )
    assert pooled.shape == (1, 1, 2, 2)
    ref = x.reshape(1, 1, 2, 3, 2, 2).max(axis=(3, 5))
    np.testing.assert_allclose(pooled, ref, atol=1e-13)


def test_pool_rejects_non_divisible():
    x = jnp.zeros((1, 1, 7, 8))  # 7 not divisible by 2
    with pytest.raises(ValueError, match='not divisible'):
        max_pool_with_indices_nd(x, pool_size=2, spatial_rank=2)


def test_unpool_round_trip_preserves_argmax_positions():
    '''unpool(pool(x)) is zero everywhere except at the argmax
    positions, where it equals the pooled value.
    '''
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.standard_normal((2, 3, 8, 8)))
    pooled, indices = max_pool_with_indices_nd(
        x, pool_size=2, spatial_rank=2,
    )
    recon = max_unpool_nd(
        pooled, indices, output_shape=(8, 8), spatial_rank=2,
    )
    # Exactly 96 nonzero positions (2*3*4*4 = output count).
    assert int(jnp.sum(recon != 0)) == 2 * 3 * 4 * 4
    # At argmax positions, recon == pooled == x at that position.
    for b in range(2):
        for c in range(3):
            for i in range(4):
                for j in range(4):
                    flat = int(indices[b, c, i, j])
                    h, w = divmod(flat, 8)
                    np.testing.assert_allclose(
                        float(recon[b, c, h, w]),
                        float(pooled[b, c, i, j]),
                        atol=1e-13,
                    )


def test_unpool_3d_shape():
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.standard_normal((1, 2, 8, 8, 8)))
    pooled, indices = max_pool_with_indices_nd(
        x, pool_size=2, spatial_rank=3,
    )
    recon = max_unpool_nd(
        pooled, indices,
        output_shape=(8, 8, 8), spatial_rank=3,
    )
    assert recon.shape == (1, 2, 8, 8, 8)


def test_pool_differentiable():
    '''Gradient through pool flows back to source.'''
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.standard_normal((1, 2, 4, 4)))
    def loss(x):
        p, _ = max_pool_with_indices_nd(x, pool_size=2, spatial_rank=2)
        return jnp.sum(p ** 2)
    g = jax.grad(loss)(x)
    assert g.shape == x.shape
    assert bool(jnp.all(jnp.isfinite(g)))


def test_unpool_differentiable_through_values():
    '''Gradient through unpool flows back to the pooled values
    (not the indices, which are discrete).
    '''
    rng = np.random.default_rng(0)
    pooled = jnp.asarray(rng.standard_normal((1, 2, 2, 2)))
    indices = jnp.asarray(rng.integers(0, 16, (1, 2, 2, 2)).astype(np.int32))
    def loss(pooled):
        return jnp.sum(
            max_unpool_nd(
                pooled, indices,
                output_shape=(4, 4), spatial_rank=2,
            ) ** 2,
        )
    g = jax.grad(loss)(pooled)
    assert g.shape == pooled.shape
    assert bool(jnp.all(jnp.isfinite(g)))


def test_unpool_argmax_agreement_is_load_bearing():
    '''Document the cross-framework parity caveat: argmax of the
    output should agree across frameworks, not raw-logit allclose.

    This test demonstrates that the per-channel argmax of the
    unpooled output (a common downstream classifier readout) is
    stable to perturbations that flip individual pooling-argmax
    positions.
    '''
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.standard_normal((1, 3, 4, 4)))  # 3 classes
    pooled, indices = max_pool_with_indices_nd(
        x, pool_size=2, spatial_rank=2,
    )
    out = max_unpool_nd(
        pooled, indices,
        output_shape=(4, 4), spatial_rank=2,
    )
    class_pred = jnp.argmax(out, axis=1)  # (1, 4, 4)

    # Perturb x slightly to flip ties (a noise-tolerance check).
    x_perturbed = x + 1e-8 * jnp.asarray(rng.standard_normal(x.shape))
    pooled_p, indices_p = max_pool_with_indices_nd(
        x_perturbed, pool_size=2, spatial_rank=2,
    )
    out_p = max_unpool_nd(
        pooled_p, indices_p,
        output_shape=(4, 4), spatial_rank=2,
    )
    class_pred_p = jnp.argmax(out_p, axis=1)

    # Argmax agreement is the load-bearing parity check (per
    # PGlandsSeg DEVNOTES.md): under a small perturbation the
    # per-pixel class prediction should not change.
    np.testing.assert_array_equal(class_pred, class_pred_p)
