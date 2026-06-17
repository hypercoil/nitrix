# -*- coding: utf-8 -*-
"""Tests for ``nitrix.stats._batching.blocked_vmap`` (the chunked-vmap spine).

The load-bearing case is **``V`` not an exact multiple of ``block``** -- the
pad-to-multiple / reshape / trim path, where an off-by-one in the trim would
silently corrupt or drop elements.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update('jax_enable_x64', True)

from nitrix.stats._batching import blocked_vmap


def _f(x):
    # a multi-output fn so the tree_map flatten/trim is exercised on a pytree
    return jnp.sum(x), x * 2.0


@pytest.mark.parametrize('v', [23, 24, 25, 1, 7, 100])
@pytest.mark.parametrize('block', [1, 5, 8, 24, 64])
def test_blocked_matches_single_vmap_all_remainders(v, block):
    """block | v, block ∤ v, block == v, and block > v must all match the
    single vmap exactly and return exactly ``v`` rows."""
    rng = np.random.default_rng(v * 1000 + block)
    X = jnp.asarray(rng.standard_normal((v, 4)))
    ref_s, ref_x = jax.vmap(_f)(X)
    got_s, got_x = blocked_vmap(_f, (X,), block=block)
    assert got_s.shape == (v,) and got_x.shape == (v, 4)
    np.testing.assert_array_equal(np.asarray(got_s), np.asarray(ref_s))
    np.testing.assert_array_equal(np.asarray(got_x), np.asarray(ref_x))


def test_blocked_none_is_plain_vmap():
    rng = np.random.default_rng(0)
    X = jnp.asarray(rng.standard_normal((23, 3)))
    none_s, none_x = blocked_vmap(_f, (X,), block=None)
    ref_s, ref_x = jax.vmap(_f)(X)
    np.testing.assert_array_equal(np.asarray(none_s), np.asarray(ref_s))
    np.testing.assert_array_equal(np.asarray(none_x), np.asarray(ref_x))


def test_blocked_multiple_batched_args_non_multiple():
    """Multiple batched arrays, V=17 with block=5 (remainder 2): the padding
    replicates row 0 of *each* arg and the trim drops it, so the fn sees no
    cross-contamination between the real tail and the padding."""
    rng = np.random.default_rng(1)
    v = 17
    A = jnp.asarray(rng.standard_normal((v, 3)))
    B = jnp.asarray(rng.standard_normal((v, 3)))
    g = lambda a, b: jnp.sum(a * b)  # noqa: E731
    ref = jax.vmap(g)(A, B)
    got = blocked_vmap(g, (A, B), block=5)
    assert got.shape == (v,)
    np.testing.assert_allclose(np.asarray(got), np.asarray(ref), atol=1e-12)
