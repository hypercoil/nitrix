# -*- coding: utf-8 -*-
"""Tests for the spin-test spatial null.

Covers the ``geometry`` generators (uniform-SO(3) ``random_rotation``;
``spin_surrogates`` rotate-and-reassign) and the ``stats.inference.spin_test``
(observed Pearson correlation, rotated null, add-one permutation p-value).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update('jax_enable_x64', True)

from nitrix.geometry import random_rotation, spin_surrogates  # noqa: E402
from nitrix.stats.inference import spin_test  # noqa: E402


def _sphere(v=200, seed=2):
    coords = jax.random.normal(jax.random.key(seed), (v, 3))
    return coords / jnp.linalg.norm(coords, axis=-1, keepdims=True)


def test_random_rotation_is_proper_orthogonal():
    R = random_rotation(jax.random.key(0), 200)
    ident = jnp.einsum('nij,nkj->nik', R, R)
    assert float(jnp.max(jnp.abs(ident - jnp.eye(3)))) < 1e-10
    np.testing.assert_allclose(jnp.linalg.det(R), 1.0, atol=1e-9)


def test_random_rotation_is_uniform_on_the_group():
    """The Haar mean rotation is the zero matrix (uniform, not Euler-biased)."""
    mean = random_rotation(jax.random.key(1), 40000).mean(0)
    assert float(jnp.max(jnp.abs(mean))) < 0.05


def test_random_rotation_single_shape():
    assert random_rotation(jax.random.key(0)).shape == (3, 3)


def test_spin_surrogates_identity_is_noop():
    coords = _sphere()
    x = jax.random.normal(jax.random.key(3), (coords.shape[0],))
    surr = spin_surrogates(coords, x, jnp.eye(3)[None])
    np.testing.assert_allclose(surr[0], x, atol=1e-10)


def test_spin_surrogates_are_a_resampling_of_x():
    coords = _sphere()
    x = jax.random.normal(jax.random.key(3), (coords.shape[0],))
    surr = spin_surrogates(coords, x, random_rotation(jax.random.key(0), 10))
    assert surr.shape == (10, coords.shape[0])
    assert np.all(
        np.isin(np.asarray(surr).round(9), np.asarray(x).round(9))
    )


def test_spin_test_self_correspondence_is_significant():
    """A map vs itself: r == 1, and spun surrogates decorrelate, so p is tiny."""
    coords = _sphere()
    x = jax.random.normal(jax.random.key(3), (coords.shape[0],))
    res = spin_test(x, x, coords, key=jax.random.key(0), n_spin=500)
    assert float(res.statistic) == 1.0 or np.isclose(float(res.statistic), 1.0)
    assert float(res.pvalue) < 0.05
    assert res.null_distribution.shape == (500,)


def test_spin_test_pvalue_in_valid_range_and_stat_is_pearson():
    coords = _sphere()
    x = jax.random.normal(jax.random.key(3), (coords.shape[0],))
    y = jax.random.normal(jax.random.key(7), (coords.shape[0],))
    res = spin_test(x, y, coords, key=jax.random.key(0), n_spin=500)
    manual_r = np.corrcoef(np.asarray(x), np.asarray(y))[0, 1]
    np.testing.assert_allclose(float(res.statistic), manual_r, atol=1e-10)
    assert 1.0 / 501.0 <= float(res.pvalue) <= 1.0


def test_spin_test_matches_manual_composition():
    """spin_test's null is exactly pearson over the spin surrogates."""
    coords = _sphere()
    x = jax.random.normal(jax.random.key(3), (coords.shape[0],))
    y = jax.random.normal(jax.random.key(7), (coords.shape[0],))
    key = jax.random.key(5)
    res = spin_test(x, y, coords, key=key, n_spin=100)
    xs = spin_surrogates(coords, x, random_rotation(key, 100))
    xm = xs - xs.mean(-1, keepdims=True)
    ym = y - y.mean()
    null = (xm * ym).sum(-1) / jnp.sqrt(
        (xm * xm).sum(-1) * (ym * ym).sum()
    )
    np.testing.assert_allclose(res.null_distribution, null, atol=1e-12)


def test_spin_test_jit():
    coords = _sphere()
    x = jax.random.normal(jax.random.key(3), (coords.shape[0],))
    y = jax.random.normal(jax.random.key(7), (coords.shape[0],))
    p = jax.jit(
        lambda x, y, c: spin_test(x, y, c, key=jax.random.key(0), n_spin=200)
        .pvalue
    )(x, y, coords)
    assert bool(jnp.isfinite(p))
