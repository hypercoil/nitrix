# -*- coding: utf-8 -*-
"""Tests for BrainSMASH variogram-matched spatial nulls (Burt 2020).

Covers ``variogram`` (rises for a smooth map, ~flat for a random one),
``brainsmash_surrogates`` (marginal-preserving under resample; reintroduces
spatial autocorrelation matched to the target), and ``brainsmash_test``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update('jax_enable_x64', True)

from nitrix.stats.inference import (  # noqa: E402
    SpatialNullResult,
    brainsmash_surrogates,
    brainsmash_test,
    spatial_null_test,
    variogram,
)


def _smooth_field(n=120, seed=0):
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0, 10, (n, 2))
    dist = jnp.asarray(
        np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    )
    x = (
        np.sin(pts[:, 0] * 0.6)
        + np.cos(pts[:, 1] * 0.5)
        + 0.1 * rng.standard_normal(n)
    )
    return jnp.asarray(x), dist


def test_variogram_rises_for_a_smooth_map():
    x, dist = _smooth_field()
    v = variogram(x, dist, n_bins=15)
    assert v.shape == (15,)
    assert float(v[-1]) > float(v[0])


def test_variogram_is_flat_for_a_random_map():
    _, dist = _smooth_field()
    x = jnp.asarray(np.random.default_rng(1).standard_normal(dist.shape[0]))
    v = np.asarray(variogram(x, dist, n_bins=15))
    # no spatial autocorrelation -> semivariance ~ constant (~ variance)
    assert v.std() < 0.35 * v.mean()


def test_brainsmash_resample_preserves_marginal():
    x, dist = _smooth_field()
    surr = brainsmash_surrogates(x, dist, 20, jax.random.key(0), n_bins=15)
    assert surr.shape == (20, x.shape[0])
    x_sorted = np.sort(np.asarray(x))
    for k in range(5):
        np.testing.assert_allclose(
            np.sort(np.asarray(surr[k])), x_sorted, atol=1e-9
        )


def test_brainsmash_reintroduces_spatial_autocorrelation():
    """Smoothed surrogates have a rising variogram, unlike a bare permutation."""
    x, dist = _smooth_field()
    surr = brainsmash_surrogates(x, dist, 30, jax.random.key(0), n_bins=15)
    v_surr = np.mean(
        [np.asarray(variogram(surr[k], dist, n_bins=15)) for k in range(30)], 0
    )
    assert v_surr[-1] > v_surr[0]  # rising = spatial autocorrelation present
    # short-range semivariance well below a bare (SA-free) permutation's level
    perm = jnp.asarray(np.random.default_rng(2).permutation(np.asarray(x)))
    perm_v = np.asarray(variogram(perm, dist, n_bins=15))
    assert v_surr[0] < 0.9 * perm_v.mean()


def test_brainsmash_no_resample_matches_moments():
    x, dist = _smooth_field()
    surr = brainsmash_surrogates(
        x, dist, 10, jax.random.key(0), n_bins=15, resample=False
    )
    np.testing.assert_allclose(
        np.asarray(surr).mean(1), float(x.mean()), atol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(surr).std(1), float(x.std()), rtol=1e-3
    )


def test_brainsmash_test_self_is_significant():
    x, dist = _smooth_field()
    res = brainsmash_test(
        x, x, dist, key=jax.random.key(1), n_surrogates=200, n_bins=15
    )
    assert np.isclose(float(res.statistic), 1.0)
    assert float(res.pvalue) < 0.05
    assert isinstance(res, SpatialNullResult)


def test_brainsmash_test_matches_manual_composition():
    x, dist = _smooth_field()
    y = jnp.asarray(np.random.default_rng(9).standard_normal(x.shape[0]))
    key = jax.random.key(5)
    res = brainsmash_test(x, y, dist, key=key, n_surrogates=80, n_bins=15)
    surr = brainsmash_surrogates(x, dist, 80, key, n_bins=15)
    manual = spatial_null_test(x, y, surr)
    np.testing.assert_allclose(
        res.null_distribution, manual.null_distribution, atol=1e-12
    )
    assert 1.0 / 81.0 <= float(res.pvalue) <= 1.0


def test_brainsmash_test_jit():
    """No eigh -> cuSolver-free -> jit/vmap-clean on any backend."""
    x, dist = _smooth_field()
    y = jnp.asarray(np.random.default_rng(9).standard_normal(x.shape[0]))
    p = jax.jit(
        lambda x, y, d: (
            brainsmash_test(
                x, y, d, key=jax.random.key(0), n_surrogates=100, n_bins=15
            ).pvalue
        )
    )(x, y, dist)
    assert bool(jnp.isfinite(p))
