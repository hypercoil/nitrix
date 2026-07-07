# -*- coding: utf-8 -*-
"""Tests for Moran spectral-randomization spatial nulls and the shared
``spatial_null_test`` seam.

Covers ``graph.moran_surrogates`` (mean/variance-preserving, spatial-
autocorrelation-matched sign-flip surrogates), ``stats.inference.moran_test``,
and the generator-agnostic ``spatial_null_test`` p-value.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update('jax_enable_x64', True)

from nitrix.graph import moran_surrogates  # noqa: E402
from nitrix.stats.inference import (  # noqa: E402
    SpatialNullResult,
    moran_test,
    spatial_null_test,
)


def _graph_and_signal(npc=15, seed=0):
    rng = np.random.default_rng(seed)
    n = 2 * npc
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            p = 0.8 if (i // npc) == (j // npc) else 0.05
            if rng.random() < p:
                A[i, j] = A[j, i] = 1.0
    x = np.concatenate([np.ones(npc), -np.ones(npc)])
    x = x + 0.2 * rng.standard_normal(n)
    return jnp.asarray(A), jnp.asarray(x)


def test_moran_surrogates_preserve_mean_and_variance():
    """Sign-flipping the Moran coefficients preserves the SA power spectrum."""
    A, x = _graph_and_signal()
    surr = moran_surrogates(A, x, 200, jax.random.key(0))
    assert surr.shape == (200, x.shape[0])
    np.testing.assert_allclose(
        np.asarray(surr).mean(1), float(x.mean()), atol=1e-9
    )
    np.testing.assert_allclose(
        np.asarray(surr).var(1), float(x.var()), atol=1e-8
    )


def test_moran_surrogates_differ_from_x():
    A, x = _graph_and_signal()
    surr = moran_surrogates(A, x, 50, jax.random.key(0))
    assert float(jnp.mean(jnp.abs(surr - x[None, :]))) > 0.1


def test_moran_test_self_correspondence_is_significant():
    A, x = _graph_and_signal()
    res = moran_test(x, x, A, key=jax.random.key(1), n_surrogates=500)
    assert np.isclose(float(res.statistic), 1.0)
    assert float(res.pvalue) < 0.05
    assert isinstance(res, SpatialNullResult)


def test_moran_test_pvalue_range_and_stat_is_pearson():
    A, x = _graph_and_signal()
    y = jnp.asarray(np.random.default_rng(9).standard_normal(x.shape[0]))
    res = moran_test(x, y, A, key=jax.random.key(1), n_surrogates=400)
    manual_r = np.corrcoef(np.asarray(x), np.asarray(y))[0, 1]
    np.testing.assert_allclose(float(res.statistic), manual_r, atol=1e-10)
    assert 1.0 / 401.0 <= float(res.pvalue) <= 1.0
    assert res.null_distribution.shape == (400,)


def test_moran_test_matches_manual_composition():
    """moran_test == spatial_null_test over moran_surrogates."""
    A, x = _graph_and_signal()
    y = jnp.asarray(np.random.default_rng(9).standard_normal(x.shape[0]))
    key = jax.random.key(5)
    res = moran_test(x, y, A, key=key, n_surrogates=100)
    manual = spatial_null_test(x, y, moran_surrogates(A, x, 100, key))
    np.testing.assert_allclose(
        res.null_distribution, manual.null_distribution, atol=1e-12
    )
    assert float(res.pvalue) == float(manual.pvalue)


def test_spatial_null_test_mechanics():
    """The shared seam: observed == Pearson, p counts the extreme surrogates."""
    v = 40
    x = jnp.asarray(np.random.default_rng(0).standard_normal(v))
    y = jnp.asarray(np.random.default_rng(1).standard_normal(v))
    extra = np.random.default_rng(2).standard_normal((4, v))
    surrogates = jnp.asarray(np.vstack([np.asarray(y), extra]))  # first == y
    res = spatial_null_test(x, y, surrogates)
    np.testing.assert_allclose(
        float(res.statistic),
        np.corrcoef(np.asarray(x), np.asarray(y))[0, 1],
        atol=1e-10,
    )
    # the y-surrogate has |corr| == 1 >= |observed|, so at least one extreme
    assert float(res.pvalue) >= 2.0 / 6.0
    assert res.null_distribution.shape == (5,)
