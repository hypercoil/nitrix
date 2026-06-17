# -*- coding: utf-8 -*-
"""Tests for ``nitrix.stats.connectivity`` (analytic shrinkage covariance)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update('jax_enable_x64', True)

from nitrix.stats.connectivity import ledoit_wolf, oas, shrunk_covariance


def _has_sklearn() -> bool:
    try:
        import sklearn.covariance  # noqa: F401

        return True
    except ImportError:
        return False


needs_sklearn = pytest.mark.skipif(
    not _has_sklearn(), reason='sklearn not installed'
)


@needs_sklearn
@pytest.mark.parametrize('n,p', [(50, 8), (30, 30), (20, 40)])
def test_ledoit_wolf_matches_sklearn(n, p):
    from sklearn.covariance import ledoit_wolf as sk

    X = np.random.default_rng(n * p).standard_normal((n, p))
    cov, shr = ledoit_wolf(jnp.asarray(X))
    sk_cov, sk_shr = sk(X)
    np.testing.assert_allclose(float(shr), sk_shr, atol=1e-12)
    np.testing.assert_allclose(np.asarray(cov), sk_cov, atol=1e-12)


@needs_sklearn
@pytest.mark.parametrize('n,p', [(50, 8), (30, 30), (20, 40)])
def test_oas_matches_sklearn(n, p):
    from sklearn.covariance import oas as sk

    X = np.random.default_rng(n * p + 1).standard_normal((n, p))
    cov, shr = oas(jnp.asarray(X))
    sk_cov, sk_shr = sk(X)
    np.testing.assert_allclose(float(shr), sk_shr, atol=1e-12)
    np.testing.assert_allclose(np.asarray(cov), sk_cov, atol=1e-12)


def test_shrunk_is_spd_in_small_sample():
    """Shrinkage keeps the covariance SPD / invertible even at p >= n, where
    the raw empirical covariance is singular."""
    rng = np.random.default_rng(0)
    X = jnp.asarray(rng.standard_normal((20, 40)))
    for method in ('ledoit_wolf', 'oas'):
        cov = shrunk_covariance(X, method=method)
        # SPD check on CPU (numpy) -- jnp.linalg.eigvalsh is cuSOLVER and fails
        # on the broken-stack GPU; the estimator itself is cuSOLVER-free.
        assert float(np.linalg.eigvalsh(np.asarray(cov)).min()) > 0
    with pytest.raises(ValueError):
        shrunk_covariance(X, method='nope')


def test_batched_and_differentiable():
    rng = np.random.default_rng(2)
    Xb = jnp.asarray(rng.standard_normal((5, 50, 8)))
    covs = jax.vmap(lambda x: ledoit_wolf(x)[0])(Xb)
    assert covs.shape == (5, 8, 8)
    g = jax.grad(lambda X: jnp.sum(ledoit_wolf(X)[0]))(
        jnp.asarray(rng.standard_normal((40, 6)))
    )
    assert bool(jnp.all(jnp.isfinite(g)))


def test_assume_centered():
    """assume_centered skips the mean subtraction (matches sklearn's flag)."""
    rng = np.random.default_rng(3)
    X = jnp.asarray(rng.standard_normal((40, 5)))
    Xc = X - jnp.mean(X, axis=0, keepdims=True)
    cov_a, _ = ledoit_wolf(Xc, assume_centered=True)
    cov_b, _ = ledoit_wolf(X, assume_centered=False)
    np.testing.assert_allclose(np.asarray(cov_a), np.asarray(cov_b), atol=1e-12)
