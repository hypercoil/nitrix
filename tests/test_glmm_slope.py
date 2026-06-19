# -*- coding: utf-8 -*-
"""Tests for non-Gaussian random *slopes* in ``glmm_fit`` (z= / structure=).

Validation triangle:

- **Gaussian == LME**: a Gaussian-family random-slope GLMM is the linear mixed
  model, so both the diagonal (``(x || g)``) and unstructured (``(1 + x | g)``)
  fits must match ``lme_fit(z=, structure=)`` -- the exact block-Woodbury oracle
  (itself validated against statsmodels MixedLM).
- **r == 1 reduces to the scalar intercept**: an unstructured slope fit with a
  single ones column ``z`` is the random-intercept GLMM, so it must reproduce the
  shipped scalar-intercept path under Poisson / binomial.
- **diagonal data -> small off-diagonal**: an unstructured fit on data simulated
  with an uncorrelated ``G`` recovers a near-zero intercept-slope correlation.

Plus parameter recovery vs simulation truth, shapes / pytree, and validation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update('jax_enable_x64', True)

from nitrix.stats import GLMMResult, glmm_fit
from nitrix.stats.lme import lme_fit


def _sim_slope(link, *, seed, q, n_per, G, beta=(0.2, 0.6), noise=0.5):
    """Random intercept + slope GLMM: ``b_g ~ N(0, G)`` over ``z = [1, x]``."""
    rng = np.random.default_rng(seed)
    group = np.repeat(np.arange(q), n_per).astype(np.int32)
    n = q * n_per
    x = rng.standard_normal(n)
    X = np.c_[np.ones(n), x]
    z = np.c_[np.ones(n), x]
    b = rng.standard_normal((q, 2)) @ np.linalg.cholesky(np.asarray(G)).T
    eta = X @ np.asarray(beta) + np.einsum('nr,nr->n', z, b[group])
    if link == 'poisson':
        y = rng.poisson(np.exp(eta)).astype(float)
    elif link == 'binomial':
        y = rng.binomial(1, 1.0 / (1.0 + np.exp(-eta))).astype(float)
    else:
        y = eta + rng.standard_normal(n) * noise
    return X, z, group, y, b


# ---------------------------------------------------------------------------
# Gaussian == LME (the exact block-Woodbury oracle)
# ---------------------------------------------------------------------------


def test_gaussian_diagonal_slope_matches_lme():
    G = np.diag([0.5, 0.3])
    X, z, group, y, _ = _sim_slope('gaussian', seed=0, q=14, n_per=20, G=G)
    Y = jnp.asarray(np.tile(y, (3, 1)))
    Xj, zj, gj = jnp.asarray(X), jnp.asarray(z), jnp.asarray(group)
    g = glmm_fit(
        Y, Xj, group=gj, z=zj, structure='diagonal', family='gaussian'
    )
    r = lme_fit(Y, Xj, z=zj, group=gj, structure='diagonal')
    assert g.tier == 'few'
    assert np.allclose(g.beta_hat, r.beta_hat, rtol=1e-3, atol=1e-3)
    assert np.allclose(
        g.re_var, np.diagonal(np.asarray(r.cov_re), axis1=1, axis2=2),
        rtol=2e-2, atol=3e-3,
    )
    assert np.allclose(g.dispersion, r.sigma_e_sq, rtol=2e-2, atol=3e-3)


def test_gaussian_unstructured_slope_matches_lme():
    G = np.array([[0.6, 0.2], [0.2, 0.4]])
    X, z, group, y, _ = _sim_slope('gaussian', seed=1, q=16, n_per=18, G=G)
    Y = jnp.asarray(np.tile(y, (3, 1)))
    Xj, zj, gj = jnp.asarray(X), jnp.asarray(z), jnp.asarray(group)
    g = glmm_fit(
        Y, Xj, group=gj, z=zj, structure='unstructured', family='gaussian'
    )
    r = lme_fit(Y, Xj, z=zj, group=gj, structure='unstructured')
    assert g.tier == 'slope'
    assert np.allclose(g.beta_hat, r.beta_hat, rtol=1e-3, atol=1e-4)
    assert np.allclose(g.re_var, r.cov_re, rtol=1e-2, atol=1e-3)
    assert np.allclose(g.dispersion, r.sigma_e_sq, rtol=1e-2, atol=1e-3)


# ---------------------------------------------------------------------------
# r == 1 reduces to the scalar random-intercept GLMM (non-Gaussian)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('link', ['poisson', 'binomial'])
def test_unstructured_r1_matches_scalar_intercept(link):
    G = np.array([[0.4]])
    rng = np.random.default_rng(7)
    q, n_per = 18, 20
    group = np.repeat(np.arange(q), n_per).astype(np.int32)
    n = q * n_per
    X = np.c_[np.ones(n), rng.standard_normal(n)]
    b = rng.standard_normal(q) * np.sqrt(0.4)
    eta = X @ np.array([0.2, 0.5]) + b[group]
    if link == 'poisson':
        y = rng.poisson(np.exp(eta)).astype(float)
    else:
        y = rng.binomial(1, 1.0 / (1.0 + np.exp(-eta))).astype(float)
    Y = jnp.asarray(y[None, :])
    Xj, gj = jnp.asarray(X), jnp.asarray(group)
    z1 = jnp.ones((n, 1))
    slope = glmm_fit(
        Y, Xj, group=gj, z=z1, structure='unstructured', family=link,
        n_outer=30,
    )
    scalar = glmm_fit(Y, Xj, group=gj, family=link)
    assert slope.tier == 'slope'
    assert np.allclose(slope.beta_hat, scalar.beta_hat, rtol=1e-2, atol=3e-3)
    assert np.allclose(
        slope.re_var[:, 0, 0], scalar.re_var, rtol=3e-2, atol=5e-3
    )


# ---------------------------------------------------------------------------
# Unstructured fit on uncorrelated data recovers a small off-diagonal
# ---------------------------------------------------------------------------


def test_poisson_unstructured_on_diagonal_data():
    G = np.diag([0.4, 0.25])
    X, z, group, y, _ = _sim_slope(
        'poisson', seed=3, q=50, n_per=15, G=G, beta=(0.3, 0.5)
    )
    g = glmm_fit(
        jnp.asarray(y[None, :]), jnp.asarray(X), group=jnp.asarray(group),
        z=jnp.asarray(z), structure='unstructured', family='poisson',
        n_outer=30,
    )
    Ghat = np.asarray(g.re_var[0])
    corr = Ghat[0, 1] / np.sqrt(Ghat[0, 0] * Ghat[1, 1])
    assert abs(corr) < 0.35
    assert abs(Ghat[0, 0] - 0.4) < 0.25
    assert abs(Ghat[1, 1] - 0.25) < 0.2


# ---------------------------------------------------------------------------
# Parameter recovery (correlated slope, non-Gaussian)
# ---------------------------------------------------------------------------


def test_poisson_correlated_slope_recovery():
    G = np.array([[0.5, 0.25], [0.25, 0.4]])
    X, z, group, y, _ = _sim_slope(
        'poisson', seed=5, q=70, n_per=15, G=G, beta=(0.3, 0.6)
    )
    g = glmm_fit(
        jnp.asarray(y[None, :]), jnp.asarray(X), group=jnp.asarray(group),
        z=jnp.asarray(z), structure='unstructured', family='poisson',
        n_outer=40,
    )
    assert abs(float(g.beta_hat[0, 1]) - 0.6) < 0.15
    Ghat = np.asarray(g.re_var[0])
    assert abs(Ghat[0, 0] - 0.5) < 0.3
    assert abs(Ghat[1, 1] - 0.4) < 0.3
    assert Ghat[0, 1] > 0.0  # positive correlation recovered


# ---------------------------------------------------------------------------
# Shapes, pytree, validation
# ---------------------------------------------------------------------------


def test_slope_shapes_and_pytree():
    G = np.diag([0.4, 0.3])
    X, z, group, y, _ = _sim_slope('poisson', seed=2, q=10, n_per=12, G=G)
    V = 4
    Y = jnp.asarray(np.tile(y, (V, 1)))
    Xj, zj, gj = jnp.asarray(X), jnp.asarray(z), jnp.asarray(group)
    gd = glmm_fit(Y, Xj, group=gj, z=zj, structure='diagonal', family='poisson')
    assert gd.blups.shape == (V, 10, 2)
    assert gd.re_var.shape == (V, 2)
    gu = glmm_fit(
        Y, Xj, group=gj, z=zj, structure='unstructured', family='poisson'
    )
    assert gu.blups.shape == (V, 10, 2)
    assert gu.re_var.shape == (V, 2, 2)
    leaves, treedef = jax.tree_util.tree_flatten(gu)
    gu2 = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(gu2, GLMMResult) and gu2.tier == 'slope'


def test_slope_validation():
    G = np.diag([0.3, 0.3])
    X, z, group, y, _ = _sim_slope('poisson', seed=0, q=8, n_per=10, G=G)
    Y = jnp.asarray(y[None, :])
    Xj, gj = jnp.asarray(X), jnp.asarray(group)
    with pytest.raises(ValueError, match='z must be'):
        glmm_fit(Y, Xj, group=gj, z=jnp.asarray(z[:-3]), family='poisson')
    with pytest.raises(ValueError, match='structure'):
        glmm_fit(
            Y, Xj, group=gj, z=jnp.asarray(z), structure='bogus',
            family='poisson',
        )
    with pytest.raises(NotImplementedError, match='pql'):
        glmm_fit(
            Y, Xj, group=gj, z=jnp.asarray(z), method='laplace',
            family='poisson',
        )
