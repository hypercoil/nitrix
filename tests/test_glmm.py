# -*- coding: utf-8 -*-
"""Tests for ``nitrix.stats.glmm`` (random-intercept GLMM via PQL).

Validation triangle, anchored on independent references:

- **few == many**: the two dispatch paths (dense ``gam_fit`` vs the structured
  Schur-complement solver) run the identical PQL iteration, so they must agree to
  the iterative tolerance -- the structured many-level solver is checked against
  the dense GAMM path bit-for-bit.
- **Gaussian == LME**: a Gaussian-family GLMM is the linear mixed model, so it
  must match ``reml_fit`` (itself validated against ``statsmodels`` ``MixedLM``).
- **dense numpy PQL**: an independent, plain-numpy penalised-IRLS + Fellner-Schall
  reference (different language/linear algebra) reproduces ``beta``, the BLUPs,
  and ``sigma_b^2`` for Poisson and binomial.

Plus parameter recovery vs simulation truth and the documented binary-PQL
attenuation (bias that shrinks as the per-cluster information grows).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update('jax_enable_x64', True)

from nitrix.stats import GLMMResult, glmm_fit
from nitrix.stats.lme import reml_fit

# ---------------------------------------------------------------------------
# Independent dense-numpy PQL reference (random intercept)
# ---------------------------------------------------------------------------


def _ref_pql(y, X, group, q, link, n_outer=20, n_inner=15):
    """Plain-numpy PQL: dense joint penalised IRLS + Fellner-Schall on lambda.

    Solves the full ``(p + q)`` penalised normal equations densely (the
    unambiguous reference the structured solver must reproduce).  Returns
    ``(beta, blups, sigma_b^2)``.
    """
    n, p = X.shape
    Z = np.eye(q)[group]
    D = np.hstack([X, Z])
    S = np.zeros((p + q, p + q))
    S[p:, p:] = np.eye(q)
    coef = np.zeros(p + q)
    lam = 1.0
    eye = 1e-8 * np.eye(p + q)

    def wz(eta):
        if link == 'poisson':
            mu = np.exp(eta)
            dmu = mu
            var = mu
        else:
            mu = 1.0 / (1.0 + np.exp(-eta))
            dmu = mu * (1.0 - mu)
            var = mu * (1.0 - mu)
        w = dmu * dmu / np.clip(var, 1e-10, None)
        z = eta + (y - mu) / np.clip(dmu, 1e-10, None)
        return w, z, mu

    for _ in range(n_outer):
        for _ in range(n_inner):
            w, z, _ = wz(D @ coef)
            a = (D * w[:, None]).T @ D + lam * S + eye
            coef = np.linalg.solve(a, (D * w[:, None]).T @ z)
        w, _, _ = wz(D @ coef)
        a = (D * w[:, None]).T @ D + lam * S + eye
        v = np.linalg.inv(a)
        b = coef[p:]
        num = max(q - lam * np.trace(v[p:, p:]), 1e-8)
        den = max(b @ b, 1e-12)
        lam = float(np.clip(num / den, 1e-6, 1e8))
    return coef[:p], coef[p:], 1.0 / lam


def _sim(link, *, seed, q, n_per, sb2=0.4, beta=(0.2, 0.6)):
    rng = np.random.default_rng(seed)
    group = np.repeat(np.arange(q), n_per).astype(np.int32)
    n = q * n_per
    X = np.ones((n, 2))
    X[:, 1] = rng.standard_normal(n)
    b = rng.standard_normal(q) * np.sqrt(sb2)
    eta = X @ np.asarray(beta) + b[group]
    if link == 'poisson':
        y = rng.poisson(np.exp(eta)).astype(float)
    else:
        y = rng.binomial(1, 1.0 / (1.0 + np.exp(-eta))).astype(float)
    return X, group, y, b


# ---------------------------------------------------------------------------
# few == many: structured solver matches the dense GAMM path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('link', ['poisson', 'binomial'])
def test_glmm_few_vs_many_paths_agree(link):
    """The dense (few-level) and structured (many-level) dispatch paths run the
    same PQL, so they must agree to the iterative tolerance."""
    X, group, y, _ = _sim(link, seed=1, q=16, n_per=15)
    Y = jnp.asarray(np.tile(y, (4, 1)))
    Xj, gj = jnp.asarray(X), jnp.asarray(group)
    few = glmm_fit(Y, Xj, group=gj, family=link, few_level_max=64)
    many = glmm_fit(Y, Xj, group=gj, family=link, few_level_max=4)
    assert few.tier == 'few' and many.tier == 'many'
    assert np.allclose(few.beta_hat, many.beta_hat, atol=1e-5)
    assert np.allclose(few.blups, many.blups, atol=1e-5)
    assert np.allclose(few.re_var, many.re_var, rtol=1e-4, atol=1e-6)
    assert np.allclose(few.deviance, many.deviance, rtol=1e-5)
    assert np.allclose(few.edf_total, many.edf_total, rtol=1e-5)


# ---------------------------------------------------------------------------
# Independent dense-numpy PQL reference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('link', ['poisson', 'binomial'])
@pytest.mark.parametrize('few_level_max', [64, 4])  # both dispatch paths
def test_glmm_matches_dense_numpy_pql(link, few_level_max):
    """Both paths reproduce an independent plain-numpy dense PQL reference."""
    X, group, y, _ = _sim(link, seed=7, q=15, n_per=20)
    rb, rblup, rrv = _ref_pql(y, X, group, 15, link)
    Y = jnp.asarray(y[None, :])
    res = glmm_fit(
        Y,
        jnp.asarray(X),
        group=jnp.asarray(group),
        family=link,
        few_level_max=few_level_max,
    )
    assert np.allclose(np.asarray(res.beta_hat[0]), rb, atol=1e-5)
    assert np.allclose(np.asarray(res.blups[0]), rblup, atol=1e-5)
    assert abs(float(res.re_var[0, 0, 0]) - rrv) < 1e-4


# ---------------------------------------------------------------------------
# Gaussian GLMM == LME (reml_fit, statsmodels-anchored)
# ---------------------------------------------------------------------------


def test_glmm_gaussian_reduces_to_reml():
    """A Gaussian-family GLMM is the linear mixed model: PQL must match the
    REML LME fit (``reml_fit``, validated against statsmodels MixedLM)."""
    rng = np.random.default_rng(3)
    n, q, p = 200, 10, 2
    group = np.repeat(np.arange(q), n // q).astype(np.int32)
    X = np.ones((n, p))
    X[:, 1] = rng.standard_normal(n)
    b = rng.standard_normal(q) * np.sqrt(0.6)
    y = (
        X @ np.array([1.0, 0.7])
        + b[group]
        + rng.standard_normal(n) * np.sqrt(0.4)
    )
    Y = jnp.asarray(np.tile(y, (3, 1)))
    Xj, gj = jnp.asarray(X), jnp.asarray(group)

    g = glmm_fit(Y, Xj, group=gj, family='gaussian')
    onehot = jax.nn.one_hot(gj, q)
    r = reml_fit(Y, Xj, onehot)
    assert np.allclose(g.beta_hat, r.beta_hat, rtol=1e-4, atol=1e-4)
    # D4: re_var is (V, 1, 1) for a scalar intercept; compare the (V,) diagonal.
    assert np.allclose(g.re_var[:, 0, 0], r.sigma_b_sq, rtol=1e-3, atol=1e-4)
    assert np.allclose(g.dispersion, r.sigma_e_sq, rtol=1e-3, atol=1e-4)


# ---------------------------------------------------------------------------
# Parameter recovery and PQL behaviour
# ---------------------------------------------------------------------------


def test_glmm_poisson_recovery():
    """Poisson GLMM recovers the fixed slope and the variance component."""
    X, group, y, _ = _sim(
        'poisson', seed=11, q=30, n_per=12, sb2=0.3, beta=(0.3, 0.6)
    )
    Y = jnp.asarray(y[None, :])
    g = glmm_fit(Y, jnp.asarray(X), group=jnp.asarray(group), family='poisson')
    assert abs(float(g.beta_hat[0, 1]) - 0.6) < 0.12
    assert abs(float(g.re_var[0, 0, 0]) - 0.3) < 0.2
    assert float(g.re_var[0, 0, 0]) > 0.0


def test_glmm_binomial_pql_attenuation_vanishes():
    """The documented binary-PQL slope attenuation shrinks as the per-cluster
    size (information) grows."""

    def slope(n_per):
        X, group, y, _ = _sim(
            'binomial', seed=0, q=40, n_per=n_per, sb2=0.5, beta=(0.0, 0.8)
        )
        Y = jnp.asarray(y[None, :])
        g = glmm_fit(
            Y, jnp.asarray(X), group=jnp.asarray(group), family='binomial'
        )
        return float(g.beta_hat[0, 1])

    s_small, s_large = slope(15), slope(120)
    # Small clusters clearly under-shoot the truth (0.8); the bias magnitude
    # shrinks toward zero as the per-cluster information grows (by which point
    # the near-unbiased estimate may overshoot slightly from sampling noise).
    assert s_small < 0.75
    assert abs(s_large - 0.8) < abs(s_small - 0.8)


# ---------------------------------------------------------------------------
# Dispatch, shapes, result pytree, validation
# ---------------------------------------------------------------------------


def test_glmm_dispatch_tier_threshold():
    """The level count vs ``few_level_max`` selects the tier; ``n_groups`` is
    read off the grouping factor."""
    X, group, y, _ = _sim('poisson', seed=2, q=20, n_per=10)
    Y = jnp.asarray(y[None, :])
    args = dict(group=jnp.asarray(group), family='poisson')
    assert glmm_fit(Y, jnp.asarray(X), few_level_max=64, **args).tier == 'few'
    assert glmm_fit(Y, jnp.asarray(X), few_level_max=10, **args).tier == 'many'
    assert glmm_fit(Y, jnp.asarray(X), **args).n_groups == 20


def test_glmm_result_shapes_and_pytree():
    X, group, y, _ = _sim('poisson', seed=5, q=12, n_per=10)
    V = 7
    Y = jnp.asarray(np.tile(y, (V, 1)))
    g = glmm_fit(Y, jnp.asarray(X), group=jnp.asarray(group), family='poisson')
    assert g.beta_hat.shape == (V, 2)
    assert g.blups.shape == (V, 12)
    # D4: re_var is the uniform (V, r, r) G shape; scalar intercept -> (V, 1, 1).
    assert g.re_var.shape == (V, 1, 1)
    assert g.dispersion.shape == (V,)
    # Fixed-dispersion family -> phi == 1.
    assert np.allclose(g.dispersion, 1.0)
    # Pytree round-trip.
    leaves, treedef = jax.tree_util.tree_flatten(g)
    g2 = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(g2, GLMMResult)
    assert g2.tier == g.tier and g2.n_groups == g.n_groups


def test_glmm_gamma_dispersion_estimated():
    """A non-fixed-dispersion family (Gamma) estimates phi (not pinned to 1)."""
    rng = np.random.default_rng(9)
    n, q = 240, 12
    group = np.repeat(np.arange(q), n // q).astype(np.int32)
    X = np.ones((n, 2))
    X[:, 1] = rng.standard_normal(n)
    b = rng.standard_normal(q) * 0.3
    mu = np.exp(X @ np.array([0.5, 0.4]) + b[group])
    y = rng.gamma(shape=4.0, scale=mu / 4.0)
    Y = jnp.asarray(y[None, :])
    g = glmm_fit(Y, jnp.asarray(X), group=jnp.asarray(group), family='gamma')
    assert float(g.dispersion[0]) > 0.0
    assert not np.isclose(float(g.dispersion[0]), 1.0)
    assert float(g.re_var[0, 0, 0]) > 0.0


def test_glmm_shape_validation():
    X, group, y, _ = _sim('poisson', seed=0, q=8, n_per=10)
    Y = jnp.asarray(y[None, :])
    with pytest.raises(ValueError, match='must match N'):
        glmm_fit(Y[:, :-3], jnp.asarray(X), group=jnp.asarray(group))
    with pytest.raises(ValueError, match='expected N'):
        glmm_fit(Y, jnp.asarray(X), group=jnp.asarray(group[:-2]))


# ---------------------------------------------------------------------------
# Laplace-approximate GLMM (method='laplace') -- the §11 follow-up to PQL
# ---------------------------------------------------------------------------


def _ref_laplace_binomial(y, X, group, q, n_mode=20):
    """numpy Laplace marginal NLL minimised by scipy -- the same estimator."""
    from scipy.optimize import minimize

    def expit(x):
        return 1.0 / (1.0 + np.exp(-x))

    p = X.shape[1]

    def nll(theta):
        beta, sb2 = theta[:p], np.exp(theta[p])
        etaf = X @ beta
        b = np.zeros(q)
        for _ in range(n_mode):
            eta = etaf + b[group]
            mu = expit(eta)
            w = mu * (1 - mu)
            sg = np.bincount(group, weights=(y - mu), minlength=q)
            sw = np.bincount(group, weights=w, minlength=q)
            b = b + (sg - b / sb2) / (sw + 1.0 / sb2)
        eta = etaf + b[group]
        mu = expit(eta)
        sw = np.bincount(group, weights=mu * (1 - mu), minlength=q)
        ll = np.bincount(
            group, weights=(y * eta - np.log1p(np.exp(eta))), minlength=q
        )
        return -np.sum(
            ll - b * b / (2 * sb2) - 0.5 * np.log(sb2) - 0.5 * np.log(sw + 1 / sb2)
        )

    r = minimize(
        nll, np.r_[np.zeros(p), 0.0], method='Nelder-Mead',
        options={'xatol': 1e-7, 'fatol': 1e-7, 'maxiter': 6000},
    )
    return r.x[:p], np.exp(r.x[p])


def _sim_binary(seed, q, n_per, sb2=0.5, slope=0.8):
    rng = np.random.default_rng(seed)
    group = np.repeat(np.arange(q), n_per).astype(np.int32)
    n = q * n_per
    X = np.ones((n, 2))
    X[:, 1] = rng.standard_normal(n)
    b = rng.standard_normal(q) * np.sqrt(sb2)
    eta = X @ np.array([0.0, slope]) + b[group]
    y = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-eta))).astype(float)
    return X, group, y


def test_glmm_laplace_matches_reference():
    """JAX Laplace fit matches the scipy-minimised numpy Laplace marginal."""
    X, group, y = _sim_binary(seed=1, q=40, n_per=12)
    rb, rsb2 = _ref_laplace_binomial(y, X, group, 40)
    res = glmm_fit(
        jnp.asarray(y[None, :]), jnp.asarray(X), group=jnp.asarray(group),
        family='binomial', method='laplace', n_outer=40,
    )
    assert res.tier == 'laplace'
    assert np.allclose(np.asarray(res.beta_hat[0]), rb, atol=5e-3)
    assert abs(float(res.re_var[0, 0, 0]) - rsb2) < 5e-3


def test_glmm_laplace_beats_pql_attenuation():
    """Laplace is less attenuated than PQL for small binary clusters (closer to
    the true slope)."""
    X, group, y = _sim_binary(seed=1, q=40, n_per=12, slope=0.8)
    Yj, Xj, gj = jnp.asarray(y[None, :]), jnp.asarray(X), jnp.asarray(group)
    lap = glmm_fit(Yj, Xj, group=gj, family='binomial', method='laplace', n_outer=40)
    pql = glmm_fit(Yj, Xj, group=gj, family='binomial', method='pql')
    lap_slope = float(lap.beta_hat[0, 1])
    pql_slope = float(pql.beta_hat[0, 1])
    assert pql_slope < lap_slope  # PQL more attenuated (smaller)
    assert abs(lap_slope - 0.8) < abs(pql_slope - 0.8)


def test_glmm_laplace_poisson_recovery():
    X, group, y, _ = _sim('poisson', seed=11, q=30, n_per=15, sb2=0.3, beta=(0.3, 0.6))
    res = glmm_fit(
        jnp.asarray(y[None, :]), jnp.asarray(X), group=jnp.asarray(group),
        family='poisson', method='laplace', n_outer=40,
    )
    assert res.tier == 'laplace'
    assert abs(float(res.beta_hat[0, 1]) - 0.6) < 0.12
    assert float(res.re_var[0, 0, 0]) > 0.0
    assert res.blups.shape == (1, 30)


def test_glmm_method_validation():
    X, group, y, _ = _sim('poisson', seed=0, q=8, n_per=10)
    with pytest.raises(ValueError, match='pql'):
        glmm_fit(
            jnp.asarray(y[None, :]), jnp.asarray(X), group=jnp.asarray(group),
            method='bogus',
        )
