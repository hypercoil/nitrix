# -*- coding: utf-8 -*-
"""Tests for Gaussian location-scale regression (``gaulss_fit``, v3 §4).

``y ~ N(mu, sigma^2)`` with ``mu = X beta_mu`` and ``log sigma = Z beta_scale``
(heteroscedasticity modelled directly).  Anchored against a ``scipy.optimize``
maximum-likelihood reference (the same joint MLE): coefficients, log-likelihood,
and the block-information standard errors.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import minimize

jax.config.update('jax_enable_x64', True)

from nitrix.stats import GauLSSResult, gaulss_fit


def _sim(seed, n=400):
    rng = np.random.default_rng(seed)
    Xm = np.c_[np.ones(n), rng.standard_normal(n)]
    Xs = np.c_[np.ones(n), rng.standard_normal(n)]
    bmu, bsig = np.array([1.0, 0.7]), np.array([-0.3, 0.5])
    y = Xm @ bmu + rng.standard_normal(n) * np.exp(Xs @ bsig)
    return Xm, Xs, y


def _negll(p, Xm, Xs, y):
    es = Xs @ p[2:]
    return -np.sum(
        -0.5 * np.log(2 * np.pi) - es - 0.5 * (y - Xm @ p[:2]) ** 2 * np.exp(-2 * es)
    )


@pytest.mark.parametrize('seed', range(4))
def test_gaulss_matches_scipy_mle(seed):
    Xm, Xs, y = _sim(seed)
    res = gaulss_fit(
        jnp.asarray(y[None, :]), jnp.asarray(Xm), scale_design=jnp.asarray(Xs),
        n_iter=60,
    )
    r = minimize(_negll, np.zeros(4), args=(Xm, Xs, y), method='BFGS')
    np.testing.assert_allclose(np.asarray(res.coef_mu[0]), r.x[:2], atol=1e-4)
    np.testing.assert_allclose(np.asarray(res.coef_scale[0]), r.x[2:], atol=1e-4)
    assert abs(float(res.log_lik[0]) + r.fun) < 1e-3


def test_gaulss_se_matches_numerical_hessian():
    Xm, Xs, y = _sim(1)
    res = gaulss_fit(
        jnp.asarray(y[None, :]), jnp.asarray(Xm), scale_design=jnp.asarray(Xs),
        n_iter=60,
    )
    r = minimize(_negll, np.zeros(4), args=(Xm, Xs, y), method='BFGS')
    eps = 1e-5

    def perturbed(i, si, j, sj):
        pp = r.x.copy()
        pp[i] += si * eps
        pp[j] += sj * eps
        return _negll(pp, Xm, Xs, y)

    H = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            H[i, j] = (
                perturbed(i, 1, j, 1)
                - perturbed(i, 1, j, -1)
                - perturbed(i, -1, j, 1)
                + perturbed(i, -1, j, -1)
            ) / (4 * eps * eps)
    se_num = np.sqrt(np.diag(np.linalg.inv(H)))
    se = np.r_[
        np.sqrt(np.diag(np.asarray(res.cov_mu[0]))),
        np.sqrt(np.diag(np.asarray(res.cov_scale[0]))),
    ]
    np.testing.assert_allclose(se, se_num, rtol=0.05)


def test_gaulss_recovers_heteroscedasticity():
    """The scale slope (variance gradient) is recovered near the simulated 0.5."""
    Xm, Xs, y = _sim(2, n=1500)
    res = gaulss_fit(
        jnp.asarray(y[None, :]), jnp.asarray(Xm), scale_design=jnp.asarray(Xs),
        n_iter=60,
    )
    assert abs(float(res.coef_scale[0, 1]) - 0.5) < 0.12


def test_gaulss_default_scale_is_ols_mean():
    Xm, _, y = _sim(0)
    res = gaulss_fit(jnp.asarray(y[None, :]), jnp.asarray(Xm), n_iter=60)
    ols = np.linalg.solve(Xm.T @ Xm, Xm.T @ y)
    np.testing.assert_allclose(np.asarray(res.coef_mu[0]), ols, atol=1e-5)
    assert res.coef_scale.shape == (1, 1)  # intercept-only scale


def test_gaulss_shapes_and_pytree():
    Xm, Xs, y = _sim(3)
    V = 5
    res = gaulss_fit(
        jnp.asarray(np.tile(y, (V, 1))), jnp.asarray(Xm),
        scale_design=jnp.asarray(Xs),
    )
    assert res.coef_mu.shape == (V, 2)
    assert res.coef_scale.shape == (V, 2)
    assert res.cov_mu.shape == (V, 2, 2) and res.cov_scale.shape == (V, 2, 2)
    assert res.log_lik.shape == (V,) and res.n_obs == Xm.shape[0]
    leaves, treedef = jax.tree_util.tree_flatten(res)
    res2 = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(res2, GauLSSResult) and res2.n_obs == res.n_obs


def test_gaulss_shape_validation():
    Xm, Xs, y = _sim(0, n=100)
    with pytest.raises(ValueError, match='must match N'):
        gaulss_fit(jnp.asarray(y[None, :-3]), jnp.asarray(Xm))
    with pytest.raises(ValueError, match='expected N'):
        gaulss_fit(
            jnp.asarray(y[None, :]), jnp.asarray(Xm),
            scale_design=jnp.asarray(Xs[:-2]),
        )
