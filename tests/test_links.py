# -*- coding: utf-8 -*-
"""Tests for non-canonical links (``Family.with_link``), v3 follow-up.

The shared IRLS core is Fisher scoring (``w = mu_eta^2 / V``, ``z = eta +
(y - mu) / mu_eta``), which is link-general -- so a non-canonical link is just a
different ``(link, linkinv, mu_eta)`` triple composed onto the distribution via
``Family.with_link``.  Each non-canonical model is anchored against the matching
``statsmodels`` ``GLM(family, link=...)`` coefficient fit.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update('jax_enable_x64', True)

import statsmodels.api as sm
from scipy.stats import norm

from nitrix.stats import (
    BINOMIAL,
    GAMMA,
    POISSON,
    PROBIT_LINK,
    Link,
    glm_fit,
)
from nitrix.stats._family import resolve_link

_links = sm.families.links


def _fit(y, X, family):
    res = glm_fit(jnp.asarray(y[None, :]), jnp.asarray(X), family=family)
    return np.asarray(res.coef[0])


def _Xrng(seed, n=500, p=2):
    rng = np.random.default_rng(seed)
    X = np.c_[np.ones(n), rng.standard_normal((n, p))]
    return rng, X


def test_binomial_probit_matches_statsmodels():
    rng, X = _Xrng(0)
    eta = X @ np.array([0.2, 0.8, -0.5])
    y = rng.binomial(1, norm.cdf(eta)).astype(float)
    beta = _fit(y, X, BINOMIAL.with_link('probit'))
    sm_res = sm.GLM(
        y, X, family=sm.families.Binomial(link=_links.Probit())
    ).fit()
    np.testing.assert_allclose(beta, sm_res.params, rtol=1e-4, atol=1e-4)


def test_binomial_cloglog_matches_statsmodels():
    rng, X = _Xrng(1)
    eta = X @ np.array([-0.3, 0.6, 0.4])
    mu = -np.expm1(-np.exp(eta))  # cloglog inverse
    y = rng.binomial(1, mu).astype(float)
    beta = _fit(y, X, BINOMIAL.with_link('cloglog'))
    sm_res = sm.GLM(
        y, X, family=sm.families.Binomial(link=_links.CLogLog())
    ).fit()
    np.testing.assert_allclose(beta, sm_res.params, rtol=1e-3, atol=1e-3)


def test_poisson_sqrt_matches_statsmodels():
    rng, X = _Xrng(2)
    eta = X @ np.array([2.0, 0.4, -0.3])  # sqrt(mu) = eta -> mu = eta^2
    y = rng.poisson(eta**2).astype(float)
    beta = _fit(y, X, POISSON.with_link('sqrt'))
    sm_res = sm.GLM(
        y, X, family=sm.families.Poisson(link=_links.Sqrt())
    ).fit()
    np.testing.assert_allclose(beta, sm_res.params, rtol=1e-4, atol=1e-4)


def test_poisson_identity_matches_statsmodels():
    rng, X = _Xrng(3)
    mu = X @ np.array([5.0, 0.8, -0.6]) + 6.0  # keep mu > 0
    y = rng.poisson(np.clip(mu, 0.1, None)).astype(float)
    beta = _fit(y, X, POISSON.with_link('identity'))
    sm_res = sm.GLM(
        y, X, family=sm.families.Poisson(link=_links.Identity())
    ).fit()
    np.testing.assert_allclose(beta, sm_res.params, rtol=1e-3, atol=1e-3)


def test_gamma_inverse_matches_statsmodels():
    # Reciprocal is the *canonical* Gamma link (nitrix GAMMA defaults to log).
    rng, X = _Xrng(4)
    eta = X @ np.array([0.5, 0.15, 0.1]) + 1.0  # eta > 0 -> mu = 1/eta > 0
    mu = 1.0 / eta
    y = rng.gamma(shape=8.0, scale=mu / 8.0)
    beta = _fit(y, X, GAMMA.with_link('inverse'))
    sm_res = sm.GLM(
        y, X, family=sm.families.Gamma(link=_links.InversePower())
    ).fit()
    np.testing.assert_allclose(beta, sm_res.params, rtol=1e-3, atol=1e-3)


def test_gamma_log_matches_statsmodels():
    rng, X = _Xrng(5)
    mu = np.exp(X @ np.array([0.5, 0.3, -0.2]))
    y = rng.gamma(shape=8.0, scale=mu / 8.0)
    beta = _fit(y, X, GAMMA.with_link('log'))
    sm_res = sm.GLM(
        y, X, family=sm.families.Gamma(link=_links.Log())
    ).fit()
    np.testing.assert_allclose(beta, sm_res.params, rtol=1e-4, atol=1e-4)


def test_with_link_preserves_distribution_swaps_link():
    probit = BINOMIAL.with_link('probit')
    # distribution (variance / deviance / fixed dispersion) is unchanged ...
    assert probit.has_fixed_dispersion is BINOMIAL.has_fixed_dispersion
    assert probit.variance is BINOMIAL.variance
    assert probit.unit_deviance is BINOMIAL.unit_deviance
    # ... only the link triple swaps.
    assert probit.linkinv is PROBIT_LINK.linkinv
    assert probit.name == 'binomial[probit]'
    # accepts a Link instance too
    assert BINOMIAL.with_link(PROBIT_LINK).linkinv is PROBIT_LINK.linkinv


def test_glmm_inverse_link_runs_finite():
    """The non-canonical (inverse) link composes through the GLMM IRLS too --
    the sign-preserving working-response floor keeps the decreasing link finite.
    """
    from nitrix.stats import glmm_fit

    rng = np.random.default_rng(6)
    q, n_per = 12, 25
    group = np.repeat(np.arange(q), n_per).astype(np.int32)
    n = q * n_per
    X = np.c_[np.ones(n), rng.standard_normal(n)]
    b = rng.standard_normal(q) * 0.1
    eta = X @ np.array([0.4, 0.1]) + 1.0 + b[group]  # eta > 0
    y = rng.gamma(shape=8.0, scale=(1.0 / eta) / 8.0)
    g = glmm_fit(
        jnp.asarray(y[None, :]), jnp.asarray(X), group=jnp.asarray(group),
        family=GAMMA.with_link('inverse'),
    )
    assert bool(np.all(np.isfinite(np.asarray(g.beta_hat))))
    assert abs(float(g.beta_hat[0, 0]) - 1.4) < 0.4  # near the true intercept
    assert float(g.re_var[0]) >= 0.0


def test_resolve_link_validation():
    assert resolve_link('probit') is PROBIT_LINK
    custom = Link(
        name='c', link=lambda m: m, linkinv=lambda e: e, mu_eta=lambda e: e
    )
    assert resolve_link(custom) is custom
    with pytest.raises(ValueError, match='unknown link'):
        resolve_link('bogus')
