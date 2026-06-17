# -*- coding: utf-8 -*-
"""Tests for ``nitrix.stats.glm`` (mass-univariate GLM).

Correctness is anchored against ``statsmodels`` (OLS / WLS / GLM) -- the same
reference ModelArray's ``lm`` ultimately tracks -- and against ``scipy.stats``
for the t / F p-values.  Plus the mass-univariate property (batched == looped),
prediction, and the cuSOLVER-free HLO guard.
"""

from __future__ import annotations

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update('jax_enable_x64', True)

from nitrix.stats.glm import (
    BINOMIAL,
    GAUSSIAN,
    POISSON,
    adj_r_squared,
    aic,
    bic,
    compare_models,
    f_contrast,
    glm_fit,
    log_likelihood,
    predict,
    r_squared,
    t_contrast,
)


def _has(mod: str) -> bool:
    try:
        __import__(mod)
        return True
    except ImportError:
        return False


needs_sm = pytest.mark.skipif(
    not _has('statsmodels'), reason='statsmodels not installed'
)
needs_scipy = pytest.mark.skipif(
    not _has('scipy'), reason='scipy not installed'
)


def _design(rng, N=80, p=3):
    cols = [np.ones(N)] + [rng.standard_normal(N) for _ in range(p - 1)]
    return np.column_stack(cols)


# ---------------------------------------------------------------------------
# OLS vs statsmodels
# ---------------------------------------------------------------------------


@needs_sm
def test_ols_matches_statsmodels():
    import statsmodels.api as sm

    rng = np.random.default_rng(0)
    X = _design(rng)
    y = X @ np.array([1.0, 0.5, -0.3]) + rng.standard_normal(80) * 0.4
    res = glm_fit(jnp.asarray(y[None, :]), jnp.asarray(X), family=GAUSSIAN)
    smf = sm.OLS(y, X).fit()

    np.testing.assert_allclose(np.asarray(res.coef[0]), smf.params, atol=1e-10)
    for j in range(3):
        eff, se, t, pv = t_contrast(res, jnp.eye(3)[j])
        np.testing.assert_allclose(float(eff[0]), smf.params[j], atol=1e-10)
        np.testing.assert_allclose(float(se[0]), smf.bse[j], atol=1e-10)
        np.testing.assert_allclose(float(t[0]), smf.tvalues[j], atol=1e-8)
        np.testing.assert_allclose(float(pv[0]), smf.pvalues[j], atol=1e-10)
    np.testing.assert_allclose(
        float(r_squared(res)[0]), smf.rsquared, atol=1e-10
    )
    np.testing.assert_allclose(
        float(adj_r_squared(res)[0]), smf.rsquared_adj, atol=1e-10
    )


@needs_sm
def test_ols_f_contrast_matches_statsmodels():
    import statsmodels.api as sm

    rng = np.random.default_rng(1)
    X = _design(rng)
    y = X @ np.array([1.0, 0.5, -0.3]) + rng.standard_normal(80) * 0.5
    res = glm_fit(jnp.asarray(y[None, :]), jnp.asarray(X))
    smf = sm.OLS(y, X).fit()
    C = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    F, Fp, df1, df2 = f_contrast(res, jnp.asarray(C))
    ftest = smf.f_test(C)
    np.testing.assert_allclose(float(F[0]), float(ftest.fvalue), atol=1e-8)
    np.testing.assert_allclose(float(Fp[0]), float(ftest.pvalue), atol=1e-10)
    assert df1 == 2.0 and df2 == 77.0


@needs_sm
def test_wls_matches_statsmodels():
    import statsmodels.api as sm

    rng = np.random.default_rng(2)
    X = _design(rng)
    y = X @ np.array([0.5, 1.0, 0.2]) + rng.standard_normal(80) * 0.4
    w = np.abs(rng.standard_normal(80)) + 0.2
    res = glm_fit(
        jnp.asarray(y[None, :]), jnp.asarray(X), weights=jnp.asarray(w)
    )
    smf = sm.WLS(y, X, weights=w).fit()
    np.testing.assert_allclose(np.asarray(res.coef[0]), smf.params, atol=1e-10)
    _, se, _, _ = t_contrast(res, jnp.eye(3)[1])
    np.testing.assert_allclose(float(se[0]), smf.bse[1], atol=1e-9)


# ---------------------------------------------------------------------------
# Exponential family (IRLS) vs statsmodels.GLM
# ---------------------------------------------------------------------------


@needs_sm
def test_poisson_matches_statsmodels():
    import statsmodels.api as sm

    rng = np.random.default_rng(3)
    X = _design(rng)
    mu = np.exp(X @ np.array([0.5, 0.3, -0.2]))
    y = rng.poisson(mu).astype(float)
    res = glm_fit(jnp.asarray(y[None, :]), jnp.asarray(X), family=POISSON, n_iter=30)
    smf = sm.GLM(y, X, family=sm.families.Poisson()).fit()
    np.testing.assert_allclose(np.asarray(res.coef[0]), smf.params, atol=1e-7)
    _, se, _, _ = t_contrast(res, jnp.eye(3)[1])
    np.testing.assert_allclose(float(se[0]), smf.bse[1], atol=1e-6)


@needs_sm
def test_binomial_matches_statsmodels():
    import statsmodels.api as sm

    rng = np.random.default_rng(4)
    X = _design(rng)
    pr = 1.0 / (1.0 + np.exp(-(X @ np.array([0.2, 0.8, -0.5]))))
    y = (rng.random(80) < pr).astype(float)
    res = glm_fit(
        jnp.asarray(y[None, :]), jnp.asarray(X), family=BINOMIAL, n_iter=30
    )
    smf = sm.GLM(y, X, family=sm.families.Binomial()).fit()
    np.testing.assert_allclose(np.asarray(res.coef[0]), smf.params, atol=1e-7)
    _, se, _, _ = t_contrast(res, jnp.eye(3)[1])
    np.testing.assert_allclose(float(se[0]), smf.bse[1], atol=1e-6)


# ---------------------------------------------------------------------------
# Mass-univariate behaviour
# ---------------------------------------------------------------------------


def test_mass_univariate_batched_matches_looped():
    """V batched fits equal V independent single-element fits (OLS + Poisson)."""
    rng = np.random.default_rng(5)
    X = jnp.asarray(_design(rng))
    Y = jnp.asarray(rng.standard_normal((16, 80)))
    res = glm_fit(Y, X)
    for v in range(16):
        rv = glm_fit(Y[v : v + 1], X)
        np.testing.assert_allclose(
            np.asarray(res.coef[v]), np.asarray(rv.coef[0]), atol=1e-10
        )

    Yp = jnp.asarray(rng.poisson(2.0, size=(10, 80)).astype(float))
    rp = glm_fit(Yp, X, family=POISSON, n_iter=25)
    for v in range(10):
        rpv = glm_fit(Yp[v : v + 1], X, family=POISSON, n_iter=25)
        np.testing.assert_allclose(
            np.asarray(rp.coef[v]), np.asarray(rpv.coef[0]), atol=1e-9
        )


def test_predict_link_and_response():
    rng = np.random.default_rng(6)
    X = jnp.asarray(_design(rng))
    Y = jnp.asarray(rng.poisson(2.0, size=(4, 80)).astype(float))
    res = glm_fit(Y, X, family=POISSON, n_iter=25)
    eta = predict(res, X, type='link')
    mu = predict(res, X, type='response')
    np.testing.assert_allclose(np.asarray(mu), np.asarray(jnp.exp(eta)), atol=1e-10)
    assert eta.shape == (4, 80)


# ---------------------------------------------------------------------------
# Information criteria and model comparison
# ---------------------------------------------------------------------------


@needs_sm
def test_loglik_aic_bic():
    """log-likelihood matches statsmodels exactly; AIC/BIC follow the R/lm
    convention (the variance is a counted parameter for Gaussian)."""
    import statsmodels.api as sm

    rng = np.random.default_rng(8)
    X = _design(rng)
    y = X @ np.array([1.0, 0.5, -0.3]) + rng.standard_normal(80) * 0.4
    res = glm_fit(jnp.asarray(y[None, :]), jnp.asarray(X))
    smf = sm.OLS(y, X).fit()
    np.testing.assert_allclose(float(log_likelihood(res)[0]), smf.llf, atol=1e-8)
    # R lm convention: k = p + 1 (counts sigma^2).
    ll = float(log_likelihood(res)[0])
    np.testing.assert_allclose(float(aic(res)[0]), -2 * ll + 2 * 4, atol=1e-8)
    np.testing.assert_allclose(
        float(bic(res)[0]), -2 * ll + 4 * np.log(80), atol=1e-8
    )

    # Poisson has fixed dispersion (k = p), so AIC matches statsmodels.GLM.
    mu = np.exp(X @ np.array([0.5, 0.3, -0.2]))
    yp = rng.poisson(mu).astype(float)
    rp = glm_fit(jnp.asarray(yp[None, :]), jnp.asarray(X), family=POISSON, n_iter=30)
    smp = sm.GLM(yp, X, family=sm.families.Poisson()).fit()
    np.testing.assert_allclose(float(log_likelihood(rp)[0]), smp.llf, atol=1e-6)
    np.testing.assert_allclose(float(aic(rp)[0]), smp.aic, atol=1e-6)


@needs_sm
def test_compare_models_f_and_lrt():
    import statsmodels.api as sm
    from scipy import stats

    rng = np.random.default_rng(9)
    X = _design(rng)
    Xr = X[:, :2]

    # Gaussian -> F-test (extra sum of squares).
    y = X @ np.array([1.0, 0.5, -0.3]) + rng.standard_normal(80) * 0.4
    full = glm_fit(jnp.asarray(y[None, :]), jnp.asarray(X))
    reduced = glm_fit(jnp.asarray(y[None, :]), jnp.asarray(Xr))
    F, Fp = compare_models(full, reduced)
    smf, smr = sm.OLS(y, X).fit(), sm.OLS(y, Xr).fit()
    F_sm = ((smr.ssr - smf.ssr) / 1) / (smf.ssr / smf.df_resid)
    np.testing.assert_allclose(float(F[0]), F_sm, atol=1e-8)
    np.testing.assert_allclose(
        float(Fp[0]), float(stats.f.sf(F_sm, 1, smf.df_resid)), atol=1e-10
    )

    # Poisson -> LRT (chi-square on the deviance / loglik difference).
    mu = np.exp(X @ np.array([0.5, 0.3, -0.2]))
    yp = rng.poisson(mu).astype(float)
    pf = glm_fit(jnp.asarray(yp[None, :]), jnp.asarray(X), family=POISSON, n_iter=30)
    pr = glm_fit(jnp.asarray(yp[None, :]), jnp.asarray(Xr), family=POISSON, n_iter=30)
    stat, sp = compare_models(pf, pr, test='LRT')
    smpf = sm.GLM(yp, X, family=sm.families.Poisson()).fit()
    smpr = sm.GLM(yp, Xr, family=sm.families.Poisson()).fit()
    lrt_sm = 2 * (smpf.llf - smpr.llf)
    np.testing.assert_allclose(float(stat[0]), lrt_sm, atol=1e-6)
    np.testing.assert_allclose(
        float(sp[0]), float(stats.chi2.sf(lrt_sm, 1)), atol=1e-8
    )


# ---------------------------------------------------------------------------
# p-value helpers vs scipy
# ---------------------------------------------------------------------------


@needs_scipy
def test_t_and_f_pvalues_match_scipy():
    from scipy import stats

    from nitrix.stats.glm import _f_sf, _t_two_sided_sf

    df = 42.0
    for tval in (0.3, 1.0, 2.5, 5.0):
        got = float(_t_two_sided_sf(jnp.asarray(tval), df))
        want = 2.0 * stats.t.sf(abs(tval), df)
        np.testing.assert_allclose(got, want, atol=1e-10)
    for fval in (0.5, 1.0, 3.0, 8.0):
        got = float(_f_sf(jnp.asarray(fval), 2.0, df))
        want = float(stats.f.sf(fval, 2, df))
        np.testing.assert_allclose(got, want, atol=1e-10)


# ---------------------------------------------------------------------------
# cuSOLVER-free HLO guard
# ---------------------------------------------------------------------------


_CUSOLVER = ('cusolver', 'syevd', 'potrf', 'getrf', 'geqrf', 'gesvd', 'cholesky', 'eigh')


def _cusolver_calls(hlo: str):
    targets = set(re.findall(r'custom_call_target="([^"]+)"', hlo))
    return [c for c in targets if any(t in c.lower() for t in _CUSOLVER)]


def test_glm_hlo_is_cusolver_free():
    """Both the OLS fast path and the IRLS path stay on cuBLAS only."""
    rng = np.random.default_rng(7)
    X = jnp.asarray(_design(rng, p=5))
    Y = jnp.asarray(rng.standard_normal((128, 80)))
    f_ols = jax.jit(lambda Y, X: glm_fit(Y, X).coef)
    hlo = f_ols.lower(Y, X).compile().as_text()
    assert not _cusolver_calls(hlo), _cusolver_calls(hlo)

    Yp = jnp.asarray(rng.poisson(2.0, size=(128, 80)).astype(float))
    f_pois = jax.jit(lambda Y, X: glm_fit(Y, X, family=POISSON, n_iter=10).coef)
    hlo_p = f_pois.lower(Yp, X).compile().as_text()
    assert not _cusolver_calls(hlo_p), _cusolver_calls(hlo_p)
