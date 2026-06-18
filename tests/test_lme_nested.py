# -*- coding: utf-8 -*-
"""Tests for the nested random-effects LME (v3 §1.1 tier R3).

``lme_fit(..., inner=g2)`` fits ``(1 | g1/g2)`` via the telescoping-Woodbury
nested solver.  Anchored two ways: an **exact dense profile-REML reference**
(build the full ``V = sigma1^2 Z1 Z1^T + sigma2^2 Z2 Z2^T + sigma_e^2 I`` and
minimise the same REML criterion) -- the authoritative same-estimator check --
and **statsmodels MixedLM** with a nested variance component (external oracle).
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import minimize

jax.config.update('jax_enable_x64', True)

from nitrix.stats.lme import NestedLMEResult, lme_fit


def _simulate_nested(
    *, seed, q1=14, subj_per=6, obs_per=5, s1=0.6, s2=0.4, se=0.5
):
    rng = np.random.default_rng(seed)
    b1 = rng.standard_normal(q1) * np.sqrt(s1)
    sub = 0
    X, y, outer, inner = [], [], [], []
    for i in range(q1):
        for _ in range(subj_per):
            b2 = rng.standard_normal() * np.sqrt(s2)
            for _ in range(obs_per):
                x1 = rng.standard_normal()
                yy = (
                    1.0
                    + 0.5 * x1
                    + b1[i]
                    + b2
                    + rng.standard_normal() * np.sqrt(se)
                )
                X.append([1.0, x1])
                y.append(yy)
                outer.append(i)
                inner.append(sub)
            sub += 1
    return (
        np.asarray(X),
        np.asarray(y),
        np.asarray(outer, np.int32),
        np.asarray(inner, np.int32),
        sub,
    )


def _dense_reml(y, X, outer, inner, q1, q2):
    n = X.shape[0]
    Z1 = np.eye(q1)[outer]
    Z2 = np.eye(q2)[inner]

    def neg2l(th):
        s1, s2, se = np.exp(th)
        V = s1 * Z1 @ Z1.T + s2 * Z2 @ Z2.T + se * np.eye(n)
        Vi = np.linalg.inv(V)
        xvx = X.T @ Vi @ X
        beta = np.linalg.solve(xvx, X.T @ Vi @ y)
        r = y - X @ beta
        _, ldv = np.linalg.slogdet(V)
        _, ldx = np.linalg.slogdet(xvx)
        return ldv + ldx + r @ Vi @ r

    res = minimize(
        neg2l,
        np.log([0.4, 0.3, 0.4]),
        method='Nelder-Mead',
        options={'xatol': 1e-9, 'fatol': 1e-9, 'maxiter': 8000},
    )
    s1, s2, se = np.exp(res.x)
    V = s1 * Z1 @ Z1.T + s2 * Z2 @ Z2.T + se * np.eye(n)
    Vi = np.linalg.inv(V)
    beta = np.linalg.solve(X.T @ Vi @ X, X.T @ Vi @ y)
    return beta, s1, s2, se


def test_nested_matches_dense_reml():
    X, y, outer, inner, q2 = _simulate_nested(seed=0)
    rb, rs1, rs2, rse = _dense_reml(y, X, outer, inner, 14, q2)
    res = lme_fit(
        jnp.asarray(np.tile(y, (3, 1))),
        jnp.asarray(X),
        group=jnp.asarray(outer),
        inner=jnp.asarray(inner),
        n_iter=40,
    )
    assert isinstance(res, NestedLMEResult) and res.tier == 'R3'
    assert np.allclose(np.asarray(res.beta_hat[0]), rb, atol=2e-3)
    assert abs(float(res.var_outer[0]) - rs1) < 5e-3
    assert abs(float(res.var_inner[0]) - rs2) < 5e-3
    assert abs(float(res.sigma_e_sq[0]) - rse) < 5e-3
    # Same fit for every (identical) voxel row.
    assert np.allclose(res.beta_hat[0], res.beta_hat[2])


def test_nested_matches_statsmodels():
    sm = pytest.importorskip('statsmodels.formula.api')
    pd = pytest.importorskip('pandas')
    X, y, outer, inner, _ = _simulate_nested(seed=1)
    df = pd.DataFrame({'y': y, 'x': X[:, 1], 'site': outer, 'subj': inner})
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        md = sm.mixedlm(
            'y ~ x',
            df,
            groups='site',
            re_formula='1',
            vc_formula={'subj': '0+C(subj)'},
        )
        mdf = md.fit(reml=True)
    res = lme_fit(
        jnp.asarray(y[None, :]),
        jnp.asarray(X),
        group=jnp.asarray(outer),
        inner=jnp.asarray(inner),
        n_iter=50,
    )
    assert np.allclose(
        np.asarray(res.beta_hat[0]), mdf.fe_params.values, atol=2e-3
    )
    assert (
        abs(float(res.var_outer[0]) - float(np.asarray(mdf.cov_re)[0, 0]))
        < 1e-2
    )
    assert abs(float(res.var_inner[0]) - float(mdf.vcomp[0])) < 1e-2
    assert abs(float(res.sigma_e_sq[0]) - float(mdf.scale)) < 1e-2


def test_nested_recovers_components():
    """A larger design recovers the simulated variance components."""
    X, y, outer, inner, _ = _simulate_nested(
        seed=3, q1=40, subj_per=8, obs_per=6, s1=0.5, s2=0.3, se=0.4
    )
    res = lme_fit(
        jnp.asarray(y[None, :]),
        jnp.asarray(X),
        group=jnp.asarray(outer),
        inner=jnp.asarray(inner),
        n_iter=50,
    )
    assert abs(float(res.var_outer[0]) - 0.5) < 0.25
    assert abs(float(res.var_inner[0]) - 0.3) < 0.15
    assert abs(float(res.sigma_e_sq[0]) - 0.4) < 0.1
    assert float(res.var_outer[0]) > 0 and float(res.var_inner[0]) > 0


def test_nested_shapes_and_dispatch():
    X, y, outer, inner, _ = _simulate_nested(seed=5)
    V = 7
    res = lme_fit(
        jnp.asarray(np.tile(y, (V, 1))),
        jnp.asarray(X),
        group=jnp.asarray(outer),
        inner=jnp.asarray(inner),
        n_iter=30,
    )
    assert res.beta_hat.shape == (V, 2)
    assert res.var_outer.shape == (V,)
    assert res.var_inner.shape == (V,)
    assert res.sigma_e_sq.shape == (V,)
    assert res.log_lik.shape == (V,)


def test_nested_rejects_random_slope():
    X, y, outer, inner, _ = _simulate_nested(seed=0)
    z = jnp.asarray(np.c_[np.ones(len(y)), X[:, 1]])
    with pytest.raises(ValueError, match='nested fit'):
        lme_fit(
            jnp.asarray(y[None, :]),
            jnp.asarray(X),
            group=jnp.asarray(outer),
            inner=jnp.asarray(inner),
            z=z,
        )
