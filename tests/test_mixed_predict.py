# -*- coding: utf-8 -*-
"""BLUP / conditional prediction for the mixed-effects fitters (nimox Tier-3 B).

``lme_predict`` / ``ranef`` (LME, opt-in ``retain_blups``) and ``glmm_predict``
(GLMM, modes always retained).  Oracles: the independent mixed-model-equation
BLUP ``b_g = G Z_g^T (Z_g G Z_g^T + sigma_e^2 I)^{-1} r_g`` per group, and the
definitional conditional mean ``X beta + Z b_group``.
"""

from __future__ import annotations

import jax

jax.config.update('jax_enable_x64', True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from nitrix.stats import glmm_fit, glmm_predict, ranef  # noqa: E402
from nitrix.stats.lme import lme_fit, lme_predict  # noqa: E402


def _blup_oracle(Y, X, beta, z, group, G, sigma_e_sq, q):
    """Per-group MME BLUP, computed independently in numpy: (V, q, r)."""
    Y, X, beta = np.asarray(Y), np.asarray(X), np.asarray(beta)
    z, group = np.asarray(z), np.asarray(group)
    G, s2 = np.asarray(G), np.asarray(sigma_e_sq)
    v, _ = beta.shape
    r = z.shape[1]
    out = np.zeros((v, q, r))
    resid = Y - beta @ X.T  # (V, N)
    for vi in range(v):
        Ginv = np.linalg.inv(G[vi])
        for g in range(q):
            m = group == g
            zg, rg = z[m], resid[vi, m]
            A = zg.T @ zg / s2[vi] + Ginv
            out[vi, g] = np.linalg.solve(A, zg.T @ rg / s2[vi])
    return out


def _corr_matrix(name, rho, ng):
    """Dense within-group correlation matrix R(rho) for an ``ng``-row group."""
    idx = np.arange(ng)
    if name == 'ar1':
        return rho ** np.abs(idx[:, None] - idx[None, :])
    if name == 'cs':
        return (1.0 - rho) * np.eye(ng) + rho * np.ones((ng, ng))
    raise ValueError(name)


def _blup_corr_oracle(Y, X, beta, z, group, G, sigma_e_sq, rho, q, name):
    """Per-group MME BLUP with a structured residual ``Sigma_g = s2 R(rho)``,
    inverting ``R`` densely per group (independent of the whitening)."""
    Y, X, beta = np.asarray(Y), np.asarray(X), np.asarray(beta)
    z, group = np.asarray(z), np.asarray(group)
    G, s2, rho = np.asarray(G), np.asarray(sigma_e_sq), np.asarray(rho)
    v, _ = beta.shape
    r = z.shape[1]
    out = np.zeros((v, q, r))
    resid = Y - beta @ X.T  # (V, N)
    for vi in range(v):
        Ginv = np.linalg.inv(G[vi])
        for g in range(q):
            m = group == g
            zg, rg = z[m], resid[vi, m]
            sig_inv = np.linalg.inv(
                s2[vi] * _corr_matrix(name, rho[vi], m.sum())
            )
            A = zg.T @ sig_inv @ zg + Ginv
            out[vi, g] = np.linalg.solve(A, zg.T @ sig_inv @ rg)
    return out


# ---------------------------------------------------------------------------
# R1 (scalar random intercept)
# ---------------------------------------------------------------------------


def _ri_data(seed=0, V=3, q=8, per=12, p=2):
    rng = np.random.default_rng(seed)
    n = q * per
    X = rng.normal(size=(n, p))
    X[:, 0] = 1.0
    group = np.repeat(np.arange(q), per).astype(np.int32)
    beta = rng.normal(size=(V, p))
    b = rng.normal(size=(V, q)) * 0.8
    Y = (X @ beta.T).T + b[:, group] + rng.normal(size=(V, n)) * 0.3
    return jnp.asarray(Y), jnp.asarray(X), jnp.asarray(group), q


def test_r1_ranef_matches_mme_oracle():
    Y, X, group, q = _ri_data()
    res = lme_fit(Y, X, group=group, retain_blups=True)
    b = np.asarray(ranef(res))
    assert b.shape == (Y.shape[0], q)  # scalar intercept -> (V, q)
    z1 = np.ones((X.shape[0], 1))
    ref = _blup_oracle(
        Y, X, res.beta_hat, z1, group, res.cov_re, res.sigma_e_sq, q
    )[..., 0]
    np.testing.assert_allclose(b, ref, atol=1e-6)


def test_r1_conditional_predict_is_fixed_plus_mode():
    Y, X, group, q = _ri_data()
    res = lme_fit(Y, X, group=group, retain_blups=True)
    pop = np.asarray(lme_predict(res, X, group=group, level='population'))
    cond = np.asarray(lme_predict(res, X, group=group, level='conditional'))
    beta = np.asarray(res.beta_hat)
    b = np.asarray(ranef(res))
    np.testing.assert_allclose(pop, beta @ np.asarray(X).T, atol=1e-6)
    np.testing.assert_allclose(
        cond, beta @ np.asarray(X).T + b[:, np.asarray(group)], atol=1e-6
    )
    # conditional fits the group-structured response better than marginal.
    assert np.mean((cond - np.asarray(Y)) ** 2) < np.mean(
        (pop - np.asarray(Y)) ** 2
    )


def test_r1_unseen_group_falls_back_to_population():
    Y, X, group, q = _ri_data()
    res = lme_fit(Y, X, group=group, retain_blups=True)
    # relabel a few rows to an unseen level q (out of range) -> marginal.
    g2 = np.asarray(group).copy()
    g2[:5] = q
    cond = np.asarray(
        lme_predict(res, X, group=jnp.asarray(g2), level='conditional')
    )
    pop = np.asarray(lme_predict(res, X, group=group, level='population'))
    np.testing.assert_allclose(cond[:, :5], pop[:, :5], atol=1e-6)


# ---------------------------------------------------------------------------
# R2 (random slope)
# ---------------------------------------------------------------------------


def test_r2_ranef_matches_mme_oracle():
    rng = np.random.default_rng(1)
    V, q, per, p = 2, 6, 16, 2
    n = q * per
    X = rng.normal(size=(n, p))
    X[:, 0] = 1.0
    xz = rng.normal(size=n)
    z = np.column_stack([np.ones(n), xz])  # (1 + x | g)
    group = np.repeat(np.arange(q), per).astype(np.int32)
    beta = rng.normal(size=(V, p))
    Y = (X @ beta.T).T + rng.normal(size=(V, n)) * 0.4
    res = lme_fit(Y, X, group=group, z=jnp.asarray(z), retain_blups=True)
    b = np.asarray(ranef(res))
    assert b.shape == (V, q, 2)  # random slope -> (V, q, r)
    ref = _blup_oracle(
        Y, X, res.beta_hat, z, group, res.cov_re, res.sigma_e_sq, q
    )
    np.testing.assert_allclose(b, ref, atol=1e-6)
    # conditional predict reassembles X beta + z . b[group].
    cond = np.asarray(
        lme_predict(res, X, z=jnp.asarray(z), group=group, level='conditional')
    )
    expect = np.asarray(res.beta_hat) @ X.T + np.einsum(
        'vnr,nr->vn', b[:, group, :], z
    )
    np.testing.assert_allclose(cond, expect, atol=1e-6)


# ---------------------------------------------------------------------------
# R2 + corr (random effect + structured within-group residual)
# ---------------------------------------------------------------------------


def _ar1_resid(rng, V, per, q, rho_true):
    """Per-group AR(1) residuals (rows already in within-group time order)."""
    e = np.zeros((V, q * per))
    for g in range(q):
        innov = rng.normal(size=(V, per))
        eg = np.zeros((V, per))
        eg[:, 0] = innov[:, 0]
        for t in range(1, per):
            eg[:, t] = (
                rho_true * eg[:, t - 1]
                + np.sqrt(1.0 - rho_true**2) * innov[:, t]
            )
        e[:, g * per : (g + 1) * per] = eg
    return e


def _corr_ri_data(seed, name, rho_true=0.6, V=2, q=6, per=14, p=2):
    rng = np.random.default_rng(seed)
    n = q * per
    X = rng.normal(size=(n, p))
    X[:, 0] = 1.0
    group = np.repeat(np.arange(q), per).astype(np.int32)
    beta = rng.normal(size=(V, p))
    b = rng.normal(size=(V, q)) * 0.8
    if name == 'ar1':
        e = _ar1_resid(rng, V, per, q, rho_true)
    else:  # cs: exchangeable -- a shared per-group shift on top of white noise.
        shared = rng.normal(size=(V, q))[:, group]
        e = np.sqrt(rho_true) * shared + np.sqrt(1.0 - rho_true) * rng.normal(
            size=(V, n)
        )
    Y = (X @ beta.T).T + b[:, group] + 0.5 * e
    return jnp.asarray(Y), jnp.asarray(X), jnp.asarray(group), q


@pytest.mark.parametrize('name', ['ar1', 'cs'])
def test_r2corr_ranef_matches_dense_R_oracle(name):
    Y, X, group, q = _corr_ri_data(10, name)
    res = lme_fit(Y, X, group=group, corr=name, retain_blups=True)
    b = np.asarray(ranef(res))
    assert b.shape == (Y.shape[0], q)  # scalar intercept -> (V, q)
    z1 = np.ones((X.shape[0], 1))
    ref = _blup_corr_oracle(
        Y,
        X,
        res.beta_hat,
        z1,
        group,
        res.cov_re,
        res.sigma_e_sq,
        res.rho,
        q,
        name,
    )[..., 0]
    np.testing.assert_allclose(b, ref, atol=1e-6)


def test_r2corr_conditional_predict_and_whitening_matters():
    Y, X, group, q = _corr_ri_data(11, 'ar1', rho_true=0.7)
    res = lme_fit(Y, X, group=group, corr='ar1', retain_blups=True)
    b = np.asarray(ranef(res))
    cond = np.asarray(lme_predict(res, X, group=group, level='conditional'))
    pop = np.asarray(lme_predict(res, X, group=group, level='population'))
    beta = np.asarray(res.beta_hat)
    np.testing.assert_allclose(pop, beta @ np.asarray(X).T, atol=1e-6)
    np.testing.assert_allclose(
        cond, beta @ np.asarray(X).T + b[:, np.asarray(group)], atol=1e-6
    )
    # the fitted AR(1) correlation is substantial, so the whitened modes are not
    # the iid-residual modes -- whitening is genuinely applied.
    assert float(np.mean(np.abs(res.rho))) > 0.2
    iid_b = _blup_oracle(
        Y,
        X,
        res.beta_hat,
        np.ones((X.shape[0], 1)),
        group,
        res.cov_re,
        res.sigma_e_sq,
        q,
    )[..., 0]
    assert not np.allclose(b, iid_b, atol=1e-3)


def test_r2corr_random_slope_ranef():
    rng = np.random.default_rng(12)
    V, q, per, p = 2, 6, 16, 2
    n = q * per
    X = rng.normal(size=(n, p))
    X[:, 0] = 1.0
    xz = rng.normal(size=n)
    z = np.column_stack([np.ones(n), xz])  # (1 + x | g)
    group = np.repeat(np.arange(q), per).astype(np.int32)
    beta = rng.normal(size=(V, p))
    Y = (X @ beta.T).T + 0.5 * _ar1_resid(rng, V, per, q, 0.5)
    res = lme_fit(
        Y, X, group=group, z=jnp.asarray(z), corr='ar1', retain_blups=True
    )
    b = np.asarray(ranef(res))
    assert b.shape == (V, q, 2)  # random slope keeps (V, q, r)
    ref = _blup_corr_oracle(
        Y,
        X,
        res.beta_hat,
        z,
        group,
        res.cov_re,
        res.sigma_e_sq,
        res.rho,
        q,
        'ar1',
    )
    np.testing.assert_allclose(b, ref, atol=1e-6)
    cond = np.asarray(
        lme_predict(res, X, z=jnp.asarray(z), group=group, level='conditional')
    )
    expect = np.asarray(res.beta_hat) @ X.T + np.einsum(
        'vnr,nr->vn', b[:, group, :], z
    )
    np.testing.assert_allclose(cond, expect, atol=1e-6)


# ---------------------------------------------------------------------------
# Opt-in contract + staged tiers
# ---------------------------------------------------------------------------


def test_ranef_raises_without_retain_blups():
    Y, X, group, q = _ri_data()
    res = lme_fit(Y, X, group=group)  # default retain_blups=False
    assert res.blups is None
    with pytest.raises(ValueError, match='retain_blups'):
        ranef(res)
    # population still works without modes.
    pop = lme_predict(res, X, group=group, level='population')
    assert pop.shape == Y.shape


def test_retain_blups_unsupported_tier_raises():
    rng = np.random.default_rng(2)
    V, n, p = 2, 80, 2
    X = jnp.asarray(rng.normal(size=(n, p)))
    Y = jnp.asarray(rng.normal(size=(V, n)))
    group = jnp.asarray(np.repeat(np.arange(8), 10).astype(np.int32))
    cross = jnp.asarray(np.tile(np.arange(4), 20).astype(np.int32))
    with pytest.raises(NotImplementedError, match='retain_blups'):
        lme_fit(Y, X, group=group, cross=cross, retain_blups=True)
    # but the crossed fit's population prediction works.
    res = lme_fit(Y, X, group=group, cross=cross)
    assert lme_predict(res, X, level='population').shape == Y.shape


# ---------------------------------------------------------------------------
# GLMM (modes always retained)
# ---------------------------------------------------------------------------


def test_glmm_predict_population_and_conditional():
    rng = np.random.default_rng(3)
    V, q, per, p = 2, 6, 15, 2
    n = q * per
    X = rng.normal(size=(n, p))
    X[:, 0] = 1.0
    group = np.repeat(np.arange(q), per).astype(np.int32)
    eta = (
        X @ rng.normal(size=(V, p)).T
        + (rng.normal(size=(V, q)) * 0.7)[:, group].T
    )
    Y = (rng.uniform(size=(n, V)) < 1 / (1 + np.exp(-eta))).astype(np.float64)
    res = glmm_fit(
        jnp.asarray(Y.T),
        jnp.asarray(X),
        group=jnp.asarray(group),
        family='binomial',
    )
    b = np.asarray(ranef(res))
    assert b.shape == (V, q)
    # link-level conditional == X beta + b[group]; response == sigmoid of it.
    link = np.asarray(
        glmm_predict(
            res,
            jnp.asarray(X),
            group=jnp.asarray(group),
            level='conditional',
            type='link',
        )
    )
    expect = np.asarray(res.beta_hat) @ X.T + b[:, group]
    np.testing.assert_allclose(link, expect, atol=1e-5)
    resp = np.asarray(
        glmm_predict(
            res,
            jnp.asarray(X),
            group=jnp.asarray(group),
            level='conditional',
            type='response',
        )
    )
    np.testing.assert_allclose(resp, 1 / (1 + np.exp(-expect)), atol=1e-5)
    # population ignores the modes.
    pop = np.asarray(
        glmm_predict(res, jnp.asarray(X), level='population', type='link')
    )
    np.testing.assert_allclose(pop, np.asarray(res.beta_hat) @ X.T, atol=1e-5)


def test_lme_predict_differentiable():
    Y, X, group, q = _ri_data()
    res = lme_fit(Y, X, group=group, retain_blups=True)
    g = jax.grad(
        lambda xx: jnp.sum(
            lme_predict(res, xx, group=group, level='conditional')
        )
    )(X)
    assert g.shape == X.shape
    assert bool(jnp.all(jnp.isfinite(g)))
