# -*- coding: utf-8 -*-
"""Multi-seed robustness guard for the variance-component Newton solvers.

This is the regression test that was missing when the saddle-point bug shipped:
the block-Woodbury R2, nested R3, GLS, and R2+corr fits all minimise a
**non-convex** profile-REML objective by Newton, and a plain damped step is not
a descent direction at the indefinite (saddle) Hessian those objectives have away
from the optimum -- so on a *seed-dependent* fraction of datasets the fit
silently stalled at the wrong variance components.  Each solver had only a
single-seed oracle test, which never hit the bad seeds.

Here every solver is checked against its **exact dense profile-REML reference**
over a sweep of seeds; a stalled fit fails loudly.  The shared saddle-free Newton
(`_optimise.damped_newton`) must keep all seeds correct.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import minimize, minimize_scalar

jax.config.update('jax_enable_x64', True)

from nitrix.stats.lme import gls_fit, lme_fit

SEEDS = list(range(8))


# ---------------------------------------------------------------------------
# R2 block-Woodbury: correlated random slope (1 + x | g)
# ---------------------------------------------------------------------------


def _dense_r2(y, X, zc, group, G, p):
    def neg2(th):
        ll = np.array([[np.exp(th[0]), 0.0], [th[1], np.exp(th[2])]])
        Gm = ll @ ll.T
        se = np.exp(th[3])
        xvx = np.zeros((p, p))
        xvy = np.zeros(p)
        yvy = 0.0
        ldv = 0.0
        for g in range(G):
            m = group == g
            Zg = zc[m]
            V = Zg @ Gm @ Zg.T + se * np.eye(int(m.sum()))
            Vi = np.linalg.inv(V)
            _, ld = np.linalg.slogdet(V)
            ldv += ld
            xvx += X[m].T @ Vi @ X[m]
            xvy += X[m].T @ Vi @ y[m]
            yvy += y[m] @ Vi @ y[m]
        beta = np.linalg.solve(xvx, xvy)
        return ldv + np.linalg.slogdet(xvx)[1] + (yvy - beta @ xvy)

    r = minimize(
        neg2,
        [0.0, 0.0, 0.0, 0.0],
        method='Nelder-Mead',
        options={'xatol': 1e-7, 'fatol': 1e-7, 'maxiter': 10000},
    )
    ll = np.array([[np.exp(r.x[0]), 0.0], [r.x[1], np.exp(r.x[2])]])
    return ll @ ll.T, np.exp(r.x[3])


@pytest.mark.parametrize('seed', SEEDS)
def test_r2_blockwoodbury_recovers_all_seeds(seed):
    rng = np.random.default_rng(seed)
    G, n_per, p = 15, 8, 2
    group = np.repeat(np.arange(G), n_per).astype(np.int32)
    N = G * n_per
    X = np.ones((N, p))
    X[:, 1] = rng.standard_normal(N)
    zc = np.c_[np.ones(N), rng.standard_normal(N)]
    b = (
        rng.standard_normal((G, 2))
        @ np.linalg.cholesky(np.array([[0.6, 0.2], [0.2, 0.4]])).T
    )
    y = (
        X @ np.array([1.0, 0.5])
        + np.einsum('nr,nr->n', zc, b[group])
        + rng.standard_normal(N) * np.sqrt(0.4)
    )
    Gd, sed = _dense_r2(y, X, zc, group, G, p)
    res = lme_fit(
        jnp.asarray(y[None, :]),
        jnp.asarray(X),
        group=jnp.asarray(group),
        z=jnp.asarray(zc),
        n_iter=40,
    )
    assert np.allclose(np.asarray(res.cov_re[0]), Gd, atol=2e-2)
    assert abs(float(res.sigma_e_sq[0]) - sed) < 2e-2


# ---------------------------------------------------------------------------
# R3 nested (1 | g1/g2)
# ---------------------------------------------------------------------------


def _dense_nested(y, X, o, i, q1, q2):
    n = X.shape[0]
    Z1, Z2 = np.eye(q1)[o], np.eye(q2)[i]

    def neg2(th):
        s1, s2, se = np.exp(th)
        V = s1 * Z1 @ Z1.T + s2 * Z2 @ Z2.T + se * np.eye(n)
        Vi = np.linalg.inv(V)
        xvx = X.T @ Vi @ X
        beta = np.linalg.solve(xvx, X.T @ Vi @ y)
        r = y - X @ beta
        return np.linalg.slogdet(V)[1] + np.linalg.slogdet(xvx)[1] + r @ Vi @ r

    r = minimize(
        neg2,
        np.log([0.4, 0.3, 0.4]),
        method='Nelder-Mead',
        options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 9000},
    )
    return np.exp(r.x)


@pytest.mark.parametrize('seed', SEEDS)
def test_r3_nested_recovers_all_seeds(seed):
    rng = np.random.default_rng(seed)
    q1, sp, op = 12, 5, 5
    b1 = rng.standard_normal(q1) * np.sqrt(0.5)
    sub = 0
    X, y, o, i = [], [], [], []
    for a in range(q1):
        for _ in range(sp):
            b2 = rng.standard_normal() * np.sqrt(0.3)
            for _ in range(op):
                x = rng.standard_normal()
                X.append([1.0, x])
                y.append(
                    1 + 0.5 * x + b1[a] + b2 + rng.standard_normal() * 0.63
                )
                o.append(a)
                i.append(sub)
            sub += 1
    X, y = np.asarray(X), np.asarray(y)
    o, i = np.asarray(o, np.int32), np.asarray(i, np.int32)
    s1, s2, se = _dense_nested(y, X, o, i, q1, sub)
    res = lme_fit(
        jnp.asarray(y[None, :]),
        jnp.asarray(X),
        group=jnp.asarray(o),
        inner=jnp.asarray(i),
        n_iter=50,
    )
    assert abs(float(res.var_outer[0]) - s1) < 1.5e-2
    assert abs(float(res.var_inner[0]) - s2) < 1.5e-2
    assert abs(float(res.sigma_e_sq[0]) - se) < 1.5e-2


# ---------------------------------------------------------------------------
# GLS (ar1 residual, no random effect)
# ---------------------------------------------------------------------------


def _dense_gls_ar1(y, X, group, G, p):
    def neg2(raw):
        rho = np.tanh(raw)
        xvx = np.zeros((p, p))
        xvy = np.zeros(p)
        yvy = 0.0
        ld = 0.0
        for g in range(G):
            m = group == g
            n = int(m.sum())
            R = rho ** np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
            Ri = np.linalg.inv(R)
            ld += np.linalg.slogdet(R)[1]
            xvx += X[m].T @ Ri @ X[m]
            xvy += X[m].T @ Ri @ y[m]
            yvy += y[m] @ Ri @ y[m]
        beta = np.linalg.solve(xvx, xvy)
        return (
            (len(y) - p) * np.log(yvy - beta @ xvy)
            + ld
            + np.linalg.slogdet(xvx)[1]
        )

    return np.tanh(
        minimize_scalar(lambda r: neg2(r), bounds=(-6, 6), method='bounded').x
    )


@pytest.mark.parametrize('seed', SEEDS)
def test_gls_ar1_recovers_all_seeds(seed):
    rng = np.random.default_rng(seed)
    G, n_per, p = 18, 8, 2
    group = np.repeat(np.arange(G), n_per).astype(np.int32)
    N = G * n_per
    X = np.ones((N, p))
    X[:, 1] = rng.standard_normal(N)
    rho_t = rng.uniform(-0.5, 0.7)
    y = np.empty(N)
    for g in range(G):
        m = group == g
        n = int(m.sum())
        R = rho_t ** np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
        y[m] = (
            X[m] @ np.array([1.0, 0.5])
            + np.linalg.cholesky(R) @ rng.standard_normal(n) * 0.7
        )
    rho_ref = _dense_gls_ar1(y, X, group, G, p)
    res = gls_fit(
        jnp.asarray(y[None, :]),
        jnp.asarray(X),
        group=jnp.asarray(group),
        corr='ar1',
    )
    assert abs(float(res.rho[0]) - rho_ref) < 1e-2


# ---------------------------------------------------------------------------
# R2 + corr (random intercept + ar1 residual)
# ---------------------------------------------------------------------------


def _dense_r2corr(y, X, group, G, p):
    def neg2(th):
        sb, se, raw = np.exp(th[0]), np.exp(th[1]), th[2]
        rho = np.tanh(raw)
        xvx = np.zeros((p, p))
        xvy = np.zeros(p)
        yvy = 0.0
        ldv = 0.0
        for g in range(G):
            m = group == g
            n = int(m.sum())
            R = rho ** np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
            V = sb * np.ones((n, n)) + se * R
            Vi = np.linalg.inv(V)
            ldv += np.linalg.slogdet(V)[1]
            xvx += X[m].T @ Vi @ X[m]
            xvy += X[m].T @ Vi @ y[m]
            yvy += y[m] @ Vi @ y[m]
        beta = np.linalg.solve(xvx, xvy)
        return ldv + np.linalg.slogdet(xvx)[1] + (yvy - beta @ xvy)

    r = minimize(
        neg2,
        [np.log(0.4), np.log(0.4), 0.0],
        method='Nelder-Mead',
        options={'xatol': 1e-7, 'fatol': 1e-7, 'maxiter': 9000},
    )
    return np.exp(r.x[0]), np.exp(r.x[1]), np.tanh(r.x[2])


@pytest.mark.parametrize('seed', SEEDS)
def test_r2corr_recovers_all_seeds(seed):
    rng = np.random.default_rng(seed)
    G, n_per, p = 20, 8, 2
    group = np.repeat(np.arange(G), n_per).astype(np.int32)
    N = G * n_per
    X = np.ones((N, p))
    X[:, 1] = rng.standard_normal(N)
    sb2, se2, rho_t = 0.5, 0.4, 0.5
    y = np.empty(N)
    for g in range(G):
        m = group == g
        n = int(m.sum())
        R = rho_t ** np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
        cov = sb2 * np.ones((n, n)) + se2 * R
        y[m] = X[m] @ np.array([1.0, 0.5]) + np.linalg.cholesky(
            cov
        ) @ rng.standard_normal(n)
    sb_r, se_r, rho_r = _dense_r2corr(y, X, group, G, p)
    res = lme_fit(
        jnp.asarray(y[None, :]),
        jnp.asarray(X),
        group=jnp.asarray(group),
        corr='ar1',
        n_iter=60,
    )
    assert abs(float(res.cov_re[0, 0, 0]) - sb_r) < 2e-2
    assert abs(float(res.sigma_e_sq[0]) - se_r) < 2e-2
    assert abs(float(res.rho[0]) - rho_r) < 1e-2
