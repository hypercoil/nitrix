# -*- coding: utf-8 -*-
"""Tests for residual variance functions (``gls_fit(weights=…)``, v3 §1.4).

A variance function makes the residual heteroscedastic: ``Var(eps_i) =
sigma_e^2 g_i^2`` with ``g`` from ``var_power`` (``|v|^delta``) or ``var_ident``
(per-stratum).  It composes with the correlation structures -- the full residual
is ``sigma_e^2 diag(g) R(rho) diag(g)``.  Each fit is anchored against an
**exact dense profile-REML reference** (build the full per-group ``diag(g) R
diag(g)`` and minimise the same REML criterion) -- the authoritative
same-estimator check.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import minimize, minimize_scalar

jax.config.update('jax_enable_x64', True)

from nitrix.stats.lme import gls_fit, var_ident, var_power


def _ar1R(n, rho):
    return rho ** np.abs(np.subtract.outer(np.arange(n), np.arange(n)))


def test_varpower_with_ar1_matches_dense_reml():
    rng = np.random.default_rng(3)
    G, n_per, p = 16, 9, 2
    group = np.repeat(np.arange(G), n_per).astype(np.int32)
    N = G * n_per
    X = np.ones((N, p))
    X[:, 1] = rng.standard_normal(N)
    v = np.abs(rng.standard_normal(N)) + 0.4
    rho_t, d_t, se_t = 0.5, 0.7, 0.6
    y = np.empty(N)
    for g in range(G):
        m = group == g
        n = int(m.sum())
        gi = v[m] ** d_t
        cov = se_t * (gi[:, None] * _ar1R(n, rho_t) * gi[None, :])
        y[m] = X[m] @ np.array([1.0, 0.5]) + np.linalg.cholesky(
            cov
        ) @ rng.standard_normal(n)

    def neg2(th):
        rho, d = np.tanh(th[0]), th[1]
        XtVX = np.zeros((p, p))
        XtVy = np.zeros(p)
        yVy = 0.0
        ld = 0.0
        for g in range(G):
            m = group == g
            n = int(m.sum())
            gi = v[m] ** d
            R = gi[:, None] * _ar1R(n, rho) * gi[None, :]
            Ri = np.linalg.inv(R)
            ld += np.linalg.slogdet(R)[1]
            XtVX += X[m].T @ Ri @ X[m]
            XtVy += X[m].T @ Ri @ y[m]
            yVy += y[m] @ Ri @ y[m]
        b = np.linalg.solve(XtVX, XtVy)
        return (N - p) * np.log(yVy - b @ XtVy) + ld + np.linalg.slogdet(XtVX)[1]

    r = minimize(
        neg2, [0.0, 0.0], method='Nelder-Mead',
        options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 9000},
    )
    res = gls_fit(
        jnp.asarray(y[None, :]), jnp.asarray(X), group=jnp.asarray(group),
        corr='ar1', weights=var_power(jnp.asarray(v)), n_iter=40,
    )
    assert res.weights == 'varPower' and res.corr == 'ar1'
    assert abs(float(res.rho[0]) - np.tanh(r.x[0])) < 5e-3
    assert abs(float(res.var_params[0, 0]) - r.x[1]) < 5e-3
    assert res.var_params.shape == (1, 1)


def test_varident_matches_dense_reml():
    """A pure heteroscedastic GLS (iid correlation, per-stratum variance)."""
    rng = np.random.default_rng(5)
    G, n_per = 20, 8
    group = np.repeat(np.arange(G), n_per).astype(np.int32)
    N = G * n_per
    X = np.ones((N, 2))
    X[:, 1] = rng.standard_normal(N)
    strata = (rng.uniform(size=N) < 0.5).astype(np.int32)
    g_t = np.where(strata == 1, np.exp(0.8), 1.0)
    y = X @ np.array([1.0, 0.5]) + rng.standard_normal(N) * np.sqrt(0.5) * g_t

    def neg2(tau):
        d = np.exp(np.where(strata == 1, tau, 0.0))
        Ri = 1.0 / d**2
        XtVX = (X * Ri[:, None]).T @ X
        XtVy = (X * Ri[:, None]).T @ y
        yVy = float(np.sum(Ri * y * y))
        b = np.linalg.solve(XtVX, XtVy)
        return (
            (N - 2) * np.log(yVy - b @ XtVy)
            + np.sum(np.log(d**2))
            + np.linalg.slogdet(XtVX)[1]
        )

    ref = minimize_scalar(neg2, bounds=(-3, 3), method='bounded').x
    res = gls_fit(
        jnp.asarray(y[None, :]), jnp.asarray(X), group=jnp.asarray(group),
        corr='iid', weights=var_ident(jnp.asarray(strata)), n_iter=40,
    )
    assert res.weights == 'varIdent' and res.corr == 'iid'
    assert abs(float(res.var_params[0, 0]) - ref) < 5e-3


def test_varpower_recovers_heteroscedasticity():
    """A strong variance gradient is recovered (delta near the simulated 1.0)."""
    rng = np.random.default_rng(7)
    G, n_per = 30, 10
    group = np.repeat(np.arange(G), n_per).astype(np.int32)
    N = G * n_per
    X = np.ones((N, 2))
    X[:, 1] = rng.standard_normal(N)
    v = np.abs(rng.standard_normal(N)) + 0.5
    y = X @ np.array([1.0, 0.5]) + rng.standard_normal(N) * (v**1.0) * 0.6
    res = gls_fit(
        jnp.asarray(y[None, :]), jnp.asarray(X), group=jnp.asarray(group),
        corr='iid', weights=var_power(jnp.asarray(v)), n_iter=40,
    )
    assert abs(float(res.var_params[0, 0]) - 1.0) < 0.25


def test_varident_three_strata_shapes():
    rng = np.random.default_rng(9)
    N = 240
    group = np.repeat(np.arange(30), 8).astype(np.int32)
    X = np.ones((N, 2))
    X[:, 1] = rng.standard_normal(N)
    strata = rng.integers(0, 3, N).astype(np.int32)
    y = rng.standard_normal((4, N))
    res = gls_fit(
        jnp.asarray(y), jnp.asarray(X), group=jnp.asarray(group),
        corr='iid', weights=var_ident(jnp.asarray(strata)), n_iter=30,
    )
    assert res.var_params.shape == (4, 2)  # S - 1 = 2 free params
    assert res.beta_hat.shape == (4, 2)


def test_gls_iid_no_weights_is_rejected():
    """'iid' with no variance function is OLS -- not a GLS fit."""
    rng = np.random.default_rng(0)
    N = 80
    group = np.repeat(np.arange(10), 8).astype(np.int32)
    X = np.ones((N, 2))
    X[:, 1] = rng.standard_normal(N)
    y = rng.standard_normal((1, N))
    with pytest.raises(ValueError, match='ordinary least squares'):
        gls_fit(jnp.asarray(y), jnp.asarray(X), group=jnp.asarray(group), corr='iid')
