# -*- coding: utf-8 -*-
"""Tests for the Kenward-Roger small-sample correction (v3 §1.3).

``lme_{t,f}_contrast(..., dof='kr', X=, Z=)`` applies the Kenward & Roger (1997)
adjusted covariance ``Phi_A`` and scaled-``F`` denominator df.  There is no
Python oracle (``pbkrtest`` is R-only), so the fit is anchored two ways:

- an **independent dense KR reference** (build the full ``V = sb2 ZZ^T + se2 I``,
  REML-fit, and apply the KR formulas in plain NumPy) -- the authoritative
  same-estimator check;
- **exact-ANOVA-df properties** KR is designed to reproduce for balanced designs:
  a between-group covariate gets ``df = G - 2``; a within-group covariate gets
  ``df ~ N - G``; and KR inflates the naive Wald SE.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import minimize

jax.config.update('jax_enable_x64', True)

from nitrix.stats.lme import lme_f_contrast, lme_t_contrast, reml_fit


def _dense_kr(y, X, Z, L):
    """Independent dense Kenward-Roger reference. Returns (PhiA, F_kr, df2)."""
    N, p = X.shape
    ZZt = Z @ Z.T
    V1 = [ZZt, np.eye(N)]

    def negreml(th):
        sb2, se2 = np.exp(th)
        V = sb2 * ZZt + se2 * np.eye(N)
        Vi = np.linalg.inv(V)
        XtViX = X.T @ Vi @ X
        beta = np.linalg.solve(XtViX, X.T @ Vi @ y)
        r = y - X @ beta
        return 0.5 * (
            np.linalg.slogdet(V)[1]
            + np.linalg.slogdet(XtViX)[1]
            + r @ Vi @ r
        )

    th = minimize(
        negreml, np.log([y.var() / 2, y.var() / 2]), method='Nelder-Mead',
        options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 12000},
    ).x
    sb2, se2 = np.exp(th)
    V = sb2 * ZZt + se2 * np.eye(N)
    Vi = np.linalg.inv(V)
    Phi = np.linalg.inv(X.T @ Vi @ X)
    beta = Phi @ (X.T @ Vi @ y)
    P = Vi - Vi @ X @ Phi @ X.T @ Vi
    info = np.array(
        [[0.5 * np.trace(P @ V1[i] @ P @ V1[j]) for j in range(2)] for i in range(2)]
    )
    W = np.linalg.inv(info)
    Pm = [X.T @ Vi @ V1[i] @ Vi @ X for i in range(2)]
    Q = [[X.T @ Vi @ V1[i] @ Vi @ V1[j] @ Vi @ X for j in range(2)] for i in range(2)]
    corr = sum(
        W[i, j] * (Q[i][j] - Pm[i] @ Phi @ Pm[j])
        for i in range(2)
        for j in range(2)
    )
    PhiA = Phi + 2 * Phi @ corr @ Phi
    q = L.shape[0]
    M = L @ PhiA @ L.T
    Minv = np.linalg.inv(M)
    Pt = [Phi @ Pm[i] @ Phi for i in range(2)]
    A1 = A2 = 0.0
    for i in range(2):
        ti = Minv @ (L @ Pt[i] @ L.T)
        for j in range(2):
            tj = Minv @ (L @ Pt[j] @ L.T)
            A1 += W[i, j] * np.trace(ti) * np.trace(tj)
            A2 += W[i, j] * np.trace(ti @ tj)
    B = (A1 + 6 * A2) / (2 * q)
    g = ((q + 1) * A1 - (q + 4) * A2) / ((q + 2) * A2)
    den = 3 * q + 2 * (1 - g)
    c1, c2, c3 = g / den, (q - g) / den, (q + 2 - g) / den
    Estar = 1 / (1 - A2 / q)
    Vstar = (2 / q) * (1 + c1 * B) / ((1 - c2 * B) ** 2 * (1 - c3 * B))
    rho = Vstar / (2 * Estar**2)
    m = 4 + (q + 2) / (q * rho - 1)
    lam = m / (Estar * (m - 2))
    Fwald = (L @ beta) @ Minv @ (L @ beta) / q
    return PhiA, lam * Fwald, m


def _sim(seed, *, between, G=10, n_per=8, se=0.5, sb=np.sqrt(0.6)):
    rng = np.random.default_rng(seed)
    N = G * n_per
    group = np.repeat(np.arange(G), n_per)
    Z = np.eye(G)[group].astype(float)
    X = np.ones((N, 2))
    if between:
        X[:, 1] = rng.standard_normal(G)[group]
    else:
        X[:, 1] = rng.standard_normal(N)
    b = rng.standard_normal(G) * sb
    y = X @ np.array([1.0, 0.5]) + b[group] + rng.standard_normal(N) * se
    return X, Z, y, group


def test_kr_matches_dense_reference():
    """JAX dof='kr' reproduces the independent dense-numpy KR (F, df, Phi_A)."""
    X, Z, y, _ = _sim(1, between=False, G=8, n_per=9)
    L = np.array([[0.0, 1.0]])
    PhiA, F_ref, df_ref = _dense_kr(y, X, Z, L)
    res = reml_fit(jnp.asarray(y[None, :]), jnp.asarray(X), jnp.asarray(Z), n_iter=60)
    fk = lme_f_contrast(
        res, jnp.asarray(L), dof='kr', X=jnp.asarray(X), Z=jnp.asarray(Z)
    )
    assert abs(float(fk.f[0]) - F_ref) < 1e-3 * max(F_ref, 1.0)
    assert abs(float(fk.df2[0]) - df_ref) < 5e-3 * df_ref


def test_kr_between_group_exact_df():
    """Balanced between-group covariate -> KR df = G - 2 (exact ANOVA df)."""
    X, Z, y, _ = _sim(2, between=True, G=12, n_per=7)
    res = reml_fit(jnp.asarray(y[None, :]), jnp.asarray(X), jnp.asarray(Z), n_iter=80)
    tk = lme_t_contrast(
        res, jnp.asarray([0.0, 1.0]), dof='kr', X=jnp.asarray(X), Z=jnp.asarray(Z)
    )
    assert abs(float(tk.df[0]) - (12 - 2)) < 0.2


def test_kr_within_group_large_df():
    """Within-group covariate -> KR df ~ N - G (lots of residual information)."""
    X, Z, y, _ = _sim(3, between=False, G=10, n_per=8)
    res = reml_fit(jnp.asarray(y[None, :]), jnp.asarray(X), jnp.asarray(Z), n_iter=60)
    tk = lme_t_contrast(
        res, jnp.asarray([0.0, 1.0]), dof='kr', X=jnp.asarray(X), Z=jnp.asarray(Z)
    )
    assert abs(float(tk.df[0]) - (80 - 10)) < 3.0


def test_kr_t_f_consistency():
    """For a single-row contrast (q=1) the KR F equals t^2 on the same df."""
    X, Z, y, _ = _sim(4, between=True, G=10, n_per=8)
    Xj, Zj = jnp.asarray(X), jnp.asarray(Z)
    res = reml_fit(jnp.asarray(np.tile(y, (3, 1))), Xj, Zj, n_iter=70)
    c = jnp.asarray([0.0, 1.0])
    tk = lme_t_contrast(res, c, dof='kr', X=Xj, Z=Zj)
    fk = lme_f_contrast(res, c[None, :], dof='kr', X=Xj, Z=Zj)
    assert np.allclose(np.asarray(fk.f), np.asarray(tk.t) ** 2, atol=1e-7)
    assert np.allclose(np.asarray(fk.df2), np.asarray(tk.df), atol=1e-7)
    assert np.allclose(tk.df[0], tk.df[2])  # identical across identical voxels


def test_kr_inflates_se_vs_naive():
    """KR inflates the naive Wald SE (Phi_A >= Phi) for a small-sample design."""
    X, Z, y, _ = _sim(5, between=False, G=6, n_per=5, sb=np.sqrt(1.0))
    Xj, Zj = jnp.asarray(X), jnp.asarray(Z)
    res = reml_fit(jnp.asarray(y[None, :]), Xj, Zj, n_iter=80)
    c = jnp.asarray([0.0, 1.0])
    naive_se = float(jnp.sqrt(res.fixed_cov[0, 1, 1]))
    kr_se = float(lme_t_contrast(res, c, dof='kr', X=Xj, Z=Zj).se[0])
    assert kr_se >= naive_se - 1e-9


def test_kr_multi_df_contrast():
    """A 2-row KR F-contrast (q=2) runs and matches the dense reference."""
    rng = np.random.default_rng(6)
    G, n_per = 9, 8
    N = G * n_per
    group = np.repeat(np.arange(G), n_per)
    Z = np.eye(G)[group].astype(float)
    X = np.c_[np.ones(N), rng.standard_normal(N), rng.standard_normal(G)[group]]
    b = rng.standard_normal(G) * np.sqrt(0.5)
    y = X @ np.array([1.0, 0.4, 0.3]) + b[group] + rng.standard_normal(N) * 0.5
    L = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    _, F_ref, df_ref = _dense_kr(y, X, Z, L)
    res = reml_fit(jnp.asarray(y[None, :]), jnp.asarray(X), jnp.asarray(Z), n_iter=80)
    fk = lme_f_contrast(
        res, jnp.asarray(L), dof='kr', X=jnp.asarray(X), Z=jnp.asarray(Z)
    )
    assert float(fk.df1[0]) == 2.0
    assert abs(float(fk.f[0]) - F_ref) < 1e-3 * max(F_ref, 1.0)
    assert abs(float(fk.df2[0]) - df_ref) < 1e-2 * df_ref


def test_kr_requires_design_and_validates_dof():
    X, Z, y, _ = _sim(7, between=True)
    res = reml_fit(jnp.asarray(y[None, :]), jnp.asarray(X), jnp.asarray(Z), n_iter=40)
    c = jnp.asarray([0.0, 1.0])
    with pytest.raises(ValueError, match='needs the original design'):
        lme_t_contrast(res, c, dof='kr')
    with pytest.raises(ValueError, match='needs the original design'):
        lme_f_contrast(res, c[None, :], dof='kr')
    with pytest.raises(ValueError, match='satterthwaite'):
        lme_t_contrast(res, c, dof='bogus')
