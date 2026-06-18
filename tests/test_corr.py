# -*- coding: utf-8 -*-
"""Tests for ``nitrix.stats.lme`` error-correlation structures (v3 §1.4).

The GLS-REML fit (``gls_fit``) with an ``ar1`` / ``car1`` / ``cs`` residual is
anchored on an **exact dense profile-REML reference** (build the per-group
correlation matrix ``R(rho)`` densely, minimise the same profile REML criterion
with scipy): same estimator, so beta / rho / sigma_e^2 must agree tightly.  A
global single-group AR(1) is additionally sanity-checked against statsmodels
``GLSAR``.  The whitening identity (``W R W^T = I``, ``|W|^2 |R| = 1``) is tested
directly on the ``CorrSpec``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import minimize_scalar

jax.config.update('jax_enable_x64', True)

from nitrix.stats.lme import ar1, car1, cs, gls_fit
from nitrix.stats.lme._corr import resolve_corr
from nitrix.stats.lme._corrfit import build_group_layout

# ---------------------------------------------------------------------------
# Exact dense profile-REML reference (same estimator as gls_fit)
# ---------------------------------------------------------------------------


def _R(kind, rho, n, t):
    if kind == 'ar1':
        return rho ** np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    if kind == 'car1':
        return rho ** np.abs(np.subtract.outer(t, t))
    return (1 - rho) * np.eye(n) + rho * np.ones((n, n))


def _to_rho(kind, raw):
    return np.tanh(raw) if kind == 'ar1' else 1.0 / (1.0 + np.exp(-raw))


def _dense_reml(y, X, groups, times, kind):
    n_obs, p = X.shape
    uniq = np.unique(groups)

    def grams(rho):
        xvx = np.zeros((p, p))
        xvy = np.zeros(p)
        yvy = 0.0
        logdet_r = 0.0
        for g in uniq:
            m = groups == g
            o = np.argsort(times[m])
            Xg, yg, tg = X[m][o], y[m][o], times[m][o]
            R = _R(kind, rho, len(yg), tg)
            Ri = np.linalg.inv(R)
            _, ld = np.linalg.slogdet(R)
            logdet_r += ld
            xvx += Xg.T @ Ri @ Xg
            xvy += Xg.T @ Ri @ yg
            yvy += yg @ Ri @ yg
        return xvx, xvy, yvy, logdet_r

    def neg2(raw):
        xvx, xvy, yvy, logdet_r = grams(_to_rho(kind, raw))
        beta = np.linalg.solve(xvx, xvy)
        rss = yvy - beta @ xvy
        _, ldx = np.linalg.slogdet(xvx)
        return (n_obs - p) * np.log(rss) + logdet_r + ldx

    raw = minimize_scalar(neg2, bounds=(-6, 6), method='bounded').x
    rho = _to_rho(kind, raw)
    xvx, xvy, yvy, _ = grams(rho)
    beta = np.linalg.solve(xvx, xvy)
    sig2 = (yvy - beta @ xvy) / (n_obs - p)
    return beta, sig2, rho


def _simulate(kind, *, seed, G=20, n_per=8, rho_t=0.5, sigma=0.7):
    rng = np.random.default_rng(seed)
    N = G * n_per
    groups = np.repeat(np.arange(G), n_per).astype(np.int32)
    if kind == 'car1':
        times = np.tile(np.sort(rng.uniform(0, 10, n_per)), G)
    else:
        times = np.tile(np.arange(n_per), G).astype(float)
    X = np.ones((N, 2))
    X[:, 1] = rng.standard_normal(N)
    y = np.empty(N)
    for g in range(G):
        m = groups == g
        o = np.argsort(times[m])
        n = int(m.sum())
        R = _R(kind, rho_t, n, times[m][o])
        eg = np.linalg.cholesky(R) @ rng.standard_normal(n) * sigma
        idxs = np.where(m)[0][o]
        y[idxs] = X[m][o] @ np.array([1.0, 0.5]) + eg
    return X, groups, times, y


# ---------------------------------------------------------------------------
# gls_fit vs exact dense REML
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('kind', ['ar1', 'car1', 'cs'])
def test_gls_matches_dense_reml(kind):
    X, groups, times, y = _simulate(kind, seed=2)
    rb, rs, rr = _dense_reml(y, X, groups, times, kind)
    use_t = jnp.asarray(times) if kind == 'car1' else None
    g = gls_fit(
        jnp.asarray(y[None, :]),
        jnp.asarray(X),
        group=jnp.asarray(groups),
        corr=kind,
        time=use_t,
    )
    assert np.allclose(np.asarray(g.beta_hat[0]), rb, atol=1e-3)
    assert abs(float(g.rho[0]) - rr) < 2e-3
    assert abs(float(g.sigma_e_sq[0]) - rs) < 1e-3


def test_gls_ar1_vs_statsmodels_glsar():
    """A single-group (global) AR(1) GLS is in the ballpark of statsmodels
    GLSAR (a different estimator -- FGLS/Yule-Walker rho -- so loose)."""
    sm = pytest.importorskip('statsmodels.api')
    rng = np.random.default_rng(0)
    N = 150
    X = np.ones((N, 2))
    X[:, 1] = rng.standard_normal(N)
    rho_t, sig = 0.6, 0.7
    e = np.zeros(N)
    e[0] = rng.standard_normal() * sig / np.sqrt(1 - rho_t**2)
    for t in range(1, N):
        e[t] = rho_t * e[t - 1] + rng.standard_normal() * sig
    y = X @ np.array([1.0, 0.5]) + e
    group = np.zeros(N, dtype=np.int32)
    g = gls_fit(
        jnp.asarray(y[None, :]),
        jnp.asarray(X),
        group=jnp.asarray(group),
        corr='ar1',
    )
    res = sm.GLSAR(y, X, rho=1).iterative_fit(maxiter=20)
    assert np.allclose(np.asarray(g.beta_hat[0]), res.params, atol=0.05)
    assert abs(float(g.rho[0]) - rho_t) < 0.12


# ---------------------------------------------------------------------------
# Whitening identity on the CorrSpec
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'kind,raw', [('ar1', 0.7), ('car1', 0.4), ('cs', 0.5)]
)
def test_whiten_identity(kind, raw):
    """W R W^T = I and the half-log-det equals 0.5 log|R| for one group."""
    spec = resolve_corr(kind)
    n = 6
    t = np.cumsum(
        np.r_[0.0, np.abs(np.random.default_rng(1).normal(1, 0.2, n - 1))]
    )
    layout = build_group_layout(
        jnp.zeros(n, dtype=jnp.int32),
        jnp.asarray(t) if kind == 'car1' else None,
    )
    rho = _to_rho(kind, raw)
    R = _R(kind, rho, n, t)
    raw_arr = jnp.asarray([raw], dtype=jnp.float64)
    # Whiten each standard-basis column: the stacked whitened design is W.
    eye = jnp.asarray(np.eye(n))[None].transpose(
        0, 1, 2
    )  # (1, n, n) = (G,T,k)
    w_pad, half_logdet = spec.whiten(
        eye, layout.gaps, layout.nsize, layout.mask, raw_arr
    )
    W = np.asarray(w_pad[0])  # (n, n): columns are W e_j -> W itself
    assert np.allclose(W @ R @ W.T, np.eye(n), atol=1e-9)
    _, ld = np.linalg.slogdet(R)
    assert abs(float(half_logdet) - 0.5 * ld) < 1e-9


# ---------------------------------------------------------------------------
# Layout, shapes, mass-univariate, validation
# ---------------------------------------------------------------------------


def test_gls_ragged_groups_and_shapes():
    """Unequal group sizes (ragged padded layout) and a multi-voxel batch."""
    rng = np.random.default_rng(4)
    sizes = [5, 9, 7, 6]
    groups = np.concatenate(
        [np.full(s, i) for i, s in enumerate(sizes)]
    ).astype(np.int32)
    N = len(groups)
    X = np.ones((N, 2))
    X[:, 1] = rng.standard_normal(N)
    Y = rng.standard_normal((5, N))
    g = gls_fit(
        jnp.asarray(Y), jnp.asarray(X), group=jnp.asarray(groups), corr='ar1'
    )
    assert g.beta_hat.shape == (5, 2)
    assert g.fixed_cov.shape == (5, 2, 2)
    assert g.df_resid == N - 2
    assert g.corr == 'ar1'
    assert np.all(np.asarray(g.sigma_e_sq) > 0)


def test_gls_unknown_corr_raises():
    with pytest.raises(ValueError, match='unknown correlation'):
        resolve_corr('nope')


def test_corr_constructors():
    """The public constructors build the named CorrSpec; a spec instance passes
    through ``corr=`` unchanged."""
    assert ar1().name == 'ar1' and ar1().n_params == 1
    assert car1().name == 'car1'
    assert cs().name == 'cs'
    rng = np.random.default_rng(0)
    N = 40
    X = np.ones((N, 2))
    X[:, 1] = rng.standard_normal(N)
    y = rng.standard_normal(N)
    group = np.repeat(np.arange(4), 10).astype(np.int32)
    g = gls_fit(
        jnp.asarray(y[None, :]),
        jnp.asarray(X),
        group=jnp.asarray(group),
        corr=ar1(),  # pass a CorrSpec instance directly
    )
    assert g.corr == 'ar1'
