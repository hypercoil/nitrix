# -*- coding: utf-8 -*-
"""Tests for non-Gaussian random *slopes* in ``glmm_fit`` (z= / structure=).

Validation triangle:

- **Gaussian == LME**: a Gaussian-family random-slope GLMM is the linear mixed
  model, so both the diagonal (``(x || g)``) and unstructured (``(1 + x | g)``)
  fits must match ``lme_fit(z=, structure=)`` -- the exact block-Woodbury oracle
  (itself validated against statsmodels MixedLM).
- **r == 1 reduces to the scalar intercept**: an unstructured slope fit with a
  single ones column ``z`` is the random-intercept GLMM, so it must reproduce the
  shipped scalar-intercept path under Poisson / binomial.
- **diagonal data -> small off-diagonal**: an unstructured fit on data simulated
  with an uncorrelated ``G`` recovers a near-zero intercept-slope correlation.

Plus parameter recovery vs simulation truth, shapes / pytree, and validation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update('jax_enable_x64', True)

from nitrix.stats import GLMMResult, glmm_fit
from nitrix.stats.lme import lme_fit


def _sim_slope(link, *, seed, q, n_per, G, beta=(0.2, 0.6), noise=0.5):
    """Random intercept + slope GLMM: ``b_g ~ N(0, G)`` over ``z = [1, x]``."""
    rng = np.random.default_rng(seed)
    group = np.repeat(np.arange(q), n_per).astype(np.int32)
    n = q * n_per
    x = rng.standard_normal(n)
    X = np.c_[np.ones(n), x]
    z = np.c_[np.ones(n), x]
    b = rng.standard_normal((q, 2)) @ np.linalg.cholesky(np.asarray(G)).T
    eta = X @ np.asarray(beta) + np.einsum('nr,nr->n', z, b[group])
    if link == 'poisson':
        y = rng.poisson(np.exp(eta)).astype(float)
    elif link == 'binomial':
        y = rng.binomial(1, 1.0 / (1.0 + np.exp(-eta))).astype(float)
    else:
        y = eta + rng.standard_normal(n) * noise
    return X, z, group, y, b


# ---------------------------------------------------------------------------
# Gaussian == LME (the exact block-Woodbury oracle)
# ---------------------------------------------------------------------------


def test_gaussian_diagonal_slope_matches_lme():
    G = np.diag([0.5, 0.3])
    X, z, group, y, _ = _sim_slope('gaussian', seed=0, q=14, n_per=20, G=G)
    Y = jnp.asarray(np.tile(y, (3, 1)))
    Xj, zj, gj = jnp.asarray(X), jnp.asarray(z), jnp.asarray(group)
    g = glmm_fit(
        Y, Xj, group=gj, z=zj, structure='diagonal', family='gaussian'
    )
    r = lme_fit(Y, Xj, z=zj, group=gj, structure='diagonal')
    assert g.tier == 'few'
    assert np.allclose(g.beta_hat, r.beta_hat, rtol=1e-3, atol=1e-3)
    # re_var is now a uniform (V, r, r) diagonal G (D4), as is lme's diagonal
    # cov_re; compare the full matrices (off-diagonals are zero in both).
    assert g.re_var.shape == r.cov_re.shape
    assert np.allclose(g.re_var, r.cov_re, rtol=2e-2, atol=3e-3)
    assert np.allclose(g.dispersion, r.sigma_e_sq, rtol=2e-2, atol=3e-3)


def test_gaussian_unstructured_slope_matches_lme():
    G = np.array([[0.6, 0.2], [0.2, 0.4]])
    X, z, group, y, _ = _sim_slope('gaussian', seed=1, q=16, n_per=18, G=G)
    Y = jnp.asarray(np.tile(y, (3, 1)))
    Xj, zj, gj = jnp.asarray(X), jnp.asarray(z), jnp.asarray(group)
    g = glmm_fit(
        Y, Xj, group=gj, z=zj, structure='unstructured', family='gaussian'
    )
    r = lme_fit(Y, Xj, z=zj, group=gj, structure='unstructured')
    assert g.tier == 'slope'
    # The REML-EM solver converges to lme_fit's REML optimum (both REML); the
    # tight tolerances pin that it is the *same* estimator, not just close.
    assert np.allclose(g.beta_hat, r.beta_hat, rtol=1e-4, atol=1e-5)
    assert np.allclose(g.re_var, r.cov_re, rtol=1e-3, atol=1e-5)
    assert np.allclose(g.dispersion, r.sigma_e_sq, rtol=1e-3, atol=1e-5)


def test_structured_slope_clamp_insensitive():
    """The REML-EM structured-slope solver is monotone, so the fit no longer
    depends on the IRLS eta clamp landing it in the right basin: the recovered G
    is identical across a wide range of eta_bound (the earlier iterated-Newton-
    REML degenerated to intercept-variance -> 0 at a looser clamp).  The clamp is
    now pure overflow safety for this path."""
    from dataclasses import replace

    from nitrix.stats import POISSON

    G = np.array([[0.5, 0.25], [0.25, 0.4]])
    X, z, group, y, _ = _sim_slope(
        'poisson', seed=5, q=70, n_per=15, G=G, beta=(0.3, 0.6)
    )
    Yj, Xj, zj, gj = (
        jnp.asarray(y[None, :]), jnp.asarray(X), jnp.asarray(z),
        jnp.asarray(group),
    )

    def fit(bound):
        return np.asarray(
            glmm_fit(
                Yj, Xj, group=gj, z=zj, structure='unstructured',
                family=replace(POISSON, eta_bound=bound), n_outer=60,
            ).re_var[0]
        )

    np.testing.assert_allclose(fit(20.0), fit(float('inf')), rtol=1e-4, atol=1e-5)


# ---------------------------------------------------------------------------
# r == 1 reduces to the scalar random-intercept GLMM (non-Gaussian)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('link', ['poisson', 'binomial'])
def test_unstructured_r1_matches_scalar_intercept(link):
    rng = np.random.default_rng(7)
    q, n_per = 18, 20
    group = np.repeat(np.arange(q), n_per).astype(np.int32)
    n = q * n_per
    X = np.c_[np.ones(n), rng.standard_normal(n)]
    b = rng.standard_normal(q) * np.sqrt(0.4)
    eta = X @ np.array([0.2, 0.5]) + b[group]
    if link == 'poisson':
        y = rng.poisson(np.exp(eta)).astype(float)
    else:
        y = rng.binomial(1, 1.0 / (1.0 + np.exp(-eta))).astype(float)
    Y = jnp.asarray(y[None, :])
    Xj, gj = jnp.asarray(X), jnp.asarray(group)
    z1 = jnp.ones((n, 1))
    slope = glmm_fit(
        Y, Xj, group=gj, z=z1, structure='unstructured', family=link,
        n_outer=30,
    )
    scalar = glmm_fit(Y, Xj, group=gj, family=link)
    assert slope.tier == 'slope'
    assert np.allclose(slope.beta_hat, scalar.beta_hat, rtol=1e-2, atol=3e-3)
    assert np.allclose(
        slope.re_var[:, 0, 0], scalar.re_var[:, 0, 0], rtol=3e-2, atol=5e-3
    )


# ---------------------------------------------------------------------------
# Unstructured fit on uncorrelated data recovers a small off-diagonal
# ---------------------------------------------------------------------------


def test_poisson_unstructured_on_diagonal_data():
    G = np.diag([0.4, 0.25])
    X, z, group, y, _ = _sim_slope(
        'poisson', seed=3, q=50, n_per=15, G=G, beta=(0.3, 0.5)
    )
    g = glmm_fit(
        jnp.asarray(y[None, :]), jnp.asarray(X), group=jnp.asarray(group),
        z=jnp.asarray(z), structure='unstructured', family='poisson',
        n_outer=30,
    )
    Ghat = np.asarray(g.re_var[0])
    corr = Ghat[0, 1] / np.sqrt(Ghat[0, 0] * Ghat[1, 1])
    assert abs(corr) < 0.35
    assert abs(Ghat[0, 0] - 0.4) < 0.25
    assert abs(Ghat[1, 1] - 0.25) < 0.2


# ---------------------------------------------------------------------------
# Parameter recovery (correlated slope, non-Gaussian)
# ---------------------------------------------------------------------------


def test_poisson_correlated_slope_recovery():
    G = np.array([[0.5, 0.25], [0.25, 0.4]])
    X, z, group, y, _ = _sim_slope(
        'poisson', seed=5, q=70, n_per=15, G=G, beta=(0.3, 0.6)
    )
    g = glmm_fit(
        jnp.asarray(y[None, :]), jnp.asarray(X), group=jnp.asarray(group),
        z=jnp.asarray(z), structure='unstructured', family='poisson',
        n_outer=40,
    )
    assert abs(float(g.beta_hat[0, 1]) - 0.6) < 0.15
    Ghat = np.asarray(g.re_var[0])
    assert abs(Ghat[0, 0] - 0.5) < 0.3
    assert abs(Ghat[1, 1] - 0.4) < 0.3
    assert Ghat[0, 1] > 0.0  # positive correlation recovered


# ---------------------------------------------------------------------------
# Shapes, pytree, validation
# ---------------------------------------------------------------------------


def test_slope_shapes_and_pytree():
    G = np.diag([0.4, 0.3])
    X, z, group, y, _ = _sim_slope('poisson', seed=2, q=10, n_per=12, G=G)
    V = 4
    Y = jnp.asarray(np.tile(y, (V, 1)))
    Xj, zj, gj = jnp.asarray(X), jnp.asarray(z), jnp.asarray(group)
    gd = glmm_fit(Y, Xj, group=gj, z=zj, structure='diagonal', family='poisson')
    assert gd.blups.shape == (V, 10, 2)
    # D4: diagonal G is the uniform (V, r, r) shape with zero off-diagonals.
    assert gd.re_var.shape == (V, 2, 2)
    assert np.allclose(gd.re_var[:, 0, 1], 0.0) and np.allclose(
        gd.re_var[:, 1, 0], 0.0
    )
    gu = glmm_fit(
        Y, Xj, group=gj, z=zj, structure='unstructured', family='poisson'
    )
    assert gu.blups.shape == (V, 10, 2)
    assert gu.re_var.shape == (V, 2, 2)
    leaves, treedef = jax.tree_util.tree_flatten(gu)
    gu2 = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(gu2, GLMMResult) and gu2.tier == 'slope'


def test_slope_validation():
    G = np.diag([0.3, 0.3])
    X, z, group, y, _ = _sim_slope('poisson', seed=0, q=8, n_per=10, G=G)
    Y = jnp.asarray(y[None, :])
    Xj, gj = jnp.asarray(X), jnp.asarray(group)
    with pytest.raises(ValueError, match='z must be'):
        glmm_fit(Y, Xj, group=gj, z=jnp.asarray(z[:-3]), family='poisson')
    with pytest.raises(ValueError, match='structure'):
        glmm_fit(
            Y, Xj, group=gj, z=jnp.asarray(z), structure='bogus',
            family='poisson',
        )
    with pytest.raises(ValueError, match='method'):
        glmm_fit(
            Y, Xj, group=gj, z=jnp.asarray(z), method='bogus',
            family='poisson',
        )


# ---------------------------------------------------------------------------
# Laplace random slope (method='laplace') -- the r-dimensional mode integral
# ---------------------------------------------------------------------------


def test_laplace_slope_r1_matches_scalar_intercept_laplace():
    """An r=1 Laplace slope (z = ones) is the scalar random-intercept Laplace
    fit -- the r-dimensional mode / determinant correction must reduce to it."""
    rng = np.random.default_rng(1)
    q, n_per = 40, 12
    group = np.repeat(np.arange(q), n_per).astype(np.int32)
    n = q * n_per
    X = np.c_[np.ones(n), rng.standard_normal(n)]
    b = rng.standard_normal(q) * np.sqrt(0.5)
    eta = X @ np.array([0.0, 0.8]) + b[group]
    y = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-eta))).astype(float)
    Y, Xj, gj = jnp.asarray(y[None, :]), jnp.asarray(X), jnp.asarray(group)
    slope = glmm_fit(
        Y, Xj, group=gj, z=jnp.ones((n, 1)), structure='unstructured',
        family='binomial', method='laplace', n_outer=40,
    )
    scalar = glmm_fit(
        Y, Xj, group=gj, family='binomial', method='laplace', n_outer=40,
    )
    assert slope.tier == 'laplace'
    np.testing.assert_allclose(slope.beta_hat[0], scalar.beta_hat[0], atol=1e-5)
    np.testing.assert_allclose(
        float(slope.re_var[0, 0, 0]), float(scalar.re_var[0, 0, 0]), atol=1e-5
    )


def _ref_laplace_slope_binomial(y, X, z, group, q, r, n_mode=40):
    """scipy-minimised r-dimensional Laplace marginal -- the same estimator,
    independent (numpy) linear algebra."""
    from scipy.optimize import minimize

    p = X.shape[1]
    tril = [(i, j) for i in range(r) for j in range(i + 1)]

    def build_G(chol):
        L = np.zeros((r, r))
        for k, (i, j) in enumerate(tril):
            L[i, j] = np.exp(chol[k]) if i == j else chol[k]
        return L @ L.T

    def nll(params):
        beta, G = params[:p], build_G(params[p:])
        Ginv = np.linalg.inv(G)
        etaf = X @ beta
        b = np.zeros((q, r))
        for _ in range(n_mode):
            eta = etaf + np.einsum('nr,nr->n', z, b[group])
            mu = 1.0 / (1.0 + np.exp(-eta))
            w = mu * (1.0 - mu)
            for g in range(q):
                m = group == g
                grad = z[m].T @ (y[m] - mu[m]) - Ginv @ b[g]
                H = z[m].T @ (w[m, None] * z[m]) + Ginv
                b[g] = b[g] + np.linalg.solve(H, grad)
        eta = etaf + np.einsum('nr,nr->n', z, b[group])
        mu = 1.0 / (1.0 + np.exp(-eta))
        w = mu * (1.0 - mu)
        _, logdetG = np.linalg.slogdet(G)
        out = 0.0
        for g in range(q):
            m = group == g
            ll = np.sum(y[m] * eta[m] - np.log1p(np.exp(eta[m])))
            H = z[m].T @ (w[m, None] * z[m]) + Ginv
            _, logdetH = np.linalg.slogdet(H)
            out -= ll - 0.5 * b[g] @ Ginv @ b[g] - 0.5 * logdetG - 0.5 * logdetH
        return out

    res = minimize(
        nll, np.r_[np.zeros(p), np.zeros(len(tril))], method='Nelder-Mead',
        options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 8000},
    )
    return res.x[:p], build_G(res.x[p:])


def test_laplace_slope_matches_scipy_reference():
    """r=2 correlated Laplace slope matches the scipy-minimised Laplace
    marginal (independent numpy reference)."""
    G = np.array([[0.6, 0.2], [0.2, 0.4]])
    X, z, group, y, _ = _sim_slope(
        'binomial', seed=2, q=25, n_per=10, G=G, beta=(0.1, 0.6)
    )
    rb, rG = _ref_laplace_slope_binomial(y, X, z, group, 25, 2)
    g = glmm_fit(
        jnp.asarray(y[None, :]), jnp.asarray(X), group=jnp.asarray(group),
        z=jnp.asarray(z), structure='unstructured', family='binomial',
        method='laplace', n_outer=60,
    )
    np.testing.assert_allclose(np.asarray(g.beta_hat[0]), rb, atol=2e-2)
    np.testing.assert_allclose(np.asarray(g.re_var[0]), rG, atol=4e-2)


def test_laplace_slope_beats_pql_attenuation():
    """For a binary random slope, PQL under-estimates the slope-variance
    component; Laplace recovers more of it (closer to the truth)."""
    G = np.diag([0.4, 0.6])
    X, z, group, y, _ = _sim_slope(
        'binomial', seed=4, q=60, n_per=14, G=G, beta=(0.0, 0.5)
    )
    Yj, Xj, zj, gj = (
        jnp.asarray(y[None, :]), jnp.asarray(X), jnp.asarray(z),
        jnp.asarray(group),
    )
    lap = glmm_fit(
        Yj, Xj, group=gj, z=zj, structure='unstructured', family='binomial',
        method='laplace', n_outer=60,
    )
    pql = glmm_fit(
        Yj, Xj, group=gj, z=zj, structure='unstructured', family='binomial',
        method='pql', n_outer=40,
    )
    slope_var_lap = float(lap.re_var[0, 1, 1])
    slope_var_pql = float(pql.re_var[0, 1, 1])
    assert slope_var_pql < slope_var_lap  # PQL more attenuated
    assert abs(slope_var_lap - 0.6) < abs(slope_var_pql - 0.6)


def _numpy_laplace_slope_nll(theta, y, X, z, group, q, r, link, n_mode):
    """Independent numpy evaluation of ``glmm._laplace_slope_nll`` for a binomial
    *non-canonical* link -- a Fisher-scoring mode + Fisher-curvature determinant
    term, the same approximation ``glmm.py`` makes (NOT the observed-Hessian
    Laplace), so it pins the link generalisation rather than re-deriving a
    different estimator."""
    from scipy.special import xlogy
    from scipy.stats import norm

    eps = 1e-10
    p = X.shape[1]
    beta = theta[:p]
    chol = theta[p:]
    tril = [(i, j) for i in range(r) for j in range(i + 1)]
    L = np.zeros((r, r))
    for k, (i, j) in enumerate(tril):
        L[i, j] = np.exp(chol[k]) if i == j else chol[k]
    G = L @ L.T
    Ginv = np.linalg.inv(G)
    _, logdetG = np.linalg.slogdet(G)

    def inv_link(eta):
        if link == 'probit':
            return norm.cdf(eta), norm.pdf(eta)
        ex = np.exp(eta)  # cloglog: mu = 1 - exp(-exp(eta))
        return -np.expm1(-ex), ex * np.exp(-ex)

    etaf = X @ beta
    b = np.zeros((q, r))
    for _ in range(n_mode):
        eta = etaf + np.einsum('nr,nr->n', z, b[group])
        mu, dmu = inv_link(eta)
        var = np.clip(mu * (1.0 - mu), eps, None)
        w = dmu * dmu / var
        score = (y - mu) * dmu / var
        for g in range(q):
            m = group == g
            grad = z[m].T @ score[m] - Ginv @ b[g]
            H = z[m].T @ (w[m, None] * z[m]) + Ginv
            b[g] = b[g] + np.linalg.solve(H, grad)
    eta = etaf + np.einsum('nr,nr->n', z, b[group])
    mu, dmu = inv_link(eta)
    var = np.clip(mu * (1.0 - mu), eps, None)
    w = dmu * dmu / var
    mc = np.clip(mu, eps, 1.0 - eps)
    out = 0.0
    for g in range(q):
        m = group == g
        ll = np.sum(xlogy(y[m], mc[m]) + xlogy(1.0 - y[m], 1.0 - mc[m]))
        H = z[m].T @ (w[m, None] * z[m]) + Ginv
        _, logdetH = np.linalg.slogdet(H)
        out -= ll - 0.5 * b[g] @ Ginv @ b[g] - 0.5 * logdetG - 0.5 * logdetH
    return out


@pytest.mark.parametrize('link', ['probit', 'cloglog'])
def test_laplace_slope_noncanonical_link_marginal_matches_numpy(link):
    """M4: the Laplace-marginal NLL for a non-canonical-link (probit / cloglog)
    random slope matches an independent numpy Fisher-scoring computation at
    arbitrary ``theta``.  The canonical (logit / log) tests cannot distinguish the
    Fisher-scoring Laplace ``glmm.py`` uses from the observed-Hessian Laplace
    (the two coincide there); a non-canonical link is where the choice bites, so
    this pins the determinant-correction curvature on that path."""
    from scipy.stats import norm

    from nitrix.stats import BINOMIAL
    from nitrix.stats.glmm._laplace import _laplace_slope_nll

    rng = np.random.default_rng(3)
    q, n_per, r, n_mode = 20, 10, 2, 25
    group = np.repeat(np.arange(q), n_per).astype(np.int32)
    n = q * n_per
    x = rng.standard_normal(n)
    X = np.c_[np.ones(n), x]
    zc = np.c_[np.ones(n), x]
    G = np.array([[0.5, 0.15], [0.15, 0.4]])
    b = rng.standard_normal((q, r)) @ np.linalg.cholesky(G).T
    # Keep eta moderate so cloglog's exp(exp(eta)) stays well in range.
    eta = X @ np.array([0.0, 0.4]) + np.einsum('nr,nr->n', zc, b[group])
    mu = norm.cdf(eta) if link == 'probit' else -np.expm1(-np.exp(eta))
    y = (rng.uniform(size=n) < mu).astype(float)

    fam = BINOMIAL.with_link(link)
    Xj, zj, gj, yj = (
        jnp.asarray(X), jnp.asarray(zc), jnp.asarray(group), jnp.asarray(y)
    )
    # A few thetas: beta = [b0, b1], chol = lower-tri (diag log) of G's factor.
    for theta in (
        np.array([0.0, 0.4, np.log(0.6), 0.1, np.log(0.5)]),
        np.array([0.2, 0.5, np.log(0.4), -0.2, np.log(0.7)]),
        np.array([-0.1, 0.3, np.log(0.8), 0.05, np.log(0.3)]),
    ):
        ref = _numpy_laplace_slope_nll(theta, y, X, zc, group, q, r, link, n_mode)
        got = float(
            _laplace_slope_nll(
                jnp.asarray(theta), yj, Xj, zj, gj, q, fam, 2, r, n_mode, False
            )
        )
        np.testing.assert_allclose(got, ref, rtol=1e-6, atol=1e-6)


def test_laplace_slope_shapes_and_diagonal():
    G = np.diag([0.4, 0.3])
    X, z, group, y, _ = _sim_slope('poisson', seed=2, q=12, n_per=12, G=G)
    V = 3
    Y = jnp.asarray(np.tile(y, (V, 1)))
    Xj, zj, gj = jnp.asarray(X), jnp.asarray(z), jnp.asarray(group)
    gu = glmm_fit(
        Y, Xj, group=gj, z=zj, structure='unstructured', family='poisson',
        method='laplace',
    )
    assert gu.tier == 'laplace'
    assert gu.blups.shape == (V, 12, 2) and gu.re_var.shape == (V, 2, 2)
    gd = glmm_fit(
        Y, Xj, group=gj, z=zj, structure='diagonal', family='poisson',
        method='laplace',
    )
    # D4: diagonal Laplace G is the uniform (V, r, r) shape.
    assert gd.re_var.shape == (V, 2, 2)
    assert np.allclose(gd.re_var[:, 0, 1], 0.0)
    leaves, treedef = jax.tree_util.tree_flatten(gu)
    gu2 = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(gu2, GLMMResult) and gu2.tier == 'laplace'


# ---------------------------------------------------------------------------
# Multi-voxel correctness (distinct y) + differentiability
# ---------------------------------------------------------------------------


def test_slope_multivoxel_matches_per_voxel():
    """Distinct y per voxel: the vmapped multi-voxel fit must equal independent
    single-voxel fits -- guards against per-voxel indexing / broadcast bugs that
    a tiled-identical-voxel test cannot catch.  Gaussian family so the fit is
    numerically stable (the Poisson slope path can amplify vmap-vs-scalar
    float-reordering near its sensitive optimum -- a separate, documented issue)."""
    rng = np.random.default_rng(11)
    q, n_per = 8, 12
    group = np.repeat(np.arange(q), n_per).astype(np.int32)
    n = q * n_per
    x = rng.standard_normal(n)
    X = np.c_[np.ones(n), x]
    z = np.c_[np.ones(n), x]
    L = np.linalg.cholesky(np.array([[0.4, 0.1], [0.1, 0.3]]))
    V = 3
    ys = []
    for v in range(V):
        b = rng.standard_normal((q, 2)) @ L.T
        eta = X @ np.array([0.2 + 0.1 * v, 0.5]) + np.einsum(
            'nr,nr->n', z, b[group]
        )
        ys.append(eta + rng.standard_normal(n) * 0.5)  # distinct Gaussian rows
    Y = jnp.asarray(np.stack(ys))  # (V, N)
    Xj, zj, gj = jnp.asarray(X), jnp.asarray(z), jnp.asarray(group)
    kw = dict(group=gj, z=zj, structure='unstructured', family='gaussian')
    multi = glmm_fit(Y, Xj, **kw)
    for v in range(V):
        one = glmm_fit(Y[v : v + 1], Xj, **kw)
        np.testing.assert_allclose(
            multi.beta_hat[v], one.beta_hat[0], atol=1e-6
        )
        np.testing.assert_allclose(multi.re_var[v], one.re_var[0], atol=1e-6)
        np.testing.assert_allclose(multi.blups[v], one.blups[0], atol=1e-6)


def test_slope_differentiable():
    """glmm_fit is differentiable through the structured-slope solver: the
    reverse-mode gradient matches finite differences."""
    rng = np.random.default_rng(0)
    q, n_per = 5, 8
    group = jnp.asarray(np.repeat(np.arange(q), n_per).astype(np.int32))
    n = q * n_per
    x = rng.standard_normal(n)
    X = jnp.asarray(np.c_[np.ones(n), x])
    z = jnp.asarray(np.c_[np.ones(n), x])
    y = jnp.asarray(rng.poisson(np.exp(0.3 + 0.5 * x)).astype(float))

    def loss(yv):
        g = glmm_fit(
            yv[None, :], X, group=group, z=z, structure='unstructured',
            family='poisson', n_outer=12,
        )
        return jnp.sum(g.beta_hat[0])

    gd = jax.grad(loss)(y)
    assert bool(jnp.all(jnp.isfinite(gd)))
    eps, i = 1e-5, 3
    fd = (loss(y.at[i].add(eps)) - loss(y.at[i].add(-eps))) / (2 * eps)
    np.testing.assert_allclose(float(gd[i]), float(fd), atol=1e-4)


# ---------------------------------------------------------------------------
# Adaptive Gauss-Hermite quadrature (method='agq') -- the tier above Laplace
# ---------------------------------------------------------------------------


def test_agq_slope_n1_matches_laplace():
    """AGQ with a single node is exactly the Laplace fit (the 1-point GH rule is
    the Laplace determinant correction)."""
    G = np.array([[0.6, 0.2], [0.2, 0.4]])
    X, z, group, y, _ = _sim_slope(
        'binomial', seed=2, q=25, n_per=10, G=G, beta=(0.1, 0.6)
    )
    Y, Xj, zj, gj = (
        jnp.asarray(y[None, :]), jnp.asarray(X), jnp.asarray(z),
        jnp.asarray(group),
    )
    kw = dict(
        group=gj, z=zj, structure='unstructured', family='binomial', n_outer=60
    )
    lap = glmm_fit(Y, Xj, method='laplace', **kw)
    agq1 = glmm_fit(Y, Xj, method='agq', n_quad=1, **kw)
    assert agq1.tier == 'agq'
    np.testing.assert_allclose(agq1.beta_hat[0], lap.beta_hat[0], atol=1e-5)
    np.testing.assert_allclose(agq1.re_var[0], lap.re_var[0], atol=1e-5)


def test_agq_slope_converges_and_corrects_laplace():
    """AGQ integrates the marginal exactly in the node limit: the fit stabilises
    as n_quad grows (AGQ(7) ~ AGQ(13)) and the marginal deviance is below
    Laplace's (= AGQ(1)) -- the Laplace bias for small binary clusters is
    corrected (the attenuated slope variance grows toward the converged value)."""
    G = np.array([[0.6, 0.2], [0.2, 0.4]])
    X, z, group, y, _ = _sim_slope(
        'binomial', seed=2, q=25, n_per=10, G=G, beta=(0.1, 0.6)
    )
    Y, Xj, zj, gj = (
        jnp.asarray(y[None, :]), jnp.asarray(X), jnp.asarray(z),
        jnp.asarray(group),
    )
    kw = dict(
        group=gj, z=zj, structure='unstructured', family='binomial', n_outer=80
    )
    lap = glmm_fit(Y, Xj, method='laplace', **kw)
    a5 = glmm_fit(Y, Xj, method='agq', n_quad=5, **kw)
    a9 = glmm_fit(Y, Xj, method='agq', n_quad=9, **kw)
    # Converged: node-count independent by 5 points.
    np.testing.assert_allclose(a5.re_var[0], a9.re_var[0], atol=2e-3)
    np.testing.assert_allclose(
        float(a5.deviance[0]), float(a9.deviance[0]), atol=2e-2
    )
    # A strictly better (lower) marginal deviance than Laplace ...
    assert float(a5.deviance[0]) < float(lap.deviance[0]) - 1e-2
    # ... and the attenuated slope variance is corrected upward.
    assert float(a5.re_var[0, 1, 1]) > float(lap.re_var[0, 1, 1])


def test_agq_shapes_and_validation():
    G = np.diag([0.4, 0.3])
    X, z, group, y, _ = _sim_slope('poisson', seed=2, q=10, n_per=12, G=G)
    V = 3
    Y = jnp.asarray(np.tile(y, (V, 1)))
    Xj, zj, gj = jnp.asarray(X), jnp.asarray(z), jnp.asarray(group)
    gu = glmm_fit(
        Y, Xj, group=gj, z=zj, structure='unstructured', family='poisson',
        method='agq', n_quad=3,
    )
    assert gu.tier == 'agq'
    assert gu.blups.shape == (V, 10, 2) and gu.re_var.shape == (V, 2, 2)
    gd = glmm_fit(
        Y, Xj, group=gj, z=zj, structure='diagonal', family='poisson',
        method='agq', n_quad=3,
    )
    # D4: diagonal AGQ G is the uniform (V, r, r) shape.
    assert gd.re_var.shape == (V, 2, 2)
    assert np.allclose(gd.re_var[:, 0, 1], 0.0)
    leaves, treedef = jax.tree_util.tree_flatten(gu)
    gu2 = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(gu2, GLMMResult) and gu2.tier == 'agq'
    # AGQ requires a slope; z=None raises.
    with pytest.raises(NotImplementedError, match='agq'):
        glmm_fit(Y, Xj, group=gj, family='poisson', method='agq')
    # P3: the n_quad**r node-count cap blocks the compile/memory cliff.
    z3 = jnp.concatenate([zj, zj[:, :1]], axis=-1)  # r = 3
    with pytest.raises(ValueError, match=r'n_quad\*\*r'):
        glmm_fit(
            Y, Xj, group=gj, z=z3, structure='unstructured', family='poisson',
            method='agq', n_quad=6,  # 6**3 = 216 > 128
        )
