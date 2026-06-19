# -*- coding: utf-8 -*-
"""Tests for crossed random effects (v3 §1.1 tier R4).

``lme_fit(..., cross=g2)`` fits ``(1 | group) + (1 | cross)`` with the two
factors **crossed** (not nested) via the Woodbury + diagonal-Schur solver.
Anchored on an exact dense profile-REML reference (build the full
``V = sigma_g^2 Zg Zg^T + sigma_c^2 Zc Zc^T + sigma_e^2 I`` with *explicit*
group/cross labels and minimise the same REML criterion) -- the authoritative
same-estimator check, including the internal factor swap that keeps the dense
solve on the smaller factor.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import minimize

jax.config.update('jax_enable_x64', True)

from nitrix.stats.lme import CrossedLMEResult, lme_fit


def _sim_crossed(*, seed, q_group, q_cross, n, sg=0.6, sc=0.4, se=0.5):
    rng = np.random.default_rng(seed)
    g = rng.integers(0, q_group, n)
    c = rng.integers(0, q_cross, n)
    X = np.ones((n, 2))
    X[:, 1] = rng.standard_normal(n)
    bg = rng.standard_normal(q_group) * np.sqrt(sg)
    bc = rng.standard_normal(q_cross) * np.sqrt(sc)
    y = (
        X @ np.array([1.0, 0.5])
        + bg[g]
        + bc[c]
        + rng.standard_normal(n) * np.sqrt(se)
    )
    return X, g, c, y


def _dense_crossed(y, X, g, c, q_group, q_cross):
    n = X.shape[0]
    ohg = np.eye(q_group)[g]
    ohc = np.eye(q_cross)[c]

    def neg2(th):
        sg, sc, se = np.exp(th)
        V = sg * ohg @ ohg.T + sc * ohc @ ohc.T + se * np.eye(n)
        Vi = np.linalg.inv(V)
        xvx = X.T @ Vi @ X
        beta = np.linalg.solve(xvx, X.T @ Vi @ y)
        r = y - X @ beta
        return np.linalg.slogdet(V)[1] + np.linalg.slogdet(xvx)[1] + r @ Vi @ r

    r = minimize(
        neg2,
        np.log([0.4, 0.4, 0.5]),
        method='Nelder-Mead',
        options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 9000},
    )
    return np.exp(r.x)  # (var_group, var_cross, sigma_e_sq)


@pytest.mark.parametrize(
    'q_group,q_cross',
    [(14, 9), (8, 18)],  # group larger (no swap) and smaller (swap)
)
def test_crossed_matches_dense_reml(q_group, q_cross):
    X, g, c, y = _sim_crossed(seed=0, q_group=q_group, q_cross=q_cross, n=300)
    sg, sc, se = _dense_crossed(y, X, g, c, q_group, q_cross)
    res = lme_fit(
        jnp.asarray(y[None, :]),
        jnp.asarray(X),
        group=jnp.asarray(g),
        cross=jnp.asarray(c),
        n_iter=40,
    )
    assert isinstance(res, CrossedLMEResult) and res.tier == 'R4'
    assert abs(float(res.var_group[0]) - sg) < 3e-3
    assert abs(float(res.var_cross[0]) - sc) < 3e-3
    assert abs(float(res.sigma_e_sq[0]) - se) < 3e-3


@pytest.mark.parametrize('seed', range(6))
def test_crossed_recovers_all_seeds(seed):
    """Multi-seed recovery vs dense REML (the regression guard)."""
    X, g, c, y = _sim_crossed(seed=seed, q_group=15, q_cross=10, n=320)
    sg, sc, se = _dense_crossed(y, X, g, c, 15, 10)
    res = lme_fit(
        jnp.asarray(y[None, :]),
        jnp.asarray(X),
        group=jnp.asarray(g),
        cross=jnp.asarray(c),
        n_iter=40,
    )
    assert abs(float(res.var_group[0]) - sg) < 5e-3
    assert abs(float(res.var_cross[0]) - sc) < 5e-3
    assert abs(float(res.sigma_e_sq[0]) - se) < 5e-3


def test_crossed_shapes_and_dispatch():
    X, g, c, y = _sim_crossed(seed=2, q_group=12, q_cross=8, n=240)
    V = 7
    res = lme_fit(
        jnp.asarray(np.tile(y, (V, 1))),
        jnp.asarray(X),
        group=jnp.asarray(g),
        cross=jnp.asarray(c),
        n_iter=30,
    )
    assert res.beta_hat.shape == (V, 2)
    assert res.var_group.shape == (V,)
    assert res.var_cross.shape == (V,)
    assert res.sigma_e_sq.shape == (V,)
    assert np.all(np.asarray(res.var_group) > 0)
    assert np.all(np.asarray(res.var_cross) > 0)


def test_crossed_rejects_incompatible():
    X, g, c, y = _sim_crossed(seed=0, q_group=8, q_cross=6, n=120)
    z = jnp.asarray(np.c_[np.ones(len(y)), X[:, 1]])
    with pytest.raises(NotImplementedError, match='crossed'):
        lme_fit(
            jnp.asarray(y[None, :]),
            jnp.asarray(X),
            group=jnp.asarray(g),
            cross=jnp.asarray(c),
            z=z,
        )
    with pytest.raises(NotImplementedError, match='crossed'):
        lme_fit(
            jnp.asarray(y[None, :]),
            jnp.asarray(X),
            group=jnp.asarray(g),
            cross=jnp.asarray(c),
            corr='ar1',
        )
