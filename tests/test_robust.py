# -*- coding: utf-8 -*-
"""Tests for robust M-estimator regression (stats.robust).

``mad`` is checked against numpy and for normal-consistency; the Huber and Tukey
estimators are checked to (a) match OLS on clean Gaussian data and (b) resist
gross contamination that catastrophically biases OLS, with the Tukey redescender
driving outlier weights to zero. Differentiability and jit are exercised.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update('jax_enable_x64', True)

from nitrix.stats.robust import (  # noqa: E402
    RobustFit,
    huber_regress,
    mad,
    tukey_bisquare_regress,
)

_BETA = np.array([2.0, -1.5, 0.7])


def _design(n=80, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 3))
    X[:, 0] = 1.0  # intercept
    y = X @ _BETA + 0.3 * rng.standard_normal(n)
    return X, y, rng


def _contaminate(X, y, rng, frac=0.15, mag=30.0):
    yc = y.copy()
    k = int(frac * len(y))
    idx = rng.choice(len(y), k, replace=False)
    yc[idx] += mag * rng.standard_normal(k)
    return yc, idx


# --- mad ---------------------------------------------------------------------


def test_mad_matches_numpy_raw():
    x = np.random.default_rng(1).standard_normal(1000)
    raw = np.median(np.abs(x - np.median(x)))
    np.testing.assert_allclose(
        float(mad(jnp.asarray(x), normalize=False)), raw, atol=1e-12
    )


def test_mad_normal_consistent():
    x = jnp.asarray(np.random.default_rng(2).standard_normal(4000))
    # normalised MAD is a consistent estimator of sigma (=1 here)
    np.testing.assert_allclose(float(mad(x)), 1.0, atol=0.06)
    # the normaliser is exactly 1.4826...
    np.testing.assert_allclose(
        float(mad(x)) / float(mad(x, normalize=False)),
        1.4826022185056018,
        rtol=1e-9,
    )


def test_mad_axis_and_center():
    y = jnp.asarray(np.random.default_rng(3).standard_normal((5, 200)))
    per_row = mad(y, axis=-1)
    assert per_row.shape == (5,)
    for i in range(5):
        np.testing.assert_allclose(
            float(per_row[i]), float(mad(y[i])), atol=1e-12
        )
    # explicit centre overrides the median
    x = jnp.asarray([1.0, 2.0, 3.0, 4.0])
    manual = 1.4826022185056018 * float(jnp.median(jnp.abs(x - 0.0)))
    np.testing.assert_allclose(float(mad(x, center=0.0)), manual, atol=1e-12)


# --- clean data: robust ~ OLS ------------------------------------------------


def test_clean_data_recovers_truth():
    X, y, _ = _design()
    Xj, yj = jnp.asarray(X), jnp.asarray(y)
    for fit in (huber_regress(Xj, yj), tukey_bisquare_regress(Xj, yj)):
        np.testing.assert_allclose(np.asarray(fit.coef), _BETA, atol=0.25)


# --- contamination resistance ------------------------------------------------


def test_huber_resists_outliers():
    X, y, rng = _design()
    yc, _ = _contaminate(X, y, rng)
    Xj, ycj = jnp.asarray(X), jnp.asarray(yc)
    ols_err = np.abs(np.linalg.lstsq(X, yc, rcond=None)[0] - _BETA).max()
    hub = huber_regress(Xj, ycj)
    rob_err = float(jnp.abs(hub.coef - _BETA).max())
    assert rob_err < 0.3  # Huber recovers the truth despite contamination
    assert ols_err > 2 * rob_err  # and clearly beats the OLS bias


def test_tukey_rejects_outliers():
    X, y, rng = _design()
    yc, idx = _contaminate(X, y, rng)
    Xj, ycj = jnp.asarray(X), jnp.asarray(yc)
    tuk = tukey_bisquare_regress(Xj, ycj)
    np.testing.assert_allclose(np.asarray(tuk.coef), _BETA, atol=0.3)
    w = np.asarray(tuk.weights)
    assert w[idx].mean() < 0.05  # outliers redescended to (near) zero weight
    assert np.delete(w, idx).mean() > 0.5  # inliers keep substantial weight


# --- result contract ---------------------------------------------------------


def test_result_fields_and_shapes():
    X, y, _ = _design()
    Xj, yj = jnp.asarray(X), jnp.asarray(y)
    fit = huber_regress(Xj, yj)
    assert isinstance(fit, RobustFit)
    assert fit.coef.shape == (3,)
    assert fit.weights.shape == (80,)
    assert fit.residuals.shape == (80,)
    assert fit.scale.shape == ()
    np.testing.assert_allclose(
        np.asarray(fit.residuals), y - X @ np.asarray(fit.coef), atol=1e-10
    )
    assert bool((fit.weights >= 0).all()) and bool((fit.weights <= 1).all())


def test_ridge_stabilises_rank_deficient():
    X, y, _ = _design()
    X[:, 2] = X[:, 1]  # collinear -> singular normal equations
    Xj, yj = jnp.asarray(X), jnp.asarray(y)
    fit = huber_regress(Xj, yj, ridge=1e-3)
    assert bool(jnp.all(jnp.isfinite(fit.coef)))


# --- differentiability + jit + vmap ------------------------------------------


def test_grad_is_finite():
    X, y, rng = _design()
    yc, _ = _contaminate(X, y, rng)
    Xj, ycj = jnp.asarray(X), jnp.asarray(yc)
    for regress in (huber_regress, tukey_bisquare_regress):
        g = jax.grad(lambda Y: regress(Xj, Y).coef.sum())(ycj)
        assert bool(jnp.all(jnp.isfinite(g)))


def test_jit_clean():
    X, y, _ = _design()
    Xj, yj = jnp.asarray(X), jnp.asarray(y)
    c = jax.jit(lambda X, Y: tukey_bisquare_regress(X, Y).coef)(Xj, yj)
    np.testing.assert_allclose(np.asarray(c), _BETA, atol=0.25)


def test_vmap_over_responses():
    X, y, rng = _design()
    Y = np.stack([y, _contaminate(X, y, rng)[0]])  # (2, N)
    Xj, Yj = jnp.asarray(X), jnp.asarray(Y)
    coefs = jax.vmap(lambda yy: huber_regress(Xj, yy).coef)(Yj)
    assert coefs.shape == (2, 3)
    # both rows recover the truth (clean + contaminated)
    for i in range(2):
        np.testing.assert_allclose(np.asarray(coefs[i]), _BETA, atol=0.3)
