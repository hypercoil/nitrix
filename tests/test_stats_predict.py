# -*- coding: utf-8 -*-
"""Self-consistency tests for the response-regression predict surface.

``beta_predict`` / ``ordinal_predict`` / ``gaulss_predict`` / ``gam_predict``
(nimox-estimators Tier-3): predict on the training design reproduces the fitted
means; shapes and link round-trips; differentiability w.r.t. the new design.
"""

from __future__ import annotations

import jax

jax.config.update('jax_enable_x64', True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from nitrix.stats import (  # noqa: E402
    beta_fit,
    beta_predict,
    gam_fit,
    gam_predict,
    gaulss_fit,
    gaulss_predict,
    ordinal_fit,
    ordinal_predict,
)
from nitrix.stats.basis import bspline_basis  # noqa: E402

# ---------------------------------------------------------------------------
# beta_predict
# ---------------------------------------------------------------------------


def test_beta_predict_self_consistent():
    rng = np.random.default_rng(0)
    n, p, v = 80, 3, 4
    X = rng.normal(size=(n, p))
    X[:, 0] = 1.0
    Y = rng.uniform(0.05, 0.95, (v, n))
    res = beta_fit(jnp.asarray(Y), jnp.asarray(X))
    coef = np.asarray(res.coef)
    eta = coef @ X.T
    # response = expit(eta); link = eta.
    pred = np.asarray(beta_predict(res, jnp.asarray(X)))
    np.testing.assert_allclose(pred, 0.5 * (1 + np.tanh(0.5 * eta)), atol=1e-6)
    link = np.asarray(beta_predict(res, jnp.asarray(X), type='link'))
    np.testing.assert_allclose(link, eta, atol=1e-6)
    assert np.all((pred > 0) & (pred < 1))


def test_beta_predict_differentiable():
    rng = np.random.default_rng(1)
    X = jnp.asarray(rng.normal(size=(40, 2)))
    Y = jnp.asarray(rng.uniform(0.1, 0.9, (2, 40)))
    res = beta_fit(Y, X)
    g = jax.grad(lambda xx: jnp.sum(beta_predict(res, xx)))(X)
    assert g.shape == X.shape
    assert bool(jnp.all(jnp.isfinite(g)))


# ---------------------------------------------------------------------------
# gaulss_predict
# ---------------------------------------------------------------------------


def test_gaulss_predict_pair_self_consistent():
    rng = np.random.default_rng(2)
    n, p, v = 100, 2, 3
    X = rng.normal(size=(n, p))
    X[:, 0] = 1.0
    Y = rng.normal(size=(v, n))
    res = gaulss_fit(jnp.asarray(Y), jnp.asarray(X))
    mu, sigma = gaulss_predict(res, jnp.asarray(X))
    np.testing.assert_allclose(
        np.asarray(mu), np.asarray(res.coef_mu) @ X.T, atol=1e-6
    )
    z = np.ones((n, 1))
    np.testing.assert_allclose(
        np.asarray(sigma), np.exp(np.asarray(res.coef_scale) @ z.T), atol=1e-6
    )
    assert np.all(np.asarray(sigma) > 0)


# ---------------------------------------------------------------------------
# ordinal_predict
# ---------------------------------------------------------------------------


def test_ordinal_predict_shapes_simplex_and_consistency():
    rng = np.random.default_rng(3)
    n, p, v, k = 120, 2, 3, 4
    X = rng.normal(size=(n, p))  # no intercept (thresholds absorb it)
    Y = rng.integers(0, k, (v, n))
    res = ordinal_fit(jnp.asarray(Y), jnp.asarray(X), n_classes=k)
    assert res.link == 'logit'

    cp = np.asarray(ordinal_predict(res, jnp.asarray(X), type='class_prob'))
    assert cp.shape == (v, n, k)
    np.testing.assert_allclose(cp.sum(-1), 1.0, atol=1e-5)  # simplex
    cum = np.asarray(ordinal_predict(res, jnp.asarray(X), type='cum_prob'))
    assert cum.shape == (v, n, k - 1)
    lab = np.asarray(ordinal_predict(res, jnp.asarray(X), type='class'))
    assert lab.shape == (v, n)
    np.testing.assert_array_equal(lab, np.argmax(cp, -1))

    # self-consistency: class_prob == manual diff-of-(logistic CDF).
    eta = np.asarray(res.coef) @ X.T  # (V, N)
    th = np.asarray(res.thresholds)  # (V, K-1)
    cumref = 1.0 / (1.0 + np.exp(-(th[:, None, :] - eta[:, :, None])))
    full = np.concatenate(
        [np.zeros((v, n, 1)), cumref, np.ones((v, n, 1))], axis=-1
    )
    np.testing.assert_allclose(
        cp, np.clip(np.diff(full, axis=-1), 1e-12, None), atol=1e-5
    )


def test_ordinal_predict_uses_stored_probit_link():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(60, 2))
    Y = rng.integers(0, 3, (2, 60))
    res = ordinal_fit(
        jnp.asarray(Y), jnp.asarray(X), n_classes=3, link='probit'
    )
    assert res.link == 'probit'
    # predict reconstructs the probit CDF from the stored link (no arg needed).
    cp = np.asarray(ordinal_predict(res, jnp.asarray(X)))
    from jax.scipy.special import ndtr

    eta = np.asarray(res.coef) @ X.T
    th = np.asarray(res.thresholds)
    cumref = np.asarray(ndtr(jnp.asarray(th[:, None, :] - eta[:, :, None])))
    full = np.concatenate(
        [np.zeros((2, 60, 1)), cumref, np.ones((2, 60, 1))], axis=-1
    )
    np.testing.assert_allclose(
        cp, np.clip(np.diff(full, axis=-1), 1e-12, None), atol=1e-5
    )


def test_ordinal_predict_differentiable_class_prob():
    rng = np.random.default_rng(5)
    X = jnp.asarray(rng.normal(size=(50, 2)))
    Y = jnp.asarray(rng.integers(0, 3, (2, 50)))
    res = ordinal_fit(Y, X, n_classes=3)
    g = jax.grad(lambda xx: jnp.sum(ordinal_predict(res, xx) ** 2))(X)
    assert g.shape == X.shape
    assert bool(jnp.all(jnp.isfinite(g)))


# ---------------------------------------------------------------------------
# gam_predict
# ---------------------------------------------------------------------------


def _smooth_data(seed=0, n=200, noise=0.2):
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0.0, 1.0, n))
    y = np.sin(2 * np.pi * x) + rng.standard_normal(n) * noise
    return x, y


def test_gam_predict_reproduces_fit():
    x, y = _smooth_data()
    sb = bspline_basis(jnp.asarray(x), 15, center=True)
    res = gam_fit(jnp.asarray(y[None, :]), [sb])
    assert res.intercept is True
    # predict on the training covariates reproduces the fitted linear predictor.
    x_fit = np.column_stack([np.ones(len(x)), np.asarray(sb.design)])
    eta_ref = np.asarray(res.coef) @ x_fit.T
    pred = np.asarray(gam_predict(res, [sb], [jnp.asarray(x)]))
    ref = np.asarray(res.family.linkinv(jnp.asarray(eta_ref)))
    np.testing.assert_allclose(pred, ref, atol=1e-5)
    link = np.asarray(gam_predict(res, [sb], [jnp.asarray(x)], type='link'))
    np.testing.assert_allclose(link, eta_ref, atol=1e-5)


def test_gam_predict_with_parametric_block():
    x, y = _smooth_data(seed=1)
    rng = np.random.default_rng(7)
    par = rng.normal(size=(len(x), 2))
    sb = bspline_basis(jnp.asarray(x), 12, center=True)
    res = gam_fit(jnp.asarray(y[None, :]), [sb], parametric=jnp.asarray(par))
    # design = [intercept | parametric | smooth]; predict must reassemble it.
    x_fit = np.column_stack([np.ones(len(x)), par, np.asarray(sb.design)])
    eta_ref = np.asarray(res.coef) @ x_fit.T
    pred = np.asarray(
        gam_predict(res, [sb], [jnp.asarray(x)], parametric=jnp.asarray(par))
    )
    np.testing.assert_allclose(
        pred, np.asarray(res.family.linkinv(jnp.asarray(eta_ref))), atol=1e-5
    )


def test_gam_predict_differentiable():
    x, y = _smooth_data(seed=2, n=120)
    sb = bspline_basis(jnp.asarray(x), 10, center=True)
    res = gam_fit(jnp.asarray(y[None, :]), [sb])
    g = jax.grad(lambda xx: jnp.sum(gam_predict(res, [sb], [xx])))(
        jnp.asarray(x)
    )
    assert g.shape == (len(x),)
    assert bool(jnp.all(jnp.isfinite(g)))
