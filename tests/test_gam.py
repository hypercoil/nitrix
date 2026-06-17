# -*- coding: utf-8 -*-
"""Tests for ``nitrix.stats.gam`` (mass-univariate GAM / GAMM).

The inner penalised fit is anchored *exactly* against the direct penalised
normal equations (the unambiguous reference); the REML / Fellner-Schall
smoothing-parameter selection is validated by smooth recovery, the correct
smoothing response to noise, and the EDF / dispersion identities -- the
properties that matter for ModelArray parity, without coupling to any single
library's lambda convention.
"""

from __future__ import annotations

import re

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update('jax_enable_x64', True)

from nitrix.stats.basis import bspline_basis
from nitrix.stats.gam import gam_fit, smooth_partial_effect
from nitrix.stats.glm import POISSON


def _smooth_data(seed=0, n=300, noise=0.2):
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0.0, 1.0, n))
    truth = np.sin(2 * np.pi * x)
    y = truth + rng.standard_normal(n) * noise
    return x, truth, y


# ---------------------------------------------------------------------------
# Inner penalised fit: exact vs direct penalised normal equations
# ---------------------------------------------------------------------------


def test_inner_fit_matches_penalised_normal_equations():
    """At a pinned lambda the GAM coefficients equal the closed-form penalised
    least-squares solution ``(X^T X + lambda S)^{-1} X^T y``."""
    x, _, y = _smooth_data(noise=0.2)
    sb = bspline_basis(jnp.asarray(x), 20, center=True)
    B = np.asarray(sb.design)
    S = np.asarray(sb.penalty)
    X = np.column_stack([np.ones(len(x)), B])
    Sfull = np.zeros((X.shape[1], X.shape[1]))
    Sfull[1:, 1:] = S
    lam = 2.5
    beta_ref = np.linalg.solve(X.T @ X + lam * Sfull, X.T @ y)
    res = gam_fit(
        jnp.asarray(y[None, :]), [sb], n_outer=1, lam_floor=lam, lam_ceil=lam
    )
    np.testing.assert_allclose(np.asarray(res.coef[0]), beta_ref, atol=1e-7)


# ---------------------------------------------------------------------------
# REML smoothing-parameter selection
# ---------------------------------------------------------------------------


def test_reml_recovers_smooth():
    x, truth, y = _smooth_data(noise=0.2)
    sb = bspline_basis(jnp.asarray(x), 20, center=True)
    res = gam_fit(jnp.asarray(y[None, :]), [sb])
    eff, se = smooth_partial_effect(res, 0, sb, jnp.asarray(x))
    fitted = float(res.coef[0, 0]) + np.asarray(eff[0])
    interior = (x > 0.05) & (x < 0.95)
    rmse = float(np.sqrt(np.mean((fitted[interior] - truth[interior]) ** 2)))
    assert rmse < 0.05
    # EDF strictly between a line (1) and the basis dimension.
    assert 2.0 < float(res.edf[0, 0]) < float(sb.dim)
    # dispersion within ~25% of the true noise variance.
    assert abs(float(res.dispersion[0]) - 0.2**2) < 0.25 * 0.2**2
    assert (se > 0).all()


def test_more_noise_gives_more_smoothing():
    """Noisier data -> larger lambda and smaller effective df."""
    x, _, y_lo = _smooth_data(seed=1, noise=0.15)
    _, _, y_hi = _smooth_data(seed=1, noise=0.7)
    sb = bspline_basis(jnp.asarray(x), 20, center=True)
    lo = gam_fit(jnp.asarray(y_lo[None, :]), [sb])
    hi = gam_fit(jnp.asarray(y_hi[None, :]), [sb])
    assert float(hi.lam[0, 0]) > float(lo.lam[0, 0])
    assert float(hi.edf[0, 0]) < float(lo.edf[0, 0])


def test_edf_equals_influence_trace():
    """Total EDF equals ``tr((X^T W X + S)^{-1} X^T W X)`` -- here, for the
    Gaussian case, computed directly from the fitted lambda."""
    x, _, y = _smooth_data(noise=0.3)
    sb = bspline_basis(jnp.asarray(x), 15, center=True)
    res = gam_fit(jnp.asarray(y[None, :]), [sb])
    B = np.asarray(sb.design)
    S = np.asarray(sb.penalty)
    X = np.column_stack([np.ones(len(x)), B])
    Sfull = np.zeros((X.shape[1], X.shape[1]))
    Sfull[1:, 1:] = S
    lam = float(res.lam[0, 0])
    xtx = X.T @ X
    edf_ref = np.trace(np.linalg.solve(xtx + lam * Sfull, xtx))
    np.testing.assert_allclose(float(res.edf_total[0]), edf_ref, atol=1e-4)


def test_poisson_gam_recovers_log_mean():
    rng = np.random.default_rng(2)
    x = np.sort(rng.uniform(0.0, 1.0, 400))
    log_mu = 0.8 * np.sin(2 * np.pi * x)
    y = rng.poisson(np.exp(log_mu)).astype(float)
    sb = bspline_basis(jnp.asarray(x), 15, center=True)
    res = gam_fit(jnp.asarray(y[None, :]), [sb], family=POISSON)
    eff, _ = smooth_partial_effect(res, 0, sb, jnp.asarray(x))
    fit = float(res.coef[0, 0]) + np.asarray(eff[0])
    interior = (x > 0.05) & (x < 0.95)
    rmse = float(np.sqrt(np.mean((fit[interior] - log_mu[interior]) ** 2)))
    assert rmse < 0.2
    assert bool(jnp.all(jnp.isfinite(res.coef)))


# ---------------------------------------------------------------------------
# Additive (multiple smooths) and mass-univariate behaviour
# ---------------------------------------------------------------------------


def test_additive_two_smooths():
    """y = f1(x1) + f2(x2): both smooths recovered, two lambdas selected."""
    rng = np.random.default_rng(3)
    n = 400
    x1 = rng.uniform(0.0, 1.0, n)
    x2 = rng.uniform(0.0, 1.0, n)
    f1 = np.sin(2 * np.pi * x1)
    f2 = (x2 - 0.5) ** 2 * 4.0
    y = f1 + f2 + rng.standard_normal(n) * 0.2
    s1 = bspline_basis(jnp.asarray(x1), 15, center=True)
    s2 = bspline_basis(jnp.asarray(x2), 15, center=True)
    res = gam_fit(jnp.asarray(y[None, :]), [s1, s2])
    assert res.lam.shape == (1, 2)
    assert res.edf.shape == (1, 2)
    # both smooths use more than a line's worth of df
    assert float(res.edf[0, 0]) > 2.0 and float(res.edf[0, 1]) > 2.0
    # quadratic f2 is smoother than the sine f1
    assert float(res.edf[0, 1]) < float(res.edf[0, 0])


def test_mass_univariate_batched_matches_looped():
    x, _, _ = _smooth_data()
    sb = bspline_basis(jnp.asarray(x), 12, center=True)
    rng = np.random.default_rng(4)
    Y = jnp.asarray(
        np.sin(2 * np.pi * x)[None, :] + rng.standard_normal((6, len(x))) * 0.3
    )
    res = gam_fit(Y, [sb])
    for v in range(6):
        rv = gam_fit(Y[v : v + 1], [sb])
        np.testing.assert_allclose(
            np.asarray(res.coef[v]), np.asarray(rv.coef[0]), atol=1e-7
        )
        np.testing.assert_allclose(
            float(res.lam[v, 0]), float(rv.lam[0, 0]), rtol=1e-6
        )


# ---------------------------------------------------------------------------
# cuSOLVER-free
# ---------------------------------------------------------------------------


def test_gam_hlo_is_cusolver_free():
    x, _, _ = _smooth_data()
    sb = bspline_basis(jnp.asarray(x), 12, center=True)
    rng = np.random.default_rng(5)
    Y = jnp.asarray(rng.standard_normal((32, len(x))))
    f = jax.jit(lambda Y: gam_fit(Y, [sb], n_outer=4, n_inner=4).coef)
    hlo = f.lower(Y).compile().as_text()
    targets = set(re.findall(r'custom_call_target="([^"]+)"', hlo))
    bad = [
        c
        for c in targets
        if any(
            t in c.lower()
            for t in ('cusolver', 'syevd', 'potrf', 'getrf', 'geqrf', 'gesvd', 'cholesky', 'eigh')
        )
    ]
    assert not bad, bad
