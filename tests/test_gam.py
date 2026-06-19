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
import pytest

jax.config.update('jax_enable_x64', True)

from nitrix.stats.basis import (
    bspline_basis,
    by_factor_smooth,
    re_smooth,
    tensor_product_basis,
    varying_coefficient_smooth,
)
from nitrix.stats.gam import (
    _assemble,
    _gam_fit_one,
    _gam_fit_one_gaussian_xprod,
    gam_fit,
    smooth_partial_effect,
)
from nitrix.stats.glm import GAUSSIAN, POISSON


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


def test_shared_lambda_recovers_smooth_and_pools():
    """Shared-lambda selects ONE smoothing parameter across all elements (the
    pooled Fellner-Schall), recovers a smooth, and on homogeneous data lands
    near the per-element median; non-Gaussian shared raises."""
    rng = np.random.default_rng(0)
    x = np.sort(rng.uniform(0.0, 1.0, 300))
    truth = np.sin(2 * np.pi * x)
    sb = bspline_basis(jnp.asarray(x), 20, center=True)
    Y = jnp.asarray(truth[None, :] + rng.standard_normal((40, 300)) * 0.2)

    shared = gam_fit(Y, [sb], lambda_mode='shared')
    per_elt = gam_fit(Y, [sb], lambda_mode='per_element')

    # one lambda for all elements
    assert bool(jnp.allclose(shared.lam, shared.lam[0]))
    # recovers the smooth
    eff, _ = smooth_partial_effect(shared, 0, sb, jnp.asarray(x))
    fit = float(shared.coef[0, 0]) + np.asarray(eff[0])
    interior = (x > 0.05) & (x < 0.95)
    assert (
        float(np.sqrt(np.mean((fit[interior] - truth[interior]) ** 2))) < 0.05
    )
    # homogeneous data: shared lambda near the per-element median
    assert 2.0 < float(shared.edf[0, 0]) < float(sb.dim)
    med = float(jnp.median(per_elt.lam[:, 0]))
    assert 0.3 < float(shared.lam[0, 0]) / med < 3.0

    from nitrix.stats.glm import POISSON

    with pytest.raises(NotImplementedError):
        gam_fit(Y, [sb], family=POISSON, lambda_mode='shared')


def test_gam_block_chunking_matches_single_vmap():
    """The GAM ``block=`` memory knob is numerically transparent, incl. when V
    is *not* a multiple of the block (the pad/trim path)."""
    x, _, _ = _smooth_data()
    sb = bspline_basis(jnp.asarray(x), 12, center=True)
    rng = np.random.default_rng(7)
    Y = jnp.asarray(
        np.sin(2 * np.pi * x)[None, :]
        + rng.standard_normal((23, len(x))) * 0.3
    )  # 23 % 8 != 0
    full = gam_fit(Y, [sb])
    chunked = gam_fit(Y, [sb], block=8)
    np.testing.assert_allclose(
        np.asarray(chunked.coef), np.asarray(full.coef), atol=1e-9
    )
    np.testing.assert_allclose(
        np.asarray(chunked.lam), np.asarray(full.lam), rtol=1e-6
    )
    assert chunked.edf.shape == full.edf.shape


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
            for t in (
                'cusolver',
                'syevd',
                'potrf',
                'getrf',
                'geqrf',
                'gesvd',
                'cholesky',
                'eigh',
            )
        )
    ]
    assert not bad, bad


# ---------------------------------------------------------------------------
# Tensor-product (te / ti) interaction smooths
# ---------------------------------------------------------------------------


def _interaction_data(seed=0, n=1200, noise=0.15):
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(0.0, 1.0, n)
    x2 = rng.uniform(0.0, 1.0, n)
    truth = np.sin(2 * np.pi * x1) * (x2 - 0.5)  # genuine interaction
    y = truth + rng.standard_normal(n) * noise
    return x1, x2, truth, y


def test_tensor_recovers_interaction_surface():
    """A tensor smooth recovers an interaction surface, selecting one smoothing
    parameter per margin (Bayesian SE positive, EDF interior)."""
    x1, x2, truth, y = _interaction_data()
    m1 = bspline_basis(jnp.asarray(x1), 8, center=True)
    m2 = bspline_basis(jnp.asarray(x2), 8, center=True)
    te = tensor_product_basis((m1, m2))
    res = gam_fit(jnp.asarray(y[None, :]), [te])

    assert res.lam.shape == (1, 2)  # one smoothing parameter per margin
    assert res.edf.shape == (1, 1)  # one EDF for the interaction term
    eff, se = smooth_partial_effect(
        res, 0, te, (jnp.asarray(x1), jnp.asarray(x2))
    )
    fit = float(res.coef[0, 0]) + np.asarray(eff[0])
    interior = (x1 > 0.1) & (x1 < 0.9) & (x2 > 0.1) & (x2 < 0.9)
    rmse = float(np.sqrt(np.mean((fit[interior] - truth[interior]) ** 2)))
    assert rmse < 0.05
    assert 2.0 < float(res.edf[0, 0]) < float(te.dim)
    assert bool((se > 0).all())
    assert abs(float(res.dispersion[0]) - 0.15**2) < 0.3 * 0.15**2


def test_tensor_anisotropic_smoothing():
    """A surface wiggly in x1 but (near) linear in x2 selects a much smaller
    lambda for the x1 margin than the x2 margin -- the point of per-margin
    smoothing the isotropic single-penalty form cannot express."""
    rng = np.random.default_rng(4)
    n = 1500
    x1 = rng.uniform(0.0, 1.0, n)
    x2 = rng.uniform(0.0, 1.0, n)
    truth = np.sin(3 * np.pi * x1) * (x2 - 0.5)  # wiggly in x1, linear in x2
    y = truth + rng.standard_normal(n) * 0.1
    m1 = bspline_basis(jnp.asarray(x1), 9, center=True)
    m2 = bspline_basis(jnp.asarray(x2), 9, center=True)
    te = tensor_product_basis((m1, m2))
    res = gam_fit(jnp.asarray(y[None, :]), [te])
    lam1, lam2 = float(res.lam[0, 0]), float(res.lam[0, 1])
    # the x2 margin is penalised orders of magnitude harder than x1.
    assert lam2 > 100.0 * lam1


def test_tensor_additive_with_marginal_smooth():
    """A tensor interaction composes with ordinary marginal smooths in one fit
    (te = s(x1) + s(x2) + ti(x1, x2) style additive model)."""
    x1, x2, _, y = _interaction_data(seed=2)
    s1 = bspline_basis(jnp.asarray(x1), 8, center=True)
    s2 = bspline_basis(jnp.asarray(x2), 8, center=True)
    m1 = bspline_basis(jnp.asarray(x1), 6, center=True)
    m2 = bspline_basis(jnp.asarray(x2), 6, center=True)
    te = tensor_product_basis((m1, m2))
    res = gam_fit(jnp.asarray(y[None, :]), [s1, s2, te])
    # three terms: 2 marginal (1 lambda each) + 1 tensor (2 lambdas) = 4 lambdas
    assert res.lam.shape == (1, 4)
    assert res.edf.shape == (1, 3)
    assert bool(jnp.all(jnp.isfinite(res.coef)))


def test_tensor_gam_cusolver_free():
    x1, x2, _, y = _interaction_data(n=300)
    m1 = bspline_basis(jnp.asarray(x1), 6, center=True)
    m2 = bspline_basis(jnp.asarray(x2), 6, center=True)
    te = tensor_product_basis((m1, m2))
    Y = jnp.asarray(y[None, :])
    f = jax.jit(lambda Y: gam_fit(Y, [te], n_outer=4, n_inner=4).coef)
    hlo = f.lower(Y).compile().as_text()
    targets = set(re.findall(r'custom_call_target="([^"]+)"', hlo))
    bad = [
        c
        for c in targets
        if any(
            t in c.lower()
            for t in (
                'cusolver',
                'syevd',
                'potrf',
                'getrf',
                'geqrf',
                'gesvd',
                'cholesky',
                'eigh',
            )
        )
    ]
    assert not bad, bad


# ---------------------------------------------------------------------------
# Gaussian cross-product fast path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('k', [12, 20])
def test_gaussian_xprod_matches_generic_irls(k):
    """The cross-product Gaussian fit (now the per-element default) reproduces
    the generic IRLS Fellner-Schall fit to floating point -- it is the SAME
    estimator, computed from c = X^T y and g = y^T y instead of the N-vector."""
    rng = np.random.default_rng(k)
    x = np.sort(rng.uniform(0.0, 1.0, 320))
    sb = bspline_basis(jnp.asarray(x), k, center=True)
    X, penalties, pen_eig, _ = _assemble(320, [sb], None, True, jnp.float64)
    p = X.shape[1]
    y = jnp.asarray(np.sin(2 * np.pi * x) + rng.standard_normal(320) * 0.25)

    bg, lg, vg, xg, pg = _gam_fit_one(
        y, X, penalties, pen_eig, GAUSSIAN, p, 20, 15, 1e-8, 1e-6, 1e8
    )
    c = X.T @ y
    g = y @ y
    bx, lx, vx, xx, px = _gam_fit_one_gaussian_xprod(
        c, g, X.T @ X, penalties, pen_eig, 320, p, 20, 1e-8, 1e-6, 1e8
    )
    np.testing.assert_allclose(np.asarray(bx), np.asarray(bg), atol=1e-10)
    np.testing.assert_allclose(np.asarray(lx), np.asarray(lg), atol=1e-9)
    np.testing.assert_allclose(np.asarray(vx), np.asarray(vg), atol=1e-10)
    np.testing.assert_allclose(float(px), float(pg), atol=1e-10)


def test_gaussian_xprod_is_the_default_and_recovers_smooth():
    """gam_fit on Gaussian data uses the cross-product path by default and still
    recovers the smooth / EDF / dispersion (an end-to-end check of the default)."""
    x, truth, y = _smooth_data(noise=0.2)
    sb = bspline_basis(jnp.asarray(x), 20, center=True)
    res = gam_fit(jnp.asarray(y[None, :]), [sb])
    eff, _ = smooth_partial_effect(res, 0, sb, jnp.asarray(x))
    fit = float(res.coef[0, 0]) + np.asarray(eff[0])
    interior = (x > 0.05) & (x < 0.95)
    assert (
        float(np.sqrt(np.mean((fit[interior] - truth[interior]) ** 2))) < 0.05
    )
    assert 2.0 < float(res.edf[0, 0]) < float(sb.dim)
    assert abs(float(res.dispersion[0]) - 0.2**2) < 0.25 * 0.2**2


# ---------------------------------------------------------------------------
# GAMM: random-effect smooth blocks (bs='re')
# ---------------------------------------------------------------------------


def test_gamm_random_intercept_recovers_variance_components():
    """A random-intercept GAMM (one re_smooth block) recovers the variance
    components: the Fellner-Schall smoothing parameter is the precision
    1 / sigma_b^2 and the dispersion is sigma_e^2 -- matched against the
    closed-form balanced-ANOVA REML estimates."""
    rng = np.random.default_rng(0)
    g, n_per, mu, sb, se = 8, 30, 2.0, 1.0, 0.5
    b = rng.standard_normal(g) * sb
    gid = np.repeat(np.arange(g), n_per)
    y = mu + b[gid] + rng.standard_normal(g * n_per) * se

    # closed-form balanced one-way REML (ANOVA) variance components
    ybar_i = np.array([y[gid == i].mean() for i in range(g)])
    ybar = y.mean()
    ms_b = n_per * np.sum((ybar_i - ybar) ** 2) / (g - 1)
    ms_w = np.sum((y - ybar_i[gid]) ** 2) / (g * n_per - g)
    sb2_ref = (ms_b - ms_w) / n_per
    se2_ref = ms_w

    re = re_smooth(jnp.asarray(gid), n_levels=g)
    res = gam_fit(jnp.asarray(y[None, :]), [re], n_outer=60)
    lam = float(res.lam[0, 0])
    disp = float(res.dispersion[0])
    assert abs(disp - se2_ref) / se2_ref < 1e-4
    assert abs(disp / lam - sb2_ref) / sb2_ref < 1e-4


def test_gamm_smooth_plus_random_intercept():
    """A GAMM with a spline smooth AND a random intercept fits both blocks: two
    smoothing parameters, the smooth recovered, the group means absorbed."""
    rng = np.random.default_rng(1)
    g, n_per = 10, 40
    gid = np.repeat(np.arange(g), n_per)
    n = g * n_per
    x = rng.uniform(0.0, 1.0, n)
    b = rng.standard_normal(g) * 0.8
    truth = np.sin(2 * np.pi * x)
    y = truth + b[gid] + rng.standard_normal(n) * 0.25

    sb = bspline_basis(jnp.asarray(x), 15, center=True)
    re = re_smooth(jnp.asarray(gid), n_levels=g)
    res = gam_fit(jnp.asarray(y[None, :]), [sb, re], n_outer=40)
    assert res.lam.shape == (1, 2)
    eff, _ = smooth_partial_effect(res, 0, sb, jnp.asarray(x))
    fit = np.asarray(eff[0])
    interior = (x > 0.05) & (x < 0.95)
    centred = truth - truth.mean()
    rmse = float(np.sqrt(np.mean((fit[interior] - centred[interior]) ** 2)))
    assert rmse < 0.1
    # random-effect block uses up to g-1 effective df
    blup, _ = smooth_partial_effect(res, 1, re, jnp.arange(g, dtype=jnp.int32))
    # recovered BLUPs correlate with the true group offsets
    assert float(np.corrcoef(np.asarray(blup[0]), b)[0, 1]) > 0.9


def test_by_factor_smooth_recovers_distinct_per_level_curves():
    """``s(x, by=f)`` recovers a *different* smooth of ``x`` for each factor
    level (the by-variable interaction)."""
    rng = np.random.default_rng(0)
    n = 600
    x = np.sort(rng.uniform(0.0, 1.0, n))
    lvl = rng.integers(0, 3, n)
    curves = (
        lambda t: np.sin(2 * np.pi * t),
        lambda t: 4.0 * (t - 0.5) ** 2 - 0.5,
        lambda t: -np.cos(3.0 * t),
    )
    f = np.array([curves[k](xi) for xi, k in zip(x, lvl)])
    for k in range(3):  # match the sum-to-zero smooth
        f[lvl == k] -= f[lvl == k].mean()
    y = f + rng.standard_normal(n) * 0.25

    parametric = np.column_stack(
        [(lvl == 1).astype(float), (lvl == 2).astype(float)]
    )
    blocks = by_factor_smooth(jnp.asarray(x), jnp.asarray(lvl), n_basis=12)
    assert len(blocks) == 3
    res = gam_fit(
        jnp.asarray(y[None, :]),
        list(blocks),
        parametric=jnp.asarray(parametric),
        intercept=True,
        n_outer=30,
    )
    xg = np.linspace(x.min(), x.max(), 100)
    for level in range(3):
        eff, _ = smooth_partial_effect(
            res, level, blocks[level], jnp.asarray(xg)
        )
        e = np.array(eff[0])
        e = e - e.mean()
        t = curves[level](xg)
        t = t - t.mean()
        assert float(np.corrcoef(e, t)[0, 1]) > 0.97


def test_varying_coefficient_smooth_recovers_smooth_coefficient():
    """``s(x, by=z)`` for continuous ``z`` recovers the coefficient surface
    ``z * f(x)`` -- the partial effect tracks ``f(x)``."""
    rng = np.random.default_rng(1)
    n = 600
    x = np.sort(rng.uniform(0.0, 1.0, n))
    z = rng.standard_normal(n)
    g = np.sin(2 * np.pi * x)
    g = g - g.mean()
    y = z * g + rng.standard_normal(n) * 0.2

    vc = varying_coefficient_smooth(jnp.asarray(x), jnp.asarray(z), n_basis=12)
    res = gam_fit(jnp.asarray(y[None, :]), [vc], intercept=True, n_outer=30)
    xg = np.linspace(x.min(), x.max(), 100)
    eff, _ = smooth_partial_effect(res, 0, vc, jnp.asarray(xg))
    e = np.array(eff[0])
    e = e - e.mean()
    gt = np.sin(2 * np.pi * xg)
    gt = gt - gt.mean()
    assert float(np.corrcoef(e, gt)[0, 1]) > 0.97
