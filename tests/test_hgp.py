# -*- coding: utf-8 -*-
"""Tests for ``nitrix.stats.hgp`` -- hierarchical / multi-level GP regression.

Anchors:

1. **Multi-block REML identity** -- the K-block diagonal penalised-REML criterion
   (``_mb_reml_nll``) equals a dense marginal-likelihood reference for the
   two-variance-component hierarchical model up to an additive constant, across
   both smoothing parameters ``(lam_pop, lam_grp)``.
2. **Hierarchical recovery** -- ``hgp_fit`` recovers the population trend *and* the
   per-group deviations (partial pooling), and ``sigma_grp^2`` reflects how much
   real group-level structure there is.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

jax.config.update('jax_enable_x64', True)

import jax.numpy as jnp

from nitrix.stats import hgp as hgpmod
from nitrix.stats.hgp import HGPResult, hgp_fit, hgp_predict

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hsgp_phi(x, rank, boundary=1.5):
    """The HSGP eigenfunction design and frequencies (mirrors hgp_fit)."""
    lo, hi = float(np.min(x)), float(np.max(x))
    c_mid, big_l = hgpmod._hsgp_domain(lo, hi, boundary)
    sqrt_lambda, phase, inv_sqrt_L = hgpmod._hsgp_eigen(
        rank, c_mid, big_l, jnp.float64
    )
    phi = hgpmod._hsgp_features(jnp.asarray(x), sqrt_lambda, phase, inv_sqrt_L)
    return np.asarray(phi), np.asarray(sqrt_lambda)


def _dense_hier_reml_m2l(y, T, Phi, Phi_factor, s, L, lam_pop, lam_grp):
    """Dense profiled ``-2 l_R`` for the hierarchical model with marginal
    ``M = I + (1/lam_pop) Phi diag(s) Phi^T + (1/lam_grp) Phi_f (I_L kron diag(s))
    Phi_f^T`` (the reference for the multi-block p-space REML)."""
    n = y.shape[0]
    m0 = T.shape[1]
    Sp = np.diag(s)
    Spop = Phi @ (Sp / lam_pop) @ Phi.T
    blk = np.kron(np.eye(L), Sp / lam_grp)
    Sgrp = Phi_factor @ blk @ Phi_factor.T
    M = np.eye(n) + Spop + Sgrp
    Minv = np.linalg.inv(M)
    A = T.T @ Minv @ T
    alpha = np.linalg.solve(A, T.T @ Minv @ y)
    r = y - T @ alpha
    rss = r @ Minv @ r
    _, ldM = np.linalg.slogdet(M)
    _, ldA = np.linalg.slogdet(A)
    return (n - m0) * np.log(rss) + ldM + ldA


def _hier_data(rng, L=8, per=20, pop_amp=1.0, dev_amp=0.4, noise=0.1):
    """Population sinusoid + per-group phase-shifted deviations + noise."""
    t = np.linspace(0.0, 1.0, per)
    x = np.tile(t, L)
    group = np.repeat(np.arange(L), per)
    pop = pop_amp * np.sin(2 * np.pi * t)
    devs = [dev_amp * np.sin(2 * np.pi * t + ph)
            for ph in np.linspace(0.0, 2.0, L)]
    y = np.concatenate([pop + devs[g] + noise * rng.standard_normal(per)
                        for g in range(L)])
    return x, group, y, t, pop, devs


# ---------------------------------------------------------------------------
# 1. Multi-block REML correctness
# ---------------------------------------------------------------------------


def test_mb_reml_matches_dense_up_to_constant():
    """The 2-block p-space REML equals the dense hierarchical marginal-likelihood
    REML up to a single additive constant, across ``(lam_pop, lam_grp)``."""
    rng = np.random.default_rng(0)
    L, per, rank = 5, 12, 8
    t = np.linspace(0.0, 1.0, per)
    x = np.tile(t, L)
    group = np.repeat(np.arange(L), per)
    n = L * per
    y = np.sin(2 * np.pi * x) + 0.2 * rng.standard_normal(n)

    Phi, sqrt_lambda = _hsgp_phi(x, rank)
    T = np.ones((n, 1))
    m0 = 1
    Phi_factor = np.asarray(
        hgpmod._factor_smooth_design(jnp.asarray(Phi), jnp.asarray(group), L)
    )
    X = np.concatenate([T, Phi, Phi_factor], axis=1)
    p = X.shape[1]
    xtx = jnp.asarray(X.T @ X)
    c = jnp.asarray(X.T @ y)
    gg = jnp.asarray(y @ y)

    s = np.asarray(
        hgpmod.spectral_density(jnp.asarray(sqrt_lambda), kernel='matern52',
                                rho=0.2)
    )
    inv_s = 1.0 / np.clip(s, 1e-30, None)
    d_blocks, ranks, _ = hgpmod._block_diag_weights(
        jnp.asarray(inv_s), m0, L
    )
    log_pdets = jnp.asarray(
        [np.sum(np.log(inv_s)), L * np.sum(np.log(inv_s))]
    )

    diffs = []
    for lam_pop in (0.5, 5.0, 50.0):
        for lam_grp in (1.0, 20.0):
            lam = jnp.asarray([lam_pop, lam_grp])
            _, logdet_h, _, _, _, d_p = hgpmod._mb_quantities(
                lam, c, gg, xtx, d_blocks, p, 0.0
            )
            pspace = float(
                hgpmod._mb_reml_nll(d_p, logdet_h, lam, ranks, log_pdets, n, m0)
            )
            dense = _dense_hier_reml_m2l(
                y, T, Phi, Phi_factor, s, L, lam_pop, lam_grp
            )
            diffs.append(pspace - dense)

    diffs = np.asarray(diffs)
    assert np.ptp(diffs) < 1e-6, (
        f'multi-block REML vs dense not constant-offset: {np.ptp(diffs):.2e}'
    )


# ---------------------------------------------------------------------------
# 2. Hierarchical recovery + partial pooling
# ---------------------------------------------------------------------------


def test_hgp_recovers_population_and_group_curves():
    """``hgp_fit`` recovers the population trend and each group's curve."""
    rng = np.random.default_rng(1)
    x, group, y, t, pop, devs = _hier_data(rng, L=8, per=20)
    Y = jnp.asarray(y[None, :])
    res = hgp_fit(Y, jnp.asarray(x), jnp.asarray(group), kernel='matern52',
                  rank=10, n_rho=16)
    assert isinstance(res, HGPResult)
    assert res.coef.shape == (1, 1 + 10 + 8 * 10)
    assert res.theta.shape == (1, 4)

    pop_mean, pop_std = hgp_predict(res, jnp.asarray(t))
    assert pop_mean.shape == (1, len(t))
    assert np.corrcoef(np.asarray(pop_mean)[0], pop)[0, 1] > 0.95
    assert np.all(np.asarray(pop_std) > 0)

    gmean, gstd = hgp_predict(res, jnp.asarray(t), levels=jnp.arange(8))
    assert gmean.shape == (1, 8, len(t))
    cors = [np.corrcoef(np.asarray(gmean)[0, g], pop + devs[g])[0, 1]
            for g in range(8)]
    assert np.mean(cors) > 0.95
    # Group curves track their own data better than the bare population curve.
    pop_cors = [np.corrcoef(np.asarray(pop_mean)[0], pop + devs[g])[0, 1]
                for g in range(8)]
    assert np.mean(cors) > np.mean(pop_cors)


def test_hgp_group_amplitude_reflects_structure():
    """``sigma_grp^2`` is large when groups genuinely differ and small when they
    don't (partial pooling shrinks absent group structure)."""
    rng = np.random.default_rng(2)
    # Strong per-group deviations.
    x1, g1, y1, *_ = _hier_data(rng, L=10, per=18, dev_amp=0.8)
    res_strong = hgp_fit(jnp.asarray(y1[None, :]), jnp.asarray(x1),
                         jnp.asarray(g1), rank=10, n_rho=16)
    # No real group deviation: all groups share the population trend.
    t = np.linspace(0.0, 1.0, 18)
    pop = np.sin(2 * np.pi * t)
    y0 = np.concatenate([pop + 0.1 * rng.standard_normal(18) for _ in range(10)])
    res_none = hgp_fit(jnp.asarray(y0[None, :]), jnp.asarray(x1),
                       jnp.asarray(g1), rank=10, n_rho=16)

    sgrp_strong = float(np.exp(res_strong.theta[0, 1]))
    sgrp_none = float(np.exp(res_none.theta[0, 1]))
    spop_none = float(np.exp(res_none.theta[0, 0]))
    assert sgrp_strong > sgrp_none
    # With no group structure the group amplitude is shrunk well below the
    # population amplitude.
    assert sgrp_none < 0.25 * spop_none


def test_hgp_mass_univariate_shared_rho():
    """``rho`` is shared across elements; replicated rows give identical fits."""
    rng = np.random.default_rng(3)
    x, group, y, *_ = _hier_data(rng, L=6, per=16)
    Y = np.stack([y, y, y])
    res = hgp_fit(jnp.asarray(Y), jnp.asarray(x), jnp.asarray(group),
                  rank=8, n_rho=14)
    log_rho = np.asarray(res.theta[:, 3])
    assert np.ptp(log_rho) < 1e-9
    assert np.allclose(res.coef[1], res.coef[0], atol=1e-8)
    assert np.allclose(res.theta[1], res.theta[0], atol=1e-8)


# ---------------------------------------------------------------------------
# 3. predict / parametric / argument validation
# ---------------------------------------------------------------------------


def test_hgp_predict_parametric():
    rng = np.random.default_rng(5)
    x, group, y, t, pop, devs = _hier_data(rng, L=6, per=18)
    z = rng.standard_normal((len(x), 1))
    y = y + 1.2 * z[:, 0]
    res = hgp_fit(jnp.asarray(y[None, :]), jnp.asarray(x), jnp.asarray(group),
                  parametric=jnp.asarray(z), rank=8, n_rho=14)
    assert res.n_fixed == 2
    assert abs(float(res.coef[0, 1]) - 1.2) < 0.25
    zt = rng.standard_normal((len(t), 1))
    mean, _ = hgp_predict(res, jnp.asarray(t), parametric=jnp.asarray(zt))
    assert mean.shape == (1, len(t))
    with pytest.raises(ValueError):
        hgp_predict(res, jnp.asarray(t))  # missing parametric


def test_hgp_argument_validation():
    rng = np.random.default_rng(7)
    x, group, y, *_ = _hier_data(rng, L=4, per=12)
    Y = jnp.asarray(y[None, :])
    with pytest.raises(NotImplementedError):
        hgp_fit(Y, jnp.asarray(x), jnp.asarray(group), model='GI')
    with pytest.raises(ValueError):
        hgp_fit(Y, jnp.asarray(x), jnp.asarray(group), rank=0)
    with pytest.raises(ValueError):
        hgp_fit(Y, jnp.asarray(x), jnp.asarray(group), boundary=0.5)
    with pytest.raises(ValueError):
        hgp_fit(Y, jnp.asarray(x[:5]), jnp.asarray(group))  # length mismatch
