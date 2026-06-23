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
    d_blocks, ranks, _ = hgpmod._block_weights(
        jnp.asarray(inv_s), m0, (L,)
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


def test_hgp_group_label_range_validation():
    """ER1: out-of-range / negative group labels would be silently zero-rowed by
    one_hot (dropping observations); hgp_fit must reject them instead."""
    rng = np.random.default_rng(8)
    x, group, y, *_ = _hier_data(rng, L=4, per=12)
    Y = jnp.asarray(y[None, :])
    # n_levels too small for the labels present
    with pytest.raises(ValueError, match='group labels must lie'):
        hgp_fit(Y, jnp.asarray(x), jnp.asarray(group), n_levels=3)
    # negative label
    bad = np.array(group)
    bad[: len(bad) // 4] = -1
    with pytest.raises(ValueError, match='group labels must lie'):
        hgp_fit(Y, jnp.asarray(x), jnp.asarray(bad))
    # nested inner-label range is validated too
    inner = np.zeros_like(group)
    with pytest.raises(ValueError, match='group_inner labels must lie'):
        hgp_fit(Y, jnp.asarray(x), jnp.asarray(group), model='nested',
                group_inner=jnp.asarray(inner), n_levels_inner=0)


def test_hgp_nested_requires_global_inner_numbering():
    """ER2: a per-outer-numbered inner factor (each outer's inner labels restart
    at 0) passes the range check but aliases distinct subjects -> mis-pool; the
    nested fit must reject it. A globally-numbered inner factor is accepted."""
    rng = np.random.default_rng(12)
    L, inner_per, per = 3, 2, 12
    t = np.linspace(0.0, 1.0, per)
    x = np.tile(t, L * inner_per)
    outer = np.repeat(np.arange(L), inner_per * per)
    # per-outer numbering: inner labels 0..inner_per-1 restart within each outer
    inner_local = np.tile(np.repeat(np.arange(inner_per), per), L)
    # global numbering: unique inner id per (outer, inner_local) pair
    inner_global = np.repeat(np.arange(L * inner_per), per)
    y = (np.sin(2 * np.pi * x) + 0.1 * rng.standard_normal(x.size))[None, :]
    Y = jnp.asarray(y)
    with pytest.raises(ValueError, match='globally-numbered inner'):
        hgp_fit(Y, jnp.asarray(x), jnp.asarray(outer), model='nested',
                group_inner=jnp.asarray(inner_local))
    res = hgp_fit(Y, jnp.asarray(x), jnp.asarray(outer), model='nested',
                  group_inner=jnp.asarray(inner_global), rank=6, n_rho=8)
    assert res.model == 'nested'


def test_gp_hgp_accept_integer_responses():
    """ER6: an integer-dtype response is promoted to float (not coerced to int,
    which would zero the whole fit)."""
    rng = np.random.default_rng(15)
    n = 60
    x = np.sort(rng.uniform(0.0, 1.0, n))
    Yi = np.rint(5 * np.sin(2 * np.pi * x) + 10).astype(np.int64)[None, :]
    from nitrix.stats.gp import gp_fit
    rg = gp_fit(jnp.asarray(Yi), jnp.asarray(x), n_rho=8)
    assert jnp.issubdtype(rg.coef.dtype, jnp.floating)
    assert np.all(np.isfinite(np.asarray(rg.coef))) and np.any(rg.coef != 0)
    g = jnp.asarray(np.repeat(np.arange(6), 10))
    rh = hgp_fit(jnp.asarray(Yi), jnp.asarray(x), g, rank=6, n_rho=8)
    assert jnp.issubdtype(rh.coef.dtype, jnp.floating)


def test_hgp_block_bounds_rho_search_without_changing_results():
    """PF1: `block=` chunks the hierarchical pooled-NLL rho search too (the
    (1+L)*m-wide design is the acuter cliff); the result is identical to the
    un-chunked search."""
    rng = np.random.default_rng(9)
    x, group, y, *_ = _hier_data(rng, L=6, per=16)
    V = 6
    Y = jnp.asarray(np.tile(y, (V, 1)) + 0.05 * rng.standard_normal((V, len(y))))
    a = hgp_fit(Y, jnp.asarray(x), jnp.asarray(group), n_rho=10, block=None)
    b = hgp_fit(Y, jnp.asarray(x), jnp.asarray(group), n_rho=10, block=2)
    np.testing.assert_allclose(np.asarray(a.coef), np.asarray(b.coef), atol=1e-10)
    np.testing.assert_allclose(
        np.asarray(a.theta), np.asarray(b.theta), atol=1e-10
    )


def test_hgp_explicit_bounds_and_n_levels():
    """ER7: explicit `bounds=` overrides the HSGP domain (changing the fit), and
    an explicit `n_levels=` matching the data is byte-identical to the inferred
    layout (the static-count jit-traceability hook)."""
    rng = np.random.default_rng(3)
    x, group, y, *_ = _hier_data(rng, L=5, per=12)
    Y = jnp.asarray(y[None, :])
    xa, ga = jnp.asarray(x), jnp.asarray(group)

    default = hgp_fit(Y, xa, ga, rank=8, n_rho=8)
    widened = hgp_fit(Y, xa, ga, rank=8, n_rho=8, bounds=(-0.5, 1.5))
    assert (widened.lo, widened.hi) == (-0.5, 1.5)
    assert (default.lo, default.hi) != (widened.lo, widened.hi)
    assert not np.allclose(np.asarray(default.coef), np.asarray(widened.coef))

    explicit = hgp_fit(Y, xa, ga, rank=8, n_rho=8, n_levels=5)
    assert explicit.n_levels == 5
    np.testing.assert_allclose(
        np.asarray(default.coef), np.asarray(explicit.coef), atol=1e-10
    )


# ---------------------------------------------------------------------------
# 4. gp_factor_smooth -- the fixed-rho factor-smooth GP basis (gam_fit drop-in)
# ---------------------------------------------------------------------------


def test_gp_factor_smooth_contract():
    """Design width ``L*m``, one shared (identity) penalty block, tuple eval."""
    from nitrix.stats import gp_factor_smooth

    rng = np.random.default_rng(0)
    x, group, _y, *_ = _hier_data(rng, L=5, per=12)
    fb = gp_factor_smooth(jnp.asarray(x), jnp.asarray(group), 8,
                          kernel='matern52', rho=0.2)
    m = fb.base.dim
    assert fb.dim == 5 * m
    assert fb.design.shape == (len(x), 5 * m)
    blocks = fb.penalty_blocks()
    assert len(blocks) == 1  # one shared smoothing parameter
    S, eig = blocks[0]
    assert S.shape == (5 * m, 5 * m)
    assert np.allclose(S, np.eye(5 * m))  # identity ridge
    # tuple eval re-evaluates to the same design at the training points.
    fe = fb.eval_design((jnp.asarray(x), jnp.asarray(group)))
    assert np.allclose(np.asarray(fe), np.asarray(fb.design), atol=1e-10)


def test_gp_factor_smooth_gam_fit_recovers_gs():
    """``gam_fit([hsgp_basis, gp_factor_smooth])`` is the GS hierarchical GP at
    fixed ``rho``: it recovers the per-group curves, with one shared group
    smoothing parameter."""
    from nitrix.stats import gam_fit, gp_factor_smooth, hsgp_basis

    rng = np.random.default_rng(1)
    x, group, y, t, pop, devs = _hier_data(rng, L=6, per=20)
    rho = 0.2
    pop_b = hsgp_basis(jnp.asarray(x), 12, kernel='matern52', rho=rho)
    fac_b = gp_factor_smooth(jnp.asarray(x), jnp.asarray(group), 10,
                             kernel='matern52', rho=rho)
    res = gam_fit(jnp.asarray(y[None, :]), [pop_b, fac_b])
    # Two smoothing parameters: population and the single shared group block.
    assert res.lam.shape == (1, 2)

    X = np.concatenate(
        [np.ones((len(x), 1)), np.asarray(pop_b.design),
         np.asarray(fac_b.design)], axis=1
    )
    fitted = np.asarray(res.coef[0]) @ X.T
    truth = np.concatenate([pop + devs[g] for g in range(6)])
    assert np.corrcoef(fitted, truth)[0, 1] > 0.99


def test_gp_factor_smooth_n_levels_stable_width():
    """``n_levels`` fixes the block width even when a level is absent."""
    from nitrix.stats import gp_factor_smooth

    rng = np.random.default_rng(2)
    x = np.sort(rng.uniform(0.0, 1.0, 40))
    group = np.where(x < 0.5, 0, 2)  # levels {0, 2}; level 1 absent
    fb = gp_factor_smooth(jnp.asarray(x), jnp.asarray(group), 6,
                          kernel='matern52', rho=0.2, n_levels=3)
    assert fb.n_levels == 3
    assert fb.dim == 3 * fb.base.dim


# ---------------------------------------------------------------------------
# 5. Nested two-level HGP (gp | g1/g2)
# ---------------------------------------------------------------------------


def _nested_data(rng, n_out=3, n_in=4, per=16, out_amp=0.5, in_amp=0.3,
                 noise=0.08):
    """Population + outer-site + inner-subject(nested) deviations."""
    t = np.linspace(0.0, 1.0, per)
    pop = np.sin(2 * np.pi * t)
    rows, xs, g1, g2 = [], [], [], []
    inner = 0
    for o in range(n_out):
        out_dev = out_amp * np.sin(2 * np.pi * t + o)
        for _ in range(n_in):
            in_dev = in_amp * np.sin(2 * np.pi * t + inner * 0.4)
            rows.append(pop + out_dev + in_dev + noise * rng.standard_normal(per))
            xs.append(t)
            g1.append(np.full(per, o))
            g2.append(np.full(per, inner))
            inner += 1
    return (
        np.concatenate(xs),
        np.concatenate(g1).astype(int),
        np.concatenate(g2).astype(int),
        np.concatenate(rows),
        t, pop, n_out, inner,
    )


def _dense_nested_reml_m2l(y, T, Phi, Phif1, Phif2, s, L1, L2,
                           lam_pop, lam_out, lam_in):
    """Dense profiled ``-2 l_R`` for the nested model (three GP components)."""
    n = y.shape[0]
    m0 = T.shape[1]
    Sp = np.diag(s)
    M = (
        np.eye(n)
        + Phi @ (Sp / lam_pop) @ Phi.T
        + Phif1 @ np.kron(np.eye(L1), Sp / lam_out) @ Phif1.T
        + Phif2 @ np.kron(np.eye(L2), Sp / lam_in) @ Phif2.T
    )
    Minv = np.linalg.inv(M)
    A = T.T @ Minv @ T
    alpha = np.linalg.solve(A, T.T @ Minv @ y)
    r = y - T @ alpha
    rss = r @ Minv @ r
    _, ldM = np.linalg.slogdet(M)
    _, ldA = np.linalg.slogdet(A)
    return (n - m0) * np.log(rss) + ldM + ldA


def test_nested_mb_reml_matches_dense():
    """The 3-block nested p-space REML equals the dense reference up to a
    constant, across the three smoothing parameters."""
    rng = np.random.default_rng(0)
    rank = 6
    x, g1, g2, y, _t, _pop, L1, L2 = _nested_data(rng, n_out=2, n_in=2, per=10)
    n = len(x)
    Phi, sqrt_lambda = _hsgp_phi(x, rank)
    T = np.ones((n, 1))
    m0 = 1
    Phif1 = np.asarray(
        hgpmod._factor_smooth_design(jnp.asarray(Phi), jnp.asarray(g1), L1)
    )
    Phif2 = np.asarray(
        hgpmod._factor_smooth_design(jnp.asarray(Phi), jnp.asarray(g2), L2)
    )
    X = np.concatenate([T, Phi, Phif1, Phif2], axis=1)
    p = X.shape[1]
    xtx = jnp.asarray(X.T @ X)
    c = jnp.asarray(X.T @ y)
    gg = jnp.asarray(y @ y)
    s = np.asarray(
        hgpmod.spectral_density(jnp.asarray(sqrt_lambda), kernel='matern52',
                                rho=0.2)
    )
    inv_s = 1.0 / np.clip(s, 1e-30, None)
    d_blocks, ranks, _ = hgpmod._block_weights(jnp.asarray(inv_s), m0, (L1, L2))
    log_pdets = jnp.asarray(
        [np.sum(np.log(inv_s)), L1 * np.sum(np.log(inv_s)),
         L2 * np.sum(np.log(inv_s))]
    )

    diffs = []
    for lp in (1.0, 10.0):
        for lo_ in (2.0, 20.0):
            for li in (5.0,):
                lam = jnp.asarray([lp, lo_, li])
                _, ldh, _, _, _, dp = hgpmod._mb_quantities(
                    lam, c, gg, xtx, d_blocks, p, 0.0
                )
                pspace = float(
                    hgpmod._mb_reml_nll(dp, ldh, lam, ranks, log_pdets, n, m0)
                )
                dense = _dense_nested_reml_m2l(
                    y, T, Phi, Phif1, Phif2, s, L1, L2, lp, lo_, li
                )
                diffs.append(pspace - dense)
    diffs = np.asarray(diffs)
    assert np.ptp(diffs) < 1e-6, f'nested REML offset spread {np.ptp(diffs):.2e}'


def test_hgp_nested_fit_and_components():
    """``hgp_fit(model='nested')`` returns three GP variance components, recovers
    the population trend, and accepts gp_aic."""
    from nitrix.stats import gp_aic

    rng = np.random.default_rng(1)
    x, g1, g2, y, t, pop, _L1, _L2 = _nested_data(rng, n_out=4, n_in=4, per=20)
    res = hgp_fit(jnp.asarray(y[None, :]), jnp.asarray(x), jnp.asarray(g1),
                  group_inner=jnp.asarray(g2), model='nested', rank=8, n_rho=18)
    assert res.model == 'nested'
    assert res.theta.shape == (1, 5)  # 3 GP + noise + rho
    assert res.edf.shape == (1, 3)
    assert tuple(res.n_levels) == (4, 16)
    assert np.isfinite(float(gp_aic(res)[0]))
    pop_mean, _ = hgp_predict(res, jnp.asarray(t))
    assert np.corrcoef(np.asarray(pop_mean)[0], pop)[0, 1] > 0.9


def test_hgp_nested_requires_group_inner():
    rng = np.random.default_rng(2)
    x, g1, _g2, y, *_ = _nested_data(rng, n_out=2, n_in=2, per=10)
    with pytest.raises(ValueError):
        hgp_fit(jnp.asarray(y[None, :]), jnp.asarray(x), jnp.asarray(g1),
                model='nested')
