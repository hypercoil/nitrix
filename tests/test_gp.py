# -*- coding: utf-8 -*-
"""Tests for ``nitrix.stats.gp`` -- HSGP Gaussian-process regression with
REML-estimated lengthscale (``gp_fit`` / ``gp_predict``).

The correctness anchors are:

1. **REML identity** -- the efficient ``p``-space profiled restricted negative
   log-likelihood (``_reml_nll``) equals a dense ``(N, N)`` marginal-likelihood
   reference up to an additive constant in ``(n, M_0)`` only, across ``(lambda,
   rho)``.  This pins the criterion that drives the lengthscale search.
2. **scikit-learn exact GP** -- ``gp_fit``'s estimated lengthscale and predictive
   mean track an exact ``GaussianProcessRegressor`` optimised on the same data.
3. **mass-univariate invariants** -- shared ``rho``, per-element amplitude/noise,
   and a cuSOLVER-free, ``N``-free final-fit HLO (no per-voxel ``(N, N)`` GP
   covariance).
"""

from __future__ import annotations

import re

import jax
import numpy as np
import pytest

jax.config.update('jax_enable_x64', True)

import jax.numpy as jnp

from nitrix.linalg.kernel import spectral_density
from nitrix.stats import gp as gpmod
from nitrix.stats.gp import GPResult, gp_aic, gp_bic, gp_fit, gp_predict

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_design(x, rank, boundary, n_fixed_param=0):
    """The fixed HSGP design ``X = [intercept | parametric? | Phi]`` and the
    eigen-frequencies, mirroring ``gp_fit``'s internal construction."""
    lo, hi = float(np.min(x)), float(np.max(x))
    c_mid, big_l = gpmod._hsgp_domain(lo, hi, boundary)
    sqrt_lambda, phase, inv_sqrt_L = gpmod._hsgp_eigen(
        rank, c_mid, big_l, jnp.float64
    )
    phi = gpmod._hsgp_features(jnp.asarray(x), sqrt_lambda, phase, inv_sqrt_L)
    return np.asarray(phi), np.asarray(sqrt_lambda), (lo, hi)


def _dense_reml_m2l(y, T, Phi, s, lam):
    """Dense profiled restricted ``-2 l_R`` (up to an additive constant in
    ``n``/``M_0``) for ``y = T alpha + Phi gamma + e``, ``gamma ~ N(0, (1/lam)
    diag(s) sigma^2)``, ``e ~ N(0, sigma^2)``, via the ``(N, N)`` marginal
    ``M = I + Phi diag(s)/lam Phi^T`` (the gold reference for ``_reml_nll``)."""
    n = y.shape[0]
    m0 = T.shape[1]
    M = np.eye(n) + Phi @ (np.diag(s) / lam) @ Phi.T
    Minv = np.linalg.inv(M)
    A = T.T @ Minv @ T
    alpha = np.linalg.solve(A, T.T @ Minv @ y)
    r = y - T @ alpha
    rss_m = r @ Minv @ r
    _, logdet_m = np.linalg.slogdet(M)
    _, logdet_a = np.linalg.slogdet(A)
    return (n - m0) * np.log(rss_m) + logdet_m + logdet_a


def _gp_draw(rng, x, rho, amp=1.0, nu=2.5, noise=0.1):
    """A noisy draw from an exact Matern GP on ``x`` (the ground-truth signal)."""
    from sklearn.gaussian_process.kernels import Matern

    K = amp * Matern(length_scale=rho, nu=nu)(x[:, None])
    K += 1e-9 * np.eye(len(x))
    f = rng.multivariate_normal(np.zeros(len(x)), K)
    y = f + noise * rng.standard_normal(len(x))
    return y, f


# ---------------------------------------------------------------------------
# 1. REML criterion correctness
# ---------------------------------------------------------------------------


def test_reml_pspace_matches_dense_up_to_constant():
    """``_reml_nll`` (p-space, profiled scale) equals the dense ``(N, N)``
    marginal-likelihood REML up to a single additive constant -- across both the
    lengthscale ``rho`` and the smoothing parameter ``lambda``."""
    rng = np.random.default_rng(0)
    n, rank, boundary = 60, 12, 1.5
    x = np.sort(rng.uniform(0.0, 1.0, n))
    y, _ = _gp_draw(rng, x, rho=0.15, noise=0.1)

    Phi, sqrt_lambda, _ = _build_design(x, rank, boundary)
    T = np.ones((n, 1))
    X = np.concatenate([T, Phi], axis=1)
    m0, p, m = 1, X.shape[1], rank
    xtx = jnp.asarray(X.T @ X)
    c = jnp.asarray(X.T @ y)
    g = jnp.asarray(y @ y)

    diffs = []
    for kernel, nu_s in [('matern52', 2.5)]:
        for rho in (0.05, 0.1, 0.2, 0.4):
            d, log_pdet_pen = gpmod._penalty_diag(
                jnp.asarray(sqrt_lambda), kernel, jnp.asarray(rho), m0
            )
            s = np.asarray(
                spectral_density(
                    jnp.asarray(sqrt_lambda), kernel=kernel, rho=rho
                )
            )
            for lam in (0.1, 1.0, 10.0, 100.0):
                _, logdet_h, _, _, _, d_p = gpmod._quantities(
                    jnp.asarray(lam), c, g, xtx, d, p, 0.0
                )
                pspace = float(
                    gpmod._reml_nll(
                        d_p, logdet_h, jnp.asarray(lam),
                        log_pdet_pen, n, m, m0,
                    )
                )
                dense = _dense_reml_m2l(y, T, Phi, s, lam)
                diffs.append(pspace - dense)

    diffs = np.asarray(diffs)
    # The p-space and dense forms differ by a constant in (n, M_0) only.
    assert np.ptp(diffs) < 1e-6, (
        f'REML p-space vs dense not constant-offset: spread={np.ptp(diffs):.2e}'
    )


def test_reml_minimum_near_truth():
    """The pooled REML profile over ``rho`` is minimised near the data-generating
    lengthscale (single long series -- the criterion is well-identified)."""
    rng = np.random.default_rng(3)
    n, rank = 200, 30
    x = np.sort(rng.uniform(0.0, 1.0, n))
    rho_true = 0.12
    y, _ = _gp_draw(rng, x, rho=rho_true, noise=0.08)

    res = gp_fit(jnp.asarray(y[None, :]), jnp.asarray(x), kernel='matern52',
                 rank=rank, n_rho=40)
    rho_hat = float(np.exp(res.theta[0, 2]))
    # Reduced-rank REML recovers the lengthscale to within a factor ~2.
    assert 0.5 * rho_true < rho_hat < 2.0 * rho_true, rho_hat


# ---------------------------------------------------------------------------
# 2. scikit-learn exact-GP anchor (whole pipeline incl. rho estimation)
# ---------------------------------------------------------------------------


def test_gp_fit_matches_sklearn_gpr():
    """``gp_fit`` lengthscale and predictive mean track an exact
    ``GaussianProcessRegressor`` (ML-optimised) on the same data."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import (
        ConstantKernel as C,
    )
    from sklearn.gaussian_process.kernels import (
        Matern,
        WhiteKernel,
    )

    rng = np.random.default_rng(11)
    n, rank = 120, 30
    x = np.sort(rng.uniform(0.0, 1.0, n))
    y, f = _gp_draw(rng, x, rho=0.18, amp=1.0, nu=2.5, noise=0.12)

    kern = (
        C(1.0, (1e-3, 1e3))
        * Matern(length_scale=0.2, nu=2.5, length_scale_bounds=(1e-2, 1e1))
        + WhiteKernel(0.05, (1e-5, 1e1))
    )
    gpr = GaussianProcessRegressor(
        kernel=kern, normalize_y=True, n_restarts_optimizer=3
    ).fit(x[:, None], y)
    ls = gpr.kernel_.k1.k2.length_scale

    res = gp_fit(jnp.asarray(y[None, :]), jnp.asarray(x), kernel='matern52',
                 rank=rank, n_rho=32)
    rho_hat = float(np.exp(res.theta[0, 2]))
    # Lengthscale parameterisation matches sklearn's Matern -- same ballpark.
    assert 0.5 * ls < rho_hat < 2.0 * ls, (rho_hat, ls)

    xg = np.linspace(0.02, 0.98, 80)
    mean_nitrix, _ = gp_predict(res, jnp.asarray(xg))
    mean_sklearn = gpr.predict(xg[:, None])
    mean_nitrix = np.asarray(mean_nitrix)[0]
    assert np.corrcoef(mean_nitrix, mean_sklearn)[0, 1] > 0.99
    assert np.sqrt(np.mean((mean_nitrix - mean_sklearn) ** 2)) < 0.15
    # And both recover the latent signal.
    f_grid = np.interp(xg, x, f)
    assert np.corrcoef(mean_nitrix, f_grid)[0, 1] > 0.95


# ---------------------------------------------------------------------------
# 3. predict / round-trip / shapes
# ---------------------------------------------------------------------------


def test_gp_predict_shapes_and_consistency():
    """``gp_predict`` returns ``(V, g)`` mean/std; the mean at the training grid
    matches ``coef @ X_train`` (the fitted values) and std is positive."""
    rng = np.random.default_rng(5)
    n, rank, V = 80, 20, 3
    x = np.sort(rng.uniform(-1.0, 1.0, n))
    Y = np.stack([_gp_draw(rng, x, rho=0.3, noise=0.1)[0] for _ in range(V)])

    res = gp_fit(jnp.asarray(Y), jnp.asarray(x), rank=rank, n_rho=20)
    assert isinstance(res, GPResult)
    assert res.coef.shape == (V, 1 + rank)
    assert res.cov_unscaled.shape == (V, 1 + rank, 1 + rank)
    assert res.theta.shape == (V, 3)
    assert res.log_mlik.shape == (V,)

    mean, std = gp_predict(res, jnp.asarray(x))
    assert mean.shape == (V, n)
    assert std.shape == (V, n)
    assert np.all(np.asarray(std) > 0)

    # The reconstructed training design reproduces the fitted values.
    Phi, _, _ = _build_design(x, rank, res.boundary)
    X = np.concatenate([np.ones((n, 1)), Phi], axis=1)
    fitted = np.asarray(res.coef) @ X.T
    assert np.allclose(np.asarray(mean), fitted, atol=1e-8)


def test_gp_predict_parametric():
    """A parametric covariate is carried through fit and predict."""
    rng = np.random.default_rng(7)
    n, rank = 90, 16
    x = np.sort(rng.uniform(0.0, 1.0, n))
    z = rng.standard_normal((n, 1))
    truth = np.sin(2 * np.pi * x) + 1.5 * z[:, 0]
    y = truth + 0.1 * rng.standard_normal(n)

    res = gp_fit(jnp.asarray(y[None, :]), jnp.asarray(x),
                 parametric=jnp.asarray(z), rank=rank, n_rho=20)
    assert res.n_fixed == 2
    # Slope on z recovered (coef[:, 1] is the parametric column).
    assert abs(float(res.coef[0, 1]) - 1.5) < 0.2

    xg = np.linspace(0.05, 0.95, 30)
    zg = rng.standard_normal((30, 1))
    mean, _ = gp_predict(res, jnp.asarray(xg), parametric=jnp.asarray(zg))
    assert mean.shape == (1, 30)
    # Missing parametric at predict is an error.
    with pytest.raises(ValueError):
        gp_predict(res, jnp.asarray(xg))


# ---------------------------------------------------------------------------
# 4. mass-univariate invariants: shared rho, per-element amplitude/noise
# ---------------------------------------------------------------------------


def test_gp_fit_shared_rho_per_voxel_amplitude():
    """``rho`` is shared across elements; amplitude and noise are per element."""
    rng = np.random.default_rng(9)
    n, rank = 150, 28
    x = np.sort(rng.uniform(0.0, 1.0, n))
    base, f = _gp_draw(rng, x, rho=0.2, amp=1.0, noise=1e-3)
    # Same shape, different per-voxel amplitude + noise.
    amps = np.array([0.5, 1.0, 2.0])
    noises = np.array([0.05, 0.1, 0.2])
    Y = np.stack([
        a * f + s * rng.standard_normal(n) for a, s in zip(amps, noises)
    ])

    res = gp_fit(jnp.asarray(Y), jnp.asarray(x), rank=rank, n_rho=24)
    log_rho = np.asarray(res.theta[:, 2])
    assert np.ptp(log_rho) < 1e-9  # one shared rho

    sigma_f2 = np.exp(np.asarray(res.theta[:, 0]))
    # Amplitude ordering tracks the injected per-voxel scale.
    assert sigma_f2[0] < sigma_f2[1] < sigma_f2[2]
    sigma_e2 = np.asarray(res.dispersion)
    assert sigma_e2[0] < sigma_e2[2]  # noisier voxel -> larger sigma_e^2


def test_gp_fit_identical_voxels_identical_fit():
    """Replicated rows give identical per-element output (vmap consistency)."""
    rng = np.random.default_rng(13)
    n, rank = 100, 20
    x = np.sort(rng.uniform(0.0, 1.0, n))
    y, _ = _gp_draw(rng, x, rho=0.25, noise=0.1)
    Y = np.tile(y, (4, 1))

    res = gp_fit(jnp.asarray(Y), jnp.asarray(x), rank=rank, n_rho=20)
    for k in range(1, 4):
        assert np.allclose(res.coef[k], res.coef[0], atol=1e-9)
        assert np.allclose(res.theta[k], res.theta[0], atol=1e-9)
        assert np.allclose(res.log_mlik[k], res.log_mlik[0], atol=1e-7)


# ---------------------------------------------------------------------------
# 5. MAP prior on rho
# ---------------------------------------------------------------------------


def test_map_rho_shrinks_lengthscale():
    """A strong prior favouring large ``rho`` pulls the estimate up relative to
    the pure-REML fit (the MAP term enters the pooled objective)."""
    rng = np.random.default_rng(17)
    n, rank = 120, 28
    x = np.sort(rng.uniform(0.0, 1.0, n))
    y, _ = _gp_draw(rng, x, rho=0.1, noise=0.1)

    res_ml = gp_fit(jnp.asarray(y[None, :]), jnp.asarray(x), rank=rank,
                    n_rho=32)
    # -log p(rho): half-normal-in-(1/rho) style prior penalising small rho.
    res_map = gp_fit(
        jnp.asarray(y[None, :]), jnp.asarray(x), rank=rank, n_rho=32,
        map_rho=lambda rho: 5.0e2 * (1.0 / rho) ** 2,
    )
    rho_ml = float(np.exp(res_ml.theta[0, 2]))
    rho_map = float(np.exp(res_map.theta[0, 2]))
    assert rho_map > rho_ml


# ---------------------------------------------------------------------------
# 6. cuSOLVER-free, N-free final fit (HLO budget)
# ---------------------------------------------------------------------------


def _cusolver_calls(hlo: str):
    targets = set(re.findall(r'custom_call_target="([^"]+)"', hlo))
    return {t for t in targets if 'cusolver' in t.lower() or t.startswith('cu')}


def test_gp_final_fit_hlo_is_cusolver_free_and_N_free():
    """The vmapped per-element final fit compiles with no cuSOLVER custom-call
    and no ``N``-sized tensor (the GP covariance never reaches ``(V, N, N)`` --
    ``N`` lives only in the one-off cross-products outside this region)."""
    V, N, rank = 512, 40, 12
    rng = np.random.default_rng(0)
    x = np.sort(rng.uniform(0.0, 1.0, N)).astype(np.float32)
    Phi, sqrt_lambda, _ = _build_design(x.astype(np.float64), rank, 1.5)
    X = np.concatenate([np.ones((N, 1)), Phi], axis=1).astype(np.float32)
    p, m = X.shape[1], rank
    xtx = jnp.asarray(X.T @ X)
    d, log_pdet = gpmod._penalty_diag(
        jnp.asarray(sqrt_lambda.astype(np.float32)), 'matern52',
        jnp.asarray(np.float32(0.2)), 1,
    )
    Y = rng.standard_normal((V, N)).astype(np.float32)
    c_all = jnp.asarray(Y @ X)
    g_all = jnp.asarray(np.sum(Y * Y, axis=1))

    def final(c_all, g_all):
        return jax.vmap(
            lambda cv, gv: gpmod._gp_fit_one(
                cv, gv, xtx, d, log_pdet, N, p, m, 1, 20, 1e-8, 1e-6, 1e8
            )[0]
        )(c_all, g_all)

    f = jax.jit(final)
    hlo = f.lower(c_all, g_all).compile().as_text()
    assert not _cusolver_calls(hlo), _cusolver_calls(hlo)
    shapes = re.findall(r'f32\[([0-9,]+)\]', hlo)
    max_size = 0
    for s in shapes:
        dims = tuple(int(t) for t in s.split(',') if t)
        sz = 1
        for dd in dims:
            sz *= dd
        max_size = max(max_size, sz)
    budget = V * N * N // 2
    assert max_size < budget, (
        f'max tensor {max_size} >= budget {budget}; a (V, N, N) GP covariance?'
    )


def test_gp_fit_runs_under_jit_partial():
    """The final-fit core is jit/vmap clean (smoke: jit the per-element fit)."""
    rng = np.random.default_rng(2)
    n, rank = 64, 16
    x = np.sort(rng.uniform(0.0, 1.0, n))
    y, _ = _gp_draw(rng, x, rho=0.2, noise=0.1)
    res = gp_fit(jnp.asarray(y[None, :]), jnp.asarray(x), rank=rank, n_rho=16)
    assert np.isfinite(float(res.log_mlik[0]))
    assert np.all(np.isfinite(np.asarray(res.coef)))


# ---------------------------------------------------------------------------
# 7. kernels + argument validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('kernel', ['matern12', 'matern32', 'matern52', 'rbf'])
def test_gp_fit_each_kernel_recovers_smooth(kernel):
    """Each supported kernel recovers a smooth signal."""
    rng = np.random.default_rng(21)
    n, rank = 140, 30
    x = np.sort(rng.uniform(0.0, 1.0, n))
    truth = np.sin(2 * np.pi * x)
    y = truth + 0.1 * rng.standard_normal(n)
    res = gp_fit(jnp.asarray(y[None, :]), jnp.asarray(x), kernel=kernel,
                 rank=rank, n_rho=24)
    mean, _ = gp_predict(res, jnp.asarray(x))
    assert np.corrcoef(np.asarray(mean)[0], truth)[0, 1] > 0.95


def test_gp_fit_argument_validation():
    rng = np.random.default_rng(0)
    x = np.sort(rng.uniform(0.0, 1.0, 30))
    y = jnp.asarray(rng.standard_normal((1, 30)))
    xj = jnp.asarray(x)
    with pytest.raises(NotImplementedError):
        gp_fit(y, xj, engine='fourier')  # type: ignore[arg-type]
    with pytest.raises(NotImplementedError):
        gp_fit(y, xj, select='per-voxel')  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        gp_fit(y, xj, corr='ar1')  # group missing
    with pytest.raises(ValueError):
        gp_fit(y, xj, rank=0)
    with pytest.raises(ValueError):
        gp_fit(y, xj, boundary=0.5)
    with pytest.raises(ValueError):
        gp_fit(y, jnp.asarray(x[:10]))  # x/Y length mismatch


# ---------------------------------------------------------------------------
# 8. engine='exact' -- full-rank kernel-eigenfeature GP (PR3a)
# ---------------------------------------------------------------------------


def test_exact_matches_reml_fit_at_fixed_rho():
    """At a fixed ``rho`` the exact engine reproduces ``lme.reml_fit`` on
    ``Z = chol(K_rho)`` -- the penalty<->variance-component identity in code: the
    variance components, their ratio, and the fixed effect all agree."""
    from nitrix.stats.lme import reml_fit

    rng = np.random.default_rng(1)
    n, V = 50, 4
    x = np.sort(rng.uniform(0.0, 1.0, n))
    Y = np.stack([_gp_draw(rng, x, rho=0.2, noise=0.1)[0] for _ in range(V)])
    rho = 0.2

    res = gp_fit(jnp.asarray(Y), jnp.asarray(x), kernel='matern52',
                 engine='exact', rank=n, rho_bounds=(rho, rho), n_rho=1,
                 n_outer=120)
    sf2 = np.exp(np.asarray(res.theta[:, 0]))
    se2 = np.exp(np.asarray(res.theta[:, 1]))
    beta = np.asarray(res.coef[:, 0])

    r = np.abs(x[:, None] - x[None, :])
    Kg = gpmod._kernel_gram(r, 'matern52', rho) + 1e-8 * np.eye(n)
    L = np.linalg.cholesky(Kg)
    rr = reml_fit(jnp.asarray(Y), jnp.asarray(np.ones((n, 1))),
                  jnp.asarray(L), n_iter=60)
    sf2_r = np.asarray(rr.sigma_b_sq)
    se2_r = np.asarray(rr.sigma_e_sq)
    beta_r = np.asarray(rr.beta_hat[:, 0])

    assert np.allclose(sf2, sf2_r, rtol=2e-3), (sf2, sf2_r)
    assert np.allclose(se2, se2_r, rtol=2e-3), (se2, se2_r)
    assert np.allclose(beta, beta_r, atol=1e-4), (beta, beta_r)


def test_exact_matches_sklearn_gpr():
    """The exact engine (full-rank) recovers an exact ``GaussianProcessRegressor``
    -- an exact-vs-exact anchor, so lengthscale and predictive mean agree tightly."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import (
        ConstantKernel as C,
    )
    from sklearn.gaussian_process.kernels import (
        Matern,
        WhiteKernel,
    )

    rng = np.random.default_rng(4)
    n = 90
    x = np.sort(rng.uniform(0.0, 1.0, n))
    y, f = _gp_draw(rng, x, rho=0.17, amp=1.0, nu=2.5, noise=0.1)

    kern = (
        C(1.0, (1e-3, 1e3))
        * Matern(length_scale=0.2, nu=2.5, length_scale_bounds=(1e-2, 1e1))
        + WhiteKernel(0.05, (1e-5, 1e1))
    )
    gpr = GaussianProcessRegressor(
        kernel=kern, normalize_y=True, n_restarts_optimizer=3
    ).fit(x[:, None], y)
    ls = gpr.kernel_.k1.k2.length_scale

    res = gp_fit(jnp.asarray(y[None, :]), jnp.asarray(x), kernel='matern52',
                 engine='exact', rank=n, n_rho=40)
    rho_hat = float(np.exp(res.theta[0, 2]))
    assert res.engine == 'exact'
    assert res.rank == n
    # Exact vs exact: lengthscale within ~30% (REML vs sklearn ML).
    assert 0.7 * ls < rho_hat < 1.4 * ls, (rho_hat, ls)

    xg = np.linspace(0.03, 0.97, 70)
    mean_nitrix, _ = gp_predict(res, jnp.asarray(xg), x_train=jnp.asarray(x))
    mean_sklearn = gpr.predict(xg[:, None])
    mean_nitrix = np.asarray(mean_nitrix)[0]
    assert np.corrcoef(mean_nitrix, mean_sklearn)[0, 1] > 0.995
    assert np.sqrt(np.mean((mean_nitrix - mean_sklearn) ** 2)) < 0.1


@pytest.mark.parametrize('kernel', ['matern12', 'matern32', 'matern52', 'rbf'])
def test_exact_each_kernel_recovers_smooth(kernel):
    rng = np.random.default_rng(31)
    n = 100
    x = np.sort(rng.uniform(0.0, 1.0, n))
    truth = np.sin(2 * np.pi * x)
    y = truth + 0.1 * rng.standard_normal(n)
    res = gp_fit(jnp.asarray(y[None, :]), jnp.asarray(x), kernel=kernel,
                 engine='exact', rank=n, n_rho=24)
    mean, _ = gp_predict(res, jnp.asarray(x), x_train=jnp.asarray(x))
    assert np.corrcoef(np.asarray(mean)[0], truth)[0, 1] > 0.95


def test_exact_truncated_rank_is_nystrom():
    """``engine='exact'`` with ``rank < N`` is the eigen-truncated (KL/Nystrom)
    approximation; it still recovers a smooth and ``rank`` is recorded."""
    rng = np.random.default_rng(33)
    n = 120
    x = np.sort(rng.uniform(0.0, 1.0, n))
    y, f = _gp_draw(rng, x, rho=0.3, noise=0.08)
    res = gp_fit(jnp.asarray(y[None, :]), jnp.asarray(x), engine='exact',
                 rank=12, n_rho=20)
    assert res.rank == 12
    assert res.coef.shape == (1, 1 + 12)
    mean, _ = gp_predict(res, jnp.asarray(x), x_train=jnp.asarray(x))
    assert np.corrcoef(np.asarray(mean)[0], f)[0, 1] > 0.95


def test_exact_default_rank_is_full():
    """``rank=None`` with ``engine='exact'`` defaults to the full rank ``N``."""
    rng = np.random.default_rng(35)
    n = 40
    x = np.sort(rng.uniform(0.0, 1.0, n))
    y, _ = _gp_draw(rng, x, rho=0.25, noise=0.1)
    res = gp_fit(jnp.asarray(y[None, :]), jnp.asarray(x), engine='exact',
                 n_rho=16)
    assert res.rank == n


def test_exact_predict_requires_x_train():
    rng = np.random.default_rng(37)
    n = 50
    x = np.sort(rng.uniform(0.0, 1.0, n))
    y, _ = _gp_draw(rng, x, rho=0.2, noise=0.1)
    res = gp_fit(jnp.asarray(y[None, :]), jnp.asarray(x), engine='exact',
                 rank=n, n_rho=16)
    with pytest.raises(ValueError):
        gp_predict(res, jnp.asarray(x))  # exact needs x_train
    mean, std = gp_predict(res, jnp.asarray(x), x_train=jnp.asarray(x))
    assert mean.shape == (1, n) and std.shape == (1, n)


# ---------------------------------------------------------------------------
# 8b. log_mlik is the full REML; gp_aic / gp_bic model selection
# ---------------------------------------------------------------------------


def test_log_mlik_is_full_reml():
    """``GPResult.log_mlik`` is the *full* restricted log marginal likelihood
    (constant included) -- it equals a dense reference at the fitted ``(lam, rho)``
    to absolute precision, not just up to a constant."""
    rng = np.random.default_rng(0)
    n, rank = 70, 20
    x = np.sort(rng.uniform(0.0, 1.0, n))
    y, _ = _gp_draw(rng, x, rho=0.2, noise=0.1)
    res = gp_fit(jnp.asarray(y[None, :]), jnp.asarray(x), rank=rank, n_rho=24)

    sf2 = np.exp(float(res.theta[0, 0]))
    se2 = np.exp(float(res.theta[0, 1]))
    rho = np.exp(float(res.theta[0, 2]))
    lam = se2 / sf2
    Phi, sqrt_lambda, _ = _build_design(x, rank, res.boundary)
    T = np.ones((n, 1))
    m0 = 1
    s = np.asarray(
        spectral_density(jnp.asarray(sqrt_lambda), kernel='matern52',
                               rho=rho)
    )
    M = np.eye(n) + Phi @ (np.diag(s) / lam) @ Phi.T
    Minv = np.linalg.inv(M)
    A = T.T @ Minv @ T
    alpha = np.linalg.solve(A, T.T @ Minv @ y)
    r = y - T @ alpha
    rss = r @ Minv @ r
    _, ldM = np.linalg.slogdet(M)
    _, ldA = np.linalg.slogdet(A)
    sig2 = rss / (n - m0)
    lR_full = -0.5 * (
        (n - m0) * np.log(2 * np.pi) + (n - m0)
        + (n - m0) * np.log(sig2) + ldM + ldA
    )
    assert abs(float(res.log_mlik[0]) - lR_full) < 1e-5


def test_gp_aic_bic_select_kernel():
    """AIC/BIC from the REML marginal likelihood prefer the data-generating
    kernel; the helpers accept GP and HGP results."""
    rng = np.random.default_rng(1)
    n = 150
    x = np.sort(rng.uniform(0.0, 1.0, n))
    # Smooth (Matern-5/2) ground truth.
    y, _ = _gp_draw(rng, x, rho=0.25, noise=0.1, nu=2.5)
    r52 = gp_fit(jnp.asarray(y[None, :]), jnp.asarray(x), kernel='matern52',
                 rank=20, n_rho=24)
    r12 = gp_fit(jnp.asarray(y[None, :]), jnp.asarray(x), kernel='matern12',
                 rank=20, n_rho=24)
    aic52, aic12 = float(gp_aic(r52)[0]), float(gp_aic(r12)[0])
    assert np.isfinite(aic52) and np.isfinite(aic12)
    assert aic52 < aic12  # the smooth kernel wins on the smooth signal
    # BIC penalises complexity more than AIC for N > e^2.
    assert float(gp_bic(r52)[0]) > float(gp_aic(r52)[0])


def test_gp_aic_bic_accepts_hgp():
    from nitrix.stats import hgp_fit

    rng = np.random.default_rng(2)
    per, L = 16, 6
    t = np.linspace(0.0, 1.0, per)
    x = np.tile(t, L)
    group = np.repeat(np.arange(L), per)
    y = np.concatenate([np.sin(2 * np.pi * t) + 0.1 * rng.standard_normal(per)
                        for _ in range(L)])
    res = hgp_fit(jnp.asarray(y[None, :]), jnp.asarray(x), jnp.asarray(group),
                  rank=8, n_rho=14)
    assert np.isfinite(float(gp_aic(res)[0]))
    assert np.isfinite(float(gp_bic(res)[0]))


# ---------------------------------------------------------------------------
# 9. corr= -- GP smooth + structured within-group residual (PR3b)
# ---------------------------------------------------------------------------


def _ar1_series(rng, n, phi, sd):
    e = np.zeros(n)
    e[0] = rng.standard_normal() * sd / np.sqrt(1 - phi**2)
    for i in range(1, n):
        e[i] = phi * e[i - 1] + sd * rng.standard_normal()
    return e


def _block_ar1(group, rho_c):
    """Dense block-diagonal AR(1) correlation ``R_{ij} = rho_c^{|i-j|}`` within
    each group (unit-gap), independent across groups."""
    g = np.asarray(group)
    n = g.shape[0]
    R = np.zeros((n, n))
    for gi in np.unique(g):
        idx = np.where(g == gi)[0]
        d = np.abs(idx[:, None] - idx[None, :])  # within-group lag (unit gaps)
        R[np.ix_(idx, idx)] = rho_c ** (np.abs(
            np.arange(len(idx))[:, None] - np.arange(len(idx))[None, :]
        ))
        # overwrite cross terms to 0 already; diagonal handled by power 0 = 1
        _ = d
    return R


def _dense_corr_reml_m2l(y, T, Phi, s, lam, R):
    """Dense profiled ``-2 l_R`` for ``y = T alpha + Phi gamma + e`` with
    ``gamma ~ N(0, (sigma^2/lam) diag(s))`` and ``e ~ N(0, sigma^2 R)`` -- the
    reference for the whitened corr REML (cf. ``_dense_reml_m2l`` with R=I)."""
    n = y.shape[0]
    m0 = T.shape[1]
    M = R + Phi @ (np.diag(s) / lam) @ Phi.T
    Minv = np.linalg.inv(M)
    A = T.T @ Minv @ T
    alpha = np.linalg.solve(A, T.T @ Minv @ y)
    r = y - T @ alpha
    rss_m = r @ Minv @ r
    _, logdet_m = np.linalg.slogdet(M)
    _, logdet_a = np.linalg.slogdet(A)
    return (n - m0) * np.log(rss_m) + logdet_m + logdet_a


def test_corr_reml_matches_dense_up_to_constant():
    """The whitened corr REML (p-space criterion + whitening Jacobian) equals the
    dense block-``R`` marginal-likelihood REML up to an additive constant -- across
    the lengthscale's ``lambda`` and the residual correlation ``rho_c``."""
    from nitrix.stats.lme._corr import ar1
    from nitrix.stats.lme._corrfit import build_group_layout

    rng = np.random.default_rng(0)
    G, Tn, rank = 6, 10, 12
    t = np.linspace(0.0, 1.0, Tn)
    x = np.tile(t, G)
    group = np.repeat(np.arange(G), Tn)
    n = G * Tn
    y = np.sin(2 * np.pi * x) + 0.2 * rng.standard_normal(n)

    Phi, sqrt_lambda, _ = _build_design(x, rank, 1.5)
    T = np.ones((n, 1))
    X = np.concatenate([T, Phi], axis=1)
    m0, p, m = 1, X.shape[1], rank

    spec = ar1()
    layout = build_group_layout(jnp.asarray(group), None)
    idx, mask = layout.idx, layout.mask
    mask_f = mask.astype(jnp.float64)
    x_pad = jnp.asarray(X)[idx] * mask_f[..., None]
    y_pad = (jnp.asarray(y)[idx] * mask_f)[..., None]  # (G, T, 1)

    diffs = []
    for rho_c in (0.0, 0.3, 0.6, -0.4):
        raw = jnp.asarray([np.arctanh(rho_c)])
        xt, half_logdet = spec.whiten(x_pad, layout.gaps, layout.nsize, mask, raw)
        yt, _ = spec.whiten(y_pad, layout.gaps, layout.nsize, mask, raw)
        xtx = jnp.einsum('gtp,gtq->pq', xt, xt)
        c = jnp.einsum('gtp,gtk->pk', xt, yt)[:, 0]
        g = jnp.einsum('gtk,gtk->', yt, yt)
        R = _block_ar1(group, rho_c)
        s = np.asarray(
            spectral_density(jnp.asarray(sqrt_lambda), kernel='matern52',
                                   rho=0.2)
        )
        d, log_pdet_pen = gpmod._penalty_diag(
            jnp.asarray(sqrt_lambda), 'matern52', jnp.asarray(0.2), m0
        )
        for lam in (0.5, 5.0, 50.0):
            _, logdet_h, _, _, _, d_p = gpmod._quantities(
                jnp.asarray(lam), c, g, xtx, d, p, 0.0
            )
            pspace = float(
                gpmod._reml_nll(d_p, logdet_h, jnp.asarray(lam),
                                log_pdet_pen, n, m, m0)
            )
            pspace_corr = pspace + 2.0 * float(half_logdet)  # + log|R|
            dense = _dense_corr_reml_m2l(y, T, Phi, s, lam, R)
            diffs.append(pspace_corr - dense)

    diffs = np.asarray(diffs)
    assert np.ptp(diffs) < 1e-6, (
        f'corr REML p-space vs dense not constant-offset: {np.ptp(diffs):.2e}'
    )


def test_corr_iid_matches_no_corr():
    """``corr='iid'`` (identity whitening) reproduces the plain ``corr=None`` fit."""
    rng = np.random.default_rng(2)
    n = 80
    x = np.sort(rng.uniform(0.0, 1.0, n))
    group = (x > 0.5).astype(np.int64)  # arbitrary grouping; iid ignores it
    y, _ = _gp_draw(rng, x, rho=0.25, noise=0.1)
    base = gp_fit(jnp.asarray(y[None, :]), jnp.asarray(x), rank=15, n_rho=20)
    iid = gp_fit(jnp.asarray(y[None, :]), jnp.asarray(x), rank=15, n_rho=20,
                 corr='iid', group=jnp.asarray(group), n_corr=3)
    assert iid.corr == 'iid'
    assert np.allclose(np.asarray(base.coef), np.asarray(iid.coef), atol=1e-6)
    assert np.allclose(np.asarray(base.theta), np.asarray(iid.theta), atol=1e-6)


def test_corr_ar1_recovers_and_beats_iid():
    """On AR(1) longitudinal data, ``corr='ar1'`` recovers a positive ``rho_c`` and
    has a higher marginal likelihood than the i.i.d. fit."""
    rng = np.random.default_rng(5)
    G, Tn = 30, 10
    t = np.linspace(0.0, 1.0, Tn)
    x = np.tile(t, G)
    group = np.repeat(np.arange(G), Tn)
    trend = np.sin(2 * np.pi * t)
    phi_true = 0.6
    y = np.concatenate([trend + _ar1_series(rng, Tn, phi_true, 0.3)
                        for _ in range(G)])

    fit_ar1 = gp_fit(jnp.asarray(y[None, :]), jnp.asarray(x), rank=15,
                     corr='ar1', group=jnp.asarray(group), n_rho=20, n_corr=15)
    fit_iid = gp_fit(jnp.asarray(y[None, :]), jnp.asarray(x), rank=15, n_rho=20)
    rho_c = float(fit_ar1.corr_rho[0])
    assert 0.35 < rho_c < 0.85, rho_c
    assert float(fit_ar1.log_mlik[0]) > float(fit_iid.log_mlik[0])


def test_corr_requires_group():
    rng = np.random.default_rng(7)
    x = np.sort(rng.uniform(0.0, 1.0, 40))
    y = jnp.asarray(rng.standard_normal((1, 40)))
    with pytest.raises(ValueError):
        gp_fit(y, jnp.asarray(x), corr='ar1')  # group missing


@pytest.mark.parametrize('corr', ['ar1', 'cs', 'car1'])
def test_corr_structures_run_each_engine(corr):
    """Each correlation structure composes with both engines and recovers the
    shared trend."""
    rng = np.random.default_rng(9)
    G, Tn = 14, 9
    t = np.linspace(0.0, 1.0, Tn)
    x = np.tile(t, G)
    group = np.repeat(np.arange(G), Tn)
    time = np.tile(t, G)  # for car1
    trend = np.sin(2 * np.pi * t)
    y = np.concatenate([trend + _ar1_series(rng, Tn, 0.4, 0.25)
                        for _ in range(G)])
    res = gp_fit(jnp.asarray(y[None, :]), jnp.asarray(x), engine='hsgp',
                 rank=12, corr=corr, group=jnp.asarray(group),
                 time=jnp.asarray(time), n_rho=16, n_corr=9)
    assert res.corr == corr
    mean, _ = gp_predict(res, jnp.asarray(t))
    assert np.corrcoef(np.asarray(mean)[0], trend)[0, 1] > 0.9


def test_corr_with_exact_engine():
    """corr= composes with the exact engine too."""
    rng = np.random.default_rng(11)
    G, Tn = 16, 8
    t = np.linspace(0.0, 1.0, Tn)
    x = np.tile(t, G)
    group = np.repeat(np.arange(G), Tn)
    trend = np.cos(2 * np.pi * t)
    y = np.concatenate([trend + _ar1_series(rng, Tn, 0.5, 0.25)
                        for _ in range(G)])
    res = gp_fit(jnp.asarray(y[None, :]), jnp.asarray(x), engine='exact',
                 rank=Tn, corr='ar1', group=jnp.asarray(group),
                 n_rho=14, n_corr=11)
    assert res.engine == 'exact' and res.corr == 'ar1'
    assert float(res.corr_rho[0]) > 0.2
    mean, _ = gp_predict(res, jnp.asarray(t), x_train=jnp.asarray(x))
    assert np.corrcoef(np.asarray(mean)[0], trend)[0, 1] > 0.9


@pytest.mark.parametrize('case', ['hsgp', 'exact', 'nd', 'corr'])
def test_block_bounds_rho_search_without_changing_results(case):
    """PF1: `block=` now chunks the pooled-NLL rho search (not just the final
    fit), so it bounds peak memory.  The fitted function (predictive mean) and
    the variance components are invariant to the chunk size.  Chunked summation
    reorders the floating-point adds by ~1e-15; the well-conditioned engines are
    unaffected, while the full-rank ``exact`` engine's near-null eigen-direction
    coefficients amplify it -- the *function* is unchanged, so we pin the
    predictive mean, not the raw coefficients."""
    rng = np.random.default_rng(21)
    v, nn = 8, 60
    if case == 'nd':
        x2 = rng.uniform(0.0, 1.0, (nn, 2))
        y = (np.sin(2 * np.pi * x2[:, 0])[None, :]
             + 0.1 * rng.standard_normal((v, nn)))
        xa = jnp.asarray(x2)
        kw = dict(n_rho=10)

        def pred(r):
            return gp_predict(r, xa)[0]
    elif case == 'corr':
        g = np.repeat(np.arange(6), 10)
        x = np.tile(np.linspace(0.0, 1.0, 10), 6)
        y = np.sin(2 * np.pi * x)[None, :] + 0.1 * rng.standard_normal((v, 60))
        xa = jnp.asarray(x)
        kw = dict(corr='ar1', group=jnp.asarray(g), n_rho=8, n_corr=6)
        xg = jnp.asarray(np.linspace(0.05, 0.95, 20))

        def pred(r):
            return gp_predict(r, xg)[0]
    else:  # hsgp / exact
        x = np.sort(rng.uniform(0.0, 1.0, nn))
        y = np.sin(2 * np.pi * x)[None, :] + 0.1 * rng.standard_normal((v, nn))
        xa = jnp.asarray(x)
        if case == 'exact':
            kw = dict(engine='exact', rank=nn, n_rho=10)

            def pred(r):
                return gp_predict(r, xa, x_train=xa)[0]
        else:
            kw = dict(rank=24, n_rho=10)

            def pred(r):
                return gp_predict(r, xa)[0]
    a = gp_fit(jnp.asarray(y), xa, block=None, **kw)
    b = gp_fit(jnp.asarray(y), xa, block=3, **kw)
    np.testing.assert_allclose(np.asarray(pred(a)), np.asarray(pred(b)), atol=1e-7)
    np.testing.assert_allclose(np.asarray(a.theta), np.asarray(b.theta), atol=1e-6)


# ---------------------------------------------------------------------------
# CV2 -- non-Gaussian GP lengthscale estimation (PQL-REML, Phase 1)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('family', ['poisson', 'binomial'])
def test_gp_glm_recovers_smooth(family):
    """CV2 Phase 1: a Binomial / Poisson GP recovers the latent smooth and a
    finite lengthscale by PQL-REML; gp_predict gives a valid response-scale mean."""
    rng = np.random.default_rng(1)
    n, V = 200, 6
    x = np.sort(rng.uniform(0.0, 1.0, n))
    truth = np.sin(2 * np.pi * x)
    grid = np.linspace(0.02, 0.98, 100)
    tgrid = np.sin(2 * np.pi * grid)
    tgrid = tgrid - tgrid.mean()
    if family == 'poisson':
        Y = rng.poisson(np.exp(0.8 + truth)[None, :] * np.ones((V, 1))).astype(float)
        res = gp_fit(jnp.asarray(Y), jnp.asarray(x), family='poisson',
                     n_rho=16, n_pql=8)
        pw = None
    else:
        ntr = 20
        pmu = 1.0 / (1.0 + np.exp(-1.5 * truth))
        Y = rng.binomial(ntr, pmu[None, :] * np.ones((V, 1))).astype(float) / ntr
        pw = jnp.full((n,), float(ntr))
        res = gp_fit(jnp.asarray(Y), jnp.asarray(x), family='binomial',
                     prior_weights=pw, n_rho=16, n_pql=8)
    assert res.family == family
    assert 0.0 < float(np.exp(res.theta[0, 2])) < 2.0
    assert np.all(np.isfinite(np.asarray(res.coef)))
    # latent (link-scale) fit tracks the smooth shape
    eta, eta_sd = gp_predict(res, jnp.asarray(grid))
    e = np.asarray(eta[0]) - float(np.mean(eta[0]))
    assert np.corrcoef(e, tgrid)[0, 1] > 0.9
    assert np.all(np.asarray(eta_sd) > 0)
    # response-scale mean is in the family's range
    mu, mu_sd = gp_predict(res, jnp.asarray(grid), type='response')
    if family == 'poisson':
        assert np.all(np.asarray(mu) > 0)
    else:
        assert np.all((np.asarray(mu) > 0) & (np.asarray(mu) < 1))
    assert np.all(np.asarray(mu_sd) > 0)


def test_gp_glm_rho_tracks_lengthscale():
    """CV2: the estimated rho is larger for a smoother (longer-scale) latent
    field -- the lengthscale is genuinely estimated, not pinned."""
    rng = np.random.default_rng(3)
    n, V = 220, 6
    x = np.sort(rng.uniform(0.0, 1.0, n))

    def fit_rho(freq):
        eta = 0.8 + np.sin(freq * np.pi * x)
        Y = rng.poisson(np.exp(eta)[None, :] * np.ones((V, 1))).astype(float)
        r = gp_fit(jnp.asarray(Y), jnp.asarray(x), family='poisson',
                   n_rho=16, n_pql=8)
        return float(np.exp(r.theta[0, 2]))

    rho_short, rho_long = fit_rho(2.0), fit_rho(0.7)
    assert rho_short < rho_long


def test_gp_gaussian_family_is_byte_identical():
    """No marquee regression: family='gaussian' takes the existing exact path,
    bit-for-bit, and gp_predict type='response' is a no-op for it."""
    rng = np.random.default_rng(4)
    n, V = 120, 5
    x = np.sort(rng.uniform(0.0, 1.0, n))
    Y = jnp.asarray(np.sin(2 * np.pi * x)[None, :] + 0.2 * rng.standard_normal((V, n)))
    a = gp_fit(Y, jnp.asarray(x), n_rho=14)
    b = gp_fit(Y, jnp.asarray(x), n_rho=14, family='gaussian')
    assert float(jnp.max(jnp.abs(a.coef - b.coef))) == 0.0
    assert float(jnp.max(jnp.abs(a.theta - b.theta))) == 0.0
    assert b.family == 'gaussian'
    grid = jnp.linspace(0.05, 0.95, 40)
    ml, sl = gp_predict(b, grid)
    mr, sr = gp_predict(b, grid, type='response')
    assert float(jnp.max(jnp.abs(ml - mr))) == 0.0
    assert float(jnp.max(jnp.abs(sl - sr))) == 0.0


def test_gp_glm_phase1_guards():
    """CV2 Phase 1 scope: non-Gaussian needs engine='hsgp', 1-D, no corr; an
    unsupported family and a bad predict type raise clearly."""
    rng = np.random.default_rng(5)
    n = 60
    x = np.sort(rng.uniform(0.0, 1.0, n))
    Y = jnp.asarray(rng.poisson(2.0, (3, n)).astype(float))
    g = jnp.asarray(np.repeat(np.arange(6), 10))
    with pytest.raises(NotImplementedError, match='engine'):
        gp_fit(Y, jnp.asarray(x), family='poisson', engine='exact')
    with pytest.raises(NotImplementedError, match='corr'):
        gp_fit(Y, jnp.asarray(x), family='poisson', corr='ar1', group=g)
    with pytest.raises(NotImplementedError, match='1-D'):
        gp_fit(Y, jnp.asarray(rng.uniform(0, 1, (n, 2))), family='poisson')
    with pytest.raises(NotImplementedError, match='binomial'):
        gp_fit(Y, jnp.asarray(x), family='gamma')
    res = gp_fit(Y, jnp.asarray(x), family='poisson', n_rho=8, n_pql=4)
    with pytest.raises(ValueError, match='type='):
        gp_predict(res, jnp.asarray(x), type='quantile')


def test_gp_corr_ar1_without_time_warns():
    """MC6: ar1/car1 with time=None assumes within-group time order, so it warns;
    cs (order-invariant) and an explicit time= are silent."""
    import warnings

    rng = np.random.default_rng(8)
    g = np.repeat(np.arange(6), 10)
    x = np.tile(np.linspace(0.0, 1.0, 10), 6)
    Y = jnp.asarray(np.sin(2 * np.pi * x)[None, :] + 0.1 * rng.standard_normal((3, 60)))

    def msgs(**kw):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            gp_fit(Y, jnp.asarray(x), group=jnp.asarray(g),
                   n_rho=6, n_corr=6, **kw)
            return [str(r.message) for r in w if 'time order' in str(r.message)]

    assert msgs(corr='ar1')
    assert not msgs(corr='cs')
    assert not msgs(corr='ar1', time=jnp.asarray(np.tile(np.arange(10.0), 6)))


def test_gp_corr_bounds_edge_clamp_warns():
    """MC5: the corr parameter is grid-quantised over corr_raw_bounds; the wider
    default (-4, 4) contains a moderate correlation (silent), but an estimate
    pinned at the search edge warns."""
    import warnings

    rng = np.random.default_rng(3)
    G, T = 16, 12
    t = np.linspace(0.0, 1.0, T)
    x = np.tile(t, G)
    g = np.repeat(np.arange(G), T)

    def _ar1(n, rho, sd):
        e = rng.standard_normal(n) * sd
        z = np.zeros(n)
        z[0] = e[0]
        for i in range(1, n):
            z[i] = rho * z[i - 1] + e[i]
        return z

    y = np.concatenate([np.cos(2 * np.pi * t) + _ar1(T, 0.7, 0.25)
                        for _ in range(G)])
    Y = jnp.asarray(y[None, :])
    tt = jnp.asarray(np.tile(np.arange(T * 1.0), G))

    def edge(**kw):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            gp_fit(Y, jnp.asarray(x), corr='ar1', group=jnp.asarray(g),
                   time=tt, n_rho=8, n_corr=9, **kw)
            return any('edge of the search grid' in str(m.message) for m in w)

    assert not edge()  # default (-4, 4) contains a ~0.7 correlation
    assert edge(corr_raw_bounds=(-2.0, 0.0))  # excludes the optimum -> clamps
