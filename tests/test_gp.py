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

from nitrix.stats import gp as gpmod
from nitrix.stats.gp import GPResult, gp_fit, gp_predict

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
                gpmod.spectral_density(
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
        gp_fit(y, xj, engine='exact')  # type: ignore[arg-type]
    with pytest.raises(NotImplementedError):
        gp_fit(y, xj, select='per-voxel')  # type: ignore[arg-type]
    with pytest.raises(NotImplementedError):
        gp_fit(y, xj, corr='ar1')
    with pytest.raises(ValueError):
        gp_fit(y, xj, rank=0)
    with pytest.raises(ValueError):
        gp_fit(y, xj, boundary=0.5)
    with pytest.raises(ValueError):
        gp_fit(y, jnp.asarray(x[:10]))  # x/Y length mismatch
