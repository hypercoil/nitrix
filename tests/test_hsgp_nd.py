# -*- coding: utf-8 -*-
"""Tests for the multi-dimensional Hilbert-space GP (``hsgp_basis_nd``) and the
``dim``-generalised spectral densities.

Anchors:

1. **D-dim spectral density** -- reduces to the 1-D closed forms at ``dim=1`` and
   matches the analytic ``D``-dimensional Fourier transform of the kernel.
2. **2-D GP recovery** -- a tensor-product HSGP smooth in ``gam_fit`` recovers a
   smooth surface and tracks an exact 2-D ``GaussianProcessRegressor``.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

jax.config.update('jax_enable_x64', True)

import jax.numpy as jnp

from nitrix.linalg.kernel import (
    matern_spectral_density,
    se_spectral_density,
    spectral_density,
)
from nitrix.stats import (
    gam_fit,
    gp_fit,
    gp_predict,
    hsgp_basis_nd,
    smooth_partial_effect,
)

# ---------------------------------------------------------------------------
# 1. dim-generalised spectral densities
# ---------------------------------------------------------------------------


def test_spectral_density_dim1_matches_closed_forms():
    """The ``dim=1`` path equals the 1-D Matern / SE closed forms (and the general
    ``D``-dim Matern normaliser reproduces them too)."""
    w = jnp.linspace(0.0, 6.0, 50)
    rho = 0.4
    for nu, lam in [(0.5, 1.0 / rho), (1.5, np.sqrt(3) / rho),
                    (2.5, np.sqrt(5) / rho)]:
        got = np.asarray(matern_spectral_density(w, rho=rho, nu=nu, dim=1))
        w2 = np.asarray(w) ** 2
        if nu == 0.5:
            ref = 2.0 * lam / (lam**2 + w2)
        elif nu == 1.5:
            ref = 4.0 * lam**3 / (lam**2 + w2) ** 2
        else:
            ref = (16.0 / 3.0) * lam**5 / (lam**2 + w2) ** 3
        assert np.allclose(got, ref, rtol=1e-10)
    se = np.asarray(se_spectral_density(w, rho=rho, dim=1))
    se_ref = np.sqrt(2 * np.pi) * rho * np.exp(-0.5 * (rho * np.asarray(w)) ** 2)
    assert np.allclose(se, se_ref, rtol=1e-10)


def test_dim_general_matern_continuous_at_dim1():
    """The general gammaln-based ``D``-dim Matern, evaluated at ``D=1``
    analytically, agrees with the closed-form branch (the normaliser is right)."""
    from jax.scipy.special import gammaln

    w = jnp.linspace(0.2, 5.0, 30)
    rho, nu, D = 0.3, 2.5, 1
    closed = np.asarray(matern_spectral_density(w, rho=rho, nu=nu, dim=1))
    # Reconstruct the general form by hand at D=1.
    half = nu + D / 2.0
    lam2 = 2 * nu / rho**2
    log_c = (D * np.log(2) + (D / 2) * np.log(np.pi)
             + float(gammaln(half)) + nu * np.log(2 * nu)
             - float(gammaln(nu)) - 2 * nu * np.log(rho))
    general = np.exp(log_c) * (lam2 + np.asarray(w) ** 2) ** (-half)
    assert np.allclose(closed, general, rtol=1e-9)


def test_spectral_density_dim_increases_normaliser():
    """The 2-D isotropic density differs from 1-D (the ``D``-dependent normaliser
    and exponent), and stays positive and decreasing in ``||w||``."""
    w = jnp.linspace(0.1, 5.0, 40)
    s1 = np.asarray(spectral_density(w, kernel='matern52', rho=0.3, dim=1))
    s2 = np.asarray(spectral_density(w, kernel='matern52', rho=0.3, dim=2))
    assert np.all(s2 > 0)
    assert np.all(np.diff(s2) < 0)  # monotone decreasing
    assert not np.allclose(s1, s2)


# ---------------------------------------------------------------------------
# 2. 2-D tensor-product HSGP basis
# ---------------------------------------------------------------------------


def _surface(X):
    return np.sin(2 * np.pi * X[:, 0]) * np.cos(2 * np.pi * X[:, 1])


def _fit_surface(b, X, y):
    res = gam_fit(jnp.asarray(y[None, :]), [b])
    eff, _ = smooth_partial_effect(res, 0, b, jnp.asarray(X))
    return (np.asarray(res.coef[:, 0:1]) + np.asarray(eff))[0]


def test_hsgp_nd_contract_and_eval_roundtrip():
    rng = np.random.default_rng(0)
    X = rng.uniform(0.0, 1.0, (200, 2))
    b = hsgp_basis_nd(jnp.asarray(X), 7, kernel='matern52', rho=0.25)
    assert b.n_dim == 2
    # 7*7 modes minus the sum-to-zero constraint column.
    assert b.dim == 7 * 7 - 1
    assert b.design.shape == (200, 7 * 7 - 1)
    blocks = b.penalty_blocks()
    assert len(blocks) == 1  # a single GP amplitude
    ed = b.eval_design(jnp.asarray(X))
    assert np.allclose(np.asarray(ed), np.asarray(b.design), atol=1e-9)


def test_hsgp_nd_recovers_2d_surface_isotropic_and_ard():
    rng = np.random.default_rng(1)
    X = rng.uniform(0.0, 1.0, (500, 2))
    truth = _surface(X)
    y = truth + 0.1 * rng.standard_normal(len(X))
    b_iso = hsgp_basis_nd(jnp.asarray(X), 8, kernel='matern52', rho=0.2)
    fit_iso = _fit_surface(b_iso, X, y)
    assert np.corrcoef(fit_iso, truth)[0, 1] > 0.98
    # Separable / ARD with per-dimension lengthscales.
    b_ard = hsgp_basis_nd(jnp.asarray(X), [8, 8], kernel='matern52',
                          rho=[0.2, 0.25])
    fit_ard = _fit_surface(b_ard, X, y)
    assert np.corrcoef(fit_ard, truth)[0, 1] > 0.98


def test_hsgp_nd_matches_sklearn_2d_gpr():
    """The 2-D tensor HSGP tracks an exact 2-D ``GaussianProcessRegressor``."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import (
        ConstantKernel as C,
    )
    from sklearn.gaussian_process.kernels import (
        Matern,
        WhiteKernel,
    )

    rng = np.random.default_rng(2)
    X = rng.uniform(0.0, 1.0, (300, 2))
    truth = _surface(X)
    y = truth + 0.1 * rng.standard_normal(len(X))

    kern = (
        C(1.0, (1e-3, 1e3))
        * Matern(length_scale=0.3, nu=2.5, length_scale_bounds=(5e-2, 2.0))
        + WhiteKernel(0.05, (1e-5, 1e1))
    )
    gpr = GaussianProcessRegressor(
        kernel=kern, normalize_y=True, n_restarts_optimizer=2
    ).fit(X, y)
    ls = gpr.kernel_.k1.k2.length_scale

    b = hsgp_basis_nd(jnp.asarray(X), 10, kernel='matern52', rho=float(ls))
    res = gam_fit(jnp.asarray(y[None, :]), [b])

    Xg = rng.uniform(0.05, 0.95, (120, 2))
    eff, _ = smooth_partial_effect(res, 0, b, jnp.asarray(Xg))
    mean_nitrix = (np.asarray(res.coef[:, 0:1]) + np.asarray(eff))[0]
    mean_sklearn = gpr.predict(Xg)
    assert np.corrcoef(mean_nitrix, mean_sklearn)[0, 1] > 0.97
    # both recover the latent surface
    assert np.corrcoef(mean_nitrix, _surface(Xg))[0, 1] > 0.95


def test_hsgp_nd_mass_univariate_and_3d():
    """Mass-univariate over voxels, and a 3-D smoke test."""
    rng = np.random.default_rng(3)
    X = rng.uniform(0.0, 1.0, (300, 2))
    truth = _surface(X)
    Y = np.stack([truth + 0.1 * rng.standard_normal(len(X)) for _ in range(4)])
    b = hsgp_basis_nd(jnp.asarray(X), 7, kernel='matern52', rho=0.25)
    res = gam_fit(jnp.asarray(Y), [b])
    assert res.coef.shape[0] == 4
    eff, _ = smooth_partial_effect(res, 0, b, jnp.asarray(X))
    fitted = np.asarray(res.coef[:, 0:1]) + np.asarray(eff)
    cors = [np.corrcoef(fitted[v], truth)[0, 1] for v in range(4)]
    assert np.min(cors) > 0.97

    X3 = rng.uniform(0.0, 1.0, (200, 3))
    b3 = hsgp_basis_nd(jnp.asarray(X3), 4, kernel='matern32', rho=0.3)
    assert b3.n_dim == 3
    assert b3.dim == 4 ** 3 - 1  # 64 modes minus constraint


def test_hsgp_nd_argument_validation():
    rng = np.random.default_rng(4)
    X = rng.uniform(0.0, 1.0, (50, 2))
    with pytest.raises(ValueError):
        hsgp_basis_nd(jnp.asarray(X[:, 0]))  # 1-D X (not (n, D))
    with pytest.raises(ValueError):
        hsgp_basis_nd(jnp.asarray(X), boundary=0.5)
    with pytest.raises(ValueError):
        hsgp_basis_nd(jnp.asarray(X), [8, 8, 8])  # wrong length for D=2
    with pytest.raises(ValueError):
        hsgp_basis_nd(jnp.asarray(X), 8, rho=[0.2, 0.3, 0.4])  # wrong rho length


# ---------------------------------------------------------------------------
# 3. Multi-D lengthscale estimation in gp_fit (isotropic + ARD)
# ---------------------------------------------------------------------------


def test_gp_fit_2d_isotropic_estimates_rho():
    """``gp_fit`` on ``(N, 2)`` X estimates a shared (isotropic) lengthscale and
    predicts the surface; the result is self-contained (no x_train)."""
    rng = np.random.default_rng(0)
    X = rng.uniform(0.0, 1.0, (500, 2))
    truth = _surface(X)
    y = truth + 0.1 * rng.standard_normal(len(X))
    res = gp_fit(jnp.asarray(y[None, :]), jnp.asarray(X), kernel='matern52',
                 rank=8, n_rho=20)
    assert res.engine == 'hsgp'
    assert res.nd_meta is not None
    rho_hat = float(np.exp(res.theta[0, 2]))
    assert 0.05 < rho_hat < 2.0
    mean, std = gp_predict(res, jnp.asarray(X))  # no x_train needed
    assert mean.shape == (1, len(X))
    assert np.all(np.asarray(std) > 0)
    assert np.corrcoef(np.asarray(mean)[0], truth)[0, 1] > 0.98


def test_gp_fit_2d_ard_recovers_anisotropy():
    """ARD recovers a longer lengthscale on the smooth axis than the wiggly one."""
    rng = np.random.default_rng(1)
    X = rng.uniform(0.0, 1.0, (600, 2))
    # smooth in axis 0, wiggly in axis 1
    truth = np.sin(2 * np.pi * X[:, 0]) * np.sin(6 * np.pi * X[:, 1])
    y = truth + 0.1 * rng.standard_normal(len(X))
    res = gp_fit(jnp.asarray(y[None, :]), jnp.asarray(X), kernel='matern52',
                 rank=10, ard=True, n_rho=16)
    rho_axes = res.nd_meta[2]
    assert rho_axes is not None
    assert rho_axes[0] > rho_axes[1]  # smooth axis has the longer lengthscale
    mean, _ = gp_predict(res, jnp.asarray(X))
    assert np.corrcoef(np.asarray(mean)[0], truth)[0, 1] > 0.97


def test_gp_fit_2d_matches_sklearn():
    """Estimated-rho 2-D gp_fit tracks an exact 2-D GaussianProcessRegressor."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import (
        ConstantKernel as C,
    )
    from sklearn.gaussian_process.kernels import (
        Matern,
        WhiteKernel,
    )

    rng = np.random.default_rng(2)
    X = rng.uniform(0.0, 1.0, (300, 2))
    truth = _surface(X)
    y = truth + 0.1 * rng.standard_normal(len(X))
    gpr = GaussianProcessRegressor(
        kernel=C(1.0) * Matern(length_scale=0.3, nu=2.5,
                               length_scale_bounds=(5e-2, 2.0))
        + WhiteKernel(0.05), normalize_y=True, n_restarts_optimizer=2,
    ).fit(X, y)
    res = gp_fit(jnp.asarray(y[None, :]), jnp.asarray(X), kernel='matern52',
                 rank=9, n_rho=24)
    Xg = rng.uniform(0.05, 0.95, (120, 2))
    mean, _ = gp_predict(res, jnp.asarray(Xg))
    assert np.corrcoef(np.asarray(mean)[0], gpr.predict(Xg))[0, 1] > 0.97


def test_gp_fit_nd_per_axis_rank_and_validation():
    rng = np.random.default_rng(3)
    X = rng.uniform(0.0, 1.0, (200, 2))
    y = _surface(X) + 0.1 * rng.standard_normal(len(X))
    res = gp_fit(jnp.asarray(y[None, :]), jnp.asarray(X), rank=[6, 8], n_rho=14)
    assert res.coef.shape == (1, 1 + 6 * 8)
    with pytest.raises(NotImplementedError):
        gp_fit(jnp.asarray(y[None, :]), jnp.asarray(X), engine='exact')
    with pytest.raises(NotImplementedError):
        gp_fit(jnp.asarray(y[None, :]), jnp.asarray(X), corr='ar1',
               group=jnp.zeros(len(X), int))
