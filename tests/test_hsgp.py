# -*- coding: utf-8 -*-
"""Tests for the Hilbert-space approximate GP (HSGP) smooth and the stationary-
kernel spectral densities.

The spectral densities are anchored two ways: against their analytic closed
forms, and against the scikit-learn ``Matern`` / ``RBF`` kernels via an
inverse-Fourier round-trip (so the lengthscale/amplitude parameterisation is
provably the reference one).  The basis is anchored against an *exact* dense GP
posterior (the unambiguous reference): the HSGP posterior mean converges to it as
the rank grows, and matches scikit-learn's GaussianProcessRegressor.  The
``gam_fit`` integration checks the basis is a drop-in penalised smooth.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update('jax_enable_x64', True)

from nitrix.linalg.kernel import (
    matern_spectral_density,
    se_spectral_density,
    spectral_density,
)
from nitrix.stats.basis import hsgp_basis, spline_design
from nitrix.stats.gam import gam_fit, smooth_partial_effect

# ---------------------------------------------------------------------------
# Spectral densities
# ---------------------------------------------------------------------------


def test_se_spectral_density_matches_analytic():
    w = jnp.array([0.0, 0.5, 1.0, 3.0, 7.0])
    rho, amp = 0.3, 1.7
    got = np.asarray(se_spectral_density(w, rho=rho, amplitude=amp))
    ref = amp**2 * np.sqrt(2 * np.pi) * rho * np.exp(-0.5 * (rho * np.asarray(w)) ** 2)
    np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize(
    'nu,rate', [(0.5, 1.0), (1.5, np.sqrt(3.0)), (2.5, np.sqrt(5.0))]
)
def test_matern_spectral_density_matches_analytic(nu, rate):
    w = jnp.array([0.0, 0.5, 1.0, 3.0, 7.0])
    rho, amp = 0.4, 1.3
    lam = rate / rho
    w2 = np.asarray(w) ** 2
    coef = {0.5: 2.0, 1.5: 4.0, 2.5: 16.0 / 3.0}[nu]
    power = {0.5: 1, 1.5: 2, 2.5: 3}[nu]
    ref = amp**2 * coef * lam ** (2 * power - 1) / (lam**2 + w2) ** power
    got = np.asarray(matern_spectral_density(w, rho=rho, nu=nu, amplitude=amp))
    np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-14)


def test_spectral_density_dispatch_aliases():
    w = jnp.array([0.0, 1.0, 2.0])
    np.testing.assert_allclose(
        np.asarray(spectral_density(w, kernel='rbf', rho=0.3)),
        np.asarray(se_spectral_density(w, rho=0.3)),
    )
    np.testing.assert_allclose(
        np.asarray(spectral_density(w, kernel='exp', rho=0.3)),
        np.asarray(matern_spectral_density(w, rho=0.3, nu=0.5)),
    )
    with pytest.raises(ValueError, match='unknown kernel'):
        spectral_density(w, kernel='cauchy', rho=0.3)
    with pytest.raises(ValueError, match='unsupported'):
        matern_spectral_density(w, rho=0.3, nu=2.0)


def test_spectral_density_integer_omega_not_corrupted():
    """MC2: an integer ``omega`` array must not coerce ``rho``/``amplitude`` to
    int (which zeroed SE and NaN'd Matern); results must match the float omega
    and a float32 omega must stay float32 (the x32 path is preserved)."""
    w_int = jnp.array([0, 1, 2, 3])
    w_flt = jnp.array([0.0, 1.0, 2.0, 3.0])
    se_i = np.asarray(se_spectral_density(w_int, rho=0.3, amplitude=1.7))
    se_f = np.asarray(se_spectral_density(w_flt, rho=0.3, amplitude=1.7))
    np.testing.assert_allclose(se_i, se_f, rtol=1e-12)
    assert np.all(np.isfinite(se_i)) and np.any(se_i != 0.0)
    mt_i = np.asarray(matern_spectral_density(w_int, rho=0.4, nu=2.5, amplitude=1.3))
    mt_f = np.asarray(matern_spectral_density(w_flt, rho=0.4, nu=2.5, amplitude=1.3))
    np.testing.assert_allclose(mt_i, mt_f, rtol=1e-12)
    assert np.all(np.isfinite(mt_i))
    assert se_spectral_density(w_flt.astype(jnp.float32), rho=0.3).dtype == jnp.float32


@pytest.mark.parametrize('kernel,nu', [('matern12', 0.5), ('matern32', 1.5),
                                       ('matern52', 2.5), ('rbf', None)])
def test_spectral_density_inverse_ft_matches_sklearn(kernel, nu):
    """k(r) = (1/pi) int_0^inf S(w) cos(w r) dw must reproduce the scikit-learn
    Matern/RBF kernel -- the parameterisation is the reference one."""
    quad = pytest.importorskip('scipy.integrate').quad
    skl = pytest.importorskip('sklearn.gaussian_process.kernels')
    rho = 0.5
    if nu is None:
        ker = skl.RBF(length_scale=rho)
    else:
        ker = skl.Matern(length_scale=rho, nu=nu)

    def s_of(w):
        return float(spectral_density(jnp.asarray(float(w)), kernel=kernel, rho=rho))

    for r in (0.0, 0.2, 0.5, 1.0, 2.0):
        # Fourier-cosine quadrature (QAWF) for the oscillatory tail; at r=0 the
        # integrand is non-oscillatory so plain adaptive quad is used.
        if r == 0.0:
            recon, _ = quad(s_of, 0.0, np.inf, limit=200)
        else:
            recon, _ = quad(s_of, 0.0, np.inf, weight='cos', wvar=r, limit=200)
        recon /= np.pi
        ref = float(ker(np.array([[0.0]]), np.array([[r]]))[0, 0])
        assert abs(recon - ref) < 1e-6, (kernel, r, recon, ref)


# ---------------------------------------------------------------------------
# Basis contract
# ---------------------------------------------------------------------------


def test_hsgp_basis_contract():
    rng = np.random.default_rng(0)
    x = np.sort(rng.uniform(-2.0, 3.0, 90))
    m = 24
    # pre-constraint penalty is identity
    b0 = hsgp_basis(jnp.asarray(x), n_basis=m, kernel='matern52', rho=0.5,
                    center=False)
    assert b0.design.shape == (90, m)
    np.testing.assert_allclose(np.asarray(b0.penalty), np.eye(m), atol=1e-12)
    assert b0.kind == 'hsgp'
    assert np.all(np.isfinite(np.asarray(b0.design)))
    # sum-to-zero constraint drops one column; re-evaluation round-trips exactly
    b = hsgp_basis(jnp.asarray(x), n_basis=m, kernel='matern52', rho=0.5)
    assert b.design.shape == (90, m - 1)
    re = spline_design(b, jnp.asarray(x))
    np.testing.assert_allclose(np.asarray(re), np.asarray(b.design), atol=1e-10)


def test_hsgp_basis_validates():
    x = jnp.linspace(0, 1, 10)
    with pytest.raises(ValueError, match='n_basis'):
        hsgp_basis(x, n_basis=0)
    with pytest.raises(ValueError, match='boundary'):
        hsgp_basis(x, n_basis=5, boundary=0.5)


def test_hsgp_basis_resolution_warning():
    """A short ``rho`` with too small a rank warns (the (m, L, rho) coupling);
    an adequate rank, or ``rho=None``, is silent."""
    import warnings

    x = jnp.linspace(0.0, 1.0, 100)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        hsgp_basis(x, n_basis=8, kernel='matern52', rho=0.02)
    assert any('under-resolve' in str(rec.message) for rec in w)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        hsgp_basis(x, n_basis=40, kernel='matern52', rho=0.3)  # adequate
        hsgp_basis(x, n_basis=8)  # rho=None default -> no resolution warning
    assert not any('under-resolve' in str(rec.message) for rec in w)


# ---------------------------------------------------------------------------
# HSGP -> exact GP (the reference)
# ---------------------------------------------------------------------------


def _dense_gp_mean(x, y, rho, nu, sigma_f, sigma_e):
    skl = pytest.importorskip('sklearn.gaussian_process.kernels')
    ker = skl.Matern(length_scale=rho, nu=nu)
    K = sigma_f**2 * ker(x[:, None], x[:, None])
    return K @ np.linalg.solve(K + sigma_e**2 * np.eye(x.size), y)


def _hsgp_mean(x, y, m, rho, nu, sigma_f, sigma_e):
    kern = {0.5: 'matern12', 1.5: 'matern32', 2.5: 'matern52'}[nu]
    b = hsgp_basis(jnp.asarray(x), n_basis=m, kernel=kern, rho=rho,
                   amplitude=sigma_f, center=False)
    Psi = np.asarray(b.design)
    beta = np.linalg.solve(Psi.T @ Psi + sigma_e**2 * np.eye(m), Psi.T @ y)
    return Psi @ beta


def test_hsgp_converges_to_exact_gp():
    rng = np.random.default_rng(1)
    x = np.sort(rng.uniform(0.0, 1.0, 70))
    y = np.sin(2 * np.pi * x) + 0.1 * rng.standard_normal(x.size)
    rho, nu, sf, se = 0.2, 2.5, 1.0, 0.1
    m_dense = _dense_gp_mean(x, y, rho, nu, sf, se)

    def rel_err(m):
        return np.linalg.norm(_hsgp_mean(x, y, m, rho, nu, sf, se) - m_dense) / \
            np.linalg.norm(m_dense)

    e_low, e_high = rel_err(8), rel_err(40)
    assert e_high < e_low, (e_low, e_high)        # more basis -> closer
    assert e_high < 3e-2, e_high                  # converged at moderate rank


def test_hsgp_matches_sklearn_gpr():
    gp = pytest.importorskip('sklearn.gaussian_process')
    k = gp.kernels
    rng = np.random.default_rng(2)
    x = np.sort(rng.uniform(0.0, 1.0, 60))
    y = np.sin(2 * np.pi * x) + 0.1 * rng.standard_normal(x.size)
    rho, nu, sf, se = 0.2, 2.5, 1.0, 0.1
    kernel = (k.ConstantKernel(sf**2, constant_value_bounds='fixed')
              * k.Matern(length_scale=rho, nu=nu, length_scale_bounds='fixed'))
    reg = gp.GaussianProcessRegressor(kernel=kernel, alpha=se**2, optimizer=None)
    reg.fit(x[:, None], y)
    skl_mean = reg.predict(x[:, None])
    # sklearn predictive mean is the dense reference; HSGP converges to it
    np.testing.assert_allclose(
        _hsgp_mean(x, y, 45, rho, nu, sf, se), skl_mean, atol=3e-2
    )


# ---------------------------------------------------------------------------
# gam_fit integration (the smooth is a drop-in penalised term)
# ---------------------------------------------------------------------------


def test_hsgp_gam_fit_recovers_smooth():
    rng = np.random.default_rng(3)
    x = np.sort(rng.uniform(0.0, 1.0, 120))
    truth = np.sin(2 * np.pi * x)
    y = truth + 0.1 * rng.standard_normal(x.size)
    b = hsgp_basis(jnp.asarray(x), n_basis=25, kernel='matern52', rho=0.15)
    res = gam_fit(jnp.asarray(y[None, :]), [b])
    grid = jnp.linspace(0.02, 0.98, 100)
    eff, se = smooth_partial_effect(res, 0, b, grid)
    t = np.sin(2 * np.pi * np.asarray(grid))
    t = t - t.mean()
    assert np.corrcoef(np.asarray(eff[0]), t)[0, 1] > 0.97
    assert eff.shape == (1, 100) and se.shape == (1, 100)
    assert np.all(np.asarray(se[0]) > 0)


def test_hsgp_gam_fit_mass_univariate():
    """Shared HSGP design, V independent responses -> per-voxel recovery."""
    rng = np.random.default_rng(4)
    x = np.sort(rng.uniform(0.0, 1.0, 100))
    truth = np.sin(2 * np.pi * x)
    V = 5
    Y = truth[None, :] + 0.1 * rng.standard_normal((V, x.size))
    b = hsgp_basis(jnp.asarray(x), n_basis=20, kernel='matern52', rho=0.15)
    res = gam_fit(jnp.asarray(Y), [b])
    assert res.coef.shape[0] == V
    grid = jnp.linspace(0.02, 0.98, 80)
    eff, _ = smooth_partial_effect(res, 0, b, grid)
    t = np.sin(2 * np.pi * np.asarray(grid))
    t = t - t.mean()
    for v in range(V):
        assert np.corrcoef(np.asarray(eff[v]), t)[0, 1] > 0.95
