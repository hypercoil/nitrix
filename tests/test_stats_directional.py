# -*- coding: utf-8 -*-
"""Tests for ``nitrix.stats.directional`` -- the von Mises-Fisher family.

Coverage keyed to the correctness mandate (theory over the legacy code):

- ``log_iv`` matches a high-precision Bessel oracle (``mpmath``) across a
  ``(nu, kappa)`` grid spanning small -> large ``kappa`` *and* large ``p`` (the
  surface-feature regime) -- the explicit refutation of the legacy
  large-``kappa``-only asymptotic, which is materially wrong at small ``kappa``.
- ``log_iv`` is grad-finite across regimes and the internal switch boundary,
  and jit-clean.
- ``vmf_log_prob`` integrates to 1 over the sphere (quadrature) and is
  grad-finite in ``mu`` / ``kappa``.
- ``vmf_fit`` recovers planted ``(mu, kappa)`` and returns the *exact*
  ``A_p(kappa) = Rbar`` root (not merely the Banerjee warm start).
- ``vmf_sample`` always returns unit-norm valid samples (Wood-1994 guaranteed
  acceptance -- no ``found=False`` path), and its empirical ``(mu, kappa)``
  recover the generating parameters.

Run on the CPU correctness floor (x64).
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

jax.config.update('jax_enable_x64', True)

import jax.numpy as jnp  # noqa: E402
from scipy.special import ive  # noqa: E402

from nitrix.stats import (  # noqa: E402
    VMFFit,
    log_iv,
    vmf_fit,
    vmf_log_prob,
    vmf_sample,
)
from nitrix.stats.directional import _a_ratio  # noqa: E402

mp = pytest.importorskip('mpmath')
mp.mp.dps = 40


def _ref_log_iv(nu: float, kappa: float) -> float:
    return float(mp.log(mp.besseli(nu, kappa)))


# --------------------------------------------------------------------------- #
# log_iv                                                                      #
# --------------------------------------------------------------------------- #

_NUS = [0.0, 0.5, 1.0, 2.0, 5.0, 8.0, 14.0, 15.0, 30.0, 100.0, 300.0]
_KAPPAS = np.array(
    [1e-3, 1e-2, 0.1, 0.5, 1.0, 3.0, 10.0, 30.0, 80.0, 120.0, 300.0, 1e3, 1e4]
)


@pytest.mark.parametrize('nu', _NUS)
def test_log_iv_matches_oracle_full_range(nu: float) -> None:
    got = np.asarray(log_iv(nu, jnp.asarray(_KAPPAS)))
    ref = np.array([_ref_log_iv(nu, float(k)) for k in _KAPPAS])
    # Tight across the whole grid (the crossover at nu=15 is the worst spot).
    np.testing.assert_allclose(got, ref, atol=1e-8, rtol=0)


def test_log_iv_refutes_legacy_large_kappa_only() -> None:
    # The legacy asymptotic (leading term only) is materially wrong at small
    # kappa; log_iv is not.  Demonstrate both against the oracle at nu=2.
    nu = 2.0
    small_kappa = np.array([0.05, 0.2, 0.8, 2.0])

    def legacy(nu: float, k: np.ndarray) -> np.ndarray:
        r = nu / k
        rad = 1 + r**2
        return (
            -0.5 * (np.log(2 * np.pi * k) + 0.5 * np.log(rad))
            + k * np.sqrt(rad)
            - nu * np.arcsinh(r)
        )

    ref = np.array([_ref_log_iv(nu, float(k)) for k in small_kappa])
    ours = np.asarray(log_iv(nu, jnp.asarray(small_kappa)))
    legacy_err = np.abs(legacy(nu, small_kappa) - ref)
    our_err = np.abs(ours - ref)
    assert our_err.max() < 1e-8
    assert legacy_err.max() > 1e-2  # legacy is materially wrong here


@pytest.mark.parametrize('nu', [0.5, 2.0, 14.0, 30.0])
@pytest.mark.parametrize('kappa', [0.05, 5.0, 119.0, 121.0, 5000.0])
def test_log_iv_grad_finite_across_regimes(nu: float, kappa: float) -> None:
    g = jax.grad(lambda k: log_iv(nu, k))(kappa)
    assert bool(jnp.isfinite(g))


def test_log_iv_grad_matches_bessel_ratio() -> None:
    # d/dkappa log I_nu(kappa) = I_{nu+1}/I_nu + nu/kappa = A-like identity;
    # equivalently I_nu'(k)/I_nu(k) = 0.5 (I_{nu-1}+I_{nu+1})/I_nu.
    nu, kappa = 3.0, 7.0
    g = float(jax.grad(lambda k: log_iv(nu, k))(kappa))
    ref = float(
        0.5
        * (mp.besseli(nu - 1, kappa) + mp.besseli(nu + 1, kappa))
        / mp.besseli(nu, kappa)
    )
    assert abs(g - ref) < 1e-6


def test_log_iv_jit() -> None:
    out = jax.jit(lambda k: log_iv(2.0, k))(jnp.asarray(_KAPPAS))
    assert out.shape == _KAPPAS.shape


# --------------------------------------------------------------------------- #
# vmf_log_prob                                                                #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize('kappa', [0.5, 2.0, 20.0])
def test_vmf_log_prob_integrates_to_one_on_s2(kappa: float) -> None:
    ng = 500
    th = np.linspace(0, np.pi, ng)
    ph = np.linspace(0, 2 * np.pi, 2 * ng)
    tt, pp = np.meshgrid(th, ph, indexing='ij')
    x = np.stack(
        [np.sin(tt) * np.cos(pp), np.sin(tt) * np.sin(pp), np.cos(tt)], -1
    )
    mu = np.array([0.0, 0.0, 1.0])
    lp = np.asarray(
        vmf_log_prob(jnp.asarray(x.reshape(-1, 3)), jnp.asarray(mu), kappa)
    ).reshape(tt.shape)
    dens = np.exp(lp) * np.sin(tt)  # spherical area element
    integral = np.trapezoid(np.trapezoid(dens, ph, axis=1), th)
    assert abs(integral - 1.0) < 1e-3


def test_vmf_log_prob_matches_closed_form_and_scipy_norm() -> None:
    # Cross-check the normaliser against scipy's scaled Bessel ive.
    rng = np.random.default_rng(0)
    p, kappa = 4, 6.0
    mu = np.array([1.0, 0.0, 0.0, 0.0])
    x = rng.standard_normal((10, p))
    x = x / np.linalg.norm(x, axis=-1, keepdims=True)
    nu = p / 2 - 1
    log_norm = (
        nu * np.log(kappa)
        - (p / 2) * np.log(2 * np.pi)
        - (np.log(ive(nu, kappa)) + kappa)  # log I_nu = log ive + kappa
    )
    expected = kappa * (x @ mu) + log_norm
    got = np.asarray(vmf_log_prob(jnp.asarray(x), jnp.asarray(mu), kappa))
    np.testing.assert_allclose(got, expected, atol=1e-7)


def test_vmf_log_prob_reduction_and_grad() -> None:
    rng = np.random.default_rng(1)
    x = rng.standard_normal((32, 3))
    x = jnp.asarray(x / np.linalg.norm(x, axis=-1, keepdims=True))
    mu = jnp.asarray([0.0, 0.0, 1.0])
    per = vmf_log_prob(x, mu, 5.0)
    assert per.shape == (32,)  # 'none' default: per-observation
    total = vmf_log_prob(x, mu, 5.0, reduction='sum')
    np.testing.assert_allclose(float(total), float(per.sum()), atol=1e-9)

    g = jax.grad(lambda mk: vmf_log_prob(x, mk[:3], mk[3], reduction='sum'))(
        jnp.asarray([0.0, 0.0, 1.0, 8.0])
    )
    assert bool(jnp.all(jnp.isfinite(g)))


# --------------------------------------------------------------------------- #
# vmf_fit                                                                     #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize('p,kappa_true', [(3, 20.0), (10, 50.0), (2, 5.0)])
def test_vmf_fit_recovers_planted(p: int, kappa_true: float) -> None:
    mu_true = np.zeros(p)
    mu_true[0], mu_true[1] = 0.6, 0.8  # a unit vector
    x = vmf_sample(
        jax.random.PRNGKey(p), jnp.asarray(mu_true), kappa_true, shape=(20000,)
    )
    fit = vmf_fit(x)
    assert isinstance(fit, VMFFit)
    assert np.linalg.norm(np.asarray(fit.mu) - mu_true) < 0.05
    assert abs(float(fit.kappa) - kappa_true) / kappa_true < 0.1


def test_vmf_fit_kappa_solves_exact_a_p_root() -> None:
    p = 5
    x = vmf_sample(
        jax.random.PRNGKey(99), jnp.asarray(np.eye(p)[0]), 30.0, shape=(5000,)
    )
    fit = vmf_fit(x)
    resultant = np.asarray(jnp.sum(x, axis=0))
    r_bar = np.linalg.norm(resultant) / x.shape[0]
    # kappa_hat is the *exact* root of A_p(kappa) = Rbar, not just Banerjee.
    assert abs(float(_a_ratio(p, fit.kappa)) - r_bar) < 1e-10
    banerjee = r_bar * (p - r_bar**2) / (1 - r_bar**2)
    assert abs(float(fit.kappa) - banerjee) > 1e-6  # refinement moved it


def test_vmf_fit_weighted_and_jit_and_grad() -> None:
    rng = np.random.default_rng(3)
    x = rng.standard_normal((100, 4))
    x = jnp.asarray(x / np.linalg.norm(x, axis=-1, keepdims=True))
    w = jnp.asarray(rng.uniform(size=100))
    fit = vmf_fit(x, weights=w)
    np.testing.assert_allclose(
        np.linalg.norm(np.asarray(fit.mu)), 1.0, atol=1e-9
    )
    jfit = jax.jit(lambda a: vmf_fit(a).kappa)(x)
    assert bool(jnp.isfinite(jfit))
    g = jax.grad(lambda a: vmf_fit(a).kappa)(x)
    assert bool(jnp.all(jnp.isfinite(g)))


# --------------------------------------------------------------------------- #
# vmf_sample                                                                  #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    'p,kappa', [(3, 20.0), (10, 50.0), (2, 5.0), (64, 100.0)]
)
def test_vmf_sample_always_unit_norm(p: int, kappa: float) -> None:
    mu = np.zeros(p)
    mu[0] = 1.0
    x = vmf_sample(
        jax.random.PRNGKey(p), jnp.asarray(mu), kappa, shape=(5000,)
    )
    norms = np.asarray(jnp.linalg.norm(x, axis=-1))
    # Guaranteed acceptance => every sample is a valid unit vector.
    np.testing.assert_allclose(norms, 1.0, atol=1e-6)
    assert x.shape == (5000, p)


def test_vmf_sample_empirical_recovers_parameters() -> None:
    kappa_true = 15.0
    mu_true = np.array([0.0, 1.0, 0.0])
    x = vmf_sample(
        jax.random.PRNGKey(0), jnp.asarray(mu_true), kappa_true, shape=(50000,)
    )
    mean = np.asarray(jnp.mean(x, axis=0))
    mu_emp = mean / np.linalg.norm(mean)
    assert np.dot(mu_emp, mu_true) > 0.999
    fit = vmf_fit(x)
    assert abs(float(fit.kappa) - kappa_true) / kappa_true < 0.05


def test_vmf_sample_jit_and_default_shape() -> None:
    mu = jnp.asarray([1.0, 0.0, 0.0])
    single = vmf_sample(jax.random.PRNGKey(1), mu, 10.0)
    assert single.shape == (3,)
    xj = jax.jit(lambda k: vmf_sample(k, mu, 10.0, shape=(256,)))(
        jax.random.PRNGKey(2)
    )
    np.testing.assert_allclose(
        np.asarray(jnp.linalg.norm(xj, axis=-1)), 1.0, atol=1e-6
    )


def test_vmf_sample_concentrates_with_kappa() -> None:
    # Larger kappa => samples cluster tighter around mu (mean resultant -> 1).
    mu = jnp.asarray([0.0, 0.0, 1.0])
    r_bars = []
    for kappa in [1.0, 10.0, 100.0]:
        x = vmf_sample(jax.random.PRNGKey(5), mu, kappa, shape=(10000,))
        r_bars.append(float(jnp.linalg.norm(jnp.mean(x, axis=0))))
    assert r_bars[0] < r_bars[1] < r_bars[2]
