# -*- coding: utf-8 -*-
"""Tests for the Watson (axial) distribution in ``nitrix.stats.directional``.

The Watson law on :math:`S^{p-1}` is antipodally symmetric with density
:math:`\\propto \\exp(\\kappa (\\mu^\\top x)^2)`, normalised by Kummer's
confluent hypergeometric :math:`M(1/2, p/2, \\kappa)`.  Coverage: the normaliser
vs an mpmath oracle (bipolar + girdle), the surface-measure density (integrates
to 1, matching the vMF convention), MLE recovery of both axial regimes, and
gradient stability (notably at :math:`\\kappa = 0`).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update('jax_enable_x64', True)

from nitrix.stats import (
    log_kummer_m,
    vmf_sample,
    watson_fit,
    watson_log_prob,
)
from nitrix.stats.directional import _watson_g

mpmath = pytest.importorskip('mpmath')
mpmath.mp.dps = 30


def _sphere_grid(n=500):
    th = np.linspace(0, np.pi, n)
    ph = np.linspace(0, 2 * np.pi, n)
    t, p = np.meshgrid(th, ph, indexing='ij')
    x = np.stack(
        [np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)], axis=-1
    )
    d_area = np.sin(t) * (th[1] - th[0]) * (ph[1] - ph[0])
    return jnp.asarray(x), d_area


# ---------------------------------------------------------------------------
# Normaliser log M(1/2, b, kappa) vs mpmath
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('p', [2, 3, 4, 10, 50])
def test_log_kummer_matches_mpmath(p):
    """Full-range (bipolar kappa>0 and girdle kappa<0) to machine precision."""
    b = p / 2.0
    zs = jnp.asarray([-500.0, -30.0, -2.0, 0.0, 2.0, 30.0, 100.0, 300.0, 1000.0])
    got = np.asarray(log_kummer_m(0.5, b, zs))
    ref = np.array([float(mpmath.log(mpmath.hyp1f1(0.5, b, float(z)))) for z in zs])
    np.testing.assert_allclose(got, ref, atol=1e-10)


def test_log_kummer_zero_is_zero():
    """M(a, b, 0) = 1, so log M = 0 for any p."""
    for p in (2, 3, 7):
        assert abs(float(log_kummer_m(0.5, p / 2.0, jnp.asarray(0.0)))) < 1e-12


# ---------------------------------------------------------------------------
# Density: surface-measure normalisation (integrates to one, like vMF)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('kappa', [3.0, -3.0, 10.0, -8.0])
def test_density_integrates_to_one(kappa):
    x, d_area = _sphere_grid(700)
    mu = jnp.asarray([0.0, 0.0, 1.0])
    dens = np.exp(np.asarray(watson_log_prob(x, mu, jnp.asarray(kappa))))
    integral = float((dens * d_area).sum())
    assert abs(integral - 1.0) < 5e-3


def test_axial_symmetry():
    """f(x) = f(-x) and f depends on mu only through +-mu."""
    x = jnp.asarray([[0.6, 0.0, 0.8], [0.1, 0.7, -0.7]])
    mu = jnp.asarray([0.0, 0.0, 1.0])
    kappa = jnp.asarray(4.0)
    lp = watson_log_prob(x, mu, kappa)
    np.testing.assert_allclose(lp, watson_log_prob(-x, mu, kappa), atol=1e-12)
    np.testing.assert_allclose(lp, watson_log_prob(x, -mu, kappa), atol=1e-12)


def test_bipolar_peaks_at_axis_girdle_at_equator():
    mu = jnp.asarray([0.0, 0.0, 1.0])
    pole = jnp.asarray([0.0, 0.0, 1.0])
    equator = jnp.asarray([1.0, 0.0, 0.0])
    # kappa > 0 (bipolar): denser at the pole than the equator.
    assert float(watson_log_prob(pole, mu, jnp.asarray(5.0))) > float(
        watson_log_prob(equator, mu, jnp.asarray(5.0))
    )
    # kappa < 0 (girdle): denser at the equator.
    assert float(watson_log_prob(equator, mu, jnp.asarray(-5.0))) > float(
        watson_log_prob(pole, mu, jnp.asarray(-5.0))
    )


# ---------------------------------------------------------------------------
# MLE fit: recovers both axial regimes, MLE-consistent concentration
# ---------------------------------------------------------------------------


def _bipolar_data(axis, kappa, n, seed):
    """Axial samples clustered at +-axis (vMF around axis, random sign)."""
    rng = np.random.default_rng(seed)
    x = np.asarray(
        vmf_sample(jax.random.PRNGKey(seed), jnp.asarray(axis), jnp.asarray(kappa), (n,))
    )
    return jnp.asarray(x * rng.choice([-1.0, 1.0], size=(n, 1)))


def _girdle_data(n, seed):
    """A ring near the equator of the z-axis (small z-component)."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 3))
    x[:, 2] *= 0.15
    return jnp.asarray(x / np.linalg.norm(x, axis=1, keepdims=True))


def test_fit_recovers_bipolar_axis_and_positive_kappa():
    data = _bipolar_data([0.0, 0.0, 1.0], 8.0, 4000, 1)
    fit = watson_fit(data)
    assert abs(float(fit.mu[2])) > 0.99  # axis ~ z
    assert float(fit.kappa) > 0.0  # bipolar
    # MLE consistency: g(kappa_hat) == mean of (mu^T x)^2.
    r_bar = float(jnp.mean((data @ fit.mu) ** 2))
    assert abs(float(_watson_g(1.5, fit.kappa)) - r_bar) < 1e-4


def test_fit_recovers_girdle_axis_and_negative_kappa():
    data = _girdle_data(4000, 0)
    fit = watson_fit(data)
    assert abs(float(fit.mu[2])) > 0.99  # girdle axis ~ z (pole of the ring)
    assert float(fit.kappa) < 0.0  # girdle
    r_bar = float(jnp.mean((data @ fit.mu) ** 2))
    assert abs(float(_watson_g(1.5, fit.kappa)) - r_bar) < 1e-4


def test_fit_apply_seam_scores_data():
    """The fitted state evaluates a finite log-density on its own data."""
    data = _bipolar_data([0.0, 1.0, 0.0], 5.0, 1000, 3)
    fit = watson_fit(data)
    lp = watson_log_prob(data, fit.mu, fit.kappa)
    assert bool(jnp.all(jnp.isfinite(lp)))
    assert lp.shape == (1000,)


def test_fit_weighted_matches_replication():
    """Integer weights match replicating observations (soft-responsibility use)."""
    data = _bipolar_data([1.0, 0.0, 0.0], 6.0, 300, 5)
    w = jnp.asarray(np.random.default_rng(0).integers(1, 4, size=300).astype(float))
    fit_w = watson_fit(data, weights=w)
    rep = jnp.concatenate([jnp.repeat(data, w.astype(int), axis=0)], axis=0)
    fit_r = watson_fit(rep)
    np.testing.assert_allclose(fit_w.kappa, fit_r.kappa, rtol=1e-6)
    # axes agree up to sign (axial)
    assert abs(float(jnp.abs(jnp.sum(fit_w.mu * fit_r.mu)))) > 0.999


# ---------------------------------------------------------------------------
# Gradients (notably the kappa = 0 boundary) and jit/vmap
# ---------------------------------------------------------------------------


def test_grad_finite_everywhere_including_zero():
    mu = jnp.asarray([0.0, 0.0, 1.0])
    x = jnp.asarray([0.3, 0.4, np.sqrt(1 - 0.25)])
    for k in (-20.0, -1.0, 0.0, 1.0, 20.0):
        g = jax.grad(lambda kk: watson_log_prob(x, mu, kk))(jnp.asarray(k))
        assert bool(jnp.isfinite(g))
    # d/dkappa log M at kappa=0 is 1/p (the uniform second moment).
    assert abs(float(_watson_g(1.5, jnp.asarray(0.0))) - 1.0 / 3.0) < 1e-9


def test_log_prob_jit_and_vmap():
    x = jax.random.normal(jax.random.PRNGKey(0), (16, 3))
    x = x / jnp.linalg.norm(x, axis=-1, keepdims=True)
    mu = jnp.asarray([0.0, 0.0, 1.0])
    out = jax.jit(lambda xx: watson_log_prob(xx, mu, jnp.asarray(2.0)))(x)
    assert out.shape == (16,)
    per = jax.vmap(lambda xi: watson_log_prob(xi, mu, jnp.asarray(2.0)))(x)
    np.testing.assert_allclose(out, per, atol=1e-12)
