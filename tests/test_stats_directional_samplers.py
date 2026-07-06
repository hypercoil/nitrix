# -*- coding: utf-8 -*-
"""Tests for the axial samplers ``watson_sample`` / ``bingham_sample``.

Both use the Angular-Central-Gaussian rejection (Kent--Ganeiber--Mardia 2018):
guaranteed acceptance, bounded efficiency uniformly in concentration and
dimension, and normaliser-free.  Validated against the exact Watson second-moment
oracle ``E[(mu^T x)^2] = d/dkappa log M(1/2, p/2, kappa)`` (Watson is the rank-one
Bingham), plus fit round-trips and the Bingham mode structure.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update('jax_enable_x64', True)

from nitrix.stats import bingham_sample, watson_fit, watson_sample
from nitrix.stats.directional import _watson_g


def _key(i=0):
    return jax.random.PRNGKey(i)


def _orthonormal(p, seed):
    a = np.random.default_rng(seed).standard_normal((p, p))
    return jnp.asarray(np.linalg.qr(a)[0])


# ---------------------------------------------------------------------------
# Watson sampler
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('p', [3, 6, 10])
@pytest.mark.parametrize('kappa', [3.0, 20.0, 60.0, -20.0, -60.0])
def test_watson_sample_second_moment_matches_oracle(p, kappa):
    """E[(mu^T x)^2] of the samples equals the exact Watson second moment
    g(kappa), for both bipolar (kappa>0) and girdle (kappa<0), any dimension."""
    mu = jnp.eye(p)[1]
    x = watson_sample(_key(0), mu, jnp.asarray(kappa), (40000,))
    assert x.shape == (40000, p)
    e_t = float(jnp.mean((x @ mu) ** 2))
    oracle = float(_watson_g(p / 2.0, jnp.asarray(kappa)))
    assert abs(e_t - oracle) < 0.01


def test_watson_sample_unit_norm_and_axial():
    mu = jnp.asarray([0.0, 0.0, 1.0])
    x = watson_sample(_key(1), mu, jnp.asarray(15.0), (5000,))
    np.testing.assert_allclose(jnp.linalg.norm(x, axis=-1), 1.0, atol=1e-10)
    # Axial: the sign of mu^T x is symmetric (mean ~ 0 for bipolar).
    assert abs(float(jnp.mean(x @ mu))) < 0.05


def test_watson_sample_deterministic():
    mu = jnp.asarray([1.0, 0.0, 0.0])
    a = watson_sample(_key(2), mu, jnp.asarray(8.0), (100,))
    b = watson_sample(_key(2), mu, jnp.asarray(8.0), (100,))
    assert bool(jnp.array_equal(a, b))


def test_watson_sample_fit_round_trip():
    """Sampling then fitting recovers the axis and a consistent concentration."""
    mu = jnp.asarray([0.0, 1.0, 0.0])
    x = watson_sample(_key(3), mu, jnp.asarray(12.0), (8000,))
    fit = watson_fit(x)
    assert abs(float(fit.mu[1])) > 0.99
    assert float(fit.kappa) > 0.0
    r_bar = float(jnp.mean((x @ fit.mu) ** 2))
    assert abs(float(_watson_g(1.5, fit.kappa)) - r_bar) < 1e-3


def test_watson_sample_girdle_concentrates_on_equator():
    mu = jnp.asarray([0.0, 0.0, 1.0])
    x = watson_sample(_key(4), mu, jnp.asarray(-30.0), (5000,))
    # girdle: mass near the equator, so |mu^T x| is small on average.
    assert float(jnp.mean(jnp.abs(x @ mu))) < 0.2


# ---------------------------------------------------------------------------
# Bingham sampler
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('kappa', [10.0, 40.0])
def test_bingham_rank_one_matches_watson_oracle(kappa):
    """Bingham with beta = (kappa, 0, ...) is the Watson law; its axis-0 second
    moment matches the Watson oracle."""
    frame = jnp.eye(3)
    beta = jnp.asarray([kappa, 0.0, 0.0])
    x = bingham_sample(_key(0), frame, beta, (40000,))
    e = float(jnp.mean(x[:, 0] ** 2))
    oracle = float(_watson_g(1.5, jnp.asarray(kappa)))
    assert abs(e - oracle) < 0.01


def test_bingham_mode_structure():
    """Larger beta_j gives larger E[(gamma_j^T x)^2] (mass concentrates on the
    top-coefficient axes)."""
    frame = _orthonormal(5, 0)
    beta = jnp.asarray([3.0, 1.0, 0.0, -1.0, -2.0])  # descending
    x = bingham_sample(_key(1), frame, beta, (40000,))
    proj2 = jnp.mean((x @ frame) ** 2, axis=0)  # E[(gamma_j^T x)^2]
    # non-increasing with j (allowing small MC noise)
    assert bool(jnp.all(jnp.diff(proj2) <= 0.02))
    # and normalised: the second moments sum to 1 (x on the sphere)
    assert abs(float(jnp.sum(proj2)) - 1.0) < 1e-9


def test_bingham_unit_norm_shape_jit():
    frame = _orthonormal(4, 2)
    beta = jnp.asarray([2.0, 1.0, -1.0, -2.0])
    x = bingham_sample(_key(2), frame, beta, (3, 500))
    assert x.shape == (3, 500, 4)
    np.testing.assert_allclose(jnp.linalg.norm(x, axis=-1), 1.0, atol=1e-10)
    xj = jax.jit(lambda k: bingham_sample(k, frame, beta, (200,)))(_key(3))
    assert bool(jnp.all(jnp.isfinite(xj)))


def test_bingham_uniform_when_beta_constant():
    """Constant beta gives the uniform distribution (no preferred axis)."""
    frame = _orthonormal(3, 5)
    x = bingham_sample(_key(4), frame, jnp.zeros(3), (20000,))
    # empirical second-moment matrix ~ I/3 (isotropic)
    m = (x.T @ x) / x.shape[0]
    np.testing.assert_allclose(m, jnp.eye(3) / 3.0, atol=0.02)
