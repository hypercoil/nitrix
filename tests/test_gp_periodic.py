# -*- coding: utf-8 -*-
"""Tests for the periodic Gaussian-process kernel (stats.gp kernel='periodic').

The reduced-rank periodic kernel is pinned by: the exponentially-scaled modified
Bessel weights matching scipy; the Fourier-Bessel basis reproducing the exact
MacKay periodic covariance; a periodic signal recovered by gp_fit with the
length-scale REML-estimated; and -- the distinctive property over a cyclic spline
-- correct *periodic extrapolation* beyond the observed range. The SE/Matern path
is checked untouched.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update('jax_enable_x64', True)

from nitrix.stats._periodic import (  # noqa: E402
    _ive,
    periodic_features,
    periodic_penalty_diag,
)
from nitrix.stats.gp import gp_fit, gp_predict  # noqa: E402


def _periodic_signal(n=120, period=1.0, seed=0):
    x = np.linspace(0.0, 3.0, n)  # three periods
    truth = np.sin(2 * np.pi * x / period) + 0.5 * np.cos(4 * np.pi * x / period)
    y = truth[None, :] + 0.1 * np.random.default_rng(seed).standard_normal((1, n))
    return x, truth, y


def test_ive_matches_scipy():
    sp = pytest.importorskip('scipy.special')
    for z in [0.05, 0.5, 2.0, 10.0, 50.0]:
        got = np.asarray(_ive(jnp.asarray(z), 20))
        ref = sp.ive(np.arange(21), z)
        np.testing.assert_allclose(got, ref, atol=1e-12)


def test_basis_reproduces_mackay_kernel():
    # Full Fourier-Bessel expansion (with the DC term) equals the MacKay kernel.
    x = np.linspace(0.0, 1.0, 15)
    period, ell, order = 1.0, 0.7, 25
    ive = np.asarray(_ive(jnp.asarray(1.0 / ell**2), order))
    phi = np.asarray(periodic_features(jnp.asarray(x), period, order))  # (n, 2m)
    design = np.concatenate([np.ones((x.shape[0], 1)), phi], axis=1)
    weights = np.concatenate([[ive[0]], np.repeat(2.0 * ive[1:], 2)])
    gram = design @ np.diag(weights) @ design.T
    tau = x[:, None] - x[None, :]
    true = np.exp(-2 * np.sin(np.pi * tau / period) ** 2 / ell**2)
    np.testing.assert_allclose(gram, true, atol=1e-7)


def test_penalty_diag_shape_and_positive():
    d, logdet = periodic_penalty_diag(10, jnp.asarray(0.8), n_fixed=1)
    assert d.shape == (1 + 2 * 10,)  # n_fixed + 2*order
    assert float(d[0]) == 0.0  # fixed (intercept) column unpenalised
    assert bool((d[1:] > 0).all())
    assert bool(jnp.isfinite(logdet))


def test_gp_fit_recovers_periodic_signal():
    x, truth, y = _periodic_signal()
    res = gp_fit(jnp.asarray(y), jnp.asarray(x), kernel='periodic', period=1.0, rank=12)
    assert res.period == 1.0
    assert res.rank == 24  # 2 * 12 harmonics
    mean, _ = gp_predict(res, jnp.asarray(x))
    corr = np.corrcoef(np.asarray(mean)[0], truth)[0, 1]
    assert corr > 0.99


def test_periodic_extrapolates_beyond_training_range():
    # The distinctive property vs a cyclic spline: the fit continues the pattern
    # into an unobserved period rather than flattening.
    x, _, y = _periodic_signal()
    res = gp_fit(jnp.asarray(y), jnp.asarray(x), kernel='periodic', period=1.0, rank=12)
    xg = np.linspace(3.0, 4.0, 60)  # a fourth, unseen period
    truth_g = np.sin(2 * np.pi * xg) + 0.5 * np.cos(4 * np.pi * xg)
    mean_g, _ = gp_predict(res, jnp.asarray(xg))
    corr = np.corrcoef(np.asarray(mean_g)[0], truth_g)[0, 1]
    assert corr > 0.98


def test_se_path_unaffected():
    x, _, y = _periodic_signal()
    res = gp_fit(jnp.asarray(y), jnp.asarray(x), kernel='matern52')
    assert res.period is None  # additive field defaults to None; no regression
    assert bool(jnp.isfinite(res.edf[0]))


def test_periodic_guards():
    x, _, y = _periodic_signal()
    Y, X = jnp.asarray(y), jnp.asarray(x)
    with pytest.raises(ValueError):
        gp_fit(Y, X, kernel='periodic')  # period required
    with pytest.raises(NotImplementedError):
        gp_fit(Y, X, kernel='periodic', period=1.0, engine='exact')
    with pytest.raises(NotImplementedError):
        gp_fit(Y, X, kernel='periodic', period=1.0, family='poisson')
