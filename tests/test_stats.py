# -*- coding: utf-8 -*-
"""Tests for ``nitrix.stats`` -- covariance, correlation, Fourier.

Coverage:

- **covariance**: match ``numpy.cov`` to machine eps on real and
  complex inputs; check Hermitian property of complex cov; ridge,
  weights (vector and matrix), and conditional cov paths.  The
  key concern -- "silently wrong on complex inputs" -- is
  explicitly tested via Hermiticity and ``np.cov`` parity.
- **fourier**: match ``scipy.signal.hilbert`` to machine eps;
  verify the analytic-signal definition on a cos / sin pair;
  shape contract for ``env_inst``; analytic-signal raises on
  complex input.
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.signal

jax.config.update('jax_enable_x64', True)

from nitrix.stats import (
    analytic_signal,
    ccorr,
    ccov,
    conditionalcorr,
    conditionalcov,
    corr,
    corrcoef,
    cov,
    env_inst,
    envelope,
    hilbert_transform,
    instantaneous_frequency,
    instantaneous_phase,
    pairedcorr,
    pairedcov,
    partialcorr,
    partialcov,
    pcorr,
    precision,
    product_filter,
    product_filtfilt,
)


# ---------------------------------------------------------------------------
# covariance: real-valued
# ---------------------------------------------------------------------------


def test_cov_matches_numpy_real():
    rng = np.random.default_rng(0)
    X_np = rng.standard_normal((5, 200))
    X = jnp.asarray(X_np)
    S = cov(X)
    S_ref = np.cov(X_np, bias=False)
    np.testing.assert_allclose(np.asarray(S), S_ref, atol=1e-13)


def test_cov_bias_matches_numpy():
    rng = np.random.default_rng(0)
    X_np = rng.standard_normal((5, 200))
    X = jnp.asarray(X_np)
    S = cov(X, bias=True)
    S_ref = np.cov(X_np, bias=True)
    np.testing.assert_allclose(np.asarray(S), S_ref, atol=1e-13)


def test_cov_ddof_overrides_bias():
    X = jnp.asarray(np.random.default_rng(0).standard_normal((4, 100)))
    np.testing.assert_allclose(
        cov(X, bias=True), cov(X, ddof=0), atol=1e-14,
    )
    np.testing.assert_allclose(
        cov(X, bias=False), cov(X, ddof=1), atol=1e-14,
    )


def test_cov_correlation_diagonal_is_one():
    X = jnp.asarray(np.random.default_rng(0).standard_normal((5, 100)))
    R = corr(X)
    np.testing.assert_allclose(jnp.diagonal(R), 1.0, atol=1e-12)


def test_cov_ridge_adds_to_diagonal():
    X = jnp.asarray(np.random.default_rng(0).standard_normal((5, 100)))
    S = cov(X)
    S_ridge = cov(X, l2=1.0)
    np.testing.assert_allclose(S_ridge - S, jnp.eye(5), atol=1e-12)


def test_cov_vector_weights_uniform_matches_unweighted():
    X = jnp.asarray(np.random.default_rng(0).standard_normal((4, 100)))
    w = jnp.ones(100)
    np.testing.assert_allclose(cov(X), cov(X, weights=w), atol=1e-14)


def test_cov_non_uniform_weights_match_numpy_aweights():
    rng = np.random.default_rng(0)
    X_np = rng.standard_normal((4, 100))
    X = jnp.asarray(X_np)
    w_np = np.abs(rng.standard_normal(100)) + 0.5
    w = jnp.asarray(w_np)
    S = cov(X, weights=w)
    S_ref = np.cov(X_np, aweights=w_np, bias=False)
    np.testing.assert_allclose(np.asarray(S), S_ref, atol=1e-13)


def test_cov_pass_both_weight_args_raises():
    X = jnp.zeros((3, 10))
    w = jnp.ones(10)
    W = jnp.eye(10)
    with pytest.raises(ValueError, match='at most one of'):
        cov(X, weights=w, weight_matrix=W)


# ---------------------------------------------------------------------------
# covariance: complex-valued (the silently-wrong concern)
# ---------------------------------------------------------------------------


def test_cov_complex_matches_numpy():
    '''Complex cov must match np.cov to machine eps; this was the
    "silently wrong" failure mode in the legacy code.
    '''
    rng = np.random.default_rng(0)
    n_chan, n_obs = 5, 100
    X_np = (
        rng.standard_normal((n_chan, n_obs))
        + 1j * rng.standard_normal((n_chan, n_obs))
    )
    X = jnp.asarray(X_np)
    S = cov(X)
    S_ref = np.cov(X_np, bias=False)
    np.testing.assert_allclose(np.asarray(S), S_ref, atol=1e-13)


def test_cov_complex_is_hermitian():
    rng = np.random.default_rng(0)
    X_np = (
        rng.standard_normal((5, 100))
        + 1j * rng.standard_normal((5, 100))
    )
    X = jnp.asarray(X_np)
    S = cov(X)
    np.testing.assert_allclose(S, S.conj().T, atol=1e-13)


def test_cov_complex_diagonal_is_real_positive():
    rng = np.random.default_rng(0)
    X = jnp.asarray(
        rng.standard_normal((5, 100))
        + 1j * rng.standard_normal((5, 100))
    )
    d = jnp.diagonal(cov(X))
    np.testing.assert_allclose(d.imag, 0.0, atol=1e-15)
    assert bool(jnp.all(d.real > 0))


def test_corr_complex_diagonal_is_one():
    rng = np.random.default_rng(0)
    X = jnp.asarray(
        rng.standard_normal((5, 100))
        + 1j * rng.standard_normal((5, 100))
    )
    R = corr(X)
    np.testing.assert_allclose(jnp.diagonal(R), 1.0, atol=1e-12)


def test_pairedcov_complex_matches_numpy_cross_block():
    rng = np.random.default_rng(0)
    n_obs = 100
    X_np = (
        rng.standard_normal((5, n_obs)) + 1j * rng.standard_normal((5, n_obs))
    )
    Y_np = (
        rng.standard_normal((3, n_obs)) + 1j * rng.standard_normal((3, n_obs))
    )
    P = pairedcov(jnp.asarray(X_np), jnp.asarray(Y_np))
    Z_np = np.concatenate([X_np, Y_np], axis=0)
    S_full = np.cov(Z_np, bias=False)
    P_ref = S_full[:5, 5:]
    np.testing.assert_allclose(np.asarray(P), P_ref, atol=1e-13)


# ---------------------------------------------------------------------------
# precision / partial / conditional
# ---------------------------------------------------------------------------


def test_precision_inverse_of_cov():
    X = jnp.asarray(np.random.default_rng(0).standard_normal((4, 200)))
    S = cov(X)
    P = precision(X)
    np.testing.assert_allclose(S @ P, jnp.eye(4), atol=1e-9)


def test_partialcorr_diagonal_is_one():
    X = jnp.asarray(np.random.default_rng(0).standard_normal((4, 200)))
    R = partialcorr(X)
    np.testing.assert_allclose(jnp.diagonal(R), 1.0, atol=1e-10)


def test_conditionalcov_residual_orthogonal_to_y():
    '''After residualisation, the un-centered inner product
    ``X_residual @ Y.T`` is ~0 (orthogonality in the OLS sense,
    which does NOT include centering -- residualise has no
    implicit intercept term).
    '''
    rng = np.random.default_rng(0)
    X = jnp.asarray(rng.standard_normal((5, 200)))
    Y = jnp.asarray(rng.standard_normal((2, 200)))
    from nitrix.linalg import residualise
    X_res = residualise(X, Y, rowvar=True)
    # X_res @ Y.T should be ~0 (orthogonal to Y's row span).
    gram = X_res @ Y.T
    assert float(jnp.abs(gram).max()) < 1e-10


def test_conditionalcov_shape_and_symmetry():
    '''Conditional cov returns symmetric ``(c, c)`` for real input.'''
    rng = np.random.default_rng(0)
    X = jnp.asarray(rng.standard_normal((5, 200)))
    Y = jnp.asarray(rng.standard_normal((2, 200)))
    S = conditionalcov(X, Y)
    assert S.shape == (5, 5)
    np.testing.assert_allclose(S, S.T, atol=1e-12)


def test_aliases_match_canonical():
    X = jnp.asarray(np.random.default_rng(0).standard_normal((4, 100)))
    Y = jnp.asarray(np.random.default_rng(1).standard_normal((2, 100)))
    np.testing.assert_allclose(corrcoef(X), corr(X), atol=1e-15)
    np.testing.assert_allclose(pcorr(X), partialcorr(X), atol=1e-15)
    np.testing.assert_allclose(ccov(X, Y), conditionalcov(X, Y), atol=1e-15)
    np.testing.assert_allclose(ccorr(X, Y), conditionalcorr(X, Y), atol=1e-15)


# ---------------------------------------------------------------------------
# fourier
# ---------------------------------------------------------------------------


def test_analytic_signal_matches_scipy():
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.standard_normal(200))
    xa = analytic_signal(x)
    xa_ref = scipy.signal.hilbert(np.asarray(x))
    np.testing.assert_allclose(np.asarray(xa), xa_ref, atol=1e-13)


def test_analytic_signal_real_part_equals_input():
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.standard_normal(200))
    xa = analytic_signal(x)
    np.testing.assert_allclose(xa.real, x, atol=1e-13)


def test_analytic_signal_on_cosine_envelope_is_unity():
    fs = 100.0
    t = np.arange(200) / fs
    x = jnp.asarray(np.cos(2 * np.pi * 5 * t))
    env = jnp.abs(analytic_signal(x))
    # Interior only -- end taper from FFT.
    np.testing.assert_allclose(env[20:-20], 1.0, atol=1e-2)


def test_analytic_signal_raises_on_complex():
    x = jnp.asarray(np.zeros(100), dtype=jnp.complex64)
    with pytest.raises(TypeError, match='must be strictly real'):
        analytic_signal(x)


def test_hilbert_transform_of_cosine_is_sine():
    fs = 100.0
    t = np.arange(200) / fs
    x = jnp.asarray(np.cos(2 * np.pi * 5 * t))
    h = hilbert_transform(x)
    np.testing.assert_allclose(h[20:-20], np.sin(2 * np.pi * 5 * t)[20:-20], atol=1e-2)


def test_envelope_matches_abs_analytic():
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.standard_normal(200))
    np.testing.assert_allclose(
        envelope(x), jnp.abs(analytic_signal(x)), atol=1e-15,
    )


def test_instantaneous_frequency_of_5hz_cosine():
    fs = 100.0
    t = np.arange(500) / fs
    x = jnp.asarray(np.cos(2 * np.pi * 5 * t))
    f = instantaneous_frequency(x, fs=fs)
    # Interior should be ~5 Hz.
    np.testing.assert_allclose(f[50:-50], 5.0, atol=0.1)


def test_env_inst_matches_individual_calls():
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.standard_normal(200))
    e, f, p = env_inst(x, fs=100.0)
    np.testing.assert_allclose(e, envelope(x), atol=1e-13)
    np.testing.assert_allclose(
        p, instantaneous_phase(x), atol=1e-13,
    )
    np.testing.assert_allclose(
        f, instantaneous_frequency(x, fs=100.0), atol=1e-13,
    )


def test_product_filter_preserves_dc_for_unit_weight():
    x = jnp.asarray(np.ones(64))
    weight = jnp.ones(33)  # rfft of length 64 has 33 freqs
    out = product_filter(x, weight)
    np.testing.assert_allclose(out, x, atol=1e-13)


def test_product_filtfilt_zero_phase():
    '''Forward-backward filter has zero phase delay even with a
    complex weight.
    '''
    n = 128
    x = jnp.asarray(np.sin(2 * np.pi * np.arange(n) * 5 / n))
    # Complex weight with magnitude 1 but nonzero phase
    phase = jnp.linspace(0, math.pi, n // 2 + 1)
    weight = jnp.exp(1j * phase)
    out = product_filtfilt(x, weight)
    # Output should remain in phase with input (after some scaling).
    # Specifically, the imaginary part should be ~0 since the filter
    # itself is zero-phase.
    assert float(jnp.abs(out.imag).max()) < 1e-10
