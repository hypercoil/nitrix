# -*- coding: utf-8 -*-
"""Tests for the continuous wavelet transform (signal.cwt).

Checks the scalogram shape/dtype, that a pure tone localises at the scale whose
Morlet Fourier period matches the tone, that the real Ricker wavelet yields real
coefficients, and batching / jit / grad.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update('jax_enable_x64', True)

from nitrix.signal import cwt  # noqa: E402


def _tone(n=256, f0=0.05, dt=1.0):
    return jnp.asarray(np.cos(2 * np.pi * f0 * np.arange(n) * dt))


def test_cwt_shape_and_complex():
    x = _tone()
    scales = jnp.asarray(np.linspace(4, 40, 20))
    W = cwt(x, scales)
    assert W.shape == (20, 256)
    assert jnp.iscomplexobj(W)


def test_morlet_tone_localises_at_expected_scale():
    f0 = 0.05
    x = _tone(f0=f0)
    scales = jnp.asarray(np.linspace(4, 40, 40))
    power = np.asarray(jnp.mean(jnp.abs(cwt(x, scales, omega0=6.0)) ** 2, -1))
    peak = float(scales[power.argmax()])
    # Morlet Fourier period lambda = 4 pi s / (w0 + sqrt(2 + w0^2)); solve = 1/f0.
    expected = (1 / f0) * (6 + math.sqrt(2 + 36)) / (4 * math.pi)
    assert abs(peak - expected) < 2.0


def test_ricker_is_real():
    x = _tone()
    scales = jnp.asarray(np.linspace(4, 40, 15))
    W = cwt(x, scales, wavelet='ricker')
    assert float(jnp.max(jnp.abs(jnp.imag(W)))) < 1e-12


def test_paul_runs_and_is_finite():
    x = _tone()
    scales = jnp.asarray(np.linspace(4, 40, 15))
    W = cwt(x, scales, wavelet='paul', order=4)
    assert jnp.iscomplexobj(W)
    assert bool(jnp.all(jnp.isfinite(jnp.abs(W))))


def test_cwt_batches_over_leading_dims():
    X = jnp.stack([_tone(f0=0.05), _tone(f0=0.1)])  # (2, n)
    scales = jnp.asarray(np.linspace(4, 40, 20))
    W = cwt(X, scales)
    assert W.shape == (20, 2, 256)
    # each row's scalogram equals the single-signal transform
    for i in range(2):
        np.testing.assert_allclose(
            np.asarray(W[:, i]), np.asarray(cwt(X[i], scales)), atol=1e-10
        )


def test_cwt_invalid_wavelet_raises():
    with pytest.raises(ValueError):
        cwt(_tone(), jnp.asarray([4.0, 8.0]), wavelet='haar')


def test_cwt_jit_and_grad():
    x = _tone()
    scales = jnp.asarray(np.linspace(4, 40, 16))
    W = jax.jit(lambda x, s: cwt(x, s))(x, scales)
    assert bool(jnp.all(jnp.isfinite(jnp.abs(W))))
    g = jax.grad(lambda x: jnp.sum(jnp.abs(cwt(x, scales)) ** 2))(x)
    assert bool(jnp.all(jnp.isfinite(g)))
