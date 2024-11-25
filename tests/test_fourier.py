# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for Fourier-domain filtering
"""
import pytest
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from importlib.resources import files
from pathlib import Path
from jax.numpy import unwrap
from scipy.fft import rfft, irfft
from scipy.signal import hilbert, chirp
from nitrix.functional import (
    product_filter,
    product_filtfilt,
    analytic_signal,
    hilbert_transform,
    envelope,
    instantaneous_frequency,
    instantaneous_phase,
    env_inst,
)


@pytest.fixture(scope='module')
def X(N: int = 100):
    return np.random.rand(7, N)

@pytest.fixture(scope='module')
def results():
    dir = Path(__file__).parent / 'artefacts' / 'fourier'
    dir.mkdir(parents=True, exist_ok=True)
    return dir

def scipy_product_filter(X, weight):
    return irfft(weight * rfft(X))

def uniform_attenuator(N: int = 100):
    return 0.5 * np.ones(N // 2 + 1)

def approx(out, ref, tol=1e-6):
    return np.allclose(out, ref, atol=tol)


def bandpass_filter(N: int = 100):
    weight = np.ones(N // 2 + 1)
    weight[:10] = 0
    weight[20:] = 0
    return weight


def test_bandpass(X):
    w = bandpass_filter()
    out = product_filter(X, w)
    ref = scipy_product_filter(X, w)
    assert approx(out, ref)


def test_zerophase_filter(X, N: int = 100):
    w = (
        np.random.rand(N // 2 + 1) +
        1j * np.random.rand(N // 2 + 1)
    )
    in_phase = jnp.angle(jnp.fft.rfft(X))
    out = product_filter(X, w)
    out_phase = jnp.angle(jnp.fft.rfft(out))
    assert not approx(in_phase, out_phase, tol=1e-5)

    out = product_filtfilt(X, w)
    out_phase = jnp.angle(jnp.fft.rfft(out))
    assert jnp.allclose(in_phase, out_phase, atol=1e-4, rtol=1e-3)


def test_attenuation(X):
    w = uniform_attenuator()
    out = product_filter(X, w)
    ref = 0.5 * X
    assert approx(out, ref)


def test_analytic_signal_hilbert(X):
    with pytest.raises(ValueError):
        analytic_signal(X * 1j)

    out = analytic_signal(X)
    ref = hilbert(X)
    assert approx(out, ref)
    assert approx(out.real, X, tol=1e-6)
    assert approx(out.imag, hilbert_transform(X), tol=1e-6)
    assert approx(out[0], analytic_signal(X[0]))

    X = np.random.randn(3, 10, 50, 5)
    ref = hilbert(X, axis=-2)
    gradient = jax.grad(lambda x: jnp.angle(analytic_signal(x)).sum())
    out = analytic_signal(X, -2)
    assert np.allclose(out, ref, atol=1e-6)
    assert np.allclose(X, out.real, atol=1e-6)
    X_grad = gradient(X)
    assert X_grad is not None


def test_hilbert_envelope(results):
    # replicating the example from the scipy documentation
    duration = 1.0
    fs = 400.0
    samples = int(fs * duration)
    t = np.arange(samples) / fs

    signal = chirp(t, 20.0, t[-1], 100.0)
    signal *= (1.0 + 0.5 * np.sin(2.0*np.pi*3.0*t) )

    amplitude_envelope = envelope(signal)
    inst_freq = instantaneous_frequency(signal, fs=400)

    fig, (ax0, ax1) = plt.subplots(nrows=2)

    ax0.plot(t, signal, label='signal')
    ax0.plot(t, amplitude_envelope, label='envelope')
    ax0.set_xlabel("time in seconds")
    ax0.legend()

    ax1.plot(t[1:], inst_freq)
    ax1.set_xlabel("time in seconds")
    ax1.set_ylim(0.0, 120.0)
    fig.tight_layout()

    fig.savefig(f'{results}/hilbert_separate.png')

    amplitude_envelope, inst_freq, _ = env_inst(signal, fs=400)

    fig, (ax0, ax1) = plt.subplots(nrows=2)

    ax0.plot(t, signal, label='signal')
    ax0.plot(t, amplitude_envelope, label='envelope')
    ax0.set_xlabel("time in seconds")
    ax0.legend()

    ax1.plot(t[1:], inst_freq)
    ax1.set_xlabel("time in seconds")
    ax1.set_ylim(0.0, 120.0)
    fig.tight_layout()

    fig.savefig(f'{results}/hilbert_onecall.png')


@pytest.mark.parametrize('n', [10, 100, 1000])
@pytest.mark.parametrize('axis', [1, 2])
@pytest.mark.parametrize(
    'fn',
    [
        analytic_signal,
        hilbert_transform,
        envelope,
        instantaneous_frequency,
        instantaneous_phase,
    ],
)
def test_shapes(fn, axis, n):
    X = np.random.randn(3, 101, 100)
    out = fn(X, axis=axis, n=n)
    ax_shape = 2 * (n // 2)
    if fn == instantaneous_frequency:
        ax_shape -= 1
    target_shape = [
        X.shape[i] if i != axis else ax_shape for i in range(X.ndim)
    ]
    assert tuple(out.shape) == tuple(target_shape)
