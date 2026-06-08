# -*- coding: utf-8 -*-
"""Tests for ``nitrix.signal`` recursive Butterworth IIR filtering.

Coverage:

- **design parity**: ``butterworth_sos`` transfer function matches
  ``scipy.signal.butter(output='sos')`` to machine precision, all band
  types and several orders (the pure-NumPy design reproduces scipy's maths).
- **apply parity**: forward ``sosfilt`` == ``scipy.signal.sosfilt`` (zero
  state); ``sosfiltfilt`` == ``scipy.signal.sosfiltfilt`` (zi + odd pad).
- **backend parity**: ``'scan'`` and ``'associative'`` agree.
- zero phase (no lag) for filtfilt; causal pass has a real phase delay.
- selectivity (band-pass keeps / band-stop notches); jit + grad; argument
  validation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update('jax_enable_x64', True)

ss = pytest.importorskip('scipy.signal')

from nitrix.signal import (  # noqa: E402
    butterworth_sos,
    iir_filter,
    sosfilt,
    sosfiltfilt,
)

FS = 0.5  # Hz (TR = 2 s)

_CASES = [
    ('lowpass', 4, None, 0.2),
    ('lowpass', 5, None, 0.1),
    ('highpass', 4, 0.15, None),
    ('highpass', 3, 0.05, None),
    ('bandpass', 4, 0.05, 0.2),
    ('bandpass', 6, 0.02, 0.1),
    ('bandstop', 4, 0.05, 0.2),
    ('bandstop', 2, 0.08, 0.12),
]


def _scipy_wn(btype, lo, hi):
    if btype in ('bandpass', 'bandstop'):
        return [lo, hi]
    return hi if btype == 'lowpass' else lo


def _mag_response(sos, w):
    z = np.exp(1j * w)
    h = np.ones_like(z)
    for b0, b1, b2, a0, a1, a2 in np.asarray(sos):
        h = h * (b0 + b1 / z + b2 / z**2) / (a0 + a1 / z + a2 / z**2)
    return np.abs(h)


# ---------------------------------------------------------------------------
# Design
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('btype, order, lo, hi', _CASES)
def test_design_matches_scipy(btype, order, lo, hi):
    mine = butterworth_sos(order=order, fs=FS, btype=btype, lo=lo, hi=hi)
    ref = ss.butter(
        order, _scipy_wn(btype, lo, hi), btype=btype, output='sos', fs=FS
    )
    assert mine.shape == ref.shape
    w = np.linspace(1e-3, np.pi - 1e-3, 400)
    np.testing.assert_allclose(
        _mag_response(mine, w), _mag_response(ref, w), atol=1e-10
    )


# ---------------------------------------------------------------------------
# Application parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('btype, order, lo, hi', _CASES)
def test_sosfilt_matches_scipy(btype, order, lo, hi):
    x = np.random.default_rng(0).normal(size=(4, 400))
    sos = butterworth_sos(order=order, fs=FS, btype=btype, lo=lo, hi=hi)
    mine = np.asarray(sosfilt(jnp.asarray(x), sos))
    ref = ss.sosfilt(sos.copy(), x, axis=-1)
    np.testing.assert_allclose(mine, ref, atol=1e-9)


@pytest.mark.parametrize('btype, order, lo, hi', _CASES)
def test_sosfiltfilt_matches_scipy(btype, order, lo, hi):
    x = np.random.default_rng(1).normal(size=(4, 400))
    sos = butterworth_sos(order=order, fs=FS, btype=btype, lo=lo, hi=hi)
    mine = np.asarray(sosfiltfilt(jnp.asarray(x), sos))
    ref = ss.sosfiltfilt(sos.copy(), x, axis=-1)
    np.testing.assert_allclose(mine, ref, atol=1e-9)


def test_backend_parity():
    x = jnp.asarray(np.random.default_rng(2).normal(size=(6, 500)))
    sos = butterworth_sos(order=4, fs=FS, btype='bandpass', lo=0.05, hi=0.2)
    a = np.asarray(sosfilt(x, sos, backend='scan'))
    b = np.asarray(sosfilt(x, sos, backend='associative'))
    np.testing.assert_allclose(a, b, atol=1e-9)


# ---------------------------------------------------------------------------
# Phase behaviour
# ---------------------------------------------------------------------------


def _tone(freq, n=512, fs=FS):
    return np.sin(2 * np.pi * freq * np.arange(n) / fs)


def test_filtfilt_zero_phase_causal_has_delay():
    # Use a low-pass with an in-band tone: its phase response has a clear
    # (monotonic) group delay, unlike a band-pass at its centre frequency
    # where the phase passes through ~0.
    tone = _tone(0.05)
    zp = np.asarray(
        iir_filter(
            jnp.asarray(tone), fs=FS, btype='lowpass', hi=0.12, zero_phase=True
        )
    )
    causal = np.asarray(
        iir_filter(
            jnp.asarray(tone),
            fs=FS,
            btype='lowpass',
            hi=0.12,
            zero_phase=False,
        )
    )

    def lag(y):
        a, b = tone[80:-80], y[80:-80]
        return int(np.argmax(np.correlate(b, a, 'full'))) - (len(a) - 1)

    assert lag(zp) == 0  # zero-phase: no delay
    assert lag(causal) > 0  # causal Butterworth: real group delay


# ---------------------------------------------------------------------------
# Selectivity
# ---------------------------------------------------------------------------


def _power_at(sig, freq, fs=FS):
    sig = np.asarray(sig)
    fr = np.fft.rfftfreq(sig.shape[-1], d=1 / fs)
    return float(np.abs(np.fft.rfft(sig)[np.argmin(np.abs(fr - freq))]))


def test_bandpass_selectivity():
    x = _tone(0.02) + _tone(0.08) + _tone(0.20)
    y = iir_filter(
        jnp.asarray(x), fs=FS, btype='bandpass', lo=0.05, hi=0.12, order=4
    )
    assert _power_at(y, 0.08) / _power_at(x, 0.08) > 0.9
    assert _power_at(y, 0.02) / _power_at(x, 0.02) < 0.05
    assert _power_at(y, 0.20) / _power_at(x, 0.20) < 0.05


def test_bandstop_notch():
    x = _tone(0.02) + _tone(0.08) + _tone(0.20)
    y = iir_filter(
        jnp.asarray(x), fs=FS, btype='bandstop', lo=0.05, hi=0.12, order=4
    )
    assert _power_at(y, 0.08) / _power_at(x, 0.08) < 0.1
    assert _power_at(y, 0.02) / _power_at(x, 0.02) > 0.95
    assert _power_at(y, 0.20) / _power_at(x, 0.20) > 0.95


# ---------------------------------------------------------------------------
# Transforms + validation
# ---------------------------------------------------------------------------


def test_jit_and_grad():
    x = jnp.asarray(_tone(0.08))
    fn = jax.jit(
        lambda z: iir_filter(z, fs=FS, btype='bandpass', lo=0.05, hi=0.12)
    )
    a = np.asarray(iir_filter(x, fs=FS, btype='bandpass', lo=0.05, hi=0.12))
    np.testing.assert_allclose(a, np.asarray(fn(x)), atol=1e-9)
    g = jax.grad(
        lambda z: jnp.sum(
            iir_filter(z, fs=FS, btype='bandpass', lo=0.05, hi=0.12) ** 2
        )
    )(x)
    assert bool(np.all(np.isfinite(np.array(g))))


def test_grad_associative_backend():
    x = jnp.asarray(_tone(0.08))
    g = jax.grad(
        lambda z: jnp.sum(
            iir_filter(
                z,
                fs=FS,
                btype='lowpass',
                hi=0.1,
                zero_phase=False,
                backend='associative',
            )
            ** 2
        )
    )(x)
    assert bool(np.all(np.isfinite(np.array(g))))


@pytest.mark.parametrize('btype, order, lo, hi', _CASES)
def test_fft_backend_matches_scipy(btype, order, lo, hi):
    # The FFT-convolution engine (default on GPU): an IIR filter is LTI, so
    # its output is exactly convolution with the (truncated) impulse response.
    x = np.random.default_rng(3).normal(size=(4, 400))
    sos = butterworth_sos(order=order, fs=FS, btype=btype, lo=lo, hi=hi)
    sf = np.asarray(sosfilt(jnp.asarray(x), sos, backend='fft'))
    ff = np.asarray(sosfiltfilt(jnp.asarray(x), sos, backend='fft'))
    np.testing.assert_allclose(
        sf, ss.sosfilt(sos.copy(), x, axis=-1), atol=1e-9
    )
    np.testing.assert_allclose(
        ff,
        ss.sosfiltfilt(sos.copy(), x, axis=-1),
        atol=1e-9,
    )


def test_fft_grad_finite():
    x = jnp.asarray(_tone(0.08))
    g = jax.grad(
        lambda z: jnp.sum(
            sosfiltfilt(
                z,
                butterworth_sos(
                    order=4, fs=FS, btype='bandpass', lo=0.05, hi=0.2
                ),
                backend='fft',
            )
        )
    )(x)
    assert bool(np.all(np.isfinite(np.array(g))))


def test_fft_impulse_atol_param():
    # A looser truncation tolerance shortens the kernel but stays accurate to
    # roughly that tolerance (geometrically bounded truncation error).
    x = np.random.default_rng(4).normal(size=(2, 600))
    sos = butterworth_sos(order=4, fs=FS, btype='bandpass', lo=0.05, hi=0.2)
    ref = ss.sosfilt(sos.copy(), x, axis=-1)
    loose = np.asarray(
        sosfilt(jnp.asarray(x), sos, backend='fft', impulse_atol=1e-6)
    )
    np.testing.assert_allclose(loose, ref, atol=1e-4)


def test_fft_too_sharp_falls_back():
    # A razor-thin high-order band-pass whose impulse response does not decay
    # within the tap cap must fall back to a recurrence (with a warning) and
    # stay correct.
    x = np.random.default_rng(5).normal(size=(2, 400))
    sos = butterworth_sos(
        order=8, fs=FS, btype='bandpass', lo=0.0004, hi=0.0008
    )
    with pytest.warns(UserWarning, match='falling back'):
        out = np.asarray(sosfilt(jnp.asarray(x), sos, backend='fft'))
    np.testing.assert_allclose(
        out, ss.sosfilt(sos.copy(), x, axis=-1), atol=1e-8
    )


def test_validation():
    x = jnp.asarray(_tone(0.08))
    with pytest.raises(ValueError):
        iir_filter(x, fs=FS, btype='bandpass', lo=0.2, hi=0.05)  # lo >= hi
    with pytest.raises(ValueError):
        iir_filter(x, fs=FS, btype='lowpass', hi=0.3)  # > Nyquist
    with pytest.raises(ValueError):
        iir_filter(x, fs=FS, btype='bandpass', lo=0.05)  # hi missing
    with pytest.raises(ValueError):
        butterworth_sos(order=4, fs=FS, btype='nope', lo=0.05, hi=0.1)
    with pytest.raises(ValueError):
        sosfilt(
            x,
            butterworth_sos(order=2, fs=FS, btype='lowpass', hi=0.1),
            backend='nope',
        )
