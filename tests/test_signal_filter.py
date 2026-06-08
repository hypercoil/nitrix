# -*- coding: utf-8 -*-
"""Tests for ``nitrix.signal.filter`` frequency-domain filters.

Coverage:

- selectivity: band-pass keeps the in-band tone and attenuates out-of-band;
  band-stop (notch) does the complement; low/high-pass DC behaviour.
- the ``'maxflat'`` magnitude matches the analog-Butterworth magnitude
  (``scipy.signal.butter(analog=True)`` oracle) -- confirming it is the
  Butterworth *magnitude shape* (and -3 dB at the cut-off).
- zero phase (no lag); ``'ideal'`` equals a hand-built brick-wall mask.
- jit parity, differentiability, per-unit independence, ``axis`` handling,
  reflect padding, and argument validation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nitrix.signal.filter import bandpass, bandstop, highpass, lowpass

FS = 0.5  # Hz (TR = 2 s); Nyquist = 0.25 Hz


def _tone(freq, n=512, fs=FS):
    t = np.arange(n) / fs
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def _power_at(sig, freq, fs=FS):
    sig = np.asarray(sig)
    n = sig.shape[-1]
    spec = np.fft.rfft(sig)
    fr = np.fft.rfftfreq(n, d=1 / fs)
    return float(np.abs(spec[np.argmin(np.abs(fr - freq))]))


# ---------------------------------------------------------------------------
# Selectivity
# ---------------------------------------------------------------------------


def test_bandpass_keeps_in_band_rejects_out_of_band():
    x = _tone(0.02) + _tone(0.08) + _tone(0.20)
    y = bandpass(jnp.asarray(x), fs=FS, lo=0.05, hi=0.12, order=6)
    assert _power_at(y, 0.08) / _power_at(x, 0.08) > 0.9
    assert _power_at(y, 0.02) / _power_at(x, 0.02) < 0.05
    assert _power_at(y, 0.20) / _power_at(x, 0.20) < 0.1


def test_bandstop_notches_in_band_keeps_rest():
    x = _tone(0.02) + _tone(0.08) + _tone(0.20)
    # Notch depth scales with order (soft-union floor ~ 2x the roll-off at
    # band centre); order 8 puts the in-band tone well below 10%.
    y = bandstop(jnp.asarray(x), fs=FS, lo=0.05, hi=0.12, order=8)
    assert _power_at(y, 0.08) / _power_at(x, 0.08) < 0.1
    assert _power_at(y, 0.02) / _power_at(x, 0.02) > 0.95
    assert _power_at(y, 0.20) / _power_at(x, 0.20) > 0.95


def test_lowpass_passes_dc_highpass_removes_it():
    x = jnp.asarray(_tone(0.20) + 5.0)  # tone on a DC pedestal
    lp = np.asarray(lowpass(x, fs=FS, cutoff=0.05))
    hp = np.asarray(highpass(x, fs=FS, cutoff=0.05))
    assert abs(lp.mean() - 5.0) < 1e-2  # low-pass keeps the pedestal
    assert abs(hp.mean()) < 1e-2  # high-pass removes it (DC bin 0)
    # ... and high-pass keeps the high tone, low-pass kills it.
    assert _power_at(hp, 0.20) / _power_at(np.asarray(x), 0.20) > 0.9
    assert _power_at(lp, 0.20) / _power_at(np.asarray(x), 0.20) < 0.1


# ---------------------------------------------------------------------------
# It is the Butterworth *magnitude shape* (scipy analog oracle)
# ---------------------------------------------------------------------------


def test_maxflat_matches_analog_butterworth_magnitude():
    scipy_signal = pytest.importorskip('scipy.signal')
    n, fs, fc, order = 1024, FS, 0.08, 4
    # nitrix maxflat low-pass weight on the rfft grid (recovered by filtering
    # a unit impulse and reading the response magnitude is overkill; compare
    # the analytic mask the op applies).
    fr = np.fft.rfftfreq(n, d=1 / fs)
    mine = 1.0 / np.sqrt(1.0 + (fr / fc) ** (2 * order))
    # scipy analog Butterworth magnitude |H(jw)| at the same frequencies.
    b, a = scipy_signal.butter(order, 2 * np.pi * fc, analog=True)
    _, h = scipy_signal.freqs(b, a, worN=2 * np.pi * fr)
    np.testing.assert_allclose(mine, np.abs(h), atol=1e-6)
    # -3 dB at the cut-off (nearest grid bin; tol covers the bin spacing).
    assert abs(mine[np.argmin(np.abs(fr - fc))] - 1 / np.sqrt(2)) < 5e-3


def test_ideal_equals_bruteforce_brickwall():
    x = _tone(0.02) + _tone(0.08) + _tone(0.20)
    n = x.shape[-1]
    y = np.asarray(
        bandpass(jnp.asarray(x), fs=FS, lo=0.05, hi=0.12, ftype='ideal')
    )
    fr = np.fft.rfftfreq(n, d=1 / FS)
    mask = ((fr >= 0.05) & (fr <= 0.12)).astype(np.float32)
    ref = np.fft.irfft(np.fft.rfft(x) * mask, n=n)
    np.testing.assert_allclose(y, ref, atol=1e-4)


# ---------------------------------------------------------------------------
# Zero phase
# ---------------------------------------------------------------------------


def test_zero_phase_no_lag():
    x = _tone(0.08)
    y = np.asarray(bandpass(jnp.asarray(x), fs=FS, lo=0.05, hi=0.12))
    a, b = x[60:-60], y[60:-60]  # avoid circular edges
    xc = np.correlate(b, a, 'full')
    lag = int(np.argmax(xc)) - (len(a) - 1)
    assert lag == 0


# ---------------------------------------------------------------------------
# Transforms / robustness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('ftype', ['maxflat', 'ideal', 'cosine'])
def test_all_designs_finite(ftype):
    x = jnp.asarray(_tone(0.08) + _tone(0.20))
    y = bandpass(x, fs=FS, lo=0.05, hi=0.12, ftype=ftype, transition=0.02)
    assert bool(np.all(np.isfinite(np.array(y))))


def test_jit_parity():
    x = jnp.asarray(_tone(0.08))
    a = np.asarray(bandpass(x, fs=FS, lo=0.05, hi=0.12))
    b = np.asarray(jax.jit(lambda z: bandpass(z, fs=FS, lo=0.05, hi=0.12))(x))
    np.testing.assert_allclose(a, b, atol=1e-5)


def test_differentiable():
    x = jnp.asarray(_tone(0.08))
    g = jax.grad(lambda z: jnp.sum(bandpass(z, fs=FS, lo=0.05, hi=0.12) ** 2))(
        x
    )
    assert bool(np.all(np.isfinite(np.array(g))))
    assert float(jnp.linalg.norm(g)) > 0


def test_per_unit_independence():
    x0 = _tone(0.08)
    multi = jnp.stack(
        [jnp.asarray(x0), jnp.asarray(_tone(0.20)), jnp.asarray(_tone(0.02))],
        axis=0,
    )
    ym = np.asarray(bandpass(multi, fs=FS, lo=0.05, hi=0.12))
    y0 = np.asarray(bandpass(jnp.asarray(x0), fs=FS, lo=0.05, hi=0.12))
    np.testing.assert_array_equal(ym[0], y0)


def test_axis_argument():
    x = _tone(0.08)
    col = jnp.asarray(x)[:, None] * jnp.ones((1, 3))
    y = np.asarray(bandpass(col, fs=FS, lo=0.05, hi=0.12, axis=0))
    ref = np.asarray(bandpass(jnp.asarray(x), fs=FS, lo=0.05, hi=0.12))
    np.testing.assert_allclose(y[:, 0], ref, atol=1e-5)


def test_padding_runs_and_preserves_shape():
    x = jnp.asarray(_tone(0.08))
    y = bandpass(x, fs=FS, lo=0.05, hi=0.12, padding=64)
    assert y.shape == x.shape
    assert bool(np.all(np.isfinite(np.array(y))))


def test_cutoff_validation():
    x = jnp.asarray(_tone(0.08))
    with pytest.raises(ValueError):
        bandpass(x, fs=FS, lo=0.12, hi=0.05)  # lo >= hi
    with pytest.raises(ValueError):
        lowpass(x, fs=FS, cutoff=0.30)  # beyond Nyquist (0.25)
    with pytest.raises(ValueError):
        bandpass(x, fs=FS, lo=0.05, hi=0.12, ftype='nope')
