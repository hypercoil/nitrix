# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
# isort: skip_file
"""
nitrix.signal -- 1D signal-processing primitives.

Submodules:

- ``window``     -- windowed-sample extraction utilities.
- ``filter``     -- linear filtering: polynomial detrend;
  zero-phase frequency-domain band-pass / band-stop / low- /
  high-pass (``bandpass`` / ``bandstop`` / ``lowpass`` /
  ``highpass``); and recursive Butterworth IIR (``iir_filter`` /
  ``butterworth_sos`` / ``sosfilt`` / ``sosfiltfilt``).
- ``fourier``    -- spectral utilities: product-filter,
  analytic-signal / Hilbert transform, instantaneous frequency /
  phase, envelope.
- ``tsconv``     -- time-series convolution (1D conv with
  channel and batch handling matching the fMRI use case).
- ``interpolate`` -- 1D / N-D interpolation utilities.

This subpackage replaces ``nitrix.functional.window`` plus the
``hypercoil.functional.{tsconv,interpolate}`` ports.
"""

from .filter import (
    bandpass,
    bandstop,
    butterworth_sos,
    highpass,
    iir_filter,
    lowpass,
    polynomial_detrend,
    sosfilt,
    sosfiltfilt,
)
from .fourier import (
    analytic_signal,
    env_inst,
    envelope,
    hilbert_transform,
    instantaneous_frequency,
    instantaneous_phase,
    product_filter,
    product_filtfilt,
)
from .interpolate import linear_interpolate
from .lomb_scargle import (
    lomb_scargle_interpolate,
    lomb_scargle_periodogram,
)
from .tsconv import tsconv
from .window import sample_windows

__all__ = [
    'analytic_signal',
    'bandpass',
    'bandstop',
    'butterworth_sos',
    'env_inst',
    'envelope',
    'highpass',
    'hilbert_transform',
    'iir_filter',
    'instantaneous_frequency',
    'instantaneous_phase',
    'linear_interpolate',
    'lomb_scargle_interpolate',
    'lomb_scargle_periodogram',
    'lowpass',
    'polynomial_detrend',
    'product_filter',
    'product_filtfilt',
    'sample_windows',
    'sosfilt',
    'sosfiltfilt',
    'tsconv',
]
