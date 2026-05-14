# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
# isort: skip_file
"""
nitrix.signal -- 1D signal-processing primitives.

Submodules:

- ``window``     -- windowed-sample extraction utilities.
- ``filter``     -- linear filtering: IIR / FIR / polynomial
  detrend / wavelet (where applicable).
- ``tsconv``     -- time-series convolution (1D conv with
  channel and batch handling matching the fMRI use case).
- ``interpolate`` -- 1D / N-D interpolation utilities.

This subpackage replaces ``nitrix.functional.window`` plus the
``hypercoil.functional.{tsconv,interpolate}`` ports.
"""
from .filter import polynomial_detrend
from .interpolate import linear_interpolate
from .lomb_scargle import (
    lomb_scargle_interpolate,
    lomb_scargle_periodogram,
)
from .tsconv import tsconv
from .window import sample_windows

__all__ = [
    'linear_interpolate',
    'lomb_scargle_interpolate',
    'lomb_scargle_periodogram',
    'polynomial_detrend',
    'sample_windows',
    'tsconv',
]
