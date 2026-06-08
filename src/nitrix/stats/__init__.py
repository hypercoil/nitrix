# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
# isort: skip_file
"""
nitrix.stats -- statistical primitives.

Two submodules:

- ``covariance`` -- (partial) (paired) (conditional) covariance
  / correlation over time series.  JIT-friendly batch handling.
- ``fourier``    -- spectral utilities: product-filter,
  analytic-signal, Hilbert transform, instantaneous frequency /
  phase, envelope.
- ``gaussian``   -- closed-form diagonal-Gaussian KL divergence and
  negative log-likelihood (log-variance parameterised).

This subpackage replaces ``nitrix.functional.covariance`` and
``nitrix.functional.fourier``.
"""

from .covariance import (
    ccorr,
    ccov,
    conditionalcorr,
    conditionalcov,
    corr,
    corrcoef,
    cov,
    pairedcorr,
    pairedcov,
    partialcorr,
    partialcov,
    pcorr,
    precision,
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
from .gaussian import gaussian_nll, kl_diagonal_gaussian

__all__ = [
    # covariance
    'ccorr',
    'ccov',
    'conditionalcorr',
    'conditionalcov',
    'corr',
    'corrcoef',
    'cov',
    'pairedcorr',
    'pairedcov',
    'partialcorr',
    'partialcov',
    'pcorr',
    'precision',
    # fourier
    'analytic_signal',
    'env_inst',
    'envelope',
    'hilbert_transform',
    'instantaneous_frequency',
    'instantaneous_phase',
    'product_filter',
    'product_filtfilt',
    # gaussian
    'kl_diagonal_gaussian',
    'gaussian_nll',
]
