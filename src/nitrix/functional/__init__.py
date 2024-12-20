# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
# isort: skip_file
from .covariance import (
    cov,
    corr,
    partialcov,
    partialcorr,
    pairedcov,
    pairedcorr,
    conditionalcov,
    conditionalcorr,
    corrcoef,
    pcorr,
    ccov,
    ccorr,
)
from .fourier import (
    product_filter,
    product_filtfilt,
    analytic_signal,
    hilbert_transform,
    envelope,
    instantaneous_frequency,
    instantaneous_phase,
    env_inst,
)
from .resid import residualise

__all__ = [
    'cov',
    'corr',
    'partialcov',
    'partialcorr',
    'pairedcov',
    'pairedcorr',
    'conditionalcov',
    'conditionalcorr',
    'corrcoef',
    'pcorr',
    'ccov',
    'ccorr',
    'product_filter',
    'product_filtfilt',
    'residualise',
]
