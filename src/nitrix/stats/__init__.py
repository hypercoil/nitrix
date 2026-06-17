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
- ``glm``        -- mass-univariate generalised linear models (OLS / WLS /
  exponential-family IRLS) with t / F contrasts and goodness-of-fit.
- ``basis``      -- penalised spline bases (P-splines) for additive models.
- ``gam``        -- mass-univariate generalised additive (mixed) models with
  REML / Fellner-Schall smoothing-parameter selection.
- ``lme``        -- voxelwise linear mixed-effects (REML / FLAME).
- ``pca``        -- principal-component analysis (fit / transform /
  inverse) via the covariance eigendecomposition.

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
from .basis import SplineBasis, bspline_basis, spline_design
from .gam import GAMResult, gam_fit, smooth_partial_effect
from .gaussian import gaussian_nll, kl_diagonal_gaussian
from .glm import (
    BINOMIAL,
    GAUSSIAN,
    POISSON,
    Family,
    GLMResult,
    adj_r_squared,
    aic,
    bic,
    compare_models,
    deviance_explained,
    f_contrast,
    glm_fit,
    log_likelihood,
    predict,
    r_squared,
    t_contrast,
)
from .pca import (
    PCAResult,
    pca_fit,
    pca_inverse_transform,
    pca_transform,
)

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
    # glm
    'Family',
    'GAUSSIAN',
    'BINOMIAL',
    'POISSON',
    'GLMResult',
    'glm_fit',
    'predict',
    't_contrast',
    'f_contrast',
    'r_squared',
    'adj_r_squared',
    'deviance_explained',
    'log_likelihood',
    'aic',
    'bic',
    'compare_models',
    # basis
    'SplineBasis',
    'bspline_basis',
    'spline_design',
    # gam
    'GAMResult',
    'gam_fit',
    'smooth_partial_effect',
    # pca
    'PCAResult',
    'pca_fit',
    'pca_transform',
    'pca_inverse_transform',
]
