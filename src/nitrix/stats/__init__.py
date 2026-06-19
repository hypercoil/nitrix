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
- ``connectivity`` -- regularised connectome estimators: analytic-shrinkage
  covariance (Ledoit-Wolf / OAS) and sparse precision (graphical LASSO) for the
  small-sample regime.
- ``lme``        -- voxelwise linear mixed-effects (REML / FLAME).
- ``inference``  -- permutation / TFCE cluster inference (the on-device FSL
  ``randomise`` engine) + FDR / Bonferroni.
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
from .basis import (
    REBasis,
    SplineBasis,
    TensorBasis,
    bspline_basis,
    by_factor_smooth,
    cyclic_cubic_basis,
    re_smooth,
    spline_design,
    tensor_product_basis,
    tensor_product_design,
    varying_coefficient_smooth,
    thinplate_regression_basis,
)
from .connectivity import (
    ebic_score,
    glasso,
    glasso_path,
    ledoit_wolf,
    oas,
    shrunk_covariance,
)
from .gam import GAMResult, gam_fit, smooth_partial_effect
from ._betareg import BetaResult, beta_fit
from .glmm import GLMMResult, glmm_fit
from .gaussian import gaussian_nll, kl_diagonal_gaussian
from .glm import (
    BINOMIAL,
    GAMMA,
    GAUSSIAN,
    NEGBINOMIAL,
    POISSON,
    TWEEDIE,
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
    negbinomial,
    predict,
    r_squared,
    sandwich_cov,
    t_contrast,
    tweedie,
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
    'GAMMA',
    'NEGBINOMIAL',
    'TWEEDIE',
    'negbinomial',
    'tweedie',
    'GLMResult',
    'glm_fit',
    'predict',
    't_contrast',
    'f_contrast',
    'sandwich_cov',
    'r_squared',
    'adj_r_squared',
    'deviance_explained',
    'log_likelihood',
    'aic',
    'bic',
    'compare_models',
    # basis
    'SplineBasis',
    'TensorBasis',
    'REBasis',
    'bspline_basis',
    'by_factor_smooth',
    'varying_coefficient_smooth',
    'cyclic_cubic_basis',
    'thinplate_regression_basis',
    'tensor_product_basis',
    're_smooth',
    'spline_design',
    'tensor_product_design',
    # gam
    'GAMResult',
    'gam_fit',
    'smooth_partial_effect',
    # glmm
    'GLMMResult',
    'glmm_fit',
    # beta regression
    'BetaResult',
    'beta_fit',
    # connectivity
    'ledoit_wolf',
    'oas',
    'shrunk_covariance',
    'glasso',
    'glasso_path',
    'ebic_score',
    # pca
    'PCAResult',
    'pca_fit',
    'pca_transform',
    'pca_inverse_transform',
]
