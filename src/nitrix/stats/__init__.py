# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
# isort: skip_file
"""
Mass-univariate statistical modelling for neuroimaging.

A collection of differentiable estimators for fitting statistical models
independently at every vertex, voxel, or edge of an image or connectome, spanning
covariance and correlation, (generalised, additive, mixed) linear models,
Gaussian-process smooths, regularised connectivity, and permutation inference.

Submodules
----------
- ``covariance`` -- (partial) (paired) (conditional) covariance
  / correlation over time series, with JIT-friendly batch handling.
- ``gaussian`` -- closed-form diagonal-Gaussian KL divergence and
  negative log-likelihood (log-variance parameterised).
- ``directional`` -- von Mises-Fisher / Watson / Kent directional statistics
  on the sphere: normalisers (:func:`log_iv` / :func:`log_kummer_m` /
  :func:`log_kent_normaliser`), densities and MLE fits, the general
  :func:`fisher_bingham_energy` (unnormalised, any dimension), and samplers
  (:func:`vmf_sample`, and :func:`watson_sample` / :func:`bingham_sample` by
  Angular-Central-Gaussian rejection).
- ``glm`` -- mass-univariate generalised linear models (OLS / WLS /
  exponential-family IRLS) with t / F contrasts and goodness-of-fit.
- ``robust`` -- outlier-resistant M-estimator regression (Huber, Tukey
  bisquare) by IRLS, and the median-absolute-deviation scale estimator.
- ``betareg`` / ``ordinal`` / ``gaulss`` -- additional response models: beta
  regression (rates / proportions), ordinal (cumulative-link), and the Gaussian
  location-scale (mean + variance) GAMLSS fit.
- ``basis`` -- penalised spline bases (P-splines) for additive models,
  plus the :func:`hsgp_basis` Hilbert-space approximate-GP smooth (reduced-rank
  Gaussian process) and the kriging :func:`gp_basis`.
- ``gam`` -- mass-univariate generalised additive (mixed) models with
  REML / Fellner-Schall smoothing-parameter selection.
- ``gp`` -- mass-univariate Gaussian-process regression (HSGP or exact
  engine) with REML-estimated kernel lengthscale and optional ``corr=``
  structured residual (:func:`gp_fit` / :func:`gp_predict`).
- ``hgp`` -- hierarchical / multi-level GP: a population smooth plus
  group-level smooth deviations sharing a kernel
  (:func:`hgp_fit` / :func:`hgp_predict`).
- ``priors`` -- lengthscale MAP-rho regularisers (half-normal / inverse-gamma
  / log-normal) for the :func:`gp_fit` / :func:`hgp_fit` kernel-range search.
- ``connectivity`` -- regularised connectome estimators: analytic-shrinkage
  covariance (Ledoit-Wolf / OAS) and sparse precision (graphical LASSO) for the
  small-sample regime.
- ``lme`` -- voxelwise linear mixed-effects (REML / FLAME).
- ``glmm`` -- generalised linear mixed models (PQL / Laplace / adaptive
  Gauss-Hermite) for non-Gaussian responses with random effects.
- ``inference`` -- permutation / TFCE cluster inference (the on-device FSL
  ``randomise`` engine) plus FDR / Bonferroni.
- ``pca`` -- principal-component analysis (fit / transform /
  inverse) via the covariance eigendecomposition.
- ``whitening`` -- zero-phase (ZCA) whitening on the fit/apply seam,
  with the cuSOLVER-free Newton-Schulz inverse square root.
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
from .basis import (
    REBasis,
    SplineBasis,
    TensorBasis,
    bspline_basis,
    by_factor_smooth,
    cr_basis,
    cyclic_cubic_basis,
    gp_basis,
    gp_factor_smooth,
    hsgp_basis,
    hsgp_basis_nd,
    mrf_smooth,
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
from .gam import (
    GAMResult,
    SmoothTest,
    gam_fit,
    gam_predict,
    smooth_partial_effect,
    smooth_significance,
)
from .gp import GPResult, gp_aic, gp_bic, gp_fit, gp_predict
from .hgp import HGPResult, hgp_fit, hgp_predict
from .priors import PriorFn, halfnormal_prior, invgamma_prior, lognormal_prior
from .betareg import BetaResult, beta_fit, beta_predict
from .gaulss import GauLSSResult, gaulss_fit, gaulss_predict
from .ordinal import OrdinalResult, ordinal_fit, ordinal_predict
from .glmm import GLMMResult, glmm_fit, glmm_predict
from .lme import (
    CorrLMEResult,
    CorrSpec,
    CrossedLMEResult,
    FLAMEResult,
    GLSResult,
    LMEContrast,
    LMEFContrast,
    LMEResult,
    NestedLMEResult,
    REMLResult,
    VarFunc,
    ar1,
    car1,
    cs,
    flame_two_level,
    gls_fit,
    iid,
    lme_f_contrast,
    lme_fit,
    lme_predict,
    lme_t_contrast,
    ranef,
    reml_fit,
    var_ident,
    var_power,
)
from .directional import (
    KentFit,
    VMFFit,
    WatsonFit,
    bingham_sample,
    fisher_bingham_energy,
    kent_fit,
    kent_log_prob,
    log_iv,
    log_kent_normaliser,
    log_kummer_m,
    vmf_fit,
    vmf_log_prob,
    vmf_sample,
    watson_fit,
    watson_log_prob,
    watson_sample,
)
from .gaussian import gaussian_nll, kl_diagonal_gaussian
from .glm import (
    BINOMIAL,
    CLOGLOG_LINK,
    GAMMA,
    GAUSSIAN,
    IDENTITY_LINK,
    INVERSE_LINK,
    LOG_LINK,
    LOGIT_LINK,
    NEGBINOMIAL,
    POISSON,
    PROBIT_LINK,
    SQRT_LINK,
    TWEEDIE,
    Family,
    GLMResult,
    Link,
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
    resolve_family,
    resolve_link,
    sandwich_cov,
    t_contrast,
    tweedie,
)
from ._effect import confidence_interval, standardized_effect
from .pca import (
    PCAResult,
    pca_fit,
    pca_inverse_transform,
    pca_transform,
)
from .whitening import (
    WhiteningState,
    whiten,
    whiten_apply,
    whiten_fit,
    whiten_inverse_apply,
)
from .robust import (
    RobustFit,
    huber_regress,
    mad,
    tukey_bisquare_regress,
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
    # gaussian
    'kl_diagonal_gaussian',
    'gaussian_nll',
    # directional (von Mises-Fisher + Watson)
    'log_iv',
    'vmf_log_prob',
    'vmf_fit',
    'vmf_sample',
    'VMFFit',
    'log_kummer_m',
    'watson_log_prob',
    'watson_fit',
    'WatsonFit',
    'log_kent_normaliser',
    'kent_log_prob',
    'kent_fit',
    'KentFit',
    'fisher_bingham_energy',
    'watson_sample',
    'bingham_sample',
    # glm
    'Family',
    'Link',
    'IDENTITY_LINK',
    'LOG_LINK',
    'LOGIT_LINK',
    'PROBIT_LINK',
    'CLOGLOG_LINK',
    'SQRT_LINK',
    'INVERSE_LINK',
    'resolve_family',
    'resolve_link',
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
    # effect size / intervals
    'confidence_interval',
    'standardized_effect',
    # basis
    'SplineBasis',
    'TensorBasis',
    'REBasis',
    'bspline_basis',
    'by_factor_smooth',
    'varying_coefficient_smooth',
    'cyclic_cubic_basis',
    'thinplate_regression_basis',
    'cr_basis',
    'gp_basis',
    'hsgp_basis',
    'hsgp_basis_nd',
    'gp_factor_smooth',
    'mrf_smooth',
    'tensor_product_basis',
    're_smooth',
    'spline_design',
    'tensor_product_design',
    # gam
    'GAMResult',
    'SmoothTest',
    'gam_fit',
    'gam_predict',
    'smooth_partial_effect',
    'smooth_significance',
    # gp
    'GPResult',
    'gp_fit',
    'gp_predict',
    'gp_aic',
    'gp_bic',
    # hgp
    'HGPResult',
    'hgp_fit',
    'hgp_predict',
    # priors (map_rho regularisers)
    'PriorFn',
    'halfnormal_prior',
    'invgamma_prior',
    'lognormal_prior',
    # glmm
    'GLMMResult',
    'glmm_fit',
    'glmm_predict',
    'ranef',
    # lme (mixed-effects: also reachable via nitrix.stats.lme, per the sibling
    # fitter convention -- glm / gam / gp / glmm are all top-level too)
    'reml_fit',
    'lme_fit',
    'lme_predict',
    'lme_t_contrast',
    'lme_f_contrast',
    'flame_two_level',
    'gls_fit',
    'REMLResult',
    'LMEResult',
    'NestedLMEResult',
    'CrossedLMEResult',
    'CorrLMEResult',
    'FLAMEResult',
    'GLSResult',
    'LMEContrast',
    'LMEFContrast',
    'CorrSpec',
    'VarFunc',
    'ar1',
    'car1',
    'cs',
    'iid',
    'var_power',
    'var_ident',
    # beta regression
    'BetaResult',
    'beta_fit',
    'beta_predict',
    # distributional (location-scale)
    'GauLSSResult',
    'gaulss_fit',
    'gaulss_predict',
    # ordinal (cumulative link)
    'OrdinalResult',
    'ordinal_fit',
    'ordinal_predict',
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
    # whitening (ZCA)
    'WhiteningState',
    'whiten',
    'whiten_fit',
    'whiten_apply',
    'whiten_inverse_apply',
    # robust M-estimation
    'mad',
    'huber_regress',
    'tukey_bisquare_regress',
    'RobustFit',
]
