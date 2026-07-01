# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Gaussian location-scale regression (``gaulss``).

The standard generalised linear model has a single linear predictor (the mean);
the residual scale is a single nuisance dispersion.  A *distributional* model
gives the scale its own linear predictor -- here
:math:`y_i \\sim \\mathcal{N}(\\mu_i, \\sigma_i^2)` with

.. math::

   \\mu_i = x_i^{\\top} \\beta_{\\mu} \\quad \\text{(identity link, the mean model)}

   \\log \\sigma_i = z_i^{\\top} \\beta_{\\sigma} \\quad \\text{(log link, the scale model)}

so heteroscedasticity is modelled directly (as in ``mgcv``'s ``gaulss`` family).
This is not the scalar IRLS core (one predictor, fixed dispersion), so it gets
its own fitter.

The estimator is joint maximum likelihood by Fisher scoring.  The Gaussian
location-scale expected information is block-diagonal between :math:`\\beta_{\\mu}`
and :math:`\\beta_{\\sigma}` (since :math:`\\mathbb{E}[y - \\mu] = 0` kills the
cross term), so each scoring step is two decoupled solves:

.. math::

   \\beta_{\\mu} \\leftarrow (X^{\\top} W X)^{-1} X^{\\top} W y, \\quad W = \\operatorname{diag}(1 / \\sigma_i^2) \\quad \\text{(WLS)}

   \\beta_{\\sigma} \\leftarrow \\beta_{\\sigma} + (2 Z^{\\top} Z)^{-1} Z^{\\top} u, \\quad u_i = -1 + (y_i - \\mu_i)^2 / \\sigma_i^2

iterated to a fixed budget (cleanly vectorised over elements).  The coefficient
covariances are the two information blocks' inverses:
:math:`\\operatorname{Cov}(\\beta_{\\mu}) = (X^{\\top} \\operatorname{diag}(1/\\sigma^2) X)^{-1}`
and :math:`\\operatorname{Cov}(\\beta_{\\sigma}) = (2 Z^{\\top} Z)^{-1}`.  Every
solve uses the cuSOLVER-free :func:`~nitrix.linalg._smalllinalg.small_inv_logdet`.

References
----------
- Rigby, R. A. & Stasinopoulos, D. M. (2005). Generalized additive models for
  location, scale and shape.  J. R. Statist. Soc. C 54, 507-554.
  :doi:`10.1111/j.1467-9876.2005.00510.x`
- Wood, S. N., Pya, N. & Saefken, B. (2016). Smoothing parameter and model
  selection for general smooth models.  JASA 111, 1548-1563 (``mgcv`` ``gaulss``).
  :doi:`10.1080/01621459.2016.1180986`
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple, cast

import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Float

from ..linalg._smalllinalg import small_inv_logdet
from ._batching import blocked_vmap
from ._result import register_result

__all__ = ['GauLSSResult', 'gaulss_fit', 'gaulss_predict']

_LOG_2PI = 1.8378770664093453


@register_result(
    children=('coef_mu', 'coef_scale', 'cov_mu', 'cov_scale', 'log_lik'),
    aux=('n_obs',),
)
@dataclass(frozen=True)
class GauLSSResult:
    """Per-element Gaussian location-scale fit output.

    A frozen container holding the fitted coefficients, their asymptotic
    covariances, and the maximised log-likelihood for every element of a
    mass-univariate Gaussian location-scale regression.

    Attributes
    ----------
    coef_mu : Float[Array, 'V p']
        Mean-model coefficients (identity link), one length-``p`` vector per
        element.
    coef_scale : Float[Array, 'V q']
        Scale-model coefficients, one length-``q`` vector per element, defined by
        the log link :math:`\\log \\sigma = Z \\beta_{\\sigma}`.
    cov_mu : Float[Array, 'V p p']
        Asymptotic mean-coefficient covariance per element,
        :math:`\\operatorname{Cov}(\\beta_{\\mu}) = (X^{\\top} \\operatorname{diag}(1/\\sigma^2) X)^{-1}`.
    cov_scale : Float[Array, 'V q q']
        Asymptotic scale-coefficient covariance per element,
        :math:`\\operatorname{Cov}(\\beta_{\\sigma}) = (2 Z^{\\top} Z)^{-1}`.
    log_lik : Float[Array, 'V']
        Maximised Gaussian location-scale log-likelihood per element.
    n_obs : int
        Number of observations :math:`N` used in the fit.
    """

    coef_mu: Float[Array, 'V p']
    coef_scale: Float[Array, 'V q']
    cov_mu: Float[Array, 'V p p']
    cov_scale: Float[Array, 'V q q']
    log_lik: Float[Array, 'V']
    n_obs: int


def _gaulss_fit_one(
    y: Float[Array, 'N'],
    X: Float[Array, 'N p'],
    Z: Float[Array, 'N q'],
    p: int,
    q: int,
    n_iter: int,
    ridge: float,
) -> Tuple[
    Float[Array, 'p'],
    Float[Array, 'q'],
    Float[Array, 'p p'],
    Float[Array, 'q q'],
    Float[Array, ''],
]:
    """Single-element Gaussian location-scale fit by block Fisher scoring.

    Fits one Gaussian location-scale regression by alternating two decoupled
    solves per scoring step: a weighted least-squares update of the mean
    coefficients and an additive information-scaled update of the log-scale
    coefficients, iterated for a fixed number of steps.

    Parameters
    ----------
    y : Float[Array, 'N']
        Response vector for the single element.
    X : Float[Array, 'N p']
        Mean-model design matrix (identity link).
    Z : Float[Array, 'N q']
        Log-scale design matrix (log link).
    p : int
        Number of mean-model coefficients (columns of ``X``).
    q : int
        Number of scale-model coefficients (columns of ``Z``).
    n_iter : int
        Number of Fisher-scoring iterations.
    ridge : float
        Small stabiliser added to the diagonal of each normal-equation matrix.

    Returns
    -------
    beta_mu : Float[Array, 'p']
        Fitted mean-model coefficients.
    beta_s : Float[Array, 'q']
        Fitted log-scale-model coefficients.
    cov_mu : Float[Array, 'p p']
        Asymptotic mean-coefficient covariance
        :math:`(X^{\\top} \\operatorname{diag}(1/\\sigma^2) X)^{-1}`.
    cov_scale : Float[Array, 'q q']
        Asymptotic scale-coefficient covariance :math:`(2 Z^{\\top} Z)^{-1}`.
    log_lik : Float[Array, '']
        Maximised Gaussian location-scale log-likelihood (scalar).
    """
    eye_p = ridge * jnp.eye(p, dtype=X.dtype)
    eye_q = ridge * jnp.eye(q, dtype=X.dtype)
    beta_mu0 = small_inv_logdet(X.T @ X + eye_p, p)[0] @ (X.T @ y)
    beta_s0 = jnp.zeros((q,), dtype=X.dtype)

    def step(
        carry: Tuple[Array, Array], _: Any
    ) -> Tuple[Tuple[Array, Array], None]:
        beta_mu, beta_s = carry
        eta_s = Z @ beta_s
        w = jnp.exp(-2.0 * eta_s)  # 1 / sigma^2
        xw = X * w[:, None]
        beta_mu = small_inv_logdet(xw.T @ X + eye_p, p)[0] @ (xw.T @ y)
        resid = y - X @ beta_mu
        u = -1.0 + resid * resid * w  # scale score per obs
        beta_s = beta_s + small_inv_logdet(2.0 * (Z.T @ Z) + eye_q, q)[0] @ (
            Z.T @ u
        )
        return (beta_mu, beta_s), None

    (beta_mu, beta_s), _ = lax.scan(
        step, (beta_mu0, beta_s0), xs=None, length=n_iter
    )
    eta_s = Z @ beta_s
    w = jnp.exp(-2.0 * eta_s)
    resid = y - X @ beta_mu
    cov_mu, _ = small_inv_logdet((X * w[:, None]).T @ X + eye_p, p)
    cov_scale, _ = small_inv_logdet(2.0 * (Z.T @ Z) + eye_q, q)
    log_lik = jnp.sum(-0.5 * _LOG_2PI - eta_s - 0.5 * resid * resid * w)
    return beta_mu, beta_s, cov_mu, cov_scale, log_lik


def gaulss_fit(
    Y: Float[Array, 'V N'],
    X: Float[Array, 'N p'],
    *,
    scale_design: Optional[Float[Array, 'N q']] = None,
    n_iter: int = 50,
    ridge: float = 1e-8,
    block: Optional[int] = None,
) -> GauLSSResult:
    """Mass-univariate Gaussian location-scale regression (``gaulss``).

    Fits, per element, :math:`y_v \\sim \\mathcal{N}(\\mu, \\sigma^2)` with
    :math:`\\mu = X \\beta_{\\mu}` and :math:`\\log \\sigma = Z \\beta_{\\sigma}` --
    modelling heteroscedasticity directly -- by joint Fisher scoring.  The mean
    and log-scale designs are shared across elements; only the responses vary.

    Parameters
    ----------
    Y : Float[Array, 'V N']
        Responses, one length-``N`` observation vector per element.
    X : Float[Array, 'N p']
        Mean-model design matrix, shared across elements; carries its own
        intercept.
    scale_design : Float[Array, 'N q'], optional
        Log-scale design matrix :math:`Z`, shared across elements.  Defaults to
        an intercept-only column -- a constant variance (the homoscedastic
        Gaussian, recovering OLS for the mean); pass a design to model the
        variance.
    n_iter : int, optional
        Number of Fisher-scoring iterations. Default ``50``.
    ridge : float, optional
        Small stabiliser added to the normal equations. Default ``1e-8``.
    block : int, optional
        Element-block size bounding peak memory; ``None`` processes all elements
        at once.

    Returns
    -------
    GauLSSResult
        Fit output holding the mean coefficients ``coef_mu``, the log-scale
        coefficients ``coef_scale``, the block information inverses ``cov_mu``
        and ``cov_scale``, and the maximised log-likelihood ``log_lik``.  A mean
        contrast is :math:`c^{\\top} \\mathrm{coef\\_mu}` with standard error
        :math:`\\sqrt{c^{\\top} \\mathrm{cov\\_mu}\\, c}`; a scale contrast is the
        analogous read-out on ``coef_scale``.

    See Also
    --------
    gaulss_predict : Evaluate the fitted mean and scale on a (new) design.
    """
    n, p = X.shape
    if Y.shape[-1] != n:
        raise ValueError(
            f'gaulss_fit: Y.shape[-1]={Y.shape[-1]} must match N={n}.'
        )
    Z = (
        jnp.ones((n, 1), dtype=X.dtype)
        if scale_design is None
        else jnp.asarray(scale_design, dtype=X.dtype)
    )
    if Z.shape[0] != n:
        raise ValueError(
            f'gaulss_fit: scale_design has {Z.shape[0]} rows; expected N={n}.'
        )
    q = Z.shape[1]

    def per_voxel(
        y: Float[Array, 'N'],
    ) -> Tuple[Array, Array, Array, Array, Array]:
        return _gaulss_fit_one(y, X, Z, p, q, n_iter, ridge)

    coef_mu, coef_scale, cov_mu, cov_scale, log_lik = cast(
        Tuple[Array, Array, Array, Array, Array],
        blocked_vmap(per_voxel, (Y,), block=block),
    )
    return GauLSSResult(
        coef_mu=coef_mu,
        coef_scale=coef_scale,
        cov_mu=cov_mu,
        cov_scale=cov_scale,
        log_lik=log_lik,
        n_obs=int(n),
    )


def gaulss_predict(
    result: GauLSSResult,
    X: Float[Array, 'N p'],
    *,
    scale_design: Optional[Float[Array, 'N q']] = None,
) -> Tuple[Float[Array, 'V N'], Float[Array, 'V N']]:
    """Per-element Gaussian location-scale prediction on a (new) design.

    Returns the pair ``(mean, scale)`` -- the heteroscedastic point of the
    model -- on the new covariates: the mean :math:`\\mu = X \\beta_{\\mu}`
    (identity link) and the standard deviation
    :math:`\\sigma = \\exp(Z \\beta_{\\sigma})` (log link).

    Parameters
    ----------
    result : GauLSSResult
        Fit output from :func:`gaulss_fit`.
    X : Float[Array, 'N p']
        Mean-model design matrix, with the same columns as the fit.
    scale_design : Float[Array, 'N q'], optional
        Scale-model design matrix; ``None`` (default) uses an intercept column
        of shape ``(N, 1)`` -- the homoscedastic default matching
        :func:`gaulss_fit`.  Must have the same column count ``q`` that the
        fit's scale model used.

    Returns
    -------
    mean : Float[Array, 'V N']
        Predicted mean for each element and observation.
    scale : Float[Array, 'V N']
        Predicted standard deviation for each element and observation.  Both
        outputs are differentiable with respect to their designs and the fitted
        coefficients.
    """
    mu = result.coef_mu @ X.T  # (V, N)
    n = X.shape[0]
    z = (
        jnp.ones((n, 1), dtype=X.dtype)
        if scale_design is None
        else jnp.asarray(scale_design, dtype=X.dtype)
    )
    sigma = jnp.exp(result.coef_scale @ z.T)  # (V, N)
    return mu, sigma
