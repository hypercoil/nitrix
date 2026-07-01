# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
The single penalised iteratively-reweighted least squares core shared by GLM
and GAM.

Iteratively-reweighted least squares for an exponential-family model, solving
the penalised normal equations :math:`(X^{\\top} W X + S)\\,\\beta = X^{\\top} W z`
over a working response :math:`z` and working weights :math:`W`.  The
generalised-linear-model path calls it with the penalty :math:`S = 0` and prior
observation weights; the generalised-additive-model path calls it with the
block smoothing penalty :math:`S(\\lambda)` and no prior weights.  This module
is the single source of truth for that recipe.

Every :math:`(p, p)` solve goes through the cuSOLVER-free
:func:`small_inv_logdet`.  The converged covariance
:math:`V = (X^{\\top} W X + S + \\text{ridge})^{-1}` and the unpenalised Gram
:math:`X^{\\top} W X` are evaluated at the converged :math:`\\beta` -- the
correct point for the coefficient covariance and the additive-model effective
degrees of freedom / Fellner-Schall traces -- so that final evaluation is
intentional, not redundant.
"""

from __future__ import annotations

from typing import Optional, Tuple, cast

import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Float

from ..linalg._smalllinalg import small_inv_logdet
from ._family import Family

__all__ = ['fit_penalised_irls', 'irls_warm_start', 'safe_dmu']

_EPS = 1e-10


def safe_dmu(dmu: Float[Array, 'N']) -> Float[Array, 'N']:
    """Floor :math:`|d\\mu|` away from zero while preserving its sign.

    The single source of truth for the IRLS working-response denominator across
    the generalised-linear, generalised-additive and mixed-model paths.  A
    *decreasing* link -- the reciprocal / inverse link has
    :math:`d\\mu = -1/\\eta^2 < 0` -- would otherwise have its derivative flipped
    to :math:`+\\varepsilon` by a naive ``clip(dmu, _EPS, None)``, exploding the
    working response :math:`z = \\eta + (y - \\mu) / d\\mu`.  For the increasing
    canonical links (:math:`d\\mu > 0`) this reduces to
    :math:`\\max(d\\mu, \\varepsilon)`.

    Parameters
    ----------
    dmu : Float[Array, 'N']
        Per-observation derivative of the mean with respect to the linear
        predictor, :math:`d\\mu/d\\eta`, evaluated at each of the ``N``
        observations.  May be positive or negative.

    Returns
    -------
    Float[Array, 'N']
        The same array with each entry pushed to have magnitude at least
        ``_EPS``, retaining its original sign, so that division by it is safe.
    """
    return jnp.where(
        dmu < 0.0, jnp.minimum(dmu, -_EPS), jnp.maximum(dmu, _EPS)
    )


def _working(
    beta: Float[Array, 'p'],
    y: Float[Array, 'N'],
    X: Float[Array, 'N p'],
    family: Family,
    prior_weights: Optional[Float[Array, 'N']],
) -> Tuple[Float[Array, 'N'], Float[Array, 'N'], Float[Array, 'N']]:
    """Compute the IRLS working weights, working response and fitted mean.

    Evaluates the linear predictor :math:`\\eta = X\\beta` (clamped to the
    family's numerically-sane range), maps it through the inverse link to the
    fitted mean :math:`\\mu`, and forms the local weights
    :math:`w = (d\\mu/d\\eta)^2 / \\operatorname{var}(\\mu)` and working response
    :math:`z = \\eta + (y - \\mu) / (d\\mu/d\\eta)`.  When prior weights are
    supplied they multiply the local weights.

    Parameters
    ----------
    beta : Float[Array, 'p']
        Current coefficient vector.
    y : Float[Array, 'N']
        Observed response for each of the ``N`` observations.
    X : Float[Array, 'N p']
        Design matrix with ``N`` rows and ``p`` columns.
    family : Family
        Exponential-family specification supplying the link, inverse link, mean
        derivative and variance function.
    prior_weights : Float[Array, 'N'] or None
        Optional non-negative per-observation weights; ``None`` leaves the
        working weights unscaled.

    Returns
    -------
    wts : Float[Array, 'N']
        Working weights :math:`W` (already scaled by ``prior_weights`` when
        given).
    z : Float[Array, 'N']
        Working response :math:`z`.
    mu : Float[Array, 'N']
        Fitted mean :math:`\\mu` at the current ``beta``.
    """
    # Clamp the linear predictor to the family's numerically-sane range: for the
    # unbounded exp links a transient overshoot otherwise blows up exp(eta) and
    # the working weights (a single observation dominates -> garbage / NaN); the
    # bounded links carry eta_bound = inf, so this is a no-op for them.
    eta = family.clip_eta(X @ beta)
    mu = family.linkinv(eta)
    dmu = family.mu_eta(eta)
    var = family.variance(mu)
    wts = dmu * dmu / jnp.clip(var, _EPS, None)
    if prior_weights is not None:
        wts = prior_weights * wts
    z = eta + (y - mu) / safe_dmu(dmu)
    return wts, z, mu


def irls_warm_start(
    y: Float[Array, 'N'],
    X: Float[Array, 'N p'],
    family: Family,
    *,
    penalty: Float[Array, 'p p'],
    ridge: float,
    prior_weights: Optional[Float[Array, 'N']] = None,
) -> Float[Array, 'p']:
    """Produce a warm-start coefficient vector for penalised IRLS.

    Fits the penalised, prior-weighted ordinary-least-squares solution of the
    linked initial mean :math:`\\eta_0 = \\text{link}(\\text{init\\_mu}(y))`.  This
    is the starting point used to seed the generalised-linear-model IRLS
    iteration.

    Parameters
    ----------
    y : Float[Array, 'N']
        Observed response for each of the ``N`` observations.
    X : Float[Array, 'N p']
        Design matrix with ``N`` rows and ``p`` columns.
    family : Family
        Exponential-family specification supplying the link and the initial-mean
        map.
    penalty : Float[Array, 'p p']
        Smoothing / regularisation penalty matrix :math:`S` added to the Gram
        before solving.  Pass a zero matrix for an unpenalised fit.
    ridge : float
        Non-negative Tikhonov ridge added to the diagonal of the Gram for
        numerical stability.
    prior_weights : Float[Array, 'N'] or None
        Optional non-negative per-observation weights; ``None`` treats every
        observation equally.

    Returns
    -------
    Float[Array, 'p']
        The warm-start coefficient vector :math:`\\beta`.
    """
    p = X.shape[-1]
    eta0 = family.link(family.init_mu(y))
    Xw = X if prior_weights is None else X * prior_weights[:, None]
    a0 = Xw.T @ X + penalty + ridge * jnp.eye(p, dtype=X.dtype)
    a0_inv, _ = small_inv_logdet(a0, p)
    return a0_inv @ (Xw.T @ eta0)


def fit_penalised_irls(
    y: Float[Array, 'N'],
    X: Float[Array, 'N p'],
    family: Family,
    *,
    penalty: Float[Array, 'p p'],
    beta0: Float[Array, 'p'],
    n_iter: int,
    ridge: float,
    prior_weights: Optional[Float[Array, 'N']] = None,
) -> Tuple[
    Float[Array, 'p'],
    Float[Array, 'p p'],
    Float[Array, 'p p'],
    Float[Array, ''],
]:
    """Run penalised IRLS from a warm start for a fixed number of iterations.

    Iterates the penalised reweighted least-squares update
    :math:`(X^{\\top} W X + S + \\text{ridge})\\,\\beta = X^{\\top} W z` for
    ``n_iter`` steps, then re-evaluates the working weights, covariance and Gram
    at the final :math:`\\beta` so that the returned covariance and unpenalised
    Gram correspond to the converged fit.

    Parameters
    ----------
    y : Float[Array, 'N']
        Observed response for each of the ``N`` observations.
    X : Float[Array, 'N p']
        Design matrix with ``N`` rows and ``p`` columns.
    family : Family
        Exponential-family specification supplying the link, mean derivative,
        variance and unit-deviance functions.
    penalty : Float[Array, 'p p']
        Smoothing / regularisation penalty matrix :math:`S` added to the Gram at
        every step.  Pass a zero matrix for an unpenalised fit.
    beta0 : Float[Array, 'p']
        Warm-start coefficient vector; typically the output of
        :func:`irls_warm_start`.
    n_iter : int
        Fixed number of IRLS iterations to perform.
    ridge : float
        Non-negative Tikhonov ridge added to the diagonal of the Gram at every
        step for numerical stability.
    prior_weights : Float[Array, 'N'] or None
        Optional non-negative per-observation weights; when given they scale the
        working weights and the deviance contribution of each observation.

    Returns
    -------
    beta : Float[Array, 'p']
        Fitted coefficient vector after ``n_iter`` iterations.
    V : Float[Array, 'p p']
        Bayesian / unscaled coefficient covariance
        :math:`V = (X^{\\top} W X + S + \\text{ridge})^{-1}`, evaluated at the
        converged ``beta``.
    xtwx : Float[Array, 'p p']
        Unpenalised weighted Gram :math:`X^{\\top} W X`, evaluated at the
        converged ``beta``.
    deviance : Float[Array, '']
        Total model deviance :math:`\\sum_i w_i\\, d(y_i, \\mu_i)` at the converged
        fit, carrying the prior weights when supplied.
    """
    p = X.shape[-1]
    ridge_eye = ridge * jnp.eye(p, dtype=X.dtype)

    def step(_: Array, beta: Float[Array, 'p']) -> Float[Array, 'p']:
        wts, z, _ = _working(beta, y, X, family, prior_weights)
        Xw = X * wts[:, None]
        a = Xw.T @ X + penalty + ridge_eye
        a_inv, _ = small_inv_logdet(a, p)
        return a_inv @ (Xw.T @ z)

    beta = cast(Float[Array, 'p'], lax.fori_loop(0, n_iter, step, beta0))

    wts, _, mu = _working(beta, y, X, family, prior_weights)
    xtwx = (X * wts[:, None]).T @ X
    v, _ = small_inv_logdet(xtwx + penalty + ridge_eye, p)
    # Deviance carries the prior weights (sum_i w_i d(y_i, mu_i)), matching the
    # weighted-RSS convention of the OLS path; a no-op when unweighted.
    unit_dev = family.unit_deviance(y, mu)
    dev = jnp.sum(
        unit_dev if prior_weights is None else prior_weights * unit_dev
    )
    return beta, v, xtwx, dev
