# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Beta regression for proportions in :math:`(0, 1)`.

Bounded continuous responses -- tissue fractions, fMRI fractional occupancy,
any proportion -- are modelled as
:math:`y \\sim \\operatorname{Beta}(\\mu\\phi,\\, (1 - \\mu)\\phi)` with mean
:math:`\\mu = \\operatorname{logit}^{-1}(X\\beta)` and a **precision**
:math:`\\phi` (larger :math:`\\phi` = less dispersion).  This is *not* a
standard exponential-family GLM: the score is **digamma**-based, not the
:math:`\\mu_\\eta^2 / V` IRLS weight, and the precision is a second parameter
estimated jointly with the mean -- so it gets its own fitter rather than
slotting into a standard GLM core.

The estimator is the Ferrari & Cribari-Neto Fisher-scoring scheme, which
alternates a reweighted-least-squares step for :math:`\\beta` and a Newton step
for :math:`\\log\\phi`, to a fixed iteration budget (``vmap``-clean over
voxels):

.. math::

    y^{*} &= \\operatorname{logit}(y), \\quad
    \\mu^{*} = \\psi(\\mu\\phi) - \\psi((1 - \\mu)\\phi) \\\\
    \\nu &= \\psi'(\\mu\\phi) + \\psi'((1 - \\mu)\\phi) \\\\
    W &= \\operatorname{diag}\\!\\left(\\phi^2
        (\\mathrm{d}\\mu / \\mathrm{d}\\eta)^2 \\nu\\right), \\quad
    z = \\eta + \\frac{y^{*} - \\mu^{*}}
        {\\phi \\, (\\mathrm{d}\\mu / \\mathrm{d}\\eta) \\, \\nu} \\\\
    \\beta &\\leftarrow (X^{\\top} W X)^{-1} X^{\\top} W z

with the precision score / information

.. math::

    U_\\phi &= \\sum \\mu(y^{*} - \\mu^{*}) + \\log(1 - y)
        - \\psi((1 - \\mu)\\phi) + \\psi(\\phi) \\\\
    K_\\phi &= \\sum \\mu^2 \\psi'(\\mu\\phi)
        + (1 - \\mu)^2 \\psi'((1 - \\mu)\\phi) - \\psi'(\\phi),

stepped as
:math:`\\log\\phi \\leftarrow \\log\\phi + U_\\phi / (\\phi K_\\phi)`.  The
fixed-effect covariance is the :math:`\\beta` block of the **joint**
:math:`(\\beta, \\phi)` Fisher information inverse (so standard errors account
for estimating :math:`\\phi`).  Every solve is a cuSOLVER-free small dense
inverse; :math:`\\psi` (digamma) and its derivatives (polygamma) come from
``jax.scipy.special``.

References
----------
- Ferrari, S. & Cribari-Neto, F. (2004). Beta regression for modelling rates and
  proportions.  Journal of Applied Statistics, 31(7), 799-815.
  https://doi.org/10.1080/0266476042000214501
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Literal, Optional, Tuple, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.scipy.special import digamma, gammaln, polygamma
from jaxtyping import Array, Float

from ..linalg._smalllinalg import small_inv_logdet
from ._batching import blocked_vmap
from ._result import register_result

__all__ = ['BetaResult', 'beta_fit', 'beta_predict']

_EPS = 1e-8


@register_result(
    children=('coef', 'precision', 'cov_unscaled', 'log_lik'),
    aux=('n_obs',),
)
@dataclass(frozen=True)
class BetaResult:
    """Per-element beta-regression fit output.

    Frozen container of the fitted mean-model coefficients, precision,
    coefficient covariance, and maximised log-likelihood for every element of a
    mass-univariate beta regression, together with the observation count.  Each
    of the ``V`` elements (voxels, vertices, ...) is fitted independently and
    shares the single design matrix passed to :func:`beta_fit`.

    Attributes
    ----------
    coef : Float[Array, 'V p']
        ``(V, p)`` mean-model coefficients :math:`\\beta` under the logit link.
    precision : Float[Array, 'V']
        ``(V,)`` precision :math:`\\phi`, so that
        :math:`\\operatorname{Var}(y) = \\mu(1 - \\mu) / (1 + \\phi)`.
    cov_unscaled : Float[Array, 'V p p']
        ``(V, p, p)`` coefficient covariance
        :math:`\\operatorname{Cov}(\\hat\\beta)` -- the :math:`\\beta` block of
        the joint :math:`(\\beta, \\phi)` Fisher-information inverse (standard
        errors thus account for estimating :math:`\\phi`).
    log_lik : Float[Array, 'V']
        ``(V,)`` maximised beta log-likelihood (for :func:`~nitrix.stats.aic`
        or a likelihood-ratio test).
    n_obs : int
        Number of observations :math:`N`.
    """

    coef: Float[Array, 'V p']
    precision: Float[Array, 'V']
    cov_unscaled: Float[Array, 'V p p']
    log_lik: Float[Array, 'V']
    n_obs: int


def _logit(p: Float[Array, 'N']) -> Float[Array, 'N']:
    return jnp.log(p / (1.0 - p))


def _expit(x: Float[Array, 'N']) -> Float[Array, 'N']:
    return 0.5 * (1.0 + jnp.tanh(0.5 * x))


def _beta_fit_one(
    y: Float[Array, 'N'],
    X: Float[Array, 'N p'],
    p: int,
    n_iter: int,
    ridge: float,
) -> Tuple[
    Float[Array, 'p'], Float[Array, ''], Float[Array, 'p p'], Float[Array, '']
]:
    """Single-element beta-regression fit by Ferrari-Cribari-Neto Fisher scoring.

    Runs the fixed-budget Fisher-scoring loop for one response vector: a logit-
    OLS warm start for :math:`\\beta` (with precision initialised at
    :math:`\\phi = 1`), then ``n_iter`` alternations of a reweighted-least-
    squares update for :math:`\\beta` and a Newton step for :math:`\\log\\phi`.
    Responses are clipped into :math:`[\\varepsilon, 1 - \\varepsilon]` before
    the logit transform.  After convergence it forms the joint
    :math:`(\\beta, \\phi)` Fisher information, extracts the :math:`\\beta`
    covariance block, and evaluates the beta log-likelihood.

    Parameters
    ----------
    y : Float[Array, 'N']
        ``(N,)`` response vector; entries should lie in :math:`(0, 1)` and are
        clipped off the boundary before fitting.
    X : Float[Array, 'N p']
        ``(N, p)`` design matrix (include your own intercept column).
    p : int
        Number of design columns, matching ``X.shape[1]``; the static width of
        the linear solves.
    n_iter : int
        Number of Fisher-scoring iterations to run.
    ridge : float
        Small ridge added to the diagonal of every normal-equation and
        information matrix for numerical stability.

    Returns
    -------
    beta : Float[Array, 'p']
        ``(p,)`` fitted mean-model coefficients.
    phi : Float[Array, '']
        Scalar fitted precision :math:`\\phi`.
    cov_beta : Float[Array, 'p p']
        ``(p, p)`` coefficient covariance -- the :math:`\\beta` block of the
        joint :math:`(\\beta, \\phi)` Fisher-information inverse.
    log_lik : Float[Array, '']
        Scalar maximised beta log-likelihood.
    """
    n = y.shape[0]
    yc = jnp.clip(y, _EPS, 1.0 - _EPS)
    ystar = _logit(yc)
    log1my = jnp.log(jnp.clip(1.0 - yc, _EPS, None))
    eye_p = ridge * jnp.eye(p, dtype=X.dtype)
    # logit-OLS warm start for beta; phi starts at 1.
    beta0 = small_inv_logdet(X.T @ X + eye_p, p)[0] @ (X.T @ ystar)

    def step(
        carry: Tuple[Array, Array], _: Any
    ) -> Tuple[Tuple[Array, Array], None]:
        beta, log_phi = carry
        phi = jnp.exp(log_phi)
        eta = X @ beta
        mu = _expit(eta)
        dmu = mu * (1.0 - mu)
        mu_star = digamma(mu * phi) - digamma((1.0 - mu) * phi)
        nu = polygamma(1, mu * phi) + polygamma(1, (1.0 - mu) * phi)
        w = phi * phi * dmu * dmu * nu
        z = eta + (ystar - mu_star) / (phi * dmu * nu)
        xw = X * w[:, None]
        beta = small_inv_logdet(xw.T @ X + eye_p, p)[0] @ (xw.T @ z)

        # precision Newton step on log phi (recompute mu at the new beta).
        eta = X @ beta
        mu = _expit(eta)
        mu_star = digamma(mu * phi) - digamma((1.0 - mu) * phi)
        u_phi = jnp.sum(
            mu * (ystar - mu_star)
            + log1my
            - digamma((1.0 - mu) * phi)
            + digamma(phi)
        )
        k_phi = jnp.sum(
            polygamma(1, mu * phi) * mu * mu
            + polygamma(1, (1.0 - mu) * phi) * (1.0 - mu) * (1.0 - mu)
        ) - n * polygamma(1, phi)
        log_phi = log_phi + jnp.clip(
            u_phi / (phi * jnp.clip(k_phi, 1e-6, None)), -2.0, 2.0
        )
        return (beta, log_phi), None

    (beta, log_phi), _ = lax.scan(
        step, (beta0, jnp.asarray(0.0, dtype=X.dtype)), xs=None, length=n_iter
    )
    phi = jnp.exp(log_phi)

    eta = X @ beta
    mu = _expit(eta)
    dmu = mu * (1.0 - mu)
    nu = polygamma(1, mu * phi) + polygamma(1, (1.0 - mu) * phi)
    # Joint (beta, phi) Fisher information; Cov(beta) is its beta block.
    k_bb = (X * (phi * phi * dmu * dmu * nu)[:, None]).T @ X  # (p, p)
    k_bphi = X.T @ (
        phi
        * (
            mu * polygamma(1, mu * phi)
            - (1.0 - mu) * polygamma(1, (1.0 - mu) * phi)
        )
        * dmu
    )  # (p,)
    k_phiphi = jnp.sum(
        polygamma(1, mu * phi) * mu * mu
        + polygamma(1, (1.0 - mu) * phi) * (1.0 - mu) * (1.0 - mu)
    ) - n * polygamma(1, phi)
    info = jnp.zeros((p + 1, p + 1), dtype=X.dtype)
    info = info.at[:p, :p].set(k_bb)
    info = info.at[:p, p].set(k_bphi)
    info = info.at[p, :p].set(k_bphi)
    info = info.at[p, p].set(k_phiphi)
    info = info + ridge * jnp.eye(p + 1, dtype=X.dtype)
    info_inv, _ = small_inv_logdet(info, p + 1)
    cov_beta = info_inv[:p, :p]

    log_lik = jnp.sum(
        gammaln(phi)
        - gammaln(mu * phi)
        - gammaln((1.0 - mu) * phi)
        + (mu * phi - 1.0) * jnp.log(yc)
        + ((1.0 - mu) * phi - 1.0) * log1my
    )
    return beta, phi, cov_beta, log_lik


def beta_fit(
    Y: Float[Array, 'V N'],
    X: Float[Array, 'N p'],
    *,
    n_iter: int = 50,
    ridge: float = 1e-8,
    block: Optional[int] = None,
) -> BetaResult:
    """Mass-univariate beta regression (logit link, estimated precision).

    Fits, per element,
    :math:`y_v \\sim \\operatorname{Beta}(\\mu\\phi,\\, (1 - \\mu)\\phi)` with
    :math:`\\mu = \\operatorname{logit}^{-1}(X\\beta_v)` and a per-element
    precision :math:`\\phi_v`, by Ferrari-Cribari-Neto Fisher scoring.
    Responses must lie strictly in :math:`(0, 1)`; values are clipped off the
    boundary by ``1e-8`` (and a host-side warning is raised for out-of-range
    inputs when not tracing).

    Parameters
    ----------
    Y : Float[Array, 'V N']
        ``(V, N)`` responses, each in :math:`(0, 1)`; one row per element.
    X : Float[Array, 'N p']
        ``(N, p)`` design matrix shared across elements (include your own
        intercept column).
    n_iter : int, optional
        Number of Fisher-scoring iterations (default ``50``).
    ridge : float, optional
        Small stabiliser added to the diagonal of the normal equations and the
        Fisher information (default ``1e-8``).
    block : int, optional
        Optional element-block size bounding peak memory; ``None`` processes
        all elements in one ``vmap``.

    Returns
    -------
    BetaResult
        A :class:`BetaResult` with fields ``coef``, ``precision``,
        ``cov_unscaled`` (the coefficient covariance
        :math:`\\operatorname{Cov}(\\hat\\beta)`), ``log_lik``, and ``n_obs``.
        A fixed-effect contrast is :math:`c^{\\top}\\hat\\beta` with standard
        error :math:`\\sqrt{c^{\\top}\\,\\mathtt{cov\\_unscaled}\\,c}` on
        :math:`N - p` degrees of freedom.
    """
    n, p = X.shape
    if Y.shape[-1] != n:
        raise ValueError(
            f'beta_fit: Y.shape[-1]={Y.shape[-1]} must match N={n}.'
        )
    # Round 4: responses outside (0, 1) are silently clipped to the boundary,
    # which corrupts the data without notice. Flag it host-side (skip under jit,
    # where Y is a tracer and the values are unavailable).
    if not isinstance(Y, jax.core.Tracer):
        y_host = np.asarray(Y)
        if np.any((y_host <= 0.0) | (y_host >= 1.0)):
            warnings.warn(
                'beta_fit: some responses lie outside the open interval (0, 1) '
                'and are clipped to [1e-8, 1 - 1e-8]; beta regression models '
                'rates / proportions strictly in (0, 1). Rescale if these are '
                'counts or unbounded values.',
                stacklevel=2,
            )

    def per_voxel(
        y: Float[Array, 'N'],
    ) -> Tuple[Array, Array, Array, Array]:
        return _beta_fit_one(y, X, p, n_iter, ridge)

    coef, precision, cov, log_lik = cast(
        Tuple[Array, Array, Array, Array],
        blocked_vmap(per_voxel, (Y,), block=block),
    )
    return BetaResult(
        coef=coef,
        precision=precision,
        cov_unscaled=cov,
        log_lik=log_lik,
        n_obs=int(n),
    )


def beta_predict(
    result: BetaResult,
    X: Float[Array, 'N p'],
    *,
    type: Literal['response', 'link'] = 'response',
) -> Float[Array, 'V N']:
    """Per-element beta-regression prediction on a (new) design ``X``.

    The :func:`beta_fit` analogue of :func:`nitrix.stats.predict`: the mean
    model is the logit link, so ``type='link'`` returns the linear predictor
    ``eta = X beta`` and ``type='response'`` (default) returns the fitted mean
    ``mu = expit(eta)`` in ``(0, 1)``.

    Parameters
    ----------
    result
        A :class:`BetaResult` from :func:`beta_fit`.
    X
        ``(N, p)`` design (same columns as the fit).
    type
        ``'response'`` (the mean ``mu``) or ``'link'`` (the linear predictor
        ``eta``).

    Returns
    -------
    ``(V, N)`` predictions.  Differentiable w.r.t. ``X`` (and the fitted
    ``result.coef``).
    """
    eta = result.coef @ X.T
    if type == 'link':
        return eta
    if type == 'response':
        return _expit(eta)
    raise ValueError(
        f"beta_predict: type={type!r}; expected 'response' or 'link'."
    )
