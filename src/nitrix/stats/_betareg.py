# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Beta regression for proportions in ``(0, 1)`` (v3 §4).

Bounded continuous responses -- tissue fractions, fMRI fractional occupancy,
any proportion -- are modelled as ``y ~ Beta(mu phi, (1 - mu) phi)`` with mean
``mu = logit^{-1}(X beta)`` and a **precision** ``phi`` (larger ``phi`` = less
dispersion).  This is *not* a standard exponential-family GLM: the score is
**digamma**-based, not the ``mu_eta^2 / V`` IRLS weight, and the precision is a
second parameter estimated jointly with the mean -- so it gets its own fitter
rather than slotting into the :class:`~nitrix.stats._family.Family` / ``glm_fit``
core.

Estimator (Ferrari & Cribari-Neto 2004).  Fisher scoring alternates a
reweighted-least-squares step for ``beta`` and a Newton step for ``log phi``,
to a fixed iteration budget (``vmap``-clean over voxels):

    y*   = logit(y),   mu*  = psi(mu phi) - psi((1 - mu) phi)
    nu   = psi'(mu phi) + psi'((1 - mu) phi)          # trigamma sum
    W    = diag(phi^2 (dmu/deta)^2 nu),   z = eta + (y* - mu*) / (phi (dmu/deta) nu)
    beta <- (X^T W X)^{-1} X^T W z

with the precision score / information

    U_phi = sum mu(y* - mu*) + log(1 - y) - psi((1-mu)phi) + psi(phi)
    K_phi = sum mu^2 psi'(mu phi) + (1-mu)^2 psi'((1-mu)phi) - psi'(phi),

stepped as ``log phi <- log phi + U_phi / (phi K_phi)``.  The fixed-effect
covariance is the ``beta`` block of the **joint** ``(beta, phi)`` Fisher
information inverse (so standard errors account for estimating ``phi``).  Every
solve is the cuSOLVER-free ``small_inv_logdet``; ``digamma`` / ``polygamma`` are
``jax.scipy.special``.

References
----------
- Ferrari, S. & Cribari-Neto, F. (2004). Beta regression for modelling rates and
  proportions.  J. Applied Statistics 31, 799-815.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple, cast

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.special import digamma, gammaln, polygamma
from jaxtyping import Array, Float

from ._batching import blocked_vmap
from ._smalllinalg import small_inv_logdet

__all__ = ['BetaResult', 'beta_fit']

_EPS = 1e-8


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class BetaResult:
    """Per-element beta-regression fit output.

    Attributes
    ----------
    coef
        ``(V, p)`` mean-model coefficients (logit link).
    precision
        ``(V,)`` precision ``phi`` (``Var(y) = mu(1-mu)/(1+phi)``).
    cov_unscaled
        ``(V, p, p)`` ``Cov(beta_hat)`` -- the ``beta`` block of the joint
        ``(beta, phi)`` Fisher-information inverse (SEs account for estimating
        ``phi``).
    log_lik
        ``(V,)`` maximised beta log-likelihood (for ``aic`` / the LRT).
    n_obs
        Number of observations ``N``.
    """

    coef: Float[Array, 'V p']
    precision: Float[Array, 'V']
    cov_unscaled: Float[Array, 'V p p']
    log_lik: Float[Array, 'V']
    n_obs: int

    def tree_flatten(
        self,
    ) -> Tuple[Tuple[Array, ...], Tuple[Any, ...]]:
        return (
            self.coef,
            self.precision,
            self.cov_unscaled,
            self.log_lik,
        ), (self.n_obs,)

    @classmethod
    def tree_unflatten(
        cls, aux: Tuple[Any, ...], children: Tuple[Any, ...]
    ) -> 'BetaResult':
        (n_obs,) = aux
        coef, precision, cov_unscaled, log_lik = children
        return cls(
            coef=coef,
            precision=precision,
            cov_unscaled=cov_unscaled,
            log_lik=log_lik,
            n_obs=n_obs,
        )


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
    """Single-element beta-regression fit (Ferrari-Cribari-Neto Fisher scoring).

    Returns ``(beta, phi, cov_beta, log_lik)``.
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
        k_phi = (
            jnp.sum(
                polygamma(1, mu * phi) * mu * mu
                + polygamma(1, (1.0 - mu) * phi) * (1.0 - mu) * (1.0 - mu)
            )
            - n * polygamma(1, phi)
        )
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
    k_phiphi = (
        jnp.sum(
            polygamma(1, mu * phi) * mu * mu
            + polygamma(1, (1.0 - mu) * phi) * (1.0 - mu) * (1.0 - mu)
        )
        - n * polygamma(1, phi)
    )
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

    Fits, per element, ``y_v ~ Beta(mu phi, (1 - mu) phi)`` with ``mu =
    logit^{-1}(X beta_v)`` and a per-element precision ``phi_v``, by
    Ferrari-Cribari-Neto Fisher scoring.  Responses must lie strictly in
    ``(0, 1)`` (values are clipped off the boundary by ``1e-8``).

    Parameters
    ----------
    Y
        ``(V, N)`` responses in ``(0, 1)``.
    X
        ``(N, p)`` shared design (include your own intercept).
    n_iter
        Fisher-scoring iterations (default ``50``).
    ridge
        Small stabiliser on the normal equations.
    block
        Optional element-block size bounding peak memory.

    Returns
    -------
    ``BetaResult`` -- ``coef``, ``precision``, ``cov_unscaled``
    (``Cov(beta_hat)``), ``log_lik``.  A fixed-effect contrast is
    ``c^T beta_hat`` with SE ``sqrt(c^T cov_unscaled c)`` on ``N - p`` df.
    """
    n, p = X.shape
    if Y.shape[-1] != n:
        raise ValueError(
            f'beta_fit: Y.shape[-1]={Y.shape[-1]} must match N={n}.'
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
