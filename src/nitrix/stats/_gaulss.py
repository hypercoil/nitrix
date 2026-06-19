# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Gaussian location-scale regression (``gaulss``, v3 §4 distributional).

The standard GLM has **one** linear predictor (the mean); the residual scale is a
single nuisance dispersion.  A *distributional* model gives the scale its own
linear predictor -- here ``y_i ~ N(mu_i, sigma_i^2)`` with

    mu_i      = x_i^T beta_mu           (identity link, the mean model)
    log sigma_i = z_i^T beta_sigma      (log link, the scale model)

so heteroscedasticity is modelled directly (``mgcv``'s ``gaulss`` family / the
``nwx`` reserved ``sigma ~ …`` predictor).  This is **not** the scalar IRLS core
(one predictor, fixed dispersion), so it gets its own fitter.

Estimator.  Joint maximum likelihood by Fisher scoring.  The Gaussian
location-scale **expected** information is block-diagonal between ``beta_mu`` and
``beta_sigma`` (``E[y - mu] = 0`` kills the cross term), so each scoring step is
two decoupled solves:

    beta_mu    <- (X^T W X)^{-1} X^T W y,        W = diag(1 / sigma_i^2)   (WLS)
    beta_sigma <- beta_sigma + (2 Z^T Z)^{-1} Z^T u,  u_i = -1 + (y_i - mu_i)^2 / sigma_i^2

iterated to a fixed budget (``vmap``-clean over voxels).  The coefficient
covariances are the two information blocks' inverses: ``Cov(beta_mu) = (X^T
diag(1/sigma^2) X)^{-1}`` and ``Cov(beta_sigma) = (2 Z^T Z)^{-1}``.  Every solve is
the cuSOLVER-free ``small_inv_logdet``.

References
----------
- Rigby, R. A. & Stasinopoulos, D. M. (2005). Generalized additive models for
  location, scale and shape.  J. R. Statist. Soc. C 54, 507-554.
- Wood, S. N., Pya, N. & Saefken, B. (2016). Smoothing parameter and model
  selection for general smooth models.  JASA 111, 1548-1563 (``mgcv`` ``gaulss``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple, cast

import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Float

from ._batching import blocked_vmap
from ._smalllinalg import small_inv_logdet

__all__ = ['GauLSSResult', 'gaulss_fit']

_LOG_2PI = 1.8378770664093453


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class GauLSSResult:
    """Per-element Gaussian location-scale fit output.

    Attributes
    ----------
    coef_mu
        ``(V, p)`` mean-model coefficients (identity link).
    coef_scale
        ``(V, q)`` scale-model coefficients (``log sigma = Z beta_scale``).
    cov_mu
        ``(V, p, p)`` ``Cov(beta_mu) = (X^T diag(1/sigma^2) X)^{-1}``.
    cov_scale
        ``(V, q, q)`` ``Cov(beta_scale) = (2 Z^T Z)^{-1}``.
    log_lik
        ``(V,)`` maximised Gaussian location-scale log-likelihood.
    n_obs
        Number of observations ``N``.
    """

    coef_mu: Float[Array, 'V p']
    coef_scale: Float[Array, 'V q']
    cov_mu: Float[Array, 'V p p']
    cov_scale: Float[Array, 'V q q']
    log_lik: Float[Array, 'V']
    n_obs: int

    def tree_flatten(
        self,
    ) -> Tuple[Tuple[Array, ...], Tuple[Any, ...]]:
        return (
            self.coef_mu,
            self.coef_scale,
            self.cov_mu,
            self.cov_scale,
            self.log_lik,
        ), (self.n_obs,)

    @classmethod
    def tree_unflatten(
        cls, aux: Tuple[Any, ...], children: Tuple[Any, ...]
    ) -> 'GauLSSResult':
        (n_obs,) = aux
        coef_mu, coef_scale, cov_mu, cov_scale, log_lik = children
        return cls(
            coef_mu=coef_mu,
            coef_scale=coef_scale,
            cov_mu=cov_mu,
            cov_scale=cov_scale,
            log_lik=log_lik,
            n_obs=n_obs,
        )


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
    """Single-element Gaussian location-scale fit (block Fisher scoring)."""
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
        beta_s = beta_s + small_inv_logdet(
            2.0 * (Z.T @ Z) + eye_q, q
        )[0] @ (Z.T @ u)
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

    Fits, per element, ``y_v ~ N(mu, sigma^2)`` with ``mu = X beta_mu`` and
    ``log sigma = Z beta_scale`` -- modelling heteroscedasticity directly -- by
    joint Fisher scoring.

    Parameters
    ----------
    Y
        ``(V, N)`` responses.
    X
        ``(N, p)`` mean-model design (shared; carries its own intercept).
    scale_design
        ``(N, q)`` log-scale design ``Z`` (shared).  Defaults to an
        intercept-only column -- a constant variance (the homoscedastic Gaussian,
        recovering OLS for the mean); pass a design to model the variance.
    n_iter
        Fisher-scoring iterations (default ``50``).
    ridge
        Small stabiliser on the normal equations.
    block
        Optional element-block size bounding peak memory.

    Returns
    -------
    ``GauLSSResult`` -- ``coef_mu``, ``coef_scale`` (log-sd model),
    ``cov_mu`` / ``cov_scale`` (the block information inverses), ``log_lik``.
    A mean contrast is ``c^T coef_mu`` with SE ``sqrt(c^T cov_mu c)``; a scale
    contrast is the analogous read-out on ``coef_scale``.
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
