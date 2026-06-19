# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
The single penalised-IRLS core shared by GLM and GAM.

Iteratively-reweighted least squares for an exponential ``Family``, solving the
penalised normal equations ``(X^T W X + S) beta = X^T W z`` over a working
response ``z`` and working weights ``W``.  ``glm`` calls it with the penalty
``S = 0`` and prior observation weights; ``gam`` calls it with the block
smoothing penalty ``S(lambda)`` and no prior weights.  v1 had two drifted copies
(the three-lens review flagged it -- one used a shared ``_EPS``, the other a
literal); this is the one source of truth.

Every ``(p, p)`` solve goes through the cuSOLVER-free ``small_inv_logdet``.  The
converged ``V = (X^T W X + S + ridge)^{-1}`` and the unpenalised Gram
``X^T W X`` are evaluated **at the converged ``beta``** (the correct point for
the coefficient covariance and the GAM EDF / Fellner-Schall traces) -- that
final evaluation is intentional, not redundant.
"""

from __future__ import annotations

from typing import Optional, Tuple, cast

import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Float

from ..linalg._smalllinalg import small_inv_logdet
from ._family import Family

__all__ = ['fit_penalised_irls', 'irls_warm_start']

_EPS = 1e-10


def _working(
    beta: Float[Array, 'p'],
    y: Float[Array, 'N'],
    X: Float[Array, 'N p'],
    family: Family,
    prior_weights: Optional[Float[Array, 'N']],
) -> Tuple[Float[Array, 'N'], Float[Array, 'N'], Float[Array, 'N']]:
    """IRLS working weights ``W``, working response ``z``, and fitted ``mu``."""
    # Clamp the linear predictor to the family's numerically-sane range: for the
    # unbounded exp links a transient overshoot otherwise blows up exp(eta) and
    # the working weights (a single observation dominates -> garbage / NaN); the
    # bounded links carry eta_bound = inf, so this is a no-op for them.
    eta = jnp.clip(X @ beta, -family.eta_bound, family.eta_bound)
    mu = family.linkinv(eta)
    dmu = family.mu_eta(eta)
    var = family.variance(mu)
    wts = dmu * dmu / jnp.clip(var, _EPS, None)
    if prior_weights is not None:
        wts = prior_weights * wts
    # Floor |dmu| away from zero *preserving sign*: a decreasing link (the
    # reciprocal / inverse link has dmu = -1/eta^2 < 0) would otherwise have its
    # derivative flipped to +_EPS by a naive clip, exploding the working response.
    safe_dmu = jnp.where(
        dmu < 0.0, jnp.minimum(dmu, -_EPS), jnp.maximum(dmu, _EPS)
    )
    z = eta + (y - mu) / safe_dmu
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
    """A warm-start ``beta`` from the (penalised, weighted) OLS fit of the
    linked initial mean ``link(init_mu(y))`` -- the GLM IRLS start."""
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
    """Penalised IRLS from a warm start.

    Returns ``(beta, V, xtwx, deviance)`` -- the coefficients, the Bayesian /
    unscaled covariance ``V = (X^T W X + S + ridge)^{-1}``, the unpenalised Gram
    ``X^T W X``, and the model deviance, all at the converged ``beta``.
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
