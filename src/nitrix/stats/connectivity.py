# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Regularised covariance estimators for connectivity.

The raw empirical covariance (``stats.covariance.cov``) is noisy and may be
singular when the number of variables approaches the number of observations
(``p ~ n`` -- the small-sample regime of resting-state connectomes).  This
module ships the **analytic shrinkage** estimators that stay well-conditioned
and invertible there: Ledoit-Wolf (2004) and OAS (Chen et al. 2010), a convex
blend of the sample covariance ``S`` toward a scaled identity ``mu I``::

    Sigma_hat = (1 - alpha) S + alpha mu I,    mu = tr(S) / p

with the shrinkage intensity ``alpha`` in **closed form** (no cross-validation).
Ledoit-Wolf is nilearn's *default* connectome covariance estimator, so this is
the missing piece for a nilearn-default-equivalent ``precision`` /
``partialcorr`` path (invert ``Sigma_hat`` with the cuSOLVER-free
``_smalllinalg`` solve).

Pure JAX -- trace / Frobenius reductions + one scalar ``alpha`` -- fully
differentiable and GPU-resident; ``vmap`` over subjects / edges for batches.
``X`` is ``(n_samples, n_features)`` (the ``sklearn.covariance`` convention).
"""

from __future__ import annotations

from typing import Literal, Tuple

import jax.numpy as jnp
from jaxtyping import Array, Float

__all__ = ['ledoit_wolf', 'oas', 'shrunk_covariance']

ShrinkageMethod = Literal['ledoit_wolf', 'oas']


def _empirical(
    X: Float[Array, 'n p'], assume_centered: bool
) -> Tuple[Float[Array, 'n p'], Float[Array, 'p p'], int, int]:
    """Centered data, the (biased, ``/n``) empirical covariance, and ``(n, p)``."""
    n, p = X.shape
    Xc = X if assume_centered else X - jnp.mean(X, axis=0, keepdims=True)
    s = (Xc.T @ Xc) / n
    return Xc, s, n, p


def _blend(
    s: Float[Array, 'p p'], alpha: Float[Array, ''], p: int
) -> Float[Array, 'p p']:
    mu = jnp.trace(s) / p
    return (1.0 - alpha) * s + alpha * mu * jnp.eye(p, dtype=s.dtype)


def ledoit_wolf(
    X: Float[Array, 'n p'],
    *,
    assume_centered: bool = False,
) -> Tuple[Float[Array, 'p p'], Float[Array, '']]:
    """Ledoit-Wolf analytic-shrinkage covariance.

    Returns ``(cov, shrinkage)``.  The shrinkage intensity is
    ``alpha = beta^2 / delta^2`` with ``delta^2 = ||S - mu I||_F^2`` and
    ``beta^2 = (1/n^2) sum_k ||x_k||^4 - (1/n) ||S||_F^2`` clipped to
    ``[0, delta^2]`` (Ledoit & Wolf 2004).  Matches
    ``sklearn.covariance.ledoit_wolf``.
    """
    Xc, s, n, p = _empirical(X, assume_centered)
    mu = jnp.trace(s) / p
    s_norm2 = jnp.sum(s * s)
    delta2 = s_norm2 - p * mu * mu
    sq_norms = jnp.sum(Xc * Xc, axis=1)  # ||x_k||^2
    beta2 = jnp.sum(sq_norms * sq_norms) / (n * n) - s_norm2 / n
    beta2 = jnp.clip(beta2, 0.0, delta2)
    alpha = jnp.where(delta2 > 0, beta2 / delta2, 0.0)
    return _blend(s, alpha, p), alpha


def oas(
    X: Float[Array, 'n p'],
    *,
    assume_centered: bool = False,
) -> Tuple[Float[Array, 'p p'], Float[Array, '']]:
    """Oracle-Approximating-Shrinkage covariance (Chen et al. 2010).

    Returns ``(cov, shrinkage)``.  Same convex blend as Ledoit-Wolf, a
    different closed-form ``alpha``.  Matches ``sklearn.covariance.oas``.
    """
    _, s, n, p = _empirical(X, assume_centered)
    mu = jnp.trace(s) / p
    s_sq_mean = jnp.mean(s * s)
    num = s_sq_mean + mu * mu
    den = (n + 1) * (s_sq_mean - mu * mu / p)
    alpha = jnp.where(den > 0, jnp.clip(num / den, 0.0, 1.0), 1.0)
    return _blend(s, alpha, p), alpha


def shrunk_covariance(
    X: Float[Array, 'n p'],
    *,
    method: ShrinkageMethod = 'ledoit_wolf',
    assume_centered: bool = False,
) -> Float[Array, 'p p']:
    """Analytic-shrinkage covariance via ``method`` (the cov only).

    ``'ledoit_wolf'`` (default, nilearn's connectome default) or ``'oas'``.
    """
    if method == 'ledoit_wolf':
        return ledoit_wolf(X, assume_centered=assume_centered)[0]
    if method == 'oas':
        return oas(X, assume_centered=assume_centered)[0]
    raise ValueError(
        f"method={method!r}; expected 'ledoit_wolf' or 'oas'."
    )
