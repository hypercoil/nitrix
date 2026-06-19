# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Regularised covariance / sparse-precision estimators for connectivity.

The raw empirical covariance (``stats.covariance.cov``) is noisy and may be
singular when the number of variables approaches the number of observations
(``p ~ n`` -- the small-sample regime of resting-state connectomes).  This
module ships the regularised estimators the connectome literature defaults to:

- **Analytic shrinkage** -- Ledoit-Wolf (2004) and OAS (Chen et al. 2010), a
  convex blend of the sample covariance ``S`` toward a scaled identity::

      Sigma_hat = (1 - alpha) S + alpha mu I,    mu = tr(S) / p

  with the shrinkage intensity ``alpha`` in **closed form** (no
  cross-validation).  Ledoit-Wolf is nilearn's *default* connectome covariance
  estimator, so this is the missing piece for a nilearn-default-equivalent
  ``precision`` / ``partialcorr`` path (invert ``Sigma_hat`` with the
  cuSOLVER-free ``_smalllinalg`` solve).

- **Graphical LASSO** -- the sparse inverse covariance (precision) that is the
  conditional-independence graph the fMRI literature has used for ~15 years::

      Theta_hat = argmin_Theta  <S, Theta> - log det Theta + lam ||Theta||_{1,off}

  solved by Friedman/Hastie/Tibshirani (2008) coordinate descent on the working
  covariance ``W = Theta^-1`` directly (no per-iteration factorisation), with
  ``glasso_path`` for a warm-started ``lam`` sweep and ``ebic_score`` for
  extended-BIC model selection (Foygel & Drton 2010).

Pure JAX -- the shrinkage path is trace / Frobenius reductions + one scalar
``alpha``; the GLASSO path is rolled coordinate descent (``lax.fori_loop`` /
``lax.scan``, so the graph stays ``O(p^2)`` and compile is flat in ``p``).
Differentiable and GPU-resident; cuSOLVER-free (the ``log det`` for EBIC goes
through the rolled Cholesky in ``_smalllinalg``).  ``vmap`` over subjects /
edges for batches.  ``X`` is ``(n_samples, n_features)`` (the
``sklearn.covariance`` convention); ``glasso`` consumes a ``(p, p)`` sample
covariance ``S`` directly (the FHT convention).
"""

from __future__ import annotations

from typing import Literal, Tuple, cast

import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Float

from ..linalg._smalllinalg import small_inv_logdet

__all__ = [
    'ebic_score',
    'glasso',
    'glasso_path',
    'ledoit_wolf',
    'oas',
    'shrunk_covariance',
]

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
    raise ValueError(f"method={method!r}; expected 'ledoit_wolf' or 'oas'.")


# ---------------------------------------------------------------------------
# Graphical LASSO -- sparse precision via FHT (2008) coordinate descent.
# ---------------------------------------------------------------------------


def _soft(x: Float[Array, '...'], t: Float[Array, '']) -> Float[Array, '...']:
    """Soft-thresholding ``sign(x) * max(|x| - t, 0)`` (the lasso proximal op)."""
    return jnp.sign(x) * jnp.clip(jnp.abs(x) - t, 0.0, None)


def _glasso_wb(
    S: Float[Array, 'p p'],
    lam: Float[Array, ''],
    W: Float[Array, 'p p'],
    B: Float[Array, 'p p'],
    n_outer: int,
    n_inner: int,
) -> Tuple[Float[Array, 'p p'], Float[Array, 'p p']]:
    """Coordinate-descent core: converged working covariance ``W`` and the
    per-column lasso coefficients ``B`` (column ``j`` is the regression of node
    ``j`` on the rest).  Every loop is **rolled** (``lax.fori_loop``) so the
    graph stays ``O(p^2)`` and compile is flat in ``p`` (the rolled-Cholesky
    lesson).  ``W``'s diagonal is held fixed (off-diagonal-only penalty:
    ``W_jj = S_jj``), so seeding ``W = S`` enforces the diagonal KKT condition.
    """
    p = S.shape[0]
    idx = jnp.arange(p)

    def per_var(
        j: Array, carry: Tuple[Float[Array, 'p p'], Float[Array, 'p p']]
    ) -> Tuple[Float[Array, 'p p'], Float[Array, 'p p']]:
        W, B = carry
        beta0 = B[:, j]

        def coord(k: Array, beta: Float[Array, 'p']) -> Float[Array, 'p']:
            # Lasso coordinate update for the j-th column regression; the j-th
            # coordinate is pinned to 0 (no self-edge in the off-diagonal block).
            r = S[k, j] - W[k] @ beta + W[k, k] * beta[k]
            return beta.at[k].set(
                jnp.where(k == j, 0.0, _soft(r, lam) / W[k, k])
            )

        def sweep(_: Array, beta: Float[Array, 'p']) -> Float[Array, 'p']:
            return cast(Float[Array, 'p'], lax.fori_loop(0, p, coord, beta))

        beta = lax.fori_loop(0, n_inner, sweep, beta0)
        # w_12 <- W_11 beta; the diagonal w_jj stays at S_jj (held fixed).
        newcol = jnp.where(idx == j, W[j, j], W @ beta)
        W = W.at[:, j].set(newcol).at[j, :].set(newcol)
        B = B.at[:, j].set(beta)
        return W, B

    def outer(
        _: Array, carry: Tuple[Float[Array, 'p p'], Float[Array, 'p p']]
    ) -> Tuple[Float[Array, 'p p'], Float[Array, 'p p']]:
        return cast(
            Tuple[Float[Array, 'p p'], Float[Array, 'p p']],
            lax.fori_loop(0, p, per_var, carry),
        )

    return cast(
        Tuple[Float[Array, 'p p'], Float[Array, 'p p']],
        lax.fori_loop(0, n_outer, outer, (W, B)),
    )


def _theta_from_wb(
    W: Float[Array, 'p p'], B: Float[Array, 'p p']
) -> Float[Array, 'p p']:
    """Recover the sparse precision ``Theta`` from the working covariance and
    the per-column lasso coefficients (FHT 2008, eq. partitioned-inverse): for
    column ``j``, ``theta_jj = 1 / (w_jj - w_12^T beta_j)`` and
    ``theta_-j,j = -beta_j theta_jj``.  Sparsity is inherited from ``B`` (the
    soft-threshold zeros exactly), so ``Theta`` carries the conditional-
    independence support, not a dense ``W^-1``.
    """
    p = W.shape[0]
    idx = jnp.arange(p)

    def rec(j: Array) -> Float[Array, 'p']:
        beta = B[:, j]
        t22 = 1.0 / (W[j, j] - W[:, j] @ beta)
        return (-beta * t22).at[j].set(t22)

    return jax.vmap(rec)(idx).T


def glasso(
    S: Float[Array, 'p p'],
    lam: float | Float[Array, ''],
    *,
    n_outer: int = 100,
    n_inner: int = 50,
) -> Float[Array, 'p p']:
    """Graphical-LASSO sparse precision (inverse covariance).

    Solves ``argmin_Theta <S, Theta> - log det Theta + lam ||Theta||_{1,off}``
    (off-diagonal-only L1 penalty -- the standard / nilearn convention) by
    Friedman/Hastie/Tibshirani (2008) block-coordinate descent: an outer sweep
    over columns, each an inner lasso on the working covariance ``W``, with the
    precision recovered from the converged per-column coefficients.  No
    factorisation, no cuSOLVER; all loops rolled (``O(p^2)`` graph), so it is
    differentiable through the fixed iteration budget and compiles flat in ``p``
    (fine for connectome ``p = 100-400``).

    Parameters
    ----------
    S
        ``(p, p)`` sample covariance (e.g. ``stats.covariance.cov`` or a
        shrinkage estimate).  Off-diagonal-only penalty fixes ``W_jj = S_jj``.
    lam
        Non-negative L1 penalty.  Larger ``lam`` -> sparser ``Theta``.
    n_outer, n_inner
        Outer column sweeps and inner lasso sweeps per column (fixed budget; the
        defaults reach KKT stationarity to machine precision on well-scaled
        connectome covariances and can be reduced for speed).

    Returns
    -------
    Theta
        ``(p, p)`` symmetric sparse precision; the inactive off-diagonals are
        exactly zero (the conditional-independence graph).
    """
    p = S.shape[0]
    lam = jnp.asarray(lam, dtype=S.dtype)
    B0 = jnp.zeros((p, p), S.dtype)
    W, B = _glasso_wb(S, lam, S, B0, n_outer, n_inner)
    return _theta_from_wb(W, B)


def glasso_path(
    S: Float[Array, 'p p'],
    lambdas: Float[Array, 'L'],
    *,
    n_outer: int = 100,
    n_inner: int = 50,
) -> Float[Array, 'L p p']:
    """Warm-started graphical-LASSO regularisation path over ``lambdas``.

    Sweeps the penalties in the **given order**, carrying the working covariance
    ``W`` and the lasso coefficients ``B`` from one ``lam`` to the next so each
    solve starts warm (far cheaper than a cold restart and the usual way to
    trace a glasso path -- pass ``lambdas`` descending, dense -> sparse, or
    ascending; either direction warm-starts).  Returns the stacked precisions
    aligned with ``lambdas``; pair with :func:`ebic_score` for model selection.
    """
    p = S.shape[0]
    lambdas = jnp.asarray(lambdas, dtype=S.dtype)
    B0 = jnp.zeros((p, p), S.dtype)

    def step(
        carry: Tuple[Float[Array, 'p p'], Float[Array, 'p p']],
        lam: Float[Array, ''],
    ) -> Tuple[
        Tuple[Float[Array, 'p p'], Float[Array, 'p p']], Float[Array, 'p p']
    ]:
        W, B = carry
        W, B = _glasso_wb(S, lam, W, B, n_outer, n_inner)
        return (W, B), _theta_from_wb(W, B)

    _, thetas = lax.scan(step, (S, B0), lambdas)
    return thetas


def ebic_score(
    theta: Float[Array, 'p p'],
    S: Float[Array, 'p p'],
    n: int | Float[Array, ''],
    *,
    gamma: float = 0.5,
    edge_tol: float = 1e-8,
) -> Float[Array, '']:
    """Extended Bayesian Information Criterion of a precision estimate.

    ``EBIC_gamma = n (tr(S Theta) - log det Theta) + E log n + 4 gamma E log p``
    (Foygel & Drton 2010), where ``E`` is the number of off-diagonal edges
    (upper-triangle entries with ``|theta| > edge_tol``, counted once).  The
    Gaussian-graphical-model term ``-2 loglik = n (tr(S Theta) - log det Theta)``
    uses the cuSOLVER-free rolled-Cholesky ``log det``.  ``gamma in [0, 1]``
    tunes the extra sparsity prior (``gamma = 0`` recovers plain BIC); pick the
    path point with the **smallest** EBIC.
    """
    p = theta.shape[0]
    _, logdet = small_inv_logdet(theta, p)
    tr_s_theta = jnp.sum(S * theta)
    neg2ll = n * (tr_s_theta - logdet)
    offdiag = jnp.abs(theta) > edge_tol
    n_edges = jnp.sum(jnp.triu(offdiag, k=1))
    return neg2ll + n_edges * jnp.log(n) + 4.0 * gamma * n_edges * jnp.log(p)
