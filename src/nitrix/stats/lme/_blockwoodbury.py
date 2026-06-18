# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Block-Woodbury REML for a single grouping factor with a correlated random
effect (v3 §1.1, tier R2).

``reml_fit`` (tier R1) is the FaST-LMM path for one *scalar* random effect
``(1 | g)``: a single rotation diagonalises ``V`` and the per-voxel work is
``O(N)``.  That trick does **not** extend to a correlated ``(1 + x | g)`` (an
``r x r`` unstructured within-group covariance ``G``) -- but for a *single*
grouping factor ``V`` is still **block-diagonal across groups**, so a per-group
Woodbury keeps the heavy algebra at the ``r x r`` (``r <= 3``) and ``(p, p)``
scale.  This is the tier-R2 solver the dispatcher (``lme_fit``) routes a
correlated / diagonal random slope onto; it never forms an ``N x N`` matrix and
is cuSOLVER-free (every solve is a tiny ``small_inv_logdet``).

The model, per voxel, with ``M`` groups (group ``i`` has ``n_i`` rows, its
random covariates ``Z_i`` are ``(n_i, r)`` -- e.g. ``[1, x]`` for ``(1 + x)``)::

    V = blockdiag_i ( sigma_e^2 I_{n_i} + Z_i G Z_i^T ),   b_i ~ N(0, G)

Per-group Woodbury with ``K_i = sigma_e^2 G^{-1} + Z_i^T Z_i`` (``r x r``)::

    X_i^T V_i^{-1} X_i = sigma_e^{-2} (X_i^T X_i - (X_i^T Z_i) K_i^{-1} (Z_i^T X_i))
    log|V_i|           = (n_i - r) log sigma_e^2 + log|G| + log|K_i|

so the whole REML objective is assembled from per-group Gram reductions
(``Z_i^T Z_i``, ``X_i^T Z_i`` -- shared across voxels; ``Z_i^T y_i`` -- per
voxel) and the ``(p, p)`` fixed-effect solve.  ``G`` is carried in **log-Cholesky**
coordinates (``G = L L^T``, ``log`` diagonal) so it stays positive-definite under
an unconstrained Newton step.  Derivatives are by autodiff through this
closed-form objective (small parameter vector ``r(r+1)/2 + 1``); the per-voxel
Newton mirrors ``_varcomp`` (damping, step clip, backtracking).
"""

from __future__ import annotations

from typing import Tuple, cast

import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Float, Int

from .._batching import blocked_vmap as _blocked_vmap
from .._smalllinalg import small_inv_logdet as _small_inv_logdet
from ._varcomp import VarCompSpec

__all__ = ['fit_blockwoodbury_reml', 'group_grams']


def _tril_layout(r: int) -> Tuple[Tuple[int, int], ...]:
    """Row-major lower-triangular ``(i, j)`` index pairs of an ``r x r`` factor."""
    return tuple((i, j) for i in range(r) for j in range(i + 1))


def _build_chol(chol_params: Float[Array, 'm'], r: int) -> Float[Array, 'r r']:
    """Lower-triangular Cholesky factor ``L`` from its free parameters.

    Diagonal entries are exponentiated (positive), off-diagonal entries are
    free -- so ``G = L L^T`` is positive-definite for any real ``chol_params``.
    The loop is over the static ``r(r+1)/2`` layout (unrolled; ``r`` is tiny).
    """
    L = jnp.zeros((r, r), dtype=chol_params.dtype)
    for k, (i, j) in enumerate(_tril_layout(r)):
        val = jnp.exp(chol_params[k]) if i == j else chol_params[k]
        L = L.at[i, j].set(val)
    return L


def group_grams(
    X: Float[Array, 'N p'],
    Z: Float[Array, 'N r'],
    group: Int[Array, 'N'],
    n_groups: int,
) -> Tuple[
    Float[Array, 'M r r'], Float[Array, 'M p r'], Float[Array, 'p p'], int
]:
    """Per-group Gram reductions shared across voxels (data-independent of ``y``).

    Returns ``(ztz, xtz, xtx, n_minus_Mr)`` with ``ztz[i] = Z_i^T Z_i`` (``M, r,
    r``), ``xtz[i] = X_i^T Z_i`` (``M, p, r``), ``xtx = X^T X`` (``p, p``), and
    the scalar ``N - M r`` (the residual multiplicity in ``log|V|``).
    """
    r = Z.shape[-1]
    zz = Z[:, :, None] * Z[:, None, :]  # (N, r, r)
    xz = X[:, :, None] * Z[:, None, :]  # (N, p, r)
    ztz = jax.ops.segment_sum(zz, group, num_segments=n_groups)
    xtz = jax.ops.segment_sum(xz, group, num_segments=n_groups)
    xtx = X.T @ X
    n_minus_mr = X.shape[0] - n_groups * r
    return ztz, xtz, xtx, n_minus_mr


def _nll_and_beta(
    theta: Float[Array, 'nt'],
    ztz: Float[Array, 'M r r'],
    xtz: Float[Array, 'M p r'],
    xtx: Float[Array, 'p p'],
    zty: Float[Array, 'M r'],
    xty: Float[Array, 'p'],
    yty: Float[Array, ''],
    n_minus_mr: int,
    r: int,
    p: int,
    ridge: float,
) -> Tuple[Float[Array, ''], Float[Array, 'p']]:
    """Profile REML negative log-likelihood (and ``beta_hat``) at ``theta``."""
    se2 = jnp.exp(theta[-1])
    L = _build_chol(theta[:-1], r)
    g_cov = L @ L.T
    g_inv, logdet_g = _small_inv_logdet(g_cov, r)

    # Per-group K_i = sigma_e^2 G^{-1} + Z_i^T Z_i  (M, r, r).
    k_mat = se2 * g_inv[None] + ztz
    k_inv, logdet_k = jax.vmap(lambda a: _small_inv_logdet(a, r))(k_mat)

    xtz_kinv = jnp.einsum('mpr,mrs->mps', xtz, k_inv)  # (M, p, r)
    a = (xtx - jnp.einsum('mpr,mqr->pq', xtz_kinv, xtz)) / se2
    a = a + ridge * jnp.eye(p, dtype=a.dtype)
    a_inv, logdet_a = _small_inv_logdet(a, p)
    b = (xty - jnp.einsum('mpr,mr->p', xtz_kinv, zty)) / se2
    beta = a_inv @ b

    y_vinv_y = (yty - jnp.einsum('mr,mrs,ms->', zty, k_inv, zty)) / se2
    rss = y_vinv_y - beta @ b
    logdet_v = (
        n_minus_mr * theta[-1] + ztz.shape[0] * logdet_g + jnp.sum(logdet_k)
    )
    nll = 0.5 * (logdet_v + logdet_a + rss)
    return nll, beta


def _fit_one(
    zty: Float[Array, 'M r'],
    xty: Float[Array, 'p'],
    yty: Float[Array, ''],
    theta_init: Float[Array, 'nt'],
    ztz: Float[Array, 'M r r'],
    xtz: Float[Array, 'M p r'],
    xtx: Float[Array, 'p p'],
    n_minus_mr: int,
    r: int,
    p: int,
    spec: VarCompSpec,
) -> Tuple[Float[Array, 'nt'], Float[Array, 'p'], Float[Array, '']]:
    """Single-voxel block-Woodbury REML fit via damped autodiff-Newton."""
    nt = theta_init.shape[0]

    def nll(theta: Float[Array, 'nt']) -> Float[Array, '']:
        return _nll_and_beta(
            theta, ztz, xtz, xtx, zty, xty, yty, n_minus_mr, r, p, spec.ridge
        )[0]

    grad_fn = jax.grad(nll)
    hess_fn = jax.hessian(nll)

    def newton(theta: Float[Array, 'nt'], _: Array) -> Tuple[Array, None]:
        g = grad_fn(theta)
        h = hess_fn(theta) + spec.damping * jnp.eye(nt, dtype=theta.dtype)
        h_inv, _ = _small_inv_logdet(h, nt)
        delta = jnp.clip(h_inv @ g, -spec.max_step, spec.max_step)
        nll_old = nll(theta)

        def bt(
            _: Array, carry: Tuple[Array, Array, Array]
        ) -> Tuple[Array, Array, Array]:
            scale, best, best_nll = carry
            trial = theta - scale * delta
            trial_nll = nll(trial)
            ok = trial_nll < best_nll
            return (
                scale * 0.5,
                jnp.where(ok, trial, best),
                jnp.where(ok, trial_nll, best_nll),
            )

        init = (jnp.asarray(1.0, theta.dtype), theta, nll_old)
        _, theta_new, _ = lax.fori_loop(0, spec.n_backtrack, bt, init)
        return theta_new, None

    theta, _ = lax.scan(newton, theta_init, xs=None, length=spec.n_iter)
    final_nll, beta = _nll_and_beta(
        theta, ztz, xtz, xtx, zty, xty, yty, n_minus_mr, r, p, spec.ridge
    )
    return theta, beta, -final_nll


def fit_blockwoodbury_reml(
    Y: Float[Array, 'V N'],
    X: Float[Array, 'N p'],
    Z: Float[Array, 'N r'],
    group: Int[Array, 'N'],
    n_groups: int,
    theta_init: Float[Array, 'V nt'],
    *,
    spec: VarCompSpec = VarCompSpec(),
    block: int | None = None,
) -> Tuple[Float[Array, 'V nt'], Float[Array, 'V p'], Float[Array, 'V']]:
    """Batched block-Woodbury REML over ``V`` voxels (single grouping factor).

    ``Z`` is the ``(N, r)`` per-observation random-covariate design (e.g.
    ``[1, x]`` for ``(1 + x | g)``); ``group`` is the ``(N,)`` group label.
    ``theta`` is ``[chol(G) params, log sigma_e^2]`` (``nt = r(r+1)/2 + 1``).
    Returns ``(theta_hat, beta_hat, log_lik)``; recover ``G = L L^T`` from the
    Cholesky params and ``sigma_e^2 = exp(theta[-1])``.
    """
    p = X.shape[-1]
    r = Z.shape[-1]
    ztz, xtz, xtx, n_minus_mr = group_grams(X, Z, group, n_groups)

    def per_voxel(
        y: Float[Array, 'N'], th: Float[Array, 'nt']
    ) -> Tuple[Float[Array, 'nt'], Float[Array, 'p'], Float[Array, '']]:
        zty = jax.ops.segment_sum(
            Z * y[:, None], group, num_segments=n_groups
        )  # (M, r)
        xty = X.T @ y  # (p,)
        yty = y @ y
        return _fit_one(
            zty, xty, yty, th, ztz, xtz, xtx, n_minus_mr, r, p, spec
        )

    return cast(
        Tuple[Float[Array, 'V nt'], Float[Array, 'V p'], Float[Array, 'V']],
        _blocked_vmap(per_voxel, (Y, theta_init), block=block),
    )
