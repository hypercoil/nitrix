# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
FLAME-style two-level mixed-effects model for fMRI group analysis.

The model
---------

At level 1, each subject's BOLD time series has been fit by a
GLM, yielding a per-voxel estimate ``beta_i`` of the per-subject
effect (e.g., the activation magnitude on a task contrast) plus
its within-subject standard error squared ``s_i^2`` (typically
the OLS residual variance from the level-1 fit).  These are the
inputs.

At level 2, the per-voxel group model is::

    beta_i = X_group gamma + b_i + eps_i
    b_i ~ N(0, sigma_b^2)        (between-subject variance, unknown)
    eps_i ~ N(0, s_i^2)          (within-subject variance, known)

So the total per-subject variance is ``sigma_b^2 + s_i^2`` --
heteroscedastic by subject.  ``gamma`` is the group-level
fixed-effect vector.

This is the **FSL FLAME** model (Beckmann, Jenkinson, Smith
2003).  Only ``sigma_b^2`` is estimated; ``gamma`` is profiled
out analytically; ``s_i^2`` is known.  The implementation is a
**single-parameter REML**: Newton iteration on a scalar log-
variance.  Avoids the identifiability problem of the generic
two-parameter REML (where ``sigma_b^2`` and a free scaling on
``var_within`` trade off).

Computational structure
-----------------------

Per voxel, the covariance ``V_v = sigma_b^2 I + diag(s_v^2)`` is
naturally diagonal.  Each Newton step is ``O(N p^2)``.  No
matrix factorisation of ``V`` is needed (it's diagonal); the
``(p, p)`` Cholesky for the fixed-effect normal equations is
the only matrix solve.  The 1-D Newton on log-variance is a
scalar division.

Memory: ``Y`` (V, N) + ``var_within`` (V, N) + per-voxel
parameters.  No ``(V, N, N)`` intermediate.

Reference
---------
Beckmann, C. F., Jenkinson, M., & Smith, S. M. (2003).  General
multilevel linear modeling for group analysis in fMRI.
NeuroImage 20, 1052-1063.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsla
from jaxtyping import Array, Float

__all__ = ['FLAMEResult', 'flame_two_level']


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class FLAMEResult:
    """Per-voxel FLAME fit output."""

    sigma_b_sq: Float[Array, 'V']
    gamma_hat: Float[Array, 'V p']
    log_lik: Float[Array, 'V']

    def tree_flatten(
        self,
    ) -> Tuple[
        Tuple[Float[Array, 'V'], Float[Array, 'V p'], Float[Array, 'V']],
        None,
    ]:
        return (self.sigma_b_sq, self.gamma_hat, self.log_lik), None

    @classmethod
    def tree_unflatten(
        cls, _aux: None, children: Tuple[Any, ...]
    ) -> 'FLAMEResult':
        return cls(*children)


def _flame_neg_loglik(
    log_sigma_b_sq: Float[Array, ''],
    beta: Float[Array, 'N'],
    X_group: Float[Array, 'N p'],
    var_within: Float[Array, 'N'],
    ridge: float = 1e-8,
) -> Float[Array, '']:
    """Profile REML negative log-likelihood for the FLAME model.

    Single parameter: ``log(sigma_b^2)``.  Within-variance
    ``var_within`` is treated as known (per FLAME convention);
    ``gamma`` profiled out analytically.
    """
    sigma_b_sq = jnp.exp(log_sigma_b_sq)
    d = sigma_b_sq + var_within
    inv_d = 1.0 / d
    Xw = X_group * inv_d[:, None]
    XtVinvX = Xw.T @ X_group
    p = XtVinvX.shape[-1]
    XtVinvX = XtVinvX + ridge * jnp.eye(p, dtype=XtVinvX.dtype)
    XtVinvy = Xw.T @ beta
    L = jnp.linalg.cholesky(XtVinvX)
    z = jsla.solve_triangular(L, XtVinvy, lower=True)
    gamma = jsla.solve_triangular(L.T, z, lower=False)
    log_det_V = jnp.sum(jnp.log(d))
    log_det_XtVinvX = 2.0 * jnp.sum(jnp.log(jnp.diagonal(L)))
    r = beta - X_group @ gamma
    rwss = jnp.sum(r * r * inv_d)
    return 0.5 * (log_det_V + log_det_XtVinvX + rwss)


def _flame_newton_step(
    log_sigma_b_sq: Float[Array, ''],
    beta: Float[Array, 'N'],
    X_group: Float[Array, 'N p'],
    var_within: Float[Array, 'N'],
    damping: float = 1e-6,
    max_step: float = 1.0,
    n_backtrack: int = 4,
) -> Float[Array, '']:
    """1-D Newton step on the FLAME negative log-likelihood with
    step clipping and backtracking.

    Since the parameter is scalar, the Hessian is a scalar and
    Newton degenerates to ``new = old - grad / hess``.  Clipped
    to ``[-max_step, max_step]`` to prevent overshoot; backtracked
    if the new nll did not decrease.
    """
    val_and_grad = jax.value_and_grad(_flame_neg_loglik)
    nll_old, grad = val_and_grad(
        log_sigma_b_sq,
        beta,
        X_group,
        var_within,
    )
    hess = jax.grad(jax.grad(_flame_neg_loglik))(
        log_sigma_b_sq,
        beta,
        X_group,
        var_within,
    )
    hess_pos = jnp.maximum(hess, damping)
    delta = grad / hess_pos
    delta = jnp.clip(delta, -max_step, max_step)

    def body(
        carry: Tuple[Float[Array, ''], Float[Array, ''], Float[Array, '']],
        _: Any,
    ) -> Tuple[
        Tuple[Float[Array, ''], Float[Array, ''], Float[Array, '']], None
    ]:
        scale, x_best, nll_best = carry
        x_try = log_sigma_b_sq - scale * delta
        nll_try = _flame_neg_loglik(
            x_try,
            beta,
            X_group,
            var_within,
        )
        accept = nll_try < nll_best
        x_new = jnp.where(accept, x_try, x_best)
        nll_new = jnp.where(accept, nll_try, nll_best)
        return (scale * 0.5, x_new, nll_new), None

    init = (
        jnp.asarray(1.0, dtype=log_sigma_b_sq.dtype),
        log_sigma_b_sq,
        nll_old,
    )
    (_, x_final, _), _ = jax.lax.scan(
        body,
        init,
        jnp.arange(n_backtrack),
    )
    return x_final


def _flame_fit_one_voxel(
    beta: Float[Array, 'N'],
    var_within: Float[Array, 'N'],
    X_group: Float[Array, 'N p'],
    log_sigma_b_sq_init: Float[Array, ''],
    n_iter: int,
    damping: float,
) -> Tuple[Float[Array, ''], Float[Array, 'p'], Float[Array, '']]:
    """Single-voxel FLAME fit via 1-D Newton.

    Returns ``(log_sigma_b_sq, gamma, log_lik)``.
    """

    def step(
        log_sigma: Float[Array, ''], _: Any
    ) -> Tuple[Float[Array, ''], None]:
        return _flame_newton_step(
            log_sigma,
            beta,
            X_group,
            var_within,
            damping,
        ), None

    log_final, _ = jax.lax.scan(
        step,
        log_sigma_b_sq_init,
        jnp.arange(n_iter),
    )

    # Compute final gamma at the converged sigma_b_sq.
    sigma_b_sq = jnp.exp(log_final)
    d = sigma_b_sq + var_within
    inv_d = 1.0 / d
    Xw = X_group * inv_d[:, None]
    XtVinvX = Xw.T @ X_group
    p = XtVinvX.shape[-1]
    XtVinvX = XtVinvX + 1e-8 * jnp.eye(p, dtype=XtVinvX.dtype)
    XtVinvy = Xw.T @ beta
    L = jnp.linalg.cholesky(XtVinvX)
    z = jsla.solve_triangular(L, XtVinvy, lower=True)
    gamma = jsla.solve_triangular(L.T, z, lower=False)
    nll = _flame_neg_loglik(log_final, beta, X_group, var_within)
    return log_final, gamma, -nll


def _flame_default_log_init(
    beta: Float[Array, 'V N'],
    var_within: Float[Array, 'V N'],
) -> Float[Array, 'V']:
    """Initial guess for ``log(sigma_b^2)`` per voxel.

    Heuristic: residual variance of ``beta`` after the per-subject
    weighted mean, minus the mean within-variance.  Clamped to
    a small positive floor so the log is finite.
    """
    # Empirical between-subject variance (per voxel, ignoring X)
    beta_mean = jnp.mean(beta, axis=-1, keepdims=True)
    var_total = jnp.mean((beta - beta_mean) ** 2, axis=-1)
    var_within_mean = jnp.mean(var_within, axis=-1)
    var_b_init = jnp.maximum(var_total - var_within_mean, 1e-4)
    return jnp.log(var_b_init)


def flame_two_level(
    beta_subject: Float[Array, 'V N'],
    var_within: Float[Array, 'V N'],
    X_group: Float[Array, 'N p'],
    *,
    log_sigma_b_sq_init: Optional[Float[Array, 'V']] = None,
    n_iter: int = 30,
    damping: float = 1e-6,
) -> FLAMEResult:
    """Voxelwise FLAME-style two-level fixed-effect group model.

    Parameters
    ----------
    beta_subject
        Per-voxel, per-subject level-1 effect estimates.
        Shape ``(V, N)``.
    var_within
        Per-voxel, per-subject within-subject variance (the
        squared standard error from the level-1 GLM).
        Shape ``(V, N)``.  All entries must be positive.
    X_group
        Group-level fixed-effect design, ``(N, p)``.  Shared across
        voxels.
    log_sigma_b_sq_init
        Per-voxel initial ``log(sigma_b^2)``.  Default heuristic:
        method-of-moments residual variance minus mean within-
        variance.
    n_iter
        Newton iterations.  Default ``30``.
    damping
        Hessian positivity guard.

    Returns
    -------
    ``FLAMEResult`` with ``sigma_b_sq``, ``gamma_hat``, ``log_lik``.

    Notes
    -----
    Single-parameter REML: only ``sigma_b^2`` is estimated;
    ``var_within`` is fixed at the user-supplied values.  This
    avoids the identifiability problem of the two-parameter
    relaxation (where the model can absorb a free scaling of
    ``var_within`` into ``sigma_b^2``).
    """
    if beta_subject.shape != var_within.shape:
        raise ValueError(
            f'flame_two_level: beta_subject.shape={beta_subject.shape} '
            f'must equal var_within.shape={var_within.shape}.'
        )
    N = beta_subject.shape[-1]
    if X_group.shape[0] != N:
        raise ValueError(
            f'flame_two_level: X_group.shape[0]={X_group.shape[0]} '
            f'must match N={N}.'
        )

    if log_sigma_b_sq_init is None:
        log_sigma_b_sq_init = _flame_default_log_init(
            beta_subject,
            var_within,
        )

    fit = jax.vmap(
        _flame_fit_one_voxel,
        in_axes=(0, 0, None, 0, None, None),
    )
    log_sigma_b_sq, gamma_hat, log_lik = fit(
        beta_subject,
        var_within,
        X_group,
        log_sigma_b_sq_init,
        n_iter,
        damping,
    )
    return FLAMEResult(
        sigma_b_sq=jnp.exp(log_sigma_b_sq),
        gamma_hat=gamma_hat,
        log_lik=log_lik,
    )
