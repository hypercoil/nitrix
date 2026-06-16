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
naturally diagonal, so FLAME is the shared diagonalised-REML fit
(``_varcomp``) with a single free component ``B = [ones]`` and a
fixed per-coordinate ``offset = var_within`` (the known
within-variance).  The fit makes **no per-voxel cuSOLVER call**:
the fixed-effect algebra is closed-form for the dominant ``p in
{1, 2}`` designs (the intercept ``p = 1`` is a scalar) and the REML
score / curvature are analytic (no second-order autodiff through a
Cholesky).  That is what unblocks ``flame_two_level`` on the
broken-cuSOLVER GPU and flattens its linear-in-``V`` compile.

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
from jaxtyping import Array, Float

from ._varcomp import VarCompSpec, fit_varcomp_diagonal

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
    block: Optional[int] = None,
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
        Levenberg guard on the (scalar) average-information curvature.
    block
        Optional voxel-block size bounding peak memory on brain-scale
        ``V`` (see ``reml_fit``).  ``None`` (default) is a single
        ``vmap`` over all voxels.

    Returns
    -------
    ``FLAMEResult`` with ``sigma_b_sq``, ``gamma_hat``, ``log_lik``.

    Notes
    -----
    Single-parameter REML: only ``sigma_b^2`` is estimated;
    ``var_within`` is fixed at the user-supplied values.  This
    avoids the identifiability problem of the two-parameter
    relaxation (where the model can absorb a free scaling of
    ``var_within`` into ``sigma_b^2``).  Implemented as the shared
    ``_varcomp`` diagonalised-REML fit with one free component
    (``B = [ones]``) and ``offset = var_within`` -- no per-voxel
    cuSOLVER call, so it runs on the broken-cuSOLVER GPU.
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

    # Single free variance component (between-subject); the known
    # within-variance enters as the fixed per-voxel diagonal offset.
    basis = jnp.ones((1, N), dtype=beta_subject.dtype)
    theta_init = log_sigma_b_sq_init[:, None]  # (V, 1)
    spec = VarCompSpec.flame(n_iter=n_iter, damping=damping)
    theta_hat, gamma_hat, log_lik = fit_varcomp_diagonal(
        beta_subject,
        X_group,
        basis,
        theta_init,
        offset=var_within,
        spec=spec,
        block=block,
    )
    return FLAMEResult(
        sigma_b_sq=jnp.exp(theta_hat[:, 0]),
        gamma_hat=gamma_hat,
        log_lik=log_lik,
    )
