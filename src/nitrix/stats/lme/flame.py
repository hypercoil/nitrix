# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
FLAME-style two-level mixed-effects model for fMRI group analysis.

The model
---------

At level 1, each subject's BOLD time series has been fit by a
GLM, yielding a per-voxel estimate :math:`\\beta_i` of the per-subject
effect (e.g., the activation magnitude on a task contrast) plus
its within-subject standard error squared :math:`s_i^2` (typically
the OLS residual variance from the level-1 fit).  These are the
inputs.

At level 2, the per-voxel group model is

.. math::

    \\beta_i &= X_{\\mathrm{group}}\\, \\gamma + b_i + \\epsilon_i \\\\
    b_i &\\sim \\mathcal{N}(0, \\sigma_b^2)
        \\quad \\text{(between-subject variance, unknown)} \\\\
    \\epsilon_i &\\sim \\mathcal{N}(0, s_i^2)
        \\quad \\text{(within-subject variance, known)}

So the total per-subject variance is :math:`\\sigma_b^2 + s_i^2` --
heteroscedastic by subject.  :math:`\\gamma` is the group-level
fixed-effect vector.

This is the **FSL FLAME** model (Beckmann, Jenkinson, Smith
2003).  Only :math:`\\sigma_b^2` is estimated; :math:`\\gamma` is profiled
out analytically; :math:`s_i^2` is known.  The implementation is a
**single-parameter REML**: Newton iteration on a scalar log-
variance.  It avoids the identifiability problem of the generic
two-parameter REML (where :math:`\\sigma_b^2` and a free scaling on
the within-variance trade off).

Computational structure
-----------------------

Per voxel, the covariance
:math:`V_v = \\sigma_b^2 I + \\operatorname{diag}(s_v^2)` is
naturally diagonal, so FLAME is the shared diagonalised-REML fit
with a single free component (an all-ones basis) and a
fixed per-coordinate offset equal to the known within-variance.  The
fit makes **no per-voxel cuSOLVER call**:
the fixed-effect algebra is closed-form for the dominant :math:`p \\in
\\{1, 2\\}` designs (the intercept :math:`p = 1` is a scalar) and the REML
score / curvature are analytic (no second-order autodiff through a
Cholesky).  That is what unblocks :func:`flame_two_level` on the
broken-cuSOLVER GPU and flattens its compile time, which would otherwise
grow linearly in the voxel count.

Memory: the effect estimates and within-variances are each ``(V, N)``,
plus per-voxel parameters.  There is no ``(V, N, N)`` intermediate.

Reference
---------
Beckmann, C. F., Jenkinson, M., & Smith, S. M. (2003).  General
multilevel linear modeling for group analysis in fMRI.
NeuroImage 20, 1052-1063.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import jax.numpy as jnp
from jaxtyping import Array, Float

from .._result import register_result
from ._varcomp import VarCompSpec, fit_varcomp_diagonal

__all__ = ['FLAMEResult', 'flame_two_level']

_EPS = 1e-12


def _huber_weight(z: Float[Array, 'V N'], c: float) -> Float[Array, 'V N']:
    """Huber down-weighting of a studentized residual.

    Returns :math:`1` where :math:`|z| \\le c` and :math:`c / |z|` otherwise.
    The weight is monotone and strictly positive -- never a hard zero, so a
    down-weighted subject's inflated variance stays finite.

    Parameters
    ----------
    z
        Studentized residuals, shape ``(V, N)`` (per voxel, per subject).
    c
        Huber tuning constant, the threshold on :math:`|z|` beyond which a
        residual is down-weighted.

    Returns
    -------
    Float[Array, 'V N']
        Per-voxel, per-subject Huber weights in :math:`(0, 1]`, shape
        ``(V, N)``.
    """
    az = jnp.abs(z)
    return jnp.where(az <= c, 1.0, c / jnp.clip(az, _EPS, None))


@register_result(children=('sigma_b_sq', 'gamma_hat', 'log_lik', 'weights'))
@dataclass(frozen=True)
class FLAMEResult:
    """Per-voxel FLAME fit output.

    Attributes
    ----------
    sigma_b_sq
        Per-voxel estimated between-subject variance
        :math:`\\sigma_b^2`, shape ``(V,)``.
    gamma_hat
        Per-voxel group-level fixed-effect estimates :math:`\\gamma`,
        shape ``(V, p)``.
    log_lik
        Per-voxel restricted log-likelihood at the fitted variance,
        shape ``(V,)``.
    weights
        Per-voxel, per-subject outlier-deweighting weight, shape ``(V, N)``:
        all ``1`` for the standard FLAME fit (``robust=False``), and the
        converged robust weights in :math:`(0, 1]` for ``robust=True`` -- a
        value well below ``1`` marks a subject whose level-1 estimate was
        down-weighted as an outlier.
    """

    sigma_b_sq: Float[Array, 'V']
    gamma_hat: Float[Array, 'V p']
    log_lik: Float[Array, 'V']
    weights: Float[Array, 'V N']


def _flame_default_log_init(
    beta: Float[Array, 'V N'],
    var_within: Float[Array, 'V N'],
) -> Float[Array, 'V']:
    """Initial guess for :math:`\\log(\\sigma_b^2)` per voxel.

    Uses a method-of-moments heuristic: the empirical between-subject
    variance of ``beta`` about its per-voxel mean, minus the mean
    within-variance, clamped to a small positive floor so the log is finite.

    Parameters
    ----------
    beta
        Per-voxel, per-subject level-1 effect estimates, shape ``(V, N)``.
    var_within
        Per-voxel, per-subject within-subject variances, shape ``(V, N)``.

    Returns
    -------
    Float[Array, 'V']
        Per-voxel initial value of :math:`\\log(\\sigma_b^2)`, shape ``(V,)``.
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
    robust: bool = False,
    n_deweight: int = 5,
    robust_c: float = 1.345,
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
        Per-voxel initial value of :math:`\\log(\\sigma_b^2)`, shape ``(V,)``.
        Default heuristic: method-of-moments residual variance minus mean
        within-variance.
    n_iter
        Number of Newton iterations.  Default ``30``.
    damping
        Levenberg guard added to the (scalar) average-information curvature.
    block
        Optional voxel-block size bounding peak memory on brain-scale
        ``V`` (see :func:`reml_fit`).  ``None`` (default) maps over all voxels
        in a single pass.
    robust
        ``True`` enables outlier deweighting (the FLAMEO-style robust group
        analysis, FSL's FLAME1 followed by FLAME-outlier): a subject whose
        level-1 estimate is an outlier relative to the group model has its
        within-variance inflated by :math:`s_i^2 / w_i`, down-weighting it in
        the precision-weighted group fit.  This is a deterministic Huber-IRLS
        deweighting (not the full Bayesian outlier-mixture inference of Woolrich
        2008); the converged per-subject weights are returned in
        :attr:`FLAMEResult.weights`.
    n_deweight
        Number of deweighting (IRLS reweight) steps when ``robust=True``.
    robust_c
        Huber tuning constant on the studentized residual (default ``1.345``,
        giving 95% efficiency under normality); smaller is more aggressive at
        down-weighting.

    Returns
    -------
    FLAMEResult
        A :class:`FLAMEResult` carrying the per-voxel between-subject variance
        ``sigma_b_sq`` (shape ``(V,)``), the group fixed-effect estimates
        ``gamma_hat`` (shape ``(V, p)``), the per-voxel restricted
        log-likelihood ``log_lik`` (shape ``(V,)``), and the per-subject
        ``weights`` (shape ``(V, N)``; all ``1`` unless ``robust=True``).

    Notes
    -----
    Single-parameter REML: only :math:`\\sigma_b^2` is estimated;
    ``var_within`` is fixed at the user-supplied values.  This
    avoids the identifiability problem of the two-parameter
    relaxation (where the model can absorb a free scaling of
    ``var_within`` into :math:`\\sigma_b^2`).  It is implemented as the shared
    diagonalised-REML fit with one free (all-ones) component and a fixed
    per-voxel offset equal to ``var_within`` -- no per-voxel cuSOLVER call,
    so it runs on the broken-cuSOLVER GPU.

    References
    ----------
    Woolrich, M. (2008).  Robust group analysis using outlier inference.
    NeuroImage 41, 286-301.  :doi:`10.1016/j.neuroimage.2008.02.042`
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
    theta = log_sigma_b_sq_init[:, None]  # (V, 1)
    spec = VarCompSpec.flame(n_iter=n_iter, damping=damping)

    def _fit(offset, theta_init):
        return fit_varcomp_diagonal(
            beta_subject,
            X_group,
            basis,
            theta_init,
            offset=offset,
            spec=spec,
            block=block,
        )

    weights = jnp.ones_like(beta_subject)  # (V, N)
    if robust:
        # IRLS outer loop: down-weight outlier subjects (inflate s_i^2 / w_i),
        # re-fit, and re-derive the Huber weight from the studentized residual.
        for _ in range(n_deweight):
            theta, gamma_hat, _ = _fit(var_within / weights, theta)
            sigma_b_sq = jnp.exp(theta[:, 0])  # (V,)
            fitted = jnp.einsum('np,vp->vn', X_group, gamma_hat)  # (V, N)
            v_total = sigma_b_sq[:, None] + var_within  # (V, N) -- true scale
            z = (beta_subject - fitted) / jnp.sqrt(
                jnp.clip(v_total, _EPS, None)
            )
            weights = _huber_weight(z, robust_c)

    # Reported estimate: a fit consistent with the (converged) weights.
    theta_hat, gamma_hat, log_lik = _fit(var_within / weights, theta)
    return FLAMEResult(
        sigma_b_sq=jnp.exp(theta_hat[:, 0]),
        gamma_hat=gamma_hat,
        log_lik=log_lik,
        weights=weights,
    )
