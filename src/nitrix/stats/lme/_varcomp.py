# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Shared variance-components REML core for the voxelwise LME family.

This module is the engine under ``reml_fit`` and ``flame_two_level`` (and,
later, the GAM smoothing-parameter selection -- a penalised GLM is a
variance-components REML, Wood's mixed-model equivalence).  It factors the
machinery the two solvers used to duplicate into one diagonalised-REML fit
with three deliberate design choices:

1. **Static size-dispatch on the fixed-effect width ``p``** (via
   ``_smalllinalg.small_inv_logdet``).  The profiled fixed-effect normal
   equations are a tiny ``(p, p)`` system: closed form for the dominant designs
   (``p == 1`` intercept; ``p == 2`` mean + covariate) and an unrolled
   hand-Cholesky + ``triangular_solve`` (cuBLAS ``trsm``) for ``p > 2``.  The
   per-voxel fit issues **no cuSOLVER custom-call** at any ``p`` -- no ``potrf``
   / ``syevd`` / ``getrf`` -- which is what unblocks ``flame_two_level`` on the
   broken-cuSOLVER GPU and flattens the linear-in-``V`` XLA:CPU compile (see
   ``docs/feature-requests/gpu-cusolver-first-call-handle-failure.md``).

2. **Analytic AI-REML derivatives** replace second-order autodiff through the
   per-voxel Cholesky.  The score is the closed-form REML gradient and the
   curvature is the *average information* matrix (Gilmour, Thompson & Cullis
   1995) -- both assembled from per-coordinate reductions over ``N`` with **no
   ``N x N`` intermediate**.  This shrinks the compiled graph (compile time)
   and removes the autodiff tape (per-voxel memory / OOM).

3. **Optional voxel-block chunking** (``_blocked_vmap``) bounds peak memory as
   a tunable knob, so a brain-scale ``V`` need not materialise every voxel's
   intermediates at once.

The diagonalised model
----------------------

Every solver in the family is the same fit in a basis where the total
covariance is diagonal.  For one element (voxel) with response ``y`` (``N``),
fixed-effect design ``X`` (``N, p``), ``K`` free variance components with
log-parameters ``theta = log(sigma^2)`` and per-coordinate basis diagonals
``B`` (``K, N``), plus a *fixed* per-coordinate ``offset`` (``N``)::

    d_i      = sum_k exp(theta_k) * B[k, i] + offset_i        # diag(V)
    V        = diag(d)
    beta_hat = (X^T V^{-1} X)^{-1} X^T V^{-1} y               # profiled out
    nll      = 0.5 * (sum_i log d_i + log|X^T V^{-1} X| + r^T V^{-1} r)

``reml_fit`` instantiates this with ``B = [lambda, 1]`` (``K = 2``,
``offset = 0``); ``flame_two_level`` with ``B = [1]`` (``K = 1``) and
``offset = var_within`` (the known within-subject variance).

Analytic derivatives (in ``theta`` space)
-----------------------------------------

With ``P = V^{-1} - V^{-1} X (X^T V^{-1} X)^{-1} X^T V^{-1}`` the REML
projection, ``g_k = exp(theta_k) * B[k]`` (``= dd/dtheta_k``), and
``r/d = P y``::

    score_k = 0.5 * sum_i g_{k,i} (P_ii - (r_i/d_i)^2)
    AI_{kl} = 0.5 * u_k^T P u_l,        u_k = g_k ⊙ (r/d)

both of which expand into reductions over ``N`` using only the ``(p, p)``
inverse and the per-coordinate leverages ``h_i = x_i^T (X^T V^{-1} X)^{-1}
x_i``.  ``score`` agrees with ``jax.grad(nll)`` to ~1e-10; ``AI`` is the
average-information curvature used for the (damped, backtracked) Newton step.

References
----------
- Gilmour, A. R., Thompson, R., & Cullis, B. R. (1995).  Average information
  REML.  Biometrics 51, 1440-1450.
- Lippert, C., Listgarten, J., et al. (2011).  FaST linear mixed models.
  Nat. Methods 8 (the diagonalising rotation the callers supply).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple, cast

import jax.numpy as jnp
from jaxtyping import Array, Float

from .._batching import blocked_vmap as _blocked_vmap
from .._smalllinalg import small_inv_logdet as _small_inv_logdet
from ._optimise import damped_newton

__all__ = ['VarCompSpec', 'fit_varcomp_diagonal', 'varcomp_inference']


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VarCompSpec:
    """Frozen, hashable configuration for a variance-components REML fit.

    Hashable so it can ride a ``jax.custom_vjp`` non-differentiable argument
    slot (the deferred implicit-function-theorem VJP) and be closed over as a
    static config today.  All fields are primitive; the defaults reproduce the
    historical ``reml_fit`` / ``flame_two_level`` robustness behaviour.

    Attributes
    ----------
    n_iter
        Fixed Newton-scoring iteration count (the AI-REML outer loop).
    damping
        Levenberg-style damping added to the average-information diagonal,
        stabilising steps near a boundary where a component collapses.
    max_step
        Per-axis clip on the log-variance update (Newton overshoot guard).
    n_backtrack
        Backtracking halvings tried when the full step does not decrease nll.
    ridge
        Small stabiliser added to ``X^T V^{-1} X`` before the (closed-form or
        LU) inverse, for near-singular fixed-effect designs.
    """

    n_iter: int = 20
    damping: float = 1e-6
    max_step: float = 1.0
    n_backtrack: int = 4
    ridge: float = 1e-8

    @classmethod
    def reml(cls, **kw: Any) -> 'VarCompSpec':
        """Defaults tuned for general two-component REML (``reml_fit``)."""
        return cls(**kw)

    @classmethod
    def flame(cls, **kw: Any) -> 'VarCompSpec':
        """Defaults tuned for the single-parameter FLAME REML."""
        return cls(**{'n_iter': 30, **kw})


# ---------------------------------------------------------------------------
# Diagonalised-REML quantities: nll, beta, analytic score + average information
# ---------------------------------------------------------------------------


def _diag_variance(
    theta: Float[Array, 'K'],
    B: Float[Array, 'K N'],
    offset: Float[Array, 'N'],
) -> Float[Array, 'N']:
    """Per-coordinate total variance ``d_i = sum_k e^{theta_k} B[k,i] + off_i``."""
    return jnp.exp(theta) @ B + offset


def _profile_beta(
    d: Float[Array, 'N'],
    y: Float[Array, 'N'],
    X: Float[Array, 'N p'],
    p: int,
    ridge: float,
) -> Tuple[
    Float[Array, 'p'],
    Float[Array, 'p p'],
    Float[Array, ''],
    Float[Array, 'N'],
]:
    """GLS fixed-effect solve in the diagonal basis.

    Returns ``(beta, A_inv, log_det_A, inv_d)`` where ``A = X^T V^{-1} X +
    ridge I`` and ``inv_d = 1/d``.  ``A_inv`` and ``log_det_A`` come from the
    size-dispatched ``_small_inv_logdet`` (no cuSOLVER call for ``p in
    {1, 2}``).
    """
    inv_d = 1.0 / d
    Xw = X * inv_d[:, None]
    A = Xw.T @ X
    A = A + ridge * jnp.eye(p, dtype=A.dtype)
    A_inv, log_det_A = _small_inv_logdet(A, p)
    beta = A_inv @ (Xw.T @ y)
    return beta, A_inv, log_det_A, inv_d


def _neg_loglik(
    theta: Float[Array, 'K'],
    y: Float[Array, 'N'],
    X: Float[Array, 'N p'],
    B: Float[Array, 'K N'],
    offset: Float[Array, 'N'],
    p: int,
    ridge: float,
) -> Float[Array, '']:
    """Profile REML negative log-likelihood in the diagonal basis."""
    d = _diag_variance(theta, B, offset)
    beta, _, log_det_A, inv_d = _profile_beta(d, y, X, p, ridge)
    r = y - X @ beta
    rss = jnp.sum(r * r * inv_d)
    return 0.5 * (jnp.sum(jnp.log(d)) + log_det_A + rss)


def _score_and_info(
    theta: Float[Array, 'K'],
    y: Float[Array, 'N'],
    X: Float[Array, 'N p'],
    B: Float[Array, 'K N'],
    offset: Float[Array, 'N'],
    p: int,
    ridge: float,
) -> Tuple[Float[Array, 'K'], Float[Array, 'K K'], Float[Array, '']]:
    """Analytic REML score, average-information curvature, and nll.

    All quantities are reductions over ``N`` built from the ``(p, p)`` inverse
    -- there is no ``N x N`` intermediate at any point.  Returns
    ``(score, AI, nll)`` so a single pass serves both the Newton step and the
    backtracking reference value.
    """
    sigma2 = jnp.exp(theta)
    d = sigma2 @ B + offset
    beta, A_inv, log_det_A, inv_d = _profile_beta(d, y, X, p, ridge)

    r = y - X @ beta
    rd = r * inv_d  # (P y)_i
    rss = jnp.sum(r * rd)
    nll = 0.5 * (jnp.sum(jnp.log(d)) + log_det_A + rss)

    # Per-coordinate leverage h_i = x_i^T A^{-1} x_i and projection diagonal.
    h = jnp.sum((X @ A_inv) * X, axis=1)
    p_diag = inv_d - h * inv_d * inv_d

    g = sigma2[:, None] * B  # (K, N) = dd/dtheta_k
    score = 0.5 * (g @ (p_diag - rd * rd))

    # Average information: AI_{kl} = 0.5 u_k^T P u_l, u_k = g_k ⊙ (r/d).
    u = g * rd[None, :]  # (K, N)
    uw = u * inv_d[None, :]  # (K, N)
    u_w_u = uw @ u.T  # (K, K)
    m = uw @ X  # (K, p)
    info = 0.5 * (u_w_u - m @ A_inv @ m.T)
    return score, info, nll


# ---------------------------------------------------------------------------
# AI-REML fit via the shared optimiser (the analytic-curvature fork)
# ---------------------------------------------------------------------------


def _fit_one(
    y: Float[Array, 'N'],
    X: Float[Array, 'N p'],
    B: Float[Array, 'K N'],
    offset: Float[Array, 'N'],
    theta_init: Float[Array, 'K'],
    p: int,
    spec: VarCompSpec,
) -> Tuple[Float[Array, 'K'], Float[Array, 'p'], Float[Array, '']]:
    """Single-element AI-REML fit.  Returns ``(theta, beta, log_lik)``.

    Uses the shared ``_optimise.damped_newton`` through its **analytic-curvature
    fork**: ``curvature`` returns the closed-form ``(score, average-information)``
    (``_score_and_info``), so no autodiff Hessian is formed, and ``step='damped'``
    is correct because the AI matrix is positive-(semi)definite by construction
    (immune to the saddle problem the autodiff solvers guard against).  This is
    the analytic counterpart of the autodiff path the block-Woodbury / nested /
    correlation solvers take through the same optimiser.
    """

    def nll(theta: Float[Array, 'K']) -> Float[Array, '']:
        return _neg_loglik(theta, y, X, B, offset, p, spec.ridge)

    def curvature(
        theta: Float[Array, 'K'],
    ) -> Tuple[Float[Array, 'K'], Float[Array, 'K K']]:
        score, info, _ = _score_and_info(theta, y, X, B, offset, p, spec.ridge)
        return score, info

    theta_final = damped_newton(
        nll, theta_init, spec=spec, curvature=curvature, step='damped'
    )
    d = _diag_variance(theta_final, B, offset)
    beta, _, _, _ = _profile_beta(d, y, X, p, spec.ridge)
    return theta_final, beta, -nll(theta_final)


# ---------------------------------------------------------------------------
# Public core entry point
# ---------------------------------------------------------------------------


def fit_varcomp_diagonal(
    Y: Float[Array, 'V N'],
    X: Float[Array, 'N p'],
    B: Float[Array, 'K N'],
    theta_init: Float[Array, 'V K'],
    *,
    offset: Optional[Float[Array, 'V N']] = None,
    spec: VarCompSpec = VarCompSpec(),
    block: Optional[int] = None,
) -> Tuple[Float[Array, 'V K'], Float[Array, 'V p'], Float[Array, 'V']]:
    """Batched diagonalised-REML fit over ``V`` elements (shared ``X``, ``B``).

    The single entry point both ``reml_fit`` and ``flame_two_level`` build on.
    ``X`` and the basis diagonals ``B`` are shared across elements; ``Y`` (and
    optionally a per-element ``offset``) vary.  ``p`` is read from ``X`` and
    drives the static size-dispatch in the tiny fixed-effect solve.

    Parameters
    ----------
    Y
        ``(V, N)`` per-element responses (already rotated into the
        diagonalising basis by the caller).
    X
        ``(N, p)`` shared fixed-effect design (rotated).
    B
        ``(K, N)`` shared variance-component basis diagonals.
    theta_init
        ``(V, K)`` per-element initial log-variances.
    offset
        Optional ``(V, N)`` fixed per-coordinate variance offset (FLAME's
        known within-variance).  ``None`` is treated as zeros.
    spec
        ``VarCompSpec`` controlling the Newton/backtracking behaviour.
    block
        Optional voxel-block size bounding peak memory (see ``_blocked_vmap``).

    Returns
    -------
    ``(theta_hat (V, K), beta_hat (V, p), log_lik (V,))``.
    """
    p = X.shape[-1]

    if offset is None:

        def per_voxel(
            y: Float[Array, 'N'], th: Float[Array, 'K']
        ) -> Tuple[Float[Array, 'K'], Float[Array, 'p'], Float[Array, '']]:
            return _fit_one(
                y, X, B, jnp.zeros(X.shape[0], dtype=Y.dtype), th, p, spec
            )

        return cast(
            Tuple[Float[Array, 'V K'], Float[Array, 'V p'], Float[Array, 'V']],
            _blocked_vmap(per_voxel, (Y, theta_init), block=block),
        )

    def per_voxel_off(
        y: Float[Array, 'N'],
        off: Float[Array, 'N'],
        th: Float[Array, 'K'],
    ) -> Tuple[Float[Array, 'K'], Float[Array, 'p'], Float[Array, '']]:
        return _fit_one(y, X, B, off, th, p, spec)

    return cast(
        Tuple[Float[Array, 'V K'], Float[Array, 'V p'], Float[Array, 'V']],
        _blocked_vmap(per_voxel_off, (Y, offset, theta_init), block=block),
    )


# ---------------------------------------------------------------------------
# Fixed-effect inference quantities at the fitted theta
# ---------------------------------------------------------------------------


def varcomp_inference(
    theta: Float[Array, 'K'],
    y: Float[Array, 'N'],
    X: Float[Array, 'N p'],
    B: Float[Array, 'K N'],
    offset: Float[Array, 'N'],
    p: int,
    spec: VarCompSpec,
) -> Tuple[Float[Array, 'p p'], Float[Array, 'K K'], Float[Array, 'K p p']]:
    """Fixed-effect inference quantities at the fitted ``theta``.

    Surfaces what the per-voxel solve already forms (and the historical fit
    discarded) for a mixed-model contrast test:

    - ``fixed_cov = (X^T V^{-1} X)^{-1}`` -- ``Cov(beta_hat)`` (``A_inv``).
    - ``theta_cov`` -- ``Cov(theta_hat)``, the inverse average-information matrix
      (the asymptotic covariance of the REML variance components).
    - ``grad_m`` -- the ``(K, p, p)`` tensors ``M_k = sum_i (g_{k,i} / d_i^2)
      x_i x_i^T`` (``g_k = dd/dtheta_k``).  For a contrast ``c`` with ``w =
      fixed_cov c`` these give the Satterthwaite gradient
      ``d(c^T fixed_cov c)/dtheta_k = w^T M_k w`` -- contrast-independent, so a
      single fit serves any contrast.  All cuSOLVER-free.
    """
    k = theta.shape[0]
    sigma2 = jnp.exp(theta)
    d = sigma2 @ B + offset
    inv_d = 1.0 / d
    _, A_inv, _, _ = _profile_beta(d, y, X, p, spec.ridge)
    _, info, _ = _score_and_info(theta, y, X, B, offset, p, spec.ridge)
    info_damped = info + spec.damping * jnp.eye(k, dtype=info.dtype)
    theta_cov, _ = _small_inv_logdet(info_damped, k)

    g = sigma2[:, None] * B  # (K, N) = dd/dtheta_k
    weight = g * (inv_d * inv_d)[None, :]  # (K, N): g_{k,i} / d_i^2
    grad_m = jnp.einsum('kn,np,nq->kpq', weight, X, X)  # (K, p, p)
    return A_inv, theta_cov, grad_m
