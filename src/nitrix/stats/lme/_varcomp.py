# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Shared variance-components REML core for the voxelwise LME family.

This module is the engine under :func:`reml_fit` and :func:`flame_two_level`
(and, later, the GAM smoothing-parameter selection -- a penalised GLM is a
variance-components REML, Wood's mixed-model equivalence).  It factors the
machinery the two solvers used to duplicate into one diagonalised-REML fit
with three deliberate design choices:

1. **Static size-dispatch on the fixed-effect width** :math:`p` (via
   :func:`small_inv_logdet`).  The profiled fixed-effect normal equations are
   a tiny :math:`(p, p)` system: closed form for the dominant designs
   (:math:`p = 1` intercept; :math:`p = 2` mean + covariate) and an unrolled
   hand-Cholesky + ``triangular_solve`` (cuBLAS ``trsm``) for :math:`p > 2`.
   The per-voxel fit issues no cuSOLVER custom-call at any :math:`p` -- no
   ``potrf`` / ``syevd`` / ``getrf`` -- which is what unblocks
   :func:`flame_two_level` on GPUs with an unreliable cuSOLVER and flattens
   the XLA:CPU compile that would otherwise grow linearly in :math:`V`.

2. **Analytic AI-REML derivatives** replace second-order autodiff through the
   per-voxel Cholesky.  The score is the closed-form REML gradient and the
   curvature is the *average information* matrix (Gilmour, Thompson & Cullis
   1995) -- both assembled from per-coordinate reductions over :math:`N` with
   no :math:`N \\times N` intermediate.  This shrinks the compiled graph
   (compile time) and removes the autodiff tape (per-voxel memory / OOM).

3. **Optional voxel-block chunking** (``blocked_vmap``) bounds peak memory as
   a tunable knob, so a brain-scale :math:`V` need not materialise every
   voxel's intermediates at once.

The diagonalised model
----------------------

Every solver in the family is the same fit in a basis where the total
covariance is diagonal.  For one element (voxel) with response :math:`y`
(length :math:`N`), fixed-effect design :math:`X` (shape :math:`(N, p)`),
:math:`K` free variance components with log-parameters
:math:`\\theta = \\log(\\sigma^2)` and per-coordinate basis diagonals
:math:`B` (shape :math:`(K, N)`), plus a *fixed* per-coordinate
``offset`` (length :math:`N`):

.. math::

    d_i &= \\sum_k \\exp(\\theta_k)\\, B_{k, i} + \\mathrm{offset}_i
        \\quad (= \\operatorname{diag} V) \\\\
    V &= \\operatorname{diag}(d) \\\\
    \\hat\\beta &= (X^{\\top} V^{-1} X)^{-1} X^{\\top} V^{-1} y
        \\quad (\\text{profiled out}) \\\\
    \\mathrm{nll} &= \\tfrac{1}{2}\\bigl(\\textstyle\\sum_i \\log d_i
        + \\log|X^{\\top} V^{-1} X| + r^{\\top} V^{-1} r\\bigr)

:func:`reml_fit` instantiates this with :math:`B = [\\lambda, 1]`
(:math:`K = 2`, ``offset = 0``); :func:`flame_two_level` with :math:`B = [1]`
(:math:`K = 1`) and ``offset = var_within`` (the known within-subject
variance).

Analytic derivatives (in :math:`\\theta` space)
-----------------------------------------------

With :math:`P = V^{-1} - V^{-1} X (X^{\\top} V^{-1} X)^{-1} X^{\\top} V^{-1}`
the REML projection, :math:`g_k = \\exp(\\theta_k)\\, B_k`
(:math:`= \\mathrm{d}d / \\mathrm{d}\\theta_k`), and :math:`r / d = P y`:

.. math::

    \\mathrm{score}_k &= \\tfrac{1}{2}\\sum_i g_{k, i}
        \\bigl(P_{ii} - (r_i / d_i)^2\\bigr) \\\\
    \\mathrm{AI}_{kl} &= \\tfrac{1}{2}\\, u_k^{\\top} P u_l,
        \\qquad u_k = g_k \\odot (r / d)

both of which expand into reductions over :math:`N` using only the
:math:`(p, p)` inverse and the per-coordinate leverages
:math:`h_i = x_i^{\\top} (X^{\\top} V^{-1} X)^{-1} x_i`.  ``score`` agrees with
``jax.grad(nll)`` to ~1e-10; ``AI`` is the average-information curvature used
for the (damped, backtracked) Newton step.

References
----------
- Gilmour, A. R., Thompson, R., & Cullis, B. R. (1995).  Average information
  REML: an efficient algorithm for variance parameter estimation in linear
  mixed models.  Biometrics 51, 1440-1450.  https://doi.org/10.2307/2533274
- Lippert, C., Listgarten, J., et al. (2011).  FaST linear mixed models for
  genome-wide association studies.  Nat. Methods 8, 833-835 (the
  diagonalising rotation the callers supply).
  https://doi.org/10.1038/nmeth.1681
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple, cast

import jax.numpy as jnp
from jaxtyping import Array, Float

from ...linalg._smalllinalg import small_inv_logdet as _small_inv_logdet
from .._batching import blocked_vmap as _blocked_vmap
from .._optimise import damped_newton

__all__ = ['VarCompSpec', 'fit_varcomp_diagonal', 'varcomp_inference']


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VarCompSpec:
    r"""Frozen, hashable configuration for a variance-components REML fit.

    Hashable so it can ride a ``jax.custom_vjp`` non-differentiable argument
    slot (the deferred implicit-function-theorem VJP) and be closed over as a
    static config today.  All fields are primitive; the defaults reproduce the
    historical :func:`reml_fit` / :func:`flame_two_level` robustness
    behaviour.

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
        Small stabiliser added to :math:`X^{\top} V^{-1} X` before the
        (closed-form or LU) inverse, for near-singular fixed-effect designs.
    """

    n_iter: int = 20
    damping: float = 1e-6
    max_step: float = 1.0
    n_backtrack: int = 4
    ridge: float = 1e-8

    @property
    def newton_kwargs(self) -> dict:
        """The iteration budget as ``stats._optimise.damped_newton`` kwargs.

        The seam decoupling the generic optimiser from this mixed-model config
        (audit O2): call sites pass ``**spec.newton_kwargs``; ``ridge`` (used by
        the solvers' own fixed-effect floor, not the Newton step) is excluded.
        """
        return {
            'n_iter': self.n_iter,
            'damping': self.damping,
            'max_step': self.max_step,
            'n_backtrack': self.n_backtrack,
        }

    @classmethod
    def flame(cls, **kw: Any) -> 'VarCompSpec':
        """Construct a spec with defaults tuned for single-parameter FLAME REML.

        The single-component FLAME fit benefits from a slightly longer Newton
        budget, so ``n_iter`` defaults to 30 here (rather than the base 20).

        Parameters
        ----------
        **kw
            Field overrides forwarded to the :class:`VarCompSpec`
            constructor.  Any field passed explicitly (including ``n_iter``)
            takes precedence over the FLAME default.

        Returns
        -------
        VarCompSpec
            A frozen spec with ``n_iter = 30`` unless overridden in ``kw``.
        """
        return cls(**{'n_iter': 30, **kw})


# ---------------------------------------------------------------------------
# Diagonalised-REML quantities: nll, beta, analytic score + average information
# ---------------------------------------------------------------------------


def _diag_variance(
    theta: Float[Array, 'K'],
    B: Float[Array, 'K N'],
    offset: Float[Array, 'N'],
) -> Float[Array, 'N']:
    """Per-coordinate total variance in the diagonal basis.

    Computes the diagonal of the total covariance,
    :math:`d_i = \\sum_k e^{\\theta_k} B_{k, i} + \\mathrm{offset}_i`.

    Parameters
    ----------
    theta
        ``(K,)`` log-variances of the free variance components.
    B
        ``(K, N)`` per-coordinate variance-component basis diagonals.
    offset
        ``(N,)`` fixed per-coordinate variance offset (e.g. FLAME's known
        within-subject variance).

    Returns
    -------
    Float[Array, 'N']
        ``(N,)`` per-coordinate total variance :math:`d`.
    """
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

    Solves the profiled fixed-effect normal equations
    :math:`A \\hat\\beta = X^{\\top} V^{-1} y` with
    :math:`A = X^{\\top} V^{-1} X + \\mathrm{ridge}\\, I`, forming the inverse
    and log-determinant of :math:`A` via the size-dispatched
    :func:`small_inv_logdet` (no cuSOLVER call for :math:`p \\in \\{1, 2\\}`).

    Parameters
    ----------
    d
        ``(N,)`` per-coordinate total variance (the diagonal of :math:`V`).
    y
        ``(N,)`` response in the diagonal basis.
    X
        ``(N, p)`` fixed-effect design in the diagonal basis.
    p
        Fixed-effect width, driving the static size-dispatch of the
        ``(p, p)`` solve.
    ridge
        Small stabiliser added to the ``(p, p)`` matrix before inversion.

    Returns
    -------
    beta : Float[Array, 'p']
        ``(p,)`` profiled fixed-effect estimate :math:`\\hat\\beta`.
    A_inv : Float[Array, 'p p']
        ``(p, p)`` inverse of :math:`A = X^{\\top} V^{-1} X + \\mathrm{ridge}\\,
        I`.
    log_det_A : Float[Array, '']
        Scalar :math:`\\log|A|`.
    inv_d : Float[Array, 'N']
        ``(N,)`` reciprocal variance :math:`1 / d`.
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
    """Profile REML negative log-likelihood in the diagonal basis.

    Evaluates
    :math:`\\tfrac{1}{2}(\\sum_i \\log d_i + \\log|X^{\\top} V^{-1} X| +
    r^{\\top} V^{-1} r)` at the given log-variances, profiling out the
    fixed effects.

    Parameters
    ----------
    theta
        ``(K,)`` log-variances of the free variance components.
    y
        ``(N,)`` response in the diagonal basis.
    X
        ``(N, p)`` fixed-effect design in the diagonal basis.
    B
        ``(K, N)`` variance-component basis diagonals.
    offset
        ``(N,)`` fixed per-coordinate variance offset.
    p
        Fixed-effect width driving the size-dispatch of the ``(p, p)`` solve.
    ridge
        Small stabiliser added to :math:`X^{\\top} V^{-1} X` before inversion.

    Returns
    -------
    Float[Array, '']
        Scalar profiled REML negative log-likelihood.
    """
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

    All quantities are reductions over :math:`N` built from the :math:`(p, p)`
    inverse -- there is no :math:`N \\times N` intermediate at any point.  A
    single pass returns all three so that the Newton step and its backtracking
    reference value share the same evaluation.

    Parameters
    ----------
    theta
        ``(K,)`` log-variances of the free variance components.
    y
        ``(N,)`` response in the diagonal basis.
    X
        ``(N, p)`` fixed-effect design in the diagonal basis.
    B
        ``(K, N)`` variance-component basis diagonals.
    offset
        ``(N,)`` fixed per-coordinate variance offset.
    p
        Fixed-effect width driving the size-dispatch of the ``(p, p)`` solve.
    ridge
        Small stabiliser added to :math:`X^{\\top} V^{-1} X` before inversion.

    Returns
    -------
    score : Float[Array, 'K']
        ``(K,)`` closed-form REML gradient of the negative log-likelihood
        with respect to :math:`\\theta`.
    info : Float[Array, 'K K']
        ``(K, K)`` average-information curvature matrix.
    nll : Float[Array, '']
        Scalar profiled REML negative log-likelihood at ``theta``.
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
    """Single-element AI-REML fit.

    Uses the shared ``damped_newton`` optimiser through its **analytic-
    curvature fork**: ``curvature`` returns the closed-form
    ``(score, average-information)`` from :func:`_score_and_info`, so no
    autodiff Hessian is formed, and ``step='damped'`` is correct because the
    average-information matrix is positive-(semi)definite by construction
    (immune to the saddle problem the autodiff solvers guard against).  This is
    the analytic counterpart of the autodiff path the block-Woodbury / nested /
    correlation solvers take through the same optimiser.

    Parameters
    ----------
    y
        ``(N,)`` response in the diagonal basis.
    X
        ``(N, p)`` fixed-effect design in the diagonal basis.
    B
        ``(K, N)`` variance-component basis diagonals.
    offset
        ``(N,)`` fixed per-coordinate variance offset.
    theta_init
        ``(K,)`` initial log-variances for the Newton iteration.
    p
        Fixed-effect width driving the size-dispatch of the ``(p, p)`` solve.
    spec
        :class:`VarCompSpec` controlling the Newton/backtracking behaviour and
        the fixed-effect ridge.

    Returns
    -------
    theta_final : Float[Array, 'K']
        ``(K,)`` fitted log-variances.
    beta : Float[Array, 'p']
        ``(p,)`` profiled fixed-effect estimate at ``theta_final``.
    log_lik : Float[Array, '']
        Scalar profiled REML log-likelihood at ``theta_final``.
    """

    def nll(theta: Float[Array, 'K']) -> Float[Array, '']:
        return _neg_loglik(theta, y, X, B, offset, p, spec.ridge)

    def curvature(
        theta: Float[Array, 'K'],
    ) -> Tuple[Float[Array, 'K'], Float[Array, 'K K']]:
        score, info, _ = _score_and_info(theta, y, X, B, offset, p, spec.ridge)
        return score, info

    theta_final = damped_newton(
        nll,
        theta_init,
        **spec.newton_kwargs,
        curvature=curvature,
        step='damped',
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

    The single entry point both :func:`reml_fit` and :func:`flame_two_level`
    build on.  ``X`` and the basis diagonals ``B`` are shared across elements;
    ``Y`` (and optionally a per-element ``offset``) vary.  ``p`` is read from
    ``X`` and drives the static size-dispatch in the tiny fixed-effect solve.

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
        :class:`VarCompSpec` controlling the Newton/backtracking behaviour.
    block
        Optional voxel-block size bounding peak memory (chunked ``vmap``).

    Returns
    -------
    theta_hat : Float[Array, 'V K']
        ``(V, K)`` per-element fitted log-variances.
    beta_hat : Float[Array, 'V p']
        ``(V, p)`` per-element profiled fixed-effect estimates.
    log_lik : Float[Array, 'V']
        ``(V,)`` per-element profiled REML log-likelihood at the fit.
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

    - ``fixed_cov`` :math:`= (X^{\\top} V^{-1} X)^{-1}` -- the fixed-effect
      covariance :math:`\\operatorname{Cov}(\\hat\\beta)` (the ``A_inv``
      returned).
    - ``theta_cov`` -- :math:`\\operatorname{Cov}(\\hat\\theta)`, the inverse
      average-information matrix (the asymptotic covariance of the REML
      variance components).
    - ``grad_m`` -- the :math:`(K, p, p)` tensors
      :math:`M_k = \\sum_i (g_{k, i} / d_i^2)\\, x_i x_i^{\\top}`, where
      :math:`g_k = \\mathrm{d}d / \\mathrm{d}\\theta_k`.  For a contrast
      :math:`c` with :math:`w = \\mathrm{fixed\\_cov}\\, c` these give the
      Satterthwaite gradient
      :math:`\\mathrm{d}(c^{\\top} \\mathrm{fixed\\_cov}\\, c) /
      \\mathrm{d}\\theta_k = w^{\\top} M_k w` -- contrast-independent, so a
      single fit serves any contrast.  All cuSOLVER-free.

    Parameters
    ----------
    theta
        ``(K,)`` fitted log-variances.
    y
        ``(N,)`` response in the diagonal basis.
    X
        ``(N, p)`` fixed-effect design in the diagonal basis.
    B
        ``(K, N)`` variance-component basis diagonals.
    offset
        ``(N,)`` fixed per-coordinate variance offset.
    p
        Fixed-effect width driving the size-dispatch of the ``(p, p)`` solve.
    spec
        :class:`VarCompSpec` supplying the fixed-effect ridge and the
        average-information damping.

    Returns
    -------
    fixed_cov : Float[Array, 'p p']
        ``(p, p)`` fixed-effect covariance
        :math:`(X^{\\top} V^{-1} X)^{-1}`.
    theta_cov : Float[Array, 'K K']
        ``(K, K)`` variance-component covariance, the (damped) inverse
        average-information matrix.
    grad_m : Float[Array, 'K p p']
        ``(K, p, p)`` Satterthwaite gradient tensors :math:`M_k` (see above).
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
