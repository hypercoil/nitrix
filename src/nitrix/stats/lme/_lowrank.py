# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Low-rank (q-rank) AI-REML for the two-component voxelwise LME.

``reml_fit`` diagonalises the total covariance by an eigendecomposition of
``ZZ^T`` (``N x N``), an ``O(N^3)`` one-off.  When the random-effect design
``Z`` is ``N x q`` with ``q < N`` (cohort group analyses: hundreds of subjects,
a handful of random-effect columns), the FaST-LMM **low-rank** formulation is
asymptotically cheaper: a ``q x q`` eig of ``Z^T Z`` (``O(N q^2 + q^3)``) gives
the ``r = rank(Z)`` nonzero eigenvalues ``s^2`` and the range left-singular
vectors ``U_r = Z W / s`` (``N x r``), so ``ZZ^T = U_r diag(s^2) U_r^T`` without
ever forming the ``N x N`` factor.

The ``N - r`` null-space directions of ``Z`` all carry the same total variance
``sigma_e^2`` (the random effect is silent there), so they never need to be
materialised individually: they enter the REML objective only through three
per-voxel aggregates -- the null-space Gram pieces

    Gxx = X^T X - X_r^T X_r            (p, p, shared)
    Gxy = X^T y - X_r^T y_r            (p,, per voxel)
    Gyy = y^T y - y_r^T y_r            (scalar, per voxel)

and the multiplicity ``n0 = N - r``.  Everything below is the same analytic
AI-REML as ``_varcomp`` (closed-form score + average-information curvature, no
``N x N`` intermediate, no second-order autodiff), specialised to the
two-component ``theta = [log sigma_b^2, log sigma_e^2]`` model and augmented with
the null-space aggregate terms.  When ``n0 = 0`` (``Z`` full row rank) the
aggregates vanish and this reduces exactly to the dense two-component fit.

This is a **separate** engine from ``_varcomp`` deliberately: ``_varcomp`` powers
both ``reml_fit`` and ``flame_two_level`` and is validated to 50/50 fresh-process
GPU trials, so the low-rank path is isolated rather than threaded through it.
Both share ``_small_inv_logdet`` (cuSOLVER-free ``(p, p)`` solve), ``VarCompSpec``
(Newton/backtracking config), and ``_blocked_vmap`` (memory chunking); the
per-voxel fit issues no cuSOLVER custom-call.

References
----------
- Lippert, C., Listgarten, J., et al. (2011).  FaST linear mixed models.
  Nat. Methods 8 (the low-rank diagonalisation).
- Gilmour, A. R., Thompson, R., & Cullis, B. R. (1995).  Average information
  REML.  Biometrics 51, 1440-1450.
"""

from __future__ import annotations

from typing import Tuple, cast

import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Float

from .._batching import blocked_vmap as _blocked_vmap
from .._smalllinalg import small_inv_logdet as _small_inv_logdet
from ._varcomp import VarCompSpec

__all__ = ['fit_lowrank_reml']


# ---------------------------------------------------------------------------
# Per-voxel quantities: GLS solve + analytic score / average information.
#
# theta = [log sigma_b^2, log sigma_e^2].  Range coordinates i = 1..r carry
# d_i = sigma_b^2 s2_i + sigma_e^2; the n0 null coordinates each carry
# d0 = sigma_e^2 and are summarised by (Gxx, Gxy, Gyy).
# ---------------------------------------------------------------------------


def _gls(
    theta: Float[Array, '2'],
    y_r: Float[Array, 'r'],
    X_r: Float[Array, 'r p'],
    s2: Float[Array, 'r'],
    gxx: Float[Array, 'p p'],
    gxy: Float[Array, 'p'],
    p: int,
    ridge: float,
) -> Tuple[
    Float[Array, 'p'],
    Float[Array, 'p p'],
    Float[Array, ''],
    Float[Array, 'r'],
    Float[Array, ''],
]:
    """GLS fixed-effect solve in the low-rank diagonal basis.

    ``A = X_r^T diag(1/d) X_r + Gxx / d0 + ridge I`` folds the null-space Gram
    in at the shared residual variance ``d0 = sigma_e^2``.  Returns ``(beta,
    A_inv, log_det_A, inv_d, d0)``.
    """
    sb2 = jnp.exp(theta[0])
    se2 = jnp.exp(theta[1])
    d = sb2 * s2 + se2
    inv_d = 1.0 / d
    d0 = se2
    Xw = X_r * inv_d[:, None]
    A = Xw.T @ X_r + gxx / d0 + ridge * jnp.eye(p, dtype=X_r.dtype)
    rhs = Xw.T @ y_r + gxy / d0
    A_inv, log_det_A = _small_inv_logdet(A, p)
    beta = A_inv @ rhs
    return beta, A_inv, log_det_A, inv_d, d0


def _neg_loglik(
    theta: Float[Array, '2'],
    y_r: Float[Array, 'r'],
    X_r: Float[Array, 'r p'],
    s2: Float[Array, 'r'],
    gxx: Float[Array, 'p p'],
    gxy: Float[Array, 'p'],
    gyy: Float[Array, ''],
    n0: int,
    p: int,
    ridge: float,
) -> Float[Array, '']:
    """Profile REML negative log-likelihood (low-rank)."""
    beta, _, log_det_A, inv_d, d0 = _gls(
        theta, y_r, X_r, s2, gxx, gxy, p, ridge
    )
    d = 1.0 / inv_d
    r = y_r - X_r @ beta
    rss_range = jnp.sum(r * r * inv_d)
    rss_null = (gyy - 2.0 * beta @ gxy + beta @ gxx @ beta) / d0
    log_det_v = jnp.sum(jnp.log(d)) + n0 * jnp.log(d0)
    return 0.5 * (log_det_v + log_det_A + rss_range + rss_null)


def _score_and_info(
    theta: Float[Array, '2'],
    y_r: Float[Array, 'r'],
    X_r: Float[Array, 'r p'],
    s2: Float[Array, 'r'],
    gxx: Float[Array, 'p p'],
    gxy: Float[Array, 'p'],
    gyy: Float[Array, ''],
    n0: int,
    p: int,
    ridge: float,
) -> Tuple[Float[Array, '2'], Float[Array, '2 2'], Float[Array, '']]:
    """Analytic REML score, average-information curvature, and nll (low-rank).

    The range terms are the ``_varcomp`` reductions over the ``r`` range
    coordinates; the null space contributes only through the aggregates
    ``(Gxx, Gxy, Gyy, n0)`` -- no ``N x N`` (or even ``r``-beyond-range)
    intermediate.
    """
    sb2 = jnp.exp(theta[0])
    se2 = jnp.exp(theta[1])
    beta, A_inv, log_det_A, inv_d, d0 = _gls(
        theta, y_r, X_r, s2, gxx, gxy, p, ridge
    )
    d = 1.0 / inv_d

    r = y_r - X_r @ beta
    rd = r * inv_d  # (P y)_i on the range
    rss_range = jnp.sum(r * rd)
    # ||y_null - X_null beta||^2 (the raw null residual sum of squares).
    rss_null_raw = gyy - 2.0 * beta @ gxy + beta @ gxx @ beta
    rss_null = rss_null_raw / d0
    nll = 0.5 * (
        jnp.sum(jnp.log(d))
        + n0 * jnp.log(d0)
        + log_det_A
        + rss_range
        + rss_null
    )

    # Range projection diagonal P_ii and the per-component dd/dtheta_k.
    h = jnp.sum((X_r @ A_inv) * X_r, axis=1)  # leverage x_i^T A^{-1} x_i
    p_diag = inv_d - h * inv_d * inv_d
    g_b = sb2 * s2  # dd/dtheta_b on the range
    g_e = se2 * jnp.ones_like(s2)  # dd/dtheta_e on the range
    g = jnp.stack([g_b, g_e], axis=0)  # (2, r)
    score_range = 0.5 * (g @ (p_diag - rd * rd))  # (2,)

    # Null-space score: only the residual component (theta_e) loads there.
    tr_ainv_gxx = jnp.sum(A_inv * gxx)  # tr(A^{-1} Gxx)
    score_e_null = (
        0.5
        * se2
        * (n0 / d0 - tr_ainv_gxx / (d0 * d0) - rss_null_raw / (d0 * d0))
    )
    score = score_range + jnp.array(
        [0.0, score_e_null], dtype=score_range.dtype
    )

    # Average information AI_{kl} = 0.5 u_k^T P u_l, u_k = g_k ⊙ (P y).
    u = g * rd[None, :]  # (2, r)
    uw = u * inv_d[None, :]  # (2, r)
    u_w_u = uw @ u.T  # (2, 2)
    m = uw @ X_r  # (2, p)

    # Null additions (residual component only): the null space adds to the
    # (e, e) self-term and the e-row of m = u_k^T V^{-1} X.
    gxx_beta = gxx @ beta
    m_e_null = (se2 / (d0 * d0)) * (gxy - gxx_beta)  # (p,)
    u_w_u = u_w_u.at[1, 1].add(se2 * se2 * rss_null_raw / (d0**3))
    m = m.at[1].add(m_e_null)

    info = 0.5 * (u_w_u - m @ A_inv @ m.T)
    return score, info, nll


# ---------------------------------------------------------------------------
# AI-REML Newton step (damped, step-clipped, backtracked) -- mirrors _varcomp.
# ---------------------------------------------------------------------------


def _newton_step(
    theta: Float[Array, '2'],
    y_r: Float[Array, 'r'],
    X_r: Float[Array, 'r p'],
    s2: Float[Array, 'r'],
    gxx: Float[Array, 'p p'],
    gxy: Float[Array, 'p'],
    gyy: Float[Array, ''],
    n0: int,
    p: int,
    spec: VarCompSpec,
) -> Float[Array, '2']:
    """One AI-REML Newton step with damping, clipping, and backtracking."""
    score, info, nll_old = _score_and_info(
        theta, y_r, X_r, s2, gxx, gxy, gyy, n0, p, spec.ridge
    )
    info_damped = info + spec.damping * jnp.eye(2, dtype=info.dtype)
    info_inv, _ = _small_inv_logdet(info_damped, 2)
    delta = jnp.clip(info_inv @ score, -spec.max_step, spec.max_step)

    def body(
        _: Array, carry: Tuple[Array, Array, Array]
    ) -> Tuple[Array, Array, Array]:
        scale, theta_best, nll_best = carry
        theta_try = theta - scale * delta
        nll_try = _neg_loglik(
            theta_try, y_r, X_r, s2, gxx, gxy, gyy, n0, p, spec.ridge
        )
        accept = nll_try < nll_best
        theta_new = jnp.where(accept, theta_try, theta_best)
        nll_new = jnp.where(accept, nll_try, nll_best)
        return (scale * 0.5, theta_new, nll_new)

    init: Tuple[Array, Array, Array] = (
        jnp.asarray(1.0, dtype=theta.dtype),
        theta,
        nll_old,
    )
    _, theta_final, _ = lax.fori_loop(0, spec.n_backtrack, body, init)
    return cast(Float[Array, '2'], theta_final)


def _fit_one(
    y_r: Float[Array, 'r'],
    X_r: Float[Array, 'r p'],
    s2: Float[Array, 'r'],
    gxx: Float[Array, 'p p'],
    gxy: Float[Array, 'p'],
    gyy: Float[Array, ''],
    theta_init: Float[Array, '2'],
    n0: int,
    p: int,
    spec: VarCompSpec,
) -> Tuple[Float[Array, '2'], Float[Array, 'p'], Float[Array, '']]:
    """Single-voxel low-rank AI-REML fit.  Returns ``(theta, beta, log_lik)``."""

    def step(
        theta: Float[Array, '2'], _: None
    ) -> Tuple[Float[Array, '2'], None]:
        return (
            _newton_step(theta, y_r, X_r, s2, gxx, gxy, gyy, n0, p, spec),
            None,
        )

    theta_final, _ = lax.scan(step, theta_init, xs=None, length=spec.n_iter)
    beta, _, _, _, _ = _gls(theta_final, y_r, X_r, s2, gxx, gxy, p, spec.ridge)
    nll = _neg_loglik(
        theta_final, y_r, X_r, s2, gxx, gxy, gyy, n0, p, spec.ridge
    )
    return theta_final, beta, -nll


# ---------------------------------------------------------------------------
# Public core entry point
# ---------------------------------------------------------------------------


def fit_lowrank_reml(
    y_range: Float[Array, 'V r'],
    X_range: Float[Array, 'r p'],
    s2: Float[Array, 'r'],
    gxx: Float[Array, 'p p'],
    gxy: Float[Array, 'V p'],
    gyy: Float[Array, 'V'],
    theta_init: Float[Array, 'V 2'],
    n0: int,
    *,
    spec: VarCompSpec = VarCompSpec(),
    block: int | None = None,
) -> Tuple[Float[Array, 'V 2'], Float[Array, 'V p'], Float[Array, 'V']]:
    """Batched low-rank two-component REML over ``V`` voxels.

    Parameters
    ----------
    y_range
        ``(V, r)`` responses projected onto the range ``U_r^T y_v``.
    X_range
        ``(r, p)`` fixed-effect design projected onto the range ``U_r^T X``.
    s2
        ``(r,)`` nonzero eigenvalues of ``ZZ^T`` (the range spectrum).
    gxx
        ``(p, p)`` null-space Gram ``X^T X - X_r^T X_r`` (shared).
    gxy, gyy
        ``(V, p)`` / ``(V,)`` per-voxel null-space aggregates.
    theta_init
        ``(V, 2)`` initial ``[log sigma_b^2, log sigma_e^2]``.
    n0
        Null-space multiplicity ``N - r`` (static).
    spec, block
        Newton/backtracking config and optional voxel-block chunking.

    Returns
    -------
    ``(theta_hat (V, 2), beta_hat (V, p), log_lik (V,))``.
    """
    p = X_range.shape[-1]

    def per_voxel(
        y: Float[Array, 'r'],
        gx: Float[Array, 'p'],
        gy: Float[Array, ''],
        th: Float[Array, '2'],
    ) -> Tuple[Float[Array, '2'], Float[Array, 'p'], Float[Array, '']]:
        return _fit_one(y, X_range, s2, gxx, gx, gy, th, n0, p, spec)

    return cast(
        Tuple[Float[Array, 'V 2'], Float[Array, 'V p'], Float[Array, 'V']],
        _blocked_vmap(per_voxel, (y_range, gxy, gyy, theta_init), block=block),
    )
