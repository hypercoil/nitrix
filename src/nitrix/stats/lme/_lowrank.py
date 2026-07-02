# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
r"""
Low-rank (q-rank) AI-REML for the two-component voxelwise LME.

The dense REML fit diagonalises the total covariance by an eigendecomposition
of :math:`Z Z^{\top}` (:math:`N \times N`), an :math:`O(N^3)` one-off.  When the
random-effect design :math:`Z` is :math:`N \times q` with :math:`q < N` (cohort
group analyses: hundreds of subjects, a handful of random-effect columns), the
FaST-LMM **low-rank** formulation is asymptotically cheaper: a :math:`q \times q`
eigendecomposition of :math:`Z^{\top} Z` (:math:`O(N q^2 + q^3)`) gives the
:math:`r = \operatorname{rank}(Z)` nonzero eigenvalues :math:`s^2` and the range
left-singular vectors :math:`U_r = Z W / s` (:math:`N \times r`), so
:math:`Z Z^{\top} = U_r \operatorname{diag}(s^2) U_r^{\top}` without ever forming
the :math:`N \times N` factor.

The :math:`N - r` null-space directions of :math:`Z` all carry the same total
variance :math:`\sigma_e^2` (the random effect is silent there), so they never
need to be materialised individually: they enter the REML objective only through
three per-voxel aggregates -- the null-space Gram pieces

- :math:`G_{xx} = X^{\top} X - X_r^{\top} X_r` (``(p, p)``, shared),
- :math:`G_{xy} = X^{\top} y - X_r^{\top} y_r` (``(p,)``, per voxel),
- :math:`G_{yy} = y^{\top} y - y_r^{\top} y_r` (scalar, per voxel),

and the multiplicity :math:`n_0 = N - r`.  Everything below is the same analytic
AI-REML as the dense variance-component engine (closed-form score plus
average-information curvature, no :math:`N \times N` intermediate, no
second-order autodiff), specialised to the two-component
:math:`\theta = [\log \sigma_b^2, \log \sigma_e^2]` model and augmented with the
null-space aggregate terms.  When :math:`n_0 = 0` (:math:`Z` full row rank) the
aggregates vanish and this reduces exactly to the dense two-component fit.

This is a **separate** engine from the dense variance-component path
deliberately, so the low-rank path is isolated rather than threaded through it.
Both share :func:`~nitrix.linalg._smalllinalg.small_inv_logdet` (a cuSOLVER-free
``(p, p)`` solve), :class:`~nitrix.stats.lme._varcomp.VarCompSpec` (the
Newton/backtracking config), and the memory-chunking batched map; the per-voxel
fit issues no cuSOLVER custom-call.

References
----------
Lippert, C., Listgarten, J., Liu, Y., Kadie, C. M., Davidson, R. I., &
Heckerman, D. (2011).  FaST linear mixed models for genome-wide association
studies.  *Nature Methods*, 8, 833-835.  https://doi.org/10.1038/nmeth.1681
(the low-rank diagonalisation).

Gilmour, A. R., Thompson, R., & Cullis, B. R. (1995).  Average information
REML: an efficient algorithm for variance parameter estimation in linear mixed
models.  *Biometrics*, 51, 1440-1450.  https://doi.org/10.2307/2533274
"""

from __future__ import annotations

from typing import Tuple, cast

import jax.numpy as jnp
from jaxtyping import Array, Float

from ...linalg._smalllinalg import small_inv_logdet as _small_inv_logdet
from .._batching import blocked_vmap as _blocked_vmap
from .._optimise import damped_newton
from ._varcomp import VarCompSpec

__all__ = ['fit_lowrank_reml', 'lowrank_inference']


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
    r"""Generalised least-squares fixed-effect solve in the low-rank diagonal
    basis.

    Forms the normal-equation matrix
    :math:`A = X_r^{\top} \operatorname{diag}(1/d) X_r + G_{xx} / d_0 + \lambda I`,
    which folds the null-space Gram in at the shared residual variance
    :math:`d_0 = \sigma_e^2`, and solves for the fixed-effect coefficients.  Here
    :math:`d_i = \sigma_b^2 s^2_i + \sigma_e^2` are the range-coordinate total
    variances and :math:`\lambda` is the ridge.

    Parameters
    ----------
    theta
        ``(2,)`` variance parameters ``[log sigma_b^2, log sigma_e^2]``.
    y_r
        ``(r,)`` response projected onto the range of the random-effect design.
    X_r
        ``(r, p)`` fixed-effect design projected onto the range.
    s2
        ``(r,)`` nonzero eigenvalues of the range spectrum.
    gxx
        ``(p, p)`` shared null-space Gram :math:`G_{xx}`.
    gxy
        ``(p,)`` per-voxel null-space aggregate :math:`G_{xy}`.
    p
        Number of fixed-effect columns (static).
    ridge
        Ridge added to the diagonal of :math:`A` for numerical stability.

    Returns
    -------
    beta : Float[Array, 'p']
        ``(p,)`` GLS fixed-effect estimate.
    A_inv : Float[Array, 'p p']
        ``(p, p)`` inverse of the normal-equation matrix :math:`A`.
    log_det_A : Float[Array, '']
        Scalar log-determinant of :math:`A`.
    inv_d : Float[Array, 'r']
        ``(r,)`` reciprocal range variances :math:`1/d_i`.
    d0 : Float[Array, '']
        Scalar shared residual variance :math:`d_0 = \sigma_e^2`.
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
    """Profile REML negative log-likelihood (low-rank).

    Evaluates the restricted (profile) negative log-likelihood at ``theta`` by
    solving the GLS problem for the fixed effects, then summing the range and
    null-space residual and log-determinant contributions.

    Parameters
    ----------
    theta
        ``(2,)`` variance parameters ``[log sigma_b^2, log sigma_e^2]``.
    y_r
        ``(r,)`` response projected onto the range.
    X_r
        ``(r, p)`` fixed-effect design projected onto the range.
    s2
        ``(r,)`` nonzero eigenvalues of the range spectrum.
    gxx
        ``(p, p)`` shared null-space Gram :math:`G_{xx}`.
    gxy
        ``(p,)`` per-voxel null-space aggregate :math:`G_{xy}`.
    gyy
        Scalar per-voxel null-space aggregate :math:`G_{yy}`.
    n0
        Null-space multiplicity :math:`n_0 = N - r` (static).
    p
        Number of fixed-effect columns (static).
    ridge
        Ridge added to the normal-equation diagonal.

    Returns
    -------
    Float[Array, '']
        Scalar profile REML negative log-likelihood at ``theta``.
    """
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
    r"""Analytic REML score, average-information curvature, and negative
    log-likelihood (low-rank).

    Computes the closed-form gradient of the profile REML negative
    log-likelihood with respect to ``theta``, the average-information
    approximation to its curvature, and the objective value itself, in a single
    pass.  The range terms are reductions over the ``r`` range coordinates
    (matching the dense variance-component engine); the null space contributes
    only through the aggregates :math:`(G_{xx}, G_{xy}, G_{yy}, n_0)` -- there is
    no :math:`N \times N` (or beyond-range) intermediate.

    Parameters
    ----------
    theta
        ``(2,)`` variance parameters ``[log sigma_b^2, log sigma_e^2]``.
    y_r
        ``(r,)`` response projected onto the range.
    X_r
        ``(r, p)`` fixed-effect design projected onto the range.
    s2
        ``(r,)`` nonzero eigenvalues of the range spectrum.
    gxx
        ``(p, p)`` shared null-space Gram :math:`G_{xx}`.
    gxy
        ``(p,)`` per-voxel null-space aggregate :math:`G_{xy}`.
    gyy
        Scalar per-voxel null-space aggregate :math:`G_{yy}`.
    n0
        Null-space multiplicity :math:`n_0 = N - r` (static).
    p
        Number of fixed-effect columns (static).
    ridge
        Ridge added to the normal-equation diagonal.

    Returns
    -------
    score : Float[Array, '2']
        ``(2,)`` REML score (gradient) with respect to ``theta``.
    info : Float[Array, '2 2']
        ``(2, 2)`` average-information curvature matrix (PSD by construction).
    nll : Float[Array, '']
        Scalar profile REML negative log-likelihood at ``theta``.
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
    """Single-voxel low-rank AI-REML fit.

    Optimises the profile REML negative log-likelihood for one voxel via the
    shared damped-Newton analytic-curvature fork (closed-form score and
    average-information curvature, with ``step='damped'`` since the
    average-information matrix is positive semi-definite by construction) -- the
    same optimiser path as the dense variance-component fit -- then recovers the
    fixed effects at the optimum.

    Parameters
    ----------
    y_r
        ``(r,)`` response projected onto the range.
    X_r
        ``(r, p)`` fixed-effect design projected onto the range.
    s2
        ``(r,)`` nonzero eigenvalues of the range spectrum.
    gxx
        ``(p, p)`` shared null-space Gram :math:`G_{xx}`.
    gxy
        ``(p,)`` per-voxel null-space aggregate :math:`G_{xy}`.
    gyy
        Scalar per-voxel null-space aggregate :math:`G_{yy}`.
    theta_init
        ``(2,)`` initial variance parameters ``[log sigma_b^2, log sigma_e^2]``.
    n0
        Null-space multiplicity :math:`n_0 = N - r` (static).
    p
        Number of fixed-effect columns (static).
    spec
        Newton/backtracking configuration (ridge, damping, Newton kwargs).

    Returns
    -------
    theta : Float[Array, '2']
        ``(2,)`` fitted variance parameters.
    beta : Float[Array, 'p']
        ``(p,)`` fitted fixed effects at the optimum.
    log_lik : Float[Array, '']
        Scalar profile REML log-likelihood at the optimum.
    """

    def nll(theta: Float[Array, '2']) -> Float[Array, '']:
        return _neg_loglik(
            theta, y_r, X_r, s2, gxx, gxy, gyy, n0, p, spec.ridge
        )

    def curvature(
        theta: Float[Array, '2'],
    ) -> Tuple[Float[Array, '2'], Float[Array, '2 2']]:
        score, info, _ = _score_and_info(
            theta, y_r, X_r, s2, gxx, gxy, gyy, n0, p, spec.ridge
        )
        return score, info

    theta_final = damped_newton(
        nll,
        theta_init,
        **spec.newton_kwargs,
        curvature=curvature,
        step='damped',
    )
    beta, _, _, _, _ = _gls(theta_final, y_r, X_r, s2, gxx, gxy, p, spec.ridge)
    return theta_final, beta, -nll(theta_final)


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
    r"""Batched low-rank two-component REML over ``V`` voxels.

    Parameters
    ----------
    y_range
        ``(V, r)`` responses projected onto the range, :math:`U_r^{\top} y_v`.
    X_range
        ``(r, p)`` fixed-effect design projected onto the range,
        :math:`U_r^{\top} X`.
    s2
        ``(r,)`` nonzero eigenvalues of :math:`Z Z^{\top}` (the range spectrum).
    gxx
        ``(p, p)`` shared null-space Gram
        :math:`G_{xx} = X^{\top} X - X_r^{\top} X_r`.
    gxy, gyy
        ``(V, p)`` / ``(V,)`` per-voxel null-space aggregates :math:`G_{xy}` and
        :math:`G_{yy}`.
    theta_init
        ``(V, 2)`` initial variance parameters
        ``[log sigma_b^2, log sigma_e^2]``.
    n0
        Null-space multiplicity :math:`n_0 = N - r` (static).
    spec
        Newton/backtracking configuration.
    block
        Optional voxel-block size for memory chunking; ``None`` maps over all
        voxels at once.

    Returns
    -------
    theta_hat : Float[Array, 'V 2']
        ``(V, 2)`` fitted variance parameters per voxel.
    beta_hat : Float[Array, 'V p']
        ``(V, p)`` fitted fixed effects per voxel.
    log_lik : Float[Array, 'V']
        ``(V,)`` profile REML log-likelihood per voxel.
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


# ---------------------------------------------------------------------------
# Fixed-effect inference quantities at the fitted theta (low-rank)
# ---------------------------------------------------------------------------


def lowrank_inference(
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
) -> Tuple[Float[Array, 'p p'], Float[Array, '2 2'], Float[Array, '2 p p']]:
    r"""Low-rank counterpart of the dense variance-component inference routine.

    Returns the same fixed-effect covariance, variance-parameter covariance, and
    Satterthwaite gradient tensor as the dense path.  The Satterthwaite gradient
    :math:`M_k = \sum_i (g_{k,i} / d_i^2)\, x_i x_i^{\top}` is the range reduction
    plus the null-space term -- which, since only the residual component loads on
    the null space (:math:`g_e = \sigma_e^2`, :math:`d_0 = \sigma_e^2`), is
    exactly :math:`G_{xx} / \sigma_e^2` added to the residual-component tensor
    :math:`M_e`.

    Parameters
    ----------
    theta
        ``(2,)`` fitted variance parameters ``[log sigma_b^2, log sigma_e^2]``.
    y_r
        ``(r,)`` response projected onto the range.
    X_r
        ``(r, p)`` fixed-effect design projected onto the range.
    s2
        ``(r,)`` nonzero eigenvalues of the range spectrum.
    gxx
        ``(p, p)`` shared null-space Gram :math:`G_{xx}`.
    gxy
        ``(p,)`` per-voxel null-space aggregate :math:`G_{xy}`.
    gyy
        Scalar per-voxel null-space aggregate :math:`G_{yy}`.
    n0
        Null-space multiplicity :math:`n_0 = N - r` (static).
    p
        Number of fixed-effect columns (static).
    spec
        Newton/backtracking configuration (supplies ridge and damping).

    Returns
    -------
    fixed_cov : Float[Array, 'p p']
        ``(p, p)`` unscaled fixed-effect covariance (the inverse normal-equation
        matrix :math:`A^{-1}` at ``theta``).
    theta_cov : Float[Array, '2 2']
        ``(2, 2)`` covariance of the variance parameters (inverse damped
        average-information matrix).
    grad_m : Float[Array, '2 p p']
        ``(2, p, p)`` per-component Satterthwaite gradient tensors
        :math:`M_k`, one ``(p, p)`` slice per variance parameter.
    """
    sb2 = jnp.exp(theta[0])
    se2 = jnp.exp(theta[1])
    _, A_inv, _, inv_d, d0 = _gls(theta, y_r, X_r, s2, gxx, gxy, p, spec.ridge)
    _, info, _ = _score_and_info(
        theta, y_r, X_r, s2, gxx, gxy, gyy, n0, p, spec.ridge
    )
    info_damped = info + spec.damping * jnp.eye(2, dtype=info.dtype)
    theta_cov, _ = _small_inv_logdet(info_damped, 2)

    g_b = sb2 * s2  # (r,) dd/dtheta_b on the range
    g_e = se2 * jnp.ones_like(s2)  # (r,) dd/dtheta_e on the range
    g = jnp.stack([g_b, g_e], axis=0)  # (2, r)
    weight = g * (inv_d * inv_d)[None, :]  # (2, r)
    grad_m = jnp.einsum('kr,rp,rq->kpq', weight, X_r, X_r)  # (2, p, p)
    # Null space: only theta_e loads there; M_e += Gxx / sigma_e^2.
    grad_m = grad_m.at[1].add(gxx / d0)
    return A_inv, theta_cov, grad_m
