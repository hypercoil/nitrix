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
(``Z_i^T Z_i``, ``X_i^T Z_i``, ``X_i^T X_i`` -- shared across voxels; ``Z_i^T
y_i``, ``X_i^T y_i``, ``y_i^T y_i`` -- per voxel) and the ``(p, p)`` fixed-effect
solve.  ``G`` is carried in **log-Cholesky** coordinates (``G = L L^T``, ``log``
diagonal) so it stays positive-definite under an unconstrained Newton step.

The per-voxel Newton uses the shared ``_optimise.damped_newton`` through its
**analytic-curvature** fork (``bw_score_and_ai``): the closed-form REML score and
the *average-information* (Gilmour-Thompson-Cullis) curvature, both assembled
from the same per-group Woodbury quantities -- no autodiff Hessian.  The AI is
PSD, so ``step='damped'`` is correct (unlike the raw autodiff Hessian, which is
indefinite away from the optimum and needs the saddle-free guard) and the fit
converges in ~8-10 iterations, like the dense ``_varcomp`` R1.
"""

from __future__ import annotations

from typing import Tuple, cast

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from .._batching import blocked_vmap as _blocked_vmap
from .._smalllinalg import small_inv_logdet as _small_inv_logdet
from ._optimise import damped_newton
from ._varcomp import VarCompSpec

__all__ = ['fit_blockwoodbury_reml', 'group_grams']


def _tril_layout(r: int) -> Tuple[Tuple[int, int], ...]:
    """Row-major lower-triangular ``(i, j)`` index pairs of an ``r x r`` factor."""
    return tuple((i, j) for i in range(r) for j in range(i + 1))


def _param_layout(
    r: int, diagonal: bool = False
) -> Tuple[Tuple[int, int], ...]:
    """Free ``(i, j)`` positions of the Cholesky factor for ``G``.

    ``diagonal=False`` -- the full lower triangle (``r(r+1)/2`` params): an
    **unstructured** ``r x r`` within-group covariance ``(1 + x | g)``.
    ``diagonal=True`` -- only the diagonal (``r`` params): an **independent**
    (diagonal-``G``) random effect ``(x || g)``, where intercept and slope share
    no covariance.  Both are tier-R2 (one grouping factor, block-diagonal ``V``).
    """
    if diagonal:
        return tuple((i, i) for i in range(r))
    return _tril_layout(r)


def _build_chol(
    chol_params: Float[Array, 'm'], r: int, diagonal: bool = False
) -> Float[Array, 'r r']:
    """Lower-triangular Cholesky factor ``L`` from its free parameters.

    Diagonal entries are exponentiated (positive), off-diagonal entries are
    free -- so ``G = L L^T`` is positive-definite for any real ``chol_params``.
    The loop is over the static layout (unrolled; ``r`` is tiny): the full lower
    triangle (``diagonal=False``) or just the diagonal (``diagonal=True`` -> a
    diagonal ``G``, i.e. an uncorrelated ``(x || g)`` random effect).
    """
    L = jnp.zeros((r, r), dtype=chol_params.dtype)
    for k, (i, j) in enumerate(_param_layout(r, diagonal)):
        val = jnp.exp(chol_params[k]) if i == j else chol_params[k]
        L = L.at[i, j].set(val)
    return L


def cov_re_from_chol(
    chol_params: Float[Array, 'm'], r: int, diagonal: bool = False
) -> Float[Array, 'r r']:
    """Random-effect covariance ``G = L L^T`` from the free Cholesky parameters.

    The single source of truth shared by the R2 (``reml.lme_fit``) and R2+corr
    (``_corrfit.fit_corr_lme``) result assembly -- both previously inlined
    ``_build_chol(th) @ _build_chol(th).T`` (recomputing ``L`` twice).
    """
    L = _build_chol(chol_params, r, diagonal)
    return L @ L.T


def bw_score_and_ai(
    theta: Float[Array, 'nt'],
    ztz: Float[Array, 'M r r'],
    xtz: Float[Array, 'M p r'],
    xtx: Float[Array, 'p p'],
    xtx_g: Float[Array, 'M p p'],
    zty: Float[Array, 'M r'],
    xty_g: Float[Array, 'M p'],
    yty_g: Float[Array, 'M'],
    nvec: Float[Array, 'M'],
    r: int,
    p: int,
    ridge: float,
    diagonal: bool,
) -> Tuple[Float[Array, 'nt'], Float[Array, 'nt nt']]:
    """Analytic REML **score** and **average-information** curvature at ``theta``.

    The closed-form alternative to autodiffing the profile-REML objective -- the
    analytic-curvature fork of ``_optimise.damped_newton``.  All quantities are
    per-group ``r x r`` / ``p x p`` reductions built from the Woodbury inverse
    ``V_i^{-1} = sigma_e^{-2}(I - Z_i M_i Z_i^T)``, ``M_i = (sigma_e^2 G^{-1} +
    Z_i^T Z_i)^{-1}`` -- no ``N x N`` factor, cuSOLVER-free.  ``theta = [chol(G)
    params, log sigma_e^2]``; the chain rule through the log-Cholesky / log-scale
    parameterisation enters via ``dG/dtheta_k`` (a tiny forward-mode jacobian of
    ``cp -> L L^T`` -- *not* the objective Hessian).

    The score is

        s_k = 0.5 [ tr(P V_k) - (Py)^T V_k (Py) ],   V_k = dV/dtheta_k

    (matches ``jax.grad`` of the objective to ~1e-9); the average information is
    ``AI_{kl} = 0.5 u_k^T P u_l``, ``u_k = V_k Py`` -- the PSD Gilmour-Thompson-
    Cullis curvature, so ``damped_newton(..., step='damped')`` is correct (no
    saddle-free guard needed) and converges fast.  Returns ``(score, AI)``.
    """
    se2 = jnp.exp(theta[-1])
    chol = theta[:-1]
    nt_g = chol.shape[0]
    inv_se = 1.0 / se2

    g_cov = _build_chol(chol, r, diagonal) @ _build_chol(chol, r, diagonal).T
    g_inv, _ = _small_inv_logdet(g_cov, r)
    # dG/dtheta_k: a tiny forward-mode jacobian of the chol map (not the Hessian).
    d_g = jax.jacfwd(
        lambda cp: (
            _build_chol(cp, r, diagonal) @ _build_chol(cp, r, diagonal).T
        )
    )(chol)  # (r, r, nt_g)
    d_g = jnp.moveaxis(d_g, -1, 0)  # (nt_g, r, r)

    s_mat = ztz  # S_i = Z_i^T Z_i
    k_mat = se2 * g_inv[None] + s_mat
    m_mat = jax.vmap(lambda a: _small_inv_logdet(a, r)[0])(
        k_mat
    )  # M_i (M,r,r)

    xz_m = jnp.einsum('gpr,grs->gps', xtz, m_mat)
    a_mat = (xtx - jnp.einsum('gpr,gqr->pq', xz_m, xtz)) / se2
    a_mat = a_mat + ridge * jnp.eye(p, dtype=a_mat.dtype)
    a_inv, _ = _small_inv_logdet(a_mat, p)
    xty = jnp.sum(xty_g, axis=0)
    b_vec = (xty - jnp.einsum('gpr,gr->p', xz_m, zty)) / se2
    beta = a_inv @ b_vec

    zr = zty - jnp.einsum('gpr,p->gr', xtz, beta)  # Z_i^T r_i
    xr = xty_g - jnp.einsum('gpq,q->gp', xtx_g, beta)  # X_i^T r_i
    rr = (
        yty_g
        - 2.0 * jnp.einsum('gp,p->g', xty_g, beta)
        + jnp.einsum('p,gpq,q->g', beta, xtx_g, beta)
    )  # r_i^T r_i

    sm = jnp.einsum('grs,gst->grt', s_mat, m_mat)  # S M
    ms = jnp.einsum('grs,gst->grt', m_mat, s_mat)  # M S
    pz = inv_se * (zr - jnp.einsum('grs,gs->gr', sm, zr))  # Z^T Py
    z_mz = jnp.einsum('gr,grs,gs->g', zr, m_mat, zr)
    msm = jnp.einsum('grs,gst->grt', ms, m_mat)  # M S M
    z_msmz = jnp.einsum('gr,grs,gs->g', zr, msm, zr)
    py2 = inv_se * inv_se * (rr - 2.0 * z_mz + z_msmz)  # ||Py_i||^2
    w_mat = inv_se * (
        s_mat - jnp.einsum('grs,gst->grt', sm, s_mat)
    )  # Z^T V^-1 Z
    c_mat = inv_se * (xtz - jnp.einsum('gpr,grs->gps', xtz, ms))  # X^T V^-1 Z
    xz_mxz = jnp.einsum('gpr,grs,gqs->gpq', xtz, m_mat, xtz)
    xz_msmxz = jnp.einsum('gpr,grs,gqs->gpq', xtz, msm, xtz)
    xt_v2_x = jnp.sum(
        inv_se * inv_se * (xtx_g - 2.0 * xz_mxz + xz_msmxz), axis=0
    )  # X^T V^-2 X
    tr_vinv = jnp.sum(inv_se * (nvec - jnp.einsum('grs,gsr->g', s_mat, m_mat)))

    sum_w = jnp.sum(w_mat, axis=0)
    score = []
    for k in range(nt_g):
        g_k = d_g[k]
        tr_vinv_vk = jnp.einsum('rs,sr->', g_k, sum_w)
        c_gk = jnp.einsum('gpr,rs->gps', c_mat, g_k)
        tr2 = jnp.einsum('pq,gqs,gps->', a_inv, c_mat, c_gk)
        quad = jnp.sum(jnp.einsum('gr,rs,gs->g', pz, g_k, pz))
        score.append(0.5 * (tr_vinv_vk - tr2 - quad))
    tr_ainv_xv2x = jnp.einsum('pq,qp->', a_inv, xt_v2_x)
    score.append(0.5 * se2 * (tr_vinv - tr_ainv_xv2x - jnp.sum(py2)))
    score_v = jnp.stack(score)

    # ----- average information AI_{kl} = 0.5 u_k^T P u_l -----
    x_py = inv_se * (
        xr - jnp.einsum('gpr,grs,gs->gp', xtz, m_mat, zr)
    )  # X^T Py
    zpy2 = inv_se * (pz - jnp.einsum('grs,gs->gr', sm, pz))  # Z^T V^-2 r
    xpy2 = inv_se * (x_py - jnp.einsum('gpr,grs,gs->gp', xtz, m_mat, pz))
    xpy2_tot = jnp.sum(xpy2, axis=0)  # X^T V^-2 r (p,)
    nt = nt_g + 1
    q_vec = jnp.stack(
        [jnp.einsum('gpr,rs,gs->p', c_mat, d_g[k], pz) for k in range(nt_g)]
    )  # (nt_g, p): u_k^T V^-1 X
    ai = jnp.zeros((nt, nt), dtype=se2.dtype)
    for k in range(nt_g):
        for ll in range(nt_g):
            t1 = jnp.sum(
                jnp.einsum(
                    'gr,rs,gst,tu,gu->g', pz, d_g[k], w_mat, d_g[ll], pz
                )
            )
            t2 = q_vec[k] @ a_inv @ q_vec[ll]
            ai = ai.at[k, ll].set(0.5 * (t1 - t2))
    for k in range(nt_g):
        cs = (
            0.5
            * se2
            * (
                jnp.sum(jnp.einsum('gr,rs,gs->g', pz, d_g[k], zpy2))
                - q_vec[k] @ a_inv @ xpy2_tot
            )
        )
        ai = ai.at[k, nt_g].set(cs)
        ai = ai.at[nt_g, k].set(cs)
    py_vinv_py = jnp.sum(
        inv_se * (py2 - jnp.einsum('gr,grs,gs->g', pz, m_mat, pz))
    )
    ai = ai.at[nt_g, nt_g].set(
        0.5 * se2 * se2 * (py_vinv_py - xpy2_tot @ a_inv @ xpy2_tot)
    )
    return score_v, ai


def group_grams(
    X: Float[Array, 'N p'],
    Z: Float[Array, 'N r'],
    group: Int[Array, 'N'],
    n_groups: int,
) -> Tuple[
    Float[Array, 'M r r'],
    Float[Array, 'M p r'],
    Float[Array, 'p p'],
    Float[Array, 'M p p'],
    Float[Array, 'M'],
    int,
]:
    """Per-group Gram reductions shared across voxels (data-independent of ``y``).

    Returns ``(ztz, xtz, xtx, xtx_g, nvec, n_minus_Mr)`` with ``ztz[i] = Z_i^T
    Z_i`` (``M, r, r``), ``xtz[i] = X_i^T Z_i`` (``M, p, r``), ``xtx = X^T X``
    (``p, p``), the **per-group** ``xtx_g[i] = X_i^T X_i`` (``M, p, p``) and group
    sizes ``nvec`` (``M``) -- the extra reductions the analytic AI-REML
    derivatives need -- and the scalar ``N - M r`` (the residual multiplicity in
    ``log|V|``).
    """
    r = Z.shape[-1]
    zz = Z[:, :, None] * Z[:, None, :]  # (N, r, r)
    xz = X[:, :, None] * Z[:, None, :]  # (N, p, r)
    xx = X[:, :, None] * X[:, None, :]  # (N, p, p)
    ztz = jax.ops.segment_sum(zz, group, num_segments=n_groups)
    xtz = jax.ops.segment_sum(xz, group, num_segments=n_groups)
    xtx_g = jax.ops.segment_sum(xx, group, num_segments=n_groups)
    nvec = jax.ops.segment_sum(
        jnp.ones(X.shape[0], X.dtype), group, num_segments=n_groups
    )
    xtx = jnp.sum(xtx_g, axis=0)
    n_minus_mr = X.shape[0] - n_groups * r
    return ztz, xtz, xtx, xtx_g, nvec, n_minus_mr


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
    diagonal: bool,
) -> Tuple[Float[Array, ''], Float[Array, 'p']]:
    """Profile REML negative log-likelihood (and ``beta_hat``) at ``theta``."""
    se2 = jnp.exp(theta[-1])
    L = _build_chol(theta[:-1], r, diagonal)
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
    xty_g: Float[Array, 'M p'],
    yty_g: Float[Array, 'M'],
    theta_init: Float[Array, 'nt'],
    ztz: Float[Array, 'M r r'],
    xtz: Float[Array, 'M p r'],
    xtx: Float[Array, 'p p'],
    xtx_g: Float[Array, 'M p p'],
    nvec: Float[Array, 'M'],
    n_minus_mr: int,
    r: int,
    p: int,
    spec: VarCompSpec,
    diagonal: bool,
) -> Tuple[Float[Array, 'nt'], Float[Array, 'p'], Float[Array, '']]:
    """Single-voxel block-Woodbury REML fit via the shared optimiser's
    **analytic-curvature** fork (``bw_score_and_ai``: closed-form score +
    average-information, ``step='damped'`` -- the AI is PSD, so no saddle-free
    guard is needed and convergence is fast, like the dense ``_varcomp`` R1)."""

    def nll(theta: Float[Array, 'nt']) -> Float[Array, '']:
        return _nll_and_beta(
            theta,
            ztz,
            xtz,
            xtx,
            zty,
            xty,
            yty,
            n_minus_mr,
            r,
            p,
            spec.ridge,
            diagonal,
        )[0]

    def curvature(
        theta: Float[Array, 'nt'],
    ) -> Tuple[Float[Array, 'nt'], Float[Array, 'nt nt']]:
        return bw_score_and_ai(
            theta,
            ztz,
            xtz,
            xtx,
            xtx_g,
            zty,
            xty_g,
            yty_g,
            nvec,
            r,
            p,
            spec.ridge,
            diagonal,
        )

    theta = damped_newton(
        nll, theta_init, spec=spec, curvature=curvature, step='damped'
    )
    final_nll, beta = _nll_and_beta(
        theta,
        ztz,
        xtz,
        xtx,
        zty,
        xty,
        yty,
        n_minus_mr,
        r,
        p,
        spec.ridge,
        diagonal,
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
    diagonal: bool = False,
) -> Tuple[Float[Array, 'V nt'], Float[Array, 'V p'], Float[Array, 'V']]:
    """Batched block-Woodbury REML over ``V`` voxels (single grouping factor).

    ``Z`` is the ``(N, r)`` per-observation random-covariate design (e.g.
    ``[1, x]`` for ``(1 + x | g)``); ``group`` is the ``(N,)`` group label.
    ``theta`` is ``[chol(G) params, log sigma_e^2]``: ``nt = r(r+1)/2 + 1`` for
    an unstructured ``G`` (``diagonal=False``), or ``nt = r + 1`` for a diagonal
    ``G`` (``diagonal=True``, the uncorrelated ``(x || g)`` random effect).
    Returns ``(theta_hat, beta_hat, log_lik)``; recover ``G = L L^T`` from the
    Cholesky params (via ``_build_chol(..., diagonal)``) and ``sigma_e^2 =
    exp(theta[-1])``.
    """
    p = X.shape[-1]
    r = Z.shape[-1]
    ztz, xtz, xtx, xtx_g, nvec, n_minus_mr = group_grams(X, Z, group, n_groups)

    def per_voxel(
        y: Float[Array, 'N'], th: Float[Array, 'nt']
    ) -> Tuple[Float[Array, 'nt'], Float[Array, 'p'], Float[Array, '']]:
        zty = jax.ops.segment_sum(
            Z * y[:, None], group, num_segments=n_groups
        )  # (M, r)
        xty_g = jax.ops.segment_sum(
            X * y[:, None], group, num_segments=n_groups
        )  # (M, p)
        yty_g = jax.ops.segment_sum(
            y * y, group, num_segments=n_groups
        )  # (M,)
        xty = jnp.sum(xty_g, axis=0)  # (p,)
        yty = jnp.sum(yty_g)
        return _fit_one(
            zty,
            xty,
            yty,
            xty_g,
            yty_g,
            th,
            ztz,
            xtz,
            xtx,
            xtx_g,
            nvec,
            n_minus_mr,
            r,
            p,
            spec,
            diagonal,
        )

    return cast(
        Tuple[Float[Array, 'V nt'], Float[Array, 'V p'], Float[Array, 'V']],
        _blocked_vmap(per_voxel, (Y, theta_init), block=block),
    )
