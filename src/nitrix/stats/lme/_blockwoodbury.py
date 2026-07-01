# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
r"""
Block-Woodbury REML for a single grouping factor with a correlated random
effect.

The FaST-LMM path is the fastest route for one *scalar* random effect
``(1 | g)``: a single rotation diagonalises :math:`V` and the per-voxel work is
:math:`O(N)`.  That trick does **not** extend to a correlated ``(1 + x | g)``
(an :math:`r \times r` unstructured within-group covariance :math:`G`) -- but
for a *single* grouping factor :math:`V` is still **block-diagonal across
groups**, so a per-group Woodbury identity keeps the heavy algebra at the
:math:`r \times r` (:math:`r \leq 3`) and :math:`(p, p)` scale.  This is the
solver the dispatcher routes a correlated or diagonal random slope onto; it
never forms an :math:`N \times N` matrix and avoids cuSOLVER entirely (every
solve is a tiny :func:`~nitrix.linalg._smalllinalg.small_inv_logdet`).

The model, per voxel, with :math:`M` groups (group :math:`i` has :math:`n_i`
rows, and its random covariates :math:`Z_i` are :math:`(n_i, r)` -- e.g.
``[1, x]`` for ``(1 + x)``) is

.. math::

    V = \operatorname{blockdiag}_i (\sigma_e^2 I_{n_i} + Z_i G Z_i^{\top}),
    \quad b_i \sim N(0, G).

The per-group Woodbury identity with :math:`K_i = \sigma_e^2 G^{-1} +
Z_i^{\top} Z_i` (:math:`r \times r`) gives

.. math::

    X_i^{\top} V_i^{-1} X_i &= \sigma_e^{-2}
        (X_i^{\top} X_i - (X_i^{\top} Z_i) K_i^{-1} (Z_i^{\top} X_i)) \\
    \log|V_i| &= (n_i - r) \log \sigma_e^2 + \log|G| + \log|K_i|,

so the whole REML objective is assembled from per-group Gram reductions
(:math:`Z_i^{\top} Z_i`, :math:`X_i^{\top} Z_i`, :math:`X_i^{\top} X_i` --
shared across voxels; :math:`Z_i^{\top} y_i`, :math:`X_i^{\top} y_i`,
:math:`y_i^{\top} y_i` -- per voxel) and the :math:`(p, p)` fixed-effect solve.
:math:`G` is carried in **log-Cholesky** coordinates (:math:`G = L L^{\top}`,
log diagonal) so it stays positive-definite under an unconstrained Newton step.

The per-voxel Newton iteration uses the shared damped-Newton optimiser through
its **analytic-curvature** fork (:func:`bw_score_and_ai`): the closed-form REML
score and the *average-information* (Gilmour-Thompson-Cullis) curvature, both
assembled from the same per-group Woodbury quantities -- no autodiff Hessian.
The average information is positive semi-definite, so a damped step is correct
(unlike the raw autodiff Hessian, which is indefinite away from the optimum and
needs a saddle-free guard) and the fit converges in roughly 8-10 iterations.
"""

from __future__ import annotations

from typing import Tuple, cast

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from ...linalg._smalllinalg import small_inv_logdet as _small_inv_logdet
from .._batching import blocked_vmap as _blocked_vmap
from .._optimise import damped_newton
from ._recov import _build_chol
from ._varcomp import VarCompSpec

__all__ = ['fit_blockwoodbury_reml', 'group_grams', 'bw_inference']


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
    r"""Analytic REML score and average-information curvature at ``theta``.

    The closed-form alternative to autodiffing the profile-REML objective -- the
    analytic-curvature fork consumed by the damped-Newton optimiser.  All
    quantities are per-group :math:`r \times r` / :math:`p \times p` reductions
    built from the Woodbury inverse :math:`V_i^{-1} = \sigma_e^{-2}
    (I - Z_i M_i Z_i^{\top})`, with :math:`M_i = (\sigma_e^2 G^{-1} +
    Z_i^{\top} Z_i)^{-1}` -- no :math:`N \times N` factor, and no cuSOLVER.  The
    parameter vector is :math:`\theta = [\operatorname{chol}(G)\ \text{params},
    \log \sigma_e^2]`; the chain rule through the log-Cholesky / log-scale
    parameterisation enters via :math:`dG/d\theta_k` (a tiny forward-mode
    Jacobian of the map :math:`cp \mapsto L L^{\top}` -- *not* the objective
    Hessian).

    The score is

    .. math::

        s_k = \tfrac{1}{2}
            [\operatorname{tr}(P V_k) - (Py)^{\top} V_k (Py)],
        \quad V_k = dV/d\theta_k,

    which matches the automatic gradient of the objective to about
    :math:`10^{-9}`; the average information is :math:`AI_{kl} =
    \tfrac{1}{2} u_k^{\top} P u_l` with :math:`u_k = V_k Py` -- the positive
    semi-definite Gilmour-Thompson-Cullis curvature, so a damped Newton step is
    correct (no saddle-free guard needed) and converges fast.

    Parameters
    ----------
    theta : Float[Array, 'nt']
        Unconstrained parameter vector :math:`[\operatorname{chol}(G)\
        \text{params}, \log \sigma_e^2]`, of length ``nt``.
    ztz : Float[Array, 'M r r']
        Per-group random-covariate Gram matrices :math:`Z_i^{\top} Z_i`.
    xtz : Float[Array, 'M p r']
        Per-group cross-Gram matrices :math:`X_i^{\top} Z_i`.
    xtx : Float[Array, 'p p']
        Total fixed-effect Gram matrix :math:`X^{\top} X`.
    xtx_g : Float[Array, 'M p p']
        Per-group fixed-effect Gram matrices :math:`X_i^{\top} X_i`.
    zty : Float[Array, 'M r']
        Per-group random-covariate moments :math:`Z_i^{\top} y_i` for this
        voxel.
    xty_g : Float[Array, 'M p']
        Per-group fixed-effect moments :math:`X_i^{\top} y_i` for this voxel.
    yty_g : Float[Array, 'M']
        Per-group response sums of squares :math:`y_i^{\top} y_i` for this
        voxel.
    nvec : Float[Array, 'M']
        Group sizes :math:`n_i`.
    r : int
        Number of random-effect covariates (columns of :math:`Z`).
    p : int
        Number of fixed-effect covariates (columns of :math:`X`).
    ridge : float
        Ridge added to the diagonal of the :math:`(p, p)` fixed-effect matrix
        before inversion, for numerical stability.
    diagonal : bool
        If ``True``, :math:`G` is constrained to be diagonal (uncorrelated
        random slopes); otherwise it is unstructured.

    Returns
    -------
    score : Float[Array, 'nt']
        The REML score (gradient of the profile objective) at ``theta``.
    ai : Float[Array, 'nt nt']
        The average-information curvature matrix at ``theta``.
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


def bw_inference(
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
    damping: float,
) -> Tuple[Float[Array, 'p p'], Float[Array, 'nt nt'], Float[Array, 'nt p p']]:
    r"""Fixed-effect contrast inference at the converged ``theta``.

    The block-Woodbury analogue of the dense variance-component inference
    routine -- the three contrast-independent tensors the Satterthwaite /
    Kenward-Roger denominator-df machinery for :math:`t` and :math:`F`
    contrasts consumes:

    - ``fixed_cov`` :math:`= (X^{\top} V^{-1} X)^{-1}`, the estimated
      :math:`\operatorname{Cov}(\hat{\beta})` (the Woodbury ``a_inv``).
    - ``theta_cov`` :math:`= (AI + \text{damping}\, I)^{-1}`, the estimated
      :math:`\operatorname{Cov}(\hat{\theta})`, from the same average
      information :func:`bw_score_and_ai` forms (so identical to the curvature
      the fit converged on).
    - ``grad_m[k]`` :math:`= M_k = X^{\top} V^{-1} (dV/d\theta_k) V^{-1} X` --
      with :math:`dV_g/d\theta_k = Z_g (dG/d\theta_k) Z_g^{\top}` for the
      :math:`G` (Cholesky) parameters, so :math:`M_k = \sum_g C_g
      (dG/d\theta_k) C_g^{\top}` with :math:`C_g = X_g^{\top} V_g^{-1} Z_g`;
      and :math:`dV/d(\log \sigma_e^2) = \sigma_e^2 I`, so the residual
      coordinate is :math:`M_{\text{last}} = \sigma_e^2 (X^{\top} V^{-2} X)`.

    Runs **once** per voxel after the fit (not in the Newton loop), so it
    recomputes the per-group Woodbury pieces it needs rather than threading them
    out of the hot path.

    Parameters
    ----------
    theta : Float[Array, 'nt']
        Converged parameter vector :math:`[\operatorname{chol}(G)\
        \text{params}, \log \sigma_e^2]`.
    ztz : Float[Array, 'M r r']
        Per-group random-covariate Gram matrices :math:`Z_i^{\top} Z_i`.
    xtz : Float[Array, 'M p r']
        Per-group cross-Gram matrices :math:`X_i^{\top} Z_i`.
    xtx : Float[Array, 'p p']
        Total fixed-effect Gram matrix :math:`X^{\top} X`.
    xtx_g : Float[Array, 'M p p']
        Per-group fixed-effect Gram matrices :math:`X_i^{\top} X_i`.
    zty : Float[Array, 'M r']
        Per-group random-covariate moments :math:`Z_i^{\top} y_i` for this
        voxel.
    xty_g : Float[Array, 'M p']
        Per-group fixed-effect moments :math:`X_i^{\top} y_i` for this voxel.
    yty_g : Float[Array, 'M']
        Per-group response sums of squares :math:`y_i^{\top} y_i` for this
        voxel.
    nvec : Float[Array, 'M']
        Group sizes :math:`n_i`.
    r : int
        Number of random-effect covariates (columns of :math:`Z`).
    p : int
        Number of fixed-effect covariates (columns of :math:`X`).
    ridge : float
        Ridge added to the diagonal of the :math:`(p, p)` fixed-effect matrix
        before inversion.
    diagonal : bool
        If ``True``, :math:`G` is constrained to be diagonal; otherwise it is
        unstructured.
    damping : float
        Ridge added to the diagonal of the average information before inverting
        it to form ``theta_cov``.

    Returns
    -------
    fixed_cov : Float[Array, 'p p']
        The fixed-effect covariance :math:`(X^{\top} V^{-1} X)^{-1}`.
    theta_cov : Float[Array, 'nt nt']
        The variance-component covariance :math:`(AI + \text{damping}\,
        I)^{-1}`.
    grad_m : Float[Array, 'nt p p']
        The stacked matrices :math:`M_k = X^{\top} V^{-1} (dV/d\theta_k)
        V^{-1} X`, one :math:`(p, p)` slice per variance-component coordinate.
    """
    # theta_cov from the same average information the fit used.
    _, ai = bw_score_and_ai(
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
        ridge,
        diagonal,
    )
    nt = ai.shape[0]
    theta_cov, _ = _small_inv_logdet(
        ai + damping * jnp.eye(nt, dtype=ai.dtype), nt
    )

    # Woodbury pieces for fixed_cov + grad_m (matching bw_score_and_ai exactly).
    se2 = jnp.exp(theta[-1])
    chol = theta[:-1]
    nt_g = chol.shape[0]
    inv_se = 1.0 / se2
    g_cov = _build_chol(chol, r, diagonal) @ _build_chol(chol, r, diagonal).T
    g_inv, _ = _small_inv_logdet(g_cov, r)
    d_g = jnp.moveaxis(
        jax.jacfwd(
            lambda cp: (
                _build_chol(cp, r, diagonal) @ _build_chol(cp, r, diagonal).T
            )
        )(chol),
        -1,
        0,
    )  # (nt_g, r, r)
    k_mat = se2 * g_inv[None] + ztz
    m_mat = jax.vmap(lambda a: _small_inv_logdet(a, r)[0])(k_mat)  # (M, r, r)
    xz_m = jnp.einsum('gpr,grs->gps', xtz, m_mat)
    a_mat = (xtx - jnp.einsum('gpr,gqr->pq', xz_m, xtz)) / se2
    a_mat = a_mat + ridge * jnp.eye(p, dtype=a_mat.dtype)
    a_inv, _ = _small_inv_logdet(a_mat, p)  # fixed_cov = (X^T V^-1 X)^-1
    ms = jnp.einsum('grs,gst->grt', m_mat, ztz)  # M S
    c_mat = inv_se * (xtz - jnp.einsum('gpr,grs->gps', xtz, ms))  # X^T V^-1 Z
    xz_mxz = jnp.einsum('gpr,grs,gqs->gpq', xtz, m_mat, xtz)
    msm = jnp.einsum('grs,gst->grt', ms, m_mat)
    xz_msmxz = jnp.einsum('gpr,grs,gqs->gpq', xtz, msm, xtz)
    xt_v2_x = jnp.sum(
        inv_se * inv_se * (xtx_g - 2.0 * xz_mxz + xz_msmxz), axis=0
    )  # X^T V^-2 X
    grad_g = jnp.stack(
        [
            jnp.einsum('gpr,rs,gqs->pq', c_mat, d_g[k], c_mat)
            for k in range(nt_g)
        ]
    )  # (nt_g, p, p)
    grad_m = jnp.concatenate([grad_g, (se2 * xt_v2_x)[None]], axis=0)
    return a_inv, theta_cov, grad_m


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
    r"""Per-group Gram reductions shared across voxels (independent of ``y``).

    Reduces the fixed-effect design :math:`X`, the random-effect design
    :math:`Z`, and the group labels into the per-group second-moment matrices
    that every per-voxel fit reuses.  Because these depend only on the design
    and not on any response, they are computed once and broadcast across all
    voxels.

    Parameters
    ----------
    X : Float[Array, 'N p']
        Fixed-effect design matrix, ``N`` observations by ``p`` covariates.
    Z : Float[Array, 'N r']
        Random-effect design matrix, ``N`` observations by ``r`` covariates.
    group : Int[Array, 'N']
        Group label in ``[0, n_groups)`` for each observation.
    n_groups : int
        Number of groups :math:`M`.

    Returns
    -------
    ztz : Float[Array, 'M r r']
        Per-group random-covariate Gram matrices :math:`Z_i^{\top} Z_i`.
    xtz : Float[Array, 'M p r']
        Per-group cross-Gram matrices :math:`X_i^{\top} Z_i`.
    xtx : Float[Array, 'p p']
        Total fixed-effect Gram matrix :math:`X^{\top} X`.
    xtx_g : Float[Array, 'M p p']
        Per-group fixed-effect Gram matrices :math:`X_i^{\top} X_i` -- the
        extra reductions the analytic average-information REML derivatives need.
    nvec : Float[Array, 'M']
        Group sizes :math:`n_i`.
    n_minus_mr : int
        The scalar :math:`N - M r`, the residual multiplicity that appears in
        :math:`\log|V|`.
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
    r"""Profile REML negative log-likelihood and ``beta_hat`` at ``theta``.

    Assembles the per-group Woodbury pieces from the shared Gram reductions and
    the per-voxel response moments, then forms the profile-REML objective
    (with the fixed effects profiled out) together with the corresponding
    generalised-least-squares estimate of the fixed effects.

    Parameters
    ----------
    theta : Float[Array, 'nt']
        Parameter vector :math:`[\operatorname{chol}(G)\ \text{params},
        \log \sigma_e^2]`.
    ztz : Float[Array, 'M r r']
        Per-group random-covariate Gram matrices :math:`Z_i^{\top} Z_i`.
    xtz : Float[Array, 'M p r']
        Per-group cross-Gram matrices :math:`X_i^{\top} Z_i`.
    xtx : Float[Array, 'p p']
        Total fixed-effect Gram matrix :math:`X^{\top} X`.
    zty : Float[Array, 'M r']
        Per-group random-covariate moments :math:`Z_i^{\top} y_i`.
    xty : Float[Array, 'p']
        Total fixed-effect moment :math:`X^{\top} y`.
    yty : Float[Array, '']
        Total response sum of squares :math:`y^{\top} y`.
    n_minus_mr : int
        The residual multiplicity :math:`N - M r`.
    r : int
        Number of random-effect covariates.
    p : int
        Number of fixed-effect covariates.
    ridge : float
        Ridge added to the diagonal of the :math:`(p, p)` fixed-effect matrix
        before inversion.
    diagonal : bool
        If ``True``, :math:`G` is constrained to be diagonal; otherwise it is
        unstructured.

    Returns
    -------
    nll : Float[Array, '']
        The profile REML negative log-likelihood at ``theta``.
    beta : Float[Array, 'p']
        The generalised-least-squares fixed-effect estimate :math:`\hat{\beta}`
        at ``theta``.
    """
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
    r"""Single-voxel block-Woodbury REML fit via the analytic-curvature Newton.

    Drives the shared damped-Newton optimiser on the profile-REML objective,
    supplying the closed-form score and average-information curvature from
    :func:`bw_score_and_ai` rather than an autodiff Hessian.  Because the
    average information is positive semi-definite, a damped step is correct (no
    saddle-free guard is needed) and convergence is fast.

    Parameters
    ----------
    zty : Float[Array, 'M r']
        Per-group random-covariate moments :math:`Z_i^{\top} y_i`.
    xty : Float[Array, 'p']
        Total fixed-effect moment :math:`X^{\top} y`.
    yty : Float[Array, '']
        Total response sum of squares :math:`y^{\top} y`.
    xty_g : Float[Array, 'M p']
        Per-group fixed-effect moments :math:`X_i^{\top} y_i`.
    yty_g : Float[Array, 'M']
        Per-group response sums of squares :math:`y_i^{\top} y_i`.
    theta_init : Float[Array, 'nt']
        Initial parameter vector for the Newton iteration.
    ztz : Float[Array, 'M r r']
        Per-group random-covariate Gram matrices :math:`Z_i^{\top} Z_i`.
    xtz : Float[Array, 'M p r']
        Per-group cross-Gram matrices :math:`X_i^{\top} Z_i`.
    xtx : Float[Array, 'p p']
        Total fixed-effect Gram matrix :math:`X^{\top} X`.
    xtx_g : Float[Array, 'M p p']
        Per-group fixed-effect Gram matrices :math:`X_i^{\top} X_i`.
    nvec : Float[Array, 'M']
        Group sizes :math:`n_i`.
    n_minus_mr : int
        The residual multiplicity :math:`N - M r`.
    r : int
        Number of random-effect covariates.
    p : int
        Number of fixed-effect covariates.
    spec : VarCompSpec
        Fit configuration: supplies the ridge, the Newton keyword arguments,
        and (for inference) the damping.
    diagonal : bool
        If ``True``, :math:`G` is constrained to be diagonal; otherwise it is
        unstructured.

    Returns
    -------
    theta : Float[Array, 'nt']
        The converged parameter vector.
    beta : Float[Array, 'p']
        The fixed-effect estimate :math:`\hat{\beta}` at the optimum.
    log_lik : Float[Array, '']
        The maximised profile REML log-likelihood (the negated final objective).
    """

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
        nll,
        theta_init,
        **spec.newton_kwargs,
        curvature=curvature,
        step='damped',
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
    inference: bool = False,
) -> Tuple[Float[Array, 'V nt'], ...]:
    r"""Batched block-Woodbury REML over ``V`` voxels (single grouping factor).

    Precomputes the design-only Gram reductions once with :func:`group_grams`,
    then runs a per-voxel block-Woodbury REML fit (:func:`_fit_one`) across all
    ``V`` voxels via a blocked vmap.  The parameter vector ``theta`` is
    :math:`[\operatorname{chol}(G)\ \text{params}, \log \sigma_e^2]`, with
    :math:`nt = r(r+1)/2 + 1` for an unstructured :math:`G` (``diagonal=False``)
    or :math:`nt = r + 1` for a diagonal :math:`G` (``diagonal=True``, the
    uncorrelated ``(x || g)`` random effect).  Recover :math:`G = L L^{\top}`
    from the Cholesky parameters and :math:`\sigma_e^2 = \exp(\theta[-1])`.

    Parameters
    ----------
    Y : Float[Array, 'V N']
        The response for each of ``V`` voxels over ``N`` observations.
    X : Float[Array, 'N p']
        Fixed-effect design matrix, shared across voxels.
    Z : Float[Array, 'N r']
        Per-observation random-covariate design (e.g. ``[1, x]`` for
        ``(1 + x | g)``), shared across voxels.
    group : Int[Array, 'N']
        Group label in ``[0, n_groups)`` for each observation.
    n_groups : int
        Number of groups :math:`M`.
    theta_init : Float[Array, 'V nt']
        Per-voxel initial parameter vectors for the Newton iteration.
    spec : VarCompSpec, optional
        Fit configuration (ridge, Newton keyword arguments, damping).
    block : int or None, optional
        Voxel block size for the blocked vmap; ``None`` processes all voxels in
        a single block.
    diagonal : bool, optional
        If ``True``, constrain :math:`G` to be diagonal (uncorrelated random
        slopes); otherwise fit an unstructured :math:`G`.
    inference : bool, optional
        If ``True``, additionally return the per-voxel contrast inference
        tensors from :func:`bw_inference`.

    Returns
    -------
    tuple of Array
        By default the 3-tuple ``(theta_hat, beta_hat, log_lik)`` with shapes
        ``(V, nt)``, ``(V, p)`` and ``(V,)``.  With ``inference=True`` the
        6-tuple ``(theta_hat, beta_hat, log_lik, fixed_cov, theta_cov,
        grad_m)``, where ``fixed_cov`` is ``(V, p, p)``, ``theta_cov`` is
        ``(V, nt, nt)`` and ``grad_m`` is ``(V, nt, p, p)`` (see
        :func:`bw_inference`).
    """
    p = X.shape[-1]
    r = Z.shape[-1]
    ztz, xtz, xtx, xtx_g, nvec, n_minus_mr = group_grams(X, Z, group, n_groups)

    def per_voxel(
        y: Float[Array, 'N'], th: Float[Array, 'nt']
    ) -> Tuple[Array, ...]:
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
        theta, beta, log_lik = _fit_one(
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
        if not inference:
            return theta, beta, log_lik
        fixed_cov, theta_cov, grad_m = bw_inference(
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
            spec.damping,
        )
        return theta, beta, log_lik, fixed_cov, theta_cov, grad_m

    return cast(
        Tuple[Float[Array, 'V nt'], ...],
        _blocked_vmap(per_voxel, (Y, theta_init), block=block),
    )
