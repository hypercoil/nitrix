# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
r"""
Shared diagonal penalised-REML core (single- and multi-block).

The penalty is a sum of :math:`K` disjoint diagonal blocks,
:math:`S_\lambda = \sum_k \lambda_k \operatorname{diag}(d_k)`, where each
:math:`d_k` is a row of ``d_blocks``.  Every quantity stays diagonal, so there is
no eigendecomposition and the Fellner-Schall penalty trace keeps the disjoint
closed form :math:`\operatorname{tr}(S_\lambda^{+} S_k) = \mathrm{rank}_k /
\lambda_k`.

This is the single source of truth for the profiled Gaussian-REML inner fit used
by both :mod:`nitrix.stats.gp` (a single block, :math:`K = 1` -- a GP smooth's
:math:`\lambda`) and :mod:`nitrix.stats.hgp` (:math:`K` blocks -- a population
smooth plus one per grouping level).  The single-block module wraps these with a
scalar :math:`\lambda`; the hierarchical module calls them directly with the
:math:`(K,)` vector.
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jaxtyping import Array, Float

from ..linalg._smalllinalg import small_inv_logdet

__all__ = ['reml_const', 'mb_quantities', 'mb_fs', 'mb_reml_nll']


def reml_const(n: int, n_fixed: int) -> float:
    r"""Additive constant of the profiled Gaussian REML objective :math:`-2 l_R`.

    Returns the term that depends only on the sample size :math:`n` and the number
    of fixed effects :math:`M_0`,
    :math:`(n - M_0)\,(\log(2\pi) + 1 - \log(n - M_0))`.  Including it makes the
    reported restricted log marginal likelihood the *full* one (so that AIC/BIC
    comparisons across models with differing fixed-effect structure are valid),
    and -- being constant in the smoothing parameters :math:`\lambda` and the
    kernel lengthscale :math:`\rho` -- it does not affect the :math:`\rho` search.

    Parameters
    ----------
    n : int
        Number of observations :math:`n`.
    n_fixed : int
        Number of fixed-effect (unpenalised) parameters :math:`M_0`.

    Returns
    -------
    float
        The :math:`(n, M_0)`-only additive constant of :math:`-2 l_R`.
    """
    dof = float(n - n_fixed)
    return dof * (float(np.log(2.0 * np.pi)) + 1.0 - float(np.log(dof)))


def mb_quantities(
    lam: Float[Array, ' K'],
    c: Float[Array, ' p'],
    g: Float[Array, ''],
    xtx: Float[Array, 'p p'],
    d_blocks: Float[Array, 'K p'],
    p: int,
    ridge: float,
) -> Tuple[
    Float[Array, 'p p'],
    Float[Array, ''],
    Float[Array, ' p'],
    Float[Array, ''],
    Float[Array, ''],
    Float[Array, ''],
]:
    r"""Penalised least-squares quantities at a given set of smoothing parameters.

    From the sufficient cross-products :math:`c = X^{\top} y` and
    :math:`g = y^{\top} y`, and at the smoothing parameters ``lam``, form the
    diagonal penalty :math:`S_\lambda = \sum_k \lambda_k \operatorname{diag}(d_k)`
    (with a ridge added to the diagonal) and return the fitted quantities of the
    ridged penalised normal equations.  The penalised Hessian is
    :math:`H = X^{\top} X + S_\lambda + \mathrm{ridge}\, I`, and everything else
    follows from its inverse.

    Parameters
    ----------
    lam : Float[Array, ' K']
        The :math:`K` smoothing parameters :math:`\lambda_k`, one per penalty
        block.
    c : Float[Array, ' p']
        Cross-product :math:`c = X^{\top} y` of the design and response.
    g : Float[Array, '']
        Scalar :math:`g = y^{\top} y`, the response sum of squares.
    xtx : Float[Array, 'p p']
        Gram matrix :math:`X^{\top} X` of the design.
    d_blocks : Float[Array, 'K p']
        The :math:`K` diagonal penalty blocks; row :math:`k` holds the diagonal
        :math:`d_k` of the :math:`k`-th penalty.
    p : int
        Number of coefficients :math:`p` (the design column count).
    ridge : float
        Non-negative ridge added to the diagonal of :math:`H` for conditioning.

    Returns
    -------
    v : Float[Array, 'p p']
        The inverse penalised Hessian :math:`V = H^{-1}`.
    logdet_h : Float[Array, '']
        Log-determinant :math:`\log|H|`.
    beta : Float[Array, ' p']
        Penalised coefficient estimate :math:`\beta = V c`.
    edf : Float[Array, '']
        Effective degrees of freedom :math:`\operatorname{tr}(V X^{\top} X)`.
    rss : Float[Array, '']
        Residual sum of squares :math:`\lVert y - X\beta \rVert^2`.
    d_p : Float[Array, '']
        Penalised residual sum of squares
        :math:`D_p = y^{\top} y - \beta^{\top} c` (equivalently
        :math:`\mathrm{rss} + \beta^{\top} S \beta`).
    """
    # sum_k lam_k diag(d_blocks[k]); written as an explicit weighted sum (not
    # lam @ d_blocks) so the K=1 case reduces bit-identically to the single-block
    # ``lam * d`` of nitrix.stats.gp (a matmul would reassociate the last bit).
    s_diag = jnp.sum(lam[:, None] * d_blocks, axis=0) + ridge  # (p,)
    h = xtx + jnp.diag(s_diag)
    v, logdet_h = small_inv_logdet(h, p)
    beta = v @ c
    edf = jnp.sum(v * xtx)  # tr(V X^T X)
    rss = g - 2.0 * (beta @ c) + beta @ (xtx @ beta)
    d_p = g - beta @ c
    return v, logdet_h, beta, edf, rss, d_p


def mb_fs(
    c: Float[Array, ' p'],
    g: Float[Array, ''],
    xtx: Float[Array, 'p p'],
    d_blocks: Float[Array, 'K p'],
    ranks: Float[Array, ' K'],
    n: int,
    p: int,
    n_outer: int,
    ridge: float,
    lam_floor: float,
    lam_ceil: float,
) -> Float[Array, ' K']:
    r"""Generalised Fellner-Schall selection of the smoothing parameters.

    Iteratively updates the :math:`K` smoothing parameters (one per diagonal
    penalty block) from the sufficient cross-products, running ``n_outer`` fixed
    Fellner-Schall sweeps and returning the final estimate.  For a disjoint
    diagonal penalty the trace simplifies to
    :math:`\operatorname{tr}(S_\lambda^{+} S_k) = \mathrm{rank}_k / \lambda_k`, so
    each sweep applies the multiplicative update
    :math:`\lambda_k \leftarrow \lambda_k\,(\mathrm{rank}_k / \lambda_k -
    \operatorname{tr}(V \operatorname{diag}(d_k))) / (\mathrm{energy}_k / \phi)`
    with energy :math:`\mathrm{energy}_k = \beta^{\top} \operatorname{diag}(d_k)
    \beta` and dispersion :math:`\phi = \mathrm{rss} / (n - \mathrm{edf})`.  Each
    updated value is clamped to ``[lam_floor, lam_ceil]``.

    Parameters
    ----------
    c : Float[Array, ' p']
        Cross-product :math:`c = X^{\top} y` of the design and response.
    g : Float[Array, '']
        Scalar :math:`g = y^{\top} y`, the response sum of squares.
    xtx : Float[Array, 'p p']
        Gram matrix :math:`X^{\top} X` of the design.
    d_blocks : Float[Array, 'K p']
        The :math:`K` diagonal penalty blocks; row :math:`k` holds the diagonal
        :math:`d_k` of the :math:`k`-th penalty.
    ranks : Float[Array, ' K']
        Penalty ranks :math:`\mathrm{rank}_k`, one per block.
    n : int
        Number of observations :math:`n`.
    p : int
        Number of coefficients :math:`p` (the design column count).
    n_outer : int
        Number of fixed Fellner-Schall sweeps to run.
    ridge : float
        Non-negative ridge added to the diagonal of :math:`H` for conditioning.
    lam_floor : float
        Lower clamp applied to each updated smoothing parameter.
    lam_ceil : float
        Upper clamp applied to each updated smoothing parameter.

    Returns
    -------
    Float[Array, ' K']
        The estimated smoothing parameters :math:`\lambda_k` after ``n_outer``
        sweeps.
    """

    def outer(
        lam: Float[Array, ' K'], _: Array
    ) -> Tuple[Float[Array, ' K'], None]:
        v, _, beta, edf, rss, _ = mb_quantities(
            lam, c, g, xtx, d_blocks, p, ridge
        )
        phi = rss / jnp.clip(n - edf, 1e-3, None)
        vdiag = jnp.diagonal(v)  # (p,)

        def fs(k: Array) -> Float[Array, '']:
            dk = d_blocks[k]
            tr_vd = jnp.sum(vdiag * dk)
            energy = (
                dk * beta
            ) @ beta  # = beta^T diag(d_k) beta; the dot form
            num = jnp.clip(ranks[k] - lam[k] * tr_vd, 1e-8, None)
            den = jnp.clip(energy / phi, 1e-12, None)
            return jnp.clip(num / den, lam_floor, lam_ceil)

        return jax.vmap(fs)(jnp.arange(d_blocks.shape[0])), None

    lam0 = jnp.ones((d_blocks.shape[0],), dtype=xtx.dtype)
    lam, _ = lax.scan(outer, lam0, xs=None, length=n_outer)
    return lam


def mb_reml_nll(
    d_p: Float[Array, ''],
    logdet_h: Float[Array, ''],
    lam: Float[Array, ' K'],
    ranks: Float[Array, ' K'],
    log_pdets: Float[Array, ' K'],
    n: int,
    n_fixed: int,
) -> Float[Array, '']:
    r"""Per-element restricted negative log-likelihood :math:`-2 l_R`.

    Assembles the full profiled Gaussian-REML objective (including the
    :math:`(n, M_0)` additive constant from :func:`reml_const`) from precomputed
    quantities,
    :math:`(n - M_0)\log D_p + \log|H| - \sum_k (\mathrm{rank}_k \log \lambda_k +
    \mathrm{logpdet}_k) + (n - M_0)(\log 2\pi + 1 - \log(n - M_0))`, where each
    per-block log pseudo-determinant is :math:`\mathrm{logpdet}_k = \sum_{j \in
    \text{block } k} \log d_j`.

    Parameters
    ----------
    d_p : Float[Array, '']
        Penalised residual sum of squares :math:`D_p` (from
        :func:`mb_quantities`).
    logdet_h : Float[Array, '']
        Log-determinant :math:`\log|H|` of the penalised Hessian.
    lam : Float[Array, ' K']
        The :math:`K` smoothing parameters :math:`\lambda_k`.
    ranks : Float[Array, ' K']
        Penalty ranks :math:`\mathrm{rank}_k`, one per block.
    log_pdets : Float[Array, ' K']
        Per-block log pseudo-determinants :math:`\mathrm{logpdet}_k =
        \sum_{j \in \text{block } k} \log d_j`.
    n : int
        Number of observations :math:`n`.
    n_fixed : int
        Number of fixed-effect (unpenalised) parameters :math:`M_0`.

    Returns
    -------
    Float[Array, '']
        The full restricted negative log-likelihood :math:`-2 l_R`.
    """
    log_pdet_s = jnp.sum(ranks * jnp.log(lam) + log_pdets)
    core = (
        (n - n_fixed) * jnp.log(jnp.clip(d_p, 1e-30, None))
        + logdet_h
        - log_pdet_s
    )
    return core + reml_const(n, n_fixed)
