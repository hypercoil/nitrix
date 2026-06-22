# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Shared diagonal penalised-REML core (single- and multi-block).

The penalty is a sum of ``K`` disjoint diagonal blocks
``S_lambda = sum_k lam_k diag(d_blocks[k])``.  Every quantity stays diagonal, so
there is no eigendecomposition and the Fellner-Schall penalty trace keeps the
disjoint closed form ``tr(S_lambda^+ S_k) = rank_k / lam_k``.

This is the single source of truth for the profiled Gaussian-REML inner fit used
by both :mod:`nitrix.stats.gp` (a single block, ``K = 1`` -- a GP smooth's
``lambda``) and :mod:`nitrix.stats.hgp` (``K`` blocks -- a population smooth plus
one per grouping level).  ``gp`` wraps these with a scalar ``lambda``; ``hgp``
calls them directly with the ``(K,)`` vector.
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
    """The ``(n, M_0)``-only additive constant of the profiled Gaussian REML
    ``-2 l_R``:  ``(n - M_0)(log(2 pi) + 1 - log(n - M_0))``.  Including it makes
    ``log_mlik`` the *full* restricted log marginal likelihood (so AIC/BIC across
    models with different fixed-effect structure are valid), and -- being constant
    in ``(lambda, rho)`` -- it does not move the ``rho`` search."""
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
    """At smoothing parameters ``lam``, return ``(V, log|H|, beta, edf, rss, D_p)``
    from the cross-products ``c = X^T y`` and ``g = y^T y``, for the diagonal
    penalty ``S_lambda = sum_k lam_k diag(d_blocks[k])``.

    ``H = X^T X + S_lambda + ridge I``; ``V = H^{-1}``; ``beta = V c``;
    ``edf = tr(V X^T X)``; ``rss = ||y - X beta||^2``; ``D_p = y^T y - beta^T c``
    is the penalised residual sum of squares (``= rss + beta^T S beta``)."""
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
    """Generalised Fellner-Schall selection of the ``K`` smoothing parameters (one
    per diagonal penalty block), from cross-products.

    ``tr(S_lambda^+ S_k) = rank_k / lam_k`` for a disjoint diagonal penalty, so the
    update is ``lam_k <- lam_k (rank_k / lam_k - tr(V diag(d_k))) / (energy_k /
    phi)`` with ``energy_k = beta^T diag(d_k) beta`` and ``phi = rss / (n - edf)``.
    """

    def outer(
        lam: Float[Array, ' K'], _: Array
    ) -> Tuple[Float[Array, ' K'], None]:
        v, _, beta, edf, rss, _ = mb_quantities(lam, c, g, xtx, d_blocks, p, ridge)
        phi = rss / jnp.clip(n - edf, 1e-3, None)
        vdiag = jnp.diagonal(v)  # (p,)

        def fs(k: Array) -> Float[Array, '']:
            dk = d_blocks[k]
            tr_vd = jnp.sum(vdiag * dk)
            energy = (dk * beta) @ beta  # = beta^T diag(d_k) beta; the dot form
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
    """Per-element restricted negative log-likelihood ``-2 l_R`` (full, incl. the
    ``(n, M_0)`` constant):  ``(n - M_0) log D_p + log|H| - sum_k (rank_k log lam_k
    + log_pdet_k) + (n - M_0)(log 2pi + 1 - log(n - M_0))`` with ``log_pdet_k =
    sum_{j in block k} log d_j``."""
    log_pdet_s = jnp.sum(ranks * jnp.log(lam) + log_pdets)
    core = (
        (n - n_fixed) * jnp.log(jnp.clip(d_p, 1e-30, None))
        + logdet_h - log_pdet_s
    )
    return core + reml_const(n, n_fixed)
