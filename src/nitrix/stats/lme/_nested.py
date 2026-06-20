# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Nested random-effects REML for a two-level hierarchy ``(1 | g1/g2)`` (tier R3).

A nested design has an outer grouping factor ``g1`` (``q1`` levels) and an inner
factor ``g2`` **nested within** ``g1`` (each inner level belongs to exactly one
outer level; the ``q2`` distinct ``(g1, g2)`` combinations are the *sublevels*).
The model, per voxel::

    y = X beta + Z1 b1 + Z2 b2 + eps
    b1 ~ N(0, sigma1^2 I_{q1}),  b2 ~ N(0, sigma2^2 I_{q2}),  eps ~ N(0, sigma_e^2 I)
    V  = sigma1^2 Z1 Z1^T + sigma2^2 Z2 Z2^T + sigma_e^2 I

Because the inner factor is nested, ``V`` is **block-diagonal across the outer
factor** (different outer blocks share no random effect), and *within* an outer
block it is a two-level telescoping structure: a compound-symmetry block per
inner sublevel (``sigma_e^2 I + sigma2^2 11^T``) plus a rank-one outer term
(``sigma1^2 11^T`` over the whole block).  Both levels invert in closed form by
Sherman-Morrison -- the **telescoping Woodbury** -- so ``V^{-1}`` / ``log|V|``
never form an ``N x N`` (or even ``n_i x n_i``) factor:

- inner: ``C_ij^{-1} = sigma_e^{-2}[I - c_ij 11^T]``, ``c_ij = sigma2^2 /
  (sigma_e^2 + n_ij sigma2^2)`` -- a rank-one update per sublevel ``j``;
- outer: ``V_i^{-1} = A_i^{-1} - d_i (A_i^{-1} 1)(A_i^{-1} 1)^T``,
  ``d_i = sigma1^2 / (1 + sigma1^2 1^T A_i^{-1} 1)``, with ``A_i^{-1}`` the
  block-diagonal inner inverse.

Every per-block quantity (``X_i^T V_i^{-1} X_i``, ``X_i^T V_i^{-1} y_i``,
``y_i^T V_i^{-1} y_i``, ``log|V_i|``) is assembled from **per-sublevel
sufficient statistics** (counts and group-sums of ``X`` / ``y``) aggregated to
the outer factor by ``segment_sum`` -- the same analytic-objective + damped
autodiff-Newton recipe as ``_blockwoodbury`` / ``_varcomp``, over the three
variance components ``theta = [log sigma1^2, log sigma2^2, log sigma_e^2]``.  No
``N x N`` intermediate, cuSOLVER-free (every solve a tiny ``small_inv_logdet``).
"""

from __future__ import annotations

from typing import NamedTuple, Optional, Tuple, cast

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int

from ...linalg._smalllinalg import small_inv_logdet
from .._batching import blocked_vmap
from .._optimise import damped_newton
from ._varcomp import VarCompSpec

__all__ = ['NestedStats', 'nested_layout', 'fit_nested_reml']


class NestedStats(NamedTuple):
    """Shared (``y``-independent) sufficient statistics for a nested design."""

    obs_sub: Int[Array, 'N']  # sublevel index per observation (0..q2-1)
    obs_outer: Int[Array, 'N']  # outer-factor index per observation (0..q1-1)
    sub_parent: Int[Array, 'q2']  # outer index of each sublevel
    nsub: Float[Array, 'q2']  # observations per sublevel
    sx: Float[Array, 'q2 p']  # per-sublevel sum of X rows
    n_outer: Float[Array, 'q1']  # observations per outer level
    gxx: Float[Array, 'q1 p p']  # per-outer-level sum of x x^T
    gx1: Float[Array, 'q1 p']  # per-outer-level sum of X rows
    q1: int
    q2: int


def nested_layout(
    outer: Int[Array, 'N'],
    inner: Int[Array, 'N'],
    X: Float[Array, 'N p'],
) -> NestedStats:
    """Build the nested-design sufficient statistics from the two factors.

    ``outer`` (``g1``) and ``inner`` (``g2``) are ``(N,)`` integer labels; each
    distinct ``(outer, inner)`` pair is a sublevel (so ``inner`` may be coded
    within-``outer`` or globally -- the pair is what counts).  The per-sublevel
    and per-outer-level reductions of the shared design ``X`` are computed once
    (data-independent of ``y``).
    """
    o_np = np.asarray(outer)
    i_np = np.asarray(inner)
    q1 = int(o_np.max()) + 1
    # Distinct (outer, inner) pairs -> contiguous sublevel ids.  np.unique returns
    # (unique, first_index, inverse) in that fixed order with both flags set.
    pairs = o_np.astype(np.int64) * (int(i_np.max()) + 1) + i_np.astype(
        np.int64
    )
    _, sub_first, obs_sub_np = np.unique(
        pairs, return_index=True, return_inverse=True
    )
    obs_sub_np = np.asarray(obs_sub_np).reshape(-1).astype(np.int64)
    q2 = int(obs_sub_np.max()) + 1
    sub_parent_np = o_np[sub_first].astype(np.int64)

    obs_sub = jnp.asarray(obs_sub_np)
    obs_outer = jnp.asarray(o_np.astype(np.int64))
    sub_parent = jnp.asarray(sub_parent_np)
    nsub = jax.ops.segment_sum(
        jnp.ones((X.shape[0],), X.dtype), obs_sub, num_segments=q2
    )
    sx = jax.ops.segment_sum(X, obs_sub, num_segments=q2)
    n_outer = jax.ops.segment_sum(
        jnp.ones((X.shape[0],), X.dtype), obs_outer, num_segments=q1
    )
    gxx = jax.ops.segment_sum(
        X[:, :, None] * X[:, None, :], obs_outer, num_segments=q1
    )
    gx1 = jax.ops.segment_sum(X, obs_outer, num_segments=q1)
    return NestedStats(
        obs_sub=obs_sub,
        obs_outer=obs_outer,
        sub_parent=sub_parent,
        nsub=nsub,
        sx=sx,
        n_outer=n_outer,
        gxx=gxx,
        gx1=gx1,
        q1=q1,
        q2=q2,
    )


def _nll_and_beta(
    theta: Float[Array, '3'],
    stats: NestedStats,
    sy: Float[Array, 'q2'],
    gxy: Float[Array, 'q1 p'],
    gyy: Float[Array, 'q1'],
    sy1: Float[Array, 'q1'],
    p: int,
    n: int,
    ridge: float,
) -> Tuple[Float[Array, ''], Float[Array, 'p']]:
    """Profile REML negative log-likelihood (and ``beta_hat``) at ``theta``."""
    s1 = jnp.exp(theta[0])  # sigma1^2 (outer)
    s2 = jnp.exp(theta[1])  # sigma2^2 (inner)
    se = jnp.exp(theta[2])  # sigma_e^2 (residual)
    parent = stats.sub_parent
    q1 = stats.q1
    nsub = stats.nsub
    sx = stats.sx

    # Inner (sublevel) Sherman-Morrison weight c_j and the outer aggregates of
    # the c_j-weighted rank-one terms.
    c = s2 / (se + nsub * s2)  # (q2,)

    def seg(v: Array) -> Array:
        return jax.ops.segment_sum(v, parent, num_segments=q1)

    csxx = seg(
        (c[:, None, None]) * (sx[:, :, None] * sx[:, None, :])
    )  # (q1, p, p) sum_j c_j sx_j sx_j^T
    csxn = seg((c * nsub)[:, None] * sx)  # (q1, p)  sum_j c_j n_j sx_j
    cnn = seg(c * nsub * nsub)  # (q1,)    sum_j c_j n_j^2
    csy = seg((c * sy)[:, None] * sx)  # (q1, p) sum_j c_j sy_j sx_j
    csyn = seg(c * sy * nsub)  # (q1,)   sum_j c_j sy_j n_j
    csyy = seg(c * sy * sy)  # (q1,)   sum_j c_j sy_j^2
    # log|C_ij| pieces aggregated to the outer level.
    logdet_inner = seg(
        (nsub - 1.0) * theta[2] + jnp.log(se + nsub * s2)
    )  # (q1,)

    # A_i^{-1} Grams (inner inverse), per outer level (divide by sigma_e^2).
    xtax = (stats.gxx - csxx) / se  # (q1, p, p)
    xta1 = (stats.gx1 - csxn) / se  # (q1, p)   X_i^T A^-1 1
    s_i = (stats.n_outer - cnn) / se  # (q1,)    1^T A^-1 1
    xtay = (gxy - csy) / se  # (q1, p)
    ya1 = (sy1 - csyn) / se  # (q1,)    y_i^T A^-1 1
    yay = (gyy - csyy) / se  # (q1,)

    # Outer Sherman-Morrison (rank-one sigma1^2 11^T over each outer block).
    d = s1 / (1.0 + s1 * s_i)  # (q1,)
    xtvx_i = xtax - d[:, None, None] * (xta1[:, :, None] * xta1[:, None, :])
    xtvy_i = xtay - (d * ya1)[:, None] * xta1
    yvy_i = yay - d * ya1 * ya1
    logdet_i = logdet_inner + jnp.log(1.0 + s1 * s_i)

    xtvx = jnp.sum(xtvx_i, axis=0) + ridge * jnp.eye(p, dtype=xtvx_i.dtype)
    xtvy = jnp.sum(xtvy_i, axis=0)
    yvy = jnp.sum(yvy_i)
    logdet_v = jnp.sum(logdet_i)

    xtvx_inv, logdet_xtvx = small_inv_logdet(xtvx, p)
    beta = xtvx_inv @ xtvy
    rss = yvy - beta @ xtvy
    nll = 0.5 * (logdet_v + logdet_xtvx + rss)
    return nll, beta


def _fit_one(
    sy: Float[Array, 'q2'],
    gxy: Float[Array, 'q1 p'],
    gyy: Float[Array, 'q1'],
    sy1: Float[Array, 'q1'],
    theta_init: Float[Array, '3'],
    stats: NestedStats,
    p: int,
    n: int,
    spec: VarCompSpec,
) -> Tuple[Float[Array, '3'], Float[Array, 'p'], Float[Array, '']]:
    """Single-voxel nested REML fit via the shared saddle-free Newton
    (``_optimise.damped_newton``)."""

    def nll(theta: Float[Array, '3']) -> Float[Array, '']:
        return _nll_and_beta(
            theta, stats, sy, gxy, gyy, sy1, p, n, spec.ridge
        )[0]

    theta = damped_newton(nll, theta_init, **spec.newton_kwargs)
    final_nll, beta = _nll_and_beta(
        theta, stats, sy, gxy, gyy, sy1, p, n, spec.ridge
    )
    return theta, beta, -final_nll


def fit_nested_reml(
    Y: Float[Array, 'V N'],
    X: Float[Array, 'N p'],
    outer: Int[Array, 'N'],
    inner: Int[Array, 'N'],
    theta_init: Float[Array, 'V 3'],
    *,
    spec: VarCompSpec = VarCompSpec(),
    block: Optional[int] = None,
) -> Tuple[Float[Array, 'V 3'], Float[Array, 'V p'], Float[Array, 'V']]:
    """Batched nested-RE REML over ``V`` voxels (one two-level hierarchy).

    Returns ``(theta_hat, beta_hat, log_lik)`` with ``theta = [log sigma1^2,
    log sigma2^2, log sigma_e^2]`` (outer / inner / residual variances).
    """
    n, p = X.shape
    stats = nested_layout(outer, inner, X)
    obs_sub, obs_outer = stats.obs_sub, stats.obs_outer
    q1, q2 = stats.q1, stats.q2

    def per_voxel(
        y: Float[Array, 'N'], th: Float[Array, '3']
    ) -> Tuple[Float[Array, '3'], Float[Array, 'p'], Float[Array, '']]:
        sy = jax.ops.segment_sum(y, obs_sub, num_segments=q2)
        gxy = jax.ops.segment_sum(X * y[:, None], obs_outer, num_segments=q1)
        gyy = jax.ops.segment_sum(y * y, obs_outer, num_segments=q1)
        sy1 = jax.ops.segment_sum(y, obs_outer, num_segments=q1)
        return _fit_one(sy, gxy, gyy, sy1, th, stats, p, n, spec)

    return cast(
        Tuple[Float[Array, 'V 3'], Float[Array, 'V p'], Float[Array, 'V']],
        blocked_vmap(per_voxel, (Y, theta_init), block=block),
    )
