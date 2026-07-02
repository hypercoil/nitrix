# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
r"""
Nested random-effects REML for a two-level hierarchy ``(1 | g1/g2)``.

A nested design has an outer grouping factor ``g1`` (:math:`q_1` levels) and an
inner factor ``g2`` **nested within** ``g1`` (each inner level belongs to
exactly one outer level; the :math:`q_2` distinct ``(g1, g2)`` combinations are
the *sublevels*).  The model, per voxel, is

.. math::

    y &= X \beta + Z_1 b_1 + Z_2 b_2 + \epsilon \\
    b_1 &\sim N(0, \sigma_1^2 I_{q_1}),\quad
      b_2 \sim N(0, \sigma_2^2 I_{q_2}),\quad
      \epsilon \sim N(0, \sigma_e^2 I) \\
    V &= \sigma_1^2 Z_1 Z_1^{\top} + \sigma_2^2 Z_2 Z_2^{\top} + \sigma_e^2 I

Because the inner factor is nested, :math:`V` is **block-diagonal across the
outer factor** (different outer blocks share no random effect), and *within* an
outer block it is a two-level telescoping structure: a compound-symmetry block
per inner sublevel (:math:`\sigma_e^2 I + \sigma_2^2 \mathbf{1}\mathbf{1}^{\top}`)
plus a rank-one outer term (:math:`\sigma_1^2 \mathbf{1}\mathbf{1}^{\top}` over
the whole block).  Both levels invert in closed form by Sherman-Morrison -- the
**telescoping Woodbury** -- so :math:`V^{-1}` / :math:`\log|V|` never form an
:math:`N \times N` (or even :math:`n_i \times n_i`) factor:

- inner: :math:`C_{ij}^{-1} = \sigma_e^{-2}[I - c_{ij}\mathbf{1}\mathbf{1}^{\top}]`,
  :math:`c_{ij} = \sigma_2^2 / (\sigma_e^2 + n_{ij}\sigma_2^2)` -- a rank-one
  update per sublevel :math:`j`;
- outer: :math:`V_i^{-1} = A_i^{-1} - d_i (A_i^{-1}\mathbf{1})(A_i^{-1}\mathbf{1})^{\top}`,
  :math:`d_i = \sigma_1^2 / (1 + \sigma_1^2 \mathbf{1}^{\top} A_i^{-1}\mathbf{1})`,
  with :math:`A_i^{-1}` the block-diagonal inner inverse.

Every per-block quantity (:math:`X_i^{\top} V_i^{-1} X_i`,
:math:`X_i^{\top} V_i^{-1} y_i`, :math:`y_i^{\top} V_i^{-1} y_i`,
:math:`\log|V_i|`) is assembled from **per-sublevel sufficient statistics**
(counts and group-sums of ``X`` / ``y``) aggregated to the outer factor by
``segment_sum`` -- the same analytic-objective plus damped autodiff-Newton
recipe as the single-level variance-component solver, over the three variance
components :math:`\theta = [\log\sigma_1^2, \log\sigma_2^2, \log\sigma_e^2]`.
There is no :math:`N \times N` intermediate, and the path is cuSOLVER-free (every
solve is a tiny :func:`small_inv_logdet`).
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
    """Shared (response-independent) sufficient statistics for a nested design."""

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
    r"""Build the nested-design sufficient statistics from the two factors.

    Each distinct ``(outer, inner)`` pair is mapped to a contiguous sublevel
    id, and the per-sublevel and per-outer-level reductions of the shared
    design ``X`` are computed once (they are independent of the response
    ``y``).  These shared statistics feed the per-voxel REML fit.

    Parameters
    ----------
    outer : Int[Array, 'N']
        Outer grouping factor (``g1``): an integer label per observation.  The
        number of outer levels :math:`q_1` is taken as ``outer.max() + 1``.
    inner : Int[Array, 'N']
        Inner grouping factor (``g2``): an integer label per observation.  Each
        distinct ``(outer, inner)`` pair defines a sublevel, so ``inner`` may be
        coded either within each ``outer`` level or globally -- only the pairing
        matters.
    X : Float[Array, 'N p']
        The shared design matrix with ``N`` observations and ``p`` fixed-effect
        columns.

    Returns
    -------
    NestedStats
        The response-independent sufficient statistics: the observation-to-
        sublevel and observation-to-outer index maps, the outer parent of each
        sublevel, per-sublevel observation counts and sums of ``X`` rows,
        per-outer-level observation counts, per-outer-level sums of the outer
        products :math:`x x^{\top}` and of the ``X`` rows, and the level counts
        :math:`q_1` and :math:`q_2`.
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
    r"""Profile REML negative log-likelihood (and ``beta_hat``) at ``theta``.

    Evaluates the restricted (profiled) negative log-likelihood of the nested
    two-level model at the log-variance vector ``theta``, together with the
    generalised-least-squares fixed-effect estimate that it profiles out.  All
    per-block quantities are assembled from the shared sufficient statistics
    and the response reductions by the telescoping Sherman-Morrison recipe, so
    no :math:`N \times N` matrix is ever formed.

    Parameters
    ----------
    theta : Float[Array, '3']
        Log-variance components
        :math:`[\log\sigma_1^2, \log\sigma_2^2, \log\sigma_e^2]` for the outer,
        inner, and residual variances.
    stats : NestedStats
        Response-independent sufficient statistics from :func:`nested_layout`.
    sy : Float[Array, 'q2']
        Per-sublevel sum of the response ``y``.
    gxy : Float[Array, 'q1 p']
        Per-outer-level sum of ``X`` rows weighted by ``y``.
    gyy : Float[Array, 'q1']
        Per-outer-level sum of :math:`y^2`.
    sy1 : Float[Array, 'q1']
        Per-outer-level sum of the response ``y``.
    p : int
        Number of fixed-effect columns (compile-time shape).
    n : int
        Number of observations ``N`` (compile-time shape).
    ridge : float
        Tikhonov regulariser added to the diagonal of
        :math:`X^{\top} V^{-1} X` before it is inverted.

    Returns
    -------
    nll : Float[Array, '']
        The profile REML negative log-likelihood at ``theta``.
    beta : Float[Array, 'p']
        The generalised-least-squares fixed-effect estimate
        :math:`\hat\beta` at ``theta``.
    """
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
    """Fit the nested REML model for a single voxel.

    Minimises the profile REML negative log-likelihood over the three log-
    variance components using the shared damped (saddle-free) Newton optimiser,
    then re-evaluates at the optimum to recover the fixed-effect estimate and
    the log-likelihood.

    Parameters
    ----------
    sy : Float[Array, 'q2']
        Per-sublevel sum of the response ``y``.
    gxy : Float[Array, 'q1 p']
        Per-outer-level sum of ``X`` rows weighted by ``y``.
    gyy : Float[Array, 'q1']
        Per-outer-level sum of :math:`y^2`.
    sy1 : Float[Array, 'q1']
        Per-outer-level sum of the response ``y``.
    theta_init : Float[Array, '3']
        Initial log-variance components for the Newton iteration.
    stats : NestedStats
        Response-independent sufficient statistics from :func:`nested_layout`.
    p : int
        Number of fixed-effect columns (compile-time shape).
    n : int
        Number of observations ``N`` (compile-time shape).
    spec : VarCompSpec
        Configuration for the variance-component solve: the ridge and the
        Newton keyword arguments.

    Returns
    -------
    theta : Float[Array, '3']
        The fitted log-variance components at the REML optimum.
    beta : Float[Array, 'p']
        The fixed-effect estimate at the optimum.
    log_lik : Float[Array, '']
        The maximised restricted log-likelihood (the negated final objective).
    """

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
    r"""Fit the nested random-effects REML model over a batch of voxels.

    Applies the two-level nested variance-component fit to each of ``V`` voxels
    that share a common design and grouping structure.  The shared sufficient
    statistics are built once from the design; per voxel the response
    reductions are formed and the profile REML objective is minimised over the
    three log-variance components.  The batch is mapped in configurable blocks
    to bound peak memory.

    Parameters
    ----------
    Y : Float[Array, 'V N']
        The response for each of ``V`` voxels, with ``N`` observations per
        voxel.
    X : Float[Array, 'N p']
        The shared fixed-effect design matrix, common to all voxels.
    outer : Int[Array, 'N']
        Outer grouping factor (``g1``): an integer label per observation.
    inner : Int[Array, 'N']
        Inner grouping factor (``g2``), nested within ``outer``: an integer
        label per observation.
    theta_init : Float[Array, 'V 3']
        Initial log-variance components per voxel.
    spec : VarCompSpec, optional
        Configuration for the variance-component solve (ridge and Newton
        settings).
    block : int, optional
        Block size for the batched map over voxels; ``None`` maps the whole
        batch at once.

    Returns
    -------
    theta_hat : Float[Array, 'V 3']
        Fitted log-variance components per voxel,
        :math:`[\log\sigma_1^2, \log\sigma_2^2, \log\sigma_e^2]` (outer, inner,
        residual variances).
    beta_hat : Float[Array, 'V p']
        Fitted fixed-effect estimates per voxel.
    log_lik : Float[Array, 'V']
        Maximised restricted log-likelihood per voxel.
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
