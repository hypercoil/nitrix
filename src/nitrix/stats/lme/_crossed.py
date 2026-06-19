# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Crossed random-effects REML for two crossed grouping factors (tier R4).

Two **crossed** scalar random intercepts ``(1 | g1) + (1 | g2)`` -- the
subjects-x-items design (each observation has a ``g1`` and a ``g2`` level, and
the factors are not nested).  The covariance::

    V = sigma1^2 Z1 Z1^T + sigma2^2 Z2 Z2^T + sigma_e^2 I

is **not** block-diagonal across either factor (the FaST-LMM rotation and the
block-Woodbury per-group trick both fail), so this is the sparse mixed-model-
equations regime the v3 ledger gates as Tier-2.

The tractable structure.  Stack ``Z = [Z1 | Z2]`` (``N x (q1+q2)``); Woodbury
gives ``V^{-1} = sigma_e^{-2}(I - Z K^{-1} Z^T)`` with the inner matrix

    K = sigma_e^2 G^{-1} + Z^T Z
      = [[ sigma_e^2/sigma1^2 I + D1 ,      N12      ],
         [        N12^T          , sigma_e^2/sigma2^2 I + D2 ]]

where ``D1 = diag(Z1^T Z1)`` / ``D2 = diag(Z2^T Z2)`` are the per-level **counts**
(both diagonal) and ``N12 = Z1^T Z2`` is the ``q1 x q2`` cross-tabulation
(incidence).  The two diagonal blocks let a **Schur complement** eliminate the
*larger* factor in ``O(q)`` and reduce to a single dense
``min(q1, q2) x min(q1, q2)`` solve::

    S = (sigma_e^2/sigma2^2 I + D2) - N12^T diag(A1^{-1}) N12,   A1 = sigma_e^2/sigma1^2 + D1
    log|K| = sum log A1 + log|S|

(taking ``q1 >= q2`` w.l.o.g.; the caller orders the factors so the dense solve
is on the smaller one).  Everything else -- ``X^T V^{-1} X``, ``X^T V^{-1} y``,
``y^T V^{-1} y``, ``log|V| = (N - q1 - q2) log sigma_e^2 + q1 log sigma1^2 +
q2 log sigma2^2 + log|K|`` -- assembles from ``K^{-1}`` applied to the shared
Grams.  cuSOLVER-free (the ``S`` and ``(p, p)`` solves go through
``small_inv_logdet``).

**Cost / HLO gate.**  The per-voxel cost is ``O(min(q1, q2)^3)`` per Newton step
(the dense Schur solve) -- cheap when one factor is small (e.g. 30 items x 1000
subjects) and expensive when both are large.  This is the tier-R4 trade the
ledger flags; ``lme_fit`` only routes here for an explicit crossed spec.  The
three variance components ``theta = [log sigma1^2, log sigma2^2, log sigma_e^2]``
are fit by the shared damped/saddle-free Newton (``_optimise``), like the nested
R3 (a well-behaved log-variance problem).
"""

from __future__ import annotations

from typing import NamedTuple, Optional, Tuple, cast

import jax.numpy as jnp
from jaxtyping import Array, Float

from ...linalg._smalllinalg import small_inv_logdet
from .._batching import blocked_vmap
from ._optimise import damped_newton
from ._varcomp import VarCompSpec

__all__ = ['CrossedStats', 'crossed_grams', 'fit_crossed_reml']


class CrossedStats(NamedTuple):
    """Shared (``y``-independent) Grams for a crossed design (``q1 >= q2``)."""

    d1: Float[Array, 'q1']  # per-level counts of factor 1 (the larger)
    d2: Float[Array, 'q2']  # per-level counts of factor 2 (the smaller)
    n12: Float[Array, 'q1 q2']  # cross-tabulation Z1^T Z2
    xtz: Float[Array, 'q p']  # [Z1 | Z2]^T X, q = q1 + q2
    xtx: Float[Array, 'p p']  # X^T X
    q1: int
    q2: int
    n_obs: int


def crossed_grams(
    X: Float[Array, 'N p'],
    oh1: Float[Array, 'N q1'],
    oh2: Float[Array, 'N q2'],
) -> CrossedStats:
    """Shared crossed-design Grams from the two one-hot designs (``q1 >= q2``)."""
    d1 = jnp.sum(oh1, axis=0)
    d2 = jnp.sum(oh2, axis=0)
    n12 = oh1.T @ oh2
    xtz = jnp.concatenate([oh1.T @ X, oh2.T @ X], axis=0)  # (q1+q2, p)
    xtx = X.T @ X
    return CrossedStats(
        d1=d1,
        d2=d2,
        n12=n12,
        xtz=xtz,
        xtx=xtx,
        q1=oh1.shape[-1],
        q2=oh2.shape[-1],
        n_obs=X.shape[0],
    )


def _nll_and_beta(
    theta: Float[Array, '3'],
    stats: CrossedStats,
    zty: Float[Array, 'q'],
    xty: Float[Array, 'p'],
    yty: Float[Array, ''],
    p: int,
    ridge: float,
) -> Tuple[Float[Array, ''], Float[Array, 'p']]:
    """Profile REML negative log-likelihood (and ``beta``) for the crossed model.

    Eliminates factor 1 (the larger, diagonal block) by Schur complement, leaving
    one dense ``q2 x q2`` solve.  ``K^{-1}`` is applied to the stacked Grams
    ``[Z1 | Z2]^T [X | y]`` to form the fixed-effect system.
    """
    q1, q2 = stats.q1, stats.q2
    s1 = jnp.exp(theta[0])
    s2 = jnp.exp(theta[1])
    se = jnp.exp(theta[2])

    a1 = se / s1 + stats.d1  # (q1,) diagonal of K block 1
    a1_inv = 1.0 / a1
    n12 = stats.n12
    # Schur complement onto factor 2: S = diag(se/s2 + D2) - N12^T A1^{-1} N12.
    s_mat = jnp.diag(se / s2 + stats.d2) - n12.T @ (a1_inv[:, None] * n12)
    s_mat = s_mat + ridge * jnp.eye(q2, dtype=s_mat.dtype)
    s_inv, logdet_s = small_inv_logdet(s_mat, q2)
    logdet_k = jnp.sum(jnp.log(a1)) + logdet_s

    # M = [Z1 | Z2]^T [X | y]  (q1+q2, p+1); apply K^{-1} via the Schur solve.
    rhs = jnp.concatenate([stats.xtz, zty[:, None]], axis=1)  # (q1+q2, p+1)
    v1 = rhs[:q1]
    v2 = rhs[q1:]
    u2 = s_inv @ (v2 - n12.T @ (a1_inv[:, None] * v1))  # (q2, p+1)
    u1 = a1_inv[:, None] * (v1 - n12 @ u2)  # (q1, p+1)
    # rhs^T K^{-1} rhs = v1^T u1 + v2^T u2  (p+1, p+1)
    quad = v1.T @ u1 + v2.T @ u2

    zt_x = stats.xtz  # for clarity
    xt_vinv_x = (stats.xtx - quad[:p, :p]) / se + ridge * jnp.eye(
        p, dtype=zt_x.dtype
    )
    xt_vinv_y = (xty - quad[:p, p]) / se
    yt_vinv_y = (yty - quad[p, p]) / se

    a_inv, logdet_a = small_inv_logdet(xt_vinv_x, p)
    beta = a_inv @ xt_vinv_y
    rss = yt_vinv_y - beta @ xt_vinv_y
    logdet_v = (
        (stats.n_obs - q1 - q2) * theta[2]
        + q1 * theta[0]
        + q2 * theta[1]
        + logdet_k
    )
    nll = 0.5 * (logdet_v + logdet_a + rss)
    return nll, beta


def _fit_one(
    zty: Float[Array, 'q'],
    xty: Float[Array, 'p'],
    yty: Float[Array, ''],
    theta_init: Float[Array, '3'],
    stats: CrossedStats,
    p: int,
    spec: VarCompSpec,
) -> Tuple[Float[Array, '3'], Float[Array, 'p'], Float[Array, '']]:
    """Single-voxel crossed REML fit via the shared saddle-free Newton."""

    def nll(theta: Float[Array, '3']) -> Float[Array, '']:
        return _nll_and_beta(theta, stats, zty, xty, yty, p, spec.ridge)[0]

    theta = damped_newton(nll, theta_init, spec=spec)
    final_nll, beta = _nll_and_beta(theta, stats, zty, xty, yty, p, spec.ridge)
    return theta, beta, -final_nll


def fit_crossed_reml(
    Y: Float[Array, 'V N'],
    X: Float[Array, 'N p'],
    oh1: Float[Array, 'N q1'],
    oh2: Float[Array, 'N q2'],
    theta_init: Float[Array, 'V 3'],
    *,
    spec: VarCompSpec = VarCompSpec(),
    block: Optional[int] = None,
) -> Tuple[Float[Array, 'V 3'], Float[Array, 'V p'], Float[Array, 'V']]:
    """Batched crossed-RE REML over ``V`` voxels.

    ``oh1`` / ``oh2`` are the one-hot designs of the two crossed factors with
    ``q1 = oh1.shape[1] >= q2 = oh2.shape[1]`` (the caller orders them so the
    dense Schur solve is on the smaller factor).  ``theta = [log sigma1^2,
    log sigma2^2, log sigma_e^2]``.  Returns ``(theta_hat, beta_hat, log_lik)``.
    """
    p = X.shape[-1]
    stats = crossed_grams(X, oh1, oh2)
    z_design = jnp.concatenate([oh1, oh2], axis=1)  # (N, q1+q2)

    def per_voxel(
        y: Float[Array, 'N'], th: Float[Array, '3']
    ) -> Tuple[Float[Array, '3'], Float[Array, 'p'], Float[Array, '']]:
        zty = z_design.T @ y  # (q1+q2,)
        xty = X.T @ y
        yty = y @ y
        return _fit_one(zty, xty, yty, th, stats, p, spec)

    return cast(
        Tuple[Float[Array, 'V 3'], Float[Array, 'V p'], Float[Array, 'V']],
        blocked_vmap(per_voxel, (Y, theta_init), block=block),
    )
