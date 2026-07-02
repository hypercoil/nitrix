# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Crossed random-effects REML for two crossed grouping factors.

Fit two *crossed* scalar random intercepts ``(1 | g1) + (1 | g2)`` -- the
subjects-by-items design, in which each observation carries one ``g1`` level and
one ``g2`` level and the two factors are not nested. The marginal covariance

.. math::

    V = \\sigma_1^2 Z_1 Z_1^{\\top}
        + \\sigma_2^2 Z_2 Z_2^{\\top}
        + \\sigma_e^2 I

is not block-diagonal across either factor, so the per-group rotations that make
a purely nested model cheap do not apply here.

The tractable structure comes from stacking the two random-effect designs as
:math:`Z = [Z_1 \\mid Z_2]` (shape :math:`N \\times (q_1 + q_2)`). The
Woodbury identity gives
:math:`V^{-1} = \\sigma_e^{-2}(I - Z K^{-1} Z^{\\top})` with inner matrix

.. math::

    K = \\sigma_e^2 G^{-1} + Z^{\\top} Z
      = \\begin{bmatrix}
        \\sigma_e^2/\\sigma_1^2\\, I + D_1 & N_{12} \\\\
        N_{12}^{\\top} & \\sigma_e^2/\\sigma_2^2\\, I + D_2
        \\end{bmatrix}

where :math:`D_1 = \\operatorname{diag}(Z_1^{\\top} Z_1)` and
:math:`D_2 = \\operatorname{diag}(Z_2^{\\top} Z_2)` are the per-level counts
(both diagonal) and :math:`N_{12} = Z_1^{\\top} Z_2` is the
:math:`q_1 \\times q_2` cross-tabulation (the incidence between the two factors).
Because the two diagonal blocks are cheap to invert, a Schur complement
eliminates the larger factor in :math:`O(q)` and reduces the problem to a single
dense :math:`\\min(q_1, q_2) \\times \\min(q_1, q_2)` solve:

.. math::

    S = (\\sigma_e^2/\\sigma_2^2\\, I + D_2)
        - N_{12}^{\\top} \\operatorname{diag}(A_1^{-1}) N_{12},
    \\qquad A_1 = \\sigma_e^2/\\sigma_1^2 + D_1

    \\log|K| = \\textstyle\\sum \\log A_1 + \\log|S|

(taking :math:`q_1 \\ge q_2` without loss of generality; the caller orders the
factors so the dense solve falls on the smaller one). Everything else --
:math:`X^{\\top} V^{-1} X`, :math:`X^{\\top} V^{-1} y`,
:math:`y^{\\top} V^{-1} y`, and
:math:`\\log|V| = (N - q_1 - q_2)\\log\\sigma_e^2 + q_1\\log\\sigma_1^2
+ q_2\\log\\sigma_2^2 + \\log|K|` -- assembles from :math:`K^{-1}` applied to
the shared Grams. The dense solves for :math:`S` and for the
:math:`(p, p)` fixed-effect system go through
:func:`~nitrix.linalg._smalllinalg.small_inv_logdet`, avoiding cuSOLVER.

The per-voxel cost is :math:`O(\\min(q_1, q_2)^3)` per Newton step (the dense
Schur solve) -- cheap when one factor is small (for example 30 items by 1000
subjects) and expensive when both are large. The three variance components
:math:`\\theta = [\\log\\sigma_1^2, \\log\\sigma_2^2, \\log\\sigma_e^2]` are fit
by the shared damped, saddle-free Newton optimiser, as in the nested case (a
well-behaved log-variance problem).
"""

from __future__ import annotations

from typing import NamedTuple, Optional, Tuple, cast

import jax.numpy as jnp
from jaxtyping import Array, Float

from ...linalg._smalllinalg import small_inv_logdet
from .._batching import blocked_vmap
from .._optimise import damped_newton
from ._varcomp import VarCompSpec

__all__ = ['CrossedStats', 'crossed_grams', 'fit_crossed_reml']


class CrossedStats(NamedTuple):
    """Shared, response-independent Grams for a crossed design.

    Holds the pieces of the mixed-model system that do not depend on the
    response ``y`` and can therefore be formed once and reused across every
    voxel of a batched fit. The two factors are ordered so that
    :math:`q_1 \\ge q_2`.

    Attributes
    ----------
    d1 : Float[Array, 'q1']
        Per-level observation counts of factor 1 (the larger), i.e. the diagonal
        of :math:`Z_1^{\\top} Z_1`.
    d2 : Float[Array, 'q2']
        Per-level observation counts of factor 2 (the smaller), i.e. the
        diagonal of :math:`Z_2^{\\top} Z_2`.
    n12 : Float[Array, 'q1 q2']
        Cross-tabulation :math:`Z_1^{\\top} Z_2` between the two factors' levels.
    xtz : Float[Array, 'q p']
        The stacked design cross-product :math:`[Z_1 \\mid Z_2]^{\\top} X`, with
        ``q = q1 + q2``.
    xtx : Float[Array, 'p p']
        The fixed-effect Gram :math:`X^{\\top} X`.
    q1 : int
        Number of levels of factor 1 (the larger).
    q2 : int
        Number of levels of factor 2 (the smaller).
    n_obs : int
        Number of observations ``N``.
    """

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
    """Form the shared crossed-design Grams from the two one-hot designs.

    Precomputes the response-independent quantities of the crossed mixed model
    (per-level counts, cross-tabulation, and fixed-effect cross-products) so
    they can be reused across every voxel of a batched fit. The factors must be
    ordered so that :math:`q_1 \\ge q_2`.

    Parameters
    ----------
    X : Float[Array, 'N p']
        Fixed-effect design matrix (``N`` observations, ``p`` covariates).
    oh1 : Float[Array, 'N q1']
        One-hot design of the first (larger) crossed factor, with ``q1`` levels.
    oh2 : Float[Array, 'N q2']
        One-hot design of the second (smaller) crossed factor, with ``q2``
        levels.

    Returns
    -------
    CrossedStats
        The shared Grams: per-level counts ``d1`` / ``d2``, the cross-tabulation
        ``n12``, the stacked design cross-product ``xtz``, the fixed-effect Gram
        ``xtx``, the two level counts ``q1`` / ``q2``, and the observation count
        ``n_obs``.
    """
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

    Evaluates the restricted negative log-likelihood at a single set of
    log-variance parameters, together with the corresponding generalised
    least-squares fixed-effect estimate. Factor 1 (the larger, diagonal block)
    is eliminated by a Schur complement, leaving one dense
    :math:`q_2 \\times q_2` solve; :math:`K^{-1}` is then applied to the stacked
    Grams :math:`[Z_1 \\mid Z_2]^{\\top} [X \\mid y]` to form the fixed-effect
    system.

    Parameters
    ----------
    theta : Float[Array, '3']
        Log-variance parameters
        :math:`[\\log\\sigma_1^2, \\log\\sigma_2^2, \\log\\sigma_e^2]`.
    stats : CrossedStats
        Shared, response-independent Grams for the crossed design.
    zty : Float[Array, 'q']
        The stacked random-effect projection
        :math:`[Z_1 \\mid Z_2]^{\\top} y`, with ``q = q1 + q2``.
    xty : Float[Array, 'p']
        The fixed-effect projection :math:`X^{\\top} y`.
    yty : Float[Array, '']
        The response sum of squares :math:`y^{\\top} y`.
    p : int
        Number of fixed-effect covariates.
    ridge : float
        Small stabiliser added to the diagonal of the Schur complement ``S`` and
        of the fixed-effect system :math:`X^{\\top} V^{-1} X` before inversion,
        guarding near-singular designs.

    Returns
    -------
    nll : Float[Array, '']
        The restricted negative log-likelihood at ``theta``.
    beta : Float[Array, 'p']
        The generalised least-squares fixed-effect estimate at ``theta``.
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
    """Fit the crossed REML model for a single voxel.

    Minimises the profile restricted negative log-likelihood over the three
    log-variance parameters with the shared damped, saddle-free Newton
    optimiser, then re-evaluates at the optimum to recover the fixed-effect
    estimate and the log-likelihood.

    Parameters
    ----------
    zty : Float[Array, 'q']
        The stacked random-effect projection
        :math:`[Z_1 \\mid Z_2]^{\\top} y`, with ``q = q1 + q2``.
    xty : Float[Array, 'p']
        The fixed-effect projection :math:`X^{\\top} y`.
    yty : Float[Array, '']
        The response sum of squares :math:`y^{\\top} y`.
    theta_init : Float[Array, '3']
        Initial log-variance parameters
        :math:`[\\log\\sigma_1^2, \\log\\sigma_2^2, \\log\\sigma_e^2]`.
    stats : CrossedStats
        Shared, response-independent Grams for the crossed design.
    p : int
        Number of fixed-effect covariates.
    spec : VarCompSpec
        Configuration for the Newton iteration and the fixed-effect ridge floor.

    Returns
    -------
    theta : Float[Array, '3']
        The fitted log-variance parameters.
    beta : Float[Array, 'p']
        The fixed-effect estimate at the optimum.
    log_lik : Float[Array, '']
        The restricted log-likelihood at the optimum (the negated final
        negative log-likelihood).
    """

    def nll(theta: Float[Array, '3']) -> Float[Array, '']:
        return _nll_and_beta(theta, stats, zty, xty, yty, p, spec.ridge)[0]

    theta = damped_newton(nll, theta_init, **spec.newton_kwargs)
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
    """Fit crossed random-effects REML over a batch of ``V`` voxels.

    Shares the response-independent Grams across the batch and maps the
    single-voxel crossed REML fit over every voxel, in blocks if requested. The
    two crossed factors must be ordered so that the dense Schur solve falls on
    the smaller one.

    Parameters
    ----------
    Y : Float[Array, 'V N']
        Responses for ``V`` voxels, each over the same ``N`` observations.
    X : Float[Array, 'N p']
        Fixed-effect design matrix shared across voxels.
    oh1 : Float[Array, 'N q1']
        One-hot design of the first (larger) crossed factor, with
        ``q1 = oh1.shape[1]``.
    oh2 : Float[Array, 'N q2']
        One-hot design of the second (smaller) crossed factor, with
        ``q2 = oh2.shape[1]`` and ``q1 >= q2`` (the caller orders the factors so
        the dense Schur solve is on the smaller one).
    theta_init : Float[Array, 'V 3']
        Per-voxel initial log-variance parameters
        :math:`[\\log\\sigma_1^2, \\log\\sigma_2^2, \\log\\sigma_e^2]`.
    spec : VarCompSpec, optional
        Configuration for the Newton iteration and the fixed-effect ridge floor.
    block : int, optional
        Voxel block size for the batched map; ``None`` maps over all voxels at
        once.

    Returns
    -------
    theta_hat : Float[Array, 'V 3']
        Fitted log-variance parameters per voxel.
    beta_hat : Float[Array, 'V p']
        Fixed-effect estimates per voxel.
    log_lik : Float[Array, 'V']
        Restricted log-likelihood per voxel.
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
