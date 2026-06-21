# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Shared GLMM container + constants for the ``glmm`` solver package.

``GLMMResult`` (the per-element fit output) and the two package-wide constants,
factored out (audit O1) so the per-method solver modules (``_pql`` / ``_laplace``
/ ``_agq``) and the ``glmm_fit`` dispatcher all import them from one place
without a cycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from jaxtyping import Array, Float

from .._family import Family
from .._result import register_result

__all__ = ['GLMMResult']

GLMMTier = Literal['few', 'many', 'slope', 'laplace', 'agq']

_EPS = 1e-10

# AGQ node-count cap: the tensor grid is ``n_quad ** r`` nodes, differentiated
# through the mode scan, so it is a compile/memory cliff for a large random-effect
# dimension r.  ``128`` admits every realistic r=2 fit (n_quad up to 11) and r=3
# up to n_quad=5, while blocking the genuine explosions (r>=4, large r=3).
_AGQ_MAX_NODES = 128


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@register_result(
    children=(
        'beta_hat',
        'blups',
        're_var',
        'dispersion',
        'deviance',
        'edf_total',
    ),
    aux=('family', 'n_obs', 'n_groups', 'tier'),
)
@dataclass(frozen=True)
class GLMMResult:
    """Per-element random-intercept GLMM fit output (PQL).

    Attributes
    ----------
    beta_hat
        ``(V, p)`` fixed-effect estimates (over the columns of ``X``).
    blups
        ``(V, q)`` per-level random intercepts (the BLUPs ``b_j``).  For a random
        slope (``z`` with ``r`` columns) this is ``(V, q, r)`` -- one ``r``-vector
        per level.
    re_var
        ``(V, r, r)`` random-effect covariance ``G`` -- **uniform across every
        tier** (D4), matching :class:`LMEResult.cov_re` so downstream ``vmap`` /
        indexing never has to branch on the fit kind.  A scalar random intercept
        is ``(V, 1, 1)`` (``G = sigma_b^2 = phi / lambda``, with ``phi`` the
        dispersion and ``lambda`` the Fellner-Schall precision on the RE block); a
        diagonal random slope (``structure='diagonal'``) carries the per-column
        variance components on the diagonal with zero off-diagonals; an
        unstructured slope (``structure='unstructured'``) is the full covariance.
    dispersion
        ``(V,)`` scale ``phi`` (residual variance for Gaussian / Gamma; ``1`` for
        the fixed-dispersion families -- binomial / Poisson / negative-binomial).
        For the ``'laplace'`` / ``'agq'`` tiers this is a placeholder ``1`` (the
        marginal-likelihood fits do not estimate a working dispersion) -- not for
        model comparison.
    deviance
        ``(V,)`` model deviance at the converged mean.  For ``'laplace'`` / ``'agq'``
        it is ``-2`` x the (approximate) marginal log-likelihood (the ``glmer``
        deviance), directly comparable across those fits.
    edf_total
        ``(V,)`` total effective degrees of freedom (fixed + random).  For the
        ``'laplace'`` / ``'agq'`` tiers this is the **fixed-effect count ``p``
        only** (a placeholder, not the marginal effective df) -- do not form an
        AIC/BIC from it.
    tier
        Which solver ran: ``'few'`` (dense GAMM ``gam_fit``) or ``'many'`` (the
        structured Schur-complement PQL); ``'laplace'`` for the Laplace fit;
        ``'slope'`` for the unstructured random-slope joint-Schur + REML-EM PQL;
        ``'agq'`` for the adaptive Gauss-Hermite random-slope fit.
    """

    beta_hat: Float[Array, 'V p']
    blups: Float[Array, 'V q']
    re_var: Float[Array, 'V r r']
    dispersion: Float[Array, 'V']
    deviance: Float[Array, 'V']
    edf_total: Float[Array, 'V']
    family: Family
    n_obs: int
    n_groups: int
    tier: GLMMTier
