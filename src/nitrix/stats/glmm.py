# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Mass-univariate generalised **linear mixed** models (GLMM) via PQL.

``glmm_fit`` fits, per element (voxel / vertex / fixel), a random-intercept
GLMM under a non-Gaussian :class:`~nitrix.stats._family.Family`::

    g(E[y | b]) = X beta + b[group],   b_j ~ N(0, sigma_b^2),   j = 1 .. q

by **penalised quasi-likelihood** (PQL; Breslow & Clayton 1993): an outer IRLS
loop forms the working response ``z`` and weights ``W`` from the current fit,
then a single variance-component (Fellner-Schall) step updates the random-effect
precision ``lambda = phi / sigma_b^2`` -- exactly the v1 "penalised GLM ==
variance-components REML" identity, now under a GLM family.  This is the §1.2
estimator: ``mgcv::gam(family=, s(g, bs="re"))`` / ``glmmPQL`` (with the
documented PQL attenuation for binary / low-count responses; Laplace is the
Tier-2 follow-up).

Performance-preserving dispatch (v3 §0.1 / §2 routing)
----------------------------------------------------

A random effect widens the per-voxel system by ``q`` = #levels, so the dense
penalised solve is ``O((p + q)^3)`` per voxel.  That is **cheap for few-level
factors** (site / scanner / batch, ``q ~ 10-50``) but a latency cliff for
**many-level** factors (random intercept per *subject*, ``q ~ 100-1000``).
``glmm_fit`` therefore dispatches on the level count:

- **few-level** (``q <= few_level_max``): the dense GAMM path -- ``gam_fit`` with
  a :func:`~nitrix.stats.basis.re_smooth` block and the GLM family.  This *is*
  the optimal solver in this regime (no cheaper exact route), and reuses the
  shipped GAM machinery verbatim.
- **many-level** (``q > few_level_max``): a **structured** PQL that never forms
  the ``(p + q) x (p + q)`` system.  Because a single grouping factor makes the
  random-effect block of the normal equations **diagonal across groups**, the
  Schur complement onto the ``p``-dimensional fixed block costs ``O(N p^2 + q)``
  per voxel (the §1.1 block-Woodbury structure, weighted by the IRLS ``W`` and
  wrapped in the PQL loop) -- linear in the level count, cuSOLVER-free.

The two paths run the *same* PQL iteration (same working response, same
Fellner-Schall update, same iteration budget), so they agree to the iterative
tolerance; the dispatch only changes the linear-algebra cost, never the answer.

Scope.  Beyond the **scalar random intercept** ``(1 | g)`` (the FR §1.2 headline:
"binary outcomes / lesion counts per subject with random intercepts"), ``glmm_fit``
also fits non-Gaussian random **slopes** via the ``z`` / ``structure`` arguments:
diagonal ``(x || g)`` as independent ``re_smooth`` blocks through ``gam_fit``, and
correlated ``(1 + x | g)`` via the block-Woodbury REML wrapped in the PQL loop (the
penalised-IRLS = iteratively-reweighted-REML identity).  The Gaussian-family slope
fit is the same REML estimator as ``lme_fit`` (R2 block-Woodbury), to the
iterative (optimiser) tolerance.  Random slopes are also
served under the **Laplace** marginal likelihood (``method='laplace'``, the
``r``-dimensional conditional-mode integral + ``r x r`` determinant correction),
which corrects the PQL attenuation for binary / low-count slopes.

References
----------
- Breslow, N. E. & Clayton, D. G. (1993). Approximate inference in generalized
  linear mixed models.  JASA 88, 9-25.
- Wood, S. N. & Fasiolo, M. (2017). A generalized Fellner-Schall method for
  smoothing parameter optimization.  Biometrics 73, 1071-1081.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.scipy.special import logsumexp
from jaxtyping import Array, Float, Int

from ..linalg._smalllinalg import small_inv_logdet, spd_chol
from ._batching import blocked_vmap
from ._family import GAUSSIAN, Family, resolve_family
from ._irls import safe_dmu
from ._result import register_result
from .basis import re_smooth
from .gam import gam_fit
from .lme._optimise import damped_newton
from .lme._varcomp import VarCompSpec

__all__ = ['GLMMResult', 'glmm_fit']

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
    tier: str


# ---------------------------------------------------------------------------
# Few-level path: dense GAMM PQL via gam_fit + re_smooth
# ---------------------------------------------------------------------------


def _glmm_few_level(
    Y: Float[Array, 'V N'],
    X: Float[Array, 'N p'],
    group: Int[Array, 'N'],
    n_groups: int,
    family: Family,
    n_outer: int,
    n_inner: int,
    ridge: float,
    lam_floor: float,
    lam_ceil: float,
    block: Optional[int],
) -> GLMMResult:
    """Dense GAMM PQL: a ``re_smooth`` block + the GLM family through ``gam_fit``.

    The fixed design ``X`` enters as the unpenalised ``parametric`` block (no
    extra intercept -- ``X`` carries its own); the random intercept is the one
    penalised ``re_smooth`` block, whose Fellner-Schall smoothing parameter *is*
    the RE precision ``1 / sigma_b^2``.  This is the optimal solver for a
    few-level factor and reuses the shipped GAM machinery verbatim.
    """
    p = X.shape[-1]
    re = re_smooth(group, n_levels=n_groups)
    res = gam_fit(
        Y,
        [re],
        parametric=X,
        intercept=False,
        family=family,
        n_outer=n_outer,
        n_inner=n_inner,
        ridge=ridge,
        lam_floor=lam_floor,
        lam_ceil=lam_ceil,
        block=block,
    )
    beta_hat = res.coef[:, :p]  # (V, p)
    blups = res.coef[:, p : p + n_groups]  # (V, q)
    # sigma_b^2 = phi / lambda (the FS precision on the identity RE penalty).
    re_var = res.dispersion / jnp.clip(res.lam[:, 0], _EPS, None)  # (V,)
    return GLMMResult(
        beta_hat=beta_hat,
        blups=blups,
        re_var=re_var[:, None, None],  # (V, 1, 1) -- uniform G shape (D4)
        dispersion=res.dispersion,
        deviance=res.deviance,
        edf_total=res.edf_total,
        family=family,
        n_obs=int(X.shape[0]),
        n_groups=n_groups,
        tier='few',
    )


def _glmm_slope_diagonal(
    Y: Float[Array, 'V N'],
    X: Float[Array, 'N p'],
    group: Int[Array, 'N'],
    n_groups: int,
    z: Float[Array, 'N r'],
    family: Family,
    n_outer: int,
    n_inner: int,
    ridge: float,
    lam_floor: float,
    lam_ceil: float,
    block: Optional[int],
) -> GLMMResult:
    """Diagonal random-slope GLMM via independent ``re_smooth`` blocks (PQL).

    For a random-effect design ``z`` with ``r`` columns, build ``r`` *independent*
    random-effect blocks -- one per column, ``re_smooth(group, by=z[:, k])`` -- and
    fit them through ``gam_fit`` under the GLM family.  Each block carries its own
    Fellner-Schall smoothing parameter, so the random-effect covariance ``G`` is
    **diagonal**: the uncorrelated ``(z_0 || g) + ... + (z_{r-1} || g)`` model
    (lme4's ``||``).  The per-column variance components are ``sigma_{b,k}^2 = phi /
    lambda_k``; ``re_var`` is ``(V, r)`` and the BLUPs are ``(V, q, r)``.

    A ones column of ``z`` is the random *intercept* (``by = 1`` -> ``one_hot(g)``);
    the remaining columns are random *slopes*.  This reuses the dense GAMM machinery
    verbatim; for a many-level factor it is the structured block-Woodbury PQL that
    avoids the ``(p + q r)``-wide solve (the unstructured / correlated path).
    """
    p = X.shape[-1]
    r = z.shape[-1]
    q = n_groups
    blocks = [re_smooth(group, by=z[:, k], n_levels=q) for k in range(r)]
    res = gam_fit(
        Y,
        blocks,
        parametric=X,
        intercept=False,
        family=family,
        n_outer=n_outer,
        n_inner=n_inner,
        ridge=ridge,
        lam_floor=lam_floor,
        lam_ceil=lam_ceil,
        block=block,
    )
    beta_hat = res.coef[:, :p]  # (V, p)
    # coef tail is [block_0 (q) | ... | block_{r-1} (q)] -> (V, q, r).
    blups = jnp.swapaxes(
        res.coef[:, p : p + r * q].reshape(-1, r, q), 1, 2
    )  # (V, q, r)
    re_var_diag = res.dispersion[:, None] / jnp.clip(
        res.lam[:, :r], _EPS, None
    )  # (V, r) per-column variance components
    # D4: embed on the diagonal of a uniform (V, r, r) G (zero off-diagonals).
    re_var = re_var_diag[..., None] * jnp.eye(r, dtype=re_var_diag.dtype)
    return GLMMResult(
        beta_hat=beta_hat,
        blups=blups,
        re_var=re_var,  # (V, r, r) diagonal
        dispersion=res.dispersion,
        deviance=res.deviance,
        edf_total=res.edf_total,
        family=family,
        n_obs=int(X.shape[0]),
        n_groups=q,
        tier='few',
    )


# ---------------------------------------------------------------------------
# Unstructured random-slope path: block-Woodbury REML wrapped in the PQL loop
# ---------------------------------------------------------------------------


def _slope_solve(
    y: Float[Array, 'N'],
    X: Float[Array, 'N p'],
    z: Float[Array, 'N r'],
    group: Int[Array, 'N'],
    n_groups: int,
    family: Family,
    r: int,
    g_inv: Float[Array, 'r r'],
    se2: Float[Array, ''],
    beta: Float[Array, 'p'],
    b: Float[Array, 'q r'],
    ridge_eye: Float[Array, 'p p'],
) -> Tuple[
    Float[Array, 'p'],
    Float[Array, 'q r'],
    Float[Array, 'N'],
    Float[Array, 'N'],
    Float[Array, 'q r r'],
    Float[Array, 'q p r'],
    Float[Array, 'q r r'],
    Float[Array, 'p p'],
]:
    """One penalised-IRLS step for an ``r``-dimensional random slope, Schur-structured.

    The ``r x r`` lift of :func:`_structured_solve`: at the current ``(beta, b)`` and
    fixed ``(G, sigma_e^2)``, form the working response and weights, then solve the
    penalised normal equations with per-group blocks ``D_g = Z_g^T W Z_g +
    sigma_e^2 G^{-1}`` (``r x r``) and ``B_g = X_g^T W Z_g`` (``p x r``).  Eliminating
    ``b`` gives the ``p x p`` Schur system ``S beta = rhs``, ``S = X^T W X - sum_g
    B_g D_g^{-1} B_g^T`` -- per-group ``r x r`` inverses (``small_inv_logdet``), never
    the ``(p + q r)``-wide system.  Returns ``(beta, b)`` and the pieces the REML-EM
    update needs: ``w``, ``z_work``, and the per-group ``Z^T W Z`` / ``X^T W Z`` /
    ``D^{-1}`` plus ``S^{-1}``.
    """
    eta = family.clip_eta(X @ beta + jnp.einsum('nr,nr->n', z, b[group]))
    mu = family.linkinv(eta)
    dmu = family.mu_eta(eta)
    var = jnp.clip(family.variance(mu), _EPS, None)
    w = dmu * dmu / var
    z_work = eta + (y - mu) / safe_dmu(dmu)

    wx = w[:, None] * X
    wzwork = w * z_work
    ztz = jax.ops.segment_sum(
        (w[:, None] * z)[:, :, None] * z[:, None, :], group, num_segments=n_groups
    )  # (q, r, r) = Z_g^T W Z_g
    xtz = jax.ops.segment_sum(
        wx[:, :, None] * z[:, None, :], group, num_segments=n_groups
    )  # (q, p, r) = X_g^T W Z_g
    zty = jax.ops.segment_sum(
        wzwork[:, None] * z, group, num_segments=n_groups
    )  # (q, r) = Z_g^T W z_work
    xtwx = wx.T @ X  # (p, p)
    xtwz = X.T @ wzwork  # (p,)

    dinv = jax.vmap(lambda a: small_inv_logdet(a, r)[0])(
        ztz + se2 * g_inv[None]
    )  # (q, r, r) = D_g^{-1}
    bdinvbt = jnp.einsum('gpr,grs,gqs->pq', xtz, dinv, xtz)  # (p, p)
    s_inv, _ = small_inv_logdet(xtwx - bdinvbt + ridge_eye, X.shape[-1])
    rhs_beta = xtwz - jnp.einsum('gpr,grs,gs->p', xtz, dinv, zty)
    beta_new = s_inv @ rhs_beta
    zr = zty - jnp.einsum('gpr,p->gr', xtz, beta_new)  # Z_g^T W (z_work - X beta)
    b_new = jnp.einsum('grs,gs->gr', dinv, zr)  # (q, r)
    return beta_new, b_new, w, z_work, ztz, xtz, dinv, s_inv


def _glmm_slope_structured_one(
    y: Float[Array, 'N'],
    X: Float[Array, 'N p'],
    z: Float[Array, 'N r'],
    group: Int[Array, 'N'],
    n_groups: int,
    family: Family,
    p: int,
    r: int,
    n_outer: int,
    n_inner: int,
    ridge: float,
) -> Tuple[
    Float[Array, 'p'],
    Float[Array, 'q r'],
    Float[Array, 'r r'],
    Float[Array, ''],
    Float[Array, ''],
    Float[Array, ''],
]:
    """Single-voxel unstructured (correlated-``G``) random-slope GLMM via PQL.

    A **joint-Schur inner IRLS + monotone REML-EM outer** (FR
    ``glmm-random-slope-robust-solver``).  Each outer step (a) runs ``n_inner``
    penalised-IRLS steps for ``(beta, b)`` at fixed ``(G, sigma_e^2)`` through the
    ``r x r`` Schur solve (:func:`_slope_solve`), then (b) updates the
    random-effect covariance by EM::

        G <- (1/q) sum_g [ b_g b_g^T + Cov(b_g) ],
        Cov(b_g) = sigma_e^2 (D_g^{-1} + D_g^{-1} B_g^T S^{-1} B_g D_g^{-1}),

    the posterior second moment of the random effects (the ``B_g^T S^{-1} B_g``
    term is the beta-uncertainty correction that makes this **REML**-EM, not ML-EM
    -- it is what reproduces ``lme_fit``'s REML for the Gaussian family).  Because
    the M-step is a sum of PSD matrices it stays positive-definite, and because EM
    is monotone in the REML objective it cannot over-shoot ``G`` the way the
    earlier iterated-Newton-REML did -- so the fit no longer depends on the IRLS
    ``eta`` clamp landing it in the right basin (the clamp reverts to pure overflow
    safety).  ``sigma_e^2`` is pinned at ``1`` for a fixed-dispersion family
    (binomial / Poisson / NB) and EM-updated (``(sum_i w_i r_i^2 + sum_g tr(Z_g^T W
    Z_g Cov(b_g))) / N``) otherwise.

    Returns ``(beta, blups (q, r), G (r, r), dispersion, deviance, edf)``.
    """
    from ._irls import irls_warm_start

    ridge_eye = ridge * jnp.eye(p, dtype=y.dtype)
    ridge_r = ridge * jnp.eye(r, dtype=y.dtype)
    n = X.shape[0]
    fixed_disp = family.has_fixed_dispersion

    # Link-scale init (see _glmm_slope_diagonal); a *small* G so the first inner
    # IRLS shrinks the BLUPs hard -- EM then grows G monotonically, no over-shoot.
    var_e = jnp.var(family.link(family.init_mu(y)))
    g_init = jnp.maximum(0.1 * var_e, 1e-4) * jnp.eye(r, dtype=y.dtype)
    se2_init = (
        jnp.asarray(1.0, dtype=y.dtype)
        if fixed_disp
        else jnp.maximum(0.5 * var_e, 1e-4)
    )
    beta_init = irls_warm_start(
        y, X, family, penalty=jnp.zeros((p, p), dtype=y.dtype), ridge=ridge
    )
    b_init = jnp.zeros((n_groups, r), dtype=y.dtype)

    def run_inner(
        g_inv: Array, se2: Array, beta: Array, b: Array
    ) -> Tuple[Array, Array, Array, Array, Array, Array, Array, Array]:
        def step(c: Tuple[Array, Array], _: Array) -> Tuple[Tuple[Array, Array], None]:
            be, bb = c
            be2, bb2, *_ = _slope_solve(
                y, X, z, group, n_groups, family, r, g_inv, se2, be, bb, ridge_eye
            )
            return (be2, bb2), None

        (beta, b), _ = lax.scan(step, (beta, b), xs=None, length=n_inner)
        return _slope_solve(
            y, X, z, group, n_groups, family, r, g_inv, se2, beta, b, ridge_eye
        )

    def outer(
        carry: Tuple[Array, Array, Array, Array], _: Array
    ) -> Tuple[Tuple[Array, Array, Array, Array], None]:
        g_cov, se2, beta, b = carry
        g_inv, _ = small_inv_logdet(g_cov, r)
        pieces = run_inner(g_inv, se2, beta, b)
        beta, b = pieces[0], pieces[1]
        g_new, se2_new, _ = _em_step(*pieces, se2)
        return (g_new, se2_new, beta, b), None

    def _em_step(
        beta: Array, b: Array, w: Array, z_work: Array, ztz: Array,
        xtz: Array, dinv: Array, s_inv: Array, se2: Array,
    ) -> Tuple[Array, Array, Array]:
        # Posterior covariance V_bb = D^{-1} + D^{-1} B^T S^{-1} B D^{-1} (the
        # beta-uncertainty term makes G REML, not ML); scaled by se2 it is Cov(b_g).
        c_mat = jnp.einsum('grs,gps->grp', dinv, xtz)  # D^{-1} B^T  (q, r, p)
        v_bb = dinv + jnp.einsum('grp,pq,gsq->grs', c_mat, s_inv, c_mat)
        cov_b = se2 * v_bb
        # M-step: G = mean over groups of (b b^T + Cov(b)) -- PSD by construction.
        g_new = (
            jnp.einsum('gr,gs->rs', b, b) + jnp.sum(cov_b, axis=0)
        ) / n_groups + ridge_r
        # Total effective df = tr(S^{-1} X^T W X) - 2 tr(S^{-1} B D^{-1} B^T)
        #                      + sum_g tr(V_bb_g Z_g^T W Z_g)  (= tr(V_full Gram)).
        bdinvbt = jnp.einsum('gpr,grs,gqs->pq', xtz, dinv, xtz)
        xtwx = (w[:, None] * X).T @ X
        edf = (
            jnp.trace(s_inv @ xtwx)
            - 2.0 * jnp.trace(s_inv @ bdinvbt)
            + jnp.sum(jnp.einsum('grs,gsr->g', v_bb, ztz))
        )
        # sigma_e^2: pinned at 1 for a fixed-dispersion family; else the REML
        # estimate rss / (N - edf) (the LMM generalisation of RSS/(N-p)).
        if fixed_disp:
            se2_new = jnp.asarray(1.0, dtype=y.dtype)
        else:
            resid = z_work - X @ beta - jnp.einsum('nr,nr->n', z, b[group])
            rss = jnp.sum(w * resid * resid)
            se2_new = rss / jnp.clip(n - edf, 1e-3, None)
        return g_new, se2_new, edf

    (g_cov, se2, beta, b), _ = lax.scan(
        outer, (g_init, se2_init, beta_init, b_init), xs=None, length=n_outer
    )

    # Final statistics at the converged (G, sigma_e^2).
    g_inv, _ = small_inv_logdet(g_cov, r)
    pieces = run_inner(g_inv, se2, beta, b)
    beta, b = pieces[0], pieces[1]
    _, _, edf = _em_step(*pieces, se2)
    eta = family.clip_eta(X @ beta + jnp.einsum('nr,nr->n', z, b[group]))
    deviance = jnp.sum(family.unit_deviance(y, family.linkinv(eta)))
    return beta, b, g_cov, se2, deviance, edf


def _glmm_slope_structured(
    Y: Float[Array, 'V N'],
    X: Float[Array, 'N p'],
    group: Int[Array, 'N'],
    n_groups: int,
    z: Float[Array, 'N r'],
    family: Family,
    n_outer: int,
    n_inner: int,
    ridge: float,
    block: Optional[int],
) -> GLMMResult:
    """Unstructured random-slope GLMM over ``V`` elements (joint-Schur + REML-EM)."""
    p = X.shape[-1]
    r = z.shape[-1]

    def per_voxel(
        y: Float[Array, 'N'],
    ) -> Tuple[Array, Array, Array, Array, Array, Array]:
        return _glmm_slope_structured_one(
            y, X, z, group, n_groups, family, p, r, n_outer, n_inner, ridge,
        )

    beta, blups, cov_re, se2, deviance, edf = blocked_vmap(
        per_voxel, (Y,), block=block
    )
    return GLMMResult(
        beta_hat=beta,
        blups=blups,
        re_var=cov_re,
        dispersion=se2,
        deviance=deviance,
        edf_total=edf,
        family=family,
        n_obs=int(X.shape[0]),
        n_groups=n_groups,
        tier='slope',
    )


# ---------------------------------------------------------------------------
# Many-level path: structured (Schur-complement) PQL, O(N p^2 + q) per voxel
# ---------------------------------------------------------------------------


def _structured_solve(
    y: Float[Array, 'N'],
    X: Float[Array, 'N p'],
    group: Int[Array, 'N'],
    n_groups: int,
    family: Family,
    lam: Float[Array, ''],
    beta: Float[Array, 'p'],
    b: Float[Array, 'q'],
    ridge_eye: Float[Array, 'p p'],
) -> Tuple[
    Float[Array, 'p'],
    Float[Array, 'q'],
    Float[Array, 'N'],
    Float[Array, 'q'],
    Float[Array, 'q p'],
    Float[Array, 'q'],
    Float[Array, 'p p'],
    Float[Array, 'p p'],
]:
    """One penalised-IRLS step for a scalar random intercept, Schur-structured.

    The penalised normal equations at smoothing parameter ``lambda`` are::

        [ X^T W X        X^T W Z   ] [beta]   [X^T W z]
        [ Z^T W X   Z^T W Z + lI_q ] [ b  ] = [Z^T W z]

    with ``Z = onehot(group)``, so ``Z^T W Z = diag(sw)`` (``sw_g = sum_{i in g}
    w_i``) is **diagonal**.  Eliminating ``b`` via its (diagonal) block gives a
    ``p x p`` Schur system ``S beta = rhs``,  ``S = X^T W X - B D^{-1} B^T``,
    ``D = diag(sw) + lambda I``, ``B = X^T W Z`` -- all assembled from the
    per-group weighted sums (segment reductions), never an ``N x N`` or
    ``q x q`` dense factor.  Returns the updated ``(beta, b)`` plus the pieces the
    dispersion / Fellner-Schall update needs (``w``, ``sw``, ``swx``, ``Dinv``,
    ``S_inv``, ``XtWX``).
    """
    eta = X @ beta + b[group]
    mu = family.linkinv(eta)
    dmu = family.mu_eta(eta)
    var = family.variance(mu)
    w = dmu * dmu / jnp.clip(var, _EPS, None)  # (N,) working weights
    z = eta + (y - mu) / safe_dmu(dmu)  # (N,) working response

    sw = jax.ops.segment_sum(w, group, num_segments=n_groups)  # (q,)
    swz = jax.ops.segment_sum(w * z, group, num_segments=n_groups)  # (q,)
    swx = jax.ops.segment_sum(
        w[:, None] * X, group, num_segments=n_groups
    )  # (q, p) = (Z^T W X)
    xtwz = X.T @ (w * z)  # (p,)
    xtwx = (X * w[:, None]).T @ X  # (p, p)

    dinv = 1.0 / (sw + lam)  # (q,) diagonal of (Z^T W Z + lambda I)^{-1}
    bdinvbt = swx.T @ (dinv[:, None] * swx)  # (p, p) = B D^{-1} B^T
    s_mat = xtwx - bdinvbt + ridge_eye
    s_inv, _ = small_inv_logdet(s_mat, X.shape[-1])
    rhs_beta = xtwz - swx.T @ (dinv * swz)  # (p,)
    beta_new = s_inv @ rhs_beta
    b_new = dinv * (swz - swx @ beta_new)  # (q,)
    return beta_new, b_new, w, sw, swx, dinv, s_inv, xtwx


def _glmm_many_one(
    y: Float[Array, 'N'],
    X: Float[Array, 'N p'],
    group: Int[Array, 'N'],
    n_groups: int,
    family: Family,
    p: int,
    n_outer: int,
    n_inner: int,
    ridge: float,
    lam_floor: float,
    lam_ceil: float,
) -> Tuple[
    Float[Array, 'p'],
    Float[Array, 'q'],
    Float[Array, ''],
    Float[Array, ''],
    Float[Array, ''],
    Float[Array, ''],
]:
    """Single-element structured PQL fit (scalar random intercept).

    Returns ``(beta, blups, re_var, dispersion, deviance, edf_total)``.  Mirrors
    ``gam._gam_fit_one`` -- outer Fellner-Schall loop over the random-effect
    precision ``lambda``, inner penalised IRLS -- but every solve goes through the
    ``O(N p^2 + q)`` Schur complement (:func:`_structured_solve`), so it never
    forms the ``(p + q)``-wide system.
    """
    n = X.shape[0]
    q = n_groups
    ridge_eye = ridge * jnp.eye(p, dtype=y.dtype)

    def run_inner(
        lam: Float[Array, ''], beta: Array, b: Array
    ) -> Tuple[Array, Array]:
        """``n_inner`` penalised-IRLS steps at fixed precision ``lam``."""

        def body(_: Array, bb: Tuple[Array, Array]) -> Tuple[Array, Array]:
            beta_, b_ = bb
            beta2, b2, *_ = _structured_solve(
                y, X, group, q, family, lam, beta_, b_, ridge_eye
            )
            return beta2, b2

        return cast(
            Tuple[Array, Array], lax.fori_loop(0, n_inner, body, (beta, b))
        )

    def outer(
        carry: Tuple[Array, Array, Array], _: Array
    ) -> Tuple[Tuple[Array, Array, Array], None]:
        lam, beta, b = carry
        beta, b = run_inner(lam, beta, b)
        beta, b, w, sw, swx, dinv, s_inv, xtwx = _structured_solve(
            y, X, group, q, family, lam, beta, b, ridge_eye
        )
        phi, _ = _dispersion(
            y, X, group, family, beta, b, sw, swx, dinv, s_inv, xtwx, n
        )
        # Generalized Fellner-Schall update for the identity RE penalty S = I_q:
        #   lambda <- lambda (tr(S_lambda^+ S) - tr(V S)) / (b^T b / phi),
        # tr(S_lambda^+ S) = q / lambda (full-rank ridge), tr(V S) = tr(V_bb).
        energy = b @ b
        tr_vbb = jnp.sum(dinv) + jnp.trace(
            s_inv @ (swx.T @ (dinv[:, None] ** 2 * swx))
        )
        num = jnp.clip(q - lam * tr_vbb, 1e-8, None)
        den = jnp.clip(energy / phi, 1e-12, None)
        lam_new = jnp.clip(num / den, lam_floor, lam_ceil)
        return (lam_new, beta, b), None

    lam0 = jnp.asarray(1.0, dtype=y.dtype)
    beta0 = jnp.zeros((p,), dtype=y.dtype)
    b0 = jnp.zeros((q,), dtype=y.dtype)
    (lam, beta, b), _ = lax.scan(
        outer, (lam0, beta0, b0), xs=None, length=n_outer
    )

    beta, b = run_inner(lam, beta, b)
    beta, b, w, sw, swx, dinv, s_inv, xtwx = _structured_solve(
        y, X, group, q, family, lam, beta, b, ridge_eye
    )
    phi, edf_total = _dispersion(
        y, X, group, family, beta, b, sw, swx, dinv, s_inv, xtwx, n
    )
    mu = family.linkinv(X @ beta + b[group])
    deviance = jnp.sum(family.unit_deviance(y, mu))
    re_var = phi / jnp.clip(lam, _EPS, None)
    return beta, b, re_var, phi, deviance, edf_total


def _dispersion(
    y: Float[Array, 'N'],
    X: Float[Array, 'N p'],
    group: Int[Array, 'N'],
    family: Family,
    beta: Float[Array, 'p'],
    b: Float[Array, 'q'],
    sw: Float[Array, 'q'],
    swx: Float[Array, 'q p'],
    dinv: Float[Array, 'q'],
    s_inv: Float[Array, 'p p'],
    xtwx: Float[Array, 'p p'],
    n: int,
) -> Tuple[Float[Array, ''], Float[Array, '']]:
    """Dispersion ``phi`` and total EDF for the structured fit.

    ``edf = tr(V G)`` with ``V`` the full penalised inverse and ``G`` the
    unpenalised Gram, assembled blockwise from the Schur pieces (no ``q x q``
    factor): ``tr(V_bb diag(sw)) = sum_g sw_g (D^{-1}_g + D^{-2}_g x_g^T S^{-1}
    x_g)``, the fixed block ``tr(S^{-1} X^T W X)``, and the cross term
    ``-2 tr(S^{-1} B D^{-1} B^T)``.  For a fixed-dispersion family ``phi = 1``;
    otherwise it is the Pearson estimate ``sum (y - mu)^2 / V(mu) / (n - edf)``.
    """
    bdinvbt = swx.T @ (dinv[:, None] * swx)  # B D^{-1} B^T
    msw = swx.T @ ((sw * dinv * dinv)[:, None] * swx)  # sum_g sw_g D^-2 x x^T
    edf_total = (
        jnp.trace(s_inv @ xtwx)
        - 2.0 * jnp.trace(s_inv @ bdinvbt)
        + jnp.sum(sw * dinv)
        + jnp.trace(s_inv @ msw)
    )
    if family.has_fixed_dispersion:
        return jnp.asarray(1.0, dtype=y.dtype), edf_total
    mu = family.linkinv(X @ beta + b[group])
    var = jnp.clip(family.variance(mu), _EPS, None)
    pearson = jnp.sum((y - mu) ** 2 / var)
    phi = pearson / jnp.clip(n - edf_total, 1e-3, None)
    return phi, edf_total


def _glmm_many_level(
    Y: Float[Array, 'V N'],
    X: Float[Array, 'N p'],
    group: Int[Array, 'N'],
    n_groups: int,
    family: Family,
    n_outer: int,
    n_inner: int,
    ridge: float,
    lam_floor: float,
    lam_ceil: float,
    block: Optional[int],
) -> GLMMResult:
    """Structured PQL over ``V`` elements (scalar random intercept)."""
    p = X.shape[-1]

    def per_voxel(
        y: Float[Array, 'N'],
    ) -> Tuple[
        Float[Array, 'p'],
        Float[Array, 'q'],
        Float[Array, ''],
        Float[Array, ''],
        Float[Array, ''],
        Float[Array, ''],
    ]:
        return _glmm_many_one(
            y,
            X,
            group,
            n_groups,
            family,
            p,
            n_outer,
            n_inner,
            ridge,
            lam_floor,
            lam_ceil,
        )

    beta, blups, re_var, phi, deviance, edf_total = blocked_vmap(
        per_voxel, (Y,), block=block
    )
    return GLMMResult(
        beta_hat=beta,
        blups=blups,
        re_var=re_var[:, None, None],  # (V, 1, 1) -- uniform G shape (D4)
        dispersion=phi,
        deviance=deviance,
        edf_total=edf_total,
        family=family,
        n_obs=int(X.shape[0]),
        n_groups=n_groups,
        tier='many',
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Laplace-approximate GLMM (scalar random intercept) -- the §11 follow-up to PQL
# ---------------------------------------------------------------------------


def _laplace_conditional_modes(
    theta: Float[Array, 'p1'],
    y: Float[Array, 'N'],
    X: Float[Array, 'N p'],
    group: Int[Array, 'N'],
    n_groups: int,
    family: Family,
    p: int,
    n_mode: int,
) -> Tuple[Float[Array, 'q'], Float[Array, 'q'], Float[Array, 'N']]:
    """Per-group conditional modes ``b_hat`` (+ the curvature ``sw`` and ``eta``).

    Given the fixed effects and ``sigma_b^2`` (in ``theta = [beta, log
    sigma_b^2]``), the mode of ``sum_i log p(y_i | beta, b_g) - b_g^2/(2
    sigma_b^2)`` is found by per-group Newton (Fisher scoring), vectorised over
    groups via ``segment_sum`` -- ``O(N)``, no random-effect-wide system.
    """
    beta = theta[:p]
    sb2 = jnp.exp(theta[p])
    eta_fix = X @ beta

    def mode_step(
        b: Float[Array, 'q'], _: Array
    ) -> Tuple[Float[Array, 'q'], None]:
        eta = eta_fix + b[group]
        mu = family.linkinv(eta)
        var = jnp.clip(family.variance(mu), _EPS, None)
        dmu = family.mu_eta(eta)
        score = (y - mu) * dmu / var  # d log p / d eta
        w = dmu * dmu / var  # Fisher weight
        sg = jax.ops.segment_sum(score, group, num_segments=n_groups) - b / sb2
        sh = jax.ops.segment_sum(w, group, num_segments=n_groups) + 1.0 / sb2
        return b + sg / sh, None

    b, _ = lax.scan(
        mode_step, jnp.zeros((n_groups,), dtype=X.dtype), xs=None, length=n_mode
    )
    eta = eta_fix + b[group]
    mu = family.linkinv(eta)
    var = jnp.clip(family.variance(mu), _EPS, None)
    sw = jax.ops.segment_sum(
        family.mu_eta(eta) ** 2 / var, group, num_segments=n_groups
    )  # curvature -ell'' at the mode (Fisher)
    return b, sw, eta


def _laplace_nll(
    theta: Float[Array, 'p1'],
    y: Float[Array, 'N'],
    X: Float[Array, 'N p'],
    group: Int[Array, 'N'],
    n_groups: int,
    family: Family,
    p: int,
    n_mode: int,
) -> Float[Array, '']:
    """Laplace-approximate marginal negative log-likelihood at ``theta``.

    ``-sum_g [ ell_g(b_hat) - b_hat^2/(2 sigma_b^2) - 0.5 log sigma_b^2
    - 0.5 log(sum_i w_i + 1/sigma_b^2) ]`` -- the Laplace approximation to the
    random-effect integral (matches Gauss-Hermite to the approximation order, and
    corrects the PQL attenuation for binary / low-count responses).
    """
    sb2 = jnp.exp(theta[p])
    b, sw, eta = _laplace_conditional_modes(
        theta, y, X, group, n_groups, family, p, n_mode
    )
    mu = family.linkinv(eta)
    ll = jax.ops.segment_sum(
        family.loglik(y, mu, jnp.asarray(1.0, dtype=X.dtype)),
        group,
        num_segments=n_groups,
    )
    log_lg = (
        ll
        - b * b / (2.0 * sb2)
        - 0.5 * jnp.log(sb2)
        - 0.5 * jnp.log(sw + 1.0 / sb2)
    )
    return -jnp.sum(log_lg)


def _glmm_laplace_one(
    y: Float[Array, 'N'],
    X: Float[Array, 'N p'],
    group: Int[Array, 'N'],
    n_groups: int,
    family: Family,
    p: int,
    n_mode: int,
    spec: VarCompSpec,
) -> Tuple[
    Float[Array, 'p'],
    Float[Array, 'q'],
    Float[Array, ''],
    Float[Array, ''],
]:
    """Single-element Laplace GLMM fit.  Returns ``(beta, blups, re_var,
    deviance)``."""

    def nll(theta: Float[Array, 'p1']) -> Float[Array, '']:
        return _laplace_nll(theta, y, X, group, n_groups, family, p, n_mode)

    theta0 = jnp.concatenate(
        [jnp.zeros((p,), dtype=X.dtype), jnp.asarray([0.0], dtype=X.dtype)]
    )  # beta = 0, log sigma_b^2 = 0
    theta = damped_newton(nll, theta0, spec=spec)
    beta = theta[:p]
    sb2 = jnp.exp(theta[p])
    b, _, _ = _laplace_conditional_modes(
        theta, y, X, group, n_groups, family, p, n_mode
    )
    deviance = 2.0 * nll(theta)  # -2 log L_Laplace (the glmer deviance)
    return beta, b, sb2, deviance


def _glmm_laplace(
    Y: Float[Array, 'V N'],
    X: Float[Array, 'N p'],
    group: Int[Array, 'N'],
    n_groups: int,
    family: Family,
    n_outer: int,
    n_mode: int,
    damping: float,
    block: Optional[int],
) -> GLMMResult:
    """Laplace GLMM over ``V`` elements (scalar random intercept)."""
    p = X.shape[-1]
    spec = VarCompSpec(n_iter=n_outer, damping=damping)

    def per_voxel(
        y: Float[Array, 'N'],
    ) -> Tuple[Array, Array, Array, Array]:
        return _glmm_laplace_one(
            y, X, group, n_groups, family, p, n_mode, spec
        )

    beta, blups, re_var, deviance = blocked_vmap(per_voxel, (Y,), block=block)
    return GLMMResult(
        beta_hat=beta,
        blups=blups,
        re_var=re_var[:, None, None],  # (V, 1, 1) -- uniform G shape (D4)
        dispersion=jnp.ones_like(re_var),
        deviance=deviance,
        edf_total=jnp.full_like(re_var, float(p)),
        family=family,
        n_obs=int(X.shape[0]),
        n_groups=n_groups,
        tier='laplace',
    )


# ---------------------------------------------------------------------------
# Laplace-approximate random *slope* -- the r-dimensional lift of the scalar fit
# ---------------------------------------------------------------------------


def _laplace_slope_modes(
    theta: Float[Array, 'pm'],
    y: Float[Array, 'N'],
    X: Float[Array, 'N p'],
    z: Float[Array, 'N r'],
    group: Int[Array, 'N'],
    n_groups: int,
    family: Family,
    p: int,
    r: int,
    n_mode: int,
    diagonal: bool,
) -> Tuple[Float[Array, 'q r'], Float[Array, 'q r r'], Float[Array, 'N']]:
    """Per-group conditional modes ``b_hat`` (r-vector) + curvature ``H`` + eta.

    The ``r``-dimensional lift of :func:`_laplace_conditional_modes`: given the
    fixed effects and ``G`` (in ``theta = [beta, chol(G)]``), the mode of
    ``sum_i log p(y_i | beta, b_g) - 0.5 b_g^T G^{-1} b_g`` is found by per-group
    ``r x r`` Newton (Fisher scoring) -- gradient ``sum_i s_i z_i - G^{-1} b_g``,
    curvature ``H_g = sum_i w_i z_i z_i^T + G^{-1}`` -- vectorised over groups via
    ``segment_sum`` of the per-observation outer products.  Returns the modes, the
    per-group curvature ``H_g`` at the mode (for the determinant correction), and
    the linear predictor.

    ``H_g`` uses the **expected** (Fisher) information ``w_i = (dmu/deta)^2 / V``,
    not the observed Hessian ``-d^2 ell/db^2`` -- the deliberate ``glmer``-
    consistent "Fisher-scoring Laplace".  For the canonical links (logit / log)
    the two coincide; for a non-canonical link (probit / cloglog slope) they
    differ at higher order, so this is an approximation choice, not the exact
    observed-information Laplace.
    """
    from .lme._blockwoodbury import cov_re_from_chol

    beta = theta[:p]
    g_inv, _ = small_inv_logdet(cov_re_from_chol(theta[p:], r, diagonal), r)
    eta_fix = X @ beta

    def _curvature(eta: Array) -> Tuple[Array, Array, Array]:
        mu = family.linkinv(eta)
        var = jnp.clip(family.variance(mu), _EPS, None)
        dmu = family.mu_eta(eta)
        w = dmu * dmu / var  # (N,) Fisher weight
        wzz = jax.ops.segment_sum(
            w[:, None, None] * (z[:, :, None] * z[:, None, :]),
            group,
            num_segments=n_groups,
        )  # (q, r, r) = sum_i w_i z_i z_i^T
        return mu, dmu / var, wzz + g_inv[None]

    def mode_step(
        b: Float[Array, 'q r'], _: Array
    ) -> Tuple[Float[Array, 'q r'], None]:
        eta = family.clip_eta(eta_fix + jnp.einsum('nr,nr->n', z, b[group]))
        mu, dlog, h_mat = _curvature(eta)
        score = (y - mu) * dlog  # (N,) d log p / d eta
        grad = jax.ops.segment_sum(
            score[:, None] * z, group, num_segments=n_groups
        ) - jnp.einsum('rs,qs->qr', g_inv, b)  # (q, r)
        h_inv = jax.vmap(lambda a: small_inv_logdet(a, r)[0])(h_mat)
        return b + jnp.einsum('qrs,qs->qr', h_inv, grad), None

    b, _ = lax.scan(
        mode_step,
        jnp.zeros((n_groups, r), dtype=X.dtype),
        xs=None,
        length=n_mode,
    )
    eta = family.clip_eta(eta_fix + jnp.einsum('nr,nr->n', z, b[group]))
    _, _, h_mat = _curvature(eta)
    return b, h_mat, eta


def _laplace_slope_nll(
    theta: Float[Array, 'pm'],
    y: Float[Array, 'N'],
    X: Float[Array, 'N p'],
    z: Float[Array, 'N r'],
    group: Int[Array, 'N'],
    n_groups: int,
    family: Family,
    p: int,
    r: int,
    n_mode: int,
    diagonal: bool,
) -> Float[Array, '']:
    """Laplace-approximate marginal NLL for a random slope.

    ``-sum_g [ ell_g(b_hat) - 0.5 b_hat^T G^{-1} b_hat - 0.5 logdet G
    - 0.5 logdet H_g ]`` -- the ``r``-dimensional generalisation of the scalar
    determinant correction (reduces to it at ``r = 1``: ``logdet G = log
    sigma_b^2``, ``logdet H_g = log(sum_i w_i + 1/sigma_b^2)``).
    """
    from .lme._blockwoodbury import cov_re_from_chol

    g_inv, logdet_g = small_inv_logdet(
        cov_re_from_chol(theta[p:], r, diagonal), r
    )
    b, h_mat, eta = _laplace_slope_modes(
        theta, y, X, z, group, n_groups, family, p, r, n_mode, diagonal
    )
    mu = family.linkinv(eta)
    ll = jax.ops.segment_sum(
        family.loglik(y, mu, jnp.asarray(1.0, dtype=X.dtype)),
        group,
        num_segments=n_groups,
    )  # (q,)
    logdet_h = jax.vmap(lambda a: small_inv_logdet(a, r)[1])(h_mat)  # (q,)
    quad = 0.5 * jnp.einsum('qr,rs,qs->q', b, g_inv, b)  # 0.5 b^T G^{-1} b
    log_lg = ll - quad - 0.5 * logdet_g - 0.5 * logdet_h
    return -jnp.sum(log_lg)


def _glmm_laplace_slope_one(
    y: Float[Array, 'N'],
    X: Float[Array, 'N p'],
    z: Float[Array, 'N r'],
    group: Int[Array, 'N'],
    n_groups: int,
    family: Family,
    p: int,
    r: int,
    n_mode: int,
    spec: VarCompSpec,
    diagonal: bool,
) -> Tuple[Float[Array, 'p'], Float[Array, 'q r'], Float[Array, 'r r'], Float[Array, '']]:
    """Single-element Laplace random-slope GLMM fit.  Returns ``(beta, blups,
    G, deviance)``."""
    from ._irls import irls_warm_start
    from .lme._blockwoodbury import _param_layout, cov_re_from_chol

    def nll(theta: Float[Array, 'pm']) -> Float[Array, '']:
        return _laplace_slope_nll(
            theta, y, X, z, group, n_groups, family, p, r, n_mode, diagonal
        )

    # Link-scale init: warm-start beta, a small G (cf. the structured PQL slope).
    var_e = jnp.var(family.link(family.init_mu(y)))
    layout = _param_layout(r, diagonal)
    diag = jnp.asarray([i == j for (i, j) in layout])
    chol0 = jnp.where(
        diag, 0.5 * jnp.log(jnp.maximum(0.1 * var_e, 1e-6)), 0.0
    )
    beta0 = irls_warm_start(
        y, X, family, penalty=jnp.zeros((p, p), dtype=y.dtype), ridge=spec.ridge
    )
    theta0 = jnp.concatenate([beta0, chol0])
    theta = damped_newton(nll, theta0, spec=spec)
    beta = theta[:p]
    g_cov = cov_re_from_chol(theta[p:], r, diagonal)
    b, _, _ = _laplace_slope_modes(
        theta, y, X, z, group, n_groups, family, p, r, n_mode, diagonal
    )
    deviance = 2.0 * nll(theta)
    return beta, b, g_cov, deviance


def _glmm_laplace_slope(
    Y: Float[Array, 'V N'],
    X: Float[Array, 'N p'],
    group: Int[Array, 'N'],
    n_groups: int,
    z: Float[Array, 'N r'],
    family: Family,
    n_outer: int,
    n_mode: int,
    damping: float,
    diagonal: bool,
    block: Optional[int],
) -> GLMMResult:
    """Laplace random-slope GLMM over ``V`` elements."""
    p = X.shape[-1]
    r = z.shape[-1]
    spec = VarCompSpec(n_iter=n_outer, damping=damping)

    def per_voxel(
        y: Float[Array, 'N'],
    ) -> Tuple[Array, Array, Array, Array]:
        return _glmm_laplace_slope_one(
            y, X, z, group, n_groups, family, p, r, n_mode, spec, diagonal
        )

    beta, blups, g_cov, deviance = blocked_vmap(per_voxel, (Y,), block=block)
    # D4: uniform (V, r, r) G -- diagonal-valued in the diagonal case.
    re_var = g_cov
    nv = beta.shape[0]
    return GLMMResult(
        beta_hat=beta,
        blups=blups,
        re_var=re_var,
        dispersion=jnp.ones((nv,), dtype=beta.dtype),
        deviance=deviance,
        edf_total=jnp.full((nv,), float(p), dtype=beta.dtype),
        family=family,
        n_obs=int(X.shape[0]),
        n_groups=n_groups,
        tier='laplace',
    )


# ---------------------------------------------------------------------------
# Adaptive Gauss-Hermite quadrature -- the accuracy tier above Laplace
# ---------------------------------------------------------------------------


def _gh_tensor_nodes(
    n_quad: int, r: int
) -> Tuple[Float[Array, 'K r'], Float[Array, 'K'], Float[Array, 'K']]:
    """Tensor-product Gauss-Hermite nodes for an ``r``-dimensional integral.

    Returns ``(nodes, log_weights, sumsq)`` for the ``K = n_quad**r`` tensor nodes
    of the physicists' rule ``int exp(-t^2) g(t) dt ~ sum_k w_k g(t_k)``: the node
    coordinates ``(K, r)``, the log product weight ``sum_j log w_{k_j}`` ``(K,)``,
    and ``||t_k||^2`` ``(K,)``.  Static (computed once per fit in numpy)."""
    x, w = np.polynomial.hermite.hermgauss(n_quad)
    if r == 1:
        nodes = x[:, None]
        logw = np.log(w)
    else:
        mesh = np.meshgrid(*([x] * r), indexing='ij')
        nodes = np.stack([m.ravel() for m in mesh], axis=-1)  # (K, r)
        wmesh = np.meshgrid(*([w] * r), indexing='ij')
        logw = sum(np.log(wm.ravel()) for wm in wmesh)  # (K,)
    sumsq = np.sum(nodes**2, axis=-1)
    return jnp.asarray(nodes), jnp.asarray(logw), jnp.asarray(sumsq)


def _agq_slope_nll(
    theta: Float[Array, 'pm'],
    y: Float[Array, 'N'],
    X: Float[Array, 'N p'],
    z: Float[Array, 'N r'],
    group: Int[Array, 'N'],
    n_groups: int,
    family: Family,
    p: int,
    r: int,
    n_mode: int,
    diagonal: bool,
    nodes: Float[Array, 'K r'],
    logw: Float[Array, 'K'],
    sumsq: Float[Array, 'K'],
) -> Float[Array, '']:
    """Adaptive Gauss-Hermite marginal NLL for a random slope.

    Centres / scales the tensor GH nodes at each group's conditional mode and
    curvature -- ``b = b_hat_g + sqrt(2) L_g t_k`` with ``L_g L_g^T = H_g^{-1}``
    (the curvature factor, principal-axis adaptive) -- so

        log L_g = (r/2) log 2 - 0.5 logdet H_g
                  + logsumexp_k [ log w_k + ||t_k||^2 + log g(b_hat_g + offset_k) ],

    ``g(b) = exp(ell_g(b)) N(b; 0, G)`` the integrand.  With ``n_quad = 1``
    (``t = 0``, ``w = sqrt(pi)``) the single term collapses to the Laplace
    determinant correction exactly; more nodes integrate the random-effect
    density directly, correcting the Laplace bias for small / low-count clusters
    (lme4's ``nAGQ``).
    """
    from .lme._blockwoodbury import cov_re_from_chol

    beta = theta[:p]
    g_inv, logdet_g = small_inv_logdet(
        cov_re_from_chol(theta[p:], r, diagonal), r
    )
    b_hat, h_mat, _ = _laplace_slope_modes(
        theta, y, X, z, group, n_groups, family, p, r, n_mode, diagonal
    )
    # Per-group curvature factor L_g (L_g L_g^T = H_g^{-1}) and logdet H_g.
    h_inv, logdet_h = jax.vmap(lambda h: small_inv_logdet(h, r))(h_mat)
    l_scale = jax.vmap(lambda hi: spd_chol(hi, r))(h_inv)  # (q, r, r)
    sqrt2 = jnp.sqrt(jnp.asarray(2.0, dtype=X.dtype))
    offset = sqrt2 * jnp.einsum('grs,ks->gkr', l_scale, nodes)  # (q, K, r)
    b_node = b_hat[:, None, :] + offset  # (q, K, r)

    eta_fix = X @ beta
    eta = family.clip_eta(
        eta_fix[:, None] + jnp.einsum('nr,nkr->nk', z, b_node[group])
    )  # (N, K)
    ll = jax.ops.segment_sum(
        family.loglik(
            y[:, None], family.linkinv(eta), jnp.asarray(1.0, dtype=X.dtype)
        ),
        group,
        num_segments=n_groups,
    )  # (q, K)
    quad = 0.5 * jnp.einsum('gkr,rs,gks->gk', b_node, g_inv, b_node)
    log_g = ll - quad - 0.5 * logdet_g - 0.5 * r * jnp.log(2.0 * jnp.pi)
    log_marg = (
        0.5 * r * jnp.log(2.0)
        - 0.5 * logdet_h
        + logsumexp(logw[None, :] + sumsq[None, :] + log_g, axis=-1)
    )
    return -jnp.sum(log_marg)


def _glmm_agq_slope_one(
    y: Float[Array, 'N'],
    X: Float[Array, 'N p'],
    z: Float[Array, 'N r'],
    group: Int[Array, 'N'],
    n_groups: int,
    family: Family,
    p: int,
    r: int,
    n_mode: int,
    spec: VarCompSpec,
    diagonal: bool,
    nodes: Float[Array, 'K r'],
    logw: Float[Array, 'K'],
    sumsq: Float[Array, 'K'],
) -> Tuple[Float[Array, 'p'], Float[Array, 'q r'], Float[Array, 'r r'], Float[Array, '']]:
    """Single-element AGQ random-slope fit.  Returns ``(beta, blups, G,
    deviance)`` -- as the Laplace fit, but the marginal is the ``n_quad``-point
    adaptive GH integral."""
    from ._irls import irls_warm_start
    from .lme._blockwoodbury import _param_layout, cov_re_from_chol

    def nll(theta: Float[Array, 'pm']) -> Float[Array, '']:
        return _agq_slope_nll(
            theta, y, X, z, group, n_groups, family, p, r, n_mode, diagonal,
            nodes, logw, sumsq,
        )

    var_e = jnp.var(family.link(family.init_mu(y)))
    layout = _param_layout(r, diagonal)
    diag = jnp.asarray([i == j for (i, j) in layout])
    chol0 = jnp.where(
        diag, 0.5 * jnp.log(jnp.maximum(0.1 * var_e, 1e-6)), 0.0
    )
    beta0 = irls_warm_start(
        y, X, family, penalty=jnp.zeros((p, p), dtype=y.dtype), ridge=spec.ridge
    )
    theta = damped_newton(nll, jnp.concatenate([beta0, chol0]), spec=spec)
    beta = theta[:p]
    g_cov = cov_re_from_chol(theta[p:], r, diagonal)
    b, _, _ = _laplace_slope_modes(
        theta, y, X, z, group, n_groups, family, p, r, n_mode, diagonal
    )
    return beta, b, g_cov, 2.0 * nll(theta)


def _glmm_agq_slope(
    Y: Float[Array, 'V N'],
    X: Float[Array, 'N p'],
    group: Int[Array, 'N'],
    n_groups: int,
    z: Float[Array, 'N r'],
    family: Family,
    n_outer: int,
    n_mode: int,
    damping: float,
    diagonal: bool,
    n_quad: int,
    block: Optional[int],
) -> GLMMResult:
    """Adaptive Gauss-Hermite random-slope GLMM over ``V`` elements."""
    p = X.shape[-1]
    r = z.shape[-1]
    spec = VarCompSpec(n_iter=n_outer, damping=damping)
    nodes, logw, sumsq = _gh_tensor_nodes(n_quad, r)

    def per_voxel(
        y: Float[Array, 'N'],
    ) -> Tuple[Array, Array, Array, Array]:
        return _glmm_agq_slope_one(
            y, X, z, group, n_groups, family, p, r, n_mode, spec, diagonal,
            nodes, logw, sumsq,
        )

    beta, blups, g_cov, deviance = blocked_vmap(per_voxel, (Y,), block=block)
    # D4: uniform (V, r, r) G -- diagonal-valued in the diagonal case.
    re_var = g_cov
    nv = beta.shape[0]
    return GLMMResult(
        beta_hat=beta,
        blups=blups,
        re_var=re_var,
        dispersion=jnp.ones((nv,), dtype=beta.dtype),
        deviance=deviance,
        edf_total=jnp.full((nv,), float(p), dtype=beta.dtype),
        family=family,
        n_obs=int(X.shape[0]),
        n_groups=n_groups,
        tier='agq',
    )


def glmm_fit(
    Y: Float[Array, 'V N'],
    X: Float[Array, 'N p'],
    *,
    group: Int[Array, 'N'],
    n_groups: Optional[int] = None,
    z: Optional[Float[Array, 'N r']] = None,
    structure: str = 'unstructured',
    family: Union[str, Family] = GAUSSIAN,
    method: str = 'pql',
    few_level_max: int = 64,
    n_outer: int = 20,
    n_inner: int = 10,
    n_mode: int = 20,
    n_quad: int = 5,
    ridge: float = 1e-8,
    lam_floor: float = 1e-6,
    lam_ceil: float = 1e8,
    damping: float = 1e-6,
    block: Optional[int] = None,
) -> GLMMResult:
    """Mass-univariate random-intercept GLMM via PQL, dispatched on level count.

    Fits, per element, ``g(E[y | b]) = X beta + b[group]`` with ``b_j ~ N(0,
    sigma_b^2)`` under the GLM ``family``, by penalised quasi-likelihood (working
    response -> one variance-component step -> repeat).  The fixed design ``X`` is
    used as given (include your own intercept column); ``group`` is the
    ``(N,)`` integer grouping factor (levels ``0 .. q-1``).

    Dispatch (the v3 §0.1 performance-preserving invariant / §2 routing)
    -------------------------------------------------------------------

    - ``q <= few_level_max`` -- **few-level**: the dense GAMM path (``gam_fit`` +
      ``re_smooth``).  Optimal here; the per-voxel cost ``O((p + q)^3)`` is tiny.
    - ``q > few_level_max`` -- **many-level**: a structured Schur-complement PQL
      costing ``O(N p^2 + q)`` per voxel (the §1.1 block-Woodbury structure,
      weighted and wrapped in the IRLS loop), avoiding the dense ``(p + q)``-wide
      solve.

    Both paths run the identical PQL iteration, so they agree to the iterative
    tolerance -- the dispatch changes only the linear-algebra cost.

    Parameters
    ----------
    Y
        ``(V, N)`` responses.
    X
        ``(N, p)`` fixed-effect design (shared across elements; carries its own
        intercept).
    group
        ``(N,)`` integer grouping factor (random intercept per level).
    n_groups
        Optional **static** level count ``q`` (levels are ``0 .. q-1``).  When
        ``None`` (default) it is derived eagerly as ``int(max(group)) + 1`` --
        byte-identical to before, but that concretises ``group`` and so makes
        ``glmm_fit`` untraceable under ``jax.jit``.  Pass the count explicitly (a
        Python ``int``) to trace the whole fit under ``jit`` -- e.g. to fuse it
        into a larger program -- for every family / structure / method.
    z
        Optional ``(N, r)`` random-effect design for a random **slope** (e.g.
        ``[1, x]`` -> random intercept + random slope of ``x``).  ``None``
        (default) is the scalar random intercept ``(1 | g)``.  Mirrors
        ``lme_fit``'s ``z=`` argument; the Gaussian-family slope GLMM is the same
        REML fit as ``lme_fit(z=, structure=)`` (to optimiser tolerance).
    structure
        Random-effect covariance for a slope (``z`` given): ``'unstructured'``
        (full ``r x r`` ``G``, the correlated ``(1 + x | g)``) or ``'diagonal'``
        (independent variance components, the uncorrelated ``(x || g)``).  Ignored
        when ``z is None``.
    family
        GLM family (``'binomial'`` / ``'poisson'`` / ``'gamma'`` /
        ``'negbinomial'`` / ``'gaussian'`` or a :class:`Family`).  Gaussian is
        accepted (it reduces to the LME) but ``lme_fit`` / ``reml_fit`` are the
        direct route for a Gaussian mixed model.
    method
        ``'pql'`` (default) -- penalised quasi-likelihood (cheap; documented
        small-cluster attenuation for binary / low-count responses).
        ``'laplace'`` -- the Laplace-approximate marginal likelihood (per-group
        conditional-mode integral + curvature correction): more accurate for
        binary / low-count GLMMs (it corrects the PQL bias), at the cost of an
        inner mode-finding loop.  Fits a scalar random intercept, or a random
        **slope** when ``z`` is given (the ``r``-dimensional mode + ``r x r``
        determinant correction, ``structure`` selecting a full / diagonal ``G``);
        the ``tier`` is ``'laplace'``.
        ``'agq'`` -- adaptive Gauss-Hermite quadrature (**random slope only**, ``z``
        required): the marginal random-effect integral by ``n_quad``-point tensor
        GH, centred / scaled at each group's mode and curvature.  ``n_quad = 1`` is
        exactly Laplace; more nodes integrate the density directly, converging to
        the exact marginal (the ``lme4`` ``nAGQ`` accuracy tier, the gold standard
        for small / low-count clusters).  ``tier`` is ``'agq'``.
    few_level_max
        Dispatch threshold on the number of levels ``q`` (default ``64`` -- the
        regime where the dense solve stops being trivially cheap).  ``'pql'``
        only.
    n_outer, n_inner, n_mode, n_quad
        PQL outer (Fellner-Schall) and inner (IRLS) iteration budgets; ``n_mode``
        is the per-group conditional-mode Newton budget for ``method='laplace'`` /
        ``'agq'``; ``n_quad`` is the GH nodes **per dimension** for ``'agq'``
        (default ``5``; the integral uses ``n_quad ** r`` tensor nodes).
    ridge, lam_floor, lam_ceil, block
        Normal-equation stabiliser, smoothing-parameter clamps, and the optional
        element-block size bounding peak memory.

    Returns
    -------
    ``GLMMResult`` -- ``beta_hat``, per-level ``blups``, ``re_var``
    (``sigma_b^2``, or the ``r``-vector / ``r x r`` ``G`` for a slope),
    ``dispersion``, ``deviance``, ``edf_total``, and the ``tier`` that ran
    (``'few'`` / ``'many'`` / ``'slope'`` / ``'laplace'`` / ``'agq'``).

    Notes
    -----
    PQL carries the documented small-cluster bias for binary / low-count
    responses (it under-estimates the variance component; the bias vanishes as
    the per-group information grows).  Random *slopes* (``z`` + ``structure``) are
    fit by PQL -- diagonal via ``gam_fit`` blocks, correlated via the joint-Schur
    + REML-EM solver (``tier='slope'``) -- by Laplace (``method='laplace'``), or by
    adaptive Gauss-Hermite quadrature (``method='agq'``, ``tier='agq'``), the
    accuracy ladder for the slope-variance attenuation: AGQ with ``n_quad``
    nodes converges to the exact marginal (``n_quad = 1`` is Laplace).
    """
    family = resolve_family(family)
    n = X.shape[0]
    group = jnp.asarray(group)
    if Y.shape[-1] != n:
        raise ValueError(
            f'glmm_fit: Y.shape[-1]={Y.shape[-1]} must match N={n}.'
        )
    if group.shape[0] != n:
        raise ValueError(
            f'glmm_fit: group has {group.shape[0]} labels; expected N={n}.'
        )
    # The level count is a *static* shape driver (segment sums, output shapes,
    # the few/many dispatch).  Deriving it from the data with int(jnp.max(...))
    # concretises a tracer, so the eager-only default makes glmm_fit untraceable
    # under jax.jit (P7); pass n_groups explicitly -- the count the caller
    # already knows -- to fuse the whole fit (mirrors lme_fit / reml_fit, where
    # the level structure is likewise caller-supplied).
    if n_groups is None:
        n_groups = int(jnp.max(group)) + 1

    if z is not None:
        z = jnp.asarray(z, dtype=X.dtype)
        if z.ndim != 2 or z.shape[0] != n:
            raise ValueError(
                f'glmm_fit: z must be (N, r) with N={n}; got shape '
                f'{tuple(z.shape)}.'
            )
        if structure not in ('unstructured', 'diagonal'):
            raise ValueError(
                f"glmm_fit: structure={structure!r}; expected 'unstructured' "
                "or 'diagonal'."
            )
        diagonal = structure == 'diagonal'
        if method == 'laplace':
            return _glmm_laplace_slope(
                Y, X, group, n_groups, z, family,
                n_outer, n_mode, damping, diagonal, block,
            )
        if method == 'agq':
            r = z.shape[-1]
            n_nodes = n_quad**r
            if n_nodes > _AGQ_MAX_NODES:
                raise ValueError(
                    f'glmm_fit: AGQ tensor grid has n_quad**r = {n_quad}**{r} '
                    f'= {n_nodes} nodes (> {_AGQ_MAX_NODES}); the per-element '
                    f'graph grows as n_quad**r and is differentiated through the '
                    f'mode scan, so a large r is a compile / memory cliff.  Use a '
                    f"smaller n_quad, method='laplace' (= AGQ with n_quad=1), or "
                    f'a lower-dimensional random effect.'
                )
            return _glmm_agq_slope(
                Y, X, group, n_groups, z, family,
                n_outer, n_mode, damping, diagonal, n_quad, block,
            )
        if method != 'pql':
            raise ValueError(
                f"glmm_fit: method={method!r}; expected 'pql', 'laplace' or "
                "'agq'."
            )
        if diagonal:
            return _glmm_slope_diagonal(
                Y, X, group, n_groups, z, family,
                n_outer, n_inner, ridge, lam_floor, lam_ceil, block,
            )
        return _glmm_slope_structured(
            Y, X, group, n_groups, z, family,
            n_outer, n_inner, ridge, block,
        )

    if method == 'laplace':
        return _glmm_laplace(
            Y, X, group, n_groups, family, n_outer, n_mode, damping, block
        )
    if method == 'agq':
        raise NotImplementedError(
            "glmm_fit: adaptive Gauss-Hermite (method='agq') currently requires "
            'a random slope -- pass z= (use z=ones((N, 1)) for a scalar '
            "intercept), or use method='laplace'."
        )
    if method != 'pql':
        raise ValueError(
            f"glmm_fit: method={method!r}; expected 'pql' or 'laplace'."
        )

    args = (
        Y,
        X,
        group,
        n_groups,
        family,
        n_outer,
        n_inner,
        ridge,
        lam_floor,
        lam_ceil,
        block,
    )
    if n_groups <= few_level_max:
        return _glmm_few_level(*args)
    return _glmm_many_level(*args)
