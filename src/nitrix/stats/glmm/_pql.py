# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""PQL (penalised quasi-likelihood) GLMM solvers -- the ``method='pql'`` family.

Few-level dense GAMM path, the uncorrelated diagonal random-slope, the
unstructured (correlated) random-slope joint-Schur + REML-EM, and the many-level
structured Schur-complement PQL.  Split from the ``glmm`` monolith (audit O1).
"""

from __future__ import annotations

from typing import Optional, Tuple, cast

import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Float, Int

from ...linalg._smalllinalg import small_inv_logdet
from .._batching import blocked_vmap
from .._family import Family
from .._irls import safe_dmu
from ..basis import re_smooth
from ..gam import gam_fit
from ._base import _EPS, GLMMResult

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
        (w[:, None] * z)[:, :, None] * z[:, None, :],
        group,
        num_segments=n_groups,
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
    zr = zty - jnp.einsum(
        'gpr,p->gr', xtz, beta_new
    )  # Z_g^T W (z_work - X beta)
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
    from .._irls import irls_warm_start

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
        def step(
            c: Tuple[Array, Array], _: Array
        ) -> Tuple[Tuple[Array, Array], None]:
            be, bb = c
            be2, bb2, *_ = _slope_solve(
                y,
                X,
                z,
                group,
                n_groups,
                family,
                r,
                g_inv,
                se2,
                be,
                bb,
                ridge_eye,
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
        beta: Array,
        b: Array,
        w: Array,
        z_work: Array,
        ztz: Array,
        xtz: Array,
        dinv: Array,
        s_inv: Array,
        se2: Array,
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
            y,
            X,
            z,
            group,
            n_groups,
            family,
            p,
            r,
            n_outer,
            n_inner,
            ridge,
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
