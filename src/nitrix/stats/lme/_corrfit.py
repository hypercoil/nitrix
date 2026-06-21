# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Generalised-least-squares REML with a structured within-group residual (§1.4).

This is the **R0 + corr** path: no random effect, a structured residual
``Cov(y_v) = sigma_e^2 R(rho)`` block-diagonal across a grouping factor, with
``R`` one of the :mod:`._corr` structures (``ar1`` / ``car1`` / ``cs``).  It is
the ``nlme::gls(correlation=corAR1 / corCAR1 / corCompSymm)`` fit, mass-
univariate over voxels with a shared design.

Method.  Each structure whitens per group (``W_i R_i W_i^T = I``), so on whitened
data the residual is i.i.d. and the profile REML criterion is the ordinary GLS
one plus the whitening Jacobian ``sum_i 0.5 log|R_i|``:

    -2 l_R(rho) = (N - p) log(rss(rho)) + log|R(rho)| + log|X~^T X~(rho)| + const

with ``X~`` / ``r~`` the whitened design / residual, ``beta_hat = (X~^T X~)^{-1}
X~^T y~`` and ``sigma_e^2 = rss / (N - p)`` profiled out.  Newton (damped,
backtracked, autodiff grad/Hessian) optimises the unconstrained ``raw`` (one
parameter for ar1/car1/cs); every solve is the cuSOLVER-free ``small_inv_logdet``
on the ``(p, p)`` whitened Gram.  Groups ride in a left-packed, time-sorted
``(G, T)`` padded layout so ragged sizes need no dynamic shapes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Optional, Tuple, Union, cast

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, Int

from ...linalg._smalllinalg import small_inv_logdet
from .._batching import blocked_vmap
from .._optimise import damped_newton
from .._result import register_result
from ._blockwoodbury import _nll_and_beta
from ._corr import CorrSpec, resolve_corr
from ._recov import _param_layout, cov_re_from_chol
from ._varcomp import VarCompSpec
from ._varfunc import VarFunc, _apply_var_scale

__all__ = [
    'GLSResult',
    'CorrLMEResult',
    'build_group_layout',
    'fit_corr_gls',
    'gls_fit',
    'fit_corr_lme',
]


class GroupLayout(NamedTuple):
    """Left-packed, time-sorted ``(G, T)`` padded group layout (shared across
    voxels; data-independent of ``y``)."""

    idx: Int[Array, 'G T']  # original row index (pad -> 0)
    gaps: Float[Array, 'G T']  # time gap to previous in-group obs (t>=1)
    nsize: Float[Array, 'G']  # real group sizes
    mask: Bool[Array, 'G T']  # validity mask


def build_group_layout(
    group: Int[Array, 'N'],
    time: Optional[Float[Array, 'N']] = None,
) -> GroupLayout:
    """Build the padded, time-sorted group layout from labels (and times).

    ``group`` is the ``(N,)`` integer grouping factor; ``time`` the ``(N,)``
    observation time (defaults to the within-group appearance order -- unit gaps,
    the AR(1) case).  Observations are sorted by ``time`` within each group and
    left-packed into a ``(G, T)`` grid (``T`` = max group size); ``gaps[g, t]`` is
    the time delta to the previous in-group observation (``0`` at ``t = 0`` and
    on the pad).  Host-side (NumPy) -- ``group`` / ``time`` are static across the
    mass axis.
    """
    g_np = np.asarray(group)
    n = g_np.shape[0]
    n_groups = int(g_np.max()) + 1 if n else 0
    t_np = (
        np.asarray(time, dtype=np.float64)
        if time is not None
        else np.zeros(n, dtype=np.float64)
    )
    # Stable order: by group, then time, then original index (ties -> appearance).
    if time is None:
        # Unit-gap (AR1): preserve appearance order within group.
        keys = (np.arange(n), g_np)
    else:
        keys = (np.arange(n), t_np, g_np)
    order = np.lexsort(keys)  # row indices, sorted by group then time
    g_sorted = g_np[order]
    sizes = np.bincount(g_sorted, minlength=n_groups)
    t_max = int(sizes.max()) if n_groups else 0

    # Left-pack vectorised (no Python loop over N): the sorted array is grouped
    # contiguously, so each entry's within-group column is its sorted position
    # minus its group's start offset.
    group_start = np.zeros(n_groups, dtype=np.int64)
    group_start[1:] = np.cumsum(sizes)[:-1]
    within = np.arange(n, dtype=np.int64) - group_start[g_sorted]  # (N,)

    idx = np.zeros((n_groups, t_max), dtype=np.int64)
    gaps = np.zeros((n_groups, t_max), dtype=np.float64)
    mask = np.zeros((n_groups, t_max), dtype=bool)
    idx[g_sorted, within] = order
    mask[g_sorted, within] = True
    if t_max:
        t_sorted = t_np[order]
        prev_t = np.empty(n, dtype=np.float64)
        prev_t[:1] = 0.0
        prev_t[1:] = t_sorted[:-1]
        # gap to the previous in-group observation (column 0 has none -> 0);
        # unit gaps when no times are given (AR1).
        gap_val = (t_sorted - prev_t) if time is not None else np.ones(n)
        gaps[g_sorted, within] = np.where(within >= 1, gap_val, 0.0)

    return GroupLayout(
        idx=jnp.asarray(idx),
        gaps=jnp.asarray(gaps),
        nsize=jnp.asarray(sizes.astype(np.float64)),
        mask=jnp.asarray(mask),
    )


@register_result(
    children=(
        'beta_hat',
        'sigma_e_sq',
        'rho',
        'var_params',
        'log_lik',
        'fixed_cov',
    ),
    aux=('df_resid', 'corr', 'weights'),
)
@dataclass(frozen=True)
class GLSResult:
    """Per-voxel GLS fit with a structured residual.

    Attributes
    ----------
    beta_hat
        ``(V, p)`` fixed-effect estimates.
    sigma_e_sq
        ``(V,)`` residual variance ``sigma_e^2``.
    rho
        ``(V,)`` natural correlation parameter ``rho`` (the structure's bounded
        parameter; the lag-1 / decay / exchangeable correlation).  ``0`` for the
        ``iid`` (no-correlation) structure.
    var_params
        ``(V, n_v)`` estimated variance-function parameters (``delta`` for
        ``varPower``; the ``S - 1`` log-ratios ``tau`` for ``varIdent``).  Empty
        ``(V, 0)`` when no ``weights`` variance function was given.
    log_lik
        ``(V,)`` profile REML log-likelihood at the fit.
    fixed_cov
        ``(V, p, p)`` ``Cov(beta_hat) = sigma_e^2 (X~^T X~)^{-1}`` (the GLS
        covariance) for a fixed-effect contrast on ``df = N - p``.
    df_resid
        Residual degrees of freedom ``N - p`` (scalar; for a t / F contrast).
    corr
        The correlation structure name.
    weights
        The variance-function name (``'varPower'`` / ``'varIdent'``), or ``None``.
    """

    beta_hat: Float[Array, 'V p']
    sigma_e_sq: Float[Array, 'V']
    rho: Float[Array, 'V']
    var_params: Float[Array, 'V n_v']
    log_lik: Float[Array, 'V']
    fixed_cov: Float[Array, 'V p p']
    df_resid: int
    corr: str
    weights: Optional[str]


def _whitened_grams(
    raw: Float[Array, 'k'],
    y_pad: Float[Array, 'G T'],
    x_pad: Float[Array, 'G T p'],
    layout: GroupLayout,
    corr: CorrSpec,
    varfunc: Optional[VarFunc],
    cov_pad: Optional[Float[Array, 'G T']],
    n_corr: int,
) -> Tuple[
    Float[Array, 'p p'], Float[Array, 'p'], Float[Array, ''], Float[Array, '']
]:
    """Whitened cross-products ``(X~^T X~, X~^T y~, y~^T y~, half_logdet)``.

    The Newton vector is ``raw = [corr_raw, var_raw]``; the variance function (if
    any) pre-scales the joint stack by ``1 / g`` and contributes ``sum_i log g_i``
    to the half-log-det before the correlation whitener runs.
    """
    # Whiten y and X jointly (stack as the channel axis) so the recurrence runs
    # once over the shared (G, T) structure.
    stack = jnp.concatenate([x_pad, y_pad[..., None]], axis=-1)  # (G, T, p+1)
    var_half = jnp.asarray(0.0, dtype=stack.dtype)
    if varfunc is not None:
        assert cov_pad is not None
        stack, var_half = _apply_var_scale(
            stack, cov_pad, layout.mask, varfunc, raw[n_corr:]
        )
    w, half_logdet = corr.whiten(
        stack, layout.gaps, layout.nsize, layout.mask, raw[:n_corr]
    )
    wx = w[..., :-1]  # (G, T, p)
    wy = w[..., -1]  # (G, T)
    xtx = jnp.einsum('gtp,gtq->pq', wx, wx)
    xty = jnp.einsum('gtp,gt->p', wx, wy)
    yty = jnp.sum(wy * wy)
    return xtx, xty, yty, half_logdet + var_half


def _fit_one(
    y_pad: Float[Array, 'G T'],
    x_pad: Float[Array, 'G T p'],
    layout: GroupLayout,
    corr: CorrSpec,
    varfunc: Optional[VarFunc],
    cov_pad: Optional[Float[Array, 'G T']],
    n_corr: int,
    raw0: Float[Array, 'k'],
    n: int,
    p: int,
    spec: VarCompSpec,
) -> Tuple[
    Float[Array, 'p'],
    Float[Array, ''],
    Float[Array, ''],
    Float[Array, 'n_v'],
    Float[Array, ''],
    Float[Array, 'p p'],
]:
    """Single-voxel GLS-REML fit via the shared saddle-free Newton
    (``_optimise.damped_newton``)."""
    dof = float(n - p)
    ridge = spec.ridge

    def neg2_reml(raw: Float[Array, 'k']) -> Float[Array, '']:
        xtx, xty, yty, half_logdet = _whitened_grams(
            raw, y_pad, x_pad, layout, corr, varfunc, cov_pad, n_corr
        )
        xtx_r = xtx + ridge * jnp.eye(p, dtype=xtx.dtype)
        xtx_inv, logdet_xtx = small_inv_logdet(xtx_r, p)
        beta = xtx_inv @ xty
        rss = jnp.clip(yty - beta @ xty, 1e-30, None)
        return dof * jnp.log(rss) + 2.0 * half_logdet + logdet_xtx

    raw = damped_newton(neg2_reml, raw0, **spec.newton_kwargs)

    xtx, xty, yty, half_logdet = _whitened_grams(
        raw, y_pad, x_pad, layout, corr, varfunc, cov_pad, n_corr
    )
    xtx_r = xtx + ridge * jnp.eye(p, dtype=xtx.dtype)
    xtx_inv, logdet_xtx = small_inv_logdet(xtx_r, p)
    beta = xtx_inv @ xty
    rss = jnp.clip(yty - beta @ xty, 1e-30, None)
    sigma2 = rss / dof
    fixed_cov = sigma2 * xtx_inv
    rho = corr.to_natural(raw[:n_corr])
    var_params = raw[n_corr:]
    # profile REML log-likelihood (full, with constants) for reporting.
    neg2 = (
        dof * jnp.log(sigma2)
        + 2.0 * half_logdet
        + logdet_xtx
        + dof
        + dof * jnp.log(2.0 * jnp.pi)
    )
    log_lik = -0.5 * neg2
    return beta, sigma2, rho, var_params, log_lik, fixed_cov


def fit_corr_gls(
    Y: Float[Array, 'V N'],
    X: Float[Array, 'N p'],
    group: Int[Array, 'N'],
    corr: CorrSpec,
    *,
    time: Optional[Float[Array, 'N']] = None,
    weights: Optional[VarFunc] = None,
    n_iter: int = 15,
    damping: float = 1e-6,
    block: Optional[int] = None,
) -> GLSResult:
    """Voxelwise GLS-REML with a structured within-group residual.

    Fits ``y_v = X beta_v + eps_v``, ``Cov(eps_v) = sigma_e^2 diag(g) R(rho)
    diag(g)`` block-diagonal across ``group``, with ``R`` the ``corr`` structure
    and ``g`` the optional ``weights`` variance function.  ``X`` is shared across
    voxels; only ``y_v`` varies.  Returns a :class:`GLSResult`.
    """
    n, p = X.shape
    n_corr = corr.n_params
    if n_corr + (0 if weights is None else weights.n_params) == 0:
        raise ValueError(
            "fit_corr_gls: nothing to estimate -- 'iid' correlation with no "
            'weights variance function is ordinary least squares (use glm_fit).'
        )
    layout = build_group_layout(group, time)
    idx = layout.idx
    mask_f = layout.mask.astype(X.dtype)
    x_pad = X[idx] * mask_f[..., None]  # (G, T, p) shared
    cov_pad = (
        None
        if weights is None
        else jnp.asarray(weights.covariate)[idx].astype(X.dtype)
    )
    raw0 = (
        corr.init_raw(X.dtype)
        if weights is None
        else jnp.concatenate(
            [corr.init_raw(X.dtype), weights.init_raw(X.dtype)]
        )
    )
    spec = VarCompSpec(n_iter=n_iter, damping=damping)

    def per_voxel(
        y: Float[Array, 'N'],
    ) -> Tuple[Array, Array, Array, Array, Array, Array]:
        y_pad = y[idx] * mask_f
        return _fit_one(
            y_pad, x_pad, layout, corr, weights, cov_pad, n_corr, raw0, n, p, spec
        )

    beta, sigma2, rho, var_params, log_lik, fixed_cov = cast(
        Tuple[Array, Array, Array, Array, Array, Array],
        blocked_vmap(per_voxel, (Y,), block=block),
    )
    return GLSResult(
        beta_hat=beta,
        sigma_e_sq=sigma2,
        rho=rho,
        var_params=var_params,
        log_lik=log_lik,
        fixed_cov=fixed_cov,
        df_resid=int(n - p),
        corr=corr.name,
        weights=None if weights is None else weights.name,
    )


def gls_fit(
    Y: Float[Array, 'V N'],
    X: Float[Array, 'N p'],
    *,
    group: Int[Array, 'N'],
    corr: Union[str, CorrSpec] = 'iid',
    time: Optional[Float[Array, 'N']] = None,
    weights: Optional[VarFunc] = None,
    n_iter: int = 15,
    damping: float = 1e-6,
    block: Optional[int] = None,
) -> GLSResult:
    """Voxelwise generalised least squares with a structured residual (§1.4).

    Fits, per voxel, ``y_v = X beta_v + eps_v`` with a within-group structured
    residual ``Cov(eps_v) = sigma_e^2 diag(g) R(rho) diag(g)`` -- ``nlme``'s
    ``gls(correlation=corAR1 / corCAR1 / corCompSymm, weights=varPower /
    varIdent)``.  ``R`` is the correlation structure; ``g`` the optional variance
    function (heteroscedasticity).  No random effect (the R0 + corr path); for a
    random effect *plus* a structured residual, the composition with the R2
    block-Woodbury is the follow-up.

    Parameters
    ----------
    Y, X
        ``(V, N)`` responses and ``(N, p)`` shared fixed-effect design (carries
        its own intercept).
    group
        ``(N,)`` integer grouping factor.  The residual is correlated *within*
        groups and independent across them (block-diagonal ``V``).
    corr
        Correlation structure: ``'ar1'`` (discrete AR(1)), ``'car1'``
        (continuous-time AR(1); pass ``time``), ``'cs'`` (compound symmetry),
        ``'iid'`` (no correlation -- the default, for a pure variance-function
        fit), or a :class:`CorrSpec`.
    time
        ``(N,)`` observation times for ``car1`` (and to order ``ar1`` when the
        rows are not already in within-group time order).  Defaults to the
        within-group appearance order (unit gaps).
    weights
        Optional residual **variance function** (heteroscedasticity):
        ``var_power(v)`` (``Var ~ |v|^{2 delta}``) or ``var_ident(strata)``
        (a separate variance per stratum).  Composes with any ``corr``; with
        ``corr='iid'`` it is a pure heteroscedastic GLS.
    n_iter, damping, block
        Newton iterations, AI/LM damping, and the optional voxel-block size.

    Returns
    -------
    ``GLSResult`` -- ``beta_hat``, ``sigma_e_sq``, ``rho``, ``var_params``,
    ``log_lik``, the GLS ``fixed_cov``, and ``df_resid = N - p``.
    """
    return fit_corr_gls(
        Y,
        X,
        jnp.asarray(group),
        resolve_corr(corr),
        time=None if time is None else jnp.asarray(time),
        weights=weights,
        n_iter=n_iter,
        damping=damping,
        block=block,
    )


# ---------------------------------------------------------------------------
# R2 + corr: a random effect *and* a structured residual (whitened block-Woodbury)
# ---------------------------------------------------------------------------


@register_result(
    children=('beta_hat', 'cov_re', 'sigma_e_sq', 'rho', 'log_lik'),
    aux=('corr', 'tier'),
)
@dataclass(frozen=True)
class CorrLMEResult:
    """Per-voxel mixed model with a structured within-group residual (R2 + corr).

    Attributes
    ----------
    beta_hat
        ``(V, p)`` fixed-effect estimates.
    cov_re
        ``(V, r, r)`` random-effect covariance ``G``.
    sigma_e_sq
        ``(V,)`` residual *scale* ``sigma_e^2`` (the residual is
        ``sigma_e^2 R(rho)``).
    rho
        ``(V,)`` natural correlation parameter of the residual structure.
    log_lik
        ``(V,)`` profile REML log-likelihood.
    corr
        Correlation-structure name.
    tier
        ``'R2+corr'``.
    """

    beta_hat: Float[Array, 'V p']
    cov_re: Float[Array, 'V r r']
    sigma_e_sq: Float[Array, 'V']
    rho: Float[Array, 'V']
    log_lik: Float[Array, 'V']
    corr: str
    tier: str


def _corr_lme_grams(
    raw: Float[Array, 'k'],
    x_pad: Float[Array, 'G T p'],
    z_pad: Float[Array, 'G T r'],
    y_pad: Float[Array, 'G T'],
    layout: GroupLayout,
    corr: CorrSpec,
    p: int,
    r: int,
) -> Tuple[
    Float[Array, 'G r r'],
    Float[Array, 'G p r'],
    Float[Array, 'p p'],
    Float[Array, 'G r'],
    Float[Array, 'p'],
    Float[Array, ''],
    Float[Array, ''],
]:
    """Whiten ``X`` / ``Z`` / ``y`` per group and form the block-Woodbury Grams.

    On whitened data the residual is i.i.d., so ``Sigma_e = sigma_e^2 R`` reduces
    to ``sigma_e^2 I`` and the per-group ``(Z^T Z, X^T Z, Z^T y)`` / total
    ``(X^T X, X^T y, y^T y)`` Grams feed ``_blockwoodbury._nll_and_beta``
    verbatim; the whitening adds ``half_logdet = 0.5 sum_i log|R_i|`` to the
    REML objective.
    """
    stack = jnp.concatenate(
        [x_pad, z_pad, y_pad[..., None]], axis=-1
    )  # (G, T, p+r+1)
    w, half_logdet = corr.whiten(
        stack, layout.gaps, layout.nsize, layout.mask, raw
    )
    wx = w[..., :p]  # (G, T, p)
    wz = w[..., p : p + r]  # (G, T, r)
    wy = w[..., -1]  # (G, T)
    ztz = jnp.einsum('gtr,gts->grs', wz, wz)  # (G, r, r)
    xtz = jnp.einsum('gtp,gtr->gpr', wx, wz)  # (G, p, r)
    zty = jnp.einsum('gtr,gt->gr', wz, wy)  # (G, r)
    xtx = jnp.einsum('gtp,gtq->pq', wx, wx)  # (p, p) total
    xty = jnp.einsum('gtp,gt->p', wx, wy)  # (p,) total
    yty = jnp.sum(wy * wy)
    return ztz, xtz, xtx, zty, xty, yty, half_logdet


def _fit_one_corr_lme(
    y_pad: Float[Array, 'G T'],
    x_pad: Float[Array, 'G T p'],
    z_pad: Float[Array, 'G T r'],
    layout: GroupLayout,
    corr: CorrSpec,
    theta_init: Float[Array, 'nt'],
    n: int,
    p: int,
    r: int,
    nt_g: int,
    n_minus_mr: int,
    diagonal: bool,
    spec: VarCompSpec,
) -> Tuple[Float[Array, 'nt'], Float[Array, 'p'], Float[Array, '']]:
    """Single-voxel R2 + corr fit via the shared saddle-free Newton over
    ``[chol(G), log sigma_e^2, corr_raw]`` (``_optimise.damped_newton``).

    The joint objective is non-convex away from the optimum (an indefinite
    Hessian at the start), so the saddle-free step rule is load-bearing here --
    it lives once in ``_optimise``.
    """

    def split(
        theta: Float[Array, 'nt'],
    ) -> Tuple[Float[Array, 'g1'], Float[Array, 'k']]:
        return theta[: nt_g + 1], theta[nt_g + 1 :]

    def nll(theta: Float[Array, 'nt']) -> Float[Array, '']:
        gse, raw = split(theta)
        ztz, xtz, xtx, zty, xty, yty, half_logdet = _corr_lme_grams(
            raw, x_pad, z_pad, y_pad, layout, corr, p, r
        )
        base, _ = _nll_and_beta(
            gse,
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
        return base + half_logdet

    theta = damped_newton(nll, theta_init, **spec.newton_kwargs)
    gse, raw = split(theta)
    ztz, xtz, xtx, zty, xty, yty, half_logdet = _corr_lme_grams(
        raw, x_pad, z_pad, y_pad, layout, corr, p, r
    )
    base, beta = _nll_and_beta(
        gse,
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
    return theta, beta, -(base + half_logdet)


def fit_corr_lme(
    Y: Float[Array, 'V N'],
    X: Float[Array, 'N p'],
    Z: Float[Array, 'N r'],
    group: Int[Array, 'N'],
    corr: CorrSpec,
    *,
    time: Optional[Float[Array, 'N']] = None,
    diagonal: bool = False,
    n_iter: int = 30,
    damping: float = 1e-6,
    block: Optional[int] = None,
) -> CorrLMEResult:
    """Voxelwise mixed model with a structured within-group residual (R2 + corr).

    Fits ``y_v = X beta_v + Z b_v + eps_v`` with ``b_v ~ N(0, G)`` and
    ``Cov(eps_v) = sigma_e^2 R(rho)`` block-diagonal across ``group`` -- the
    ``nlme::lme(random=~Z|g, correlation=corAR1/…)`` model.  Whitening by ``R``
    reduces each group to a standard block-Woodbury, so the per-group ``r x r``
    Woodbury algebra is reused verbatim with ``rho`` joining the REML
    ``theta``-vector; cuSOLVER-free.
    """
    n, p = X.shape
    r = Z.shape[-1]
    layout = build_group_layout(group, time)
    n_groups = int(layout.nsize.shape[0])
    idx = layout.idx
    mask_f = layout.mask.astype(X.dtype)
    x_pad = X[idx] * mask_f[..., None]  # (G, T, p) shared
    z_pad = Z[idx] * mask_f[..., None]  # (G, T, r) shared
    nt_g = len(_param_layout(r, diagonal))
    n_minus_mr = n - n_groups * r

    var_y = jnp.var(Y, axis=-1)  # (V,)
    layout_g = _param_layout(r, diagonal)
    diag_mask = jnp.asarray([i == j for (i, j) in layout_g], dtype=bool)
    chol_diag = 0.5 * jnp.log(jnp.maximum(0.1 * var_y, 1e-6))  # (V,)
    chol = jnp.where(diag_mask[None, :], chol_diag[:, None], 0.0)  # (V, nt_g)
    log_se2 = jnp.log(jnp.maximum(0.5 * var_y, 1e-6))[:, None]  # (V, 1)
    raw0 = jnp.broadcast_to(
        corr.init_raw(X.dtype)[None, :], (Y.shape[0], corr.n_params)
    )
    theta_init = jnp.concatenate([chol, log_se2, raw0], axis=1)  # (V, nt)
    spec = VarCompSpec(n_iter=n_iter, damping=damping)

    def per_voxel(
        y: Float[Array, 'N'], th: Float[Array, 'nt']
    ) -> Tuple[Array, Array, Array]:
        y_pad = y[idx] * mask_f
        return _fit_one_corr_lme(
            y_pad,
            x_pad,
            z_pad,
            layout,
            corr,
            th,
            n,
            p,
            r,
            nt_g,
            n_minus_mr,
            diagonal,
            spec,
        )

    theta_hat, beta_hat, log_lik = cast(
        Tuple[Array, Array, Array],
        blocked_vmap(per_voxel, (Y, theta_init), block=block),
    )
    cov_re = jax.vmap(lambda th: cov_re_from_chol(th[:nt_g], r, diagonal))(
        theta_hat
    )  # (V, r, r)
    sigma_e_sq = jnp.exp(theta_hat[:, nt_g])
    rho = jax.vmap(lambda th: corr.to_natural(th[nt_g + 1 :]))(theta_hat)
    return CorrLMEResult(
        beta_hat=beta_hat,
        cov_re=cov_re,
        sigma_e_sq=sigma_e_sq,
        rho=rho,
        log_lik=log_lik,
        corr=corr.name,
        tier='R2+corr',
    )
