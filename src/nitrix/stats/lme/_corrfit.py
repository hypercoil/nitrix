# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Generalised-least-squares REML with a structured within-group residual.

This is the residual-only path: no random effect, a structured residual
:math:`\\operatorname{Cov}(y_v) = \\sigma_e^2 R(\\rho)` block-diagonal across a
grouping factor, with :math:`R` one of the correlation structures (``'ar1'`` /
``'car1'`` / ``'cs'``).  It matches the
``nlme::gls(correlation=corAR1 / corCAR1 / corCompSymm)`` fit, mass-univariate
over voxels with a shared design.

Method.  Each structure whitens per group
(:math:`W_i R_i W_i^{\\top} = I`), so on whitened data the residual is i.i.d.
and the profile REML criterion is the ordinary GLS one plus the whitening
Jacobian :math:`\\sum_i \\tfrac{1}{2} \\log|R_i|`:

.. math::

    -2\\,l_R(\\rho) = (N - p) \\log(\\mathrm{rss}(\\rho)) + \\log|R(\\rho)|
    + \\log|\\tilde{X}^{\\top} \\tilde{X}(\\rho)| + \\mathrm{const}

with :math:`\\tilde{X}` / :math:`\\tilde{r}` the whitened design / residual,
:math:`\\hat{\\beta} = (\\tilde{X}^{\\top} \\tilde{X})^{-1} \\tilde{X}^{\\top}
\\tilde{y}` and :math:`\\sigma_e^2 = \\mathrm{rss} / (N - p)` profiled out.
Newton (damped, backtracked, autodiff gradient/Hessian) optimises the
unconstrained ``raw`` correlation parameter (one parameter for each of the
``'ar1'`` / ``'car1'`` / ``'cs'`` structures); every solve is the cuSOLVER-free
``small_inv_logdet`` on the :math:`(p, p)` whitened Gram.  Groups ride in a
left-packed, time-sorted :math:`(G, T)` padded layout so ragged sizes need no
dynamic shapes.
"""

from __future__ import annotations

import warnings
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
from ._blup import _solve_blup_system
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
    """Left-packed, time-sorted padded group layout.

    A :math:`(G, T)` grid describing how the :math:`N` observations distribute
    over ``G`` groups (``T`` the maximum group size).  The layout is shared
    across voxels and data-independent of the responses ``y``.

    Attributes
    ----------
    idx
        ``(G, T)`` original row index of each packed observation (pad entries
        map to ``0``).
    gaps
        ``(G, T)`` time gap to the previous in-group observation (``0`` at
        column ``0`` and on the pad).
    nsize
        ``(G,)`` real (unpadded) group sizes.
    mask
        ``(G, T)`` boolean validity mask (``True`` for real observations).
    """

    idx: Int[Array, 'G T']  # original row index (pad -> 0)
    gaps: Float[Array, 'G T']  # time gap to previous in-group obs (t>=1)
    nsize: Float[Array, 'G']  # real group sizes
    mask: Bool[Array, 'G T']  # validity mask


def build_group_layout(
    group: Int[Array, 'N'],
    time: Optional[Float[Array, 'N']] = None,
) -> GroupLayout:
    """Build the padded, time-sorted group layout from labels (and times).

    Observations are sorted by ``time`` within each group and left-packed into a
    :math:`(G, T)` grid (``T`` the maximum group size).  The computation is
    host-side (NumPy) because ``group`` / ``time`` are static across the mass
    (voxel) axis.

    With ``time=None`` the within-group **appearance order** is taken as the time
    order (consecutive rows one lag apart).  For the order-dependent ``'ar1'`` /
    ``'car1'`` structures this is a real assumption: pass ``time=`` when the rows
    are not already time-ordered, or the correlation is estimated over the wrong
    adjacencies (the fitting routines warn in this case).

    Parameters
    ----------
    group
        ``(N,)`` integer grouping factor assigning each observation to a group.
    time
        ``(N,)`` observation times, used to order and space observations within
        each group.  Defaults to the within-group appearance order (unit gaps,
        the discrete AR(1) case).

    Returns
    -------
    GroupLayout
        The left-packed, time-sorted :math:`(G, T)` layout: original row
        indices, previous-observation time gaps, real group sizes, and the
        validity mask.
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
        ``(V,)`` residual variance :math:`\\sigma_e^2`.
    rho
        ``(V,)`` natural correlation parameter :math:`\\rho` (the structure's
        bounded parameter; the lag-1 / decay / exchangeable correlation).  ``0``
        for the ``'iid'`` (no-correlation) structure.
    var_params
        :math:`(V, n_v)` estimated variance-function parameters
        (:math:`\\delta` for ``'varPower'``; the :math:`S - 1` log-ratios
        :math:`\\tau` for ``'varIdent'``).  Empty :math:`(V, 0)` when no
        ``weights`` variance function was given.
    log_lik
        ``(V,)`` profile REML log-likelihood at the fit.
    fixed_cov
        :math:`(V, p, p)` GLS covariance
        :math:`\\operatorname{Cov}(\\hat{\\beta}) = \\sigma_e^2
        (\\tilde{X}^{\\top} \\tilde{X})^{-1}` for a fixed-effect contrast on
        :math:`\\mathrm{df} = N - p`.
    df_resid
        Residual degrees of freedom :math:`N - p` (scalar; for a t / F
        contrast).
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
    """Whitened cross-products of the design and response.

    Whitens ``X`` and ``y`` jointly per group under the correlation structure and
    forms the whitened Grams :math:`\\tilde{X}^{\\top} \\tilde{X}`,
    :math:`\\tilde{X}^{\\top} \\tilde{y}`, :math:`\\tilde{y}^{\\top} \\tilde{y}`,
    together with the half-log-determinant Jacobian of the whitening.

    The Newton vector is ``raw = [corr_raw, var_raw]``; the variance function (if
    any) pre-scales the joint stack by :math:`1 / g` and contributes
    :math:`\\sum_i \\log g_i` to the half-log-determinant before the correlation
    whitener runs.

    Parameters
    ----------
    raw
        ``(k,)`` unconstrained Newton vector ``[corr_raw, var_raw]``: the first
        ``n_corr`` entries parameterise the correlation structure, the remainder
        the variance function.
    y_pad
        ``(G, T)`` left-packed, mask-zeroed response.
    x_pad
        ``(G, T, p)`` left-packed, mask-zeroed fixed-effect design.
    layout
        The :class:`GroupLayout` supplying gaps, group sizes, and the validity
        mask for the whitening recurrence.
    corr
        The correlation structure providing the per-group whitener.
    varfunc
        Optional residual variance function (heteroscedasticity), or ``None``.
    cov_pad
        ``(G, T)`` left-packed variance-function covariate; required (non-``None``)
        when ``varfunc`` is given, otherwise unused.
    n_corr
        Number of leading correlation parameters in ``raw``.

    Returns
    -------
    xtx
        :math:`(p, p)` whitened cross-product :math:`\\tilde{X}^{\\top}
        \\tilde{X}`.
    xty
        :math:`(p,)` whitened cross-product :math:`\\tilde{X}^{\\top}
        \\tilde{y}`.
    yty
        Scalar whitened cross-product :math:`\\tilde{y}^{\\top} \\tilde{y}`.
    half_logdet
        Scalar half-log-determinant Jacobian of the joint whitening (correlation
        plus variance-function contributions).
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
    """Single-voxel GLS-REML fit.

    Minimises the profile REML criterion in the unconstrained ``raw`` vector via
    the shared saddle-free damped Newton, then re-evaluates the whitened Grams at
    the optimum to recover the fixed effects, residual variance, natural
    correlation parameter, and reporting log-likelihood.

    Parameters
    ----------
    y_pad
        ``(G, T)`` left-packed, mask-zeroed response for this voxel.
    x_pad
        ``(G, T, p)`` left-packed, mask-zeroed fixed-effect design.
    layout
        The :class:`GroupLayout` for the whitening recurrence.
    corr
        The correlation structure.
    varfunc
        Optional residual variance function, or ``None``.
    cov_pad
        ``(G, T)`` left-packed variance-function covariate, or ``None``.
    n_corr
        Number of leading correlation parameters in the Newton vector.
    raw0
        ``(k,)`` initial unconstrained Newton vector.
    n
        Total number of observations :math:`N`.
    p
        Number of fixed-effect columns.
    spec
        Newton/variance-component specification supplying the ridge and
        Newton keyword arguments.

    Returns
    -------
    beta
        :math:`(p,)` fixed-effect estimates.
    sigma2
        Scalar residual variance :math:`\\sigma_e^2`.
    rho
        Scalar natural correlation parameter :math:`\\rho`.
    var_params
        ``(n_v,)`` estimated variance-function parameters (empty when no
        variance function was given).
    log_lik
        Scalar profile REML log-likelihood at the fit.
    fixed_cov
        :math:`(p, p)` GLS covariance of the fixed effects.
    """
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

    Fits, per voxel, :math:`y_v = X \\beta_v + \\varepsilon_v` with
    :math:`\\operatorname{Cov}(\\varepsilon_v) = \\sigma_e^2
    \\operatorname{diag}(g) R(\\rho) \\operatorname{diag}(g)` block-diagonal
    across ``group``, with :math:`R` the ``corr`` structure and :math:`g` the
    optional ``weights`` variance function.  ``X`` is shared across voxels; only
    :math:`y_v` varies.

    Parameters
    ----------
    Y
        ``(V, N)`` responses (one row per voxel).
    X
        ``(N, p)`` shared fixed-effect design.
    group
        ``(N,)`` integer grouping factor; the residual is correlated within
        groups and independent across them.
    corr
        The within-group correlation structure (a :class:`CorrSpec`).
    time
        ``(N,)`` observation times for the continuous-time structure and to
        order the discrete structure when the rows are not already in
        within-group time order.  Defaults to the within-group appearance order.
    weights
        Optional residual variance function (heteroscedasticity), or ``None``.
    n_iter
        Number of Newton iterations.
    damping
        Newton damping factor.
    block
        Optional voxel-block size for the batched vmap over ``Y``.

    Returns
    -------
    GLSResult
        The per-voxel fit: fixed effects, residual variance, correlation
        parameter, variance-function parameters, log-likelihood, GLS covariance,
        and residual degrees of freedom.
    """
    n, p = X.shape
    n_corr = corr.n_params
    if n_corr + (0 if weights is None else weights.n_params) == 0:
        raise ValueError(
            "fit_corr_gls: nothing to estimate -- 'iid' correlation with no "
            'weights variance function is ordinary least squares (use glm_fit).'
        )
    if time is None and corr.name in ('ar1', 'car1'):
        # MC6: without `time`, build_group_layout pairs *consecutive rows* within
        # each group (appearance order). For the order-dependent ar1 / car1
        # structures that silently estimates the correlation over the wrong
        # adjacencies if the rows are not already in within-group time order.
        warnings.warn(
            f"fit_corr_gls: corr='{corr.name}' with time=None assumes each "
            "group's rows are already in within-group time order (consecutive "
            'rows one lag apart). Pass time= if they are not.',
            stacklevel=2,
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
            y_pad,
            x_pad,
            layout,
            corr,
            weights,
            cov_pad,
            n_corr,
            raw0,
            n,
            p,
            spec,
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
    """Voxelwise generalised least squares with a structured residual.

    Fits, per voxel, :math:`y_v = X \\beta_v + \\varepsilon_v` with a
    within-group structured residual
    :math:`\\operatorname{Cov}(\\varepsilon_v) = \\sigma_e^2
    \\operatorname{diag}(g) R(\\rho) \\operatorname{diag}(g)` -- ``nlme``'s
    ``gls(correlation=corAR1 / corCAR1 / corCompSymm, weights=varPower /
    varIdent)``.  :math:`R` is the correlation structure; :math:`g` the optional
    variance function (heteroscedasticity).  There is no random effect here; for
    a random effect *plus* a structured residual, use :func:`fit_corr_lme`.

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
        ``(N,)`` observation times for ``'car1'`` (and to order ``'ar1'`` when
        the rows are not already in within-group time order).  Defaults to the
        within-group appearance order (unit gaps).
    weights
        Optional residual **variance function** (heteroscedasticity):
        :func:`var_power` (:math:`\\operatorname{Var} \\sim |v|^{2\\delta}`) or
        :func:`var_ident` (a separate variance per stratum).  Composes with any
        ``corr``; with ``corr='iid'`` it is a pure heteroscedastic GLS.
    n_iter, damping, block
        Newton iterations, damping factor, and the optional voxel-block size.

    Returns
    -------
    GLSResult
        The per-voxel fit: ``beta_hat``, ``sigma_e_sq``, ``rho``, ``var_params``,
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
    children=('beta_hat', 'cov_re', 'sigma_e_sq', 'rho', 'log_lik', 'blups'),
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
        ``(V, r, r)`` random-effect covariance :math:`G`.
    sigma_e_sq
        ``(V,)`` residual *scale* :math:`\\sigma_e^2` (the residual covariance
        is :math:`\\sigma_e^2 R(\\rho)`).
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
    blups: Optional[Float[Array, 'V q r']] = None
    """``(V, q, r)`` per-group random-effect modes on the whitened residual, or
    ``None`` when the fit did not retain them (the default ``retain_blups=False``).
    Read via :func:`~nitrix.stats.ranef`; used by
    :func:`~nitrix.stats.lme_predict` at ``level='conditional'``."""

    @property
    def re_labels(self) -> Tuple[str, ...]:
        """Names of the ``r`` within-factor random-effect dimensions of
        :attr:`cov_re` (D2)."""
        return tuple(f're{j}' for j in range(self.cov_re.shape[-1]))

    @property
    def coef(self) -> Float[Array, 'V p']:
        """Alias for :attr:`beta_hat` -- the fixed-effect coefficients, named
        ``coef`` for cross-suite parity with GLM / GAM / GP / HGP results (UX1)."""
        return self.beta_hat


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

    On whitened data the residual is i.i.d., so :math:`\\Sigma_e = \\sigma_e^2 R`
    reduces to :math:`\\sigma_e^2 I` and the per-group
    (:math:`Z^{\\top} Z`, :math:`X^{\\top} Z`, :math:`Z^{\\top} y`) / total
    (:math:`X^{\\top} X`, :math:`X^{\\top} y`, :math:`y^{\\top} y`) Grams feed
    the block-Woodbury objective verbatim; the whitening adds
    :math:`\\mathrm{half\\_logdet} = \\tfrac{1}{2} \\sum_i \\log|R_i|` to the
    REML objective.

    Parameters
    ----------
    raw
        ``(k,)`` unconstrained correlation parameters.
    x_pad
        ``(G, T, p)`` left-packed, mask-zeroed fixed-effect design.
    z_pad
        ``(G, T, r)`` left-packed, mask-zeroed random-effect design.
    y_pad
        ``(G, T)`` left-packed, mask-zeroed response.
    layout
        The :class:`GroupLayout` for the whitening recurrence.
    corr
        The correlation structure providing the per-group whitener.
    p
        Number of fixed-effect columns.
    r
        Number of random-effect columns.

    Returns
    -------
    ztz
        :math:`(G, r, r)` per-group whitened :math:`Z^{\\top} Z` blocks.
    xtz
        :math:`(G, p, r)` per-group whitened :math:`X^{\\top} Z` blocks.
    xtx
        :math:`(p, p)` total whitened :math:`X^{\\top} X`.
    zty
        :math:`(G, r)` per-group whitened :math:`Z^{\\top} y`.
    xty
        :math:`(p,)` total whitened :math:`X^{\\top} y`.
    yty
        Scalar total whitened :math:`y^{\\top} y`.
    half_logdet
        Scalar half-log-determinant Jacobian of the whitening.
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
    """Single-voxel mixed-model fit with a structured residual.

    Minimises the joint REML objective over the parameter vector
    :math:`[\\operatorname{chol}(G),\\ \\log \\sigma_e^2,\\ \\mathrm{corr\\_raw}]`
    via the shared saddle-free damped Newton, then re-evaluates at the optimum
    to recover the fixed effects and log-likelihood.

    The joint objective is non-convex away from the optimum (an indefinite
    Hessian at the start), so the saddle-free step rule is load-bearing here.

    Parameters
    ----------
    y_pad
        ``(G, T)`` left-packed, mask-zeroed response for this voxel.
    x_pad
        ``(G, T, p)`` left-packed, mask-zeroed fixed-effect design.
    z_pad
        ``(G, T, r)`` left-packed, mask-zeroed random-effect design.
    layout
        The :class:`GroupLayout` for the whitening recurrence.
    corr
        The correlation structure.
    theta_init
        ``(nt,)`` initial parameter vector
        :math:`[\\operatorname{chol}(G),\\ \\log \\sigma_e^2,\\
        \\mathrm{corr\\_raw}]`.
    n
        Total number of observations :math:`N`.
    p
        Number of fixed-effect columns.
    r
        Number of random-effect columns.
    nt_g
        Number of Cholesky parameters of :math:`G` (the leading block of
        ``theta``).
    n_minus_mr
        Residual degrees-of-freedom term :math:`N - G r` used by the REML
        objective.
    diagonal
        Whether the random-effect covariance :math:`G` is constrained diagonal.
    spec
        Newton/variance-component specification supplying the ridge and Newton
        keyword arguments.

    Returns
    -------
    theta
        ``(nt,)`` converged parameter vector.
    beta
        :math:`(p,)` fixed-effect estimates at the optimum.
    log_lik
        Scalar REML log-likelihood at the fit.
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


def _blups_corr(
    Y: Float[Array, 'V N'],
    X: Float[Array, 'N p'],
    z_pad: Float[Array, 'G T r'],
    idx: Int[Array, 'G T'],
    mask_f: Float[Array, 'G T'],
    layout: GroupLayout,
    corr: CorrSpec,
    beta_hat: Float[Array, 'V p'],
    cov_re: Float[Array, 'V r r'],
    sigma_e_sq: Float[Array, 'V'],
    raw_hat: Float[Array, 'V k'],
    r: int,
) -> Float[Array, 'V q r']:
    """Per-group random-effect modes (BLUPs) for the mixed-model + corr tier.

    The standard mixed-model-equation BLUP on **whitened**
    :math:`(Z_g, r_g)`: with the per-voxel converged :math:`\\rho` the group
    whitener :math:`W_g` gives :math:`Z_g^{\\top} R_g^{-1} Z_g = (W_g Z_g)^{\\top}
    (W_g Z_g)` and :math:`Z_g^{\\top} R_g^{-1} r_g = (W_g Z_g)^{\\top} (W_g r_g)`,
    so the Grams are formed on the whitened padded stack (einsum over the
    within-group time axis) and fed the shared :math:`r \\times r` solve.
    Whitening uses the per-voxel :math:`\\rho`, so -- unlike the homoscedastic
    residual-only paths -- the :math:`Z^{\\top} Z` Gram is *not* shared across
    voxels; the whole pass is mapped over voxels.

    Parameters
    ----------
    Y
        ``(V, N)`` responses (one row per voxel).
    X
        ``(N, p)`` shared fixed-effect design.
    z_pad
        ``(G, T, r)`` left-packed, mask-zeroed random-effect design.
    idx
        ``(G, T)`` original row indices from the group layout.
    mask_f
        ``(G, T)`` validity mask cast to the working float dtype.
    layout
        The :class:`GroupLayout` for the whitening recurrence.
    corr
        The correlation structure providing the per-group whitener.
    beta_hat
        ``(V, p)`` per-voxel fixed-effect estimates.
    cov_re
        ``(V, r, r)`` per-voxel random-effect covariance :math:`G`.
    sigma_e_sq
        ``(V,)`` per-voxel residual scale :math:`\\sigma_e^2`.
    raw_hat
        ``(V, k)`` per-voxel converged unconstrained correlation parameters.
    r
        Number of random-effect columns.

    Returns
    -------
    Float[Array, 'V q r']
        Per-group random-effect modes, where ``q`` is the number of groups.
    """

    def _one(
        y: Float[Array, 'N'],
        beta_v: Float[Array, 'p'],
        g_v: Float[Array, 'r r'],
        se_v: Float[Array, ''],
        raw_v: Float[Array, 'k'],
    ) -> Float[Array, 'q r']:
        resid_pad = (y - X @ beta_v)[idx] * mask_f  # (G, T)
        stack = jnp.concatenate(
            [z_pad, resid_pad[..., None]], axis=-1
        )  # (G, T, r+1)
        w, _ = corr.whiten(
            stack, layout.gaps, layout.nsize, layout.mask, raw_v
        )
        wz = w[..., :r]  # (G, T, r)
        wr = w[..., -1]  # (G, T)
        ztz = jnp.einsum('gtr,gts->grs', wz, wz)  # (G, r, r)
        ztr = jnp.einsum('gtr,gt->gr', wz, wr)  # (G, r)
        g_inv = small_inv_logdet(g_v, r)[0]  # (r, r)
        inv_s = 1.0 / se_v
        a = ztz * inv_s + g_inv[None]  # (G, r, r)
        rhs = ztr * inv_s  # (G, r)
        return _solve_blup_system(a, rhs, r)  # (G, r)

    return jax.vmap(_one)(Y, beta_hat, cov_re, sigma_e_sq, raw_hat)


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
    retain_blups: bool = False,
) -> CorrLMEResult:
    """Voxelwise mixed model with a structured within-group residual.

    Fits, per voxel, :math:`y_v = X \\beta_v + Z b_v + \\varepsilon_v` with
    :math:`b_v \\sim N(0, G)` and
    :math:`\\operatorname{Cov}(\\varepsilon_v) = \\sigma_e^2 R(\\rho)`
    block-diagonal across ``group`` -- the
    ``nlme::lme(random=~Z|g, correlation=corAR1/…)`` model.  Whitening by
    :math:`R` reduces each group to a standard block-Woodbury, so the per-group
    :math:`r \\times r` Woodbury algebra is reused verbatim with :math:`\\rho`
    joining the REML ``theta``-vector; cuSOLVER-free.

    Parameters
    ----------
    Y
        ``(V, N)`` responses (one row per voxel).
    X
        ``(N, p)`` shared fixed-effect design.
    Z
        ``(N, r)`` shared random-effect design.
    group
        ``(N,)`` integer grouping factor; the random effect and the residual
        correlation act within groups.
    corr
        The within-group residual correlation structure (a :class:`CorrSpec`).
    time
        ``(N,)`` observation times for the continuous-time structure and to
        order the discrete structure when the rows are not already in
        within-group time order.  Defaults to the within-group appearance order.
    diagonal
        Whether to constrain the random-effect covariance :math:`G` to be
        diagonal.
    n_iter
        Number of Newton iterations.
    damping
        Newton damping factor.
    block
        Optional voxel-block size for the batched vmap over ``Y``.
    retain_blups
        Whether to compute and retain the per-group random-effect modes (BLUPs)
        in the result.  Off by default.

    Returns
    -------
    CorrLMEResult
        The per-voxel fit: fixed effects, random-effect covariance, residual
        scale, correlation parameter, log-likelihood, and (optionally) the
        retained BLUPs.
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
    blups = (
        _blups_corr(
            Y,
            X,
            z_pad,
            idx,
            mask_f,
            layout,
            corr,
            beta_hat,
            cov_re,
            sigma_e_sq,
            theta_hat[:, nt_g + 1 :],  # the per-voxel corr raw params
            r,
        )
        if retain_blups
        else None
    )
    return CorrLMEResult(
        beta_hat=beta_hat,
        cov_re=cov_re,
        sigma_e_sq=sigma_e_sq,
        rho=rho,
        log_lik=log_lik,
        corr=corr.name,
        tier='R2+corr',
        blups=blups,
    )
