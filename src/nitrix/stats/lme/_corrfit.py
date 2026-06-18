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

from typing import NamedTuple, Optional, Tuple, Union, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jaxtyping import Array, Bool, Float, Int

from .._batching import blocked_vmap
from .._smalllinalg import small_inv_logdet
from ._corr import CorrSpec, resolve_corr
from ._varcomp import VarCompSpec

__all__ = ['GLSResult', 'build_group_layout', 'fit_corr_gls', 'gls_fit']


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
    order = np.lexsort(keys)
    g_sorted = g_np[order]
    sizes = np.bincount(g_sorted, minlength=n_groups)
    t_max = int(sizes.max()) if n_groups else 0

    idx = np.zeros((n_groups, t_max), dtype=np.int64)
    gaps = np.zeros((n_groups, t_max), dtype=np.float64)
    mask = np.zeros((n_groups, t_max), dtype=bool)
    pos = np.zeros(n_groups, dtype=np.int64)
    for orig in order:
        gid = g_np[orig]
        p = pos[gid]
        idx[gid, p] = orig
        mask[gid, p] = True
        if p >= 1:
            prev = idx[gid, p - 1]
            gaps[gid, p] = (
                float(t_np[orig] - t_np[prev]) if time is not None else 1.0
            )
        pos[gid] += 1

    return GroupLayout(
        idx=jnp.asarray(idx),
        gaps=jnp.asarray(gaps),
        nsize=jnp.asarray(sizes.astype(np.float64)),
        mask=jnp.asarray(mask),
    )


class GLSResult(NamedTuple):
    """Per-voxel GLS fit with a structured residual.

    Attributes
    ----------
    beta_hat
        ``(V, p)`` fixed-effect estimates.
    sigma_e_sq
        ``(V,)`` residual variance ``sigma_e^2``.
    rho
        ``(V,)`` natural correlation parameter ``rho`` (the structure's bounded
        parameter; the lag-1 / decay / exchangeable correlation).
    log_lik
        ``(V,)`` profile REML log-likelihood at the fit.
    fixed_cov
        ``(V, p, p)`` ``Cov(beta_hat) = sigma_e^2 (X~^T X~)^{-1}`` (the GLS
        covariance) for a fixed-effect contrast on ``df = N - p``.
    df_resid
        Residual degrees of freedom ``N - p`` (scalar; for a t / F contrast).
    corr
        The correlation structure name.
    """

    beta_hat: Float[Array, 'V p']
    sigma_e_sq: Float[Array, 'V']
    rho: Float[Array, 'V']
    log_lik: Float[Array, 'V']
    fixed_cov: Float[Array, 'V p p']
    df_resid: int
    corr: str


def _whitened_grams(
    raw: Float[Array, 'k'],
    y_pad: Float[Array, 'G T'],
    x_pad: Float[Array, 'G T p'],
    layout: GroupLayout,
    corr: CorrSpec,
) -> Tuple[
    Float[Array, 'p p'], Float[Array, 'p'], Float[Array, ''], Float[Array, '']
]:
    """Whitened cross-products ``(X~^T X~, X~^T y~, y~^T y~, half_logdet)``."""
    # Whiten y and X jointly (stack as the channel axis) so the recurrence runs
    # once over the shared (G, T) structure.
    stack = jnp.concatenate([x_pad, y_pad[..., None]], axis=-1)  # (G, T, p+1)
    w, half_logdet = corr.whiten(
        stack, layout.gaps, layout.nsize, layout.mask, raw
    )
    wx = w[..., :-1]  # (G, T, p)
    wy = w[..., -1]  # (G, T)
    xtx = jnp.einsum('gtp,gtq->pq', wx, wx)
    xty = jnp.einsum('gtp,gt->p', wx, wy)
    yty = jnp.sum(wy * wy)
    return xtx, xty, yty, half_logdet


def _fit_one(
    y_pad: Float[Array, 'G T'],
    x_pad: Float[Array, 'G T p'],
    layout: GroupLayout,
    corr: CorrSpec,
    raw0: Float[Array, 'k'],
    n: int,
    p: int,
    spec: VarCompSpec,
) -> Tuple[
    Float[Array, 'p'],
    Float[Array, ''],
    Float[Array, ''],
    Float[Array, ''],
    Float[Array, 'p p'],
]:
    """Single-voxel GLS-REML fit: damped Newton on the unconstrained ``raw``."""
    dof = float(n - p)
    ridge = spec.ridge

    def neg2_reml(raw: Float[Array, 'k']) -> Float[Array, '']:
        xtx, xty, yty, half_logdet = _whitened_grams(
            raw, y_pad, x_pad, layout, corr
        )
        xtx_r = xtx + ridge * jnp.eye(p, dtype=xtx.dtype)
        xtx_inv, logdet_xtx = small_inv_logdet(xtx_r, p)
        beta = xtx_inv @ xty
        rss = jnp.clip(yty - beta @ xty, 1e-30, None)
        return dof * jnp.log(rss) + 2.0 * half_logdet + logdet_xtx

    grad_fn = jax.grad(neg2_reml)
    hess_fn = jax.hessian(neg2_reml)
    k = raw0.shape[0]

    def newton(raw: Float[Array, 'k'], _: Array) -> Tuple[Array, None]:
        g = grad_fn(raw)
        h = hess_fn(raw) + spec.damping * jnp.eye(k, dtype=raw.dtype)
        h_inv, _ = small_inv_logdet(h, k)
        delta = jnp.clip(h_inv @ g, -spec.max_step, spec.max_step)
        f0 = neg2_reml(raw)

        def bt(
            _: Array, carry: Tuple[Array, Array, Array]
        ) -> Tuple[Array, Array, Array]:
            scale, best, best_f = carry
            trial = raw - scale * delta
            trial_f = neg2_reml(trial)
            ok = trial_f < best_f
            return (
                scale * 0.5,
                jnp.where(ok, trial, best),
                jnp.where(ok, trial_f, best_f),
            )

        init = (jnp.asarray(1.0, raw.dtype), raw, f0)
        _, raw_new, _ = lax.fori_loop(0, spec.n_backtrack, bt, init)
        return raw_new, None

    raw, _ = lax.scan(newton, raw0, xs=None, length=spec.n_iter)

    xtx, xty, yty, half_logdet = _whitened_grams(
        raw, y_pad, x_pad, layout, corr
    )
    xtx_r = xtx + ridge * jnp.eye(p, dtype=xtx.dtype)
    xtx_inv, logdet_xtx = small_inv_logdet(xtx_r, p)
    beta = xtx_inv @ xty
    rss = jnp.clip(yty - beta @ xty, 1e-30, None)
    sigma2 = rss / dof
    fixed_cov = sigma2 * xtx_inv
    rho = corr.to_natural(raw)
    # profile REML log-likelihood (full, with constants) for reporting.
    neg2 = (
        dof * jnp.log(sigma2)
        + 2.0 * half_logdet
        + logdet_xtx
        + dof
        + dof * jnp.log(2.0 * jnp.pi)
    )
    log_lik = -0.5 * neg2
    return beta, sigma2, rho, log_lik, fixed_cov


def fit_corr_gls(
    Y: Float[Array, 'V N'],
    X: Float[Array, 'N p'],
    group: Int[Array, 'N'],
    corr: CorrSpec,
    *,
    time: Optional[Float[Array, 'N']] = None,
    n_iter: int = 30,
    damping: float = 1e-6,
    block: Optional[int] = None,
) -> GLSResult:
    """Voxelwise GLS-REML with a structured within-group residual.

    Fits ``y_v = X beta_v + eps_v``, ``Cov(eps_v) = sigma_e^2 R(rho)``
    block-diagonal across ``group``, with ``R`` the ``corr`` structure.  ``X`` is
    shared across voxels; only ``y_v`` varies.  Returns a :class:`GLSResult`.
    """
    n, p = X.shape
    layout = build_group_layout(group, time)
    idx = layout.idx
    mask_f = layout.mask.astype(X.dtype)
    x_pad = X[idx] * mask_f[..., None]  # (G, T, p) shared
    raw0 = corr.init_raw(X.dtype)
    spec = VarCompSpec.reml(n_iter=n_iter, damping=damping)

    def per_voxel(
        y: Float[Array, 'N'],
    ) -> Tuple[Array, Array, Array, Array, Array]:
        y_pad = y[idx] * mask_f
        return _fit_one(y_pad, x_pad, layout, corr, raw0, n, p, spec)

    beta, sigma2, rho, log_lik, fixed_cov = cast(
        Tuple[Array, Array, Array, Array, Array],
        blocked_vmap(per_voxel, (Y,), block=block),
    )
    return GLSResult(
        beta_hat=beta,
        sigma_e_sq=sigma2,
        rho=rho,
        log_lik=log_lik,
        fixed_cov=fixed_cov,
        df_resid=int(n - p),
        corr=corr.name,
    )


def gls_fit(
    Y: Float[Array, 'V N'],
    X: Float[Array, 'N p'],
    *,
    group: Int[Array, 'N'],
    corr: Union[str, CorrSpec],
    time: Optional[Float[Array, 'N']] = None,
    n_iter: int = 30,
    damping: float = 1e-6,
    block: Optional[int] = None,
) -> GLSResult:
    """Voxelwise generalised least squares with a structured residual (§1.4).

    Fits, per voxel, ``y_v = X beta_v + eps_v`` with a within-group correlated
    residual ``Cov(eps_v) = sigma_e^2 R(rho)`` -- ``nlme``'s
    ``gls(correlation=corAR1 / corCAR1 / corCompSymm)``.  No random effect (the
    R0 + corr path); for a random effect *plus* a structured residual, the
    composition with the R2 block-Woodbury is the follow-up.

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
        (continuous-time AR(1); pass ``time``), ``'cs'`` (compound symmetry), or
        a :class:`CorrSpec`.
    time
        ``(N,)`` observation times for ``car1`` (and to order ``ar1`` when the
        rows are not already in within-group time order).  Defaults to the
        within-group appearance order (unit gaps).
    n_iter, damping, block
        Newton iterations, AI/LM damping, and the optional voxel-block size.

    Returns
    -------
    ``GLSResult`` -- ``beta_hat``, ``sigma_e_sq``, ``rho``, ``log_lik``, the GLS
    ``fixed_cov``, and ``df_resid = N - p``.
    """
    return fit_corr_gls(
        Y,
        X,
        jnp.asarray(group),
        resolve_corr(corr),
        time=None if time is None else jnp.asarray(time),
        n_iter=n_iter,
        damping=damping,
        block=block,
    )
