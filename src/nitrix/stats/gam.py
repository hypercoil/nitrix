# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Mass-univariate generalised additive (mixed) models.

``gam_fit`` fits, per element (voxel / vertex / fixel), a GAM::

    g(E[y]) = X_parametric beta + sum_k f_k(x_k),   f_k = B_k(x_k) gamma_k

with each smooth ``f_k`` a penalised spline (``stats.basis``) carrying a
roughness penalty ``lambda_k gamma_k^T S_k gamma_k``.  This is ModelArray's
``gam`` / ``mgcv``-style fit; a **GAMM** adds explicit random-effect blocks,
which enter as just more penalty components (a random effect is a ridge
penalty), so the same machinery covers both.

Two nested loops, one per element
---------------------------------

- **Inner** (fixed ``lambda``): penalised IRLS -- the same cuSOLVER-free
  weighted normal-equations solve as ``glm``, with the block penalty
  ``S(lambda) = sum_k lambda_k S_k`` added.  OLS / WLS / exponential family all
  reduce to it.
- **Outer** (select ``lambda``): the **generalized Fellner-Schall** update
  (Wood & Fasiolo 2017) -- a multiplicative, positivity-preserving generalized-
  REML step ``lambda_k <- lambda_k (tr(S_lambda^- S_k) - tr(V S_k)) / (gamma_k^T
  S_k gamma_k / phi)`` that increases the (Laplace) marginal likelihood each
  iteration.  Because GAM smooths occupy disjoint coefficient blocks,
  ``tr(S_lambda^- S_k) = rank(S_k) / lambda_k`` -- no generalized inverse of the
  summed penalty is needed.  This is the operational form of the penalty <->
  variance-component REML equivalence (the GAM smoothing parameter is the
  ratio ``phi / sigma_b^2`` of a mixed model).

Both loops run a fixed number of iterations (``vmap``-clean over elements) and
every solve is cuSOLVER-free (``linalg._smalllinalg``), so the whole fit runs on
the broken-cuSOLVER GPU.

Outputs (ModelArray ``gam`` parity)
-----------------------------------

``GAMResult`` carries per-element coefficients, selected ``lambda``, per-smooth
**effective degrees of freedom** (``edf_k = tr`` of the smooth's influence
block) and total EDF, dispersion, deviance, and the Bayesian coefficient
covariance ``V = (X^T W X + S_lambda)^{-1}`` (for smooth-term confidence bands
and the approximate F / chi-square tests).  Partial effects are rendered with
``smooth_partial_effect``.

References
----------
- Wood, S. N. & Fasiolo, M. (2017). A generalized Fellner-Schall method for
  smoothing parameter optimization.  Biometrics 73, 1071-1081.
- Wood, S. N. (2017). Generalized Additive Models, 2nd ed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Any,
    Literal,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.scipy.special import betainc, gammaincc
from jaxtyping import Array, Float

from ..linalg._smalllinalg import small_inv_logdet, spd_chol, sym_eig_jacobi
from ._batching import blocked_vmap
from ._family import GAUSSIAN, Family, resolve_family
from ._irls import fit_penalised_irls
from ._result import register_result
from .basis import SmoothBasis

__all__ = [
    'GAMResult',
    'SmoothTest',
    'gam_fit',
    'smooth_partial_effect',
    'smooth_significance',
]

# A pooled penalty eigenvalue at or below this is the unpenalised null space
# (exact zero after the basis's eigenvalue floor), excluded from the
# smoothing-parameter trace ``tr(S_lambda^+ S_k)``.
_EIG_EPS = 0.0

# Any object conforming to the :class:`SmoothBasis` Protocol (audit D8): the
# built-in SplineBasis / TensorBasis / REBasis, or a user basis implementing
# ``design`` / ``dim`` / ``penalty_blocks()`` / ``eval_design()``.
Smooth = SmoothBasis


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@register_result(
    children=(
        'coef',
        'lam',
        'edf',
        'edf_total',
        'dispersion',
        'deviance',
        'null_deviance',
        'cov_unscaled',
    ),
    aux=('family', 'n_obs', 'col_slices'),
)
@dataclass(frozen=True)
class GAMResult:
    """Per-element GAM fit output.

    Attributes
    ----------
    coef
        ``(V, p)`` coefficients over the assembled design
        ``[intercept | parametric | smooth_1 | ... ]``.
    lam
        ``(V, m)`` selected smoothing parameters (one per smooth).
    edf
        ``(V, m)`` per-smooth effective degrees of freedom.
    edf_total
        ``(V,)`` total effective degrees of freedom (incl. parametric).
    dispersion
        ``(V,)`` scale estimate (residual variance for Gaussian; ``1`` for
        fixed-dispersion families).
    deviance, null_deviance
        ``(V,)`` model and intercept-only deviance.
    cov_unscaled
        ``(V, p, p)`` Bayesian covariance ``(X^T W X + S_lambda)^{-1}``.
    """

    coef: Float[Array, 'V p']
    lam: Float[Array, 'V m']
    edf: Float[Array, 'V m']
    edf_total: Float[Array, 'V']
    dispersion: Float[Array, 'V']
    deviance: Float[Array, 'V']
    null_deviance: Float[Array, 'V']
    cov_unscaled: Float[Array, 'V p p']
    family: Family
    n_obs: int
    col_slices: Tuple[Tuple[int, int], ...]


# ---------------------------------------------------------------------------
# Design + penalty assembly (data-independent: shared across elements)
# ---------------------------------------------------------------------------


def _assemble(
    n: int,
    smooths: Sequence[Smooth],
    parametric: Optional[Float[Array, 'N q']],
    intercept: bool,
    dtype: Any,
) -> Tuple[
    Float[Array, 'N p'],
    Float[Array, 'K p p'],
    Float[Array, 'K p'],
    Tuple[Tuple[int, int], ...],
]:
    """Build the full design ``X``, the stacked full-size penalties ``S_k``,
    their block eigenvalues ``pen_eig`` (for the Fellner-Schall trace), and the
    per-smooth column slices.  A smooth may carry **multiple** penalties (a
    tensor product has one per margin); ``K >= len(smooths)``."""
    blocks = []
    if intercept:
        blocks.append(jnp.ones((n, 1), dtype=dtype))
    if parametric is not None:
        blocks.append(jnp.asarray(parametric, dtype=dtype))
    smooth_start = sum(b.shape[1] for b in blocks)
    slices = []
    col = smooth_start
    pen_blocks = []  # (lo, hi, S_block, eig_block) per penalty
    for sm in smooths:
        blocks.append(jnp.asarray(sm.design, dtype=dtype))
        kb = sm.dim
        slices.append((col, col + kb))
        for s_block, eig_block in sm.penalty_blocks():
            pen_blocks.append((col, col + kb, s_block, eig_block))
        col += kb
    X = jnp.concatenate(blocks, axis=1)
    p = X.shape[1]

    big_k = len(pen_blocks)
    pen_full = np.zeros((big_k, p, p))
    pen_eig = np.zeros((big_k, p))
    for k, (lo, hi, s_block, eig_block) in enumerate(pen_blocks):
        pen_full[k, lo:hi, lo:hi] = s_block
        pen_eig[k, lo:hi] = eig_block
    return (
        X,
        jnp.asarray(pen_full, dtype=dtype),
        jnp.asarray(pen_eig, dtype=dtype),
        tuple(slices),
    )


# ---------------------------------------------------------------------------
# Inner penalised IRLS (returns the pieces the Fellner-Schall step needs)
# ---------------------------------------------------------------------------


def _penalised_irls(
    y: Float[Array, 'N'],
    X: Float[Array, 'N p'],
    s_lambda: Float[Array, 'p p'],
    family: Family,
    p: int,
    n_iter: int,
    ridge: float,
    beta0: Float[Array, 'p'],
) -> Tuple[Float[Array, 'p'], Float[Array, 'p p'], Float[Array, 'p p']]:
    """Penalised IRLS from a warm start, via the shared core.  Returns
    ``(beta, V, xtwx)`` -- the coefficients, ``V = (X^T W X + S_lambda +
    ridge)^{-1}``, and the unpenalised Gram ``X^T W X`` (for the EDF / FS
    traces), all at the converged ``beta``."""
    beta, v, xtwx, _ = fit_penalised_irls(
        y, X, family, penalty=s_lambda, beta0=beta0, n_iter=n_iter, ridge=ridge
    )
    return beta, v, xtwx


def _trace_slinv_sk(
    ek: Float[Array, 'p'], s_lambda_eig: Float[Array, 'p']
) -> Float[Array, '']:
    """``tr(S_lambda^+ S_k)`` from the precomputed block eigenvalues.

    With every penalty diagonal in its block's joint eigenbasis, the trace is an
    elementwise sum ``sum_i [s_lambda_eig_i > 0] eig_k_i / s_lambda_eig_i`` --
    basis-invariant, so it is exact even though the *fit* is carried in the
    original (non-rotated) basis.  For a lone disjoint penalty this reduces to
    ``rank_k / lambda_k`` (the old shortcut); for overlapping tensor-product
    penalties it is the correct general trace, no pseudo-inverse needed.
    """
    safe = jnp.where(s_lambda_eig > _EIG_EPS, s_lambda_eig, 1.0)
    return jnp.sum(jnp.where(s_lambda_eig > _EIG_EPS, ek / safe, 0.0))


# ---------------------------------------------------------------------------
# Shared-lambda fit (Gaussian): pooled Fellner-Schall on sufficient statistics
# ---------------------------------------------------------------------------


def _gam_fit_shared_gaussian(
    Y: Float[Array, 'V N'],
    X: Float[Array, 'N p'],
    penalties: Float[Array, 'm p p'],
    pen_eig: Float[Array, 'm p'],
    n_outer: int,
    ridge: float,
    lam_floor: float,
    lam_ceil: float,
) -> Tuple[
    Float[Array, 'V p'],
    Float[Array, 'V m'],
    Float[Array, 'V p p'],
    Float[Array, 'V p p'],
    Float[Array, 'V'],
]:
    """One smoothing parameter shared across all ``V`` elements (Gaussian).

    For the Gaussian identity link the influence ``V = (X^T X + S_lambda)^{-1}``
    is *shared* (no ``y`` dependence), so the **pooled** Fellner-Schall update
    is a function only of the ``(p, p)`` sufficient statistics ``X^T X`` and
    ``C = (Y X)^T (Y X) = sum_v (X^T y_v)(X^T y_v)^T`` and the scalar
    ``tr(sum_v y_v y_v^T)``.  The outer loop is therefore ``O(n_outer p^3)`` --
    **independent of ``V``** -- removing the per-element outer loop entirely;
    only the final coefficient fit and the sufficient statistics touch ``V``.

    Returns the same ``(beta, lam, V, xtwx, dispersion)`` per-element tuple as
    the per-element path (``V`` / ``xtwx`` broadcast from the shared ``(p, p)``),
    so the result assembly is identical.
    """
    v_count, n = Y.shape
    m = penalties.shape[0]
    p = X.shape[1]
    xtx = X.T @ X
    yx = Y @ X  # (V, p): row v is (X^T y_v)^T
    c = yx.T @ yx  # (p, p): sum_v (X^T y_v)(X^T y_v)^T
    g_tr = jnp.sum(Y * Y)  # tr(sum_v y_v y_v^T)
    ridge_eye = ridge * jnp.eye(p, dtype=Y.dtype)

    def outer(
        lam: Float[Array, 'm'], _: Array
    ) -> Tuple[Float[Array, 'm'], None]:
        s_lambda = jnp.tensordot(lam, penalties, axes=(0, 0))
        vmat, _ = small_inv_logdet(xtx + s_lambda + ridge_eye, p)
        edf = jnp.trace(vmat @ xtx)
        pooled_rss = (
            g_tr - 2.0 * jnp.trace(vmat @ c) + jnp.trace(vmat @ xtx @ vmat @ c)
        )
        phi = pooled_rss / jnp.clip(v_count * (n - edf), 1e-3, None)
        s_lambda_eig = lam @ pen_eig  # (p,) pooled block eigenvalues

        def fs(k: Array) -> Float[Array, '']:
            sk = penalties[k]
            tr_vsk = jnp.sum(vmat * sk)  # tr(V S_k)
            energy_sum = jnp.trace(sk @ vmat @ c @ vmat)  # sum_v b_v^T S_k b_v
            tr_slinv_sk = _trace_slinv_sk(pen_eig[k], s_lambda_eig)
            num = jnp.clip(lam[k] * tr_slinv_sk - lam[k] * tr_vsk, 1e-8, None)
            den = jnp.clip(energy_sum / (v_count * phi), 1e-12, None)
            return jnp.clip(num / den, lam_floor, lam_ceil)

        return jax.vmap(fs)(jnp.arange(m)), None

    lam, _ = lax.scan(outer, jnp.ones((m,), Y.dtype), xs=None, length=n_outer)

    s_lambda = jnp.tensordot(lam, penalties, axes=(0, 0))
    vmat, _ = small_inv_logdet(xtx + s_lambda + ridge_eye, p)
    coef = yx @ vmat  # (V, p): beta_v = V (X^T y_v)
    resid = Y - coef @ X.T
    edf = jnp.trace(vmat @ xtx)
    phi_v = jnp.sum(resid * resid, axis=1) / jnp.clip(n - edf, 1e-3, None)

    v_bc = jnp.broadcast_to(vmat, (v_count, p, p))
    xtwx_bc = jnp.broadcast_to(xtx, (v_count, p, p))
    lam_bc = jnp.broadcast_to(lam, (v_count, m))
    return coef, lam_bc, v_bc, xtwx_bc, phi_v


# ---------------------------------------------------------------------------
# Per-element fit: Fellner-Schall outer loop over the inner penalised IRLS
# ---------------------------------------------------------------------------


def _gam_fit_one(
    y: Float[Array, 'N'],
    X: Float[Array, 'N p'],
    penalties: Float[Array, 'm p p'],
    pen_eig: Float[Array, 'm p'],
    family: Family,
    p: int,
    n_outer: int,
    n_inner: int,
    ridge: float,
    lam_floor: float,
    lam_ceil: float,
) -> Tuple[
    Float[Array, 'p'],
    Float[Array, 'm'],
    Float[Array, 'p p'],
    Float[Array, 'p p'],
    Float[Array, ''],
]:
    """Single-element GAM fit.  Returns ``(beta, lam, V, xtwx, dispersion)``."""
    m = penalties.shape[0]
    n = X.shape[0]

    def outer(
        carry: Tuple[Float[Array, 'm'], Float[Array, 'p']], _: Array
    ) -> Tuple[Tuple[Float[Array, 'm'], Float[Array, 'p']], None]:
        lam, beta = carry
        s_lambda = jnp.tensordot(lam, penalties, axes=(0, 0))  # (p, p)
        beta, v, xtwx = _penalised_irls(
            y, X, s_lambda, family, p, n_inner, ridge, beta
        )
        # Dispersion: Pearson/residual scale for Gaussian, fixed otherwise.
        if family.has_fixed_dispersion:
            phi = jnp.asarray(1.0, dtype=y.dtype)
        else:
            edf_tot = jnp.trace(v @ xtwx)
            resid = y - X @ beta
            phi = jnp.sum(resid * resid) / jnp.clip(n - edf_tot, 1e-3, None)

        # Generalized Fellner-Schall update per penalty.  The penalty trace
        # tr(S_lambda^+ S_k) is read off the precomputed block eigenvalues (it
        # equals rank_k/lambda_k for a disjoint penalty, the general elementwise
        # sum for overlapping tensor-product penalties).
        s_lambda_eig = lam @ pen_eig  # (p,) pooled block eigenvalues

        def fs(k: Array) -> Float[Array, '']:
            sk = penalties[k]
            tr_vsk = jnp.sum(v * sk)  # tr(V S_k)
            energy = beta @ (sk @ beta)
            tr_slinv_sk = _trace_slinv_sk(pen_eig[k], s_lambda_eig)
            # lambda_k * [tr(S_lambda^+ S_k) - tr(V S_k)] / [energy / phi].
            num = jnp.clip(lam[k] * tr_slinv_sk - lam[k] * tr_vsk, 1e-8, None)
            den = jnp.clip(energy / phi, 1e-12, None)
            return jnp.clip(num / den, lam_floor, lam_ceil)

        lam_new = jax.vmap(fs)(jnp.arange(m))
        return (lam_new, beta), None

    lam0 = jnp.ones((m,), dtype=y.dtype)
    beta_init = jnp.zeros((p,), dtype=y.dtype)
    (lam, beta), _ = lax.scan(
        outer, (lam0, beta_init), xs=None, length=n_outer
    )

    # Final fit at the selected lambda.
    s_lambda = jnp.tensordot(lam, penalties, axes=(0, 0))
    beta, v, xtwx = _penalised_irls(
        y, X, s_lambda, family, p, n_inner, ridge, beta
    )
    if family.has_fixed_dispersion:
        phi = jnp.asarray(1.0, dtype=y.dtype)
    else:
        edf_tot = jnp.trace(v @ xtwx)
        resid = y - X @ beta
        phi = jnp.sum(resid * resid) / jnp.clip(n - edf_tot, 1e-3, None)
    return beta, lam, v, xtwx, phi


# ---------------------------------------------------------------------------
# Per-element Gaussian fast path: the cross-product (sufficient-statistic) fit
# ---------------------------------------------------------------------------


def _gam_fit_one_gaussian_xprod(
    c: Float[Array, 'p'],
    g: Float[Array, ''],
    xtx: Float[Array, 'p p'],
    penalties: Float[Array, 'm p p'],
    pen_eig: Float[Array, 'm p'],
    n: int,
    p: int,
    n_outer: int,
    ridge: float,
    lam_floor: float,
    lam_ceil: float,
) -> Tuple[
    Float[Array, 'p'],
    Float[Array, 'm'],
    Float[Array, 'p p'],
    Float[Array, 'p p'],
    Float[Array, ''],
]:
    """Exact per-voxel Gaussian GAM fit from the cross-products ``c = X^T y_v``
    and ``g = y_v^T y_v``.

    For the Gaussian identity link the penalised IRLS converges in **one** step
    to ``beta = (X^T X + S_lambda)^{-1} c`` -- a function of ``y`` only through
    ``c`` -- and the dispersion (``phi = (g - 2 beta^T c + beta^T X^T X beta) /
    (N - edf)``), the EDF, and the Fellner-Schall traces all reduce to ``(c, g,
    X^T X)``.  So the whole per-voxel Fellner-Schall loop runs in ``p``-space
    with **no N-dimensional vector in the loop** -- ``N`` enters only the one-off
    cross-products ``X^T Y`` and ``diag(Y Y^T)`` -- and the result is identical
    (to floating point) to ``_gam_fit_one`` for the Gaussian family.  Returns the
    same ``(beta, lam, V, xtwx, dispersion)`` tuple; ``xtwx = X^T X`` is shared.
    """
    m = penalties.shape[0]
    ridge_eye = ridge * jnp.eye(p, dtype=xtx.dtype)

    def quantities(
        lam: Float[Array, 'm'],
    ) -> Tuple[Float[Array, 'p p'], Float[Array, 'p'], Float[Array, '']]:
        s_lambda = jnp.tensordot(lam, penalties, axes=(0, 0))
        v, _ = small_inv_logdet(xtx + s_lambda + ridge_eye, p)
        beta = v @ c
        edf = jnp.trace(v @ xtx)
        rss = g - 2.0 * (beta @ c) + beta @ (xtx @ beta)
        phi = rss / jnp.clip(n - edf, 1e-3, None)
        return v, beta, phi

    def outer(
        lam: Float[Array, 'm'], _: Array
    ) -> Tuple[Float[Array, 'm'], None]:
        v, beta, phi = quantities(lam)
        s_lambda_eig = lam @ pen_eig

        def fs(k: Array) -> Float[Array, '']:
            sk = penalties[k]
            tr_vsk = jnp.sum(v * sk)
            energy = beta @ (sk @ beta)
            tr_slinv_sk = _trace_slinv_sk(pen_eig[k], s_lambda_eig)
            num = jnp.clip(lam[k] * tr_slinv_sk - lam[k] * tr_vsk, 1e-8, None)
            den = jnp.clip(energy / phi, 1e-12, None)
            return jnp.clip(num / den, lam_floor, lam_ceil)

        return jax.vmap(fs)(jnp.arange(m)), None

    lam0 = jnp.ones((m,), dtype=xtx.dtype)
    lam, _ = lax.scan(outer, lam0, xs=None, length=n_outer)
    v, beta, phi = quantities(lam)
    return beta, lam, v, xtx, phi


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def gam_fit(
    Y: Float[Array, 'V N'],
    smooths: Sequence[Smooth],
    *,
    parametric: Optional[Float[Array, 'N q']] = None,
    intercept: bool = True,
    family: Union[str, Family] = GAUSSIAN,
    lambda_mode: Literal['per_element', 'shared'] = 'per_element',
    n_outer: int = 20,
    n_inner: int = 10,
    ridge: float = 1e-8,
    lam_floor: float = 1e-6,
    lam_ceil: float = 1e8,
    block: Optional[int] = None,
) -> GAMResult:
    """Fit a mass-univariate GAM: shared smooth bases, per-element responses.

    Parameters
    ----------
    Y
        ``(V, N)`` responses.
    smooths
        Penalised smooth bases (one per smooth term): a ``SplineBasis``
        (``bspline_basis`` / ``thinplate_regression_basis`` / ``cyclic_cubic_basis``)
        or a ``TensorBasis`` (``tensor_product_basis``) for an anisotropic
        interaction.  A tensor smooth carries one smoothing parameter per margin,
        all selected by the same Fellner-Schall loop.
    parametric
        Optional ``(N, q)`` unpenalised linear design (covariates entering
        linearly).  The intercept is added separately (see ``intercept``).
    intercept
        Prepend an intercept column (default ``True``).
    family
        Exponential family (default ``GAUSSIAN``).
    lambda_mode
        ``'per_element'`` (default) selects a smoothing parameter per element
        (ModelArray parity).  ``'shared'`` selects **one** smoothing parameter
        across all elements via a pooled Fellner-Schall update on sufficient
        statistics -- an ``O(n_outer p^3)``, ``V``-independent outer loop (much
        faster when smoothness is homogeneous across the brain).  Gaussian only.
    n_outer, n_inner
        Fellner-Schall outer iterations and penalised-IRLS inner iterations.
    ridge
        Small stabiliser on the penalised normal equations.
    lam_floor, lam_ceil
        Clamp on each smoothing parameter.
    block
        Optional element-block size bounding peak memory (the per-element
        ``(V, p, p)`` covariances) on brain-scale ``V``.  ``None`` (default) is
        a single ``vmap``.

    Returns
    -------
    ``GAMResult`` (coefficients, selected ``lambda``, per-smooth EDF, dispersion,
    deviance, Bayesian covariance).
    """
    family = resolve_family(family)
    n = X_n = Y.shape[-1]
    for sm in smooths:
        if sm.design.shape[0] != n:
            raise ValueError(
                f'gam_fit: smooth design has {sm.design.shape[0]} rows; '
                f'expected N={n}.'
            )
    if not smooths:
        raise ValueError('gam_fit: provide at least one smooth term.')

    X, penalties, pen_eig, slices = _assemble(
        X_n, smooths, parametric, intercept, Y.dtype
    )
    p = X.shape[1]

    if lambda_mode == 'shared':
        if family.name != 'gaussian':
            raise NotImplementedError(
                "lambda_mode='shared' is implemented for the Gaussian family "
                'only (the pooled sufficient statistics require a shared, '
                "y-independent influence matrix); use 'per_element' otherwise."
            )
        coef, lam, v, xtwx, phi = _gam_fit_shared_gaussian(
            Y, X, penalties, pen_eig, n_outer, ridge, lam_floor, lam_ceil
        )
    elif family.name == 'gaussian':
        # Exact cross-product fast path: the Gaussian fit depends on each y_v
        # only through c_v = X^T y_v and g_v = y_v^T y_v, so the per-voxel
        # Fellner-Schall loop runs in p-space with no N-dimensional vector in
        # the loop (N enters only the one-off cross-products).
        xtx = X.T @ X
        c_all = Y @ X  # (V, p)
        g_all = jnp.sum(Y * Y, axis=1)  # (V,)

        def per_voxel_g(
            c: Float[Array, 'p'], g: Float[Array, '']
        ) -> Tuple[
            Float[Array, 'p'],
            Float[Array, 'm'],
            Float[Array, 'p p'],
            Float[Array, 'p p'],
            Float[Array, ''],
        ]:
            return _gam_fit_one_gaussian_xprod(
                c,
                g,
                xtx,
                penalties,
                pen_eig,
                n,
                p,
                n_outer,
                ridge,
                lam_floor,
                lam_ceil,
            )

        coef, lam, v, xtwx, phi = blocked_vmap(
            per_voxel_g, (c_all, g_all), block=block
        )
    else:

        def per_voxel(
            y: Float[Array, 'N'],
        ) -> Tuple[
            Float[Array, 'p'],
            Float[Array, 'm'],
            Float[Array, 'p p'],
            Float[Array, 'p p'],
            Float[Array, ''],
        ]:
            return _gam_fit_one(
                y,
                X,
                penalties,
                pen_eig,
                family,
                p,
                n_outer,
                n_inner,
                ridge,
                lam_floor,
                lam_ceil,
            )

        coef, lam, v, xtwx, phi = blocked_vmap(per_voxel, (Y,), block=block)

    # Effective degrees of freedom: the diagonal of the influence F = V X^T W X,
    # computed directly as diag(V @ xtwx)_i = sum_j V[v,i,j] xtwx[v,j,i] so the
    # full (V, p, p) influence matrix is never materialised (it was the dominant
    # unbounded epilogue allocation -- ~3.2 GB at V=1e6, p=20, fp64).
    edf_diag = jnp.einsum('vij,vji->vi', v, xtwx)  # (V, p)
    edf = jnp.stack(
        [jnp.sum(edf_diag[:, lo:hi], axis=-1) for (lo, hi) in slices], axis=-1
    )
    edf_total = jnp.sum(edf_diag, axis=-1)

    fitted = family.linkinv(coef @ X.T)
    deviance = jnp.sum(family.unit_deviance(Y, fitted), axis=-1)
    y_bar = jnp.mean(Y, axis=-1, keepdims=True)
    null_dev = jnp.sum(
        family.unit_deviance(Y, jnp.broadcast_to(y_bar, Y.shape)), axis=-1
    )

    return GAMResult(
        coef=coef,
        lam=lam,
        edf=edf,
        edf_total=edf_total,
        dispersion=phi,
        deviance=deviance,
        null_deviance=null_dev,
        cov_unscaled=v,
        family=family,
        n_obs=int(n),
        col_slices=slices,
    )


def smooth_partial_effect(
    result: GAMResult,
    smooth_index: int,
    basis: Smooth,
    x: Union[Float[Array, ' g'], Tuple[Float[Array, ' g'], ...]],
) -> Tuple[Float[Array, 'V g'], Float[Array, 'V g']]:
    """Per-element partial effect of one smooth on a covariate grid ``x``.

    Returns ``(effect, se)``: the fitted smooth ``B(x) gamma_k`` and its
    pointwise standard error from the Bayesian covariance block (for a
    credible band).  ``basis`` is the smooth used to build the term; for a
    ``TensorBasis`` pass ``x`` as a tuple of matched per-margin grids (all
    length ``g``) and the effect is the interaction surface along that path.
    For a ``REBasis`` pass ``x`` as the integer level indices to read off -- the
    effect is then the per-level random effect (the BLUP intercept, or the
    random-slope coefficient) at those levels.
    """
    lo, hi = result.col_slices[smooth_index]
    design = basis.eval_design(x)  # (g, k) -- D8: per-basis via the Protocol
    gamma = result.coef[:, lo:hi]  # (V, k)
    effect = gamma @ design.T  # (V, g)
    cov_block = result.cov_unscaled[:, lo:hi, lo:hi]  # (V, k, k)
    var = jnp.einsum('gi,vij,gj->vg', design, cov_block, design)
    se = jnp.sqrt(jnp.clip(result.dispersion[:, None] * var, 1e-12, None))
    return effect, se


# ---------------------------------------------------------------------------
# Smooth-term significance (Wood 2013) -- the mgcv summary.gam p-value
# ---------------------------------------------------------------------------


class SmoothTest(NamedTuple):
    """Per-smooth approximate-significance test (Wood 2013, integer-rank).

    Each field is ``(V, m)`` -- one column per smooth (in ``smooths`` order), one
    row per element.  ``stat`` is the test statistic ``T_r``, ``rank`` its
    reference degrees of freedom, ``p_value`` the upper-tail p (chi-square for a
    fixed-dispersion family, ``F`` with ``N - edf_total`` denominator df
    otherwise), and ``edf`` the smooth's effective degrees of freedom (the
    ``mgcv::summary.gam`` "edf" column).
    """

    stat: Float[Array, 'V m']
    edf: Float[Array, 'V m']
    rank: Float[Array, 'V m']
    p_value: Float[Array, 'V m']


def _smooth_test_block(
    R: Float[Array, 'm m'],
    beta: Float[Array, 'V m'],
    v_block: Float[Array, 'V m m'],
    edf_k: Float[Array, 'V'],
    m: int,
    known_scale: bool,
    res_df: Float[Array, 'V'],
) -> Tuple[Float[Array, 'V'], Float[Array, 'V'], Float[Array, 'V']]:
    """Per-smooth Wood-2013 integer-rank statistic + p-value, vmapped over V.

    ``T_r = beta^T V_r^- beta`` with ``V_r^-`` the rank-``r`` pseudo-inverse of
    the QR-projected covariance ``R V R^T`` (``r`` = rounded edf), built by
    *masking* the eigen-spectrum rather than a dynamic slice so the per-voxel
    rank composes under ``vmap``.
    """
    eps9 = jnp.finfo(R.dtype).eps**0.9
    rbeta = beta @ R.T  # (V, m) -- row v is R @ beta_v
    vr = jnp.einsum('as,vst,bt->vab', R, v_block, R)  # R V R^T (V, m, m)

    def per_voxel(
        vr_v: Array, rbeta_v: Array, edf_v: Array
    ) -> Tuple[Array, Array]:
        vals, vecs = sym_eig_jacobi(vr_v, m)  # unsorted
        order = jnp.argsort(-vals)
        vals = vals[order]
        vecs = vecs[:, order]
        # type=1 integer rank from the effective df (mgcv:::testStat).
        fl = jnp.floor(edf_v)
        k_int = fl + jnp.where((edf_v > fl + 0.05) | (fl == 0.0), 1.0, 0.0)
        val_ok = vals > jnp.max(vals) * eps9
        k_rank = jnp.minimum(k_int, jnp.sum(val_ok))
        mask = (jnp.arange(m) < k_rank) & val_ok
        proj = vecs.T @ rbeta_v  # (m,) = u_i^T R beta
        contrib = jnp.where(
            mask, proj * proj / jnp.clip(vals, eps9, None), 0.0
        )
        return jnp.sum(contrib), jnp.sum(mask).astype(R.dtype)

    d, rank = jax.vmap(per_voxel)(vr, rbeta, edf_k)
    rank_c = jnp.clip(rank, 1.0, None)
    if known_scale:
        # chi-square upper tail: P(chi^2_r > d) = Q(r/2, d/2).
        pval = gammaincc(rank_c / 2.0, d / 2.0)
    else:
        # F upper tail: P(F_{r, res_df} > d/r) = I_x(res_df/2, r/2),
        # x = res_df / (res_df + d).
        x = res_df / (res_df + d)
        pval = betainc(res_df / 2.0, rank_c / 2.0, x)
    return d, rank, jnp.clip(pval, 0.0, 1.0)


def smooth_significance(
    result: GAMResult,
    smooths: Sequence[Smooth],
) -> SmoothTest:
    """Approximate significance test for each smooth term (Wood 2013).

    The ``mgcv::summary.gam`` smooth-term test in its integer-rank form
    (``mgcv:::testStat`` with ``type=1``): project the smooth's coefficients
    through its design's QR (``R``, ``R^T R = X^T X``), eigendecompose the
    projected Bayesian covariance ``R V R^T``, and form the rank-truncated
    quadratic form ``T_r = beta^T V_r^- beta`` with the rank set to the rounded
    effective df.  ``T_r`` is approximately ``chi^2_r`` for a fixed-dispersion
    family, or ``F_{r, N - edf_total}`` for an estimated scale -- the single
    "is ``s(x)`` significant, edf, p" line a developmental / brain-age GAM
    analyst reads off ``summary.gam``.

    ``smooths`` is the same sequence passed to :func:`gam_fit` (their ``.design``
    supplies each term's model matrix).  The test uses the forward-only
    ``sym_eig_jacobi`` and is **not** differentiated through.

    Returns a :class:`SmoothTest` of ``(V, m)`` arrays.  The fractional-rank
    refinement (the ``mgcv`` default ``type=0``, which needs the weighted-
    chi-square CDF) is a documented follow-up; the integer-rank test reproduces
    ``testStat(..., type=1)`` and tracks the default closely.
    """
    n = result.n_obs
    known_scale = result.family.has_fixed_dispersion
    res_df = jnp.clip(n - result.edf_total, 1.0, None)  # (V,) F denom df
    stats, edfs, ranks, pvals = [], [], [], []
    for k, sm in enumerate(smooths):
        lo, hi = result.col_slices[k]
        m = hi - lo
        Xk = jnp.asarray(sm.design, dtype=result.coef.dtype)  # (N, m)
        R = spd_chol(Xk.T @ Xk, m).T  # upper, R^T R = X^T X
        v_block = (
            result.dispersion[:, None, None]
            * result.cov_unscaled[:, lo:hi, lo:hi]
        )  # (V, m, m) scaled Bayesian covariance
        edf_k = result.edf[:, k]  # (V,)
        d, rank, pval = _smooth_test_block(
            R, result.coef[:, lo:hi], v_block, edf_k, m, known_scale, res_df
        )
        stats.append(d)
        edfs.append(edf_k)
        ranks.append(rank)
        pvals.append(pval)
    return SmoothTest(
        stat=jnp.stack(stats, axis=-1),
        edf=jnp.stack(edfs, axis=-1),
        rank=jnp.stack(ranks, axis=-1),
        p_value=jnp.stack(pvals, axis=-1),
    )
