# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
r"""
Mass-univariate generalised additive (mixed) models.

:func:`gam_fit` fits, per element (voxel / vertex / fixel), a generalised
additive model

.. math::

    g(\mathbb{E}[y]) = X_{\text{parametric}} \beta
        + \sum_k f_k(x_k), \qquad f_k = B_k(x_k)\,\gamma_k

with each smooth :math:`f_k` a penalised spline (from the sibling basis module)
carrying a roughness penalty :math:`\lambda_k\,\gamma_k^{\top} S_k\,\gamma_k`.
This is the ModelArray ``gam`` / ``mgcv``-style fit; a *generalised additive
mixed model* (GAMM) adds explicit random-effect blocks, which enter as just
more penalty components (a random effect is a ridge penalty), so the same
machinery covers both.

Two nested loops, one per element
---------------------------------

- **Inner** (fixed :math:`\lambda`): penalised iteratively reweighted least
  squares -- the same solver-free weighted normal-equations solve as the
  generalised linear model, with the block penalty
  :math:`S(\lambda) = \sum_k \lambda_k S_k` added.  Ordinary and weighted least
  squares and the exponential family all reduce to it.
- **Outer** (select :math:`\lambda`): the generalised Fellner-Schall update
  (Wood & Fasiolo, 2017) -- a multiplicative, positivity-preserving
  generalised-REML step

  .. math::

      \lambda_k \leftarrow \lambda_k\,
        \frac{\operatorname{tr}(S_\lambda^{-} S_k) - \operatorname{tr}(V S_k)}
             {\gamma_k^{\top} S_k\,\gamma_k / \phi}

  that increases the (Laplace) marginal likelihood each iteration.  Because GAM
  smooths occupy disjoint coefficient blocks,
  :math:`\operatorname{tr}(S_\lambda^{-} S_k) = \operatorname{rank}(S_k)/\lambda_k`
  -- no generalised inverse of the summed penalty is needed.  This is the
  operational form of the penalty / variance-component REML equivalence (the GAM
  smoothing parameter is the ratio :math:`\phi / \sigma_b^2` of a mixed model).

Both loops run a fixed number of iterations (clean under
:func:`jax.vmap` over elements) and every solve avoids the vendor dense
eigensolver, so the whole fit runs on hardware where that solver is unreliable.

Outputs (ModelArray ``gam`` parity)
-----------------------------------

:class:`GAMResult` carries per-element coefficients, selected :math:`\lambda`,
per-smooth **effective degrees of freedom** (the trace of the smooth's influence
block) and total effective degrees of freedom, dispersion, deviance, and the
Bayesian coefficient covariance :math:`V = (X^{\top} W X + S_\lambda)^{-1}` (for
smooth-term confidence bands and the approximate :math:`F` / :math:`\chi^2`
tests).  Partial effects are rendered with :func:`smooth_partial_effect`.

References
----------
Wood, S. N. & Fasiolo, M. (2017). A generalized Fellner-Schall method for
smoothing parameter optimization with application to Tweedie location, scale
and shape models. Biometrics, 73(4), 1071-1081.
https://doi.org/10.1111/biom.12666

Wood, S. N. (2017). Generalized Additive Models: An Introduction with R,
2nd ed. Chapman and Hall/CRC.
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
    'gam_predict',
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
    aux=('family', 'n_obs', 'col_slices', 'intercept'),
)
@dataclass(frozen=True)
class GAMResult:
    r"""Per-element GAM fit output.

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
        ``(V, p, p)`` Bayesian covariance
        :math:`(X^{\top} W X + S_\lambda)^{-1}`.
    col_slices
        Per-smooth ``(lo, hi)`` column ranges into ``coef`` (the smooth blocks
        only; the intercept / parametric columns precede the first smooth).
    intercept
        Whether the fitted design carried a leading intercept column -- stored
        so :func:`gam_predict` reassembles the design identically.
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
    intercept: bool


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
    r"""Assemble the shared design matrix, penalties, and column layout.

    Builds the full design :math:`X`, the stacked full-size penalties
    :math:`S_k` embedded into the coefficient space, their block eigenvalues
    (used by the Fellner-Schall penalty trace), and the per-smooth column
    slices.  A smooth may carry **multiple** penalties (a tensor product has
    one per margin), so the number of penalties ``K`` may exceed the number of
    smooth terms.

    Parameters
    ----------
    n
        Number of observation rows :math:`N` (the design row count).
    smooths
        Penalised smooth bases (one per smooth term).  Each supplies its
        ``design`` matrix, its column count ``dim``, and its
        ``penalty_blocks()`` (per-penalty coefficient-space block matrix plus
        block eigenvalues).
    parametric
        Optional ``(N, q)`` unpenalised linear design placed immediately after
        the intercept; ``None`` if there are no parametric covariates.
    intercept
        Whether to prepend a leading column of ones.
    dtype
        Floating dtype for the assembled arrays.

    Returns
    -------
    X : Float[Array, 'N p']
        The assembled design ``[intercept | parametric | smooth_1 | ... ]``.
    pen_full : Float[Array, 'K p p']
        The ``K`` penalty matrices, each embedded at its smooth's coefficient
        block within the full :math:`(p, p)` coefficient space.
    pen_eig : Float[Array, 'K p']
        The block eigenvalues of each penalty (zero outside its block).
    slices : tuple of (int, int)
        Per-smooth ``(lo, hi)`` column ranges into the assembled design (one
        entry per smooth term).
    """
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
    r"""Run penalised iteratively reweighted least squares from a warm start.

    A thin wrapper over the shared penalised-IRLS core that returns the extra
    quantities the Fellner-Schall step needs, all evaluated at the converged
    coefficients.

    Parameters
    ----------
    y
        ``(N,)`` response for this element.
    X
        ``(N, p)`` design matrix.
    s_lambda
        ``(p, p)`` summed penalty :math:`S_\lambda = \sum_k \lambda_k S_k`.
    family
        Exponential family supplying the link and variance functions.
    p
        Number of design columns :math:`p` (static, for the linear solve).
    n_iter
        Number of inner IRLS iterations.
    ridge
        Small stabiliser added to the penalised normal equations.
    beta0
        ``(p,)`` warm-start coefficients.

    Returns
    -------
    beta : Float[Array, 'p']
        The converged coefficients.
    V : Float[Array, 'p p']
        The Bayesian covariance
        :math:`V = (X^{\top} W X + S_\lambda + \text{ridge})^{-1}`.
    xtwx : Float[Array, 'p p']
        The unpenalised weighted Gram matrix :math:`X^{\top} W X` (for the
        effective-degrees-of-freedom and Fellner-Schall traces).
    """
    beta, v, xtwx, _ = fit_penalised_irls(
        y, X, family, penalty=s_lambda, beta0=beta0, n_iter=n_iter, ridge=ridge
    )
    return beta, v, xtwx


def _trace_slinv_sk(
    ek: Float[Array, 'p'], s_lambda_eig: Float[Array, 'p']
) -> Float[Array, '']:
    r"""Compute the Fellner-Schall penalty trace from block eigenvalues.

    Evaluates :math:`\operatorname{tr}(S_\lambda^{+} S_k)` from the precomputed
    block eigenvalues.  With every penalty diagonal in its block's joint
    eigenbasis, the trace is an elementwise sum
    :math:`\sum_i [\,s_i > 0\,]\, e_{k,i} / s_i` (with :math:`s_i` the pooled
    penalty eigenvalues and :math:`e_{k,i}` those of :math:`S_k`) --
    basis-invariant, so it is exact even though the *fit* is carried in the
    original (non-rotated) basis.  For a lone disjoint penalty this reduces to
    :math:`\operatorname{rank}_k / \lambda_k` (the simple shortcut); for
    overlapping tensor-product penalties it is the correct general trace, with
    no pseudo-inverse needed.

    Parameters
    ----------
    ek
        ``(p,)`` block eigenvalues of the single penalty :math:`S_k`.
    s_lambda_eig
        ``(p,)`` pooled block eigenvalues of the summed penalty
        :math:`S_\lambda`.

    Returns
    -------
    Float[Array, '']
        The scalar trace :math:`\operatorname{tr}(S_\lambda^{+} S_k)`.
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
    r"""Fit with one smoothing parameter shared across all elements (Gaussian).

    For the Gaussian identity link the influence matrix
    :math:`V = (X^{\top} X + S_\lambda)^{-1}` is *shared* (it has no :math:`y`
    dependence), so the **pooled** Fellner-Schall update is a function only of
    the :math:`(p, p)` sufficient statistics :math:`X^{\top} X` and
    :math:`C = (Y X)^{\top} (Y X) = \sum_v (X^{\top} y_v)(X^{\top} y_v)^{\top}`
    and the scalar :math:`\operatorname{tr}(\sum_v y_v y_v^{\top})`.  The outer
    loop is therefore :math:`O(n_{\text{outer}}\, p^3)` -- **independent of**
    the number of elements -- removing the per-element outer loop entirely; only
    the final coefficient fit and the sufficient statistics touch the element
    axis.

    Parameters
    ----------
    Y
        ``(V, N)`` responses (one row per element).
    X
        ``(N, p)`` shared design matrix.
    penalties
        ``(m, p, p)`` stacked penalty matrices embedded in coefficient space.
    pen_eig
        ``(m, p)`` per-penalty block eigenvalues.
    n_outer
        Number of pooled Fellner-Schall outer iterations.
    ridge
        Small stabiliser added to the penalised normal equations.
    lam_floor, lam_ceil
        Clamp on each smoothing parameter.

    Returns
    -------
    coef : Float[Array, 'V p']
        Per-element coefficients :math:`\beta_v = V (X^{\top} y_v)`.
    lam : Float[Array, 'V m']
        The shared smoothing parameters, broadcast over elements.
    V : Float[Array, 'V p p']
        The Bayesian covariance, broadcast over elements.
    xtwx : Float[Array, 'V p p']
        The Gram matrix :math:`X^{\top} X`, broadcast over elements.
    dispersion : Float[Array, 'V']
        Per-element residual-variance scale estimate.
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
    r"""Fit the GAM for a single element (general exponential family).

    Runs the Fellner-Schall outer loop over the inner penalised-IRLS solve,
    then re-fits at the selected smoothing parameters.

    Parameters
    ----------
    y
        ``(N,)`` response for this element.
    X
        ``(N, p)`` design matrix.
    penalties
        ``(m, p, p)`` stacked penalty matrices embedded in coefficient space.
    pen_eig
        ``(m, p)`` per-penalty block eigenvalues.
    family
        Exponential family supplying the link, variance, and dispersion
        convention.
    p
        Number of design columns :math:`p` (static, for the linear solve).
    n_outer
        Number of Fellner-Schall outer iterations.
    n_inner
        Number of penalised-IRLS inner iterations.
    ridge
        Small stabiliser added to the penalised normal equations.
    lam_floor, lam_ceil
        Clamp on each smoothing parameter.

    Returns
    -------
    beta : Float[Array, 'p']
        The fitted coefficients.
    lam : Float[Array, 'm']
        The selected smoothing parameters (one per penalty).
    V : Float[Array, 'p p']
        The Bayesian coefficient covariance.
    xtwx : Float[Array, 'p p']
        The unpenalised weighted Gram matrix :math:`X^{\top} W X`.
    dispersion : Float[Array, '']
        The scale estimate (residual scale for Gaussian; ``1`` for
        fixed-dispersion families).
    """
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
    r"""Fit the Gaussian GAM for one element from its cross-products.

    Exact per-element Gaussian fit from the cross-products
    :math:`c = X^{\top} y_v` and :math:`g = y_v^{\top} y_v`.  For the Gaussian
    identity link the penalised IRLS converges in **one** step to
    :math:`\beta = (X^{\top} X + S_\lambda)^{-1} c` -- a function of :math:`y`
    only through :math:`c` -- and the dispersion
    :math:`\phi = (g - 2\,\beta^{\top} c + \beta^{\top} X^{\top} X\,\beta) /
    (N - \text{edf})`, the effective degrees of freedom, and the Fellner-Schall
    traces all reduce to :math:`(c, g, X^{\top} X)`.  So the whole per-element
    Fellner-Schall loop runs in :math:`p`-space with **no N-dimensional vector
    in the loop** -- :math:`N` enters only the one-off cross-products
    :math:`X^{\top} Y` and :math:`\operatorname{diag}(Y Y^{\top})` -- and the
    result is identical (to floating point) to :func:`_gam_fit_one` for the
    Gaussian family.

    Parameters
    ----------
    c
        ``(p,)`` design-response cross-product :math:`X^{\top} y_v`.
    g
        The scalar response energy :math:`y_v^{\top} y_v`.
    xtx
        ``(p, p)`` shared Gram matrix :math:`X^{\top} X`.
    penalties
        ``(m, p, p)`` stacked penalty matrices embedded in coefficient space.
    pen_eig
        ``(m, p)`` per-penalty block eigenvalues.
    n
        Number of observation rows :math:`N` (for the dispersion denominator).
    p
        Number of design columns :math:`p` (static, for the linear solve).
    n_outer
        Number of Fellner-Schall outer iterations.
    ridge
        Small stabiliser added to the penalised normal equations.
    lam_floor, lam_ceil
        Clamp on each smoothing parameter.

    Returns
    -------
    beta : Float[Array, 'p']
        The fitted coefficients.
    lam : Float[Array, 'm']
        The selected smoothing parameters (one per penalty).
    V : Float[Array, 'p p']
        The Bayesian coefficient covariance.
    xtwx : Float[Array, 'p p']
        The shared Gram matrix :math:`X^{\top} X`.
    dispersion : Float[Array, '']
        The residual-variance scale estimate.
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
    r"""Fit a mass-univariate GAM: shared smooth bases, per-element responses.

    Parameters
    ----------
    Y
        ``(V, N)`` responses.
    smooths
        Penalised smooth bases (one per smooth term): a
        :class:`SplineBasis` (from :func:`bspline_basis`,
        :func:`thinplate_regression_basis`, or :func:`cyclic_cubic_basis`) or a
        :class:`TensorBasis` (from :func:`tensor_product_basis`) for an
        anisotropic interaction.  A tensor smooth carries one smoothing
        parameter per margin, all selected by the same Fellner-Schall loop.
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
        statistics -- an :math:`O(n_{\text{outer}}\, p^3)`, element-count
        independent outer loop (much faster when smoothness is homogeneous
        across the brain).  Gaussian only.
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
    GAMResult
        The per-element fit: coefficients, selected smoothing parameters,
        per-smooth effective degrees of freedom, dispersion, deviance, and the
        Bayesian covariance.
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
        intercept=intercept,
    )


def gam_predict(
    result: GAMResult,
    smooths: Sequence[Smooth],
    x_smooths: Sequence[Any],
    *,
    parametric: Optional[Float[Array, 'N q']] = None,
    type: Literal['response', 'link'] = 'response',
) -> Float[Array, 'V N']:
    """Per-element GAM prediction on (new) covariates.

    Reassembles the fitted design ``[intercept | parametric | B_1(x_1) | ...]``
    at the new covariates -- evaluating each smooth's basis at ``x_smooths[k]``
    with the **same** fitted knots/penalty (``smooth.eval_design``, the
    machinery :func:`smooth_partial_effect` uses for one block) and the stored
    ``result.col_slices`` / ``result.intercept`` layout -- then returns
    ``eta = design @ coef`` (``type='link'``) or ``family.linkinv(eta)``
    (``type='response'``, default).

    Parameters
    ----------
    result
        A :class:`GAMResult` from :func:`gam_fit`.
    smooths
        The same :class:`SmoothBasis` terms passed to :func:`gam_fit`, in the
        same order (used here only to evaluate the basis at the new covariates).
    x_smooths
        New covariate(s) per smooth (one entry per ``smooths`` term, each in the
        form that ``smooth.eval_design`` accepts -- an ``(N,)`` grid, a tuple of
        per-margin grids for a tensor smooth, or integer level indices for a
        random-effect basis).  All terms must evaluate to the same row count
        ``N``.
    parametric
        ``(N, q)`` parametric design at the new covariates; ``None`` if the fit
        had none.  Must match the fit's parametric column count.
    type
        ``'response'`` (the mean, via the link inverse) or ``'link'`` (the
        linear predictor ``eta``).

    Returns
    -------
    ``(V, N)`` predictions, differentiable w.r.t. the new covariates / the
    parametric design (and the fitted coefficients).
    """
    blocks = []
    n_ref: Optional[int] = None
    if parametric is not None:
        parametric = jnp.asarray(parametric)
        n_ref = parametric.shape[0]
    designs = []
    for smooth, x in zip(smooths, x_smooths):
        d = smooth.eval_design(x)  # (N, k)
        designs.append(d)
        n_ref = d.shape[0] if n_ref is None else n_ref
    if n_ref is None:
        raise ValueError(
            'gam_predict: provide at least one smooth (with covariates) or a '
            'parametric design to define the prediction rows.'
        )
    if result.intercept:
        blocks.append(jnp.ones((n_ref, 1), dtype=result.coef.dtype))
    if parametric is not None:
        blocks.append(parametric.astype(result.coef.dtype))
    blocks.extend(d.astype(result.coef.dtype) for d in designs)
    design = jnp.concatenate(blocks, axis=1)  # (N, p)
    eta = result.coef @ design.T  # (V, N)
    if type == 'link':
        return eta
    if type == 'response':
        return result.family.linkinv(eta)
    raise ValueError(
        f"gam_predict: type={type!r}; expected 'response' or 'link'."
    )


def smooth_partial_effect(
    result: GAMResult,
    smooth_index: int,
    basis: Smooth,
    x: Union[Float[Array, ' g'], Tuple[Float[Array, ' g'], ...]],
) -> Tuple[Float[Array, 'V g'], Float[Array, 'V g']]:
    r"""Evaluate one smooth's partial effect on a covariate grid.

    Computes the per-element fitted smooth :math:`B(x)\,\gamma_k` and its
    pointwise standard error from the Bayesian covariance block (for a credible
    band).

    Parameters
    ----------
    result
        A :class:`GAMResult` from :func:`gam_fit`.
    smooth_index
        Index of the smooth term (into ``result.col_slices``) to render.
    basis
        The smooth basis used to build this term.  For a :class:`TensorBasis`
        pass ``x`` as a tuple of matched per-margin grids (all length ``g``) and
        the effect is the interaction surface along that path.  For a
        :class:`REBasis` pass ``x`` as the integer level indices to read off,
        and the effect is the per-level random effect (the best linear unbiased
        predictor intercept, or the random-slope coefficient) at those levels.
    x
        ``(g,)`` covariate grid, or a tuple of per-margin grids for a tensor
        smooth, at which to evaluate the effect.

    Returns
    -------
    effect : Float[Array, 'V g']
        The fitted partial effect at each grid point, per element.
    se : Float[Array, 'V g']
        The pointwise standard error of the effect, per element.
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
    r"""Per-smooth approximate-significance test (Wood, 2013; integer-rank).

    Each field is ``(V, m)`` -- one column per smooth (in ``smooths`` order),
    one row per element.  ``stat`` is the test statistic :math:`T_r`, ``rank``
    its reference degrees of freedom, ``p_value`` the upper-tail p-value
    (:math:`\chi^2` for a fixed-dispersion family, :math:`F` with
    :math:`N - \text{edf}_{\text{total}}` denominator degrees of freedom
    otherwise), and ``edf`` the smooth's effective degrees of freedom (the
    ``mgcv::summary.gam`` "edf" column).

    References
    ----------
    Wood, S. N. (2013). On p-values for smooth components of an extended
    generalized additive model. Biometrika, 100(1), 221-228.
    https://doi.org/10.1093/biomet/ass048
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
    r"""Compute the Wood-2013 integer-rank statistic and p-value per element.

    Forms :math:`T_r = \beta^{\top} V_r^{-}\,\beta` with :math:`V_r^{-}` the
    rank-:math:`r` pseudo-inverse of the QR-projected covariance
    :math:`R V R^{\top}` (:math:`r` = rounded effective degrees of freedom),
    built by *masking* the eigen-spectrum rather than by a dynamic slice so the
    per-element rank composes under :func:`jax.vmap`.

    Parameters
    ----------
    R
        ``(m, m)`` upper-triangular Cholesky factor with
        :math:`R^{\top} R = X_k^{\top} X_k` for this smooth's design.
    beta
        ``(V, m)`` per-element coefficients for this smooth block.
    v_block
        ``(V, m, m)`` scaled Bayesian covariance for this smooth block.
    edf_k
        ``(V,)`` effective degrees of freedom of the smooth, per element.
    m
        Number of columns :math:`m` in the smooth block (static).
    known_scale
        Whether the dispersion is fixed (chi-square reference) rather than
        estimated (:math:`F` reference).
    res_df
        ``(V,)`` residual degrees of freedom (the :math:`F` denominator).

    Returns
    -------
    stat : Float[Array, 'V']
        The test statistic :math:`T_r`, per element.
    rank : Float[Array, 'V']
        The effective reference rank used, per element.
    p_value : Float[Array, 'V']
        The upper-tail p-value, per element.
    """
    eps9 = jnp.finfo(R.dtype).eps ** 0.9
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
    r"""Approximate significance test for each smooth term (Wood, 2013).

    The ``mgcv::summary.gam`` smooth-term test in its integer-rank form
    (``mgcv:::testStat`` with ``type=1``): project the smooth's coefficients
    through its design's QR factor (:math:`R`, with :math:`R^{\top} R =
    X^{\top} X`), eigendecompose the projected Bayesian covariance
    :math:`R V R^{\top}`, and form the rank-truncated quadratic form
    :math:`T_r = \beta^{\top} V_r^{-}\,\beta` with the rank set to the rounded
    effective degrees of freedom.  :math:`T_r` is approximately
    :math:`\chi^2_r` for a fixed-dispersion family, or
    :math:`F_{r,\,N - \text{edf}_{\text{total}}}` for an estimated scale -- the
    single "is ``s(x)`` significant, edf, p" line a developmental / brain-age
    GAM analyst reads off ``summary.gam``.

    The test uses the forward-only :func:`sym_eig_jacobi` eigensolver and is
    **not** differentiated through.  The fractional-rank refinement (the
    ``mgcv`` default ``type=0``, which needs the weighted-chi-square CDF) is a
    documented follow-up; the integer-rank test reproduces
    ``testStat(..., type=1)`` and tracks the default closely.

    Parameters
    ----------
    result
        A :class:`GAMResult` from :func:`gam_fit`.
    smooths
        The same sequence of smooth terms passed to :func:`gam_fit`, in the
        same order (their ``design`` supplies each term's model matrix).

    Returns
    -------
    SmoothTest
        The per-smooth test statistic, effective degrees of freedom, reference
        rank, and p-value, each a ``(V, m)`` array.

    References
    ----------
    Wood, S. N. (2013). On p-values for smooth components of an extended
    generalized additive model. Biometrika, 100(1), 221-228.
    https://doi.org/10.1093/biomet/ass048
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
