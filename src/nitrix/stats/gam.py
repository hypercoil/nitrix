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
every solve is cuSOLVER-free (``stats._smalllinalg``), so the whole fit runs on
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
from typing import Any, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jaxtyping import Array, Float

from ._batching import blocked_vmap
from ._family import GAUSSIAN, Family, resolve_family
from ._irls import fit_penalised_irls
from .basis import SplineBasis, spline_design

__all__ = ['GAMResult', 'gam_fit', 'smooth_partial_effect']


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@jax.tree_util.register_pytree_node_class
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

    def tree_flatten(
        self,
    ) -> Tuple[Tuple[Array, ...], Tuple[Any, ...]]:
        children = (
            self.coef,
            self.lam,
            self.edf,
            self.edf_total,
            self.dispersion,
            self.deviance,
            self.null_deviance,
            self.cov_unscaled,
        )
        return children, (self.family, self.n_obs, self.col_slices)

    @classmethod
    def tree_unflatten(
        cls, aux: Tuple[Any, ...], children: Tuple[Any, ...]
    ) -> 'GAMResult':
        family, n_obs, col_slices = aux
        (
            coef,
            lam,
            edf,
            edf_total,
            dispersion,
            deviance,
            null_deviance,
            cov_unscaled,
        ) = children
        return cls(
            coef=coef,
            lam=lam,
            edf=edf,
            edf_total=edf_total,
            dispersion=dispersion,
            deviance=deviance,
            null_deviance=null_deviance,
            cov_unscaled=cov_unscaled,
            family=family,
            n_obs=n_obs,
            col_slices=col_slices,
        )


# ---------------------------------------------------------------------------
# Design + penalty assembly (data-independent: shared across elements)
# ---------------------------------------------------------------------------


def _assemble(
    n: int,
    smooths: Sequence[SplineBasis],
    parametric: Optional[Float[Array, 'N q']],
    intercept: bool,
    dtype: Any,
) -> Tuple[
    Float[Array, 'N p'],
    Float[Array, 'm p p'],
    Tuple[Tuple[int, int], ...],
    np.ndarray,
]:
    """Build the full design ``X``, stacked full-size penalties ``S_k``, the
    per-smooth column slices, and the per-smooth penalty ranks."""
    blocks = []
    if intercept:
        blocks.append(jnp.ones((n, 1), dtype=dtype))
    if parametric is not None:
        blocks.append(jnp.asarray(parametric, dtype=dtype))
    smooth_start = sum(b.shape[1] for b in blocks)
    slices = []
    col = smooth_start
    for sm in smooths:
        blocks.append(jnp.asarray(sm.design, dtype=dtype))
        slices.append((col, col + sm.dim))
        col += sm.dim
    X = jnp.concatenate(blocks, axis=1)
    p = X.shape[1]

    pen_full = []
    ranks = []
    for sm, (lo, hi) in zip(smooths, slices):
        s_full = np.zeros((p, p))
        s_block = np.asarray(sm.penalty)
        s_full[lo:hi, lo:hi] = s_block
        pen_full.append(s_full)
        ranks.append(int(np.linalg.matrix_rank(s_block)))
    penalties = jnp.asarray(np.stack(pen_full), dtype=dtype) if pen_full else (
        jnp.zeros((0, p, p), dtype=dtype)
    )
    return X, penalties, tuple(slices), np.asarray(ranks, dtype=np.float64)


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


# ---------------------------------------------------------------------------
# Per-element fit: Fellner-Schall outer loop over the inner penalised IRLS
# ---------------------------------------------------------------------------


def _gam_fit_one(
    y: Float[Array, 'N'],
    X: Float[Array, 'N p'],
    penalties: Float[Array, 'm p p'],
    ranks: Float[Array, 'm'],
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

        # Generalized Fellner-Schall update per smooth (disjoint blocks ->
        # tr(S_lambda^- S_k) = rank_k / lambda_k).
        def fs(k: Array) -> Float[Array, '']:
            sk = penalties[k]
            tr_vsk = jnp.sum(v * sk)  # tr(V S_k)
            energy = beta @ (sk @ beta)
            # Generalized Fellner-Schall: lambda_k * [rank_k/lambda_k -
            # tr(V S_k)] / [energy / phi] = [rank_k - lambda_k tr(V S_k)] /
            # [energy / phi].
            num = jnp.clip(ranks[k] - lam[k] * tr_vsk, 1e-8, None)
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
# Public API
# ---------------------------------------------------------------------------


def gam_fit(
    Y: Float[Array, 'V N'],
    smooths: Sequence[SplineBasis],
    *,
    parametric: Optional[Float[Array, 'N q']] = None,
    intercept: bool = True,
    family: Union[str, Family] = GAUSSIAN,
    n_outer: int = 20,
    n_inner: int = 15,
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
        Penalised spline bases (one per smooth term), built with
        ``stats.basis.bspline_basis`` on the (shared) covariates.
    parametric
        Optional ``(N, q)`` unpenalised linear design (covariates entering
        linearly).  The intercept is added separately (see ``intercept``).
    intercept
        Prepend an intercept column (default ``True``).
    family
        Exponential family (default ``GAUSSIAN``).
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

    X, penalties, slices, ranks = _assemble(
        X_n, smooths, parametric, intercept, Y.dtype
    )
    p = X.shape[1]
    ranks_j = jnp.asarray(ranks, dtype=Y.dtype)

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
            y, X, penalties, ranks_j, family, p,
            n_outer, n_inner, ridge, lam_floor, lam_ceil,
        )

    coef, lam, v, xtwx, phi = blocked_vmap(per_voxel, (Y,), block=block)

    # Effective degrees of freedom: F = V X^T W X (influence on coefficients).
    influence = jnp.einsum('vij,vjk->vik', v, xtwx)  # (V, p, p)
    edf_diag = jnp.diagonal(influence, axis1=-2, axis2=-1)  # (V, p)
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
    basis: SplineBasis,
    x: Float[Array, ' g'],
) -> Tuple[Float[Array, 'V g'], Float[Array, 'V g']]:
    """Per-element partial effect of one smooth on a covariate grid ``x``.

    Returns ``(effect, se)``: the fitted smooth ``B(x) gamma_k`` and its
    pointwise standard error from the Bayesian covariance block (for a
    credible band).  ``basis`` is the ``SplineBasis`` used to build the smooth.
    """
    lo, hi = result.col_slices[smooth_index]
    design = spline_design(basis, x)  # (g, k)
    gamma = result.coef[:, lo:hi]  # (V, k)
    effect = gamma @ design.T  # (V, g)
    cov_block = result.cov_unscaled[:, lo:hi, lo:hi]  # (V, k, k)
    var = jnp.einsum('gi,vij,gj->vg', design, cov_block, design)
    se = jnp.sqrt(jnp.clip(result.dispersion[:, None] * var, 1e-12, None))
    return effect, se
