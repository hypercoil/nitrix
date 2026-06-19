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

Scope.  This ships the **scalar random intercept** ``(1 | g)`` -- the FR §1.2
headline ("binary outcomes / lesion counts per subject with random intercepts").
Random *slopes* (correlated ``(1 + x | g)`` or diagonal ``(x || g)``) under a
non-Gaussian family are the structured-RE GLMM follow-up (Tier-2); the Gaussian
random slope is already served by ``lme_fit`` (R2 block-Woodbury).

References
----------
- Breslow, N. E. & Clayton, D. G. (1993). Approximate inference in generalized
  linear mixed models.  JASA 88, 9-25.
- Wood, S. N. & Fasiolo, M. (2017). A generalized Fellner-Schall method for
  smoothing parameter optimization.  Biometrics 73, 1071-1081.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union, cast

import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Float, Int

from ._batching import blocked_vmap
from ._family import GAUSSIAN, Family, resolve_family
from ._smalllinalg import small_inv_logdet
from .basis import re_smooth
from .gam import gam_fit

__all__ = ['GLMMResult', 'glmm_fit']

_EPS = 1e-10


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class GLMMResult:
    """Per-element random-intercept GLMM fit output (PQL).

    Attributes
    ----------
    beta_hat
        ``(V, p)`` fixed-effect estimates (over the columns of ``X``).
    blups
        ``(V, q)`` per-level random intercepts (the BLUPs ``b_j``).
    re_var
        ``(V,)`` random-effect variance ``sigma_b^2 = phi / lambda`` (``phi`` the
        dispersion, ``lambda`` the Fellner-Schall precision on the RE block).
    dispersion
        ``(V,)`` scale ``phi`` (residual variance for Gaussian / Gamma; ``1`` for
        the fixed-dispersion families -- binomial / Poisson / negative-binomial).
    deviance
        ``(V,)`` model deviance at the converged mean.
    edf_total
        ``(V,)`` total effective degrees of freedom (fixed + random).
    tier
        Which solver ran: ``'few'`` (dense GAMM ``gam_fit``) or ``'many'`` (the
        structured Schur-complement PQL).
    """

    beta_hat: Float[Array, 'V p']
    blups: Float[Array, 'V q']
    re_var: Float[Array, 'V']
    dispersion: Float[Array, 'V']
    deviance: Float[Array, 'V']
    edf_total: Float[Array, 'V']
    family: Family
    n_obs: int
    n_groups: int
    tier: str

    def tree_flatten(
        self,
    ) -> Tuple[Tuple[Array, ...], Tuple[Any, ...]]:
        children = (
            self.beta_hat,
            self.blups,
            self.re_var,
            self.dispersion,
            self.deviance,
            self.edf_total,
        )
        return children, (
            self.family,
            self.n_obs,
            self.n_groups,
            self.tier,
        )

    @classmethod
    def tree_unflatten(
        cls, aux: Tuple[Any, ...], children: Tuple[Any, ...]
    ) -> 'GLMMResult':
        family, n_obs, n_groups, tier = aux
        (beta_hat, blups, re_var, dispersion, deviance, edf_total) = children
        return cls(
            beta_hat=beta_hat,
            blups=blups,
            re_var=re_var,
            dispersion=dispersion,
            deviance=deviance,
            edf_total=edf_total,
            family=family,
            n_obs=n_obs,
            n_groups=n_groups,
            tier=tier,
        )


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
    re_var = res.dispersion / jnp.clip(res.lam[:, 0], _EPS, None)
    return GLMMResult(
        beta_hat=beta_hat,
        blups=blups,
        re_var=re_var,
        dispersion=res.dispersion,
        deviance=res.deviance,
        edf_total=res.edf_total,
        family=family,
        n_obs=int(X.shape[0]),
        n_groups=n_groups,
        tier='few',
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
    z = eta + (y - mu) / jnp.clip(dmu, _EPS, None)  # (N,) working response

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
        re_var=re_var,
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


def glmm_fit(
    Y: Float[Array, 'V N'],
    X: Float[Array, 'N p'],
    *,
    group: Int[Array, 'N'],
    family: Union[str, Family] = GAUSSIAN,
    few_level_max: int = 64,
    n_outer: int = 20,
    n_inner: int = 10,
    ridge: float = 1e-8,
    lam_floor: float = 1e-6,
    lam_ceil: float = 1e8,
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
    family
        GLM family (``'binomial'`` / ``'poisson'`` / ``'gamma'`` /
        ``'negbinomial'`` / ``'gaussian'`` or a :class:`Family`).  Gaussian is
        accepted (it reduces to the LME) but ``lme_fit`` / ``reml_fit`` are the
        direct route for a Gaussian mixed model.
    few_level_max
        Dispatch threshold on the number of levels ``q`` (default ``64`` -- the
        regime where the dense solve stops being trivially cheap).
    n_outer, n_inner
        PQL outer (Fellner-Schall) and inner (IRLS) iteration budgets.
    ridge, lam_floor, lam_ceil, block
        Normal-equation stabiliser, smoothing-parameter clamps, and the optional
        element-block size bounding peak memory.

    Returns
    -------
    ``GLMMResult`` -- ``beta_hat``, per-level ``blups``, ``re_var``
    (``sigma_b^2``), ``dispersion``, ``deviance``, ``edf_total``, and the
    ``tier`` that ran (``'few'`` / ``'many'``).

    Notes
    -----
    PQL carries the documented small-cluster bias for binary / low-count
    responses (it under-estimates the variance component; the bias vanishes as
    the per-group information grows).  Laplace / adaptive-quadrature is the
    Tier-2 follow-up (v3 §11).  Random *slopes* under a non-Gaussian family are
    also Tier-2; the Gaussian random slope is served by ``lme_fit`` (R2).
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
    n_groups = int(jnp.max(group)) + 1

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
