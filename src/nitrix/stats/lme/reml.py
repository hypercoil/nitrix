# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Variance-components REML for voxelwise LMEs.

The model
---------

For each voxel ``v``, the LME is::

    y_v = X beta_v + Z b_v + eps_v
    b_v ~ N(0, sigma_b^2 I_q)
    eps_v ~ N(0, sigma_e^2 I_N)
    Cov(y_v) = V = sigma_b^2 ZZ^T + sigma_e^2 I_N

The fixed-effect design ``X`` and random-effect design ``Z`` are
**shared** across voxels (the typical fMRI / dMRI case: one
group-level design, applied to every voxel's response).  Only
``y_v`` varies per voxel.

The profile REML (Restricted Maximum Likelihood) negative log-
likelihood, after profiling out ``beta``, is::

    nll(theta) = 0.5 * [log|V| + log|X^T V^{-1} X| + r^T V^{-1} r]

where ``r = y - X beta_hat`` and ``beta_hat = (X^T V^{-1} X)^{-1}
X^T V^{-1} y``.  We parameterise ``theta = log(sigma^2)`` (log-
space) so optimisation is unconstrained.

The FaST-LMM spectral trick
---------------------------

Lippert et al. 2011 (FaST-LMM): eigendecompose ``ZZ^T = U Lambda
U^T``.  In the rotated basis ``y_rot = U^T y``, ``X_rot = U^T X``,
the total covariance is diagonal::

    V_rot = sigma_b^2 Lambda + sigma_e^2 I = diag(d)
    d_i = sigma_b^2 lambda_i + sigma_e^2

Every operation in the Newton iteration becomes elementwise on
``d``: ``log|V| = sum_i log d_i``; ``V^{-1}`` is ``diag(1/d)``;
``X^T V^{-1} X = sum_i x_i x_i^T / d_i``.  Per-iteration cost
drops from ``O(N^3)`` (naive) to ``O(N p^2 + N)`` (rotated).

The eigendecomposition of ``ZZ^T`` is computed **once** at the
outer call -- shared across all voxels via vmap closure.

Memory regime
-------------

For ``V`` voxels, ``N`` subjects, ``p`` fixed-effect coefficients:

- Shared (computed once):
  - ``U``, ``Lambda``: ``(N, N)`` + ``(N,)``  --  ``~N^2 * 4`` bytes.
  - ``X_rot``: ``(N, p)``.
- Per-voxel (vmap):
  - ``y_rot``: ``(N,)``.
  - ``XtVinvX``: ``(p, p)``.  Tiny.
  - ``beta``: ``(p,)``.
- Per-Newton-step intermediates: ``(N,)`` arrays.

Total HBM at ``V = 100k``, ``N = 30``, ``p = 5``:

- ``Y``: ``100k * 30 * 4 = 12 MB``.
- Per-voxel results: ``100k * (2 + 5 + 1) * 4 = ~3 MB``.
- Newton-step intermediates (vmapped): ``100k * 30 * 4 = 12 MB`` peak.

Total ~30 MB.  Fits trivially.  At ``V = 1M``, ``N = 100``,
``p = 10``: ``~500 MB`` -- still comfortable on a 24 GB GPU.

Solver
------

The per-voxel inner loop is the shared ``_varcomp`` AI-REML engine:
size-dispatched fixed-effect algebra (closed-form for ``p <= 2``,
``LU`` for ``p > 2`` -- no per-voxel cuSOLVER ``potrf`` / ``syevd``)
and *analytic* REML score + average-information curvature (no
second-order autodiff through a Cholesky).  See ``_varcomp.py`` for
the derivation.  The ``ZZ^T`` eigendecomposition that supplies the
diagonalising basis is the one shared, one-off ``safe_eigh`` call.

Differentiability
-----------------

Implementation uses ``jax.lax.scan`` over a fixed number of
Newton steps; each step is fully differentiable, so backward-
mode AD through the fit works (unrolled gradient through the
scan).  At ``n_iter = 20`` the unrolled grad has ~20 stacked
sub-graphs -- a real memory cost.  For applications that
differentiate through the fit (e.g., differentiable model
selection), pass a smaller ``n_iter`` or wait for the implicit-
function-theorem VJP follow-up.

References
----------
- Lindstrom, M. J., & Bates, D. M. (1990). Newton-Raphson and
  EM algorithms for linear mixed-effects models.  JASA 83.
- Lippert, C., Listgarten, J., et al. (2011). FaST linear mixed
  models for genome-wide association studies. Nat. Methods 8.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, NamedTuple, Optional, Tuple, Union, cast

import jax
import jax.numpy as jnp
from jax.scipy.special import betainc
from jaxtyping import Array, Float, Int

from .._batching import blocked_vmap
from .._smalllinalg import small_inv_logdet, sym_eig_jacobi
from ._corr import CorrSpec, resolve_corr
from ._corrfit import CorrLMEResult
from ._varcomp import VarCompSpec, fit_varcomp_diagonal, varcomp_inference

__all__ = [
    'REMLResult',
    'LMEResult',
    'NestedLMEResult',
    'LMEContrast',
    'LMEFContrast',
    'reml_fit',
    'lme_fit',
    'lme_t_contrast',
    'lme_f_contrast',
]

Structure = Literal['unstructured', 'diagonal']


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class REMLResult:
    """Per-voxel REML fit output.

    Attributes
    ----------
    theta_hat
        ``(V, 2)`` -- ``[log(sigma_b^2), log(sigma_e^2)]`` per voxel.
        Take ``jnp.exp`` for the natural variance scale.
    beta_hat
        ``(V, p)`` -- fixed-effect estimates per voxel.
    log_lik
        ``(V,)`` -- profile REML log-likelihood at the fit.
    fixed_cov
        ``(V, p, p)`` -- ``Cov(beta_hat) = (X^T V^{-1} X)^{-1}`` at the fit.
        The fixed-effect covariance for a contrast test (``lme_t_contrast``).
    theta_cov
        ``(V, 2, 2)`` -- ``Cov(theta_hat)`` (inverse average-information), for
        the Satterthwaite denominator degrees of freedom.
    grad_m
        ``(V, 2, p, p)`` -- the contrast-independent Satterthwaite gradient
        tensors ``M_k`` (see ``_varcomp.varcomp_inference``).
    """

    theta_hat: Float[Array, 'V 2']
    beta_hat: Float[Array, 'V p']
    log_lik: Float[Array, 'V']
    fixed_cov: Float[Array, 'V p p']
    theta_cov: Float[Array, 'V 2 2']
    grad_m: Float[Array, 'V 2 p p']

    @property
    def sigma_b_sq(self) -> Float[Array, 'V']:
        return jnp.exp(self.theta_hat[..., 0])

    @property
    def sigma_e_sq(self) -> Float[Array, 'V']:
        return jnp.exp(self.theta_hat[..., 1])

    def tree_flatten(
        self,
    ) -> Tuple[Tuple[Array, ...], None]:
        return (
            self.theta_hat,
            self.beta_hat,
            self.log_lik,
            self.fixed_cov,
            self.theta_cov,
            self.grad_m,
        ), None

    @classmethod
    def tree_unflatten(
        cls, _aux: None, children: Tuple[Any, ...]
    ) -> REMLResult:
        return cls(*children)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _default_theta_init(
    y_rot: Float[Array, '... N'],
    V_basis_diag: Float[Array, 'K N'],
) -> Float[Array, '... K']:
    """Initialise log-variance components from the empirical data.

    Heuristic: start with a small random-effect variance and most
    of the variance assigned to the residual component.  This
    starting point is closer to the "no random effects" boundary
    than to the "all random" boundary, which makes Newton more
    likely to converge to the true optimum without overshooting
    when the true ``sigma_b^2`` is small.

    Concretely: ``sigma_e^2_init = 0.5 * var(y)``,
    ``sigma_b^2_init = 0.1 * var(y) / mean(basis)`` (so the
    contribution to the diagonal is comparable in magnitude to
    ``sigma_e^2_init``), floored at ``1e-6`` so the log is finite.

    This is the default for ``reml_fit`` only; ``flame_two_level``
    supplies its own initialiser (``_flame_default_log_init``).
    Caveat: for a random-effect design whose basis (``ZZ^T`` spectrum)
    has a large mean, ``sigma_b^2_init`` can hit the ``1e-6`` floor and
    start Newton near the ``sigma_b^2 -> 0`` boundary; the
    damped + backtracked iteration still climbs off it given enough
    iterations, but raise ``n_iter`` for such designs.
    """
    y_var = jnp.var(y_rot, axis=-1, keepdims=True)  # (..., 1)
    K = V_basis_diag.shape[0]
    # Average diagonal contribution of each basis matrix to V.
    basis_scale = jnp.mean(V_basis_diag, axis=-1)  # (K,)
    basis_scale = jnp.where(basis_scale > 1e-12, basis_scale, 1.0)
    # Allocate var(y) primarily to the LAST component (typically the
    # residual identity); 10% to the others.
    weights = jnp.full((K,), 0.1, dtype=y_var.dtype).at[-1].set(0.5)
    weights = weights / jnp.sum(weights)
    # Per-component initial variance: var(y) * weight / basis_scale.
    init = jnp.log(
        jnp.maximum(y_var * weights / basis_scale, 1e-6),
    )
    return init


def _lowrank_theta_init(
    Y: Float[Array, 'V N'],
    s2: Float[Array, 'q'],
    n: int,
) -> Float[Array, 'V 2']:
    """Low-rank counterpart of ``_default_theta_init``.

    Reproduces the dense heuristic's split (10% of ``var(y)`` to the random
    component, 50% to the residual, normalised) using the *full* basis scale
    ``mean_N(lambda) = sum(s2) / N`` -- so the low-rank fit starts from the same
    ``theta`` as the dense path and converges to the same optimum.
    """
    y_var = jnp.var(Y, axis=-1, keepdims=True)  # (V, 1)
    basis_scale_b = jnp.maximum(jnp.sum(s2) / n, 1e-12)  # mean_N(lambda)
    w = jnp.asarray([0.1, 0.5], dtype=Y.dtype)
    w = w / jnp.sum(w)
    sb2 = jnp.maximum(y_var * w[0] / basis_scale_b, 1e-6)
    se2 = jnp.maximum(y_var * w[1], 1e-6)
    return jnp.concatenate([jnp.log(sb2), jnp.log(se2)], axis=-1)  # (V, 2)


def _reml_fit_lowrank(
    Y: Float[Array, 'V N'],
    X: Float[Array, 'N p'],
    Z: Float[Array, 'N q'],
    theta_init: Optional[Float[Array, 'V 2']],
    n_iter: int,
    damping: float,
    block: Optional[int],
) -> REMLResult:
    """FaST-LMM low-rank REML: ``q x q`` eig of ``Z^T Z`` instead of ``N x N``.

    Requires ``q <= N`` with ``Z`` of full column rank (the usual random-effect
    design).  ``ZZ^T = U_r diag(s2) U_r^T`` with ``U_r = Z W / sqrt(s2)``
    (``N x q``), and the ``N - q`` null directions enter only through per-voxel
    Gram aggregates -- never an ``N x N`` factor (see ``_lowrank``).
    """
    from ...linalg._solver import safe_eigh
    from ._lowrank import fit_lowrank_reml, lowrank_inference

    n, q = Z.shape
    if q > n:
        raise ValueError(
            f'reml_fit(low_rank=True): needs q <= N, got q={q}, N={n}; the '
            'low-rank form assumes a tall random-effect design.'
        )

    ZtZ = Z.T @ Z  # (q, q) -- the cheap factor
    ZtZ = 0.5 * (ZtZ + ZtZ.T)
    evals, W = safe_eigh(ZtZ)
    s2 = jnp.clip(evals, 0.0, None)  # (q,) range spectrum (ZZ^T eigenvalues)
    s = jnp.sqrt(jnp.clip(s2, jnp.finfo(Y.dtype).tiny, None))
    u_r = (Z @ W) / s[None, :]  # (N, q) range left-singular vectors

    y_range = Y @ u_r  # (V, q)
    x_range = u_r.T @ X  # (q, p)
    # Null-space Gram aggregates: X^T X - X_r^T X_r etc. (the orthogonal
    # complement of the range, summarised -- no per-null-coordinate work).
    gxx = X.T @ X - x_range.T @ x_range  # (p, p) shared
    gxy = Y @ X - y_range @ x_range  # (V, p): X^T y_v - X_r^T y_r,v
    gyy = jnp.sum(Y * Y, axis=-1) - jnp.sum(y_range * y_range, axis=-1)  # (V,)
    n0 = n - q

    if theta_init is None:
        theta_init = _lowrank_theta_init(Y, s2, n)

    spec = VarCompSpec.reml(n_iter=n_iter, damping=damping)
    theta_hat, beta_hat, log_lik = fit_lowrank_reml(
        y_range,
        x_range,
        s2,
        gxx,
        gxy,
        gyy,
        theta_init,
        n0,
        spec=spec,
        block=block,
    )
    p = X.shape[-1]

    def _inf(
        y: Float[Array, 'q'],
        gx: Float[Array, 'p'],
        gy: Float[Array, ''],
        th: Float[Array, '2'],
    ) -> Tuple[
        Float[Array, 'p p'], Float[Array, '2 2'], Float[Array, '2 p p']
    ]:
        return lowrank_inference(th, y, x_range, s2, gxx, gx, gy, n0, p, spec)

    fixed_cov, theta_cov, grad_m = cast(
        Tuple[Array, Array, Array],
        blocked_vmap(_inf, (y_range, gxy, gyy, theta_hat), block=block),
    )
    return REMLResult(
        theta_hat=theta_hat,
        beta_hat=beta_hat,
        log_lik=log_lik,
        fixed_cov=fixed_cov,
        theta_cov=theta_cov,
        grad_m=grad_m,
    )


# ---------------------------------------------------------------------------
# Fixed-effect contrast inference (Satterthwaite)
# ---------------------------------------------------------------------------


class LMEContrast(NamedTuple):
    """Per-voxel mixed-model t-contrast result.

    Attributes
    ----------
    effect
        ``(V,)`` -- the contrast estimate ``c^T beta_hat``.
    se
        ``(V,)`` -- its standard error ``sqrt(c^T Cov(beta_hat) c)``.
    t
        ``(V,)`` -- the t-statistic ``effect / se``.
    df
        ``(V,)`` -- Satterthwaite denominator degrees of freedom.
    p_value
        ``(V,)`` -- two-sided p-value from ``t`` on ``df`` degrees of freedom.
    """

    effect: Float[Array, 'V']
    se: Float[Array, 'V']
    t: Float[Array, 'V']
    df: Float[Array, 'V']
    p_value: Float[Array, 'V']


def lme_t_contrast(
    result: REMLResult,
    contrast: Float[Array, 'p'],
    *,
    dof: str = 'satterthwaite',
) -> LMEContrast:
    """Per-voxel t-test of a mixed-model fixed-effect contrast ``c^T beta = 0``.

    The standard error is ``sqrt(c^T (X^T V^{-1} X)^{-1} c)`` from the fitted
    ``result.fixed_cov``; the denominator degrees of freedom are
    **Satterthwaite** (the default, ``dof='satterthwaite'``):

        df = 2 (c^T C c)^2 / (g^T Cov(theta_hat) g),
        g_k = d(c^T C c)/dtheta_k = w^T M_k w,   w = C c

    using the per-voxel ``theta_cov`` / ``grad_m`` the fit already formed
    (``_varcomp.varcomp_inference``).  cuSOLVER-free and differentiable.  Kenward-
    Roger (a vcov bias correction) is the heavier follow-up (v3 §1.3, Tier-2).

    Parameters
    ----------
    result
        A :class:`REMLResult` from ``reml_fit``.
    contrast
        ``(p,)`` contrast vector ``c`` over the fixed effects.

    Returns
    -------
    ``LMEContrast`` ``(effect, se, t, df, p_value)``.
    """
    if dof != 'satterthwaite':
        raise ValueError(
            f'lme_t_contrast: dof={dof!r}; only "satterthwaite" is implemented '
            '(Kenward-Roger is v3 Tier-2).'
        )
    c = jnp.asarray(contrast, dtype=result.beta_hat.dtype)
    effect = result.beta_hat @ c  # (V,)
    w = jnp.einsum('vij,j->vi', result.fixed_cov, c)  # C c, (V, p)
    var = jnp.clip(jnp.einsum('vi,i->v', w, c), 1e-30, None)  # c^T C c
    se = jnp.sqrt(var)
    t = effect / se

    # Satterthwaite: g_k = w^T M_k w; df = 2 var^2 / (g^T theta_cov g).
    g = jnp.einsum('vi,vkij,vj->vk', w, result.grad_m, w)  # (V, K)
    denom = jnp.clip(
        jnp.einsum('vk,vkl,vl->v', g, result.theta_cov, g), 1e-30, None
    )
    df = 2.0 * var * var / denom
    x = df / (df + t * t)
    p_value = betainc(0.5 * df, 0.5, x)  # two-sided
    return LMEContrast(effect=effect, se=se, t=t, df=df, p_value=p_value)


class LMEFContrast(NamedTuple):
    """Per-voxel mixed-model F-contrast result.

    Attributes
    ----------
    f
        ``(V,)`` -- the Wald F-statistic
        ``(C beta)^T (C Cov(beta) C^T)^{-1} (C beta) / L`` for the ``L``-row
        contrast matrix ``C``.
    df1
        ``(V,)`` -- numerator degrees of freedom ``= L`` (the contrast rank;
        constant across voxels, returned per-voxel for a uniform record).
    df2
        ``(V,)`` -- Satterthwaite (Fai-Cornelius) denominator degrees of freedom.
    p_value
        ``(V,)`` -- upper-tail p-value of ``f`` on ``(df1, df2)``.
    """

    f: Float[Array, 'V']
    df1: Float[Array, 'V']
    df2: Float[Array, 'V']
    p_value: Float[Array, 'V']


def lme_f_contrast(
    result: REMLResult,
    contrast: Float[Array, 'L p'],
    *,
    dof: str = 'satterthwaite',
) -> LMEFContrast:
    """Per-voxel F-test of a multi-row mixed-model contrast ``C beta = 0``.

    For an ``L``-row contrast matrix ``C`` (a single ``(p,)`` row is promoted to
    ``L = 1``), the Wald F-statistic is

        F = (C beta)^T (C Cov(beta) C^T)^{-1} (C beta) / L

    with ``Cov(beta) = (X^T V^{-1} X)^{-1}`` from ``result.fixed_cov``.  The
    denominator degrees of freedom use the **Fai-Cornelius / lmerTest**
    multivariate-Satterthwaite construction (``dof='satterthwaite'``, the
    default):

    1. eigendecompose ``M = C Cov(beta) C^T = P diag(d) P^T`` (a small ``L x L``
       symmetric solve, cuSOLVER-free ``sym_eig_jacobi``);
    2. each eigendirection ``l_m = (P^T C)_m`` is an independent 1-df contrast
       with variance ``d_m`` and a per-direction Satterthwaite
       ``nu_m = 2 d_m^2 / (g_m^T Cov(theta) g_m)``, ``g_{m,k} = w_m^T M_k w_m``,
       ``w_m = Cov(beta) l_m`` (the same ``M_k`` / ``Cov(theta)`` the fit formed,
       ``result.grad_m`` / ``result.theta_cov``);
    3. combine by the E-method: ``E = sum_m nu_m / (nu_m - 2)`` over
       ``nu_m > 2``, and ``df2 = 2 E / (E - L)``.

    For ``L = 1`` this collapses **exactly** to ``lme_t_contrast`` (``F = t^2``,
    ``df2`` the t-Satterthwaite df, same p-value).  The F-statistic is
    differentiable; the denominator df is not (the iterative eigensolver feeds it
    under ``stop_gradient`` -- a model-selection-style scalar, per v3 §1.3).
    Kenward-Roger (``dof='kr'``) is the heavier Tier-2 follow-up.

    Parameters
    ----------
    result
        A :class:`REMLResult` from ``reml_fit`` (the R1 FaST-LMM path; carries
        the ``fixed_cov`` / ``theta_cov`` / ``grad_m`` inference fields).
    contrast
        ``(L, p)`` contrast matrix (or a ``(p,)`` single-row contrast).

    Returns
    -------
    ``LMEFContrast`` ``(f, df1, df2, p_value)``.
    """
    if dof != 'satterthwaite':
        raise ValueError(
            f'lme_f_contrast: dof={dof!r}; only "satterthwaite" is implemented '
            '(Kenward-Roger is v3 Tier-2).'
        )
    if not isinstance(result, REMLResult):
        raise TypeError(
            'lme_f_contrast: needs a REMLResult from reml_fit (the R1 path); '
            'the R2 block-Woodbury LMEResult does not yet surface the '
            'fixed-effect inference fields.'
        )
    dtype = result.beta_hat.dtype
    c_mat = jnp.atleast_2d(jnp.asarray(contrast, dtype=dtype))  # (L, p)
    n_rows = c_mat.shape[0]
    tiny = jnp.finfo(dtype).tiny

    def per_voxel(
        beta: Float[Array, 'p'],
        sigma: Float[Array, 'p p'],
        theta_cov: Float[Array, 'K K'],
        grad_m: Float[Array, 'K p p'],
    ) -> Tuple[Float[Array, ''], Float[Array, '']]:
        cb = c_mat @ beta  # (L,)
        m_cov = c_mat @ sigma @ c_mat.T  # (L, L) = C Cov(beta) C^T
        m_inv, _ = small_inv_logdet(m_cov, n_rows)
        f = (cb @ m_inv @ cb) / n_rows  # the Wald F (differentiable)

        # Denominator df (Fai-Cornelius E-method) -- not differentiated through.
        d, eigvecs = sym_eig_jacobi(jax.lax.stop_gradient(m_cov), n_rows)
        l_proj = eigvecs.T @ c_mat  # (L, p): per-eigendirection contrasts
        w = l_proj @ sigma  # (L, p): w_m = Cov(beta) l_m
        g = jnp.einsum('mp,kpq,mq->mk', w, grad_m, w)  # (L, K)
        denom = jnp.einsum('mk,kl,ml->m', g, theta_cov, g)  # (L,)
        d_safe = jnp.clip(d, tiny, None)
        nu = 2.0 * d_safe * d_safe / jnp.clip(denom, tiny, None)  # (L,)
        ratio = jnp.where(nu > 2.0, nu / (nu - 2.0), 0.0)
        e_sum = jnp.sum(ratio)
        # E-method df2 = 2E/(E-L) is only valid when E > L; in the degenerate
        # small-sample / near-boundary case (no eigendirection has nu > 2, so
        # E <= L) it would give df2 <= 0 and a NaN F p-value.  Fall back to a
        # conservative df2 = 1 there (wide tails -> larger p), keeping betainc
        # well-defined.
        df2 = jnp.where(
            e_sum > n_rows,
            2.0 * e_sum / jnp.maximum(e_sum - n_rows, tiny),
            1.0,
        )
        return f, df2

    f, df2 = jax.vmap(per_voxel)(
        result.beta_hat,
        result.fixed_cov,
        result.theta_cov,
        result.grad_m,
    )
    df1 = jnp.full_like(f, float(n_rows))
    x = df2 / (df2 + df1 * f)
    p_value = betainc(0.5 * df2, 0.5 * df1, x)  # upper-tail F survival
    return LMEFContrast(f=f, df1=df1, df2=df2, p_value=p_value)


def reml_fit(
    Y: Float[Array, 'V N'],
    X: Float[Array, 'N p'],
    Z: Float[Array, 'N q'],
    *,
    theta_init: Optional[Float[Array, 'V 2']] = None,
    n_iter: int = 20,
    damping: float = 1e-6,
    block: Optional[int] = None,
    low_rank: bool = False,
) -> REMLResult:
    """Voxelwise variance-components REML fit.

    Fits the LME

    .. code::

        y_v = X beta_v + Z b_v + eps_v
        b_v ~ N(0, sigma_b^2 I_q)
        eps_v ~ N(0, sigma_e^2 I_N)

    independently for each voxel, sharing the fixed-effect design
    ``X`` and the random-effect design ``Z`` across voxels.  Uses
    the FaST-LMM spectral trick: eigendecompose ``ZZ^T`` once,
    rotate ``y`` and ``X`` into the diagonalising basis, then
    Newton-score on the variance components in log-space.

    Parameters
    ----------
    Y
        Response tensor, ``(V, N)`` -- ``V`` voxels, ``N`` subjects
        (or observations).
    X
        Fixed-effect design, ``(N, p)``.  Shared across voxels.
    Z
        Random-effect design, ``(N, q)``.  Shared across voxels.
        The random-effect covariance is ``sigma_b^2 I_q`` (single
        variance component); pass each component's design matrix
        column to ``Z`` to model multiple random effects (currently
        all share ``sigma_b^2``; for separate variance components
        per random effect, the lower-level ``_varcomp`` core accepts a
        general ``(K, N)`` basis).
    theta_init
        Per-voxel initial log-variances, ``(V, 2)``.  Defaults to a
        heuristic split of empirical variance.
    n_iter
        Fixed Newton-scoring iteration count.  Default ``20`` --
        typically converges in 5-10 for well-conditioned data.
    damping
        Levenberg-Marquardt-style damping on the average-information
        matrix.  Default ``1e-6``; raise if Newton steps are unstable
        near boundaries.
    block
        Optional voxel-block size: cap the number of voxels whose
        per-Newton intermediates are live at once, bounding peak memory
        on brain-scale ``V``.  ``None`` (default) is a single ``vmap``
        over all voxels (identical numerics and HLO to before).
    low_rank
        Use the FaST-LMM **low-rank** decomposition: a ``q x q`` eig of
        ``Z^T Z`` (``O(N q^2 + q^3)``) instead of the dense ``N x N`` eig of
        ``ZZ^T`` (``O(N^3)``).  Requires ``q <= N`` with ``Z`` of full column
        rank (the usual tall random-effect design); the ``N - q`` null
        directions enter only through per-voxel Gram aggregates.  Default
        ``False`` (the dense path, bit-for-bit unchanged); set ``True`` for
        brain-scale cohorts where ``q << N`` -- it matches the dense fit to the
        iterative tolerance.

    Returns
    -------
    ``REMLResult`` with per-voxel ``theta_hat`` (log-variances),
    ``beta_hat`` (fixed effects), and ``log_lik``.

    Notes
    -----
    Eigendecomposition of ``ZZ^T`` uses ``safe_eigh`` (cuSolver-
    robust fallback) -- the one shared, one-off solver call.  The
    per-voxel inner loop (``_varcomp``) makes no cuSOLVER call: the
    fixed-effect algebra is closed-form for ``p <= 2`` and ``LU``-based
    for ``p > 2``, and the REML derivatives are analytic.
    """
    from ...linalg._solver import safe_eigh

    n = X.shape[0]
    if Z.shape[0] != n:
        raise ValueError(
            f'reml_fit: Z.shape[0]={Z.shape[0]} must match X.shape[0]={n}.'
        )
    if Y.shape[-1] != n:
        raise ValueError(
            f'reml_fit: Y.shape[-1]={Y.shape[-1]} must match N={n}.'
        )

    if low_rank:
        return _reml_fit_lowrank(Y, X, Z, theta_init, n_iter, damping, block)

    # Eigendecompose ZZ^T (shared across voxels).
    ZZt = Z @ Z.T
    ZZt = 0.5 * (ZZt + ZZt.T)  # symmetrise against drift
    eigvals, U = safe_eigh(ZZt)
    # Clamp negative eigenvalues from roundoff.
    lambdas = jnp.clip(eigvals, 0.0, None)

    # Rotate Y (per-voxel) and X (shared).
    Y_rot = Y @ U  # (V, N)
    X_rot = U.T @ X  # (N, p)

    # V_basis_diag = (lambdas, ones) -- shared across voxels.
    V_basis_diag = jnp.stack(
        [lambdas, jnp.ones_like(lambdas)],
        axis=0,
    )

    if theta_init is None:
        theta_init = _default_theta_init(Y_rot, V_basis_diag)

    spec = VarCompSpec.reml(n_iter=n_iter, damping=damping)
    theta_hat, beta_hat, log_lik = fit_varcomp_diagonal(
        Y_rot,
        X_rot,
        V_basis_diag,
        theta_init,
        spec=spec,
        block=block,
    )
    p = X.shape[-1]
    zero_off = jnp.zeros(n, dtype=Y.dtype)

    def _inf(
        y: Float[Array, 'N'], th: Float[Array, '2']
    ) -> Tuple[
        Float[Array, 'p p'], Float[Array, '2 2'], Float[Array, '2 p p']
    ]:
        return varcomp_inference(th, y, X_rot, V_basis_diag, zero_off, p, spec)

    fixed_cov, theta_cov, grad_m = cast(
        Tuple[Array, Array, Array],
        blocked_vmap(_inf, (Y_rot, theta_hat), block=block),
    )
    return REMLResult(
        theta_hat=theta_hat,
        beta_hat=beta_hat,
        log_lik=log_lik,
        fixed_cov=fixed_cov,
        theta_cov=theta_cov,
        grad_m=grad_m,
    )


# ---------------------------------------------------------------------------
# The structure-dispatched LME (v3 §1.1): cheapest exact solver per random spec
# ---------------------------------------------------------------------------


class NestedLMEResult(NamedTuple):
    """Nested two-level mixed-model fit output (the ``lme_fit`` R3 path).

    Attributes
    ----------
    beta_hat
        ``(V, p)`` fixed-effect estimates.
    var_outer
        ``(V,)`` outer random-intercept variance ``sigma1^2`` (``(1 | g1)``).
    var_inner
        ``(V,)`` inner (nested) random-intercept variance ``sigma2^2``
        (``(1 | g1:g2)``).
    sigma_e_sq
        ``(V,)`` residual variance.
    log_lik
        ``(V,)`` profile REML log-likelihood.
    tier
        ``'R3'`` (the telescoping-Woodbury nested solver).
    """

    beta_hat: Float[Array, 'V p']
    var_outer: Float[Array, 'V']
    var_inner: Float[Array, 'V']
    sigma_e_sq: Float[Array, 'V']
    log_lik: Float[Array, 'V']
    tier: str


class LMEResult(NamedTuple):
    """General mixed-model fit output (the ``lme_fit`` dispatcher result).

    Uniform across the dispatch tiers: the random-effect covariance is always a
    full ``r x r`` matrix (``r = 1`` for a scalar random effect, ``r >= 2`` for a
    correlated / diagonal random slope).

    Attributes
    ----------
    beta_hat
        ``(V, p)`` fixed-effect estimates.
    cov_re
        ``(V, r, r)`` random-effect covariance ``G`` (the within-group
        covariance of the random effects).
    sigma_e_sq
        ``(V,)`` residual variance.
    log_lik
        ``(V,)`` profile REML log-likelihood.
    tier
        Which solver the dispatcher ran: ``'R1'`` (FaST-LMM spectral, one scalar
        factor) or ``'R2'`` (block-Woodbury, one correlated / diagonal factor).
    """

    beta_hat: Float[Array, 'V p']
    cov_re: Float[Array, 'V r r']
    sigma_e_sq: Float[Array, 'V']
    log_lik: Float[Array, 'V']
    tier: str


def lme_fit(
    Y: Float[Array, 'V N'],
    X: Float[Array, 'N p'],
    *,
    group: Int[Array, 'N'],
    z: Optional[Float[Array, 'N r']] = None,
    inner: Optional[Int[Array, 'N']] = None,
    corr: Optional[Union[str, CorrSpec]] = None,
    time: Optional[Float[Array, 'N']] = None,
    structure: Structure = 'unstructured',
    n_iter: int = 20,
    damping: float = 1e-6,
    block: Optional[int] = None,
    low_rank: bool = False,
) -> Union[REMLResult, LMEResult, NestedLMEResult, CorrLMEResult]:
    """Voxelwise mixed model, dispatched to the cheapest exact solver.

    A single grouping factor ``group`` with per-observation random covariates
    ``z`` (``None`` -> a scalar random intercept).  The dispatcher honours the
    **performance-preserving** invariant (v3 §0.1): it routes to the cheapest
    solver that is *exact* for the given random structure, so the shipped fast
    paths stay the realised path for the cases they already serve.

    Dispatch ladder
    ---------------

    - **R1** -- one **scalar** random effect (``z`` is ``None`` or a single
      column): the FaST-LMM spectral ``reml_fit`` (dense, or q-rank via
      ``low_rank=``).  Returns a :class:`REMLResult` (with the §1.3
      fixed-effect inference fields).  *Bit-for-bit the shipped path.*
    - **R2** -- one **correlated / diagonal** random effect (``z`` has ``r >= 2``
      columns, e.g. ``[1, x]`` for ``(1 + x | g)``): the block-Woodbury REML
      (``_blockwoodbury``) -- per-group ``r x r`` Woodbury, no ``N x N``
      intermediate, cuSOLVER-free.  Returns an :class:`LMEResult`.
      ``structure='unstructured'`` fits a full ``r x r`` ``G``;
      ``structure='diagonal'`` constrains ``G`` to be diagonal -- the
      uncorrelated ``(x || g)`` random effect (independent intercept and slope),
      fitting only the ``r`` variances (``nt = r + 1`` parameters).

    - **R3** -- a **nested** two-level hierarchy ``(1 | g1/g2)`` (pass the inner
      factor as ``inner``; ``z`` must be ``None``): the telescoping-Woodbury
      nested solver (``_nested``) over the three variance components
      ``[sigma1^2, sigma2^2, sigma_e^2]`` -- block-diagonal across ``g1``,
      Sherman-Morrison at each nested level, no ``N x N`` intermediate,
      cuSOLVER-free.  Returns a :class:`NestedLMEResult`.
    - **R2 + corr** -- a random effect **and** a structured within-group residual
      ``Cov(eps) = sigma_e^2 R(rho)`` (pass ``corr=`` and, for ``car1``,
      ``time=``).  Whitening by ``R`` reduces each group to a standard
      block-Woodbury, so the per-group Woodbury algebra is reused with ``rho``
      joining the REML ``theta``; cuSOLVER-free.  Returns a
      :class:`CorrLMEResult`.  ``z=None`` is the random intercept; ``z`` a random
      slope (``structure='diagonal'`` for ``(x || g)``).

    Crossed / multiple grouping factors (R4) are the general sparse-MME follow-up
    (v3 §1.1, Tier-2).

    Parameters
    ----------
    Y, X
        ``(V, N)`` responses and ``(N, p)`` shared fixed-effect design.
    group
        ``(N,)`` integer group labels (``0 .. M-1``).  The **outer** factor
        ``g1`` when ``inner`` is given (R3).
    z
        ``(N, r)`` per-observation random covariates; ``None`` is the scalar
        random intercept (``r = 1``, ``z = 1``).
    inner
        ``(N,)`` **nested** inner factor ``g2`` (each ``(group, inner)`` pair is
        a sublevel ``g1:g2``).  When given, routes to the R3 nested solver
        (``z`` must be ``None``); ``None`` (default) keeps the single-factor
        R1/R2 dispatch.
    corr
        Within-group residual correlation structure (``'ar1'`` / ``'car1'`` /
        ``'cs'`` or a :class:`CorrSpec`).  When given, routes to the R2 + corr
        whitened block-Woodbury (a random effect plus a structured residual);
        ``None`` (default) is the ``sigma_e^2 I`` residual.
    time
        ``(N,)`` observation times for a ``car1`` residual (see ``corr``).
    structure
        ``'unstructured'`` (full ``r x r`` ``G``, ``(1 + x | g)``) or
        ``'diagonal'`` (independent variance components, ``(x || g)`` -- the
        off-diagonal of ``G`` is held at zero).  Both run the R2 block-Woodbury.
    n_iter, damping, block, low_rank
        Newton iterations, AI damping, voxel-block chunking, and (R1) the q-rank
        toggle.

    Returns
    -------
    ``REMLResult`` (R1), ``LMEResult`` (R2), ``NestedLMEResult`` (R3), or
    ``CorrLMEResult`` (R2 + corr).
    """
    group = jnp.asarray(group)
    n_groups = int(jnp.max(group)) + 1
    r = 1 if z is None else jnp.asarray(z).shape[-1]

    if corr is not None:
        # R2 + corr: a random effect *and* a structured within-group residual.
        if inner is not None:
            raise NotImplementedError(
                'lme_fit: a structured residual (corr=) on a nested design '
                '(inner=) is a further composition; not yet supported.'
            )
        from ._corrfit import fit_corr_lme

        z_design = (
            jnp.ones((group.shape[0], 1), dtype=Y.dtype)
            if z is None
            else jnp.asarray(z, dtype=Y.dtype)
        )
        return fit_corr_lme(
            Y,
            X,
            z_design,
            group,
            resolve_corr(corr),
            time=None if time is None else jnp.asarray(time),
            diagonal=structure == 'diagonal',
            n_iter=n_iter,
            damping=damping,
            block=block,
        )

    if inner is not None:
        # R3: nested two-level hierarchy (1 | g1/g2).
        if z is not None:
            raise ValueError(
                'lme_fit: nested fit (inner=...) takes scalar intercepts at '
                'both levels; pass z=None (random slopes nested under a factor '
                'are a Tier-2 follow-up).'
            )
        from ._nested import fit_nested_reml

        inner_arr = jnp.asarray(inner)
        var_y = jnp.var(Y, axis=-1, keepdims=True)  # (V, 1)
        w = jnp.asarray([0.25, 0.25, 0.5], dtype=Y.dtype)
        theta_init = jnp.log(jnp.maximum(var_y * w, 1e-6))  # (V, 3)
        spec = VarCompSpec.reml(n_iter=n_iter, damping=damping)
        theta_hat, beta_hat, log_lik = fit_nested_reml(
            Y, X, group, inner_arr, theta_init, spec=spec, block=block
        )
        return NestedLMEResult(
            beta_hat=beta_hat,
            var_outer=jnp.exp(theta_hat[:, 0]),
            var_inner=jnp.exp(theta_hat[:, 1]),
            sigma_e_sq=jnp.exp(theta_hat[:, 2]),
            log_lik=log_lik,
            tier='R3',
        )

    if r == 1:
        # R1: one scalar random effect -> the FaST-LMM spectral fast path.
        onehot = jax.nn.one_hot(group, n_groups, dtype=Y.dtype)  # (N, M)
        z_design = onehot if z is None else onehot * jnp.asarray(z)
        return reml_fit(
            Y,
            X,
            z_design,
            n_iter=n_iter,
            damping=damping,
            block=block,
            low_rank=low_rank,
        )

    # R2: one correlated / diagonal random effect -> block-Woodbury REML.
    from ._blockwoodbury import (
        _build_chol,
        _param_layout,
        fit_blockwoodbury_reml,
    )

    diagonal = structure == 'diagonal'
    z_arr = jnp.asarray(z, dtype=Y.dtype)
    layout = _param_layout(r, diagonal)
    diag_mask = jnp.asarray([i == j for (i, j) in layout], dtype=bool)  # (m,)
    var_y = jnp.var(Y, axis=-1)  # (V,)
    chol_diag = 0.5 * jnp.log(jnp.maximum(0.1 * var_y, 1e-6))  # (V,)
    chol = jnp.where(diag_mask[None, :], chol_diag[:, None], 0.0)  # (V, m)
    log_se2 = jnp.log(jnp.maximum(0.5 * var_y, 1e-6))[:, None]  # (V, 1)
    theta_init = jnp.concatenate([chol, log_se2], axis=1)  # (V, nt)

    spec = VarCompSpec.reml(n_iter=n_iter, damping=damping)
    theta_hat, beta_hat, log_lik = fit_blockwoodbury_reml(
        Y,
        X,
        z_arr,
        group,
        n_groups,
        theta_init,
        spec=spec,
        block=block,
        diagonal=diagonal,
    )
    cov_re = jax.vmap(
        lambda th: (
            _build_chol(th[:-1], r, diagonal)
            @ _build_chol(th[:-1], r, diagonal).T
        )
    )(theta_hat)  # (V, r, r)
    sigma_e_sq = jnp.exp(theta_hat[:, -1])
    return LMEResult(
        beta_hat=beta_hat,
        cov_re=cov_re,
        sigma_e_sq=sigma_e_sq,
        log_lik=log_lik,
        tier='R2',
    )
