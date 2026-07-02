# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
r"""
Mass-univariate generalised linear models.

:func:`glm_fit` fits the same design ``X`` (``N, p``) to every element (voxel /
vertex / fixel) of a response tensor ``Y`` (``V, N``) -- the mass-univariate
"one model per element, shared design" workload behind ModelArray's ``lm`` and
the parametric backbone of GAM (``gam.py``) and the permutation engine
(``inference/randomise``).

Two paths, one surface
----------------------

The fit is **IRLS-shaped**: a weighted-least-squares solve
:math:`(X^{\top} W X + S) \beta = X^{\top} W z` over a working response ``z`` and
working weights ``W``.  OLS, WLS, and the full exponential family are then the
same machinery at different ``(W, z)``:

- **Gaussian identity, no penalty** -- the dominant ``lm`` case -- takes a
  vectorised fast path: :math:`X^{\top} X` is *shared* across elements, so a single
  ``(p, p)`` inverse and a couple of matmuls fit all ``V`` elements at once
  (no per-element factorisation).
- **Any other family** runs Penalised IRLS per element (``vmap``), reusing the
  same cuSOLVER-free ``(p, p)`` solve.  ``gam.py`` calls the same inner solve
  with a non-zero penalty ``S``; here ``S = 0``.

Every ``(p, p)`` solve goes through ``linalg._smalllinalg.small_inv_logdet``
(closed-form for ``p <= 2``, hand-Cholesky + ``trsm`` for ``p > 2``) -- no
cuSOLVER custom-call, so the fit runs on the broken-cuSOLVER GPU.

Inference
---------

:func:`t_contrast` / :func:`f_contrast` return the per-element effect, standard
error, t / F statistic, and the (Student-t / F) p-value -- the ModelArray ``lm``
output columns.  p-values use the regularised incomplete beta
(``jax.scipy.special.betainc``), so they are exact and cuSOLVER-free.
:func:`r_squared` / :func:`deviance_explained` use the stored null-model deviance.

The exponential family is a frozen :class:`Family` value (a small record of pure
link / variance / deviance functions); :data:`GAUSSIAN` / :data:`BINOMIAL` /
:data:`POISSON` ship, and a custom family is just another ``Family(...)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import betainc, gammaincc
from jaxtyping import Array, Float, Int

from ..linalg._smalllinalg import small_inv_logdet
from ._batching import blocked_vmap
from ._family import (
    BINOMIAL,
    CLOGLOG_LINK,
    GAMMA,
    GAUSSIAN,
    IDENTITY_LINK,
    INVERSE_LINK,
    LOG_LINK,
    LOGIT_LINK,
    NEGBINOMIAL,
    POISSON,
    PROBIT_LINK,
    SQRT_LINK,
    TWEEDIE,
    Family,
    Link,
    negbinomial,
    resolve_family,
    resolve_link,
    tweedie,
)
from ._irls import fit_penalised_irls, irls_warm_start
from ._result import register_result

__all__ = [
    'Family',
    'Link',
    'IDENTITY_LINK',
    'LOG_LINK',
    'LOGIT_LINK',
    'PROBIT_LINK',
    'CLOGLOG_LINK',
    'SQRT_LINK',
    'INVERSE_LINK',
    'resolve_family',
    'resolve_link',
    'GAUSSIAN',
    'BINOMIAL',
    'POISSON',
    'GAMMA',
    'NEGBINOMIAL',
    'TWEEDIE',
    'negbinomial',
    'tweedie',
    'GLMResult',
    'glm_fit',
    'predict',
    't_contrast',
    'f_contrast',
    'sandwich_cov',
    'r_squared',
    'adj_r_squared',
    'deviance_explained',
    'log_likelihood',
    'aic',
    'bic',
    'compare_models',
]

_EPS = 1e-10


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@register_result(
    children=(
        'coef',
        'cov_unscaled',
        'dispersion',
        'deviance',
        'null_deviance',
        'log_lik',
    ),
    aux=('family', 'n_obs', 'rank'),
)
@dataclass(frozen=True)
class GLMResult:
    """Per-element GLM fit output.

    Attributes
    ----------
    coef
        ``(V, p)`` fixed-effect estimates.
    cov_unscaled
        Unscaled coefficient covariance :math:`(X^{\\top} W X)^{-1}`.  Shared
        ``(p, p)`` for the Gaussian-identity fast path; per-element
        ``(V, p, p)`` for the IRLS path.  Multiply by ``dispersion`` for the
        covariance.
    dispersion
        ``(V,)`` dispersion estimate (residual variance for Gaussian; ``1`` for
        binomial / Poisson unless overdispersion is estimated).
    deviance
        ``(V,)`` model deviance (residual sum of squares for Gaussian).
    null_deviance
        ``(V,)`` deviance of the intercept-only model (used by
        :func:`r_squared` / :func:`deviance_explained`).
    log_lik
        ``(V,)`` maximised log-likelihood (used by :func:`aic` / :func:`bic`
        and the likelihood-ratio test).
    """

    coef: Float[Array, 'V p']
    cov_unscaled: Float[Array, '... p p']
    dispersion: Float[Array, 'V']
    deviance: Float[Array, 'V']
    null_deviance: Float[Array, 'V']
    log_lik: Float[Array, 'V']
    family: Family
    n_obs: int
    rank: int

    @property
    def dof_resid(self) -> float:
        return float(self.n_obs - self.rank)

    @property
    def n_params(self) -> int:
        """Number of estimated parameters (coefficients + a dispersion)."""
        return self.rank + (0 if self.family.has_fixed_dispersion else 1)


# ---------------------------------------------------------------------------
# Fitting -- the OLS fast path and per-element penalised IRLS
# ---------------------------------------------------------------------------


def _ols_fit(
    Y: Float[Array, 'V N'],
    X: Float[Array, 'N p'],
    w: Float[Array, 'N'],
    ridge: float,
) -> Tuple[Float[Array, 'V p'], Float[Array, 'p p'], Float[Array, 'V']]:
    """Vectorised Gaussian-identity weighted least squares.

    Because the normal-equation matrix :math:`X^{\\top} W X` does not depend on
    the element, all ``V`` fits reduce to a single shared ``(p, p)`` inverse
    plus a couple of matrix multiplies, with no per-element factorisation.

    Parameters
    ----------
    Y
        ``(V, N)`` responses -- ``V`` elements, ``N`` observations.
    X
        ``(N, p)`` design matrix, shared across all elements.
    w
        ``(N,)`` prior observation weights (all ones for an unweighted fit).
    ridge
        L2 stabiliser added to the diagonal of :math:`X^{\\top} W X` before the
        inverse.

    Returns
    -------
    coef : Float[Array, 'V p']
        ``(V, p)`` weighted-least-squares coefficient estimates.
    xtwx_inv : Float[Array, 'p p']
        Shared ``(p, p)`` inverse
        :math:`(X^{\\top} W X + \\mathrm{ridge}\\,I)^{-1}` (the unscaled
        coefficient covariance).
    wrss : Float[Array, 'V']
        ``(V,)`` weighted residual sum of squares, one per element.
    """
    p = X.shape[-1]
    Xw = X * w[:, None]
    xtwx = Xw.T @ X
    xtwx = xtwx + ridge * jnp.eye(p, dtype=X.dtype)
    xtwx_inv, _ = small_inv_logdet(xtwx, p)
    coef = (Y @ Xw) @ xtwx_inv  # (V, p)
    resid = Y - coef @ X.T  # (V, N)
    wrss = jnp.sum(w[None, :] * resid * resid, axis=-1)  # (V,)
    return coef, xtwx_inv, wrss


def _pirls_one(
    y: Float[Array, 'N'],
    X: Float[Array, 'N p'],
    w: Float[Array, 'N'],
    penalty: Float[Array, 'p p'],
    family: Family,
    p: int,
    n_iter: int,
    ridge: float,
) -> Tuple[Float[Array, 'p'], Float[Array, 'p p'], Float[Array, '']]:
    """Fit a single element by penalised iteratively reweighted least squares.

    Warm-starts a coefficient vector and runs penalised IRLS through the shared
    fitting core.  A plain (unpenalised) GLM passes an all-zeros ``penalty``.

    Parameters
    ----------
    y
        ``(N,)`` response for this element.
    X
        ``(N, p)`` design matrix, shared across elements.
    w
        ``(N,)`` prior observation weights.
    penalty
        ``(p, p)`` smoothing penalty added to the normal equations (all zeros
        for a plain GLM).
    family
        Exponential family and link defining the link, variance, and deviance
        functions.
    p
        Number of design columns (coefficients).
    n_iter
        Number of IRLS iterations.
    ridge
        L2 stabiliser added to the normal-equation matrix.

    Returns
    -------
    beta : Float[Array, 'p']
        ``(p,)`` fitted coefficients for this element.
    cov_unscaled : Float[Array, 'p p']
        ``(p, p)`` unscaled coefficient covariance.
    deviance : Float[Array, '']
        Scalar model deviance for this element.
    """
    beta0 = irls_warm_start(
        y, X, family, penalty=penalty, ridge=ridge, prior_weights=w
    )
    beta, v, _, dev = fit_penalised_irls(
        y,
        X,
        family,
        penalty=penalty,
        beta0=beta0,
        n_iter=n_iter,
        ridge=ridge,
        prior_weights=w,
    )
    return beta, v, dev


def glm_fit(
    Y: Float[Array, 'V N'],
    X: Float[Array, 'N p'],
    *,
    family: Union[str, Family] = GAUSSIAN,
    weights: Optional[Float[Array, 'N']] = None,
    rank: Optional[int] = None,
    n_iter: int = 25,
    ridge: float = 0.0,
    block: Optional[int] = None,
) -> GLMResult:
    """Fit a mass-univariate GLM: shared design ``X``, per-element responses.

    Parameters
    ----------
    Y
        ``(V, N)`` responses -- ``V`` elements, ``N`` observations.
    X
        ``(N, p)`` design, shared across elements (include your own intercept
        column -- no intercept is added).
    family
        Exponential family + link: a built-in name (``'gaussian'`` /
        ``'binomial'`` / ``'poisson'``) or a :class:`Family` instance.  Default
        :data:`GAUSSIAN` identity, i.e. ordinary least squares.
    weights
        Optional ``(N,)`` prior observation weights (WLS / known precision).
        ``None`` is unweighted.
    rank
        Effective rank of ``X`` for the residual degrees of freedom (dispersion /
        SE / AIC).  ``None`` (default) detects it host-side (``matrix_rank``) when
        ``X`` is concrete, falling back to ``p`` under ``jax.jit``; pass it
        explicitly for a known rank-deficient design traced under ``jit``.
    n_iter
        IRLS iterations for non-Gaussian families (ignored on the OLS path).
    ridge
        Optional L2 stabiliser added to the normal-equation matrix.
    block
        Optional element-block size bounding peak memory on the per-element
        IRLS path (non-Gaussian families).  ``None`` (default) is a single
        ``vmap``; the OLS fast path is already vectorised and ignores it.

    Returns
    -------
    GLMResult
        A :class:`GLMResult` carrying the per-element coefficients, unscaled
        covariance, dispersion, model deviance, null deviance, and maximised
        log-likelihood.
    """
    family = resolve_family(family)
    n, p = X.shape
    if Y.shape[-1] != n:
        raise ValueError(
            f'glm_fit: Y.shape[-1]={Y.shape[-1]} must match N={n}.'
        )
    w = jnp.ones((n,), dtype=Y.dtype) if weights is None else weights

    # Effective rank for the residual dof (dispersion / SE / AIC): a rank-deficient
    # design (collinear columns) has rank < p, so n - p over-counts the dof -- the
    # ridge / pivot floor keeps the solve finite but the SEs would be wrong.
    # Detect it host-side when X is concrete; under jit (X a tracer) fall back to p
    # (pass `rank=` for a known-deficient design, as with glmm's n_groups).  A
    # genuinely full-rank design gives rank == p, so dof is byte-unchanged.
    if rank is not None:
        rank_eff = int(rank)
    elif isinstance(X, jax.core.Tracer):
        rank_eff = p
    else:
        rank_eff = int(np.linalg.matrix_rank(np.asarray(X)))

    # Null-model deviance: the intercept-only MLE is the **weighted** mean
    # mu0 = sum(w y)/sum(w) (the ordinary mean when unweighted), and the
    # deviance carries the same prior weights as the model deviance, so
    # r_squared / deviance_explained are a consistent ratio under weights=.
    # (Assumes a canonical link; a non-canonical-link null mean is not mean(y).)
    y_bar = (Y @ w / jnp.sum(w))[:, None]  # (V, 1) weighted mean
    null_dev = jnp.sum(
        w[None, :] * family.unit_deviance(Y, jnp.broadcast_to(y_bar, Y.shape)),
        axis=-1,
    )

    is_ols = family.name == 'gaussian'
    if is_ols:
        coef, cov_unscaled, wrss = _ols_fit(Y, X, w, ridge)
        dof = float(n - rank_eff)
        dispersion = wrss / dof if dof > 0 else wrss
        deviance = wrss
    else:
        penalty = jnp.zeros((p, p), dtype=X.dtype)
        coef, cov_unscaled, deviance = blocked_vmap(
            lambda y: _pirls_one(y, X, w, penalty, family, p, n_iter, ridge),
            (Y,),
            block=block,
        )
        if family.has_fixed_dispersion:
            dispersion = jnp.ones((Y.shape[0],), dtype=Y.dtype)
        else:
            dof = float(n - rank_eff)
            dispersion = deviance / dof if dof > 0 else deviance

    # Maximised log-likelihood.  Gaussian uses the MLE dispersion (RSS / N),
    # which is what the deviance-free AIC convention expects.
    fitted = family.linkinv(coef @ X.T)
    disp_ll = (deviance / n) if not family.has_fixed_dispersion else dispersion
    log_lik = jnp.sum(family.loglik(Y, fitted, disp_ll[:, None]), axis=-1)

    return GLMResult(
        coef=coef,
        cov_unscaled=cov_unscaled,
        dispersion=dispersion,
        deviance=deviance,
        null_deviance=null_dev,
        log_lik=log_lik,
        family=family,
        n_obs=int(n),
        rank=rank_eff,
    )


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


PredictType = Literal['response', 'link']


def predict(
    result: GLMResult,
    X: Float[Array, 'N p'],
    *,
    type: PredictType = 'response',
) -> Float[Array, 'V N']:
    """Per-element prediction on a (possibly new) design.

    Forms the linear predictor :math:`\\eta = X \\beta` for every element and,
    for ``type='response'``, maps it through the family's inverse link.

    Parameters
    ----------
    result
        A fitted :class:`GLMResult` supplying the per-element coefficients and
        the family (for the inverse link).
    X
        ``(N, p)`` design matrix to predict on (need not be the design used to
        fit; must share the coefficient layout).
    type
        ``'response'`` (default) applies the inverse link to return fitted
        values on the response scale; ``'link'`` returns the linear predictor
        :math:`\\eta = X \\beta` on the link scale.

    Returns
    -------
    Float[Array, 'V N']
        ``(V, N)`` predictions, one row per element over the ``N`` rows of
        ``X``.
    """
    eta = result.coef @ X.T
    if type == 'link':
        return eta
    if type == 'response':
        return result.family.linkinv(eta)
    raise ValueError(f"predict: type={type!r}; expected 'response' or 'link'.")


# ---------------------------------------------------------------------------
# Contrasts and inference
# ---------------------------------------------------------------------------


def _t_two_sided_sf(t: Array, df: float) -> Array:
    """Two-sided Student-t survival :math:`P(|T| > |t|)` via regularised beta.

    Parameters
    ----------
    t
        Observed t statistic(s), any shape.
    df
        Degrees of freedom.

    Returns
    -------
    Array
        Two-sided tail probability, matching the shape of ``t``.
    """
    x = df / (df + t * t)
    return betainc(0.5 * df, 0.5, x)


def _f_sf(f: Array, df1: float, df2: float) -> Array:
    """F survival :math:`P(F > f)` via the regularised incomplete beta.

    Parameters
    ----------
    f
        Observed F statistic(s), any shape (clipped at zero).
    df1
        Numerator degrees of freedom.
    df2
        Denominator degrees of freedom.

    Returns
    -------
    Array
        Upper-tail probability, matching the shape of ``f``.
    """
    x = df2 / (df2 + df1 * jnp.clip(f, 0.0, None))
    return betainc(0.5 * df2, 0.5 * df1, x)


def t_contrast(
    result: GLMResult,
    contrast: Float[Array, 'p'],
    *,
    cov: Optional[Float[Array, '... p p']] = None,
) -> Tuple[
    Float[Array, 'V'], Float[Array, 'V'], Float[Array, 'V'], Float[Array, 'V']
]:
    r"""Per-element t-test of a linear contrast :math:`c^{\top} \beta`.

    Computes, for every element, the contrast estimate, its standard error, the
    t statistic, and the two-sided p-value at ``result.dof_resid`` degrees of
    freedom (the ModelArray ``lm`` columns ``estimate`` / ``std.error`` /
    ``statistic`` / ``p.value``).

    Parameters
    ----------
    result
        A fitted :class:`GLMResult` supplying the coefficients, unscaled
        covariance, dispersion, and residual degrees of freedom.
    contrast
        ``(p,)`` contrast vector :math:`c`; the tested effect is
        :math:`c^{\top} \beta`.
    cov
        Optional coefficient covariance override, either a shared ``(p, p)`` or
        a per-element ``(V, p, p)`` matrix -- e.g. the robust output of
        :func:`sandwich_cov`.  When given, the variance is :math:`c^{\top}
        \Sigma\, c` taken directly (already on the variance scale).  The
        default (``None``) uses the model-based covariance
        :math:`\mathrm{dispersion} \cdot (X^{\top} W X)^{-1}`.

    Returns
    -------
    effect : Float[Array, 'V']
        ``(V,)`` contrast estimate :math:`c^{\top} \beta`.
    se : Float[Array, 'V']
        ``(V,)`` standard error of the contrast.
    t : Float[Array, 'V']
        ``(V,)`` t statistic ``effect / se``.
    p_value : Float[Array, 'V']
        ``(V,)`` two-sided p-value at ``result.dof_resid`` degrees of freedom.
    """
    c = jnp.asarray(contrast)
    effect = result.coef @ c  # (V,)
    if cov is None:
        var = result.dispersion * jnp.einsum(
            '...ij,i,j->...', result.cov_unscaled, c, c
        )
    else:
        var = jnp.einsum('...ij,i,j->...', cov, c, c)
    se = jnp.sqrt(jnp.clip(var, _EPS, None))
    t = effect / se
    p_value = _t_two_sided_sf(t, result.dof_resid)
    return effect, se, t, p_value


def f_contrast(
    result: GLMResult,
    contrast: Float[Array, 'm p'],
    *,
    cov: Optional[Float[Array, '... p p']] = None,
) -> Tuple[Float[Array, 'V'], Float[Array, 'V'], float, float]:
    r"""Per-element F-test of a multi-row contrast :math:`C \beta = 0`.

    Forms the Wald quadratic form of the ``m`` contrast rows against the fitted
    coefficients and returns an F statistic per element.

    Parameters
    ----------
    result
        A fitted :class:`GLMResult` supplying the coefficients, unscaled
        covariance, dispersion, and residual degrees of freedom.
    contrast
        ``(m, p)`` contrast matrix :math:`C`; the tested hypothesis is
        :math:`C \beta = 0`.
    cov
        Optional coefficient covariance override, either a shared ``(p, p)`` or
        a per-element ``(V, p, p)`` matrix -- e.g. the robust output of
        :func:`sandwich_cov`.  When given, the Wald quadratic form uses
        :math:`C \Sigma\, C^{\top}` directly and the statistic is
        :math:`\mathrm{quad} / m` (no dispersion rescale).  The default
        (``None``) uses the model-based covariance
        :math:`\mathrm{dispersion} \cdot (X^{\top} W X)^{-1}`.

    Returns
    -------
    F : Float[Array, 'V']
        ``(V,)`` F statistic, one per element.
    p_value : Float[Array, 'V']
        ``(V,)`` upper-tail p-value on ``(df1, df2)`` degrees of freedom.
    df1 : float
        Numerator degrees of freedom, equal to the contrast rank ``m``.
    df2 : float
        Denominator degrees of freedom, equal to ``result.dof_resid``.
    """
    C = jnp.asarray(contrast)
    m = C.shape[0]
    cb = result.coef @ C.T  # (V, m)
    cov_mat = result.cov_unscaled if cov is None else cov
    scale = (
        result.dispersion if cov is None else jnp.ones_like(result.dispersion)
    )

    # M = C cov C^T -- shared (m, m) on the OLS path, else (V, m, m).
    if cov_mat.ndim == 2:
        mat = C @ cov_mat @ C.T  # (m, m)
        mat_inv, _ = small_inv_logdet(mat, m)
        quad = jnp.einsum('vi,ij,vj->v', cb, mat_inv, cb)
    else:
        mat = jnp.einsum('mi,vij,nj->vmn', C, cov_mat, C)
        mat_inv = jax.vmap(lambda a: small_inv_logdet(a, m)[0])(mat)
        quad = jnp.einsum('vi,vij,vj->v', cb, mat_inv, cb)

    f = quad / (m * scale)
    df1 = float(m)
    df2 = result.dof_resid
    p_value = _f_sf(f, df1, df2)
    return f, p_value, df1, df2


_HC_KINDS = ('HC0', 'HC1', 'HC2', 'HC3')
HCKind = Literal['HC0', 'HC1', 'HC2', 'HC3']


def sandwich_cov(
    result: GLMResult,
    Y: Float[Array, 'V N'],
    X: Float[Array, 'N p'],
    *,
    kind: HCKind = 'HC3',
    groups: Optional[Int[Array, 'N']] = None,
    n_groups: Optional[int] = None,
    weights: Optional[Float[Array, 'N']] = None,
) -> Float[Array, 'V p p']:
    r"""Robust (sandwich) coefficient covariance for a fitted GLM.

    Returns the per-element ``(V, p, p)`` covariance :math:`A^{-1} B A^{-1}`
    with bread :math:`A^{-1} = (X^{\top} W X)^{-1}` (the fit's ``cov_unscaled``)
    and a robust meat :math:`B` built from the per-observation score
    contributions :math:`u_i = x_i g_i`, where :math:`g_i = (y_i - \mu_i)
    (\mathrm{d}\mu_i / \mathrm{d}\eta_i) / V(\mu_i)` (for OLS, :math:`g_i` is
    the raw residual :math:`y_i - \mu_i` -- the classic Huber-White form).  The
    computation avoids cuSOLVER and maps over elements; the sandwich weights are
    treated as data-dependent constants and are not differentiated through.

    Pass the result as ``cov=`` to :func:`t_contrast` / :func:`f_contrast`.

    Parameters
    ----------
    result, Y, X
        The fitted :class:`GLMResult` and the data it was fit on (``X`` the
        shared ``(N, p)`` design, ``Y`` the ``(V, N)`` responses).
    kind
        Heteroscedasticity-consistent variant when ``groups`` is ``None``:
        ``'HC0'`` (raw), ``'HC1'`` (:math:`n/(n-p)` correction), ``'HC2'``
        (:math:`/(1 - h_i)`), ``'HC3'`` (:math:`/(1 - h_i)^2`); :math:`h_i` the
        GLM hat diagonal.  Ignored when ``groups`` is given.
    groups
        ``(N,)`` cluster labels for a **cluster-robust** covariance (one-way):
        :math:`B = \sum_g s_g s_g^{\top}`, :math:`s_g = \sum_{i \in g} x_i g_i`,
        with the :math:`G/(G-1) \cdot (N-1)/(N-p)` small-sample factor.  This is
        **one-way** clustering -- the analogue of FSL ``randomise``'s ``-e``
        exchangeability blocks.  It is **not** a variance-group model (FSL
        ``-g`` / PALM ``-vg``): a separate residual *variance* per group
        (heteroscedastic two-sample / FLAME-style grouped variance) is a
        different estimator and is not provided here -- the cluster-robust meat
        assumes a common model, only correlating scores *within* a cluster.

        The cluster path is **eager-only** by default: the distinct-cluster count
        is found with ``jnp.unique`` (a data-dependent shape).  To trace the whole
        estimator under ``jax.jit``, pre-densify ``groups`` to contiguous
        ``0..n_groups-1`` labels and pass ``n_groups``.
    n_groups
        Optional distinct-cluster count.  ``None`` (default) densifies ``groups``
        host-side; supplying it (with contiguous labels) keeps ``sandwich_cov``
        jit-traceable.  Ignored when ``groups`` is ``None``.
    weights
        Optional ``(N,)`` prior weights matching the fit's ``weights=``.

    Returns
    -------
    ``(V, p, p)`` robust covariance (one per element).
    """
    if groups is None and kind not in _HC_KINDS:
        raise ValueError(
            f'sandwich_cov: kind={kind!r}; expected one of {_HC_KINDS}.'
        )
    fam = result.family
    X = jnp.asarray(X)
    Y = jnp.asarray(Y)
    n, p = X.shape
    coef = result.coef  # (V, p)
    n_elem = coef.shape[0]
    w_prior = None if weights is None else jnp.asarray(weights, dtype=X.dtype)

    cl_factor = 0.0
    ng = 0
    if groups is not None:
        groups = jnp.asarray(groups)
        if n_groups is None:
            # Eager-only path: densify the cluster labels to the true *distinct*
            # cluster count (B3) -- the G/(G-1) finite-sample factor needs it, and
            # non-contiguous labels (e.g. from subject exclusion) would otherwise
            # inflate G and under-correct the SE.  jnp.unique has a data-dependent
            # shape, so this branch is not jit-traceable.
            uniq = jnp.unique(groups)
            groups = jnp.searchsorted(uniq, groups)
            ng = int(uniq.shape[0])
        else:
            # Caller-supplied count: `groups` is assumed already densified to
            # contiguous 0..n_groups-1 labels, which lets the whole estimator
            # trace under jax.jit.
            ng = int(n_groups)
        cl_factor = (ng / (ng - 1.0)) * ((n - 1.0) / (n - p))

    def per_elem(
        y: Float[Array, 'N'], b: Float[Array, 'p'], ainv: Float[Array, 'p p']
    ) -> Float[Array, 'p p']:
        eta = X @ b
        mu = fam.linkinv(eta)
        dmu = fam.mu_eta(eta)
        var = jnp.clip(fam.variance(mu), _EPS, None)
        g = (y - mu) * dmu / var  # (N,) score scalar
        w = dmu * dmu / var  # (N,) working weight
        if w_prior is not None:
            g = w_prior * g
            w = w_prior * w
        if groups is None:
            h = w * jnp.einsum('np,pq,nq->n', X, ainv, X)  # GLM hat diagonal
            if kind == 'HC0':
                fac = g * g
            elif kind == 'HC1':
                fac = g * g * (n / (n - p))
            elif kind == 'HC2':
                fac = g * g / jnp.clip(1.0 - h, _EPS, None)
            else:  # HC3
                fac = g * g / jnp.clip(1.0 - h, _EPS, None) ** 2
            meat = jnp.einsum('n,np,nq->pq', fac, X, X)
        else:
            u = X * g[:, None]  # (N, p) per-observation score
            s = jax.ops.segment_sum(u, groups, num_segments=ng)  # (G, p)
            meat = (s.T @ s) * cl_factor
        return ainv @ meat @ ainv

    bread = result.cov_unscaled
    if bread.ndim == 2:
        bread = jnp.broadcast_to(bread, (n_elem, p, p))
    return jax.vmap(per_elem)(Y, coef, bread)


# ---------------------------------------------------------------------------
# Goodness of fit
# ---------------------------------------------------------------------------


def deviance_explained(result: GLMResult) -> Float[Array, 'V']:
    """Per-element fraction of null deviance explained.

    Computes :math:`1 - D / D_0`, where :math:`D` is the model deviance and
    :math:`D_0` the intercept-only (null) deviance.

    Parameters
    ----------
    result
        A fitted :class:`GLMResult` supplying the model and null deviances.

    Returns
    -------
    Float[Array, 'V']
        ``(V,)`` fraction of null deviance explained, one per element.
    """
    return 1.0 - result.deviance / jnp.clip(result.null_deviance, _EPS, None)


def r_squared(result: GLMResult) -> Float[Array, 'V']:
    """Per-element :math:`R^2` (Gaussian) or deviance explained (general).

    For the Gaussian family the deviance is the residual sum of squares and the
    null deviance the total sum of squares, so this is the ordinary coefficient
    of determination :math:`R^2`; for other families it is the fraction of
    deviance explained, i.e. :func:`deviance_explained`.

    Parameters
    ----------
    result
        A fitted :class:`GLMResult` supplying the model and null deviances.

    Returns
    -------
    Float[Array, 'V']
        ``(V,)`` :math:`R^2` (or deviance explained), one per element.
    """
    return deviance_explained(result)


def adj_r_squared(result: GLMResult) -> Float[Array, 'V']:
    """Per-element adjusted :math:`R^2` (Gaussian).

    Applies the standard degrees-of-freedom adjustment to :func:`r_squared`,
    penalising the number of fitted parameters relative to the number of
    observations.

    Parameters
    ----------
    result
        A fitted :class:`GLMResult` supplying the deviances, observation count,
        and residual degrees of freedom.

    Returns
    -------
    Float[Array, 'V']
        ``(V,)`` adjusted :math:`R^2`, one per element.
    """
    n = float(result.n_obs)
    dof = result.dof_resid
    return 1.0 - (1.0 - r_squared(result)) * (n - 1.0) / jnp.clip(
        dof, _EPS, None
    )


def log_likelihood(result: GLMResult) -> Float[Array, 'V']:
    """Per-element maximised log-likelihood.

    Parameters
    ----------
    result
        A fitted :class:`GLMResult` carrying the stored maximised
        log-likelihood.

    Returns
    -------
    Float[Array, 'V']
        ``(V,)`` maximised log-likelihood, one per element.
    """
    return result.log_lik


def aic(result: GLMResult) -> Float[Array, 'V']:
    r"""Per-element Akaike information criterion :math:`-2\ell + 2k`.

    Here :math:`\ell` is the maximised log-likelihood and :math:`k` the number
    of estimated parameters (coefficients plus a dispersion where free).

    Parameters
    ----------
    result
        A fitted :class:`GLMResult` supplying the log-likelihood and parameter
        count.

    Returns
    -------
    Float[Array, 'V']
        ``(V,)`` Akaike information criterion, one per element.
    """
    return -2.0 * result.log_lik + 2.0 * result.n_params


def bic(result: GLMResult) -> Float[Array, 'V']:
    r"""Per-element Bayesian information criterion :math:`-2\ell + k \log N`.

    Here :math:`\ell` is the maximised log-likelihood, :math:`k` the number of
    estimated parameters, and :math:`N` the number of observations.

    Parameters
    ----------
    result
        A fitted :class:`GLMResult` supplying the log-likelihood, parameter
        count, and observation count.

    Returns
    -------
    Float[Array, 'V']
        ``(V,)`` Bayesian information criterion, one per element.
    """
    return -2.0 * result.log_lik + result.n_params * jnp.log(
        jnp.asarray(float(result.n_obs))
    )


def _chi2_sf(x: Array, df: float) -> Array:
    """Chi-square survival :math:`P(X > x)` via the upper incomplete gamma.

    Parameters
    ----------
    x
        Observed chi-square statistic(s), any shape (clipped at zero).
    df
        Degrees of freedom.

    Returns
    -------
    Array
        Upper-tail probability, matching the shape of ``x``.
    """
    return gammaincc(0.5 * df, 0.5 * jnp.clip(x, 0.0, None))


CompareTest = Literal['auto', 'F', 'LRT']


def compare_models(
    full: GLMResult,
    reduced: GLMResult,
    *,
    test: CompareTest = 'auto',
) -> Tuple[Float[Array, 'V'], Float[Array, 'V']]:
    r"""Per-element nested-model comparison (ModelArray full vs. reduced).

    Compares a ``reduced`` model nested within a ``full`` model (fewer
    parameters, fit on the same data) element by element, returning a test
    statistic and its p-value.

    Parameters
    ----------
    full
        The larger fitted :class:`GLMResult` (more parameters).
    reduced
        The nested fitted :class:`GLMResult` (fewer parameters, same data).
    test
        Which test to run.  ``'F'`` uses the extra-sum-of-squares F-test,
        :math:`F = ((D_r - D_f) / (k_f - k_r)) / (D_f / \mathrm{dof}_f)` on
        :math:`(k_f - k_r, \mathrm{dof}_f)` degrees of freedom.  ``'LRT'`` uses
        the likelihood-ratio test, :math:`2(\ell_f - \ell_r) \sim
        \chi^2(k_f - k_r)`.  ``'auto'`` (default) selects ``'F'`` for a
        Gaussian family and ``'LRT'`` otherwise.

    Returns
    -------
    statistic : Float[Array, 'V']
        ``(V,)`` test statistic (F or likelihood-ratio), one per element.
    p_value : Float[Array, 'V']
        ``(V,)`` upper-tail p-value of the statistic.
    """
    d_rank = float(full.rank - reduced.rank)
    if d_rank <= 0:
        raise ValueError(
            'compare_models: `full` must have more parameters than `reduced`.'
        )
    if full.n_obs != reduced.n_obs:
        raise ValueError(
            'compare_models: models must be fit on the same number of '
            'observations.'
        )
    mode = test
    if mode == 'auto':
        mode = 'F' if full.family.name == 'gaussian' else 'LRT'

    if mode == 'F':
        dof_f = full.dof_resid
        num = (reduced.deviance - full.deviance) / d_rank
        den = full.deviance / jnp.clip(dof_f, _EPS, None)
        f = num / jnp.clip(den, _EPS, None)
        return f, _f_sf(f, d_rank, dof_f)
    if mode == 'LRT':
        stat = 2.0 * (full.log_lik - reduced.log_lik)
        return stat, _chi2_sf(stat, d_rank)
    raise ValueError(
        f"compare_models: test={test!r}; expected 'auto'/'F'/'LRT'."
    )
