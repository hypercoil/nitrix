# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Mass-univariate generalised linear models.

``glm_fit`` fits the same design ``X`` (``N, p``) to every element (voxel /
vertex / fixel) of a response tensor ``Y`` (``V, N``) -- the mass-univariate
"one model per element, shared design" workload behind ModelArray's ``lm`` and
the parametric backbone of GAM (``gam.py``) and the permutation engine
(``inference/randomise``).

Two paths, one surface
----------------------

The fit is **IRLS-shaped**: a weighted-least-squares solve
``(X^T W X + S) beta = X^T W z`` over a working response ``z`` and working
weights ``W``.  OLS, WLS, and the full exponential family are then the same
machinery at different ``(W, z)``:

- **Gaussian identity, no penalty** -- the dominant ``lm`` case -- takes a
  vectorised fast path: ``X^T X`` is *shared* across elements, so a single
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

``t_contrast`` / ``f_contrast`` return the per-element effect, standard error,
t / F statistic, and the (Student-t / F) p-value -- the ModelArray ``lm``
output columns.  p-values use the regularised incomplete beta
(``jax.scipy.special.betainc``), so they are exact and cuSOLVER-free.
``r_squared`` / ``deviance_explained`` use the stored null-model deviance.

The exponential family is a frozen ``Family`` value (a small record of pure
link / variance / deviance functions); ``GAUSSIAN`` / ``BINOMIAL`` / ``POISSON``
ship, and a custom family is just another ``Family(...)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple, Union

import jax
import jax.numpy as jnp
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
        Unscaled coefficient covariance ``(X^T W X)^{-1}``.  Shared ``(p, p)``
        for the Gaussian-identity fast path; per-element ``(V, p, p)`` for the
        IRLS path.  Multiply by ``dispersion`` for the covariance.
    dispersion
        ``(V,)`` dispersion estimate (residual variance for Gaussian; ``1`` for
        binomial / Poisson unless overdispersion is estimated).
    deviance
        ``(V,)`` model deviance (residual sum of squares for Gaussian).
    null_deviance
        ``(V,)`` deviance of the intercept-only model (for ``r_squared`` /
        ``deviance_explained``).
    log_lik
        ``(V,)`` maximised log-likelihood (for ``aic`` / ``bic`` / the LRT).
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
    """Vectorised Gaussian-identity WLS: one shared ``(p, p)`` inverse.

    ``X^T W X`` does not depend on the element, so all ``V`` fits are a single
    ``(p, p)`` solve plus matmuls -- no per-element factorisation.
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
    """Single-element penalised IRLS via the shared core.  Returns
    ``(beta, cov_unscaled, deviance)``.  ``penalty`` is the ``(p, p)`` smoothing
    penalty (zeros for a plain GLM)."""
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
        ``'binomial'`` / ``'poisson'``) or a ``Family`` instance.  Default
        ``GAUSSIAN`` identity = OLS.
    weights
        Optional ``(N,)`` prior observation weights (WLS / known precision).
        ``None`` is unweighted.
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
    ``GLMResult`` (coefficients, unscaled covariance, dispersion, deviance,
    null deviance).
    """
    family = resolve_family(family)
    n, p = X.shape
    if Y.shape[-1] != n:
        raise ValueError(
            f'glm_fit: Y.shape[-1]={Y.shape[-1]} must match N={n}.'
        )
    w = jnp.ones((n,), dtype=Y.dtype) if weights is None else weights

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
        dof = float(n - p)
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
            dof = float(n - p)
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
        rank=int(p),
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
    """Per-element prediction on a (new) design ``X``.

    ``type='link'`` returns the linear predictor ``eta = X beta``;
    ``type='response'`` (default) applies the inverse link.
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
    """Two-sided Student-t survival ``P(|T| > |t|)`` via regularised beta."""
    x = df / (df + t * t)
    return betainc(0.5 * df, 0.5, x)


def _f_sf(f: Array, df1: float, df2: float) -> Array:
    """F survival ``P(F > f)`` via the regularised incomplete beta."""
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
    """Per-element t-test of a linear contrast ``c^T beta``.

    Returns ``(effect, se, t, p_value)``: the contrast estimate, its standard
    error, the t statistic, and the two-sided p-value at ``dof_resid`` degrees
    of freedom (ModelArray ``lm`` columns ``estimate`` / ``std.error`` /
    ``statistic`` / ``p.value``).

    ``cov`` overrides the coefficient covariance: pass a ``(V, p, p)`` (or shared
    ``(p, p)``) matrix -- e.g. the robust ``sandwich_cov`` output -- to use
    ``c^T cov c`` directly (already on the variance scale).  The default
    (``None``) is the model-based ``dispersion * (X^T W X)^{-1}`` (unchanged).
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
    """Per-element F-test of a multi-row contrast ``C beta = 0``.

    Returns ``(F, p_value, df1, df2)`` with ``df1 = m`` (contrast rank) and
    ``df2 = dof_resid``.

    ``cov`` overrides the coefficient covariance (e.g. the robust
    ``sandwich_cov`` output): the Wald quadratic form uses ``C cov C^T`` directly
    and the statistic is ``quad / m`` (no dispersion rescale).  The default
    (``None``) is the model-based ``dispersion * (X^T W X)^{-1}`` (unchanged).
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
    weights: Optional[Float[Array, 'N']] = None,
) -> Float[Array, 'V p p']:
    """Robust (sandwich) coefficient covariance for a fitted GLM.

    Returns the per-element ``(V, p, p)`` covariance
    ``A^{-1} B A^{-1}`` with bread ``A^{-1} = (X^T W X)^{-1}`` (the fit's
    ``cov_unscaled``) and a robust meat ``B`` built from the per-observation
    score contributions ``u_i = x_i g_i``, ``g_i = (y_i - mu_i) (dmu_i/deta_i) /
    V(mu_i)`` (for OLS, ``g_i`` is the raw residual ``y_i - mu_i`` -- the
    classic Huber-White form).  cuSOLVER-free, ``vmap`` over elements; the
    sandwich weights are data-dependent constants (not differentiated through).

    Pass the result as ``cov=`` to ``t_contrast`` / ``f_contrast``.

    Parameters
    ----------
    result, Y, X
        The fitted ``GLMResult`` and the data it was fit on (``X`` shared,
        ``Y`` the ``(V, N)`` responses).
    kind
        Heteroscedasticity-consistent variant when ``groups`` is ``None``:
        ``'HC0'`` (raw), ``'HC1'`` (``n/(n-p)`` correction), ``'HC2'``
        (``/(1 - h_i)``), ``'HC3'`` (``/(1 - h_i)^2``); ``h_i`` the GLM hat
        diagonal.  Ignored when ``groups`` is given.
    groups
        ``(N,)`` cluster labels for a **cluster-robust** covariance (one-way):
        ``B = sum_g s_g s_g^T``, ``s_g = sum_{i in g} x_i g_i``, with the
        ``G/(G-1) * (N-1)/(N-p)`` small-sample factor.  This is **one-way**
        clustering -- FSL ``randomise``'s ``-e`` exchangeability-block analogue
        (audit N6).  It is **not** a variance-group model (FSL ``-g`` / PALM
        ``-vg``): a separate residual *variance* per group (heteroscedastic
        two-sample / FLAME-style grouped variance) is a different estimator and
        is not provided here -- the cluster-robust meat assumes a common model,
        only correlating scores *within* a cluster.
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
    n_groups = 0
    if groups is not None:
        # Densify the cluster labels: the G/(G-1) finite-sample factor needs the
        # true *distinct* cluster count, not max(label)+1 -- non-contiguous labels
        # (e.g. from subject exclusion) would otherwise inflate G and under-correct
        # the SE.  (Unlike the likelihood paths, this count has no cancellation.)
        groups = jnp.asarray(groups)
        uniq = jnp.unique(groups)
        groups = jnp.searchsorted(uniq, groups)
        n_groups = int(uniq.shape[0])
        cl_factor = (n_groups / (n_groups - 1.0)) * ((n - 1.0) / (n - p))

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
            s = jax.ops.segment_sum(u, groups, num_segments=n_groups)  # (G, p)
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
    """Per-element fraction of null deviance explained: ``1 - dev / null_dev``."""
    return 1.0 - result.deviance / jnp.clip(result.null_deviance, _EPS, None)


def r_squared(result: GLMResult) -> Float[Array, 'V']:
    """Per-element ``R^2`` (Gaussian) / deviance explained (general).

    For the Gaussian family ``deviance`` is the residual sum of squares and
    ``null_deviance`` the total sum of squares, so this is the ordinary
    ``R^2``; for other families it is the fraction of deviance explained.
    """
    return deviance_explained(result)


def adj_r_squared(result: GLMResult) -> Float[Array, 'V']:
    """Per-element adjusted ``R^2`` (Gaussian)."""
    n = float(result.n_obs)
    dof = result.dof_resid
    return 1.0 - (1.0 - r_squared(result)) * (n - 1.0) / jnp.clip(
        dof, _EPS, None
    )


def log_likelihood(result: GLMResult) -> Float[Array, 'V']:
    """Per-element maximised log-likelihood."""
    return result.log_lik


def aic(result: GLMResult) -> Float[Array, 'V']:
    """Per-element Akaike information criterion ``-2 ll + 2 k``."""
    return -2.0 * result.log_lik + 2.0 * result.n_params


def bic(result: GLMResult) -> Float[Array, 'V']:
    """Per-element Bayesian information criterion ``-2 ll + k log N``."""
    return -2.0 * result.log_lik + result.n_params * jnp.log(
        jnp.asarray(float(result.n_obs))
    )


def _chi2_sf(x: Array, df: float) -> Array:
    """Chi-square survival ``P(X > x)`` via the upper incomplete gamma."""
    return gammaincc(0.5 * df, 0.5 * jnp.clip(x, 0.0, None))


CompareTest = Literal['auto', 'F', 'LRT']


def compare_models(
    full: GLMResult,
    reduced: GLMResult,
    *,
    test: CompareTest = 'auto',
) -> Tuple[Float[Array, 'V'], Float[Array, 'V']]:
    """Per-element nested-model comparison (ModelArray ``full vs. reduced``).

    ``reduced`` must be nested in ``full`` (fewer parameters, same data).
    Returns ``(statistic, p_value)``:

    - ``test='F'`` (or ``'auto'`` for a Gaussian family) -- the extra-sum-of-
      squares F-test, ``F = ((D_r - D_f) / (k_f - k_r)) / (D_f / dof_f)`` on
      ``(k_f - k_r, dof_f)`` degrees of freedom.
    - ``test='LRT'`` (or ``'auto'`` otherwise) -- the likelihood-ratio test,
      ``2 (ll_f - ll_r) ~ chi^2(k_f - k_r)``.
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
