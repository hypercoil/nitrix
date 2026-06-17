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

Every ``(p, p)`` solve goes through ``stats._smalllinalg.small_inv_logdet``
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
from typing import Any, Callable, Optional, Tuple, cast

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.special import betainc, gammaincc, gammaln, xlogy
from jaxtyping import Array, Float

from ._smalllinalg import small_inv_logdet

__all__ = [
    'Family',
    'GAUSSIAN',
    'BINOMIAL',
    'POISSON',
    'GLMResult',
    'glm_fit',
    'predict',
    't_contrast',
    'f_contrast',
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
# Exponential family
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Family:
    """An exponential-family + link, as a record of pure functions.

    Frozen and hashable so it rides as a static config (and a future
    ``custom_vjp`` nondiff arg).  The fields are the GLM/IRLS primitives:

    - ``link`` / ``linkinv`` -- ``g(mu)`` and ``g^{-1}(eta)``.
    - ``mu_eta`` -- ``d mu / d eta`` as a function of ``eta`` (IRLS weight).
    - ``variance`` -- the variance function ``V(mu)``.
    - ``unit_deviance`` -- per-observation deviance contribution ``d(y, mu)``.
    - ``init_mu`` -- a fitted-mean initialiser from ``y`` (IRLS start).
    - ``loglik`` -- per-observation log-likelihood ``l(y, mu, dispersion)``
      (the dispersion argument is ignored by the fixed-dispersion families).
    - ``has_fixed_dispersion`` -- ``True`` when the dispersion is 1 (binomial /
      Poisson); ``False`` when it is estimated (Gaussian).
    """

    name: str
    has_fixed_dispersion: bool
    link: Callable[[Array], Array]
    linkinv: Callable[[Array], Array]
    mu_eta: Callable[[Array], Array]
    variance: Callable[[Array], Array]
    unit_deviance: Callable[[Array, Array], Array]
    init_mu: Callable[[Array], Array]
    loglik: Callable[[Array, Array, Array], Array]


def _identity(x: Array) -> Array:
    return x


def _ones_like(x: Array) -> Array:
    return jnp.ones_like(x)


def _gaussian_loglik(y: Array, mu: Array, dispersion: Array) -> Array:
    phi = jnp.clip(dispersion, _EPS, None)
    return -0.5 * (jnp.log(2.0 * jnp.pi * phi) + (y - mu) ** 2 / phi)


GAUSSIAN = Family(
    name='gaussian',
    has_fixed_dispersion=False,
    link=_identity,
    linkinv=_identity,
    mu_eta=_ones_like,
    variance=_ones_like,
    unit_deviance=lambda y, mu: (y - mu) ** 2,
    init_mu=_identity,
    loglik=_gaussian_loglik,
)


def _logit(mu: Array) -> Array:
    m = jnp.clip(mu, _EPS, 1.0 - _EPS)
    return jnp.log(m / (1.0 - m))


def _expit(eta: Array) -> Array:
    return 0.5 * (1.0 + jnp.tanh(0.5 * eta))


def _binomial_mu_eta(eta: Array) -> Array:
    mu = _expit(eta)
    return mu * (1.0 - mu)


def _binomial_deviance(y: Array, mu: Array) -> Array:
    m = jnp.clip(mu, _EPS, 1.0 - _EPS)
    return 2.0 * (xlogy(y, y / m) + xlogy(1.0 - y, (1.0 - y) / (1.0 - m)))


def _binomial_loglik(y: Array, mu: Array, dispersion: Array) -> Array:
    m = jnp.clip(mu, _EPS, 1.0 - _EPS)
    return xlogy(y, m) + xlogy(1.0 - y, 1.0 - m)


BINOMIAL = Family(
    name='binomial',
    has_fixed_dispersion=True,
    link=_logit,
    linkinv=_expit,
    mu_eta=_binomial_mu_eta,
    variance=lambda mu: mu * (1.0 - mu),
    unit_deviance=_binomial_deviance,
    init_mu=lambda y: (y + 0.5) / 2.0,
    loglik=_binomial_loglik,
)


def _poisson_deviance(y: Array, mu: Array) -> Array:
    m = jnp.clip(mu, _EPS, None)
    return 2.0 * (xlogy(y, y / m) - (y - m))


def _poisson_loglik(y: Array, mu: Array, dispersion: Array) -> Array:
    m = jnp.clip(mu, _EPS, None)
    return xlogy(y, m) - m - gammaln(y + 1.0)


POISSON = Family(
    name='poisson',
    has_fixed_dispersion=True,
    link=lambda mu: jnp.log(jnp.clip(mu, _EPS, None)),
    linkinv=jnp.exp,
    mu_eta=jnp.exp,
    variance=_identity,
    unit_deviance=_poisson_deviance,
    init_mu=lambda y: y + 0.1,
    loglik=_poisson_loglik,
)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@jax.tree_util.register_pytree_node_class
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

    def tree_flatten(
        self,
    ) -> Tuple[Tuple[Array, ...], Tuple[Family, int, int]]:
        children = (
            self.coef,
            self.cov_unscaled,
            self.dispersion,
            self.deviance,
            self.null_deviance,
            self.log_lik,
        )
        return children, (self.family, self.n_obs, self.rank)

    @classmethod
    def tree_unflatten(
        cls, aux: Tuple[Family, int, int], children: Tuple[Any, ...]
    ) -> 'GLMResult':
        family, n_obs, rank = aux
        coef, cov_unscaled, dispersion, deviance, null_deviance, log_lik = (
            children
        )
        return cls(
            coef=coef,
            cov_unscaled=cov_unscaled,
            dispersion=dispersion,
            deviance=deviance,
            null_deviance=null_deviance,
            log_lik=log_lik,
            family=family,
            n_obs=n_obs,
            rank=rank,
        )


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
    """Single-element penalised IRLS.  Returns ``(beta, cov_unscaled, deviance)``.

    ``penalty`` is the ``(p, p)`` smoothing penalty ``S`` (zeros for a plain
    GLM; ``gam.py`` passes ``S(lambda)``).
    """
    ridge_eye = ridge * jnp.eye(p, dtype=X.dtype)

    def normal_eqs(
        wts: Float[Array, 'N'], z: Float[Array, 'N']
    ) -> Tuple[Float[Array, 'p p'], Float[Array, 'p']]:
        Xw = X * wts[:, None]
        a = Xw.T @ X + penalty + ridge_eye
        b = Xw.T @ z
        return a, b

    # Initialise beta by an OLS fit of the linked initial mean.
    eta0 = family.link(family.init_mu(y))
    a0, b0 = normal_eqs(w, eta0)
    a0_inv, _ = small_inv_logdet(a0, p)
    beta0: Float[Array, 'p'] = a0_inv @ b0

    def step(_: Array, beta: Float[Array, 'p']) -> Float[Array, 'p']:
        eta = X @ beta
        mu = family.linkinv(eta)
        dmu = family.mu_eta(eta)
        var = family.variance(mu)
        wts = w * dmu * dmu / jnp.clip(var, _EPS, None)
        z = eta + (y - mu) / jnp.clip(dmu, _EPS, None)
        a, b = normal_eqs(wts, z)
        a_inv, _ = small_inv_logdet(a, p)
        return a_inv @ b

    beta = cast(Float[Array, 'p'], lax.fori_loop(0, n_iter, step, beta0))

    eta = X @ beta
    mu = family.linkinv(eta)
    dmu = family.mu_eta(eta)
    var = family.variance(mu)
    wts = w * dmu * dmu / jnp.clip(var, _EPS, None)
    a, _ = normal_eqs(wts, eta)
    a_inv, _ = small_inv_logdet(a, p)
    dev = jnp.sum(family.unit_deviance(y, mu))
    return beta, a_inv, dev


def glm_fit(
    Y: Float[Array, 'V N'],
    X: Float[Array, 'N p'],
    *,
    family: Family = GAUSSIAN,
    weights: Optional[Float[Array, 'N']] = None,
    n_iter: int = 25,
    ridge: float = 0.0,
) -> GLMResult:
    """Fit a mass-univariate GLM: shared design ``X``, per-element responses.

    Parameters
    ----------
    Y
        ``(V, N)`` responses -- ``V`` elements, ``N`` observations.
    X
        ``(N, p)`` design, shared across elements.
    family
        Exponential family + link (default ``GAUSSIAN`` identity = OLS).
    weights
        Optional ``(N,)`` prior observation weights (WLS / known precision).
        ``None`` is unweighted.
    n_iter
        IRLS iterations for non-Gaussian families (ignored on the OLS path).
    ridge
        Optional L2 stabiliser added to the normal-equation matrix.

    Returns
    -------
    ``GLMResult`` (coefficients, unscaled covariance, dispersion, deviance,
    null deviance).
    """
    n, p = X.shape
    if Y.shape[-1] != n:
        raise ValueError(
            f'glm_fit: Y.shape[-1]={Y.shape[-1]} must match N={n}.'
        )
    w = jnp.ones((n,), dtype=Y.dtype) if weights is None else weights

    # Null-model deviance (intercept-only MLE mu = mean(y) for canonical links).
    y_bar = jnp.mean(Y, axis=-1, keepdims=True)
    null_dev = jnp.sum(family.unit_deviance(Y, jnp.broadcast_to(y_bar, Y.shape)), axis=-1)

    is_ols = family.name == 'gaussian'
    if is_ols:
        coef, cov_unscaled, wrss = _ols_fit(Y, X, w, ridge)
        dof = float(n - p)
        dispersion = wrss / dof if dof > 0 else wrss
        deviance = wrss
    else:
        penalty = jnp.zeros((p, p), dtype=X.dtype)
        fit = jax.vmap(
            lambda y: _pirls_one(y, X, w, penalty, family, p, n_iter, ridge)
        )
        coef, cov_unscaled, deviance = fit(Y)
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


def predict(
    result: GLMResult,
    X: Float[Array, 'N p'],
    *,
    type: str = 'response',
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
) -> Tuple[
    Float[Array, 'V'], Float[Array, 'V'], Float[Array, 'V'], Float[Array, 'V']
]:
    """Per-element t-test of a linear contrast ``c^T beta``.

    Returns ``(effect, se, t, p_value)``: the contrast estimate, its standard
    error, the t statistic, and the two-sided p-value at ``dof_resid`` degrees
    of freedom (ModelArray ``lm`` columns ``estimate`` / ``std.error`` /
    ``statistic`` / ``p.value``).
    """
    c = jnp.asarray(contrast)
    effect = result.coef @ c  # (V,)
    var_unscaled = jnp.einsum('...ij,i,j->...', result.cov_unscaled, c, c)
    se = jnp.sqrt(jnp.clip(result.dispersion * var_unscaled, _EPS, None))
    t = effect / se
    p_value = _t_two_sided_sf(t, result.dof_resid)
    return effect, se, t, p_value


def f_contrast(
    result: GLMResult,
    contrast: Float[Array, 'm p'],
) -> Tuple[Float[Array, 'V'], Float[Array, 'V'], float, float]:
    """Per-element F-test of a multi-row contrast ``C beta = 0``.

    Returns ``(F, p_value, df1, df2)`` with ``df1 = m`` (contrast rank) and
    ``df2 = dof_resid``.
    """
    C = jnp.asarray(contrast)
    m = C.shape[0]
    cb = result.coef @ C.T  # (V, m)

    # M = C (X^T W X)^{-1} C^T -- shared (m, m) on the OLS path, else (V, m, m).
    if result.cov_unscaled.ndim == 2:
        mat = C @ result.cov_unscaled @ C.T  # (m, m)
        mat_inv, _ = small_inv_logdet(mat, m)
        quad = jnp.einsum('vi,ij,vj->v', cb, mat_inv, cb)
    else:
        mat = jnp.einsum('mi,vij,nj->vmn', C, result.cov_unscaled, C)
        mat_inv = jax.vmap(lambda a: small_inv_logdet(a, m)[0])(mat)
        quad = jnp.einsum('vi,vij,vj->v', cb, mat_inv, cb)

    f = quad / (m * result.dispersion)
    df1 = float(m)
    df2 = result.dof_resid
    p_value = _f_sf(f, df1, df2)
    return f, p_value, df1, df2


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
    return 1.0 - (1.0 - r_squared(result)) * (n - 1.0) / jnp.clip(dof, _EPS, None)


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


def compare_models(
    full: GLMResult,
    reduced: GLMResult,
    *,
    test: str = 'auto',
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
    raise ValueError(f"compare_models: test={test!r}; expected 'auto'/'F'/'LRT'.")
