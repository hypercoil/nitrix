# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Ordinal regression -- the cumulative-link (proportional-odds) model (v3 §4).

An *ordered* categorical response ``y in {0, .., K-1}`` (clinical stage, a Likert
rating) is modelled through ``K - 1`` cumulative thresholds and a **single**
linear predictor shared across them (the proportional-odds assumption):

    P(y_i <= k) = F(theta_k - x_i^T beta),   theta_1 < theta_2 < ... < theta_{K-1}

so ``P(y_i = k) = F(theta_k - eta_i) - F(theta_{k-1} - eta_i)`` with ``theta_0 =
-inf``, ``theta_K = +inf``.  ``F`` is the logistic CDF (``link='logit'`` -- the
proportional-odds model) or the standard-normal CDF (``link='probit'`` -- ordered
probit).  ``beta`` carries **no intercept** (the thresholds play that role).

This is not the scalar single-predictor IRLS (the likelihood couples ``K - 1``
ordered thresholds with the regression coefficients), so it gets its own fitter.
The ordered thresholds are kept ordered by an unconstrained parameterisation
``theta = theta_1 + cumsum([0, exp(delta)])``; the (concave) log-likelihood is
maximised by the shared damped Newton (``lme._optimise.damped_newton`` with
analytic autodiff curvature).  cuSOLVER-free.

References
----------
- McCullagh, P. (1980). Regression models for ordinal data.  J. R. Statist. Soc.
  B 42, 109-142.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional, Tuple, cast

import jax
import jax.numpy as jnp
from jax.scipy.special import ndtr
from jaxtyping import Array, Float, Int

from ..linalg._smalllinalg import small_inv_logdet
from ._batching import blocked_vmap
from ._optimise import damped_newton
from ._result import register_result

__all__ = ['OrdinalResult', 'ordinal_fit', 'ordinal_predict']

_EPS = 1e-12


def _link_cdf(link: str) -> Callable[[Array], Array]:
    """The cumulative-link CDF ``F`` for ``link`` (shared by fit and predict)."""
    if link == 'logit':
        return cast(Callable[[Array], Array], jax.nn.sigmoid)
    if link == 'probit':
        return cast(Callable[[Array], Array], ndtr)
    raise ValueError(f"link={link!r}; expected 'logit' or 'probit'.")


@register_result(
    children=('coef', 'thresholds', 'cov_coef', 'log_lik'),
    aux=('n_obs', 'n_classes', 'link'),
)
@dataclass(frozen=True)
class OrdinalResult:
    """Per-element ordinal (cumulative-link) fit output.

    Attributes
    ----------
    coef
        ``(V, p)`` regression coefficients (no intercept; shared across the
        cumulative thresholds -- the proportional-odds slope).
    thresholds
        ``(V, K-1)`` ordered cutpoints ``theta_1 < ... < theta_{K-1}``.
    cov_coef
        ``(V, p, p)`` ``Cov(coef)`` -- the coefficient block of the inverse
        observed information (accounts for estimating the thresholds).
    log_lik
        ``(V,)`` maximised log-likelihood.
    n_obs, n_classes
        Number of observations ``N`` and ordered classes ``K``.
    link
        The cumulative link used at fit (``'logit'`` / ``'probit'``), stored so
        :func:`ordinal_predict` reconstructs the same CDF without the caller
        re-specifying it.
    """

    coef: Float[Array, 'V p']
    thresholds: Float[Array, 'V Km1']
    cov_coef: Float[Array, 'V p p']
    log_lik: Float[Array, 'V']
    n_obs: int
    n_classes: int
    link: str


def _thresholds(raw: Float[Array, 'nt'], k: int) -> Float[Array, 'Km1']:
    """Ordered cutpoints from the unconstrained ``[theta_1, delta_2..delta_{K-1}]``
    head: ``theta = theta_1 + cumsum([0, exp(delta)])``."""
    t1 = raw[0]
    incr = jnp.exp(raw[1 : k - 1])  # (K-2,)
    steps = jnp.concatenate([jnp.zeros((1,), dtype=raw.dtype), incr])
    return t1 + jnp.cumsum(steps)


def _ordinal_nll(
    raw: Float[Array, 'nt'],
    y: Int[Array, 'N'],
    X: Float[Array, 'N p'],
    k: int,
    p: int,
    cdf: Any,
) -> Float[Array, '']:
    """Negative cumulative-link log-likelihood at the packed parameters."""
    theta = _thresholds(raw, k)  # (K-1,)
    beta = raw[k - 1 :]  # (p,)
    eta = X @ beta  # (N,)
    cum = cdf(theta[None, :] - eta[:, None])  # (N, K-1) = P(y <= j)
    zeros = jnp.zeros((eta.shape[0], 1), dtype=X.dtype)
    ones = jnp.ones((eta.shape[0], 1), dtype=X.dtype)
    cum_full = jnp.concatenate([zeros, cum, ones], axis=1)  # (N, K+1)
    probs = jnp.clip(jnp.diff(cum_full, axis=1), _EPS, None)  # (N, K)
    py = jnp.take_along_axis(probs, y[:, None], axis=1)[:, 0]  # (N,)
    return -jnp.sum(jnp.log(py))


def _ordinal_fit_one(
    y: Int[Array, 'N'],
    X: Float[Array, 'N p'],
    k: int,
    p: int,
    cdf: Any,
    n_iter: int,
    ridge: float,
    raw0: Float[Array, 'nt'],
) -> Tuple[
    Float[Array, 'Km1'],
    Float[Array, 'p'],
    Float[Array, 'p p'],
    Float[Array, ''],
]:
    """Single-element ordinal fit.  Returns ``(thresholds, coef, cov_coef,
    log_lik)``."""

    def nll(raw: Float[Array, 'nt']) -> Float[Array, '']:
        return _ordinal_nll(raw, y, X, k, p, cdf)

    raw = damped_newton(nll, raw0, n_iter=n_iter, damping=ridge, step='damped')
    theta = _thresholds(raw, k)
    beta = raw[k - 1 :]
    nt = raw.shape[0]
    hess = jax.hessian(nll)(raw)
    hess_inv, _ = small_inv_logdet(
        hess + ridge * jnp.eye(nt, dtype=X.dtype), nt
    )
    cov_coef = hess_inv[k - 1 :, k - 1 :]
    return theta, beta, cov_coef, -nll(raw)


def ordinal_fit(
    Y: Int[Array, 'V N'],
    X: Float[Array, 'N p'],
    *,
    n_classes: int,
    link: str = 'logit',
    n_iter: int = 50,
    ridge: float = 1e-8,
    block: Optional[int] = None,
) -> OrdinalResult:
    """Mass-univariate ordinal regression (cumulative-link / proportional odds).

    Fits, per element, the ordered-categorical model ``P(y_v <= k) =
    F(theta_k - X beta_v)`` for ``y_v in {0, .., K-1}``, with ``F`` the logistic
    (``link='logit'``) or normal (``link='probit'``) CDF.  ``X`` carries **no**
    intercept (the ``K - 1`` thresholds absorb it).

    Parameters
    ----------
    Y
        ``(V, N)`` ordered class labels in ``{0, .., n_classes-1}``.
    X
        ``(N, p)`` shared design (no intercept column).
    n_classes
        Number of ordered classes ``K`` (``>= 2``).
    link
        ``'logit'`` (proportional odds, default) or ``'probit'`` (ordered probit).
    n_iter
        Newton iterations (default ``50``).
    ridge
        Small stabiliser on the Newton / information solves.
    block
        Optional element-block size bounding peak memory.

    Returns
    -------
    ``OrdinalResult`` -- ``coef`` (proportional-odds slopes), ordered
    ``thresholds``, ``cov_coef`` (the coefficient information block), ``log_lik``.
    A coefficient contrast is ``c^T coef`` with SE ``sqrt(c^T cov_coef c)``.
    """
    n, p = X.shape
    if n_classes < 2:
        raise ValueError(f'ordinal_fit: n_classes={n_classes} must be >= 2.')
    if Y.shape[-1] != n:
        raise ValueError(
            f'ordinal_fit: Y.shape[-1]={Y.shape[-1]} must match N={n}.'
        )
    cdf = _link_cdf(link)
    k = n_classes
    # Spread the initial thresholds across the response, zero slope.
    raw0 = jnp.concatenate(
        [
            jnp.asarray([-1.0], dtype=X.dtype),
            jnp.zeros((k - 2,), dtype=X.dtype),
            jnp.zeros((p,), dtype=X.dtype),
        ]
    )

    def per_voxel(
        y: Int[Array, 'N'],
    ) -> Tuple[Array, Array, Array, Array]:
        return _ordinal_fit_one(
            y.astype(jnp.int32), X, k, p, cdf, n_iter, ridge, raw0
        )

    thresholds, coef, cov_coef, log_lik = cast(
        Tuple[Array, Array, Array, Array],
        blocked_vmap(per_voxel, (Y,), block=block),
    )
    return OrdinalResult(
        coef=coef,
        thresholds=thresholds,
        cov_coef=cov_coef,
        log_lik=log_lik,
        n_obs=int(n),
        n_classes=int(k),
        link=link,
    )


def ordinal_predict(
    result: OrdinalResult,
    X: Float[Array, 'N p'],
    *,
    type: Literal['class_prob', 'cum_prob', 'class'] = 'class_prob',
) -> Float[Array, '...']:
    """Per-element ordinal prediction on a (new) design ``X``.

    Applies the fitted cumulative-link model ``P(y <= j) = F(theta_j - X beta)``
    (``F`` the stored ``result.link`` CDF) at the new design and returns,
    per ``type``:

    - ``'class_prob'`` (default): the per-class simplex ``P(y = k)``, shape
      ``(V, N, K)`` -- the diff of the cumulative probabilities.
    - ``'cum_prob'``: the cumulative probabilities ``P(y <= j)`` for the
      ``K - 1`` interior thresholds, shape ``(V, N, K - 1)``.
    - ``'class'``: the arg-max class label, shape ``(V, N)`` (**non**
      -differentiable -- a hard label).

    Parameters
    ----------
    result
        An :class:`OrdinalResult` from :func:`ordinal_fit`.
    X
        ``(N, p)`` design (no intercept column, as at fit -- the thresholds
        absorb it).
    type
        Output kind (see above).

    Returns
    -------
    Predictions of the shape listed above.  ``'class_prob'`` / ``'cum_prob'``
    are differentiable w.r.t. ``X`` (and the fitted coefficients / thresholds);
    ``'class'`` is the non-differentiable arg-max.
    """
    cdf = _link_cdf(result.link)
    eta = result.coef @ X.T  # (V, N)
    # cum[v, n, j] = P(y <= j) = F(theta_vj - eta_vn).
    cum = cdf(result.thresholds[:, None, :] - eta[:, :, None])  # (V, N, K-1)
    if type == 'cum_prob':
        return cum
    v, n = eta.shape
    zeros = jnp.zeros((v, n, 1), dtype=cum.dtype)
    ones = jnp.ones((v, n, 1), dtype=cum.dtype)
    cum_full = jnp.concatenate([zeros, cum, ones], axis=-1)  # (V, N, K+1)
    probs = jnp.clip(jnp.diff(cum_full, axis=-1), _EPS, None)  # (V, N, K)
    if type == 'class_prob':
        return probs
    if type == 'class':
        return jnp.argmax(probs, axis=-1)  # (V, N)
    raise ValueError(
        f"ordinal_predict: type={type!r}; expected 'class_prob', 'cum_prob', "
        "or 'class'."
    )
