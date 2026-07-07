# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
r"""
Robust M-estimator regression and the MAD scale estimator.

Outlier-resistant linear regression by iteratively reweighted least squares
(IRLS): each observation is down-weighted by an *influence function* of its
current standardised residual, so a handful of gross outliers (motion-corrupted
fMRI frames, a mislabelled subject) cannot dominate the fit the way they do
ordinary least squares.

- :func:`mad` -- the median absolute deviation, the robust scale estimator that
  pairs with the M-estimators (normal-consistent by default).
- :func:`huber_regress` -- Huber's monotone M-estimator: quadratic near zero,
  linear in the tails. Bounded influence but never zero -- large outliers still
  contribute a constant pull.
- :func:`tukey_bisquare_regress` -- the Tukey bisquare *redescender*: influence
  rises, then falls back to zero beyond the tuning radius, so extreme outliers
  are rejected outright (at the cost of a non-convex objective needing a decent
  start; the OLS warm start suffices in practice).

Both estimators are pure composition: the IRLS loop is a fixed-iteration
:func:`jax.lax.fori_loop` over the cuSOLVER-free ``(p, p)`` weighted-least-
squares solve (:func:`nitrix.linalg._smalllinalg.small_inv_logdet`, the same
inner solve the GLM/GAM suite uses), so the fit is differentiable and carries no
factorisation that could invoke a fragile cuSOLVER path. Fit a single response
``y`` of shape ``(N,)``; ``jax.vmap`` over a stack of responses for the
mass-univariate case.

References
----------
Huber PJ (1964). Robust estimation of a location parameter. *Annals of
Mathematical Statistics*, 35(1), 73-101.
https://doi.org/10.1214/aoms/1177703732

Beaton AE, Tukey JW (1974). The fitting of power series, meaning polynomials,
illustrated on band-spectroscopic data. *Technometrics*, 16(2), 147-185.
https://doi.org/10.1080/00401706.1974.10489171
"""

from __future__ import annotations

from typing import Callable, NamedTuple, Optional, Union

import jax.lax as lax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ..linalg._smalllinalg import small_inv_logdet

__all__ = [
    'RobustFit',
    'huber_regress',
    'mad',
    'tukey_bisquare_regress',
]

# 1 / Phi^{-1}(3/4) = 1 / 0.674489...: scales the raw MAD to a consistent
# estimator of the standard deviation under Gaussian data.
_MAD_TO_SIGMA = 1.4826022185056018


class RobustFit(NamedTuple):
    """A fitted robust regression (a NamedTuple of arrays, not a module).

    Attributes
    ----------
    coef : Float[Array, 'p']
        The M-estimated coefficient vector.
    scale : Float[Array, '']
        The final robust residual scale (normal-consistent :func:`mad` of the
        residuals), in the same units as ``y``.
    weights : Float[Array, 'N']
        The final per-observation IRLS weights in :math:`[0, 1]`; near-zero
        weights flag observations the estimator treated as outliers.
    residuals : Float[Array, 'N']
        The final residuals ``y - X @ coef``.
    """

    coef: Float[Array, 'p']
    scale: Float[Array, '']
    weights: Float[Array, 'N']
    residuals: Float[Array, 'N']


def mad(
    x: Float[Array, '...'],
    *,
    axis: int = -1,
    center: Optional[Union[float, Float[Array, '...']]] = None,
    normalize: bool = True,
) -> Float[Array, '...']:
    r"""Median absolute deviation of ``x`` along an axis.

    :math:`\operatorname{MAD}(x) = \operatorname{median}_i\,
    |x_i - \operatorname{median}(x)|`, a breakdown-point-½ robust dispersion
    estimator. With ``normalize=True`` (default) the result is multiplied by
    :math:`1/\Phi^{-1}(3/4) \approx 1.4826`, making it a consistent estimator of
    the standard deviation :math:`\sigma` for Gaussian data (so it is
    interchangeable with the sample standard deviation but resistant to
    outliers).

    Parameters
    ----------
    x : Float[Array, '...']
        Input values.
    axis : int, optional
        Axis to reduce. Default ``-1``.
    center : float or Float[Array, '...'], optional
        Centre to deviate about. Default ``None`` uses the median of ``x`` along
        ``axis`` (broadcast back over the reduced axis).
    normalize : bool, optional
        If ``True`` (default), scale by :math:`1.4826` for consistency with the
        Gaussian standard deviation; if ``False``, return the raw MAD.

    Returns
    -------
    Float[Array, '...']
        The (optionally normalised) MAD, with ``axis`` reduced.
    """
    med = jnp.median(x, axis=axis, keepdims=True) if center is None else center
    dev = jnp.median(jnp.abs(x - med), axis=axis)
    factor = _MAD_TO_SIGMA if normalize else 1.0
    return factor * dev


def _huber_weight(u: Float[Array, 'N'], delta: float) -> Float[Array, 'N']:
    """Huber IRLS weight ``w(u) = min(1, delta / |u|)`` (double-``where`` safe)."""
    au = jnp.abs(u)
    inner = au <= delta
    safe = jnp.where(inner, 1.0, au)  # avoid 0-div in the unused branch
    return jnp.where(inner, 1.0, delta / safe)


def _tukey_weight(u: Float[Array, 'N'], c: float) -> Float[Array, 'N']:
    """Tukey bisquare IRLS weight ``(1 - (u/c)^2)^2`` inside ``|u| <= c``, else 0."""
    t = u / c
    return jnp.where(jnp.abs(u) <= c, (1.0 - t * t) ** 2, 0.0)


def _wls_solve(
    X: Float[Array, 'N p'],
    y: Float[Array, 'N'],
    w: Float[Array, 'N'],
    ridge: float,
    p: int,
) -> Float[Array, 'p']:
    """One cuSOLVER-free weighted-least-squares solve of the normal equations."""
    Xw = X * w[:, None]
    xtwx = Xw.T @ X + ridge * jnp.eye(p, dtype=X.dtype)
    inv, _ = small_inv_logdet(xtwx, p)
    return inv @ (Xw.T @ y)


def _robust_regress(
    X: Float[Array, 'N p'],
    y: Float[Array, 'N'],
    weight_fn: Callable[[Float[Array, 'N']], Float[Array, 'N']],
    *,
    n_iter: int,
    ridge: float,
) -> RobustFit:
    """IRLS shared by the Huber and Tukey estimators.

    Warm-starts at OLS, then reweights by ``weight_fn`` of the residuals
    standardised by their (normal-consistent) MAD, solving the weighted normal
    equations each iteration. When the residual scale collapses (a near-exact
    fit), it falls back to unit scale so the reweighting reduces to OLS.
    """
    n, p = X.shape
    eps = jnp.finfo(X.dtype).eps
    ones = jnp.ones((n,), X.dtype)
    beta0 = _wls_solve(X, y, ones, ridge, p)

    def scaled_weights(beta: Float[Array, 'p']) -> Float[Array, 'N']:
        resid = y - X @ beta
        s = mad(resid, normalize=True)
        s = jnp.where(s > eps, s, 1.0)
        return weight_fn(resid / s)

    def step(_: int, beta: Float[Array, 'p']) -> Float[Array, 'p']:
        return _wls_solve(X, y, scaled_weights(beta), ridge, p)

    beta = lax.fori_loop(0, n_iter, step, beta0)
    resid = y - X @ beta
    scale = jnp.where(mad(resid, normalize=True) > eps, mad(resid), 1.0)
    return RobustFit(
        coef=beta,
        scale=scale,
        weights=weight_fn(resid / scale),
        residuals=resid,
    )


def huber_regress(
    X: Float[Array, 'N p'],
    y: Float[Array, 'N'],
    *,
    delta: float = 1.345,
    n_iter: int = 20,
    ridge: float = 0.0,
) -> RobustFit:
    r"""Huber M-estimator linear regression.

    Minimises :math:`\sum_i \rho_\delta(r_i / s)` where :math:`\rho_\delta` is
    quadratic for :math:`|r| \le \delta s` and linear beyond, and :math:`s` is
    the robust residual scale. The influence is bounded but non-redescending, so
    the objective is convex and the fit is stable, but very extreme outliers
    still exert a constant (not vanishing) pull -- use
    :func:`tukey_bisquare_regress` to reject them outright.

    Parameters
    ----------
    X : Float[Array, 'N p']
        Design matrix (``N`` observations, ``p`` predictors). Include an
        intercept column explicitly if wanted.
    y : Float[Array, 'N']
        Response vector. ``vmap`` over a stack for many responses.
    delta : float, optional
        Tuning constant in units of the robust scale. Default ``1.345`` gives
        95% efficiency relative to OLS at the Gaussian model.
    n_iter : int, optional
        IRLS iterations (fixed, for differentiability). Default ``20``.
    ridge : float, optional
        Optional L2 stabiliser on the normal-equation diagonal (for rank-
        deficient or collinear ``X``). Default ``0.0``.

    Returns
    -------
    RobustFit
        ``(coef, scale, weights, residuals)``.
    """
    return _robust_regress(
        X, y, lambda u: _huber_weight(u, delta), n_iter=n_iter, ridge=ridge
    )


def tukey_bisquare_regress(
    X: Float[Array, 'N p'],
    y: Float[Array, 'N'],
    *,
    c: float = 4.685,
    n_iter: int = 20,
    ridge: float = 0.0,
) -> RobustFit:
    r"""Tukey bisquare (biweight) redescending M-estimator regression.

    Minimises :math:`\sum_i \rho_c(r_i / s)` with the bisquare
    :math:`\rho_c`, whose influence rises then *redescends* to zero for
    :math:`|r| > c\,s`: observations beyond the tuning radius are rejected
    outright (zero weight), giving maximal outlier resistance. The objective is
    non-convex, so the estimate is the redescender basin reachable from the OLS
    warm start -- adequate in practice for the moderate-contamination regime this
    targets.

    Parameters
    ----------
    X : Float[Array, 'N p']
        Design matrix (``N`` observations, ``p`` predictors).
    y : Float[Array, 'N']
        Response vector. ``vmap`` over a stack for many responses.
    c : float, optional
        Tuning radius in units of the robust scale. Default ``4.685`` gives 95%
        efficiency relative to OLS at the Gaussian model.
    n_iter : int, optional
        IRLS iterations (fixed, for differentiability). Default ``20``.
    ridge : float, optional
        Optional L2 stabiliser on the normal-equation diagonal. Default ``0.0``.

    Returns
    -------
    RobustFit
        ``(coef, scale, weights, residuals)``.
    """
    return _robust_regress(
        X, y, lambda u: _tukey_weight(u, c), n_iter=n_iter, ridge=ridge
    )
