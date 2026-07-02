# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Closed-form diagonal-Gaussian quantities.

- :func:`kl_diagonal_gaussian` -- KL divergence of a diagonal Gaussian
  :math:`\\mathcal{N}(\\mu, \\operatorname{diag} e^{\\ell})` from the
  standard normal :math:`\\mathcal{N}(0, I)`, in closed form.
- :func:`gaussian_nll` -- negative log-likelihood of observations under a
  diagonal Gaussian.

Both are parameterised by the log-variance ``log_var`` (the log of the
variance) rather than the variance or standard deviation: this keeps the
variance positive without a clamp and turns the :math:`1 / \\sigma^2` and
:math:`\\log \\sigma^2` terms into a numerically benign
:math:`e^{-\\ell}` / :math:`\\ell`, where :math:`\\ell` denotes the
log-variance.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import jax.numpy as jnp
from jaxtyping import Array, Float

from .._internal.reductions import Reduction, reduce

__all__ = ['kl_diagonal_gaussian', 'gaussian_nll']

_AxisArg = Union[int, Tuple[int, ...]]


def kl_diagonal_gaussian(
    mean: Float[Array, '...'],
    log_var: Float[Array, '...'],
    *,
    axis: Optional[_AxisArg] = None,
    reduction: Reduction = 'sum',
) -> Float[Array, '...']:
    """Kullback-Leibler divergence of a diagonal Gaussian from the standard normal.

    Evaluates, in closed form, the KL divergence of the diagonal Gaussian
    :math:`\\mathcal{N}(\\mu, \\operatorname{diag} e^{\\ell})` from the
    standard normal :math:`\\mathcal{N}(0, I)`, where :math:`\\mu` is
    ``mean`` and :math:`\\ell` is the log-variance ``log_var``.

    The closed form for a diagonal Gaussian against the standard normal,
    per dimension, is

    .. math::

        \\tfrac{1}{2}\\left(\\mu^2 + e^{\\ell} - 1 - \\ell\\right),

    which is non-negative and zero exactly at :math:`\\mu = 0`,
    :math:`\\ell = 0`.

    Parameters
    ----------
    mean : Float[Array, '...']
        The Gaussian's mean :math:`\\mu`, of arbitrary shape.
    log_var : Float[Array, '...']
        The Gaussian's log-variance :math:`\\ell`, of the same shape as
        ``mean``.
    axis : int or tuple of int, optional
        Axes to reduce over. Default ``None`` reduces over all axes. The
        latent KL is conventionally *summed* over the latent dimension, so
        pass the latent axis together with ``reduction="sum"`` for a
        per-sample KL.
    reduction : {'sum', 'mean', 'none'}, optional
        How to reduce the per-dimension divergence over ``axis``.
        ``"sum"`` (the default, since the per-dimension KL is additive),
        ``"mean"``, or ``"none"`` to leave it unreduced.

    Returns
    -------
    Float[Array, '...']
        The reduced KL divergence: a scalar when ``axis`` is ``None``,
        per-sample when ``axis`` selects the latent axis, or the
        full per-dimension divergence (same shape as ``mean``) when
        ``reduction="none"``.
    """
    per_dim = 0.5 * (mean**2 + jnp.exp(log_var) - 1.0 - log_var)
    return reduce(per_dim, axis=axis, reduction=reduction)


def gaussian_nll(
    x: Float[Array, '...'],
    mean: Float[Array, '...'],
    log_var: Float[Array, '...'],
    *,
    axis: Optional[_AxisArg] = None,
    reduction: Reduction = 'mean',
) -> Float[Array, '...']:
    """Negative log-likelihood of observations under a diagonal Gaussian.

    Evaluates the negative log-likelihood of the observations ``x`` under
    the diagonal Gaussian :math:`\\mathcal{N}(\\mu, e^{\\ell})`, where
    :math:`\\mu` is ``mean`` and :math:`\\ell` is the log-variance
    ``log_var``.

    Per element, this is

    .. math::

        \\tfrac{1}{2}\\left(\\log(2\\pi) + \\ell
        + (x - \\mu)^2\\, e^{-\\ell}\\right).

    The :math:`e^{-\\ell}` precision and the additive :math:`\\ell`
    normaliser are the log-variance rewrite of the
    :math:`1 / (2\\sigma^2)` and :math:`\\tfrac{1}{2}\\log \\sigma^2`
    terms of the Gaussian density.

    Parameters
    ----------
    x : Float[Array, '...']
        Observations, of arbitrary shape.
    mean : Float[Array, '...']
        Predicted mean :math:`\\mu`, broadcastable to ``x``.
    log_var : Float[Array, '...']
        Predicted log-variance :math:`\\ell`, broadcastable to ``x``.
    axis : int or tuple of int, optional
        Axes to reduce over. Default ``None`` reduces over all axes.
    reduction : {'mean', 'sum', 'none'}, optional
        How to reduce the per-element NLL over ``axis``. ``"mean"`` (the
        default), ``"sum"``, or ``"none"`` to leave it unreduced.

    Returns
    -------
    Float[Array, '...']
        The reduced negative log-likelihood: a scalar when ``axis`` is
        ``None``, otherwise the reduction over ``axis``; or the
        per-element NLL (the broadcast shape of ``x``, ``mean`` and
        ``log_var``) when ``reduction="none"``.
    """
    log_2pi = math.log(2.0 * math.pi)
    per_elem = 0.5 * (log_2pi + log_var + (x - mean) ** 2 * jnp.exp(-log_var))
    return reduce(per_elem, axis=axis, reduction=reduction)
