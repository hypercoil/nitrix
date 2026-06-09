# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Closed-form diagonal-Gaussian quantities.

- ``kl_diagonal_gaussian`` -- KL divergence of a diagonal Gaussian
  ``N(mean, diag exp(log_var))`` from the standard normal ``N(0, I)``,
  in closed form.
- ``gaussian_nll`` -- negative log-likelihood of observations under a
  diagonal Gaussian.

Both are parameterised by ``log_var`` (the log of the variance) rather
than the variance or standard deviation: this keeps the variance
positive without a clamp and turns the ``1 / var`` and ``log var`` terms
into a numerically benign ``exp(-log_var)`` / ``log_var``.
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
    """KL divergence of ``N(mean, diag exp(log_var))`` from ``N(0, I)``.

    The closed form for a diagonal Gaussian against the standard
    normal, per dimension:

    ``0.5 * (mean**2 + exp(log_var) - 1 - log_var)``

    which is non-negative and zero exactly at ``mean = 0``,
    ``log_var = 0``.

    Parameters
    ----------
    mean, log_var
        The Gaussian's mean and log-variance, identical shape.
    axis
        Axes to reduce.  Default ``None`` (all).  The latent KL is
        conventionally *summed* over the latent dimension, so pass the
        latent axis with ``reduction="sum"`` for a per-sample KL.
    reduction
        ``"sum"`` (default -- the per-dimension KL is additive),
        ``"mean"``, or ``"none"``.

    Returns
    -------
    The reduced KL (scalar, or per-sample when ``axis`` is the latent
    axis), or the per-dimension KL when ``reduction="none"``.
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
    """Negative log-likelihood of ``x`` under ``N(mean, exp(log_var))``.

    Per element:

    ``0.5 * (log(2 pi) + log_var + (x - mean)**2 * exp(-log_var))``

    The ``exp(-log_var)`` precision and the additive ``log_var``
    normaliser are the log-variance rewrite of the ``1 / (2 var)`` and
    ``0.5 log var`` terms of the Gaussian density.

    Parameters
    ----------
    x
        Observations.
    mean, log_var
        Predicted mean and log-variance, broadcastable to ``x``.
    axis
        Axes to reduce.  Default ``None`` (all).
    reduction
        ``"mean"`` (default), ``"sum"``, or ``"none"``.

    Returns
    -------
    The reduced NLL, or the per-element NLL when ``reduction="none"``.
    """
    log_2pi = math.log(2.0 * math.pi)
    per_elem = 0.5 * (log_2pi + log_var + (x - mean) ** 2 * jnp.exp(-log_var))
    return reduce(per_elem, axis=axis, reduction=reduction)
