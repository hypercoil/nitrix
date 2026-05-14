# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Per-tensor / per-channel intensity normalisation.

Four canonical normalisations used across neuroimaging
pipelines:

- ``zscore_normalize``: subtract mean, divide by std (the
  standard z-score).
- ``psc_normalize``: percent-signal-change relative to the mean
  (the fMRI BOLD convention).
- ``robust_zscore_normalize``: median + MAD instead of mean + std
  (robust to outliers; useful for raw intensity images with
  bright artefacts).
- ``intensity_normalize``: percentile-clip then rescale to
  ``[0, 1]`` (the synthstrip / SynthSeg pre-training
  convention).

All four are differentiable and support per-axis reduction via
the ``axis`` argument; the default is to reduce over the
trailing axis (the observation / time axis convention).

The optional ``weight`` argument enables weighted normalisation:
the mean / std / median are computed using the named weights.
For unit weights this matches the unweighted normalisation
exactly.

The legacy ``hypercoil.functional.linear``'s normalisation
helpers were tightly coupled to the compartmentalised-linear
layer; this module pulls them out as standalone primitives.
"""
from __future__ import annotations

from typing import Optional, Tuple, Union

import jax.numpy as jnp
from jaxtyping import Array, Float, Num


__all__ = [
    'zscore_normalize',
    'psc_normalize',
    'robust_zscore_normalize',
    'intensity_normalize',
    'demean',
]


_AxisArg = Union[int, Tuple[int, ...]]


def _weighted_mean_std(
    x: Num[Array, '...'],
    *,
    axis: _AxisArg,
    weights: Optional[Num[Array, '...']] = None,
) -> Tuple[Float[Array, '...'], Float[Array, '...']]:
    '''Weighted mean and std along ``axis``.

    For ``weights=None``, equivalent to ``mean`` / ``std`` over
    ``axis``.  ``ddof = 0`` (population, not sample), since
    normalisation is per-cell and the Bessel correction is
    inappropriate for centering / scaling.
    '''
    if weights is None:
        mean = jnp.mean(x, axis=axis, keepdims=True)
        std = jnp.std(x, axis=axis, keepdims=True)
        return mean, std
    w_sum = jnp.sum(weights, axis=axis, keepdims=True)
    mean = jnp.sum(x * weights, axis=axis, keepdims=True) / w_sum
    var = jnp.sum(weights * (x - mean) ** 2, axis=axis, keepdims=True) / w_sum
    return mean, jnp.sqrt(var)


def demean(
    x: Num[Array, '...'],
    *,
    axis: _AxisArg = -1,
    weights: Optional[Num[Array, '...']] = None,
) -> Num[Array, '...']:
    '''Subtract the (weighted) mean along ``axis``.'''
    mean, _ = _weighted_mean_std(x, axis=axis, weights=weights)
    return x - mean


def zscore_normalize(
    x: Num[Array, '...'],
    *,
    axis: _AxisArg = -1,
    weights: Optional[Num[Array, '...']] = None,
    eps: float = 1e-12,
) -> Num[Array, '...']:
    '''Z-score normalisation: ``(x - mean) / (std + eps)`` along ``axis``.

    Population standard deviation (no Bessel correction).
    Per-cell normalisation; output has zero mean and unit std
    along the named axis.
    '''
    mean, std = _weighted_mean_std(x, axis=axis, weights=weights)
    return (x - mean) / (std + eps)


def psc_normalize(
    x: Num[Array, '...'],
    *,
    axis: _AxisArg = -1,
    weights: Optional[Num[Array, '...']] = None,
    eps: float = 1e-12,
) -> Num[Array, '...']:
    '''Percent signal change: ``100 * (x - mean) / (mean + eps)``.

    The fMRI BOLD convention.  Per-cell normalisation;
    interpretable as "deviation from baseline as a percent of
    baseline".
    '''
    mean, _ = _weighted_mean_std(x, axis=axis, weights=weights)
    return 100.0 * (x - mean) / (mean + eps)


def _median_mad(
    x: Num[Array, '...'], *, axis: _AxisArg,
) -> Tuple[Float[Array, '...'], Float[Array, '...']]:
    '''Median and median-absolute-deviation along ``axis``.'''
    median = jnp.median(x, axis=axis, keepdims=True)
    mad = jnp.median(jnp.abs(x - median), axis=axis, keepdims=True)
    return median, mad


def robust_zscore_normalize(
    x: Num[Array, '...'],
    *,
    axis: _AxisArg = -1,
    eps: float = 1e-12,
) -> Num[Array, '...']:
    '''Median / MAD-based robust z-score normalisation.

    Less sensitive to outliers than the mean / std version.
    The factor 1.4826 makes the MAD a consistent estimator of
    the standard deviation for Gaussian inputs.
    '''
    median, mad = _median_mad(x, axis=axis)
    return (x - median) / (1.4826 * mad + eps)


def intensity_normalize(
    x: Num[Array, '...'],
    *,
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
    axis: Optional[_AxisArg] = None,
    eps: float = 1e-12,
) -> Num[Array, '...']:
    '''Percentile clip then rescale to ``[0, 1]``.

    The synthstrip / SynthSeg pre-training convention: clip the
    intensity histogram to the ``[low_percentile, high_percentile]``
    range to suppress outliers (motion artefacts, bias-field
    inhomogeneity), then rescale the clipped range to ``[0, 1]``.

    Parameters
    ----------
    x
        Intensity tensor.
    low_percentile, high_percentile
        Clipping percentiles.  Default ``(1, 99)``.
    axis
        Axes over which to compute the percentiles.  ``None``
        (default) computes over the entire tensor (one pair of
        percentiles for the whole input).  Pass an axis tuple
        for per-slice normalisation (e.g. ``axis=(-3, -2, -1)``
        on a ``(B, H, W, D)`` volume gives per-batch
        normalisation).
    eps
        Stabiliser for the division when ``high == low``.
    '''
    lo = jnp.percentile(x, low_percentile, axis=axis, keepdims=True)
    hi = jnp.percentile(x, high_percentile, axis=axis, keepdims=True)
    clipped = jnp.clip(x, lo, hi)
    return (clipped - lo) / (hi - lo + eps)
