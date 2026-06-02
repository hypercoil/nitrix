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
- ``percentile_rescale``: shift by a low percentile, scale by a
  high percentile, then clip -- the strict min--p99--clip recipe
  used by the Synth* pipelines (distinct from
  ``intensity_normalize``; see its docstring).

The z-score path additionally supports ``nonzero_mask=True`` for
the per-channel BraTS / nnUNet convention (statistics over the
nonzero foreground; background left at ``0``).

All are differentiable and support per-axis reduction via
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
    'percentile_rescale',
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
    nonzero_mask: bool = False,
) -> Num[Array, '...']:
    '''Z-score normalisation: ``(x - mean) / (std + eps)`` along ``axis``.

    Population standard deviation (no Bessel correction).
    Per-cell normalisation; output has zero mean and unit std
    along the named axis.

    Parameters
    ----------
    x
        Input tensor.
    axis
        Axis (or axes) over which the mean / std are computed.
        Default ``-1`` (the trailing observation axis).  For a
        per-channel statistic on a channel-first multimodal volume
        ``(C, *spatial)``, pass ``axis`` covering the spatial axes
        (e.g. ``axis=(-3, -2, -1)``).
    weights
        Optional weights for the mean / std (see module docstring).
    eps
        Denominator stabiliser.
    nonzero_mask
        If ``True``, compute the mean / std over the **nonzero**
        entries only (``mask = x != 0``) and leave the zero entries
        at exactly ``0`` in the output (rather than mapping them to
        ``-mean / std``).  This is the per-channel BraTS / nnUNet
        convention: the background outside the brain stays ``0``
        while the foreground is z-scored against foreground
        statistics.  Mutually exclusive with an explicit
        ``weights`` (the mask *is* the weight).

    Returns
    -------
    Z-scored tensor, same shape as ``x``.
    '''
    if nonzero_mask:
        if weights is not None:
            raise ValueError(
                'zscore_normalize: pass either weights= or '
                'nonzero_mask=True, not both -- the nonzero mask is '
                'the weight.'
            )
        mask = x != 0
        mean, std = _weighted_mean_std(
            x, axis=axis, weights=mask.astype(x.dtype),
        )
        z = (x - mean) / (std + eps)
        return jnp.where(mask, z, jnp.zeros_like(z))
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


def percentile_rescale(
    x: Num[Array, '...'],
    *,
    lo: float = 0.0,
    hi: float = 99.0,
    clip: bool = True,
    axis: Optional[_AxisArg] = None,
    eps: float = 1e-12,
) -> Num[Array, '...']:
    '''Shift by the ``lo``-percentile, scale by the ``hi``-percentile,
    then optionally clip to ``[0, 1]``.

    Computes ``(x - p_lo) / p_hi`` where ``p_lo`` / ``p_hi`` are the
    ``lo`` / ``hi`` percentiles of ``x``, then clips to ``[0, 1]``
    when ``clip`` is set.  With the defaults ``lo=0`` (the minimum)
    and ``hi=99`` this is the **min--p99--clip** recipe used by
    SynthStrip / SynthDist / SynthSR (and SynthSR after its CT
    intensity clip): ``clip((x - min) / p99, 0, 1)``.

    Distinct from ``intensity_normalize`` in two ways, both
    deliberate:

    1. The scale is the **raw high percentile** ``p_hi`` (not the
       inter-percentile width ``p_hi - p_lo``), so the upper
       reference is an absolute intensity, not a range endpoint.
    2. It rescales **then** clips (``intensity_normalize`` clips
       then rescales), so values above ``p_hi`` saturate at ``1``
       rather than defining the top of the range.

    Use ``intensity_normalize`` for symmetric two-sided
    percentile-window rescaling; use this for the strict
    min/high-percentile convention the Synth* pipelines expect.

    Parameters
    ----------
    x
        Intensity tensor.
    lo, hi
        Lower / upper percentiles.  Defaults ``(0, 99)`` (minimum
        and 99th percentile).
    clip
        Clip the result to ``[0, 1]``.  Default ``True``.
    axis
        Axes over which to compute the percentiles.  ``None``
        (default) uses the whole tensor; pass an axis tuple for
        per-slice / per-channel rescaling.
    eps
        Stabiliser added to the ``p_hi`` denominator (guards the
        all-constant / all-zero input).

    Returns
    -------
    Rescaled tensor, same shape as ``x``.  In ``[0, 1]`` when
    ``clip`` is set.
    '''
    p_lo = jnp.percentile(x, lo, axis=axis, keepdims=True)
    p_hi = jnp.percentile(x, hi, axis=axis, keepdims=True)
    out = (x - p_lo) / (p_hi + eps)
    if clip:
        out = jnp.clip(out, 0.0, 1.0)
    return out
