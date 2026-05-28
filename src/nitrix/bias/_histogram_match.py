# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Nyul-Udupa landmark histogram matching.

Remap a ``source`` image's intensities so its CDF matches that of a
``reference`` image.  Nyul & Udupa (1999), "On Standardizing the MR Image
Intensity Scale" (IEEE TMI): pick ``N`` landmarks at evenly-spaced
quantile positions on both CDFs, then apply a piecewise-linear
interpolant from source-landmark to reference-landmark intensities.

Distinct from ``sharpen_histogram`` (N3 Wiener log-histogram
*deconvolution* of a single image's histogram against a Gaussian PSF, to
de-blur within-image contrast).  Histogram *matching* operates on two
images and remaps one toward the other's distribution; it does not
deconvolve.

ITK parity target.  The implementation is the ITK
``HistogramMatchingImageFilter`` algorithm:

- The **outer landmarks are always the image extrema** ``(image.min(),
  image.max())`` -- not the extrema of the histogram-contributing
  population.  This is what keeps the apply step interpolation-only: any
  source voxel lies inside ``[src.min(), src.max()]`` by construction,
  so no extrapolation regime is needed.
- With ``threshold_at_mean=True`` (ITK default) **and** no explicit
  weight, the image mean is inserted as an additional landmark right
  after the lower extremum: the landmark array becomes ``[image.min(),
  image.mean(), m_1, ..., m_N, image.max()]``, with match points
  ``m_1..m_N`` at evenly-spaced quantile positions on the
  above-mean voxels.  This is what ITK's ``ThresholdAtMeanIntensityOn``
  actually does -- the mean is **both** the histogram threshold *and* a
  literal landmark, so every below-mean voxel is interpolated between
  ``(image.min(), image.mean())`` instead of being lumped into the
  above-mean ladder.
- With ``threshold_at_mean=False`` (or an explicit weight given), the
  landmark array is the simpler ``[image.min(), m_1, ..., m_N,
  image.max()]`` with match points on the full (or weight-filtered)
  population.

Static shapes throughout (``n_histogram_levels`` and ``n_match_points``
are fixed at trace time; the weighting is a multiplicative mask, not a
boolean-index resize), so the op JITs cleanly.

Not differentiable through the histogram binning (piecewise constant in
source values) -- the same posture as ``sharpen_histogram``.  Gradients
flow through reference landmarks via ``jnp.interp``'s autograd; the
landmark identification step is treated as a constant.

See ``docs/design/bias-field.md`` for the place this primitive holds
inside the bias-correction surface and the recommended N4 -> match
composition.
"""

from __future__ import annotations

from typing import Optional, Tuple

import jax.numpy as jnp
from jaxtyping import Array, Float

__all__ = ['histogram_match']


def _match_points(
    image: Float[Array, '...'],
    weight: Float[Array, '...'],
    bin_min: Float[Array, ''],
    bin_max: Float[Array, ''],
    *,
    n_bins: int,
    n_match_points: int,
    dtype: jnp.dtype,
) -> Float[Array, 'n_match_points']:
    '''Quantile-positioned landmark intensities from a weighted histogram.

    Builds an ``n_bins``-resolution histogram of ``image`` weighted by
    ``weight`` over the interval ``[bin_min, bin_max]``, then for each
    of ``n_match_points`` cumulative-frequency targets
    ``i / (n_match_points + 1)`` (``i = 1..n_match_points``) locates the
    intensity at which the cumulative frequency equals the target
    exactly, via ``searchsorted`` + linear interpolation across the bin
    boundary.  The latter sub-bin step is what makes the landmarks not
    quantised to ``(bin_max - bin_min) / n_bins`` precision.
    '''
    flat = image.reshape(-1).astype(dtype)
    w = weight.reshape(-1).astype(dtype)
    span = bin_max - bin_min
    slope = jnp.where(span > 0, span / (n_bins - 1), 1.0)

    idx = jnp.floor((flat - bin_min) / slope).astype(jnp.int32)
    idx = jnp.clip(idx, 0, n_bins - 1)
    hist = jnp.zeros((n_bins,), dtype=dtype).at[idx].add(w)

    cum = jnp.cumsum(hist)
    total = cum[-1]
    safe_total = jnp.where(total > 0, total, 1.0)
    cum_frac = cum / safe_total

    quantiles = (
        jnp.arange(1, n_match_points + 1, dtype=dtype)
        / (n_match_points + 1)
    )

    bin_hi = jnp.searchsorted(cum_frac, quantiles, side='left')
    bin_hi = jnp.clip(bin_hi, 0, n_bins - 1)
    bin_lo = jnp.maximum(bin_hi - 1, 0)
    cum_hi = cum_frac[bin_hi]
    cum_lo = jnp.where(bin_hi > 0, cum_frac[bin_lo], 0.0)
    denom = cum_hi - cum_lo
    frac = jnp.where(denom > 0, (quantiles - cum_lo) / denom, 0.0)
    # The cumulative-frequency target lands somewhere inside bin
    # ``bin_hi``; the intensity at the left edge of ``bin_hi`` is
    # ``bin_min + bin_hi * slope`` and the bin is of width ``slope``, so
    # the landmark is ``bin_min + (bin_hi + frac) * slope``.
    return bin_min + (bin_hi.astype(dtype) + frac) * slope


def _landmarks_for(
    image: Float[Array, '...'],
    weight: Optional[Float[Array, '...']],
    threshold_at_mean: bool,
    *,
    n_bins: int,
    n_match_points: int,
    dtype: jnp.dtype,
) -> Tuple[Float[Array, '...'], bool]:
    '''Build the full landmark array for one image.

    Returns ``(landmarks, has_mean)`` -- the array is one entry longer
    when the mean is inserted as an extra landmark (the ITK
    ``ThresholdAtMeanIntensityOn``-with-no-explicit-weight case).
    Both source and reference must agree on ``has_mean`` (or the apply
    step would be malformed) -- the caller is responsible for
    consistency.
    '''
    image = image.astype(dtype)
    bin_min = jnp.min(image)
    bin_max = jnp.max(image)

    has_mean = threshold_at_mean and weight is None
    if has_mean:
        mean = jnp.mean(image)
        w = (image > mean).astype(dtype)
        # Match-point histogram lives over [mean, bin_max] -- below-mean
        # voxels are zero-weighted and would otherwise concentrate in
        # the lowest bin with no contribution, wasting half the
        # resolution.
        m = _match_points(
            image, w, mean, bin_max,
            n_bins=n_bins, n_match_points=n_match_points, dtype=dtype,
        )
        return jnp.concatenate(
            [bin_min[None], mean[None], m, bin_max[None]]
        ), True

    if weight is None:
        w = jnp.ones_like(image)
    else:
        w = jnp.broadcast_to(jnp.asarray(weight).astype(dtype), image.shape)

    m = _match_points(
        image, w, bin_min, bin_max,
        n_bins=n_bins, n_match_points=n_match_points, dtype=dtype,
    )
    return jnp.concatenate([bin_min[None], m, bin_max[None]]), False


def histogram_match(
    source: Float[Array, '*spatial'],
    reference: Float[Array, '*spatial'],
    *,
    source_weight: Optional[Float[Array, '*spatial']] = None,
    reference_weight: Optional[Float[Array, '*spatial']] = None,
    n_match_points: int = 7,
    n_histogram_levels: int = 1024,
    threshold_at_mean: bool = True,
) -> Float[Array, '*spatial']:
    '''Remap ``source`` intensities so its CDF matches ``reference``'s.

    Nyul-Udupa landmark histogram matching: pick ``n_match_points``
    landmarks at evenly-spaced quantile positions on the source and
    reference cumulative distributions, then apply a piecewise-linear
    interpolant from source-landmark intensities to the corresponding
    reference-landmark intensities.

    Parameters
    ----------
    source
        Image to remap, ``(*spatial)``.  The entire tensor is treated as
        a single intensity population (one histogram).  For independent
        per-volume matching, ``jax.vmap`` over the batch.
    reference
        Image whose intensity distribution defines the target CDF.
        Shape does not have to match ``source``.
    source_weight, reference_weight
        Optional masks / confidences, broadcastable to the matching
        image's shape.  Only voxels with non-zero weight contribute to
        the corresponding histogram for match-point identification; the
        outer landmarks are always the full ``image.min()`` /
        ``image.max()``.  Passing an explicit weight overrides the
        ITK-style ``threshold_at_mean`` policy on that side.
    n_match_points
        Number of landmark match points (excluding the outer extrema
        and the optional mean landmark).  Default ``7``, matching ITK's
        ``HistogramMatchingImageFilter`` default.
    n_histogram_levels
        Histogram resolution used to locate match-point intensities.
        Default ``1024`` (ITK default).  Sub-bin resolution is recovered
        by linear interpolation across the bin boundary at the
        cumulative-frequency target, so the landmarks are not quantised
        to ``1 / n_histogram_levels``.
    threshold_at_mean
        ITK's ``ThresholdAtMeanIntensityOn``: when ``True`` (default)
        and the corresponding ``..._weight`` is ``None``, only voxels
        with intensity above the image's mean contribute to its
        match-point histogram, **and** the mean itself is inserted as
        an additional landmark just above the lower extremum.  Set to
        ``False`` to use every voxel and skip the mean landmark.
        Ignored for an image whose weight is supplied explicitly.

    Returns
    -------
    The remapped image, same shape and dtype as ``source``.  Outer
    landmarks are the actual image extrema, so the apply is
    interpolation-only -- no extrapolation regime to choose.

    Notes
    -----
    Not differentiable through the bin assignment (the landmark search
    is piecewise constant in source values).  Gradients through the
    apply step flow via ``jnp.interp``'s autograd; treat the landmarks
    as constants from the source side.  Static shapes throughout, so
    the op JITs cleanly under ``jit`` / ``vmap``.

    For the apply step to be well-defined, the source and reference
    landmark arrays must have the same length.  This requires the
    ``threshold_at_mean`` policy to resolve to the same answer on both
    sides (either both insert a mean landmark or neither does); when it
    does not -- e.g.  ``threshold_at_mean=True`` with an explicit
    ``source_weight`` but ``reference_weight=None`` -- ``ValueError`` is
    raised.

    See Also
    --------
    sharpen_histogram :
        N3 Wiener log-histogram deconvolution -- a different op on a
        single image's histogram (no reference involved).
    '''
    if n_match_points < 1:
        raise ValueError(
            f'n_match_points must be >= 1, got {n_match_points}'
        )
    if n_histogram_levels < 2:
        raise ValueError(
            f'n_histogram_levels must be >= 2, got {n_histogram_levels}'
        )
    src_has_mean = threshold_at_mean and source_weight is None
    ref_has_mean = threshold_at_mean and reference_weight is None
    if src_has_mean != ref_has_mean:
        raise ValueError(
            'threshold_at_mean inserts a mean landmark, but only when '
            'no explicit weight is given on that side; the source and '
            'reference must agree.  Pass weights on both sides or '
            'neither, or set threshold_at_mean=False.'
        )

    src = jnp.asarray(source)
    ref = jnp.asarray(reference)
    out_dtype = src.dtype
    dtype = jnp.result_type(src.dtype, ref.dtype, jnp.float32)

    src_landmarks, _ = _landmarks_for(
        src, source_weight, threshold_at_mean,
        n_bins=n_histogram_levels, n_match_points=n_match_points,
        dtype=dtype,
    )
    ref_landmarks, _ = _landmarks_for(
        ref, reference_weight, threshold_at_mean,
        n_bins=n_histogram_levels, n_match_points=n_match_points,
        dtype=dtype,
    )

    mapped = jnp.interp(
        src.astype(dtype).reshape(-1), src_landmarks, ref_landmarks,
    )
    return mapped.reshape(src.shape).astype(out_dtype)
