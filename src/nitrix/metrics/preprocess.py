# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Intensity preprocessing for registration (the fMRIPrep front-end).

The two within-/cross-modality conditioning steps an ``antsRegistration`` run
applies before the metric sees the data:

- :func:`winsorize` -- clip each image to its low/high intensity percentiles
  (``--winsorize-image-intensities [0.005, 0.995]``), so outliers (hot voxels,
  skull, artifacts) do not dominate an SSD / Demons force.  Also relieves the
  step-normalisation global-outlier sensitivity (the clipped image has no
  extreme voxel to set the trust-region cap).
- :func:`match_histogram` -- remap the moving image's intensity distribution onto
  the reference's by CDF / quantile transport (``--use-histogram-matching 1``), so
  a within-modality SSD / CC metric is not fighting a global intensity
  gain/offset.

Both are pure and differentiable (clip + sort-based quantiles + linear
interpolation), so they compose with the differentiable-layer story.
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float

__all__ = ['winsorize', 'match_histogram']


def winsorize(
    image: Float[Array, '...'],
    *,
    lower: float = 0.005,
    upper: float = 0.995,
) -> Float[Array, '...']:
    """Clip ``image`` to its ``[lower, upper]`` intensity percentiles.

    The fMRIPrep ``--winsorize-image-intensities [0.005, 0.995]`` step: the
    bottom / top tails are clamped to the percentile values, so a handful of
    extreme voxels cannot dominate the intensity-based cost.  Pure and
    differentiable.

    Parameters
    ----------
    image
        Any-shape intensity array (the percentiles are over **all** elements).
    lower, upper
        Lower / upper clip percentiles in ``[0, 1]`` (default ``0.005`` /
        ``0.995``, the ANTs / fMRIPrep convention).

    Returns
    -------
    Float[Array, '...']
        The intensity array with the same shape as ``image``, its values
        clamped to lie within the ``lower`` and ``upper`` intensity percentiles.
    """
    if not 0.0 <= lower < upper <= 1.0:
        raise ValueError(
            f'require 0 <= lower < upper <= 1; got lower={lower}, upper={upper}.'
        )
    lo, hi = jnp.quantile(
        image, jnp.asarray([lower, upper], dtype=image.dtype)
    )
    return jnp.clip(image, lo, hi)


def match_histogram(
    moving: Float[Array, '...'],
    reference: Float[Array, '...'],
    *,
    bins: int = 256,
) -> Float[Array, '...']:
    """Match the intensity histogram of ``moving`` to that of ``reference``.

    Remaps the intensities of ``moving`` by quantile transport so that its
    distribution matches ``reference``'s.

    The fMRIPrep ``--use-histogram-matching 1`` step for within-modality stages:
    each moving intensity is mapped to the reference intensity at the same
    quantile (CDF transport), so the two images share an intensity distribution
    and an SSD / CC metric is not pulled off by a global gain / offset.

    Implemented by interpolating ``moving`` against the two images' quantile
    curves (``bins`` quantiles each) -- differentiable, ``O(N log N)`` (the
    quantile sorts), paid once before the pyramid.

    Parameters
    ----------
    moving
        Image to remap.
    reference
        Image whose distribution is matched (the registration ``fixed``).
    bins
        Number of quantile knots in the transport map (default ``256``).

    Returns
    -------
    Float[Array, '...']
        The remapped moving image, with the same shape as ``moving``, whose
        intensity distribution matches that of ``reference``.
    """
    qs = jnp.linspace(0.0, 1.0, bins, dtype=moving.dtype)
    m_q = jnp.quantile(moving.reshape(-1), qs)  # moving's quantile values
    r_q = jnp.quantile(
        reference.reshape(-1), qs
    )  # reference's, same quantiles
    # For each moving voxel: its position on the moving quantile curve (its
    # quantile) read off the reference quantile curve = the matched intensity.
    matched = jnp.interp(moving.reshape(-1), m_q, r_q)
    return matched.reshape(moving.shape)
