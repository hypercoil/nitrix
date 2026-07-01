# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Intensity preprocessing applied to ``(moving, fixed)`` at a recipe's entry.

The fMRIPrep front-end (4a / 4b): winsorize both images to their intensity
percentiles (``--winsorize-image-intensities``), then optionally match the
moving distribution to the fixed by CDF transport (``--use-histogram-matching``).
Both default off, so an unconfigured recipe is byte-unchanged.

Applied **once** at the recipe entry, so the conditioned images feed both the
pyramid (the registration the transform is estimated on) and the ``warped``
output.  The estimated transform (``matrix`` / ``displacement``) is the real
deliverable -- apply it to the *original* moving for the original intensities;
the returned ``warped`` is the conditioned moving resampled.
"""

from __future__ import annotations

from typing import Optional, Tuple

from jaxtyping import Array

from ..metrics import match_histogram, winsorize

__all__ = ['preprocess_images']


def preprocess_images(
    moving: Array,
    fixed: Array,
    *,
    winsorize_range: Optional[Tuple[float, float]],
    histogram_match: bool,
) -> Tuple[Array, Array]:
    """Condition an image pair before registration.

    Winsorise both images to a shared intensity percentile range, then
    optionally remap the moving image's intensity distribution onto the fixed
    image's by cumulative-distribution transport. When both steps are disabled
    the pair is returned byte-unchanged.

    Winsorisation uses :func:`~nitrix.metrics.winsorize` and histogram matching
    uses :func:`~nitrix.metrics.match_histogram`.

    Parameters
    ----------
    moving : Array
        The moving image, whose intensities are conditioned and (when histogram
        matching is enabled) mapped onto the fixed distribution.
    fixed : Array
        The fixed (reference) image. It is winsorised alongside the moving image
        and, when histogram matching is enabled, serves as the target
        distribution.
    winsorize_range : tuple of float or None
        A ``(lower, upper)`` pair of intensity percentiles to clip both images
        to. ``None`` disables winsorisation, leaving both images unchanged.
    histogram_match : bool
        If ``True``, remap the (winsorised) moving image onto the (winsorised)
        fixed distribution. If ``False``, the moving image is left as is.

    Returns
    -------
    moving : Array
        The conditioned moving image (winsorised, then optionally
        histogram-matched to ``fixed``).
    fixed : Array
        The conditioned fixed image (winsorised only).
    """
    if winsorize_range is not None:
        lo, hi = winsorize_range
        moving = winsorize(moving, lower=lo, upper=hi)
        fixed = winsorize(fixed, lower=lo, upper=hi)
    if histogram_match:
        moving = match_histogram(moving, fixed)
    return moving, fixed
