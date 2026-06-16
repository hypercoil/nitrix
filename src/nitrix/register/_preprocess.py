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
    """Winsorize both images, then (optionally) histogram-match moving -> fixed.

    ``winsorize_range`` is a ``(lower, upper)`` percentile pair (``None`` -> no
    winsorisation); ``histogram_match`` remaps the (winsorised) moving onto the
    (winsorised) fixed distribution.  A no-op when both are off.
    """
    if winsorize_range is not None:
        lo, hi = winsorize_range
        moving = winsorize(moving, lower=lo, upper=hi)
        fixed = winsorize(fixed, lower=lo, upper=hi)
    if histogram_match:
        moving = match_histogram(moving, fixed)
    return moving, fixed
