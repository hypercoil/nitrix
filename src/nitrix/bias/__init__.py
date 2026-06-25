# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
# isort: skip_file
"""
nitrix.bias -- intensity bias-field correction.

Two public entry points:

- ``n4_bias_field_correction`` -- the Tustison (2010) N4 algorithm (the
  improved successor to N3), GPU-accelerated in pure JAX and validated to
  ITK / ANTs parity.  This is *specifically* N4 (the Lee--Wolberg--Shin MBA
  field fit).
- ``bias_field_correction`` -- a dispatcher over the same N3/N4 iteration
  with a selectable field estimator: ``method='n4'`` (parity, == above),
  ``'least_squares'`` and ``'psplines'`` (unbiased, higher-accuracy
  estimators for new / internal pipelines where ANTs bit-compatibility is
  not required).

Plus three reusable substrate primitives the above are built from
(``bspline_approximate`` / ``sharpen_histogram``) or compose with
naturally (``histogram_match``):

- ``bspline_approximate`` -- separable cubic B-spline scattered-data
  approximation on a regular grid (MBA, least-squares, or P-spline fit),
  a fast differentiable smoother for any "fit a smooth low-DOF surface to
  a noisy grid" task (registration fields, surface fitting).
- ``sharpen_histogram`` -- N3 / N4 Wiener log-histogram deconvolution
  (single-image histogram de-blurring).
- ``histogram_match`` -- Nyul-Udupa landmark histogram matching (Nyul &
  Udupa 1999): remap a source image's intensities so its CDF matches a
  reference image's, ITK-faithful to
  ``HistogramMatchingImageFilter``.  Composes with N4 as an "N4 then
  match to a reference template" intensity-standardise recipe.  The
  ``histogram_match_fit`` / ``histogram_match_apply`` split exposes its
  fit/apply seam (§6.5) -- fit a reference's ~9 standard-scale landmarks
  once, apply to many sources -- with ``histogram_match`` the convenience
  composition.

See ``docs/design/bias-field.md`` for the derivation, the parity-vs-
correctness discussion, and the "no Pallas needed" rationale.
"""

from ._bspline import bspline_approximate
from ._histogram_match import (
    histogram_match,
    histogram_match_apply,
    histogram_match_fit,
)
from ._sharpen import sharpen_histogram
from .correction import bias_field_correction
from .n4 import n4_bias_field_correction

__all__ = [
    'n4_bias_field_correction',
    'bias_field_correction',
    'bspline_approximate',
    'sharpen_histogram',
    'histogram_match',
    'histogram_match_fit',
    'histogram_match_apply',
]
