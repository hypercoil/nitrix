# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
# isort: skip_file
"""
Intensity bias-field correction.

Differentiable, pure-JAX estimation and removal of the slowly varying
multiplicative intensity inhomogeneity ("bias field") that corrupts MR
images, together with the reusable histogram and B-spline primitives the
correction algorithms are built from.

There are two public entry points for correction:

- :func:`n4_bias_field_correction` -- the N4 algorithm, the improved
  successor to N3, GPU-accelerated in pure JAX and validated to ITK / ANTs
  parity. This is *specifically* N4 (the Lee--Wolberg--Shin multilevel
  B-spline field fit).
- :func:`bias_field_correction` -- a dispatcher over the same N3/N4
  iteration with a selectable field estimator: ``method='n4'`` (parity,
  identical to the above), ``'least_squares'`` and ``'psplines'``
  (unbiased, higher-accuracy estimators for new or internal pipelines where
  bit-for-bit ANTs compatibility is not required).

Alongside these are three reusable substrate primitives that the
correction algorithms are built from (:func:`bspline_approximate`,
:func:`sharpen_histogram`) or compose with naturally
(:func:`histogram_match`):

- :func:`bspline_approximate` -- separable cubic B-spline scattered-data
  approximation on a regular grid (multilevel B-spline, least-squares, or
  P-spline fit); a fast differentiable smoother for any "fit a smooth
  low-degree-of-freedom surface to a noisy grid" task, such as registration
  fields or surface fitting.
- :func:`sharpen_histogram` -- the N3 / N4 Wiener log-histogram
  deconvolution (single-image histogram de-blurring).
- :func:`histogram_match` -- Nyul--Udupa landmark histogram matching: remap
  a source image's intensities so that its cumulative distribution matches
  a reference image's, faithful to ITK's
  ``HistogramMatchingImageFilter``. It composes with N4 as an "N4 then
  match to a reference template" intensity-standardisation recipe. The
  :func:`histogram_match_fit` / :func:`histogram_match_apply` split exposes
  the fit/apply seam -- fit a reference's standard-scale landmarks once and
  apply them to many sources -- with :func:`histogram_match` the
  single-call convenience composition.

References
----------
Tustison, N. J., Avants, B. B., Cook, P. A., Zheng, Y., Egan, A.,
Yushkevich, P. A., & Gee, J. C. (2010). N4ITK: improved N3 bias
correction. *IEEE Transactions on Medical Imaging*, 29(6), 1310--1320.
:doi:`10.1109/TMI.2010.2046908`

Nyul, L. G., & Udupa, J. K. (1999). On standardizing the MR image
intensity scale. *Magnetic Resonance in Medicine*, 42(6), 1072--1081.
:doi:`10.1002/(SICI)1522-2594(199912)42:6<1072::AID-MRM11>3.0.CO;2-M`
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
