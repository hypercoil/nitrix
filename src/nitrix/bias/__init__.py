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

Plus two reusable substrate primitives the above are built from:

- ``bspline_approximate`` -- separable cubic B-spline scattered-data
  approximation on a regular grid (MBA, least-squares, or P-spline fit),
  a fast differentiable smoother for any "fit a smooth low-DOF surface to
  a noisy grid" task (registration fields, surface fitting).
- ``sharpen_histogram`` -- N3 / N4 Wiener log-histogram deconvolution.

See ``docs/design/bias-field.md`` for the derivation, the parity-vs-
correctness discussion, and the "no Pallas needed" rationale.
"""

from ._bspline import bspline_approximate
from ._sharpen import sharpen_histogram
from .correction import bias_field_correction
from .n4 import n4_bias_field_correction

__all__ = [
    'n4_bias_field_correction',
    'bias_field_correction',
    'bspline_approximate',
    'sharpen_histogram',
]
