# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
# isort: skip_file
"""
nitrix.bias -- intensity bias-field correction.

The marquee item is ``n4_bias_field_correction`` -- the Tustison (2010)
N4 algorithm (the improved successor to N3), GPU-accelerated in pure JAX
and following the ITK / ANTs defaults.  It is a composite of two reusable
substrate primitives, both exposed in their own right:

- ``bspline_approximate`` -- separable cubic B-spline scattered-data
  approximation on a regular grid (the Lee--Wolberg--Shin MBA specialised
  to the image lattice; the field-smoothing engine of N4).  A fast,
  differentiable smoother for any "fit a smooth low-DOF surface to a noisy
  grid" task (registration fields, surface fitting).
- ``sharpen_histogram`` -- N3 / N4 Wiener log-histogram deconvolution (the
  intensity-deblurring engine of N4).

See ``docs/design/bias-field.md`` for the derivation, the equivalence to
ITK's control-point-lattice accumulation, and the "no Pallas needed"
rationale.
"""

from ._bspline import bspline_approximate
from ._sharpen import sharpen_histogram
from .n4 import n4_bias_field_correction

__all__ = [
    'n4_bias_field_correction',
    'bspline_approximate',
    'sharpen_histogram',
]
