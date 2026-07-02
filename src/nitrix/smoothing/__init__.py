# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
# isort: skip_file
"""
Edge-preserving and non-edge-preserving smoothing.

This subpackage provides two tiers of smoothing.

- :func:`gaussian` -- separable n-D Gaussian; the unconditional baseline
  for cases where edge preservation is not wanted.
- :func:`bilateral_gaussian` -- the marquee edge-preserving capability: a
  bounded high-dimensional bilateral filter over a feature-space metric
  (a :class:`FeatureMetric`) and a bounded neighbourhood (grid box, mesh
  k-ring, geodesic ball, or feature-space k-NN). It is built on the
  semiring ELL matrix-multiply primitive as one gather plus one weighted
  reduction, and is clean under ``jit`` / ``vmap`` / ``grad``. It
  supersedes the retired permutohedral lattice for the feature
  dimensionalities targeted here (and, via a low-rank metric, beyond).

:func:`susan_emulator` is a convenience wrapper composing
:func:`bilateral_gaussian` with the median filter from the ``morphology``
subpackage.
"""

from .gaussian import gaussian
from .metric import (
    DiagonalMetric,
    FactorMetric,
    FeatureMetric,
    block_diagonal_metric,
    metric_from_spd,
)
from .bilateral import bilateral_gaussian, brute_force_knn
from .susan import susan_emulator, spatial_cube_neighbourhood

__all__ = [
    'gaussian',
    'FeatureMetric',
    'DiagonalMetric',
    'FactorMetric',
    'block_diagonal_metric',
    'metric_from_spd',
    'bilateral_gaussian',
    'brute_force_knn',
    'susan_emulator',
    'spatial_cube_neighbourhood',
]
