# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
# isort: skip_file
"""
nitrix.smoothing -- edge-preserving / non-edge-preserving smoothing.

Two tiers (SPEC §4.4):

- ``gaussian`` -- separable n-D Gaussian; unconditional baseline for
  cases where edge preservation is not wanted.
- ``bilateral_gaussian`` -- the marquee edge-preserving capability: a
  bounded high-dimensional bilateral filter over a feature-space metric
  (``FeatureMetric``) and a bounded neighbourhood (grid box, mesh
  k-ring, geodesic ball, or feature-space k-NN).  Built on
  ``semiring_ell_matmul``; one gather plus one weighted reduction,
  ``jit`` / ``vmap`` / ``grad`` clean.  Supersedes the retired
  permutohedral lattice for the feature dimensionalities we target (and,
  via a low-rank metric, beyond).

``susan_emulator`` is a convenience wrapper composing
``bilateral_gaussian`` with ``morphology.median_filter``.
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
