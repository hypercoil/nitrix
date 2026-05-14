# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
# isort: skip_file
"""
nitrix.smoothing -- edge-preserving / non-edge-preserving smoothing.

Three tiers per SPEC_UPDATE §3.3:

- ``gaussian`` -- separable n-D Gaussian; unconditional baseline.
- ``bilateral_gaussian`` -- direct N-body bilateral with arbitrary
  feature dimensionality; the marquee edge-preserving capability.
  Built on ``semiring_ell_matmul`` over a feature-distance-weighted
  adjacency.  Practical for ``d_f <= 5`` and spatial neighbourhoods
  up to ~7³ voxels.
- ``permutohedral_lattice`` -- target with tripwire; not currently
  shipped.  The namespace is reserved; the symbol raises
  ``NotImplementedError`` pointing at ``bilateral_gaussian`` for
  ``d_f <= 5`` until the tripwire criteria are met.
"""
from .gaussian import gaussian
from .bilateral import bilateral_gaussian, brute_force_knn
from .susan import susan_emulator, spatial_cube_neighbourhood

__all__ = [
    'gaussian',
    'bilateral_gaussian',
    'brute_force_knn',
    'susan_emulator',
    'spatial_cube_neighbourhood',
]
