# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
# isort: skip_file
"""
nitrix.stats.lme -- voxelwise linear mixed-effects models.

Two layered primitives:

- ``reml_fit`` -- general variance-components REML via the
  FaST-LMM spectral trick (Lippert et al. 2011).  For a model
  ``V = sigma_b^2 ZZ^T + sigma_e^2 I``, eigendecompose ``ZZ^T``
  once (shared across voxels); rotate ``y`` and ``X`` into the
  diagonalising basis; per-voxel inner loop becomes ``O(N)``
  per Newton step instead of the naive ``O(N^3)``.  Vmap-batched
  over voxels with shared design.
- ``flame_two_level`` -- the FSL FLAME equivalent: per-subject
  within-variance is known (from a level-1 GLM); between-subject
  variance is estimated.  Naturally diagonal-``V``, so the
  spectral trick is trivial.  Built on top of the general REML
  inner solver.

Both fits use Newton scoring on the log-parameterised variance
components (positivity automatic; quadratic local convergence).
Fixed-iteration scan (default 20) for clean vmap composition;
caller can tune via ``n_iter``.  Differentiable through the
output statistics via the implicit-function-theorem VJP (see
``docs/design/lme.md``).
"""

from ._corr import CorrSpec, ar1, car1, cs
from ._corrfit import GLSResult, gls_fit
from .flame import FLAMEResult, flame_two_level
from .reml import (
    LMEContrast,
    LMEFContrast,
    LMEResult,
    NestedLMEResult,
    REMLResult,
    lme_f_contrast,
    lme_fit,
    lme_t_contrast,
    reml_fit,
)


__all__ = [
    'REMLResult',
    'LMEResult',
    'NestedLMEResult',
    'LMEContrast',
    'LMEFContrast',
    'FLAMEResult',
    'GLSResult',
    'CorrSpec',
    'reml_fit',
    'lme_fit',
    'lme_t_contrast',
    'lme_f_contrast',
    'flame_two_level',
    'gls_fit',
    'ar1',
    'car1',
    'cs',
]
