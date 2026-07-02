# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
# isort: skip_file
"""
Voxelwise linear mixed-effects models.

Two layered primitives are exposed:

- :func:`reml_fit` -- general variance-components REML via the
  FaST-LMM spectral trick.  For a model
  :math:`V = \\sigma_b^2 ZZ^{\\top} + \\sigma_e^2 I`, the term
  :math:`ZZ^{\\top}` is eigendecomposed once (shared across
  voxels); ``y`` and ``X`` are rotated into the diagonalising
  basis, so the per-voxel inner loop becomes :math:`O(N)` per
  Newton step instead of the naive :math:`O(N^3)`.  The fit is
  vmap-batched over voxels with a shared design.
- :func:`flame_two_level` -- the FSL FLAME equivalent: the
  per-subject within-variance is known (from a level-1 GLM) and
  the between-subject variance is estimated.  Here :math:`V` is
  naturally diagonal, so the spectral trick is trivial.  This is
  built on top of the general REML inner solver.

Both fits use Newton scoring on the log-parameterised variance
components, which makes positivity automatic and gives quadratic
local convergence.  A fixed-iteration scan (default 20) is used
for clean vmap composition; the caller can tune this via the
``n_iter`` argument.  The output statistics are differentiable
via an implicit-function-theorem VJP.

References
----------
.. [1] Lippert, C., Listgarten, J., Liu, Y., Kadie, C. M.,
   Davidson, R. I., & Heckerman, D. (2011). FaST linear mixed
   models for genome-wide association studies. Nature Methods,
   8(10), 833-835. https://doi.org/10.1038/nmeth.1681
"""

from ._blup import lme_predict, ranef
from ._corr import CorrSpec, ar1, car1, cs, iid
from ._corrfit import CorrLMEResult, GLSResult, gls_fit
from ._varfunc import VarFunc, var_ident, var_power
from .flame import FLAMEResult, flame_two_level
from .reml import (
    CrossedLMEResult,
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
    'CrossedLMEResult',
    'LMEContrast',
    'LMEFContrast',
    'FLAMEResult',
    'GLSResult',
    'CorrLMEResult',
    'CorrSpec',
    'VarFunc',
    'reml_fit',
    'lme_fit',
    'lme_predict',
    'ranef',
    'lme_t_contrast',
    'lme_f_contrast',
    'flame_two_level',
    'gls_fit',
    'ar1',
    'car1',
    'cs',
    'iid',
    'var_power',
    'var_ident',
]
