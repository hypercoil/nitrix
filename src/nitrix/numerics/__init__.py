# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
# isort: skip_file
"""
nitrix.numerics -- general-purpose tensor utilities.

Two submodules:

- ``tensor_ops`` -- shape / layout utilities (orient_and_conform,
  fold/unfold, broadcasting helpers).  These are the "low-level
  building blocks" that downstream layers compose; they were
  scattered across ``hypercoil.functional.linear`` and
  ``ilex.core.adapters`` before consolidation here.
- ``normalize``  -- intensity normalisation routines (synthstrip
  median, robust z-scoring, etc.).

This subpackage didn't exist in the legacy code (the utilities
lived in ``functional.linear`` etc.); it's a Phase 1 rename and
consolidation.
"""

from . import normalize, tensor_ops
from .normalize import (
    demean,
    intensity_normalize,
    psc_normalize,
    robust_zscore_normalize,
    zscore_normalize,
)
from .tensor_ops import (
    apply_mask,
    broadcast_ignoring,
    complex_decompose,
    complex_recompose,
    conform_mask,
    fold_axis,
    orient_and_conform,
    promote_to_rank,
    unfold_axes,
)

__all__ = [
    # normalize
    'demean',
    'intensity_normalize',
    'psc_normalize',
    'robust_zscore_normalize',
    'zscore_normalize',
    # tensor_ops (most useful subset)
    'apply_mask',
    'broadcast_ignoring',
    'complex_decompose',
    'complex_recompose',
    'conform_mask',
    'fold_axis',
    'orient_and_conform',
    'promote_to_rank',
    'unfold_axes',
]
