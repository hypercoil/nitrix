# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
# isort: skip_file
"""
nitrix.numerics -- general-purpose tensor utilities.

Submodules:

- ``tensor_ops`` -- shape / layout utilities (orient_and_conform,
  fold/unfold, broadcasting helpers).  These are the "low-level
  building blocks" that downstream layers compose; they were
  scattered across ``hypercoil.functional.linear`` and
  ``ilex.core.adapters`` before consolidation here.
- ``normalize``  -- intensity normalisation routines (synthstrip
  median, robust z-scoring, etc.).
- ``spatial``    -- volumetric shape / windowing utilities: pad / crop
  to a multiple, nonzero bounding box, Gaussian patch window +
  overlap-add normalisation.
- ``ode``        -- ODE integrators (euler / midpoint / rk4 /
  local_linearization)
  for continuous-time / neural-ODE models.
- ``fixed_point`` -- :func:`fixed_point_solve` (the implicit-VJP
  fixed-point iteration), re-exported at the package top level.
"""

from . import cluster, normalize, ode, spatial, tensor_ops
from .cluster import (
    KMeansState,
    Similarity,
    kmeans,
    kmeans_fit,
    kmeans_predict,
)
from .fixed_point import fixed_point_solve
from .ode import euler, local_linearization, midpoint, odeint, rk4
from .spatial import (
    crop_to_multiple,
    gaussian_window,
    nonzero_bounding_box,
    overlap_add,
    pad_to_multiple,
)
from .normalize import (
    demean,
    instance_norm,
    intensity_normalize,
    l2_normalize,
    lp_normalize,
    percentile_rescale,
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
    # clustering
    'kmeans',
    'kmeans_fit',
    'kmeans_predict',
    'KMeansState',
    'Similarity',
    # fixed point
    'fixed_point_solve',
    # ode integrators
    'euler',
    'midpoint',
    'rk4',
    'local_linearization',
    'odeint',
    # spatial shape / windowing
    'pad_to_multiple',
    'crop_to_multiple',
    'nonzero_bounding_box',
    'gaussian_window',
    'overlap_add',
    # normalize
    'demean',
    'instance_norm',
    'intensity_normalize',
    'l2_normalize',
    'lp_normalize',
    'percentile_rescale',
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
