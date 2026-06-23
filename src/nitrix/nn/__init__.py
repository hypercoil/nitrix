# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
# isort: skip_file
"""
nitrix.nn -- functional neural-network forward-block kernels.

Pure ``(Array, ...) -> Array`` contractions lifted out of the ilex/nimox
transformer and state-space modules, behind the standard ``backend=``
dispatch (a ``pallas-cuda`` fused fast path with a ``jax`` reference
fallback).  These are *functions*, not Equinox modules: the learnable
modules that hold the parameters stay in nimox; nitrix owns only the
inner kernel.

Submodules:

- ``attention`` -- ``scaled_dot_product_attention`` (dense / windowed
  additive-bias / causal / cross), with a flash-attention fast path.
- ``ssm``       -- ``selective_scan`` (Mamba / S6), with a parallel
  ``associative_scan`` GPU path + a fused chunked-scan kernel.
- ``norm``      -- ``layer_norm`` / ``group_norm`` / ``instance_norm`` with the
  curse-of-depth ``out_scale`` hook; fused single-pass kernel is perf-gated.

See ``docs/feature-requests/nn-forward-kernels-suite.md``.
"""

from . import attention, norm, ssm
from .attention import scaled_dot_product_attention
from .norm import group_norm, instance_norm, layer_norm
from .ssm import selective_scan

__all__ = [
    'attention',
    'ssm',
    'norm',
    'scaled_dot_product_attention',
    'selective_scan',
    'layer_norm',
    'group_norm',
    'instance_norm',
]
