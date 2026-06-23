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
- ``ssm``       -- ``selective_scan`` (Mamba / S6).  *(planned -- suite P1)*
- ``norm``      -- fused layer / group / instance norm.  *(planned -- P3)*

See ``docs/feature-requests/nn-forward-kernels-suite.md``.
"""

from . import attention
from .attention import scaled_dot_product_attention

__all__ = [
    'attention',
    'scaled_dot_product_attention',
]
