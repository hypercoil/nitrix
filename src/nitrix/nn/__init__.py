# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
# isort: skip_file
"""
Functional neural-network forward-block kernels.

Pure ``(Array, ...) -> Array`` contractions for the inner kernels of
transformer and state-space model forward blocks, behind the standard
``backend=`` dispatch (a ``pallas-cuda`` fused fast path with a ``jax``
reference fallback). These are *functions*, not stateful modules: the
learnable modules that hold the parameters live in downstream libraries;
this subpackage owns only the inner kernel.

Submodules
----------
attention
    :func:`scaled_dot_product_attention` (dense / windowed additive-bias /
    causal / cross), with a flash-attention fast path.
ssm
    :func:`selective_scan` (the selective state-space scan of Mamba / S6),
    with a parallel associative-scan GPU path and a fused chunked-scan
    kernel.
norm
    :func:`layer_norm`, :func:`group_norm`, and :func:`instance_norm`, each
    exposing the curse-of-depth ``out_scale`` hook; the fused single-pass
    kernel is performance-gated.
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
