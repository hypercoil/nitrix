# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Pallas Triton fused normalisations (LayerNorm / GroupNorm / InstanceNorm).

**Gated placeholder.** The norm references (XLA) are correct and adequate today;
the fused single-pass kernels (fork of
``jax.experimental.pallas.ops.gpu.{layer_norm, rms_norm}`` -- mean/var/normalise/
affine in one pass, recompute-in-backward VJP, ``out_scale`` folded into the
affine) are **perf-gated**: promoted only when a profiler shows norm bandwidth on
a model's critical path (suite §7.3).  Unlike attention / selective-scan there is
no activation cliff to remove -- the win is pure bandwidth, which the sibling perf
suite must certify against XLA.  Until then these raise ``PallasNotTileable`` and
the public ops run the JAX reference (loud fallback).

Implementation detail: never import from ``nitrix._kernels.cuda`` directly.  Use
``nitrix.nn.norm.{layer_norm, group_norm, instance_norm}``.
"""

from __future__ import annotations

from typing import Optional

from jaxtyping import Array, Float

__all__ = [
    'layer_norm_pallas',
    'group_norm_pallas',
    'instance_norm_pallas',
    'PallasNotTileable',
]


class PallasNotTileable(RuntimeError):
    """The Pallas kernel rejected the requested shape / host.

    Caught by the dispatcher in ``nitrix.nn.norm`` and translated into a
    ``NitrixBackendFallback`` warning (the JAX reference runs instead).
    """


def layer_norm_pallas(
    x: Float[Array, '... c'],
    weight: Optional[Float[Array, 'c']] = None,
    bias: Optional[Float[Array, 'c']] = None,
    *,
    eps: float = 1e-5,
    out_scale: float = 1.0,
) -> Float[Array, '... c']:
    """Fused LayerNorm (perf-gated); currently always falls back."""
    raise PallasNotTileable(
        'fused norm kernels are perf-gated (suite §7.3); using the JAX '
        'reference.'
    )


def group_norm_pallas(
    x: Float[Array, 'n c *spatial'],
    num_groups: int,
    weight: Optional[Float[Array, 'c']] = None,
    bias: Optional[Float[Array, 'c']] = None,
    *,
    eps: float = 1e-5,
    out_scale: float = 1.0,
) -> Float[Array, 'n c *spatial']:
    """Fused GroupNorm (perf-gated); currently always falls back."""
    raise PallasNotTileable(
        'fused norm kernels are perf-gated (suite §7.3); using the JAX '
        'reference.'
    )


def instance_norm_pallas(
    x: Float[Array, 'n c *spatial'],
    weight: Optional[Float[Array, 'c']] = None,
    bias: Optional[Float[Array, 'c']] = None,
    *,
    eps: float = 1e-5,
    out_scale: float = 1.0,
) -> Float[Array, 'n c *spatial']:
    """Fused InstanceNorm (perf-gated); currently always falls back."""
    raise PallasNotTileable(
        'fused norm kernels are perf-gated (suite §7.3); using the JAX '
        'reference.'
    )
