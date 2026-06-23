# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Pure-JAX references for the fused normalisations.

- ``layer_norm`` -- normalise over the trailing feature axis ``c`` (transformer
  convention); affine ``(c,)``.
- ``group_norm`` -- channels-first n-D ``(N, C, *spatial)``; split ``C`` into
  ``num_groups`` groups and normalise over ``(C/num_groups, *spatial)`` per
  sample/group; affine per-channel ``(C,)``.
- ``instance_norm`` -- channels-first n-D; normalise over ``*spatial`` per
  sample/channel (== ``group_norm`` with ``num_groups == C``).

All use biased variance (``/n``), the ``equinox`` / ``rsqrt`` convention, and
the **curse-of-depth `out_scale`** hook: ``out = out_scale·(x̂·w + b)``, which
folds the constant depth-scaling family (LayerNorm Scaling ``1/√l``, residual
``1/√(2N)``, DeepNorm-α, ReZero) in at zero marginal cost.  Autodiff-native --
the fused-kernel custom VJP lives in ``_kernels/cuda/norm.py``.  See
``docs/feature-requests/nn-forward-kernels-suite.md`` §7.3.
"""

from __future__ import annotations

from typing import Optional

from jax import lax
from jaxtyping import Array, Float

__all__ = [
    'reference_layer_norm',
    'reference_group_norm',
    'reference_instance_norm',
]


def reference_layer_norm(
    x: Float[Array, '... c'],
    weight: Optional[Float[Array, 'c']] = None,
    bias: Optional[Float[Array, 'c']] = None,
    *,
    eps: float = 1e-5,
    out_scale: float = 1.0,
) -> Float[Array, '... c']:
    """LayerNorm over the trailing axis (oracle)."""
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    x_hat = (x - mean) * lax.rsqrt(var + eps)
    if weight is not None:
        x_hat = x_hat * weight
    if bias is not None:
        x_hat = x_hat + bias
    return out_scale * x_hat


def reference_group_norm(
    x: Float[Array, 'n c *spatial'],
    num_groups: int,
    weight: Optional[Float[Array, 'c']] = None,
    bias: Optional[Float[Array, 'c']] = None,
    *,
    eps: float = 1e-5,
    out_scale: float = 1.0,
) -> Float[Array, 'n c *spatial']:
    """GroupNorm over ``(C/num_groups, *spatial)`` per sample/group (oracle)."""
    n, c = x.shape[0], x.shape[1]
    spatial = x.shape[2:]
    grouped = x.reshape(n, num_groups, c // num_groups, *spatial)
    axes = tuple(range(2, grouped.ndim))
    mean = grouped.mean(axis=axes, keepdims=True)
    var = grouped.var(axis=axes, keepdims=True)
    x_hat = ((grouped - mean) * lax.rsqrt(var + eps)).reshape(x.shape)
    affine_shape = (1, c) + (1,) * len(spatial)
    if weight is not None:
        x_hat = x_hat * weight.reshape(affine_shape)
    if bias is not None:
        x_hat = x_hat + bias.reshape(affine_shape)
    return out_scale * x_hat


def reference_instance_norm(
    x: Float[Array, 'n c *spatial'],
    weight: Optional[Float[Array, 'c']] = None,
    bias: Optional[Float[Array, 'c']] = None,
    *,
    eps: float = 1e-5,
    out_scale: float = 1.0,
) -> Float[Array, 'n c *spatial']:
    """InstanceNorm == GroupNorm with one group per channel (oracle)."""
    return reference_group_norm(
        x, x.shape[1], weight, bias, eps=eps, out_scale=out_scale
    )
