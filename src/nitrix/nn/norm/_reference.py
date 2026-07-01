# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Pure-JAX references for the fused normalisations.

- :func:`reference_layer_norm` -- normalise over the trailing feature axis
  ``c`` (transformer convention); per-feature affine of shape ``(c,)``.
- :func:`reference_group_norm` -- channels-first n-D input ``(N, C, *spatial)``;
  split ``C`` into ``num_groups`` groups and normalise over
  ``(C/num_groups, *spatial)`` per sample and group; per-channel affine of shape
  ``(C,)``.
- :func:`reference_instance_norm` -- channels-first n-D input; normalise over
  ``*spatial`` per sample and channel (equivalently, group normalisation with
  one group per channel, ``num_groups == C``).

All three use the biased variance estimator (division by :math:`n`) and the
reciprocal-square-root normalisation convention, and all expose the depth-scaling
``out_scale`` hook :math:`\\mathrm{out} = \\mathrm{out\\_scale}\\cdot
(\\hat{x}\\cdot w + b)`. That constant post-scaling folds in the family of
depth-scaling schemes -- LayerNorm Scaling :math:`1/\\sqrt{l}`, residual scaling
:math:`1/\\sqrt{2N}`, DeepNorm-:math:`\\alpha`, and ReZero -- at zero marginal
cost. These references are autodiff-native; the fused-kernel custom VJP lives
alongside them in the CUDA kernel module.
"""

from __future__ import annotations

from typing import Optional

import jax.numpy as jnp
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
    """Layer normalisation over the trailing axis (reference oracle).

    Normalise ``x`` over its trailing feature axis ``c`` using the mean and
    biased variance computed over that axis, apply the optional per-feature
    affine transform, and scale the result by ``out_scale``.

    The reductions (mean and variance) and the affine transform run in at least
    single precision: a ``float16`` / ``bfloat16`` input is upcast to
    ``float32`` and the result cast back to the input dtype, while ``float32``
    and ``float64`` inputs are left unchanged.

    Parameters
    ----------
    x : Float[Array, '... c']
        Input tensor. Normalisation is performed independently over the trailing
        feature axis ``c`` for every entry of the leading axes.
    weight : Float[Array, 'c'], optional
        Per-feature affine scale. If ``None``, no multiplicative scaling is
        applied.
    bias : Float[Array, 'c'], optional
        Per-feature affine shift. If ``None``, no additive shift is applied.
    eps : float, optional
        Small constant added to the variance before taking the reciprocal square
        root, for numerical stability.
    out_scale : float, optional
        Constant post-normalisation scale applied to the affine output,
        implementing the depth-scaling hook :math:`\\mathrm{out} =
        \\mathrm{out\\_scale}\\cdot(\\hat{x}\\cdot w + b)`.

    Returns
    -------
    Float[Array, '... c']
        The normalised, affine-transformed and scaled tensor, with the same
        shape and dtype as ``x``.
    """
    io_dtype = x.dtype
    acc_dtype = jnp.promote_types(io_dtype, jnp.float32)
    upcast = acc_dtype != io_dtype
    if upcast:
        x = x.astype(acc_dtype)
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    x_hat = (x - mean) * lax.rsqrt(var + eps)
    if weight is not None:
        x_hat = x_hat * (weight.astype(acc_dtype) if upcast else weight)
    if bias is not None:
        x_hat = x_hat + (bias.astype(acc_dtype) if upcast else bias)
    out = out_scale * x_hat
    return out.astype(io_dtype) if upcast else out


def reference_group_norm(
    x: Float[Array, 'n c *spatial'],
    num_groups: int,
    weight: Optional[Float[Array, 'c']] = None,
    bias: Optional[Float[Array, 'c']] = None,
    *,
    eps: float = 1e-5,
    out_scale: float = 1.0,
) -> Float[Array, 'n c *spatial']:
    """Group normalisation per sample and group (reference oracle).

    Split the ``C`` channels of a channels-first input into ``num_groups``
    contiguous groups, then normalise over each group's channels together with
    all spatial axes, independently per sample and group, using the mean and
    biased variance. The optional per-channel affine transform is applied
    channel-wise, and the result is scaled by ``out_scale``.

    The reductions (mean and variance) and the affine transform run in at least
    single precision: a ``float16`` / ``bfloat16`` input is upcast to
    ``float32`` and the result cast back to the input dtype, while ``float32``
    and ``float64`` inputs are left unchanged.

    Parameters
    ----------
    x : Float[Array, 'n c *spatial']
        Channels-first input tensor: leading sample axis ``n``, channel axis
        ``c``, then zero or more spatial axes.
    num_groups : int
        Number of groups into which the ``c`` channels are partitioned.
        ``c`` must be divisible by ``num_groups``.
    weight : Float[Array, 'c'], optional
        Per-channel affine scale. If ``None``, no multiplicative scaling is
        applied.
    bias : Float[Array, 'c'], optional
        Per-channel affine shift. If ``None``, no additive shift is applied.
    eps : float, optional
        Small constant added to the variance before taking the reciprocal square
        root, for numerical stability.
    out_scale : float, optional
        Constant post-normalisation scale applied to the affine output,
        implementing the depth-scaling hook :math:`\\mathrm{out} =
        \\mathrm{out\\_scale}\\cdot(\\hat{x}\\cdot w + b)`.

    Returns
    -------
    Float[Array, 'n c *spatial']
        The normalised, affine-transformed and scaled tensor, with the same
        shape and dtype as ``x``.
    """
    io_dtype = x.dtype
    acc_dtype = jnp.promote_types(io_dtype, jnp.float32)
    upcast = acc_dtype != io_dtype
    if upcast:
        x = x.astype(acc_dtype)
    n, c = x.shape[0], x.shape[1]
    spatial = x.shape[2:]
    grouped = x.reshape(n, num_groups, c // num_groups, *spatial)
    axes = tuple(range(2, grouped.ndim))
    mean = grouped.mean(axis=axes, keepdims=True)
    var = grouped.var(axis=axes, keepdims=True)
    x_hat = ((grouped - mean) * lax.rsqrt(var + eps)).reshape(x.shape)
    affine_shape = (1, c) + (1,) * len(spatial)
    if weight is not None:
        w = weight.astype(acc_dtype) if upcast else weight
        x_hat = x_hat * w.reshape(affine_shape)
    if bias is not None:
        b = bias.astype(acc_dtype) if upcast else bias
        x_hat = x_hat + b.reshape(affine_shape)
    out = out_scale * x_hat
    return out.astype(io_dtype) if upcast else out


def reference_instance_norm(
    x: Float[Array, 'n c *spatial'],
    weight: Optional[Float[Array, 'c']] = None,
    bias: Optional[Float[Array, 'c']] = None,
    *,
    eps: float = 1e-5,
    out_scale: float = 1.0,
) -> Float[Array, 'n c *spatial']:
    """Instance normalisation per sample and channel (reference oracle).

    Normalise each channel of a channels-first input independently over its
    spatial axes, per sample. This is exactly group normalisation with one group
    per channel, so the call delegates to :func:`reference_group_norm` with
    ``num_groups`` equal to the number of channels.

    Parameters
    ----------
    x : Float[Array, 'n c *spatial']
        Channels-first input tensor: leading sample axis ``n``, channel axis
        ``c``, then zero or more spatial axes.
    weight : Float[Array, 'c'], optional
        Per-channel affine scale. If ``None``, no multiplicative scaling is
        applied.
    bias : Float[Array, 'c'], optional
        Per-channel affine shift. If ``None``, no additive shift is applied.
    eps : float, optional
        Small constant added to the variance before taking the reciprocal square
        root, for numerical stability.
    out_scale : float, optional
        Constant post-normalisation scale applied to the affine output,
        implementing the depth-scaling hook :math:`\\mathrm{out} =
        \\mathrm{out\\_scale}\\cdot(\\hat{x}\\cdot w + b)`.

    Returns
    -------
    Float[Array, 'n c *spatial']
        The normalised, affine-transformed and scaled tensor, with the same
        shape and dtype as ``x``.
    """
    return reference_group_norm(
        x, x.shape[1], weight, bias, eps=eps, out_scale=out_scale
    )
