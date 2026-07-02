# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Pallas Triton fused normalisations (LayerNorm / GroupNorm / InstanceNorm).

Gated placeholder for single-pass fused normalisation kernels. The pure-JAX
normalisation references are correct and adequate today; the fused single-pass
kernels (a fork of ``jax.experimental.pallas.ops.gpu.layer_norm`` and
``rms_norm`` that compute the mean, variance, normalisation and affine transform
in one pass, recompute the statistics in the backward pass, and fold
``out_scale`` into the affine transform) are perf-gated. They are to be promoted
only once profiling shows normalisation bandwidth on a model's critical path.

Unlike attention or selective-scan there is no activation cliff to remove here --
the win is pure memory bandwidth, which the sibling perf-benchmark suite must
certify against XLA. Until then the kernels raise
:class:`PallasNotTileable` and the public ops run the JAX reference by way of a
loud fallback.

Notes
-----
This module is an implementation detail. Never import from
``nitrix._kernels.cuda`` directly; use ``nitrix.nn.norm.layer_norm``,
``nitrix.nn.norm.group_norm`` and ``nitrix.nn.norm.instance_norm`` instead.
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
    """Raised when the Pallas kernel rejects the requested shape or host.

    Caught by the dispatcher in ``nitrix.nn.norm`` and translated into a
    :class:`NitrixBackendFallback` warning, after which the JAX reference runs
    instead.
    """


def layer_norm_pallas(
    x: Float[Array, '... c'],
    weight: Optional[Float[Array, 'c']] = None,
    bias: Optional[Float[Array, 'c']] = None,
    *,
    eps: float = 1e-5,
    out_scale: float = 1.0,
) -> Float[Array, '... c']:
    """Fused single-pass layer normalisation over the last axis.

    Perf-gated placeholder: this kernel is not yet promoted, so every call
    raises :class:`PallasNotTileable` and the dispatcher falls back to the JAX
    reference. When enabled it would normalise ``x`` over its final ``c`` axis,
    apply the optional affine transform, and fold ``out_scale`` into the affine.

    Parameters
    ----------
    x : Float[Array, '... c']
        Input tensor; normalisation is computed over the trailing ``c`` axis,
        with any leading axes treated as a batch.
    weight : Float[Array, 'c'], optional
        Per-channel affine scale applied after normalisation. If ``None``, no
        scaling is applied.
    bias : Float[Array, 'c'], optional
        Per-channel affine shift applied after normalisation. If ``None``, no
        shift is applied.
    eps : float, optional
        Small constant added to the variance for numerical stability. Defaults
        to ``1e-5``.
    out_scale : float, optional
        Scalar factor folded into the affine transform, scaling the output.
        Defaults to ``1.0``.

    Returns
    -------
    Float[Array, '... c']
        The normalised tensor with the same shape as ``x``.

    Raises
    ------
    PallasNotTileable
        Always, while the fused kernel is perf-gated.
    """
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
    """Fused single-pass group normalisation over channel groups.

    Perf-gated placeholder: this kernel is not yet promoted, so every call
    raises :class:`PallasNotTileable` and the dispatcher falls back to the JAX
    reference. When enabled it would partition the ``c`` channels into
    ``num_groups`` groups, normalise over each group and its spatial extent,
    apply the optional affine transform, and fold ``out_scale`` into the affine.

    Parameters
    ----------
    x : Float[Array, 'n c *spatial']
        Input tensor with a batch axis ``n``, a channel axis ``c``, and any
        number of trailing spatial axes.
    num_groups : int
        Number of groups the ``c`` channels are partitioned into; normalisation
        statistics are computed within each group.
    weight : Float[Array, 'c'], optional
        Per-channel affine scale applied after normalisation. If ``None``, no
        scaling is applied.
    bias : Float[Array, 'c'], optional
        Per-channel affine shift applied after normalisation. If ``None``, no
        shift is applied.
    eps : float, optional
        Small constant added to the variance for numerical stability. Defaults
        to ``1e-5``.
    out_scale : float, optional
        Scalar factor folded into the affine transform, scaling the output.
        Defaults to ``1.0``.

    Returns
    -------
    Float[Array, 'n c *spatial']
        The normalised tensor with the same shape as ``x``.

    Raises
    ------
    PallasNotTileable
        Always, while the fused kernel is perf-gated.
    """
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
    """Fused single-pass instance normalisation over spatial axes.

    Perf-gated placeholder: this kernel is not yet promoted, so every call
    raises :class:`PallasNotTileable` and the dispatcher falls back to the JAX
    reference. When enabled it would normalise each channel of each batch
    element independently over its spatial extent, apply the optional affine
    transform, and fold ``out_scale`` into the affine.

    Parameters
    ----------
    x : Float[Array, 'n c *spatial']
        Input tensor with a batch axis ``n``, a channel axis ``c``, and any
        number of trailing spatial axes; statistics are computed per
        batch-channel pair over the spatial axes.
    weight : Float[Array, 'c'], optional
        Per-channel affine scale applied after normalisation. If ``None``, no
        scaling is applied.
    bias : Float[Array, 'c'], optional
        Per-channel affine shift applied after normalisation. If ``None``, no
        shift is applied.
    eps : float, optional
        Small constant added to the variance for numerical stability. Defaults
        to ``1e-5``.
    out_scale : float, optional
        Scalar factor folded into the affine transform, scaling the output.
        Defaults to ``1.0``.

    Returns
    -------
    Float[Array, 'n c *spatial']
        The normalised tensor with the same shape as ``x``.

    Raises
    ------
    PallasNotTileable
        Always, while the fused kernel is perf-gated.
    """
    raise PallasNotTileable(
        'fused norm kernels are perf-gated (suite §7.3); using the JAX '
        'reference.'
    )
