# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
# isort: skip_file
"""
nitrix.nn.norm -- fused LayerNorm / GroupNorm / InstanceNorm.

Public ``layer_norm`` / ``group_norm`` / ``instance_norm`` with three-level
backend selection.  The ``jax`` reference is the bit-faithful oracle (and the
path that ships today); ``pallas-cuda`` is the fused single-pass fast path,
certified ``pallas-cuda ≈ jax`` only inside nitrix and **gated** -- promoted
only when a profiler shows norm bandwidth on a model's critical path.  Each
op carries the curse-of-depth ``out_scale`` hook (see ``_reference``).  Until
the fused kernel lands, a ``pallas-cuda`` request falls back loudly to the
reference (whose XLA fusion is already correct + adequate).
"""

from __future__ import annotations

from typing import Optional

from jaxtyping import Array, Float

from ..._internal.backend import Backend, fallback, resolve_backend
from ._reference import (
    reference_group_norm,
    reference_instance_norm,
    reference_layer_norm,
)

__all__ = [
    'layer_norm',
    'group_norm',
    'instance_norm',
    'reference_layer_norm',
    'reference_group_norm',
    'reference_instance_norm',
]


def _check_affine(
    name: str,
    weight: Optional[Array],
    bias: Optional[Array],
    channels: int,
) -> None:
    for label, p in (('weight', weight), ('bias', bias)):
        if p is not None and p.shape != (channels,):
            raise ValueError(
                f'{name}: {label} must have shape ({channels},); got {p.shape}.'
            )


def _norm_pallas(op: str, *args: object, **kwargs: object) -> Optional[Array]:
    """Pallas dispatch; ``None`` if the kernel rejects the shape / host."""
    try:
        from ..._kernels.cuda import norm as _k
    except Exception:
        return None
    try:
        fn = getattr(_k, f'{op}_pallas')
        return fn(*args, **kwargs)  # type: ignore[no-any-return]
    except _k.PallasNotTileable:
        return None


def _fallback(op: str, x: Array) -> None:
    fallback(
        function=op,
        requested='pallas-cuda',
        resolved='jax',
        reason=(
            'no fused norm kernel available for this shape/host '
            '(the fused path is perf-gated; suite §7.3)'
        ),
        shapes=(tuple(x.shape),),
        dtype=x.dtype,
    )


def layer_norm(
    x: Float[Array, '... c'],
    weight: Optional[Float[Array, 'c']] = None,
    bias: Optional[Float[Array, 'c']] = None,
    *,
    eps: float = 1e-5,
    out_scale: float = 1.0,
    backend: Backend = 'auto',
) -> Float[Array, '... c']:
    """LayerNorm over the trailing axis, with backend dispatch.

    ``out`` is ``out_scale·(x̂·weight + bias)`` (see
    :func:`reference_layer_norm`); ``out_scale`` is the curse-of-depth hook.
    """
    _check_affine('layer_norm', weight, bias, x.shape[-1])
    if resolve_backend(backend) == 'pallas-cuda':
        out = _norm_pallas(
            'layer_norm', x, weight, bias, eps=eps, out_scale=out_scale
        )
        if out is not None:
            return out
        _fallback('layer_norm', x)
    return reference_layer_norm(x, weight, bias, eps=eps, out_scale=out_scale)


def group_norm(
    x: Float[Array, 'n c *spatial'],
    num_groups: int,
    weight: Optional[Float[Array, 'c']] = None,
    bias: Optional[Float[Array, 'c']] = None,
    *,
    eps: float = 1e-5,
    out_scale: float = 1.0,
    backend: Backend = 'auto',
) -> Float[Array, 'n c *spatial']:
    """GroupNorm (channels-first n-D), with backend dispatch."""
    if x.ndim < 2:
        raise ValueError(
            f'group_norm: x must be (N, C, *spatial); got ndim {x.ndim}.'
        )
    c = x.shape[1]
    if c % num_groups != 0:
        raise ValueError(
            f'group_norm: C={c} not divisible by num_groups={num_groups}.'
        )
    _check_affine('group_norm', weight, bias, c)
    if resolve_backend(backend) == 'pallas-cuda':
        out = _norm_pallas(
            'group_norm',
            x,
            num_groups,
            weight,
            bias,
            eps=eps,
            out_scale=out_scale,
        )
        if out is not None:
            return out
        _fallback('group_norm', x)
    return reference_group_norm(
        x, num_groups, weight, bias, eps=eps, out_scale=out_scale
    )


def instance_norm(
    x: Float[Array, 'n c *spatial'],
    weight: Optional[Float[Array, 'c']] = None,
    bias: Optional[Float[Array, 'c']] = None,
    *,
    eps: float = 1e-5,
    out_scale: float = 1.0,
    backend: Backend = 'auto',
) -> Float[Array, 'n c *spatial']:
    """InstanceNorm (channels-first n-D), with backend dispatch."""
    if x.ndim < 2:
        raise ValueError(
            f'instance_norm: x must be (N, C, *spatial); got ndim {x.ndim}.'
        )
    _check_affine('instance_norm', weight, bias, x.shape[1])
    if resolve_backend(backend) == 'pallas-cuda':
        out = _norm_pallas(
            'instance_norm', x, weight, bias, eps=eps, out_scale=out_scale
        )
        if out is not None:
            return out
        _fallback('instance_norm', x)
    return reference_instance_norm(
        x, weight, bias, eps=eps, out_scale=out_scale
    )
