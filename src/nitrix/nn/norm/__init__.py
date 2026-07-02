# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
# isort: skip_file
"""
Fused LayerNorm / GroupNorm / InstanceNorm.

Public :func:`layer_norm`, :func:`group_norm` and :func:`instance_norm`, each
with backend selection between a pure-JAX reference and a fused CUDA kernel.
The JAX reference is the bit-faithful oracle and the path that ships today; the
``pallas-cuda`` backend is a fused single-pass fast path, certified numerically
equivalent to the reference within tolerance and promoted only when profiling
shows normalisation bandwidth on a model's critical path.

Each operation carries the curse-of-depth ``out_scale`` hook (see the
:func:`reference_layer_norm` oracle). Until the fused kernel is available, a
``pallas-cuda`` request falls back loudly to the reference, whose XLA fusion is
already correct and adequate.
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
    """Attempt to run a fused Pallas normalisation kernel.

    Imports the CUDA norm kernel module lazily and calls ``{op}_pallas`` with
    the supplied positional and keyword arguments. Returns ``None`` (rather than
    raising) whenever the kernel cannot run on the current host or rejects the
    requested shape, so the caller can fall back to the JAX reference.

    Parameters
    ----------
    op : str
        Name of the normalisation operation, used to select the kernel entry
        point ``{op}_pallas`` (e.g. ``'layer_norm'``, ``'group_norm'``,
        ``'instance_norm'``).
    *args : object
        Positional arguments forwarded verbatim to the selected kernel
        (typically the input array followed by the affine parameters).
    **kwargs : object
        Keyword arguments forwarded verbatim to the selected kernel (e.g.
        ``eps`` and ``out_scale``).

    Returns
    -------
    Array or None
        The kernel output if the fused path ran, otherwise ``None`` -- either
        because the kernel module could not be imported (no CUDA host) or
        because the kernel reported the shape as not tileable.
    """
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
    """LayerNorm over the trailing feature axis, with backend dispatch.

    Standardises ``x`` over its trailing axis to zero mean and unit variance
    (biased, dividing by the axis length), then applies the optional affine
    transform and the depth-scaling factor. The output is
    :math:`\\text{out\\_scale} \\cdot (\\hat{x} \\cdot \\text{weight} +
    \\text{bias})`, where :math:`\\hat{x}` is the standardised input; see the
    :func:`reference_layer_norm` oracle for the exact recipe. The ``out_scale``
    factor is the curse-of-depth hook that folds constant depth-scaling schemes
    in at zero marginal cost.

    When the resolved backend is ``pallas-cuda`` the fused kernel is attempted;
    if it cannot run for the given shape or host, the call falls back loudly to
    the JAX reference.

    Parameters
    ----------
    x : Float[Array, '... c']
        Input array; normalisation runs over the trailing channel axis ``c``.
    weight : Float[Array, 'c'], optional
        Per-channel affine scale of shape ``(c,)``. If omitted, no scaling is
        applied.
    bias : Float[Array, 'c'], optional
        Per-channel affine shift of shape ``(c,)``. If omitted, no shift is
        applied.
    eps : float, default=1e-5
        Constant added to the variance before the reciprocal square root, for
        numerical stability.
    out_scale : float, default=1.0
        Depth-scaling multiplier applied to the full affine output.
    backend : Backend, default='auto'
        Execution backend. ``'auto'`` resolves to ``'pallas-cuda'`` on a
        capable GPU and ``'jax'`` otherwise.

    Returns
    -------
    Float[Array, '... c']
        The normalised, affine-transformed and depth-scaled array, with the
        same shape and dtype as ``x``.
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
    """GroupNorm for channels-first n-D inputs, with backend dispatch.

    Splits the channel axis of a channels-first ``(N, C, *spatial)`` input into
    ``num_groups`` groups and standardises each sample and group over its
    ``(C / num_groups, *spatial)`` elements to zero mean and unit variance
    (biased). It then applies the optional per-channel affine transform and the
    ``out_scale`` depth-scaling factor, matching the
    :func:`reference_group_norm` oracle.

    When the resolved backend is ``pallas-cuda`` the fused kernel is attempted;
    if it cannot run for the given shape or host, the call falls back loudly to
    the JAX reference.

    Parameters
    ----------
    x : Float[Array, 'n c *spatial']
        Channels-first input with a batch axis ``n``, channel axis ``c`` and
        zero or more trailing spatial axes.
    num_groups : int
        Number of channel groups; must divide the channel count ``c`` exactly.
    weight : Float[Array, 'c'], optional
        Per-channel affine scale of shape ``(c,)``. If omitted, no scaling is
        applied.
    bias : Float[Array, 'c'], optional
        Per-channel affine shift of shape ``(c,)``. If omitted, no shift is
        applied.
    eps : float, default=1e-5
        Constant added to the variance before the reciprocal square root, for
        numerical stability.
    out_scale : float, default=1.0
        Depth-scaling multiplier applied to the full affine output.
    backend : Backend, default='auto'
        Execution backend. ``'auto'`` resolves to ``'pallas-cuda'`` on a
        capable GPU and ``'jax'`` otherwise.

    Returns
    -------
    Float[Array, 'n c *spatial']
        The normalised, affine-transformed and depth-scaled array, with the
        same shape and dtype as ``x``.

    Raises
    ------
    ValueError
        If ``x`` has fewer than two dimensions, or if the channel count is not
        divisible by ``num_groups``.
    """
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
    """InstanceNorm for channels-first n-D inputs, with backend dispatch.

    Standardises each sample and channel of a channels-first
    ``(N, C, *spatial)`` input over its spatial elements to zero mean and unit
    variance (biased), then applies the optional per-channel affine transform
    and the ``out_scale`` depth-scaling factor. This is equivalent to
    :func:`group_norm` with one group per channel; see the
    :func:`reference_instance_norm` oracle.

    When the resolved backend is ``pallas-cuda`` the fused kernel is attempted;
    if it cannot run for the given shape or host, the call falls back loudly to
    the JAX reference.

    Parameters
    ----------
    x : Float[Array, 'n c *spatial']
        Channels-first input with a batch axis ``n``, channel axis ``c`` and
        zero or more trailing spatial axes.
    weight : Float[Array, 'c'], optional
        Per-channel affine scale of shape ``(c,)``. If omitted, no scaling is
        applied.
    bias : Float[Array, 'c'], optional
        Per-channel affine shift of shape ``(c,)``. If omitted, no shift is
        applied.
    eps : float, default=1e-5
        Constant added to the variance before the reciprocal square root, for
        numerical stability.
    out_scale : float, default=1.0
        Depth-scaling multiplier applied to the full affine output.
    backend : Backend, default='auto'
        Execution backend. ``'auto'`` resolves to ``'pallas-cuda'`` on a
        capable GPU and ``'jax'`` otherwise.

    Returns
    -------
    Float[Array, 'n c *spatial']
        The normalised, affine-transformed and depth-scaled array, with the
        same shape and dtype as ``x``.

    Raises
    ------
    ValueError
        If ``x`` has fewer than two dimensions.
    """
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
