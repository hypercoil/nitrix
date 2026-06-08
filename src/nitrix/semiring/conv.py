# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Public ``semiring_conv`` -- the third member of the SPEC §3.1 trinity.

For each output spatial position, the kernel is multiplied with the
input receptive field via the algebra's ``binary_op`` and then
reduced via the algebra's ``monoid``::

    C[..., *spatial_out, c_out]
        = (+)_{*ks, c_in} (
            x[..., *spatial_at(spatial_out, ks), c_in]
              (*)
            k[*ks, c_in, c_out]
          )

Forward implementation strategy
-------------------------------

We reduce the conv to a matmul over patches:

1. Extract patches via ``lax.conv_general_dilated_patches`` with the
   requested window strides / padding / dilation.  The result is
   ``(batch, c_in * prod(kspatial), *spatial_out)`` with the channel
   axis ordered ``c_in_outer, kspatial_inner``.
2. Reshape patches to ``(batch * prod(spatial_out), c_in * prod(kspatial))``.
3. Reshape ``k`` to ``(c_in * prod(kspatial), c_out)`` with the same
   axis ordering.
4. Call ``semiring_matmul`` -- the existing custom_vjp on matmul
   gives us the backward "for free" via the JAX autograd pipeline.

The patches tensor is materialised (this is the price for re-using
``semiring_matmul``).  For typical conv shapes (~spatial × c_in ×
kspatial ≲ 64 MB) this is fine; a streaming-Pallas conv kernel that
avoids materialisation is a Phase 2.A.6 follow-up.

Backward
--------

Composed via JAX autograd from:

- ``lax.conv_general_dilated_patches`` (XLA-registered VJP)
- The ``reshape`` / ``moveaxis`` glue (trivial VJPs)
- ``semiring_matmul`` (our per-algebra custom_vjp)

We do *not* need a separate ``conv_vjp`` field on ``Semiring``: the
chain rule through the matmul VJP already handles all five
differentiable algebras correctly.

Layout
------

Channel-last (``NHWC``-style) for the public surface: ``x`` has shape
``(..., *spatial, c_in)``; ``k`` has shape ``(*kspatial, c_in,
c_out)``.  This matches Keras / TF convention and is the most natural
for users who think of channels as features.  Internally we transpose
to JAX-default ``NCHW`` for the lax patches call and convert back.

Backend selection
-----------------

Three-level resolution as everywhere else.  The Pallas-CUDA path is
a stub that raises ``PallasConvNotTileable``; the dispatcher catches
this and falls back to JAX with one ``NitrixBackendFallback``
warning per ``(shape, dtype, algebra)`` signature.  A native Pallas
conv kernel is a future-work item.
"""

from __future__ import annotations

from functools import reduce
from operator import mul
from typing import Any, Optional, Sequence, Union, cast

import jax.numpy as jnp
from jaxtyping import Array, Num

from .._internal.backend import (
    Backend,
    ResolvedBackend,
    fallback,
    resolve_backend,
)
from ._types import Semiring
from .algebras import REAL
from .matmul import semiring_matmul

__all__ = [
    'semiring_conv',
    'reference_semiring_conv',
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prod(xs: Sequence[int]) -> int:
    return reduce(mul, xs, 1)


def _normalise_n_tuple(
    x: Union[int, Sequence[int]], n: int, name: str
) -> tuple[int, ...]:
    if isinstance(x, int):
        return (x,) * n
    out = tuple(x)
    if len(out) != n:
        raise ValueError(
            f'{name} must be an int or a length-{n} sequence; got {x!r}.'
        )
    return out


def _normalise_padding(
    padding: Union[str, Sequence[tuple[int, int]]], spatial_rank: int
) -> Union[str, tuple[tuple[int, int], ...]]:
    """Accept ``'SAME'``, ``'VALID'``, or an explicit per-dim sequence."""
    if isinstance(padding, str):
        if padding.upper() not in ('SAME', 'VALID'):
            raise ValueError(
                f'padding={padding!r} must be "SAME", "VALID", or a '
                'per-dim sequence of (lo, hi) pairs.'
            )
        return padding.upper()
    out = tuple(padding)
    if len(out) != spatial_rank:
        raise ValueError(
            f'padding must have {spatial_rank} (lo, hi) pairs; '
            f'got {padding!r}.'
        )
    return tuple((int(lo), int(hi)) for lo, hi in out)


# ---------------------------------------------------------------------------
# Reference implementation
# ---------------------------------------------------------------------------


def _resolve_pad(
    pad: Union[str, tuple[tuple[int, int], ...]],
    spatial_in: tuple[int, ...],
    kspatial: tuple[int, ...],
    strides: tuple[int, ...],
    dilations: tuple[int, ...],
) -> tuple[tuple[int, int], ...]:
    """Resolve string padding to explicit ``(lo, hi)`` per spatial dim.

    Mirrors the ``"SAME"`` / ``"VALID"`` semantics of
    ``lax.conv_general_dilated`` exactly (per-dim symmetric or
    near-symmetric padding for SAME; zero for VALID).
    """
    n = len(spatial_in)
    if pad == 'VALID':
        return tuple((0, 0) for _ in range(n))
    if pad == 'SAME':
        out = []
        for d in range(n):
            eff_k = (kspatial[d] - 1) * dilations[d] + 1
            out_size = -(-spatial_in[d] // strides[d])  # ceil div
            total_pad = max(
                (out_size - 1) * strides[d] + eff_k - spatial_in[d],
                0,
            )
            lo = total_pad // 2
            hi = total_pad - lo
            out.append((lo, hi))
        return tuple(out)
    # Already an explicit per-dim tuple (the str cases returned above).
    return cast(tuple[tuple[int, int], ...], pad)


def _extract_patches_nan_safe(
    x_nhwc: Num[Array, '...'],
    *,
    kspatial: tuple[int, ...],
    strides: tuple[int, ...],
    dilations: tuple[int, ...],
    padding_explicit: Sequence[tuple[int, int]],
    identity: Any,
) -> Num[Array, '...']:
    """NaN-safe patch extraction.

    ``lax.conv_general_dilated_patches`` lowers via a multiply-with-
    one-hot trick that produces ``NaN`` when ``x`` contains ``-inf``
    (because ``0 * -inf == NaN``).  For the tropical / log semirings
    we *need* ``-inf`` to flow through cleanly, and we *need* the
    padding to be the algebra's identity (so padded positions are
    no-ops under the reduction).

    Strategy: pad ``x`` with ``identity`` on the spatial dims, then
    build ``(out_d, k_d)`` index matrices and gather along each dim
    via ``jnp.take``.  No multiplication, no NaN injection.

    Parameters
    ----------
    x_nhwc
        Input, shape ``(batch, *spatial, c_in)``.
    kspatial, strides, dilations, padding_explicit
        Standard conv hyperparameters; ``padding_explicit`` is a
        sequence of ``(lo, hi)`` pairs.
    identity
        Scalar (or ``None``) padding value; defaults to ``0``.

    Returns
    -------
    Patches array of shape
    ``(batch, *spatial_out, *kspatial, c_in)``.
    """
    spatial_rank = len(kspatial)
    if identity is None:
        identity = 0
    # Pad the spatial dims; leave batch and channel untouched.
    pad_widths = (
        [(0, 0)]
        + [tuple(padding_explicit[d]) for d in range(spatial_rank)]
        + [(0, 0)]
    )
    x_padded = jnp.pad(
        x_nhwc,
        pad_widths,
        mode='constant',
        constant_values=identity,
    )

    spatial_padded = tuple(x_padded.shape[1 + d] for d in range(spatial_rank))
    spatial_out = []
    for d in range(spatial_rank):
        eff_k = (kspatial[d] - 1) * dilations[d] + 1
        out_d = (spatial_padded[d] - eff_k) // strides[d] + 1
        spatial_out.append(max(out_d, 0))

    # Gather along each spatial dim.  Walk the dims in order; each
    # ``jnp.take`` inserts a new axis after the gathered dim.
    out = x_padded
    for d in range(spatial_rank):
        # Source axis (after the d previous insertions) is 1 + 2 * d.
        ax = 1 + 2 * d
        idx = (
            jnp.arange(spatial_out[d])[:, None] * strides[d]
            + jnp.arange(kspatial[d])[None, :] * dilations[d]
        )  # (out_d, k_d)
        out = jnp.take(out, idx, axis=ax)
        # New axes layout after take at axis ``ax``: the source axis
        # of size ``spatial_padded[d]`` is replaced by two axes of
        # sizes ``(out_d, k_d)``, in that order.
    # ``out`` now has shape:
    #   (batch, out_0, k_0, out_1, k_1, ..., c_in)
    # We want (batch, *spatial_out, *kspatial, c_in).  Permute the
    # interleaved ``(out_d, k_d)`` pairs into two contiguous groups.
    perm = [0]  # batch
    for d in range(spatial_rank):
        perm.append(1 + 2 * d)  # spatial_out[d]
    for d in range(spatial_rank):
        perm.append(2 + 2 * d)  # kspatial[d]
    perm.append(out.ndim - 1)  # c_in
    out = jnp.transpose(out, perm)
    return out


def reference_semiring_conv(
    x: Num[Array, '... *spatial c_in'],
    k: Num[Array, '*kspatial c_in c_out'],
    *,
    semiring: Semiring[Any],
    stride: Union[int, Sequence[int]] = 1,
    padding: Union[str, Sequence[tuple[int, int]]] = 'SAME',
    dilation: Union[int, Sequence[int]] = 1,
    backend: Backend = 'jax',
) -> Num[Array, '... *spatial_out c_out']:
    """Pure-JAX reference for ``semiring_conv``.

    NaN-safe patches extraction (pads with the algebra's identity,
    gathers via ``jnp.take``) followed by ``semiring_matmul`` for
    the per-output-position reduction.  See module docstring.
    """
    spatial_rank = k.ndim - 2
    if spatial_rank < 1:
        raise ValueError(
            'semiring_conv: kernel must have at least one spatial dim '
            f'plus c_in, c_out; got k.shape={k.shape}.'
        )
    if x.ndim < spatial_rank + 1:
        raise ValueError(
            f'semiring_conv: x.ndim={x.ndim} too small for spatial_rank='
            f'{spatial_rank} (needs at least {spatial_rank + 1}).'
        )

    batch_shape = tuple(x.shape[: -spatial_rank - 1])
    spatial_in = tuple(x.shape[-spatial_rank - 1 : -1])
    c_in = int(x.shape[-1])
    c_out = int(k.shape[-1])
    kspatial = tuple(int(d) for d in k.shape[:-2])
    if int(k.shape[-2]) != c_in:
        raise ValueError(
            f'semiring_conv: k.shape[-2]={int(k.shape[-2])} must match '
            f'x.shape[-1]={c_in}.'
        )

    strides = _normalise_n_tuple(stride, spatial_rank, 'stride')
    dilations = _normalise_n_tuple(dilation, spatial_rank, 'dilation')
    pad_str_or_explicit = _normalise_padding(padding, spatial_rank)
    pad_explicit = _resolve_pad(
        pad_str_or_explicit,
        spatial_in,
        kspatial,
        strides,
        dilations,
    )

    # Flatten leading batch dims into one.
    bsz = _prod(batch_shape) if batch_shape else 1
    x_flat = x.reshape((bsz,) + spatial_in + (c_in,))

    # Patches: (batch, *spatial_out, *kspatial, c_in).
    patches = _extract_patches_nan_safe(
        x_flat,
        kspatial=kspatial,
        strides=strides,
        dilations=dilations,
        padding_explicit=pad_explicit,
        identity=semiring.identity,
    )
    spatial_out = tuple(patches.shape[1 : 1 + spatial_rank])
    spatial_out_prod = _prod(spatial_out) if spatial_out else 1

    # Reshape patches to (batch * prod(out), prod(kspatial) * c_in).
    # The axis ordering after extraction matches the kernel's
    # natural (*kspatial, c_in) layout, so no extra moveaxis needed.
    patches_2d = patches.reshape(
        bsz * spatial_out_prod,
        _prod(kspatial) * c_in,
    )

    # Kernel: (*kspatial, c_in, c_out) -> (prod(kspatial) * c_in, c_out)
    k_2d = k.reshape(_prod(kspatial) * c_in, c_out)

    # Inner reduction via the existing matmul surface (custom_vjp
    # registered there gives us the backward composition).
    out_2d = semiring_matmul(
        patches_2d,
        k_2d,
        semiring=semiring,
        backend=backend,
    )

    return out_2d.reshape(*batch_shape, *spatial_out, c_out)


# ---------------------------------------------------------------------------
# Public semiring_conv with backend dispatch
# ---------------------------------------------------------------------------


def _semiring_conv_pallas(
    x: Num[Array, '... *spatial c_in'],
    k: Num[Array, '*kspatial c_in c_out'],
    *,
    semiring: Semiring[Any],
    stride: Union[int, Sequence[int]],
    padding: Union[str, Sequence[tuple[int, int]]],
    dilation: Union[int, Sequence[int]],
) -> Optional[Array]:
    """Pallas dispatch; returns ``None`` if the kernel rejects the request.

    Currently always rejects: there is no dedicated Pallas conv kernel
    yet (would require a separate ``_kernels/cuda/semiring_conv.py``
    that streams over kspatial × c_in instead of materialising patches).
    The dispatcher catches the ``None`` and falls back to JAX with a
    ``NitrixBackendFallback`` warning.
    """
    return None


def semiring_conv(
    x: Num[Array, '... *spatial c_in'],
    k: Num[Array, '*kspatial c_in c_out'],
    *,
    semiring: Semiring[Any] = REAL,
    stride: Union[int, Sequence[int]] = 1,
    padding: Union[str, Sequence[tuple[int, int]]] = 'SAME',
    dilation: Union[int, Sequence[int]] = 1,
    backend: Backend = 'auto',
) -> Num[Array, '... *spatial_out c_out']:
    """Semiring-generalised convolution.

    Channel-last layout: ``x: (..., *spatial, c_in)``,
    ``k: (*kspatial, c_in, c_out)``.  Leading batch dims are
    arbitrary; the convolution slides ``k`` over ``x`` along the
    spatial dims and reduces via the supplied ``semiring``.

    Parameters
    ----------
    x
        Input array, ``(..., *spatial, c_in)``.
    k
        Kernel, ``(*kspatial, c_in, c_out)``.  The number of trailing
        spatial dims of ``x`` is inferred from ``k.ndim - 2``.
    semiring
        Algebra to reduce under.  Defaults to ``REAL`` (i.e., standard
        linear convolution).
    stride
        Per-spatial-dim stride.  Either an ``int`` (applied to every
        spatial dim) or a per-dim sequence.
    padding
        ``"SAME"``, ``"VALID"``, or an explicit per-spatial-dim
        sequence of ``(lo, hi)`` pairs.
    dilation
        Per-spatial-dim kernel dilation (rhs_dilation in lax terms).
    backend
        ``"auto"``, ``"pallas-cuda"``, or ``"jax"``.  At first GA
        there is no native Pallas conv kernel; ``"pallas-cuda"`` is
        accepted (so callers don't have to special-case conv) but
        always falls back to the JAX patches path with a warning.

    Returns
    -------
    Output array of shape ``(..., *spatial_out, c_out)``.

    Notes
    -----
    The forward path reuses ``semiring_matmul`` over reshaped
    patches, so:

    - ``backend`` is honoured for the inner reduction: ``pallas-cuda``
      uses the native Pallas matmul kernel for the per-output-position
      reduction.
    - The backward composes through ``semiring_matmul``'s registered
      VJP automatically; no per-algebra ``conv_vjp`` is required.

    For ``REAL``, this produces the same answer as
    ``lax.conv_general_dilated`` modulo TF32 vs strict-fp32 precision
    (see ``semiring_matmul``'s docstring).
    """
    # Try the Pallas-specific kernel first (currently a stub).
    out = _semiring_conv_pallas(
        x,
        k,
        semiring=semiring,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )
    if out is not None:
        return out

    resolved: ResolvedBackend = resolve_backend(backend)
    if resolved == 'pallas-cuda':
        # No dedicated Pallas conv kernel yet, but the inner matmul
        # has one.  We warn once that the *conv-level* kernel is
        # missing, then route the inner reduction through Pallas
        # where it can tile.  If the inner matmul also can't tile
        # the requested shape, it emits its own separate warning at
        # the matmul layer -- that's a legitimate second observability
        # event (different layer, different reason) and the
        # deduplication-per-signature still holds at each layer.
        fallback(
            function='semiring_conv',
            requested='pallas-cuda',
            resolved='jax',
            reason=(
                f'algebra={semiring.name!r}: no dedicated Pallas conv '
                'kernel; patches extraction runs on JAX while the '
                'inner matmul keeps the Pallas path when shapes tile'
            ),
            shapes=(tuple(x.shape), tuple(k.shape)),
            dtype=x.dtype,
        )
        inner_backend: Backend = 'pallas-cuda'
    else:
        inner_backend = 'jax'

    return reference_semiring_conv(
        x,
        k,
        semiring=semiring,
        stride=stride,
        padding=padding,
        dilation=dilation,
        backend=inner_backend,
    )
