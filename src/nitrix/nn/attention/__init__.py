# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
# isort: skip_file
"""
nitrix.nn.attention -- scaled-dot-product / flash attention.

Public ``scaled_dot_product_attention`` with three-level backend
selection (SPEC §7.2).  The two-tier parity contract (suite plan §4): the
``jax`` reference is the bit-faithful oracle; ``pallas-cuda`` is the fused
flash path (suite Phase 2), certified ``pallas-cuda ≈ jax`` only inside
nitrix.  ilex pins its own forward-parity to ``backend='jax'``.

Until the fused kernel lands, an explicit / auto-resolved ``pallas-cuda``
request falls back to the reference with a loud ``NitrixBackendFallback``.
"""

from __future__ import annotations

from typing import Optional

from jaxtyping import Array, Bool, Float

from ..._internal.backend import Backend, fallback, resolve_backend
from ._reference import (
    default_scale,
    qk_rms_norm,
    reference_scaled_dot_product_attention,
)

__all__ = [
    'scaled_dot_product_attention',
    'reference_scaled_dot_product_attention',
]


def _validate(
    q: Float[Array, '... h s d'],
    k: Float[Array, '... h t d'],
    v: Float[Array, '... h t d_v'],
) -> None:
    if q.ndim < 3 or k.ndim < 3 or v.ndim < 3:
        raise ValueError(
            'scaled_dot_product_attention: q, k, v must have at least 3 '
            'dims (..., h, s, d); got ndim '
            f'{q.ndim}, {k.ndim}, {v.ndim}.'
        )
    if q.shape[-1] != k.shape[-1]:
        raise ValueError(
            'scaled_dot_product_attention: q/k head-dim mismatch '
            f'(q d={q.shape[-1]}, k d={k.shape[-1]}).'
        )
    if k.shape[-2] != v.shape[-2]:
        raise ValueError(
            'scaled_dot_product_attention: k/v key-length mismatch '
            f'(k t={k.shape[-2]}, v t={v.shape[-2]}).'
        )


def _attention_pallas(
    q: Float[Array, '... h s d'],
    k: Float[Array, '... h t d'],
    v: Float[Array, '... h t d_v'],
    *,
    bias: Optional[Float[Array, '... h s t']],
    mask: Optional[Bool[Array, '... h s t']],
    scale: float,
    causal: bool,
) -> Optional[Float[Array, '... h s d_v']]:
    """Pallas dispatch; ``None`` if the kernel rejects the shape / host.

    Import-on-call so a Pallas-broken install still reaches the JAX
    fallback.  Catches only ``PallasNotTileable`` -- a real kernel failure
    is not silently swallowed.
    """
    try:
        from ..._kernels.cuda.attention import (
            PallasNotTileable,
            scaled_dot_product_attention_pallas,
        )
    except Exception:
        return None
    try:
        return scaled_dot_product_attention_pallas(
            q,
            k,
            v,
            bias=bias,
            mask=mask,
            scale=scale,
            causal=causal,
        )
    except PallasNotTileable:
        return None


def scaled_dot_product_attention(
    q: Float[Array, '... h s d'],
    k: Float[Array, '... h t d'],
    v: Float[Array, '... h t d_v'],
    *,
    bias: Optional[Float[Array, '... h s t']] = None,
    mask: Optional[Bool[Array, '... h s t']] = None,
    scale: Optional[float] = None,
    causal: bool = False,
    qk_norm: bool = False,
    backend: Backend = 'auto',
) -> Float[Array, '... h s d_v']:
    """Scaled-dot-product attention with backend dispatch.

    Computes ``softmax(scale * q @ kᵀ + bias  [masked]) @ v`` over the key
    axis ``t`` (see :func:`reference_scaled_dot_product_attention` for the
    math).  ``backend`` selects the implementation:

    - ``'jax'`` -- the pure-JAX reference (the bit-faithful oracle).
    - ``'pallas-cuda'`` -- the fused flash path on Ampere+ (suite Phase 2);
      currently falls back to the reference with a loud warning.
    - ``'auto'`` (default) -- ``pallas-cuda`` on Ampere+ NVIDIA, else
      ``'jax'`` (see ``nitrix._internal.backend``).

    Parameters
    ----------
    q, k, v
        Query / key / value, layout ``(..., h, s, d)`` / ``(..., h, t, d)``
        / ``(..., h, t, d_v)``.  The module owns heads / windowing; the
        kernel sees already-partitioned ``q``/``k``/``v``.
    bias, mask, scale, causal, qk_norm
        As in :func:`reference_scaled_dot_product_attention`.
    backend
        ``'auto'`` / ``'pallas-cuda'`` / ``'jax'``.

    Returns
    -------
    Attention output of shape ``(..., h, s, d_v)``.

    Differentiability
    -----------------
    The reference path is autodiff-native.  On the fused path, ``qk_norm`` is a
    pure-elementwise RMS pre-op applied *outside* the kernel's ``custom_vjp``,
    so autodiff chains its VJP onto the unchanged fused forward / backward (the
    in-kernel tile-fusion of the norm is a later bandwidth-only optimisation).
    ``qk_norm=False`` is byte-identical to the no-norm path.
    """
    _validate(q, k, v)
    resolved_scale = default_scale(q.shape[-1]) if scale is None else scale
    resolved = resolve_backend(backend)
    if resolved == 'pallas-cuda':
        # RMS-norm outside the fused core: the kernel + its custom_vjp are
        # unchanged; autodiff composes the norm's VJP with the fused VJP.
        q_in = qk_rms_norm(q) if qk_norm else q
        k_in = qk_rms_norm(k) if qk_norm else k
        out = _attention_pallas(
            q_in,
            k_in,
            v,
            bias=bias,
            mask=mask,
            scale=resolved_scale,
            causal=causal,
        )
        if out is not None:
            return out
        fallback(
            function='scaled_dot_product_attention',
            requested='pallas-cuda',
            resolved='jax',
            reason=(
                'no fused attention kernel available for this shape/host '
                '(the fused path lands in suite Phase 2)'
            ),
            shapes=(tuple(q.shape), tuple(k.shape), tuple(v.shape)),
            dtype=q.dtype,
        )
    return reference_scaled_dot_product_attention(
        q,
        k,
        v,
        bias=bias,
        mask=mask,
        scale=resolved_scale,
        causal=causal,
        qk_norm=qk_norm,
    )
