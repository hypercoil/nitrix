# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Pure-JAX reference for scaled-dot-product attention.

This is the bit-faithful oracle the fused Pallas path (suite Phase 2) is
certified against, and the implementation ilex pins its forward-parity
oracle to (``backend='jax'``).  It is the canonical
``einsum -> (+ bias) -> (mask / causal) -> softmax -> einsum`` with at
least float32 score / softmax accumulation, differentiated by ordinary
XLA autodiff (no hand-written VJP -- that lives on the fused path).

See ``docs/feature-requests/nn-forward-kernels-suite.md`` §7.1.
"""

from __future__ import annotations

import math
from typing import Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

__all__ = [
    'reference_scaled_dot_product_attention',
    'default_scale',
    'qk_rms_norm',
]

# RMSNorm epsilon for QK-norm (the Gemma2 / ViT-22B convention).
_QK_RMS_EPS = 1e-6


def default_scale(head_dim: int) -> float:
    """Softmax temperature ``1 / sqrt(head_dim)`` (the SDPA default)."""
    return 1.0 / math.sqrt(float(head_dim))


def qk_rms_norm(z: Float[Array, '... d']) -> Float[Array, '... d']:
    """RMSNorm over the trailing head dim (QK-norm; no mean, no learnable scale).

    ``z̄ = z / sqrt(mean(z², axis=-1) + eps)``.  A cheap pure-elementwise pre-op
    applied to ``q`` and ``k`` before the dot; autodiff-native, so it composes
    with both the reference and the fused kernel without touching either.
    """
    ms = jnp.mean(jnp.square(z), axis=-1, keepdims=True)
    return z * jax.lax.rsqrt(ms + _QK_RMS_EPS)


def reference_scaled_dot_product_attention(
    q: Float[Array, '... h s d'],
    k: Float[Array, '... h t d'],
    v: Float[Array, '... h t d_v'],
    *,
    bias: Optional[Float[Array, '... h s t']] = None,
    mask: Optional[Bool[Array, '... h s t']] = None,
    scale: Optional[float] = None,
    causal: bool = False,
    qk_norm: bool = False,
) -> Float[Array, '... h s d_v']:
    """Reference scaled-dot-product attention (oracle).

    Computes ``softmax(scale * q @ kᵀ + bias  [masked]) @ v`` over the key
    axis ``t``.  Leading dims ``...`` (batch) and the head axis ``h`` are
    handled by ``einsum`` broadcasting; ``bias`` / ``mask`` broadcast
    against ``(..., h, s, t)``.

    Parameters
    ----------
    q, k, v
        Query / key / value.  ``q`` and ``k`` share head dim ``d``; ``k``
        and ``v`` share key length ``t``.  ``s`` may differ from ``t``
        (cross-attention).
    bias
        Optional additive logit bias (e.g. a Swin relative-position
        table), broadcast to ``(..., h, s, t)``.
    mask
        Optional boolean keep-mask (``True`` keeps an entry); broadcast to
        ``(..., h, s, t)``.  Combined with ``causal`` by logical-and.
    scale
        Softmax temperature; defaults to ``1 / sqrt(d)``.
    causal
        If ``True``, query ``i`` attends only to keys ``j <= i`` (indices
        aligned at 0 -- the standard self-attention convention).
    qk_norm
        If ``True``, RMS-normalise ``q`` and ``k`` over the head dim before the
        dot (QK-norm; logit-growth control at depth/scale).

    Returns
    -------
    Attention output of shape ``(..., h, s, d_v)`` in
    ``result_type(q, k, v)``.

    Notes
    -----
    Scores and softmax accumulate in at least float32 (float64 inputs stay
    float64); the output is cast back to the input result type.  A fully
    masked query row yields ``nan`` (degenerate; callers keep >= 1 key).
    """
    if scale is None:
        scale = default_scale(q.shape[-1])
    if qk_norm:
        q = qk_rms_norm(q)
        k = qk_rms_norm(k)

    out_dtype = jnp.result_type(q, k, v)
    acc_dtype = jnp.promote_types(out_dtype, jnp.float32)

    logits = jnp.einsum(
        '...hsd,...htd->...hst',
        q,
        k,
        preferred_element_type=acc_dtype,
    )
    logits = logits * scale
    if bias is not None:
        logits = logits + bias.astype(acc_dtype)

    keep = mask
    if causal:
        s = logits.shape[-2]
        t = logits.shape[-1]
        causal_keep = jnp.arange(s)[:, None] >= jnp.arange(t)[None, :]
        keep = causal_keep if keep is None else (keep & causal_keep)
    if keep is not None:
        logits = jnp.where(keep, logits, jnp.asarray(-jnp.inf, acc_dtype))

    weights = jax.nn.softmax(logits, axis=-1)
    out = jnp.einsum(
        '...hst,...htd->...hsd',
        weights,
        v,
        preferred_element_type=acc_dtype,
    )
    return out.astype(out_dtype)
