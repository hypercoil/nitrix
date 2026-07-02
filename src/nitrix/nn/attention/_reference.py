# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Pure-JAX reference for scaled-dot-product attention.

This is the bit-faithful oracle that the fused Pallas path is certified
against, and the implementation that consumers may pin as a forward-parity
oracle by requesting ``backend='jax'``.  It is the canonical
``einsum -> (+ bias) -> (mask / causal) -> softmax -> einsum`` recipe with at
least float32 score and softmax accumulation, differentiated by ordinary
XLA autodiff (no hand-written vector-Jacobian product -- that lives on the
fused path).
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
    """Softmax temperature :math:`1 / \\sqrt{d}` (the SDPA default).

    Parameters
    ----------
    head_dim
        Size :math:`d` of the per-head query/key dimension.

    Returns
    -------
    float
        The scalar softmax temperature :math:`1 / \\sqrt{d}` applied to the
        query-key logits.
    """
    return 1.0 / math.sqrt(float(head_dim))


def qk_rms_norm(z: Float[Array, '... d']) -> Float[Array, '... d']:
    """Root-mean-square normalisation over the trailing head dimension.

    Applies QK-norm --- root-mean-square normalisation with no mean
    subtraction and no learnable scale ---
    :math:`\\bar z = z / \\sqrt{\\operatorname{mean}(z^2) + \\varepsilon}`,
    where the mean is taken over the trailing axis and
    :math:`\\varepsilon` is a small constant.  This is a cheap, purely
    elementwise pre-operation applied to the queries and keys before their
    dot product; it is autodiff-native, so it composes with both the
    reference and the fused kernel without touching either.

    Parameters
    ----------
    z
        Input tensor of shape ``(..., d)``; normalisation is over the
        trailing head dimension ``d``.

    Returns
    -------
    Float[Array, '... d']
        The input scaled elementwise so that each trailing-axis vector has
        unit root-mean-square, with the same shape and dtype as ``z``.
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

    Computes
    :math:`\\operatorname{softmax}(\\mathrm{scale} \\cdot q\\,k^{\\top} + b)\\,v`
    over the key axis ``t``, where the softmax logits are masked (set to
    :math:`-\\infty`) at excluded entries before normalisation and the
    additive bias :math:`b` is optional.  Leading dimensions ``...`` (batch)
    and the head axis ``h`` are handled by ``einsum`` broadcasting, and
    ``bias`` / ``mask`` broadcast against ``(..., h, s, t)``.

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
    Float[Array, '... h s d_v']
        Attention output of shape ``(..., h, s, d_v)`` in the common
        result type of ``q``, ``k`` and ``v``.

    Notes
    -----
    All compute runs in at least float32: a float16 or bfloat16 input is
    upcast to float32 for QK-norm, scores, softmax and value aggregation, and
    cast back at the end, so the oracle is platform-stable (no bfloat16
    tensor-core multiply variance); float64 inputs stay float64.  A fully
    masked query row yields ``nan`` (a degenerate case; callers should keep at
    least one key per query).
    """
    if scale is None:
        scale = default_scale(q.shape[-1])

    out_dtype = jnp.result_type(q, k, v)
    acc_dtype = jnp.promote_types(out_dtype, jnp.float32)
    if acc_dtype != out_dtype:
        # fp32-accumulation invariant (SPEC §2 tenet 11): a reduced-precision
        # (bf16/fp16) attention runs QK-norm, scores, softmax, and the value
        # aggregation in >= float32 and casts back to the I/O dtype at the end,
        # so the oracle is platform-stable (no bf16 tensor-core multiply
        # variance).  No-op for float32 / float64 inputs (acc_dtype == out_dtype).
        q = q.astype(acc_dtype)
        k = k.astype(acc_dtype)
        v = v.astype(acc_dtype)
    if qk_norm:
        q = qk_rms_norm(q)
        k = qk_rms_norm(k)

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
