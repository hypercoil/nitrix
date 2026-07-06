# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
# isort: skip_file
"""
Scaled-dot-product / flash attention.

The public entry point :func:`scaled_dot_product_attention` supports
three-level backend selection.  Under the two-tier parity contract the
``'jax'`` reference is the bit-faithful oracle and ``'pallas-cuda'`` is the
fused flash path, certified equivalent to the reference (:math:`\\text{pallas-cuda}
\\approx \\text{jax}`) only inside nitrix; downstream consumers pin their own
forward-parity oracle to ``backend='jax'``.

On Ampere+ NVIDIA hardware the fused flash kernel runs; a shape it cannot
tile -- non-float32, ``d_v != d``, a non-power-of-two head dimension, or
non-power-of-two query / key lengths -- declines and falls back to the
reference with a loud ``NitrixBackendFallback`` warning.
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
    """Dispatch to the fused Pallas attention kernel, or signal it is unusable.

    The kernel module is imported on call so that a Pallas-broken install
    still reaches the JAX fallback rather than failing at import time.  Only
    ``PallasNotTileable`` (the kernel declining an unsupported shape or host)
    is caught; any other exception is a genuine kernel failure and is not
    silently swallowed.

    Parameters
    ----------
    q, k, v
        Query / key / value, layout ``(..., h, s, d)`` / ``(..., h, t, d)`` /
        ``(..., h, t, d_v)``.
    bias
        Optional additive logit bias broadcast to ``(..., h, s, t)``, or
        ``None``.
    mask
        Optional boolean keep-mask (``True`` keeps an entry) broadcast to
        ``(..., h, s, t)``, or ``None``.
    scale
        Softmax temperature applied to the query-key dot product.
    causal
        If ``True``, apply a causal keep-mask so query ``i`` attends only to
        keys ``j <= i``.

    Returns
    -------
    Optional[Float[Array, '... h s d_v']]
        Attention output of shape ``(..., h, s, d_v)`` when the fused kernel
        runs, or ``None`` if the kernel is unavailable or rejects the shape /
        host (signalling the caller to fall back to the reference path).
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

    Computes :math:`\\operatorname{softmax}(\\text{scale} \\cdot q k^{\\top} +
    \\text{bias}\\ [\\text{masked}])\\, v` over the key axis ``t`` (see
    :func:`reference_scaled_dot_product_attention` for the full definition).
    The ``backend`` argument selects the implementation:

    - ``'jax'`` -- the pure-JAX reference (the bit-faithful oracle).
    - ``'pallas-cuda'`` -- the fused flash path on Ampere+; a shape it cannot
      tile falls back to the reference with a loud warning.
    - ``'auto'`` (default) -- ``'pallas-cuda'`` on Ampere+ NVIDIA hardware,
      else ``'jax'``.

    Parameters
    ----------
    q, k, v
        Query / key / value, layout ``(..., h, s, d)`` / ``(..., h, t, d)`` /
        ``(..., h, t, d_v)``.  Head partitioning and windowing are the
        caller's responsibility; this function sees already-partitioned
        ``q`` / ``k`` / ``v``.  ``q`` and ``k`` share head dim ``d``; ``k``
        and ``v`` share key length ``t``.  Query length ``s`` may differ from
        ``t`` (cross-attention).
    bias
        Optional additive logit bias (e.g. a relative-position table),
        broadcast to ``(..., h, s, t)``.
    mask
        Optional boolean keep-mask (``True`` keeps an entry), broadcast to
        ``(..., h, s, t)`` and combined with ``causal`` by logical-and.
    scale
        Softmax temperature; defaults to :math:`1 / \\sqrt{d}`.
    causal
        If ``True``, query ``i`` attends only to keys ``j <= i`` (indices
        aligned at 0, the standard self-attention convention).
    qk_norm
        If ``True``, RMS-normalise ``q`` and ``k`` over the head dim before
        the dot product (QK-norm; logit-growth control at depth / scale).
    backend
        Execution engine: ``'auto'``, ``'pallas-cuda'`` or ``'jax'``.

    Returns
    -------
    Float[Array, '... h s d_v']
        Attention output of shape ``(..., h, s, d_v)``.

    Notes
    -----
    The reference path is autodiff-native.  On the fused path, ``qk_norm`` is
    a pure-elementwise RMS pre-op applied *outside* the kernel's custom VJP,
    so autodiff chains its VJP onto the unchanged fused forward / backward
    (in-kernel tile-fusion of the norm is a later bandwidth-only
    optimisation).  ``qk_norm=False`` is byte-identical to the no-norm path.
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
                'the fused attention kernel does not support this shape/host '
                '(needs Ampere+, float32, d_v == d, and power-of-two head / '
                'token dimensions)'
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
