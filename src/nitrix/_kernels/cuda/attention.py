# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Pallas Triton fused scaled-dot-product (flash) attention.

Forked from ``jax.experimental.pallas.ops.gpu.attention`` (the Triton
``mha`` forward) and extended with the two capability gaps the stock kernel
lacks: an **additive bias** tile (Swin relative-position table) threaded into
the online softmax, and an explicit **boolean mask** tile (SAM-style padding /
cross-attention), in addition to ``causal``.  The score matrix ``(s, t)`` is
never materialised: the K loop streams a running ``(max, sum_exp)`` softmax
state (the same pattern as nitrix's ``LOG`` semiring monoid).

Differentiation is via a ``custom_vjp`` whose backward recomputes the gradient
through the JAX reference (the "xla backward"): correct ``dq/dk/dv`` and the
learnable-``bias`` gradient ``d_bias`` (broadcasting reduced by autodiff),
finite-difference checked.  The fully-fused Triton backward (recompute-in-tile
+ in-kernel ``d_bias``) is the next suite increment; this one already delivers
the forward / inference activation-memory win.

Scope (else ``PallasNotTileable`` -> loud JAX fallback): NVIDIA Ampere+,
power-of-two head dim with ``d_v == d``, and query / key lengths divisible by
the block size.  Anisotropic / awkward shapes fall back; pad-to-multiple is a
later refinement.

Implementation detail: never import from ``nitrix._kernels.cuda`` directly.
Use ``nitrix.nn.attention.scaled_dot_product_attention`` which handles backend
dispatch and fallback observability.
"""

from __future__ import annotations

import math
from functools import partial
from typing import Any, Optional, cast

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu
from jaxtyping import Array, Bool, Float

__all__ = [
    'scaled_dot_product_attention_pallas',
    'PallasNotTileable',
]

# Base-2 softmax: ``exp(x) = exp2(x * log2(e))``.  All logits (scores and the
# additive bias) are lifted into base-2 once, so the streamed ``exp2`` matches
# the natural-log reference.
_LOG2E = math.log2(math.e)
# Large negative sentinel for masked logits (survives the base-2 exp -> 0).
_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.float32).max)

# Block tile sizes.  64 (not the stock 128) keeps the live SRAM footprint --
# q / k / v / qk / bias / mask tiles -- inside the ~99 KB shared-memory budget
# of Ampere/Lovelace SMs once the additive-bias and mask tiles are added on top
# of the stock flash working set.  Perf-tuning the tile ladder is the sibling
# perf suite's job; correctness + fitting SRAM is this suite's.
_BLOCK_Q = 64
_BLOCK_K = 64


class PallasNotTileable(RuntimeError):
    """The Pallas kernel rejected the requested shape / host.

    Caught by the dispatcher in ``nitrix.nn.attention`` and translated into a
    ``NitrixBackendFallback`` warning (the JAX reference runs instead).
    """


# ---------------------------------------------------------------------------
# Forward kernel (one (block_q) row-tile per program; streams the K axis)
# ---------------------------------------------------------------------------


def _fwd_kernel(
    q_ref: Any,
    k_ref: Any,
    v_ref: Any,
    bias_ref: Any,
    mask_ref: Any,
    o_ref: Any,
    *,
    sm_scale: float,
    causal: bool,
    block_q: int,
    block_k: int,
) -> None:
    seq_k = k_ref.shape[0]
    start_q = pl.program_id(0)
    d = q_ref.shape[-1]

    m_i = jnp.full((block_q,), -jnp.inf, dtype=jnp.float32)
    l_i = jnp.zeros((block_q,), dtype=jnp.float32)
    o = jnp.zeros((block_q, d), dtype=jnp.float32)
    q = q_ref[...]
    qk_scale = sm_scale * _LOG2E

    def body(start_k: int, carry: Any) -> Any:
        o_prev, m_prev, l_prev = carry
        ks = pl.ds(start_k * block_k, block_k)
        k = k_ref[ks, :]
        qk = pl.dot(q, k.T) * qk_scale  # (block_q, block_k), fp32
        if bias_ref is not None:
            qk = qk + bias_ref[:, ks] * _LOG2E
        keep = None
        if mask_ref is not None:
            keep = mask_ref[:, ks]
        if causal:
            span_q = start_q * block_q + jnp.arange(block_q)
            span_k = start_k * block_k + jnp.arange(block_k)
            cm = span_q[:, None] >= span_k[None, :]
            keep = cm if keep is None else (keep & cm)
        if keep is not None:
            qk = jnp.where(keep, qk, _MASK_VALUE)

        m_curr = jnp.max(qk, axis=-1)
        m_next = jnp.maximum(m_prev, m_curr)
        corr = jnp.exp2(m_prev - m_next)
        s = jnp.exp2(qk - m_next[:, None])
        l_next = corr * l_prev + s.sum(axis=-1)
        v = v_ref[ks, :]
        o_next = corr[:, None] * o_prev + pl.dot(s.astype(v.dtype), v)
        # Pin carry dtypes (strong float32) so the fori_loop carry types match
        # regardless of the global x64 setting -- the python-scalar constants
        # (qk_scale, _MASK_VALUE) above can otherwise weak-type the carry and
        # trip "carry input/output type mismatch" under jax_enable_x64.
        return (
            o_next.astype(jnp.float32),
            m_next.astype(jnp.float32),
            l_next.astype(jnp.float32),
        )

    if causal:
        # Skip K blocks entirely past the diagonal for this Q tile.  Pin the
        # divisor dtype to start_q's (int32) so lax.div is x64-robust.
        upper = lax.div(
            block_q * (start_q + 1) + block_k - 1,
            jnp.asarray(block_k, start_q.dtype),
        )
    else:
        upper = pl.cdiv(seq_k, block_k)
    o, m_i, l_i = lax.fori_loop(0, upper, body, (o, m_i, l_i))
    o = o / l_i[:, None]
    o_ref[...] = o.astype(o_ref.dtype)


def _mha_forward(
    q: Float[Array, 'b s h d'],
    k: Float[Array, 'b t h d'],
    v: Float[Array, 'b t h d'],
    bias: Optional[Float[Array, 'b h s t']],
    mask: Optional[Bool[Array, 'b h s t']],
    *,
    sm_scale: float,
    causal: bool,
    block_q: int,
    block_k: int,
) -> Float[Array, 'b s h d']:
    """Pallas ``pallas_call`` driver on the stock ``(b, s, h, d)`` layout."""
    b, s, h, d = q.shape
    t = k.shape[1]
    grid = (pl.cdiv(s, block_q), b, h)
    kernel = partial(
        _fwd_kernel,
        sm_scale=sm_scale,
        causal=causal,
        block_q=block_q,
        block_k=block_k,
    )
    in_specs: list[Optional[pl.BlockSpec]] = [
        pl.BlockSpec((None, block_q, None, d), lambda i, j, hh: (j, i, hh, 0)),
        pl.BlockSpec((None, t, None, d), lambda i, j, hh: (j, 0, hh, 0)),
        pl.BlockSpec((None, t, None, d), lambda i, j, hh: (j, 0, hh, 0)),
        None
        if bias is None
        else pl.BlockSpec(
            (None, None, block_q, t), lambda i, j, hh: (j, hh, i, 0)
        ),
        None
        if mask is None
        else pl.BlockSpec(
            (None, None, block_q, t), lambda i, j, hh: (j, hh, i, 0)
        ),
    ]
    out_specs = pl.BlockSpec(
        (None, block_q, None, d), lambda i, j, hh: (j, i, hh, 0)
    )
    num_warps = 4 if d <= 64 else 8
    out = pl.pallas_call(
        kernel,
        grid=grid,
        in_specs=in_specs,
        out_specs=out_specs,
        out_shape=jax.ShapeDtypeStruct((b, s, h, d), q.dtype),
        compiler_params=plgpu.CompilerParams(
            num_warps=num_warps, num_stages=2
        ),
        name='nitrix_mha_forward',
    )(q, k, v, bias, mask)
    return cast(Float[Array, 'b s h d'], out)


# ---------------------------------------------------------------------------
# Layout adapter: public ``(... h s d)`` <-> kernel ``(b, s, h, d)``
# ---------------------------------------------------------------------------


def _fused_forward(
    q: Float[Array, '... h s d'],
    k: Float[Array, '... h t d'],
    v: Float[Array, '... h t d_v'],
    bias: Optional[Float[Array, '... h s t']],
    mask: Optional[Bool[Array, '... h s t']],
    scale: float,
    causal: bool,
) -> Float[Array, '... h s d_v']:
    *batch, h, s, d = q.shape
    t = k.shape[-2]
    nb = math.prod(batch) if batch else 1
    block_q = min(_BLOCK_Q, s)
    block_k = min(_BLOCK_K, t)

    def to_bshd(x: Array, length: int) -> Array:
        # (*batch, h, length, d) -> (nb, length, h, d)
        return x.reshape(nb, h, length, d).transpose(0, 2, 1, 3)

    qf = to_bshd(q, s)
    kf = to_bshd(k, t)
    vf = to_bshd(v, t)
    bf = (
        None
        if bias is None
        else jnp.broadcast_to(bias, (*batch, h, s, t)).reshape(nb, h, s, t)
    )
    mf = (
        None
        if mask is None
        else jnp.broadcast_to(mask, (*batch, h, s, t)).reshape(nb, h, s, t)
    )
    out = _mha_forward(
        qf,
        kf,
        vf,
        bf,
        mf,
        sm_scale=scale,
        causal=causal,
        block_q=block_q,
        block_k=block_k,
    )
    # (nb, s, h, d) -> (*batch, h, s, d)
    return out.transpose(0, 2, 1, 3).reshape(*batch, h, s, d)


# ---------------------------------------------------------------------------
# Differentiable wrapper (custom_vjp; backward via the reference recompute)
# ---------------------------------------------------------------------------


@partial(jax.custom_vjp, nondiff_argnums=(5, 6))
def _sdpa_p(
    q: Array,
    k: Array,
    v: Array,
    bias: Optional[Array],
    mask: Optional[Array],
    scale: float,
    causal: bool,
) -> Array:
    return _fused_forward(q, k, v, bias, mask, scale, causal)


def _sdpa_p_fwd(
    q: Array,
    k: Array,
    v: Array,
    bias: Optional[Array],
    mask: Optional[Array],
    scale: float,
    causal: bool,
) -> tuple[Array, tuple[Any, ...]]:
    out = _fused_forward(q, k, v, bias, mask, scale, causal)
    return out, (q, k, v, bias, mask)


def _sdpa_p_bwd(
    scale: float,
    causal: bool,
    res: tuple[Any, ...],
    g: Array,
) -> tuple[Any, ...]:
    # Gradient of the *reference* (the function the fused forward approximates):
    # correct dq/dk/dv and the learnable-bias gradient d_bias, with bias
    # broadcasting reduced automatically by autodiff.  mask is boolean -> None.
    from ...nn.attention._reference import (
        reference_scaled_dot_product_attention as _ref,
    )

    q, k, v, bias, mask = res
    if bias is None:

        def f_nb(q: Array, k: Array, v: Array) -> Array:
            return _ref(q, k, v, mask=mask, scale=scale, causal=causal)

        _, vjp = jax.vjp(f_nb, q, k, v)
        dq, dk, dv = vjp(g)
        return dq, dk, dv, None, None

    def f_b(q: Array, k: Array, v: Array, bias: Array) -> Array:
        return _ref(q, k, v, bias=bias, mask=mask, scale=scale, causal=causal)

    _, vjp = jax.vjp(f_b, q, k, v, bias)
    dq, dk, dv, dbias = vjp(g)
    return dq, dk, dv, dbias, None


_sdpa_p.defvjp(_sdpa_p_fwd, _sdpa_p_bwd)


# ---------------------------------------------------------------------------
# Tileability gate + public entry point
# ---------------------------------------------------------------------------


def _check_tileable(
    q: Float[Array, '... h s d'],
    k: Float[Array, '... h t d'],
    v: Float[Array, '... h t d_v'],
) -> None:
    """Static rejection (-> loud JAX fallback) of shapes the kernel can't run."""
    d = q.shape[-1]
    d_v = v.shape[-1]
    s = q.shape[-2]
    t = k.shape[-2]
    if q.dtype != jnp.float32:
        raise PallasNotTileable(
            f'fused attention currently supports float32 only; got '
            f'{q.dtype} (use backend="jax" for other dtypes).'
        )
    if d_v != d:
        raise PallasNotTileable(
            f'fused attention requires d_v == d; got d={d}, d_v={d_v}.'
        )
    if d != pl.next_power_of_2(d) or d < 16:
        raise PallasNotTileable(
            f'fused attention requires a power-of-two head dim >= 16; got {d}.'
        )
    block_q = min(_BLOCK_Q, s)
    block_k = min(_BLOCK_K, t)
    if s % block_q != 0 or t % block_k != 0:
        raise PallasNotTileable(
            f'fused attention requires s % {block_q} == 0 and t % {block_k} '
            f'== 0; got s={s}, t={t} (pad to a friendly shape or use '
            'backend="jax").'
        )


def scaled_dot_product_attention_pallas(
    q: Float[Array, '... h s d'],
    k: Float[Array, '... h t d'],
    v: Float[Array, '... h t d_v'],
    *,
    scale: float,
    bias: Optional[Float[Array, '... h s t']] = None,
    mask: Optional[Bool[Array, '... h s t']] = None,
    causal: bool = False,
) -> Float[Array, '... h s d_v']:
    """Fused flash attention (NVIDIA Ampere+); differentiable via ``custom_vjp``.

    Raises
    ------
    PallasNotTileable
        If the shape is outside the kernel's supported set; the dispatcher
        catches this and runs the JAX reference with a loud fallback.
    """
    _check_tileable(q, k, v)
    return _sdpa_p(q, k, v, bias, mask, scale, causal)
