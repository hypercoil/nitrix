# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
r"""
Pallas-Triton fused scaled-dot-product (flash) attention -- forward + backward.

Forked from ``jax.experimental.pallas.ops.gpu.attention`` (the Triton ``mha``)
and extended with the two capability gaps the stock kernel lacks: an *additive
bias* tile (e.g. a Swin relative-position table) threaded into the online
softmax, and an explicit *boolean mask* tile (e.g. SAM-style padding /
cross-attention), in addition to ``causal``.  The :math:`(s, t)` score matrix is
never materialised: the forward streams a running :math:`(\max, \sum \exp)`
state (the same pattern as the log-domain semiring monoid), and the backward
recomputes the score tiles in shared memory rather than reading a stored
:math:`(s, t)` activation.

The backward pass is two kernels (with no ``q_blocks == k_blocks`` constraint):
one accumulates :math:`dk` / :math:`dv` and, when a bias is supplied, writes the
learnable-bias gradient :math:`d_{bias}` tile-by-tile; the other accumulates
:math:`dq`.  Both recompute :math:`p = \exp_2(\text{scores} + \text{bias} -
\text{lse})` from the forward pass's ``lse`` residual, so no per-step softmax
state is stored.  The wrapper's :func:`jax.custom_vjp` reduces :math:`d_{bias}`
over the broadcast axes back to the caller's bias shape.

The supported shapes (anything else raises :class:`PallasNotTileable`, which the
dispatcher turns into a loud JAX fallback) are: NVIDIA Ampere+ hardware, float32
data, a power-of-two head dimension :math:`\geq 16` with ``d_v == d``, and
power-of-two query / key lengths.  Anisotropic or awkward shapes fall
back; pad-to-multiple is a later refinement, and the at-scale
wall-clock-versus-reference certification is the sibling perf suite's job.

This is a private implementation detail: never import from
``nitrix._kernels.cuda`` directly.  Use
:func:`nitrix.nn.attention.scaled_dot_product_attention`, which handles backend
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

# One block size for forward and backward so a shape that tiles the forward
# tiles the backward too.  32 keeps the live SRAM footprint -- q / k / v / qk /
# bias / mask tiles plus the backward dk/dv/dq accumulators -- inside the ~99 KB
# shared-memory budget of Ampere/Lovelace SMs.  Perf-tuning the tile ladder is
# the sibling perf suite's job; correctness + fitting SRAM is this suite's.
_BLOCK = 32


class PallasNotTileable(RuntimeError):
    """The Pallas kernel rejected the requested shape / host.

    Caught by the dispatcher in :mod:`nitrix.nn.attention` and translated into a
    ``NitrixBackendFallback`` warning, after which the JAX reference runs
    instead.
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
    lse_ref: Any,
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
    lse_ref[...] = (m_i + jnp.log2(l_i)).astype(lse_ref.dtype)


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
) -> tuple[Float[Array, 'b s h d'], Float[Array, 'b h s']]:
    """Run the fused forward attention kernel on the ``(b, s, h, d)`` layout.

    Launches the flash-attention forward Pallas kernel over a grid of one
    query-row tile per program, streaming the key axis and accumulating the
    online-softmax output plus the log-sum-exp residual needed by the backward
    pass.

    Parameters
    ----------
    q : Float[Array, 'b s h d']
        Query activations, laid out as ``(batch, query length, heads, head
        dim)``.
    k : Float[Array, 'b t h d']
        Key activations, laid out as ``(batch, key length, heads, head dim)``.
    v : Float[Array, 'b t h d']
        Value activations, laid out as ``(batch, key length, heads, head dim)``.
    bias : Float[Array, 'b h s t'] or None
        Optional additive attention bias, broadcast to ``(batch, heads, query
        length, key length)`` and added to the logits before the softmax.  When
        ``None`` no bias tile is threaded through the kernel.
    mask : Bool[Array, 'b h s t'] or None
        Optional boolean keep-mask over ``(batch, heads, query length, key
        length)``; positions that are ``False`` are driven to a large negative
        sentinel before the softmax.  When ``None`` no mask tile is used.
    sm_scale : float
        Softmax scale applied to the raw query-key dot products.
    causal : bool
        If ``True``, apply a lower-triangular causal mask so each query attends
        only to keys at or before its own position, and skip key blocks that lie
        wholly past the diagonal.
    block_q : int
        Query-tile height (rows per program).
    block_k : int
        Key-tile width (columns streamed per inner-loop step).

    Returns
    -------
    out : Float[Array, 'b s h d']
        Attention output in the same ``(batch, query length, heads, head dim)``
        layout as ``q``.
    lse : Float[Array, 'b h s']
        Per-query base-2 log-sum-exp residual, in float32, retained so the
        backward pass can recompute the softmax weights without storing the
        score matrix.
    """
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
    out_specs = [
        pl.BlockSpec((None, block_q, None, d), lambda i, j, hh: (j, i, hh, 0)),
        pl.BlockSpec((None, None, block_q), lambda i, j, hh: (j, hh, i)),
    ]
    out_shape = [
        jax.ShapeDtypeStruct((b, s, h, d), q.dtype),
        jax.ShapeDtypeStruct((b, h, s), jnp.float32),
    ]
    num_warps = 4 if d <= 64 else 8
    out, lse = pl.pallas_call(
        kernel,
        grid=grid,
        in_specs=in_specs,
        out_specs=out_specs,
        out_shape=out_shape,
        compiler_params=plgpu.CompilerParams(
            num_warps=num_warps, num_stages=2
        ),
        name='nitrix_mha_forward',
    )(q, k, v, bias, mask)
    return (
        cast(Float[Array, 'b s h d'], out),
        cast(Float[Array, 'b h s'], lse),
    )


# ---------------------------------------------------------------------------
# Backward kernels (recompute the score tiles; never read a stored (s, t))
# ---------------------------------------------------------------------------


def _bwd_kv_kernel(
    q_ref: Any,
    do_ref: Any,
    k_ref: Any,
    v_ref: Any,
    lse_ref: Any,
    delta_ref: Any,
    bias_ref: Any,
    mask_ref: Any,
    dk_ref: Any,
    dv_ref: Any,
    *out_extra: Any,
    sm_scale: float,
    causal: bool,
    block_q: int,
    block_k: int,
    num_q: int,
    has_bias: bool,
) -> None:
    # dk / dv for one K tile (+ d_bias stripe when has_bias); loop over all Q.
    dbias_ref: Any = out_extra[0] if has_bias else None
    start_k = pl.program_id(0)
    d = k_ref.shape[-1]
    k = k_ref[...]
    v = v_ref[...]
    qk_scale = sm_scale * _LOG2E
    span_k = start_k * block_k + jnp.arange(block_k)

    def body(start_q: int, carry: Any) -> Any:
        dk, dv = carry
        qs = pl.ds(start_q * block_q, block_q)
        q = q_ref[qs, :]
        do = do_ref[qs, :]
        lse_i = lse_ref[qs]
        delta_i = delta_ref[qs]
        qk = pl.dot(q, k.T) * qk_scale
        if has_bias:
            qk = qk + bias_ref[qs, :] * _LOG2E
        keep = None
        if mask_ref is not None:
            keep = mask_ref[qs, :]
        if causal:
            span_q = start_q * block_q + jnp.arange(block_q)
            cm = span_q[:, None] >= span_k[None, :]
            keep = cm if keep is None else (keep & cm)
        if keep is not None:
            qk = jnp.where(keep, qk, _MASK_VALUE)
        p = jnp.exp2(qk - lse_i[:, None])  # true softmax weights
        dv = dv + pl.dot(p.astype(do.dtype).T, do)
        dp = pl.dot(do, v.T) - delta_i[:, None]
        ds = p * dp  # dL/dscore == dL/dbias (unscaled)
        if has_bias:
            dbias_ref[qs, :] = ds.astype(dbias_ref.dtype)
        dk = dk + sm_scale * pl.dot(ds.astype(q.dtype).T, q)
        return dk.astype(jnp.float32), dv.astype(jnp.float32)

    dk0 = jnp.zeros((block_k, d), dtype=jnp.float32)
    dv0 = jnp.zeros((block_k, d), dtype=jnp.float32)
    dk, dv = lax.fori_loop(0, num_q, body, (dk0, dv0))
    dk_ref[...] = dk.astype(dk_ref.dtype)
    dv_ref[...] = dv.astype(dv_ref.dtype)


def _bwd_q_kernel(
    q_ref: Any,
    do_ref: Any,
    k_ref: Any,
    v_ref: Any,
    lse_ref: Any,
    delta_ref: Any,
    bias_ref: Any,
    mask_ref: Any,
    dq_ref: Any,
    *,
    sm_scale: float,
    causal: bool,
    block_q: int,
    block_k: int,
    num_k: int,
    has_bias: bool,
) -> None:
    # dq for one Q tile; loop over all K.
    start_q = pl.program_id(0)
    d = q_ref.shape[-1]
    q = q_ref[...]
    do = do_ref[...]
    lse_i = lse_ref[...]
    delta_i = delta_ref[...]
    qk_scale = sm_scale * _LOG2E
    span_q = start_q * block_q + jnp.arange(block_q)

    def body(start_k: int, dq: Any) -> Any:
        ks = pl.ds(start_k * block_k, block_k)
        k = k_ref[ks, :]
        v = v_ref[ks, :]
        qk = pl.dot(q, k.T) * qk_scale
        if has_bias:
            qk = qk + bias_ref[:, ks] * _LOG2E
        keep = None
        if mask_ref is not None:
            keep = mask_ref[:, ks]
        if causal:
            span_k = start_k * block_k + jnp.arange(block_k)
            cm = span_q[:, None] >= span_k[None, :]
            keep = cm if keep is None else (keep & cm)
        if keep is not None:
            qk = jnp.where(keep, qk, _MASK_VALUE)
        p = jnp.exp2(qk - lse_i[:, None])
        dp = pl.dot(do, v.T) - delta_i[:, None]
        ds = p * dp
        dq = dq + sm_scale * pl.dot(ds.astype(k.dtype), k)
        return dq.astype(jnp.float32)

    dq0 = jnp.zeros((block_q, d), dtype=jnp.float32)
    dq = lax.fori_loop(0, num_k, body, dq0)
    dq_ref[...] = dq.astype(dq_ref.dtype)


def _mha_backward(
    q: Float[Array, 'b s h d'],
    k: Float[Array, 'b t h d'],
    v: Float[Array, 'b t h d'],
    do: Float[Array, 'b s h d'],
    lse: Float[Array, 'b h s'],
    delta: Float[Array, 'b h s'],
    bias: Optional[Float[Array, 'b h s t']],
    mask: Optional[Bool[Array, 'b h s t']],
    *,
    sm_scale: float,
    causal: bool,
    block: int,
) -> tuple[Array, Array, Array, Optional[Array]]:
    b, s, h, d = q.shape
    t = k.shape[1]
    num_q = s // block
    num_k = t // block
    has_bias = bias is not None
    num_warps = 4 if d <= 64 else 8
    cp = plgpu.CompilerParams(num_warps=num_warps, num_stages=2)

    # --- dk / dv (+ d_bias): grid over K tiles, loop over Q ---
    q_full = pl.BlockSpec((None, s, None, d), lambda kb, j, hh: (j, 0, hh, 0))
    do_full = pl.BlockSpec((None, s, None, d), lambda kb, j, hh: (j, 0, hh, 0))
    kv_tile = pl.BlockSpec(
        (None, block, None, d), lambda kb, j, hh: (j, kb, hh, 0)
    )
    svec_full = pl.BlockSpec((None, None, s), lambda kb, j, hh: (j, hh, 0))
    stripe = pl.BlockSpec(
        (None, None, s, block), lambda kb, j, hh: (j, hh, 0, kb)
    )
    kv_in: list[Optional[pl.BlockSpec]] = [
        q_full,
        do_full,
        kv_tile,
        kv_tile,
        svec_full,
        svec_full,
        None if bias is None else stripe,
        None if mask is None else stripe,
    ]
    kv_out = [kv_tile, kv_tile]  # dk, dv
    kv_out_shape = [
        jax.ShapeDtypeStruct((b, t, h, d), q.dtype),
        jax.ShapeDtypeStruct((b, t, h, d), q.dtype),
    ]
    if has_bias:
        kv_out = kv_out + [stripe]
        kv_out_shape = kv_out_shape + [
            jax.ShapeDtypeStruct((b, h, s, t), jnp.float32)
        ]
    kv_results = pl.pallas_call(
        partial(
            _bwd_kv_kernel,
            sm_scale=sm_scale,
            causal=causal,
            block_q=block,
            block_k=block,
            num_q=num_q,
            has_bias=has_bias,
        ),
        grid=(num_k, b, h),
        in_specs=kv_in,
        out_specs=kv_out,
        out_shape=kv_out_shape,
        compiler_params=cp,
        name='nitrix_mha_backward_kv',
    )(q, do, k, v, lse, delta, bias, mask)
    dk = kv_results[0]
    dv = kv_results[1]
    dbias = kv_results[2] if has_bias else None

    # --- dq: grid over Q tiles, loop over K ---
    q_tile = pl.BlockSpec(
        (None, block, None, d), lambda qb, j, hh: (j, qb, hh, 0)
    )
    kv_full = pl.BlockSpec((None, t, None, d), lambda qb, j, hh: (j, 0, hh, 0))
    qvec = pl.BlockSpec((None, None, block), lambda qb, j, hh: (j, hh, qb))
    q_row = pl.BlockSpec(
        (None, None, block, t), lambda qb, j, hh: (j, hh, qb, 0)
    )
    q_in: list[Optional[pl.BlockSpec]] = [
        q_tile,
        q_tile,
        kv_full,
        kv_full,
        qvec,
        qvec,
        None if bias is None else q_row,
        None if mask is None else q_row,
    ]
    dq = pl.pallas_call(
        partial(
            _bwd_q_kernel,
            sm_scale=sm_scale,
            causal=causal,
            block_q=block,
            block_k=block,
            num_k=num_k,
            has_bias=has_bias,
        ),
        grid=(num_q, b, h),
        in_specs=q_in,
        out_specs=q_tile,
        out_shape=jax.ShapeDtypeStruct((b, s, h, d), q.dtype),
        compiler_params=cp,
        name='nitrix_mha_backward_q',
    )(q, do, k, v, lse, delta, bias, mask)
    return dq, dk, dv, dbias


# ---------------------------------------------------------------------------
# Layout adapter: public ``(... h s d)`` <-> kernel ``(b, s, h, d)``
# ---------------------------------------------------------------------------


def _to_bshd(x: Array, nb: int, h: int, length: int, d: int) -> Array:
    return x.reshape(nb, h, length, d).transpose(0, 2, 1, 3)


def _from_bshd(
    x: Array, batch: tuple[int, ...], h: int, length: int, d: int
) -> Array:
    return x.transpose(0, 2, 1, 3).reshape(*batch, h, length, d)


def _unbroadcast(g: Array, shape: tuple[int, ...]) -> Array:
    """Reduce a broadcast cotangent down to a target shape.

    Implements the vector-Jacobian product of a broadcast: leading axes that
    were introduced by the broadcast are summed away, then any axis that the
    target shape holds at size one (but which the cotangent expanded) is summed
    with the dimension kept, and finally the result is reshaped to ``shape``.

    Parameters
    ----------
    g : Array
        The cotangent to reduce, whose shape is a broadcast of ``shape``.
    shape : tuple of int
        The target shape (the original, pre-broadcast shape of the operand).

    Returns
    -------
    Array
        The cotangent summed and reshaped to exactly ``shape``.
    """
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    axes = tuple(
        i for i, dim in enumerate(shape) if dim == 1 and g.shape[i] != 1
    )
    if axes:
        g = g.sum(axis=axes, keepdims=True)
    return g.reshape(shape)


def _fused_forward_with_lse(
    q: Float[Array, '... h s d'],
    k: Float[Array, '... h t d'],
    v: Float[Array, '... h t d_v'],
    bias: Optional[Float[Array, '... h s t']],
    mask: Optional[Bool[Array, '... h s t']],
    scale: float,
    causal: bool,
) -> tuple[Float[Array, '... h s d_v'], Float[Array, '... h s']]:
    *batch, h, s, d = q.shape
    t = k.shape[-2]
    nb = math.prod(batch) if batch else 1
    block = min(_BLOCK, s)
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
    out, lse = _mha_forward(
        _to_bshd(q, nb, h, s, d),
        _to_bshd(k, nb, h, t, d),
        _to_bshd(v, nb, h, t, d),
        bf,
        mf,
        sm_scale=scale,
        causal=causal,
        block_q=block,
        block_k=min(_BLOCK, t),
    )
    return (
        _from_bshd(out, tuple(batch), h, s, d),
        lse.reshape(*batch, h, s),
    )


def _fused_forward(
    q: Float[Array, '... h s d'],
    k: Float[Array, '... h t d'],
    v: Float[Array, '... h t d_v'],
    bias: Optional[Float[Array, '... h s t']],
    mask: Optional[Bool[Array, '... h s t']],
    scale: float,
    causal: bool,
) -> Float[Array, '... h s d_v']:
    out, _ = _fused_forward_with_lse(q, k, v, bias, mask, scale, causal)
    return out


def _fused_backward(
    q: Float[Array, '... h s d'],
    k: Float[Array, '... h t d'],
    v: Float[Array, '... h t d_v'],
    bias: Optional[Float[Array, '... h s t']],
    mask: Optional[Bool[Array, '... h s t']],
    o: Float[Array, '... h s d_v'],
    lse: Float[Array, '... h s'],
    do: Float[Array, '... h s d_v'],
    scale: float,
    causal: bool,
) -> tuple[Array, Array, Array, Optional[Array]]:
    *batch, h, s, d = q.shape
    t = k.shape[-2]
    nb = math.prod(batch) if batch else 1
    block = min(_BLOCK, s)
    qf = _to_bshd(q, nb, h, s, d)
    kf = _to_bshd(k, nb, h, t, d)
    vf = _to_bshd(v, nb, h, t, d)
    of = _to_bshd(o, nb, h, s, d)
    dof = _to_bshd(do, nb, h, s, d)
    # delta_i = sum_j p_ij dp_ij = o_i . do_i (cheap; no (s, t) materialised).
    delta = (of * dof).sum(axis=-1).transpose(0, 2, 1)  # (nb, h, s)
    lsef = lse.reshape(nb, h, s)
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
    dq_f, dk_f, dv_f, dbias_f = _mha_backward(
        qf,
        kf,
        vf,
        dof,
        lsef,
        delta,
        bf,
        mf,
        sm_scale=scale,
        causal=causal,
        block=block,
    )
    dq = _from_bshd(dq_f, tuple(batch), h, s, d)
    dk = _from_bshd(dk_f, tuple(batch), h, t, d)
    dv = _from_bshd(dv_f, tuple(batch), h, t, d)
    dbias: Optional[Array] = None
    if bias is not None and dbias_f is not None:
        dbias = _unbroadcast(dbias_f.reshape(*batch, h, s, t), bias.shape)
    return dq, dk, dv, dbias


# ---------------------------------------------------------------------------
# Differentiable wrapper (custom_vjp; fused forward + fused backward)
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
    out, lse = _fused_forward_with_lse(q, k, v, bias, mask, scale, causal)
    return out, (q, k, v, bias, mask, out, lse)


def _sdpa_p_bwd(
    scale: float,
    causal: bool,
    res: tuple[Any, ...],
    g: Array,
) -> tuple[Any, ...]:
    q, k, v, bias, mask, o, lse = res
    dq, dk, dv, dbias = _fused_backward(
        q, k, v, bias, mask, o, lse, g, scale, causal
    )
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
    """Reject, at trace time, shapes the fused kernel cannot run.

    Raises :class:`PallasNotTileable` (which the dispatcher turns into a loud JAX
    fallback) unless the inputs satisfy every kernel requirement: float32 dtype,
    ``d_v == d``, a power-of-two head dimension of at least 16, and power-of-two
    query / key lengths.

    Parameters
    ----------
    q : Float[Array, '... h s d']
        Query activations; supplies the head dimension :math:`d`, the query
        length :math:`s`, and the checked dtype.
    k : Float[Array, '... h t d']
        Key activations; supplies the key length :math:`t`.
    v : Float[Array, '... h t d_v']
        Value activations; supplies the value head dimension :math:`d_v`, which
        must equal :math:`d`.

    Raises
    ------
    PallasNotTileable
        If any requirement above is violated.
    """
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
    # The forward loads the full key length ``t`` as one SRAM tile and the
    # backward the full query length ``s``; the Triton lowering requires every
    # tile dimension to be a power of 2 (the block sizes and their divisibility
    # then follow automatically).  Non-power-of-two token counts -- including
    # ones the block size happens to divide evenly, e.g. ``28 = min(32, 28)`` --
    # must decline here so the dispatcher falls back to JAX, rather than pass
    # the gate and crash the lowering mid-trace.
    if s != pl.next_power_of_2(s) or t != pl.next_power_of_2(t):
        raise PallasNotTileable(
            f'fused attention requires power-of-two query/key lengths; got '
            f's={s}, t={t} (non-power-of-two token counts fall back to '
            'backend="jax"; pad to a power of 2 to use the fused kernel).'
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
    """Compute fused flash attention on NVIDIA Ampere+ hardware.

    Runs the Pallas-Triton fused scaled-dot-product attention kernel, which
    streams the online softmax without materialising the score matrix and is
    differentiable via :func:`jax.custom_vjp`.  The supported shape set is
    checked first (see :func:`_check_tileable`); shapes outside it raise
    :class:`PallasNotTileable` so the caller can fall back to the JAX reference.

    Parameters
    ----------
    q : Float[Array, '... h s d']
        Query activations, with leading batch axes, then ``(heads, query
        length, head dim)``.
    k : Float[Array, '... h t d']
        Key activations, with the same leading batch axes and ``(heads, key
        length, head dim)``.
    v : Float[Array, '... h t d_v']
        Value activations, with ``(heads, key length, value head dim)``; the
        kernel requires ``d_v == d``.
    scale : float
        Softmax scale applied to the query-key dot products.
    bias : Float[Array, '... h s t'] or None, optional
        Additive attention bias over ``(heads, query length, key length)``,
        broadcast against the batch axes and added to the logits before the
        softmax.  Defaults to ``None`` (no bias).
    mask : Bool[Array, '... h s t'] or None, optional
        Boolean keep-mask over ``(heads, query length, key length)``; ``False``
        positions are excluded from the softmax.  Defaults to ``None`` (no
        mask).
    causal : bool, optional
        If ``True``, apply a causal mask so each query attends only to keys at
        or before its own position.  Defaults to ``False``.

    Returns
    -------
    Float[Array, '... h s d_v']
        Attention output with the same leading batch axes as the inputs and
        trailing shape ``(heads, query length, value head dim)``.

    Raises
    ------
    PallasNotTileable
        If the shape is outside the kernel's supported set; the dispatcher
        catches this and runs the JAX reference with a loud fallback.
    """
    _check_tileable(q, k, v)
    return _sdpa_p(q, k, v, bias, mask, scale, causal)
