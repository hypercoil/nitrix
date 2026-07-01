# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Pallas Triton fused selective state-space scan (Mamba / S6) -- forward + backward.

Clean-room kernel.  Triton-Pallas lowers neither ``cumprod`` /
``associative_scan`` nor element / ``slice`` indexing of register tiles -- only
whole-tile ops and ``cumsum``.  The linear recurrence
:math:`h_t = \\exp(\\Delta_t A)\\, h_{t-1} + \\Delta_t B_t x_t` is therefore
evaluated as a **chunked cumsum closed form**: with
:math:`\\log P_t = A \\cdot \\operatorname{cumsum}(\\Delta)_t` over a chunk,
:math:`h_t = \\exp(\\log P_t) \\cdot \\bigl(\\operatorname{cumsum}(\\Delta_i B_i x_i \\exp(-\\log P_i)) + h_{\\mathrm{start}}\\bigr)`.
The chunk size keeps :math:`\\exp(\\pm \\log P)` bounded; the chunk-final state
is carried to the next chunk via whole-tile sums (no indexing).  The
:math:`(l, d, n)` state trajectory is never materialised in HBM -- only
:math:`y\\,(l, d)` and the per-chunk start states.

Backward is the fully-fused recompute-adjoint (the *training*-memory win: no
:math:`(l, d, n)` trajectory in HBM in either pass).  The reverse linear
recurrence :math:`a_t = dy_t C_t + dA_{t+1} a_{t+1}` is the same chunked cumsum
form run in reverse (:math:`\\operatorname{rev\\_cumsum}(z) = \\sum z - \\operatorname{cumsum}(z) + z`,
no flip); the forward states :math:`h_{t-1}` are recomputed from the saved
per-chunk start states as :math:`(h_t - \\Delta_t B_t x_t) / \\exp(\\Delta_t A)`.
:math:`dx` / :math:`d\\Delta` are per-channel direct writes; the grid-shared
gradients :math:`dB` / :math:`dC` (summed over channels) use
``plgpu.atomic_add`` with zero-init via ``input_output_aliases``, and
:math:`dA` / :math:`dD` (summed over batch) are emitted as per-program partials
reduced in JAX.

Scope (else :class:`PallasNotTileable` -> loud JAX fallback): NVIDIA Ampere+,
float32, sequence length divisible by the chunk size.  The sibling perf suite
owns the at-scale wall-clock-vs-reference certification (the reference's
parallel ``associative_scan`` is already the GPU throughput path).

Implementation detail: never import from ``nitrix._kernels.cuda`` directly.  Use
``nitrix.nn.ssm.selective_scan`` which handles dispatch and fallback.
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
from jaxtyping import Array, Float

__all__ = [
    'selective_scan_pallas',
    'PallasNotTileable',
]

# Chunk size for the cumsum closed form.  Small enough that exp(±A·cumsum(Δ))
# stays in float32 range over a chunk (Δ is post-softplus, typically O(<=1)).
_CHUNK = 16


class PallasNotTileable(RuntimeError):
    """The Pallas kernel rejected the requested shape / host.

    Caught by the dispatcher in ``nitrix.nn.ssm`` and translated into a
    :class:`NitrixBackendFallback` warning (the JAX reference runs instead).
    """


# ---------------------------------------------------------------------------
# Forward kernel (one (batch, channel) program; chunked cumsum scan over l)
# ---------------------------------------------------------------------------


def _fwd_kernel(
    x_ref: Any,
    delta_ref: Any,
    a_ref: Any,
    b_ref: Any,
    c_ref: Any,
    d_ref: Any,
    y_ref: Any,
    hstart_ref: Any,
    *,
    length: int,
    chunk: int,
    has_d: bool,
) -> None:
    A = a_ref[...][None, :]  # (1, n)
    nc = length // chunk

    def body(c: int, h_start: Any) -> Any:  # h_start: (1, n)
        cs = pl.ds(c * chunk, chunk)
        hstart_ref[pl.ds(c, 1), :] = h_start
        xf = x_ref[cs]  # (L,)
        df = delta_ref[cs]  # (L,)
        xc = xf[:, None]  # (L, 1)
        dc = df[:, None]  # (L, 1)
        bc = b_ref[cs, :]  # (L, n)
        cc = c_ref[cs, :]  # (L, n)
        log_p = jnp.cumsum(dc, axis=0) * A  # (L, n)
        p = jnp.exp(log_p)
        bip = (dc * xc * bc) * jnp.exp(-log_p)  # (L, n)
        h_true = p * (jnp.cumsum(bip, axis=0) + h_start)  # (L, n)
        yc = jnp.sum(cc * h_true, axis=1)  # (L,)
        if has_d:
            yc = yc + d_ref[...] * xf
        y_ref[cs] = yc
        log_p_tot = jnp.sum(dc, axis=0, keepdims=True) * A  # (1, n)
        h_next = jnp.exp(log_p_tot) * (
            jnp.sum(bip, axis=0, keepdims=True) + h_start
        )
        return h_next

    lax.fori_loop(0, nc, body, jnp.zeros_like(A))


def _mha_like_forward(
    x: Float[Array, 'b l d'],
    delta: Float[Array, 'b l d'],
    A: Float[Array, 'd n'],
    B: Float[Array, 'b l n'],
    C: Float[Array, 'b l n'],
    D: Optional[Float[Array, 'd']],
    *,
    chunk: int,
) -> tuple[Float[Array, 'b l d'], Float[Array, 'b nc d n']]:
    """Pallas driver for the fused forward scan on the ``(b, l, d)`` layout.

    Launches the forward kernel on a ``(b, d)`` grid -- one program per
    (batch, channel) pair -- and returns both the output sequence and the
    per-chunk start states needed to recompute the state trajectory in the
    backward pass.

    Parameters
    ----------
    x : Float[Array, 'b l d']
        Input sequence: ``b`` batch elements, sequence length ``l``, ``d``
        channels.
    delta : Float[Array, 'b l d']
        Per-step, per-channel discretisation step :math:`\\Delta` (already
        post-softplus), same shape as ``x``.
    A : Float[Array, 'd n']
        State transition parameter over the ``n``-dimensional state, one row
        per channel.
    B : Float[Array, 'b l n']
        Input projection into the state, shared across channels.
    C : Float[Array, 'b l n']
        Output projection from the state, shared across channels.
    D : Float[Array, 'd'] or None
        Optional per-channel skip (feed-through) term added to the output. If
        ``None`` no skip term is applied.
    chunk : int
        Length of each cumsum chunk; ``l`` must be divisible by ``chunk``.

    Returns
    -------
    y : Float[Array, 'b l d']
        Output sequence, same shape and dtype as ``x``.
    hstart : Float[Array, 'b nc d n']
        Per-chunk start states (``nc = l // chunk`` chunks), always float32.
        ``hstart[j, c, dd, :]`` is the state entering chunk ``c`` for batch
        ``j`` and channel ``dd``; consumed by the backward pass.
    """
    b, length, d = x.shape
    n = A.shape[1]
    nc = length // chunk
    has_d = D is not None
    grid = (b, d)
    seq = pl.BlockSpec((None, length, None), lambda j, dd: (j, 0, dd))
    bc_spec = pl.BlockSpec((None, length, n), lambda j, dd: (j, 0, 0))
    a_spec = pl.BlockSpec((None, n), lambda j, dd: (dd, 0))
    d_spec = None if D is None else pl.BlockSpec((None,), lambda j, dd: (dd,))
    in_specs = [seq, seq, a_spec, bc_spec, bc_spec, d_spec]
    out_specs = [
        pl.BlockSpec((None, length, None), lambda j, dd: (j, 0, dd)),  # y
        pl.BlockSpec(
            (None, nc, None, n), lambda j, dd: (j, 0, dd, 0)
        ),  # hstart -> (b, nc, d, n) as hstart[j, c, dd, :]
    ]
    out_shape = [
        jax.ShapeDtypeStruct((b, length, d), x.dtype),
        jax.ShapeDtypeStruct((b, nc, d, n), jnp.float32),
    ]
    y, hstart = pl.pallas_call(
        partial(_fwd_kernel, length=length, chunk=chunk, has_d=has_d),
        grid=grid,
        in_specs=in_specs,
        out_specs=out_specs,
        out_shape=out_shape,
        compiler_params=plgpu.CompilerParams(num_warps=4, num_stages=1),
        name='nitrix_selective_scan_fwd',
    )(x, delta, A, B, C, D)
    return (
        cast(Float[Array, 'b l d'], y),
        cast(Float[Array, 'b nc d n'], hstart),
    )


# ---------------------------------------------------------------------------
# Layout adapter: public ``(... l d)`` <-> kernel ``(b, l, d)``
# ---------------------------------------------------------------------------


def _flat(x: Array, nb: int, length: int, last: int) -> Array:
    return x.reshape(nb, length, last)


def _fused_forward(
    x: Float[Array, '... l d'],
    delta: Float[Array, '... l d'],
    A: Float[Array, 'd n'],
    B: Float[Array, '... l n'],
    C: Float[Array, '... l n'],
    D: Optional[Float[Array, 'd']],
    chunk: int,
) -> tuple[Float[Array, '... l d'], Any]:
    *batch, length, d = x.shape
    n = A.shape[1]
    nb = math.prod(batch) if batch else 1
    y, hstart = _mha_like_forward(
        _flat(x, nb, length, d),
        _flat(delta, nb, length, d),
        A,
        _flat(B, nb, length, n),
        _flat(C, nb, length, n),
        D,
        chunk=chunk,
    )
    return y.reshape(*batch, length, d), hstart


# ---------------------------------------------------------------------------
# Backward kernel (recompute-adjoint; reverse chunked cumsum, no (l, d, n) HBM)
# ---------------------------------------------------------------------------


def _bwd_kernel(
    x_ref: Any,
    delta_ref: Any,
    a_ref: Any,
    b_ref: Any,
    c_ref: Any,
    d_ref: Any,
    dy_ref: Any,
    hstart_ref: Any,
    dbz_ref: Any,  # zero-init input aliased to dB (unused directly)
    dcz_ref: Any,  # zero-init input aliased to dC (unused directly)
    dx_ref: Any,
    ddelta_ref: Any,
    da_ref: Any,
    dd_ref: Any,
    db_ref: Any,
    dc_ref: Any,
    *,
    length: int,
    chunk: int,
    has_d: bool,
) -> None:
    del dbz_ref, dcz_ref  # only their aliasing (zero-init of db/dc) is used
    A = a_ref[...][None, :]  # (1, n)
    n = a_ref.shape[0]
    nc = length // chunk
    d_val = d_ref[...] if has_d else None

    def body(i: int, carry: Any) -> Any:
        adj_carry, da_acc, dd_acc = carry  # (1, n), (n,), scalar
        c = nc - 1 - i  # reverse order
        cs = pl.ds(c * chunk, chunk)
        xf = x_ref[cs]
        df = delta_ref[cs]
        dyf = dy_ref[cs]
        bc = b_ref[cs, :]
        cc = c_ref[cs, :]
        xc = xf[:, None]
        dc = df[:, None]
        dyc = dyf[:, None]
        hstart_c = hstart_ref[pl.ds(c, 1), :]  # (1, n)

        log_p = jnp.cumsum(dc, axis=0) * A  # (L, n)
        p = jnp.exp(log_p)
        inv_p = jnp.exp(-log_p)
        tr = jnp.exp(dc * A)  # per-step transition
        bi_x = dc * xc * bc  # Δ B x
        bi_p = bi_x * inv_p
        h = p * (jnp.cumsum(bi_p, axis=0) + hstart_c)  # h_t (recomputed)
        h_prev = (h - bi_x) * jnp.exp(-dc * A)  # h_{t-1}

        # adjoint a_t = invP * (rev_cumsum(exp(logP) dy C) + exp(logP_end) carry)
        z = p * dyc * cc
        rev = jnp.sum(z, axis=0, keepdims=True) - jnp.cumsum(z, axis=0) + z
        e_end = jnp.exp(jnp.sum(dc, axis=0, keepdims=True) * A)  # (1, n)
        adj = inv_p * (rev + e_end * adj_carry)
        carry_out = jnp.sum(z, axis=0, keepdims=True) + e_end * adj_carry

        adj_b = jnp.sum(adj * bc, axis=1)  # Σ_n adj B  (L,)
        dx_chunk = adj_b * df
        if has_d:
            dx_chunk = dx_chunk + dyf * d_val
        ddelta_chunk = adj_b * xf + jnp.sum(adj * h_prev * A * tr, axis=1)
        dx_ref[cs] = dx_chunk
        ddelta_ref[cs] = ddelta_chunk
        plgpu.atomic_add(db_ref, (cs, slice(None)), adj * dc * xc)
        plgpu.atomic_add(dc_ref, (cs, slice(None)), dyc * h)
        da_acc = da_acc + jnp.sum(adj * h_prev * dc * tr, axis=0)
        dd_acc = dd_acc + jnp.sum(dyf * xf)
        return (
            carry_out.astype(jnp.float32),
            da_acc.astype(jnp.float32),
            dd_acc.astype(jnp.float32),
        )

    init = (
        jnp.zeros((1, n), jnp.float32),
        jnp.zeros((n,), jnp.float32),
        jnp.zeros((), jnp.float32),
    )
    _, da_acc, dd_acc = lax.fori_loop(0, nc, body, init)
    da_ref[...] = da_acc
    dd_ref[...] = dd_acc


def _mha_like_backward(
    x: Float[Array, 'b l d'],
    delta: Float[Array, 'b l d'],
    A: Float[Array, 'd n'],
    B: Float[Array, 'b l n'],
    C: Float[Array, 'b l n'],
    D: Optional[Float[Array, 'd']],
    dy: Float[Array, 'b l d'],
    hstart: Float[Array, 'b nc d n'],
    *,
    chunk: int,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    b, length, d = x.shape
    n = A.shape[1]
    nc = length // chunk
    has_d = D is not None
    seq = pl.BlockSpec((None, length, None), lambda j, dd: (j, 0, dd))
    bc = pl.BlockSpec((None, length, n), lambda j, dd: (j, 0, 0))
    a_spec = pl.BlockSpec((None, n), lambda j, dd: (dd, 0))
    d_spec = None if D is None else pl.BlockSpec((None,), lambda j, dd: (dd,))
    hs = pl.BlockSpec((None, nc, None, n), lambda j, dd: (j, 0, dd, 0))
    da_spec = pl.BlockSpec((None, None, n), lambda j, dd: (j, dd, 0))
    dd_spec = pl.BlockSpec((None, None), lambda j, dd: (j, dd))
    zeros_bln = jnp.zeros((b, length, n), x.dtype)
    in_specs = [seq, seq, a_spec, bc, bc, d_spec, seq, hs, bc, bc]
    out_specs = [seq, seq, da_spec, dd_spec, bc, bc]
    # A ``None`` D operand is dropped from the flattened inputs, so the
    # zeros->dB/dC alias indices shift down by one when D is absent.
    dbz_idx = 8 if has_d else 7
    aliases = {dbz_idx: 4, dbz_idx + 1: 5}
    out_shape = [
        jax.ShapeDtypeStruct((b, length, d), x.dtype),  # dx
        jax.ShapeDtypeStruct((b, length, d), x.dtype),  # ddelta
        jax.ShapeDtypeStruct(
            (b, d, n), jnp.float32
        ),  # dA partial (over batch)
        jax.ShapeDtypeStruct((b, d), jnp.float32),  # dD partial (over batch)
        jax.ShapeDtypeStruct((b, length, n), x.dtype),  # dB (atomic)
        jax.ShapeDtypeStruct((b, length, n), x.dtype),  # dC (atomic)
    ]
    dx, ddelta, da_p, dd_p, dbg, dcg = pl.pallas_call(
        partial(_bwd_kernel, length=length, chunk=chunk, has_d=has_d),
        grid=(b, d),
        in_specs=in_specs,
        out_specs=out_specs,
        out_shape=out_shape,
        input_output_aliases=aliases,  # zeros_bln inputs -> dB / dC
        compiler_params=plgpu.CompilerParams(num_warps=4, num_stages=1),
        name='nitrix_selective_scan_bwd',
    )(x, delta, A, B, C, D, dy, hstart, zeros_bln, zeros_bln)
    dA = jnp.sum(da_p, axis=0)  # (d, n)
    dD = jnp.sum(dd_p, axis=0)  # (d,)
    return dx, ddelta, dA, dbg, dcg, dD


def _fused_backward(
    x: Float[Array, '... l d'],
    delta: Float[Array, '... l d'],
    A: Float[Array, 'd n'],
    B: Float[Array, '... l n'],
    C: Float[Array, '... l n'],
    D: Optional[Float[Array, 'd']],
    hstart: Any,
    dy: Float[Array, '... l d'],
    chunk: int,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    *batch, length, d = x.shape
    n = A.shape[1]
    nb = math.prod(batch) if batch else 1
    dx, ddelta, dA, dB, dC, dD = _mha_like_backward(
        _flat(x, nb, length, d),
        _flat(delta, nb, length, d),
        A,
        _flat(B, nb, length, n),
        _flat(C, nb, length, n),
        D,
        _flat(dy, nb, length, d),
        hstart,
        chunk=chunk,
    )
    return (
        dx.reshape(*batch, length, d),
        ddelta.reshape(*batch, length, d),
        dA,
        dB.reshape(*batch, length, n),
        dC.reshape(*batch, length, n),
        dD,
    )


# ---------------------------------------------------------------------------
# Differentiable wrapper (custom_vjp; fused forward + fused backward)
# ---------------------------------------------------------------------------


@partial(jax.custom_vjp, nondiff_argnums=(6,))
def _scan_p(
    x: Array,
    delta: Array,
    A: Array,
    B: Array,
    C: Array,
    D: Optional[Array],
    chunk: int,
) -> Array:
    y, _ = _fused_forward(x, delta, A, B, C, D, chunk)
    return y


def _scan_p_fwd(
    x: Array,
    delta: Array,
    A: Array,
    B: Array,
    C: Array,
    D: Optional[Array],
    chunk: int,
) -> tuple[Array, tuple[Any, ...]]:
    y, hstart = _fused_forward(x, delta, A, B, C, D, chunk)
    return y, (x, delta, A, B, C, D, hstart)


def _scan_p_bwd(chunk: int, res: tuple[Any, ...], g: Array) -> tuple[Any, ...]:
    x, delta, A, B, C, D, hstart = res
    dx, ddelta, dA, dB, dC, dD = _fused_backward(
        x, delta, A, B, C, D, hstart, g, chunk
    )
    # D passed as None -> its cotangent is None (non-differentiable slot).
    return dx, ddelta, dA, dB, dC, (None if D is None else dD)


_scan_p.defvjp(_scan_p_fwd, _scan_p_bwd)


# ---------------------------------------------------------------------------
# Tileability gate + public entry point
# ---------------------------------------------------------------------------


def _is_pow2(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0


def _resolve_chunk(length: int) -> int:
    chunk = min(_CHUNK, length)
    if length % chunk != 0 or not _is_pow2(length // chunk):
        raise PallasNotTileable(
            f'fused selective scan requires length % {chunk} == 0 with a '
            f'power-of-two chunk count; got length={length} (pad to a friendly '
            'shape or use backend="jax").'
        )
    return chunk


def _check_tileable(
    x: Float[Array, '... l d'],
    A: Float[Array, 'd n'],
) -> None:
    # Triton tiles must have power-of-two extents; non-conforming shapes fall
    # back to the JAX reference rather than raising an uncaught Triton error.
    if x.dtype != jnp.float32:
        raise PallasNotTileable(
            f'fused selective scan currently supports float32 only; got '
            f'{x.dtype} (use backend="jax" for other dtypes).'
        )
    n = A.shape[1]
    if not _is_pow2(n):
        raise PallasNotTileable(
            f'fused selective scan requires a power-of-two state dim n; got {n}.'
        )
    _resolve_chunk(x.shape[-2])


def selective_scan_pallas(
    x: Float[Array, '... l d'],
    delta: Float[Array, '... l d'],
    A: Float[Array, 'd n'],
    B: Float[Array, '... l n'],
    C: Float[Array, '... l n'],
    D: Optional[Float[Array, 'd']] = None,
) -> Float[Array, '... l d']:
    """Fused selective scan (NVIDIA Ampere+); differentiable via ``custom_vjp``.

    Evaluates the selective state-space recurrence
    :math:`h_t = \\exp(\\Delta_t A)\\, h_{t-1} + \\Delta_t B_t x_t`,
    :math:`y_t = C_t h_t + D x_t`, using the fused chunked-cumsum Pallas kernel
    (see the module docstring). The leading batch axes are arbitrary and are
    flattened before dispatch.

    Parameters
    ----------
    x : Float[Array, '... l d']
        Input sequence with arbitrary leading batch axes, sequence length
        ``l``, and ``d`` channels.
    delta : Float[Array, '... l d']
        Per-step, per-channel discretisation step :math:`\\Delta` (already
        post-softplus), same shape as ``x``.
    A : Float[Array, 'd n']
        State transition parameter over the ``n``-dimensional state, one row
        per channel. ``n`` must be a power of two.
    B : Float[Array, '... l n']
        Input projection into the state, shared across channels.
    C : Float[Array, '... l n']
        Output projection from the state, shared across channels.
    D : Float[Array, 'd'] or None, optional
        Optional per-channel skip (feed-through) term added to the output. If
        ``None`` (default) no skip term is applied and its cotangent is
        omitted.

    Returns
    -------
    Float[Array, '... l d']
        Output sequence ``y``, same shape and dtype as ``x``.

    Raises
    ------
    PallasNotTileable
        If the shape is outside the kernel's supported set (non-float32 input,
        a state dim ``n`` that is not a power of two, or a sequence length not
        divisible by the chunk size into a power-of-two chunk count); the
        dispatcher catches this and runs the JAX reference with a loud
        fallback.
    """
    _check_tileable(x, A)
    chunk = _resolve_chunk(x.shape[-2])
    return _scan_p(x, delta, A, B, C, D, chunk)
