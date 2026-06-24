# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Pure-JAX reference for the Mamba / S6 selective state-space scan.

The discretised selective recurrence (per channel ``d`` and state ``n``):

    h_t = exp(Δ_t · A) ⊙ h_{t-1} + (Δ_t · B_t) ⊙ x_t      # state update
    y_t = Σ_n C_{t,n} · h_{t,d,n}  (+ D_d ⊙ x_t)           # readout

This is a first-order linear recurrence ``h_t = a_t ⊙ h_{t-1} + b_t`` with
``a_t = exp(Δ_t A)`` and ``b_t = Δ_t B_t x_t``, so it admits both a sequential
``lax.scan`` form (the bit-exact oracle) and a parallel
``lax.associative_scan`` form (combinator
``(a₁,b₁)∘(a₂,b₂) = (a₁a₂, a₂b₁+b₂)``).  ``driver='auto'`` picks the parallel
prefix scan on GPU (``O(log L)`` depth) and the sequential scan on CPU -- the
same platform-dependent choice as ``signal._iir`` -- so the GPU gets the
work-parallel speedup before any fused kernel exists.

Differentiated by ordinary XLA autodiff (no hand-written VJP -- that lives on
the fused path).  See ``docs/feature-requests/nn-forward-kernels-suite.md`` §7.2.
"""

from __future__ import annotations

from typing import Literal, Optional, cast

import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Float

from ..._internal.backend import default_backend_is_gpu
from ..._internal.config import resolve_driver

__all__ = ['reference_selective_scan']

Driver = Literal['auto', 'sequential', 'associative', 'chunked']

# The 'nn.ssm.selective_scan' divergent-op contract is registered centrally in
# nitrix._internal._divergent_ops; this module only resolves against it.


def _lin_combine(
    e1: tuple[Array, Array], e2: tuple[Array, Array]
) -> tuple[Array, Array]:
    # Linear-recurrence prefix combinator: (a,b) o (a',b') = (a a', a' b + b').
    a1, b1 = e1
    a2, b2 = e2
    return a1 * a2, a2 * b1 + b2


def _discretize(
    x: Float[Array, '... l d'],
    delta: Float[Array, '... l d'],
    A: Float[Array, 'd n'],
    B: Float[Array, '... l n'],
) -> tuple[Float[Array, '... l d n'], Float[Array, '... l d n']]:
    # dA_{t,d,n} = exp(Δ_{t,d} A_{d,n});  dBx_{t,d,n} = Δ_{t,d} B_{t,n} x_{t,d}
    dA = jnp.exp(delta[..., None] * A)
    dBx = delta[..., None] * B[..., None, :] * x[..., None]
    return dA, dBx


def _readout(
    H: Float[Array, '... l d n'],
    C: Float[Array, '... l n'],
    x: Float[Array, '... l d'],
    D: Optional[Float[Array, 'd']],
) -> Float[Array, '... l d']:
    y = (H * C[..., None, :]).sum(axis=-1)
    if D is not None:
        y = y + D * x
    return y


def _scan_sequential(
    dA: Float[Array, '... l d n'], dBx: Float[Array, '... l d n']
) -> Float[Array, '... l d n']:
    dA_l = jnp.moveaxis(dA, -3, 0)  # (l, ..., d, n)
    dBx_l = jnp.moveaxis(dBx, -3, 0)
    h0 = jnp.zeros(dA.shape[:-3] + dA.shape[-2:], dtype=dA.dtype)

    def step(h: Array, ab: tuple[Array, Array]) -> tuple[Array, Array]:
        a, b = ab
        h = a * h + b
        return h, h

    _, h_l = lax.scan(step, h0, (dA_l, dBx_l))
    return jnp.moveaxis(h_l, 0, -3)


def _scan_associative(
    dA: Float[Array, '... l d n'], dBx: Float[Array, '... l d n']
) -> Float[Array, '... l d n']:
    _, h = lax.associative_scan(_lin_combine, (dA, dBx), axis=-3)
    return cast(Float[Array, '... l d n'], h)


def _chunk_front(arr: Array, nc: int, chunk: int, last: int) -> Array:
    # (..., l, last) -> (nc, ..., chunk, last)  [nc moved to the scan axis]
    batch = arr.shape[:-2]
    return jnp.moveaxis(arr.reshape(*batch, nc, chunk, last), -3, 0)


def _scan_chunked(
    x: Float[Array, '... l d'],
    delta: Float[Array, '... l d'],
    A: Float[Array, 'd n'],
    B: Float[Array, '... l n'],
    C: Float[Array, '... l n'],
    D: Optional[Float[Array, 'd']],
    chunk: int,
) -> Float[Array, '... l d']:
    """Memory-sparing pure-XLA scan: chunked ``lax.scan`` (carry = (d, n) state),
    parallel ``associative_scan`` within each chunk, ``n`` collapsed into ``y``
    inside the body so the ``(l, d, n)`` trajectory is never materialised; the
    body is rematerialised (``jax.checkpoint``) so the backward stores only the
    per-chunk carry.  Same math as the other methods (XLA-stable scan, no fp32
    range limit)."""
    *batch, length, d = x.shape
    n = A.shape[1]
    nc = length // chunk
    acc_dtype = jnp.result_type(x, delta, A, B)
    xs = (
        _chunk_front(x, nc, chunk, d),
        _chunk_front(delta, nc, chunk, d),
        _chunk_front(B, nc, chunk, n),
        _chunk_front(C, nc, chunk, n),
    )
    h0 = jnp.zeros((*batch, d, n), dtype=acc_dtype)

    def body(
        h_carry: Array, blk: tuple[Array, Array, Array, Array]
    ) -> tuple[Array, Array]:
        x_c, delta_c, b_c, c_c = blk  # (..., chunk, d/d/n/n)
        dA_c = jnp.exp(delta_c[..., None] * A)  # (..., chunk, d, n)
        dBx_c = delta_c[..., None] * b_c[..., None, :] * x_c[..., None]
        cum_a, h_local = lax.associative_scan(
            _lin_combine, (dA_c, dBx_c), axis=-3
        )
        # Propagate the chunk-entry state: h_t = h_local_t + (Π_{j<=t} dA_j) h_in.
        h = h_local + cum_a * jnp.expand_dims(h_carry, axis=-3)
        y_c = (h * c_c[..., None, :]).sum(axis=-1)  # collapse n
        if D is not None:
            y_c = y_c + D * x_c
        return h[..., -1, :, :], y_c

    _, y_blocks = lax.scan(jax.checkpoint(body), h0, xs)
    # y_blocks: (nc, ..., chunk, d) -> (..., l, d)
    y = jnp.moveaxis(y_blocks, 0, -3)
    return y.reshape(*batch, length, d)


def reference_selective_scan(
    x: Float[Array, '... l d'],
    delta: Float[Array, '... l d'],
    A: Float[Array, 'd n'],
    B: Float[Array, '... l n'],
    C: Float[Array, '... l n'],
    D: Optional[Float[Array, 'd']] = None,
    *,
    driver: Driver = 'auto',
    chunk_size: int = 64,
) -> Float[Array, '... l d']:
    """Reference Mamba / S6 selective scan (oracle).

    Parameters
    ----------
    x, delta
        Input sequence and per-step / per-channel ``Δ`` (already post-softplus),
        both ``(..., l, d)``.
    A
        State matrix in diagonal-plus form, ``(d, n)`` (typically negative for a
        contractive recurrence).
    B, C
        Selective input / output projections, ``(..., l, n)``.
    D
        Optional per-channel skip / residual, ``(d,)``.
    driver
        Numerical variant (the ``driver`` axis).  ``'sequential'`` (the
        bit-exact ``lax.scan`` oracle -- the canonical variant), ``'associative'``
        (parallel prefix, materialises the ``(l, d, n)`` trajectory),
        ``'chunked'`` (pure-XLA memory-sparing: chunked scan that never
        materialises ``(l, d, n)`` -- the Pallas-free analogue of the fused
        kernel), or ``'auto'`` (parallel on GPU, sequential on CPU).
        ``nitrix.reproducible()`` forces the canonical ``'sequential'``; the
        variants agree only to a documented tolerance
        (``nitrix.divergent_ops()``), not bit-for-bit.
    chunk_size
        Chunk length for ``method='chunked'`` (must divide ``l``); the
        memory / parallel-depth trade-off knob.  Size-based dispatch is left to
        the perf suite.

    Returns
    -------
    Output sequence ``(..., l, d)``.

    Notes
    -----
    All methods agree up to floating-point reassociation.  ``'chunked'`` keeps
    peak memory at ``O(chunk·d·n + l·d)`` instead of ``O(l·d·n)`` and, being
    XLA-stable, has none of the fused kernel's fp32 within-chunk range limit.
    Degenerate ``A -> 0`` (``dA -> 1``) reduces the update to a cumulative sum of
    ``Δ_t B_t x_t``.  A float16/bfloat16 input is upcast to float32 for the
    recurrence and readout and cast back at the end (the fp32-accumulation
    invariant, SPEC §2 tenet 11); float32 / float64 inputs are unchanged.
    """
    resolved = resolve_driver(
        driver,
        op='nn.ssm.selective_scan',
        fast=lambda: (
            'associative' if default_backend_is_gpu() else 'sequential'
        ),
    )
    # fp32-accumulation invariant (SPEC §2 tenet 11): the first-order recurrence
    # accumulates in >= float32 regardless of the I/O dtype, so a bf16/fp16 scan
    # does not reassociate the state update in reduced precision.  Upcast at the
    # boundary, run the existing machinery, cast back -- a no-op for float32 /
    # float64 inputs (acc_dtype == io_dtype), byte-identical to before.
    io_dtype = (
        jnp.result_type(x, delta, A, B, C)
        if D is None
        else jnp.result_type(x, delta, A, B, C, D)
    )
    acc_dtype = jnp.promote_types(io_dtype, jnp.float32)
    if acc_dtype != io_dtype:
        x = x.astype(acc_dtype)
        delta = delta.astype(acc_dtype)
        A = A.astype(acc_dtype)
        B = B.astype(acc_dtype)
        C = C.astype(acc_dtype)
        if D is not None:
            D = D.astype(acc_dtype)
    if resolved == 'chunked':
        chunk = min(chunk_size, x.shape[-2])
        if x.shape[-2] % chunk != 0:
            raise ValueError(
                f"driver='chunked' requires l % chunk_size == 0; got "
                f'l={x.shape[-2]}, chunk_size={chunk_size}.'
            )
        out = _scan_chunked(x, delta, A, B, C, D, chunk)
    else:
        dA, dBx = _discretize(x, delta, A, B)
        if resolved == 'associative':
            h = _scan_associative(dA, dBx)
        else:  # 'sequential' (resolve_driver guarantees a registered value)
            h = _scan_sequential(dA, dBx)
        out = _readout(h, C, x, D)
    return out if out.dtype == io_dtype else out.astype(io_dtype)
