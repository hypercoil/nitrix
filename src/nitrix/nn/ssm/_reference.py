# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Pure-JAX reference for the Mamba / S6 selective state-space scan.

The discretised selective recurrence (per channel :math:`d` and state
:math:`n`) is a state update followed by a readout:

.. math::

    h_t = \\exp(\\Delta_t \\cdot A) \\odot h_{t-1}
          + (\\Delta_t \\cdot B_t) \\odot x_t

    y_t = \\sum_n C_{t,n} \\cdot h_{t,d,n} \\; (+ D_d \\odot x_t)

This is a first-order linear recurrence
:math:`h_t = a_t \\odot h_{t-1} + b_t` with
:math:`a_t = \\exp(\\Delta_t A)` and :math:`b_t = \\Delta_t B_t x_t`, so it
admits both a sequential :func:`jax.lax.scan` form (the bit-exact oracle) and
a parallel :func:`jax.lax.associative_scan` form with combinator
:math:`(a_1, b_1) \\circ (a_2, b_2) = (a_1 a_2, \\; a_2 b_1 + b_2)`.
``driver='auto'`` picks the parallel prefix scan on GPU (:math:`O(\\log L)`
depth) and the sequential scan on CPU, so the GPU gets the work-parallel
speed-up before any fused kernel exists.

Differentiated by ordinary XLA autodiff (no hand-written VJP; that lives on
the fused path).
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
    """Memory-sparing pure-XLA selective scan.

    The sequence is split into chunks and processed by an outer
    :func:`jax.lax.scan` whose carry is the ``(d, n)`` state at the chunk
    boundary.  Within each chunk the recurrence is run as a parallel
    :func:`jax.lax.associative_scan`, then the state axis :math:`n` is collapsed
    into the output inside the scan body so the full ``(l, d, n)`` trajectory is
    never materialised.  The body is rematerialised with :func:`jax.checkpoint`,
    so the backward pass stores only the per-chunk carry.  The mathematics is
    identical to the sequential and associative variants (an XLA-stable scan with
    no within-chunk float32 range limit).

    Parameters
    ----------
    x : Float[Array, '... l d']
        Input sequence of length ``l`` with ``d`` channels.
    delta : Float[Array, '... l d']
        Per-step, per-channel step size :math:`\\Delta` (already post-softplus),
        same shape as ``x``.
    A : Float[Array, 'd n']
        Diagonal-plus state matrix, one ``n``-vector of state coefficients per
        channel.
    B : Float[Array, '... l n']
        Selective input projection onto the ``n`` state dimensions.
    C : Float[Array, '... l n']
        Selective output projection from the ``n`` state dimensions.
    D : Float[Array, 'd'] or None
        Optional per-channel skip / residual connection; omitted when ``None``.
    chunk : int
        Chunk length along the sequence axis; must divide ``l``.  Trades peak
        memory and parallel depth against per-chunk work.

    Returns
    -------
    Float[Array, '... l d']
        Output sequence with the same batch, length and channel shape as ``x``.
    """
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
    x, delta : Float[Array, '... l d']
        Input sequence and per-step, per-channel step size :math:`\\Delta`
        (already post-softplus), both ``(..., l, d)``.
    A : Float[Array, 'd n']
        State matrix in diagonal-plus form, ``(d, n)`` (typically negative for a
        contractive recurrence).
    B, C : Float[Array, '... l n']
        Selective input and output projections, both ``(..., l, n)``.
    D : Float[Array, 'd'] or None, optional
        Optional per-channel skip / residual, ``(d,)``.  Omitted when ``None``.
    driver : {'auto', 'sequential', 'associative', 'chunked'}, optional
        Numerical variant. ``'sequential'`` is the bit-exact
        :func:`jax.lax.scan` oracle and the canonical variant; ``'associative'``
        is the parallel prefix scan, which materialises the ``(l, d, n)``
        trajectory; ``'chunked'`` is a pure-XLA memory-sparing chunked scan that
        never materialises ``(l, d, n)`` (the Pallas-free analogue of the fused
        kernel); and ``'auto'`` selects the parallel scan on GPU and the
        sequential scan on CPU.  ``nitrix.reproducible()`` forces the canonical
        ``'sequential'``; the variants agree only to a documented tolerance (see
        ``nitrix.divergent_ops()``), not bit-for-bit.
    chunk_size : int, optional
        Chunk length used when ``driver='chunked'`` (must divide ``l``); the
        memory / parallel-depth trade-off knob.  Size-based dispatch is left to
        the performance suite.

    Returns
    -------
    Float[Array, '... l d']
        Output sequence ``(..., l, d)``.

    Notes
    -----
    All variants agree up to floating-point reassociation.  ``'chunked'`` keeps
    peak memory at :math:`O(\\mathrm{chunk} \\cdot d \\cdot n + l \\cdot d)`
    instead of :math:`O(l \\cdot d \\cdot n)` and, being XLA-stable, has none of
    the fused kernel's float32 within-chunk range limit.  A degenerate
    :math:`A \\to 0` (so :math:`\\mathrm{dA} \\to 1`) reduces the update to a
    cumulative sum of :math:`\\Delta_t B_t x_t`.  A float16 / bfloat16 input is
    upcast to float32 for the recurrence and readout and cast back at the end
    (the float32-accumulation invariant); float32 / float64 inputs are unchanged.
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
