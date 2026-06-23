# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Pallas Triton fused selective state-space scan (Mamba / S6) -- forward + backward.

Clean-room kernel.  Triton-Pallas lowers neither ``cumprod`` /
``associative_scan`` nor element / ``slice`` indexing of register tiles -- only
whole-tile ops and ``cumsum``.  The linear recurrence
``h_t = exp(Δ_t A) h_{t-1} + Δ_t B_t x_t`` is therefore evaluated as a **chunked
cumsum closed form**: with ``logP_t = A · cumsum(Δ)_t`` over a chunk,
``h_t = exp(logP_t) · (cumsum(Δ_i B_i x_i · exp(-logP_i)) + h_start)``.  The
chunk size keeps ``exp(±logP)`` bounded; the chunk-final state is carried to the
next chunk via whole-tile sums (no indexing).  The ``(l, d, n)`` state trajectory
is never materialised in HBM -- only ``y (l, d)`` and the per-chunk start states.

Backward (current): a ``custom_vjp`` whose backward recomputes the gradient
through the JAX reference -- correct ``dx/ddelta/dA/dB/dC/dD`` for all input
ranges.  The forward already delivers the inference activation-memory win (no
``(l, d, n)`` in HBM); the forward also emits the per-chunk start states the
fully-fused backward needs.  The fully-fused recompute-adjoint backward (reverse
chunked cumsum ``a_t = dy_t C_t + dA_{t+1} a_{t+1}`` via
``rev_cumsum(z) = sum(z) - cumsum(z) + z``; ``h_{t-1}`` recovered as
``(h_t - Δ_t B_t x_t) / exp(Δ_t A)``; shared-grad reductions via
``plgpu.atomic_add``) is the next increment for the *training*-memory win.

Scope (else ``PallasNotTileable`` -> loud JAX fallback): NVIDIA Ampere+, float32,
sequence length divisible by the chunk size.  The sibling perf suite owns the
at-scale wall-clock-vs-reference certification (the reference's parallel
``associative_scan`` is already the GPU throughput path).

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
    ``NitrixBackendFallback`` warning (the JAX reference runs instead).
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
    """Pallas driver; returns (y, chunk-start states) on the (b, l, d) layout."""
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
            (None, None, nc, n), lambda j, dd: (j, dd, 0, 0)
        ),  # hstart
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
# Differentiable wrapper (custom_vjp; fused forward, reference backward for now)
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
    from ...nn.ssm._reference import reference_selective_scan as _ref

    x, delta, A, B, C, D, _hstart = res
    if D is None:

        def f_nd(
            x: Array, delta: Array, A: Array, B: Array, C: Array
        ) -> Array:
            return _ref(x, delta, A, B, C, method='sequential')

        _, vjp = jax.vjp(f_nd, x, delta, A, B, C)
        dx, dd, dA, dB, dC = vjp(g)
        return dx, dd, dA, dB, dC, None

    def f_d(
        x: Array, delta: Array, A: Array, B: Array, C: Array, D: Array
    ) -> Array:
        return _ref(x, delta, A, B, C, D, method='sequential')

    _, vjp = jax.vjp(f_d, x, delta, A, B, C, D)
    return tuple(vjp(g))


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

    Raises
    ------
    PallasNotTileable
        If the shape is outside the kernel's supported set; the dispatcher
        catches this and runs the JAX reference with a loud fallback.
    """
    _check_tileable(x, A)
    chunk = _resolve_chunk(x.shape[-2])
    return _scan_p(x, delta, A, B, C, D, chunk)
