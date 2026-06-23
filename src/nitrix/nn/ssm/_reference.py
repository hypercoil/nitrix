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
``(a₁,b₁)∘(a₂,b₂) = (a₁a₂, a₂b₁+b₂)``).  ``method='auto'`` picks the parallel
prefix scan on GPU (``O(log L)`` depth) and the sequential scan on CPU -- the
same platform-dependent choice as ``signal._iir`` -- so the GPU gets the
work-parallel speedup before any fused kernel exists.

Differentiated by ordinary XLA autodiff (no hand-written VJP -- that lives on
the fused path).  See ``docs/feature-requests/nn-forward-kernels-suite.md`` §7.2.
"""

from __future__ import annotations

from typing import Literal, Optional, cast

import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Float

from ..._internal.backend import default_backend_is_gpu

__all__ = ['reference_selective_scan']

Method = Literal['auto', 'sequential', 'associative']


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
    def combine(
        e1: tuple[Array, Array], e2: tuple[Array, Array]
    ) -> tuple[Array, Array]:
        a1, b1 = e1
        a2, b2 = e2
        return a1 * a2, a2 * b1 + b2

    _, h = lax.associative_scan(combine, (dA, dBx), axis=-3)
    return cast(Float[Array, '... l d n'], h)


def reference_selective_scan(
    x: Float[Array, '... l d'],
    delta: Float[Array, '... l d'],
    A: Float[Array, 'd n'],
    B: Float[Array, '... l n'],
    C: Float[Array, '... l n'],
    D: Optional[Float[Array, 'd']] = None,
    *,
    method: Method = 'auto',
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
    method
        ``'sequential'`` (the bit-exact ``lax.scan`` oracle), ``'associative'``
        (parallel prefix), or ``'auto'`` (parallel on GPU, sequential on CPU).

    Returns
    -------
    Output sequence ``(..., l, d)``.

    Notes
    -----
    The two scan forms agree up to floating-point reassociation; ``'auto'`` is
    the public default and gives the GPU the ``O(log L)`` parallel speedup.
    Degenerate ``A -> 0`` (``dA -> 1``) reduces the state update to a cumulative
    sum of ``Δ_t B_t x_t``.
    """
    dA, dBx = _discretize(x, delta, A, B)
    resolved = method
    if resolved == 'auto':
        resolved = 'associative' if default_backend_is_gpu() else 'sequential'
    if resolved == 'sequential':
        h = _scan_sequential(dA, dBx)
    elif resolved == 'associative':
        h = _scan_associative(dA, dBx)
    else:
        raise ValueError(
            f"method must be 'auto'/'sequential'/'associative'; got {method!r}."
        )
    return _readout(h, C, x, D)
