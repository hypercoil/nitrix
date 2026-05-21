# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Pallas Triton implementation of ``semiring_matmul``.

Computes ``C[i, j] = (+)_k (A[i, k] (*) B[k, j])`` on NVIDIA GPUs.  The
K loop carries a pytree-shaped accumulator (``Monoid`` state) through
``lax.fori_loop``; the inner per-K-step folds a rank-1 outer combine
into the accumulator without ever materialising the ``(BM, BK, BN)``
value tensor.  This is the KeOps-style streaming pattern from SPEC §3.1.

The kernel does **not** issue ``dot``/tensor-core primitives: every
``(*)`` is a plain elementwise op on the chosen semigroup, so the same
code lowers for ``REAL``, ``LOG``, ``TROPICAL_*``, ``EUCLIDEAN``, and
``BOOLEAN``.  Downstream consumers wanting tensor-core throughput on
the real semiring should call ``jnp.matmul`` directly.

This module is an implementation detail: never import from
``nitrix._kernels.cuda`` directly.  Use ``nitrix.semiring.semiring_matmul``
which handles backend dispatch and fallback observability.
"""
from __future__ import annotations

from typing import Any, Callable, Optional, cast

import jax
import jax.lax as lax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu
from jaxtyping import Array, Num

from ...semiring._types import Semiring


__all__ = [
    'semiring_matmul_pallas',
    'PallasNotTileable',
]


class PallasNotTileable(RuntimeError):
    '''The Pallas Triton kernel rejected the requested shape.

    Caught by the public dispatcher in ``nitrix.semiring.matmul`` and
    translated into a ``NitrixBackendFallback`` warning.
    '''


# ---------------------------------------------------------------------------
# Tile selection
# ---------------------------------------------------------------------------


def _pick_blocks(m: int, k: int, n: int) -> tuple[int, int, int]:
    '''Choose ``(BM, BK, BN)`` tiles that divide ``(m, k, n)``.

    We prefer the largest of ``{128, 64, 32, 16}`` that divides each
    extent, since the streaming kernel's arithmetic intensity grows
    with ``BM * BN`` (each ``(*)`` fold reuses the (BM, 1) and (1, BN)
    column / row across the entire tile).  ``BK`` is the per-step
    granularity; choosing it too large compiles slowly, too small
    leaves accumulator-flush overhead.  The defaults match the
    refstubs experimental sweet spot on Ampere (sm_80) cards.
    '''
    def largest_divisor(
        x: int, candidates: tuple[int, ...]
    ) -> Optional[int]:
        for c in candidates:
            if x % c == 0:
                return c
        return None
    bm = largest_divisor(m, (128, 64, 32, 16))
    bn = largest_divisor(n, (128, 64, 32, 16))
    bk = largest_divisor(k, (32, 16, 8))
    if bm is None or bn is None or bk is None:
        raise PallasNotTileable(
            f'cannot find tile sizes dividing (m={m}, k={k}, n={n}); '
            'pad to a friendly shape or use backend="jax".'
        )
    return bm, bk, bn


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------


def _build_kernel(
    semiring: Semiring[Any], bm: int, bk: int, bn: int, k: int
) -> Callable[..., None]:
    monoid = semiring.monoid
    binary_op = semiring.binary_op

    def kernel(a_ref: Any, b_ref: Any, o_ref: Any) -> None:
        # a_ref: (bm, k); b_ref: (k, bn); o_ref: (bm, bn)
        acc_init = monoid.init((bm, bn), o_ref.dtype)

        def body(kk: int, acc: Any) -> Any:
            # Slice 1-wide stripes off the *refs* (Triton supports pl.ds on
            # refs but not lax.dynamic_slice on intermediate arrays).
            a_col = a_ref[:, pl.ds(kk, 1)]                       # (bm, 1)
            b_row = b_ref[pl.ds(kk, 1), :]                       # (1, bn)
            value = binary_op.combine(a_col, b_row)              # (bm, bn)
            return monoid.update(acc, value)

        acc = lax.fori_loop(0, k, body, acc_init)
        o_ref[...] = monoid.finalize(acc)

    return kernel


def semiring_matmul_pallas(
    A: Num[Array, 'm k'],
    B: Num[Array, 'k n'],
    *,
    semiring: Semiring[Any],
) -> Num[Array, 'm n']:
    '''Pallas Triton ``semiring_matmul`` for a single 2-D ``(A, B)`` pair.

    Batching is handled by ``vmap`` upstream in
    ``nitrix.semiring.matmul``.

    Raises
    ------
    PallasNotTileable
        If a viable ``(BM, BK, BN)`` tile cannot be chosen for the
        requested shape.  The public dispatcher catches this and falls
        back to JAX with a ``NitrixBackendFallback`` warning.
    '''
    if A.ndim != 2 or B.ndim != 2:
        raise PallasNotTileable(
            'semiring_matmul_pallas only handles 2-D inputs; '
            'higher rank should be vmapped from the dispatcher.'
        )
    m, k = A.shape
    k2, n = B.shape
    if k != k2:
        raise PallasNotTileable(
            f'semiring_matmul_pallas: contraction dim mismatch '
            f'k={k} != {k2}.'
        )

    bm, bk, bn = _pick_blocks(int(m), int(k), int(n))
    kernel = _build_kernel(semiring, bm, bk, bn, int(k))

    # ``pl.pallas_call`` is untyped (returns Any); restore the array type.
    return cast(
        Num[Array, 'm n'],
        pl.pallas_call(
            kernel,
            grid=(m // bm, n // bn),
            in_specs=[
                pl.BlockSpec((bm, k), lambda i, j: (i, 0)),
                pl.BlockSpec((k, bn), lambda i, j: (0, j)),
            ],
            out_specs=pl.BlockSpec((bm, bn), lambda i, j: (i, j)),
            out_shape=jax.ShapeDtypeStruct((m, n), A.dtype),
            compiler_params=plgpu.CompilerParams(),
            name=f'semiring_matmul_{semiring.name}',
        )(A, B),
    )
