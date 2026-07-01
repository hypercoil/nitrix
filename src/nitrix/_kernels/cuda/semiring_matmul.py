# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Pallas Triton implementation of :func:`semiring_matmul`.

Computes :math:`C_{ij} = \\bigoplus_k \\left( A_{ik} \\otimes B_{kj} \\right)`
on NVIDIA GPUs, where :math:`\\oplus` is the semiring's additive reduction
and :math:`\\otimes` its multiplicative combine. The contraction loop over
:math:`k` carries a pytree-shaped accumulator (:class:`Monoid` state)
through ``lax.fori_loop``; each per-step iteration folds a rank-1 outer
combine into the accumulator without ever materialising the
``(BM, BK, BN)`` value tensor. This is the streaming reduction pattern in
the spirit of KeOps.

The kernel does **not** issue ``dot`` / tensor-core primitives: every
:math:`\\otimes` is a plain elementwise op on the chosen semigroup, so the
same code lowers for the :data:`REAL`, :data:`LOG`, ``TROPICAL_*``,
:data:`EUCLIDEAN`, and ``BOOLEAN`` algebras. Downstream consumers wanting
tensor-core throughput on the real semiring should call ``jnp.matmul``
directly.

This module is an implementation detail: never import from
``nitrix._kernels.cuda`` directly. Use :func:`nitrix.semiring.semiring_matmul`,
which handles backend dispatch and fallback observability.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, cast

import jax
import jax.lax as lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu
from jaxtyping import Array, Num

from ...semiring._types import Semiring

__all__ = [
    'semiring_matmul_pallas',
    'PallasNotTileable',
]


class PallasNotTileable(RuntimeError):
    """The Pallas Triton kernel rejected the requested shape.

    Raised when no viable tiling of the requested matrix shape exists.
    Caught by the public dispatcher in :mod:`nitrix.semiring` and
    translated into a :class:`~nitrix._internal.backend.NitrixBackendFallback`
    warning.
    """


# ---------------------------------------------------------------------------
# Tile selection
# ---------------------------------------------------------------------------


def _pick_blocks(m: int, k: int, n: int) -> tuple[int, int, int]:
    """Choose ``(BM, BK, BN)`` tiles that divide ``(m, k, n)``.

    For the row and column extents the largest of ``{128, 64, 32, 16}``
    that divides the extent is preferred, since the streaming kernel's
    arithmetic intensity grows with ``BM * BN`` (each :math:`\\otimes` fold
    reuses the ``(BM, 1)`` column and ``(1, BN)`` row across the entire
    tile). ``BK`` is the per-step contraction granularity, drawn from
    ``{32, 16, 8}``; choosing it too large compiles slowly, too small
    leaves accumulator-flush overhead. The defaults match an experimental
    sweet spot on Ampere (sm_80) cards.

    Parameters
    ----------
    m : int
        Number of rows of the left operand (the ``m`` extent of the
        ``(m, k)`` design).
    k : int
        Shared contraction extent (the ``k`` dimension common to both
        operands).
    n : int
        Number of columns of the right operand (the ``n`` extent of the
        ``(k, n)`` design).

    Returns
    -------
    tuple of int
        The chosen ``(BM, BK, BN)`` block sizes tiling the ``m``, ``k``,
        and ``n`` extents respectively.

    Raises
    ------
    PallasNotTileable
        If any extent admits no divisor from its candidate set, so that no
        valid tiling exists.
    """

    def largest_divisor(x: int, candidates: tuple[int, ...]) -> Optional[int]:
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
            a_col = a_ref[:, pl.ds(kk, 1)]  # (bm, 1)
            b_row = b_ref[pl.ds(kk, 1), :]  # (1, bn)
            value = binary_op.combine(a_col, b_row)  # (bm, bn)
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
    """Pallas Triton :func:`semiring_matmul` for a single 2-D ``(A, B)`` pair.

    Computes the semiring product of two dense matrices on a single
    NVIDIA GPU, contracting over the shared inner dimension. Batching is
    handled by ``vmap`` upstream in the public dispatcher; this kernel
    itself accepts only rank-2 operands.

    Parameters
    ----------
    A : Num[Array, 'm k']
        Left operand, an ``(m, k)`` matrix whose rows are contracted
        against the columns of ``B``.
    B : Num[Array, 'k n']
        Right operand, a ``(k, n)`` matrix sharing the contraction extent
        ``k`` with ``A``.
    semiring : Semiring
        The algebra defining the additive reduction and multiplicative
        combine used at each contraction step (for example :data:`REAL`,
        :data:`LOG`, or :data:`EUCLIDEAN`).

    Returns
    -------
    Num[Array, 'm n']
        The ``(m, n)`` semiring product ``C`` with
        :math:`C_{ij} = \\bigoplus_k \\left( A_{ik} \\otimes B_{kj} \\right)`,
        in the dtype of ``A``.

    Raises
    ------
    PallasNotTileable
        If either operand is not rank-2, if the contraction dimensions do
        not match, or if a viable ``(BM, BK, BN)`` tile cannot be chosen
        for the requested shape. The public dispatcher catches this and
        falls back to JAX with a
        :class:`~nitrix._internal.backend.NitrixBackendFallback` warning.
    """
    if A.ndim != 2 or B.ndim != 2:
        raise PallasNotTileable(
            'semiring_matmul_pallas only handles 2-D inputs; '
            'higher rank should be vmapped from the dispatcher.'
        )
    m, k = A.shape
    k2, n = B.shape
    if k != k2:
        raise PallasNotTileable(
            f'semiring_matmul_pallas: contraction dim mismatch k={k} != {k2}.'
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
