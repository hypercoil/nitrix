# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Public ``semiring_matmul`` with three-level backend selection.

The function shape matches SPEC §3.1::

    semiring_matmul(
        A: Num[Array, "... m k"],
        B: Num[Array, "... k n"],
        *,
        semiring: Semiring = REAL,
        backend: Backend = "auto",
    ) -> Num[Array, "... m n"]

Leading batch dims are broadcast-compatible.  The 2-D core is dispatched
to either ``reference_semiring_matmul`` (pure JAX) or
``_kernels.cuda.semiring_matmul.semiring_matmul_pallas`` (Pallas /
Triton).  If the Pallas path cannot tile a given shape × algebra
combination, the call falls back to JAX and emits exactly one
``NitrixBackendFallback`` warning per ``(function, shape, dtype,
backend)`` per process; see ``_internal.backend`` for the env-var knobs.
"""
from __future__ import annotations

from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array, Num

from .._internal.backend import (
    Backend,
    ResolvedBackend,
    fallback,
    resolve_backend,
)
from ._reference import reference_semiring_matmul
from ._types import Semiring
from .algebras import REAL


__all__ = [
    'semiring_matmul',
    'reference_semiring_matmul',
]


def _check_2d_compat(A_shape, B_shape, name: str) -> tuple[int, int, int]:
    if len(A_shape) < 2 or len(B_shape) < 2:
        raise ValueError(
            f'{name}: A and B must have at least 2 dimensions; got '
            f'A.shape={A_shape}, B.shape={B_shape}.'
        )
    m, k = A_shape[-2], A_shape[-1]
    k2, n = B_shape[-2], B_shape[-1]
    if k != k2:
        raise ValueError(
            f'{name}: contraction dim mismatch -- A has k={k}, B has '
            f'k={k2} (A.shape={A_shape}, B.shape={B_shape}).'
        )
    return int(m), int(k), int(n)


def _broadcast_batch(A, B):
    '''Broadcast leading batch dims of ``A`` and ``B`` together.

    Returns the broadcasted operands; the trailing two dims are
    unchanged.  We avoid materialising broadcast copies where possible
    by relying on ``jnp.broadcast_to``.
    '''
    A_batch = A.shape[:-2]
    B_batch = B.shape[:-2]
    out_batch = jnp.broadcast_shapes(A_batch, B_batch)
    A_b = jnp.broadcast_to(A, out_batch + A.shape[-2:])
    B_b = jnp.broadcast_to(B, out_batch + B.shape[-2:])
    return A_b, B_b, out_batch


def _run_jax_core(A2, B2, *, semiring: Semiring) -> Array:
    '''2-D JAX-reference core, expected to be ``vmap``-ed over batch dims.'''
    return reference_semiring_matmul(A2, B2, semiring=semiring)


def _vmap_over_batch(fn, n_batch_dims: int):
    out = fn
    for _ in range(n_batch_dims):
        out = jax.vmap(out, in_axes=(0, 0))
    return out


def _semiring_matmul_jax(
    A: Array, B: Array, *, semiring: Semiring
) -> Array:
    Ab, Bb, batch = _broadcast_batch(A, B)
    core = partial(_run_jax_core, semiring=semiring)
    return _vmap_over_batch(core, len(batch))(Ab, Bb)


def _semiring_matmul_pallas(
    A: Array, B: Array, *, semiring: Semiring
) -> Optional[Array]:
    '''Pallas dispatch; returns ``None`` if the kernel rejects the shape.

    Import-on-call so that a Pallas-broken JAX install can still use
    the JAX fallback.  Catches only ``PallasNotTileable`` -- a real
    kernel failure (an unexpected lowering error, a CUDA error) is not
    silently swallowed.
    '''
    try:
        from .._kernels.cuda.semiring_matmul import (
            PallasNotTileable,
            semiring_matmul_pallas,
        )
    except Exception:
        return None
    Ab, Bb, batch = _broadcast_batch(A, B)
    try:
        core = partial(semiring_matmul_pallas, semiring=semiring)
        return _vmap_over_batch(core, len(batch))(Ab, Bb)
    except PallasNotTileable:
        return None


def semiring_matmul(
    A: Num[Array, '... m k'],
    B: Num[Array, '... k n'],
    *,
    semiring: Semiring = REAL,
    backend: Backend = 'auto',
) -> Num[Array, '... m n']:
    '''Semiring-generalised matrix multiplication.

    Computes ``C[..., i, j] = (+)_k ( A[..., i, k] (*) B[..., k, j] )``
    under the supplied ``semiring``.

    Parameters
    ----------
    A
        Left operand; trailing two dims are ``(m, k)``.
    B
        Right operand; trailing two dims are ``(k, n)``.  Leading batch
        dims of ``A`` and ``B`` are broadcast-compatible.
    semiring
        Algebra (``Semiring`` or ``StrictSemiring``) defining the
        reduction and combine.  Defaults to ``REAL``.
    backend
        ``"auto"``, ``"pallas-cuda"``, or ``"jax"``.  See
        ``nitrix._internal.backend`` for the resolution rules and
        env-var overrides.

    Returns
    -------
    Array of shape ``(*broadcast_batch, m, n)``.

    Notes
    -----
    For ``REAL`` on real-valued inputs, the same answer is produced as
    ``A @ B``; downstream consumers wanting tensor-core throughput on
    the real semiring should call ``jnp.matmul`` directly.  ``nitrix``'s
    advantage is the *other* algebras and the streaming kernel for
    log / tropical / euclidean.
    '''
    _check_2d_compat(A.shape, B.shape, name='semiring_matmul')

    resolved: ResolvedBackend = resolve_backend(backend)

    if resolved == 'pallas-cuda':
        out = _semiring_matmul_pallas(A, B, semiring=semiring)
        if out is None:
            resolved = fallback(
                function='semiring_matmul',
                requested='pallas-cuda',
                resolved='jax',
                reason=(
                    f'algebra={semiring.name!r}: Pallas Triton cannot '
                    'tile the requested (M, K, N) for the chosen block '
                    'sizes, or the kernel is unavailable on this host'
                ),
                shapes=(tuple(A.shape), tuple(B.shape)),
                dtype=A.dtype,
            )
        else:
            return out

    return _semiring_matmul_jax(A, B, semiring=semiring)
