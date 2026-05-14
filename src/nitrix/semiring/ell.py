# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Public ``semiring_ell_matmul`` with three-level backend selection.

The function shape matches SPEC §3.1::

    semiring_ell_matmul(
        values: Num[Array, "m k_max"],
        indices: Int[Array, "m k_max"],
        B: Num[Array, "n_cols ncol"],
        *,
        semiring: Semiring = REAL,
        n_cols: int | None = None,
        backend: Backend = "auto",
    ) -> Num[Array, "m ncol"]

The ELL row's pad positions are expected to be filled with the algebra's
identity (e.g., ``0`` for ``REAL``, ``-inf`` for ``LOG``).  See
``nitrix.sparse.ell.ell_pad`` for the helper.

Per SPEC_UPDATE_v0.2 §4, ``semiring_ell_matmul`` is the central op for
brain-geometry workloads and is the load-bearing kernel behind the G0
gate.  At first GA we ship the JAX path unconditionally and the Pallas
path opt-in (pending the Ampere benchmark in ``bench/g0_ampere_ell.py``).
"""
from __future__ import annotations

from functools import partial
from typing import Optional

import jax
from jaxtyping import Array, Int, Num

from .._internal.backend import (
    Backend,
    ResolvedBackend,
    fallback,
    resolve_backend,
)
from ._reference import reference_semiring_ell_matmul
from ._types import Semiring
from .algebras import REAL


__all__ = [
    'semiring_ell_matmul',
    'reference_semiring_ell_matmul',
]


def _check_ell_shapes(values_shape, indices_shape, B_shape, name: str):
    if len(values_shape) < 2:
        raise ValueError(
            f'{name}: values must be at least 2-D '
            f'(got values.shape={values_shape}).'
        )
    if values_shape != indices_shape:
        raise ValueError(
            f'{name}: values.shape={values_shape} must equal '
            f'indices.shape={indices_shape}.'
        )
    if len(B_shape) < 2:
        raise ValueError(
            f'{name}: B must be at least 2-D (got B.shape={B_shape}).'
        )


def _semiring_ell_matmul_pallas(values, indices, B, *, semiring):
    '''Pallas dispatch; returns ``None`` if the kernel rejects the request.'''
    try:
        from .._kernels.cuda.semiring_ell_matmul import (
            semiring_ell_matmul_pallas,
            PallasELLNotTileable,
        )
    except Exception:
        return None
    try:
        return semiring_ell_matmul_pallas(
            values, indices, B, semiring=semiring,
        )
    except PallasELLNotTileable:
        return None


def semiring_ell_matmul(
    values: Num[Array, '... m kmax'],
    indices: Int[Array, '... m kmax'],
    B: Num[Array, '... n_cols ncol'],
    *,
    semiring: Semiring = REAL,
    n_cols: Optional[int] = None,
    backend: Backend = 'auto',
) -> Num[Array, '... m ncol']:
    '''Semiring-generalised ELL-sparse matrix multiplication.

    Computes::

        C[..., i, j] = (+)_p ( values[..., i, p] (*) B[..., indices[..., i, p], j] )

    where the implicit ``M × N`` sparse left operand has the per-row
    neighbour list ``indices[..., i, :]`` with values
    ``values[..., i, :]``.  Padding positions in ``indices`` must point
    to a valid row of ``B``, and the corresponding ``values`` entries
    must be the semiring identity so the contribution is a no-op.

    Parameters
    ----------
    values
        ELL values, shape ``(..., m, k_max)``.
    indices
        ELL column indices into ``B``'s outer dim, shape ``(..., m, k_max)``.
    B
        Dense right operand, shape ``(..., n_cols, ncol)``.  ``n_cols``
        is the outer dim of the implicit sparse matrix.
    semiring
        Algebra to reduce under.
    n_cols
        The implicit sparse-matrix outer dim.  Required for the public
        contract but defaults to ``B.shape[-2]``.
    backend
        ``"auto"``, ``"pallas-cuda"``, or ``"jax"``.

    Returns
    -------
    Array of shape ``(*broadcast_batch, m, ncol)``.
    '''
    _check_ell_shapes(
        values.shape, indices.shape, B.shape, 'semiring_ell_matmul',
    )
    if n_cols is None:
        n_cols = int(B.shape[-2])

    resolved: ResolvedBackend = resolve_backend(backend)

    if resolved == 'pallas-cuda':
        out = _semiring_ell_matmul_pallas(
            values, indices, B, semiring=semiring,
        )
        if out is None:
            resolved = fallback(
                function='semiring_ell_matmul',
                requested='pallas-cuda',
                resolved='jax',
                reason=(
                    f'algebra={semiring.name!r}: Pallas Triton kernel '
                    'unavailable or cannot tile the requested shape'
                ),
                shapes=(
                    tuple(values.shape),
                    tuple(indices.shape),
                    tuple(B.shape),
                ),
                dtype=B.dtype,
            )
        else:
            return out

    # JAX path (batched over leading dims via vmap).
    batch_dims = len(values.shape) - 2
    core = partial(
        reference_semiring_ell_matmul,
        semiring=semiring,
        n_cols=n_cols,
    )
    for _ in range(batch_dims):
        core = jax.vmap(core, in_axes=(0, 0, 0))
    return core(values, indices, B)
