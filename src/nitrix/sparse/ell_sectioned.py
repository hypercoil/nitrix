# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Sectioned ELL -- bucketed-row format for variable-degree adjacencies.

Per SPEC_UPDATE §3.2, the naive ELL layout pads every row to the
global ``k_max``.  That's memory-efficient when ``k_max ≈
median(k)`` but pathological when the worst-case row is much
larger -- e.g. distance-thresholded neighbourhoods in irregular
point clouds (k_max can be 10-100× the median) or atlas-parcel
adjacencies (parcel sizes vary by 1-2 orders of magnitude).

The sectioned-ELL trick: bucket rows by ``ceil(log2(degree))``, run
``semiring_ell_matmul`` once per bucket with the bucket's *local*
``k_max``, and scatter the results back to the original row order.
Memory cost is dominated by the largest bucket, not by the worst-
case row.

This is a Python-level wrapper around the existing
``semiring_ell_matmul`` kernel -- no new kernel code.  Construction
is *not* JIT-compatible because the per-bucket shapes are data-
dependent.  Reuse is JIT-friendly (the per-bucket matmuls trace
once and the dispatch is a static Python loop).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Int, Num
from numpy.typing import NDArray

from .._internal.backend import Backend
from ..semiring import semiring_ell_matmul
from ..semiring._types import Semiring
from ..semiring.algebras import REAL
from .ell import ELL


__all__ = [
    'SectionedELL',
    'sectioned_ell_from_ragged',
    'sectioned_semiring_ell_matmul',
]


@dataclass(frozen=True)
class SectionedELL:
    '''Bucketed-row ELL.

    Attributes
    ----------
    sections
        One ``ELL`` per bucket.  Each bucket has its own
        ``k_max``; this is the whole point of the structure.
    row_groups
        ``len(sections)`` arrays of original-row indices, one
        per bucket.  ``sections[b]`` holds the rows whose
        original indices are ``row_groups[b]``.  Stored as
        ``np.ndarray[int32]`` rather than ``jax.Array`` so the
        Python-level dispatch can use them as static indices for
        scatter-back.
    n_rows
        Total number of rows in the original (un-bucketed) matrix.
    n_cols
        Outer dim of the implicit sparse matrix; same as a flat ELL.
    identity
        The semiring identity used for padding within each bucket.
    '''

    sections: Tuple[ELL, ...]
    row_groups: Tuple[NDArray[Any], ...]
    n_rows: int
    n_cols: int
    identity: Any = None

    @property
    def n_buckets(self) -> int:
        return len(self.sections)

    @property
    def total_storage(self) -> int:
        '''Total per-row entries summed across all buckets.

        Use this to verify the memory win vs. a flat ELL of the same
        adjacency: a flat ELL would store ``n_rows * max(k_max for
        b in sections)`` entries.
        '''
        return sum(s.values.size for s in self.sections)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def _default_bucket(degree: int) -> int:
    '''Default bucketing: ``ceil(log2(max(degree, 1)))``.

    - degree 0 -> bucket 0 with k_max = 1 (and an identity-only row).
    - degree 1 -> bucket 0 with k_max = 1.
    - degree 2 -> bucket 1 with k_max = 2.
    - degree 3-4 -> bucket 2 with k_max = 4.
    - degree 5-8 -> bucket 3 with k_max = 8.  etc.
    '''
    if degree <= 1:
        return 0
    return (degree - 1).bit_length()


def _bucket_kmax(bucket: int) -> int:
    '''Largest degree placeable in bucket ``b``.'''
    return 1 if bucket == 0 else 2 ** bucket


def sectioned_ell_from_ragged(
    values_per_row: Sequence[Num[Array, '...']],
    indices_per_row: Sequence[Int[Array, '...']],
    *,
    n_cols: int,
    identity: Any = 0.0,
    pad_index: int = 0,
    bucket_by: Optional[Callable[[int], int]] = None,
) -> SectionedELL:
    '''Build a ``SectionedELL`` from ragged per-row neighbour lists.

    Parameters
    ----------
    values_per_row, indices_per_row
        Length-``m`` sequences of 1-D arrays.  ``values_per_row[i]``
        and ``indices_per_row[i]`` must have the same length, equal
        to row ``i``'s degree.
    n_cols
        Outer dim of the implicit sparse matrix.
    identity
        Semiring identity used to pad rows within each bucket up to
        the bucket's ``k_max``.
    pad_index
        Row of the dense operand that pad positions reference.
        Default ``0``; must be ``< n_cols``.
    bucket_by
        Custom bucketing function ``degree -> bucket_idx``.  Defaults
        to ``ceil(log2(max(degree, 1)))``.

    Returns
    -------
    A ``SectionedELL`` with one ``ELL`` per non-empty bucket.

    Notes
    -----
    Runs on the host (not JIT-compatible).  Construction is
    typically done once per adjacency (e.g., once per mesh or once
    per pre-computed k-NN graph); the resulting ``SectionedELL``
    can then be passed through JIT'd code many times.
    '''
    n_rows = len(values_per_row)
    if len(indices_per_row) != n_rows:
        raise ValueError(
            f'len(values_per_row)={n_rows} != '
            f'len(indices_per_row)={len(indices_per_row)}.'
        )
    if not (0 <= pad_index < n_cols):
        raise ValueError(
            f'pad_index={pad_index} not in [0, n_cols={n_cols}).'
        )
    if bucket_by is None:
        bucket_by = _default_bucket

    # Collect (bucket, row, values, indices) tuples.
    buckets: dict[
        int, list[tuple[int, NDArray[Any], NDArray[Any]]]
    ] = {}
    for i in range(n_rows):
        v = np.asarray(values_per_row[i])
        idx = np.asarray(indices_per_row[i])
        if v.shape != idx.shape:
            raise ValueError(
                f'row {i}: values.shape={v.shape} != '
                f'indices.shape={idx.shape}.'
            )
        if v.ndim != 1:
            raise ValueError(
                f'row {i}: expected 1-D per-row neighbour list, '
                f'got shape {v.shape}.'
            )
        b = bucket_by(v.size)
        buckets.setdefault(b, []).append((i, v, idx))

    # Empty buckets: skipped (no section emitted).  Order sections by
    # bucket index for determinism.
    sections: list[ELL] = []
    row_groups: list[NDArray[Any]] = []
    # Pick a dtype: float32 if no rows (degenerate); otherwise inherit
    # from the first row's values.
    if n_rows > 0:
        sample = np.asarray(values_per_row[0])
        v_dtype = sample.dtype
        i_dtype = np.asarray(indices_per_row[0]).dtype
        if not np.issubdtype(i_dtype, np.integer):
            i_dtype = np.dtype(np.int32)
    else:
        v_dtype = np.dtype(np.float32)
        i_dtype = np.dtype(np.int32)

    for b in sorted(buckets):
        rows = buckets[b]
        k_max = max(r[1].size for r in rows)
        # Pad bucket k_max up to the bucket's nominal max-degree for
        # better tile / kernel reuse across calls (and to make
        # k_max == 2^b for bucket b >= 1).  This costs a small
        # amount of memory but means re-using the same SectionedELL
        # across many calls compiles each bucket only once.
        k_max_aligned = _bucket_kmax(b)
        if k_max < k_max_aligned:
            k_max = k_max_aligned
        n_rows_b = len(rows)
        values = np.full((n_rows_b, k_max), identity, dtype=v_dtype)
        indices = np.full((n_rows_b, k_max), pad_index, dtype=i_dtype)
        row_idx = np.empty(n_rows_b, dtype=np.int32)
        for j, (orig_i, v, idx) in enumerate(rows):
            row_idx[j] = orig_i
            k_i = v.size
            if k_i > 0:
                values[j, :k_i] = v
                indices[j, :k_i] = idx
        sections.append(ELL(
            values=jnp.asarray(values),
            indices=jnp.asarray(indices),
            n_cols=n_cols,
            identity=identity,
        ))
        row_groups.append(row_idx)

    return SectionedELL(
        sections=tuple(sections),
        row_groups=tuple(row_groups),
        n_rows=n_rows,
        n_cols=n_cols,
        identity=identity,
    )


# ---------------------------------------------------------------------------
# Matmul
# ---------------------------------------------------------------------------


def sectioned_semiring_ell_matmul(
    sectioned: SectionedELL,
    B: Num[Array, 'n_cols ncol'],
    *,
    semiring: Semiring[Any] = REAL,
    backend: Backend = 'auto',
) -> Num[Array, 'm ncol']:
    '''Sectioned-ELL semiring matrix multiplication.

    For each bucket, run ``semiring_ell_matmul`` and scatter the
    result back to the original row positions.  The bucket loop is
    Python-level (one trace per bucket); scatter-back uses
    ``jnp.zeros(...).at[row_indices].set(...)``.

    Parameters
    ----------
    sectioned
        The bucketed adjacency.
    B
        Dense right operand, ``(n_cols, ncol)``.
    semiring
        Algebra to reduce under.
    backend
        ``"auto"``, ``"pallas-cuda"``, or ``"jax"``.  Passed through
        to the per-bucket ``semiring_ell_matmul`` calls.

    Returns
    -------
    Array of shape ``(n_rows, ncol)``.

    Notes
    -----
    Empty rows (degree 0) live in bucket 0 with all-identity
    values; their output is the algebra's identity, broadcast over
    ``ncol``.  This is consistent with "no neighbours -> no
    contribution".

    For the ``LOG`` and ``TROPICAL_*`` algebras with ``-inf`` /
    ``+inf`` identities, empty rows produce ``-inf`` / ``+inf``
    outputs respectively, which is the algebra-correct answer.
    '''
    if B.shape[-2] != sectioned.n_cols:
        raise ValueError(
            f'B.shape[-2]={B.shape[-2]} must equal '
            f'sectioned.n_cols={sectioned.n_cols}.'
        )
    ncol = int(B.shape[-1])
    out_dtype = jnp.result_type(B.dtype, sectioned.sections[0].dtype)

    # Initialise output with the algebra identity (so any rows the
    # bucketing missed -- shouldn't happen, but defensive -- get a
    # sensible value).
    if sectioned.identity is not None:
        init = jnp.full(
            (sectioned.n_rows, ncol), sectioned.identity, dtype=out_dtype,
        )
    else:
        init = jnp.zeros((sectioned.n_rows, ncol), dtype=out_dtype)

    out = init
    for ell, row_idx in zip(sectioned.sections, sectioned.row_groups):
        bucket_out = semiring_ell_matmul(
            ell.values, ell.indices, B,
            semiring=semiring,
            n_cols=sectioned.n_cols,
            backend=backend,
        )
        row_idx_jax = jnp.asarray(row_idx)
        out = out.at[row_idx_jax].set(bucket_out)
    return out
