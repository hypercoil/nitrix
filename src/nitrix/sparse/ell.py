# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
ELL sparse format primitives.

ELL is the primary format in nitrix.sparse (SPEC §3.2): the per-row
neighbour list is stored as two arrays, ``values: (m, k_max)`` and
``indices: (m, k_max)``.  Rows with fewer than ``k_max`` neighbours
have the trailing entries padded with the semiring identity in
``values`` and a fixed (in-range) sentinel index in ``indices``.

The format is intentionally a thin pair of dense arrays: no BCOO, no
``jax.experimental.sparse`` import.  Sectioned ELL (variable-degree;
SPEC_UPDATE §3.2) will live in ``nitrix.sparse.ell_sectioned``.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Optional

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Int, Num


__all__ = [
    'ELL',
    'ell_from_dense',
    'ell_to_dense',
    'ell_pad',
    'ell_mask',
]


@dataclass(frozen=True)
class ELL:
    '''ELL-format sparse matrix.

    Attributes
    ----------
    values
        Per-row non-pad weights, shape ``(m, k_max)``.
    indices
        Per-row neighbour indices, shape ``(m, k_max)``.  Pad positions
        must point to a valid row of the dense operand so a stray
        ``gather`` does not OOB; ``ell_pad`` enforces this.
    n_cols
        The implicit ``n``-dim of the sparse matrix (i.e., the outer
        dim of the dense operand it will be contracted with).
    identity
        The semiring identity that pad entries of ``values`` have been
        filled with.  ``None`` if the caller has handled sentinels.
    '''

    values: Num[Array, 'm k_max']
    indices: Int[Array, 'm k_max']
    n_cols: int
    identity: Any = None

    @property
    def n_rows(self) -> int:
        return int(self.values.shape[-2])

    @property
    def k_max(self) -> int:
        return int(self.values.shape[-1])

    @property
    def shape(self) -> tuple[int, int]:
        return (self.n_rows, self.n_cols)

    @property
    def dtype(self):
        return self.values.dtype


def ell_pad(
    values: Num[Array, 'm k_actual'],
    indices: Int[Array, 'm k_actual'],
    *,
    k_max: int,
    n_cols: int,
    identity: Any = 0.0,
    pad_index: int = 0,
) -> ELL:
    '''Pad ragged neighbour lists to a fixed ``k_max``.

    Use this when constructing an ``ELL`` from per-row lists whose
    lengths vary up to ``k_max``: ``values[i, k_i:]`` is filled with
    ``identity`` and ``indices[i, k_i:]`` is filled with ``pad_index``.
    ``pad_index`` must be a valid row of the dense operand the ELL
    will be contracted with (``0`` works whenever ``n_cols >= 1``).

    Parameters
    ----------
    values
        Right-aligned per-row values, shape ``(m, k_actual)`` with
        ``k_actual <= k_max``.
    indices
        Right-aligned per-row indices, shape ``(m, k_actual)``.
    k_max
        Target padded width.
    n_cols
        Outer dim of the dense operand; recorded on the ``ELL``.
    identity
        Semiring identity to write into the pad positions.  Defaults
        to ``0`` (the ``REAL`` identity).
    pad_index
        Row of the dense operand that pad positions reference.  Must
        satisfy ``0 <= pad_index < n_cols``.

    Returns
    -------
    The padded ``ELL``.
    '''
    if values.shape != indices.shape:
        raise ValueError(
            f'ell_pad: values.shape={values.shape} must equal '
            f'indices.shape={indices.shape}.'
        )
    m, k_actual = values.shape
    if k_actual > k_max:
        raise ValueError(
            f'ell_pad: k_actual={k_actual} > k_max={k_max}.'
        )
    if not (0 <= pad_index < n_cols):
        raise ValueError(
            f'ell_pad: pad_index={pad_index} not in [0, n_cols={n_cols}).'
        )
    if k_actual == k_max:
        return ELL(
            values=values,
            indices=indices,
            n_cols=n_cols,
            identity=identity,
        )
    pad_w = k_max - k_actual
    pad_v = jnp.full((m, pad_w), identity, dtype=values.dtype)
    pad_i = jnp.full((m, pad_w), pad_index, dtype=indices.dtype)
    return ELL(
        values=jnp.concatenate([values, pad_v], axis=-1),
        indices=jnp.concatenate([indices, pad_i], axis=-1),
        n_cols=n_cols,
        identity=identity,
    )


def ell_from_dense(
    dense: Num[Array, 'm n'],
    *,
    k_max: Optional[int] = None,
    threshold: float = 0.0,
    identity: Any = 0.0,
) -> ELL:
    '''Convert a dense matrix to ELL by selecting top-``k_max`` non-pad entries
    per row.

    Entries with absolute value ``<= threshold`` are treated as
    structural zeros.  If a row has more than ``k_max`` non-zero
    entries, the largest-by-magnitude ``k_max`` are kept.  If ``k_max``
    is ``None``, it is chosen as the maximum row degree present.

    Parameters
    ----------
    dense
        Dense matrix, shape ``(m, n)``.
    k_max
        Target column count of the ELL.  Defaults to the worst-case
        per-row non-zero count.
    threshold
        Absolute-value cutoff for "structurally zero".
    identity
        Pad fill in ``values``.

    Returns
    -------
    ``ELL`` of shape ``(m, n)``.

    Notes
    -----
    Implemented with NumPy on the host: this is a one-shot
    construction helper, not a JIT-traceable op.  Treat it as a
    convenience for tests and small workloads.
    '''
    if dense.ndim != 2:
        raise ValueError(
            f'ell_from_dense: dense must be 2-D, got {dense.shape}.'
        )
    dense_np = np.asarray(dense)
    m, n = dense_np.shape
    abs_d = np.abs(dense_np)
    mask = abs_d > threshold
    deg = mask.sum(axis=1)
    if k_max is None:
        k_max = int(deg.max()) if deg.size else 0
        k_max = max(k_max, 1)
    values = np.full((m, k_max), identity, dtype=dense_np.dtype)
    indices = np.zeros((m, k_max), dtype=np.int32)
    for i in range(m):
        row_idx = np.flatnonzero(mask[i])
        if row_idx.size > k_max:
            order = np.argsort(-abs_d[i, row_idx])
            row_idx = row_idx[order[:k_max]]
        else:
            row_idx = row_idx[np.argsort(-abs_d[i, row_idx])]
        k_i = row_idx.size
        values[i, :k_i] = dense_np[i, row_idx]
        indices[i, :k_i] = row_idx
    return ELL(
        values=jnp.asarray(values),
        indices=jnp.asarray(indices),
        n_cols=n,
        identity=identity,
    )


def ell_mask(
    ell: ELL,
    valid: Num[Array, '...'],
    *,
    identity: Any,
) -> ELL:
    '''Mask edges out of an ELL by setting their value to a semiring identity.

    Brain geometries are routinely *incomplete*: the cortical-surface
    medial wall is excluded from analysis, and volumetric work is often
    restricted to a grey-matter (or otherwise thresholded) mask.  Edges
    that reach a masked vertex / voxel must contribute the **semiring
    identity** to every reduction -- so no FLOPs and, crucially, no
    *signal* leak in from the uninteresting region.

    This helper rewrites the masked positions of ``ell.values`` to
    ``identity`` and stamps ``identity`` onto the returned ELL's
    ``identity`` field, so the result is a drop-in operand for
    ``semiring_ell_matmul`` / ``semiring_ell_edge_aggregate`` under a
    semiring whose identity is ``identity``.

    **Why this is a true no-op (and the one algebra where it is not).**
    An edge with value ``z`` is a no-op iff ``z`` is the ``(*)``
    *annihilator* of the algebra -- the element with ``z (*) b ==
    monoid_identity`` for every ``b`` -- because the reduction computes
    ``(+)_p ( values[i, p] (*) B[indices[i, p]] )``.  For the built-in
    algebras the annihilator equals ``semiring.identity``:

    ===================  ============  ====================
    Algebra              ``(*)``       annihilator (= id)
    ===================  ============  ====================
    ``REAL``             ``a * b``     ``0``
    ``LOG``              ``a + b``     ``-inf``
    ``TROPICAL_MAX_PLUS````a + b``     ``-inf``
    ``TROPICAL_MIN_PLUS````a + b``     ``+inf``
    ``BOOLEAN``          ``a and b``   ``False``
    ===================  ============  ====================

    Pass ``identity=semiring.identity`` and the masked edge vanishes
    regardless of where its (still in-range) index points.

    ``EUCLIDEAN`` is the exception: its ``(*)`` is ``(a - b)**2``, which
    has **no** annihilator (``(z - b)**2`` cannot be ``0`` for all
    ``b``), so its ``identity`` of ``0`` does **not** mask -- it injects
    ``B[idx]**2`` instead.  Mask EUCLIDEAN neighbourhoods by dropping the
    columns from the index structure (do not include the edge), not via
    this helper.

    Parameters
    ----------
    ell
        Adjacency / operator in ELL format.
    valid
        Boolean keep-mask.  Either:

        - shape ``(n_cols,)`` -- a per-column (target vertex / voxel)
          mask; an edge is masked when ``valid[ell.indices[i, p]]`` is
          ``False`` (the medial-wall / grey-matter case); or
        - shape ``ell.indices.shape`` -- an explicit per-edge mask.

    identity
        The value written into masked positions.  Use
        ``semiring.identity`` for the algebra you will reduce under.

    Returns
    -------
    A new ``ELL`` with masked values set to ``identity`` and
    ``.identity = identity``.  ``indices`` is unchanged (masked indices
    stay in-range, so no gather goes out of bounds).
    '''
    valid = jnp.asarray(valid)
    if valid.shape == (ell.n_cols,):
        edge_valid = valid[ell.indices]
    elif valid.shape == tuple(ell.indices.shape):
        edge_valid = valid
    else:
        raise ValueError(
            f'ell_mask: valid.shape={tuple(valid.shape)} must be '
            f'(n_cols,)=({ell.n_cols},) for a column mask or '
            f'{tuple(ell.indices.shape)} for an edge mask.'
        )
    id_val = jnp.asarray(identity, dtype=ell.values.dtype)
    new_values = jnp.where(edge_valid, ell.values, id_val)
    return replace(ell, values=new_values, identity=identity)


def ell_to_dense(ell: ELL) -> Num[Array, 'm n']:
    '''Scatter an ELL back to a dense ``(m, n_cols)`` matrix.

    Useful in tests and in fallback paths.  Pad entries (assumed to
    carry the algebra's identity) are written via ``scatter_add`` with
    the identity, which is a no-op for ``REAL``; callers using
    non-additive semirings should not rely on this for round-trip
    correctness.
    '''
    m, k_max = ell.values.shape
    dense = jnp.zeros((m, ell.n_cols), dtype=ell.values.dtype)
    row_idx = jnp.broadcast_to(
        jnp.arange(m)[:, None], (m, k_max)
    ).reshape(-1)
    col_idx = ell.indices.reshape(-1)
    vals = ell.values.reshape(-1)
    return dense.at[row_idx, col_idx].add(vals)
