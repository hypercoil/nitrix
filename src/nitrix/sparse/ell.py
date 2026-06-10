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

import warnings
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal, Optional, Tuple

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int, Num

if TYPE_CHECKING:
    # Type-only import: keeps ``nitrix.sparse.ell`` free of a *runtime*
    # ``nitrix.semiring`` dependency (which would cycle, since semiring
    # imports sparse).  The annihilator is read duck-typed at runtime.
    from ..semiring._types import Semiring


# Sentinel distinguishing "identity not passed" from an explicit
# ``identity=None`` (which legitimately disables pad-fill).
_UNSET: Any = object()


__all__ = [
    'ELL',
    'ell_from_dense',
    'ell_to_dense',
    'ell_pad',
    'ell_mask',
    'ell_add_self_loops',
]


@dataclass(frozen=True)
class ELL:
    """ELL-format sparse matrix.

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
    """

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
    def dtype(self) -> np.dtype[Any]:
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
    """Pad ragged neighbour lists to a fixed ``k_max``.

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
    """
    if values.shape != indices.shape:
        raise ValueError(
            f'ell_pad: values.shape={values.shape} must equal '
            f'indices.shape={indices.shape}.'
        )
    m, k_actual = values.shape
    if k_actual > k_max:
        raise ValueError(f'ell_pad: k_actual={k_actual} > k_max={k_max}.')
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
    symmetrize: bool = False,
) -> ELL:
    """Convert a dense matrix to ELL by selecting top-``k_max`` non-pad entries
    per row.

    Entries with absolute value ``<= threshold`` are treated as
    structural zeros.  If a row has more than ``k_max`` non-zero
    entries, the largest-by-magnitude ``k_max`` are kept.  If ``k_max``
    is ``None``, it is chosen as the maximum row degree present.

    .. warning::

       Top-``k_max``-per-row selection is **not** symmetry-preserving: row
       ``i`` may keep edge ``(i, j)`` while row ``j`` drops ``(j, i)``, so
       the stored pattern is generally **asymmetric** even when ``dense``
       is symmetric.  This matters for the spectral consumers
       (``laplacian_eigenmap`` / ``diffusion_embedding`` and
       ``eigsolve_top_k``), whose iterative solvers assume a symmetric
       operator.  Two remedies: pass the result with the connectopy
       default ``promise_symmetry=False`` (the operator is symmetrised at
       the matvec, storage-free, ``½(A x + Aᵀ x)``); or build a symmetric
       ELL here with ``symmetrize=True`` (below).

    Parameters
    ----------
    dense
        Dense matrix, shape ``(m, n)``.
    k_max
        Target column count of the ELL.  Defaults to the worst-case
        per-row non-zero count.  With ``symmetrize=True`` the *effective*
        column count may exceed this (the symmetric closure can add edges).
    threshold
        Absolute-value cutoff for "structurally zero".
    identity
        Pad fill in ``values``.
    symmetrize
        When ``True``, return the symmetric kNN graph of the selection:
        take the top-``k_max`` pattern ``P``, then store ``½(S + Sᵀ)`` over
        the closure ``P ∪ Pᵀ`` where ``S = dense ⊙ P``.  The result is an
        exactly-symmetric ELL (``ell_to_dense`` is symmetric); ``m`` must
        equal ``n``.  Default ``False`` (the historical asymmetric top-k
        behaviour).

        **Storage is not bounded by a small multiple of** ``k_max``.  The
        effective column count is ``max_i |row_i(P) ∪ col_i(P)|`` --
        i.e. ``k_max`` plus the largest top-k *in-degree* (how many rows
        selected a given column).  In-degree is **not** limited by
        ``k_max``: a hub/star (one column selected by every row) drives the
        closure of that row to the full ``n``, so ``symmetrize=True`` can
        densify hub rows entirely.  For hub-heavy or unknown-degree graphs
        prefer the storage-free matvec symmetrisation
        (``promise_symmetry=False`` on the spectral consumers), which never
        materialises ``P ∪ Pᵀ``.

    Returns
    -------
    ``ELL`` of shape ``(m, n)``.

    Notes
    -----
    Implemented with NumPy on the host: this is a one-shot
    construction helper, not a JIT-traceable op.  Treat it as a
    convenience for tests and small workloads.
    """
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

    # Per-row top-k selection pattern.
    selection = np.zeros((m, n), dtype=bool)
    for i in range(m):
        row_idx = np.flatnonzero(mask[i])
        if row_idx.size > k_max:
            order = np.argsort(-abs_d[i, row_idx])
            row_idx = row_idx[order[:k_max]]
        selection[i, row_idx] = True

    if symmetrize:
        if m != n:
            raise ValueError(
                f'ell_from_dense(symmetrize=True): requires a square matrix, '
                f'got {dense_np.shape}.'
            )
        # Symmetric kNN graph: ½(S + Sᵀ) over the closure P ∪ Pᵀ.
        masked = np.where(selection, dense_np, np.zeros_like(dense_np))
        sym_vals = 0.5 * (masked + masked.T)
        closure = selection | selection.T
        eff_k = max(int(closure.sum(axis=1).max()) if closure.size else 0, 1)
        values = np.full((m, eff_k), identity, dtype=dense_np.dtype)
        indices = np.zeros((m, eff_k), dtype=np.int32)
        for i in range(m):
            cols = np.flatnonzero(closure[i])
            cols = cols[np.argsort(-np.abs(sym_vals[i, cols]))]
            k_i = cols.size
            values[i, :k_i] = sym_vals[i, cols]
            indices[i, :k_i] = cols
        return ELL(
            values=jnp.asarray(values),
            indices=jnp.asarray(indices),
            n_cols=n,
            identity=identity,
        )

    values = np.full((m, k_max), identity, dtype=dense_np.dtype)
    indices = np.zeros((m, k_max), dtype=np.int32)
    for i in range(m):
        row_idx = np.flatnonzero(selection[i])
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
    identity: Any = _UNSET,
    semiring: Optional['Semiring[Any]'] = None,
) -> ELL:
    """Mask edges out of an ELL by setting their value to a semiring annihilator.

    Brain geometries are routinely *incomplete*: the cortical-surface
    medial wall is excluded from analysis, and volumetric work is often
    restricted to a grey-matter (or otherwise thresholded) mask.  Edges
    that reach a masked vertex / voxel must contribute the **semiring
    identity** to every reduction -- so no FLOPs and, crucially, no
    *signal* leak in from the uninteresting region.

    This helper rewrites the masked positions of ``ell.values`` to the
    algebra's ``(*)``-annihilator and stamps that value onto the
    returned ELL's ``identity`` field, so the result is a drop-in
    operand for ``semiring_ell_matmul`` / ``semiring_ell_edge_aggregate``.

    **Why this is a true no-op (and the one algebra where it is not).**
    An edge with value ``z`` is a no-op iff ``z`` is the ``(*)``
    *annihilator* of the algebra -- the element with ``z (*) b ==
    monoid_identity`` for every ``b`` -- because the reduction computes
    ``(+)_p ( values[i, p] (*) B[indices[i, p]] )``.  For most built-in
    algebras the annihilator coincides with ``semiring.identity`` (the
    *additive* / monoid identity), which is why a bare ``identity=``
    historically doubled as the mask value:

    ===================  ============  ============  ============
    Algebra              ``(*)``       identity      annihilator
    ===================  ============  ============  ============
    ``REAL``             ``a * b``     ``0``         ``0``
    ``LOG``              ``a + b``     ``-inf``      ``-inf``
    ``TROPICAL_MAX_PLUS````a + b``     ``-inf``      ``-inf``
    ``TROPICAL_MIN_PLUS````a + b``     ``+inf``      ``+inf``
    ``BOOLEAN``          ``a and b``   ``False``     ``False``
    ``EUCLIDEAN``        ``(a-b)**2``  ``0``         **none**
    ===================  ============  ============  ============

    **Prefer the ``semiring=`` form.**  Pass ``semiring=`` and the
    annihilator is read from ``semiring.annihilator`` -- the correct
    masking value by construction, with no chance of the
    identity-vs-annihilator confusion below.  The explicit
    ``identity=`` form remains for callers that already hold the scalar.
    Pass exactly one of the two.

    ``EUCLIDEAN`` is the exception: its ``(*)`` is ``(a - b)**2``, which
    has **no** annihilator (``(z - b)**2`` cannot be ``0`` for all
    ``b``), so ``EUCLIDEAN.annihilator is None`` and ``semiring=EUCLIDEAN``
    raises here.  Its ``identity`` of ``0`` does **not** mask -- it would
    inject ``B[idx]**2`` instead.  Mask EUCLIDEAN neighbourhoods by
    dropping the columns from the index structure (do not include the
    edge), not via this helper.

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
        The scalar written into masked positions.  Use
        ``semiring.identity`` for the algebra you will reduce under.
        Mutually exclusive with ``semiring=``; pass exactly one.
    semiring
        The algebra the masked ELL will be reduced under.  Its
        ``annihilator`` field supplies the mask value; raises if the
        algebra has no annihilator (``EUCLIDEAN``).  Mutually exclusive
        with ``identity=``; pass exactly one.

    Returns
    -------
    A new ``ELL`` with masked values set to the annihilator and
    ``.identity`` set to that value.  ``indices`` is unchanged (masked
    indices stay in-range, so no gather goes out of bounds).

    Raises
    ------
    ValueError
        If neither or both of ``identity`` / ``semiring`` are given; if
        ``semiring`` has no annihilator (``EUCLIDEAN``); or if ``valid``
        has an unrecognised shape.
    """
    if semiring is not None:
        if identity is not _UNSET:
            raise ValueError(
                'ell_mask: pass exactly one of identity= or semiring=, '
                'not both.'
            )
        mask_value = semiring.annihilator
        if mask_value is None:
            raise ValueError(
                f'ell_mask: semiring {semiring.name!r} has no '
                '(*)-annihilator, so masking by a value cannot be a '
                'no-op (e.g. EUCLIDEAN: (a - b)**2 never annihilates).  '
                'Drop the masked columns from the index structure '
                'instead of calling ell_mask.'
            )
    elif identity is not _UNSET:
        warnings.warn(
            'ell_mask(identity=...) is deprecated; pass semiring= '
            "instead.  The masking value is the algebra's "
            '(*)-annihilator, read for you from semiring.annihilator -- '
            'which is *not* always the monoid identity (EUCLIDEAN has '
            'no annihilator, so identity=0 silently injects B[idx]**2 '
            'rather than masking).  The identity= form will be removed '
            'in a future release.',
            DeprecationWarning,
            stacklevel=2,
        )
        mask_value = identity
    else:
        raise ValueError(
            'ell_mask: pass exactly one of identity= (the annihilator '
            'scalar) or semiring= (whose annihilator is read for you).'
        )
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
    id_val = jnp.asarray(mask_value, dtype=ell.values.dtype)
    new_values = jnp.where(edge_valid, ell.values, id_val)
    return replace(ell, values=new_values, identity=mask_value)


def ell_add_self_loops(
    ell: ELL,
    edge_attr: Optional[Float[Array, 'm k_max f']] = None,
    *,
    fill: Literal['mean', 'zero', 'add'] = 'mean',
    self_value: float = 1.0,
) -> Tuple[ELL, Optional[Float[Array, 'm k_max_plus_1 f']]]:
    """Append a self-loop ``(i, i)`` to every row of an ELL adjacency.

    Returns a new ELL with one extra neighbour slot per row whose index
    is the row itself (``indices[i, -1] = i``) and whose ``values`` entry
    is ``self_value`` -- a non-identity marker so the slot reads as a real
    edge (e.g. it survives ``ell_row_softmax``'s padding mask), not as a
    pad.  ``k_max`` grows by one.

    Self-loops are a graph-convolution construct, not part of the geometric
    mesh adjacency: ``mesh_k_ring_adjacency`` and friends are self-loop-free.
    Graph attention attends each vertex to its own features -- the
    neighbourhood in Velickovic et al. (2018) explicitly *includes* node
    ``i`` -- and the GCN renormalisation trick (Kipf & Welling 2017) adds
    the self-connection ``A_hat = A + I``.  A convolution that wants either
    adds the self-edge to the bare adjacency before aggregating; without it
    the row's reduction (and any row-softmax over it) simply omits the
    self-term, which is a different operator.

    Per-edge attributes
    -------------------
    When ``edge_attr`` is supplied (the per-edge vector tensor consumed by
    ``semiring_ell_edge_aggregate``), the synthesised self-edge needs an
    attribute.  Lacking an intrinsic self-feature, ``fill`` derives one
    from the row's existing **valid** (non-pad) edges:

    - ``'mean'`` -- the per-row mean of the vertex's other edge attributes;
      the natural default when no self-feature is defined.
    - ``'add'``  -- the per-row sum.
    - ``'zero'`` -- a zero vector.

    Padding slots are excluded from the reduction via ``ell.values !=
    ell.identity`` (so a partially-padded row averages only its real
    edges).  ``fill`` applies only to ``edge_attr``; the scalar ``values``
    self-slot is always ``self_value`` (the geometric-weight channel,
    typically overwritten downstream -- e.g. by attention weights).

    Parameters
    ----------
    ell
        Adjacency in ELL format, assumed self-loop-free (the geometric
        mesh case).  If a row already contains an ``(i, i)`` edge it is
        not removed; a duplicate self-slot is appended.
    edge_attr
        Optional per-edge attributes, shape ``(m, k_max, *f)`` aligned
        with the ELL pattern.  ``None`` skips attribute handling.
    fill
        Self-edge attribute rule; see above.  Ignored when ``edge_attr``
        is ``None``.
    self_value
        Scalar written into the self-slot of ``values``.  Must differ from
        ``ell.identity`` for the slot to read as a real edge (the default
        ``1.0`` is correct for every built-in algebra except a degenerate
        ``identity == 1``).

    Returns
    -------
    ``(ell_with_loops, edge_attr_with_loops)``.  The ELL has ``k_max + 1``
    slots; ``edge_attr_with_loops`` is ``None`` iff ``edge_attr`` was
    ``None``.
    """
    m, k_max = ell.indices.shape[-2], ell.indices.shape[-1]
    self_idx = jnp.broadcast_to(
        jnp.arange(m, dtype=ell.indices.dtype)[:, None], (m, 1)
    )
    self_val = jnp.full((m, 1), self_value, dtype=ell.values.dtype)
    new_ell = replace(
        ell,
        values=jnp.concatenate([ell.values, self_val], axis=-1),
        indices=jnp.concatenate([ell.indices, self_idx], axis=-1),
    )

    if edge_attr is None:
        return new_ell, None

    if edge_attr.shape[:2] != (m, k_max):
        raise ValueError(
            f'ell_add_self_loops: edge_attr.shape={tuple(edge_attr.shape)} '
            f'must lead with (m, k_max)={(m, k_max)} matching the ELL '
            'pattern.'
        )

    # Reduce the row's existing valid (non-pad) attributes for the self-edge.
    if ell.identity is None:
        valid = jnp.ones((m, k_max), dtype=bool)
    else:
        valid = ell.values != ell.identity  # (m, k_max)
    feat_rank = edge_attr.ndim - 2
    vmask = valid.reshape((m, k_max) + (1,) * feat_rank)
    masked = jnp.where(vmask, edge_attr, jnp.zeros_like(edge_attr))
    summed = jnp.sum(masked, axis=1, keepdims=True)  # (m, 1, *f)
    if fill == 'mean':
        count = jnp.maximum(jnp.sum(valid, axis=-1), 1)
        count = count.reshape((m, 1) + (1,) * feat_rank)
        loop_attr = summed / count
    elif fill == 'add':
        loop_attr = summed
    elif fill == 'zero':
        loop_attr = jnp.zeros_like(summed)
    else:
        raise ValueError(
            f"ell_add_self_loops: fill={fill!r}; expected 'mean', 'add', "
            "or 'zero'."
        )
    new_edge_attr = jnp.concatenate([edge_attr, loop_attr], axis=1)
    return new_ell, new_edge_attr


def ell_to_dense(ell: ELL) -> Num[Array, 'm n']:
    """Scatter an ELL back to a dense ``(m, n_cols)`` matrix.

    Useful in tests and in fallback paths.  Pad entries (assumed to
    carry the algebra's identity) are written via ``scatter_add`` with
    the identity, which is a no-op for ``REAL``; callers using
    non-additive semirings should not rely on this for round-trip
    correctness.
    """
    m, k_max = ell.values.shape
    dense = jnp.zeros((m, ell.n_cols), dtype=ell.values.dtype)
    row_idx = jnp.broadcast_to(jnp.arange(m)[:, None], (m, k_max)).reshape(-1)
    col_idx = ell.indices.reshape(-1)
    vals = ell.values.reshape(-1)
    return dense.at[row_idx, col_idx].add(vals)
