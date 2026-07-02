# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Graph Laplacian variants and the shared sparse-matvec machinery.

This module is intentionally pure "graph Laplacians": the
modularity-related operators live in ``community.py`` because their
*role* is community detection even though their algebraic shape
overlaps with Laplacians.  The split keeps the Laplacian import
graph free of community-detection concerns.

Three Laplacian variants exposed via :func:`laplacian`
(selected by ``normalisation=...``):

- ``"combinatorial"`` -- :math:`L = D - A`.  Eigenvalues unbounded.
- ``"symmetric"`` -- :math:`L = I - D^{-1/2} A D^{-1/2}`.  Symmetric,
  positive semidefinite, eigenvalues in :math:`[0, 2]`; the canonical
  "normalised Laplacian" used in Laplacian-eigenmap embeddings.
- ``"random_walk"`` -- :math:`L = I - D^{-1} A`.  Not symmetric in
  general but shares the spectrum of ``"symmetric"`` and is the
  generator of the random walk on the graph.

Multi-format support: :func:`degree_vector` and :func:`laplacian_matvec`
accept dense ``(..., n, n)``, :class:`~nitrix.sparse.ELL`, or
:class:`~nitrix.sparse.SectionedELL` inputs.  The explicit
:func:`laplacian` returns dense -- by definition, since the symmetric
Laplacian's diagonal is generally non-zero and does not fit the ELL
pattern of the input.  For *operator* use (eigenvalue solvers, etc.)
call :func:`laplacian_matvec`, which never materialises the dense matrix.
"""

from __future__ import annotations

from typing import Literal, Union

import jax.numpy as jnp
from jaxtyping import Array, Num

# TypeIs (PEP 742) narrows in *both* branches of a guard; sourced from
# typing_extensions (a guaranteed transitive dependency via jaxtyping) so the
# 3.11 baseline keeps it -- ``typing.TypeIs`` only lands in 3.13.
from typing_extensions import TypeIs

from ..semiring import REAL, semiring_ell_matmul, semiring_ell_rmatvec
from ..sparse import (
    ELL,
    SectionedELL,
    sectioned_semiring_ell_matmul,
    sectioned_semiring_ell_rmatvec,
)

__all__ = [
    'laplacian',
    'laplacian_matvec',
    'degree_vector',
    'in_degree_vector',
    'symmetric_degree_vector',
]


_Normalisation = Literal['combinatorial', 'symmetric', 'random_walk']
_GraphInput = Union[Num[Array, '... n n'], ELL, SectionedELL]


# ---------------------------------------------------------------------------
# Shared helpers (also used by community.py)
# ---------------------------------------------------------------------------


def _delete_diagonal(A: Num[Array, '... n n']) -> Num[Array, '... n n']:
    n = A.shape[-1]
    mask = ~jnp.eye(n, dtype=bool)
    return A * mask


def _safe_max(x: Num[Array, '...']) -> Num[Array, '...']:
    """Maximum over the trailing two axes, clipped to be at least one.

    Used to normalise unbounded logits by a stable scale.

    Parameters
    ----------
    x
        Input array, ``(...)``.  The reduction is taken over the last
        two axes.

    Returns
    -------
    Num[Array, '...']
        The maximum of ``x`` over its trailing two axes, floored at
        ``1.0``, with those axes kept as singletons for broadcasting.
    """
    return jnp.maximum(1.0, x.max(axis=(-1, -2), keepdims=True))


def _is_sparse(A: _GraphInput) -> TypeIs[Union[ELL, SectionedELL]]:
    return isinstance(A, (ELL, SectionedELL))


def _ell_matvec(
    A: Union[ELL, SectionedELL],
    x: Num[Array, '... n k'],
    *,
    promise_symmetry: bool = True,
) -> Num[Array, '... n k']:
    """Sparse matrix-block product :math:`A x` for ELL / SectionedELL.

    Dispatched once to keep callers simple.  Always uses the REAL
    semiring; for other algebras the caller should use
    :func:`~nitrix.semiring.semiring_ell_matmul` /
    :func:`~nitrix.sparse.sectioned_semiring_ell_matmul` directly.

    When ``promise_symmetry`` is ``False`` the symmetric part
    :math:`\\tfrac{1}{2}(A x + A^{\\top} x)` is applied instead of the bare
    :math:`A x` -- for adjacencies whose stored pattern is not symmetric
    (e.g. top-k affinity sparsification).  The operator must be square.

    Parameters
    ----------
    A
        Sparse adjacency, an :class:`~nitrix.sparse.ELL` or
        :class:`~nitrix.sparse.SectionedELL` operator.
    x
        Block of vectors to apply the operator to, ``(..., n, k)``.
    promise_symmetry
        When ``True`` (default) the stored operator is applied directly.
        When ``False`` the symmetric part
        :math:`\\tfrac{1}{2}(A x + A^{\\top} x)` is applied instead.

    Returns
    -------
    Num[Array, '... n k']
        The product :math:`A x` (or its symmetric part when
        ``promise_symmetry`` is ``False``), shape ``(..., n, k)``.
    """
    if isinstance(A, ELL):
        Ax = semiring_ell_matmul(
            A.values,
            A.indices,
            x,
            semiring=REAL,
            n_cols=A.n_cols,
            backend='jax',
        )
        if promise_symmetry:
            return Ax
        Atx = semiring_ell_rmatvec(
            A.values, A.indices, x, semiring=REAL, n_cols=A.n_cols
        )
        return 0.5 * (Ax + Atx)
    Ax = sectioned_semiring_ell_matmul(
        A,
        x,
        semiring=REAL,
        backend='jax',
    )
    if promise_symmetry:
        return Ax
    Atx = sectioned_semiring_ell_rmatvec(A, x, semiring=REAL)
    return 0.5 * (Ax + Atx)


# ---------------------------------------------------------------------------
# Degree
# ---------------------------------------------------------------------------


def degree_vector(A: _GraphInput) -> Num[Array, '... n']:
    """Per-node degree (row sum) of an adjacency operator.

    For an :class:`~nitrix.sparse.ELL` adjacency this is the sum of the
    stored values along the last axis (pad positions contribute the
    additive identity ``0``).  For :class:`~nitrix.sparse.SectionedELL`,
    each bucket is summed and scattered back to the original row order.
    For dense ``A`` it is the row sum ``A.sum(-1)``.

    For directed graphs (asymmetric ``A``) this returns the out-degree;
    the corresponding in-degree is given by :func:`in_degree_vector`.

    Parameters
    ----------
    A
        Adjacency operator: dense ``(..., n, n)``,
        :class:`~nitrix.sparse.ELL`, or
        :class:`~nitrix.sparse.SectionedELL`.

    Returns
    -------
    Num[Array, '... n']
        Per-node (out-)degree, shape ``(..., n)``.
    """
    if isinstance(A, ELL):
        return A.values.sum(axis=-1)
    if isinstance(A, SectionedELL):
        # Sum values within each section, scatter back to original rows.
        out = jnp.zeros((A.n_rows,), dtype=A.sections[0].dtype)
        for ell, row_idx in zip(A.sections, A.row_groups):
            row_idx_jax = jnp.asarray(row_idx)
            out = out.at[row_idx_jax].set(ell.values.sum(axis=-1))
        return out
    return A.sum(axis=-1)


def in_degree_vector(A: _GraphInput) -> Num[Array, '... n']:
    """Per-node in-degree (column sum) -- the adjoint of :func:`degree_vector`.

    For an asymmetric ``A`` this is the column sum
    :math:`\\sum_i A_{ij}`; it equals the (row-sum) out-degree if and only
    if ``A`` is symmetric.  Computed without materialising
    :math:`A^{\\top}`: dense ``A.sum(-2)``;
    :class:`~nitrix.sparse.ELL` / :class:`~nitrix.sparse.SectionedELL` via
    the additive adjoint matvec applied to the ones vector
    (:math:`A^{\\top} \\mathbf{1}`) -- one ``O(nnz)`` pass over the stored
    sparsity pattern (the same adjoint the spectral solvers use for the
    :math:`\\tfrac{1}{2}(A x + A^{\\top} x)` symmetric matvec).

    Parameters
    ----------
    A
        Adjacency operator: dense ``(..., n, n)``,
        :class:`~nitrix.sparse.ELL`, or
        :class:`~nitrix.sparse.SectionedELL`.

    Returns
    -------
    Num[Array, '... n']
        Per-node in-degree, shape ``(..., n)``.
    """
    if isinstance(A, ELL):
        ones = jnp.ones(A.values.shape[:-1] + (1,), dtype=A.values.dtype)
        return semiring_ell_rmatvec(
            A.values, A.indices, ones, semiring=REAL, n_cols=A.n_cols
        )[..., 0]
    if isinstance(A, SectionedELL):
        ones = jnp.ones((A.n_rows, 1), dtype=A.sections[0].dtype)
        return sectioned_semiring_ell_rmatvec(A, ones, semiring=REAL)[..., 0]
    return A.sum(axis=-2)


def symmetric_degree_vector(A: _GraphInput) -> Num[Array, '... n']:
    """Degree of the symmetrised adjacency :math:`W = \\tfrac{1}{2}(A + A^{\\top})`.

    The per-node degree is
    :math:`d_i = \\tfrac{1}{2}(\\sum_j A_{ij} + \\sum_j A_{ji}) =
    \\tfrac{1}{2}(\\text{out-degree} + \\text{in-degree})` -- the degree the
    symmetric normalised Laplacian / affinity must normalise by when ``A``
    is **not** symmetric (e.g. a top-k-per-row sparsified affinity).
    Normalising by the bare row-sum out-degree instead uses the wrong
    diagonal, so :math:`\\tfrac{1}{2}(D^{-1/2} A D^{-1/2} + (\\cdot)^{\\top})`
    is no longer :math:`D_W^{-1/2} W D_W^{-1/2}` and its trivial (constant)
    eigenvalue drifts off :math:`1` -- the implicit "symmetrise the
    Laplacian, not the adjacency" error.  Equals :func:`degree_vector`
    exactly for symmetric ``A``.

    Parameters
    ----------
    A
        Adjacency operator: dense ``(..., n, n)``,
        :class:`~nitrix.sparse.ELL`, or
        :class:`~nitrix.sparse.SectionedELL`.

    Returns
    -------
    Num[Array, '... n']
        Per-node degree of the symmetrised adjacency, shape ``(..., n)``.
    """
    return 0.5 * (degree_vector(A) + in_degree_vector(A))


# ---------------------------------------------------------------------------
# Dense Laplacian
# ---------------------------------------------------------------------------


def laplacian(
    A: Num[Array, '... n n'],
    *,
    normalisation: _Normalisation = 'combinatorial',
    eps: float = 1e-12,
) -> Num[Array, '... n n']:
    """Graph Laplacian in one of three standard normalisations.

    *Dense only*: the symmetric Laplacian has a non-zero diagonal
    that does not fit the sparse pattern of the input ``A``.  For
    sparse inputs use :func:`laplacian_matvec` to apply the Laplacian
    to a vector without materialising the matrix.

    Parameters
    ----------
    A
        Adjacency matrix block, ``(..., n, n)``.  Non-negative;
        self-loops in ``A`` flow through (they are not subtracted).
    normalisation
        Which Laplacian variant to build: ``"combinatorial"``
        (:math:`D - A`), ``"symmetric"`` (:math:`I - D^{-1/2} A D^{-1/2}`),
        or ``"random_walk"`` (:math:`I - D^{-1} A`).
    eps
        Floor on degree before division (avoids ``0/0`` for
        isolated nodes; those end up with a zero row in the
        normalised variants).

    Returns
    -------
    Num[Array, '... n n']
        The dense Laplacian in the requested normalisation, shape
        ``(..., n, n)``.
    """
    if normalisation not in ('combinatorial', 'symmetric', 'random_walk'):
        raise ValueError(
            f'normalisation={normalisation!r}; expected one of '
            '"combinatorial", "symmetric", "random_walk".'
        )

    n = A.shape[-1]
    deg = degree_vector(A)
    eye = jnp.eye(n, dtype=A.dtype)
    while eye.ndim < A.ndim:
        eye = eye[None, ...]

    if normalisation == 'combinatorial':
        D = jnp.zeros_like(A).at[..., jnp.arange(n), jnp.arange(n)].set(deg)
        return D - A

    safe_deg = jnp.maximum(deg, eps)
    if normalisation == 'symmetric':
        inv_sqrt_d = (1.0 / jnp.sqrt(safe_deg))[..., :, None]
        norm_A = A * inv_sqrt_d * inv_sqrt_d.swapaxes(-1, -2)
        return eye - norm_A
    # random_walk
    inv_d = (1.0 / safe_deg)[..., :, None]
    return eye - inv_d * A


# ---------------------------------------------------------------------------
# Sparse-friendly matvec (the path the eigenvalue solvers use)
# ---------------------------------------------------------------------------


def laplacian_matvec(
    A: _GraphInput,
    x: Num[Array, '... n k'],
    *,
    normalisation: _Normalisation = 'symmetric',
    eps: float = 1e-12,
    promise_symmetry: bool = True,
) -> Num[Array, '... n k']:
    """Apply the Laplacian to a block of vectors, sparse-friendly.

    Computes :math:`L x` without ever materialising :math:`L`.  Accepts
    dense, :class:`~nitrix.sparse.ELL`, or
    :class:`~nitrix.sparse.SectionedELL` ``A``; works for the three
    Laplacian variants.  This is the operator passed to
    ``jax.experimental.sparse.linalg.lobpcg_standard`` for top-k
    eigenvalue decomposition on graphs too large for a dense
    eigendecomposition.

    Parameters
    ----------
    A
        Adjacency, ``(..., n, n)`` dense or ``ELL`` / ``SectionedELL``.
    x
        Block of vectors, ``(..., n, k)``.  For a single vector
        pass ``x[..., None]`` and squeeze the result.
    normalisation
        Which Laplacian variant to apply.
    eps
        Floor on degree.
    promise_symmetry
        Sparse (:class:`~nitrix.sparse.ELL` /
        :class:`~nitrix.sparse.SectionedELL`) only.  When ``True``
        (default) the stored ``A`` matvec is applied directly.  When
        ``False`` the adjacency application is symmetrised to
        :math:`\\tfrac{1}{2}(A x + A^{\\top} x)` -- use this when the stored
        pattern is not guaranteed symmetric (e.g. top-k affinity
        sparsification).  The **degree** term still uses the row-sum of the
        stored ``A``, so for ``'symmetric'`` the result is exactly the
        normalised Laplacian of :math:`\\tfrac{1}{2}(A + A^{\\top})`; for
        ``'combinatorial'`` / ``'random_walk'`` on a genuinely asymmetric
        ``A`` the degree and the symmetrised off-diagonal are not mutually
        consistent (the out-degree :math:`D` no longer matches the
        symmetrised adjacency) -- symmetrise the adjacency at construction
        if you need that.  Ignored for dense ``A``.

    Returns
    -------
    Num[Array, '... n k']
        The product :math:`L x`, shape ``(..., n, k)``.
    """
    deg = degree_vector(A)
    safe_deg = jnp.maximum(deg, eps)

    if normalisation == 'combinatorial':
        # L x = D x - A x = deg * x - A x.
        if _is_sparse(A):
            Ax = _ell_matvec(A, x, promise_symmetry=promise_symmetry)
        else:
            Ax = jnp.matmul(A, x)
        return deg[..., None] * x - Ax
    if normalisation == 'symmetric':
        # L_sym x = x - D^(-1/2) A D^(-1/2) x
        inv_sqrt_d = (1.0 / jnp.sqrt(safe_deg))[..., None]
        scaled = inv_sqrt_d * x
        if _is_sparse(A):
            Ax = _ell_matvec(A, scaled, promise_symmetry=promise_symmetry)
        else:
            Ax = jnp.matmul(A, scaled)
        return x - inv_sqrt_d * Ax
    if normalisation == 'random_walk':
        # L_rw x = x - D^(-1) A x
        if _is_sparse(A):
            Ax = _ell_matvec(A, x, promise_symmetry=promise_symmetry)
        else:
            Ax = jnp.matmul(A, x)
        return x - Ax / safe_deg[..., None]
    raise ValueError(
        f'normalisation={normalisation!r}; expected one of '
        '"combinatorial", "symmetric", "random_walk".'
    )
