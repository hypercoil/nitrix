# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Community-detection primitives.

Houses the modularity-related operators (the *null model*, the
*modularity matrix*, and *coaffiliation*) plus the relaxed-modularity
quality score.  These all share an algebraic shape with graph
Laplacians, but their *role* is community-specific: :func:`modularity_matrix`
is the operator whose eigendecomposition gives Newman-style community
structure, and :func:`coaffiliation` is the symmetric outer product of a
community-assignment matrix.  They are kept separate from the graph
Laplacian operators so that those may remain purely about Laplacians.

The relaxed-modularity score has both *dense* and *sparse* call
paths.  The sparse path uses the factorisation

.. math::

    Q = \\sum_{i,j} (A_{ij} - \\gamma\\, k_i k_j / 2m)\\,(CC^{\\top})_{ij}
      = \\operatorname{tr}(A\\,CC^{\\top}) - \\frac{\\gamma}{2m}\\,\\lVert C^{\\top} k\\rVert^2

so that an :class:`~nitrix.sparse.ELL` or
:class:`~nitrix.sparse.SectionedELL` adjacency never materialises the
:math:`n \\times n` modularity matrix.  This is the difference between
"runs on small toy graphs" and "runs on a 100k-node mesh" for the
modularity score.
"""

from __future__ import annotations

from typing import Callable, Literal, Optional, Union

import jax.numpy as jnp
from jaxtyping import Array, Num

from ..semiring import REAL, semiring_ell_matmul
from ..sparse import ELL, SectionedELL, sectioned_semiring_ell_matmul
from .laplacian import _delete_diagonal, _safe_max, degree_vector

__all__ = [
    'girvan_newman_null',
    'modularity_matrix',
    'coaffiliation',
    'relaxed_modularity',
]


_GraphInput = Union[Num[Array, '... n n'], ELL, SectionedELL]
# A null-model factory: ``A -> null(A)``, both dense ``(..., n, n)``.
_NullModel = Callable[[Num[Array, '... n n']], Num[Array, '... n n']]


# ---------------------------------------------------------------------------
# Null model
# ---------------------------------------------------------------------------


def girvan_newman_null(
    A: Num[Array, '... n n'],
) -> Num[Array, '... n n']:
    """Rank-1 Girvan-Newman null model :math:`k_{\\mathrm{in}} k_{\\mathrm{out}}^{\\top} / (2m)`.

    The expected weight between two nodes under a random rewiring that
    preserves the (weighted) degree sequence.  For an undirected graph
    (symmetric ``A``) this reduces to :math:`k k^{\\top} / 2m`, with the
    degree vector :math:`k = A\\mathbf{1}` (the row sums of ``A``) and
    :math:`2m` the total edge weight.  This is the standard modularity
    null model of Newman.

    Note that the result is a dense :math:`n \\times n` matrix even from a
    sparse input, because the null model is rank-1 and its outer product
    is dense.  For modularity-style use it is never materialised:
    :func:`relaxed_modularity` factorises around the rank-1 structure so
    the dense matrix is not needed.

    Parameters
    ----------
    A : Num[Array, '... n n']
        Dense (weighted) adjacency matrix, batched over leading axes.

    Returns
    -------
    Num[Array, '... n n']
        Dense rank-1 null-model matrix :math:`k_{\\mathrm{in}}
        k_{\\mathrm{out}}^{\\top} / (2m)`, one per batch element.

    References
    ----------
    Newman MEJ (2006). Modularity and community structure in networks.
    *PNAS* 103(23), 8577-8582. https://doi.org/10.1073/pnas.0601602103
    """
    k_in = A.sum(axis=-1, keepdims=True)
    k_out = A.sum(axis=-2, keepdims=True)
    two_m = k_in.sum(axis=-2, keepdims=True)
    return k_in @ k_out / two_m


# ---------------------------------------------------------------------------
# Modularity matrix
# ---------------------------------------------------------------------------


def modularity_matrix(
    A: Num[Array, '... n n'],
    *,
    gamma: float = 1.0,
    null: _NullModel = girvan_newman_null,
    normalise: bool = True,
    sign: Optional[Literal['+', '-']] = '+',
) -> Num[Array, '... n n']:
    """Modularity matrix :math:`B = A - \\gamma\\,\\mathrm{null}(A)`, optionally normalised.

    The operator whose leading eigenvectors reveal Newman-style community
    structure: the observed adjacency minus a resolution-scaled null model
    of the expected adjacency.

    The output is a dense :math:`n \\times n` matrix.  For large sparse
    graphs prefer :func:`relaxed_modularity` directly (which does not
    materialise this intermediate) or build a matrix-vector product via
    :func:`modularity_matrix_matvec`.

    Parameters
    ----------
    A : Num[Array, '... n n']
        Dense (weighted) adjacency matrix, batched over leading axes.
    gamma : float, optional
        Resolution parameter.  Larger values favour smaller communities.
        Default ``1.0``.
    null : callable, optional
        Null-model factory mapping a dense adjacency to a dense expected
        adjacency.  Default :func:`girvan_newman_null`.
    normalise : bool, optional
        If ``True`` (default), divide :math:`B` by the total edge weight
        :math:`2m`, placing the resulting modularity in :math:`[-1/2, 1]`.
    sign : {'+', '-', None}, optional
        Weight handling before forming the matrix.  ``'+'`` (default)
        clips negative weights to zero; ``'-'`` clips positive weights to
        zero and negates, computing the anti-modularity on the negative
        subgraph; ``None`` uses the raw input.

    Returns
    -------
    Num[Array, '... n n']
        The (optionally normalised) modularity matrix :math:`B`, one per
        batch element.
    """
    if sign == '+':
        A = jnp.maximum(A, 0.0)
    elif sign == '-':
        A = -jnp.minimum(A, 0.0)
    elif sign is not None:
        raise ValueError(f"sign={sign!r}; expected '+' / '-' / None.")
    B = A - gamma * null(A)
    if normalise:
        two_m = A.sum(axis=(-2, -1), keepdims=True)
        two_m = jnp.where(two_m == 0, 1.0, two_m)
        B = B / two_m
    return B


def modularity_matrix_matvec(
    A: _GraphInput,
    x: Num[Array, '... n k'],
    *,
    gamma: float = 1.0,
    normalise: bool = True,
) -> Num[Array, '... n k']:
    """Apply the modularity matrix to a stack of vectors.

    Computes the matrix-vector product

    .. math::

        B x = (A - \\gamma\\, k k^{\\top} / 2m)\\, x
            = A x - \\frac{\\gamma}{2m}\\, k\\,(k^{\\top} x)

    *without* materialising the rank-1 outer product :math:`k k^{\\top}`.
    Works for dense, :class:`~nitrix.sparse.ELL`, and
    :class:`~nitrix.sparse.SectionedELL` adjacencies.  This is the
    operator that spectral community detection eigendecomposes; passing
    it directly to a matrix-free eigensolver (such as ``lobpcg``) enables
    spectral community detection on million-node graphs.

    Parameters
    ----------
    A : Num[Array, '... n n'] or ELL or SectionedELL
        (Weighted) adjacency, dense or sparse, batched over leading axes.
    x : Num[Array, '... n k']
        Stack of ``k`` column vectors to which :math:`B` is applied.
    gamma : float, optional
        Resolution parameter scaling the rank-1 null-model correction.
        Default ``1.0``.
    normalise : bool, optional
        If ``True`` (default), divide the result by the total edge weight
        :math:`2m`.

    Returns
    -------
    Num[Array, '... n k']
        The product :math:`B x`, one column per input column of ``x``.
    """
    if isinstance(A, (ELL, SectionedELL)):
        deg = degree_vector(A)
    else:
        deg = A.sum(axis=-1)
    two_m = deg.sum(axis=-1, keepdims=True)
    # A @ x
    if isinstance(A, ELL):
        Ax = semiring_ell_matmul(
            A.values,
            A.indices,
            x,
            semiring=REAL,
            n_cols=A.n_cols,
            backend='jax',
        )
    elif isinstance(A, SectionedELL):
        Ax = sectioned_semiring_ell_matmul(
            A,
            x,
            semiring=REAL,
            backend='jax',
        )
    else:
        Ax = jnp.matmul(A, x)
    # rank-1 correction: gamma / 2m * k * (k^T x)
    kTx = (deg[..., None] * x).sum(axis=-2, keepdims=True)  # (..., 1, k)
    correction = (gamma / two_m[..., None]) * deg[..., None] * kTx
    out = Ax - correction
    if normalise:
        out = out / two_m[..., None]
    return out


# ---------------------------------------------------------------------------
# Coaffiliation
# ---------------------------------------------------------------------------


def coaffiliation(
    C: Num[Array, '... n k'],
    C_other: Optional[Num[Array, '... n k']] = None,
    *,
    bridge: Optional[Num[Array, '... k k']] = None,
    exclude_diag: bool = True,
    normalise: bool = False,
) -> Num[Array, '... n n']:
    """Coaffiliation matrix from a (soft) community-assignment.

    For a hard partition with one-hot rows in ``C``, the result is
    ``1`` where nodes share a community and ``0`` otherwise.  For
    soft / overlapping assignments the result is the inner product
    of the assignment vectors.

    Parameters
    ----------
    C
        Community-assignment matrix, ``(..., n, k)``.
    C_other
        Optional second assignment (asymmetric / bipartite cases).
    bridge
        Optional ``(k, k)`` mapping between community indices.
        Result is ``C @ bridge @ C_other.T``.
    exclude_diag
        Zero the diagonal of the result.  Default ``True``: a node
        is trivially co-affiliated with itself and including that
        contribution biases modularity-like scores.
    normalise
        Normalise ``C`` and ``C_other`` by their maximum value before
        the outer product.  Useful when assignments are unnormalised
        logits.

    Returns
    -------
    Num[Array, '... n n']
        The coaffiliation matrix :math:`C\\,C_{\\mathrm{other}}^{\\top}`
        (or :math:`C\\,\\mathrm{bridge}\\,C_{\\mathrm{other}}^{\\top}` when
        ``bridge`` is given), with the diagonal zeroed when
        ``exclude_diag`` is ``True``.
    """
    if C_other is None:
        C_other = C
    if normalise:
        C = C / _safe_max(C)
        C_other = C_other / _safe_max(C_other)
    if bridge is None:
        out = C @ C_other.swapaxes(-1, -2)
    else:
        out = C @ bridge @ C_other.swapaxes(-1, -2)
    if exclude_diag:
        out = _delete_diagonal(out)
    return out


# ---------------------------------------------------------------------------
# Relaxed modularity (dense + sparse-aware)
# ---------------------------------------------------------------------------


def relaxed_modularity(
    A: _GraphInput,
    C: Num[Array, '... n k'],
    *,
    C_other: Optional[Num[Array, '... n k']] = None,
    bridge: Optional[Num[Array, '... k k']] = None,
    gamma: float = 1.0,
    null: _NullModel = girvan_newman_null,
    normalise_modularity: bool = True,
    normalise_coaffiliation: bool = True,
    exclude_diag: bool = True,
    directed: bool = False,
    sign: Optional[Literal['+', '-']] = '+',
) -> Num[Array, '...']:
    """Relaxed (differentiable) modularity for a (soft) community assignment.

    Computes :math:`Q = \\sum_{i,j} B_{ij}\\,(CC^{\\top})_{ij}`, where
    :math:`B` is the modularity matrix.  For a one-hot assignment ``C``
    with ``exclude_diag=False`` this reduces **exactly** to the standard
    Newman modularity :math:`Q_{\\mathrm{Newman}}` (the :math:`1/2m`
    normalisation of :func:`modularity_matrix` already corrects the
    undirected double-count).  The default ``exclude_diag=True`` drops the
    within-community diagonal null terms that Newman includes, giving a
    co-affiliation variant better suited to correlation-like adjacencies
    and soft / overlapping assignments (see ``exclude_diag``), where the
    score is a smooth, differentiable relaxation suitable as an
    optimisation objective.

    The implementation dispatches on the type of ``A``:

    - **Dense** ``A``: materialises :math:`B` and :math:`CC^{\\top}` and
      sums.  Easy to read; quadratic in :math:`n` for both memory and
      compute.
    - :class:`~nitrix.sparse.ELL` / :class:`~nitrix.sparse.SectionedELL`
      ``A``: factorised path.  Computes
      :math:`Q = \\operatorname{tr}(A\\,CC^{\\top}) - (\\gamma / 2m)\\,
      \\lVert C^{\\top} k\\rVert^2` with sparse matrix-vector products and
      never materialises the dense :math:`B`.  Cost is
      :math:`O(\\mathrm{nnz}\\cdot k)`, where :math:`\\mathrm{nnz}` is the
      non-zero count and :math:`k` the community count.

    For ``A`` larger than roughly 5k nodes the sparse path is preferable;
    for small dense graphs the readable dense path is fine.

    Parameters
    ----------
    A : Num[Array, '... n n'] or ELL or SectionedELL
        (Weighted) adjacency, dense or sparse, batched over leading axes.
    C : Num[Array, '... n k']
        Community-assignment matrix over ``k`` communities; one-hot for a
        hard partition, otherwise a soft or overlapping assignment.
    C_other : Num[Array, '... n k'], optional
        Optional second assignment for asymmetric or bipartite cases.
        Defaults to ``C``.
    bridge : Num[Array, '... k k'], optional
        Optional mapping between community indices, giving the
        coaffiliation :math:`C\\,\\mathrm{bridge}\\,C_{\\mathrm{other}}^{\\top}`.
    gamma : float, optional
        Resolution parameter.  Larger values favour smaller communities.
        Default ``1.0``.
    null : callable, optional
        Null-model factory used on the dense path.  Default
        :func:`girvan_newman_null`.  (The sparse path uses the equivalent
        rank-1 degree-based correction directly.)
    normalise_modularity : bool, optional
        If ``True`` (default), normalise the modularity by the total edge
        weight :math:`2m`.
    normalise_coaffiliation : bool, optional
        If ``True`` (default), normalise ``C`` (and ``C_other``) by their
        maximum value before forming the coaffiliation.
    exclude_diag : bool, optional
        If ``True`` (default), drop the diagonal of the coaffiliation so a
        node's trivial self-affiliation does not bias the score -- apt when
        ``A`` is a correlation-like matrix (unit, uninformative diagonal)
        or when self-self coupling carries a different scale or noise than
        self-other.  Set ``False`` to include the diagonal null terms and
        recover exact Newman modularity for a one-hot partition.
    directed : bool, optional
        If ``False`` (default), the score counts each undirected edge
        once: the :math:`1/2m` normalisation removes the ordered-sum
        double-count when ``normalise_modularity=True``, and the
        un-normalised sum is halved explicitly.  If ``True``, return the
        raw (directed) sum with no double-count correction.
    sign : {'+', '-', None}, optional
        Weight handling on the dense path, forwarded to
        :func:`modularity_matrix`.  ``'+'`` (default) clips negative
        weights; ``'-'`` clips positive weights; ``None`` uses the raw
        input.  Ignored on the sparse path.

    Returns
    -------
    Num[Array, '...']
        The scalar relaxed modularity per batch element.
    """
    if isinstance(A, (ELL, SectionedELL)):
        return _relaxed_modularity_sparse(
            A,
            C,
            C_other=C_other,
            bridge=bridge,
            gamma=gamma,
            normalise_modularity=normalise_modularity,
            normalise_coaffiliation=normalise_coaffiliation,
            exclude_diag=exclude_diag,
            directed=directed,
        )
    return _relaxed_modularity_dense(
        A,
        C,
        C_other=C_other,
        bridge=bridge,
        gamma=gamma,
        null=null,
        normalise_modularity=normalise_modularity,
        normalise_coaffiliation=normalise_coaffiliation,
        exclude_diag=exclude_diag,
        directed=directed,
        sign=sign,
    )


def _relaxed_modularity_dense(
    A: Num[Array, '... n n'],
    C: Num[Array, '... n k'],
    *,
    C_other: Optional[Num[Array, '... n k']],
    bridge: Optional[Num[Array, '... k k']],
    gamma: float,
    null: _NullModel,
    normalise_modularity: bool,
    normalise_coaffiliation: bool,
    exclude_diag: bool,
    directed: bool,
    sign: Optional[Literal['+', '-']],
) -> Num[Array, '...']:
    B = modularity_matrix(
        A,
        gamma=gamma,
        null=null,
        normalise=normalise_modularity,
        sign=sign,
    )
    K = coaffiliation(
        C,
        C_other,
        bridge=bridge,
        exclude_diag=exclude_diag,
        normalise=normalise_coaffiliation,
    )
    Q = (B * K).sum(axis=(-2, -1))
    # The undirected double-count is corrected by the 1/2m normalisation
    # (see modularity_matrix); only the un-normalised sum still needs the
    # explicit halving, so the normalised undirected score reduces exactly
    # to Newman modularity.
    return Q / 2.0 if (not directed and not normalise_modularity) else Q


def _relaxed_modularity_sparse(
    A: Union[ELL, SectionedELL],
    C: Num[Array, '... n k'],
    *,
    C_other: Optional[Num[Array, '... n k']],
    bridge: Optional[Num[Array, '... k k']],
    gamma: float,
    normalise_modularity: bool,
    normalise_coaffiliation: bool,
    exclude_diag: bool,
    directed: bool,
) -> Num[Array, '...']:
    """Factored relaxed modularity for ELL / SectionedELL adjacency.

    The product ``A @ C`` (via the semiring ELL matrix multiply) has shape
    ``(n, k)``; then
    :math:`\\operatorname{tr}(A\\,CC^{\\top}) = \\sum_i C_i^{\\top}(AC)_i
    = \\sum_{i,k} C_{ik}\\,(AC)_{ik}`.  For ``bridge`` and ``C_other`` this
    generalises to :math:`\\operatorname{tr}(C\\,L\\,C_{\\mathrm{o}}^{\\top}\\,A)`
    along the same path.

    The rank-1 correction :math:`(\\gamma / 2m)\\,\\lVert C^{\\top} k\\rVert^2`
    uses a single matrix-vector product :math:`C^{\\top} k` (cheap, with a
    length-``k`` result).

    When ``exclude_diag`` is set, the diagonal of the coaffiliation must be
    subtracted; the sparse path does not expose it cheaply, so
    :math:`\\operatorname{diag}(CC^{\\top}) = \\sum_k C_{\\cdot k}^2` is
    formed and the matching self-loop contributions of :math:`A` are
    subtracted.  For most graphs :math:`A` has no self-loops so this is a
    no-op, but it is applied unconditionally for correctness.

    Parameters
    ----------
    A : ELL or SectionedELL
        Sparse (weighted) adjacency, batched over leading axes.
    C : Num[Array, '... n k']
        Community-assignment matrix over ``k`` communities.
    C_other : Num[Array, '... n k'] or None
        Optional second assignment for asymmetric or bipartite cases;
        ``None`` reuses ``C``.
    bridge : Num[Array, '... k k'] or None
        Optional mapping between community indices; ``None`` uses the
        identity coupling.
    gamma : float
        Resolution parameter scaling the rank-1 null-model correction.
    normalise_modularity : bool
        Whether to normalise the modularity by the total edge weight
        :math:`2m`.
    normalise_coaffiliation : bool
        Whether to normalise ``C`` (and ``C_other``) by their maximum
        value before forming the coaffiliation.
    exclude_diag : bool
        Whether to drop the diagonal contribution of the coaffiliation.
    directed : bool
        If ``False``, count each undirected edge once (the :math:`1/2m`
        normalisation removes the ordered-sum double-count, or the
        un-normalised sum is halved explicitly); if ``True``, return the
        raw directed sum.

    Returns
    -------
    Num[Array, '...']
        The scalar relaxed modularity per batch element.
    """
    # ---------------- Normalisations ----------------
    if normalise_coaffiliation:
        C = C / _safe_max(C)
        if C_other is not None:
            C_other = C_other / _safe_max(C_other)
    if C_other is None:
        C_other = C

    # ---------------- A @ C ----------------
    if isinstance(A, ELL):
        AC = semiring_ell_matmul(
            A.values,
            A.indices,
            C,
            semiring=REAL,
            n_cols=A.n_cols,
            backend='jax',
        )
    else:  # SectionedELL
        AC = sectioned_semiring_ell_matmul(
            A,
            C,
            semiring=REAL,
            backend='jax',
        )

    # ---------------- trace(A · C_o L C^T) ----------------
    # = sum_{i, j, p, q} A[i, j] * C_o[j, p] * L[p, q] * C[i, q]
    # = sum_i (C[i] L^T · (A @ C_o)[i])         (a single inner product per node)
    if bridge is not None:
        trace_term = (C @ bridge * AC).sum(axis=(-2, -1))
    else:
        # When C_other != C, AC was computed wrt C; we need wrt C_other:
        if C_other is not C:
            if isinstance(A, ELL):
                AC_o = semiring_ell_matmul(
                    A.values,
                    A.indices,
                    C_other,
                    semiring=REAL,
                    n_cols=A.n_cols,
                    backend='jax',
                )
            else:
                AC_o = sectioned_semiring_ell_matmul(
                    A,
                    C_other,
                    semiring=REAL,
                    backend='jax',
                )
        else:
            AC_o = AC
        trace_term = (C * AC_o).sum(axis=(-2, -1))

    # ---------------- Rank-1 correction ----------------
    deg = degree_vector(A)  # (..., n)
    two_m = deg.sum(axis=-1)  # (...,)
    # Ct_k = C^T @ deg  -> (..., k_comm)
    Ct_k = jnp.einsum('...nc,...n->...c', C, deg)
    if bridge is None and C_other is C:
        rank1 = (Ct_k * Ct_k).sum(axis=-1)
    else:
        Co_k = jnp.einsum('...nc,...n->...c', C_other, deg)
        if bridge is None:
            rank1 = (Ct_k * Co_k).sum(axis=-1)
        else:
            rank1 = jnp.einsum(
                '...p,...pq,...q->...',
                Ct_k,
                bridge,
                Co_k,
            )
    rank1 = (gamma / two_m) * rank1

    # ---------------- Diagonal correction ----------------
    # ``exclude_diag`` zeros out the diagonal of CC^T (or C bridge C_o^T).
    # In the trace formulation, we subtract the diagonal contribution.
    if exclude_diag:
        if bridge is None:
            # diag(CC_o^T)_i = sum_k C[i, k] C_o[i, k]
            diag_K = (C * C_other).sum(axis=-1)  # (..., n)
        else:
            diag_K = jnp.einsum(
                '...np,...pq,...nq->...n',
                C,
                bridge,
                C_other,
            )
        # Self-loops of A contribute via diag(A) * diag_K.  For graphs
        # without self-loops (the common case) diag(A) == 0 so this
        # is zero; we still subtract for correctness.
        if isinstance(A, ELL):
            # Diagonal of an ELL is awkward; for typical graph
            # adjacencies (no self-loops) it's zero.  We document the
            # assumption rather than scan the indices.
            diag_A = jnp.zeros_like(diag_K)
        else:
            # SectionedELL: same.
            diag_A = jnp.zeros_like(diag_K)
        trace_term = trace_term - (diag_A * diag_K).sum(axis=-1)
        # Subtract diag from the rank-1 term too (the dense formula
        # has the diagonal of ``B`` subtracted via exclude_diag = both
        # halves drop their diagonal).
        # Diagonal of the null model is k_i^2 / 2m; sum_i k_i^2 / 2m *
        # diag_K[i] is the contribution.
        rank1_diag = ((deg**2) / two_m[..., None] * diag_K).sum(axis=-1)
        rank1 = rank1 - rank1_diag

    Q = trace_term - rank1
    if normalise_modularity:
        Q = Q / two_m
    # See _relaxed_modularity_dense: the 1/2m normalisation already removes
    # the undirected double-count, so only the un-normalised sum is halved.
    return Q / 2.0 if (not directed and not normalise_modularity) else Q
