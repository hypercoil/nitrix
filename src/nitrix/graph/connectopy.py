# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Low-dimensional embeddings of graph nodes from spectral decompositions
of (modified) Laplacians.

This module provides two node embeddings built on the spectrum of a
normalised graph affinity operator.  Both map each node to a point in a
low-dimensional Euclidean space whose coordinates are the leading
non-trivial eigenvectors of that operator, so that proximity in the
embedding reflects connectivity on the graph.

- :func:`laplacian_eigenmap` -- the Laplacian eigenmap of Belkin and
  Niyogi.  Embeds nodes in the bottom-``k`` non-trivial eigenspace of
  the symmetric normalised Laplacian.
- :func:`diffusion_embedding` -- the diffusion map of Coifman and Lafon.
  Embeds nodes in the top-``k`` non-trivial eigenspace of the
  density-normalised diffusion operator, scaled by :math:`\\lambda^{t}`.

Both build the symmetric affinity operator :math:`M` (dense,
:class:`~nitrix.sparse.ELL`, or :class:`~nitrix.sparse.SectionedELL`) and
hand the extremal top-``k`` eigendecomposition to the dedicated
dispatcher :func:`~nitrix.linalg._eigsolve.eigsolve_top_k`.  The
embedding-specific spectral conventions -- skipping the trivial constant
eigenvector, mapping :math:`\\lambda_L = 1 - \\lambda_M` for the
Laplacian, recovering the random-walk right eigenvectors, and scaling by
:math:`\\lambda^{t}` for the diffusion map -- are applied here,
uniformly, on the dispatcher's largest-first return; the solver choice
does not change that post-processing.

The solver knobs (``solver`` / ``preconditioner``) map onto the
dispatcher's :class:`~nitrix.linalg._eigsolve.SolverSpec`:

- ``"eigh"`` -- dense full spectrum, sliced to the top-``k``.  Required
  for dense input; not available for
  :class:`~nitrix.sparse.ELL` / :class:`~nitrix.sparse.SectionedELL`.
- ``"lobpcg"`` -- iterative top-``k``; the default for sparse input,
  scaling to ``n ~ 1M+``.
- ``"shift_invert"`` -- matrix-free shift-invert (far fewer outer
  iterations on clustered low-Laplacian spectra; dense,
  :class:`~nitrix.sparse.ELL`, or :class:`~nitrix.sparse.SectionedELL`).
- ``preconditioner="polynomial"`` -- the matvec-only shifted-power
  spectral filter (dense,
  :class:`~nitrix.sparse.ELL`, or :class:`~nitrix.sparse.SectionedELL`).
- ``"auto"`` (default) -- ``eigh`` for dense, ``lobpcg`` for sparse.
  (Shift-invert and polynomial on sparse input are served only on
  explicit request, not auto-selected.)

Differentiability
-----------------

- ``"eigh"``: differentiable via the registered VJP of ``jnp.linalg.eigh``.
- ``"lobpcg"`` / ``"shift_invert"`` / ``"polynomial"`` with dense,
  :class:`~nitrix.sparse.ELL`, or :class:`~nitrix.sparse.SectionedELL`
  input: differentiable via the implicit VJP of the eigensolver (a
  Hellmann-Feynman eigenvalue gradient plus the F-matrix subspace
  projector for the eigenvector gradient; exact for in-subspace losses,
  biased for losses on the orthogonal complement of the returned
  top-``k`` subspace; near-degenerate denominators are clamped).  For
  :class:`~nitrix.sparse.ELL` the gradient is projected onto the sparsity
  pattern; for :class:`~nitrix.sparse.SectionedELL` onto each section's
  ``values``.

Every required step is expressible in plain JAX.

References
----------
.. [1] Belkin M, Niyogi P (2003). Laplacian eigenmaps for dimensionality
       reduction and data representation. Neural Computation 15(6),
       1373-1396. https://doi.org/10.1162/089976603321780317
.. [2] Coifman RR, Lafon S (2006). Diffusion maps. Applied and
       Computational Harmonic Analysis 21(1), 5-30.
       https://doi.org/10.1016/j.acha.2006.04.006
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple, Union

import jax.numpy as jnp
from jaxtyping import Array, Float, Num

from ..linalg._eigsolve import SolverSpec, eigsolve_top_k
from ..sparse import ELL, SectionedELL
from .laplacian import _is_sparse, degree_vector, symmetric_degree_vector

__all__ = ['laplacian_eigenmap', 'diffusion_embedding']


_Solver = Literal['auto', 'eigh', 'lobpcg', 'shift_invert']
_Preconditioner = Literal['none', 'polynomial']
_GraphInput = Union[Num[Array, '... n n'], ELL, SectionedELL]

# Shift-invert solver tuning (dense path).  A small negative shift keeps
# ``(L - sigma I)`` SPD/CG-solvable; these defaults reach ~1e-4 eigenvalue
# accuracy in ~15 outer x 12 inner-CG iterations (~2x off cupy eigsh on the
# clustered Laplacian spectrum at n=1024, vs ~12x for plain lobpcg).
_SI_SIGMA = -0.5
_SI_OUTER_ITERS = 15
_SI_CG_ITERS = 12

# Polynomial preconditioner tuning (the matvec-only spectral filter on the
# plain LOBPCG path).  LOBPCG runs on ``(M + shift I)^degree`` to amplify the
# wanted top of the spectrum; these defaults reach ~1e-3 accuracy in ~16 outer
# iterations (~1.5-2x off cupy at n=1024 -- faster than shift-invert since it
# is matvec-only, no inner solve).  ``shift = 1`` makes ``M + shift I`` PSD for
# the normalized affinity (eigenvalues in [-1, 1]).
_POLY_DEGREE = 4
_POLY_SHIFT = 1.0
_POLY_OUTER_ITERS = 16

# Default LOBPCG residual tolerance.  ``lobpcg_standard`` runs to its iteration
# cap unless a tolerance lets it stop on convergence; the Laplacian's smallest
# eigenvalues are tightly clustered, so without this it wastes ~half its
# iterations past convergence.  ``1e-7`` early-stops at ~1e-5 eigenvalue
# accuracy (~2.5-4x faster than running to the cap on the L4) -- ample for
# spectral embedding, and differentiability is unaffected (the iteration loop
# lives inside the implicit-VJP ``custom_vjp`` forward, never autodiffed).
# In fp32 the achievable residual floor (~1e-6) is above this, so the fp32
# path simply runs to the cap as before (full accuracy).  Loosen via
# ``lobpcg_tol=`` to trade accuracy for speed.
_LOBPCG_DEFAULT_TOL = 1e-7


# ---------------------------------------------------------------------------
# Solver-spec mapping
# ---------------------------------------------------------------------------


def _spec_from(
    solver: _Solver,
    preconditioner: _Preconditioner,
    *,
    lobpcg_iters: int,
    lobpcg_tol: Optional[float],
    promise_symmetry: bool,
) -> SolverSpec:
    """Map the ``(solver, preconditioner)`` vocabulary to a solver spec.

    Translates this module's ``(solver, preconditioner)`` selection into a
    :class:`~nitrix.linalg._eigsolve.SolverSpec` for
    :func:`~nitrix.linalg._eigsolve.eigsolve_top_k`.

    ``preconditioner='polynomial'`` selects the matvec-only spectral filter
    in place of plain LOBPCG -- it overrides ``'auto'`` / ``'eigh'`` /
    ``'lobpcg'`` but not an explicit ``'shift_invert'`` (the shift-invert
    path ignores the preconditioner).  ``'auto'`` is deferred to the
    dispatcher's format-based policy (dense to ``eigh``, sparse to
    ``lobpcg``), carrying the LOBPCG knobs in case it resolves to
    ``lobpcg``.

    Parameters
    ----------
    solver
        Requested solver family: ``'auto'``, ``'eigh'``, ``'lobpcg'``, or
        ``'shift_invert'``.
    preconditioner
        LOBPCG acceleration: ``'none'`` or ``'polynomial'``.  When
        ``'polynomial'`` and ``solver`` is not ``'shift_invert'``, the
        polynomial-filter spec is returned regardless of ``solver``.
    lobpcg_iters
        Maximum LOBPCG iterations, forwarded to the ``lobpcg`` and
        ``auto`` specs.
    lobpcg_tol
        LOBPCG residual tolerance (or ``None`` for the JAX default),
        forwarded to the ``lobpcg`` and ``auto`` specs.
    promise_symmetry
        Whether the caller guarantees the operator is symmetric.  Forwarded
        to every iterative spec; a no-op for dense ``eigh``, whose operator
        is symmetrised when built.

    Returns
    -------
    SolverSpec
        The solver specification passed to
        :func:`~nitrix.linalg._eigsolve.eigsolve_top_k`.
    """
    if preconditioner == 'polynomial' and solver != 'shift_invert':
        return SolverSpec.poly(
            degree=_POLY_DEGREE,
            shift=_POLY_SHIFT,
            outer_iters=_POLY_OUTER_ITERS,
            promise_symmetry=promise_symmetry,
        )
    if solver == 'eigh':
        return SolverSpec.eigh()
    if solver == 'shift_invert':
        return SolverSpec.shift_invert(
            sigma=_SI_SIGMA,
            outer_iters=_SI_OUTER_ITERS,
            cg_iters=_SI_CG_ITERS,
            promise_symmetry=promise_symmetry,
        )
    if solver == 'lobpcg':
        return SolverSpec.lobpcg(
            n_iters=lobpcg_iters,
            tol=lobpcg_tol,
            promise_symmetry=promise_symmetry,
        )
    return SolverSpec.auto(
        n_iters=lobpcg_iters,
        tol=lobpcg_tol,
        promise_symmetry=promise_symmetry,
    )


# ---------------------------------------------------------------------------
# Internal: build the normalised-affinity operator
# ---------------------------------------------------------------------------


def _scale_by_outer(
    A: _GraphInput,
    left: Float[Array, '... n'],
    right: Float[Array, '... n'],
) -> _GraphInput:
    """Scale entry :math:`A_{ij}` by ``left[i] * right[j]``.

    Applies a two-sided diagonal scaling to a dense,
    :class:`~nitrix.sparse.ELL`, or :class:`~nitrix.sparse.SectionedELL`
    operand.  The sparsity pattern is preserved (diagonal scaling on both
    sides is structure-preserving); a new operand of the same type is
    returned with only its values scaled.

    Parameters
    ----------
    A
        The operator to scale: dense ``(..., n, n)``,
        :class:`~nitrix.sparse.ELL`, or :class:`~nitrix.sparse.SectionedELL`.
    left
        Left (row) scaling vector, shape ``(..., n)``.
    right
        Right (column) scaling vector, shape ``(..., n)``.

    Returns
    -------
    _GraphInput
        The scaled operator, same type and sparsity pattern as ``A``.
    """
    if isinstance(A, ELL):
        right_at_idx = right[A.indices]
        return ELL(
            values=A.values * (left[:, None] * right_at_idx),
            indices=A.indices,
            n_cols=A.n_cols,
            identity=A.identity,
        )
    if isinstance(A, SectionedELL):
        new_sections = []
        for ell, row_idx in zip(A.sections, A.row_groups):
            left_rows = left[jnp.asarray(row_idx)]
            right_at = right[ell.indices]
            new_sections.append(
                ELL(
                    values=ell.values * (left_rows[:, None] * right_at),
                    indices=ell.indices,
                    n_cols=ell.n_cols,
                    identity=ell.identity,
                )
            )
        return SectionedELL(
            sections=tuple(new_sections),
            row_groups=A.row_groups,
            n_rows=A.n_rows,
            n_cols=A.n_cols,
            identity=A.identity,
        )
    return A * left[..., :, None] * right[..., None, :]


def _build_affinity_operator(
    A: _GraphInput,
    *,
    alpha: float = 0.0,
    eps: float = 1e-12,
    promise_symmetry: bool = False,
) -> Tuple[_GraphInput, Float[Array, '... n']]:
    """Build the normalised affinity operator and its inverse-sqrt degree.

    The graph is treated as the symmetrised adjacency
    :math:`W = \\tfrac{1}{2}(A + A^{\\top})` (``A`` need not be symmetric),
    so every degree below is the *symmetric* degree
    :math:`\\tfrac{1}{2}(\\mathrm{out} + \\mathrm{in})` of :math:`W` -- not
    the row-sum out-degree of :math:`A` (which would normalise by the wrong
    diagonal and shift the trivial eigenvalue off :math:`1`).

    For ``alpha == 0`` the operator is
    :math:`M = D^{-1/2} A D^{-1/2}`, symmetrised to
    :math:`D_W^{-1/2} W D_W^{-1/2}` (the affinity matrix of the symmetric
    normalised Laplacian of :math:`W`).  For ``alpha > 0`` the operator is
    the Coifman-Lafon diffusion operator
    :math:`M_\\alpha = D_2^{-1/2} K_\\alpha D_2^{-1/2}`, where
    :math:`K_\\alpha = A / (d^{\\alpha} (d^{\\alpha})^{\\top})` is the
    density-normalised affinity and :math:`d_2` is its row sums.

    Parameters
    ----------
    A
        Adjacency / affinity matrix: dense ``(..., n, n)``,
        :class:`~nitrix.sparse.ELL`, or :class:`~nitrix.sparse.SectionedELL`.
    alpha
        Density-normalisation exponent.  ``0`` yields the plain symmetric
        normalised affinity; ``> 0`` yields the density-normalised diffusion
        operator.  Default ``0``.
    eps
        Degree floor, so that isolated nodes do not divide by zero.
        Default ``1e-12``.
    promise_symmetry
        Sparse inputs only.  When ``True`` the adjacency is asserted
        symmetric, so the plain out-degree is used and the adjoint
        (in-degree) pass is skipped; when ``False`` (default) the symmetric
        degree is used, matching the matvec symmetrisation applied by the
        solver.  Ignored for dense ``A`` (always symmetrised).

    Returns
    -------
    M : _GraphInput
        The concrete normalised affinity operator (same format as ``A``,
        not a closure), passed directly to
        :func:`~nitrix.linalg._eigsolve.eigsolve_top_k`.
    inv_sqrt_d2 : Float[Array, '... n']
        The inverse square root of the (density-normalised) degree, shape
        ``(..., n)``.  Multiplying the eigenvectors of :math:`M_\\alpha` by
        it recovers the right eigenvectors of the random-walk diffusion
        operator :math:`P`.  Ignored on the Laplacian-eigenmap path
        (``alpha = 0``).
    """
    # The normalisation degree MUST match the operator symmetrisation, and both
    # are governed by the same ``promise_symmetry`` opt-in:
    #
    # - symmetrised operator (dense -- always; or sparse with the default
    #   ``promise_symmetry=False``, where the solver applies ½(A x + Aᵀ x)):
    #   normalise by the *symmetric* degree d_W = ½(out + in) of W = ½(A + Aᵀ),
    #   so the operator is the true ``D_W^{-1/2} W D_W^{-1/2}`` (trivial
    #   eigenvalue exactly 1).  Using the bare out-degree here would symmetrise
    #   the *Laplacian* while normalising by the wrong diagonal.
    # - bare operator (sparse, ``promise_symmetry=True``): the caller asserts
    #   A == Aᵀ, so out == in == d_W; use the plain out-degree and skip the
    #   in-degree adjoint pass -- the perf opt-in is honoured on the degree too,
    #   and the result is identical for the (promised) symmetric input.
    symmetrised = (not _is_sparse(A)) or (not promise_symmetry)
    degree_fn = symmetric_degree_vector if symmetrised else degree_vector
    deg = degree_fn(A)
    safe_deg = jnp.maximum(deg, eps)

    if alpha != 0.0:
        d_alpha = safe_deg**alpha
        inv_d_alpha = 1.0 / d_alpha
        K = _scale_by_outer(A, inv_d_alpha, inv_d_alpha)
        safe_d2 = jnp.maximum(degree_fn(K), eps)
    else:
        K = A
        safe_d2 = safe_deg

    inv_sqrt_d2 = 1.0 / jnp.sqrt(safe_d2)
    M = _scale_by_outer(K, inv_sqrt_d2, inv_sqrt_d2)
    # Dense M is symmetrised here (cheap; also corrects float drift).  A
    # *sparse* M carries only whatever symmetry its stored pattern has --
    # which top-k affinity construction (``ell_from_dense``) does NOT
    # guarantee.  Symmetry of the sparse operator is therefore handled in
    # the eigensolver via ``promise_symmetry=False`` (the ``½(A x + Aᵀ x)``
    # matvec), threaded from the public API below.  Because ``inv_sqrt_d2``
    # is a symmetric diagonal scaling (built from the *symmetric* degree
    # ``d_W``), that matvec yields exactly the operator this dense branch
    # builds: ``D_W^{-1/2} sym(A) D_W^{-1/2} = D_W^{-1/2} W D_W^{-1/2}`` -- the
    # genuine symmetric normalised affinity of ``W = ½(A + Aᵀ)``.
    if not _is_sparse(M):
        M = 0.5 * (M + M.swapaxes(-1, -2))
    return M, inv_sqrt_d2


def _affinity_top_k(
    A: _GraphInput,
    *,
    alpha: float,
    n_components: int,
    skip_trivial: bool,
    eps: float,
    spec: SolverSpec,
    seed: int,
    promise_symmetry: bool,
) -> Tuple[
    Float[Array, 'n_components'],
    Float[Array, 'n n_components'],
    Float[Array, '... n'],
]:
    """Build the affinity operator and return its top non-trivial eigenpairs.

    Constructs the normalised affinity operator, solves for its
    largest-first extremal eigenpairs, and drops the trivial leading
    eigenpair (the constant eigenvector, the *largest* affinity eigenvalue)
    when ``skip_trivial`` is set.  The convention-specific transforms
    (:math:`1 - \\mu`, :math:`\\lambda^{t}`, right-eigenvector recovery) are
    left to the caller.

    Parameters
    ----------
    A
        Adjacency / affinity matrix: dense ``(..., n, n)``,
        :class:`~nitrix.sparse.ELL`, or :class:`~nitrix.sparse.SectionedELL`.
    alpha
        Density-normalisation exponent passed to the affinity operator.
    n_components
        Number of non-trivial eigenpairs to return.
    skip_trivial
        Drop the trivial leading eigenpair before returning.  When set, one
        extra eigenpair is solved for and discarded.
    eps
        Degree floor for the normalisation.
    spec
        The eigensolver specification
        (:class:`~nitrix.linalg._eigsolve.SolverSpec`).
    seed
        PRNG seed for the iterative solver's initial guess.
    promise_symmetry
        Whether to assume the adjacency is symmetric; selects the
        normalisation degree (symmetric vs out-degree) to match the solver's
        matvec.

    Returns
    -------
    vals : Float[Array, 'n_components']
        The non-trivial eigenvalues of the affinity operator, largest-first,
        shape ``(n_components,)``.
    vecs : Float[Array, 'n n_components']
        The corresponding eigenvectors, shape ``(n, n_components)``.
    inv_sqrt_d2 : Float[Array, '... n']
        The inverse square root of the (density-normalised) degree, shape
        ``(..., n)``, for random-walk right-eigenvector recovery.
    """
    M, inv_sqrt_d2 = _build_affinity_operator(
        A, alpha=alpha, eps=eps, promise_symmetry=promise_symmetry
    )
    k_total = n_components + (1 if skip_trivial else 0)
    vals, vecs = eigsolve_top_k(M, k_total, spec=spec, seed=seed)
    # ``eigsolve_top_k`` returns largest-first; the trivial eigenpair is the
    # leading (largest) one.
    if skip_trivial:
        vals = vals[1:]
        vecs = vecs[:, 1:]
    return vals, vecs, inv_sqrt_d2


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def laplacian_eigenmap(
    A: _GraphInput,
    *,
    n_components: int,
    solver: _Solver = 'auto',
    preconditioner: _Preconditioner = 'none',
    promise_symmetry: bool = False,
    skip_trivial: bool = True,
    eps: float = 1e-12,
    lobpcg_iters: int = 200,
    lobpcg_tol: Optional[float] = _LOBPCG_DEFAULT_TOL,
    seed: int = 0,
) -> Tuple[
    Float[Array, 'n n_components'],
    Float[Array, 'n_components'],
]:
    """Laplacian-eigenmap embedding of graph nodes.

    Embeds nodes in the bottom-``n_components`` non-trivial eigenspace of
    the symmetric normalised Laplacian
    :math:`L = I - D^{-1/2} A D^{-1/2}`.  The smallest eigenvalue is the
    trivial zero (constant eigenvector); ``skip_trivial=True`` (default)
    discards it.

    Parameters
    ----------
    A
        Adjacency matrix: dense ``(n, n)``, :class:`~nitrix.sparse.ELL`, or
        :class:`~nitrix.sparse.SectionedELL`.  Non-negative.  Treated as
        symmetric: dense is symmetrised when the operator is built, and
        sparse inputs are symmetrised at the matvec by default
        (``promise_symmetry=False``), so a top-``k``-sparsified affinity
        needs no pre-symmetrisation.
    n_components
        Embedding dimensionality.
    solver
        ``"eigh"`` (dense full spectrum), ``"lobpcg"`` (iterative
        top-k), ``"shift_invert"`` (matrix-free shift-invert via inner CG --
        far fewer outer iterations than ``lobpcg`` on the clustered
        low-Laplacian spectrum, ~2x off cupy ``eigsh``; dense / ELL /
        SectionedELL, differentiable via the same implicit VJP), or
        ``"auto"`` (default: ``lobpcg`` for sparse, ``eigh`` for dense).
    preconditioner
        Acceleration for the LOBPCG path: ``"none"`` (default) or
        ``"polynomial"`` -- a matvec-only shifted-power spectral filter
        (LOBPCG runs on ``(M + I)^degree`` to amplify the wanted top of the
        spectrum, eigenvalues recovered by Rayleigh quotient).  Faster than
        both plain ``lobpcg`` and ``shift_invert`` on clustered spectra
        (~1.5-2x off cupy), differentiable via the same implicit VJP; dense
        / ELL / SectionedELL.  Requesting it routes the solve to the
        polynomial-filter method so the preconditioner is honoured.
    promise_symmetry
        Sparse (:class:`~nitrix.sparse.ELL` /
        :class:`~nitrix.sparse.SectionedELL`) inputs only.  ``False``
        (default) symmetrises the operator at the matvec level -- the solver
        applies
        :math:`\\tfrac{1}{2}(A x + A^{\\top} x) = D_W^{-1/2}\\,\\operatorname{sym}(A)\\,D_W^{-1/2} x`,
        matching the dense path exactly -- and normalises by the *symmetric
        degree* :math:`d_W = \\tfrac{1}{2}(\\mathrm{out} + \\mathrm{in})` of
        :math:`W = \\tfrac{1}{2}(A + A^{\\top})`.  This is the safe choice for
        affinities built by top-``k``-per-row sparsification, whose stored
        pattern is generally *not* symmetric.  Set ``True`` only when the
        adjacency is known symmetric (regular meshes / grids): the solver
        applies the bare :math:`A x` and the normalisation uses the plain
        out-degree, skipping both adjoint passes (roughly twice as cheap).
        Because :math:`A = A^{\\top}` is then promised, out-degree equals
        :math:`d_W`, so the result is identical to the default path for a
        genuinely symmetric input; on an asymmetric input it is the caller's
        contract violation.  Ignored for dense ``A`` (always symmetrised,
        always the symmetric degree).
    skip_trivial
        Drop the trivial zero eigenvalue and the corresponding
        constant eigenvector.  Default ``True``.
    eps
        Degree floor (isolated nodes don't divide by zero).
    lobpcg_iters
        Maximum iterations for the iterative solver.  Default 200.
        Increase if convergence warnings appear; typical convergence
        is < 50 iterations.
    lobpcg_tol
        LOBPCG residual tolerance.  Default ``1e-7`` (early-stops at ~1e-5
        eigenvalue accuracy, ~2.5-4x faster than running to the iteration cap);
        loosen to trade accuracy for speed, or pass ``None`` for the JAX
        default (runs to the cap).  Differentiability is unaffected (the
        iteration loop is inside the implicit-VJP forward).
    seed
        PRNG seed for the lobpcg initial guess.

    Returns
    -------
    ``(embedding, eigenvalues)`` of shapes ``(n, n_components)`` and
    ``(n_components,)``.  Eigenvalues are sorted smallest-first
    (the Laplacian convention, *not* the diffusion-map convention).

    Notes
    -----
    - The Laplacian's smallest eigenvalues are computed via the *largest*
      eigenvalues of the affinity operator
      :math:`M = D^{-1/2} A D^{-1/2}` (using the identity :math:`L = I - M`,
      so :math:`\\lambda_L = 1 - \\lambda_M`).  This lets the iterative
      solvers chase the *largest* eigenvalues of :math:`M` -- a much better
      fit than chasing the smallest with shift-and-invert iterations.
    - The ``lobpcg`` solver requires ``5 * (n_components + 1) < n`` (a JAX
      constraint on the search subspace).  For tiny graphs, use
      ``solver="eigh"`` instead.
    """
    if n_components < 1:
        raise ValueError('n_components must be >= 1.')
    spec = _spec_from(
        solver,
        preconditioner,
        lobpcg_iters=lobpcg_iters,
        lobpcg_tol=lobpcg_tol,
        promise_symmetry=promise_symmetry,
    )
    vals_M, vecs_M, _ = _affinity_top_k(
        A,
        alpha=0.0,
        n_components=n_components,
        skip_trivial=skip_trivial,
        eps=eps,
        spec=spec,
        seed=seed,
        promise_symmetry=promise_symmetry,
    )
    # Laplacian convention: lambda_L = 1 - lambda_M, smallest-first.  The
    # iterative methods may not return perfectly ordered eigenvalues, so we
    # re-sort (a no-op for the exact eigh path).
    vals_L = 1.0 - vals_M
    order = jnp.argsort(vals_L)
    return vecs_M[:, order], vals_L[order]


def diffusion_embedding(
    A: _GraphInput,
    *,
    n_components: int,
    alpha: float = 0.5,
    t: float = 0.0,
    solver: _Solver = 'auto',
    preconditioner: _Preconditioner = 'none',
    promise_symmetry: bool = False,
    skip_trivial: bool = True,
    eps: float = 1e-12,
    lobpcg_iters: int = 200,
    lobpcg_tol: Optional[float] = _LOBPCG_DEFAULT_TOL,
    seed: int = 0,
) -> Tuple[
    Float[Array, 'n n_components'],
    Float[Array, 'n_components'],
]:
    """Coifman-Lafon diffusion-map embedding.

    Eigendecomposes the symmetric companion of the diffusion operator and
    scales each eigenvector by :math:`\\lambda^{t}`.  Returns the top
    ``n_components`` largest eigenpairs (the diffusion-map convention:
    largest eigenvalue first, descending).

    Parameters
    ----------
    A
        Affinity / adjacency matrix: dense, :class:`~nitrix.sparse.ELL`, or
        :class:`~nitrix.sparse.SectionedELL`.  Non-negative.  Treated as
        symmetric (see ``promise_symmetry``); a top-``k``-sparsified affinity
        needs no pre-symmetrisation.
    n_components
        Embedding dimensionality.
    alpha
        Density-normalisation parameter.  ``0`` is graph-Laplacian;
        ``1`` is Fokker-Planck (geometry-only); ``0.5`` is the
        canonical "diffusion map".  Default ``0.5``.
    t
        Diffusion time (real).  ``0`` returns raw eigenvectors;
        ``t > 0`` emphasises low-frequency components.
    solver, preconditioner, promise_symmetry, skip_trivial, eps, lobpcg_iters, lobpcg_tol, seed
        See :func:`laplacian_eigenmap`.

    Returns
    -------
    ``(embedding, eigenvalues)`` of shapes ``(n, n_components)`` and
    ``(n_components,)``.  Eigenvalues sorted largest-first.
    """
    if n_components < 1:
        raise ValueError('n_components must be >= 1.')
    spec = _spec_from(
        solver,
        preconditioner,
        lobpcg_iters=lobpcg_iters,
        lobpcg_tol=lobpcg_tol,
        promise_symmetry=promise_symmetry,
    )
    vals_M, vecs_M, inv_sqrt_d2 = _affinity_top_k(
        A,
        alpha=alpha,
        n_components=n_components,
        skip_trivial=skip_trivial,
        eps=eps,
        spec=spec,
        seed=seed,
        promise_symmetry=promise_symmetry,
    )
    # Recover the right eigenvectors of the random-walk operator P from the
    # symmetric-companion eigenvectors: psi = D^(-1/2) phi.  Eigenvalues stay
    # largest-first (the diffusion-map convention).
    vecs = vecs_M * inv_sqrt_d2[:, None]
    if t != 0.0:
        vecs = _scale_by_lambda_t(vecs, vals_M, t)
    return vecs, vals_M


def _scale_by_lambda_t(
    vecs: Float[Array, 'n k'],
    vals: Float[Array, 'k'],
    t: float,
) -> Float[Array, 'n k']:
    """Scale each eigenvector by :math:`|\\lambda|^{t}\\,\\operatorname{sign}(\\lambda)`.

    Used for the diffusion-map ``t > 0`` case.  Negative eigenvalues are
    guarded via the absolute value to avoid complex powers; the sign is
    preserved so the eigenvector retains its parity.

    Parameters
    ----------
    vecs
        Eigenvectors to scale, shape ``(n, k)`` (one column per eigenpair).
    vals
        Corresponding eigenvalues, shape ``(k,)``.
    t
        Diffusion time; the exponent applied to each eigenvalue's magnitude.

    Returns
    -------
    Float[Array, 'n k']
        The eigenvectors scaled column-wise by
        :math:`|\\lambda|^{t}\\,\\operatorname{sign}(\\lambda)`, shape
        ``(n, k)``.
    """
    scale = jnp.where(
        vals >= 0,
        jnp.abs(vals) ** t,
        -(jnp.abs(vals) ** t),
    )
    return vecs * scale[None, :]
