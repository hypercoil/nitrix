# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Connectopy: low-dim embeddings of graph nodes from spectral
decompositions of (modified) Laplacians.

Two embeddings:

- ``laplacian_eigenmap`` -- Belkin & Niyogi 2003.  Embed nodes in
  the bottom-``k`` non-trivial eigenspace of the symmetric
  normalised Laplacian.
- ``diffusion_embedding`` -- Coifman & Lafon 2006.  Embed nodes in
  the top-``k`` non-trivial eigenspace of the density-normalised
  diffusion operator, scaled by ``lambda^t``.

Both build the symmetric affinity operator ``M`` (dense / ELL /
SectionedELL) and hand the extremal top-k eigendecomposition to the
dedicated dispatcher ``nitrix.linalg.eigsolve`` (``eigsolve_top_k``).
The connectopy-specific spectral conventions -- skip the trivial
constant eigenvector, map ``lambda_L = 1 - lambda_M`` for the
Laplacian, recover the random-walk right eigenvectors and scale by
``lambda^t`` for diffusion -- are applied here, uniformly, on the
dispatcher's largest-first return; the solver choice no longer changes
that post-processing.

Solver methods (``solver=`` / ``preconditioner=``) map onto the
dispatcher's ``SolverSpec``:

- ``"eigh"``         -- dense full spectrum, sliced to the top-k.
  Required for dense; not available for ELL / SectionedELL.
- ``"lobpcg"``       -- iterative top-k; the default for sparse, scales
  to ``n ~ 1M+``.
- ``"shift_invert"`` -- matrix-free shift-invert (far fewer outer
  iterations on clustered low-Laplacian spectra; dense / ELL /
  SectionedELL).
- ``preconditioner="polynomial"`` -- the matvec-only shifted-power
  spectral filter (dense / ELL / SectionedELL).
- ``"auto"`` (default) -- ``eigh`` for dense, ``lobpcg`` for sparse.
  (Shift-invert / polynomial on sparse are served only on explicit
  request, not auto-selected.)

Differentiability
-----------------

- ``"eigh"``: differentiable via ``jnp.linalg.eigh``'s registered VJP.
- ``"lobpcg"`` / ``"shift_invert"`` / ``"polynomial"`` with **dense,
  ELL, or SectionedELL** input: differentiable via the implicit-VJP in
  ``linalg._eigsolve`` (Hellmann-Feynman eigenvalue gradient + the
  F-matrix subspace projector for the eigenvector gradient; exact for
  in-subspace losses, biased for losses on the orthogonal complement of
  the returned top-``k`` subspace; near-degenerate denominators clamped
  by ``eps_clamp``).  For ELL the gradient is projected onto the
  sparsity pattern; for SectionedELL onto each section's ``values``.

The brainspace dependency from the legacy code is dropped: every
required step is expressible in plain JAX.
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple, Union

import jax.numpy as jnp
from jaxtyping import Array, Float, Num

from ..linalg._eigsolve import SolverSpec, eigsolve_top_k
from ..sparse import ELL, SectionedELL
from .laplacian import _is_sparse, degree_vector

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
) -> SolverSpec:
    """Map the connectopy ``(solver, preconditioner)`` vocabulary to a
    ``SolverSpec`` for ``eigsolve_top_k``.

    ``preconditioner='polynomial'`` selects the matvec-only spectral filter
    in place of plain LOBPCG -- it overrides ``'auto'`` / ``'eigh'`` /
    ``'lobpcg'`` but not an explicit ``'shift_invert'`` (the historical
    behaviour: the shift-invert path ignored the preconditioner).  ``'auto'``
    is deferred to the dispatcher's format-based policy (dense -> ``eigh``,
    sparse -> ``lobpcg``), carrying the LOBPCG knobs in case it resolves to
    ``lobpcg``.
    """
    if preconditioner == 'polynomial' and solver != 'shift_invert':
        return SolverSpec.poly(
            degree=_POLY_DEGREE,
            shift=_POLY_SHIFT,
            outer_iters=_POLY_OUTER_ITERS,
        )
    if solver == 'eigh':
        return SolverSpec.eigh()
    if solver == 'shift_invert':
        return SolverSpec.shift_invert(
            sigma=_SI_SIGMA,
            outer_iters=_SI_OUTER_ITERS,
            cg_iters=_SI_CG_ITERS,
        )
    if solver == 'lobpcg':
        return SolverSpec.lobpcg(n_iters=lobpcg_iters, tol=lobpcg_tol)
    return SolverSpec.auto(n_iters=lobpcg_iters, tol=lobpcg_tol)


# ---------------------------------------------------------------------------
# Internal: build the normalised-affinity operator
# ---------------------------------------------------------------------------


def _scale_by_outer(
    A: _GraphInput,
    left: Float[Array, '... n'],
    right: Float[Array, '... n'],
) -> _GraphInput:
    """Scale ``A_{ij}`` by ``left[i] * right[j]`` over dense / ELL /
    SectionedELL.

    The sparsity pattern is preserved (diagonal scaling on both sides
    is structure-preserving).  Returns a new operand of the same
    type; values are scaled.
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
) -> Tuple[_GraphInput, Float[Array, '... n']]:
    """Return ``(M, inv_sqrt_d2)`` for the normalised affinity operator.

    For ``alpha == 0`` the operator is
    ``M = D^(-1/2) A D^(-1/2)`` (the affinity matrix of the
    symmetric normalised Laplacian).  For ``alpha > 0`` the operator
    is the Coifman-Lafon diffusion operator
    ``M_alpha = D_2^(-1/2) K_alpha D_2^(-1/2)`` where
    ``K_alpha = A / (d^alpha d^alpha.T)`` is the density-normalised
    affinity and ``d_2`` is its row sums.

    Returns the *concrete* operator (dense or ELL / SectionedELL,
    matching the input format), not a closure.  The returned operator
    is passed directly to ``eigsolve_top_k``.

    ``inv_sqrt_d2`` is what one multiplies the eigenvectors of
    ``M_alpha`` by to recover the right eigenvectors of the
    random-walk diffusion operator ``P``.  For the Laplacian-
    eigenmap path (``alpha = 0``) the caller ignores it.
    """
    deg = degree_vector(A)
    safe_deg = jnp.maximum(deg, eps)

    if alpha != 0.0:
        d_alpha = safe_deg**alpha
        inv_d_alpha = 1.0 / d_alpha
        K = _scale_by_outer(A, inv_d_alpha, inv_d_alpha)
        safe_d2 = jnp.maximum(degree_vector(K), eps)
    else:
        K = A
        safe_d2 = safe_deg

    inv_sqrt_d2 = 1.0 / jnp.sqrt(safe_d2)
    M = _scale_by_outer(K, inv_sqrt_d2, inv_sqrt_d2)
    # Symmetrize dense M against floating-point drift; the sparse
    # paths inherit symmetry from the (symmetric by assumption)
    # input sparsity pattern.
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
) -> Tuple[
    Float[Array, 'n_components'],
    Float[Array, 'n n_components'],
    Float[Array, '... n'],
]:
    """Build the affinity operator and return its top non-trivial
    eigenpairs (largest-first) plus ``inv_sqrt_d2``.

    Drops the trivial leading eigenpair (the constant eigenvector, the
    *largest* affinity eigenvalue) when ``skip_trivial``.  The
    convention-specific transforms (``1 - mu`` / ``lambda^t`` / right-
    eigenvector recovery) are applied by the caller.
    """
    M, inv_sqrt_d2 = _build_affinity_operator(A, alpha=alpha, eps=eps)
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

    Embeds nodes in the bottom-``n_components`` non-trivial
    eigenspace of the symmetric normalised Laplacian
    ``L = I - D^(-1/2) A D^(-1/2)``.  The smallest eigenvalue is the
    trivial zero (constant eigenvector); ``skip_trivial=True``
    (default) discards it.

    Parameters
    ----------
    A
        Adjacency matrix: dense ``(n, n)``, ``ELL``, or
        ``SectionedELL``.  Non-negative and symmetric.
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
    - The Laplacian's smallest eigenvalues are computed via the
      *largest* eigenvalues of the affinity operator
      ``M = D^(-1/2) A D^(-1/2)`` (using the identity ``L = I - M``,
      so ``lambda_L = 1 - lambda_M``).  This lets the iterative
      solvers chase the *largest* eigenvalues of ``M`` -- a much
      better fit than chasing the smallest with shift-and-invert
      iterations.
    - ``lobpcg`` requires ``5 * (n_components + 1) < n`` (a JAX
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
    )
    vals_M, vecs_M, _ = _affinity_top_k(
        A,
        alpha=0.0,
        n_components=n_components,
        skip_trivial=skip_trivial,
        eps=eps,
        spec=spec,
        seed=seed,
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

    Eigendecomposes the symmetric companion of the diffusion
    operator and scales by ``lambda^t``.  Returns the top
    ``n_components`` largest eigenpairs (the diffusion-map
    convention: largest eigenvalue first, descending).

    Parameters
    ----------
    A
        Affinity / adjacency matrix: dense, ``ELL``, or
        ``SectionedELL``.  Non-negative, symmetric.
    n_components
        Embedding dimensionality.
    alpha
        Density-normalisation parameter.  ``0`` is graph-Laplacian;
        ``1`` is Fokker-Planck (geometry-only); ``0.5`` is the
        canonical "diffusion map".  Default ``0.5``.
    t
        Diffusion time (real).  ``0`` returns raw eigenvectors;
        ``t > 0`` emphasises low-frequency components.
    solver, preconditioner, skip_trivial, eps, lobpcg_iters, lobpcg_tol, seed
        See ``laplacian_eigenmap``.

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
    )
    vals_M, vecs_M, inv_sqrt_d2 = _affinity_top_k(
        A,
        alpha=alpha,
        n_components=n_components,
        skip_trivial=skip_trivial,
        eps=eps,
        spec=spec,
        seed=seed,
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
    """Scale each eigenvector by ``|lambda|^t * sign(lambda)``.

    Used for the diffusion-map ``t > 0`` case.  Negative eigenvalues
    are guarded via ``|.|`` to avoid complex powers; sign is
    preserved so the eigenvector retains its parity.
    """
    scale = jnp.where(
        vals >= 0,
        jnp.abs(vals) ** t,
        -(jnp.abs(vals) ** t),
    )
    return vecs * scale[None, :]
