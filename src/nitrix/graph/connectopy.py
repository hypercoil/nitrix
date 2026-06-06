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

Two solver paths, selectable via ``solver=``:

- ``"eigh"``   -- dense ``jnp.linalg.eigh``.  Full spectrum,
  ``O(n^3)`` time / ``O(n^2)`` memory.  Required for dense inputs;
  optional for sparse.  Practical up to ``n ~ 5000`` on GPU.
- ``"lobpcg"`` -- iterative top-``k`` via
  ``jax.experimental.sparse.linalg.lobpcg_standard``.  Matrix-free
  (matvec callable), ``O((nnz + n * k^2) * iter)``.  Scales to
  ``n ~ 1M+`` for sparse adjacencies.  Default for ELL /
  SectionedELL inputs.

The brainspace dependency from the legacy code is dropped: every
required step is expressible in plain JAX.

Performance notes (A10G fp32):

============  ============  ============  =====================
``n``         dense eigh    sparse lobpcg ratio (lobpcg / eigh)
============  ============  ============  =====================
~1k                  ~50 ms        ~20 ms ~0.4×
~10k                  ~5 s         ~80 ms ~0.02× (sparse wins)
~100k                 OOM           ~1 s  n/a (sparse only)
============  ============  ============  =====================

The eigh path is fine for small dense graphs (atlas-parcel
connectomes, ROI-to-ROI connectivity); the lobpcg path is required
for any sparse graph at scale (mesh adjacencies, voxelwise
connectomes).

Differentiability
-----------------

- ``solver="eigh"``: **differentiable**.  ``jnp.linalg.eigh``
  ships with a registered VJP that JAX wires up automatically.
  Works for ``jax.grad`` over both eigenvalues and eigenvectors.
- ``solver="lobpcg"`` with **dense or flat ELL** input:
  **differentiable** via the implicit-VJP wrapper in
  ``_lobpcg_diff``.  Hellmann-Feynman for the eigenvalue gradient;
  the F-matrix subspace projector for the eigenvector gradient.
  Exact for losses that depend only on eigenvalues or on
  in-subspace functionals of eigenvectors (the typical use cases);
  biased for losses that depend on the orthogonal complement of
  the returned top-``k`` subspace.  Near-degenerate denominators
  are clamped (``eps_clamp`` knob, default ``1e-8``); pairs with
  ``|lambda_i - lambda_j| < eps_clamp`` contribute zero rather
  than a blow-up.
- ``solver="lobpcg"`` with **SectionedELL** input: forward-only.
  ``jax.grad`` raises the native JAX while-loop AD error; convert
  to flat ELL first if you need gradients.  Math sketch for an
  eventual SectionedELL-aware backward in
  ``docs/design/lobpcg-implicit-vjp.md``.
"""
from __future__ import annotations

from typing import Any, Callable, Literal, Optional, Tuple, Union, cast

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Num

from ..linalg._solver import (
    device_of_concrete as _device_of_concrete,
    eigh_device as _eigh_device,
    safe_eigh as _safe_eigh,
    solver_device as _solver_device,
    source_device as _source_device,
)
from ..semiring import REAL, semiring_ell_matmul
from ..sparse import ELL, SectionedELL, sectioned_semiring_ell_matmul
from ._lobpcg_diff import (
    lobpcg_top_k_dense,
    lobpcg_top_k_ell,
    lobpcg_top_k_sectioned_ell,
    poly_filtered_top_k_dense,
    shift_invert_top_k_dense,
)
from .laplacian import _ell_matvec, _is_sparse, degree_vector


__all__ = ['laplacian_eigenmap', 'diffusion_embedding']


# The cuSolver-fallback ``eigh`` lives in ``nitrix.linalg._solver``
# now (it's used here, in ``linalg.spd``, and in ``_lobpcg_diff``);
# import names are aliased above with a leading underscore to keep
# the in-module call sites unchanged.


def _device_put_graph(A: _GraphInput, target: Any) -> _GraphInput:
    '''Move all array fields of a graph operand to ``target``.

    Handles dense, ELL, and SectionedELL.  Non-array fields
    (``n_cols``, ``identity``, etc.) pass through unchanged.
    '''
    if isinstance(A, ELL):
        return ELL(
            values=jax.device_put(A.values, target),
            indices=jax.device_put(A.indices, target),
            n_cols=A.n_cols,
            identity=A.identity,
        )
    if isinstance(A, SectionedELL):
        new_sections = tuple(
            ELL(
                values=jax.device_put(ell.values, target),
                indices=jax.device_put(ell.indices, target),
                n_cols=ell.n_cols,
                identity=ell.identity,
            )
            for ell in A.sections
        )
        return SectionedELL(
            sections=new_sections,
            row_groups=A.row_groups,
            n_rows=A.n_rows,
            n_cols=A.n_cols,
            identity=A.identity,
        )
    # ``A`` is dense here; ``jax.device_put`` is untyped (returns Any).
    return cast(Num[Array, '... n n'], jax.device_put(A, target))


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
# Internal: build matvec for the normalised-affinity operator
# ---------------------------------------------------------------------------


def _scale_by_outer(
    A: _GraphInput,
    left: Float[Array, '... n'],
    right: Float[Array, '... n'],
) -> _GraphInput:
    '''Scale ``A_{ij}`` by ``left[i] * right[j]`` over dense / ELL /
    SectionedELL.

    The sparsity pattern is preserved (diagonal scaling on both sides
    is structure-preserving).  Returns a new operand of the same
    type; values are scaled.
    '''
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
            new_sections.append(ELL(
                values=ell.values * (left_rows[:, None] * right_at),
                indices=ell.indices,
                n_cols=ell.n_cols,
                identity=ell.identity,
            ))
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
    '''Return ``(M, inv_sqrt_d2)`` for the normalised affinity operator.

    For ``alpha == 0`` the operator is
    ``M = D^(-1/2) A D^(-1/2)`` (the affinity matrix of the
    symmetric normalised Laplacian).  For ``alpha > 0`` the operator
    is the Coifman-Lafon diffusion operator
    ``M_alpha = D_2^(-1/2) K_alpha D_2^(-1/2)`` where
    ``K_alpha = A / (d^alpha d^alpha.T)`` is the density-normalised
    affinity and ``d_2`` is its row sums.

    Returns the *concrete* operator (dense or ELL / SectionedELL,
    matching the input format), not a closure.  The returned
    operator can be passed directly to ``lobpcg_top_k_dense /
    lobpcg_top_k_ell`` for differentiable use, or wrapped in a
    closure for forward-only use (SectionedELL).

    ``inv_sqrt_d2`` is what one multiplies the eigenvectors of
    ``M_alpha`` by to recover the right eigenvectors of the
    random-walk diffusion operator ``P``.  For the Laplacian-
    eigenmap path (``alpha = 0``) the caller ignores it.
    '''
    deg = degree_vector(A)
    safe_deg = jnp.maximum(deg, eps)

    if alpha != 0.0:
        d_alpha = safe_deg ** alpha
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


def _operator_matvec(
    M: _GraphInput,
) -> Callable[[Float[Array, 'n k']], Float[Array, 'n k']]:
    '''Build a matvec closure for a concrete operator.

    Used for the SectionedELL forward-only LOBPCG path, where there
    is no differentiable wrapper for the format.
    '''
    if _is_sparse(M):
        return lambda X: _ell_matvec(M, X)
    return lambda X: jnp.matmul(M, X)


def _auto_solver(A: _GraphInput) -> _Solver:
    '''Default solver per format: dense -> eigh, sparse -> lobpcg.'''
    return 'lobpcg' if _is_sparse(A) else 'eigh'


# ---------------------------------------------------------------------------
# Dense eigh paths
# ---------------------------------------------------------------------------


def _eigh_normalised_affinity(
    A: Num[Array, '... n n'],
    *,
    alpha: float,
    eps: float,
) -> Tuple[
    Float[Array, '... n'],
    Float[Array, '... n n'],
    Float[Array, '... n'],
]:
    '''Eigh of the (alpha-normalised) symmetric affinity operator.

    Returns ``(eigenvalues_asc, eigenvectors, inv_sqrt_d2)``.
    Eigenvalues are ascending; the *largest* are at the end.
    '''
    M, inv_sqrt_d2 = _build_affinity_operator(A, alpha=alpha, eps=eps)
    # ``A`` is dense, so ``M`` is dense; narrow the operator union for eigh.
    assert not _is_sparse(M)
    eigvals, eigvecs = _safe_eigh(M)
    return eigvals, eigvecs, inv_sqrt_d2


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
    '''Laplacian-eigenmap embedding of graph nodes.

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
        low-Laplacian spectrum, ~2x off cupy ``eigsh``; dense input only,
        differentiable via the same implicit VJP), or ``"auto"`` (default:
        ``lobpcg`` for sparse, ``eigh`` for dense).
    preconditioner
        Acceleration for the LOBPCG path: ``"none"`` (default) or
        ``"polynomial"`` -- a matvec-only shifted-power spectral filter
        (LOBPCG runs on ``(M + I)^degree`` to amplify the wanted top of the
        spectrum, eigenvalues recovered by Rayleigh quotient).  Faster than
        both plain ``lobpcg`` and ``shift_invert`` on clustered spectra
        (~1.5-2x off cupy), differentiable via the same implicit VJP; dense
        input only.  Requesting it on a dense graph routes ``"auto"`` to the
        LOBPCG path (rather than ``eigh``) so the preconditioner is honoured.
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
      so ``lambda_L = 1 - lambda_M``).  This lets us use
      ``lobpcg_standard`` which returns the *largest* eigenvalues --
      a much better fit than chasing the smallest with shift-and-
      invert iterations.
    - ``lobpcg`` requires ``5 * (n_components + 1) < n`` (a JAX
      constraint on the search subspace).  For tiny graphs, use
      ``solver="eigh"`` instead.
    '''
    if n_components < 1:
        raise ValueError('n_components must be >= 1.')
    if solver == 'auto':
        solver = _auto_solver(A)
    if preconditioner != 'none' and solver == 'eigh':
        # A preconditioner is an iterative-solver concept; honour the request
        # by routing the dense ``eigh`` default to the LOBPCG path.
        solver = 'lobpcg'

    if solver == 'eigh':
        if _is_sparse(A):
            raise ValueError(
                'solver="eigh" not supported for ELL / SectionedELL '
                'inputs; use solver="lobpcg" or "auto".'
            )
        eigvals, eigvecs, _ = _eigh_normalised_affinity(
            A, alpha=0.0, eps=eps,
        )
        # eigenvalues of M ascending; convert to L = I - M -> reverse.
        # Smallest of L = 1 - largest of M (which is at the end).
        n = eigvals.shape[-1]
        start = n - n_components - (1 if skip_trivial else 0)
        # Take largest of M (skipping the very largest if skip_trivial),
        # reverse so smallest-L (= largest-M) comes first.
        if skip_trivial:
            top_M_vals = eigvals[..., start:-1][..., ::-1]
            top_M_vecs = eigvecs[..., :, start:-1][..., ::-1]
        else:
            top_M_vals = eigvals[..., start:][..., ::-1]
            top_M_vecs = eigvecs[..., :, start:][..., ::-1]
        return top_M_vecs, 1.0 - top_M_vals

    if solver == 'lobpcg':
        source = _source_device(A)
        target = _solver_device()
        A_solver = _device_put_graph(A, target)
        M, _ = _build_affinity_operator(A_solver, alpha=0.0, eps=eps)
        if preconditioner == 'polynomial':
            vecs, vals = _poly_top_k(
                M,
                n=_n_from(A_solver),
                n_components=n_components,
                skip_trivial=skip_trivial,
                seed=seed,
                transform_eigvals=lambda mu: 1.0 - mu,
            )
        else:
            vecs, vals = _lobpcg_top_k(
                M,
                n=_n_from(A_solver),
                n_components=n_components,
                skip_trivial=skip_trivial,
                iters=lobpcg_iters,
                tol=lobpcg_tol,
                seed=seed,
                transform_eigvals=lambda mu: 1.0 - mu,
            )
        if source is not None and source != target:
            vecs = jax.device_put(vecs, source)
            vals = jax.device_put(vals, source)
        return vecs, vals

    if solver == 'shift_invert':
        source = _source_device(A)
        target = _solver_device()
        A_solver = _device_put_graph(A, target)
        M, _ = _build_affinity_operator(A_solver, alpha=0.0, eps=eps)
        vecs, vals = _shift_invert_top_k(
            M,
            n=_n_from(A_solver),
            n_components=n_components,
            skip_trivial=skip_trivial,
            seed=seed,
            transform_eigvals=lambda mu: 1.0 - mu,
        )
        if source is not None and source != target:
            vecs = jax.device_put(vecs, source)
            vals = jax.device_put(vals, source)
        return vecs, vals

    raise ValueError(
        f'solver={solver!r}; expected auto/eigh/lobpcg/shift_invert.'
    )


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
    '''Coifman-Lafon diffusion-map embedding.

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
    solver, skip_trivial, eps, lobpcg_iters, lobpcg_tol, seed
        See ``laplacian_eigenmap``.

    Returns
    -------
    ``(embedding, eigenvalues)`` of shapes ``(n, n_components)`` and
    ``(n_components,)``.  Eigenvalues sorted largest-first.
    '''
    if n_components < 1:
        raise ValueError('n_components must be >= 1.')
    if solver == 'auto':
        solver = _auto_solver(A)
    if preconditioner != 'none' and solver == 'eigh':
        # A preconditioner is an iterative-solver concept; honour the request
        # by routing the dense ``eigh`` default to the LOBPCG path.
        solver = 'lobpcg'

    if solver == 'eigh':
        if _is_sparse(A):
            raise ValueError(
                'solver="eigh" not supported for ELL / SectionedELL '
                'inputs; use solver="lobpcg" or "auto".'
            )
        eigvals, eigvecs, inv_sqrt_d2 = _eigh_normalised_affinity(
            A, alpha=alpha, eps=eps,
        )
        # Recover right eigenvectors of the random-walk operator P
        # from eigenvectors of M_sym: psi = D^(-1/2) phi.
        right_eigvecs = eigvecs * inv_sqrt_d2[..., :, None]
        # Take largest k (or k+1 if skipping trivial).
        n = eigvals.shape[-1]
        if skip_trivial:
            top_vals = eigvals[..., n - n_components - 1:-1][..., ::-1]
            top_vecs = right_eigvecs[..., :, n - n_components - 1:-1][..., ::-1]
        else:
            top_vals = eigvals[..., n - n_components:][..., ::-1]
            top_vecs = right_eigvecs[..., :, n - n_components:][..., ::-1]
        if t != 0.0:
            top_vecs = _scale_by_lambda_t(top_vecs, top_vals, t)
        return top_vecs, top_vals

    if solver == 'lobpcg':
        source = _source_device(A)
        target = _solver_device()
        A_solver = _device_put_graph(A, target)
        M, inv_sqrt_d2 = _build_affinity_operator(
            A_solver, alpha=alpha, eps=eps,
        )
        if preconditioner == 'polynomial':
            vecs, vals = _poly_top_k(
                M,
                n=_n_from(A_solver),
                n_components=n_components,
                skip_trivial=skip_trivial,
                seed=seed,
                transform_eigvals=None,
            )
        else:
            vecs, vals = _lobpcg_top_k(
                M,
                n=_n_from(A_solver),
                n_components=n_components,
                skip_trivial=skip_trivial,
                iters=lobpcg_iters,
                tol=lobpcg_tol,
                seed=seed,
                transform_eigvals=None,
            )
        # Recover right eigenvectors of P from eigenvectors of M_sym.
        vecs = vecs * inv_sqrt_d2[:, None]
        if t != 0.0:
            vecs = _scale_by_lambda_t(vecs, vals, t)
        if source is not None and source != target:
            vecs = jax.device_put(vecs, source)
            vals = jax.device_put(vals, source)
        return vecs, vals

    if solver == 'shift_invert':
        source = _source_device(A)
        target = _solver_device()
        A_solver = _device_put_graph(A, target)
        M, inv_sqrt_d2 = _build_affinity_operator(
            A_solver, alpha=alpha, eps=eps,
        )
        vecs, vals = _shift_invert_top_k(
            M,
            n=_n_from(A_solver),
            n_components=n_components,
            skip_trivial=skip_trivial,
            seed=seed,
            transform_eigvals=None,
        )
        vecs = vecs * inv_sqrt_d2[:, None]
        if t != 0.0:
            vecs = _scale_by_lambda_t(vecs, vals, t)
        if source is not None and source != target:
            vecs = jax.device_put(vecs, source)
            vals = jax.device_put(vals, source)
        return vecs, vals

    raise ValueError(
        f'solver={solver!r}; expected auto/eigh/lobpcg/shift_invert.'
    )


# ---------------------------------------------------------------------------
# LOBPCG plumbing
# ---------------------------------------------------------------------------


def _n_from(A: _GraphInput) -> int:
    if isinstance(A, ELL):
        return A.n_cols
    if isinstance(A, SectionedELL):
        return A.n_cols
    return int(A.shape[-1])


def _lobpcg_initial_subspace(
    n: int, k_total: int, seed: int, device: Any,
) -> Float[Array, 'n k_total']:
    '''Random initial subspace for LOBPCG, on the requested device.

    ``seed`` controls reproducibility across runs.  ``device`` is the
    solver device (chosen by ``_solver_device()``); LOBPCG's internal
    QR / Cholesky use cuSolver, which we route to CPU when the
    GPU-side build has a broken handle.
    '''
    key = jax.random.key(seed)
    # ``jax.device_put`` is untyped (returns Any); restore the subspace type.
    return cast(
        Float[Array, 'n k_total'],
        jax.device_put(jax.random.normal(key, (n, k_total)), device),
    )


def _run_lobpcg_on_operator(
    M: _GraphInput,
    X0: Float[Array, 'n k_total'],
    *,
    iters: int,
    tol: Optional[float],
    eps_clamp: float = 1e-8,
) -> Tuple[Float[Array, 'k_total'], Float[Array, 'n k_total']]:
    '''Dispatch LOBPCG to the differentiable variant when possible.

    Dispatch table:

    - dense ``M``: ``lobpcg_top_k_dense`` (implicit-VJP).
    - ``ELL`` ``M``: ``lobpcg_top_k_ell`` (sparsity-projected VJP).
    - ``SectionedELL`` ``M``: ``lobpcg_top_k_sectioned_ell``
      (per-section sparsity-projected VJP; differentiable through
      each section's ``values``).

    The skip-trivial / transform / sort post-processing is handled
    by the caller in plain JAX so the chain rule flows naturally.
    '''
    if isinstance(M, ELL):
        return lobpcg_top_k_ell(
            M.values, M.indices, X0, M.n_cols, iters, tol, eps_clamp,
        )
    if isinstance(M, SectionedELL):
        # Sectioned: pull out per-section values / indices / row_groups
        # and route to the diff wrapper.  Row groups are np.ndarray
        # in the SectionedELL; convert to jnp once at the boundary.
        values_tuple = tuple(s.values for s in M.sections)
        indices_tuple = tuple(s.indices for s in M.sections)
        row_groups_tuple = tuple(jnp.asarray(rg) for rg in M.row_groups)
        return lobpcg_top_k_sectioned_ell(
            values_tuple, indices_tuple, row_groups_tuple, X0,
            n_cols=M.n_cols, n_iters=iters, tol=tol, eps_clamp=eps_clamp,
        )
    # Dense
    return lobpcg_top_k_dense(M, X0, iters, tol, eps_clamp)


def _lobpcg_top_k(
    M: _GraphInput,
    *,
    n: int,
    n_components: int,
    skip_trivial: bool,
    iters: int,
    tol: Optional[float],
    seed: int,
    transform_eigvals: Optional[
        Callable[[Float[Array, 'k']], Float[Array, 'k']]
    ],
) -> Tuple[Float[Array, 'n n_components'], Float[Array, 'n_components']]:
    '''Run LOBPCG on a concrete operator and post-process.

    Returns ``(eigenvectors, eigenvalues)`` after optionally
    skipping the trivial top eigenvector and applying
    ``transform_eigvals`` (e.g. ``mu -> 1 - mu`` for the Laplacian
    convention).

    The post-processing (skip / transform / sort) is in plain JAX
    and is therefore differentiable end-to-end when ``M`` is dense
    or flat ELL.  SectionedELL is forward-only; see
    ``_run_lobpcg_on_operator``.
    '''
    k_total = n_components + (1 if skip_trivial else 0)
    target = _solver_device()
    X0 = _lobpcg_initial_subspace(n, k_total, seed, target)
    eigvals, eigvecs = _run_lobpcg_on_operator(
        M, X0, iters=iters, tol=tol,
    )
    # ``lobpcg_standard`` returns largest-first.  Drop the trivial
    # *first* column if requested.
    if skip_trivial:
        eigvals = eigvals[1:]
        eigvecs = eigvecs[:, 1:]
    if transform_eigvals is not None:
        eigvals = transform_eigvals(eigvals)
        # When we transform mu -> 1 - mu, largest mu => smallest
        # 1 - mu.  We re-sort smallest-first to match the
        # Laplacian convention.
        order = jnp.argsort(eigvals)
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
    return eigvecs, eigvals


def _shift_invert_top_k(
    M: _GraphInput,
    *,
    n: int,
    n_components: int,
    skip_trivial: bool,
    seed: int,
    transform_eigvals: Optional[
        Callable[[Float[Array, 'k']], Float[Array, 'k']]
    ],
) -> Tuple[Float[Array, 'n n_components'], Float[Array, 'n_components']]:
    '''Matrix-free shift-invert top-k on a *dense* operator, post-processed.

    Same contract as ``_lobpcg_top_k`` (skip trivial, transform, sort), but the
    forward is the shift-invert/CG eigensolver -- far fewer outer iterations on
    clustered Laplacian spectra.  Dense only; ELL / SectionedELL keep
    ``solver='lobpcg'``.
    '''
    if _is_sparse(M):
        raise ValueError(
            "solver='shift_invert' supports dense input only; use "
            "solver='lobpcg' (or 'auto') for ELL / SectionedELL."
        )
    k_total = n_components + (1 if skip_trivial else 0)
    target = _solver_device()
    X0 = _lobpcg_initial_subspace(n, k_total, seed, target)
    eigvals, eigvecs = shift_invert_top_k_dense(
        M, X0, _SI_SIGMA, _SI_OUTER_ITERS, _SI_CG_ITERS,
    )
    if skip_trivial:
        eigvals = eigvals[1:]
        eigvecs = eigvecs[:, 1:]
    if transform_eigvals is not None:
        eigvals = transform_eigvals(eigvals)
        order = jnp.argsort(eigvals)
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
    return eigvecs, eigvals


def _poly_top_k(
    M: _GraphInput,
    *,
    n: int,
    n_components: int,
    skip_trivial: bool,
    seed: int,
    transform_eigvals: Optional[
        Callable[[Float[Array, 'k']], Float[Array, 'k']]
    ],
) -> Tuple[Float[Array, 'n n_components'], Float[Array, 'n_components']]:
    '''Polynomial-preconditioned (matvec-only) LOBPCG top-k on dense ``M``.

    Same contract as ``_lobpcg_top_k`` but LOBPCG runs on the shifted-power
    spectral filter (eigenvalues recovered by Rayleigh quotient) -- faster than
    plain LOBPCG *and* shift-invert on clustered spectra, using only matvecs.
    Dense only for now (the differentiable sparse-pattern backward is the ELL
    follow-up); ELL / SectionedELL keep the plain ``preconditioner='none'``
    LOBPCG path.
    '''
    if _is_sparse(M):
        raise ValueError(
            "preconditioner='polynomial' supports dense input only; use "
            "preconditioner='none' for ELL / SectionedELL."
        )
    k_total = n_components + (1 if skip_trivial else 0)
    target = _solver_device()
    X0 = _lobpcg_initial_subspace(n, k_total, seed, target)
    eigvals, eigvecs = poly_filtered_top_k_dense(
        M, X0, _POLY_DEGREE, _POLY_SHIFT, _POLY_OUTER_ITERS,
    )
    if skip_trivial:
        eigvals = eigvals[1:]
        eigvecs = eigvecs[:, 1:]
    if transform_eigvals is not None:
        eigvals = transform_eigvals(eigvals)
        order = jnp.argsort(eigvals)
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
    return eigvecs, eigvals


def _scale_by_lambda_t(
    vecs: Float[Array, 'n k'],
    vals: Float[Array, 'k'],
    t: float,
) -> Float[Array, 'n k']:
    '''Scale each eigenvector by ``|lambda|^t * sign(lambda)``.

    Used for the diffusion-map ``t > 0`` case.  Negative eigenvalues
    are guarded via ``|.|`` to avoid complex powers; sign is
    preserved so the eigenvector retains its parity.
    '''
    scale = jnp.where(
        vals >= 0,
        jnp.abs(vals) ** t,
        -(jnp.abs(vals) ** t),
    )
    return vecs * scale[None, :]
