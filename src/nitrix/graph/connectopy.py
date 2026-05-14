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

import functools
from typing import Literal, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax.experimental.sparse.linalg import lobpcg_standard
from jaxtyping import Array, Float, Num

from ..semiring import REAL, semiring_ell_matmul
from ..sparse import ELL, SectionedELL, sectioned_semiring_ell_matmul
from ._lobpcg_diff import lobpcg_top_k_dense, lobpcg_top_k_ell
from .laplacian import _ell_matvec, _is_sparse, degree_vector


__all__ = ['laplacian_eigenmap', 'diffusion_embedding']


# ---------------------------------------------------------------------------
# Robust ``eigh`` device selection
#
# Certain CUDA / JAX combinations have a broken cuSolver handle (an
# ABI mismatch between the cuSolver library and the GPU driver
# manifests as ``gpusolverDnCreate(&handle) failed``).  We probe once
# at first use; if GPU eigh fails we route eigh to CPU.  For the eigh
# use case (small dense matrices, ``n ≲ 5000``) the GPU vs CPU
# difference is small enough not to matter.
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _eigh_device():
    '''Pick a device for dense ``eigh`` that's known to work.

    Returns ``jax.devices()[0]`` if GPU eigh succeeds on a probe;
    otherwise ``jax.devices('cpu')[0]``.  Cached so the probe runs
    at most once per process.
    '''
    try:
        probe = jnp.eye(2, dtype=jnp.float32)
        out = jnp.linalg.eigh(probe)
        jax.block_until_ready(out)
        return jax.devices()[0]
    except Exception:
        cpu_devs = jax.devices('cpu')
        return cpu_devs[0] if cpu_devs else jax.devices()[0]


def _device_of_concrete(arr):
    '''Return the device of a concrete array, or ``None`` for tracers.

    ``arr.devices()`` raises ``ConcretizationTypeError`` inside a
    JAX trace; we treat tracers as "no fixed device" and let JAX's
    abstract evaluation handle dispatch.
    '''
    if not hasattr(arr, 'devices'):
        return None
    try:
        devs = arr.devices()
    except jax.errors.ConcretizationTypeError:
        return None
    return next(iter(devs), None)


def _safe_eigh(A: Float[Array, '... n n']):
    '''``jnp.linalg.eigh`` with the cuSolver-robust device pick.

    Results are moved back to the input's original device so the
    caller doesn't see a surprise CPU array when ``A`` was on GPU.
    Inside ``jax.grad`` the input is a tracer with no concrete
    device; we then skip the move-back and rely on JAX dispatch.
    '''
    target = _eigh_device()
    source = _device_of_concrete(A)
    A_dev = jax.device_put(A, target) if source is not None else A
    eigvals, eigvecs = jnp.linalg.eigh(A_dev)
    if source is not None and source != target:
        eigvals = jax.device_put(eigvals, source)
        eigvecs = jax.device_put(eigvecs, source)
    return eigvals, eigvecs


def _solver_device():
    '''Pick the device for matrix-free iterative solvers (LOBPCG, etc.).

    LOBPCG internally calls cuSolver-backed QR / Cholesky, so it
    inherits the same broken-cuSolver failure mode as ``eigh``.  We
    reuse the eigh probe's verdict: if eigh works on GPU, lobpcg
    likely does too; if eigh fell back to CPU, route lobpcg there
    as well.
    '''
    return _eigh_device()


def _devices_of(tree) -> set:
    '''Unique devices of every concrete-array leaf in a pytree.

    Skips tracers (they have no concrete device) and non-array
    leaves.
    '''
    leaves = jax.tree_util.tree_leaves(tree)
    devs = set()
    for leaf in leaves:
        if not hasattr(leaf, 'devices'):
            continue
        try:
            devs.update(leaf.devices())
        except jax.errors.ConcretizationTypeError:
            continue
        except Exception:
            continue
    return devs


def _source_device(tree):
    '''The "originating" device for a tree of arrays.

    If all leaves share a device, return it.  If multiple, return
    the first found.  ``None`` if no array leaves.
    '''
    devs = _devices_of(tree)
    return next(iter(devs), None) if devs else None


def _device_put_graph(A: _GraphInput, target) -> _GraphInput:
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
    return jax.device_put(A, target)


_Solver = Literal['auto', 'eigh', 'lobpcg']
_GraphInput = Union[Num[Array, '... n n'], ELL, SectionedELL]


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


def _operator_matvec(M: _GraphInput):
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
    skip_trivial: bool = True,
    eps: float = 1e-12,
    lobpcg_iters: int = 200,
    lobpcg_tol: Optional[float] = None,
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
        top-k), or ``"auto"`` (default: ``lobpcg`` for sparse,
        ``eigh`` for dense).
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
        Convergence tolerance; ``None`` uses the JAX default.
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

    raise ValueError(f'solver={solver!r}; expected auto/eigh/lobpcg.')


def diffusion_embedding(
    A: _GraphInput,
    *,
    n_components: int,
    alpha: float = 0.5,
    t: float = 0.0,
    solver: _Solver = 'auto',
    skip_trivial: bool = True,
    eps: float = 1e-12,
    lobpcg_iters: int = 200,
    lobpcg_tol: Optional[float] = None,
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

    raise ValueError(f'solver={solver!r}; expected auto/eigh/lobpcg.')


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
    n: int, k_total: int, seed: int, device,
) -> Float[Array, 'n k_total']:
    '''Random initial subspace for LOBPCG, on the requested device.

    ``seed`` controls reproducibility across runs.  ``device`` is the
    solver device (chosen by ``_solver_device()``); LOBPCG's internal
    QR / Cholesky use cuSolver, which we route to CPU when the
    GPU-side build has a broken handle.
    '''
    key = jax.random.key(seed)
    return jax.device_put(jax.random.normal(key, (n, k_total)), device)


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
    - ``SectionedELL`` ``M``: closure-based ``lobpcg_standard`` call;
      forward-only, raises under ``jax.grad`` with a native JAX
      while-loop AD error.  Convert to flat ELL first if you need
      differentiability through SectionedELL.

    The skip-trivial / transform / sort post-processing is handled
    by the caller in plain JAX so the chain rule flows naturally.
    '''
    if isinstance(M, ELL):
        return lobpcg_top_k_ell(
            M.values, M.indices, X0, M.n_cols, iters, tol, eps_clamp,
        )
    if isinstance(M, SectionedELL):
        # Forward-only.  No diff-capable wrapper for SectionedELL.
        matvec = _operator_matvec(M)
        eigvals, eigvecs, _ = lobpcg_standard(matvec, X0, m=iters, tol=tol)
        return eigvals, eigvecs
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
    transform_eigvals,
):
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


def _scale_by_lambda_t(vecs, vals, t):
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
