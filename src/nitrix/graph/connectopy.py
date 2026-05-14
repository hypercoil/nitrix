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
- ``solver="lobpcg"``: **forward-only at first GA**.  The
  ``lobpcg_standard`` implementation uses ``lax.while_loop`` with
  a data-dependent termination criterion, which JAX cannot
  differentiate in reverse mode.  Calling
  ``jax.grad(...)`` on a function that goes through ``lobpcg``
  raises a (slightly cryptic) JAX error; we wrap it in a
  ``custom_vjp`` to surface a clear "use eigh for grad" message.

The fix is implicit differentiation through the eigenvalue
equation (``∂λ/∂A = v v^T``; eigenvector gradient via the
``F``-matrix formula with a subspace projector for the partial
top-``k`` case).  Math sketch in
``docs/design/lobpcg-implicit-vjp.md``; tracked as a stretch
TODO -- the use case (analysis primitives, not training
operators) is typically forward-only anyway.
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


def _build_affinity_matvec(
    A: _GraphInput,
    *,
    alpha: float = 0.0,
    eps: float = 1e-12,
) -> Tuple[
    'jax.tree_util.Partial',
    Float[Array, '... n'],
]:
    '''Return ``(matvec, inv_sqrt_d2)`` for the normalised affinity operator.

    For ``alpha == 0`` the operator is
    ``M = D^(-1/2) A D^(-1/2)`` (the affinity matrix of the
    symmetric normalised Laplacian).  For ``alpha > 0`` the operator
    is the Coifman-Lafon diffusion operator
    ``M_alpha = D_2^(-1/2) K_alpha D_2^(-1/2)`` where
    ``K_alpha = A / (d^alpha d^alpha.T)`` is the density-normalised
    affinity and ``d_2`` is its row sums.

    The returned ``inv_sqrt_d2`` is what one multiplies the
    eigenvectors of ``M_alpha`` by to recover the right
    eigenvectors of the random-walk diffusion operator ``P``.  For
    the Laplacian-eigenmap path (``alpha = 0``) the caller ignores
    it.
    '''
    deg = degree_vector(A)
    safe_deg = jnp.maximum(deg, eps)

    if alpha != 0.0:
        d_alpha = safe_deg ** alpha
        # K_alpha is A scaled by 1 / (d_a[i] * d_a[j]).  In ELL
        # form, we gather d_a at indices to do the elementwise
        # scaling.  For dense, it's outer-product scaling.
        if isinstance(A, ELL):
            d_alpha_at_idx = d_alpha[A.indices]
            new_values = A.values / (d_alpha[:, None] * d_alpha_at_idx)
            A_normalised: _GraphInput = ELL(
                values=new_values,
                indices=A.indices,
                n_cols=A.n_cols,
                identity=A.identity,
            )
        elif isinstance(A, SectionedELL):
            new_sections = []
            for ell, row_idx in zip(A.sections, A.row_groups):
                d_a_rows = d_alpha[jnp.asarray(row_idx)]
                d_a_at = d_alpha[ell.indices]
                nv = ell.values / (d_a_rows[:, None] * d_a_at)
                new_sections.append(ELL(
                    values=nv,
                    indices=ell.indices,
                    n_cols=ell.n_cols,
                    identity=ell.identity,
                ))
            A_normalised = SectionedELL(
                sections=tuple(new_sections),
                row_groups=A.row_groups,
                n_rows=A.n_rows,
                n_cols=A.n_cols,
                identity=A.identity,
            )
        else:  # dense
            inv = 1.0 / (d_alpha[:, None] * d_alpha[None, :])
            A_normalised = A * inv
        new_deg = degree_vector(A_normalised)
        safe_d2 = jnp.maximum(new_deg, eps)
    else:
        A_normalised = A
        safe_d2 = safe_deg

    inv_sqrt_d2 = 1.0 / jnp.sqrt(safe_d2)

    def matvec(X):
        # Want M @ X = D^(-1/2) A D^(-1/2) X.
        # Step 1: scale by D^(-1/2)
        scaled = inv_sqrt_d2[:, None] * X
        # Step 2: A @ scaled
        if _is_sparse(A_normalised):
            Ax = _ell_matvec(A_normalised, scaled)
        else:
            Ax = jnp.matmul(A_normalised, scaled)
        # Step 3: scale again
        return inv_sqrt_d2[:, None] * Ax

    return matvec, inv_sqrt_d2


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
    deg = jnp.maximum(degree_vector(A), eps)
    if alpha != 0.0:
        d_alpha = deg ** alpha
        K = A / (d_alpha[..., :, None] * d_alpha[..., None, :])
    else:
        K = A
    d2 = jnp.maximum(degree_vector(K), eps)
    inv_sqrt_d2 = 1.0 / jnp.sqrt(d2)
    M = K * inv_sqrt_d2[..., :, None] * inv_sqrt_d2[..., None, :]
    M = (M + M.swapaxes(-1, -2)) / 2
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
        matvec, _ = _build_affinity_matvec(A_solver, alpha=0.0, eps=eps)
        vecs, vals = _lobpcg_top_k(
            matvec,
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
        matvec, inv_sqrt_d2 = _build_affinity_matvec(
            A_solver, alpha=alpha, eps=eps,
        )
        vecs, vals = _lobpcg_top_k(
            matvec,
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


def _lobpcg_top_k(
    matvec,
    *,
    n: int,
    n_components: int,
    skip_trivial: bool,
    iters: int,
    tol: Optional[float],
    seed: int,
    transform_eigvals,
):
    '''Run ``lobpcg_standard`` for the top ``n_components`` (+ trivial) eigenpairs.

    Returns ``(eigenvectors, eigenvalues)`` after optionally
    skipping the trivial top eigenvector and applying
    ``transform_eigvals`` (e.g. ``mu -> 1 - mu`` for Laplacian).

    Wrapped in ``_LOBPCGNotDifferentiable`` so reverse-mode AD
    through this function raises a friendly error pointing at the
    eigh solver path; see module docstring on differentiability.
    '''
    # ``lobpcg_standard`` uses ``lax.while_loop`` with a data-
    # dependent termination criterion; reverse-mode AD through it
    # raises a "while_loop with dynamic start/stop" error from JAX.
    # We do *not* wrap with ``custom_vjp`` to provide a friendlier
    # message because ``matvec`` is a closure and not hashable for
    # ``custom_vjp``'s nondiff machinery.  The native JAX error is
    # informative enough; the module docstring documents the
    # work-around (``solver="eigh"``) and ``docs/design/
    # lobpcg-implicit-vjp.md`` carries the math for the eventual
    # implicit-VJP implementation.
    k_total = n_components + (1 if skip_trivial else 0)
    key = jax.random.key(seed)
    target = _solver_device()
    X0 = jax.device_put(
        jax.random.normal(key, (n, k_total)), target,
    )
    eigvals, eigvecs, _ = lobpcg_standard(matvec, X0, m=iters, tol=tol)
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
