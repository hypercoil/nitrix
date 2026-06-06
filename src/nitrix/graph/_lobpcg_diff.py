# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Differentiable LOBPCG via implicit-VJP through the eigenvalue equation.

The underlying ``jax.experimental.sparse.linalg.lobpcg_standard`` is not
natively differentiable (its iterative ``lax.while_loop`` has a data-
dependent termination criterion).  We wrap the forward call in
``jax.custom_vjp`` and supply a hand-derived backward built from the
returned eigenpairs alone -- no replay of the iteration.

The math follows ``docs/design/lobpcg-implicit-vjp.md``.  The shipped
backward is "option 1" from that document: Hellmann-Feynman for the
eigenvalue gradient plus an F-matrix correction restricted to the
top-``k`` subspace for the eigenvector gradient.  Contributions from
the (unknown) orthogonal complement are dropped; the approximation is
**exact** for losses that depend only on the eigenvalues or only on
in-subspace functionals of the eigenvectors (the typical analysis
use cases: Laplacian-eigenmap embeddings, diffusion embeddings,
spectral clustering objectives).  For losses that depend on the
discarded part of the spectrum it is biased; option 2 (an iterative
Krylov correction) is the follow-up.

Two entry points:

- ``lobpcg_top_k_dense`` -- ``M: (n, n)`` symmetric, diff w.r.t. ``M``.
- ``lobpcg_top_k_ell`` -- ``M`` stored as ELL ``(values, indices)``;
  diff w.r.t. ``values`` only.  The gradient is **projected onto the
  existing sparsity pattern** so the backward cost is
  ``O(nnz * k + n * k^2)`` -- consistent with the forward.  Off-pattern
  derivatives are not returned (the user typically wouldn't be able to
  use them anyway, since the sparse forward never touched those
  entries).

Both wrappers symmetrize the resulting gradient so the cotangent lives
in the symmetric matrix subspace; this matches ``jnp.linalg.eigh``'s
VJP convention.  Near-degenerate denominators in the F-matrix are
clamped (``|lambda_j - lambda_i| < eps_clamp`` -> ``F[i, j] = 0``);
callers exercising near-degenerate spectra should pass a small
positive ``eps_clamp`` and treat individual eigenvectors as not
meaningful within the cluster.

The wrappers carry ``X0`` as a non-differentiated companion -- the
initial search subspace is not a function of the loss, and its
gradient is undefined under the iteration anyway.

Cross-reference: SPEC_UPDATE §3.x (no specific spec entry; this
ships ahead of formal spec as a stretch beyond the Phase 3 graph
exit checklist).
"""
from __future__ import annotations

from functools import partial
from typing import Optional, Tuple, cast

import jax
import jax.numpy as jnp
from jax.experimental.sparse.linalg import lobpcg_standard
from jax.scipy.sparse.linalg import cg
from jaxtyping import Array, Float, Int

# Per-section component tuples for the SectionedELL LOBPCG path.
_ValuesTuple = Tuple[Float[Array, 'n_s k_max_s'], ...]
_IndicesTuple = Tuple[Int[Array, 'n_s k_max_s'], ...]
_RowGroupsTuple = Tuple[Int[Array, 'n_s'], ...]
# ``(eigvals, eigvecs)`` largest-first, shapes ``(k,)`` and ``(n, k)``.
_EigPair = Tuple[Float[Array, 'k'], Float[Array, 'n k']]


__all__ = [
    'lobpcg_top_k_dense',
    'lobpcg_top_k_ell',
    'shift_invert_top_k_dense',
    'poly_filtered_top_k_dense',
]


# ---------------------------------------------------------------------------
# Shared backward machinery
# ---------------------------------------------------------------------------


def _subspace_vjp_kernel(
    eigvals: Float[Array, 'k'],
    eigvecs: Float[Array, 'n k'],
    g_eigvals: Float[Array, 'k'],
    g_eigvecs: Float[Array, 'n k'],
    eps_clamp: float,
) -> Float[Array, 'k k']:
    '''Build the ``k x k`` core matrix ``K`` of the implicit VJP.

    The full eigendecomposition VJP for symmetric ``M`` is::

        dM = V (diag(g_lambda) + S \\odot F) V^T

    where ``V`` is the eigenvector matrix, ``g_lambda`` the eigenvalue
    cotangent, ``S = (V^T g_V - g_V^T V) / 2`` the antisymmetric part
    of the projected eigenvector cotangent, and
    ``F[i, j] = 1 / (lambda_j - lambda_i)`` (zero on the diagonal).
    Returning ``K = diag(g_lambda) + S \\odot F`` lets dense and sparse
    backwards share this expensive piece and apply ``V K V^T`` (dense)
    or its sparsity-pattern projection (ELL) on top.

    The clamp converts ``|lambda_j - lambda_i| < eps_clamp`` into a
    zero entry of ``F`` rather than a blow-up; the entire pair-(i, j)
    correction is dropped in that case.  Documented in the public
    docstrings.
    '''
    k = eigvals.shape[0]
    # F[i, j] = 1 / (lambda_j - lambda_i), with diagonal handled separately.
    diff = eigvals[None, :] - eigvals[:, None]
    safe = jnp.where(jnp.abs(diff) < eps_clamp, 1.0, diff)
    F = jnp.where(jnp.abs(diff) < eps_clamp, 0.0, 1.0 / safe)
    F = F * (1.0 - jnp.eye(k, dtype=F.dtype))
    G = eigvecs.T @ g_eigvecs
    S = 0.5 * (G - G.T)
    K = jnp.diag(g_eigvals) + S * F
    return K


# ---------------------------------------------------------------------------
# Dense path
# ---------------------------------------------------------------------------


@partial(jax.custom_vjp, nondiff_argnums=(2, 3, 4))
def lobpcg_top_k_dense(
    M: Float[Array, 'n n'],
    X0: Float[Array, 'n k'],
    n_iters: int = 200,
    tol: Optional[float] = None,
    eps_clamp: float = 1e-8,
) -> Tuple[Float[Array, 'k'], Float[Array, 'n k']]:
    '''Differentiable LOBPCG for a concrete dense symmetric ``M``.

    Parameters
    ----------
    M
        Symmetric ``(n, n)`` operand.  Caller is responsible for
        symmetry; we do not symmetrize ``M`` because doing so would
        materialise a copy.
    X0
        ``(n, k)`` initial search subspace.  Not differentiated --
        the gradient of an iterative solver's output w.r.t. its
        initial guess is undefined when the iteration converges, and
        zero otherwise.
    n_iters
        Maximum LOBPCG iterations; passed to ``lobpcg_standard.m``.
    tol
        Convergence tolerance; passed to ``lobpcg_standard.tol``.
    eps_clamp
        Denominator floor for the F-matrix in the eigenvector
        gradient.  Pairs with ``|lambda_i - lambda_j| < eps_clamp``
        contribute zero to the gradient instead of a blow-up.

    Returns
    -------
    ``(eigvals, eigvecs)`` of shapes ``(k,)`` and ``(n, k)``.
    Eigenvalues are largest-first (the ``lobpcg_standard``
    convention).

    Notes
    -----
    The backward implements the subspace-projector approximation: it
    drops the contribution of the orthogonal complement of the top-
    ``k`` subspace to the eigenvector gradient.  Exact for purely
    in-subspace losses; biased otherwise.  See
    ``docs/design/lobpcg-implicit-vjp.md``.
    '''
    return _dense_forward(M, X0, n_iters, tol)


def _dense_forward(
    M: Float[Array, 'n n'],
    X0: Float[Array, 'n k'],
    n_iters: int,
    tol: Optional[float],
) -> _EigPair:
    eigvals, eigvecs, _ = lobpcg_standard(M, X0, m=n_iters, tol=tol)
    # ``lobpcg_standard`` is untyped (returns Any); restore the eigenpair.
    return cast(_EigPair, (eigvals, eigvecs))


def _dense_fwd(
    M: Float[Array, 'n n'],
    X0: Float[Array, 'n k'],
    n_iters: int,
    tol: Optional[float],
    eps_clamp: float,
) -> Tuple[_EigPair, _EigPair]:
    eigvals, eigvecs = _dense_forward(M, X0, n_iters, tol)
    return (eigvals, eigvecs), (eigvals, eigvecs)


def _dense_bwd(
    n_iters: int,
    tol: Optional[float],
    eps_clamp: float,
    residuals: _EigPair,
    cotangents: _EigPair,
) -> Tuple[Float[Array, 'n n'], None]:
    eigvals, eigvecs = residuals
    g_eigvals, g_eigvecs = cotangents
    K = _subspace_vjp_kernel(
        eigvals, eigvecs, g_eigvals, g_eigvecs, eps_clamp,
    )
    # dM = V K V^T.  K = diag(g_lambda) + S \\odot F is symmetric
    # because both terms are symmetric (diag is symmetric; S antisym
    # times F antisym is symmetric elementwise).  So V K V^T is
    # symmetric without further symmetrization.
    dM = eigvecs @ K @ eigvecs.T
    return (dM, None)


lobpcg_top_k_dense.defvjp(_dense_fwd, _dense_bwd)


# ---------------------------------------------------------------------------
# Shift-invert path (matrix-free, for clustered low-Laplacian spectra)
# ---------------------------------------------------------------------------


def _si_dense_forward(
    M: Float[Array, 'n n'],
    X0: Float[Array, 'n k'],
    sigma: float,
    outer_iters: int,
    cg_iters: int,
) -> _EigPair:
    '''Largest eigenpairs of symmetric ``M`` via shift-invert on ``L = I - M``.

    For ``sigma < 0`` the shifted Laplacian ``(L - sigma I) = (1 - sigma) I - M``
    is SPD (``L`` is PSD), so its inverse is applied column-wise by CG; LOBPCG on
    ``(L - sigma I)^{-1}`` finds ``L``'s smallest (``M``'s largest) eigenpairs and
    converges in *few* outer iterations because the wanted subspace is well
    separated from the rest of the spectrum under the shift-invert map, even
    when the eigenvalues themselves are tightly clustered.
    '''
    def shifted(y: Array) -> Array:
        return (1.0 - sigma) * y - M @ y

    def si_op(x: Array) -> Array:
        sol, _ = cg(shifted, x, maxiter=cg_iters)
        return sol

    mu, eigvecs, _ = lobpcg_standard(si_op, X0, m=outer_iters)
    eigvals = 1.0 - sigma - 1.0 / mu  # recover M eigenvalues, largest-first
    return cast(_EigPair, (eigvals, eigvecs))


@partial(jax.custom_vjp, nondiff_argnums=(2, 3, 4, 5))
def shift_invert_top_k_dense(
    M: Float[Array, 'n n'],
    X0: Float[Array, 'n k'],
    sigma: float = -0.5,
    outer_iters: int = 12,
    cg_iters: int = 10,
    eps_clamp: float = 1e-8,
) -> Tuple[Float[Array, 'k'], Float[Array, 'n k']]:
    '''Differentiable shift-invert top-k eigensolver for dense symmetric ``M``.

    Returns ``M``'s ``k`` largest ``(eigvals, eigvecs)`` (largest-first),
    identical in contract to ``lobpcg_top_k_dense`` but computed by matrix-free
    shift-invert (``O(2x)`` off cupy ``eigsh`` on clustered Laplacian spectra,
    vs ``O(12x)`` for plain LOBPCG).  The forward's CG / LOBPCG ``while_loop``s
    are opaque to autodiff (they live inside this ``custom_vjp``); the backward
    is the *same* implicit VJP as ``lobpcg_top_k_dense`` -- it depends only on
    the converged eigenpair, so the analytic gradient is solver-independent.

    Parameters
    ----------
    sigma
        Negative spectral shift; ``(1 - sigma) I - M`` is the SPD system CG
        solves.  Larger ``|sigma|`` conditions CG better (fewer inner iters)
        but separates the wanted subspace less (more outer iters).
    outer_iters, cg_iters
        LOBPCG outer iterations and inner CG iterations per matvec.
    '''
    return _si_dense_forward(M, X0, sigma, outer_iters, cg_iters)


def _si_dense_fwd(
    M: Float[Array, 'n n'],
    X0: Float[Array, 'n k'],
    sigma: float,
    outer_iters: int,
    cg_iters: int,
    eps_clamp: float,
) -> Tuple[_EigPair, _EigPair]:
    eigvals, eigvecs = _si_dense_forward(M, X0, sigma, outer_iters, cg_iters)
    return (eigvals, eigvecs), (eigvals, eigvecs)


def _si_dense_bwd(
    sigma: float,
    outer_iters: int,
    cg_iters: int,
    eps_clamp: float,
    residuals: _EigPair,
    cotangents: _EigPair,
) -> Tuple[Float[Array, 'n n'], None]:
    eigvals, eigvecs = residuals
    g_eigvals, g_eigvecs = cotangents
    K = _subspace_vjp_kernel(
        eigvals, eigvecs, g_eigvals, g_eigvecs, eps_clamp,
    )
    dM = eigvecs @ K @ eigvecs.T  # same implicit VJP as the LOBPCG path
    return (dM, None)


shift_invert_top_k_dense.defvjp(_si_dense_fwd, _si_dense_bwd)


# ---------------------------------------------------------------------------
# Polynomial spectral-filter path (matvec-only preconditioner)
# ---------------------------------------------------------------------------


def _poly_forward(
    M: Float[Array, 'n n'],
    X0: Float[Array, 'n k'],
    degree: int,
    shift: float,
    outer_iters: int,
) -> _EigPair:
    '''Largest eigenpairs of symmetric ``M`` via a shifted-power spectral filter.

    LOBPCG is run on ``(M + shift I)^degree`` instead of ``M``: the filter is
    monotone in ``M``'s eigenvalues (``M + shift >= 0`` for the normalized
    affinity), so it preserves eigenvectors and *amplifies the top of the
    spectrum* relative to the rest (the wanted subspace separates as
    ``((lambda_k + shift)/(lambda_{k+1} + shift))^degree``), giving far faster
    convergence on a clustered spectrum -- using only matvecs (no inner solve,
    so it also accelerates the sparse path).  ``M``'s eigenvalues are recovered
    from the converged eigenvectors by the Rayleigh quotient.
    '''
    def filt(x: Array) -> Array:
        y = x
        for _ in range(degree):
            y = M @ y + shift * y
        return y

    _, eigvecs, _ = lobpcg_standard(filt, X0, m=outer_iters)
    mv = M @ eigvecs
    eigvals = jnp.sum(eigvecs * mv, axis=0) / jnp.sum(eigvecs * eigvecs, axis=0)
    return cast(_EigPair, (eigvals, eigvecs))


@partial(jax.custom_vjp, nondiff_argnums=(2, 3, 4, 5))
def poly_filtered_top_k_dense(
    M: Float[Array, 'n n'],
    X0: Float[Array, 'n k'],
    degree: int = 4,
    shift: float = 1.0,
    outer_iters: int = 12,
    eps_clamp: float = 1e-8,
) -> Tuple[Float[Array, 'k'], Float[Array, 'n k']]:
    '''Differentiable polynomial-filtered top-k eigensolver for dense ``M``.

    Same contract as ``lobpcg_top_k_dense`` (``M``'s ``k`` largest eigenpairs,
    largest-first) but LOBPCG runs on the shifted-power filter
    ``(M + shift I)^degree`` for accelerated convergence on clustered spectra;
    the eigenvalues are recovered by Rayleigh quotient.  Matvec-only (no inner
    solve).  The forward's ``while_loop`` is opaque to autodiff (inside this
    ``custom_vjp``); the backward is the *same* implicit VJP as the plain
    LOBPCG path -- solver-independent, depending only on the converged pair.
    '''
    return _poly_forward(M, X0, degree, shift, outer_iters)


def _poly_dense_fwd(
    M: Float[Array, 'n n'],
    X0: Float[Array, 'n k'],
    degree: int,
    shift: float,
    outer_iters: int,
    eps_clamp: float,
) -> Tuple[_EigPair, _EigPair]:
    eigvals, eigvecs = _poly_forward(M, X0, degree, shift, outer_iters)
    return (eigvals, eigvecs), (eigvals, eigvecs)


def _poly_dense_bwd(
    degree: int,
    shift: float,
    outer_iters: int,
    eps_clamp: float,
    residuals: _EigPair,
    cotangents: _EigPair,
) -> Tuple[Float[Array, 'n n'], None]:
    eigvals, eigvecs = residuals
    g_eigvals, g_eigvecs = cotangents
    K = _subspace_vjp_kernel(
        eigvals, eigvecs, g_eigvals, g_eigvecs, eps_clamp,
    )
    dM = eigvecs @ K @ eigvecs.T
    return (dM, None)


poly_filtered_top_k_dense.defvjp(_poly_dense_fwd, _poly_dense_bwd)


# ---------------------------------------------------------------------------
# ELL path
# ---------------------------------------------------------------------------


@partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5, 6))
def lobpcg_top_k_ell(
    values: Float[Array, 'n k_max'],
    indices: Int[Array, 'n k_max'],
    X0: Float[Array, 'n k'],
    n_cols: int,
    n_iters: int = 200,
    tol: Optional[float] = None,
    eps_clamp: float = 1e-8,
) -> Tuple[Float[Array, 'k'], Float[Array, 'n k']]:
    '''Differentiable LOBPCG for an ELL-stored symmetric operator.

    The ELL operator is interpreted as ``M[i, indices[i, p]] +=
    values[i, p]`` (with pad rows pointing to a valid in-range index,
    contributing the semiring identity; for REAL that's zero).  The
    forward uses ELL matvec; the backward returns the projection of
    the dense gradient onto the existing ``(i, indices[i, p])``
    pattern.  Off-pattern derivatives are not returned -- in a sparse
    optimisation, the user typically can't write them anyway.

    Parameters
    ----------
    values
        ELL value array ``(n, k_max)``.  This is the differentiated
        operand.
    indices
        ELL index array ``(n, k_max)``.  Not differentiated.
    X0
        Initial search subspace ``(n, k)``.  Not differentiated.
    n_cols
        The number of dense columns the ELL implicitly represents.
    n_iters, tol, eps_clamp
        See ``lobpcg_top_k_dense``.

    Returns
    -------
    ``(eigvals, eigvecs)`` of shapes ``(k,)`` and ``(n, k)``.

    Notes
    -----
    For *symmetric* operators, the sparsity pattern must itself be
    symmetric (every off-diagonal entry has a matching transpose
    entry in the ELL).  If the pattern is not symmetric the forward
    is ill-defined regardless of differentiability.  The backward
    does not enforce symmetry of the gradient -- doing so would
    require reading the transpose entry, which costs an additional
    gather per nnz.  Callers that need a symmetric gradient should
    compute ``(g + g^T) / 2`` after the backward.
    '''
    return _ell_forward(values, indices, X0, n_cols, n_iters, tol)


def _ell_forward(
    values: Float[Array, 'n k_max'],
    indices: Int[Array, 'n k_max'],
    X0: Float[Array, 'n k'],
    n_cols: int,
    n_iters: int,
    tol: Optional[float],
) -> _EigPair:
    # Local import to avoid a cycle (this module is imported from
    # ``connectopy`` which is itself imported from ``graph/__init__``).
    from ..semiring import REAL, semiring_ell_matmul

    def matvec(X: Float[Array, 'n k']) -> Float[Array, 'n k']:
        return semiring_ell_matmul(
            values, indices, X,
            semiring=REAL, n_cols=n_cols, backend='jax',
        )

    eigvals, eigvecs, _ = lobpcg_standard(matvec, X0, m=n_iters, tol=tol)
    # ``lobpcg_standard`` is untyped (returns Any); restore the eigenpair.
    return cast(_EigPair, (eigvals, eigvecs))


def _ell_fwd(
    values: Float[Array, 'n k_max'],
    indices: Int[Array, 'n k_max'],
    X0: Float[Array, 'n k'],
    n_cols: int,
    n_iters: int,
    tol: Optional[float],
    eps_clamp: float,
) -> Tuple[_EigPair, Tuple[Int[Array, 'n k_max'], Float[Array, 'k'], Float[Array, 'n k']]]:
    eigvals, eigvecs = _ell_forward(
        values, indices, X0, n_cols, n_iters, tol,
    )
    return (eigvals, eigvecs), (indices, eigvals, eigvecs)


def _ell_bwd(
    n_cols: int,
    n_iters: int,
    tol: Optional[float],
    eps_clamp: float,
    residuals: Tuple[Int[Array, 'n k_max'], Float[Array, 'k'], Float[Array, 'n k']],
    cotangents: _EigPair,
) -> Tuple[Float[Array, 'n k_max'], Int[Array, 'n k_max'], None]:
    indices, eigvals, eigvecs = residuals
    g_eigvals, g_eigvecs = cotangents
    K = _subspace_vjp_kernel(
        eigvals, eigvecs, g_eigvals, g_eigvecs, eps_clamp,
    )
    # The dense gradient is V K V^T = V_row K V_col^T.  We project
    # onto sparsity: g_values[i, p] = (V K V^T)[i, indices[i, p]]
    # = (V K)[i, :] @ V[indices[i, p], :].
    VK = eigvecs @ K  # (n, k)
    V_at_idx = eigvecs[indices]  # (n, k_max, k)  via fancy indexing
    g_values = jnp.einsum('ij,ipj->ip', VK, V_at_idx)
    # g_indices is undefined (integer index); return zeros of matching
    # shape and dtype as required by JAX's custom_vjp contract.
    g_indices = jnp.zeros_like(indices)
    return (g_values, g_indices, None)


lobpcg_top_k_ell.defvjp(_ell_fwd, _ell_bwd)


# ---------------------------------------------------------------------------
# SectionedELL path
# ---------------------------------------------------------------------------


__all__.append('lobpcg_top_k_sectioned_ell')


@partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6))
def _lobpcg_top_k_sectioned_impl(
    values_tuple: _ValuesTuple,
    indices_tuple: _IndicesTuple,
    row_groups_tuple: _RowGroupsTuple,
    X0: Float[Array, 'n k'],
    n_cols: int,
    n_iters: int,
    tol_eps_clamp: Tuple[Optional[float], float],
) -> _EigPair:
    '''Internal entry point.  All "static" args are at the end:

    - ``n_cols`` -- int, ELL n_cols.
    - ``n_iters`` -- int, LOBPCG iteration cap.
    - ``tol_eps_clamp`` -- ``(tol, eps_clamp)`` tuple; packed so the
      whole tuple is a single nondiff arg (hashable).
    '''
    tol, _eps_clamp = tol_eps_clamp
    return _sectioned_forward(
        values_tuple, indices_tuple, row_groups_tuple,
        X0, n_cols, n_iters, tol,
    )


def _sectioned_matvec(
    values_tuple: _ValuesTuple,
    indices_tuple: _IndicesTuple,
    row_groups_tuple: _RowGroupsTuple,
    X: Float[Array, 'n k'],
    n_cols: int,
    n_rows: int,
) -> Float[Array, 'n_rows k']:
    '''SectionedELL matvec: per-bucket matmul + scatter-back.'''
    from ..semiring import REAL, semiring_ell_matmul

    out = jnp.zeros((n_rows,) + X.shape[1:], dtype=X.dtype)
    for vals, idx, row_idx in zip(values_tuple, indices_tuple, row_groups_tuple):
        bucket_out = semiring_ell_matmul(
            vals, idx, X, semiring=REAL, n_cols=n_cols, backend='jax',
        )
        out = out.at[row_idx].set(bucket_out)
    return out


def _sectioned_forward(
    values_tuple: _ValuesTuple,
    indices_tuple: _IndicesTuple,
    row_groups_tuple: _RowGroupsTuple,
    X0: Float[Array, 'n k'],
    n_cols: int,
    n_iters: int,
    tol: Optional[float],
) -> _EigPair:
    n_rows = X0.shape[0]

    def matvec(X: Float[Array, 'n k']) -> Float[Array, 'n k']:
        return _sectioned_matvec(
            values_tuple, indices_tuple, row_groups_tuple, X,
            n_cols=n_cols, n_rows=n_rows,
        )

    eigvals, eigvecs, _ = lobpcg_standard(matvec, X0, m=n_iters, tol=tol)
    # ``lobpcg_standard`` is untyped (returns Any); restore the eigenpair.
    return cast(_EigPair, (eigvals, eigvecs))


_SectionedResiduals = Tuple[
    _IndicesTuple, _RowGroupsTuple, Float[Array, 'k'], Float[Array, 'n k']
]


def _sectioned_fwd(
    values_tuple: _ValuesTuple,
    indices_tuple: _IndicesTuple,
    row_groups_tuple: _RowGroupsTuple,
    X0: Float[Array, 'n k'],
    n_cols: int,
    n_iters: int,
    tol_eps_clamp: Tuple[Optional[float], float],
) -> Tuple[_EigPair, _SectionedResiduals]:
    tol, _ = tol_eps_clamp
    eigvals, eigvecs = _sectioned_forward(
        values_tuple, indices_tuple, row_groups_tuple,
        X0, n_cols, n_iters, tol,
    )
    return (eigvals, eigvecs), (
        indices_tuple, row_groups_tuple, eigvals, eigvecs,
    )


def _sectioned_bwd(
    n_cols: int,
    n_iters: int,
    tol_eps_clamp: Tuple[Optional[float], float],
    residuals: _SectionedResiduals,
    cotangents: _EigPair,
) -> Tuple[_ValuesTuple, _IndicesTuple, _RowGroupsTuple, None]:
    indices_tuple, row_groups_tuple, eigvals, eigvecs = residuals
    _, eps_clamp = tol_eps_clamp
    g_eigvals, g_eigvecs = cotangents

    # Shared subspace-projector kernel.
    K = _subspace_vjp_kernel(
        eigvals, eigvecs, g_eigvals, g_eigvecs, eps_clamp,
    )
    VK = eigvecs @ K  # (n_rows, k)

    # Per-section sparsity-projected gradient.  For section ``s``:
    #   g_s_values[i, p] = (V K V^T)[row_groups[s][i], indices[s][i, p]]
    # Expand the per-section gather:
    #   V_at_rows = V[row_groups[s]]   # (n_rows_s, k)
    #   V_at_cols = V[indices[s]]      # (n_rows_s, k_max_s, k)
    #   VK_at_rows = V_at_rows @ K  # but we use VK directly indexed.
    g_values_list = []
    g_indices_list = []
    g_row_groups_list = []
    for idx, row_idx in zip(indices_tuple, row_groups_tuple):
        VK_at_rows = VK[row_idx]                   # (n_rows_s, k)
        V_at_cols = eigvecs[idx]                   # (n_rows_s, k_max_s, k)
        g_vals_s = jnp.einsum(
            'ij,ipj->ip', VK_at_rows, V_at_cols,
        )                                          # (n_rows_s, k_max_s)
        g_values_list.append(g_vals_s)
        g_indices_list.append(jnp.zeros_like(idx))
        g_row_groups_list.append(jnp.zeros_like(row_idx))

    return (
        tuple(g_values_list),
        tuple(g_indices_list),
        tuple(g_row_groups_list),
        None,  # X0
    )


_lobpcg_top_k_sectioned_impl.defvjp(_sectioned_fwd, _sectioned_bwd)


def lobpcg_top_k_sectioned_ell(
    values_tuple: _ValuesTuple,
    indices_tuple: _IndicesTuple,
    row_groups_tuple: _RowGroupsTuple,
    X0: Float[Array, 'n k'],
    *,
    n_cols: int,
    n_iters: int = 200,
    tol: Optional[float] = None,
    eps_clamp: float = 1e-8,
) -> _EigPair:
    '''Differentiable LOBPCG for a ``SectionedELL`` operator.

    Closes the last non-differentiable graph path in
    ``nitrix.graph.connectopy``: previously SectionedELL inputs were
    forward-only because the natural ``vmap``-over-sections backward
    didn't fit the JAX custom_vjp shape contract for variable-length
    sections.  This wrapper accepts the section components as
    parallel tuples (values, indices, row_groups), each a pytree of
    JAX arrays, so the custom_vjp can return per-section gradient
    tuples of matching structure.

    Parameters
    ----------
    values_tuple
        Per-section ``values`` arrays, one per bucket.  Each is
        ``(n_rows_section, k_max_section)``.
    indices_tuple
        Per-section ``indices`` arrays.  Same per-section shapes as
        ``values_tuple``.
    row_groups_tuple
        Per-section original-row index arrays.  Each entry maps the
        section's local row index ``i`` back to the global row
        index ``row_groups[s][i]``.
    X0
        Initial search subspace, ``(n_rows, k)``.  Not differentiated.
    n_cols
        SectionedELL ``n_cols`` -- the implicit ``n`` of the sparse
        matrix.
    n_iters, tol, eps_clamp
        See ``lobpcg_top_k_dense``.

    Returns
    -------
    ``(eigvals, eigvecs)`` of shapes ``(k,)`` and ``(n_rows, k)``.

    Notes
    -----
    Each section's gradient is computed in isolation via the same
    sparsity projection used for flat ELL -- the only difference is
    that the row indices index *through* ``row_groups`` (each
    section's local row ``i`` corresponds to global row
    ``row_groups[s][i]``).  Cost per backward:
    ``sum_s O(n_rows_s * k_max_s * k)`` where ``k`` is the
    eigenpair count -- consistent with the forward.
    '''
    return _lobpcg_top_k_sectioned_impl(
        values_tuple, indices_tuple, row_groups_tuple, X0,
        n_cols, n_iters, (tol, eps_clamp),
    )
