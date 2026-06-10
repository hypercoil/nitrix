# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Dedicated extremal (top-k) eigendecomposition dispatcher.

Consolidates the solver suite that previously lived inlined in
``nitrix.graph.connectopy``.  The design rationale is in
``docs/design/eigsolve-dispatcher.md``; the implicit-VJP math is in
``docs/design/lobpcg-implicit-vjp.md``.

The load-bearing factoring: the solver *forward* (which method) is
orthogonal to the gradient *backward* (which operand format).  The
backward is solver-independent -- it depends only on the converged
eigenpair -- so it is shared across methods and specialised only by
format (dense ``V K Vᵀ``; ELL gather-einsum onto the sparsity pattern;
SectionedELL per-section gather).  The forward is format-independent --
every iterative method is written against an abstract matvec
``apply: X -> M @ X``.  This module therefore ships:

- one frozen, hashable ``SolverSpec`` carrying all static config (it
  rides the ``custom_vjp`` ``nondiff_argnums`` as a single argument);
- three ``custom_vjp`` entry points, one per operand format, each
  branching its forward on ``spec.method`` and sharing the format's
  single backward;
- ``eigsolve_top_k`` -- the user-facing dispatcher: validity check,
  ``auto`` policy, cuSolver-robust device routing, and (for ``eigh``)
  the dense full-spectrum solve sliced to the extremal top-k.

``eigh`` is folded in *only in its extremal role* (full ``safe_eigh``
then slice top-k).  The full-spectrum uses of ``safe_eigh`` (``spd``,
``reml``, ``lomb_scargle``) keep calling it directly; this module is
built *on* ``safe_eigh``, it does not subsume it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial
from typing import (
    Any,
    Callable,
    Literal,
    Mapping,
    NamedTuple,
    Optional,
    Tuple,
    Union,
    cast,
)

import jax
import jax.numpy as jnp
from jax.experimental.sparse.linalg import lobpcg_standard
from jax.scipy.sparse.linalg import cg
from jaxtyping import Array, Float, Int

from ..semiring import REAL, semiring_ell_matmul, semiring_ell_rmatvec
from ..sparse import ELL, SectionedELL
from ._solver import safe_eigh, solver_device, source_device

__all__ = [
    'EigPair',
    'SolverSpec',
    'eigsolve_top_k',
]


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

Method = Literal['auto', 'eigh', 'lobpcg', 'shift_invert', 'poly']
_Format = Literal['dense', 'ell', 'sectioned']
EigOperand = Union[Float[Array, '... n n'], ELL, SectionedELL]

# Internal plain-tuple eigenpair (largest-first), the ``custom_vjp`` output
# structure; the public dispatcher wraps it in ``EigPair``.
_EigPair = Tuple[Float[Array, 'k'], Float[Array, 'n k']]
_ValuesTuple = Tuple[Float[Array, 'n_s k_max_s'], ...]
_IndicesTuple = Tuple[Int[Array, 'n_s k_max_s'], ...]
_RowGroupsTuple = Tuple[Int[Array, 'n_s'], ...]


class EigPair(NamedTuple):
    """Top-k eigenpair, eigenvalues largest-first."""

    values: Float[Array, 'k']
    vectors: Float[Array, 'n k']


@dataclass(frozen=True)
class SolverSpec:
    """Frozen, hashable static configuration for a single solve.

    Hashability is load-bearing: the whole spec rides the
    ``jax.custom_vjp`` ``nondiff_argnums`` as one argument, so the
    iterative forwards branch on ``method`` at trace time.  Only the
    fields relevant to ``method`` are read; the rest sit at their
    defaults.  Build via the classmethods rather than the raw
    constructor.
    """

    method: Method
    # lobpcg
    n_iters: int = 200
    tol: Optional[float] = None
    # shift-invert
    sigma: float = -0.5
    outer_iters: int = 12
    cg_iters: int = 10
    # polynomial filter
    degree: int = 4
    shift: float = 1.0
    # shared backward
    eps_clamp: float = 1e-8
    # operator symmetry.  When True (the default), the iterative forwards
    # apply the operand's matvec directly -- the caller *promises* the
    # operator is symmetric.  When False, ELL / SectionedELL forwards apply
    # the symmetric part ``½(A X + Aᵀ X) = sym(A) X`` instead, so a stored
    # operator whose construction did not preserve symmetry (e.g. top-k
    # affinity sparsification) still presents the symmetric operator the
    # solvers (and shift-invert's inner CG) require.  Ignored for dense
    # operands (symmetry is the caller's to enforce on the materialised M).
    promise_symmetry: bool = True

    @classmethod
    def auto(
        cls,
        *,
        n_iters: int = 200,
        tol: Optional[float] = None,
        eps_clamp: float = 1e-8,
        promise_symmetry: bool = True,
    ) -> 'SolverSpec':
        # Carry the lobpcg knobs: ``auto`` resolves to ``lobpcg`` on sparse
        # input, and the dispatcher keeps the other fields via ``replace``.
        return cls(
            method='auto',
            n_iters=n_iters,
            tol=tol,
            eps_clamp=eps_clamp,
            promise_symmetry=promise_symmetry,
        )

    @classmethod
    def eigh(cls, *, eps_clamp: float = 1e-8) -> 'SolverSpec':
        # Dense-only; symmetry is enforced on the materialised M upstream,
        # so ``promise_symmetry`` is left at its (unused) default.
        return cls(method='eigh', eps_clamp=eps_clamp)

    @classmethod
    def lobpcg(
        cls,
        *,
        n_iters: int = 200,
        tol: Optional[float] = None,
        eps_clamp: float = 1e-8,
        promise_symmetry: bool = True,
    ) -> 'SolverSpec':
        return cls(
            method='lobpcg',
            n_iters=n_iters,
            tol=tol,
            eps_clamp=eps_clamp,
            promise_symmetry=promise_symmetry,
        )

    @classmethod
    def shift_invert(
        cls,
        *,
        sigma: float = -0.5,
        outer_iters: int = 12,
        cg_iters: int = 10,
        eps_clamp: float = 1e-8,
        promise_symmetry: bool = True,
    ) -> 'SolverSpec':
        return cls(
            method='shift_invert',
            sigma=sigma,
            outer_iters=outer_iters,
            cg_iters=cg_iters,
            eps_clamp=eps_clamp,
            promise_symmetry=promise_symmetry,
        )

    @classmethod
    def poly(
        cls,
        *,
        degree: int = 4,
        shift: float = 1.0,
        outer_iters: int = 12,
        eps_clamp: float = 1e-8,
        promise_symmetry: bool = True,
    ) -> 'SolverSpec':
        return cls(
            method='poly',
            degree=degree,
            shift=shift,
            outer_iters=outer_iters,
            eps_clamp=eps_clamp,
            promise_symmetry=promise_symmetry,
        )


# Which operand formats each method serves.  ``shift_invert`` / ``poly`` are
# matvec-only -- their forward runs on the ELL / sectioned matvec and their
# backward is the same solver-independent per-format projection as ``lobpcg``
# -- so they cover all three formats.  ``eigh`` is dense-only by construction
# (it materialises the full spectrum).  ``auto`` still routes sparse to
# ``lobpcg`` (the benchmarked sparse default); shift-invert / poly on sparse
# are served only on explicit request.
_SUPPORTED: Mapping[Method, frozenset[str]] = {
    'eigh': frozenset({'dense'}),
    'lobpcg': frozenset({'dense', 'ell', 'sectioned'}),
    'shift_invert': frozenset({'dense', 'ell', 'sectioned'}),
    'poly': frozenset({'dense', 'ell', 'sectioned'}),
}


# ---------------------------------------------------------------------------
# Shared backward: the k x k core matrix and the per-format projections
# ---------------------------------------------------------------------------


def _subspace_vjp_kernel(
    eigvals: Float[Array, 'k'],
    eigvecs: Float[Array, 'n k'],
    g_eigvals: Float[Array, 'k'],
    g_eigvecs: Float[Array, 'n k'],
    eps_clamp: float,
) -> Float[Array, 'k k']:
    """The ``K = diag(g_lambda) + S ⊙ F`` core of the implicit VJP.

    ``F[i, j] = 1 / (lambda_j - lambda_i)`` (zero on the diagonal and
    where ``|lambda_j - lambda_i| < eps_clamp``); ``S`` is the
    antisymmetric part of the projected eigenvector cotangent.  Shared by
    every method (the gradient depends only on the converged eigenpair)
    and every format (the format-specific projection applies ``V K Vᵀ``
    or its sparsity restriction on top).  See
    ``docs/design/lobpcg-implicit-vjp.md``.
    """
    k = eigvals.shape[0]
    diff = eigvals[None, :] - eigvals[:, None]
    safe = jnp.where(jnp.abs(diff) < eps_clamp, 1.0, diff)
    F = jnp.where(jnp.abs(diff) < eps_clamp, 0.0, 1.0 / safe)
    F = F * (1.0 - jnp.eye(k, dtype=F.dtype))
    G = eigvecs.T @ g_eigvecs
    S = 0.5 * (G - G.T)
    return jnp.diag(g_eigvals) + S * F


def _project_dense(
    K: Float[Array, 'k k'],
    eigvecs: Float[Array, 'n k'],
) -> Float[Array, 'n n']:
    """``dM = V K Vᵀ`` -- symmetric by construction."""
    return eigvecs @ K @ eigvecs.T


def _project_ell(
    K: Float[Array, 'k k'],
    eigvecs: Float[Array, 'n k'],
    indices: Int[Array, 'n k_max'],
) -> Float[Array, 'n k_max']:
    """Project ``V K Vᵀ`` onto the ELL pattern:
    ``g_values[i, p] = (V K)[i, :] @ V[indices[i, p], :]``.

    This is **unchanged** under the ``promise_symmetry=False`` symmetric
    forward, and exactly correct for it: the implicit-VJP cotangent
    ``G = V K Vᵀ`` is symmetric, and a stored ``values[i, p]`` enters
    ``sym(A)`` at both ``[i, c]`` and ``[c, i]`` (each ∂ = ½, ``c =
    indices[i, p]``), so ``∂L/∂values[i, p] = ½G[i, c] + ½G[c, i] = G[i, c]``
    -- precisely the entry this projection returns.  The symmetrisation
    lives entirely in the *forward* matvec; the backward needs no change.
    """
    VK = eigvecs @ K
    V_at_idx = eigvecs[indices]
    return jnp.einsum('ij,ipj->ip', VK, V_at_idx)


def _project_sectioned(
    K: Float[Array, 'k k'],
    eigvecs: Float[Array, 'n k'],
    indices_tuple: _IndicesTuple,
    row_groups_tuple: _RowGroupsTuple,
) -> _ValuesTuple:
    """Per-section ELL projection, reading the mapped global row index
    from ``row_groups[s]``."""
    VK = eigvecs @ K
    g_values = []
    for idx, row_idx in zip(indices_tuple, row_groups_tuple):
        VK_at_rows = VK[row_idx]
        V_at_cols = eigvecs[idx]
        g_values.append(jnp.einsum('ij,ipj->ip', VK_at_rows, V_at_cols))
    return tuple(g_values)


# ---------------------------------------------------------------------------
# Method forwards (operate on an abstract matvec ``apply: X -> M @ X``)
# ---------------------------------------------------------------------------


def _run_lobpcg_method(
    operator: Any,
    X0: Float[Array, 'n k'],
    n_iters: int,
    tol: Optional[float],
) -> _EigPair:
    """Plain LOBPCG.  ``operator`` is a dense array (passed through for the
    blocked-matmul fast path) or a matvec callable (ELL / sectioned)."""
    eigvals, eigvecs, _ = lobpcg_standard(operator, X0, m=n_iters, tol=tol)
    return cast(_EigPair, (eigvals, eigvecs))


def _run_shift_invert_method(
    apply: Callable[[Array], Array],
    X0: Float[Array, 'n k'],
    sigma: float,
    outer_iters: int,
    cg_iters: int,
) -> _EigPair:
    """Shift-invert on ``L = I - M`` via inner CG.  Matvec-only, so it
    serves any operand format.  For ``sigma < 0`` the shifted Laplacian is
    SPD and CG-solvable; LOBPCG on ``(L - sigma I)^{-1}`` recovers ``M``'s
    largest eigenpairs in few outer iterations on clustered spectra."""

    def shifted(y: Array) -> Array:
        return (1.0 - sigma) * y - apply(y)

    def si_op(x: Array) -> Array:
        sol, _ = cg(shifted, x, maxiter=cg_iters)
        # ``cg`` is untyped (returns Any); restore the solution array type.
        return cast(Array, sol)

    mu, eigvecs, _ = lobpcg_standard(si_op, X0, m=outer_iters)
    eigvals = 1.0 - sigma - 1.0 / mu  # recover M eigenvalues, largest-first
    return cast(_EigPair, (eigvals, eigvecs))


def _run_poly_method(
    apply: Callable[[Array], Array],
    X0: Float[Array, 'n k'],
    degree: int,
    shift: float,
    outer_iters: int,
) -> _EigPair:
    """Shifted-power spectral filter ``(M + shift I)^degree``.  Matvec-only;
    eigenvalues recovered by the Rayleigh quotient."""

    def filt(x: Array) -> Array:
        y = x
        for _ in range(degree):
            y = apply(y) + shift * y
        return y

    _, eigvecs, _ = lobpcg_standard(filt, X0, m=outer_iters)
    mv = apply(eigvecs)
    eigvals = jnp.sum(eigvecs * mv, axis=0) / jnp.sum(
        eigvecs * eigvecs, axis=0
    )
    return cast(_EigPair, (eigvals, eigvecs))


def _forward_for_spec(
    spec: SolverSpec,
    apply: Callable[[Array], Array],
    lobpcg_operator: Any,
    X0: Float[Array, 'n k'],
) -> _EigPair:
    """Run the iterative method selected by ``spec.method`` (a static,
    trace-time branch) against the operand's matvec."""
    if spec.method == 'lobpcg':
        return _run_lobpcg_method(lobpcg_operator, X0, spec.n_iters, spec.tol)
    if spec.method == 'shift_invert':
        return _run_shift_invert_method(
            apply,
            X0,
            spec.sigma,
            spec.outer_iters,
            spec.cg_iters,
        )
    if spec.method == 'poly':
        return _run_poly_method(
            apply,
            X0,
            spec.degree,
            spec.shift,
            spec.outer_iters,
        )
    raise ValueError(
        f'_forward_for_spec: method={spec.method!r} is not iterative.'
    )


def _sectioned_matvec(
    values_tuple: _ValuesTuple,
    indices_tuple: _IndicesTuple,
    row_groups_tuple: _RowGroupsTuple,
    X: Float[Array, 'n k'],
    n_cols: int,
    n_rows: int,
) -> Float[Array, 'n_rows k']:
    """SectionedELL matvec: per-bucket matmul + scatter-back."""
    out = jnp.zeros((n_rows,) + X.shape[1:], dtype=X.dtype)
    for vals, idx, row_idx in zip(
        values_tuple, indices_tuple, row_groups_tuple
    ):
        bucket_out = semiring_ell_matmul(
            vals,
            idx,
            X,
            semiring=REAL,
            n_cols=n_cols,
            backend='jax',
        )
        out = out.at[row_idx].set(bucket_out)
    return out


def _sectioned_rmatvec(
    values_tuple: _ValuesTuple,
    indices_tuple: _IndicesTuple,
    row_groups_tuple: _RowGroupsTuple,
    X: Float[Array, 'n k'],
    n_cols: int,
    n_rows: int,
) -> Float[Array, 'n_cols k']:
    """SectionedELL adjoint matvec ``Aᵀ X``: per-bucket gather of the source
    rows + scatter into the shared ``n_cols`` axis, summed over buckets.

    The tuple-level companion of ``sparse.sectioned_semiring_ell_rmatvec``
    (kept here so the ``custom_vjp`` forward runs on the same flat arrays it
    differentiates, mirroring ``_sectioned_matvec``)."""
    out = jnp.zeros((n_cols,) + X.shape[1:], dtype=X.dtype)
    for vals, idx, row_idx in zip(
        values_tuple, indices_tuple, row_groups_tuple
    ):
        out = out + semiring_ell_rmatvec(
            vals,
            idx,
            X[row_idx],
            semiring=REAL,
            n_cols=n_cols,
        )
    return out


# ---------------------------------------------------------------------------
# Per-format custom_vjp entry points (parameterised by the static SolverSpec)
# ---------------------------------------------------------------------------


@partial(jax.custom_vjp, nondiff_argnums=(2,))
def _eig_top_k_dense(
    M: Float[Array, 'n n'],
    X0: Float[Array, 'n k'],
    spec: SolverSpec,
) -> _EigPair:
    """Iterative top-k for a concrete dense symmetric ``M``; diff w.r.t.
    ``M``.  Forward branches on ``spec.method``; backward is the shared
    dense projection."""
    return _forward_for_spec(spec, lambda X: M @ X, M, X0)


def _eig_top_k_dense_fwd(
    M: Float[Array, 'n n'],
    X0: Float[Array, 'n k'],
    spec: SolverSpec,
) -> Tuple[_EigPair, _EigPair]:
    eigvals, eigvecs = _forward_for_spec(spec, lambda X: M @ X, M, X0)
    return (eigvals, eigvecs), (eigvals, eigvecs)


def _eig_top_k_dense_bwd(
    spec: SolverSpec,
    residuals: _EigPair,
    cotangents: _EigPair,
) -> Tuple[Float[Array, 'n n'], None]:
    eigvals, eigvecs = residuals
    g_eigvals, g_eigvecs = cotangents
    K = _subspace_vjp_kernel(
        eigvals,
        eigvecs,
        g_eigvals,
        g_eigvecs,
        spec.eps_clamp,
    )
    return (_project_dense(K, eigvecs), None)


_eig_top_k_dense.defvjp(_eig_top_k_dense_fwd, _eig_top_k_dense_bwd)


@partial(jax.custom_vjp, nondiff_argnums=(3, 4))
def _eig_top_k_ell(
    values: Float[Array, 'n k_max'],
    indices: Int[Array, 'n k_max'],
    X0: Float[Array, 'n k'],
    n_cols: int,
    spec: SolverSpec,
) -> _EigPair:
    """Iterative top-k for an ELL-stored operator; diff w.r.t. ``values``
    (projected onto the sparsity pattern)."""
    return _ell_forward(values, indices, X0, n_cols, spec)


def _ell_forward(
    values: Float[Array, 'n k_max'],
    indices: Int[Array, 'n k_max'],
    X0: Float[Array, 'n k'],
    n_cols: int,
    spec: SolverSpec,
) -> _EigPair:
    if not spec.promise_symmetry and values.shape[0] != n_cols:
        raise ValueError(
            'eigsolve_top_k(promise_symmetry=False): symmetrising requires a '
            f'square operator, but the ELL has n_rows={values.shape[0]} != '
            f'n_cols={n_cols}.'
        )

    def matvec(X: Float[Array, 'n k']) -> Float[Array, 'n k']:
        return semiring_ell_matmul(
            values,
            indices,
            X,
            semiring=REAL,
            n_cols=n_cols,
            backend='jax',
        )

    if spec.promise_symmetry:
        apply = matvec
    else:
        # Apply the symmetric part ½(A X + Aᵀ X); the backward
        # (_project_ell) is already exact for this -- see its docstring.
        def apply(X: Float[Array, 'n k']) -> Float[Array, 'n k']:
            Ax = matvec(X)
            Atx = semiring_ell_rmatvec(
                values, indices, X, semiring=REAL, n_cols=n_cols
            )
            return 0.5 * (Ax + Atx)

    return _forward_for_spec(spec, apply, apply, X0)


def _eig_top_k_ell_fwd(
    values: Float[Array, 'n k_max'],
    indices: Int[Array, 'n k_max'],
    X0: Float[Array, 'n k'],
    n_cols: int,
    spec: SolverSpec,
) -> Tuple[
    _EigPair,
    Tuple[Int[Array, 'n k_max'], Float[Array, 'k'], Float[Array, 'n k']],
]:
    eigvals, eigvecs = _ell_forward(values, indices, X0, n_cols, spec)
    return (eigvals, eigvecs), (indices, eigvals, eigvecs)


def _eig_top_k_ell_bwd(
    n_cols: int,
    spec: SolverSpec,
    residuals: Tuple[
        Int[Array, 'n k_max'], Float[Array, 'k'], Float[Array, 'n k']
    ],
    cotangents: _EigPair,
) -> Tuple[Float[Array, 'n k_max'], Int[Array, 'n k_max'], None]:
    indices, eigvals, eigvecs = residuals
    g_eigvals, g_eigvecs = cotangents
    K = _subspace_vjp_kernel(
        eigvals,
        eigvecs,
        g_eigvals,
        g_eigvecs,
        spec.eps_clamp,
    )
    g_values = _project_ell(K, eigvecs, indices)
    return (g_values, jnp.zeros_like(indices), None)


_eig_top_k_ell.defvjp(_eig_top_k_ell_fwd, _eig_top_k_ell_bwd)


@partial(jax.custom_vjp, nondiff_argnums=(4, 5))
def _eig_top_k_sectioned(
    values_tuple: _ValuesTuple,
    indices_tuple: _IndicesTuple,
    row_groups_tuple: _RowGroupsTuple,
    X0: Float[Array, 'n k'],
    n_cols: int,
    spec: SolverSpec,
) -> _EigPair:
    """Iterative top-k for a SectionedELL operator; diff w.r.t. each
    section's ``values`` independently."""
    return _sectioned_forward(
        values_tuple,
        indices_tuple,
        row_groups_tuple,
        X0,
        n_cols,
        spec,
    )


def _sectioned_forward(
    values_tuple: _ValuesTuple,
    indices_tuple: _IndicesTuple,
    row_groups_tuple: _RowGroupsTuple,
    X0: Float[Array, 'n k'],
    n_cols: int,
    spec: SolverSpec,
) -> _EigPair:
    n_rows = X0.shape[0]
    if not spec.promise_symmetry and n_rows != n_cols:
        raise ValueError(
            'eigsolve_top_k(promise_symmetry=False): symmetrising requires a '
            f'square operator, but the SectionedELL has n_rows={n_rows} != '
            f'n_cols={n_cols}.'
        )

    def matvec(X: Float[Array, 'n k']) -> Float[Array, 'n k']:
        return _sectioned_matvec(
            values_tuple,
            indices_tuple,
            row_groups_tuple,
            X,
            n_cols=n_cols,
            n_rows=n_rows,
        )

    if spec.promise_symmetry:
        apply = matvec
    else:

        def apply(X: Float[Array, 'n k']) -> Float[Array, 'n k']:
            Ax = matvec(X)
            Atx = _sectioned_rmatvec(
                values_tuple,
                indices_tuple,
                row_groups_tuple,
                X,
                n_cols=n_cols,
                n_rows=n_rows,
            )
            return 0.5 * (Ax + Atx)

    return _forward_for_spec(spec, apply, apply, X0)


_SectionedResiduals = Tuple[
    _IndicesTuple, _RowGroupsTuple, Float[Array, 'k'], Float[Array, 'n k']
]


def _eig_top_k_sectioned_fwd(
    values_tuple: _ValuesTuple,
    indices_tuple: _IndicesTuple,
    row_groups_tuple: _RowGroupsTuple,
    X0: Float[Array, 'n k'],
    n_cols: int,
    spec: SolverSpec,
) -> Tuple[_EigPair, _SectionedResiduals]:
    eigvals, eigvecs = _sectioned_forward(
        values_tuple,
        indices_tuple,
        row_groups_tuple,
        X0,
        n_cols,
        spec,
    )
    return (eigvals, eigvecs), (
        indices_tuple,
        row_groups_tuple,
        eigvals,
        eigvecs,
    )


def _eig_top_k_sectioned_bwd(
    n_cols: int,
    spec: SolverSpec,
    residuals: _SectionedResiduals,
    cotangents: _EigPair,
) -> Tuple[_ValuesTuple, _IndicesTuple, _RowGroupsTuple, None]:
    indices_tuple, row_groups_tuple, eigvals, eigvecs = residuals
    g_eigvals, g_eigvecs = cotangents
    K = _subspace_vjp_kernel(
        eigvals,
        eigvecs,
        g_eigvals,
        g_eigvecs,
        spec.eps_clamp,
    )
    g_values = _project_sectioned(K, eigvecs, indices_tuple, row_groups_tuple)
    g_indices = tuple(jnp.zeros_like(i) for i in indices_tuple)
    g_row_groups = tuple(jnp.zeros_like(r) for r in row_groups_tuple)
    return (g_values, g_indices, g_row_groups, None)


_eig_top_k_sectioned.defvjp(
    _eig_top_k_sectioned_fwd,
    _eig_top_k_sectioned_bwd,
)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def _format_of(operand: EigOperand) -> _Format:
    if isinstance(operand, ELL):
        return 'ell'
    if isinstance(operand, SectionedELL):
        return 'sectioned'
    return 'dense'


def _n_of(operand: EigOperand) -> int:
    if isinstance(operand, (ELL, SectionedELL)):
        return operand.n_cols
    return int(operand.shape[-1])


def _dtype_of(operand: EigOperand) -> Any:
    if isinstance(operand, ELL):
        return operand.values.dtype
    if isinstance(operand, SectionedELL):
        return operand.sections[0].values.dtype
    return operand.dtype


def _resolve_method(method: Method, fmt: _Format) -> Method:
    """``auto`` -> ``eigh`` for dense, ``lobpcg`` for sparse.  Format-based
    only (static, JIT-stable); no runtime tracer inspection."""
    if method != 'auto':
        return method
    return 'eigh' if fmt == 'dense' else 'lobpcg'


def _device_put_operand(operand: EigOperand, target: Any) -> EigOperand:
    """Move all array fields of an operand to ``target`` (dense / ELL /
    SectionedELL); non-array fields pass through."""
    if isinstance(operand, ELL):
        return ELL(
            values=jax.device_put(operand.values, target),
            indices=jax.device_put(operand.indices, target),
            n_cols=operand.n_cols,
            identity=operand.identity,
        )
    if isinstance(operand, SectionedELL):
        new_sections = tuple(
            ELL(
                values=jax.device_put(s.values, target),
                indices=jax.device_put(s.indices, target),
                n_cols=s.n_cols,
                identity=s.identity,
            )
            for s in operand.sections
        )
        return SectionedELL(
            sections=new_sections,
            row_groups=operand.row_groups,
            n_rows=operand.n_rows,
            n_cols=operand.n_cols,
            identity=operand.identity,
        )
    return cast(EigOperand, jax.device_put(operand, target))


def _initial_subspace(
    n: int,
    k: int,
    seed: int,
    device: Any,
    dtype: Any,
) -> Float[Array, 'n k']:
    """Random initial subspace on the solver device, dtype-matched to the
    operator (so the x64 backward returns a matching cotangent dtype)."""
    key = jax.random.key(seed)
    return cast(
        Float[Array, 'n k'],
        jax.device_put(jax.random.normal(key, (n, k), dtype=dtype), device),
    )


def _eigh_top_k(
    M: Float[Array, '... n n'],
    k: int,
) -> _EigPair:
    """Dense full ``safe_eigh`` sliced to the ``k`` largest, largest-first."""
    eigvals, eigvecs = safe_eigh(M)  # ascending
    top_vals = eigvals[..., ::-1][..., :k]
    top_vecs = eigvecs[..., ::-1][..., :k]
    return top_vals, top_vecs


def _dispatch_iterative(
    operand: EigOperand,
    X0: Float[Array, 'n k'],
    spec: SolverSpec,
) -> _EigPair:
    if isinstance(operand, ELL):
        return _eig_top_k_ell(
            operand.values,
            operand.indices,
            X0,
            operand.n_cols,
            spec,
        )
    if isinstance(operand, SectionedELL):
        values_tuple = tuple(s.values for s in operand.sections)
        indices_tuple = tuple(s.indices for s in operand.sections)
        row_groups_tuple = tuple(jnp.asarray(rg) for rg in operand.row_groups)
        return _eig_top_k_sectioned(
            values_tuple,
            indices_tuple,
            row_groups_tuple,
            X0,
            operand.n_cols,
            spec,
        )
    return _eig_top_k_dense(operand, X0, spec)


def eigsolve_top_k(
    operand: EigOperand,
    k: int,
    *,
    spec: Optional[SolverSpec] = None,
    seed: int = 0,
) -> EigPair:
    """The ``k`` largest eigenpairs of a symmetric operator, largest-first.

    Parameters
    ----------
    operand
        Symmetric operator: dense ``(..., n, n)``, ``ELL``, or
        ``SectionedELL``.  The iterative methods (and shift-invert's inner
        CG) assume a symmetric operator.  For dense input the caller owns
        symmetry (symmetrise ``M`` first).  For ELL / SectionedELL, set
        ``spec.promise_symmetry=False`` when the *stored* operator is not
        guaranteed symmetric -- e.g. an affinity sparsified by top-k-per-row
        (``ell_from_dense``), whose pattern is generally asymmetric -- and
        the forward will apply the symmetric part ``½(A X + Aᵀ X)`` instead
        of trusting the stored pattern.  Leaving it ``True`` (the default)
        applies the stored matvec directly and is correct only when the
        operator really is symmetric (regular meshes / grids).
    k
        Number of extremal (largest) eigenpairs to return.
    spec
        Static solver configuration; defaults to ``SolverSpec.auto()``
        (dense -> ``eigh``, sparse -> ``lobpcg``).  Carries
        ``promise_symmetry`` (see ``operand``).
    seed
        PRNG seed for the iterative-solver initial subspace.

    Returns
    -------
    ``EigPair(values, vectors)`` -- ``values`` shape ``(k,)`` largest-first,
    ``vectors`` shape ``(n, k)``.

    Notes
    -----
    Differentiable w.r.t. ``operand`` (dense ``M``; ELL / SectionedELL
    ``values``) for every method: ``eigh`` via the native ``eigh`` VJP, the
    iterative methods via the shared implicit VJP.  ``skip_trivial`` /
    spectral-convention post-processing is the *caller's* concern -- this
    returns the raw extremal top-k.
    """
    if spec is None:
        spec = SolverSpec.auto()
    fmt = _format_of(operand)
    method = _resolve_method(spec.method, fmt)
    if fmt not in _SUPPORTED[method]:
        supported = sorted(_SUPPORTED[method])
        detail = (
            f'{method!r} supports dense input only'
            if supported == ['dense']
            else f'{method!r} supports {supported}'
        )
        raise ValueError(
            f'eigsolve_top_k: method={method!r} does not support {fmt!r} '
            f'input; {detail}.'
        )
    if method != spec.method:
        spec = replace(spec, method=method)

    source = source_device(operand)
    target = solver_device()
    operand = _device_put_operand(operand, target)

    if method == 'eigh':
        # ``eigh`` is dense-only (enforced by the validity table above), so
        # ``operand`` is a dense array here; narrow the operand union.
        eigvals, eigvecs = _eigh_top_k(
            cast(Float[Array, '... n n'], operand),
            k,
        )
    else:
        n = _n_of(operand)
        X0 = _initial_subspace(n, k, seed, target, _dtype_of(operand))
        eigvals, eigvecs = _dispatch_iterative(operand, X0, spec)

    if source is not None and source != target:
        eigvals = jax.device_put(eigvals, source)
        eigvecs = jax.device_put(eigvecs, source)
    return EigPair(eigvals, eigvecs)
