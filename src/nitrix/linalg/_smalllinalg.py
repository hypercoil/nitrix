# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Tiny dense symmetric linear algebra, size-dispatched and cuSOLVER-free.

Inverse-plus-log-determinant of a small ``(p, p)`` symmetric-positive-definite
matrix, and the symmetric eigendecomposition of one -- the per-element kernels
behind a ``vmap`` over up to ~1M independent small solves.  At that batch the
choice of routine matters twice over:

- **Performance / compile.**  Differentiating a batched ``cholesky`` makes the
  XLA:CPU compile scale with the batch and inflates the autodiff tape.
- **GPU availability.**  On the dev L4 the cuSOLVER handle pool is dead --
  ``potrf`` (Cholesky), ``syevd`` (eigh), and even ``getrf`` (LU) fail to
  create a handle as the first cuSOLVER routine in a process (see
  ``docs/feature-requests/gpu-cusolver-first-call-handle-failure.md``).  Only
  cuBLAS (matmul, ``triangular_solve`` / ``trsm``) and the elementwise surface
  work.

So we issue **no cuSOLVER custom-call at any** :math:`p`: closed-form algebra for
:math:`p \\in \\{1, 2\\}` (the common small designs), and for :math:`p > 2` a
hand-Cholesky whose inverse goes through ``triangular_solve`` (cuBLAS ``trsm``).
Everything is differentiable.  :func:`sym_eig_jacobi` is the matching
eigendecomposition: a fixed-sweep cyclic Jacobi rotation (matmul-only, no
cuSOLVER ``syevd``).

Why a *rolled* Cholesky
-----------------------

The Cholesky loop is run as a ``lax.fori_loop`` over the :math:`p` columns,
**not** unrolled at trace time.  A trace-time unroll produces :math:`O(p^3)`
scalar ops in the graph, so compile time grows cubically in :math:`p` --
negligible at :math:`p \\le 5` (~0.4 s) but ruinous once :math:`p` reaches 20-30
(the unrolled form measured ~29 s for a single :math:`p = 30` inverse).  Rolling
the loop makes the graph :math:`O(p^2)` (one compiled body), so compile is ~flat
in :math:`p` (~0.4 s at :math:`p = 30`); runtime is unchanged (still
:math:`O(p^3)` flops).  The closed forms for :math:`p \\in \\{1, 2\\}` are
untouched.

Numerical precision (x64 expectation)
-------------------------------------

These kernels factor the *normal-equation* matrix :math:`X^{\\top} W X` (or a
per-group Gram), whose condition number is the **square** of the design's -- so
an ill-conditioned design squares into a near-singular pivot.  The
modified-Cholesky pivot floor (:func:`_pivot_rel_floor`,
``~1e2 x finfo(dtype).eps``) keeps such a
solve *finite and regularised* rather than ``NaN``, but it cannot restore digits
roundoff has already lost.  For ill-conditioned designs the suite **expects
float64** (``jax.config.update('jax_enable_x64', True)``): in float32 the floor
fires far sooner (eps ~1.2e-5 vs ~2.2e-14), so the fit silently leans on the
regulariser.  Prefer x64, or pre-condition / ridge the design, when the
normal-equation condition number approaches ``1 / eps``.
"""

from __future__ import annotations

from typing import Tuple, cast

import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Float

__all__ = [
    'small_det',
    'small_inv',
    'small_inv_logdet',
    'spd_inv_logdet_chol',
    'spd_chol',
    'sym_eig_jacobi',
]

# Largest matrix the closed-form adjugate inverse covers: the cofactor graph
# grows factorially, so it is reserved for the small general blocks (affine /
# triangular, n in {2, 3, 4}) the callers actually pass.
_ADJ_INV_MAX_N = 4

# Relative pivot-floor multiplier (modified Cholesky), ~1e2 x the dtype's
# machine epsilon.
_PIVOT_REL_EPS_MULT = 100.0


def _pivot_rel_floor(dtype: jnp.dtype) -> Float[Array, '']:
    """Relative pivot floor, **dtype-aware**: :math:`\\sim 10^{2} \\times`
    ``finfo(dtype).eps``.

    A pivot or determinant is clamped to this fraction of the matrix's diagonal
    scale before a ``sqrt`` or division, so a near-singular or boundary system
    (where the caller's ridge was too small to keep the pivot positive against
    roundoff) yields a *regularised, finite* solve instead of a silent ``NaN``.

    Tying it to ``eps`` (fp32 ~1.2e-5, fp64 ~2.2e-14) keeps the floor just above
    the dtype's roundoff scale, so it sits below the smallest pivot of any
    well-conditioned matrix (it never perturbs a healthy solve) yet still
    catches a pivot that roundoff has driven non-positive.  The previous fixed
    ``1e-12`` was ~4 orders *below* fp32 eps, so in fp32 it was **inert** -- it
    could never rescue an fp32 pivot that had already gone negative -- and only
    ever did anything in fp64.

    Parameters
    ----------
    dtype : jnp.dtype
        Floating-point dtype of the matrix being factored; its machine epsilon
        sets the scale of the returned floor.

    Returns
    -------
    Float[Array, '']
        Scalar relative floor, :math:`10^{2} \\times` the machine epsilon of
        ``dtype``.  Callers multiply it by the matrix's diagonal scale to obtain
        the absolute clamp.
    """
    return jnp.asarray(_PIVOT_REL_EPS_MULT * jnp.finfo(dtype).eps)


def spd_inv_logdet_chol(
    A: Float[Array, 'n n'], n: int
) -> Tuple[Float[Array, 'n n'], Float[Array, '']]:
    """Inverse and log-determinant of a small SPD matrix via a rolled,
    cuSOLVER-free Cholesky.

    For :math:`n > 2` we deliberately avoid ``jnp.linalg.{inv,cholesky,slogdet}``
    -- on the broken-cuSOLVER L4 their ``getrf`` / ``potrf`` custom-calls fail
    to create a handle.  Instead we factor :math:`A = L L^{\\top}`
    column-by-column with a ``lax.fori_loop`` (rolled, so the graph stays
    :math:`O(n^2)` -- see the module docstring), take the inverse through
    ``triangular_solve`` (cuBLAS ``trsm``, which *does* work on these stacks),
    and read the log-determinant off the factor diagonal.  No cuSOLVER routine
    is issued; everything is differentiable (``sqrt``, division and ``trsm`` all
    have VJPs).

    The column update uses ``L @ L[j]``: at column :math:`j` the not-yet-filled
    entries of ``L`` are zero, so the full dot automatically restricts to the
    :math:`k < j` terms of the Cholesky-Banachiewicz recurrence.

    Parameters
    ----------
    A : Float[Array, 'n n']
        Symmetric positive-definite matrix to invert (callers add a ridge to
        keep it SPD).
    n : int
        Static matrix dimension (a Python int, the compile-time shape).

    Returns
    -------
    inv : Float[Array, 'n n']
        The inverse :math:`A^{-1} = L^{-\\top} L^{-1}`.
    log_det : Float[Array, '']
        The log-determinant :math:`\\log \\det A = 2 \\sum_i \\log L_{ii}`.
    """
    L = spd_chol(A, n)
    log_det = 2.0 * jnp.sum(jnp.log(jnp.diagonal(L)))
    eye = jnp.eye(n, dtype=A.dtype)
    l_inv = lax.linalg.triangular_solve(
        L, eye, left_side=True, lower=True, transpose_a=False
    )
    inv = l_inv.T @ l_inv  # A^{-1} = L^{-T} L^{-1}
    return inv, log_det


def spd_chol(A: Float[Array, 'n n'], n: int) -> Float[Array, 'n n']:
    """Lower-triangular Cholesky factor :math:`L` (:math:`A = L L^{\\top}`) of a
    small SPD matrix, rolled and cuSOLVER-free.

    The shared factor behind :func:`spd_inv_logdet_chol`: a column-by-column
    ``lax.fori_loop`` (rolled, :math:`O(n^2)` graph -- see the module docstring)
    using only ``sqrt`` and division (all with VJPs, so it is
    **differentiable**).  The modified-Cholesky pivot floor (relative to the
    diagonal scale) keeps a degenerate ``A`` finite (a regularised factor)
    rather than ``sqrt(negative) = NaN``.  Used to scale adaptive-quadrature
    nodes by a curvature factor (:math:`L L^{\\top} = H^{-1}`) as well as to
    invert or take the log-det of small SPD blocks.

    Parameters
    ----------
    A : Float[Array, 'n n']
        Symmetric positive-definite matrix to factor.  A degenerate input is
        regularised by the pivot floor rather than producing ``NaN``.
    n : int
        Static matrix dimension (a Python int, the compile-time shape).

    Returns
    -------
    Float[Array, 'n n']
        Lower-triangular factor :math:`L` with :math:`A = L L^{\\top}` (entries
        above the diagonal are zero).
    """
    idx = jnp.arange(n)
    floor = (
        _pivot_rel_floor(A.dtype) * jnp.max(jnp.diagonal(A))
        + jnp.finfo(A.dtype).tiny
    )

    def body(j: Array, L: Float[Array, 'n n']) -> Float[Array, 'n n']:
        Lj = lax.dynamic_index_in_dim(L, j, axis=0, keepdims=False)  # (n,)
        s = L @ Lj  # s_i = sum_{k<j} L[i,k] L[j,k]
        diag = jnp.sqrt(jnp.maximum(A[j, j] - s[j], floor))
        col = (A[:, j] - s) / diag
        new_col = jnp.where(idx == j, diag, jnp.where(idx > j, col, 0.0))
        return cast(
            Float[Array, 'n n'],
            lax.dynamic_update_index_in_dim(L, new_col, j, axis=1),
        )

    return cast(
        Float[Array, 'n n'],
        lax.fori_loop(0, n, body, jnp.zeros((n, n), A.dtype)),
    )


def small_inv_logdet(
    A: Float[Array, 'n n'], n: int
) -> Tuple[Float[Array, 'n n'], Float[Array, '']]:
    """Inverse and log-determinant of a small SPD ``(n, n)`` matrix.

    ``n`` is a Python int (a compile-time shape), so the branch is a
    Python-level ``if`` and **no branch issues a cuSOLVER custom-call**:

    - ``n == 1``  -- a reciprocal and a ``log``.
    - ``n == 2``  -- the explicit symmetric inverse ``[[c,-b],[-b,a]]/det`` and
      :math:`\\log(\\det)` with :math:`\\det = a c - b^2`.
    - ``n > 2``   -- a rolled hand-Cholesky and ``triangular_solve`` (cuBLAS
      ``trsm``); see :func:`spd_inv_logdet_chol`.

    ``A`` is assumed SPD (the callers add a ridge).

    Parameters
    ----------
    A : Float[Array, 'n n']
        Symmetric positive-definite matrix to invert.
    n : int
        Static matrix dimension (a Python int, the compile-time shape) that
        selects the closed-form or Cholesky branch.

    Returns
    -------
    inv : Float[Array, 'n n']
        The inverse :math:`A^{-1}`.
    log_det : Float[Array, '']
        The log-determinant :math:`\\log \\det A`.
    """
    tiny = jnp.finfo(A.dtype).tiny
    rel = _pivot_rel_floor(A.dtype)
    if n == 1:
        # Floor a non-positive / zero pivot (roundoff or constant input) so the
        # reciprocal and log stay finite; never perturbs an SPD scalar.
        a = jnp.maximum(A[0, 0], rel * jnp.abs(A[0, 0]) + tiny)
        return (1.0 / a)[None, None], jnp.log(a)
    if n == 2:
        a = A[0, 0]
        b = A[0, 1]
        c = A[1, 1]
        # Floor the determinant relative to the diagonal product, so a
        # near-singular / indefinite 2x2 gives a finite regularised inverse.
        det = jnp.maximum(a * c - b * b, rel * a * c + tiny)
        inv = jnp.array([[c, -b], [-b, a]], dtype=A.dtype) / det
        return inv, jnp.log(det)
    return spd_inv_logdet_chol(A, n)


def _delete_row_col(
    A: Float[Array, '... n n'], i: int, j: int, n: int
) -> Float[Array, '... m m']:
    """The ``(n-1, n-1)`` minor of ``A`` with row ``i`` and column ``j`` removed.

    ``i``, ``j`` and ``n`` are Python ints (compile-time), so the kept-index
    lists are static and the gather is a plain advanced index -- traceable and
    differentiable.

    Parameters
    ----------
    A : Float[Array, '... n n']
        Batched square matrix (leading dimensions are broadcast).
    i : int
        Index of the row to remove (a Python int, compile-time).
    j : int
        Index of the column to remove (a Python int, compile-time).
    n : int
        Static matrix dimension of ``A`` (a Python int, compile-time).

    Returns
    -------
    Float[Array, '... m m']
        The minor with the given row and column deleted, of trailing shape
        ``(n-1, n-1)`` (i.e. ``m = n - 1``); the leading batch dimensions are
        preserved.
    """
    rows = [r for r in range(n) if r != i]
    cols = [c for c in range(n) if c != j]
    return A[..., rows, :][..., :, cols]


def small_det(A: Float[Array, '... n n'], n: int) -> Float[Array, '...']:
    """Determinant of a small general matrix by cofactor expansion.

    Computes the determinant of a batched ``(..., n, n)`` matrix by Laplace
    (cofactor) expansion -- cuSOLVER-free, batched over the leading dimensions,
    and differentiable.

    ``n`` is a Python int (compile-time), so the expansion unrolls into pure
    multiply and add (no ``getrf`` LU custom-call); it is reserved for small
    ``n`` as the term count grows factorially.  Closed forms are used for
    :math:`n \\in \\{1, 2\\}`.

    Parameters
    ----------
    A : Float[Array, '... n n']
        Batched square matrix (need not be symmetric).
    n : int
        Static matrix dimension (a Python int, the compile-time shape).

    Returns
    -------
    Float[Array, '...']
        The determinant, one scalar per leading batch element.
    """
    if n == 1:
        return A[..., 0, 0]
    if n == 2:
        return A[..., 0, 0] * A[..., 1, 1] - A[..., 0, 1] * A[..., 1, 0]
    acc = None
    for j in range(n):
        term = A[..., 0, j] * small_det(_delete_row_col(A, 0, j, n), n - 1)
        signed = term if j % 2 == 0 else -term
        acc = signed if acc is None else acc + signed
    return cast(Float[Array, '...'], acc)


def _adjugate(A: Float[Array, '... n n'], n: int) -> Float[Array, '... n n']:
    """Adjugate (transpose of the cofactor matrix) of a small general matrix.

    Entrywise, :math:`\\operatorname{adj}(A)_{ij} = (-1)^{i+j}
    \\det(\\text{minor}_{ji})`, so :math:`A^{-1} = \\operatorname{adj}(A) /
    \\det(A)` (Cramer's rule).  Built from pure arithmetic, hence batched over
    the leading dimensions and differentiable.

    Parameters
    ----------
    A : Float[Array, '... n n']
        Batched square matrix (need not be symmetric).
    n : int
        Static matrix dimension (a Python int, the compile-time shape).

    Returns
    -------
    Float[Array, '... n n']
        The adjugate matrix, matching the shape of ``A``.
    """
    if n == 1:
        return jnp.ones_like(A)
    rows_out = []
    for i in range(n):
        row = []
        for j in range(n):
            cof = small_det(_delete_row_col(A, j, i, n), n - 1)
            row.append(cof if (i + j) % 2 == 0 else -cof)
        rows_out.append(jnp.stack(row, axis=-1))
    return jnp.stack(rows_out, axis=-2)


def small_inv(A: Float[Array, '... n n'], n: int) -> Float[Array, '... n n']:
    """General (non-symmetric) inverse of a small ``(..., n, n)`` matrix.

    The cuSOLVER-free counterpart to :func:`small_inv_logdet` for the case the
    caller's matrix is a *general* small block (an affine, a triangular
    scale/shear factor) rather than an SPD Gram.  Closed-form adjugate inverse
    :math:`A^{-1} = \\operatorname{adj}(A) / \\det(A)` for :math:`n \\le 4`: pure
    multiply, add and divide, so it is batched over the leading dimensions,
    differentiable, and issues no cuSOLVER custom-call (unlike
    ``jnp.linalg.inv``'s ``getrf``).

    Unlike the SPD kernels this does **not** regularise -- a singular ``A``
    yields a non-finite result, matching ``jnp.linalg.inv`` (callers supply an
    invertible matrix: an affine, or a ridged system).  Cramer's rule is exact
    to roundoff for the well-conditioned tiny matrices this serves; prefer x64
    for an ill-conditioned input (see the module docstring).

    Parameters
    ----------
    A : Float[Array, '... n n']
        Batched square matrix (need not be symmetric); assumed invertible.
    n : int
        Static matrix dimension (a Python int, the compile-time shape).  Must be
        at most 4, else a ``ValueError`` is raised.

    Returns
    -------
    Float[Array, '... n n']
        The inverse :math:`A^{-1}`, matching the shape of ``A``.

    Raises
    ------
    ValueError
        If ``n`` exceeds the maximum dimension the adjugate inverse covers (4).
    """
    if n > _ADJ_INV_MAX_N:
        raise ValueError(
            f'small_inv supports n <= {_ADJ_INV_MAX_N}; got n={n}.'
        )
    det = small_det(A, n)
    return _adjugate(A, n) / det[..., None, None]


def sym_eig_jacobi(
    A: Float[Array, 'n n'], n: int, n_sweeps: int = 8
) -> Tuple[Float[Array, 'n'], Float[Array, 'n n']]:
    """Symmetric eigendecomposition of a small ``(n, n)`` matrix, cuSOLVER-free.

    Returns ``(evals, evecs)`` with :math:`A \\approx V \\operatorname{diag}(
    \\lambda) V^{\\top}`; the columns of ``evecs`` are orthonormal eigenvectors.
    Eigenvalues are **not sorted** (cyclic Jacobi visits pairs in a fixed
    order).

    Why a hand-rolled Jacobi.  ``jnp.linalg.eigh`` issues a ``syevd`` cuSOLVER
    custom-call, which is dead on the dev L4 (see :func:`small_inv_logdet`); and
    :func:`~nitrix.linalg._solver.safe_eigh`'s CPU fallback is eager (not ``jit`` /
    ``vmap`` -able), so it cannot run *inside* a per-element batched
    computation.  A fixed-sweep cyclic Jacobi is pure arithmetic (Givens
    rotations): jittable, ``vmap``-clean, and exact to machine precision in a
    handful of sweeps for the tiny ``n`` this serves (an :math:`L \\times L`
    contrast covariance, :math:`L` a few rows; the Fai-Cornelius
    denominator-df eigendirections of :func:`~nitrix.stats.lme_f_contrast`).

    **Forward-only.**  The decomposition is for the value, not its gradient: a
    naive reverse-mode sweep accumulates :math:`1 / \\text{off-diagonal}^2`
    rotation-angle terms that overflow as the iteration converges (the standard
    hazard of differentiating an iterative eigensolver).  Its sole consumer --
    the Satterthwaite F denominator df -- is not differentiated through, so that
    is by design; wrap a call in ``lax.stop_gradient`` if it ever feeds a
    differentiated path.

    Each rotation zeroes one off-diagonal ``A[p, q]`` via the orthogonal
    :math:`J(p, q)` (the stable smaller-root angle), applied as
    :math:`A \\leftarrow J^{\\top} A J` and :math:`V \\leftarrow V J`; ``n`` is a
    Python int, so the :math:`n(n-1)/2` pair sweep is unrolled and ``n_sweeps``
    cyclic passes drive the off-diagonal to zero quadratically.  ``n_sweeps=8``
    reaches machine precision for :math:`n \\le 7` (the largest matrix this
    serves -- a 7-component block-Woodbury Hessian; empirically :math:`n = 7`
    converges by 6 sweeps, :math:`n \\le 5` by 5).  The rotation is applied as
    the dense :math:`J^{\\top} A J` matmul rather than an explicit rank-2
    row/column update: the latter is fewer flops but unrolls into many small
    dynamic-update-slice fusions that the XLA:CPU JIT fails to materialise, and
    at this ``n`` the matmul cost is negligible.

    Parameters
    ----------
    A : Float[Array, 'n n']
        Matrix to decompose; it is symmetrised as :math:`(A + A^{\\top}) / 2`
        before the sweeps.
    n : int
        Static matrix dimension (a Python int, the compile-time shape).
    n_sweeps : int, optional
        Number of cyclic Jacobi passes over all :math:`n(n-1)/2` off-diagonal
        pairs.  The default of 8 reaches machine precision for :math:`n \\le 7`.

    Returns
    -------
    evals : Float[Array, 'n']
        Eigenvalues, read off the converged diagonal (**not sorted**).
    evecs : Float[Array, 'n n']
        Matrix whose columns are the corresponding orthonormal eigenvectors.
    """
    A = 0.5 * (A + A.T)
    dtype = A.dtype
    tiny = jnp.finfo(dtype).tiny
    pairs = [(p, q) for p in range(n) for q in range(p + 1, n)]

    def one_sweep(
        carry: Tuple[Float[Array, 'n n'], Float[Array, 'n n']], _: Array
    ) -> Tuple[Tuple[Float[Array, 'n n'], Float[Array, 'n n']], None]:
        mat, vecs = carry
        for p, q in pairs:
            apq = mat[p, q]
            app = mat[p, p]
            aqq = mat[q, q]
            do_rot = jnp.abs(apq) > tiny  # skip an already-zero off-diagonal
            denom = jnp.where(do_rot, 2.0 * apq, 1.0)
            tau = (aqq - app) / denom
            sgn = jnp.where(tau >= 0, 1.0, -1.0)
            t = sgn / (
                jnp.abs(tau) + jnp.sqrt(tau * tau + 1.0)
            )  # smaller root
            t = jnp.where(do_rot, t, 0.0)
            cos = 1.0 / jnp.sqrt(t * t + 1.0)
            sin = t * cos
            rot = jnp.eye(n, dtype=dtype)
            rot = rot.at[p, p].set(cos)
            rot = rot.at[q, q].set(cos)
            rot = rot.at[p, q].set(sin)
            rot = rot.at[q, p].set(-sin)
            mat = rot.T @ mat @ rot
            vecs = vecs @ rot
        return (mat, vecs), None

    (mat, vecs), _ = lax.scan(
        one_sweep, (A, jnp.eye(n, dtype=dtype)), xs=None, length=n_sweeps
    )
    return jnp.diagonal(mat), vecs
