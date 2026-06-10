# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Pure-JAX reference implementations of the semiring kernels.

These functions are the **correctness floor** for the library.  Every
Pallas kernel is checked against the corresponding reference in the
backend-parity tests; if a Pallas kernel is unable to tile a given shape
× algebra combination it falls back to the reference (with a
``NitrixBackendFallback`` warning).

The reference walks the contraction dim with ``lax.fori_loop`` and the
supplied ``Monoid.update``, so the pytree-state pattern is exercised on
the JAX side identically to the Pallas side.
"""

from __future__ import annotations

from typing import TypeVar, cast

import jax.lax as lax
import jax.numpy as jnp
from jaxtyping import Array, Num

from ._types import Semiring

S = TypeVar('S')


def reference_semiring_matmul(
    A: Num[Array, 'm k'],
    B: Num[Array, 'k n'],
    *,
    semiring: Semiring[S],
) -> Num[Array, 'm n']:
    """Pure-JAX reference for ``semiring_matmul``.

    Computes ``C[i, j] = (+)_k ( A[i, k] (*) B[k, j] )``
    via ``lax.fori_loop`` over the contraction axis with the supplied
    ``semiring.monoid.update``.

    Parameters
    ----------
    A
        Left operand, shape ``(m, k)``.
    B
        Right operand, shape ``(k, n)``.
    semiring
        The algebra to reduce under.  Either ``Semiring`` (relaxed) or
        ``StrictSemiring``; this function fixes the reduction order so
        the relaxed type is sufficient.

    Returns
    -------
    Array of shape ``(m, n)`` and the same dtype as ``A``.

    Notes
    -----
    No batching dims here; ``semiring_matmul`` in ``matmul.py``
    handles broadcast over leading batch dimensions by ``vmap``-ing
    over this reference.  Keeping the reference 2-D simplifies the
    Pallas parity contract.
    """
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError(
            f'reference_semiring_matmul: expected 2-D A and B, '
            f'got A.shape={A.shape}, B.shape={B.shape}'
        )
    if A.shape[1] != B.shape[0]:
        raise ValueError(
            f'reference_semiring_matmul: A.shape={A.shape}, '
            f'B.shape={B.shape} not compatible.'
        )
    m, k = A.shape
    _, n = B.shape
    monoid = semiring.monoid
    binary_op = semiring.binary_op
    acc_init = monoid.init((m, n), A.dtype)

    def body(kk: int, acc: S) -> S:
        a_col = lax.dynamic_slice_in_dim(A, kk, 1, axis=1)  # (m, 1)
        b_row = lax.dynamic_slice_in_dim(B, kk, 1, axis=0)  # (1, n)
        value = binary_op.combine(a_col, b_row)  # (m, n)
        return monoid.update(acc, value)

    acc = lax.fori_loop(0, k, body, acc_init)
    return monoid.finalize(acc)


def reference_semiring_ell_matmul(
    values: Num[Array, 'm kmax'],
    indices: Num[Array, 'm kmax'],
    B: Num[Array, 'n_cols ncol'],
    *,
    semiring: Semiring[S],
    n_cols: int | None = None,
) -> Num[Array, 'm ncol']:
    """Pure-JAX reference for ``semiring_ell_matmul``.

    The implicit M×N sparse operand has the per-row neighbour list
    ``indices[i, :]`` with weights ``values[i, :]``.  Output is::

        C[i, j] = (+)_p ( values[i, p] (*) B[indices[i, p], j] )

    where ``p`` runs over the ``k_max`` columns of the ELL row.

    Parameters
    ----------
    values
        ELL values, shape ``(m, k_max)``.
    indices
        ELL column indices into ``B``'s outer dim, shape ``(m, k_max)``.
        Padding positions must point at a valid row of ``B`` and have
        ``values`` set to the semiring identity (i.e., ``B``'s row at
        that index contributes a no-op).  The caller is responsible for
        this; see ``nitrix.sparse.ell.ell_pad``.
    B
        Dense right operand, shape ``(n_cols, ncol)``.
    semiring
        Algebra to reduce under.
    n_cols
        Outer dim of the implicit sparse operand (``B.shape[0]``).
        Optional; defaults to ``B.shape[0]`` and is asserted equal.

    Returns
    -------
    ``C``, shape ``(m, ncol)``.
    """
    if values.shape != indices.shape:
        raise ValueError(
            f'reference_semiring_ell_matmul: values.shape={values.shape}, '
            f'indices.shape={indices.shape} mismatch.'
        )
    if B.ndim != 2:
        raise ValueError(
            f'reference_semiring_ell_matmul: B must be 2-D, '
            f'got shape={B.shape}'
        )
    if n_cols is not None and n_cols != B.shape[0]:
        raise ValueError(
            f'reference_semiring_ell_matmul: n_cols={n_cols} does not '
            f'match B.shape[0]={B.shape[0]}.'
        )
    m, kmax = values.shape
    _, ncol = B.shape
    monoid = semiring.monoid
    binary_op = semiring.binary_op
    acc_init = monoid.init((m, ncol), B.dtype)

    def body(p: int, acc: S) -> S:
        v_p = lax.dynamic_slice_in_dim(values, p, 1, axis=1)  # (m, 1)
        idx_p = lax.dynamic_slice_in_dim(indices, p, 1, axis=1)
        idx_p = idx_p[:, 0]  # (m,)
        gathered = B[idx_p]  # (m, ncol)
        value = binary_op.combine(v_p, gathered)  # (m, ncol)
        return monoid.update(acc, value)

    acc = lax.fori_loop(0, kmax, body, acc_init)
    return monoid.finalize(acc)


def reference_semiring_ell_rmatvec(
    values: Num[Array, 'm kmax'],
    indices: Num[Array, 'm kmax'],
    X: Num[Array, 'm ncol'],
    *,
    semiring: Semiring[S],
    n_cols: int,
) -> Num[Array, 'n_cols ncol']:
    """REAL adjoint (transpose) of the ELL matvec: ``Y = Aᵀ X``.

    Where ``semiring_ell_matmul`` gathers -- ``(A X)[i] = Σ_p values[i, p]
    · X[indices[i, p]]`` -- this scatters::

        Y[c, j] = Σ_{(i, p) : indices[i, p] == c} values[i, p] · X[i, j]

    for the same implicit ``m × n_cols`` operand ``A``.  This is the
    **additive** adjoint: the scatter reduction is ``+`` (not a general
    monoid), so pad positions -- whose ``values`` carry REAL's additive
    identity ``0`` -- contribute nothing, exactly as in the gather
    direction.  It is the single source of truth for both the ``g_B`` term
    of ``real_ell_matmul_vjp`` and the symmetric matvec ``½(A X + Aᵀ X)``
    used by the spectral solvers on adjacencies whose top-k construction
    did not preserve symmetry.

    ``semiring`` mirrors ``reference_semiring_ell_matmul``'s signature, but
    only **REAL** is implemented: the additive scatter has no meaning for a
    general monoid (``½(A + Aᵀ)`` averaging needs the linear structure, and
    a non-zero pad identity would inject spurious mass).  Any other algebra
    raises ``NotImplementedError`` rather than silently mis-reducing.

    Parameters
    ----------
    values
        ELL values, shape ``(m, k_max)``.  Pad positions must hold ``0``.
    indices
        ELL column indices, shape ``(m, k_max)``; the scatter targets.
    X
        Dense operand indexed by the ELL *row*, shape ``(m, ncol)``.
    semiring
        The algebra to reduce under.  REAL only (see above).
    n_cols
        Outer dim of the implicit operand ``A`` -- the length of ``Y``'s
        leading axis (the scatter range).

    Returns
    -------
    ``Y``, shape ``(n_cols, ncol)``.
    """
    if semiring.name != 'real':
        raise NotImplementedError(
            f'reference_semiring_ell_rmatvec: the additive ELL adjoint is '
            f'defined for the REAL semiring only; got {semiring.name!r}.  '
            '(½(A + Aᵀ) symmetrisation needs linear structure, and a '
            'non-zero pad identity would scatter spurious mass.)'
        )
    if values.shape != indices.shape:
        raise ValueError(
            f'reference_semiring_ell_rmatvec: values.shape={values.shape}, '
            f'indices.shape={indices.shape} mismatch.'
        )
    if X.ndim != 2:
        raise ValueError(
            f'reference_semiring_ell_rmatvec: X must be 2-D, got shape={X.shape}'
        )
    _, kmax = values.shape
    _, ncol = X.shape
    out_dtype = jnp.result_type(values.dtype, X.dtype)
    acc_init = jnp.zeros((n_cols, ncol), dtype=out_dtype)

    def body(p: int, acc: Num[Array, 'n_cols ncol']) -> Num[Array, 'n_cols ncol']:
        idx_p = lax.dynamic_slice_in_dim(indices, p, 1, axis=1)[:, 0]  # (m,)
        v_p = lax.dynamic_slice_in_dim(values, p, 1, axis=1)  # (m, 1)
        return acc.at[idx_p].add(v_p * X)

    return cast(
        Num[Array, 'n_cols ncol'], lax.fori_loop(0, kmax, body, acc_init)
    )


__all__ = [
    'reference_semiring_matmul',
    'reference_semiring_ell_matmul',
    'reference_semiring_ell_rmatvec',
]
