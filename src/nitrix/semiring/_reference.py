# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Pure-JAX reference implementations of the semiring kernels.

These functions are the **correctness floor** for the library.  Every
Pallas kernel is checked against the corresponding reference in the
backend-parity tests; if a Pallas kernel is unable to tile a given shape
× algebra combination it falls back to the reference (with a
:class:`NitrixBackendFallback` warning).

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
    """Pure-JAX reference for :func:`semiring_matmul`.

    Computes the dense semiring matrix product
    :math:`C_{ij} = \\bigoplus_k \\left( A_{ik} \\otimes B_{kj} \\right)`,
    where :math:`\\oplus` is the monoid reduction and :math:`\\otimes` the
    per-element combine of the supplied algebra.  The contraction axis is
    walked with :func:`jax.lax.fori_loop`, folding each rank-one outer
    product into the accumulator via ``semiring.monoid.update``.

    Parameters
    ----------
    A : Num[Array, 'm k']
        Left operand, shape ``(m, k)``.
    B : Num[Array, 'k n']
        Right operand, shape ``(k, n)``.  Its leading axis is the
        contraction dimension shared with ``A``.
    semiring : Semiring
        The algebra to reduce under.  Either a relaxed :class:`Semiring`
        or a :class:`StrictSemiring`; this function fixes the reduction
        order, so the relaxed type is sufficient.

    Returns
    -------
    Num[Array, 'm n']
        The product ``C``, shape ``(m, n)``, with the same dtype as ``A``.

    Notes
    -----
    There are no batching dimensions here.  The public
    :func:`semiring_matmul` handles broadcast over leading batch
    dimensions by ``vmap``-ing over this reference.  Keeping the reference
    two-dimensional simplifies the Pallas parity contract.
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
    """Pure-JAX reference for :func:`semiring_ell_matmul`.

    The implicit :math:`m \\times n` sparse left operand is held in ELL
    layout: row ``i`` has neighbour list ``indices[i, :]`` with weights
    ``values[i, :]``.  Multiplying it against the dense right operand
    ``B`` gathers those neighbours from the rows of ``B`` and reduces them
    under the algebra,
    :math:`C_{ij} = \\bigoplus_p \\left( \\mathtt{values}_{ip} \\otimes
    B_{\\,\\mathtt{indices}_{ip},\\, j} \\right)`, where :math:`p` runs
    over the ``k_max`` columns of the ELL row.  The reduction is performed
    with :func:`jax.lax.fori_loop` over ``p``.

    Parameters
    ----------
    values : Num[Array, 'm kmax']
        ELL values, shape ``(m, k_max)``.
    indices : Num[Array, 'm kmax']
        ELL column indices into ``B``'s outer dimension, shape
        ``(m, k_max)``.  Padding positions must point at a valid row of
        ``B`` and carry a ``values`` entry equal to the semiring identity,
        so that ``B``'s row at that index contributes a no-op.  The caller
        is responsible for this; see :func:`ell_pad`.
    B : Num[Array, 'n_cols ncol']
        Dense right operand, shape ``(n_cols, ncol)``.
    semiring : Semiring
        The algebra to reduce under.
    n_cols : int or None, optional
        Outer dimension of the implicit sparse operand (``B.shape[0]``).
        Defaults to ``B.shape[0]`` and, when supplied, is asserted equal.

    Returns
    -------
    Num[Array, 'm ncol']
        The product ``C``, shape ``(m, ncol)``, with the same dtype as
        the accumulator initialised from ``B``.
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
    """REAL adjoint (transpose) of the ELL matvec: :math:`Y = A^{\\top} X`.

    Where :func:`semiring_ell_matmul` gathers --
    :math:`(A X)_i = \\sum_p \\mathtt{values}_{ip} \\cdot
    X_{\\,\\mathtt{indices}_{ip}}` -- this routine scatters,

    .. math::

        Y_{cj} = \\sum_{(i, p)\\,:\\,\\mathtt{indices}_{ip} = c}
            \\mathtt{values}_{ip} \\cdot X_{ij},

    for the same implicit :math:`m \\times n_{\\mathrm{cols}}` operand
    :math:`A`, accumulating with :func:`jax.lax.fori_loop` over the ELL
    columns.  This is the **additive** adjoint: the scatter reduction is
    :math:`+` (not a general monoid), so pad positions -- whose ``values``
    carry REAL's additive identity ``0`` -- contribute nothing, exactly as
    in the gather direction.  It is the single source of truth for both
    the ``g_B`` term of :func:`real_ell_matmul_vjp` and the symmetric
    matvec :math:`\\tfrac{1}{2}(A X + A^{\\top} X)` used by the spectral
    solvers on adjacencies whose top-``k`` construction did not preserve
    symmetry.

    ``semiring`` mirrors :func:`reference_semiring_ell_matmul`'s
    signature, but only **REAL** is implemented: the additive scatter has
    no meaning for a general monoid (the :math:`\\tfrac{1}{2}(A +
    A^{\\top})` averaging needs the linear structure, and a non-zero pad
    identity would inject spurious mass).  Any other algebra raises
    ``NotImplementedError`` rather than silently mis-reducing.

    Parameters
    ----------
    values : Num[Array, 'm kmax']
        ELL values, shape ``(m, k_max)``.  Pad positions must hold ``0``.
    indices : Num[Array, 'm kmax']
        ELL column indices, shape ``(m, k_max)``; the scatter targets.
    X : Num[Array, 'm ncol']
        Dense operand indexed by the ELL *row*, shape ``(m, ncol)``.
    semiring : Semiring
        The algebra to reduce under.  REAL only (see above).
    n_cols : int
        Outer dimension of the implicit operand :math:`A` -- the length of
        ``Y``'s leading axis (the scatter range).

    Returns
    -------
    Num[Array, 'n_cols ncol']
        The adjoint product ``Y``, shape ``(n_cols, ncol)``, with dtype
        given by ``jnp.result_type(values.dtype, X.dtype)``.
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

    def body(
        p: int, acc: Num[Array, 'n_cols ncol']
    ) -> Num[Array, 'n_cols ncol']:
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
