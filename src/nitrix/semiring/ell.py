# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Semiring-generalised ELL-sparse matrix multiplication.

This module provides :func:`semiring_ell_matmul`, which contracts a sparse
left operand held in ELL (ELLPACK) layout against a dense right operand
under an arbitrary semiring, together with its transpose companion
:func:`semiring_ell_rmatvec`.  The forward :func:`semiring_ell_matmul`
dispatches across a three-level backend selection (auto, Pallas-CUDA, or
pure JAX); :func:`semiring_ell_rmatvec` is pure JAX.

The pad positions of each ELL row are expected to be filled with the
algebra's identity (for example ``0`` for the real semiring, or
:math:`-\\infty` for the log semiring), so that padded entries contribute
nothing to the reduction.  See :func:`nitrix.sparse.ell.ell_pad` for the
helper that produces such padding.

This is the central operator for brain-geometry workloads and the
load-bearing sparse kernel of the library.  The pure-JAX path is always
available; the Pallas-CUDA path is opt-in on supported hardware.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Int, Num

from .._internal.backend import (
    Backend,
    ResolvedBackend,
    fallback,
    resolve_backend,
)
from ._backward import real_ell_rmatvec_vjp
from ._reference import (
    reference_semiring_ell_matmul,
    reference_semiring_ell_rmatvec,
)
from ._types import Semiring
from .algebras import REAL

# ---------------------------------------------------------------------------
# Differentiable 2-D core via jax.custom_vjp.  The wrapper mirrors the
# matmul pattern: `indices`, `semiring`, `n_cols`, `backend` are non-diff
# (we never want a gradient w.r.t. integer indices or static config).
# Batching composes via ``jax.vmap`` upstream.
# ---------------------------------------------------------------------------


def _forward_only_ell_2d(
    values: Num[Array, 'm kmax'],
    indices: Int[Array, 'm kmax'],
    B: Num[Array, 'n_cols ncol'],
    *,
    semiring: Semiring[Any],
    n_cols: Optional[int],
    backend: Backend,
) -> Num[Array, 'm ncol']:
    resolved: ResolvedBackend = resolve_backend(backend)
    if resolved == 'pallas-cuda':
        out = _semiring_ell_matmul_pallas(
            values,
            indices,
            B,
            semiring=semiring,
        )
        if out is None:
            resolved = fallback(
                function='semiring_ell_matmul',
                requested='pallas-cuda',
                resolved='jax',
                reason=(
                    f'algebra={semiring.name!r}: Pallas Triton kernel '
                    'unavailable or cannot tile the requested shape'
                ),
                shapes=(tuple(values.shape), tuple(B.shape)),
                dtype=B.dtype,
            )
        else:
            return out
    return reference_semiring_ell_matmul(
        values,
        indices,
        B,
        semiring=semiring,
        n_cols=n_cols,
    )


@partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5))
def _diff_ell_matmul_2d(
    values: Array,
    indices: Array,
    B: Array,
    semiring: Semiring[Any],
    n_cols: Optional[int],
    backend: Backend,
) -> Array:
    # ``indices`` cannot live in ``nondiff_argnums`` -- JAX (>= 0.10)
    # refuses to accept Tracers there.  It is carried as a regular
    # array argument; the backward returns a zero gradient for it,
    # which is consistent with the "indices are integer-valued and
    # therefore non-differentiable" semantics.
    return _forward_only_ell_2d(
        values,
        indices,
        B,
        semiring=semiring,
        n_cols=n_cols,
        backend=backend,
    )


def _diff_ell_matmul_2d_fwd(
    values: Array,
    indices: Array,
    B: Array,
    semiring: Semiring[Any],
    n_cols: Optional[int],
    backend: Backend,
) -> Tuple[Array, Tuple[Array, Array, Array, Array]]:
    out = _forward_only_ell_2d(
        values,
        indices,
        B,
        semiring=semiring,
        n_cols=n_cols,
        backend=backend,
    )
    # Stash (values, indices, B, out) -- backward rules pick what they need.
    return out, (values, indices, B, out)


def _diff_ell_matmul_2d_bwd(
    semiring: Semiring[Any],
    n_cols: Optional[int],
    backend: Backend,
    residuals: Tuple[Array, Array, Array, Array],
    g_out: Array,
) -> Tuple[Array, Array, Array]:
    rule = semiring.ell_matmul_vjp
    if rule is None:
        raise TypeError(
            f'semiring={semiring.name!r} is forward-only for ELL; '
            'supply ``ell_matmul_vjp`` to enable gradients (or wrap the '
            'call in your own ``jax.custom_vjp``).'
        )
    _values, indices, _B, _out = residuals
    g_values, g_B = rule(residuals, g_out)
    # Diff-arg order: (values, indices, B).  Indices is integer-valued;
    # by convention we return a zero gradient of matching shape/dtype.
    return g_values, jnp.zeros_like(indices), g_B


__all__ = [
    'semiring_ell_matmul',
    'reference_semiring_ell_matmul',
    'semiring_ell_rmatvec',
    'reference_semiring_ell_rmatvec',
]


def _check_ell_shapes(
    values_shape: tuple[int, ...],
    indices_shape: tuple[int, ...],
    B_shape: tuple[int, ...],
    name: str,
) -> None:
    if len(values_shape) < 2:
        raise ValueError(
            f'{name}: values must be at least 2-D '
            f'(got values.shape={values_shape}).'
        )
    if values_shape != indices_shape:
        raise ValueError(
            f'{name}: values.shape={values_shape} must equal '
            f'indices.shape={indices_shape}.'
        )
    if len(B_shape) < 2:
        raise ValueError(
            f'{name}: B must be at least 2-D (got B.shape={B_shape}).'
        )


def _semiring_ell_matmul_pallas(
    values: Array,
    indices: Array,
    B: Array,
    *,
    semiring: Semiring[Any],
) -> Optional[Array]:
    """Attempt the Pallas-CUDA ELL matmul, returning ``None`` on rejection.

    Imports the CUDA Pallas kernel lazily and invokes it.  If the kernel
    module cannot be imported, or the kernel signals that it cannot tile
    the requested shape, ``None`` is returned so the caller can fall back
    to the pure-JAX reference path.

    Parameters
    ----------
    values
        ELL values, shape ``(m, k_max)``.
    indices
        ELL column indices into ``B``'s outer dimension, shape
        ``(m, k_max)``.
    B
        Dense right operand, shape ``(n_cols, ncol)``.
    semiring
        Algebra under which the ELL contraction is reduced.

    Returns
    -------
    Array of shape ``(m, ncol)`` holding the semiring contraction, or
    ``None`` if the Pallas kernel is unavailable or cannot tile the
    requested shape.
    """
    try:
        from .._kernels.cuda.semiring_ell_matmul import (
            PallasELLNotTileable,
            semiring_ell_matmul_pallas,
        )
    except Exception:
        return None
    try:
        return semiring_ell_matmul_pallas(
            values,
            indices,
            B,
            semiring=semiring,
        )
    except PallasELLNotTileable:
        return None


def semiring_ell_matmul(
    values: Num[Array, '... m kmax'],
    indices: Int[Array, '... m kmax'],
    B: Num[Array, '... n_cols ncol'],
    *,
    semiring: Semiring[Any] = REAL,
    n_cols: Optional[int] = None,
    backend: Backend = 'auto',
) -> Num[Array, '... m ncol']:
    """Semiring-generalised ELL-sparse matrix multiplication.

    Computes::

        C[..., i, j] = (+)_p ( values[..., i, p] (*) B[..., indices[..., i, p], j] )

    where the implicit :math:`M \\times N` sparse left operand has the per-row
    neighbour list ``indices[..., i, :]`` with values
    ``values[..., i, :]``.  Padding positions in ``indices`` must point
    to a valid row of ``B``, and the corresponding ``values`` entries
    must be the semiring identity so the contribution is a no-op.

    Parameters
    ----------
    values
        ELL values, shape ``(..., m, k_max)``.
    indices
        ELL column indices into ``B``'s outer dim, shape ``(..., m, k_max)``.
    B
        Dense right operand, shape ``(..., n_cols, ncol)``.  ``n_cols``
        is the outer dim of the implicit sparse matrix.
    semiring
        Algebra to reduce under.
    n_cols
        The implicit sparse-matrix outer dim.  Required for the public
        contract but defaults to ``B.shape[-2]``.
    backend
        ``"auto"``, ``"pallas-cuda"``, or ``"jax"``.

    Returns
    -------
    Array of shape ``(*broadcast_batch, m, ncol)`` holding the semiring
    contraction of the sparse left operand against ``B``.
    """
    _check_ell_shapes(
        values.shape,
        indices.shape,
        B.shape,
        'semiring_ell_matmul',
    )
    if n_cols is None:
        n_cols = int(B.shape[-2])

    batch_dims = len(values.shape) - 2
    core: Callable[[Array, Array, Array], Array]
    if semiring.ell_matmul_vjp is None:

        def core(v_: Array, i_: Array, B_: Array) -> Array:
            return _forward_only_ell_2d(
                v_,
                i_,
                B_,
                semiring=semiring,
                n_cols=n_cols,
                backend=backend,
            )
    else:

        def core(v_: Array, i_: Array, B_: Array) -> Array:
            return _diff_ell_matmul_2d(
                v_,
                i_,
                B_,
                semiring,
                n_cols,
                backend,
            )

    fn: Callable[..., Array] = core
    for _ in range(batch_dims):
        fn = jax.vmap(fn, in_axes=(0, 0, 0))
    return fn(values, indices, B)


# Bind the VJP definitions now that the forward helper is in scope.
_diff_ell_matmul_2d.defvjp(_diff_ell_matmul_2d_fwd, _diff_ell_matmul_2d_bwd)


# ---------------------------------------------------------------------------
# Adjoint (transpose) ELL matvec -- the REAL-only scatter direction.
#
# ``semiring_ell_matmul`` gathers (``A @ X``); this scatters (``Aᵀ @ X``).
# Together they give the symmetric-part matvec ``½(A X + Aᵀ X)`` that the
# spectral solvers need when an adjacency's top-k construction did not
# preserve symmetry (see ``linalg._eigsolve`` / ``graph.connectopy``).  The
# scatter reduction is additive, so only REAL is defined; the ``semiring``
# argument mirrors ``semiring_ell_matmul`` for surface consistency and
# raises ``NotImplementedError`` for any other algebra.
# ---------------------------------------------------------------------------


@partial(jax.custom_vjp, nondiff_argnums=(3, 4))
def _diff_ell_rmatvec_2d(
    values: Array,
    indices: Array,
    X: Array,
    semiring: Semiring[Any],
    n_cols: int,
) -> Array:
    # ``indices`` rides as a regular array arg (JAX refuses Tracers in
    # ``nondiff_argnums``); the backward returns a zero gradient for it.
    return reference_semiring_ell_rmatvec(
        values, indices, X, semiring=semiring, n_cols=n_cols
    )


def _diff_ell_rmatvec_2d_fwd(
    values: Array,
    indices: Array,
    X: Array,
    semiring: Semiring[Any],
    n_cols: int,
) -> Tuple[Array, Tuple[Array, Array, Array]]:
    out = reference_semiring_ell_rmatvec(
        values, indices, X, semiring=semiring, n_cols=n_cols
    )
    return out, (values, indices, X)


def _diff_ell_rmatvec_2d_bwd(
    semiring: Semiring[Any],
    n_cols: int,
    residuals: Tuple[Array, Array, Array],
    g_out: Array,
) -> Tuple[Array, Array, Array]:
    # REAL-only (enforced at the forward); the backward is the fixed REAL
    # adjoint rule.  Diff-arg order mirrors the primal: (values, indices, X).
    values, indices, X = residuals
    g_values, g_X = real_ell_rmatvec_vjp((values, indices, X), g_out)
    return g_values, jnp.zeros_like(indices), g_X


_diff_ell_rmatvec_2d.defvjp(_diff_ell_rmatvec_2d_fwd, _diff_ell_rmatvec_2d_bwd)


def semiring_ell_rmatvec(
    values: Num[Array, '... m kmax'],
    indices: Int[Array, '... m kmax'],
    X: Num[Array, '... m ncol'],
    *,
    semiring: Semiring[Any] = REAL,
    n_cols: int,
) -> Num[Array, '... n_cols ncol']:
    """Adjoint (transpose) ELL matvec: :math:`Y = A^{\\top} X` for the operand.

    The transpose companion of :func:`semiring_ell_matmul`.  Where the
    matmul *gathers* --
    :math:`(A X)_i = \\sum_p \\mathtt{values}_{i, p} \\cdot X_{\\mathtt{indices}_{i, p}}`
    -- this *scatters*:

    .. math::

        Y_{c, j} = \\sum_{(i, p)\\,:\\,\\mathtt{indices}_{i, p} = c}
            \\mathtt{values}_{i, p} \\cdot X_{i, j}

    for the same implicit :math:`m \\times n_{\\mathrm{cols}}` operand
    :math:`A`.  Composing the two gives the symmetric-part matvec
    :math:`\\tfrac{1}{2}(A X + A^{\\top} X) = \\operatorname{sym}(A)\\,X`,
    which is what the extremal eigensolvers require when an adjacency was
    built by a construction (for example top-k-per-row affinity
    sparsification, :func:`nitrix.sparse.ell_from_dense`) that does **not**
    preserve symmetry.

    Parameters
    ----------
    values
        ELL values, shape ``(..., m, k_max)``.  Pad positions must hold the
        real additive identity ``0``.
    indices
        ELL column indices, shape ``(..., m, k_max)``; the scatter targets.
    X
        Dense operand indexed by the ELL *row*, shape ``(..., m, ncol)``.
    semiring
        The algebra to reduce under.  **Real semiring only** -- the additive
        scatter is meaningless for a general monoid (the symmetric part
        :math:`\\tfrac{1}{2}(A + A^{\\top})` needs linear structure, and a
        non-zero pad identity would scatter spurious mass).  Any other
        algebra raises :class:`NotImplementedError`.  The argument is
        retained for surface parity with :func:`semiring_ell_matmul`.
    n_cols
        Outer dimension of the implicit operand :math:`A` and the leading
        axis of the output.  Required: unlike the gather direction it is
        **not** recoverable from ``X`` (which is indexed by the row axis
        ``m``).

    Returns
    -------
    Array of shape ``(*broadcast_batch, n_cols, ncol)`` holding the
    scattered adjoint matvec.

    Notes
    -----
    JAX-only: there is no Pallas scatter kernel (the matmul backward's
    gradient-with-respect-to-``B`` scatter is likewise JAX-only), so no
    ``backend`` argument is exposed.  The result is differentiable with
    respect to ``values`` and ``X`` via the registered real-semiring adjoint
    VJP (``indices`` is integer-valued and therefore non-differentiable).
    """
    if semiring is not REAL:
        raise NotImplementedError(
            f'semiring_ell_rmatvec: the additive ELL adjoint is implemented '
            f'for the REAL semiring only; got {semiring.name!r}.'
        )
    _check_ell_shapes(
        values.shape, indices.shape, X.shape, 'semiring_ell_rmatvec'
    )

    batch_dims = len(values.shape) - 2

    def core(v_: Array, i_: Array, X_: Array) -> Array:
        return _diff_ell_rmatvec_2d(v_, i_, X_, semiring, n_cols)

    fn: Callable[..., Array] = core
    for _ in range(batch_dims):
        fn = jax.vmap(fn, in_axes=(0, 0, 0))
    return fn(values, indices, X)
