# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Format-agnostic application of a sparse mesh / graph operator to a field.

A single dispatch over the sparse operator formats so the surface-algorithm
layer (curvature, distortion, geodesic smoothing, DEC) never branches on
whether an operator is stored as a flat ``ELL`` or a bucketed
``SectionedELL``.  Mirrors the ``linalg._eigsolve`` / ``Interpolator``
"factor the independent axes, don't hand-write the cross product" discipline:
the *method* (algebra) and the *task* (which operator) compose, the storage
*format* is dispatched behind one seam.

``ELL`` (near-uniform degree, e.g. the icosphere) and ``SectionedELL``
(irregular valence, e.g. a ``recon-all`` white surface) thus flow through the
same algorithm code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Union, cast

import jax
from jaxtyping import Array, Num

from .._internal.backend import Backend
from .ell import ELL
from .ell_sectioned import SectionedELL, sectioned_semiring_ell_matmul

if TYPE_CHECKING:
    from ..semiring._types import Semiring

__all__ = [
    'MeshOperator',
    'apply_operator',
]

# A sparse operator in either supported storage format.
MeshOperator = Union[ELL, SectionedELL]


def apply_operator(
    op: MeshOperator,
    x: Num[Array, '... n d'],
    *,
    semiring: 'Semiring[Any] | None' = None,
    backend: Backend = 'jax',
) -> Num[Array, '... m d']:
    """Apply a sparse operator to a per-vertex field, format-agnostically.

    Computes the semiring contraction ``op @ x`` whether ``op`` is a flat
    ``ELL`` or a bucketed ``SectionedELL``.  Leading axes of ``x`` are treated
    as batch and vmapped over the 2-D core (the operator is shared across the
    batch -- it is not itself batched), matching the shape contract of the
    ``sparse.mesh`` cross-level wrappers.

    Parameters
    ----------
    op
        The sparse operator, ``ELL`` or ``SectionedELL`` of shape
        ``(m, n)``.
    x
        Per-vertex field, shape ``(..., n, d)``.  For a scalar field use a
        trailing singleton ``(..., n, 1)``.  Must be at least 2-D.
    semiring
        Algebra to reduce under.  Defaults to ``REAL`` (resolved lazily to
        avoid an import cycle).
    backend
        Backend passed through to the underlying kernel (``'jax'`` is the
        only GA path for the ELL gather; see ``ell-on-triton.md``).

    Returns
    -------
    ``(..., m, d)`` -- the contracted field.

    Raises
    ------
    ValueError
        If ``x`` is not at least 2-D (it must carry an ``(n, d)`` core).
    TypeError
        If ``op`` is neither an ``ELL`` nor a ``SectionedELL``.
    """
    if x.ndim < 2:
        raise ValueError(
            f'apply_operator: x must be (..., n, d) with a trailing feature '
            f'axis (use (n, 1) for a scalar field); got shape {x.shape}.'
        )
    if semiring is None:
        from ..semiring import REAL

        semiring = REAL

    if isinstance(op, ELL):
        from ..semiring import semiring_ell_matmul

        def core(b2d: Num[Array, 'n d']) -> Num[Array, 'm d']:
            return semiring_ell_matmul(
                op.values,
                op.indices,
                b2d,
                semiring=semiring,
                n_cols=op.n_cols,
                backend=backend,
            )
    elif isinstance(op, SectionedELL):

        def core(b2d: Num[Array, 'n d']) -> Num[Array, 'm d']:
            return sectioned_semiring_ell_matmul(
                op, b2d, semiring=semiring, backend=backend
            )
    else:
        raise TypeError(
            f'apply_operator: op must be an ELL or SectionedELL; got '
            f'{type(op).__name__}.'
        )

    fn: Callable[..., Any] = core
    for _ in range(x.ndim - 2):
        fn = jax.vmap(fn)
    # ``jax.vmap`` erases the return type to Any; restore it.
    return cast(Num[Array, '... m d'], fn(x))
