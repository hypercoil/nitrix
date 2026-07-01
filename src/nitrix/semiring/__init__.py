# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
# isort: skip_file
"""
KeOps-style streaming reductions over arbitrary algebras.

This subpackage provides matrix and graph reductions that run over a
user-selected algebraic structure rather than the fixed
(add, multiply) ring of ordinary linear algebra. The same streaming
kernels back the max-plus, min-plus, log-sum-exp, boolean and Euclidean
reductions used throughout the wider library, so that morphology,
smoothing and mesh aggregation can all be expressed as reductions over a
chosen semiring.

The public surface is organised as follows.

- :class:`Semiring` (relaxed) and :class:`StrictSemiring`
  (which asserts associativity) describe the algebraic structure supplied
  to a reduction, built on the :class:`Semigroup` and :class:`Monoid`
  protocols.
- The built-in algebras :data:`REAL`, :data:`LOG`,
  :data:`TROPICAL_MAX_PLUS`, :data:`TROPICAL_MIN_PLUS`, :data:`BOOLEAN`
  and :data:`EUCLIDEAN` cover the common cases.
- :func:`semiring_matmul` and :func:`semiring_ell_matmul` perform dense
  and ELL-sparse matrix products over a chosen semiring.
- :func:`semiring_ell_edge_aggregate` performs edge-functional
  aggregation over an ELL adjacency structure, with optional per-edge
  ``edge_attr``, and :func:`ell_row_softmax` provides the row-normalising
  attention pre-pass used by graph-attention aggregation.
"""

from ._types import (
    Monoid,
    Semigroup,
    Semiring,
    StrictSemiring,
)
from .algebras import (
    REAL,
    LOG,
    TROPICAL_MAX_PLUS,
    TROPICAL_MIN_PLUS,
    BOOLEAN,
    EUCLIDEAN,
    LogSumExpAcc,
)
from .matmul import (
    semiring_matmul,
    reference_semiring_matmul,
)
from ._reference import (
    reference_semiring_ell_matmul,
    reference_semiring_ell_rmatvec,
)
from .ell import semiring_ell_matmul, semiring_ell_rmatvec
from .ell_edge import ell_row_softmax, semiring_ell_edge_aggregate
from .conv import (
    semiring_conv,
    reference_semiring_conv,
)

__all__ = [
    'Monoid',
    'Semigroup',
    'Semiring',
    'StrictSemiring',
    'REAL',
    'LOG',
    'TROPICAL_MAX_PLUS',
    'TROPICAL_MIN_PLUS',
    'BOOLEAN',
    'EUCLIDEAN',
    'LogSumExpAcc',
    'semiring_matmul',
    'reference_semiring_matmul',
    'semiring_ell_matmul',
    'reference_semiring_ell_matmul',
    'semiring_ell_rmatvec',
    'reference_semiring_ell_rmatvec',
    'semiring_ell_edge_aggregate',
    'ell_row_softmax',
    'semiring_conv',
    'reference_semiring_conv',
]
