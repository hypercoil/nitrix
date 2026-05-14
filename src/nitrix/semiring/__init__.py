# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
# isort: skip_file
"""
nitrix.semiring -- KeOps-style streaming reductions over arbitrary algebras.

Public surface:

- ``Semiring`` (relaxed) / ``StrictSemiring`` (asserts associativity)
- ``Semigroup``, ``Monoid`` Protocols
- Built-in algebras: ``REAL``, ``LOG``, ``TROPICAL_MAX_PLUS``,
  ``TROPICAL_MIN_PLUS``, ``BOOLEAN``, ``EUCLIDEAN``
- ``semiring_matmul``, ``semiring_ell_matmul``

See ``nitrix/SPEC.md`` §3.1 and ``SPEC_UPDATE.md`` §3.1.
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
from ._reference import reference_semiring_ell_matmul
from .ell import semiring_ell_matmul
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
    'semiring_conv',
    'reference_semiring_conv',
]
