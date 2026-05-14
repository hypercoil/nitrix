# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
# isort: skip_file
"""
nitrix.sparse -- geometry-aware sparse formats.

ELL is the primary format. The submodule ``nitrix.sparse.ell`` exports
the construction / gather / scatter / pad primitives; the section
implementation lives in ``nitrix.sparse.ell_sectioned`` (Phase 2.A.8).

No ``jax.experimental.sparse`` dependency anywhere.
"""
from .ell import (
    ELL,
    ell_from_dense,
    ell_to_dense,
    ell_pad,
)
from .ell_sectioned import (
    SectionedELL,
    sectioned_ell_from_ragged,
    sectioned_semiring_ell_matmul,
)

__all__ = [
    'ELL',
    'ell_from_dense',
    'ell_to_dense',
    'ell_pad',
    'SectionedELL',
    'sectioned_ell_from_ragged',
    'sectioned_semiring_ell_matmul',
]
