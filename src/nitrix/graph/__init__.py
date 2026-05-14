# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
# isort: skip_file
"""
nitrix.graph -- graph-theoretic primitives.

Submodules:

- ``laplacian``  -- ``laplacian``, ``laplacian_matvec``,
  ``degree_vector``.  Multi-format (dense, ELL, SectionedELL) where
  the math admits.
- ``community``  -- modularity-related operators: ``girvan_newman_null``,
  ``modularity_matrix`` (dense), ``modularity_matrix_matvec``
  (sparse-friendly), ``coaffiliation``, ``relaxed_modularity``
  (dense + sparse-aware factored path).
- ``connectopy`` -- ``laplacian_eigenmap`` and ``diffusion_embedding``
  with both dense (``eigh``) and sparse (``lobpcg``) solver paths.

See SPEC §4.5, SPEC §6.1, and IMPLEMENTATION_PLAN §6.1 tasks 3.5-3.7.
"""
from .laplacian import (
    degree_vector,
    laplacian,
    laplacian_matvec,
)
from .community import (
    coaffiliation,
    girvan_newman_null,
    modularity_matrix,
    modularity_matrix_matvec,
    relaxed_modularity,
)
from .connectopy import (
    diffusion_embedding,
    laplacian_eigenmap,
)

__all__ = [
    # laplacian
    'degree_vector',
    'laplacian',
    'laplacian_matvec',
    # community
    'coaffiliation',
    'girvan_newman_null',
    'modularity_matrix',
    'modularity_matrix_matvec',
    'relaxed_modularity',
    # connectopy
    'diffusion_embedding',
    'laplacian_eigenmap',
]
