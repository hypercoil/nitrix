# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
# isort: skip_file
"""
Graph-theoretic primitives.

Differentiable operators over graphs and meshes, spanning graph Laplacians,
modularity and community structure, spectral embeddings, and functional
parcellation. Operators are multi-format (dense, ELL, and SectionedELL sparse
storage) wherever the underlying mathematics admits it.

The subsystem is organised into four groups of operators.

Laplacian
    :func:`laplacian`, :func:`laplacian_matvec`, and the degree vectors
    :func:`degree_vector`, :func:`in_degree_vector`, and
    :func:`symmetric_degree_vector`.

Community structure
    Modularity-related operators: :func:`girvan_newman_null`,
    :func:`modularity_matrix` (dense), :func:`modularity_matrix_matvec`
    (sparse-friendly), :func:`coaffiliation`, and :func:`relaxed_modularity`
    (dense plus a sparse-aware factored path).

Connectopy
    Spectral embeddings: :func:`laplacian_eigenmap` and
    :func:`diffusion_embedding`, each with both a dense (``eigh``) and a
    sparse (``lobpcg``) solver path.

Parcellation
    Functional-parcellation steps: :func:`surface_boundary_map`
    (connectivity-profile boundary detection) and :func:`eta_squared`
    similarity, together with :func:`mesh_watershed` (host-side
    priority-flood basins on a vertex field).

Spectral graph wavelets
    The Chebyshev-approximated spectral graph wavelet transform
    :func:`graph_wavelet_transform` (matvec-only, no eigensolve) and its default
    band-pass :func:`mexican_hat_kernel`.
"""

from .laplacian import (
    degree_vector,
    in_degree_vector,
    laplacian,
    laplacian_matvec,
    symmetric_degree_vector,
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
    heat_kernel,
    laplacian_eigenmap,
    moran_surrogates,
    normalized_cut,
)
from .parcellation import (
    eta_squared,
    mesh_watershed,
    surface_boundary_map,
)
from .wavelet import graph_wavelet_transform, mexican_hat_kernel

__all__ = [
    # laplacian
    'degree_vector',
    'in_degree_vector',
    'symmetric_degree_vector',
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
    'heat_kernel',
    'laplacian_eigenmap',
    'moran_surrogates',
    'normalized_cut',
    # parcellation (functional boundary map -> watershed)
    'eta_squared',
    'surface_boundary_map',
    'mesh_watershed',
    # spectral graph wavelets (SGWT)
    'graph_wavelet_transform',
    'mexican_hat_kernel',
]
