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
    ell_add_self_loops,
    ell_from_dense,
    ell_mask,
    ell_to_dense,
    ell_pad,
)
from .ell_sectioned import (
    SectionedELL,
    sectioned_ell_from_ragged,
    sectioned_semiring_ell_matmul,
)
from .grid import (
    grid_identity,
    grid_laplacian,
    regular_grid_stencil,
)
from .mesh import (
    IcosphereHierarchy,
    Mesh,
    icosphere,
    icosphere_bary_upsampler,
    icosphere_cross_level_adjacency,
    icosphere_hierarchy,
    icosphere_hierarchy_from_levels,
    mesh_bary_upsample,
    mesh_coarsen_meanpool,
    mesh_cotangent_laplacian,
    mesh_k_ring_adjacency,
    mesh_pool_max,
    mesh_unpool_max,
)

__all__ = [
    # ell
    'ELL',
    'ell_add_self_loops',
    'ell_from_dense',
    'ell_mask',
    'ell_to_dense',
    'ell_pad',
    # sectioned ell
    'SectionedELL',
    'sectioned_ell_from_ragged',
    'sectioned_semiring_ell_matmul',
    # grid stencils
    'grid_identity',
    'grid_laplacian',
    'regular_grid_stencil',
    # mesh
    'IcosphereHierarchy',
    'Mesh',
    'icosphere',
    'icosphere_bary_upsampler',
    'icosphere_cross_level_adjacency',
    'icosphere_hierarchy',
    'icosphere_hierarchy_from_levels',
    'mesh_bary_upsample',
    'mesh_coarsen_meanpool',
    'mesh_cotangent_laplacian',
    'mesh_k_ring_adjacency',
    'mesh_pool_max',
    'mesh_unpool_max',
]
