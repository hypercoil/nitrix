# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
# isort: skip_file
"""
nitrix.geometry -- geometric primitives for neuroimaging.

Submodules:

- ``grid``   -- regular-grid deformable-registration primitives:
  ``identity_grid``, ``spatial_transform``, ``integrate_velocity_field``,
  ``resample``, ``center_of_mass_grid``.
- ``sphere`` -- 2-sphere **mesh** primitives: coordinate
  conversions, geodesic distance, and ``spherical_conv`` re-backed
  on ``semiring_ell_matmul`` for ``O(n · k)`` instead of ``O(n²)``.
- ``sphere_grid`` -- 2-sphere **regular-grid** topology helpers
  (parameterised equirectangular sphere): ``sphere_grid_pad_2d``
  / ``sphere_grid_unpad_2d`` with pole-flip + longitudinal-wrap
  padding.  Distinct from ``sphere`` because the parameterised-
  grid case and the mesh case have different storage and adjacency
  models.
- ``coords`` -- coordinate utilities: ``center_of_mass_points``,
  ``displacement_from_reference_*``, ``compactness_penalty``.

See SPEC §4.4, SPEC §6.1, and IMPLEMENTATION_PLAN §6.
"""
from .grid import (
    center_of_mass_grid,
    identity_grid,
    integrate_velocity_field,
    jacobian_det_displacement,
    jacobian_displacement,
    resample,
    spatial_transform,
    spatial_transform_batched,
    # legacy aliases (removed at v0.1)
    cmass_regular_grid,
    rescale,
    vec_int,
)
from .sphere import (
    cartesian_to_latlong,
    latlong_to_cartesian,
    spherical_conv,
    spherical_geodesic_distance,
)
from .sphere_grid import (
    sphere_grid_pad_2d,
    sphere_grid_unpad_2d,
)
from .coords import (
    center_of_mass_points,
    compactness_penalty,
    displacement_from_reference_grid,
    displacement_from_reference_points,
    # legacy aliases
    cmass_coor,
    cmass_reference_displacement_coor,
    cmass_reference_displacement_grid,
    diffuse,
)

__all__ = [
    # grid
    'identity_grid',
    'spatial_transform',
    'spatial_transform_batched',
    'integrate_velocity_field',
    'jacobian_displacement',
    'jacobian_det_displacement',
    'resample',
    'center_of_mass_grid',
    # sphere (mesh)
    'cartesian_to_latlong',
    'latlong_to_cartesian',
    'spherical_conv',
    'spherical_geodesic_distance',
    # sphere_grid (parameterised regular grid)
    'sphere_grid_pad_2d',
    'sphere_grid_unpad_2d',
    # coords
    'center_of_mass_points',
    'compactness_penalty',
    'displacement_from_reference_grid',
    'displacement_from_reference_points',
    # legacy aliases (kept for migration; removed at v0.1)
    'cmass_regular_grid',
    'rescale',
    'vec_int',
    'cmass_coor',
    'cmass_reference_displacement_coor',
    'cmass_reference_displacement_grid',
    'diffuse',
]
