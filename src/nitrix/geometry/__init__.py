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
- ``sphere`` -- 2-sphere primitives: coordinate conversions,
  geodesic distance, and ``spherical_conv`` re-backed on
  ``semiring_ell_matmul`` for ``O(n · k)`` instead of ``O(n²)``.
- ``coords`` -- coordinate utilities: ``center_of_mass_points``,
  ``displacement_from_reference_*``, ``compactness_penalty``.

See SPEC §4.4, SPEC §6.1, and IMPLEMENTATION_PLAN §6.
"""
from .grid import (
    center_of_mass_grid,
    identity_grid,
    integrate_velocity_field,
    resample,
    spatial_transform,
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
    'integrate_velocity_field',
    'resample',
    'center_of_mass_grid',
    # sphere
    'cartesian_to_latlong',
    'latlong_to_cartesian',
    'spherical_conv',
    'spherical_geodesic_distance',
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
