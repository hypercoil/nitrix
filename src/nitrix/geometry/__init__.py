# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
# isort: skip_file
"""
nitrix.geometry -- geometric primitives for neuroimaging.

Submodules:

- ``grid``   -- regular-grid deformable-registration primitives:
  ``identity_grid``, ``spatial_transform``, ``integrate_velocity_field``,
  ``resample``, ``center_of_mass_grid``.  ``resample`` /
  ``spatial_transform`` dispatch over an ``Interpolator`` kernel
  (``Linear`` default, ``NearestNeighbour``, ``Lanczos``, ``CubicBSpline``,
  ``MultiLabel``) -- see ``geometry._interpolate``.
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
- ``transform`` -- the Lie-group chart (``rigid_exp`` / ``affine_exp``)
  + ``apply_affine`` / ``affine_grid``.
- ``affine``  -- the geometric-parameter affine algebra (Euler-angle /
  scale / shear ``T @ R @ S @ E`` compose / decompose) and the
  closed-form least-squares ``fit_affine`` between point sets;
  complementary to the ``transform`` exponential chart.

See SPEC §4.4, SPEC §6.1, and IMPLEMENTATION_PLAN §6.
"""

from ._interpolate import (
    CubicBSpline,
    CubicBSplineBoundaryWarning,
    Interpolator,
    Lanczos,
    Linear,
    MultiLabel,
    NearestNeighbour,
)
from .grid import (
    center_of_mass_grid,
    identity_grid,
    integrate_velocity_field,
    jacobian_det_displacement,
    jacobian_displacement,
    resample,
    sample_at_points,
    spatial_transform,
    spatial_transform_batched,
    # legacy aliases (removed at v0.1)
    cmass_regular_grid,
    rescale,
    vec_int,
)
from .deformation import (
    compose_displacement,
    compose_velocity,
    field_log,
    invert_displacement,
)
from .algebra import (
    fuse_transforms,
    transform_geodesic,
    transform_mean,
    velocity_mean,
)
from .differential import spatial_gradient
from .pyramid import downsample, gaussian_pyramid, upsample
from .transform import (
    affine_exp,
    affine_grid,
    apply_affine,
    rigid_exp,
    rigid_log,
)
from .affine import (
    affine_matrix_to_params,
    angles_to_rotation_matrix,
    compose_affine,
    fit_affine,
    invert_affine,
    make_square_affine,
    params_to_affine_matrix,
    rotation_matrix_to_angles,
)
from .sphere import (
    cartesian_to_latlong,
    is_bijective_sphere_map,
    latlong_to_cartesian,
    signed_spherical_areas,
    spectral_sphere_embedding,
    spherical_conv,
    spherical_geodesic_distance,
    spherical_parameterize,
    surface_resample,
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
from .surface import (
    areal_distortion,
    cortical_thickness,
    deform_to_sdf,
    gaussian_curvature,
    inflate_surface,
    mean_curvature,
    principal_curvatures,
    ribbon_map,
    strain_distortion,
    surface_smooth,
)
from .intersection import (
    find_self_intersections,
    remove_self_intersections,
)
from .isosurface import marching_cubes, mesh_to_sdf
from .topology import (
    euler_characteristic,
    genus,
)

__all__ = [
    # grid
    'identity_grid',
    'spatial_transform',
    'spatial_transform_batched',
    'sample_at_points',
    'integrate_velocity_field',
    'jacobian_displacement',
    'jacobian_det_displacement',
    'resample',
    'center_of_mass_grid',
    # differential / multi-resolution
    'spatial_gradient',
    'downsample',
    'upsample',
    'gaussian_pyramid',
    # deformation / velocity-field algebra
    'compose_displacement',
    'compose_velocity',
    'invert_displacement',
    'field_log',
    # transform parametrisation (rigid / affine Lie chart)
    'rigid_exp',
    'rigid_log',
    'affine_exp',
    'transform_mean',
    'fuse_transforms',
    'transform_geodesic',
    'velocity_mean',
    'apply_affine',
    'affine_grid',
    # affine algebra (geometric params: T @ R @ S @ E; point-set fit)
    'make_square_affine',
    'invert_affine',
    'compose_affine',
    'fit_affine',
    'angles_to_rotation_matrix',
    'rotation_matrix_to_angles',
    'params_to_affine_matrix',
    'affine_matrix_to_params',
    # interpolation-method ADT
    'Interpolator',
    'Linear',
    'NearestNeighbour',
    'Lanczos',
    'CubicBSpline',
    'CubicBSplineBoundaryWarning',
    'MultiLabel',
    # sphere (mesh)
    'cartesian_to_latlong',
    'latlong_to_cartesian',
    'spherical_conv',
    'spherical_geodesic_distance',
    'signed_spherical_areas',
    'is_bijective_sphere_map',
    'spectral_sphere_embedding',
    'spherical_parameterize',
    'surface_resample',
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
    # surface algorithms (differential geometry on meshes)
    'mean_curvature',
    'gaussian_curvature',
    'principal_curvatures',
    'areal_distortion',
    'strain_distortion',
    'surface_smooth',
    'deform_to_sdf',
    'cortical_thickness',
    'inflate_surface',
    'ribbon_map',
    # topology (the genus-0 defect gate)
    'euler_characteristic',
    'genus',
    # volume <-> surface conversion
    'marching_cubes',
    'mesh_to_sdf',
    # self-intersection (host-side QA / cleanup)
    'find_self_intersections',
    'remove_self_intersections',
]
