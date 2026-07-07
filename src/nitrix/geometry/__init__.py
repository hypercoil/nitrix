# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
# isort: skip_file
"""
Geometric primitives for neuroimaging.

This subpackage collects the differentiable geometry primitives that
underpin spatial registration, resampling, and surface/mesh analysis. It
spans both regular-grid (voxel/volume) and mesh/graph representations of
spatial data.

Submodules
----------
- ``grid`` -- regular-grid deformable-registration primitives:
  :func:`identity_grid`, :func:`spatial_transform`,
  :func:`integrate_velocity_field`, :func:`resample`, and
  :func:`center_of_mass_grid`. :func:`resample` and
  :func:`spatial_transform` dispatch over an :class:`Interpolator` kernel
  (:class:`Linear` default, :class:`NearestNeighbour`, :class:`Lanczos`,
  :class:`CubicBSpline`, :class:`CatmullRomCubic`, :class:`MultiLabel`).
- ``sphere`` -- 2-sphere **mesh** primitives: coordinate conversions,
  geodesic distance, and :func:`spherical_conv` re-backed on the sparse
  semiring matmul for :math:`O(n \\cdot k)` cost (with :math:`n` vertices
  and :math:`k` neighbours) instead of :math:`O(n^2)`.
- ``sphere_grid`` -- 2-sphere **regular-grid** topology helpers
  (parameterised equirectangular sphere): :func:`sphere_grid_pad_2d` and
  :func:`sphere_grid_unpad_2d` with pole-flip and longitudinal-wrap
  padding. Distinct from ``sphere`` because the parameterised-grid case
  and the mesh case have different storage and adjacency models.
- ``coords`` -- coordinate utilities: :func:`center_of_mass_points`,
  the displacement-from-reference helpers, and
  :func:`compactness_penalty`.
- ``transform`` -- the Lie-group chart (:func:`rigid_exp` /
  :func:`affine_exp`) together with :func:`apply_affine` and
  :func:`affine_grid`.
- ``affine`` -- the geometric-parameter affine algebra (Euler-angle /
  scale / shear :math:`T R S E` compose / decompose) and the closed-form
  least-squares :func:`fit_affine` between point sets; complementary to
  the ``transform`` exponential chart.
"""

from ._interpolate import (
    CatmullRomCubic,
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
    random_rotation,
    signed_spherical_areas,
    spectral_sphere_embedding,
    spherical_conv,
    spherical_geodesic_distance,
    spherical_parameterize,
    spin_surrogates,
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
from ._mesh_distance import (
    point_set_nearest_sq_dist,
    segment_segment_sq_dist,
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
    'CatmullRomCubic',
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
    'random_rotation',
    'spin_surrogates',
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
    # differentiable dense distance kernels (mesh-loss substrate)
    'segment_segment_sq_dist',
    'point_set_nearest_sq_dist',
]
