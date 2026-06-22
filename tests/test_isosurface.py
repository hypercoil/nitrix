# -*- coding: utf-8 -*-
"""Marching cubes / isosurface extraction (geometry-suite P2.2 / GS-3).

Marching-tetrahedra engine -> watertight, manifold, correctly-oriented meshes.
Anchored on the sphere SDF (area = 4 pi R^2, genus 0, every edge in exactly two
faces, outward normals) across grid-aligned and non-aligned radii and
anisotropic spacing, with an end-to-end MC -> curvature check.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from nitrix.geometry import (
    euler_characteristic,
    genus,
    marching_cubes,
    mean_curvature,
    mesh_to_sdf,
)
from nitrix.sparse import Mesh, compute_vertex_normals, face_areas, icosphere


def _sphere_sdf(n: int, radius: float, centre: float) -> np.ndarray:
    ax = np.arange(n)
    x, y, z = np.meshgrid(ax, ax, ax, indexing='ij')
    return (
        np.sqrt((x - centre) ** 2 + (y - centre) ** 2 + (z - centre) ** 2)
        - radius
    )


def _all_edges_shared_by_two(faces: np.ndarray) -> bool:
    e = np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0
    )
    _, counts = np.unique(np.sort(e, axis=1), axis=0, return_counts=True)
    return bool((counts == 2).all())


@pytest.mark.parametrize(
    'n,radius,centre',
    [(48, 15.0, 24.0), (40, 12.0, 20.0), (48, 14.7, 23.5), (36, 10.3, 17.5)],
)
def test_sphere_isosurface(n: int, radius: float, centre: float) -> None:
    mesh = marching_cubes(_sphere_sdf(n, radius, centre), level=0.0)
    f = np.asarray(mesh.faces)
    v = np.asarray(mesh.vertices)
    # Watertight, genus-0, manifold (grid-aligned and non-aligned alike).
    assert euler_characteristic(mesh) == 2
    assert genus(mesh) == 0
    assert _all_edges_shared_by_two(f)
    # Area within ~1% of 4 pi R^2.
    area = float(jnp.sum(face_areas(mesh)))
    assert abs(area - 4 * np.pi * radius**2) / (4 * np.pi * radius**2) < 0.01
    # All faces oriented outward (normal . radial > 0).
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    fc = (v[f[:, 0]] + v[f[:, 1]] + v[f[:, 2]]) / 3.0 - centre
    assert np.all((fn * fc).sum(1) > 0)


def test_vertex_normals_point_outward() -> None:
    mesh = marching_cubes(_sphere_sdf(48, 14.7, 23.5), level=0.0)
    normals = np.asarray(compute_vertex_normals(mesh.vertices, mesh.faces))
    radial = np.asarray(mesh.vertices) - 23.5
    assert float((((normals * radial).sum(1)) > 0).mean()) > 0.99


def test_anisotropic_spacing() -> None:
    mesh = marching_cubes(
        _sphere_sdf(40, 12.0, 20.0), level=0.0, spacing=(2.0, 1.0, 1.0)
    )
    assert euler_characteristic(mesh) == 2
    # x stretched by 2 -> bounding box wider in x than y/z.
    v = np.asarray(mesh.vertices)
    ext = v.max(0) - v.min(0)
    assert ext[0] > 1.8 * ext[1]


def test_curvature_of_isosurface_recovers_one_over_r() -> None:
    # End-to-end: marching cubes -> mean curvature ~ 1/R on the extracted
    # sphere.  ~12% median error is expected -- the cotangent curvature is
    # sensitive to MC's irregular triangle quality (slivers), unlike the
    # regular icosphere where it is exact; this is a sanity check, not parity.
    radius = 14.7
    mesh = marching_cubes(_sphere_sdf(56, radius, 27.5), level=0.0)
    h = np.asarray(mean_curvature(mesh))
    assert abs(np.median(h) - 1.0 / radius) / (1.0 / radius) < 0.15


def test_empty_when_level_outside_range() -> None:
    mesh = marching_cubes(_sphere_sdf(20, 5.0, 10.0), level=1000.0)
    assert mesh.n_vertices == 0
    assert mesh.n_faces == 0


def test_input_validation() -> None:
    with pytest.raises(ValueError, match='3-D'):
        marching_cubes(np.zeros((4, 4)))
    with pytest.raises(ValueError, match='length >= 2'):
        marching_cubes(np.zeros((1, 4, 4)))


# --------------------------------------------------------------------------- #
# mesh_to_sdf (the inverse direction)
# --------------------------------------------------------------------------- #


def _sphere_mesh(n_sub: int, radius: float, centre: float) -> Mesh:
    m = icosphere(n_sub)
    return Mesh(m.vertices * radius + centre, m.faces)


def test_mesh_to_sdf_matches_analytic_sphere() -> None:
    radius, centre, n = 10.0, 16.0, 32
    sphere = _sphere_mesh(3, radius, centre)
    sdf = np.asarray(mesh_to_sdf(sphere, (n, n, n)))
    ax = np.arange(n)
    x, y, z = np.meshgrid(ax, ax, ax, indexing='ij')
    analytic = (
        np.sqrt((x - centre) ** 2 + (y - centre) ** 2 + (z - centre) ** 2)
        - radius
    )
    # The mesh approximates the sphere -> small deviation from the exact SDF.
    assert np.abs(sdf - analytic).max() < 0.5
    # Sign agrees away from the surface.
    assert float((np.sign(sdf) == np.sign(analytic)).mean()) > 0.99


def test_mesh_to_sdf_sign() -> None:
    sphere = _sphere_mesh(3, 10.0, 16.0)
    sdf = np.asarray(mesh_to_sdf(sphere, (32, 32, 32)))
    assert sdf[16, 16, 16] < 0.0  # centre is inside
    assert sdf[0, 0, 0] > 0.0  # corner is outside


def test_sdf_marching_cubes_roundtrip_preserves_area() -> None:
    sphere = _sphere_mesh(3, 10.0, 16.0)
    sdf = np.asarray(mesh_to_sdf(sphere, (32, 32, 32)))
    recovered = marching_cubes(sdf, level=0.0)
    assert euler_characteristic(recovered) == 2
    a0 = float(jnp.sum(face_areas(sphere)))
    a1 = float(jnp.sum(face_areas(recovered)))
    assert abs(a1 - a0) / a0 < 0.03


def test_mesh_to_sdf_validation() -> None:
    sphere = _sphere_mesh(2, 5.0, 8.0)
    with pytest.raises(ValueError, match='3-D'):
        mesh_to_sdf(sphere, (16, 16))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match='no faces'):
        mesh_to_sdf(Mesh(sphere.vertices, sphere.faces[:0]), (16, 16, 16))
