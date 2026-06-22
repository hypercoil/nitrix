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
)
from nitrix.sparse import compute_vertex_normals, face_areas


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
