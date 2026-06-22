# -*- coding: utf-8 -*-
"""Signed spherical area + bijectivity check (geometry-suite P3.2 / GS-2a).

``signed_spherical_areas`` is the signed solid angle per triangle (the
unit-sphere spherical-triangle area, radius-independent), and
``is_bijective_sphere_map`` the degree-1-cover + fold-free test the spherical
parameterisation asserts against at every step.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from nitrix.geometry import is_bijective_sphere_map, signed_spherical_areas
from nitrix.sparse import icosphere


def test_sphere_areas_positive_and_sum_to_4pi() -> None:
    m = icosphere(3)
    areas = np.asarray(signed_spherical_areas(m.vertices, m.faces))
    assert np.all(areas > 0)  # outward-oriented, no folds
    assert np.isclose(areas.sum(), 4 * np.pi, rtol=1e-4)
    assert is_bijective_sphere_map(m.vertices, m.faces)


def test_radius_independent() -> None:
    # Solid angle does not depend on sphere radius -> still sums to 4*pi.
    m = icosphere(3)
    for radius in (0.5, 5.0, 100.0):
        areas = np.asarray(
            signed_spherical_areas(m.vertices * radius, m.faces)
        )
        assert np.isclose(areas.sum(), 4 * np.pi, rtol=1e-4)
        assert is_bijective_sphere_map(m.vertices * radius, m.faces)


def test_flipped_triangle_is_detected() -> None:
    m = icosphere(2)
    faces = np.asarray(m.faces).copy()
    faces[0] = faces[0][[0, 2, 1]]  # reverse winding of one triangle -> a fold
    faces = jnp.asarray(faces)
    areas = np.asarray(signed_spherical_areas(m.vertices, faces))
    assert areas[0] < 0  # the flipped triangle is negative
    assert not is_bijective_sphere_map(m.vertices, faces)  # strict: rejected
    # ... but tolerated under a small flipped-area budget (recon-surf style).
    assert is_bijective_sphere_map(m.vertices, faces, flip_area_tol=0.05)


def test_non_cover_rejected() -> None:
    # Collapse all vertices onto a small cap -> not a degree-1 cover.
    m = icosphere(2)
    v = np.asarray(m.vertices).copy()
    v[:, 2] = np.abs(v[:, 2]) + 5.0  # push everything to one side (z >> 0)
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    assert not is_bijective_sphere_map(jnp.asarray(v), m.faces)


def test_signed_areas_differentiable() -> None:
    m = icosphere(2)

    def loss(v: jax.Array) -> jax.Array:
        return jnp.sum(signed_spherical_areas(v, m.faces) ** 2)

    g = jax.grad(loss)(m.vertices)
    assert g.shape == m.vertices.shape
    assert np.all(np.isfinite(np.asarray(g)))
