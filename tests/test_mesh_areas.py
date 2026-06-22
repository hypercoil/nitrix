# -*- coding: utf-8 -*-
"""Per-vertex / per-face areas + lumped mass matrix (geometry-suite P0.2 / GS-5).

Anchored on three oracles:
- the exact partition invariant ``sum(vertex_areas) == sum(face_areas)``;
- the unit-sphere area ``4 pi`` (icosphere analytic oracle);
- the **obtuse-triangle branch** of the mixed Voronoi rule (the case the
  icosphere never exercises but real cortical surfaces do).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nitrix.sparse import (
    Mesh,
    apply_operator,
    face_areas,
    icosphere,
    mesh_mass_matrix,
    vertex_areas,
)

_EQUILATERAL = Mesh(
    vertices=jnp.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, float(np.sqrt(3) / 2), 0.0]]
    ),
    faces=jnp.array([[0, 1, 2]]),
)
# Obtuse at vertex 1 (B): B->A . B->C = -4 < 0.  area = 2.
_OBTUSE = Mesh(
    vertices=jnp.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [5.0, 1.0, 0.0]]),
    faces=jnp.array([[0, 1, 2]]),
)


def test_face_area_single_triangle() -> None:
    assert np.allclose(np.asarray(face_areas(_OBTUSE)), [2.0])
    assert np.allclose(
        np.asarray(face_areas(_EQUILATERAL)), [float(np.sqrt(3) / 4)]
    )


@pytest.mark.parametrize('scheme', ['voronoi', 'barycentric'])
def test_vertex_areas_partition_invariant(scheme: str) -> None:
    # Each triangle's area is fully distributed among its vertices.
    mesh = icosphere(3)
    va = vertex_areas(mesh, scheme=scheme)
    assert np.allclose(
        float(jnp.sum(va)), float(jnp.sum(face_areas(mesh))), rtol=1e-5
    )


def test_unit_sphere_area_approaches_4pi() -> None:
    mesh = icosphere(5)  # 10242 vertices
    total = float(jnp.sum(face_areas(mesh)))
    # Inscribed geodesic polyhedron: area < 4*pi, converging quadratically.
    assert total <= 4 * np.pi
    assert abs(total - 4 * np.pi) / (4 * np.pi) < 0.01


def test_voronoi_equals_barycentric_on_equilateral() -> None:
    # An acute (equilateral) triangle: Voronoi == barycentric == area/3.
    expected = float(np.sqrt(3) / 4) / 3.0
    vor = vertex_areas(_EQUILATERAL, scheme='voronoi')
    bary = vertex_areas(_EQUILATERAL, scheme='barycentric')
    assert np.allclose(np.asarray(vor), expected)
    assert np.allclose(np.asarray(bary), expected)


def test_obtuse_branch_assigns_half_and_quarters() -> None:
    # The load-bearing obtuse branch: obtuse vertex -> area/2, others -> area/4.
    vor = vertex_areas(_OBTUSE, scheme='voronoi')
    assert np.allclose(np.asarray(vor), [0.5, 1.0, 0.5])  # area == 2
    # Barycentric ignores obtuseness (area/3 each) -> the schemes differ here.
    bary = vertex_areas(_OBTUSE, scheme='barycentric')
    assert np.allclose(np.asarray(bary), [2.0 / 3, 2.0 / 3, 2.0 / 3])
    assert not np.allclose(np.asarray(vor), np.asarray(bary))


def test_vertex_areas_rejects_bad_scheme() -> None:
    with pytest.raises(ValueError, match='voronoi.*barycentric'):
        vertex_areas(icosphere(0), scheme='mixed')


def test_mass_matrix_is_diagonal_vertex_areas() -> None:
    mesh = icosphere(2)
    m = mesh_mass_matrix(mesh, scheme='voronoi')
    assert m.k_max == 1
    va = vertex_areas(mesh, scheme='voronoi')
    assert np.allclose(np.asarray(m.values[:, 0]), np.asarray(va))
    # As an operator, M @ x == vertex_areas[:, None] * x.
    x = jax.random.normal(jax.random.PRNGKey(0), (mesh.n_vertices, 3))
    mx = apply_operator(m, x)
    assert np.allclose(np.asarray(mx), np.asarray(va[:, None] * x), atol=1e-5)


def test_mass_matrix_lumped_false_raises() -> None:
    with pytest.raises(NotImplementedError, match='consistent'):
        mesh_mass_matrix(icosphere(0), lumped=False)


def test_vertex_areas_differentiable() -> None:
    mesh = icosphere(2)

    def total_area(v: jax.Array) -> jax.Array:
        return jnp.sum(vertex_areas(Mesh(v, mesh.faces), scheme='voronoi'))

    g = jax.grad(total_area)(mesh.vertices)
    assert g.shape == mesh.vertices.shape
    assert np.all(np.isfinite(np.asarray(g)))


# --------------------------------------------------------------------------- #
# Obtuse-triangle mixed-Voronoi partition (audit AI-B4)
# --------------------------------------------------------------------------- #


def test_obtuse_triangle_mixed_voronoi_partition() -> None:
    # The Meyer mixed rule: in an OBTUSE triangle the Voronoi partition is
    # replaced by area/2 at the obtuse vertex and area/4 at the other two.  The
    # icosphere never exercises this branch; a single hand-built obtuse triangle
    # does.  Triangle (0,0,0)-(4,0,0)-(5,1,0): obtuse at vertex 1 (dot of its two
    # edge vectors = -4 < 0); face area = 2.
    tri = Mesh(
        vertices=jnp.array(
            [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [5.0, 1.0, 0.0]]
        ),
        faces=jnp.array([[0, 1, 2]]),
    )
    area = float(face_areas(tri)[0])
    assert np.isclose(area, 2.0, atol=1e-5)
    va = np.asarray(vertex_areas(tri, scheme='voronoi'))
    # Partition of unity (sums to the face area).
    assert np.isclose(va.sum(), area, atol=1e-5)
    # Obtuse vertex (index 1) gets area/2; the two acute vertices area/4.
    assert np.allclose(va, [area / 4, area / 2, area / 4], atol=1e-5)
