# -*- coding: utf-8 -*-
"""Self-intersection detection / removal (geometry-suite P2.6 / GS-8).

Host-side QA: a clean convex mesh has none (adjacent faces excluded); a
deliberate triangle crossing is detected; a noised sphere's many
self-intersections are relaxed away while topology is preserved.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from nitrix.geometry import find_self_intersections, remove_self_intersections
from nitrix.sparse import Mesh, icosphere


def test_clean_sphere_has_none() -> None:
    # A convex mesh self-intersects nowhere -- and adjacent (vertex-sharing)
    # faces must NOT be reported.
    pairs = np.asarray(find_self_intersections(icosphere(3)))
    assert pairs.shape[0] == 0


def test_crossing_triangles_detected() -> None:
    # T2's vertical edge pierces T1 at (2, 1, 0), inside T1.
    verts = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [2.0, 3.0, 0.0],
            [2.0, 1.0, -1.0],
            [2.0, 1.0, 1.0],
            [2.0, 4.0, 0.0],
        ]
    )
    faces = jnp.array([[0, 1, 2], [3, 4, 5]])
    pairs = np.asarray(find_self_intersections(Mesh(verts, faces)))
    assert pairs.shape[0] == 1
    assert set(pairs[0].tolist()) == {0, 1}


def test_removal_reduces_intersections_and_keeps_topology() -> None:
    m = icosphere(3)
    noise = np.asarray(
        jax.random.normal(jax.random.PRNGKey(0), (m.n_vertices, 3))
    )
    noisy = Mesh(jnp.asarray(np.asarray(m.vertices) + 0.2 * noise), m.faces)
    before = int(np.asarray(find_self_intersections(noisy)).shape[0])
    assert before > 0  # noising a sphere creates many crossings

    fixed = remove_self_intersections(noisy, n_iterations=20)
    after = int(np.asarray(find_self_intersections(fixed)).shape[0])
    assert after <= before // 10  # relaxation removes the vast majority
    assert np.array_equal(np.asarray(fixed.faces), np.asarray(noisy.faces))


def test_removal_noop_on_clean_mesh() -> None:
    clean = icosphere(2)
    fixed = remove_self_intersections(clean, n_iterations=5)
    # Nothing to fix -> vertices unchanged.
    assert np.allclose(np.asarray(fixed.vertices), np.asarray(clean.vertices))
