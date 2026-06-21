# -*- coding: utf-8 -*-
"""Phase-0 substrate on a *real* FreeSurfer surface (geometry-suite P0.5).

The user directive: validate on real meshes early, so the suite does not break
when it leaves the mathematical abstraction of the unit icosphere.  These run
the Phase-0 substrate (pytree containers, areas/mass, sectioned emission, the
apply-seam) on fsaverage5's white surface -- irregular, folded, mm-scale,
fp32, with ~27% obtuse triangles -- and assert robust geometric invariants
(the FS overlays are not tight oracles here; see ``_real_meshes``).

Skips cleanly when nilearn / the fsaverage download is unavailable.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from _real_meshes import fsaverage_white

from nitrix.sparse import (
    ELL,
    Mesh,
    SectionedELL,
    apply_operator,
    face_areas,
    mesh_cotangent_laplacian,
    mesh_k_ring_adjacency,
    vertex_areas,
)


def _real_mesh() -> Mesh:
    v, f, _ = fsaverage_white()
    return Mesh(jnp.asarray(v), jnp.asarray(f))


# --------------------------------------------------------------------------- #
# P0.1 -- the Mesh pytree on a real surface (the array-handoff seam)
# --------------------------------------------------------------------------- #


def test_real_mesh_pytree_roundtrip_and_jit() -> None:
    mesh = _real_mesh()
    assert mesh.n_vertices == 10242
    rebuilt = jax.tree_util.tree_unflatten(
        *reversed(jax.tree_util.tree_flatten(mesh))
    )
    assert isinstance(rebuilt, Mesh)
    centroid = jax.jit(lambda m: m.vertices.mean(0))(mesh)
    assert np.all(np.isfinite(np.asarray(centroid)))


# --------------------------------------------------------------------------- #
# P0.2 -- areas / mass: geometric invariants on the real (folded) mesh
# --------------------------------------------------------------------------- #


def test_real_area_partition_invariant() -> None:
    # Each triangle's area is fully distributed -> sum matches, both schemes.
    mesh = _real_mesh()
    fa = float(jnp.sum(face_areas(mesh)))
    for scheme in ('voronoi', 'barycentric'):
        va = vertex_areas(mesh, scheme=scheme)
        assert np.all(np.asarray(va) > 0.0)  # positive on a real surface
        assert np.all(np.isfinite(np.asarray(va)))  # fp32 at mm scale, no NaN
        assert np.allclose(float(jnp.sum(va)), fa, rtol=1e-4)


def test_real_obtuse_branch_is_exercised() -> None:
    # ~27% of real white-surface triangles are obtuse (icosphere: ~0%), so the
    # mixed-Voronoi obtuse branch genuinely fires and diverges from barycentric.
    mesh = _real_mesh()
    v, f = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    da = ((b - a) * (c - a)).sum(-1)
    db = ((a - b) * (c - b)).sum(-1)
    dc = ((a - c) * (b - c)).sum(-1)
    obtuse_frac = float(((da < 0) | (db < 0) | (dc < 0)).mean())
    assert obtuse_frac > 0.1  # the real surface is full of obtuse triangles

    vor = np.asarray(vertex_areas(mesh, scheme='voronoi'))
    bary = np.asarray(vertex_areas(mesh, scheme='barycentric'))
    # Same total, but per-vertex they differ where triangles are obtuse.
    assert not np.allclose(vor, bary, atol=1e-3)


# --------------------------------------------------------------------------- #
# P0.3 / P0.4 -- format dispatch + apply-seam on the real operator
# --------------------------------------------------------------------------- #


def test_real_auto_format_stays_flat() -> None:
    # fsaverage is icosphere-derived (valence 5-6) -> 'auto' keeps it flat.
    mesh = _real_mesh()
    assert isinstance(mesh_cotangent_laplacian(mesh, format='auto'), ELL)
    assert isinstance(mesh_k_ring_adjacency(mesh, format='auto'), ELL)


def test_real_sectioned_matches_ell_through_seam() -> None:
    mesh = _real_mesh()
    x = jax.random.normal(jax.random.PRNGKey(0), (mesh.n_vertices, 3))
    ell = mesh_cotangent_laplacian(mesh, format='ell')
    sec = mesh_cotangent_laplacian(mesh, format='sectioned')
    assert isinstance(sec, SectionedELL)
    assert np.allclose(
        np.asarray(apply_operator(ell, x)),
        np.asarray(apply_operator(sec, x)),
        atol=1e-3,
    )


def test_real_laplacian_annihilates_constants() -> None:
    # The defining property of any Laplacian: rows sum to zero, so L @ 1 == 0.
    # Holds on the real obtuse-heavy mesh (negative cotangents and all).
    mesh = _real_mesh()
    ones = jnp.ones((mesh.n_vertices, 1))
    lap = mesh_cotangent_laplacian(mesh)
    out = apply_operator(lap, ones)
    # mm-scale cotangent weights -> use an absolute floor scaled by the operator.
    scale = float(jnp.max(jnp.abs(lap.values)))
    assert float(jnp.max(jnp.abs(out))) < 1e-3 * scale
