# -*- coding: utf-8 -*-
"""Surface functional parcellation (geometry-suite P5.1 / 12.16 + 12.17).

``surface_boundary_map`` (connectivity-profile boundary detection composing
``semiring_ell_edge_aggregate``) + ``eta_squared`` similarity, and
``mesh_watershed`` (host-side priority-flood basins).  Anchored on synthetic
two-region profiles (boundary high at the seam; watershed recovers the two
parcels) and exercised on a real fsaverage5 adjacency.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _real_meshes import fsaverage_white

from nitrix.graph import eta_squared, mesh_watershed, surface_boundary_map
from nitrix.sparse import Mesh, icosphere, mesh_k_ring_adjacency


def _two_region_profiles(mesh: Mesh, noise: float = 0.02) -> np.ndarray:
    z = np.asarray(mesh.vertices[:, 2])
    prof = np.zeros((mesh.n_vertices, 6), dtype=np.float32)
    prof[z >= 0] = np.array([1, 1, 1, 0, 0, 0])
    prof[z < 0] = np.array([0, 0, 0, 1, 1, 1])
    prof += noise * np.random.default_rng(0).standard_normal(prof.shape)
    return prof


def _is_connected(labels: np.ndarray, label: int, neighbours) -> bool:
    members = set(np.where(labels == label)[0].tolist())
    if not members:
        return True
    start = next(iter(members))
    seen = {start}
    stack = [start]
    while stack:
        u = stack.pop()
        for v in neighbours[u]:
            if v in members and v not in seen:
                seen.add(v)
                stack.append(v)
    return seen == members


# --------------------------------------------------------------------------- #
# eta_squared
# --------------------------------------------------------------------------- #


def test_eta_squared_identical_and_reversed() -> None:
    a = jnp.array([1.0, 2.0, 3.0, 4.0])
    assert np.isclose(float(eta_squared(a, a)), 1.0, atol=1e-6)
    rev = jnp.array([4.0, 3.0, 2.0, 1.0])
    assert float(eta_squared(a, rev)) < 0.1  # anti-aligned -> near 0


def test_eta_squared_batched_and_range() -> None:
    rng = np.random.default_rng(1)
    x = jnp.asarray(rng.standard_normal((20, 8)))
    y = jnp.asarray(rng.standard_normal((20, 8)))
    e = np.asarray(eta_squared(x, y))
    assert e.shape == (20,)
    assert np.all(e <= 1.0 + 1e-6)


def test_eta_squared_differentiable() -> None:
    a = jnp.array([1.0, 2.0, 3.0, 4.0])
    b = jnp.array([1.5, 1.0, 3.5, 3.0])
    g = jax.grad(lambda u: eta_squared(u, b))(a)
    assert np.all(np.isfinite(np.asarray(g)))


# --------------------------------------------------------------------------- #
# surface_boundary_map
# --------------------------------------------------------------------------- #


def test_boundary_high_at_seam_low_in_interior() -> None:
    m = icosphere(3)
    adj = mesh_k_ring_adjacency(m, k=1)
    z = np.asarray(m.vertices[:, 2])
    prof = _two_region_profiles(m)
    b = np.asarray(surface_boundary_map(jnp.asarray(prof), adj))
    assert b[np.abs(z) < 0.2].mean() > 10 * b[np.abs(z) > 0.6].mean()


@pytest.mark.parametrize('similarity', ['eta_squared', 'pearson'])
@pytest.mark.parametrize('aggregate', ['mean', 'max'])
def test_boundary_modes_run_and_separate(
    similarity: str, aggregate: str
) -> None:
    m = icosphere(3)
    adj = mesh_k_ring_adjacency(m, k=1)
    z = np.asarray(m.vertices[:, 2])
    b = np.asarray(
        surface_boundary_map(
            jnp.asarray(_two_region_profiles(m)),
            adj,
            similarity=similarity,
            aggregate=aggregate,
        )
    )
    assert np.all(np.isfinite(b))
    assert b[np.abs(z) < 0.2].mean() > b[np.abs(z) > 0.6].mean()


def test_boundary_bad_args_raise() -> None:
    m = icosphere(2)
    adj = mesh_k_ring_adjacency(m, k=1)
    prof = jnp.asarray(_two_region_profiles(m))
    with pytest.raises(ValueError, match="'eta_squared' or 'pearson'"):
        surface_boundary_map(prof, adj, similarity='nope')
    with pytest.raises(ValueError, match="'mean' or 'max'"):
        surface_boundary_map(prof, adj, aggregate='nope')


def test_boundary_differentiable() -> None:
    m = icosphere(2)
    adj = mesh_k_ring_adjacency(m, k=1)
    prof = jnp.asarray(_two_region_profiles(m))

    def loss(p: jax.Array) -> jax.Array:
        return jnp.sum(surface_boundary_map(p, adj) ** 2)

    g = jax.grad(loss)(prof)
    assert g.shape == prof.shape
    assert np.all(np.isfinite(np.asarray(g)))


# --------------------------------------------------------------------------- #
# mesh_watershed
# --------------------------------------------------------------------------- #


def test_watershed_two_wells_two_contiguous_basins() -> None:
    m = icosphere(3)
    adj = mesh_k_ring_adjacency(m, k=1)
    from nitrix.graph.parcellation import _neighbour_lists

    z = np.asarray(m.vertices[:, 2])
    field = jnp.asarray((-np.abs(z)).astype(np.float32))  # minima at the poles
    lab = np.asarray(mesh_watershed(field, adj))
    labels = np.unique(lab)
    assert labels.size == 2
    nbrs = _neighbour_lists(adj)
    for label in labels:
        assert _is_connected(lab, int(label), nbrs)


def test_watershed_constant_field_single_basin() -> None:
    m = icosphere(2)
    adj = mesh_k_ring_adjacency(m, k=1)
    field = jnp.zeros((m.n_vertices,))  # one flat plateau -> one basin
    lab = np.asarray(mesh_watershed(field, adj))
    assert np.unique(lab).size == 1


def test_watershed_depth_merge_cleans_noise() -> None:
    # A noisy boundary map floods into many shallow basins; h_min collapses
    # them to the two true regions.
    m = icosphere(3)
    adj = mesh_k_ring_adjacency(m, k=1)
    b = surface_boundary_map(jnp.asarray(_two_region_profiles(m)), adj)
    raw = np.unique(np.asarray(mesh_watershed(b, adj))).size
    merged = np.unique(np.asarray(mesh_watershed(b, adj, h_min=0.05))).size
    assert raw > merged
    assert merged == 2


def test_watershed_min_basin_size_reduces_count() -> None:
    m = icosphere(3)
    adj = mesh_k_ring_adjacency(m, k=1)
    b = surface_boundary_map(jnp.asarray(_two_region_profiles(m)), adj)
    raw = np.unique(np.asarray(mesh_watershed(b, adj))).size
    fewer = np.unique(
        np.asarray(mesh_watershed(b, adj, min_basin_size=20))
    ).size
    assert fewer < raw


def test_watershed_labels_are_contiguous_range() -> None:
    m = icosphere(2)
    adj = mesh_k_ring_adjacency(m, k=1)
    field = jnp.asarray(np.asarray(m.vertices[:, 0]).astype(np.float32))
    lab = np.asarray(mesh_watershed(field, adj))
    u = np.unique(lab)
    assert u.min() == 0 and u.max() == u.size - 1  # 0..n_basins-1


# --------------------------------------------------------------------------- #
# Pipeline + real mesh
# --------------------------------------------------------------------------- #


def test_pipeline_recovers_two_parcels() -> None:
    m = icosphere(3)
    adj = mesh_k_ring_adjacency(m, k=1)
    b = surface_boundary_map(jnp.asarray(_two_region_profiles(m)), adj)
    parcels = mesh_watershed(b, adj, h_min=0.05)
    assert np.unique(np.asarray(parcels)).size == 2


def test_real_fsaverage_parcellation_smoke() -> None:
    # Real adjacency + smooth synthetic profiles: the boundary map is finite
    # and the watershed yields a sensible, in-range parcellation.
    v, f, _ = fsaverage_white()
    mesh = Mesh(jnp.asarray(v), jnp.asarray(f))
    adj = mesh_k_ring_adjacency(mesh, k=1)
    coords = np.asarray(v)
    coords = (coords - coords.mean(0)) / (coords.std(0) + 1e-6)
    # 4 smooth spatial profile channels (so neighbours are mostly similar).
    prof = np.stack(
        [
            np.sin(0.5 * coords[:, 0]),
            np.cos(0.5 * coords[:, 1]),
            np.sin(0.5 * coords[:, 2]),
            coords[:, 0] * coords[:, 1],
        ],
        axis=1,
    ).astype(np.float32)
    b = surface_boundary_map(jnp.asarray(prof), adj)
    assert np.all(np.isfinite(np.asarray(b)))
    parcels = np.asarray(mesh_watershed(b, adj, min_basin_size=30))
    n_parcels = np.unique(parcels).size
    assert 1 <= n_parcels < mesh.n_vertices
    assert parcels.min() == 0 and parcels.max() == n_parcels - 1


# --------------------------------------------------------------------------- #
# ROI masking (audit AI-B1)
# --------------------------------------------------------------------------- #


def test_boundary_map_roi_zeroes_outside_and_ignores_offroi_edges() -> None:
    m = icosphere(3)
    adj = mesh_k_ring_adjacency(m, k=1)
    z = np.asarray(m.vertices[:, 2])
    roi = jnp.asarray(z >= 0.0)
    prof = _two_region_profiles(m)
    b = np.asarray(surface_boundary_map(jnp.asarray(prof), adj, roi=roi))
    roi_np = np.asarray(roi)
    # Out-of-ROI vertices are exactly zero.
    assert np.allclose(b[~roi_np], 0.0)
    # In-ROI values are finite and present.
    assert np.all(np.isfinite(b[roi_np]))


def test_watershed_roi_labels_background_minus_one() -> None:
    m = icosphere(3)
    adj = mesh_k_ring_adjacency(m, k=1)
    z = np.asarray(m.vertices[:, 2])
    roi = jnp.asarray(z >= 0.0)
    field = jnp.asarray((-np.abs(z)).astype(np.float32))  # poles are minima
    lab = np.asarray(mesh_watershed(field, adj, roi=roi))
    roi_np = np.asarray(roi)
    assert np.all(lab[~roi_np] == -1)  # background outside ROI
    assert np.all(lab[roi_np] >= 0)  # every ROI vertex is in a basin
    # Basins are confined to the ROI (the north well only).
    assert set(np.unique(lab[roi_np]).tolist()) == {0}
