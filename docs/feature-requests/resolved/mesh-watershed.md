# Watershed segmentation on meshes — `nitrix.graph.parcellation.watershed`

> **Status (2026-07-06): SHIPPED** (via the geometry suite).
> `graph.parcellation.mesh_watershed` (priority-flood, `h_min` depth-merge),
> tested in `test_parcellation.py`.
> Provenance: `docs/feature-requests catalogue §12.17`.

**What.** Priority-flood watershed on a vertex-valued scalar field with
arbitrary mesh adjacency.

**Proposed surface.**

```python
def mesh_watershed(
    field: Float[Array, 'n_vertices'],
    adjacency: ELL,
    *,
    min_basin_size: int = 1,
    h_min: float = 0.0,
) -> Int[Array, 'n_vertices']:
    '''Label each vertex by its watershed basin.'''
```

**Composition.** The boundary-mapping output
([`surface-boundary-map.md`](surface-boundary-map.md), §12.16) is the
natural input: local minima of `−B` are catchment-basin seeds; flooding from
sorted-by-value vertices yields the basin labels. The algorithm is
fundamentally serial (priority-queue flooding), so this is a **host-side
NumPy** primitive that returns a JAX int array — same pattern as the shipped
`mesh_k_ring_adjacency` (host-side BFS, JAX-array output). Composition with
the substrate is via the input `adjacency` `ELL` — works on any mesh
adjacency, not just icosphere.

**Likely consumer.** Gordon / Schaefer parcellation regeneration; any
boundary-map → discrete-parcels step; medical-image post-processing wanting
marker-based segmentation on a surface.

**Effort.** M. Priority-flood is well-understood (Barnes 2014 linear-time
variant); the engineering surface is correctness on irregular
triangulations with degenerate basins.

**Live-code status.** No `mesh_watershed`. The host-side-construct →
JAX-array-output pattern is established (`sparse.mesh_k_ring_adjacency`); the
`ELL` adjacency input type is shipped.

## Cross-references

- `docs/feature-requests catalogue §12.17` — origin entry; `§13` — acceptance protocol;
  `§12.20` — strategy survey (Gordon row: boundary_map + watershed).
- [`surface-boundary-map.md`](surface-boundary-map.md) — the upstream input.
- `src/nitrix/sparse/mesh.py` — `mesh_k_ring_adjacency`, the host-side
  precedent.
