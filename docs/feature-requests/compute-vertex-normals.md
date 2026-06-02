# `compute_vertex_normals` — `nitrix.sparse.mesh`

> **Status (2026-06-02): not started — CONVENIENCE (residual mesh/UNet
> primitive).** Consumer-pipeline substrate for the ilex → thrux migration.
> Provenance: 2026-06-02 ilex vendored-model survey
> ([`ilex-pipeline-substrate.md`](ilex-pipeline-substrate.md), residual mesh
> item B); ilex `UPSTREAM.md` "Pending primitives".

**What.** Per-vertex unit normals from triangle faces (face cross-products →
scatter-add to vertices → normalise).

**Driver.** `topofit`.

**Home.** `nitrix.sparse.mesh` (next to the `Mesh` dataclass).

## Cross-references

- [`ilex-pipeline-substrate.md`](ilex-pipeline-substrate.md) — survey
  context + the residual mesh/UNet tier.
- [`mesh-curvature.md`](mesh-curvature.md) (§12.6) — shares the
  per-face/per-vertex arithmetic on the `Mesh` container.
- `src/nitrix/sparse/mesh.py` — the `Mesh` dataclass.
