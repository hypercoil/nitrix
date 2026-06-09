# `compute_vertex_normals` — `nitrix.sparse.mesh`

> **Status (2026-06-09): SHIPPED.** `sparse/mesh.py` adds
> `compute_vertex_normals(vertices, faces)` — area-weighted face normals
> (`(v1−v0)×(v2−v0)`) scattered onto the three incident vertices then
> L2-normalised (zero-area → zero vector, not NaN). Verified radial on the
> icosphere. Consumer-pipeline substrate
> ([`ilex-pipeline-substrate.md`](ilex-pipeline-substrate.md), residual mesh
> item B); drivers `topofit` / `cortex_ode` / `surfnet`.

**What.** Per-vertex unit normals from triangle faces (face cross-products →
scatter-add to vertices → normalise).

**Drivers.** `topofit`; **and (2026-06-08 audit) `cortex_ode` / `surfnet`** —
the new ODE / diffeomorphic surface models hand-roll the same face-cross /
scatter-add / normalise (`cortex_ode/_pipeline_helpers.py:166`, numpy
`np.add.at`; the JAX form is `jnp.cross` + `segment_sum`). Recurrence is now
≥3 surface consumers.

**Home.** `nitrix.sparse.mesh` (next to the `Mesh` dataclass). Pairs with
[`mesh-laplacian-smoothing`](mesh-laplacian-smoothing.md) and
[`mesh-curvature`](mesh-curvature.md) as the per-vertex mesh family.

## Cross-references

- [`ilex-pipeline-substrate.md`](ilex-pipeline-substrate.md) — original
  survey context + the residual mesh/UNet tier.
- [`ilex-training-substrate.md`](ilex-training-substrate.md) — the 2026-06-08
  audit that added the `cortex_ode`/`surfnet` drivers.
- [`mesh-curvature.md`](mesh-curvature.md) (§12.6),
  [`mesh-laplacian-smoothing.md`](mesh-laplacian-smoothing.md) — share the
  per-face/per-vertex arithmetic on the `Mesh` container.
- `src/nitrix/sparse/mesh.py` — the `Mesh` dataclass.
