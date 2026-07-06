# `spatial_transform_batched` — `nitrix.geometry.grid`

> **Status (2026-06-02): SHIPPED.** `geometry.spatial_transform_batched`
> `vmap`s `spatial_transform` over a single leading batch axis, broadcasting
> whichever of image / deformation lacks it (the shared-operand case
> `spatial_transform` deliberately refuses). See `IMPLEMENTATION_PLAN.md
> §10.3` (2026-06-02 entry). Provenance: 2026-06-02 ilex vendored-model survey
> ([`ilex-pipeline-substrate.md`](../ilex-pipeline-substrate.md), residual mesh
> item D); originally a JOSA-port convenience request.

**What.** A leading-batch convenience that internally `vmap`s
`spatial_transform` — saves one wrap line per consuming model and keeps the
cross-model surface uniform.

**Home.** `nitrix.geometry.grid`.

## Cross-references

- [`ilex-pipeline-substrate.md`](../ilex-pipeline-substrate.md) — survey
  context + the residual mesh/UNet tier.
- [`spatial-transform-linear-extrap.md`](../spatial-transform-linear-extrap.md)
  (G1) — the other open `spatial_transform` extension (boundary mode).
- `src/nitrix/geometry/grid.py` — `spatial_transform`.
