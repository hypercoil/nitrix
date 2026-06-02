# `upsample_nearest_nd` — `nitrix.numerics` / `geometry`

> **Status (2026-06-02): not started — CONVENIENCE (residual mesh/UNet
> primitive).** Consumer-pipeline substrate for the ilex → thrux migration.
> Provenance: 2026-06-02 ilex vendored-model survey
> ([`ilex-pipeline-substrate.md`](ilex-pipeline-substrate.md), residual mesh
> item C); ilex `UPSTREAM.md` "Pending primitives".

**What.** Nearest-neighbour spatial upsample — a pure-array resize, so
`nitrix`-shaped.

**Drivers.** Inlined by ~7 neurite-family UNet decoders (`synthseg` /
`synthsr` / `fsm_seg` / `synthstrip` / `exvivo_*` / `voxelmorph`).

**Home.** `nitrix.numerics` / `geometry`.

## Cross-references

- [`ilex-pipeline-substrate.md`](ilex-pipeline-substrate.md) — survey
  context + the residual mesh/UNet tier.
- `src/nitrix/morphology/pooling.py` — `max_unpool_nd` (the existing
  upsampling-shaped op; nearest-upsample is the cheaper sibling).
