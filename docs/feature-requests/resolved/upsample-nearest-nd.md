# `upsample_nearest_nd` — `nitrix.numerics` / `geometry`

> **Status (2026-06-07): ADDRESSED via the resample dispatcher.**
> Nearest-neighbour spatial resize (up *or* down) ships as
> `geometry.resample(image, target_shape, method=NearestNeighbour())` —
> the order-0 path of the interpolation-method dispatcher
> (`geometry/_interpolate.py`; `IMPLEMENTATION_PLAN.md §10.3`, 2026-06-07).
> A bare `upsample_nearest_nd(image, factors)` convenience wrapper (integer
> scale factors rather than a target shape) could still be added if a
> consumer prefers the factor-based call, but the capability is present.
> Provenance: 2026-06-02 ilex vendored-model survey
> ([`ilex-pipeline-substrate.md`](../ilex-pipeline-substrate.md), residual mesh
> item C); ilex `UPSTREAM.md` "Pending primitives".

**What.** Nearest-neighbour spatial upsample — a pure-array resize, so
`nitrix`-shaped.

**Drivers.** Inlined by ~7 neurite-family UNet decoders (`synthseg` /
`synthsr` / `fsm_seg` / `synthstrip` / `exvivo_*` / `voxelmorph`).

**Home.** `nitrix.numerics` / `geometry`.

## Cross-references

- [`ilex-pipeline-substrate.md`](../ilex-pipeline-substrate.md) — survey
  context + the residual mesh/UNet tier.
- `src/nitrix/morphology/pooling.py` — `max_unpool_nd` (the existing
  upsampling-shaped op; nearest-upsample is the cheaper sibling).
