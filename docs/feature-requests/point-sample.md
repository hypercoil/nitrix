# `point_sample` / `sample_volume_at_points` — `nitrix.geometry.grid`

> **Status (2026-06-07): capability available via `spatial_transform`; a
> point-list convenience wrapper is still unshipped.** Zero-fill trilinear
> sampling at arbitrary float coordinates is
> `spatial_transform(volume, points, method=Linear(), mode='constant',
> cval=0.0)` — the dispatcher's shared sampler handles scattered coordinate
> fields, and `mode='constant', cval=0` is exactly the zero-fill this asks
> for (distinct from the edge-replicate `mode='nearest'`). What is *not* yet
> shipped is a thin `sample_at_points(volume, points, *, mode)` taking a flat
> `(n_points, ndim)` list (vs a `(*spatial, ndim)` field) — a reshape-and-call
> convenience, not new numerics. Provenance: 2026-06-02 ilex vendored-model
> survey ([`ilex-pipeline-substrate.md`](ilex-pipeline-substrate.md),
> residual mesh item A); ilex `UPSTREAM.md` "Pending primitives".

**What.** Trilinear interpolation of a 3-D volume at floating-point vertex
coordinates with **zero-fill** out-of-bounds — distinct from
`spatial_transform`'s edge-replicate `mode='nearest'`.

**Driver.** `topofit` (sampling the image-UNet feature volume at deforming
mesh vertices); future voxel-features-on-vertices consumers.

**Shape.** A `mode='zero'` point sampler, or an explicit
`sample_at_points(volume, points, *, mode)`.

**Home.** `nitrix.geometry.grid`.

## Cross-references

- [`ilex-pipeline-substrate.md`](ilex-pipeline-substrate.md) — survey
  context + the residual mesh/UNet tier.
- `src/nitrix/geometry/grid.py` — `spatial_transform` (the edge-replicate
  sampler this complements with a zero-fill point variant).
