# `point_sample` / `sample_volume_at_points` — `nitrix.geometry.grid`

> **Status (2026-06-09): SHIPPED.** `geometry.sample_at_points(volume,
> points, *, method, mode, cval)` samples a volume at a flat `(*n, ndim)`
> list of index-space points (the `align_corners` convention), wrapping the
> `_interpolate._sample_at_coords` dispatcher seam — so it inherits the full
> `Interpolator` ADT (`Linear`, `NearestNeighbour`, …), zero-fill
> (`mode='constant'`) **and** border-clamp (`mode='nearest'`), and handles
> both channel-free `(*spatial)` → `(*n)` and channels-last `(*spatial, c)` →
> `(*n, c)`. Resolves the `grid_sample` gap (3 duplicates across
> `cortex_ode`/`surfnet`; planned task #138). Provenance: 2026-06-02 ilex
> vendored-model survey
> ([`ilex-pipeline-substrate.md`](ilex-pipeline-substrate.md), residual mesh
> item A).

**What.** Trilinear interpolation of a 3-D volume at floating-point vertex
coordinates with **zero-fill** out-of-bounds — distinct from
`spatial_transform`'s edge-replicate `mode='nearest'`.

**Drivers.** `topofit` (sampling the image-UNet feature volume at deforming
mesh vertices); **and (2026-06-08 audit) `cortex_ode` / `surfnet`** — the new
surface models carry **3 live duplicates** of an arbitrary-point trilinear
sampler (`cortex_ode/_deformation_net.py:158` `_trilinear_sample_at`;
`surfnet/_surf_nnode.py:107` `_sample_feature_cube`; a 3rd in SurfNet_d).
These add two requirements beyond the zero-fill point sampler:

- **`align_corners=True` + border-clamp** (vs the zero-fill `mode='constant'`
  above) — the convention the trained surface checkpoints depend on.
- **multichannel** `(C, *spatial)` sampled at `(N, ndim)` points (one copy is
  single-channel, the other vmaps over channels).

`surfnet/_surf_nnode.py:60` already references this as **planned nitrix task
#138 `nitrix.image.grid_sample_3d`** with an `align_corners` / `cube_offsets`
contract. This is the `grid_sample` gap: nitrix's `_interpolate` dispatcher
samples on *grids*; this is sampling at an arbitrary *point set*.

**Shape.** A unified `sample_at_points(volume, points, *, mode, align_corners)`
on the existing `_interpolate` `_sample_at_coords` seam — `mode='constant'`
(zero-fill, topofit) and `mode='nearest'`/`'border'` + `align_corners`
(cortex_ode/surfnet) as the two modes; multichannel via a leading axis.

**Home.** `nitrix.geometry.grid` (on the `_interpolate` dispatcher).

## Cross-references

- [`ilex-pipeline-substrate.md`](ilex-pipeline-substrate.md) — original
  survey context + the residual mesh/UNet tier.
- [`ilex-training-substrate.md`](ilex-training-substrate.md) — the 2026-06-08
  audit that added the `cortex_ode`/`surfnet` `grid_sample` drivers + the
  `align_corners`/border-clamp/multichannel scope.
- `src/nitrix/geometry/grid.py`, `src/nitrix/geometry/_interpolate.py` —
  `spatial_transform` and the `_sample_at_coords` dispatcher seam this
  extends.
