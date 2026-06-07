# G1. `spatial_transform(mode='linear_extrap')` — genuine linear extrapolator

> **Status (2026-06-02): parked — genuinely open, explicitly low priority
> (not needed for voxelmorph / JOSA parity).** Effort **S**. Provenance:
> the one genuinely-open leftover migrated from the sealed
> `NITRIX_FEEDBACK_JOSA.md`; ledger context in
> [`internal-backlog.md`](internal-backlog.md).

`geometry.spatial_transform`'s `BoundaryMode` is
`{'constant','nearest','wrap','mirror','reflect'}` — edge-replicate
(`'nearest'`, the voxelmorph `fill_value=None` convention) is covered, but
there is no true *linear extrapolation* at the boundary. Distinct from
`'nearest'`: pad-by-1 + edge-replicate the image + adjust coords by 1 to
continue the end-segment slope.

**Trigger.** A consumer where gradient continuity at the sampling boundary
matters (a sampler inside a learned generative model). **Not** required for
voxelmorph / JOSA parity (those use edge-replicate), so explicitly low
priority. Originally a JOSA-port request (low priority).

**Effort.** S — one `BoundaryMode` value handled in `_boundary_index`
(`geometry/_interpolate.py`); the explicit separable gather makes a
`'linear_extrap'` fold straightforward to add there (it would not, however,
have a `map_coordinates` equivalent, so the affected kernels would always
take the gather engine for that mode).

## Cross-references

- [`internal-backlog.md`](internal-backlog.md) — the engineering-backlog
  ledger.
- [`spatial-transform-batched.md`](spatial-transform-batched.md) — the other
  open `spatial_transform` extension (leading-batch convenience).
- `src/nitrix/geometry/_interpolate.py` — `_boundary_index` (the five
  boundary folds a sixth would join); `src/nitrix/geometry/grid.py` —
  `spatial_transform`.
