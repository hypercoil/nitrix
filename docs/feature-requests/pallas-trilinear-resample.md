# B7. Pallas kernel for 3-D trilinear resampling

> **Status (2026-06-02): parked (engineering backlog) — blocked on the ELL
> gather-lowering risk; a cheap pure-JAX interim win exists.** Not a
> commitment — gated on the **Trigger** below. Provenance: migrated from the
> retired top-level `BACKLOG.md` (B-numbering preserved); ledger context in
> [`internal-backlog.md`](internal-backlog.md).

`geometry.spatial_transform` / `integrate_velocity_field` resample via
`map_coordinates(order=1)`. (See also [`cubic-resample.md`](cubic-resample.md)
— cubic order-3 resampling is a separate, parity-driven gap.)

**Trigger.** Both of: (a) the baseline shows resampling is a real bottleneck
in a consumer training loop, and (b) a pointer-load (`pl.load`/`pl.ds`)
Pallas prototype clears the gather-lowering risk — i.e. avoids the HLO
`gather` primitive that Triton on the pinned JAX cannot lower (the ELL
blocker; `bench/G0_ELL_REPORT.md`).

**Notes.** Trilinear resampling is structurally a gather (8 corner voxels at
data-dependent positions), so it inherits the ELL gather blocker. Realistic
near-term outcome: JAX-default until upstream Pallas grows gather, unless the
pointer-load formulation works. A kernel must register a `custom_vjp`
(backward is scatter-add) and keep `map_coordinates` as the floor.

**Baseline (A10G, jax 0.10.0; `bench/PERF_TRILINEAR.md`).** Forward
`spatial_transform`: 256³ ~3.1 ms, 192³ ~1.4 ms; fwd+bwd 256³ ~13 ms.
**Cheap interim win, no Pallas:** an explicit pure-JAX 8-corner gather is
~1.5–1.7× faster than `map_coordinates` (256³: 1.80 ms vs 3.14 ms) —
`map_coordinates` carries dispatch overhead this op doesn't need. If a
consumer hits a resampling wall before the Pallas gate clears, swap
`_gather_coords_linear` to the explicit 8-corner form first.

## Cross-references

- [`internal-backlog.md`](internal-backlog.md) — the engineering-backlog
  ledger.
- [`cubic-resample.md`](cubic-resample.md) — the separate order-3 parity gap.
- `src/nitrix/geometry/grid.py`; `bench/G0_ELL_REPORT.md`.
