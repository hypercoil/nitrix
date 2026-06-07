# B7. Pallas kernel for 3-D trilinear resampling

> **Status (2026-06-07): interim pure-JAX win SHIPPED; the Pallas kernel
> itself stays parked** (blocked on the ELL gather-lowering risk). The
> explicit separable gather landed with the interpolation-method dispatcher
> (`geometry/_interpolate.py`, `_separable_gather`): it is now the GPU engine
> for `Linear` / `NearestNeighbour` (and the always-engine for `Lanczos`),
> **platform-branched** because the win inverts on CPU (see below). The Pallas
> pointer-load kernel remains a commitment-free, gated follow-up. Provenance:
> migrated from the retired top-level `BACKLOG.md` (B-numbering preserved);
> ledger context in [`internal-backlog.md`](internal-backlog.md);
> `IMPLEMENTATION_PLAN.md §10.3` (2026-06-07).

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
**Interim win, no Pallas — SHIPPED (2026-06-07):** the explicit pure-JAX
separable gather (`_interpolate._separable_gather`, generalising the
8-corner gather to `T**ndim` taps so it also serves `Lanczos`) is
~1.5–1.7× faster than `map_coordinates` **on GPU** (256³: 1.80 ms vs
3.14 ms). The win is GPU-specific: on CPU the gather is ~1.3× *slower*
(the XLA `map_coordinates` lowering is tighter, and CPU interpolation is
the B15 throughput-sensitive path), so `Linear` / `NearestNeighbour`
select the engine per platform (`default_backend_is_gpu`, the
`signal._iir` precedent) — explicit gather on GPU, `map_coordinates` on
CPU, parity-equal to a ULP.

## Cross-references

- [`internal-backlog.md`](internal-backlog.md) — the engineering-backlog
  ledger.
- [`cubic-resample.md`](cubic-resample.md) — the separate order-3 parity gap.
- `src/nitrix/geometry/_interpolate.py` (`_separable_gather`,
  `_gather_sample` — the shipped interim win); `src/nitrix/geometry/grid.py`;
  `bench/G0_ELL_REPORT.md`.
