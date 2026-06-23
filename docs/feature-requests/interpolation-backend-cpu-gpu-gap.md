# B15. Interpolation backend: CPU-weak vs scipy, single-shot GPU-lag vs cupyx

> **Status (2026-06-04): open (perf characterisation).** Measured, not a
> commitment — gated on the **Trigger** below. Provenance: surfaced across
> several `nitrix-perf-bench` geometry cases (`integrate_velocity_field`,
> `spatial_transform`, `resample`); ledger context in
> [`internal-backlog.md`](internal-backlog.md) and
> [`perf-bench-feedback.md`](perf-bench-feedback.md). The companion research
> note on alternative backends is
> [`alternative-interp-backends-xla`](alternative-interp-backends-xla.md).
>
> **R8 narrowing (2026-06-10).** The "CPU-weak everywhere" framing is too broad
> and was mis-cited as *the* registration CPU lag. Reconciling the dated
> measurements: **single-shot** `spatial_transform` (4.0–4.9× vs scipy at 256²)
> and `resample` (5.6× at 128²→256²) on CPU are **faster** than scipy — only the
> **iterated** `integrate_velocity_field` (the 7× scaling-and-squaring scan, at
> small grids; table below) is the 5–9× loss, where XLA-CPU per-call/scan
> overhead dominates and scipy's tight C loop wins. So this is a **narrow** gap:
> CPU-side SVF-exponential preprocessing at small grids — *not* the warp gather
> in general, and **not** the rigid/affine recipes (which never call
> `integrate_velocity_field`; their CPU lag is the optimiser, addressed
> separately). Demoted from "highest" in the registration perf program.

Every nitrix op that resamples by coordinate gather routes through
`jax.scipy.ndimage.map_coordinates` (order=1) — `spatial_transform`,
`resample`, and (iterated 7×) `integrate_velocity_field`. The benchmark shows
this gather is nitrix's clearest **relative** weak spot, and the gap has two
distinct faces:

**1. CPU: jax's `map_coordinates` loses to scipy's C across the board.**
`integrate_velocity_field` (scaling-and-squaring; the reference is a numpy loop
whose composition core is `scipy.ndimage.map_coordinates`, order=1,
mode='nearest', verified bit-equal to ~5e-16 in fp64):

| grid (d³, ndim=3) | nitrix-jax CPU | scipy.ndimage CPU | scipy speedup |
|---|---|---|---|
| 16³ | 47.0 ms | 5.34 ms | **8.8×** |
| 24³ | 78.1 ms | 16.5 ms | **4.7×** |
| 32³ | 392.7 ms | 48.0 ms | **8.2×** |

So scipy's optimised-C gather is ~5–9× faster than the XLA-CPU lowering for the
same linear-interpolation work. The same backend powers `spatial_transform` /
`resample`, so they inherit the CPU lag.

**2. GPU: split.** When the gather is **jit-fused into a larger graph**, nitrix
wins decisively — `integrate_velocity_field` on the L4 is **13×** over the cupy
reimplementation (and 220× over the CPU-bound scipy floor): XLA fuses the
7-step unrolled composition, while cupy pays a kernel launch per step. But for a
**single-shot** gather, cupyx's specialised kernel edges nitrix out:
`spatial_transform`'s benchmark records `gpu_ref_ratio ≈ 0.56` vs
`cupyx.scipy.ndimage.map_coordinates` (i.e. nitrix ~1.8× slower on GPU for the
one-shot warp). So the GPU weakness is narrow (single-shot interpolation), not
the iterated case.

**Takeaway.** GPU-strong when fused, CPU-weak everywhere, single-shot GPU-lag
vs cupyx. All three ops are *numerically correct* (fidelity passes everywhere);
this is purely a throughput characterisation. Fixing the CPU path is the
high-value half — fMRI/registration preprocessing that runs CPU-side eats the
5–9× directly.

**Trigger.** A consumer running CPU-side registration / resampling preprocessing
at scale (the 5–9× is a real wall-clock cost), or the
[`alternative-interp-backends-xla`](alternative-interp-backends-xla.md)
research note concluding a backend swap is cheap enough to land.

**Effort.** S to characterise further (already mostly done here); the fix is
M–L and is the subject of the companion research note.

## Cross-references

- [`alternative-interp-backends-xla`](alternative-interp-backends-xla.md) — the
  research note on scipy / cupy backends spliced into the XLA graph.
- [`pallas-trilinear-resample`](pallas-trilinear-resample.md) (B7) — a Pallas
  resample kernel: an in-XLA alternative to an external backend.
- [`spatial-transform-linear-extrap`](spatial-transform-linear-extrap.md),
  [`boundary-mode-parity`](boundary-mode-parity.md) — sibling
  `map_coordinates` items.
- [`cubic-bspline-prefilter-backend-parity`](cubic-bspline-prefilter-backend-parity.md)
  — the order-3 *numeric parity* sibling to this order-1 *perf* gap: the
  `CubicBSpline` prefilter's `associative=default_backend_is_gpu()` scan is
  backend-dependent (this gap's order-1 path is not).
- `src/nitrix/geometry/_interpolate.py` (`_map_coordinates_sample`,
  `_separable_gather`, `_gather_sample` — the per-platform engine choice that
  this gap motivated); `src/nitrix/geometry/grid.py` (`spatial_transform`,
  `resample`, `integrate_velocity_field`); nitrix-perf-bench
  `integrate_velocity_field` / `spatial_transform` / `resample` cases.
