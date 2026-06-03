# B13. Boundary-mode parity for windowed / neighbourhood ops

> **Status (2026-06-03): parked (API refinement / drop-in parity) — measured
> boundary divergences from the de-facto reference tools.** Not a commitment —
> gated on the **Trigger** below. Effort **M**, additive (new boundary modes;
> no change to existing defaults). Provenance: surfaced building the
> `nitrix-perf-bench` `median_filter` + `bilateral_gaussian` cases; ledger
> context in [`internal-backlog.md`](internal-backlog.md), evidence in
> [`perf-bench-feedback.md`](perf-bench-feedback.md).

Several windowed / bounded-neighbourhood ops use **boundary conventions that
diverge from the de-facto standards** (scipy.ndimage, ITK). The *interiors*
match to round-off, but the border pixels do not — so a downstream user
expecting a drop-in scipy/ITK replacement gets different edge values, and
`nitrix-perf-bench` cannot use a shared full-image fp64 oracle for these ops
(it falls back to **interior-parity / no-oracle**, the
`fp64_reference=None` path). Two concrete cases:

**1. `morphology.median_filter` — NaN-shrink ≠ scipy `reflect`.** The op pads
the spatial dims with `NaN` and takes `nanmedian`, i.e. the border median is
over the *shrunken* in-bounds window ("ignore boundary positions"). This is
documented and intentional, but it **diverges from
`scipy.ndimage.median_filter`'s default `mode='reflect'`** (verified). A
`mode=` argument selecting the scipy boundary family (`reflect` / `nearest` /
`constant` / `wrap`) would make it a drop-in scipy replacement; the NaN-shrink
stays the (sensible) default.

**2. `sparse.regular_grid_stencil` — no `'valid'`/shrink mode (blocks ITK
bilateral parity).** Its boundary modes are `replicate` / `periodic` /
`reflect` — all **domain-extending** (every centre keeps a full `k_max`
window). ITK's `BilateralImageFilter` instead **shrinks the window at the edge
and renormalises** over only the in-bounds neighbours. With the stencil as the
neighbourhood, `smoothing.bilateral_gaussian` reproduces ITK's image bilateral
in the interior to **~1e-4**, but the `r`-pixel border diverges (replicate /
reflect both miss it). A **`'valid'` (mask-out-of-bounds) boundary mode** — the
stencil marks out-of-domain taps as padding so the consumer masks + renormalises
over real neighbours only — would close the border (and is the natural mode for
*any* "average over real neighbours only" reduction, e.g. masked smoothing,
mesh-boundary vertices).

**Scope note.** This is *not* universal: `morphology.{erode,dilate}` already
match `scipy.ndimage.grey_{erosion,dilation}` bit-for-bit (verified in
perf-bench), so the divergence is specific to `median_filter` and the
stencil-extended ops, not the morphology family as a whole.

**Trigger.** A consumer needing a drop-in scipy/ITK-parity boundary (porting a
pipeline, or matching a published result at the edge); or a `nitrix-perf-bench`
sweep wanting a full-image fp64 oracle (not just interior parity) for these
ops.

**Notes (evidence; `../nitrix-perf-bench/`).** `median_filter` case carries
`fp64_reference=None` ("nitrix NaN-pad shrink-window vs scipy reflect; interiors
match"). `bilateral_gaussian` case: interior matches `sitk.Bilateral` to ~1e-4,
boundary max-error ~7e-2 (replicate) / ~6e-2 (reflect) at `sigma_d=2`. Both
cases assert *interior* parity in tests in lieu of a full oracle.

**Effort.** M — additive boundary modes: `median_filter(mode=...)` (reuse the
existing pad machinery) + `regular_grid_stencil(boundary='valid')` (emit a
padding tap the consumer masks). No change to existing defaults / call sites.

**Cross-refs.** `../nitrix-perf-bench/reports/PERF_MEDIAN_FILTER.md`,
`PERF_BILATERAL_GAUSSIAN.md`; `src/nitrix/morphology/_median.py`;
`src/nitrix/sparse/grid.py` (`regular_grid_stencil`, `BoundaryMode`);
nitrix-perf-bench `9ebd933` (bilateral).

## Cross-references

- [`internal-backlog.md`](internal-backlog.md) — the engineering-backlog
  ledger.
- [`perf-bench-feedback.md`](perf-bench-feedback.md) — the perf-bench-surfaced
  ledger.
