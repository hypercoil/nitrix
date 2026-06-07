# Cubic (order-3) resample — `nitrix.geometry.grid`

> **Status (2026-06-02): docstring deviation flagged; full cubic path still
> deferred.** The "at minimum, flag the deviation" ask is done — `resample`
> and `spatial_transform` now document that they are linear-only (order 0/1,
> via `map_coordinates`) and that order-3 bit-parity with nnUNet / `hd_bet`
> is **not** achieved (see `IMPLEMENTATION_PLAN.md §10.3`, 2026-06-02 entry).
> The separable B-spline prefilter + cubic sampler itself remains **not
> started** (a genuinely new sampling path; deferred pending a consumer that
> requires bit-parity). Provenance: 2026-06-02 ilex vendored-model survey
> ([`ilex-pipeline-substrate.md`](ilex-pipeline-substrate.md), volumetric
> item D); ilex SKILL FM #17.

**What.** A cubic-spline resampler. As of 2026-06-07 `geometry.resample` /
`spatial_transform` dispatch over an `Interpolator` set — `Linear`,
`NearestNeighbour`, `Lanczos` (windowed sinc), `MultiLabel` — so they are no
longer "linear-only", **but there is still no order-3 B-spline path.**
`Lanczos` is the high-fidelity option, yet it is a windowed-sinc kernel, not
a cubic B-spline: it does **not** give bit-parity with `hd_bet`'s nnUNet
order-3 preprocessing (or `scipy.ndimage.zoom(order=3)`). A separable
B-spline prefilter + cubic sampler is still the gap if that specific parity
is needed.

**What it needs.** A separable B-spline prefilter + cubic sampling. With the
dispatcher in place this now slots in cleanly as a **new `Interpolator`
record** (e.g. `CubicBSpline(order=3)`) in `geometry/_interpolate.py` — the
prefilter is a per-axis recursive IIR pass, the sampler a 4-tap separable
gather (the existing `_separable_gather` / `_separable_resample` machinery
takes any tap rule). No new top-level surface; just another `method=`.

**Priority.** Lower than the ENABLING items A–C (linear is "good enough" for
most consumers). The deviation is flagged in the `resample` docstring.

**Home.** `nitrix.geometry._interpolate` (a new `Interpolator` record).

## Cross-references

- [`ilex-pipeline-substrate.md`](ilex-pipeline-substrate.md) — survey
  context + the volumetric tier.
- [`pallas-trilinear-resample.md`](pallas-trilinear-resample.md) (B7) — the
  *linear* resampling perf track; explicitly a **separate** concern from
  this order-3 parity gap.
- `src/nitrix/geometry/_interpolate.py` — the `Interpolator` dispatcher a
  cubic method would extend; `src/nitrix/geometry/grid.py` —
  `spatial_transform` / `resample`.
