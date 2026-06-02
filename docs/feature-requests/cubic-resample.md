# Cubic (order-3) resample — `nitrix.geometry.grid`

> **Status (2026-06-02): not started — MISMATCH (a documented parity-
> relevant gap, lower priority than the ENABLING items).** Consumer-pipeline
> substrate for the ilex → thrux migration. Provenance: 2026-06-02 ilex
> vendored-model survey
> ([`ilex-pipeline-substrate.md`](ilex-pipeline-substrate.md), volumetric
> item D); ilex SKILL FM #17.

**What.** A cubic-spline resampler. `geometry.spatial_transform` /
`resample` are **linear-only** — they wrap `jax.scipy.ndimage.map_coordinates`,
which supports `order` 0/1 only (the documented FreeSurfer-port deviation).
`hd_bet`'s nnUNet preprocessing resamples with an order-3 spline; the
linear-only path is a documented parity deviation, not a match.

**What it needs.** A separable B-spline prefilter + cubic sampling, rather
than living under the `map_coordinates` order cap. If bit-parity with nnUNet
preprocessing is wanted, this closes the gap.

**Priority.** Lower than the ENABLING items A–C (linear is "good enough" for
most consumers). At minimum, flag the deviation in the `resample` docstring.

**Home.** `nitrix.geometry.grid`.

## Cross-references

- [`ilex-pipeline-substrate.md`](ilex-pipeline-substrate.md) — survey
  context + the volumetric tier.
- [`pallas-trilinear-resample.md`](pallas-trilinear-resample.md) (B7) — the
  *linear* resampling perf track; explicitly a **separate** concern from
  this order-3 parity gap.
- `src/nitrix/geometry/grid.py` — `spatial_transform` / `resample`.
