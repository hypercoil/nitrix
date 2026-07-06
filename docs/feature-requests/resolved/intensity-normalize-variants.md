# Intensity-normalize variants — `nitrix.numerics.normalize`

> **Status (2026-06-02): SHIPPED.** Added `percentile_rescale(x, *, lo=0.0,
> hi=99.0, clip=True, axis=None)` (the min–p99–clip recipe) and a
> `nonzero_mask=` option on `zscore_normalize` (per-channel foreground
> z-score, background left at 0) on the existing `numerics.normalize`
> surface — no new top-level concept. See `IMPLEMENTATION_PLAN.md §10.3`
> (2026-06-02 entry). Provenance: 2026-06-02 ilex vendored-model survey
> ([`ilex-pipeline-substrate.md`](../ilex-pipeline-substrate.md), volumetric
> item E).

**What.** `numerics.normalize` ships `zscore_normalize`,
`robust_zscore_normalize`, `intensity_normalize`, `psc_normalize`, `demean`
— but two recurring upstream recipes are not exactly reachable:

- **min–p99–clip** (`x → clip((x − min) / p99, 0, 1)`): `synthstrip`,
  `synthdist`, `synthsr` (after CT-clip). The existing two-sided percentile
  path is not this strict min/p99 form.
- **per-channel *nonzero*-masked z-score**: `brats_segresnet`,
  `brainsegfounder` (BraTS multi-channel). Needs `mask = x != 0` per channel,
  distinct from the global/robust z-score.

**Shape.** Add these as variants/kwargs on the existing `normalize` surface
(e.g. a `percentile_rescale(x, *, lo=0.0, hi=99.0, clip=True)` and a
`nonzero_mask=` / `axis=` channel-wise option) rather than new top-level
names.

**Home.** `nitrix.numerics.normalize`.

**Remaining gap (2026-06-08 audit) — RESOLVED 2026-06-09.** FM
`percentile_normalize` (`ilex/train/augment/intensity.py:166`) computes its
clip percentiles over **non-zero voxels only** (the skull-strip-aware
variant). `percentile_rescale` now takes a `mask=` argument: percentiles are
computed over the masked voxels via `nanpercentile` (falling back to the
global min/max on empty slices), while every voxel is still rescaled. Pass
`mask = x != 0` for the non-zero recipe. Small extension, not a new symbol. See
[`intensity-augmentation-ops.md`](intensity-augmentation-ops.md) for the
training-side intensity perturbations this normaliser pairs with.

## Cross-references

- [`ilex-pipeline-substrate.md`](../ilex-pipeline-substrate.md) — survey
  context + the volumetric tier.
- `src/nitrix/numerics/normalize.py` — the existing normalisation family.
