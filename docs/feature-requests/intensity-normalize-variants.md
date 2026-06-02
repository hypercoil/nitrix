# Intensity-normalize variants — `nitrix.numerics.normalize`

> **Status (2026-06-02): not started — CONVENIENCE (close, but the two
> recipes below are not exactly reachable today).** Consumer-pipeline
> substrate for the ilex → thrux migration. Provenance: 2026-06-02 ilex
> vendored-model survey
> ([`ilex-pipeline-substrate.md`](ilex-pipeline-substrate.md), volumetric
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

## Cross-references

- [`ilex-pipeline-substrate.md`](ilex-pipeline-substrate.md) — survey
  context + the volumetric tier.
- `src/nitrix/numerics/normalize.py` — the existing normalisation family.
