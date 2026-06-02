# Sliding-window weighting kernel + overlap-add stitch — `nitrix.numerics`

> **Status (2026-06-02): not started — CONVENIENCE (borderline; the
> orchestration is out of scope, only the two numeric pieces are
> `nitrix`-shaped).** Consumer-pipeline substrate for the ilex → thrux
> migration. Provenance: 2026-06-02 ilex vendored-model survey
> ([`ilex-pipeline-substrate.md`](ilex-pipeline-substrate.md), volumetric
> item F).

**What.** `hd_bet`, `fastcsr`, `wholebrain_unest`, `brats_segresnet`,
`synthstrip`, `bme_x` all tile a large volume and blend overlapping patch
logits with a Gaussian-weighted window. The *orchestration* (tile
scheduling, model dispatch) belongs in `nimox.inference` / `thrux`, but two
numeric pieces are `nitrix`-shaped and reused by every consumer:

1. the separable **Gaussian patch-weight window**, and
2. the **weighted overlap-add** accumulation
   `out += w * patch; norm += w` then `out /= norm`.

Ship those as kernels; leave the loop to the orchestrator.

**Home.** `nitrix.numerics` (window) + the reduction; orchestration stays
out.

## Cross-references

- [`ilex-pipeline-substrate.md`](ilex-pipeline-substrate.md) — survey
  context; the sliding-window scope boundary (tiling/scheduling →
  `nimox.inference` / `thrux`).
- [`compensated-summation.md`](compensated-summation.md) (§12.10) — the
  overlap-add accumulator is a natural drop-in site for a stable reduction.
