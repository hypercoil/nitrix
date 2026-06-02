# `pad_to_multiple` / `crop_to_multiple` (+ unpad) — `nitrix.numerics`

> **Status (2026-06-02): not started — ENABLING, near-universal volumetric
> pre-step.** Consumer-pipeline substrate for the ilex → thrux migration.
> Provenance: 2026-06-02 ilex vendored-model survey
> ([`ilex-pipeline-substrate.md`](ilex-pipeline-substrate.md), volumetric
> item B).

**What.** Pad a volume so each spatial axis is a multiple of the net pooling
factor (and record the crop-back slice). Almost every volumetric net
requires this; each port currently inlines an ad-hoc pad. The *array* pad +
the recorded crop-back slice are pure-`nitrix`; the affine-origin update is
`thrux`'s half.

**Drivers (ilex ports).** `synthstrip` ×64; `synthseg` / `wmh_synthseg` /
`brain_ldm(_vae)` ×16/×8; `synthdist` ×32; `bme_x` ×8.

**API sketch.**

```python
def pad_to_multiple(
    x: Float[Array, '*spatial c'],
    multiple: int | Sequence[int],
    *,
    spatial_rank: int,
    mode: BoundaryMode = 'constant',
    cval: float = 0.0,
) -> Tuple[Float[Array, '*padded c'], Tuple[Tuple[int, int], ...]]:
    '''Returns the padded array and the per-axis (lo, hi) pad widths so the
    caller (or thrux) can unpad / update the affine.'''
    ...
```

**Home.** `nitrix.numerics` (sibling to `tensor_ops`).

## Cross-references

- [`ilex-pipeline-substrate.md`](ilex-pipeline-substrate.md) — survey
  context + the volumetric tier.
- [`crop-to-nonzero.md`](crop-to-nonzero.md) — the sibling index-math
  pre-step (data-dependent bbox crop).
