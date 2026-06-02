# `crop_to_nonzero` / bounding-box crop — `nitrix.numerics`

> **Status (2026-06-02): not started — ENABLING (nnUNet / template
> pre-step).** Consumer-pipeline substrate for the ilex → thrux migration.
> Provenance: 2026-06-02 ilex vendored-model survey
> ([`ilex-pipeline-substrate.md`](ilex-pipeline-substrate.md), volumetric
> item C).

**What.** The bounding box of the nonzero / above-threshold region. The
output *shape* of a crop is data-dependent, so the **index math** (the bbox)
is the `nitrix` piece — a pure index-computing primitive keeps `nitrix`
jit-clean; the actual slice + affine update is `thrux`.

**Drivers (ilex ports).** `hd_bet`, `fastcsr` (nnUNet `crop_to_nonzero`),
`pglands_seg` (MNI-template crop), `brainiac`.

**API sketch.**

```python
def nonzero_bounding_box(
    x: Float[Array, '*spatial'] | Bool[Array, '*spatial'],
    *,
    threshold: float = 0.0,
) -> Tuple[Int[Array, 'ndim'], Int[Array, 'ndim']]:   # (lo, hi) per axis
    ...
```

**Home.** `nitrix.numerics`.

## Cross-references

- [`ilex-pipeline-substrate.md`](ilex-pipeline-substrate.md) — survey
  context + the volumetric tier; the `conform` scope boundary (the slice +
  affine update belongs to `thrux`).
- [`pad-to-multiple.md`](pad-to-multiple.md) — the sibling pre-step.
