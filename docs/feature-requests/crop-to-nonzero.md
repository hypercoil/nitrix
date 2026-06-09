# `crop_to_nonzero` / bounding-box crop — `nitrix.numerics`

> **Status (2026-06-09): SHIPPED.** `numerics/spatial.py` adds
> `nonzero_bounding_box(x, *, threshold) -> (lo, hi)` index arrays (the
> half-open box of the above-threshold region; empty → all-zero box). Only
> the **index math** is in nitrix (jit-clean — the box, not a
> data-dependent-shape crop); the slice + affine update stay in `thrux`.
> Verified against `np.argwhere` (2-D / 3-D). Consumer-pipeline substrate
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
