# Connected-components / largest-component labelling — `nitrix.morphology`

> **Status (2026-06-02): not started — ENABLING, highest recurrence across
> the ilex volumetric ports.** Consumer-pipeline substrate for the
> ilex → thrux migration. Provenance: 2026-06-02 ilex vendored-model survey
> ([`ilex-pipeline-substrate.md`](ilex-pipeline-substrate.md), volumetric
> item A).

**What.** N-D connected-components labelling (+ a thin largest-CC helper).
`nitrix.morphology` ships `dilate`/`erode`/`open`/`close`/`distance_transform`/
`median_filter` but **no connected-components** (verified: nothing matches
`connected_comp|label_components` in `src/nitrix/`). It is the single most
common omitted *post*-processing step.

**Drivers (ilex ports).**

- `exvivo_strip`, `synthstrip` — largest-CC to clean the brain mask after
  the SDT/SDF→mask threshold.
- `synthseg`, `wmh_synthseg`, `supersynth`, `hd_bet` — morphological cleanup
  / hole-fill / largest-CC on the label map.

**API sketch** (N-D, channel-free, label image out; plus a largest-CC helper):

```python
def connected_components(
    mask: Bool[Array, '*spatial'],
    *,
    connectivity: int = 1,            # 1 = faces; ndim = full (incl. diagonals)
) -> Int[Array, '*spatial']:          # 0 = background, 1..K = component ids
    ...

def largest_connected_component(
    mask: Bool[Array, '*spatial'], *, connectivity: int = 1,
) -> Bool[Array, '*spatial']:
    ...
```

**XLA note.** Static-shape label propagation (iterated `max`-relabel to
fixed point via `lax.while_loop`, or Playne–Equivalence) keeps it jit-able;
the component *count* stays data-independent (a fixed label-image, not a
ragged list).

**Home.** `nitrix.morphology`.

## Cross-references

- [`ilex-pipeline-substrate.md`](ilex-pipeline-substrate.md) — the survey
  context, scope boundary, and the rest of the volumetric tier.
- `src/nitrix/morphology/_mm.py` — the existing morphology ops it joins.
