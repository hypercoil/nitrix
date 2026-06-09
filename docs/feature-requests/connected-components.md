# Connected-components / largest-component labelling — `nitrix.morphology`

> **Status (2026-06-09): SHIPPED.** `morphology/_label.py` adds
> `connected_components(mask, *, connectivity)` (N-D, contiguous `1..K`
> labels, scipy connectivity convention) and `largest_connected_component`.
> Implemented as jit-able label propagation with **pointer jumping**
> (`lax.while_loop`: a neighbour-max hop + a `L = L[L-1]` pointer-jump per
> pass) + contiguous renumber. Pointer jumping doubles the flood's reach
> per pass, so a component of diameter `d` converges in `O(log d)` passes
> (same fixed point as a pure flood), overcoming the original `O(d)` cost
> the review flagged (§5.4). Benched on GPU: a solid 256³ block
> (diameter ~765) labels in ~24 ms (≈10 passes, not 765). Verified against
> `scipy.ndimage.label` (2-D / 3-D, face + full connectivity), a
> high-diameter snake, and the diagonal-merge / empty / jit cases. Consumer-pipeline substrate
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
