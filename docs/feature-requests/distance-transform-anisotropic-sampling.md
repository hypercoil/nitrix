# B20. `distance_transform` euclidean has no `sampling=` (anisotropic spacing)

> **Status (2026-06-06): open — feature gap from `nitrix-perf-bench`.** Surfaced
> while hardening the euclidean EDT case (B18 Win 1): the tight gate is exact vs
> scipy *with unit spacing*, but nitrix has no way to express anisotropic voxel
> spacing, which real medical volumes have. Authored perf-bench-side; nitrix
> disposes.

## TL;DR

`distance_transform(mask, metric='euclidean')` bakes **unit spacing** into the
squared-distance matrix ``D2[q, p] = (q - p)**2`` (`_edt_along_axis` in
`morphology/_mm.py`). There is no `sampling=` parameter, so it cannot compute
the EDT on an anisotropic grid (e.g. a 1x1x3 mm MRI). scipy's
`distance_transform_edt(input, sampling=(...))` and ITK both support this; nitrix
silently returns the unit-spacing answer, which is **wrong** for anisotropic
data unless the caller pre-rescales (and pre-rescaling the grid is not
equivalent -- it changes the sampling lattice).

## Verified (perf-bench)

On a 32^3 structured blob mask (CPU, fp64):

```
nitrix distance_transform(m)        vs scipy edt (unit)          max|Δ| = 3.1e-08   (exact)
nitrix distance_transform(m)        vs scipy edt(sampling=1,1,3) max|Δ| = 1.24      (the gap)
```

nitrix matches scipy exactly at unit spacing and diverges by >1 voxel once the
reference uses non-unit spacing -- i.e. the op has no anisotropic mode at all,
not merely a precision difference.

## Suggested direction (nitrix disposes)

The separable min-plus-matmul engine makes this **cheap**: per-axis spacing
`s[axis]` scales that axis's squared-distance matrix,
``D2[q, p] = (s[axis] * (q - p))**2``. A `sampling: float | Sequence[float] |
None = None` kwarg (scipy's name/semantics; `None` => unit, the current
behaviour) threaded into `_edt_along_axis` covers it without touching the
streaming kernel. The chamfer engine is a separate question (its step kernel
encodes integer steps); the euclidean engine is the high-value case.

## Perf-bench follow-through

The euclidean `distance_transform` case asserts the unit-spacing contract and
the anisotropic gap explicitly (`test_distance_transform.py::
test_anisotropic_gap_is_visible`), so when a `sampling=` parameter lands, that
test becomes the parity check (nitrix `sampling=(...)` vs scipy `sampling=(...)`)
rather than a gap marker.

## Cross-references

- [`perf-bench-case-hardening.md`](perf-bench-case-hardening.md) (B18, Win 1 --
  the EDT seams; this is the "anisotropic spacing unsupported" item).
- `morphology/_mm.py` (`_edt_along_axis`, `_distance_transform_edt`).
- [`internal-backlog.md`](internal-backlog.md) — ledger index (B20 pointer).
