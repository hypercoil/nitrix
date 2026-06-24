# B20. `distance_transform` euclidean has no `sampling=` (anisotropic spacing)

> **Status (2026-06-24): RESOLVED — `sampling=` shipped.**
> `distance_transform(metric='euclidean', sampling=...)` and
> `distance_transform_edt(..., sampling=...)` now take a scalar or per-axis voxel
> spacing (scipy semantics); each axis pass scales its squared-distance matrix
> by that spacing, so the separable composition is the exact anisotropic squared
> distance `Σ_axis (spacing_axis·Δ_axis)²`. Verified exact vs
> `scipy.ndimage.distance_transform_edt(sampling=...)` (2-D + 3-D, fp32) and
> byte-identical to the unit grid when `sampling` is `None`/`1.0`. The chamfer
> engine rejects `sampling=` (it encodes integer steps in its kernel). Surfaced
> for the surface-metrics tier (`classification-surface-metrics.md`), which this
> unblocks.
>
> **FH dispatch evaluated and rejected (same session).** The suggestion below
> notes the euclidean engine is an O(n²) brute-force min-plus matmul; a
> Felzenszwalb–Huttenlocher O(n) per-line scan was prototyped and benchmarked
> (`bench/PERF_EDT_FH.md`). The brute-force matmul **wins at every size on both
> CPU and GPU** (20–100× on GPU; 2.4–3.7× on CPU; no crossover), because FH's
> per-line stack scan is dominated by data-dependent control flow under `vmap`.
> No size dispatch added — the matmul engine carries `sampling=` for free.
>
> _Original (2026-06-06): open — feature gap from `nitrix-perf-bench`. Surfaced
> while hardening the euclidean EDT case (B18 Win 1): the tight gate is exact vs
> scipy with unit spacing, but nitrix had no way to express anisotropic voxel
> spacing, which real medical volumes have. Authored perf-bench-side; nitrix
> disposes._

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
