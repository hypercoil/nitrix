# CubicBSpline prefilter: backend-dependent numerics (parity flag)

> **Status (2026-06-23): open (consumer correctness flag, ilex → nitrix).**
> Verified against `nitrix main@6449cfa`. Not a perf item — sibling to the
> *perf* gap [`interpolation-backend-cpu-gpu-gap`](interpolation-backend-cpu-gpu-gap.md)
> (B15), which is order-1 only and explicitly "numerically correct everywhere."
> This is the **order-3** path and concerns cross-backend *numeric* parity.

**What.** `CubicBSpline`'s B-spline prefilter chooses its scan algorithm by
**backend**: `src/nitrix/geometry/_interpolate.py:601`

```python
associative = default_backend_is_gpu()
...
_bspline_prefilter_1d(row, _BSPLINE_POLE, associative=associative)
```

> **Generalised (2026-06-24):** this is now the motivating instance of a
> codebase-wide design principle — [`reproducible-dispatch`](reproducible-dispatch.md)
> — governing *every* numerically-divergent auto-dispatch (5 sites; this is #5).
> Its P2/P3 deliver the resolution here: an honest docstring, a `driver` field on
> `CubicBSpline`, a tested cross-variant tolerance, and the global `reproducible`
> switch — satisfying all three options below at once.

So the order-3 prefilter runs a **parallel associative scan on GPU** and a
**sequential recurrence on CPU**. Those are mathematically equivalent but
**floating-point-associativity different**, so `spatial_transform` /
`resample` with `method=CubicBSpline()` can produce results that differ
**CPU-vs-GPU beyond ULP** (a long IIR pole filter accumulates the reordering).
By contrast `Linear` / `NearestNeighbour` (order 0/1) route through
`_gather_sample` → `_map_coordinates_sample` with **no** backend branch
(`_interpolate.py:381`), so they are bit-identical across backends — verified
while migrating ilex's `nimox.warp` onto `spatial_transform(Linear())`.

**Why it matters for a consumer.** ilex's forward-parity oracle pins
`NITRIX_BACKEND=jax` (CPU reference) precisely so fixtures are
backend-deterministic. The published model bundles, however, are intended to
run on GPU. For order-1 paths that's safe (bit-identical). But once an ilex
port migrates a **cubic** resample (the nnUNet / scipy-`order=3` preprocessing
path) onto `CubicBSpline` and runs the forward on GPU, its output could drift
from the CPU-generated parity fixture through this prefilter — a
backend-dependent parity break that the two-tier strategy (ilex pins jax;
nitrix golden corpus certifies pallas≈jax) only catches if nitrix's own
corpus exercises the prefilter cross-backend within a pinned tolerance.

**When it bites.** Not today — ilex's first warp migration is order-1 only and
unaffected. It bites the first cubic-resample migration that runs cross-backend
(GPU forward vs CPU fixture). Filing now so it's known before that migration,
not discovered as a mysterious GPU-only parity failure.

**Asked of nitrix (pick one).**
1. Make the prefilter scan order **backend-independent for the reference** —
   e.g. always sequential on the `jax` backend (already the case) *and* assert
   the GPU associative path stays within the golden-corpus tolerance of it, so
   `CubicBSpline` carries the same `pallas ≈ jax` guarantee `Linear` does; or
2. expose the prefilter scan choice (or a `backend=`/`deterministic=` flag) so a
   parity-sensitive consumer can force the reference path on GPU; or
3. if (1) already holds in the golden corpus, just **document** the
   cross-backend tolerance for `CubicBSpline` in the `_interpolate` docstring
   (the order-0/1 "parity-equal to a ULP on both platforms" note does **not**
   extend to order-3, and currently reads as if it might).

**Effort.** S to confirm whether the golden corpus already pins
`CubicBSpline` pallas-vs-jax (likely the quickest resolution — may be a doc
fix); M if a deterministic-scan option is wanted.

## Cross-references

- [`interpolation-backend-cpu-gpu-gap`](interpolation-backend-cpu-gpu-gap.md)
  (B15) — the order-1 *perf* sibling (this is the order-3 *parity* sibling).
- [`cubic-resample`](resolved/cubic-resample.md) — where `CubicBSpline` (scipy order-3
  parity) was added.
- `src/nitrix/geometry/_interpolate.py:601` (`associative =
  default_backend_is_gpu()`), `:381` (`_gather_sample` → `_map_coordinates_sample`,
  the order-0/1 no-branch path).
