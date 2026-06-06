# B17. `jnp.median` / `jnp.percentile` CPU sort cliff (robust-z / intensity norm)

> **Status (2026-06-04): open (perf characterisation).** Measured, not a
> commitment -- gated on the **Trigger** below. Provenance: surfaced building
> the `nitrix-perf-bench` numerics-normalize family; ledger context in
> [`internal-backlog.md`](internal-backlog.md). Sibling of
> [`interpolation-backend-cpu-gpu-gap`](interpolation-backend-cpu-gpu-gap.md)
> (B15) -- same shape (a jax CPU primitive losing to the optimised library),
> different primitive (sort/median/percentile).

`robust_zscore_normalize` (median + MAD) and `intensity_normalize` (percentile
clip) are **catastrophically slow on the CPU backend** -- both route through
`jnp.median` / `jnp.percentile`, which lower to a full **sort**.

**Measured (L4, perf-bench, fp32):**

| op | n=2048 (jax CPU) | n=4096 (jax CPU) | numpy CPU | numpy speedup |
|---|---|---|---|---|
| `robust_zscore_normalize` | 1.18 s | 5.25 s | 0.10 / 0.62 s | **8-11×** |
| `intensity_normalize` | 1.18 s | 5.15 s | 0.095 / 0.40 s | **12×** |

Seconds, in absolute terms, for a normalisation op. On the **GPU** backend
nitrix is fine (robust beats the cupy reimpl 9-13×; intensity ~parity), so this
is a CPU-only cliff. `zscore_normalize` / `psc_normalize` (mean/std, no sort)
are clean wins -- the cliff is specifically the order-statistic ops.

**Measured 2026-06-05 (negative result — the "double sort" is a red herring).**
`intensity_normalize` / `percentile_rescale` call `jnp.percentile` *twice*
(p_lo and p_hi), which *looks* like two sorts to remove by batching into one
length-2 quantile call. It is not: **XLA already CSEs the two identical
`lax.sort` subexpressions into one** (verified by HLO dump — `old` two-call and
`new` one-call forms compile to the identical sort count; GPU steady **1.41 ms**
and peak **184.55 MB HBM** are unchanged to the decimal). So the GPU `intensity`
gap (≈1.6× vs cupy at n=2048, parity-to-faster by n=4096) is the cost of the
**single** full `lax.sort` (4M elements → ~184 MB, ~11× cupy's 16.8 MB), not a
double sort. The single-call refactor was measured perf-neutral and reverted.
The only real levers remain those listed below (pure_callback / accept /
upstream quickselect) — there is no cheap pure-JAX win here, consistent with the
`top_k`-doesn't-help-the-tail finding above.

**The obvious fix does NOT work in jax.** The natural idea -- "use a partition
(O(n) introselect) instead of a full sort" -- fails *inside* jax, because
`jnp.partition` is itself sort-based on CPU (XLA has no O(n) selection
primitive). Measured for a (2048, 2048) row-reduction on this box:

| call | CPU time | note |
|---|---|---|
| `np.median(axis=-1)` | 55 ms | introselect partition |
| `np.partition(axis=-1)` | **8 ms** | true O(n) introselect |
| `jnp.median(axis=-1)` | 609 ms | == sort |
| `jnp.sort(axis=-1)` | 607 ms | the lowering |
| `jnp.partition(axis=-1)` | 550 ms | **also sort-based** -- no win |

So `jnp.median → jnp.partition` buys ~10% (550 vs 609 ms), nowhere near numpy's
8 ms. The gap is two-fold: XLA's CPU sort is ~10× slower than numpy's sort, and
it is `O(n log n)` where numpy's `np.partition` is `O(n)` (~70× faster than
even XLA's sort).

**`jax.lax.top_k` (partial selection) is a partial win at best, not a fix.**
`top_k` *is* faster than a full sort on CPU when `k` is small, so it was worth
testing. Measured:
- *Row median* (k = n/2, the (2048, 2048) row-reduction): `lax.top_k(n//2)`
  alone is 264 ms (vs 595 ms sort, 2.25×), but a *correct* top_k median (the two
  middle order statistics, exact vs `np.median`) lands ~510 ms -- only ~1.2×
  over `jnp.median`, because at `k ≈ n/2` the partial-selection advantage is
  small. Still ~10× off numpy's 51 ms.
- *Whole-tensor percentile* (intensity_normalize, N = 16.7 M, the p1/p99 tails
  at k ≈ 0.01 N ≈ 167 k): `top_k` is **slower** -- 8.96 s vs `jnp.percentile`'s
  5.31 s -- because `k` is still large in absolute terms and the two tails
  double the work. (numpy: 0.36 s.)

So `top_k` modestly helps the per-row median and *hurts* the large-N tail
percentile; neither closes the ~10-14× gap. It is not the lever.

**Actionable paths (none is a one-liner):**
1. **scipy-backed CPU path** for the order-statistic reduction -- a
   `jax.pure_callback` to `numpy.median` / `numpy.percentile` on the CPU
   backend (cheap there: no host transfer). Differentiability via a
   `custom_vjp` (the median's gradient is a selection mask; the percentile's is
   the interpolated index pair). This is exactly the B16 mechanism applied to a
   different primitive -- cross-ref
   [`alternative-interp-backends-xla`](alternative-interp-backends-xla.md).
   Keep `jnp.median` on GPU (it's fine there); branch on backend.
2. **Upstream**: an XLA quickselect / `top_k`-style partial-sort lowering would
   fix this library-wide (jnp.median, jnp.percentile, jnp.partition all benefit)
   -- a jax/XLA feature ask, not a nitrix change.
3. **Accept on CPU** (status quo) -- fine if the order-statistic normalisers are
   only ever run on GPU.

**Trigger.** A consumer running `robust_zscore_normalize` / `intensity_normalize`
(or any `jnp.median`/`jnp.percentile` op -- this is library-wide) CPU-side at
scale; the seconds are a real wall-clock cost in preprocessing.

**Effort.** M (the pure_callback + custom_vjp CPU path, per op or factored into
a shared median/percentile helper).

## Cross-references

- [`interpolation-backend-cpu-gpu-gap`](interpolation-backend-cpu-gpu-gap.md)
  (B15) + [`alternative-interp-backends-xla`](alternative-interp-backends-xla.md)
  (B16) -- the sibling jax-CPU-primitive finding and its backend-swap research
  (same `pure_callback` + `custom_vjp` mechanism).
- `src/nitrix/numerics/normalize.py` (`robust_zscore_normalize`,
  `intensity_normalize`, `_median_mad`); nitrix-perf-bench
  `robust_zscore_normalize` / `intensity_normalize` cases.
