# FR: TFCE — compute the threshold ladder in fp64 to fix fp32 boundary flips

**Status:** open · **Severity:** medium (fp32 accuracy at brain-scale volumes;
exact in fp64) · **Source:** nitrix-perf-bench `tfce` case.

## Summary

`stats.inference.tfce` loses accuracy in fp32 at larger volumes — vs the exact
fp64 Smith-Nichols integral it reaches `rel_to_tol` **2.7× at 64³ and 8.4× at
96³** (gate rtol=1e-3/atol=1e-4). The cause is the **fp32 threshold ladder**:
the Riemann thresholds `h = i * dh` with `dh = max(stat)/n_steps` accumulate
fp32 rounding, so the supra-threshold comparison `stat > h` flips a few
boundary voxels relative to fp64 — changing connected-component *extents* at the
high thresholds that determine the peak voxels' TFCE. Computing the **ladder in
fp64** (the `hmax`, `dh`, and `h = i*dh` scalars) — while the input data and the
connected-component labelling stay fp32 — fixes it to **rel_to_tol 0.001×**.

## Evidence (L4, stat = smoothed-blob map, n_steps=100, one-sided)

`rel_to_tol = max|tfce − fp64_oracle| / (1e-4 + 1e-3·|oracle|)`:

| volume | nitrix fp32 | numpy SAME algo fp32 | fp64 threshold ladder (data fp32) |
|--------|-------------|----------------------|-----------------------------------|
| 32³    | 0.000       | 0.000                | —                                 |
| 48³    | 0.000       | 0.000                | —                                 |
| 64³    | 2.721       | 2.721                | —                                 |
| 96³    | **8.354**   | **8.354**            | **0.001**                         |

Two controls localize it:

- **Not nitrix-specific / not the accumulation.** An independent numpy
  reimplementation of the *same* algorithm in fp32 reproduces nitrix's error to
  `max|Δ| = 3e-5` (8.354× both). Doing only the *accumulation* of
  `extent^E·h^H·dh` in fp64 does **not** help (still 8.354×).
- **It is the threshold ladder.** Computing just `h = i·dh` (and `hmax`, `dh`)
  in fp64 — so `stat(fp32) > h(fp64)` compares against an accurate threshold —
  drops the error to **0.001×**. The fp32 `i·dh` was producing slightly-off
  thresholds; near the top of the ladder (where a peak's cluster is one or two
  voxels) that flips inclusion and changes the extent.

## Suggested fix

Compute the threshold ladder in fp64 regardless of input dtype:

```python
hmax = jnp.max(pos).astype(jnp.float64)          # (or float of the host)
dh   = hmax / n_steps
heights = jnp.arange(1, n_steps + 1, dtype=jnp.float64) * dh
# compare pos (fp32, promoted) > h (fp64) at each step
```

This is O(n_steps) scalar work (100 values) — negligible cost — and leaves the
expensive per-voxel labelling/integration in fp32. (If x64 is disabled, do the
`i*dh` in a Kahan/compensated form, or hold the ladder as a cumulative sum to
avoid the `i*dh` round-off; the key is that the *thresholds* be accurate, not
the data.) Optional: also accumulate the integral in fp64/compensated — not
required here (the accumulation alone was not the cause), but cheap insurance at
very large `n_steps`.

## Impact

The perf-bench `tfce` case currently gates nitrix against the fp32 reference
(fp32-honest: nitrix matches the numpy fp32 algorithm to 3e-5) and documents
this fp32-ladder limit, pending this fix; with the fp64 ladder it could gate
against the exact fp64 oracle at all volumes. The same ladder pattern appears in
the `permutation_test` TFCE enhancement path.
