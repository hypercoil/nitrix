# EDT engine — FH vs brute-force matmul (2026-06-24, L4)

**Question.** `morphology.distance_transform(metric='euclidean')` computes each
1D axis pass as a tropical **min-plus matmul** against the `(n, n)`
squared-distance matrix — exact, dense, control-flow-free, but **O(n²)** work
per line. The Felzenszwalb–Huttenlocher (FH) algorithm computes the same exact
1D EDT in **O(n)** via a per-line parabola-envelope stack scan. Should we add an
FH path and **dispatch on axis size** (brute-force small, FH large), as the
box-sum filter does (`_BOX_SHIFT_MAX_WINDOW`)?

**Method.** A faithful vmappable FH 1D EDT (`bench/perf_edt_fh.py`): pass-1
lower-envelope construction (`lax.fori_loop` over positions, inner `lax.while_loop`
to pop dominated parabolas) + pass-2 envelope walk, `vmap`-ed over lines,
composed separably over axes. Verified **exact** against both scipy
`distance_transform_edt` and the nitrix brute-force (max|Δ| ~5e-8, fp32). Timed
the full 3D EDT on random 50%-density `D³` volumes, jitted, mean of 5 iters
after warmup, on CPU and the L4 GPU.

**Results (full 3D EDT, ms; `FH_speedup = brute / FH`, >1 ⇒ FH wins).**

GPU (L4):

| D | brute_ms | fh_ms | FH_speedup |
|---|---|---|---|
| 32 | 0.19 | 11.69 | 0.02× |
| 64 | 0.28 | 23.58 | 0.01× |
| 128 | 0.52 | 52.99 | 0.01× |
| 192 | 3.38 | 101.62 | 0.03× |
| 256 | 7.38 | 155.00 | 0.05× |

CPU:

| D | brute_ms | fh_ms | FH_speedup |
|---|---|---|---|
| 32 | 2.44 | 6.71 | 0.36× |
| 64 | 17.72 | 65.26 | 0.27× |
| 96 | 67.30 | 222.41 | 0.30× |
| 128 | 267.15 | 639.97 | 0.42× |
| 160 | 683.72 | 1713.11 | 0.40× |

**Reading.** The brute-force matmul wins at **every** tested size on **both**
platforms — by 20–100× on GPU, 2.4–3.7× on CPU. FH's O(n) work never amortizes
against the matmul's O(n²): the per-line stack scan is dominated by data-dependent
control flow (the inner `while_loop` pops, which under `vmap` serialize to the
slowest lane across the batch), whereas the min-plus matmul is exactly the dense,
tensor-core-friendly kernel XLA/the semiring Pallas path is built for. FH does
close the gap *relatively* with D (GPU ratio 0.01→0.05×), but the constant-factor
deficit is so large there is no crossover within any realistic volume size
(`D ≤ 256`; brute-force is still only 7 ms at `D=256` on GPU), and CPU shows no
crossover trend either (`CPU perf is not a goal` regardless — SPEC §2 tenet 3).

**Decision: do NOT add the FH dispatch.** The brute-force min-plus matmul wins
everywhere measured. This vindicates the original design (`_edt_along_axis`
docstring: a "no-control-flow implementation instead of a per-line stack scan").
The euclidean EDT keeps the matmul engine; the value this session ships is the
**anisotropic `sampling=`** parameter (B20), which the matmul engine carries for
free (a per-axis scale on the squared-distance matrix).

**Reproduce.** `JAX_PLATFORMS=cpu python bench/perf_edt_fh.py` and (GPU)
`python bench/perf_edt_fh.py`.
