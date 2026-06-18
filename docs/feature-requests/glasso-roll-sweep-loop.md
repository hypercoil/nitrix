# Graphical-LASSO — roll the sweep loop (GPU-compile-hostile as shipped)

> **Status (2026-06-18): perf finding + fix request.** `nitrix.stats.glasso`
> / `glasso_path` UNROLL their `n_outer × n_inner` block-coordinate-descent
> sweeps into a single XLA graph. On the GPU this makes **compile infeasible**
> (>5 min at p=32, timing out at every size the perf-bench tried), while the
> CPU path runs fine. The ask is the same one that fixed the registration
> recipes: **roll the sweep loop with `lax.scan` / `lax.fori_loop`** so compile
> is flat in the iteration counts. Surfaced by the nitrix-perf-bench
> `glasso` / `glasso_path` cases (stats-v2 breadth).

## 1. The measurement (nitrix-perf-bench, L4)

`glasso(S, lam=0.1)` at the defaults (`n_outer=100, n_inner=50` = 5000 inner
sweeps), well-conditioned sparse-precision `S[p, p]`:

| p | nitrix **CPU** steady | nitrix **GPU** | sklearn `graphical_lasso` (CPU, fp64) |
|---|---|---|---|
| 40 | 0.29 s | **73.8 s** steady (+ a long compile) | < 0.05 s (early-exits on the duality gap) |
| 80 | 0.98 s | **timeout** (> 600 s) | < 0.1 s |
| 160 | 12 s | timeout | — |
| 320 | 60 s | timeout | — |

A direct probe — `jax.jit(lambda s: glasso(s, 0.1))` on a 32×32 `S` — did **not
finish compiling on the GPU within 5 minutes** (killed). So the 73.8 s at p=40
is not a steady-state cost story: it is a giant-graph compile + a giant-graph
execution. The op is effectively **unusable on the GPU**, which is exactly the
platform nitrix exists to win on.

## 2. The cause (same shape as the pre-v4 registration recipes)

`glasso` runs its block-coordinate descent as a **Python-unrolled** double loop
(`for _ in range(n_outer): for _ in range(n_inner): …`), so XLA traces the full
`5000 ×` graph. Compile time then scales with the *unrolled* iteration count
(and the graph is enormous), the precise pathology that
`perf/registration-roll-loops` (commit `ddc2e10`) fixed for the optimiser /
Demons / SVF loops — there, rolling the loops to `lax.scan` **collapsed compile
10–20× and made it flat in iterations** (`L2x20` == `L2x40`). `glasso` and
`glasso_path` were not part of that pass and still unroll.

(`glasso_path` is worse: it is `L` warm-started `glasso` solves, so the
unrolled graph is `L ×` larger again.)

## 3. The fix

Roll the BCD sweeps with `lax.fori_loop` / `lax.scan` (carry = the current
precision / its Cholesky factor), exactly as the registration recipes now do.
This should:

- make GPU **compile flat** in `n_outer` / `n_inner` (seconds, not unbounded);
- let the op actually run on the GPU at p ≥ 80;
- preserve the result (the sweep recurrence is unchanged — only its trace
  representation changes), so the existing fidelity vs `sklearn` (≈ 4e-4 on the
  precision at p=40, same support) is untouched.

**Optional, separate (do not block the roll):** an ANTs/sklearn-style
early-exit on the duality gap (a `lax.while_loop`, single-problem only — the
same trade-off documented for the registration `convergence` knob). The
fixed-sweep-vs-early-stop confound is why nitrix runs all 5000 sweeps while
`sklearn` stops in tens of iterations; that is a wall-clock read, secondary to
the compile blow-up above.

## 4. Why it matters

`glasso` / `glasso_path` are the sparse-functional-connectivity workhorse
(connectome precision estimation, the GGM model-selection path with
`ebic_score`). As shipped, a user who puts them on the GPU pays an unbounded
compile and then a graph that is **~250× slower than the CPU** (p=40: 73.8 s
GPU vs 0.29 s CPU) — the opposite of the expected GPU win. Until the loop is
rolled the perf-bench measures these CPU-only (nitrix-CPU vs `sklearn`-CPU),
with the GPU column documented as the finding.

## References

- `perf/registration-roll-loops` / commit `ddc2e10` — the loop-roll precedent
  (optimiser / Demons / SVF), the model fix.
- nitrix-perf-bench `cases/glasso.py`, `cases/glasso_path.py` — the cases that
  surfaced this (complexity notes carry the same finding).
