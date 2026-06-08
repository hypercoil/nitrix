# B23. A perf win must certify at brain scale (not just the benched size)

> **Status (2026-06-08): open — benchmark-integrity principle from
> `nitrix-perf-bench`.** The size-axis sibling of [`perf-bench-case-hardening.md`](
> perf-bench-case-hardening.md) (B18): B18 hardened *which branch / what accuracy*
> a number reflects; this adds *at what scale*.  The discipline + tooling live in
> perf-bench; this is the nitrix-facing pointer so the optimisation loop honours
> it.  Authored perf-bench-side (COVERAGE_MANDATE §2.6); nitrix disposes.

## TL;DR

Several nitrix wins are **constant-factor GPU wins on an algorithm with worse
asymptotic FLOP or memory growth** — they win at the small size the bench
happened to measure and *invert, or OOM,* before the size practitioners run (a
256³ MRI volume, a subject cohort).  Optimising for the benched size while the
realistic-scale behaviour is worse is **scale-gaming**, and a perf-bench number
at one small size does not certify a win.

A win on a scale-sensitive op is only real if it is measured **across the size
curve, up to brain scale, with the cost law stated** — so the crossover is
predictable from the algorithm, not hidden behind one number.

## Why this is real (measured)

- **EDT** (`morphology.distance_transform`): the separable min-plus matmul is
  O(n^(d+1)) per axis and materialises O(n^d) buffers; scipy/cupy
  Felzenszwalb–Huttenlocher EDT is O(n^d) and in-place.  On the L4 nitrix is
  *competitive in speed* but a **5–2051× HBM multiplier** → ~5× less OOM
  headroom; the **batched cohort** regime is where it bites.  (Working
  hypothesis for the small-scale win: a low-depth, high-FLOP brute force beats a
  deeper low-FLOP sequential scan while GPU wall-clock is depth-bound — and
  loses once flop/HBM-bound.  The substrate is differentiable, but that is a
  *bonus*, not why it was chosen.)
- **Morphology disk/ball** ([`morphology-explicit-se-im2col-cost.md`](
  morphology-explicit-se-im2col-cost.md), B21): im2col is O(N·k^d) in time *and*
  HBM — 336 MB at a single 64³ ball; a 256³ ball is an outright OOM.
- **Spectral eigensolver**: the dense implicit-VJP backward is O(n²); at a
  vertex-wise graph (n~100k) the dense path cannot exist, only the sparse one —
  whose at-scale behaviour is the regime that motivates the op.

## What perf-bench now does (so a win is certifiable)

`nitrix-perf-bench` hardened the size axis (COVERAGE_MANDATE §2.6, the
distance_transform template):

- a **size tier** on each scale-sensitive case (brain-scale single **and batched
  cohort** sizes), distinct from the small dev points;
- a stated **cost law** per op (time + HBM asymptotic, nitrix vs the reference);
- `tools/scaling_report.py` → the speed crossover, the HBM multiplier, a
  **projected-OOM headroom** vs the baseline, and OOM-as-signal.

## What this asks of the nitrix optimisation loop

Mirrors the *ships-with-a-case* SLA, extended to scale (perf-bench
`OPTIMIZATION_LOOP.md` §2b):

1. **Certify at scale.** For a scale-sensitive op, validate the win across the
   size curve up to brain scale (single + batched), not just the rep point.
   Report the crossover size and the HBM slope alongside the headline ratio.
2. **State the cost law.** A one-line time + HBM asymptotic (vs the reference),
   so the crossover is predictable from the algorithm.  If a small-scale win
   will not generalise, say *why* (e.g. "depth-bound brute force — wins small,
   flop/HBM-bound at scale").
3. **A crossover / OOM before brain scale is a documented limitation, not a
   win.** Surface it (the size-axis analogue of an approximate-but-fast
   baseline: the tradeoff is the signal); do not optimise the small-size number
   in isolation.

This does **not** demand every op scale perfectly — a differentiable,
small-scale-competitive op (the EDT) is a legitimate, useful win.  It demands
the *limitation be measured and stated*, so a consumer (and the next
optimisation pass) chooses with eyes open: nitrix for moderate-scale +
differentiable; the in-place baseline for the at-scale, non-differentiable path.

## Cross-references

- [`perf-bench-case-hardening.md`](perf-bench-case-hardening.md) (B18 — the
  branch/accuracy axes this completes on the size axis).
- [`morphology-explicit-se-im2col-cost.md`](morphology-explicit-se-im2col-cost.md)
  (B21 — the morphology OOM exemplar), [`spectral-embedding-gpu-solver.md`](
  spectral-embedding-gpu-solver.md) (B14 — the sparse eigensolver scale regime).
- perf-bench: `COVERAGE_MANDATE.md` §2.6 + §7-D (ships-with-a-scalability-case),
  `OPTIMIZATION_LOOP.md` §2b, `tools/scaling_report.py`, `reports/SCALING.md`,
  `Case.large_param_points` / `Case.complexity`.
- [`internal-backlog.md`](internal-backlog.md) — ledger index (B23 pointer).
