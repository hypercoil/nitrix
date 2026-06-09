# Perf: registration recipes have a pathological cold compile (the Python-unrolled optimizer loop)

> **Status (2026-06-09): OPEN — high-impact perf candidate.** Surfaced
> diagnosing a "registration is slow on GPU" report while building the
> `nitrix-perf-bench` recipe case (Phase 4). Not a correctness bug: the steady
> state is fast; the cost is a one-time XLA **cold compile** that scales with
> the unrolled iteration count.

## Observation (L4, 48³ rigid; cold = first call incl. compile, steady = warm)

| recipe / spec | cold (compile) | steady | first-call / steady |
|---|---|---|---|
| `rigid_register` L1 × 5 | 12.6 s | 4.3 ms | ~2900× |
| `rigid_register` L1 × 30 | 38.4 s | 16.8 ms | ~2300× |
| **`rigid_register` L3 × 30 (default)** | **144.6 s** | 37.8 ms | **~3800×** |
| `diffeomorphic_demons_register` L2 × 20 | 46.8 s | 7.0 ms | ~6700× |

The default `rigid_register` **compiles for ~2.4 minutes, then runs in 38 ms**.
That one-time cold compile is the "slow on GPU" — undifferentiated from steady
state it reads as "registration is slow," when the steady state is in fact fast.

## Cause (confirmed in code)

**The optimizer loop is Python-unrolled** — `for _ in range(n_iters):` in
`linalg/optimize.py` (`gauss_newton` / `levenberg_marquardt`) and
`register/diffeomorphic.py` (the Demons iterations + the pyramid levels). So XLA
compiles the *entire* `levels × iters` graph as one expression; compile time
scales ~linearly with the total unrolled iteration count (12.6 → 38.4 → 144.6 s
above), which is the unrolled-loop signature. The unroll is deliberate (the
docstring notes it is "so `jax.grad`" works — differentiating the recipe as a
layer).

**Compounded by the per-iteration internal Jacobian.** Each iteration's
Gauss-Newton/LM step builds the metric Jacobian *matrix-free by autodiff*
(`jax.linearize` + `jax.linear_transpose` through the warp, applied as `JᵀJ`
matvecs inside CG — `optimize.py:84,88`), with **no closed-form affine
Jacobian**. Measured weight: a GN iteration is ~4.5× a bare warp+SSD in steady
state (0.50 ms vs 0.11 ms). Modest in steady state, but each unrolled iteration
carries that whole `linearize` subgraph, so it also **inflates the graph XLA
must compile** — the two effects compound.

## Fix levers

1. **Roll the loop (primary).** `lax.scan` / `fori_loop` for the optimizer
   iterations (and the Demons inner loop) compiles *one* iteration, making
   compile time ~constant in `iters`/`levels` — collapsing the 145 s toward
   seconds. Differentiability is preserved: `lax.scan` is differentiable, and
   the R3 `implicit_least_squares` path already provides an O(1)-memory
   gradient — so the "unrolled so `jax.grad`" rationale is covered without the
   unroll.
2. **Closed-form affine Jacobian (secondary).** For the rigid/affine update,
   the analytic `∂(M(θ)·x)/∂θ` is cheap and closed-form; using it instead of
   `linearize`-through-the-warp would cut the per-iteration steady cost (the
   4.5×) and shrink each iteration's compiled subgraph (helping the compile
   too).

## Scope note

Steady state is fine; this is purely **first-call latency**. It amortises over
repeated registrations of the same shape (one compile, then 38 ms each), but a
single registration pays the full ~145 s. Rolling the loop helps both the
first-call latency and the per-shape recompiles.

## Cross-references

- `linalg/optimize.py` (`gauss_newton` / `levenberg_marquardt`, the `for _ in
  range` loop + `jax.linearize` Jacobian), `register/recipes.py`,
  `register/diffeomorphic.py` (the unrolled Demons + pyramid loops).
- nitrix-perf-bench Phase 4 recipe diagnostic (`reg_diag.py`).
