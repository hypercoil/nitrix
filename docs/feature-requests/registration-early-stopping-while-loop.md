# Registration optimisers: try a `while_loop` early-exit forward (implicit backward already supports it)

> **Status (2026-06-10): proposed, acceptance-gated.** Surfaced by
> `nitrix-perf-bench` while benching the registration recipes across scale on an
> L4. **Accept only if it is a clean win on hard, brain-realistic cases** — not
> the easy toy warp (where early-exit trivially helps but is unrepresentative).

## The finding

The registration recipes minimise over a **fixed** step count: every optimiser
is `jax.lax.scan(step, …, length=n_iters)` (`linalg/optimize.py` —
`gauss_newton`, `levenberg_marquardt`; `register/diffeomorphic.py` —
`_demons_level`), with **no convergence-gated early exit** (LM's accept/reject is
a `jnp.where`, but it still runs all `n_iters`). The loop-roll to `lax.scan` was
the right call for cold-compile (it is now ~flat in `n_iters`) — this is a
*separate* axis.

The cost: on an easy problem the solver converges long before the budget and
then spins. Measured (perf-bench planted rigid warp, `levels=2, iters=20`, L4):
the finest level's cost reaches **99 % of its improvement by iteration 2 of 20**
and is then **flat for the remaining ~18 iterations** (cost `6324.016`,
unchanged iter 3→20). nitrix is, correctly, **difficulty-independent** (easy
221 ms vs hard 211 ms CPU rigid — it always runs the full count); ANTs and dipy
use convergence-gated termination, so on easy problems they do fewer *effective*
iterations.

**Why this matters for "can we beat ANTs by optimising the loop?":** a faster
per-iteration kernel (e.g. the proposed Pallas ESM-force kernel,
[`pallas-demons-esm-force.md`](pallas-demons-esm-force.md)) attacks
*per-iteration cost*; it **cannot** recover an *iteration-count* deficit. If a
tool wins partly by stopping early, no amount of kernel tuning closes that part
of the gap — only early stopping does. The two levers are orthogonal.

## Why nitrix is well-placed to try it

The backward over these loops is **implicit** (`implicit_least_squares` /
`implicit_minimize`, the IFT adjoint). The adjoint is solved at the converged
`x*` and needs **only `x*` + the residual/objective**, *not* the forward
trajectory or its length. So the forward iteration scheme is free to change
without touching the backward: a **`lax.while_loop`** (convergence-gated:
cost-plateau, ‖∇‖, or ‖δ‖ below tol) can be wrapped in the *same* `custom_vjp`,
with the existing implicit backward unchanged. (A fixed-length `scan` with a
masked no-op after convergence preserves differentiability but saves **no**
wall-clock — it still executes every step; `while_loop` is the only variant that
actually skips work.)

## The proposal

Prototype a `while_loop` early-exit forward for the GN/LM solver (and the Demons
iteration) under the existing implicit-diff wrapper, with a documented
convergence criterion + a hard `max_iters` cap.

## The acceptance gate (important)

Accept **only on a clean win on hard, brain-realistic cases** — large
deformations / real anatomy at brain scale, where the solver may legitimately
need most of its budget and early-exit may buy little. The easy planted warp
will *always* favour early-exit, so it must **not** be the deciding benchmark.
Weigh against `while_loop`'s real costs:

- it **breaks `vmap`-batching** over a cohort of problems with heterogeneous
  trip counts (the batch runs to the max, or needs per-element masking — much of
  the early-exit saving evaporates under batched/cohort registration);
- **data-dependent, less reproducible wall-clock** (the trip count varies with
  the input), which complicates benchmarking and CI timing;
- it loses the **fixed-shape `cost_history`** the recipes return;
- possible XLA recompile / lowering wrinkles vs the settled `scan` path.

If hard-case + cohort benchmarks don't show a clean net win, the fixed `scan`
stays — the simplicity and batch-friendliness are worth more than speculative
easy-case savings.

## How perf-bench would measure it

The registration scale tier (`reports/REGISTRATION_SCALING.md`) plus an
**iso-accuracy** comparison (wall-clock to reach a target recovery NCC, which
disentangles per-iteration cost from iteration count) — the `while_loop` variant
slots in as a baseline against the `scan` path, on both easy and
large-deformation pairs, single and batched.

## Cross-references

- [`pallas-demons-esm-force.md`](pallas-demons-esm-force.md) — the *orthogonal*
  per-iteration lever (kernel cost, not iteration count).
- [`registration-recipe-cold-compile.md`](registration-recipe-cold-compile.md)
  — the loop-roll finding; `scan`-vs-`while_loop` is a different axis from it.
- `src/nitrix/linalg/optimize.py` (`gauss_newton`, `levenberg_marquardt`,
  `implicit_least_squares`, `implicit_minimize`); `src/nitrix/register/
  diffeomorphic.py` (`_demons_level`).
