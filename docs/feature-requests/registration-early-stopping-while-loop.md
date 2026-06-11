# Registration optimisers: try a `while_loop` early-exit forward (implicit backward already supports it)

> **Status (2026-06-11): GATE PASSED → SHIPPED electively for single-pair IC
> (V4e `a9cc855`).** The gate probe (`bench/early_exit_probe.py`, single-pair
> rigid IC, 96²) showed a **clean win on the hard case**: 28°/14-vox warp
> converges in 18 iters vs 90 fixed, **1.69× faster with identical recovery**
> (ncc 0.9924 == 0.9924); easy 10 iters, 2.09×. Shipped as opt-in
> `RegistrationSpec.convergence = Convergence(threshold, window)` (ANTs-style
> windowed slope criterion) on the single-pair inverse-compositional path
> (rigid + affine); the fixed `lax.scan` stays the default (reproducible +
> `vmap`-batchable for the cohort/volreg path). Not reverse-differentiable
> (use the implicit path for a differentiable layer). The clamp-vs-scale
> revisit below is **not** triggered: `while_loop` landed for the *matrix* IC
> path, not greedy SyN (which uses its full budget, so early-exit is moot) —
> the R7 clamp decision stands.
>
> *(Original gate, for the record:)* Surfaced by `nitrix-perf-bench` while
> benching the registration recipes across scale on an L4. **Accept only if it
> is a clean win on hard, brain-realistic cases** — not the easy toy warp
> (where early-exit trivially helps but is unrepresentative).
>
> **R8 convergence measurement (`bench/conv_trajectory.py`, 96³ f32) resolves
> *which* recipes this can help:**
> - **Greedy SyN — no benefit.** It uses its *full* iteration budget (finest
>   level reaches 95 % only at step 36/40; cost still descending at budget end) —
>   the gradual first-order descent under the **non-vanishing LNCC force** has no
>   early plateau to exploit (the same property that makes the R7 clamp correct;
>   cf. `_svf.py::_normalise_step`). So the clamp-vs-scale revisit this doc
>   contemplated is **not triggered** — the fixed scan stays for SyN.
> - **Cohort volreg — contraindicated.** It is a `vmap` over frames; `while_loop`
>   with heterogeneous per-frame trip counts runs to the max (or needs masking),
>   so the saving evaporates — and the batched-GPU throughput is itself the
>   structural win, not to be sacrificed.
> - **Single-pair rigid/affine — the live target.** Second-order GN/LM converges
>   in **~5/20 steps even on a hard case** (NCC 0.625→0.996), so ~75 % of the
>   finest-level budget is spin. This is *also the recipe class nitrix is most
>   behind on* (single-pair latency = a cost-normalised loss vs ANTs-CPU), so a
>   `while_loop` early-exit is both viable (no `vmap` to break in the single-pair
>   path) and economically pointed. **This is the case to prototype.**

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

**Downstream coupling (greedy SyN force normalisation).** If this lands, it
re-opens a coupled decision in `register/_svf.py::_normalise_step`: the greedy-SyN
LNCC force is normalised by a trust-region **clamp** rather than ANTS-style
scale-to-step *because* the forward is a fixed-length `scan` with no convergence
gate (the clamp stops a constant-magnitude step from dithering forever). A
convergence-gated `while_loop` bounds that dithering, making scale-to-step viable
again — so adopting `while_loop` should trigger a clamp-vs-scale revisit there
(the LNCC force does not vanish at the optimum, so the normalisation genuinely
shapes the trajectory). Noted in that function's docstring.

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
