# Matrix-recipe (rigid/affine) per-iteration perf levers

> **Status (2026-06-10): logged, measured, not yet implemented.** Surfaced by
> the R8 perf round (`docs/design/registration-suite-v2.md` §6 + §8-R8) on the
> L4. Single-pair rigid/affine is the recipe class nitrix is **most behind on** —
> 235 ms GPU vs 274 ms ANTs-CPU at 128³ is a ~1.16× edge, which is a
> **cost-normalised loss** once the GPU hardware premium (10–100×) is paid (see
> [`../design/registration-suite-v2.md`](../design/registration-suite-v2.md) §6
> "economic verdict"). These levers aim to turn it into a *qualitative* win
> (≥10× vs ANTs-CPU). **A design discussion on model configurability and the
> composition-vs-performance balance is owed before implementing** — these
> levers are the menu that discussion will choose a structure for.

## The diagnosis (where the per-iteration cost goes)

Every forward Gauss-Newton / Levenberg-Marquardt iteration on the `assembled`
path (the rigid/affine default, `linalg/optimize.py`) costs roughly **P + 2
full-field warps**, and **re-linearises the moving image every iteration**:

- `jax.jacfwd(residual_fn)(x)` (`optimize.py:137`) — **P forward-mode tangent
  warps** (P = 6 rigid, 12 affine), materialising the M×P Jacobian;
- the residual warp itself (`:136`);
- LM's trial-cost warp for the accept/reject `jnp.where` (`:261`).

The warp (interpolation gather) is the hot kernel, so per-iteration cost tracks
that warp count.

## The levers

### A — single-pair inverse-compositional (rigid). **Highest impact, lowest risk.**

The Baker-Matthews constant-template-Hessian kernel already exists
(`register/_inverse_compositional.py`) and is exercised by `volreg`; it
linearises the **reference** once per level (steepest-descent images + Gauss-
Newton Hessian `H = SDᵀSD` built once), so each iteration is **one warp + one
(M,P) projection + a closed-form compositional matrix update** — no per-iteration
`jacfwd`. `ic_register_core` already returns a `RegistrationResult`. Wiring it
into `rigid_register` is a dispatch: `IndexSpace + SSD + optimizer∈{auto,lm,gn}
+ Rigid` → IC; everything else → the forward path (unchanged).

**Measured (`bench/ic_vs_forward.py`, L4 f32, single pair, L3×20, identical
recovery NCC):**

| size | forward GN/LM | IC | speedup |
|---|---|---|---|
| 64³ | 24.1 ms | 3.4 ms | **7.1×** |
| 96³ | 66.3 ms | 10.2 ms | **6.5×** |
| 128³ | 134.2 ms | 36.6 ms | **3.7×** |

At 128³ that alone is ~7.5× vs ANTs-CPU (274 ms). Effort **S**, risk **low**
(code exists + tested by `volreg`); ships with an IC-vs-forward parity test.

### A′ — affine inverse-compositional.

Generalise `_steepest_descent` to the 12-DOF affine generators (linear-block
`Eᵢⱼ·(x−c)` columns + translation columns `= ∇F`), swap `_rigid_inverse` for a
general affine inverse (a small `safe_inv` of the (ndim+1) matrix), and use
`affine_exp` for the compositional update `T ← T · exp(Δθ)⁻¹`. Affine has the
largest P, hence the largest forward-path penalty (≈14 warps/iter), so the win
is at least as large as rigid. Effort **M**, risk **low–med**.

### B — single-pair `while_loop` early-exit (iteration-count lever).

Tracked in [`registration-early-stopping-while-loop.md`](resolved/registration-early-stopping-while-loop.md)
(R8-scoped to single-pair matrix recipes). R8 measured rigid converging in
**~5/20 steps even on a hard case**, so ~3–4× of the budget is spin.
**Multiplies** with A/A′ (per-iteration × iteration-count are orthogonal axes).
Single-pair sidesteps the `vmap`-cohort objection that gates this generally;
the implicit backward is already trajectory-independent. Effort **M**, risk
**med** (the `custom_vjp` + convergence-criterion work).

### C — closed-form steepest-descent for the *forward* path.

For the cases IC cannot cover — `WorldSpace` (IC is IndexSpace-only) and any
forced-forward run — replace `jacfwd` (P autodiff JVP warps) with the analytic
`SD[x,j] = ∇warped(x) · ∂W/∂θ_j` (one gradient of the warped image + P
closed-form generator terms), the same construction `_steepest_descent` uses for
IC. ~P× cheaper Jacobian without changing the optimiser. This is the long-
deferred "item C." Effort **M**, risk **low–med**.

### E — per-level iteration schedule (cheap).

The finest level is the costliest per iteration yet converges fastest; today
`RegistrationSpec.iterations` is constant across levels. Making it accept a
per-level tuple (front-load the cheap coarse levels, starve the expensive fine
one) is a few lines and composes with B.

## The compound payoff

A (≈7× small / 3.7× at 128³) **×** B (≈3–4× iteration-count) compounds toward
rigid 128³ ≈ 10 ms — **~25× vs ANTs-CPU 274 ms**, clearing the ≥10×
hardware-premium bar that turns single-pair rigid/affine from "not a win" into a
qualitative one. A′ carries it to affine.

## Secondary finding (a later refinement, not a blocker)

IC's speedup erodes 7.1× → 3.7× from 64³ → 128³ because the per-iteration
`sd.T @ err` re-reads the full M×P steepest-descent array — a bandwidth cost at
scale. A fusion of the SD build + projection (or a compacter SD layout) would
recover the small-size ratio at brain scale.

## The open design question (next loop, before implementation)

How these levers *select* is itself a design problem — IC-vs-forward dispatch,
per-level schedules, and closed-form-vs-autodiff Jacobians are all **model
configurability** choices, and they trade **composition** (one clean driver over
the `CoordinateSpace`/`TransformModel`/`Metric` ADTs) against **performance**
(specialised kernels that bypass the general path). Resolve that balance — what
is configuration vs. a separate constructor vs. an automatic dispatch — before
landing A–E, so the perf wins don't fragment the recipe surface.

## Cross-references

- [`../design/registration-suite-v2.md`](../design/registration-suite-v2.md) §6
  (perf program + economic verdict), §8-R8 (the measurements).
- [`registration-early-stopping-while-loop.md`](resolved/registration-early-stopping-while-loop.md)
  — lever B.
- `src/nitrix/register/_inverse_compositional.py` (the IC kernel — levers A/A′);
  `src/nitrix/linalg/optimize.py` (`gauss_newton`/`levenberg_marquardt`,
  `jacfwd` assembly — levers C/E); `src/nitrix/register/_core.py`
  (`register_core`, `optimize_objective` — the dispatch site); `bench/ic_vs_-
  forward.py` (the A measurement).
