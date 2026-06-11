# Registration suite v3 — metric/instrument decoupling, transform algebra, ANTs-parity SyN

> **Status (2026-06-10): scoped + locked, plan only (no code).** The third
> registration block, on top of the shipped v2 suite (volreg, BBR, greedy SyN,
> the physical-space `CoordinateSpace` foundation; merged to `main` `60a0d70`).
> Branch `registration-suite-v3` off `main`. This document is the durable plan;
> per-phase landings update it in place (the v2 precedent).

Reads on top of [`registration-suite-v2.md`](registration-suite-v2.md) (R4–R8)
and [`registration.md`](registration.md) (the R0–R3 core).

## 0. Confirmed decisions (2026-06-10, user)

1. **Keystone abstraction first.** The metric↔instrument decoupling (the `Force`
   protocol + `MetricForce` escape hatch) lands before the matrix perf levers,
   so those levers arrive in their final structural home rather than as ad-hoc
   specialisations.
2. **Multimodal-deformable as a capability** (`MetricForce(MI/NCC)` driving the
   diffeomorphic recipes), **plus a transform-algebra + batched-application
   pillar** — Lie-group composition / **fusion** / **Fréchet (Karcher) mean** of
   transforms, batched — the substrate groupwise/template construction and
   multi-stage fusion need. Full atlas/template *data structures* stay out of
   scope (→ `thrux`; the SPEC §1 non-goal).
3. **Greedy-SyN to ANTs parity first; LDDMM is an end-of-round decision.** Most
   ANTs "SyN" users run greedy symmetric diffeomorphic registration — which v2
   already ships — so feature-completeness (multi-metric, affine-init, masks,
   multi-stage) is the near-term headline. Whether to *also* build geodesic
   LDDMM (the §12.11 ODE-adjoint + time-varying velocity) is decided at the end
   of the round, with its realistic added value weighed then (§8).

## 1. The diagnosis: the metric↔instrument gap is real and asymmetric

The v2 work surfaced that our registration *instruments* are, in many cases,
algorithmically welded to a *single* metric/force — and that welding is the
source of the lingering isotropic-sampling / matching-grid assumptions (a
generic driver would have forced those to be parameters). Grounded in the code:

- **Matrix family (rigid/affine) is already metric-generic.**
  `_core.optimize_objective` (`_core.py:167-199`) dispatches purely on
  `Metric.is_least_squares` — SSD → `gauss_newton`/`levenberg_marquardt`,
  LNCC/MI/CR → BFGS — and the driver knows nothing about which metric is bound
  (`MetricObjective`, `_core.py:260`). Anisotropy / different grids are handled
  by the `CoordinateSpace` ADT (`_space.py`, IndexSpace/WorldSpace). **This is
  the pattern to replicate.**
- **Dense-field family (Demons/SyN) is metric-welded.** The driving force is
  inlined and hardcoded: the ESM/SSD force `u = (diff/denom)·j`
  (`diffeomorphic.py:189`) and the LNCC force `lncc_grad(a,b)·∇a`
  (`_syn.py:221`). `Metric` exposes only `cost`/`residual` (`_metric.py:56-83`)
  — **no force/gradient method** — and **no `Force` abstraction exists**.
- **Matching grids are mandatory in the dense-field family.** Both recipes
  enforce `moving.shape == fixed.shape` (`diffeomorphic.py:231`, `_syn.py:262`),
  put the field on the fixed grid from identity, have **no affine-init
  composition**, and carry only relative `spacing` (anisotropy ratio,
  `_svf._relative_spacing`), not a `CoordinateSpace`.
- **Metric windows are isotropic-in-voxels.** `lncc`/`lncc_grad` take `radius`
  in voxels (`intensity.py:192`, `:256`); no mm/spacing. MI/CR are differentiable
  (soft Parzen, `information.py`) but have **no analytic force** — autodiff is
  their only route.
- **The SVF scaffold already has the seam.** `svf_coarse_to_fine`
  (`_svf.py:91-148`) takes a `level_solve` *closure*; each recipe passes its own,
  with the force inlined. A pluggable-force protocol slots in exactly here.

## 2. The design thesis: stratify, don't choose

Our genuine wins come from *specific* forces / closed-form gradients driving
*specific* algorithms (`lncc_grad`→SyN, ESM→Demons, inverse-compositional→rigid).
Make everything generic (any metric × any algorithm via autodiff) and we lose
them; keep everything specific and we get a combinatorial explosion of welded
recipes with no composability. The resolution — applied on **every** axis where
performance and generality conflict:

> **A generic, composable path is the substrate** (always available,
> autodiff-backed, correctness-guaranteed, **no** performance guarantee);
> **fast specialised implementers sit behind the *same* protocol, auto-selected
> when their preconditions hold.** The driver is written once over the protocol;
> specialisation is a *dispatch*, not a new recipe.

Performance becomes an *optimisation of a correct general path*, not a separate
API. Three consequences make this more than a slogan:

- **The generic path is the parity oracle** for the fast path — exactly how
  `lncc_grad`==autodiff and IC==forward were already validated. Every fast
  implementer ships with a generic-path equality test.
- **The recipe surface stays a *sum*, not a *product*.** A new fast kernel
  (analytic MI-force, a Pallas ESM kernel) is another implementer; the driver
  and the recipe count don't grow.
- **Configurability lives in the spec/args, not the function count.** The user
  picks a metric/force/schedule/space; the fast path engages silently when the
  inputs permit, the generic path carries the rest.

The four axes where this applies:

| Axis | Generic escape hatch | Fast specialised path (auto when preconditions hold) |
|---|---|---|
| **Dense-field force** | `MetricForce(metric)` — `∇(metric.cost)·∇warped`, any metric | `LNCCForce`, `DemonsForce` (closed-form) |
| **Matrix solver** | forward GN/LM/BFGS, any metric/space | inverse-compositional (SSD+IndexSpace+rigid/affine); closed-form SD |
| **Iteration count** | fixed `scan` (cohort-safe, reproducible) | `while_loop` early-exit (single-pair) |
| **Pyramid schedule** | one metric/force everywhere | per-level force/metric (fast coarse, high-signal fine) |

## 3. V1 — the `Force` keystone (the central deliverable) ✅ SHIPPED

A **`Force` protocol** in `register` — the dense-field analogue of the existing
`Objective`. Narrow contract: `(warped, fixed) → raw per-voxel update field`
(channel-last, `ndim`), plus a `cost(warped, fixed)` for the history (referencing
a `Metric`). Implementers in two tiers:

- **Tier 1 (performant; the named recipes' defaults):** `LNCCForce(radius)`
  (wraps `lncc_grad`), `DemonsForce(alpha)` (the ESM closed form, with its
  symmetrised gradient `j = ½(∇F+∇warped)`). These *are* the forces we ship
  today, now behind the protocol.
- **Tier 2 (escape hatch):** `MetricForce(metric)` — `∇_warped metric.cost ·
  ∇warped` via `jax.grad` for **any** `Metric` (MI, CR, NCC, custom), explicitly
  **no perf guarantee**.

`metrics` stays pure similarity kernels (the consumer-agnostic discipline,
[[nitrix-docstring-no-consumers]]); the metric→force adapter is a registration
concern, so `Force` and `MetricForce` live in `register`, mirroring how
`Objective`/`MetricObjective` do. The closed-form gradient kernels (`lncc_grad`,
a trivial `ssd_grad`) stay in `metrics` as general gradients.

**Driver changes.** `_demons_level`/`_syn_level` are rewritten to consume a
`Force` through the existing `level_solve` seam; each driver keeps only its
*structure* (single-sided vs symmetric-midpoint) + regularisation +
step-normalisation (spec concerns, not force concerns). The driver accepts a
**per-level force** (a `Force` or a length-`levels` schedule) — enabling
fast-coarse / high-signal-fine pyramids.

**Gate.** `LNCCForce`/`DemonsForce` reproduce the current Demons/SyN recovery
byte-for-byte (the welded path == the protocol path); `MetricForce(LNCC)` ≈
`LNCCForce` and `MetricForce(SSD)` ≈ `DemonsForce` to parity tolerance; a
**multimodal** (MI/NCC via `MetricForce`) deformable-recovery test; a per-level
metric-schedule test.

## 4. V2 — geometry/anisotropy + warm-start + multi-stage ✅ SHIPPED

Closes the matching-grid / isotropy assumptions with targeted fixes, no new
heavy machinery:

- **Warm-start / multi-stage init (V2a, `9d5610d`).** One mechanism —
  **pre-warp + compose** — serves three use cases: `init_displacement`
  (SynthMorph seed-then-refine), `init_affine` (the ANTs `Rigid→Affine→SyN`
  multi-stage, from a prior matrix recipe), and **different grids** (the
  pre-warp resamples moving onto the fixed grid, so the `shape ==` constraint
  applies only without an init). The recipe registers the *residual* and
  composes; `displacement`/`warped`/`jacobian_det` are the total map.
- **Physical LNCC windows (V2b, `e1ce09e`).** `LNCCForce.bind` already receives
  `rel_spacing`, so `_BoundLNCC` scales its voxel radius by `1/rel_spacing` to a
  **physically isotropic** window (same mm extent per axis) — the convention the
  regularisation sigmas already follow. The metric kernels stay spacing-agnostic
  (the conversion lives in the force, where the geometry context already flows).

**`GeometryContext` — declined (not deferred), a reasoned call.** The plan
floated a shared spacing+grid+affine context unifying the matrix
`CoordinateSpace` and the dense `spacing`. After V2a/V2b the two families use
geometry *genuinely differently*: the matrix family composes voxel→world affines
into the sampling matrix (`A_m⁻¹·T·A_f`), while the dense family needs only the
fixed-grid **anisotropy ratio** (sigmas/windows/gradients) plus **index-space
affine-init** (the pre-warp consumes the moving's world affine *before* the
deformable stage, which then runs on the common fixed grid). A unified context
would be a thin bag-of-fields the two paths read divergently — a forced
abstraction with no consumer demanding the unified entry. Declined for the same
reason the R7c matrix+SVF driver merge was: different state, not shared. A
`spacing_from_affine` convenience can land later if a pipeline asks for it.

**Gate (met).** Seeded-refinement recovery (no regression, reaches 0.99);
rigid→SyN multi-stage recovers a large rigid offset + deformation; a
coarser-grid moving registers via init; anisotropic physical-window SyN
recovers; physically-isotropic per-axis radii; init mutual-exclusion + no-init
shape-mismatch raise.

## 5. V3 — transform algebra + batched application (the Lie-group pillar) ✅ SHIPPED

Landed in `geometry.algebra` (V3a `b7f8079`, V3b `1597775`):
- **`transform_mean`** — the Fréchet (Karcher) mean of homogeneous transforms
  (rigid `SE(n)` ⊂ affine) via the log/exp fixed point. Unified on the **true
  matrix chart** (`linalg.matrix_log`/`matrix_exp`), not the closed-form
  `rigid_exp`/`rigid_log`, because the latter is split-param (the geodesic
  ½-point would not square to `T`) and singular at identity (which the Karcher
  init hits). The mean of rigids is rigid.
- **`linalg.matrix_log`** (NEW) — the general matrix logarithm its affine mean
  warranted: inverse scaling-and-squaring with Denman–Beavers square roots via
  `safe_inv`; round-trips `matrix_exp` to machine precision incl. large
  translations. Graduates the deferred §12.2 `matrix_log`.
- **`transform_geodesic`** — `exp(t·log T)`, a true geodesic (½-point squares
  to `T`); **`velocity_mean`** — the SVF barycentre (weighted arithmetic mean).
- **`fuse_transforms`** — collapse a multi-stage chain (matrices + displacement
  fields) into one displacement → **one** resample (quality: no compounded
  blur; throughput: one gather). Pure resampling, GPU-native.
- **Batched application** is already a primitive (`geometry.spatial_transform_-
  batched` / `vmap`), so it was demonstrated, not re-added.

*Design note:* `transform_mean`/`transform_geodesic` use `matrix_log` (hence
`safe_inv`), so they are offline barycentre / interpolation ops — jit/grad-clean
on a healthy GPU, forward/eager via the CPU fallback on the wedged-cuSolver dev
box (hence a Python loop, not `lax.scan`, since `safe_inv`'s fallback can't
engage inside a traced scan). **Gate (met):** `matrix_log` round-trip; mean of
identical / symmetric (rigid + affine) / pure-translation; geodesic endpoints +
halfway; A∘A⁻¹ fuses to ~zero; fused == sequential warp (ncc > 0.99);
vmap-batched fused warp.

### Original plan (for reference)

The algebraic backbone for groupwise/template work, multi-stage fusion, and
cohort throughput. A transform is a group element with a log (algebra) and exp
(group): rigid `SE(n)`, affine, and dense SVF/displacement. Operations, all
**batched** (`vmap` over a cohort) and differentiable:

- **compose / invert** — group composition + inverse. The primitives exist
  (`rigid_exp/log`, `affine_exp`, `compose_affine`/`invert_affine`,
  `compose_displacement`/`compose_velocity`, `invert_displacement`,
  `matrix_log`); this pillar gives them a coherent surface.
- **fuse-to-single-resampling.** Collapse a chain (rigid∘affine∘deformable) into
  **one** sampling grid so the moving is interpolated **once**, not three times —
  a *quality* win (no compounded interpolation blur) and a *perf* win (one
  gather). matrix∘matrix is a matmul; matrix∘dense folds the affine into the
  deformation grid.
- **Fréchet (Karcher) mean** of a set `{T_i}` — the Riemannian barycenter. For
  `SE(n)`/affine: the iterate `μ_{k+1} = μ_k ∘ exp(mean_i log(μ_k⁻¹ ∘ T_i))`; for
  SVF: the velocity-field mean (linear in the log domain). This is the
  template-centre primitive *and* the motion-summary primitive.
- **Geodesic interpolation / weighted mean** — `exp(t·log(T))` (slerp-like for
  rotations) for transform interpolation and temporal regularisation.
- **Batched application** primitives — apply a (batch of) transform(s)/warp(s) to
  (a batch of) images/points, with the volreg hoisting discipline (shared work
  out of the batch) and the bandwidth-aware kernel.

Homes: the math in `geometry`/`linalg` (Fréchet mean, fuse-to-grid); a thin
`register` layer for `RegistrationResult`-level fusion/mean. **Gate.** Fréchet
mean recovers a known centre + matches a reference Karcher iterate; fused
single-resample == sequential resamples to interpolation tolerance (and faster);
batched-apply throughput certified at cohort scale.

## 6. V4 — matrix perf levers, inside the config design

The last-round levers (`registration-matrix-recipe-perf-levers.md`) become the
**matrix-solver row** of the §2 table — fast implementers auto-selected on
preconditions, the forward/generic path as the escape hatch:

- **A** single-pair inverse-compositional (rigid) — wire the existing
  `_inverse_compositional` kernel into `rigid_register` (SSD+IndexSpace
  dispatch). **Measured 3.7–7.1×, identical recovery** (`bench/ic_vs_forward.py`).
- **A′** affine inverse-compositional (generalise the steepest-descent to the
  affine generators + a general compositional inverse).
- **C** closed-form forward steepest-descent (`∇warped·∂grid/∂θ`) for the cases
  IC can't cover (WorldSpace, affine-forward).
- **E** per-level iteration schedule (shared with the dense-field side).
- **B** single-pair `while_loop` early-exit (the iteration-count fast path;
  rigid converges ~5/20 even when hard) — and the `_normalise_step`
  clamp-vs-scale revisit it triggers.

**Gate.** IC==forward parity; the measured speedups certified at scale; the
`while_loop` iso-accuracy on a hard single-pair case.

## 7. V5 — ANTs-parity SyN completion + multimodal/groupwise capability

With V1 (metric-generic force) + V2 (multi-stage, anisotropy) + V3 (algebra)
landed, greedy SyN reaches ANTs feature parity: multi-metric (incl. multimodal
via `MetricForce`), affine-init, **masks** (windowed-metric + force masking),
multi-stage orchestration helpers, and the per-level metric schedule. The
**multimodal template construction** use case is delivered at the *capability*
level: `MetricForce(MI)` deformable + the V3 Fréchet mean as the template centre
+ batched application over the cohort. Template *data structures* / the full
atlas-build loop remain out of scope (→ `thrux`).

## 8. LDDMM / geodesic shooting — scoped, decided at end-of-round

**Substrate gaps (confirmed):** `numerics.ode` has euler/rk4 but **no
memory-efficient adjoint** (§12.11); velocity ops are **stationary-only** (no
`v(t)`). True geodesic-shooting LDDMM needs both: initial-momentum + EPDiff
geodesic + the ODE-adjoint backward, *or* time-varying relaxation (Beg-style
`v(t)` over `T` steps, differentiable through the existing scan-AD, memory linear
in `T`).

**The end-of-round decision (with the value question framed now).** Greedy SyN
already covers what most ANTs-SyN users run. Geodesic LDDMM earns its substrate
cost only for the cohort that needs *true geodesics*: momentum-based shape
statistics, geodesic regression, guaranteed-shortest-path deformations,
parallel transport for cross-subject statistics. The decision at the end of the
round weighs that realistic added value against the substrate investment, and
chooses between (a) defer, (b) time-varying relaxation LDDMM (lighter), or
(c) full geodesic shooting (the ODE-adjoint is independently valuable). The
adjoint substrate, if built, lands on the same `f(t, y)` interface
(`numerics.ode`) and the V1 `Force` protocol.

## 9. Engineering rigour & clean-abstraction vectors (corpus review)

- **The `Force` protocol + `MetricForce` adapter + closed-form implementers** —
  the keystone; collapses the per-recipe inlined forces into a sum of
  implementers behind one protocol (§3).
- ~~**A shared `GeometryContext`**~~ — **declined in V2** (§4): the matrix and
  dense families use geometry divergently (world-affine sampling composition vs
  anisotropy-ratio + index-space affine-init), so a unified context is a forced
  abstraction; a `spacing_from_affine` convenience can land if a pipeline asks.
- **The IC SD-projection bandwidth refinement** — the 7.1→3.7× erosion at 128³
  is the per-iteration `sd.T@err` re-reading the M×P steepest-descent array; a
  fusion / compacter layout recovers the small-size ratio at scale.
- **Warm-start plumbing** (`init_velocity`/`init_affine`) across both families.
- **The flagged consistency nit** — `integrate_velocity_field` default
  `n_steps=7` vs `DemonsSpec.n_steps=6`.
- **Parity-oracle test discipline made explicit** — every fast path ships a
  generic-path equality test; this is the design's safety net, not an extra.
- **Batched-application throughput** — the cohort/structural-win axis, with the
  volreg hoisting precedent generalised.

## 10. Sequencing & gates

- **V1 — `Force` keystone.** ✅ SHIPPED. Metric↔instrument decoupling; per-level
  force; `MetricForce` escape hatch (voxel-count rescale → exact closed-form
  parity).
- **V2 — geometry/anisotropy + warm-start + multi-stage.** ✅ SHIPPED. Pre-warp +
  compose init (`init_affine`/`init_displacement`); physical LNCC windows;
  `GeometryContext` declined.
- **V3 — transform algebra + batched application.** ✅ SHIPPED. `transform_mean`
  (Fréchet) / `transform_geodesic` / `velocity_mean` / `fuse_transforms` +
  `linalg.matrix_log` (graduated).
- **V4 — matrix perf levers** (A/A′/B/C/E) inside the config design. ← next.
- **V5 — ANTs-parity SyN + multimodal/groupwise capability.** Then the **LDDMM
  decision** (§8), then the perf round / hand-back to the perf agent.

Each phase: pure-functional surface, JAX fallback floor, rigorous typing
(Protocols where they earn it), immutable/frozen specs + NamedTuple results,
jaxtyping, ruff + ruff-format + mypy, `custom_vjp` where stability/efficiency
needs it, **a fast-path-vs-generic-path parity test**, and ships-with-a-(scaling)
case — the standing non-negotiables.

## 11. Out of scope (scope discipline)

Atlas/template *data structures* and the full atlas-build loop (→ `thrux`);
surface *data structures* / `bbregister` coupling; any I/O; PyTree / module
wrappers (→ `nimox` / `entense`). Geodesic LDDMM is a §8 end-of-round decision,
not a committed deliverable.

## 12. Cross-references

- [`registration-suite-v2.md`](registration-suite-v2.md) — the v2 block (R4–R8)
  this extends; §6 (perf program + economic verdict).
- [`registration.md`](registration.md) — the R0–R3 core.
- `docs/feature-requests/{registration-matrix-recipe-perf-levers,
  registration-early-stopping-while-loop, pallas-demons-esm-force,
  interpolation-backend-cpu-gpu-gap}.md` — the levers folded in above.
- `src/nitrix/{register,metrics,geometry,linalg,numerics}/` — the homes.
- `IMPLEMENTATION_PLAN.md` §10.A (deviation log; §12.11 ODE-adjoint for LDDMM).
