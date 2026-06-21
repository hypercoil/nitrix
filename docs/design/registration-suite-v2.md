# Registration suite v2 — volreg, BBR, greedy SyN (+ physical-space foundation)

> **TL;DR.**  The second registration block, building on the shipped
> rigid/affine + log-Demons core (`docs/design/registration.md`).  It adds
> the three most in-demand targets — **batched volreg** (motion
> realignment, per-frame matrix), **volumetric BBR** (boundary-driven
> rigid), and **greedy SyN** (symmetric, LNCC-driven diffeomorphic) — on
> top of a **physical-space foundation** that fixes the suite's one
> silent correctness bug: today everything optimises in *index/voxel*
> space, so on anisotropic voxels a "rigid" transform shears and the
> Demons regulariser is physically wrong.  Full SyN / LDDMM
> (geodesic-shooting, ODE-adjoint) is scoped here but **deferred** to the
> increment after this block.  Closes with one performance round so the
> new recipes are benchmarked before the registration spike closes.

Reads on top of `docs/design/registration.md` (the R0–R3 core),
`docs/design/geometry.md` (the SVF stack), `SPEC.md` §1 (non-goals:
no atlas/template *data structures* — those are `thrux`; this block adds
*algorithms*, not atlas structures), and the perf-agent feature requests
under `docs/feature-requests/` (cited inline).

## 0. Confirmed decisions (2026-06-10)

1. **Physical-space foundation first** (R4), ahead of the new algorithms —
   they all inherit the index-space bug otherwise.
2. **Greedy SyN only** this block; full SyN / LDDMM scoped as the next
   increment (needs the §12.11 ODE-adjoint layer; `L`-effort, SPEC-level).
3. **Correctness-first sequencing:** R4 → volreg → BBR → greedy SyN →
   one perf round.  The perf round comes last so it benchmarks the
   *finished* recipes.

## 1. The critical deficiency: everything is index-space

The recipes and almost the entire `geometry` layer operate in **voxel /
index** coordinates.  The only spacing-aware primitives are
`spatial_gradient` and `jacobian_displacement` / `jacobian_det_displacement`
(optional `spacing=`).  Every transform builder and both drivers are pure
index space: `affine_grid`, `identity_grid`, `apply_affine`, `rigid_exp`,
`affine_exp`, `integrate_velocity_field`; and the Demons fluid/diffusion
sigmas (`diffeomorphic.py::_smooth_vector`) are isotropic in **voxels**.

Consequences on anisotropic voxels (1×1×3 mm fMRI, 0.9×0.9×4 mm clinical
T1 — i.e. most real data):

- A rotation parametrised in index space is **not rigid in physical
  space** — it shears.  `rigid_register` returns a physically wrong matrix
  whenever spacing is anisotropic, with no knob to correct it.  This is
  the "fails under anisotropic voxel sizes" failure mode, and it is real
  and silent.
- Demons regularisation is anisotropically wrong (a voxel-isotropic
  Gaussian is physically anisotropic), so the deformation is over-/under-
  smoothed along the thick axis.
- Pyramid downsampling is isotropic-by-factor, ignoring that the
  thick-slice axis usually wants to stop downsampling early.

A second, related limitation: **`multi_resolution_register` requires
`moving.shape == fixed.shape`** (`_core.py:202`; same in
`diffeomorphic.py:199`).  Real inter-image registration (subject→template,
EPI→T1) lives on *different* grids with *different* world affines.

Both stem from one root: **no voxel→world affine in the model.**

### 1.1 The fix (R4)

Make the recipes world-aware.  Each image carries an optional `affine`
(voxel→world, `(d+1, d+1)`) or, as the convenience reduction, an
isotropic/per-axis `spacing`.  Then:

- The optimisation is over the **physical** (world→world) rigid/affine
  transform, not the index-space one.
- Sampling composes `fixed-voxel → world → moving-world → moving-voxel`
  via `geometry.affine.{compose_affine, invert_affine, make_square_affine}`
  (already shipped) at grid-build time — a few matrix products, no new
  heavy substrate.
- Demons fluid/diffusion sigmas are specified in **mm** and converted
  per-axis before the separable Gaussian.
- Gradients pass `spacing` (already supported).
- Dropping the equal-affine assumption removes the **same-shape
  constraint** for free: `fixed` and `moving` may live on different grids.

As shipped (§8, R4), this is **not** plumbed through the geometry
primitives (which are correctly coordinate-agnostic) but factored behind a
`CoordinateSpace` ADT consulted by the driver — keeping `IndexSpace` as a
distinct on-device fast path.  The gate is an **anisotropic-voxel recovery
test** that plants a known physical transform on an anisotropic grid and
recovers it under `WorldSpace` where `IndexSpace` cannot.

## 2. Batched volreg (motion realignment)

**What.**  Per-frame rigid registration of a 4-D series to a reference,
returning a `(T, d+1, d+1)` matrix stack + the realigned series.

| Need | Status | Home |
|---|---|---|
| physical-space rigid (anisotropic EPI) | ❌→R4 | foundation |
| batched warp | ✅ `spatial_transform_batched` | `geometry.grid` |
| batched `rigid_exp` (leading dims) | ✅ | `geometry.transform` |
| **batched optimiser** (vmap GN/LM over frames) | ❌→R5 | `linalg.optimize` / driver |
| **inverse-compositional rigid step** (constant template Hessian) | ❌→R5 | `linalg.optimize` (item C, specialised) |
| batched pyramid / `affine_grid` | ⚠️ vmap | `geometry` |
| reference policy (frame / mean / 2-pass) | recipe | `register.volreg` |

- **Inverse-compositional is the marquee perf lever** and the reason AFNI
  `3dvolreg` is fast where we lag.  Today the residual Jacobian is taken
  through the warp of the *moving* frame every iteration (re-evaluated per
  step *and* per frame).  In the inverse-compositional Lucas–Kanade scheme
  the steepest-descent images `∇F·(Gⱼx)` and the Gauss–Newton Hessian are
  evaluated **on the fixed template once** and reused across all iterations
  and all frames; the per-iteration cost collapses to a warp + one gather +
  a constant 6×6 solve.  This is the closed-form steepest-descent (item C
  of `registration-typing-metric-adt.md`), specialised to the constant-
  template case.
- The `assembled` small-P path (`cg` on a P×P Gram, `cg` is batched)
  `vmap`s cleanly over a frame axis; volreg is SSD/LM so the non-vmappable
  BFGS path is irrelevant.
- **Tension to record:** the `while_loop` early-exit
  (`registration-early-stopping-while-loop.md`) **breaks vmap-batching** (a
  cohort runs to the max trip count).  Volreg takes the fixed-`scan` +
  constant-Hessian win; `while_loop` is reserved for single-volume
  affine/SyN.

## 3. Volumetric BBR

**What.**  Rigid alignment driven by the WM/GM boundary (Greve & Fischl
2009).  The cost is over boundary point samples, not an image pair: for
each boundary point, sample the moving (EPI) intensity at `±δ` along the
boundary normal and accumulate a normalized cross-boundary contrast.

| Need | Status | Home |
|---|---|---|
| transform points by candidate `T` | ✅ `apply_affine` | `geometry.transform` |
| sample along normals | ✅ `sample_at_points` | `geometry.grid` |
| physical normals + `δ` in mm | ❌→R4 | foundation |
| **`Objective` protocol** (θ ↦ cost over closed-over data) | ❌→R6 | `register` |
| **BBR cost** (Greve–Fischl `tanh` contrast + robust weighting) | ❌→R6 | `register.bbr` |
| rigid optimise (BFGS / `implicit_minimize`) | ✅ | `linalg` |

- The **`Objective` protocol** is the clean-abstraction centrepiece.
  `registration-typing-metric-adt.md` already foreshadows it: generalise
  `Metric.cost(warped, fixed)` to `Objective` (θ ↦ cost/residual over
  closed-over data), of which the image-pair `Metric` is one constructor
  and BBR is another (`BBRObjective`, over boundary points), and the
  dense-field force a third.  Introducing it *with* BBR is what prevents a
  parallel structure.
- BBR depends on physical space intrinsically (normals are physical
  directions, `δ` is in mm) — a second reason R4 comes first.
- **Scope boundary:** surfaces as a *data structure* are deferred (your
  note + SPEC "no atlas/template structures").  BBR here consumes boundary
  points + normals **as arrays**; building them from a surface is a
  `thrux`/surface-features concern, and the `bbregister` surface coupling
  waits for surface features.  `Rigid()` is the transform model (the typing
  doc: "BBR uses `Rigid()`").

### 3.1 Recovery recipe — grid multistart + step annealing (2026-06-21)

**Problem.** The shipped BBR (R6b) was a single local BFGS solve at a fixed
sampling `step`.  On a real WM/GM boundary it recovered rotation but barely
moved translation (a `~2.7°/3 mm` plant -> `~1.4°/2.5 mm` residual): BBR's cost
is shallow and markedly **non-convex**, and at a fine `step` the translation
basin is narrow and noisy (diagnosed by sweeping the cost — the minimum *is* at
the truth, so the cost/normals were correct; it was a basin/tuning gap, not a
formulation bug).

**Fix (FSL `flirt --bbr` parity).** Two ingredients, both default-on:

- **Step annealing** (`BBRSpec.schedule`, default `(4,2,1)`) — a *large*
  sampling `step` samples far either side of the boundary, smoothing the cost
  and widening its basin (it captures the gross translation a fine step is
  blind to); finer stages refine, warm-started.  This mirrors FSL's annealed
  `bbrstep`.
- **Grid multistart** (`BBRSpec.search`, default on) — before refining, the
  cost is evaluated under one `vmap` on a coarse grid of rigid seeds
  (`3**6 = 729` for 3-D rigid) and the lowest-cost seed initialises the solve,
  so a single descent cannot latch a wrong boundary alignment.  Mirrors FSL's
  `gridmeasurecost`.  Pure annealing *without* the grid diverged on the folded
  brain at `>3°` (wrong-basin latch); the grid cures it.

Recovery on the real boundary: `2.7°/3 mm -> 0.1°/0.7 mm`, robust across plant
sizes and boundary samples (and on a 3-D box synthetic, `7°/6 vox -> 0.5°/0.3 vox`).

**Optimiser — normalised-block GD, and why not BFGS / Newton.** The per-stage
solve replaces `jax.scipy` BFGS with a *normalised-block gradient descent*: the
rotation and translation gradient blocks are unit-normalised and stepped by
physical (rad / mm) step lengths with a geometric decay.  This is forced by the
cost's structure — damped Newton **diverges** (the Hessian is indefinite on the
non-convex tanh cost) and a single-rate GD cannot serve both the rad and mm
blocks (the unit mismatch BFGS otherwise learns).  The normalised blocks make
the step **amplitude- and unit-invariant** (the contrast `Q` is already a ratio),
so the step lengths are image-independent defaults (`50×` intensity scaling ->
identical recovery up to the `eps` denominator guard).

**Scan/while toggle (aligned with the other recipes).** The GD step runs through
the shared `run_iterations`, so `BBRSpec.convergence` gives the same toggle:
`None` (and `'auto'` for BBR) is the fixed `lax.scan` (reproducible, reverse-
differentiable); an explicit `Convergence` is the windowed-slope `lax.while_loop`
early-exit (opt in; the early-exit barrier raises on reverse-mode, as elsewhere).
Unlike the inverse-compositional recipes, **`'auto'` resolves to the scan** here:
the GD early-exit only beats the (already fast) scan at a loosened threshold (at
the tight default it is slower than the scan from while-loop overhead).

**Performance (the decision driver).** Measured on the real boundary (L4 GPU,
`N=5000`): grid+anneal GD-scan **8.9 ms** at 30 iters/stage (`11.6 ms` at the
default 50) vs the grid+anneal BFGS path `11.3 ms` — so GD is *not slower*, and
recovers better and more consistently (BFGS recovery is backend-inconsistent:
`1.4°` GPU vs `0.1°` CPU on the same problem).  CPU ~320–450 ms.  vs FSL
`flirt --bbr` `7.35 s` -> still `~650×`.  Unrolled reverse-mode diff through the
scan is a by-product, not a goal — prefer `linalg.implicit_minimize` on
`bbr_cost` (adjoint at the optimum, trajectory-independent) for gradients.

## 4. Greedy SyN

**What.**  The design doc's stated upgrade — "a symmetric forward+inverse
formulation with an LNCC metric is greedy-SyN."  Represented as
**symmetric, LNCC-driven log-Demons**, reusing the SVF machinery rather
than a separate greedy-displacement integrator.

| Need | Status | Home |
|---|---|---|
| **LNCC analytic force** (per-voxel CC gradient image) | ❌→R7 | `metrics` / `register` |
| symmetric driver (two half-warps, midpoint) | ❌→R7 | `register.greedy_syn` |
| compose / invert velocity & displacement | ✅ `compose_velocity` / `invert_displacement` / `compose_displacement` | `geometry.deformation` |
| SVF exp, Gaussian fluid+diffusion, pyramid | ✅ | `geometry` / `smoothing` |
| explicit penalties (`bending_energy`, …) | ✅ | `register.regulariser` |

- **LNCC analytic force is the one genuinely new kernel** and a dual-
  purpose win (correctness + perf).  Demons currently uses the closed-form
  SSD ESM force; SyN needs the local-cross-correlation force.  The ANTs CC
  gradient has a closed form that **reuses the exact box-sums `lncc`
  already computes** (`metrics/_common._box_sum`) — far cheaper than
  autodiff-through-the-windowed-CC and numerically matched to ANTs.  Ship
  it as `lncc`'s per-voxel gradient w.r.t. the moving image (a force
  field).
- **Symmetric driver:** two half-warps (moving→mid, fixed→mid); warp both
  to the midpoint; compute the symmetric LNCC force; update both
  velocities; final transform `φ_fwd ∘ φ_inv⁻¹`.  Verify
  `invert_displacement`'s fixed 50-iter Picard (`deformation.py:114`)
  converges under large symmetric deformation at brain scale — it is now in
  the inner loop.
- Most of the body is the existing `_demons_level` with the force swapped
  and a midpoint split — which is exactly why the **shared driver
  refactor** (§6) is worth doing first.

## 5. Full SyN / LDDMM — scoped, deferred

Time-dependent velocity + geodesic shooting needs the **ODE-adjoint**
layer (§12.11).  `numerics.ode` has fixed-step `euler`/`rk4` but **no
adjoint**; the adjoint is the Chen-2018 pattern = `fixed_point_solve` +
`cg` (both shipped), so the composition is sketchable — but it is `L`
effort and SPEC-level per §13.  Greedy SyN is the representative
diffeomorphic-quality jump; full SyN/LDDMM is the geodesic refinement, and
it lands *after* the perf round certifies greedy SyN.

## 6. Performance program (closing the ANTs/ITK gap)

Orthogonal levers, prioritised against the "lag on CPU, barely ahead on
GPU" finding:

| Lever | Target | Status / source | Priority |
|---|---|---|---|
| **CPU interpolation backend** | ~~the CPU lag is mostly this~~ — **R8 correction:** single-shot `spatial_transform`/`resample` on CPU are **faster** than scipy (4–5.6×); the 5–9× lag is **narrow** (iterated `integrate_velocity_field`, CPU, small grids) and rigid/affine never hit it | `interpolation-backend-cpu-gpu-gap.md` (R8 status); `geometry/_interpolate.py` | **demoted** |
| **Inverse-compositional / constant-template Hessian** | rigid + volreg per-iteration cost + cross-frame redundancy | item C; `linalg/optimize.py` | high (AFNI-parity lever) ✅ R5b |
| **LNCC analytic force** | SyN/Demons per-iteration cost; enables §4 | new; `metrics`/`register` | high (dual-purpose) ✅ R7a |
| **Pallas ESM-force kernel** | SVF family HBM-bound at scale — **R8: gate (a) confirmed for *both* Demons and SyN** (96³→128³ scaling exponent 1.37, identical) | `pallas-demons-esm-force.md`, **promoted** (gate b: a bottlenecked consumer) | medium |
| **`while_loop` early-exit** | iteration-count deficit vs convergence-gated ANTs — **R8: SyN uses full budget (no benefit); cohort breaks vmap; but single-pair rigid/affine converge ~5/20 even when hard (the lever for the case we lose on)** | `registration-early-stopping-while-loop.md`, acceptance-gated | medium (single-pair matrix only) |
| **Adaptive `matrix_exp` squaring / SVF `n_steps`** | trim affine + SVF graph for near-identity generators | item E | low |

Discipline (`perf-wins-must-certify-at-scale.md`): each lever is certified
**across the size curve to brain scale (single + cohort)** with a stated
cost law, not at the dev size; the new recipes ship with a scaling case.

**R8 economic verdict (an edge is not a win).** GPUs are scarce/expensive vs
CPUs (a cluster has 1000s of CPUs / 10s of GPUs; cloud GPU is 10–100× the cost),
so nitrix-GPU must beat ANTs-CPU by a margin that *clears the hardware premium*
(≈10×), not merely edge it. Classified by recipe: **single-pair rigid/affine =
NOT a win** (235 ms GPU vs 274 ms ANTs-CPU at 128³ = 1.16×, a cost-normalised
loss — the §6 review target); **diffeomorphic demons/SyN = qualitative win**
(~20× vs fixed-count ITK demons, confound-free; SyN same regime); **cohort
volreg = structural win** (one GPU runs the whole 4-D series as one batched
kernel — 4.1 ms/frame, flat in T; serial CPU mcflirt/3dvolreg can't match
per-dollar); **differentiable layer = capability win** (no ANTs analogue). Lead
with the qualitative/structural wins. The honest open measurement is an
**iso-accuracy** point vs early-stopping ANTs-SyN (needs ANTs in-env).

## 7. Engineering rigour & clean-abstraction vectors (corpus review)

- **Unify the coarse-to-fine driver.**  `multi_resolution_register`
  (matrix) and `diffeomorphic_demons_register` (SVF) duplicate the
  scaffold — pyramid build, coarse→fine loop, warm-start prolong/rescale,
  history concat.  *Revised plan (R5):* volreg turned out to reuse the
  matrix driver (via the extracted `register_core`) rather than need a
  matrix+SVF unification, and the inverse-compositional path is a
  genuinely distinct loop — so unifying all three speculatively risks the
  wrong abstraction.  ✅ *Resolved (R7c):* the warranted factoring was the
  **SVF×2** one — Demons + greedy SyN share `_svf.svf_coarse_to_fine`.  The
  matrix driver stays in `_core`: it carries a parameter *vector* + a
  coordinate-space sampler, a genuinely different state machine, so a
  matrix+SVF merge was *declined*, not deferred (BBR also turned out to be
  single-resolution, not a coarse-to-fine instance).
- **Introduce the `Objective` protocol** (§3) as the metric-side unifier:
  `Metric` (image pair), `BBRObjective` (boundary points), `DenseFieldForce`
  (Demons/SyN) become constructors.  ✅ *Done (R6):* `Objective` +
  `MetricObjective` + `BoundaryObjective`, with `optimize_objective` the
  shared dispatch.  Demons/SyN force is not yet an `Objective` (the SVF
  driver is a different loop) — folds in at R7 if it earns it.
- **Spec hierarchy.**  `RegistrationSpec` / `DemonsSpec` (+ incoming
  `VolregSpec` / `BBRSpec` / `SynSpec`) share schedule fields (`levels`,
  `iterations`, `pyramid_factor`, `pyramid_sigma`, `boundary_mode`).  A
  shared frozen, hashable, jit-static base stops them drifting; spacing /
  world-affine lands here in R4.
- **Pin MI/CR histogram ranges per level.**  `joint_histogram` defaults to
  data min/max ⇒ piecewise-constant gradient, unstable across the
  optimisation.  The recipes pin intensity ranges; the kernel default
  stays.
- **Validation gaps to close before the spike ends:** (a) the
  anisotropic-voxel recovery test (R4); (b) **real-data parity** vs
  `3dvolreg` / `mcflirt` / ANTs (synthetic recovery only so far); (c)
  `invert_displacement` convergence under large deformation; (d)
  consistency nit — `integrate_velocity_field` default `n_steps=7` vs
  `DemonsSpec.n_steps=6`.
- **Keep the pure-functional / NamedTuple surface; decide pytrees
  consciously.**  Results stay NamedTuples (already pytrees); specs stay
  frozen/static.  Register a record as a pytree only if it must cross
  `vmap` as *data* (the B22 concern) — batched volreg is a leading axis on
  arrays, so no new pytree is needed.

## 8. Sequencing & gates

- **R4 — physical-space foundation.** ✅ **SHIPPED** (branch
  `registration-suite-v2`).  Rather than thread world affines through the
  geometry primitives (which are correctly coordinate-agnostic), the
  index-vs-physical axis is factored behind a **`CoordinateSpace` ADT**
  (`register/_space.py`): `IndexSpace` (default; voxel-space, shared-grid,
  fully on-device, the lean path and the future inverse-compositional
  frame) and `WorldSpace(fixed_affine, moving_affine)` (physical space via
  `A_moving⁻¹·T_world·A_fixed`, per-level align-corners scale; one
  `safe_inv` per registration).  The driver is rewritten **once** over the
  space (shared `_warp` on `sampler.index_sampling`); the same-shape
  constraint is dropped (the warp builds on the fixed grid).  The Demons
  recipe gains a `spacing` knob applied via the **relative** (anisotropy-
  only, level-independent) spacing, axis-correcting the ESM force and the
  fluid/diffusion regularisation; isotropic data is byte-unchanged.
  **Gate (met):** an independent raw-matrix oracle recovers a known *world*
  rigid transform on an anisotropic grid under `WorldSpace` (ncc > 0.95,
  rotation tight) where `IndexSpace` cannot; 3-D recovery; Demons
  anisotropic recovery stays diffeomorphic.  84 registration tests green,
  mypy + ruff clean.  *Deferred to R5:* the closed-form steepest-descent /
  inverse-compositional kernel that `IndexSpace` is now the frame for.
- **R5 — batched volreg.** ✅ **SHIPPED** (branch `registration-suite-v2`).
  `register.volreg` rigidly realigns a `(T, *spatial)` series to a common
  reference and returns the per-frame transform stack + realigned series.
  Rather than `vmap` the whole single-pair driver (which would recompute
  the shared reference per frame), the reference work is **hoisted out of
  the batch**: `register_core(moving, pyr_f, sampler, …)` is extracted from
  the driver (behaviour-preserving), the reference pyramid + sampler are
  built once, and only the per-frame core is `vmap`-ed (closing over the
  shared reference — which also sidesteps pytree registration).  Reference
  policy (`"mean"` / frame index / explicit), two-pass (`passes=2`), and
  the `CoordinateSpace` arg (so motion rigid in *physical* space is
  recovered on anisotropic grids).  **R5b — inverse-compositional kernel
  (`_inverse_compositional.py`):** the constant-template Hessian — the
  reference steepest-descent + `H⁻¹` are built **once per level for the
  whole series** (all frames, all iterations); each per-frame iteration is
  warp → error → `SDᵀe` → `H⁻¹` matvec → compositional matrix update.
  IndexSpace + SSD (the frame R4 set up for it); `method="auto"` selects it
  for IndexSpace, forward for WorldSpace.  **Gate (met):** per-frame
  recovery 2-D/3-D, reference policies, two-pass, batched WorldSpace
  anisotropy; IC == forward at convergence (realigned ncc > 0.999);
  measured **7.3× warm** vs forward on a 16-frame 32³ series.  *Deferred:*
  the matrix+SVF driver unification (§7) — to be factored when BBR/SyN give
  a third concrete instance, not speculatively; single-pair IC fast path.
- **R6 — BBR.** ✅ **SHIPPED** (branch `registration-suite-v2`).  **R6a —
  `Objective` protocol** (`_objective.py`): the `θ ↦ cost` generalisation
  the `Metric` ADT foreshadowed.  `MetricObjective` (image pair + warp) and
  `BoundaryObjective` (BBR) implement it; `_optimize_level` became
  `optimize_objective(objective, …)` — the LM/BFGS dispatch written once,
  consumed by both the coarse-to-fine driver and BBR (behaviour-preserving
  for the image recipes).  This is the second-instance factoring, not
  speculation.  **R6b — `bbr_register`** (`_bbr.py`): the Greve-Fischl
  boundary cost (`bbr_cost`: sample `moving` at `±step` along the
  transformed normal, normalised cross-boundary contrast, `mean(1 +
  tanh(slope·(Q−q0)))`), optimised over the rigid params via the shared
  BFGS path.  The transform rotates about the **boundary centroid** (not
  the origin — the same centring lesson as R4; without it the rotation
  parameter's origin-distance leverage collapses BFGS).  Surfaces stay
  out of scope (points + normals as arrays); `moving_affine` makes `step`
  physical and the normals physical directions (anisotropy-correct).
  **Gate (met):** planted-offset recovery (smoothed disk), cost strictly
  lower at the truth, differentiable w.r.t. the moving image via
  `implicit_minimize` (FD-checked), anisotropic world-circle recovery,
  validation.
- **R7 — greedy SyN.** ✅ **SHIPPED** (branch `registration-suite-v2`).
  **R7a — `metrics.lncc_grad`:** the analytic local-CC force, the closed-
  form `∂(Σ cc)/∂moving` from the same box-sums `lncc` forms plus a second
  box-sum pass — **machine-precision-identical to autodiff** across the
  whole field (the reflect box filter is self-adjoint), with no autodiff
  tape.  **R7b — `greedy_syn_register`** (`_syn.py`): symmetric forward/
  inverse SVF flowed to a midpoint by `lncc_grad·∇warped`, composed via
  `invert_displacement` + `compose_displacement`; diffeomorphic by
  construction.  **R7c — SVF driver unification** (`_svf.py`):
  `svf_coarse_to_fine` + the spacing/smoothing helpers shared by Demons
  (n_fields=1) and SyN (n_fields=2) — the §7 factoring, scoped to the SVF
  family (the matrix driver is a different state machine, left in `_core`).
  **Gate (met):** `lncc_grad` == autodiff (2-D/3-D); symmetric-warp
  recovery (ncc > 0.99 / 0.96, min `det J > 0`); **LNCC bias-robustness**
  (recovers under a smooth multiplicative bias that defeats an SSD force);
  symmetric identity (net `φ ≈ id` by forward/inverse cancellation).  Two
  decisions worked through: force normalisation is a trust-region *clamp*
  (not scale-to-step) under the fixed scan budget — **coupled to the
  `while_loop` early-exit decision** (recorded at the site + the
  feature-request); and the LNCC force does not vanish at a match (eps in
  low-variance windows — standard), handled by symmetric cancellation.
  *Deferred:* per-window variance gating of the force (so the velocities
  settle, not just the net map).
- **R8 — perf round.** ✅ **SHIPPED** (branch `registration-suite-v2`).  A
  certification + decision round, not a new-kernel round — the gated kernels
  stayed gated, decided from evidence.  **Certification** (`bench/perf_-
  registration_v2.py`, L4 f32): greedy SyN single-pair 64³ 53 ms / 96³ 199 ms /
  128³ 650 ms (warm/Mvox 204→225→310 — super-linear, **same bandwidth-bound
  regime as Demons**, exponent 1.37); volreg cohort IC 4.1 ms/frame at 64³,
  **flat as T doubles** (reference-hoisting amortises) — the structural cohort
  win.  **Convergence/iso-accuracy** (`bench/conv_trajectory.py`): rigid
  converges ~5/20 steps even when hard; **greedy SyN uses its full budget**
  (gradual 1st-order descent under the non-vanishing LNCC force).  **Decisions:**
  (1) *CPU-interp backend dropped* — single-shot interpolation already beats
  scipy on CPU; the 5–9× is narrow (iterated SVF-exp, small grids) and misses
  rigid/affine.  (2) *Pallas ESM-force promoted* — gate (a) HBM-bound confirmed
  for the whole SVF family (Demons + SyN); awaits a bottlenecked consumer (gate
  b).  (3) *`while_loop` decided* — no benefit for SyN (full budget), breaks the
  cohort path, but **viable for single-pair rigid/affine** (the recipe class §6
  targets); the R7 clamp-vs-scale call stands.  The economic verdict (above)
  is the headline.  *Deferred:* the iso-accuracy point vs ANTs-SyN (needs ANTs
  in-env); full SyN/LDDMM is the next increment.

Each phase: pure-functional surface, JAX fallback floor, jaxtyping, ruff +
ruff-format + mypy, `custom_vjp` where stability/efficiency needs it,
ships-with-a-(scaling-)case — the standing non-negotiables.

## 9. Out of scope (scope discipline)

Atlas/template data structures and template-aware ops (→ `thrux`); surface
*data structures* / `bbregister` surface coupling (→ surface-features
block); any I/O; PyTree / module wrappers (→ `nimox` / `entense`); full
geodesic-shooting SyN / LDDMM in this block (the §12.11 ODE-adjoint
increment that follows).

## Cross-references

- `docs/design/registration.md` — the R0–R3 core this extends.
- `docs/design/geometry.md` — the SVF stack the diffeomorphic recipes lower
  onto.
- `docs/feature-requests/{registration-typing-metric-adt,
  registration-early-stopping-while-loop, pallas-demons-esm-force,
  interpolation-backend-cpu-gpu-gap, perf-wins-must-certify-at-scale,
  field-regularisers}.md` — the perf-agent findings folded in above.
- `src/nitrix/{geometry,linalg,numerics,metrics,register}/` — the homes.
- `IMPLEMENTATION_PLAN.md` §10.A (deviation log where the §12 graduations
  and this block's landings are recorded).
