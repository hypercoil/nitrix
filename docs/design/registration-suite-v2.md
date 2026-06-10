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

The cost is mostly plumbing a `spacing`/`affine` through the specs and the
grid builders.  The gate is an **anisotropic-voxel recovery test** that
plants a known physical transform on an anisotropic grid and recovers it —
a test that *currently fails*, which is the point.

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
| **CPU interpolation backend** | the CPU lag is *mostly this*: jax `map_coordinates` is **5–9× slower than scipy-C** for the order-1 gather every warp/resample/SVF-step routes through | `interpolation-backend-cpu-gpu-gap.md`; `geometry/_interpolate.py` | **highest** |
| **Inverse-compositional / constant-template Hessian** | rigid + volreg per-iteration cost + cross-frame redundancy | item C; `linalg/optimize.py` | high (AFNI-parity lever) |
| **LNCC analytic force** | SyN/Demons per-iteration cost; enables §4 | new; `metrics`/`register` | high (dual-purpose) |
| **Pallas ESM-force kernel** | Demons HBM-bound at scale (speedup 43×→28×, 48³→160³) | `pallas-demons-esm-force.md`, profile-gated | medium |
| **`while_loop` early-exit** | iteration-count deficit vs convergence-gated ANTs/dipy | `registration-early-stopping-while-loop.md`, acceptance-gated; single-volume only (conflicts with batched volreg) | medium |
| **Adaptive `matrix_exp` squaring / SVF `n_steps`** | trim affine + SVF graph for near-identity generators | item E | low |

Discipline (`perf-wins-must-certify-at-scale.md`): each lever is certified
**across the size curve to brain scale (single + cohort)** with a stated
cost law, not at the dev size; the new recipes ship with a scaling case.

## 7. Engineering rigour & clean-abstraction vectors (corpus review)

- **Unify the coarse-to-fine driver.**  `multi_resolution_register`
  (matrix) and `diffeomorphic_demons_register` (SVF) duplicate the
  scaffold — pyramid build, coarse→fine loop, warm-start prolong/rescale,
  history concat.  With volreg + BBR + greedy SyN incoming, factor one
  shared driver parameterised by `(init, level_solve, prolong, finalize)`.
  Do this **before** the new recipes so each plugs in instead of
  re-deriving the loop.
- **Introduce the `Objective` protocol** (§3) as the metric-side unifier:
  `Metric` (image pair), `BBRObjective` (boundary points), `DenseFieldForce`
  (Demons/SyN) become constructors.
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

- **R4 — physical-space foundation.**  Spacing/world-affine through specs +
  grid builders + Demons sigmas; drop the same-shape constraint.
  **Gate:** anisotropic-voxel physical-transform recovery (rigid + affine);
  isotropic results unchanged (regression).
- **R5 — batched volreg.**  Shared-driver refactor (§7) +
  inverse-compositional rigid step + `volreg` recipe + reference policy.
  **Gate:** per-frame recovery on a planted motion series; constant-Hessian
  numerics == per-iteration-Jacobian; scaling case (single + cohort).
- **R6 — BBR.**  `Objective` protocol + `bbr_register` + robust weighting.
  **Gate:** boundary-recovery on a planted rigid offset; differentiable-
  layer grad == FD via `implicit_minimize`.
- **R7 — greedy SyN.**  LNCC analytic force + symmetric SVF driver +
  `greedy_syn_register`.  **Gate:** force == FD-grad of `1 − lncc`;
  symmetric-warp recovery (ncc) with min `det J > 0`; inverse consistency.
- **R8 — perf round.**  CPU-interp backend; certify all new recipes at
  scale; revisit Pallas / `while_loop` on profile evidence.  Then the
  benchmark round, then close the spike (full SyN/LDDMM scoped as the next
  increment).

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
