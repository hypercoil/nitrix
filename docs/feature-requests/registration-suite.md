# Registration suite — audit ledger & improvement roadmap (`nitrix.register`)

> **Status (2026-06-22): audit/ledger doc.** Frames the registration suite,
> credits the shipped substrate, indexes the already-filed registration FRs
> that own part of the audit (the duplicate-issue guard, per `README.md`), and
> specs the **genuinely-new numerical gaps** surfaced by a six-lens review.
> Numerics-only scope; image/file I/O is **out of scope** (→ `thrux`). No item
> here duplicates an existing FR — those are referenced, not re-opened.

## 1. Why this doc exists

A six-dimension audit of the whole suite (mathematical correctness, engineering
rigour, community value, performance, design/abstraction, hardware/GPU —
fan-out review, 42 findings, 40 verified against the code) returned a clear
verdict: **the suite is algorithmically strong and architecturally
sophisticated — the gaps are at the *seams and the surface*, not in the
kernels.** The core math is sound where it matters (LNCC center-force is
ITK-anchored, CR's autodiff gradient is well-conditioned, WorldSpace and the
closed-form Jacobian are well tested), and the recent recovery round (§3)
hardened BBR, affine-MI, and validated the diffeomorphic paths.

Five themes dominate the improvement surface; this doc turns them into a
trackable roadmap.

1. **Public ergonomics** — no one-call rigid→affine→SyN pipeline (the single
   most-run neuroimaging command), no `apply_transform` for an in-memory
   array/label, no masked cost on the matrix recipes. Every downstream
   re-derives the staging and gets it subtly wrong.
2. **Convention correctness** — `IndexSpace` `RegistrationResult.matrix` is not
   self-contained (centering is applied to the warp but not baked into the
   returned matrix, unlike `WorldSpace`), and `convergence='auto'` resolves to
   *opposite* behaviours across recipes.
3. **Cross-modal accuracy** — NMI (the preferred deformable cross-modal
   criterion) has no closed-form force and routes to the slow, non-deterministic
   autodiff path.
4. **Hot-loop performance** — the SVF drivers recompute the cost from scratch
   alongside the force every iteration; the symmetric driver keys early-exit on
   the *forward-half* cost only; an O(N log N) percentile sort runs every
   iteration.
5. **GPU/float32 rigour** — the test suite validates only float64 while
   float32/GPU is the declared production path; the MI scatter non-determinism
   has no determinism contract or GPU-gated test.

## 2. Scope boundary

**In scope — numerical primitives only**, consistent with the SPEC §6
dependency contract (`numpy` + `jax`; no `nibabel`, no filesystem, no image
I/O). Kept in scope and specced below: in-memory **transform algebra**
(compose / invert / fuse — owned by [`affine-matrix-algebra`](affine-matrix-algebra.md)),
applying a recovered transform to an **in-memory** array/label
(`apply_transform`), masked cost, and the numerics of `WorldSpace`.

**Out of scope → `thrux`** (explicitly curated out of this roadmap; the audit
raised an I/O vector that does **not** belong in `nitrix`):

- Reading/writing image files (NIfTI / `nibabel`), and any "optional NIfTI
  extra" dependency.
- Serialising transforms to/from disk (ITK `.mat` / `.h5` / FSL / AFNI formats).
- Deriving a `WorldSpace` affine *from a file header* / NIfTI `sform`/`qform`
  round-tripping.

`nitrix` consumes arrays + an affine **already in memory** and returns arrays +
transforms in memory; the file layer is `thrux`'s.

## 3. Already shipped (credit)

- **Recovery round (2026-06, branch `accuracy/bbr-recovery`, not yet merged):**
  BBR recovery recipe (grid-multistart + step-annealing + normalised-block-GD
  optimiser with the scan/while toggle, `a360f40`); `affine_register(restarts=)`
  multi-start for the GPU affine-MI non-determinism (`35bf540`); `MetricForce(SSD)`
  docstring note — it is the raw optical-flow gradient, under-recovers vs the
  Thirion-normalised `DemonsForce` (`87751b6`). The diffeomorphic paths
  (Demons+DemonsForce, SyN+LNCCForce, ±MI multimodal) were validated as
  recovering sub-voxel in 2D + 3D with no fix needed.
- **Substrate (shipped, see the referenced FRs):** Metric ADT + TransformModel
  ([`registration-typing-metric-adt`](registration-typing-metric-adt.md)),
  rolled-loop cold compile ([`registration-recipe-cold-compile`](registration-recipe-cold-compile.md)),
  single-pair IC early-exit ([`registration-early-stopping-while-loop`](registration-early-stopping-while-loop.md)),
  affine small-grid trust region ([`register-affine-small-grid-divergence`](register-affine-small-grid-divergence.md)),
  Demons ESM 0/0 guard ([`register-demons-force-divide-by-zero`](register-demons-force-divide-by-zero.md)),
  metric-convention verification ([`metrics-convention-vs-domain-tools`](metrics-convention-vs-domain-tools.md)).

## 4. Audit findings already owned by an existing FR (duplicate guard)

These audit items are **not re-specced here** — add to the referenced doc instead:

| Audit finding | Owned by | State |
|---|---|---|
| Per-iteration matrix-recipe perf levers (rigid/affine) | [`registration-matrix-recipe-perf-levers`](registration-matrix-recipe-perf-levers.md) | logged/measured |
| Pallas ESM / SVF force kernel (bandwidth-bound family) | [`pallas-demons-esm-force`](pallas-demons-esm-force.md) | promoted, gate (b) pending |
| Hopper/Blackwell registration kernels (Mosaic GPU) | [`mosaic-hopper-registration-kernels`](mosaic-hopper-registration-kernels.md) | hardware-blocked |
| `jnp.percentile` CPU sort cliff (general) | [`median-percentile-cpu-sort-cliff`](median-percentile-cpu-sort-cliff.md) | open — D3 is the *registration hot-loop* application |
| Differentiable recipe (implicit-VJP wrapper) | [`registration-recipe-transparent-differentiability`](registration-recipe-transparent-differentiability.md) | proposed — interacts with B2 |
| In-memory transform algebra (compose/invert/fuse) | [`affine-matrix-algebra`](affine-matrix-algebra.md) | ENABLING — A1's `fuse` consumes it |
| `register.regulariser` field penalties | [`field-regularisers`](field-regularisers.md) | ENABLING |
| Demons 0/0 → NaN; affine small-grid divergence | [`register-demons-force-divide-by-zero`](register-demons-force-divide-by-zero.md), [`register-affine-small-grid-divergence`](register-affine-small-grid-divergence.md) | RESOLVED (so F3 is the *remaining* guards only) |
| Perf wins must certify at brain scale; interp backend gap | [`perf-wins-must-certify-at-scale`](perf-wins-must-certify-at-scale.md), [`interpolation-backend-cpu-gpu-gap`](interpolation-backend-cpu-gpu-gap.md) | open — E1's scale discipline |

## 5. New gaps — the roadmap

Each item: **What · Why (code-grounded) · Change · Impact/Effort · Acceptance ·
Lens.** All verified against the live `src/nitrix/register` surface. Backward
compatibility is **not** a constraint.

### Workstream A — Public API & ergonomics

**A1. `syn_pipeline` — a composed, staged rigid→affine→SyN recipe.** · *high / medium · lens: community*
- **Why.** `register/__init__.py` exports only the leaf recipes; `affine_register`'s
  own docstring (`recipes.py:423–425`) tells users to *manually* extend rigid
  params with a zero linear block, and `greedy_syn_register` takes `init_affine`
  but nothing stages it. The most-run neuroimaging command
  (`antsRegistrationSyN.sh`) therefore requires hand-chaining three recipes plus
  rigid→affine param surgery plus a manual fuse — and downstream tools get the
  staging subtly wrong (we hit the coarsest-level translation-rescale foot-gun
  ourselves while validating affine multi-start).
- **Change.** Add `register.syn_pipeline(moving, fixed, *, transform='r'|'a'|'s', spec=...)`
  → `PipelineResult` that stages rigid (moment init) → affine (warm-started via
  `_affine_params_from_matrix`, MI default) → `greedy_syn` (`init_affine`, LNCC
  default), threads each stage's matrix forward, and returns a fused composite
  (via [`affine-matrix-algebra`](affine-matrix-algebra.md)'s fuse) plus per-stage
  cost histories, with `antsRegistrationSyN`-matching per-stage metric defaults.
- **Acceptance.** One call recovers a planted rigid+affine+nonlinear chain on a
  textured/real volume to the same accuracy as the hand-staged sequence; the
  returned composite, applied via A2, reproduces `result.warped`.
- **Depends on:** B1 (self-contained `.matrix` makes the composite unambiguous).

**A2. `apply_transform` — apply a recovered transform to an in-memory array/label.** · *high / low–medium · lens: community*
- **Why.** Recipes return only the warped *moving intensity*. Applying the
  recovered transform to a label map / atlas / second contrast means
  hand-assembling `affine_grid` + `spatial_transform` and knowing to pass
  `MultiLabel`/`NearestNeighbour` (a silent correctness bug otherwise). This is
  in-memory resampling — **numerics, not I/O.**
- **Change.** `register.apply_transform(image, result, *, reference_shape=None, method=None)`
  dispatching matrix-vs-displacement, defaulting to `MultiLabel` for
  integer/label input and `Linear` otherwise. Cross-reference the existing
  `metrics.dice`/`jaccard` in the docstring for overlap evaluation (do **not**
  re-implement them).
- **Acceptance.** Round-trips `result.warped`; label input stays integer-valued
  (no interpolation bleed); honours `reference_shape` for grid changes.

**A3. Masked / weighted cost through `Metric`, with `mask=` on every recipe.** · *high / medium · lens: community*
- **Why.** Real registration is masked (brain mask, lesion exclusion, FoV).
  The `Metric` ADT (`_metric.py`) computes the cost over the full array; there
  is no way to restrict/weight it, so users pre-mask images (changing the
  histogram for MI/CR) instead of masking the cost.
- **Change.** Thread an optional per-voxel weight/mask into `Metric.cost` (and
  the matching `Force`), expose `mask=`/`moving_mask=` on the recipes; for
  histogram metrics the mask must gate the scatter, not just the reduction.
- **Acceptance.** A masked SSD/LNCC/MI cost ignores out-of-mask voxels (parity
  vs cropping for SSD; histogram excludes masked voxels for MI); recovery on a
  partially-corrupted image improves with the corrupt region masked.

### Workstream B — Convention correctness

**B1. Make `RegistrationResult.matrix` self-contained and uniform across spaces.** · *high / medium · lens: design + community*
- **Why.** `_IndexSampler.result_transform` (`_space.py:145–148`) is a
  pass-through — the `(shape−1)/2` centering is supplied to `affine_grid` but
  **not** baked into the returned matrix, whereas `_WorldSampler` bakes
  `t_pos @ T @ t_neg` (`_space.py:188`). So the *same* field has two application
  conventions: an `IndexSpace` `.matrix` mis-warps under a plain `apply_affine`,
  a `WorldSpace` one does not, and composing the two silently mis-centers — a
  sharp foot-gun for exactly the A1/A2 composition workflow. (We documented this
  in `registration.md` during the recovery round; this is the fix.)
- **Change.** `_IndexSampler.result_transform` bakes the centering in: capture
  full-res `(shape−1)/2` at sampler construction and return `T_pos @ transform @ T_neg`,
  so `IndexSpace` and `WorldSpace` `.matrix` share `apply_affine(coords, matrix)`
  semantics. Keep the raw about-centre params on `RegistrationResult.params`.
  Extract a `_conjugate_about(T, center)` helper shared by `_WorldSampler` and
  BBR (BBR keeps its boundary-centroid centre — intentionally different, not a
  duplicate).
- **Acceptance.** For both spaces, `apply_transform(moving, result)` reproduces
  `result.warped`; composing an `IndexSpace` and a `WorldSpace` result is
  geometrically correct.

**B2. Collapse the `convergence='auto'` polysemy into orthogonal fields.** · *high / medium · lens: design*
- **Why.** `convergence='auto'` resolves to **early-exit** on IC/SSD, **raises**
  on MI-forward, and **scan** on BBR (`_bbr.py:453`) — three meanings for one
  sentinel, validated at three scattered guard sites.
- **Change.** Replace `Union[Convergence, None, 'auto']` with two orthogonal
  Spec fields: `mode: Literal['fixed','early_exit']` (default `'fixed'`,
  differentiable everywhere) and `convergence: Convergence` (threshold/window,
  used only when `mode='early_exit'`); validate eligibility in **one**
  `_validate_convergence_mode` site; set `early_exit` as the explicit IC-path
  default. Keep `_early_exit_barrier` (the runtime reverse-mode guard).
- **Acceptance.** Identical behaviour selectable explicitly across all recipes;
  one validation site; the `_converge`/recipe docstrings agree.
- **Interacts with:** [`registration-recipe-transparent-differentiability`](registration-recipe-transparent-differentiability.md).

### Workstream C — Cross-modal accuracy

**C1. Closed-form NMI force (`nmi_grad`) + `MIForce(normalized=True)`.** · *high / medium · lens: math*
- **Why.** NMI is the preferred deformable cross-modal criterion (ANTs default
  for many cross-modal tasks). `MIForce`/`mi_grad` (`metrics/information.py`)
  is **unnormalised Mattes MI only**; NMI registration falls back to the generic
  `MetricForce(MI(normalized=True))` autodiff path — slow *and* on the
  non-deterministic scatter (E2).
- **Change.** Derive and ship the closed-form NMI gradient (`nmi_grad`, the
  quotient-rule form over the joint/marginal entropies) and expose
  `MIForce(normalized=True)` routing to it; parity-test vs `jax.grad` on
  populated bins (the same oracle `mi_grad` uses).
- **Acceptance.** `MIForce(normalized=True)` is `jax.grad`-parity on populated
  bins and recovers a planted cross-modal warp at SyN/Demons quality.

### Workstream D — Hot-loop performance

**D1. Fuse the cost into the force computation in the SVF drivers.** · *high / medium · lens: performance + math*
- **Why.** `diffeomorphic.py`/`_syn.py` compute the per-iteration cost from
  scratch *alongside* the force (duplicate box-sums for LNCC, duplicate
  histogram scatter for MI) — roughly double the per-iteration metric work.
- **Change.** Have the `Force` return (or share) the metric value it already
  computes internally, so the convergence trace reuses it; one metric evaluation
  per iteration.
- **Acceptance.** Identical recovery + cost trace; measurable per-iteration
  speedup (LNCC/MI), no extra compile.

**D2. Key the symmetric early-exit on the *total* cost.** *(quick win)* · *medium / low · lens: math + performance*
- **Why.** `symmetric_level`/`group_symmetric_level` return the **forward-half**
  cost for the convergence test; for an asymmetric metric (MI/CR) the inverse
  half can still be improving when the forward plateaus → premature stop.
- **Change.** Return `0.5*(bound_fwd.cost(a) + bound_inv.cost(b))` — one line
  each; a no-op for SSD/LNCC, a real fix for MI/CR.
- **Acceptance.** SSD/LNCC byte-identical; MI/CR symmetric runs no longer
  early-stop with the inverse half unconverged.

**D3. Replace the per-iteration `jnp.percentile` with `lax.top_k`.** *(quick win)* · *medium / low · lens: performance*
- **Why.** `_normalise_step` calls `jnp.percentile(norm, 99)` every SVF
  iteration — a full O(N log N) sort per step (the registration consumer of the
  general [`median-percentile-cpu-sort-cliff`](median-percentile-cpu-sort-cliff.md) cliff).
- **Change.** `lax.top_k(norm.ravel(), ceil(0.01*N))[0].min()` — static `k`,
  jit-safe, identical semantics, removes the sort.
- **Acceptance.** Identical normalisation; no full sort in the per-iteration HLO.

**D4. Reduce algebra-mode SVF re-integration; extend batch-aggregate early-exit to a cohort SVF driver.** · *medium / high · lens: performance*
- **Why.** Algebra-mode Demons/SyN re-integrate the velocity field
  (`integrate_velocity_field`, scaling-and-squaring) every iteration; and the
  batch-aggregate early-exit (shipped for volreg) is not available to a cohort
  SVF driver. (Sibling of [`registration-early-stopping-while-loop`](registration-early-stopping-while-loop.md).)
- **Change.** Cache/incrementally update the integrated field where the velocity
  delta is small; add a `vmap`'d cohort SVF driver honouring the
  all-lanes-exit `while_loop` (the volreg pattern). Fix the stale
  "not vmap-batchable" docstrings (`_converge.py:14–19`, `_syn.py:136`) — volreg
  already proves it works.
- **Acceptance.** Cohort SVF recovers per-subject warps with adaptive early-exit;
  algebra-mode per-iteration integration cost drops with unchanged recovery.

### Workstream E — GPU / float32 rigour & determinism

**E1. Parametrize core recovery tests over dtype; add a float32 conditioning guard.** · *high / medium · lens: engineering + gpu*
- **Why.** All 12 registration test files enable x64 and build float64 inputs —
  **zero** dtype parametrization — while float32/GPU is the declared production
  path. A real conditioning regression would pass CI silently. (Scale discipline:
  [`perf-wins-must-certify-at-scale`](perf-wins-must-certify-at-scale.md).)
- **Change.** Parametrize the IC-vs-forward parity and hard-affine recovery tests
  (`test_ic_dispatch.py`) over `dtype ∈ {float32, float64}` with
  dtype-appropriate tolerances; add one float32 conditioning guard. Derive the
  dtype-unaware constants (`1e-12` in `_trust_scale`, `_RMS_EPS` in `_force.py`)
  from `jnp.finfo(dtype)`.
- **Acceptance.** float32 recovery within tolerance in CI; no hard-coded
  float64-only epsilons on the hot paths.

**E2. Deterministic joint-histogram path (one-hot matmul) at affine sizes + a GPU-gated MI determinism/recovery test.** · *high / medium · lens: gpu + engineering*
- **Why.** The joint-histogram scatter (`information.py:67–71`, `.at[idx].add`)
  is the GPU non-determinism source, mitigated **only** by `affine_register(restarts=k)`
  — a 4–6× compute multiplier papering over one non-associative scatter, with no
  determinism contract or GPU-gated test. At coarse affine pyramid sizes
  (N ≲ 200K) a one-hot-matmul histogram (`onehot_m.T @ onehot_f`) is **faster
  *and* deterministic** — it removes the failure mode at its source and demotes
  `restarts` to a pure basin-of-attraction lever.
- **Change.** Add a size-gated deterministic path in `_joint_hist_from_softbins`:
  one-hot-matmul below ~200K voxels, scatter-add above; auto-engage for the
  affine-MI forward dispatch. Add a GPU-gated test asserting affine-MI recovery
  determinism (and that NMI/C1 inherits it).
- **Acceptance.** Affine-MI on GPU is deterministic and recovers without
  `restarts`; the determinism test runs on the L4.

**E3. Remat the differentiable SVF scan; fix the latent float64 literal; document Pallas tileability.** · *medium / medium · lens: gpu*
- **Why.** The differentiable (fixed-`scan`) SVF path materialises the full
  trajectory for reverse-mode (memory); `_grid_scale` silently downcasts a
  `jnp.float64` literal to float32; the Pallas kernels' tileability is
  undocumented. (Kernels themselves: [`pallas-demons-esm-force`](pallas-demons-esm-force.md),
  [`mosaic-hopper-registration-kernels`](mosaic-hopper-registration-kernels.md).)
- **Change.** `jax.checkpoint` the SVF scan body on the differentiable path;
  compute `_grid_scale` in pure Python over static int shapes (no float64
  literal); add a tileability note to the kernel docs.
- **Acceptance.** `grad` through the fixed-scan SVF path with bounded memory; no
  float64 literal in the float32 graph.

### Workstream F — Numerical robustness

**F1. Cost-decrease (Armijo/LM) guard on the single-pair IC step.** · *medium / medium · lens: math*
- **Why.** The IC Gauss-Newton step has a trust-region clamp
  ([`register-affine-small-grid-divergence`](register-affine-small-grid-divergence.md))
  but no *cost-decrease* check — a clamped step can still increase the cost on a
  hard case (no monotonicity guarantee).
- **Change.** Add a backtracking/LM damping that rejects a non-decreasing step
  (the dense-numpy reference accepts only cost-decreasing steps).
- **Acceptance.** Monotone cost on the IC path; no recovery regression on the
  easy cases (byte-unchanged where the step already decreased).

**F2. Align/guard the SVF integration step count.** · *medium / low · lens: math*
- **Why.** `integrate_velocity_field` (scaling-and-squaring) uses a fixed step
  count; for a large velocity the squaring count may be too low (integration
  error / folding) and is not tied to the velocity magnitude.
- **Change.** Derive the squaring count from `max|v|` (the standard
  `ceil(log2(max|v|/0.5))` rule) or document/guard the fixed choice; assert the
  resulting field stays diffeomorphic.
- **Acceptance.** Jacobian stays positive across the tested warp-magnitude range;
  integration error bounded.

**F3. Remaining robustness guards + the genuine test-matrix holes.** · *medium / low · lens: engineering*
- **Why.** Smaller, verified gaps (the demons-NaN and small-grid items are
  already RESOLVED, so they are excluded): `_bbr._grid_seed` `argmin` is not
  NaN-safe; three recipes (volreg, demons, bbr) lack a jit-wrapped smoke test
  (rigid/affine/SyN have them).
- **Change.** NaN-safe grid seed (`argmin` over `where(isnan(costs), inf, costs)`);
  add jit-wrapped smoke tests for volreg/demons/bbr.
- **Acceptance.** Grid seed robust to NaN candidates; all recipes have a
  jit-coverage test.

### Workstream G — Design consolidation

**G1. Consolidate the Spec / dispatch / ownership seams.** · *medium / high · lens: design*
- **Why.** `RegistrationSpec` / `DemonsSpec` / `SyNSpec` / `BBRSpec` duplicate
  the shared schedule fields (`levels`, `iterations`, `pyramid_factor`,
  `pyramid_sigma`); the IC-vs-forward dispatch is decided at scattered sites;
  `Convergence` lives in `_core` but is a cross-cutting concern; `init`/`space`/
  `restarts` are exposed inconsistently across recipes. (Builds on the shipped
  ADT work, [`registration-typing-metric-adt`](registration-typing-metric-adt.md).)
- **Change.** A shared `ScheduleSpec` base the recipe Specs embed; one
  `resolve_ic_dispatch(...)` resolver; relocate `Convergence` to `_converge`;
  unify `init`/`space`/`restarts` exposure (restarts generalises to any forward
  recipe, not just affine). Backward-compat irrelevant — pick the clean API.
- **Acceptance.** Adding a metric/force/model/space touches one place each;
  recipe signatures are uniform; no duplicated schedule docstrings.

## 6. Quick wins (one-liners, non-I/O)

- **D2** symmetric early-exit total cost — one line each in
  `symmetric_level`/`group_symmetric_level`.
- **D3** `percentile → top_k` in `_normalise_step` — removes a per-iteration sort.
- **F3** NaN-safe `_bbr._grid_seed` argmin.
- **E1/E3** dtype-match `_trust_scale` `1e-12` + `_force.py` `_RMS_EPS` to
  `jnp.finfo(dtype)`; pure-Python static `_grid_scale` (drop the float64 literal).
- **D4** fix the stale "not vmap-batchable" docstrings (`_converge.py:14–19`,
  `_syn.py:136`) — volreg disproves them.
- **Docs** metric-choice + anisotropy guidance on `rigid_register`/`affine_register`
  (SSD within-modality; MI/CR/LNCC cross-modal; `IndexSpace` assumes a shared
  isotropic grid) and jit-wrapped smoke tests for volreg/demons/bbr.

## 7. Sequencing

1. **B1 → A1/A2.** The self-contained `.matrix` (B1) unblocks unambiguous
   composition, so land it before the pipeline/apply layer.
2. **Quick wins (§6) anytime** — independent, low-risk, mostly one-liners.
3. **E2 then C1.** Deterministic histogram first; NMI force inherits it.
4. **A3, D1, E1** are independent and high-value — parallelisable.
5. **G1** is the largest refactor; do it once A/B have settled the public API so
   the consolidation targets the final surface.

## 8. Source

Full verified findings (per-dimension, with the adversarial verdicts and
refined recommendations) backing this roadmap: the 2026-06-22 six-lens audit
(`/scratch/regprof/registration-audit-2026-06.md`).

## 9. New capability — functional alignment (representation-space)

Owned by [`register-functional-alignment`](register-functional-alignment.md)
(do not duplicate here). Extends the suite's scope from **spatial** registration
to **alignment in representation space**: a closed-form orthogonal-Procrustes
aligner with an optional **matrix von Mises–Fisher** (matrix-Langevin) spatial
prior on the rotation (the ProMises model), on the §6.5 `fit`/`apply` seam, with
a **method ADT** (ProMises now; other hyperalignment algorithms follow) so
``functional_align`` is a family, not a synonym for one algorithm.  A sibling
recipe family — *not* a `TransformModel` chart (closed-form, not an `exp`-map
optimised by GN/LM) and *not* a `CoordinateSpace` (feature space, not
voxel/world). Reframes `register/__init__.py` from "pairwise registration
recipes" to "pairwise **alignment** recipes (spatial + functional)". Solver
dependency: [`linalg-orthogonal-procrustes`](linalg-orthogonal-procrustes.md)
(shipped). The whole-brain regime needs the **efficient ProMises** subspace
method (tracked in the FR §6). Migrated theory-faithfully (not ported) from the
deprecated `hypercoil-examples` repo — see the
[migration ledger](hypercoil-examples-migration.md). Numerics-only.
