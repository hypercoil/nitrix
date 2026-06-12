# Registration suite v4 — brain-scale performance + fMRIPrep parity (forces, kernels, greedy driver)

> **Status (2026-06-11): PROPOSED — comprehensive plan, no code yet.** The
> fourth registration block, on top of the shipped v3 suite (Force keystone,
> transform algebra, matrix IC/early-exit perf levers, masks; merged to local
> `main` `356c768`). Two intertwined threads: **(1) make the GPU wins real at
> brain scale** (256³ single + cohort), closing the economic gap the perf agent
> flagged (floor 4×, target 20× vs ANTs-CPU; CPU no longer times out); and **(2)
> reach fMRIPrep / `antsRegistration` feature parity**, headlined by a
> **closed-form MI force**. Folds in the post-v3 review
> (`registration-suite-v3-followups.md`, items A/B/C/D) and the standing perf
> ledger (`registration-matrix-recipe-perf-levers`, `pallas-demons-esm-force`,
> `perf-wins-must-certify-at-scale`, `interpolation-backend-cpu-gpu-gap`).

Reads on top of [`registration-suite-v3.md`](registration-suite-v3.md) (V1–V5),
[`registration-suite-v2.md`](registration-suite-v2.md) (R4–R8), and
[`registration.md`](registration.md) (R0–R3).

## 0. The diagnosis (why v3 did not clear the bar)

The hot kernel under every recipe is the **interpolation gather**. Recipe cost
tracks (i) how many gathers per iteration, (ii) the HBM each touches, and (iii)
the iteration count. Against that lens, v3's shipped levers fall short for five
separable, concrete reasons — all structural, which is why a *qualitative* (not
incremental) jump is on the table:

1. **Affine IC diverges at small coarse grids**
   (`register-affine-small-grid-divergence`). Root cause:
   `_inverse_compositional._ic_level` is pure Gauss-Newton with a fixed ridge and
   **no accept/reject**, unlike the forward `levenberg_marquardt`. On an
   ill-conditioned ≤14³ level the full GN step overshoots and `matrix_exp(−Δθ)`
   blows the linear block up; the across-level translation rescale compounds it.
   So affine does not actually get the IC win — it anti-correlates.
2. **The rigid/affine IC win erodes at scale** (7.1×→3.7× by 128³, worse at
   256³). `_ic_level` re-reads the full `(M, P)` steepest-descent buffer every
   iteration (`h_inv @ (sd.T @ err)`); at 256³ affine that is ~800 MB re-read per
   step plus a large resident buffer — the certify-at-scale wall.
3. **Early-exit is implemented but off by default.** The compound-payoff
   projection assumed lever B engaged; by default single-pair rigid runs the full
   `30 × 3` iterations though it converges in ~5/20 even on hard cases.
4. **Demons/SyN scale super-linearly** (96³→160³ time exponent 1.37; speedup
   erodes 43×→28×). Algorithmically O(N) per registration, so this is the HBM
   wall (~3 KB/voxel/iter over several `d`-component fields at low arithmetic
   intensity) **plus** a structural redundancy (item 5).
5. **The diffeomorphic recipes re-exponentiate from scratch every iteration.**
   `_svf.symmetric_level` / `single_sided_level` call
   `integrate_velocity_field(v, n_steps=5/6)` for **each** field **every**
   iteration — ~12 gathers/iter (SyN), ~7 (Demons). But these are documented as
   *greedy* recipes, and greedy SyN composes incremental smoothed displacement
   updates (~2–4 gathers/iter); it does not maintain a stationary velocity it
   re-exponentiates. The implementation pays the geodesic-SVF cost while
   delivering the greedy algorithm.
6. **Demons returns all-NaN on every real image**
   (`register-demons-force-divide-by-zero`): the unguarded ESM `diff/denom` is
   `0/0` on uniform background, so no Demons perf number on real anatomy is
   currently meaningful.

The unifying theme: **the suite does far more gathers than necessary, and the
survivors are not bandwidth-efficient.** v4 = cut redundant gathers
(structural), make the survivors bandwidth-efficient (reformulation + Pallas),
cut the iteration count (default early-exit), and — orthogonally — close the
fMRIPrep feature gaps.

## 1. Force-layer HBM analysis (ESM vs LNCC vs MI — which warrant kernels)

The perf agent flagged the ESM force as HBM-bound and scoped a Pallas kernel
(`pallas-demons-esm-force`). The natural question (user, 2026-06-11): do **LNCC**
and **MI** carry the analogous inefficiency? They do — but with *different
shapes*, so they warrant *different* kernels and different priority. The cost
laws (per force evaluation, d = 3, "round-trip" = a full-field HBM read+write):

| Force | Dominant traffic | Round-trips / eval | Shape | Kernel verdict |
|---|---|---|---|---|
| **ESM (Demons)** | `∇warped` stencil + `j`/`denom`/`u` elementwise + 2 Gaussian smooths | ~5–6 | stencil + elementwise + channel-reduce | **gather-free fused stencil+force** (already scoped). High tractability. |
| **LNCC (SyN)** | **9 separable box sums** (5 windowed stats + 4 re-sums) + `spatial_gradient` + assembly | **~35+** (27 conv passes) | two halo'd stencil-reduce stages | **the worst** — warrants both an algorithmic fix *and* a fused windowed-stats kernel. |
| **MI (Mattes)** | joint-histogram **scatter** (N→B²) + per-voxel bin-derivative gather + ∇warped | low field volume, but a **scatter** (+ autodiff tape if not closed-form) | scatter-reduce + tiny-table gather | **closed-form first** (removes the tape — the real win); a shared-memory histogram+gradient kernel is a distinct, lower-priority second. |

**LNCC is the most HBM-bound of the three** — ~8× the ESM traffic — because
`lncc_grad` runs ~27 separable-conv passes, each a full-field round-trip XLA does
not fuse across axes (each `correlate1d` is its own op with a `moveaxis`). It is
the dominant SyN per-iteration cost. Three tiers of fix, increasing depth:

- **(L-i) Separable integral-image box sums.** Replace each
  `correlate1d(ones(2r+1))` with a per-axis cumsum-difference (`S[i+r] −
  S[i−r−1]`): **O(N), radius-independent** (today O(N·(2r+1))). Because the box
  filter is already separable, the running sum is 1-D (bounded by axis length, ~
  256·max ≈ 2.6e5), so fp32 cancellation is **not** the brain-scale hazard the
  full 3-D summed-area table would be — numerically safe at the default radii.
  Makes the larger ANTs windows free. Lands in `metrics/_common._box_sum`;
  benefits `lncc`/`lncc_grad` everywhere; the conv path stays the parity oracle.
- **(L-ii) Hoist the fixed-only windowed sums in `bind`.** `_BoundLNCC.bind`
  precomputes **nothing** today; `sum_f`, `sum_ff`, `var_f`, `fbar` depend only
  on `fixed` and are recomputed every iteration. For the **single-sided** path
  (Demons-with-LNCC) `fixed` is constant → hoist them (≈40% of the stage-1
  sums). *Not* available to symmetric SyN, whose "fixed" is the other image at
  the moving midpoint (changes per step) — state that scope precisely.
- **(L-iii) Fused windowed-statistics Pallas kernel.** Two halo'd stencil-reduce
  passes (halo = radius): pass 1 accumulates the 5 windowed sums from
  (warped, fixed) per tile; pass 2 re-sums (p, q) and assembles with `∇warped` —
  ~2 passes vs ~35 round-trips. The HBM endgame; gated like all Pallas work.

**MI's inefficiency is the autodiff-through-scatter tape, not field round-trips.**
`MetricForce(MI())` runs `jax.grad` of the soft-histogram cost per iteration —
differentiating through the `scatter_add` that builds the B² joint histogram,
materialising a backward tape. The closed-form Mattes gradient (§2) removes the
tape (exactly as `lncc_grad` beats `jax.grad(lncc)`); the residual scatter
(low-volume, reduces N→B²) is the only HBM-awkward piece, addressed by an
optional fused shared-memory histogram+gradient kernel — distinct in shape from
the stencil kernels, and the lowest kernel priority because the closed form
already captures most of the win.

## 2. Phase plan

Seven phases. Phase 0 is a hard gate (benchmarks lie until it lands); Phases 1–4
are the substance and parallelise; Phase 5 (Pallas) is profile-gated; Phase 6
consolidates. Each item cites the followup it folds in. Standing discipline
(every phase): pure-functional surface, frozen/`NamedTuple`, Protocols where they
earn it, jaxtyping, ruff + ruff-format + mypy, a **fast-path-vs-generic parity
test**, and a **scaling case certified to brain scale** (single + cohort) with a
stated cost law (`perf-wins-must-certify-at-scale`).

### Phase 0 — Correctness & honesty gates (blocking)

| Item | Fix | Home | Fold-in |
|---|---|---|---|
| 0a | Epsilon-guard the ESM denom (`where(denom>eps, diff/denom, 0)` — zero force where no gradient + no mismatch); audit LNCC/`MetricForce` for the same 0/0 | `_force.py` | `register-demons-force-divide-by-zero` |
| 0b | Cost-guarded (LM accept/reject or backtracking) IC step — recompute cost after the trial compositional update, reject/grow ridge on increase | `_inverse_compositional._ic_level(_converge)` | `register-affine-small-grid-divergence` |
| 0c | `MetricForce` magnitude normalisation for non-spatial-mean metrics (RMS/max-norm) so an unclamped Demons driver gets a controlled MI/CR step | `_force.py` | B2 |
| 0d | `_normalise_step`: global-max → RMS / high-percentile (one outlier no longer starves the field) | `_svf._normalise_step` | B4 |
| 0e | `invert_displacement` robustness — Anderson/Newton acceleration + residual check; large-deformation SyN test | `geometry/deformation.py` | B5 |
| 0f | `transform_mean` / `matrix_log` convergence + domain guards (tangent-norm tolerance; negative-eigenvalue / Taylor-radius guard; DB balancing) | `geometry/algebra.py`, `linalg/matrix_function.py` | B3 |
| 0g | Honest docs: "ANTs-*style*, synthetic-recovery-validated"; real-data/ANTs-ref parity delegated to perf-bench | design docs + `register` docstring | D3 |

0a + 0b are the load-bearing pair: 0a makes Demons usable on real anatomy at all;
0b makes affine reliable *and* is the precondition for default early-exit (Phase
3b). Both are XS–S effort, low risk.

### Phase 1 — The force layer (metric↔force completion + fMRIPrep's MI)

The keystone of v4. Completes the Tier-1 fast-force set so every fMRIPrep metric
has a closed-form force; `MetricForce` stays the escape hatch / parity oracle.

**1a — Closed-form MI force (the headline; B1).** fMRIPrep's cross-modal metric
is Mattes MI; today it is only reachable via the autodiff escape hatch. Add the
analytic gradient. For the linear-Parzen joint histogram the suite already forms
(`joint_histogram`), the per-voxel derivative has the clean Mattes form (the
+1 and `P_f` terms vanish because the soft weights sum to 1, so
`Σ_a ∂β_m(x,a)/∂m = 0`):

```
∂MI/∂m(x) = (1/N) Σ_{a,b} ∂β_m(x,a)/∂m · β_f(x,b) · log( P[a,b] / P_m[a] )
```

i.e. a per-voxel sum over the ≤4 touched bin pairs of `(±s)·β_f·W[a,b]`, reading
from the small precomputed `W[a,b] = log((P[a,b]+ε)/(P_m[a]+ε))` table
(`s = (bins−1)/span_m`). No tape through the scatter.

- **Deliver:** `metrics.information.mi_grad(moving, fixed, *, bins, value_range,
  normalized)` + a Tier-1 `MIForce(bins, value_range, normalized)` in `_force.py`
  (wraps `mi_grad · ∇warped`). **Pin the histogram range** (compute once from the
  full-res images, thread per level) — a force with a drifting data-min/max range
  is ill-posed (this *is* followup A6, mandatory here, not optional).
- **Gate:** `mi_grad == jax.grad(mutual_information)` to tolerance (pinned
  range); `MIForce(MI)` drives a cross-modal (T1↔T2-style) deformable recovery to
  convergence; NMI variant covered. Ships a SyN-MI scaling case.
- **Why:** converts the multimodal headline from "works but slow (escape hatch)"
  to a first-class fast path — the metric fMRIPrep actually runs.

**1b — Closed-form CR gradient + `CRForce` (B1 tail).** Same construction for
Roche's η² (the `fixed`-binned group-variance ratio); `cr_grad` +
`CRForce(bins, value_range)`. Lower priority than MI (FSL-lineage, no SimpleITK
co-oracle), but cheap once 1a's machinery exists.

**1c — LNCC force efficiency (L-i + L-ii from §1).** Separable integral-image box
sums (`metrics/_common`) + hoist the fixed-only windowed sums in
`_BoundLNCC.bind`. Algorithmic; benefits every LNCC consumer; conv path is the
oracle.

**1d — `SumForce` (A5).** `SumForce([(w1, f1), (w2, f2), …])` implementer:
`update = Σ wᵢ·fᵢ.update`, `cost = Σ wᵢ·fᵢ.cost`. Composes with the protocol, no
driver change — gives ANTs' in-stage multi-metric summation (`-m MI -m CC`).

**1e — Restrict-deformation (A7).** A length-`ndim` `restrict` weight multiplied
into the force after masking (reuse `_svf._mask_force`) — ANTs
`--restrict-deformation 1x1x0`.

### Phase 2 — Greedy dense-field driver (the structural perf win, P1a)

The largest single lever. Restructure the `_svf` inner loop to carry the
**displacement** field(s) and compose an incremental smoothed update per
iteration, instead of re-exponentiating a stationary velocity — ~12→~2–4
gathers/iter for SyN. Lands as a fast implementer **behind the existing
`level_solve` seam** (the §2-v3 "stratify, don't choose" thesis): the log-domain
SVF path stays available as the parity oracle, for the velocity-field outputs,
and as the future geodesic-LDDMM base.

- **2a — Greedy composition loop.** Per iteration: warp moving by `(id+s_fwd)` and
  fixed by `(id+s_inv)` to the midpoint (1 gather each), compute the force
  (Phase 1), fluid-smooth (the regulariser's Green's function), compose the small
  smoothed step onto each displacement, periodically re-symmetrise. Diffeomorphism
  preserved as ANTS does (compose small diffeos `id+ε·smoothed`; one
  `invert_displacement` per re-symmetrise — leans on 0e). Directly flattens the
  super-linear GPU scaling **and** the CPU timeout (the CPU lag is precisely the
  iterated `integrate_velocity_field`, `interpolation-backend-cpu-gpu-gap`).
- **2b — Adaptive `n_steps` + reconcile the 5/6/7 mismatch (C5).** Scale the
  final/residual integration steps to `‖v‖_max`; one documented `n_steps`
  convention; fix the test ground-truth that builds warps with the default 7
  while recipes integrate with 5/6.
- **2c — SVF early-exit (A3 / P2b).** `DemonsSpec.convergence` /
  `SyNSpec.convergence` + a `while_loop` variant of the level drivers using the
  shared windowed-slope test (Phase 6 C7). Single-pair only (breaks `vmap`).
  Re-derive clamp-vs-scale here (a convergence gate makes ANTS scale-to-step
  viable again).
- **Gate:** greedy == log-SVF recovery to tolerance (parity oracle); diffeomorphism
  (all `det J > 0`) preserved; inversion residual bounded under large deformation;
  SyN/Demons scaling re-certified to 256³ with the new cost law (target: ≤ linear).

### Phase 3 — Matrix path at brain scale

- **3a — IC moment reformulation (P1c).** Eliminate the `(M, P)` steepest-descent
  buffer. The per-iteration projection `SDᵀe` is exactly the moment tensor
  `m_ij = Σ_x ∇F_i(x)·(x−c)_j·e(x)` (affine linear block) + `Σ_x ∇F_i·e`
  (translation), with the rigid rotation columns a fixed contraction of the same
  moments against the generators. Store only `∇F` `(M, ndim)` per level; build `H`
  once from the same moments; compute `SDᵀe` per iteration as a fused reduction
  reading `∇F` + `e`, writing P scalars. Cuts resident memory `(M,P)→(M,ndim)` and
  per-iteration bandwidth `M·P → M·(ndim+1)` (3× affine). **Exact** — current IC
  is its own parity oracle. Recovers the eroded 7.1×→3.7× and certifies the win at
  256³.
- **3b — Default-on early-exit, single-pair (P2a).** Make `while_loop` early-exit
  the default for the single-pair matrix path (already implemented). Multiplies
  with 3a. Requires 0b (guarded step). Differentiable-layer entry point stays the
  trajectory-independent implicit path; cohort/`vmap` keeps the fixed `scan`. Also
  fixes **C3** (raise when `convergence` is set on a path that cannot honour it).
- **3c — Affine multi-start / moment init (A8).** A cheap centre-of-mass / moment
  initialisation (and an optional small best-of-N rotation search) before the
  affine optimise — the affine basin is narrow and a single zero start fails
  silently on a large misalignment.

### Phase 4 — Preprocessing parity (fMRIPrep front-end)

Cheap, independent, high parity-value; all opt-in, default-off (byte-unchanged).

- **4a — Winsorization (A1, highest parity value).** `metrics.winsorize(image,
  lower=0.005, upper=0.995)` (percentile clip) + a `winsorize` spec knob applied
  before the pyramid. fMRIPrep runs this on every call; it also mitigates the
  `_normalise_step` outlier sensitivity (0d).
- **4b — Histogram matching (A2).** `metrics.match_histogram(moving, reference,
  bins)` (CDF / quantile transport) + recipe knob, for within-modality stages.
- **4c — Decouple smoothing from shrink (A4).** Per-level `smoothing_sigma`
  schedule independent of the shrink factor (ANTs `-f 4x2x1 -s 2x1x0vox`);
  default reproduces today. (A6 range-pinning already lands with 1a.)

### Phase 5 — Custom kernels (Pallas; profile-gated)

Build only when a profile confirms the op dominates per-iteration HBM **and** a
consumer is bottlenecked (`pallas-demons-esm-force` Trigger). Each registers a
`custom_vjp` and keeps the pure-JAX path as floor + oracle. Priority order tracks
§1's verdict:

- **5a — Fused ESM stencil+force** (`pallas-demons-esm-force`). Gather-free →
  dodges the parked-trilinear ELL gather-lowering blocker. Reuses the separable
  Gaussian kernel for steps 5/7.
- **5b — Fused LNCC windowed-statistics kernel** (L-iii). The largest HBM to
  recover; two halo'd stencil-reduce passes. Build after Phase 1c (the algorithmic
  win may suffice for many sizes; the kernel is the brain-scale endgame).
- **5c — Fused MI histogram+gradient kernel.** Shared-memory partial histograms +
  the bin-derivative gather. Lowest priority (the Phase-1a closed form captures
  most of the win; ANTs-style histogram subsampling is an optional further lever).
- **5d — Trilinear gather** (`pallas-trilinear-resample`). The universal warp
  kernel; **blocked** on pinned-JAX gather lowering. The algorithmic levers
  (Phase 2, 3a) cut gather *count* and are higher-ROI until it unblocks. Tracked,
  not committed.

### Phase 6 — Abstraction consolidation + validation hygiene

Threads through; close the round here.

- **C1/C2 — Unify the fast/slow dispatch.** Lift the matrix `method=` string to an
  ADT mirroring `Force`, or at minimum share `resolve_ic_method` between
  `recipes` and `volreg` (one precondition/error-string source).
- **C4 — Factor the SVF preamble/postamble.** `_svf.prepare_svf_inputs(...)` so the
  two recipes collapse to prepare → `level_solve` → drive → assemble → finalise
  (the `_svf` docstring already claims this).
- **C5 — Shared `PyramidSchedule` mixin** for the copy-pasted pyramid fields;
  per-level `iterations` symmetric across matrix/SVF specs.
- **C6 — Normalise/document `cost_history`** (format varies by resolved path).
- **C7 — Hoist the windowed-slope test** to a shared `_converge` helper (consumed
  by 2c and the matrix IC path).
- **C8 — Guard the affine-IC-under-`volreg` landmine** (the `safe_inv` would land
  inside the `vmap`).
- **B6 — Mask semantics** documented (soft-by-smoothing, weight contract) + a hard
  variant restricting the metric domain (ANTs `-x`); out-of-mask-suppression test.
- **D1 — Adversarial tests:** folding deformation, near-singular/large-shear
  affine, antipodal-rotation Karcher mean, flat-region image, degenerate
  single-axis grid, NaN/inf inputs, too-large BBR offset (graceful failure).
- **D2 — Re-certify early-exit** with the as-shipped windowed-slope criterion.

## 3. Sequencing & parallelisation

```
Phase 0  ────────────────────────────────────────────────  (gate; everything waits on 0a/0b)
            │
            ├── Phase 1 (force layer)   ──┐
            ├── Phase 3 (matrix scale)  ──┤  parallel; independent state machines
            ├── Phase 4 (preprocessing) ──┤
            │                             │
            └── Phase 2 (greedy driver) ──┘  consumes Phase 1 forces; leans on 0e
                          │
                  Phase 5 (Pallas; profile-gated, after the algorithmic levers)
                          │
                  Phase 6 (consolidation; threads through, closes the round)
```

- **Critical path:** 0a/0b → 1a (MI force) and 2a (greedy driver) are the two
  headline deliverables; they are independent and can run concurrently after the
  gate.
- **Lowest-effort / highest-parity first:** 0a, 0b, 4a (winsorize), 1c (LNCC
  integral image) are all small and high-leverage — land them early for a quick
  bar-moving increment.
- **Profile before Pallas:** Phases 1–3 may already clear the bar; Phase 5 is
  built only against a recorded roofline, never speculatively.

## 4. Economic verdict the round targets

| Recipe | v3 status | v4 levers | v4 target |
|---|---|---|---|
| rigid / affine (single pair) | NOT a win (1.16× @128³); affine diverges | 0b + 3a + 3b | qualitative win, **certified at 256³** |
| Demons / SyN | qualitative win that **erodes** (super-linear) | 0a + 1c + 2a (+5a/5b) | qualitative win, **≤ linear to 256³** |
| multimodal (MI) deformable | escape-hatch only (slow) | **1a (MIForce)** + A6 | **first-class fast path** (fMRIPrep metric) |
| cohort volreg | structural win | unchanged (+ 3a memory) | structural win, lower memory |
| differentiable layer | capability win | unchanged | capability win |

## 5. Out of scope (scope discipline)

Geodesic-shooting LDDMM (the v3 §8 deferral stands — the greedy driver does not
preclude it; the SVF path it keeps is its base); atlas/template **data
structures** and the groupwise build loop (→ `thrux`); surface/`bbregister`
coupling; any I/O; PyTree/module wrappers (→ `nimox`/`entense`). ANTs-reference
and real-data parity remain **delegated to the perf-bench agent**.

## 6. Cross-references

- [`registration-suite-v3.md`](registration-suite-v3.md),
  [`registration-suite-v2.md`](registration-suite-v2.md),
  [`registration.md`](registration.md) — the prior rounds.
- `docs/feature-requests/`: `registration-suite-v3-followups` (A/B/C/D),
  `registration-matrix-recipe-perf-levers` (IC/early-exit),
  `pallas-demons-esm-force` (5a), `pallas-trilinear-resample` (5d),
  `perf-wins-must-certify-at-scale` (the scaling discipline),
  `interpolation-backend-cpu-gpu-gap`, `register-affine-small-grid-divergence`
  (0b), `register-demons-force-divide-by-zero` (0a).
- `src/nitrix/{register,metrics,geometry,linalg,numerics}/` — the homes.
- `IMPLEMENTATION_PLAN.md` §10.A (deviation log).
