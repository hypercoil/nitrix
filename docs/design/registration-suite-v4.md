# Registration suite v4 — brain-scale performance + fMRIPrep parity

> **Status (2026-06-12): PROPOSED — comprehensive plan, no code yet.** Self-
> contained (supersedes the first v4 draft; folds in a three-perspective
> performance / mathematical / engineering review). The fourth registration
> block, on top of the shipped v3 suite (Force keystone, transform algebra,
> matrix inverse-compositional + early-exit perf levers, masks; merged to local
> `main` `356c768`). Two intertwined goals: **(1) make the GPU wins real at brain
> scale** (256³ single + cohort) — closing the economic gap the perf agent
> flagged (floor 4×, target 20× vs ANTs-CPU; CPU no longer timing out); and
> **(2) reach fMRIPrep / `antsRegistration` feature parity**, headlined by a
> **closed-form Mattes mutual-information force**. The organising idea is a clean
> **algebra↔group split for the dense-field family** (§2) mirroring the matrix
> transform-algebra v3 already shipped.

Reads on top of [`registration-suite-v3.md`](registration-suite-v3.md) (V1–V5),
[`registration-suite-v2.md`](registration-suite-v2.md) (R4–R8), and
[`registration.md`](registration.md) (R0–R3); folds in the post-v3 review
(`registration-suite-v3-followups.md`, items A/B/C/D) and the perf ledger
(`registration-matrix-recipe-perf-levers`, `pallas-demons-esm-force`,
`perf-wins-must-certify-at-scale`, `interpolation-backend-cpu-gpu-gap`).

---

## 0. Diagnosis — why v3 did not clear the bar

The hot kernel under every recipe is the **interpolation gather**; cost tracks
(i) gathers per iteration, (ii) HBM each touches, (iii) iteration count. Against
that lens, v3's shipped levers fall short for six concrete, structural reasons:

1. **Affine inverse-compositional (IC) diverges at small coarse grids**
   (`register-affine-small-grid-divergence`). `_inverse_compositional._ic_level`
   takes a full Gauss–Newton step with a fixed ridge and **no accept/reject**,
   unlike the forward `levenberg_marquardt`. On an ill-conditioned ≤14³ level the
   step overshoots and `matrix_exp(−Δθ)` blows the linear block up; the
   across-level translation rescale compounds it. Affine never gets the IC win.
2. **The rigid/affine IC win erodes at scale** (7.1×→3.7× by 128³). `_ic_level`
   re-reads the whole `(M, P)` steepest-descent buffer every iteration
   (`h_inv @ (sd.T @ err)`): at 256³ affine that is ~805 MB re-read per step plus
   a large resident buffer — the certify-at-scale wall.
3. **Early-exit is implemented but off by default** (`RegistrationSpec.convergence
   = None`). Single-pair rigid runs the full `30 × 3` iterations though it
   converges in ~5/20 even on hard cases.
4. **Demons/SyN scale super-linearly** (measured 96³→160³ time exponent ≈1.37;
   GPU/CPU speedup erodes 43×→28×). The recipes are algorithmically O(N) per
   registration, so this is a **hardware/bandwidth** effect (HBM saturation), not
   an asymptotic complexity bug — a working hypothesis that must be confirmed by
   an HLO roofline before the ≤-linear target (§5) is committed.
5. **The diffeomorphic recipes re-exponentiate from scratch every iteration.**
   `_svf.symmetric_level` / `single_sided_level` call
   `integrate_velocity_field(v, n_steps=5/6)` for **each** field **every**
   iteration — 12 gathers/iter (SyN), 7 (Demons). They are documented as *greedy*
   recipes, yet pay the full stationary-velocity exponential the greedy method
   does not need. This is the largest single source of redundant gathers, and the
   resolution (§2) is the centrepiece of v4.
6. **Demons returns all-NaN on every real image**
   (`register-demons-force-divide-by-zero`): the unguarded ESM `diff/denom` is
   `0/0` on uniform background, so no Demons number on real anatomy is meaningful.

The theme: **the suite does far more gathers than necessary, and the survivors
are not bandwidth-efficient.** v4 = cut redundant gathers (§2), make the
survivors O(N) and bandwidth-efficient (§3, §4), cut the iteration count, and
close the fMRIPrep feature gaps — each lever certified at brain scale.

---

## 1. The two design axes that organise v4

Everything below is a point on one of two orthogonal stratification axes, both
instances of the v3 "**a generic correct path is the substrate; fast specialised
implementers sit behind the same surface, auto-selected when their preconditions
hold**" thesis:

- **Representation axis (dense fields): algebra ↔ group.** A dense deformation
  can be carried as a **Lie-algebra** element (a stationary velocity field `v`,
  the log/SVF domain) or as a **group** element (the diffeomorphism itself, a
  displacement field `φ = id + s`). The two are bridged by `exp` (scaling-and-
  squaring) and `log` (its inverse). The *algebra* path is correctness-faithful
  and exact-velocity; the *group* path is the performant greedy method. §2.
- **Metric→force axis: closed-form ↔ autodiff.** Every metric can drive a
  dense recipe generically via `MetricForce` (autodiff of the cost); a metric
  that earns a closed-form gradient gets a Tier-1 fast `Force`. §3.

---

## 2. The dense-field algebra↔group design (the centrepiece)

### 2.1 The representation choice, stated precisely

A dense deformation is an element of the diffeomorphism group `Diff`. Two ways to
carry the optimiser's state:

- **Group element** — the diffeomorphism, stored as a displacement `s`
  (`φ = id + s`). Composition `φ₁∘φ₂` is the group product
  (`compose_displacement`); inverse is `invert_displacement`. *Greedy*
  registration works here: each iteration composes a small smoothed update onto
  `φ`. One warp per iteration; **no** per-iteration exponential. This is what ANTs
  `SyN` actually does.
- **Algebra element** — a stationary velocity `v` in the Lie algebra (vector
  fields). The exponential `exp: v ↦ φ` is `integrate_velocity_field`
  (scaling-and-squaring); its inverse `log: φ ↦ v` is scaling-and-squaring
  *backward*. *Log-domain* registration works here: carry `v`, update additively
  / by BCH, exponentiate to warp. One `exp(v)` (n_steps gathers) per iteration.

**The load-bearing fact:** `exp` is a local diffeomorphism but **not
surjective** — a general diffeomorphism built by composing many small,
non-commuting warps (exactly what greedy registration produces) is **not**
`exp(v)` for any single stationary `v`. So the two representations are not
interchangeable state: solving in the algebra stays on the SVF submanifold (the
velocity is exact); solving in the group explores all of `Diff` (the velocity is
recoverable only as a *best-fit* `log(φ)`, exact iff `φ` happens to lie in the
image of `exp`).

This is the same structure the matrix family already has in v3 (`matrix_exp` /
`matrix_log`, the group `SE(n)`/affine vs the algebra `se(n)`/`gl(n)`, the Fréchet
mean computed in the algebra). v4 gives the **dense** family the same two-sided
structure.

### 2.2 The bridge — `field_log` (the recovery mechanism)

> Directly answers "if we solve in the group, do we have the log to recover the
> algebra element / the SVF?" — **yes**, via a new dense field logarithm, the
> exact analogue of v3's `matrix_log`.

Deliver **`geometry.field_log(φ) → v`** (group→algebra) to pair with the existing
`integrate_velocity_field` (algebra→group):

- **Algorithm — scaling-and-squaring backward.** Take `n` successive
  *diffeomorphism square roots* `ψ = φ^{1/2}` (the fixed point of `ψ∘ψ = φ`,
  solved with the existing `compose_displacement` / `invert_displacement`
  machinery — the inverse of the `_double` step in `integrate_velocity_field`),
  until `φ^{1/2^n}` is in the near-identity regime where `log ≈ id`; then
  `v ≈ 2^n · (φ^{1/2^n} − id)`. Differentiable (fixed-point + compose, the
  implicit-function path), GPU-native.
- **Contract & gate.** On the SVF submanifold the round-trip
  `integrate_velocity_field(field_log(φ)) == φ` holds to integration tolerance
  (the exact gate). On a general greedy `φ` it returns the **best stationary-
  velocity fit**, and ships a *documented residual* (the projection error) rather
  than claiming an exact inverse — the honest contract.
- **Why it matters beyond the recipes.** It unlocks the dense velocity-barycentre
  / Fréchet-mean template path: v3's `velocity_mean` / `transform_mean` consume
  velocities, so `field_log` lets a *group*-solved warp feed the groupwise/
  template-construction machinery. The algebra↔group bridge is now first-class on
  both the matrix and dense sides.

### 2.3 The two drivers (shared scaffold, distinct state contracts)

Both drivers reuse the coarse-to-fine scaffold `_svf.svf_coarse_to_fine` — its
prolongation (`upsample · ratio`) is **representation-agnostic** because both `v`
and `s` are voxel-unit `(*spatial, ndim)` fields. What differs is the per-level
**update closure** and the **finalisation**; these are *sibling* level-drivers,
not one masquerading as a fast implementer of the other.

- **Algebra driver** (`algebra_level`; the refactored
  `single_sided_level`/`symmetric_level`): carry `v`; per iteration warp by
  `exp(v)`, compute the force, fluid-smooth the *update*, accumulate `v += u`
  (or BCH), diffusion-smooth the *accumulated velocity*. Output: `v` (exact) +
  `φ = exp(v)`. **This is log-Demons / the SVF path** — the correctness oracle
  (it stays byte-equal to the current shipped recipes, so it anchors the test
  suite exactly), the exact-velocity output, and the future geodesic-LDDMM base.
- **Group driver** (`group_level`, NEW): carry `s` (`φ = id + s`); per iteration
  warp by `φ` directly (one gather), compute the force, fluid-smooth the
  *increment*, compose it onto `φ` (`φ ← φ ∘ (id + ε·u_smoothed)`),
  diffusion-smooth the **total displacement** `s`, and — symmetric variant —
  periodically re-invert to keep `φ_fwd`/`φ_inv` consistent. Output: `φ` (group
  element) + `v = field_log(φ)` **on demand** (computed once at finalisation, not
  per iteration). **This is greedy SyN / greedy Demons** — the performance path.

**Two correctness obligations the group driver carries (do not omit):**

1. **Per-step diffeomorphism bound — on the Jacobian, not the magnitude.** Each
   increment must satisfy `det(I + ε∇u_smoothed) > 0`, i.e.
   `ε·‖∇·u_smoothed‖_∞ < 1` — a trust-region clamp on the *gradient* of the
   increment, which Gaussian fluid-smoothing bounds by construction. A clamp on
   displacement magnitude alone can still fold. The QA gate `det J > 0` on the
   *total* `φ` is necessary but not sufficient (an intermediate composition can
   fold while the total stays positive-det), so the bound is enforced *per step*.
2. **Total-field (diffusion) regularisation is retained.** In the algebra driver,
   fluid smooths the *update* and diffusion smooths the *accumulated velocity*. In
   the group driver, fluid smooths the *increment* and diffusion smooths the
   *total displacement* `s` each iteration (the elastic regulariser of the group
   element — ANTs' `SyN[grad, updateFieldSigma, totalFieldSigma]`). Dropping the
   total-field smoothing would silently turn the recipe into fluid-only
   regularisation — a behavioural change, not a perf-neutral reformulation.

### 2.4 Recipe wiring, outputs, and the honest test story

- **`representation` is a spec field** (`{'group', 'algebra'}`) on the SVF specs.
  `greedy_syn_register` defaults to **group** (it *is* greedy SyN);
  `diffeomorphic_demons_register` defaults to **group** as well (the v4
  perf/CPU-timeout win is the point), with **algebra** the opt-in exact-SVF /
  oracle / geodesic-base path. The velocity output is always available — carried
  directly in algebra mode, recovered by `field_log` in group mode — so the
  result records (`velocity` / `forward_velocity` / `inverse_velocity`) keep
  their meaning either way. (Naming note: log-Demons' "log" now refers to the
  recoverable velocity via `field_log`; `representation='algebra'` is the literal
  log-domain solve.)
- **The finalisation simplifies in group mode.** Greedy SyN already carries
  `s_fwd`/`s_inv` directly, so `φ = (id+s_fwd) ∘ (id+s_inv)⁻¹` is the same two
  calls (`invert_displacement` + `compose_displacement`) with **no** per-field
  `exp` — and the only `field_log` cost is paid once, on demand, if a velocity is
  requested.
- **Parity oracles (named explicitly, because greedy has no exact oracle):**
  (a) the **algebra driver == the current shipped recipe**, byte/ULP — an exact
  oracle anchoring correctness; (b) `field_log`/`exp` **round-trip** exact on the
  SVF submanifold; (c) **both drivers recover the same synthetic ground-truth
  warp to tolerance** — a *recovery* gate, since the greedy and SVF operators are
  genuinely different fixed points (greedy SyN is not a one-parameter subgroup),
  named as a discipline exception rather than dressed up as field-wise equality;
  (d) **diffeomorphism** (`det J > 0`) and **inversion residual** bounded under
  large deformation.

### 2.5 Realistic perf accounting (no overclaim)

Per SyN iteration: group mode is 2 midpoint warps + 2 increment composes = **4
gathers**, vs algebra mode's **12** (the 2× scaling-and-squaring) → **3×**. Group
Demons is **2** (warp + compose) vs **7** → ~3.5×. Crucially there is **no
per-iteration inversion**: the symmetric half-warps are driven to the midpoint by
the symmetric force, and inverse-consistency is realised by the **single**
`invert_displacement` at finalisation — exactly as the current algebra SyN, which
also inverts only once. (`field_log` adds a one-time ~120-gather finalisation cost
if a velocity output is requested — negligible.) This is the largest single
dense-field lever, and it removes the iterated-`integrate_velocity_field` cost that
is the CPU-timeout culprit (`interpolation-backend-cpu-gpu-gap`). The *total*
speedup also depends on the surviving Gaussian-regulariser and force costs, which
Phase 1c/1d attack — so Phase 2 is necessary but not alone sufficient for the
≤-linear-at-256³ target. **Full implementation design:**
[`registration-suite-v4-phase2.md`](registration-suite-v4-phase2.md).

---

## 3. Force layer — closed-form forces + the HBM analysis

### 3.1 Mattes MI force vs NMI — the use-case decision

fMRIPrep's cross-modal stages run `antsRegistration --metric MI[...]`, which is
**unnormalised Mattes MI**. Today MI can only drive a dense recipe via
`MetricForce(MI())` — the autodiff escape hatch, which differentiates through the
joint-histogram **scatter** every iteration (a backward tape over a `scatter_add`),
explicitly "no perf guarantee." MI *has* a closed-form gradient.

**Closed-form Mattes MI force (ships in v4).** For the linear-Parzen joint
histogram the suite already forms (`joint_histogram`), the per-voxel gradient is

```
∂MI/∂m(x) = (1/N) Σ_{a,b} ∂β_m(x,a)/∂m · β_f(x,b) · log( P[a,b] / P_m[a] )
```

— a per-voxel sum over the ≤4 touched bin pairs, reading from the small
precomputed table `W[a,b] = log(P[a,b] / P_m[a])` (`s = (bins−1)/span_m`; the
`+1` and `P_f[b]` terms vanish because the soft weights sum to one, so
`Σ_a ∂β_m(x,a)/∂m = 0`). This is verified correct to machine precision and is the
ITK/ANTs metric. Deliver `metrics.information.mi_grad(moving, fixed, *, bins,
range_moving, range_fixed)` + a Tier-1 `MIForce` in `_force.py`.

**NMI (Studholme) — deferred, on use-case grounds.** Normalised MI
`(H_m+H_f)/H_mf` is a **quotient**, so its gradient does **not** reduce to the
formula above (the `+1`/`P_f` terms no longer cancel — it needs the quotient-rule
table, with a near-convergence `1/H_mf` sensitivity). NMI's distinctive value is
**overlap invariance**, which matters when the *overlapping FOV changes
substantially during optimisation* — a predominantly **global/affine**, large-FOV
or partial-overlap concern, much less so for the *dense deformable* force where
per-level overlap is ~constant. Since fMRIPrep's deformable cross-modal metric is
Mattes MI, the call is: **ship the closed-form Mattes MI force; defer the
closed-form NMI force** (scoped — same per-voxel contraction machinery, only the
`W[a,b]` table changes; validate against autodiff; guard the `1/H_mf` blow-up),
with `MetricForce(MI(normalized=True))` the correct-but-slower escape hatch
meanwhile. Revisit if a deformable-NMI consumer with large FOV variation
materialises. **Correlation-ratio** gets the same treatment: a `cr_grad` +
`CRForce` if it earns one, with `jax.grad(correlation_ratio)` its only oracle.

**Pinning the histogram range is mandatory for the MI force** (and is the matrix
side of followup A6, §4 Phase 3): with a data-min/max range the bin grid drifts
as the moving image deforms, making the objective **non-stationary** across
iterations (the gradient is finite, but the target moves), and the two extreme
voxels sit at the clip boundary where the force truncates to 0. MI needs **two**
ranges — `range_fixed` pinned, `range_moving` pinned to cover the full intensity
support (pinning below the data extent silently zeroes the force on over-range
voxels). The MIForce parity gate must mirror the metric's empty-bin convention
(`where(hist>0, …)`, not an ε-floor inside the log) or it tests a different
function. A `span` floor tied to intensity scale (not a fixed `1e-12`) guards
flat coarse levels where `s = (bins−1)/span` would otherwise explode.

### 3.2 HBM profiles — which forces warrant custom kernels

The perf agent flagged the ESM force as HBM-bound. The analogous question for
LNCC and MI has **different answers per force** (cost laws below; a "round-trip" is
a full-field HBM read+write that breaks XLA fusion):

| Force | Dominant traffic | Pass count | Kernel verdict |
|---|---|---|---|
| **ESM (Demons)** | `∇warped` stencil + `j`/`denom`/`u` elementwise + 2 Gaussian smooths | ~5–6 fusion-breaking passes | **gather-free fused stencil+force** (scoped). Highest tractability. |
| **LNCC (SyN)** | **9 box sums** (5 windowed stats + 4 re-sums) + `spatial_gradient` + assembly | **~27 conv passes**, ~**5× the ESM** traffic | **the worst** — warrants an algorithmic fix *and* a fused windowed-stats kernel. |
| **MI (Mattes)** | joint-histogram **scatter** (N→B²) + per-voxel bin-derivative gather | low field volume, but a **scatter** (+ the autodiff tape if not closed-form) | **closed-form first** (removes the tape — the real win); a shared-memory histogram kernel is a distinct, lower-priority second. |

- **LNCC is the most HBM-bound** (~5× ESM, not the 8× an earlier framing
  overstated — but the ranking holds): `lncc_grad` runs ~27 separable-conv passes
  XLA does not fuse across axes (each `correlate1d` is its own op with a
  `moveaxis`). Three tiers of fix in Phase 1.
- **MI's inefficiency is the autodiff-through-scatter tape**, not field
  round-trips; the closed form (§3.1) removes it. The residual **forward** scatter
  (16.7M scattered adds into ~1024 buckets at 256³ — atomics contention) is the
  brain-scale risk that keeps "first-class fast path *at brain scale*" honest: it
  is measured before the claim, and the shared-memory histogram kernel (Phase 5c)
  — or ANTs-style histogram subsampling — closes it.

---

## 4. Phase plan

Seven phases. **Phase 0 is a hard gate** (benchmarks lie until it lands); Phases
1–4 are the substance and largely parallelise; Phase 5 (Pallas) is profile-gated;
Phase 6 consolidates *after* Phase 2 (it refactors the same files). Standing
discipline on the **perf** phases (2/3/5): a fast-path-vs-generic parity test (or
the named recovery-gate exception) **and** a scaling case certified to brain scale
(single + cohort) with a stated cost law. Phases 0/4/6 are correctness/abstraction
work and are *not* expected to ship a scaling case. Every phase: pure-functional,
frozen/`NamedTuple`, Protocols where they earn it, jaxtyping, ruff + mypy,
`custom_vjp` where stability needs it.

### Phase 0 — Correctness & honesty gates (blocking)

| Item | Fix | Home | Fold-in |
|---|---|---|---|
| 0a | Guard the ESM denom: `where(denom>eps, diff/denom, 0)` (zero force where no gradient *and* no mismatch — does not bias the fixed point). `eps` on the squared scale of `denom`. The `_normalise_step` global-max cannot rescue a NaN, so the guard must be at the force. Audit LNCC/`MetricForce` for the same 0/0 (cross-ref D1 flat-region test) | `_force.py` | `register-demons-force-divide-by-zero` |
| 0b | **Backtracking** (step-halving along the fixed IC direction) cost-guarded IC step — *not* a per-iteration ridge re-solve, which would break the constant-Hessian invariant the whole module is built on. Recompute cost after the trial compositional update, halve `Δθ` on increase | `_inverse_compositional._ic_level(_converge)` | `register-affine-small-grid-divergence` |
| 0c | `MetricForce` magnitude normalisation (RMS/max-norm) for non-spatial-mean metrics so an unclamped Demons driver gets a controlled MI/CR step | `_force.py` | B2 |
| 0d | `_normalise_step`: global-max → RMS / high-percentile | `_svf._normalise_step` | B4 |
| 0e | **`invert_displacement` robustness — a hard predecessor of Phase 2** (the group driver's re-symmetrise depends on it converging): Anderson/Newton acceleration **+ a residual assertion** (today the Picard loop silently returns a non-converged inverse at `max_iter`) | `geometry/deformation.py` | B5 |
| 0f | `transform_mean` / `matrix_log` convergence + domain guards (tangent-norm tolerance; negative-eigenvalue / Taylor-radius guard; DB balancing) | `geometry/algebra.py`, `linalg/matrix_function.py` | B3 |
| 0g | Honest docs: "ANTs-*style*, synthetic-recovery-validated"; real-data/ANTs-ref parity delegated to perf-bench | design docs + `register` docstring | D3 |

0a + 0b are load-bearing: 0a makes Demons usable on real anatomy at all; 0b makes
affine reliable *and* is the precondition for default early-exit (Phase 3b).

### Phase 1 — Force layer + O(N) separable operators

- **1a — Closed-form Mattes MI force (headline; B1).** `metrics.mi_grad` +
  Tier-1 `MIForce` (§3.1), with the two-range pinning, the matching empty-bin
  convention, and the `span` floor. Gate: `mi_grad == jax.grad(mutual_information)`
  to tolerance (pinned range); cross-modal deformable recovery to convergence;
  a SyN-MI scaling case. **NMI deferred** to the `MetricForce` escape hatch
  (§3.1); `mi_grad` ships the unnormalised form only.
- **1b — `cr_grad` + `CRForce` (B1 tail).** Optional; oracle =
  `jax.grad(correlation_ratio)`.
- **1c — LNCC efficiency (L-i + L-ii).** **(L-i)** Replace the box-sum conv with a
  **separable per-axis integral image** (cumsum-difference) in
  `metrics/_common._box_sum`: O(N), radius-independent. **It must cumsum over the
  reflect-*padded* axis** — a cumsum over the unpadded signal yields a
  valid-neighbourhood trim, not the box sum, differing from the oracle by up to a
  full window at the boundary. fp32-safety (the cancellation hazard scales with
  the prefix-sum magnitude, ~axis_length·max) is a **gate** — an fp64-vs-fp32
  parity test at the real intensity range — not an assertion; safe at the default
  radii. **(L-ii)** Hoist the fixed-only windowed sums in `_BoundLNCC.bind` (which
  precomputes nothing today): `sum_f`/`sum_ff`/`var_f`/`fbar` for the
  *single-sided* path only (symmetric SyN's "fixed" changes per step). Conv path
  stays the parity oracle, with the documented interior-exact / boundary-divergent
  contract `lncc_grad` already carries.
- **1d — Recursive (IIR) Gaussian regulariser.** *After Phase 2 removes the
  integration gathers, the fluid+diffusion Gaussian smooths (2–4 separable convs/
  iter) become co-dominant with the warp.* Add a Deriche / Young–van Vliet
  recursive-Gaussian backend to `smoothing.gaussian`: O(N), radius-independent —
  the regulariser analogue of L-i. The `signal._iir` machinery (and
  `_first_order_causal`) already exist. Gate: parity to the truncated-FIR Gaussian
  to tolerance. (Lands in `smoothing`; benefits every regularised recipe.)
- **1e — `SumForce` (A5)** and **1f — restrict-deformation (A7)**: behind the
  `Force` protocol with no driver change (`SumForce.update = Σ wᵢ·fᵢ.update`;
  restrict is a per-axis weight at the `_svf._mask_force` site).

### Phase 2 — Dense-field algebra↔group drivers (the structural win)

§2 in full. Deliverables: **`geometry.field_log`** (the recovery bridge, 2.2); the
**group driver** `group_level` with its per-step Jacobian bound and retained
total-field smoothing (2.3); the **algebra driver** as the refactored existing
recipe (the exact oracle); the `representation` spec field and the on-demand
velocity recovery (2.4). Also folds **C5's `n_steps` reconciliation** (5/6/7 →
one convention; fix the test ground-truth that builds warps with the default 7
while recipes integrate with 5/6) and **A3 SVF early-exit** (`while_loop` variant
of the level drivers using the shared windowed-slope helper, single-pair only,
re-deriving clamp-vs-scale). Gate: §2.4 (a)–(d); SyN/Demons scaling re-certified
to 256³ with the new cost law (target ≤ linear, **gated on the Phase-0 roofline
confirming the bandwidth hypothesis** of §0.4).

### Phase 3 — Matrix path at brain scale

- **3a — IC moment reformulation.** Eliminate the `(M, P)` steepest-descent
  buffer: `SDᵀe` is exactly the moment tensor `m_ij = Σ_x ∇F_i(x)·(x−c)_j·e(x)`
  (affine linear block) + `Σ_x ∇F_i·e` (translation), with the rigid rotation
  columns a fixed contraction of the same moments against the `so(n)` generators
  (verified algebraically to roundoff). Store only `∇F` `(M, ndim)`; build `H`
  once from the same moments **reproducing `_hessian_inv`'s Jacobi
  preconditioner**; compute `SDᵀe` per iteration as a fused reduction. Cuts
  resident memory `(M,P)→(M,ndim)` and per-iteration projection bandwidth
  `M·P→M·(ndim+1)` (3× affine; the projection is ~72% of per-iteration HBM at
  256³ affine, so this halves total per-iteration traffic — the warp does *not*
  dominate it). **Algebraically identical, parity to fp tolerance** (different
  summation order → ~1e-4 fp32, amplified by `H⁻¹`; the gate is a tolerance on the
  recovered matrix *after* `H⁻¹`, not bit-exact on `SDᵀe`). The lowest-risk
  brain-scale win — **lead the §5 certification with it.**
- **3b — Default-on early-exit, single-pair — with an explicit differentiability
  contract.** Make the `while_loop` early-exit the default for the single-pair
  matrix path (already implemented; requires 0b). Because `while_loop` has **no
  reverse rule**, this would otherwise *silently* break existing
  `jax.grad(rigid_register)` callers (the documented differentiable-layer
  capability). The contract: (i) the **implicit-function entry points**
  (`implicit_least_squares` / `implicit_minimize`) are the blessed differentiable
  API (trajectory-independent — they do not care about the forward trip count);
  (ii) `convergence=None` restores the reverse-differentiable fixed `scan` as a
  first-class, documented opt-out; (iii) a reverse-mode trace through the
  `while_loop` path raises a **loud, actionable error** pointing to (i)/(ii) — the
  "loud fallbacks" tenet, never a silent break. **C6 (`cost_history` format)
  ships *with* 3b** (the padded early-exit trace becomes the default output, so it
  must be normalised/documented here, not deferred to Phase 6). Also resolves
  **C3** (raise when `convergence` is set on a path that cannot honour it).
- **3c — Affine multi-start / moment init (A8)** — a centre-of-mass / moment
  initialisation (+ optional best-of-N rotation search) before the affine
  optimise; the narrow affine basin fails silently on a large misalignment from a
  single zero start.

### Phase 4 — fMRIPrep front-end + matrix-side range pinning

Cheap, independent, opt-in / default-off (byte-unchanged unless enabled).

- **4a — Winsorization (A1, highest parity value).** `metrics.winsorize(image,
  lower=0.005, upper=0.995)` + a spec knob applied before the pyramid (also
  mitigates 0d's outlier sensitivity).
- **4b — Histogram matching (A2).** `metrics.match_histogram(moving, reference,
  bins)` (CDF / quantile transport) + recipe knob, within-modality stages.
- **4c — Decouple smoothing from shrink (A4).** Per-level `smoothing_sigma`
  schedule independent of the shrink factor (ANTs `-f 4x2x1 -s 2x1x0vox`); default
  reproduces today.
- **4d — Matrix-path range pinning (the *other half* of A6).** A6 was originally
  raised against the **matrix** MI/CR path: `_metric.MI` / `CorrelationRatio` call
  the kernels with **no** range, so `affine_register(metric=MI())` drifts. Add a
  `range_moving`/`range_fixed` field to the `MI`/`CorrelationRatio` records (note:
  the kernels already take `range_moving`/`range_fixed`, not a single
  `value_range`), have the matrix driver compute the range once from full-res and
  inject it per level (`_core.register_core` → `MetricObjective`). This is a real
  spec change spanning `_metric.py` + `_core.py`, distinct from the *force*-side
  pinning in 1a — both halves of A6 must land.

### Phase 5 — Custom kernels (Pallas; profile-gated)

Built only against a recorded roofline confirming the op dominates per-iteration
HBM **and** a bottlenecked consumer (`pallas-demons-esm-force` Trigger); each
registers a `custom_vjp` and keeps the pure-JAX path as floor + oracle. Priority
tracks §3.2:

- **5a — Fused ESM stencil+force.** Gather-free → dodges the parked-trilinear ELL
  gather-lowering blocker; reuses the recursive Gaussian (1d) for the smooths.
- **5b — Fused LNCC windowed-statistics kernel** (two halo'd stencil-reduce
  passes). The largest HBM to recover; built after 1c (the algorithmic win may
  suffice below brain scale; this is the 256³ endgame).
- **5c — Fused MI histogram+gradient kernel.** Shared-memory partial histograms +
  the bin-derivative gather — closes the §3.2 forward-scatter brain-scale risk.
  Lowest priority (1a captures most of the win).
- **5d — Trilinear gather** (`pallas-trilinear-resample`). The universal warp
  kernel; **blocked** on pinned-JAX gather lowering. The §2/§3a levers cut gather
  *count* and are higher-ROI until it unblocks. Tracked, not committed — and the
  §5 ≤-linear targets that depend on the *surviving* warp bandwidth are flagged as
  contingent on 5a/5b/5d, not on the algorithmic levers alone.

### Phase 6 — Abstraction consolidation + validation hygiene (after Phase 2)

- **C1/C2** unify the fast/slow dispatch (lift the matrix `method=` string to an
  ADT mirroring `Force`, or at minimum share `resolve_ic_method` between `recipes`
  and `volreg`).
- **C4 — Factor the SVF preamble/postamble** (`_svf.prepare_svf_inputs`) —
  **sequenced strictly after Phase 2**, so it factors the *final* (two-driver)
  shape rather than colliding with the greedy rewrite of the same files.
- **C5 — Shared `PyramidSchedule` mixin**; per-level `iterations` symmetric across
  matrix/SVF specs.
- **C7 — Hoist the windowed-slope convergence test** to a shared `_converge`
  helper (consumed by Phase 2's A3 and the matrix IC path).
- **C8 — Guard the affine-IC-under-`volreg` landmine** (the `safe_inv` in
  `_affine_params_from_matrix` would land inside the `vmap`).
- **B6 — Mask semantics** (soft-by-smoothing, weight contract) + a hard variant
  restricting the metric domain (ANTs `-x`); out-of-mask-suppression test.
- **D1 — Adversarial tests:** folding deformation, near-singular/large-shear
  affine, antipodal-rotation Karcher mean, flat-region / uniform-background image
  (cross-ref 0a), degenerate single-axis grid, NaN/inf inputs, too-large BBR
  offset (graceful failure).
- **D2 — Re-certify early-exit** with the as-shipped windowed-slope criterion.

---

## 5. Sequencing & gates

```
Phase 0  ───────────────────────────────────────────────  (gate; everything waits on 0a/0b; 0e gates Phase 2)
            │
            ├── Phase 1 (forces + O(N) ops) ──┐
            ├── Phase 3 (matrix scale)        ──┤  parallel; independent state machines
            ├── Phase 4 (preprocessing + A6)  ──┤
            │                                  │
            └── Phase 2 (algebra↔group drivers)┘  consumes Phase 1 forces; hard-requires 0e
                          │
                  Phase 5 (Pallas; profile-gated, after the algorithmic levers)
                          │
                  Phase 6 (consolidation; strictly after Phase 2)
```

- **Lead the brain-scale certification with 3a** — exact, low-risk, verified to
  halve per-iteration HBM at 256³ affine. Treat Phase 2's ≤-linear claim as
  **hypothesis-gated** on the Phase-0 roofline (the 1.37 exponent must be
  confirmed as the bandwidth wall, §0.4) before committing the greedy rewrite's
  scaling target.
- **Quick early wins** (small, high-leverage, land first for an early
  bar-moving increment): 0a, 0b, 4a, 1c.
- **Critical path:** 0a/0b → {1a Mattes MI force, 2a group driver} — the two
  headline deliverables, independent after the gate (2 additionally needs 0e).
- **Profile before Pallas:** Phases 1–3 may already clear the bar; Phase 5 is
  built only against a recorded roofline, never speculatively.

---

## 6. Economic verdict the round targets

| Recipe | v3 status | v4 levers | v4 target |
|---|---|---|---|
| rigid / affine (single pair) | NOT a win (1.16× @128³); affine diverges | 0b + 3a + 3b | qualitative win, **certified at 256³** |
| Demons / SyN | qualitative win that **erodes** (super-linear) | 0a + 1c + 1d + 2 (+5a/5b) | qualitative win, **≤ linear to 256³** (roofline-gated) |
| multimodal (Mattes MI) deformable | escape-hatch only (slow) | **1a** + 4d | **first-class fast path** (scatter measured at 256³; 5c if needed) |
| cohort volreg | structural win | unchanged (+ 3a memory) | structural win, lower memory |
| differentiable layer | capability win | 3b implicit-path contract | capability win (explicit contract) |

---

## 7. Out of scope (scope discipline)

Geodesic-shooting LDDMM (the v3 §8 deferral stands — the algebra driver and
`field_log` are precisely its base, not a precluder); atlas/template **data
structures** and the groupwise build loop (→ `thrux`); surface/`bbregister`
coupling; any I/O; PyTree/module wrappers (→ `nimox`/`entense`). ANTs-reference and
real-data parity remain **delegated to the perf-bench agent**. The closed-form
**NMI** and **CR** forces are scoped-but-deferred (§3.1), on the `MetricForce`
escape hatch meanwhile.

---

## 8. Cross-references

- [`registration-suite-v3.md`](registration-suite-v3.md),
  [`registration-suite-v2.md`](registration-suite-v2.md),
  [`registration.md`](registration.md) — the prior rounds.
- `docs/feature-requests/`: `registration-suite-v3-followups` (A/B/C/D),
  `registration-matrix-recipe-perf-levers` (IC/early-exit), `pallas-demons-esm-force`
  (5a), `pallas-trilinear-resample` (5d), `perf-wins-must-certify-at-scale` (the
  scaling discipline), `interpolation-backend-cpu-gpu-gap`,
  `register-affine-small-grid-divergence` (0b), `register-demons-force-divide-by-zero`
  (0a).
- `src/nitrix/{register,metrics,geometry,linalg,numerics,smoothing}/` — the homes.
- `IMPLEMENTATION_PLAN.md` §10.A (deviation log).
