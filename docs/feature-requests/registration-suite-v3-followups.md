# Registration suite v3 — post-merge follow-ups (from the pre-merge review)

> **Status (2026-06-11): scoped, post-merge.** A three-perspective pre-merge
> review (engineering rigour vs fMRIPrep/ANTs; clean abstraction/parsimony;
> performance) of the `registration-suite-v3` round surfaced the items below.
> The **blockers** (closed-form forward-Jacobian memory blow-up; affine
> inverse-compositional conditioning; two stale/over-claimed docstrings) were
> fixed *in* the merge and are **not** listed here. **ANTs-reference parity and
> real-data validation are delegated to the perf-bench agent** (not done in this
> repo) and are likewise out of scope here. Everything else the review raised is
> captured below, self-contained, for the next increment(s).

This document is self-contained: each item states the problem, the fix, where it
lands, and why it matters, without relying on the review transcript.

---

## A. ANTs / fMRIPrep feature-parity gaps (the suite is *not* a drop-in `antsRegistration`)

These are real features a fMRIPrep-equivalent run uses that the suite lacks.
Until they land, the docs must **not** claim unqualified "ANTs parity" — say
"ANTs-*style*; synthetic-recovery-validated; winsorize / histogram-matching /
multi-metric-sum / restrict-deformation not yet implemented."

### A1. Intensity winsorization. **[highest parity value]**
fMRIPrep runs `antsRegistration --winsorize-image-intensities [0.005, 0.995]` on
every call: clip each image to its 0.5/99.5 intensity percentiles before the
metric sees it, so outliers (hot voxels, skull, artifacts) do not dominate the
SSD / Demons force. The suite has no such op, so the within-modality (SSD /
Demons) path is fully exposed to outliers ANTs removes.
- **Fix:** a `metrics.winsorize(image, lower=0.005, upper=0.995)` (percentile
  clip; pure, differentiable-enough) and an opt-in `winsorize` knob on the
  recipe specs (applied to `moving`/`fixed` before the pyramid).
- **Lands:** `nitrix.metrics` (the op) + the recipe specs (the knob).
- **Interaction:** also mitigates **B4** (`_normalise_step`'s global-max outlier
  sensitivity) — the winsorize removes the outlier that throttles the field.

### A2. Histogram matching (same-contrast stages).
fMRIPrep uses `--use-histogram-matching 1` for within-modality stages: match the
moving image's intensity histogram to the fixed before registering, so an SSD /
CC metric is not fighting a global intensity offset/gain.
- **Fix:** a `metrics.match_histogram(moving, reference, bins=...)` (CDF
  matching; the standard quantile transport), opt-in on the recipes.
- **Lands:** `nitrix.metrics` + recipe knob.

### A3. SyN / Demons early-exit (extend the elective convergence to the SVF path).
The V4 elective `while_loop` early-exit (`RegistrationSpec.convergence =
Convergence(threshold, window)`, ANTs-style windowed cost-slope) is wired **only**
into the single-pair *matrix* inverse-compositional path. The diffeomorphic
recipes (Demons, greedy SyN) still run a fixed `lax.scan` over the full
`spec.iterations` every level. ANTs `-c [100x70x50, 1e-6, 10]` early-stops every
stage; without it nitrix SyN cannot match ANTs' effective iteration count, which
is the blocking piece for the iso-accuracy comparison.
- **Fix:** a `DemonsSpec.convergence` / `SyNSpec.convergence` and a
  `while_loop` variant of `_svf.single_sided_level` / `symmetric_level` with the
  same windowed criterion (reuse the slope test currently inside
  `_inverse_compositional._ic_level_converge` — hoist it to a shared helper, see
  **C7**). Single-pair only (it breaks `vmap`-batching, like the matrix path).
- **Caveat:** revisits the `_svf._normalise_step` clamp-vs-scale decision (the
  clamp was chosen *because* the forward was a fixed-budget scan; a convergence
  gate makes ANTS-style scale-to-step viable again). Re-derive that there.
- **Lands:** `_svf.py`, `diffeomorphic.py`, `_syn.py`.

### A4. Decouple smoothing from shrink in the multi-resolution schedule.
ANTs specifies shrink factors (`-f 4x2x1`) and smoothing sigmas (`-s 2x1x0vox`)
**independently**. The suite couples them: `gaussian_pyramid` downsamples by one
`pyramid_factor` with an anti-alias sigma *derived from the factor*. There is no
way to express "smooth 2 vox at a 4× shrink." Same-data parity needs the split.
- **Fix:** let the specs take a per-level `smoothing_sigma` schedule (a tuple,
  like the per-level `iterations` from V4 lever E) applied to the pyramid levels
  independently of the shrink factor; default reproduces today's behaviour.
- **Lands:** the specs + `gaussian_pyramid` call sites / a thin pyramid builder.

### A5. Multi-metric *summation* per stage.
ANTs sums metrics in one stage (`-m MI[...] -m CC[...]`). The suite has a
per-level metric/force *schedule* (a different force per level) but not a
weighted *sum* of forces within a level.
- **Fix:** a `SumForce([(w1, force1), (w2, force2), ...])` implementer of the
  `Force` protocol (its `update` returns `Σ wᵢ · forceᵢ.update`, its `cost` the
  weighted sum) — composes with the existing protocol with no driver change.
- **Lands:** `_force.py`.

### A6. Pin MI / CR histogram range per level in the recipes. **[a stated-but-unmet v2 gate]**
The v2 plan committed "the recipes pin intensity ranges," but `MI` / `Correlation
Ratio` (`register._metric`) call `mutual_information` / `correlation_ratio` with
**no** explicit `value_range`, so the kernel falls back to per-batch data
min/max. Across the optimisation the moving image's range drifts, making the
soft-histogram bin assignment (and hence the gradient) piecewise-unstable.
- **Fix:** have the matrix driver (and the dense MI path) compute a fixed
  intensity range once from the full-resolution images and thread it into the
  metric calls per level (a `value_range` on the `MI` / `CorrelationRatio`
  records, pinned by the recipe).
- **Lands:** `_metric.py` + `_core.py` / the recipes.

### A7. Restrict-deformation (deformation-axis masking).
ANTs `--restrict-deformation 1x1x0` zeroes the deformation along chosen axes
(e.g. in-plane-only for 2-D-acquired data). Trivial given the `Force` protocol:
a per-axis weight applied to the velocity update.
- **Fix:** a `restrict` (length-`ndim`) weight on the SVF recipes, multiplied
  into the force after masking (reuse the `_svf._mask_force` site).
- **Lands:** `_svf.py` + the SVF recipes.

### A8. Affine multi-start / search initialisation.
ANTs' affine stage has a search over initial transforms (the basin is narrow).
The suite has a single zero/`init_affine` start. For a large initial
misalignment this fails silently.
- **Fix:** an optional small grid/random search over initial rotations (+ the
  best-of-N by initial metric) before the affine optimise; or document the
  expectation that the caller supplies a good `init_affine` (the moment-based /
  center-of-mass init is a cheap default worth adding).
- **Lands:** the matrix recipes (a thin pre-optimise search).

---

## B. Correctness / robustness

### B1. Closed-form **MI force** (the user-added priority — fMRIPrep's metric, so performance matters). **[high value]**
fMRIPrep's cross-modal registration uses **Mutual Information** (the Mattes
form). Today MI can only drive a diffeomorphic recipe via `MetricForce(MI())`,
which is the generic autodiff escape hatch (a full `jax.grad` of the
soft-histogram cost **per iteration** — slow, "no performance guarantee"). But
MI *has* a known closed-form gradient: the Mattes-MI analytic derivative (Mattes
et al. 2003; what ITK/ANTs actually compute), built from the same soft (Parzen /
B-spline) joint-histogram the suite's `joint_histogram` already forms.
- **Investigate + implement:** a closed-form `metrics.mi_grad(moving, fixed, *,
  bins, ...)` = `∂(MI)/∂(moving intensity)` from the soft-histogram bin
  derivatives (the Parzen-window weight gradients), analogous to the existing
  `lncc_grad`. Then a Tier-1 `MIForce(bins, ...)` implementer (wraps `mi_grad ·
  ∇warped`), so MI-driven SyN/Demons gets the fast path instead of the autodiff
  hatch. **Validate against autodiff** (the parity oracle: `mi_grad` ==
  `jax.grad(mutual_information)` to tolerance), exactly as `lncc_grad` was.
- **Why it matters:** MI is *the* fMRIPrep cross-modal metric, so the deformable
  multimodal path's per-iteration cost is performance-critical, not a niche
  escape hatch. This converts the multimodal headline from "works but slow" to a
  first-class fast path.
- **Lands:** `nitrix.metrics.information` (the gradient) + `register._force`
  (`MIForce`). Same as the correlation-ratio gradient if it earns one.

### B2. `MetricForce(MI)` driving Demons is unclamped + magnitude-arbitrary.
`MetricForce` rescales the cost gradient by `warped.size` to undo the metric's
*spatial-mean* reduction so it matches the sum-convention closed forms. That is
exact for spatial-mean metrics (LNCC, SSD) but for MI/CR (a histogram scalar, not
a per-voxel mean) the `· size` is a dimensionally-arbitrary constant. The greedy
SyN driver clamps the force (`step` set), so the constant cancels; the **Demons**
driver runs with `step=None` (no clamp), so a `MetricForce(MI())`-driven Demons
gets an unclamped, size-scaled MI gradient added straight to the velocity — a
magnitude no one has tuned, and there is no test that MI *drives Demons to
convergence* (only that the force is finite/nonzero).
- **Fix:** when the metric is not a spatial mean (or generally, for the escape
  hatch under an unclamped driver), normalise the `MetricForce` output by its own
  max/RMS norm to a controlled magnitude (so the step is metric-scale-invariant);
  add a `MetricForce(MI)` → Demons recovery test.
- **Largely mooted by B1** (a closed-form `MIForce` is the real path); the
  escape-hatch normalisation is still worth fixing for arbitrary user metrics.
- **Lands:** `_force.py` + tests.

### B3. `transform_mean` / `matrix_log` need convergence + domain guards.
- `transform_mean` (geometry.algebra) runs a **fixed 10** Karcher iterations from
  `transforms[0]` with no residual check; for a widely-dispersed cohort (rotation
  spread ≳ π/2) the log/exp fixed point can converge slowly or to the wrong
  branch and silently return a non-converged mean. Tests only cover tightly
  clustered symmetric inputs.
- `matrix_log` runs **fixed** `n_sqrt=6` / `db_iters=8` with no convergence check
  and **no domain check** for eigenvalues on the negative real axis (its stated
  precondition) — a reflection (negative-determinant linear block, which a
  user-supplied matrix can carry even though `affine_exp` cannot) returns
  NaN/garbage silently. Denman–Beavers has no scaling/balancing, so a
  large-`‖A−I‖` / poorly-scaled affine may not converge in 8 iterations.
- **Fix:** (a) `transform_mean` — iterate to a tangent-norm tolerance with a hard
  cap (or document the cap + add a dispersed-cohort test); (b) `matrix_log` — a
  cheap domain guard (warn / NaN-with-reason on a non-positive real eigenvalue or
  `‖A−I‖` beyond the Taylor radius), and a balancing step before Denman–Beavers
  for ill-scaled inputs; add a large-`‖A−I‖` / near-singular stress test.
- **Lands:** `geometry/algebra.py`, `linalg/matrix_function.py` + tests.

### B4. `_normalise_step` global-max sensitivity → RMS normalisation.
The SyN trust-region clamp uses a single global `max(‖u‖)` over the field, so one
outlier voxel (edge/boundary/hot voxel) sets the cap for the whole field and
starves the real signal. Normalise by the field's RMS / a high percentile
instead of the global max.
- **Lands:** `_svf._normalise_step`. **Interacts with A1** (winsorize removes the
  usual outlier source).

### B5. `invert_displacement` robustness under large symmetric deformation.
Greedy SyN's final composition `φ = (id+s_fwd) ∘ (id+s_inv)⁻¹` calls
`invert_displacement` (a fixed 50-iteration Picard fixed-point that converges for
`‖∇s‖ < 1`). Under the large symmetric deformation SyN exists for, the half-warp
can approach that boundary and the inversion silently under-converges, corrupting
the returned `displacement` / `jacobian_det`. The v2 plan flagged this as a
to-do; no test plants a large-deformation case and checks the inversion residual.
- **Fix:** add an inversion-residual check (or an Anderson/Newton acceleration of
  the fixed point), and an adversarial large-deformation SyN test asserting
  `(id+s)∘(id+s_inv) ≈ id` to tolerance.
- **Lands:** `geometry/deformation.py` (the solver), `register/_syn.py` (usage),
  tests.

### B6. Mask semantics: document soft-by-smoothing + the weight contract.
The region mask gates the **raw** force before fluid smoothing, so the effective
masked region grows with `sigma_fluid` (the regulariser bleeds force across the
mask edge) — defensible ("the rest follows by regularisation") but it means a
binary mask is *soft in effect*, and coarse pyramid levels turn it fractional.
Also values outside `[0, 1]` scale the force up (it's a weight field, not a hard
mask). Document both; optionally offer a hard variant that also restricts the
metric sampling domain (closer to ANTs `-x` semantics). Add a test that out-of-
mask deformation is actually suppressed (not just that in-mask recovers).
- **Lands:** docstrings + `test_masks.py`.

---

## C. Design / abstraction / parsimony

### C1. Unify the three fast/slow-dispatch vocabularies.
The same conceptual axis — "take the fast specialised path when its preconditions
hold, else the generic path" — is exposed three different ways: a `method=`
string on the matrix recipes (`"auto"/"forward"/"inverse_compositional"`, with a
hand-written precondition boolean), a `Force` *object* on the dense recipes
(dispatch carried by the implementer's type), and a `Convergence` *record* for
early-exit. The `method=` string is the odd one out (re-implements precondition
logic). Either lift the matrix-solver choice to an ADT mirroring `Force`, or at
minimum share the validation (**C2**). Document the divergence if kept.

### C2. De-duplicate the `method` resolution (`recipes.py` ↔ `_volreg.py`).
`recipes._use_inverse_compositional` and the inline `method` branch in `volreg`
implement the same auto/forward/inverse_compositional tri-state with the same
error strings (volreg just drops the metric/model checks). Factor a shared
`resolve_ic_method(method, *, supported) -> bool`; one error-string source.

### C3. `spec.convergence` is silently inert off the IC path.
`RegistrationSpec.convergence` sits on the shared spec but is honoured **only** by
the matrix IC path. Setting it on a `method="forward"` / `WorldSpace` / non-SSD
run is silently ignored (no early-exit, no error). **Fix:** raise when
`convergence is not None` and the resolved path cannot honour it (parallels the
`method="inverse_compositional"` precondition raise), or move `convergence` off
the shared `RegistrationSpec`. (A3 — extending early-exit to the SVF path — also
relieves this on the dense side.)

### C4. Factor the SVF recipe preamble/postamble. **[clearest round-introduced under-factoring]**
`diffeomorphic_demons_register` and `greedy_syn_register` are ~80 % identical by
line: same ndim validation, same `resolve_init_displacement` + shape-mismatch
raise, same dual `gaussian_pyramid` builds, same `_relative_spacing`, same
`resolve_force_schedule`, same mask-pyramid ternary, same `level_solve` shape,
same `finalize_with_init`. The genuine differences are only `n_fields` (1 vs 2),
the per-level update (single-sided vs symmetric), and the residual assembly
(`integrate` vs `s_fwd ∘ s_inv⁻¹`). **Fix:** a `_svf.prepare_svf_inputs(moving,
fixed, *, levels, factor, sigma, boundary_mode, spacing, force, default_force,
init_affine, init_displacement, mask) -> (pyr_m, pyr_f, pyr_mask, forces,
rel_spacing, init_disp)` that both recipes call, collapsing them to: prepare →
define `level_solve` → `svf_coarse_to_fine` → assemble-residual → `finalize_with_-
init`. The `_svf` module docstring already *claims* the recipes are "just the
per-level update + finalisation" — make it true.

### C5. Spec proliferation: shared schedule fields + the `n_steps` mismatch.
`RegistrationSpec` / `DemonsSpec` / `SyNSpec` / `BBRSpec` carry overlapping fields
with drifting defaults: `pyramid_factor` / `pyramid_sigma` are copy-pasted;
`iterations` is per-level (`int | tuple`) on `RegistrationSpec` but plain `int` on
the SVF specs (asymmetric with their per-level *force* support);
`boundary_mode` defaults differ (`'constant'` matrix vs `'nearest'` SVF —
deliberate but undocumented); and **`n_steps` is 6 (Demons) / 5 (SyN) / 7
(`integrate_velocity_field` default)** — the long-flagged nit, still open, which
also leaks into the *tests* (they build ground-truth warps with the default
`n_steps=7` while the recipes integrate with 5/6). **Fix:** a shared
`PyramidSchedule` mixin/record for the pyramid fields; reconcile or explicitly
justify the `n_steps` values per recipe (and fix the test ground-truth to match
the recipe's `n_steps`).

### C6. `cost_history` format varies by path.
`RegistrationResult.cost_history` is the GN/LM per-step trace on the forward path,
`0.5·Σerr²` per IC step on the IC path, `[init, final]` on the BFGS path, and a
fixed-length **padded** trace on the early-exit path. The fast paths advertise as
"interchangeable" with the parity oracle, but `cost_history` is not. **Fix:**
document the field as "format/length depends on the resolved path," or normalise.

### C7. Hoist the windowed-convergence slope test.
The least-squares-slope convergence criterion currently lives inside
`_inverse_compositional._ic_level_converge` but is pure numerics with no IC
dependency. When early-exit extends to the SVF path (**A3**) it would duplicate;
hoist it to a shared helper (`register/_converge.py` or `_core`) now or as part of
A3.

### C8. Document / guard the affine-IC-under-`volreg` landmine.
`volreg` is rigid-only (it hardcodes `Rigid()` + `_rigid_params_from_matrix`), so
the affine IC path is never reached under its `vmap`. But the generalised
`ic_register_core` *accepts* an affine kernel, and `_affine_params_from_matrix`
calls `matrix_log` (→ `safe_inv`); wiring affine into `volreg` would put that
`safe_inv` **inside the `vmap`**, where the wedged-cuSolver fallback breaks.
**Fix:** an assertion / comment at the volreg dispatch (affine + vmap
unsupported), so the landmine is documented before someone steps on it.

---

## D. Validation hygiene (in-repo portion)

### D1. Strengthen the synthetic test suite with adversarial cases.
Everything is currently Gaussian-blob synthetic recovery. Add (in-repo, not the
delegated real-data parity): a folding-inducing large deformation; a near-singular
/ large-shear affine (the stiff-affine IC recovery guard + the Jacobi
ill-conditioned-Hessian unit test, **already added as a merge blocker**); an
antipodal-rotation Karcher mean; a zero-variance / flat-region
image; a degenerate single-axis grid (`_space._grid_scale` guard is untested);
NaN/inf input handling; and a too-large BBR offset that should fail *gracefully*.

### D2. Re-certify the early-exit speedup with the as-shipped criterion.
The 1.69× hard-case gate was measured with a `‖δ‖` step-norm criterion in the
probe, but the shipped `_ic_level_converge` uses the windowed cost-slope
criterion. Re-bench the shipped criterion (the recovery parity is already tested;
the speedup is not certified as-shipped).

### D3. Mark the docs honestly.
Update the v3 / v2 design docs and the `register` package docstring: "ANTs-*style*,
synthetic-recovery-validated; winsorize / histogram-matching / multi-metric-sum /
restrict-deformation / MI-fast-force not yet implemented; real-data + ANTs-
reference parity is delegated to nitrix-perf-bench." (The over-claim of unqualified
"ANTs parity" was the review's headline doc finding.)

---

## Out of scope here (delegated)

- **ANTs-reference parity** (compare nitrix transforms/warps to an
  `antsRegistration` reference) and **real-data validation** (vs `3dvolreg` /
  `mcflirt` / ANTs on real volumes) — **delegated to the nitrix-perf-bench
  agent**, which owns the cross-tool harness and the iso-accuracy measurement.

## Cross-references

- `docs/design/registration-suite-v3.md` (the round), `registration-suite-v2.md`
  (the prior round + its open validation gaps).
- `docs/feature-requests/{registration-matrix-recipe-perf-levers,
  registration-early-stopping-while-loop, pallas-demons-esm-force,
  interpolation-backend-cpu-gpu-gap}.md`.
- `src/nitrix/{register,metrics,geometry,linalg}/`.
