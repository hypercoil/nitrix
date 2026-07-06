# Statistical modelling suite — mass-univariate GLM / GAM / GAMM, LME size-dispatch, and TFCE `randomise` (ledger)

> **Status (2026-06-16): planned — agreed, not started.** Context/ledger doc
> for a consumer-driven statistical-modelling sprint spanning three sibling
> requests that share **one substrate**: (1) the perf-bench agent's LME
> **size-dispatch + cuSOLVER bypass** ([`lme-family-tiny-linalg-gpu-block-and-perf`](lme-family-tiny-linalg-gpu-block-and-perf.md),
> [`gpu-cusolver-first-call-handle-failure`](gpu-cusolver-first-call-handle-failure.md));
> (2) ModelArray-parity **GAMs / GAMMs / GLMs** on GPU; (3) niffi's flagged
> **TFCE-backed `randomise`** cluster-correction kernel it cannot lift out of
> FSL. This doc is the shared framing + the atomised work-item index; the
> per-workstream detail lives in the sections below. Scope is governed by
> SPEC §1 (substrate role) and the **v0.5 §1 score-kernel ↔ scalarisation
> boundary** (these are *score kernels* — arrays → statistic arrays — not
> losses, not containers).

## §0. The organising insight (why this is one sprint, not three)

The three requests look independent. They collapse onto a small set of shared
primitives:

1. **One mass-univariate batching spine.** LME, GLM, GAM, and `randomise` all
   do *"fit the same design to `V` elements (voxels / vertices / fixels),
   shared design, only the response varies."* That is one chunked-`vmap`
   pattern with a memory knob — the pattern `reml_fit` already uses
   (`reml.py:467`), generalised into the common spine and given the
   block-chunking the OOM fix needs.

2. **Penalised GLM ≡ variance-components REML ≡ mixed model.** Wood's
   mixed-model equivalence (the basis of `mgcv`) means a GAM's smoothing-
   parameter selection *is* a variance-components REML fit (each spline
   penalty is a random-effect precision). So the **same REML engine** that
   powers `reml_fit` / `flame_two_level` powers GAM smoothness selection, and
   a **GAMM is just "add explicit random-effect components to the same
   `K`-component model."** This is what makes ModelArray parity tractable
   *inside the existing substrate* rather than as a parallel implementation.

3. **The dispatch pattern already exists in-house.** `linalg/_eigsolve.py`'s
   `forward(method) ⟂ backward(format)` design — a frozen hashable
   `SolverSpec` (`_eigsolve.py:93`) riding `custom_vjp` nondiff args, a
   `Protocol` for the open kernel set, `isinstance` / `Literal` for closed
   sets, a pure testable `auto` policy, an explicit validity table — is
   exactly the template the perf-bench agent asks for on LME size-dispatch.
   We reuse that design language verbatim. See
   [`../design/eigsolve-dispatcher.md`](../design/eigsolve-dispatcher.md).

So the deliverable is **one variance-components core, one GLM core, one
spline-basis core, one permutation/TFCE inference core, and one chunked-`vmap`
utility**, wired together — not three bolted-on features.

## §1. Scope boundary (nitrix vs the ecosystem)

Per SPEC §1 and v0.5 §1 (normative):

- nitrix ships **score kernels**: arrays in → statistic arrays out
  (coefficients, t/F maps, EDF, smooth partial effects, p-maps, null
  distributions). Same classification `lme` and `gaussian_nll` already carry
  in the v0.5 §1.3 table.
- nitrix does **not** ship: the FSL-`randomise` CLI, `.mat` / `.con` / `.grp`
  design parsing, HDF5 / ConFixel containers, BIDS, exchangeability-block file
  formats, or any "loss" / scalarisation. Those are **niffi / thrux / bitsjax
  / nimox**. niffi flagged it needs the *kernel* it cannot lift out of FSL —
  nitrix provides the on-device permutation + TFCE engine; niffi wraps it as
  the `randomise` emulation.
- **RNG policy** stays the caller's; nitrix exposes **keyed pure generators**
  (permutations / sign-flips), blessed by v0.5 §2 ("keyed pure generators are
  in scope").
- **Differentiability.** The *fits* (GLM / GAM / LME) are differentiable
  (linear solves, scan-unrolled Newton). The *inference loop* (TFCE /
  connected-components / permutation) is a **non-differentiable inference
  kernel**, consistent with how `morphology.connected_components` is already
  treated (subgradients "where appropriate", SPEC §2 tenet 2). Stated as an explicit
  design decision, not an omission.

## §2. Locked decisions (2026-06-16)

| # | Decision | Consequence |
|---|---|---|
| D1 | **Sequencing: ModelArray first.** After the LME fix, build the GLM/GAM/GAMM modelling layer; `randomise`/TFCE follows. | Phases below ordered A → B → C. |
| D2 | **GAM families: full exponential family at first GA.** P-IRLS in the core from day one, not deferred. | The penalised solve is IRLS-shaped from the start; λ-selection is Laplace-approximate REML for non-Gaussian families. |
| D3 | **Refactor sealed LME onto the shared `_varcomp` core now.** | `reml_fit` / `flame_two_level` re-expressed with frozen public surface + HLO/statsmodels regression guards (golden-output stability). |

## §3. Workstream A — LME size-dispatch + cuSOLVER bypass

> **Status (2026-06-16): SHIPPED on `feat/stats-suite-modelarray-tfce`.** The
> shared `stats/lme/_varcomp.py` core landed and `reml_fit` /
> `flame_two_level` are re-expressed on it (public surface frozen + a new
> optional `block=` chunking knob). The per-voxel path is cuSOLVER-free at
> every `p` (closed-form for `p <= 2`; an unrolled hand-Cholesky +
> `triangular_solve`/cuBLAS `trsm` for `p > 2`), with analytic AI-REML score +
> average-information replacing the second-order autodiff. `flame_two_level`
> now runs on the broken-cuSOLVER L4 (the original skip is gone). Gate met:
> statsmodels parity held (5e-3); ANOVA closed form to 2e-11; analytic score
> vs autodiff to 1e-10; per-voxel HLO carries only cuBLAS custom-calls;
> **50 / 50 fresh-process GPU trials pass** (`bench/validate_lme_gpu.py`),
> satisfying the cuSOLVER FR's repeated-measurement demand. Deferred within A:
> the q-rank `Z`-decomposition (the shared `safe_eigh(ZZ^T)` stays, routing to
> CPU via its latch -- secondary for runtime and GPU-risky to swap; tracked
> for a follow-up).

**Driver.** [`lme-family-tiny-linalg-gpu-block-and-perf`](lme-family-tiny-linalg-gpu-block-and-perf.md)
(measured: 3–6× CPU steady, 3–5× flatter compile, ~700× GPU at `V=65536`) +
[`gpu-cusolver-first-call-handle-failure`](gpu-cusolver-first-call-handle-failure.md)
(the `flame_two_level` GPU skip). Highest-confidence, lowest-risk, measured —
**lands first**.

**New shared core `stats/lme/_varcomp.py`:**

- **Static size-dispatch on `p`** (fixed-effect width — a compile-time shape,
  so a Python `if`, not `lax.cond`): `p==1` → scalar closed form; `p==2` →
  explicit symmetric `2×2` (`det=ac−b²`, closed inverse + `log_det`); `p>2` →
  an **LU-based** solve (`linalg.solve` / `safe_solve`), *not* Cholesky/eigh.
  Removes every per-voxel cuSOLVER `potrf` / `syevd` from the hot path → (a)
  unblocks `flame_two_level` on GPU, (b) flattens the linear-in-`V` compile.
  Replaces `reml.py:188-190` and `flame.py:113-115`.
- **Analytic REML score & Fisher information** (AI-REML for the `K`-component
  diagonalised model; closed-form single-parameter for FLAME) replacing
  `jax.hessian` / `grad(grad)` (`reml.py:225,231`; `flame.py:140,147`). Keep
  the Newton + Levenberg damping + backtracking control flow (fixed scans →
  vmap-clean); feed it analytic derivatives. Shrinks the graph (compile) and
  the per-voxel autodiff tape (memory / OOM).
- **q-rank decomposition.** Replace `safe_eigh(ZZ^T)` (`reml.py:448`; O(N³),
  full `N×N`, the part that latches to CPU) with `Z`'s thin SVD / a `q×q` eig
  → ZZ^T eigenpairs in O(Nq²). Changes the cuSOLVER routine set → verify on
  the target GPU (FR caveat).
- **Voxel-block chunking.** `_blocked_vmap(fn, *, block)` (`lax.map` over voxel
  blocks, `vmap` inside) bounds peak memory as a tunable knob → fixes the
  `V=262144` OOM. Default `block=None` preserves today's behaviour.
- **`VarCompSpec`** — frozen / hashable dataclass (`n_iter`, `damping`,
  `max_step`, `n_backtrack`, `ridge`, solver policy) with classmethod
  builders, riding the `custom_vjp` nondiff slot — mirroring `SolverSpec`.
- **Working-response entry point** (for Workstream B): in addition to the
  exact-Gaussian REML path, `_varcomp` exposes a P-IRLS working-response form
  so GAM's Laplace-REML λ-selection feeds the same analytic-score Newton
  machinery. Exact-Gaussian REML (LME) stays the fast special case.

**`reml_fit` / `flame_two_level` re-expressed** on this core, **public
signatures and return types frozen** (LME design is sealed). Regression guard
(per the cuSOLVER FR's verification demands):

1. `statsmodels.MixedLM` parity unchanged at the pinned tolerances.
2. HLO `custom_call_target` diff proving **zero** per-voxel cuSOLVER calls
   survive (the `potrf` / `syevd` must be gone, not merely warmed-up).
3. **≥50 fresh-process GPU runs** for availability — one green run is treated
   as *no evidence*; require a stable pass rate.

## §4. Workstream B — mass-univariate GLM + GAM / GAMM (ModelArray parity)

> **Status (2026-06-17): SHIPPED on `feat/stats-suite-modelarray-tfce`.**
> `stats/glm.py`, `stats/basis.py`, `stats/gam.py` landed, exported from
> `nitrix.stats`. **GLM** (`glm_fit`): OLS fast path + exponential-family
> P-IRLS, `Family` (Gaussian/Binomial/Poisson + custom), `t_contrast` /
> `f_contrast`, `r_squared`/`adj_r_squared`/`deviance_explained`, `aic`/`bic`/
> `log_likelihood`, `compare_models` (F / LRT). **basis** (`bspline_basis`):
> P-spline design + difference penalty + Householder sum-to-zero (cuSOLVER-free).
> **GAM/GAMM** (`gam_fit`): penalised IRLS inner + generalized Fellner-Schall
> REML outer (per-element `lambda`), per-smooth EDF, Bayesian covariance,
> `smooth_partial_effect`. Validated: GLM vs statsmodels (OLS coef 2e-16 +
> se/t/p/R²/F exact; WLS exact; Poisson 1e-11; Binomial 7e-13; llf exact; AIC =
> R/`lm` convention; F/LRT vs scipy 1e-8); GAM inner fit == penalised normal
> equations (1e-7), REML recovers smooths, EDF == influence trace (1e-4),
> noise→smoothing monotone, Poisson GAM, additive 2-smooth, batched==looped;
> everything cuSOLVER-free on GPU. **Compile perf:** a diagnostic decomposition
> found GAM compile was ≈cubic in design width `p` (96 s at `p=30`) due to the
> *unrolled* per-element Cholesky (`O(p^3)` trace-time ops; iteration count was
> already flat — loops roll); switched to a *rolled* `lax.fori_loop` Cholesky
> (`O(p^2)` graph) → compile flat in `p` (96→1.8 s at `p=30`, 14–53× across
> `k`), guarded by a jaxpr-size test. Deferred: cyclic / thin-plate bases;
> shared-`lambda` fast mode; per-element-`lambda` Newton (FS is the shipped
> selector).

**`stats/glm.py`** — `glm_fit(Y, X, *, weights, l2, family, method) →
GLMResult`, shared-design batched over elements (+ a per-element-design
variant), reusing `residualise`'s Cholesky / SVD solver paths
(`residual.py:73,99`) and a shared `(p,p)` factor for SEs. The core solve is
**IRLS-shaped** (takes working weights `W` and working response `z`): OLS =
one iteration, WLS = weighted, P-IRLS = the inner loop. `GLMResult` (frozen
pytree): `betas (V,p)`, the shared `(p,p)` factor, `dof_resid`, `sigma2 (V,)`,
`deviance`. Inference primitives:

- `t_contrast(result, c) → (effect, se, t, p)`,
  `f_contrast(result, C) → (F, df1, df2, p)`.
- `r_squared` / `adj_r_squared`, `deviance_explained`, `aic` / `bic`.
- `compare_models(full, reduced) → (F | LRT, p)` — ModelArray's
  `fullVs.reduced`.

**`stats/basis.py`** — penalised-spline basis builders, returning
`(design_block, penalty S)` with sum-to-zero identifiability reparameterisation:

- `bspline_basis`, `cyclic_cubic_basis`, `thinplate_regression_basis` (TPRS via
  eigen-truncation of the TPS penalty).
- **Reuse**: extract the 1-D cubic B-spline evaluator + difference penalty
  already in `bias/_bspline.py` (`_uniform_bspline_weights:78`,
  `_difference_penalty:305`, `_weighted_gram:276`) so N4 and GAM share one
  basis primitive (the per-axis `R_d` weight row *is* a B-spline design row).

**`stats/gam.py`** — `gam_fit(Y, design, *, family, method) → GAMResult`.
Assemble `X = [parametric | smooth bases]`, block penalty `S(λ) = Σ_k λ_k S_k`:

- **Gaussian-identity** → penalised least squares (one solve).
- **Non-Gaussian** → P-IRLS outer loop (Fisher-scoring weights) wrapping the
  penalised solve, `Family` a frozen-registry **Protocol** of pure
  `(link, variance, deviance, mu_eta)` callables (open set → Protocol, exactly
  as eigsolve reserves Protocol for its forward-kernel set).
- **λ-selection via the Workstream-A `_varcomp` engine** (penalty ↔
  random-effect precision). Per-smooth eigen-reparameterisation diagonalises
  each `S_k` → the FaST-LMM diagonal trick still applies per block, and the
  same size-dispatch logic carries over. Exact REML for Gaussian;
  Laplace-approximate REML (Fellner–Schall / outer Newton on the LAML) for
  non-Gaussian.
- **GAMM** = add explicit random-effect components to the same `K`-component
  REML (falls out for free).
- `GAMResult` (ModelArray output parity): coefficients, per-smooth **EDF**
  (trace of the influence matrix) + `ref.df`, smooth **partial effects** on a
  grid, per-smooth F/p, parametric-term stats, R² / deviance-explained, AIC /
  BIC, λ̂.
- Mass-univariate via the chunked-vmap spine; **per-element λ-selection** is
  the ModelArray-parity default, **shared/global λ** a fast mode.

## §5. Workstream C — TFCE + `randomise` inference (niffi gap)

> **Status (2026-06-17): SHIPPED on `feat/stats-suite-modelarray-tfce`.**
> `stats/inference/` landed: `permutation.py` (keyed `sign_flips` /
> `permutations` + exchangeability blocks, identity-first), `tfce.py` (TFCE on
> `morphology.connected_components`, two-sided, matches scipy.ndimage to 1e-5),
> `cluster.py` (size / mass maps), `randomise.py` (`permutation_test`:
> OLS refit + Freedman-Lane nuisance, per-permutation enhancement, FWE +
> uncorrected p-maps + null-max — observed captured as permutation 0 *inside*
> the scan so the identity reproduces it bit-exactly and `p_fwe >= 1/n_perm`),
> `multiple_comparisons.py` (`fdr_bh` / `bonferroni` vs statsmodels 1e-10). All
> cuSOLVER-free on GPU. Validated: observed t == scipy one/two-sample (1e-10);
> FWE floor; signal recovery; null FWE control; Freedman-Lane observed == GLM
> t-contrast. Deferred: cluster-extent/mass *enhancement* modes in the driver
> (the maps exist in `cluster.py`); F-contrast statistic; tail-acceleration for
> very large `n_perm`. niffi consumes `permutation_test` (arrays in/out); the
> `randomise` CLI / `.con`/`.grp` parsing / containers stay in niffi.

**`stats/inference/permutation.py`** — keyed pure generators `sign_flips` /
`permutations` honouring **exchangeability blocks** (within-block permute /
whole-block swap), plus a **Freedman–Lane** helper (partition design into
effect + nuisance, regress nuisance out, permute residuals — the standard
nuisance-aware scheme). Returns `(n_perm, N)` index / sign arrays. (v0.5 §2.)

**`stats/inference/tfce.py`** — `tfce(stat_map, *, E=0.5, H=2.0, dh,
connectivity, mask) → enhanced`, over a fixed threshold grid: per step
threshold → `morphology.connected_components` (`morphology/_label.py`) →
per-voxel cluster extent (bincount gather) → accumulate `e(h)^E · h^H · dh`.
Fixed-shape, jit-able; two-sided support; surface / 1-D parameterisations
(connectivity + E/H). Built directly on the existing connected-components
engine.

**`stats/inference/cluster.py`** — `cluster_extent` / `cluster_mass`
thresholding + the spatial-max extractors (voxelwise-max / cluster-max /
tfce-max) for null-distribution building.

**`stats/inference/randomise.py`** — `permutation_test(Y, X, contrast, *, stat,
enhancement, n_perm, key, blocks, var_smooth, mask) → PermResult`: fit observed
GLM → statistic (t/F; optional variance-smoothed pseudo-t via
`smoothing.gaussian`, FSL `-v`) → enhancement → `lax.map` / `scan` over
permutations (Freedman–Lane refit, recompute statistic + enhancement, spatial
max) → null max distribution → **FWE-corrected** p-map (mean over perms of
`max ≥ observed`) + **uncorrected** voxelwise p-map. `PermResult` (frozen
pytree): `stat_map`, `enhanced_map`, `p_fwe`, `p_uncorr`, `null_max`. Arrays
only; niffi wraps the CLI / containers / design parsing.

**`stats/inference/multiple_comparisons.py`** — `fdr_bh`, `bonferroni`,
`cluster_pvalues` companions (pure array ops).

## §6. Workstream D — corpus review findings & cross-cutting cleanups

Concrete improvement vectors found while reviewing the existing stats corpus;
several *are* the same refactor as a feature above:

1. **DRY the REML machinery.** `reml.py` and `flame.py` duplicate the Newton /
   backtracking / log-parameterisation / final-`beta` blocks. Extracting
   `_varcomp` (§3) removes the duplication *and* yields the GAM engine — one
   refactor, two payoffs.
2. **Single reduction surface.** v0.5 §1.4 collapses the duplicated `_reduce`
   helpers into `nitrix._internal.reductions.reduce` (`reductions.py:42`,
   already shipped). New stats modules consume that surface; they must **not**
   add a fourth `_reduce`.
3. **Closed-form derivatives over autodiff** in hot inner loops (compile +
   memory) — a general principle the review applies beyond LME.
4. **One chunking utility** (`_blocked_vmap`) shared by lme / glm / gam /
   randomise — the mass-univariate memory knob, currently absent (the OOM root
   cause).
5. **Route PCA top-k / aCompCor through `eigsolve_top_k`** (`_eigsolve.py:829`)
   rather than bespoke iteration. The eigsolve design doc names **aCompCor as
   the second consumer that promotes `eigsolve_top_k` to public** — a
   nuisance-Gram component extractor in this suite *is* that consumer, closing
   that deferred promotion. See [`pca-svd`](resolved/pca-svd.md).
6. **covariance.py audit.** Confirm the matrix-weight bias correction and
   complex-Hermitian paths against the documented tests, and that
   `conditionalcov` / `precision` route through `residualise` consistently —
   verifying the "silently wrong" JIT-trap stays closed
   ([`../design/stats.md`](../design/stats.md)). Doc/regression pass; no new
   code expected unless a gap surfaces.
7. **Typing / immutability discipline.** Every result is a frozen-dataclass /
   NamedTuple registered pytree; every config a frozen **hashable** `Spec`;
   **Protocol** for the open kernel sets (solver / family / enhancement),
   `Literal` / `isinstance` for closed sets; jaxtyping on all boundaries; ruff
   house style (double-quote docstrings / single-quote strings,
   `from __future__ import annotations`, `__all__`, emacs header).

## §7. Proposed module layout

```
nitrix/stats/
  glm.py                 # GLMResult, glm_fit (IRLS-shaped), t_contrast, f_contrast, compare_models, r2/aic/bic
  gam.py                 # GAMResult, gam_fit (penalised GLM + Laplace-REML λ-selection; GAMM; Family protocol)
  basis.py               # bspline_basis, cyclic_cubic_basis, thinplate_regression_basis (reuse bias/_bspline 1-D core)
  lme/
    _varcomp.py          # VarCompSpec; shared REML core: size-dispatch on p, analytic AI-REML, q-rank, working-response, _blocked_vmap
    reml.py  flame.py    # re-expressed on _varcomp; public surface frozen
  inference/
    permutation.py       # keyed sign_flips / permutations (exchangeability blocks) + Freedman–Lane
    tfce.py              # tfce() on connected_components
    cluster.py           # cluster_extent / cluster_mass / spatial-max extractors
    randomise.py         # PermResult, permutation_test (on-device randomise engine)
    multiple_comparisons.py  # fdr_bh, bonferroni, cluster_pvalues
```

## §8. Types & protocols (sketch)

```python
# Closed sets → Literal/isinstance; open sets → Protocol; configs → frozen+hashable.
Method = Literal['cholesky', 'lu', 'svd', 'closed_form']         # GLM/varcomp solver
Enhancement = Literal['none', 'tfce', 'cluster_extent', 'cluster_mass']
Statistic = Literal['t', 'f']

class Family(Protocol):                       # the one genuinely open set (B/D7)
    def link(self, mu: Array) -> Array: ...
    def mu_eta(self, eta: Array) -> Array: ...
    def variance(self, mu: Array) -> Array: ...
    def deviance(self, y: Array, mu: Array) -> Array: ...

@dataclass(frozen=True)                        # hashable → rides custom_vjp nondiff args
class VarCompSpec:
    n_iter: int = 20; damping: float = 1e-6; max_step: float = 1.0
    n_backtrack: int = 4; ridge: float = 1e-8
    # classmethod builders: .reml(), .flame(), .gam(...)

class GLMResult(NamedTuple): betas; xtx_factor; dof_resid; sigma2; deviance
class GAMResult(NamedTuple): coef; edf; ref_df; smooths; lam; r2; dev_explained; aic; bic
class PermResult(NamedTuple): stat_map; enhanced_map; p_fwe; p_uncorr; null_max
```

## §9. Phasing & gates

- **Phase A — `_varcomp` + LME re-expression.** §3 + §6.1/§6.2/§6.4. Unblocks
  the perf-bench agent. **Gate:** statsmodels parity held + HLO no-cuSOLVER
  diff + ≥50 fresh-process GPU runs.
- **Phase B — GLM + GAM (Gaussian + full family) + GAMM.** §4. Bulk of
  ModelArray parity. Side-quest §6.5 (eigsolve/PCA routing) + §6.6 (covariance
  audit). **Gate:** parity vs `statsmodels.GLM` and `mgcv::gam` (coefficients,
  EDF, smooth F/p, R²/dev-expl); no-large-intermediate HLO audit (like
  `reml`'s `test_reml_max_tensor_size_within_budget`) extended to glm/gam;
  brain-scale-`V` GPU certification (the repo's "certify at scale" rule).
- **Phase C — TFCE + `randomise`.** §5. Needs Phase-B GLM + existing
  `connected_components` / `smoothing.gaussian`. Unblocks niffi. **Gate:**
  parity vs **FSL `randomise`** (TFCE map, FWE p-map, cluster-extent/mass) and
  **ModelArray** on a fixed seed / permutation set.

## §10. Validation / oracles

External oracles live in `tests/` only (never runtime deps, SPEC §6.2):
`statsmodels.MixedLM` / `statsmodels.GLM`, `mgcv::gam` (rpy2 or pinned
fixtures), FSL `randomise` reference outputs, ModelArray. Each new
mass-univariate op carries the no-large-intermediate HLO audit and a
brain-scale-`V` GPU certification, plus backend-parity where kernels exist
(here mostly XLA + the existing connected-components / gaussian-smoothing).

## §11. Cross-references

- Drivers: [`lme-family-tiny-linalg-gpu-block-and-perf`](lme-family-tiny-linalg-gpu-block-and-perf.md),
  [`gpu-cusolver-first-call-handle-failure`](gpu-cusolver-first-call-handle-failure.md),
  [`pca-svd`](resolved/pca-svd.md).
- Related substrate already filed: [`graphical-lasso`](graphical-lasso.md)
  (§12.14), [`robust-statistics`](robust-statistics.md) (§12.7),
  [`ledoit-wolf-shrinkage`](ledoit-wolf-shrinkage.md),
  [`gaussian-kl-nll`](resolved/gaussian-kl-nll.md),
  [`contrastive-ssl-losses`](resolved/contrastive-ssl-losses.md).
- Design docs: [`../design/lme.md`](../design/lme.md),
  [`../design/stats.md`](../design/stats.md),
  [`../design/eigsolve-dispatcher.md`](../design/eigsolve-dispatcher.md)
  (the dispatch template), [`../design/bias-field.md`](../design/bias-field.md)
  (the B-spline basis to extract).
- Reused code: `src/nitrix/stats/lme/{reml,flame}.py`,
  `src/nitrix/linalg/{residual,_solver,_eigsolve}.py`,
  `src/nitrix/morphology/_label.py`, `src/nitrix/bias/_bspline.py`,
  `src/nitrix/_internal/reductions.py`, `src/nitrix/smoothing/gaussian.py`.
- Governing spec: SPEC §1 (substrate role), §2.2 (non-negotiables),
  SPEC §5 (score-kernel ↔ scalarisation boundary), §2 (keyed pure
  generators); SPEC §4.6 (LME promotion).
