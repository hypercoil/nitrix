# `nitrix.stats` comprehensive audit — findings register — `nitrix.stats`

> **Status (2026-06-20): open register.** Consolidated findings from a seven-lens
> fan-out review of the *entire standing* stats suite (~12.5k LOC) — mathematical
> correctness, engineering rigour, neuroimaging community use, code organisation,
> design / abstraction, performance (algorithm choice + XLA/JIT compile), and
> hardware-aware algorithms. High-severity items were **empirically verified**
> (the reviewers could not run code); the verification evidence is recorded inline
> so findings are not re-litigated. This is a tracking ledger: update the
> **Status** column (`open` → `wip` → `done` / `deferred` / `refuted`) as items
> are worked through.

## Verdict

The suite is in **strong shape**: the math is **sound** (no correctness bug found
in any subsystem; every load-bearing derivation hand-verified against references),
it is **very well matched** to the dead-cuSOLVER L4 (the per-element path is
grep-clean of cuSOLVER custom-calls), and the design / organisation is deliberate
and mostly excellent. The single-source-of-truth factoring (`_irls` / `safe_dmu`
/ `_varcomp` / `small_inv_logdet`), the analytic AI-REML, the FaST-LMM / Woodbury
structure exploitation, and the calibration-aware permutation engine were praised
unprompted by multiple reviewers. The findings below are a **small set of
contained bugs** plus organisational / contract refinements and capability gaps —
none threatening the suite's core trustworthiness.

## ⚠️ Record correction — C1 ("gappy group labels bias REML") is REFUTED

The engineering-rigour lens flagged, as its sole **Critical**, that non-contiguous
group labels (e.g. `{0, 2, …}` from subject exclusion; `n_groups = max(group)+1`
creates phantom empty groups) silently bias the R2 block-Woodbury REML fit. **This
is false** — do not re-investigate:

- **Empirical:** fitting the same data with contiguous labels vs. remapped
  `g → 2g` (inserting `q−1` phantom empty groups) gives a **bit-identical** fit —
  `cov_re = [0.4633, 0.2509, 0.2509, 0.3816]`, `sigma_e_sq = 0.28656`,
  `beta = [−0.0802, 0.4431]` in both.
- **Why:** a phantom empty group contributes `K_i = sigma_e^2 G^{-1}` (since
  `Z_i^T Z_i = 0`), and its terms **cancel exactly** in the block-Woodbury
  `log|V| = n_minus_mr·log sigma_e^2 + M·logdet_g + sum_g logdet_k`: the
  `logdet_k = r·log sigma_e^2 − logdet_g` it adds is exactly offset by the
  `+logdet_g` (from `M·logdet_g`) and the `−r·log sigma_e^2` (from `n_minus_mr`
  losing `r`). It contributes nothing to the score/data terms (`ztz = zty = 0`).
  The GLMM-Laplace version cancels identically (`−0.5 log sb2 − 0.5 log(1/sb2) =
  0`), and the REML-EM is self-consistent at its fixed point (an empty group's
  posterior = prior contributes exactly `G` to the mean, neutral at `G = S/q`).

The **narrower** sibling the same lens bundled under H1 — the `sandwich_cov`
**cluster-count** correction — **is** real (see **B3**), because there a *count*
of groups enters a normaliser with no cancellation. Lesson logged: count-normalised
paths are the only ones the `max+1` pattern can bite; the likelihood paths are
immune.

## Verified bugs (fix candidates)

| ID | Sev | Status | Location | Issue & evidence | Recommendation |
|----|-----|--------|----------|------------------|----------------|
| **B1** | High | ✅ **done** | `stats/covariance.py` `precision` (~358) | `precision` / `partialcov` / `partialcorr` inverted via bare `jnp.linalg.inv` / `pinv` — dead-cuSOLVER `getrf`/`gesvd`. **The only dead call on a compute path** (3 reviewers + read). | **FIXED:** `inv` branch → `safe_inv`; `pinv` branch → new `_sym_pinv` (eigen-truncated, `safe_eigh`). Tests: inverse + rank-deficient pinv vs numpy; also closes **M5** (partialcorr off-diagonal now pinned). |
| **B2** | Med-High | ✅ **done** | `stats/basis.py` `_bspline_design` (~85) | `span` clamped but `frac = s − span` not clamped to `[0,1]`, so out-of-range queries lose partition-of-unity and blow up (row-sums −57/−793, `max|w|` ~655). | **FIXED:** `frac = clip(s − span, 0, 1)` (constant boundary extrapolation) — verified partition-of-unity restored, `max|w| ≤ 0.67` out-of-range, flat extrapolation. The cyclic design already wraps mod-period (no bug). |
| **B3** | Med | ✅ **done** | `stats/glm.py` `sandwich_cov` (~531) | Cluster factor `G/(G−1)` with `G = int(max(groups))+1` — gappy labels inflate `G` (SEs shifted ~3–4%). | **FIXED:** densify via `jnp.unique` + `searchsorted` before counting `G`; SE now label-encoding-invariant (gappy / permuted == contiguous). **Scope:** the few-level GLMM count and the REML paths are **immune** (verified — Fellner-Schall / `log|V|` cancellation, like the C1 refutation), so `sandwich_cov` is the *only* affected path; no shared helper needed. |
| **B4** | — | **refuted** | `lme/reml.py`, `lme/_blockwoodbury.py` | C1 "gappy labels bias REML" — **refuted**, see the record-correction section above (bit-identical fits; exact log|V| cancellation). | None. Kept for provenance so it is not re-raised. |

## Design / abstraction & contract findings

| ID | Sev | Status | Location | Issue | Recommendation |
|----|-----|--------|----------|-------|----------------|
| **D1** | High | open | 8 Result types (`glm.py`/`gam.py`/`glmm.py`/`_betareg.py`/`_gaulss.py`/`_ordinal.py`/`flame.py`/`reml.py`/`randomise.py`) | Pytree `tree_flatten`/`tree_unflatten` hand-rolled ~8× (add-a-field-touch-3-places footgun); and **inconsistent** — the `lme_fit` dispatch returns `NamedTuple`s (`LMEResult`/`NestedLMEResult`/`CrossedLMEResult`/`CorrLMEResult`/`GLSResult`) that flatten `tier`/`corr`/`df_resid` strings/ints as **dynamic leaves**, opposite to the careful static-aux split the dataclass results take. (design F1 + code-org) | One shared `@stats_result(children=(...), aux=(...))` decorator / base; convert the `NamedTuple` results to the registered-pytree + static-aux convention. |
| **D2** | High | open | `lme/reml.py` `lme_fit` (~901) | 5-way `Union` return with the RE variance under a different name/shape per tier (`sigma_b_sq` / `cov_re` / `var_outer`+`var_inner` / `var_group`+`var_cross`); no tier-agnostic accessor (the `LMEResult` docstring already aspires to "uniform across tiers"). | Expose a uniform `cov_re (V,r,r)` + `re_labels` accessor across all five. |
| **D3** | High | conf'd | `lme/reml.py` `lme_f_contrast`/`lme_t_contrast` (~568) | Accept only `REMLResult`, so `lme_f_contrast(lme_fit(...))` raises `TypeError` on R2/R3/R4/+corr. **Verified** — graceful (clear error explaining R2 lacks the inference fields), not silent; but no F/t-contrasts on random-slope/nested/crossed fits. | Fold the inference fields into the shared contract (even if only R1 populates them) or type the contrast fns to the union + dispatch internally. |
| **D4** | Med | open | `stats/glmm.py` `GLMMResult.re_var` (~108) | Silently shape-polymorphic `(V,)` / `(V,r)` / `(V,r,r)` by `z`/`structure` — a `vmap`/indexing footgun; contrast `LMEResult.cov_re` which is *always* `(V,r,r)`. (design F5 + slope review) | Adopt the `LMEResult` convention: always `(V,r,r)` (and `(V,)` → `(V,1,1)` for the scalar intercept), shape stable & tier-independent. |
| **D5** | Low | ✅ **done** | `_family.py` / `glm.py` / `stats/__init__.py` | `resolve_family` reachable but not exported, while `resolve_link` is (3×). | **FIXED:** added to `glm.__all__` + imported / exported from `stats.__init__`. |
| **D6** | Low-Med | open | `glm_fit`/`glmm_fit`/`lme_fit`/`predict`/`sandwich_cov`/… | Dispatch axes (`family`/`method`/`structure`/`type`/`kind`) are bare `str` validated by deep `raise ValueError`; only `gam_fit`/`pca_fit`/`reml.Structure` use `Literal`. `tier` is a free `str` return. | Promote taxonomies to `Literal` (incl. `tier`) so legal values are in the signature / type-checker. |
| **D7** | Low | open | `lme/_blockwoodbury.py` `_param_layout`/`cov_re_from_chol` | Shared RE-covariance (log-Cholesky) helpers reached across module boundaries from `reml.py`, `_corrfit.py`, and `glmm.py` (4× in-function private-name imports). (code-org + design) | Lift to a shared `lme/_recov.py` (or have `fit_blockwoodbury_reml` return `cov_re` directly). |
| **D8** | Low-Med | open | `gam.py` `Smooth = Union[...]`, `_smooth_penalties` `isinstance` chain | New basis type ⇒ edit the union *and* every `isinstance` branch — the open-set registry story `Family`/`CorrSpec` get right is absent. | A `SmoothBasis` `Protocol` (shared `dim`/`design`/`penalty`) or a `penalty_blocks()` method on each basis. |
| **D9** | Taste | open | various | `n_iter` vs `n_outer`/`n_inner`/`n_mode`/`n_quad` naming drift; intercept policy differs (`X`-carries-own vs `intercept=` vs forbidden); `VarCompSpec.reml` is pure ceremony (`= cls(**kw)`); `low_rank` is an R1-only silent no-op. | Standardise single-loop fitters on `n_iter`; document the intercept policy per signature; drop `.reml`; validate/doc `low_rank` R1-only. |

## Code organisation

| ID | Sev | Status | Location | Issue | Recommendation |
|----|-----|--------|----------|-------|----------------|
| **O1** | High | deferred? | `stats/glmm.py` (1556 LOC) | 6-solver monolith (few-level / structured slope / many-level Schur / Laplace / Laplace-slope / AGQ) behind one dispatcher; banners already mark the seams. | Split into a `glmm/` package mirroring `lme/`: `__init__.py` (dispatcher + `GLMMResult`), `_pql.py`, `_slope.py`, `_laplace.py`, `_agq.py`. Pure mechanical relocation. |
| **O2** | High | open | `lme/_optimise.py` `damped_newton` (~127) | The "one shared Newton" is housed *inside* `lme/` yet driven by `_ordinal`/`_betareg`/`_gaulss` (non-mixed-model), and typed on `VarCompSpec` though it reads only primitive fields — forcing `_ordinal` to build a variance-components spec it has no use for. | Move `damped_newton` to a stats-core module (`stats/_optimise.py` beside `_irls`/`_batching`, or `numerics/`); take a small generic config / kwargs; `VarCompSpec` stays in `lme/_varcomp.py`. |
| **O3** | Low | open | `_betareg.py`/`_gaulss.py`/`_ordinal.py` | `_`-prefixed yet export public API (`beta_fit`/…), unlike peer public fitters `glm`/`gam`/`glmm`. | Rename to `betareg.py`/`gaulss.py`/`ordinal.py`, or document "secondary public fitters". |
| **O4** | Low | open | `tests/test_stats.py` | `covariance`/`pca`/`gaussian`/`_irls` fold into `test_stats.py` (no 1:1 test file) while newer fitters are 1:1. | Split `test_stats.py` as those grow. |

## Performance / hardware

| ID | Sev | Status | Location | Issue | Recommendation |
|----|-----|--------|----------|-------|----------------|
| **P1** | High → **none** | **deferred (measured)** | `lme/_corrfit.py` (~241/529), `lme/_crossed.py` (~126/172) | Flagged as "the largest compile graph / OOM risk" — but **the premise did not hold under measurement.** corr-GLS ar1 compiles in **3.4–4.2 s with no OOM** even at the agent's "large groups" sizes (G=120 T=15 n=1800; G=60 T=40 n=2400) — *comparable to* the analytic-curvature R1 sibling (4.2 s) and running faster. The cost is **graph-dominated**; the autodiff Hessian is only a small `theta_dim^2` (~3–5) multiplier, so an analytic curvature (sharing the same graph) would land at the same ~4 s. Same lesson as #36. **No measured impact → deferred** (feasible but not worth the substantial ar1 `∂R/∂rho` AI-REML derivation). The `_crossed` `q2` sequential-Cholesky note stands as a separate (latency-only) item if very large crossed factors ever arise. |
| **P2** | Med | ✅ **done** | `lme/reml.py` R1 (~744) | `low_rank=False` default forced the dense `(N,N)` `ZZ^T` eig where the FaST-LMM `(q,q)` form suffices for `q ≪ N`. | **FIXED:** `low_rank: Optional[bool] = None` auto-selects low-rank when `q < N` (the brain-scale case); explicit `True`/`False` override. Verified `auto == dense` to 1e-6 (incl. gappy/rank-deficient `Z`, absorbed by the null space); 81 LME tests green. |
| **P3** | Med | ✅ **done** | `stats/glmm.py` AGQ | `K = n_quad**r` tensor nodes differentiated through the mode scan — explodes at r≥3. | **FIXED:** `_AGQ_MAX_NODES = 128` cap in the dispatch raises a clear `ValueError` (admits r=2 n_quad≤11, r=3 n_quad≤5; blocks r≥4 / large r=3) pointing to a smaller `n_quad` or Laplace. |
| **P4** | Med | open | `stats/glmm.py` Laplace/Laplace-slope/AGQ | All three marginal GLMM fits take the autodiff-Hessian-through-mode-scan fork. See the **#36** FR (`glmm-random-slope-robust-solver.md`): the clean fix is implicit-diff of the mode (`custom_vjp`/IFT); deferred on ROI (cold-compile, amortised). | Tracked separately (#36). AGQ would benefit identically. |
| **P5** | Low | open | `gam.py` `_smooth_penalties` (~195) | Data-independent penalty eigendecomp (`eigh` of `k×k`) recomputed every `gam_fit` call — wasteful across CV/λ loops. | Cache `penalty_eigs` in the `SplineBasis` container at construction. |
| **P6** | Low | open | `linalg/_smalllinalg.py` `_PIVOT_REL_FLOOR=1e-12` (~62); `covariance.py:134` | Pivot floor / `ridge=1e-8` defaults sit below fp32 eps (~1.2e-7) → inert in fp32 on the squared-condition (X^T X) Cholesky; `covariance.py:134` hard-codes `float32`. Suite quietly assumes x64. (engineering + hardware) | Make `_PIVOT_REL_FLOOR` dtype-aware (`~1e2·finfo(dtype).eps`); document the x64 expectation for ill-conditioned designs; fix the hard-coded dtype. |

## Mathematical correctness — test-gaps & doc (no bugs; all Low)

| ID | Sev | Status | Location | Issue | Recommendation |
|----|-----|--------|----------|-------|----------------|
| **M1** | Med | ✅ **done** (doc) | `glmm.py` Laplace/AGQ (`edf_total=float(p)`, `dispersion=1`) | Stubs: `edf_total` fixed-effect-only, `dispersion=1` wrong for a Gaussian Laplace fit. (engineering + math) | **DOCUMENTED:** `GLMMResult` now states `dispersion`/`edf_total` are placeholders for the laplace/agq tiers (not for AIC), and that `deviance` *is* the comparable -2·marginal-loglik. (Computing the real marginal edf is left as a future enhancement.) |
| **M2** | Med | open | `tests/test_lme.py` `_flame_hand_iter` | The independent hand-computed FLAME oracle is **defined but never asserted** (dead code); FLAME pinned only by recover-truth + self-consistency. | Wire `_flame_hand_iter` into an `assert_allclose`, or delete the dead reference + its docstring claim. |
| **M3** | Low | open | `_blockwoodbury.py` `diagonal=True` | Uncorrelated `(x‖g)` diagonal-G slope has **no** dense oracle (only the unstructured case is checked vs `_dense_r2`). | Add a `diagonal=True` case to the dense-REML battery. |
| **M4** | Low | open | `glmm.py` Laplace/AGQ | Fisher-curvature Laplace is only tested on logit/log (canonical, Fisher = observed Hessian); the non-canonical-link (probit/cloglog slope) path is unpinned. | Add a probit/cloglog slope test vs an independent quadrature. |
| **M5** | Low | ✅ **done** | `covariance.py` `partialcorr` | Off-diagonal not pinned to an external oracle (only `diag==1`). | Closed by **B1**'s `test_partialcorr_offdiagonal_matches_numpy`. |
| **M6** | Low | open | `gam.py` λ-selection | Absolute λ / EDF never compared to mgcv (only inner fit, EDF, FS-identity, and λ-responds). | One mgcv-anchored λ/EDF regression per basis kind. |
| **M7** | Low | open | `linalg/residual.py` JS docstring | "James-Stein dominates OLS in MSE for k≥3" overstates — it's a sound heuristic (plug-in σ²), not the strict theorem. | Soften the wording. |

## Neuroimaging community gaps (capability, not defects)

| ID | Sev | Status | Issue | Recommendation |
|----|-----|--------|-------|----------------|
| **N1** | High value | ✅ **done** | No GAM smooth-term p-values. | **SHIPPED:** `smooth_significance(result, smooths)` → `SmoothTest` (`stat`/`edf`/`rank`/`p_value`, `(V, m)`), the Wood-2013 integer-rank test (QR-projected eigen-pseudo-inverse, χ² for fixed-dispersion / F otherwise), mask-truncated for `vmap`. **Validated to EXACT `mgcv:::testStat(type=1)` parity** (both F and χ² branches, stat/rank/p to 6 digits, R 4.5.3 / mgcv 1.9.4). Regression vs an independent numpy+scipy reference. The fractional-rank default (`type=0`, needs the weighted-χ² CDF / Davies) is a documented follow-up. |
| **N2** | High (surface/dMRI) | open | **TFCE/clustering are lattice-only** — no spin test (Alexander-Bloch/Váša) or mesh/graph adjacency; blocks surface (CIFTI) / fixel (ModelArray) workflows. | A mesh/graph `connected_components` path (the `morphology` dep exists) + a spin-test permutation operator. |
| **N3** | Med | open | **No RFT / ACF-FWHM smoothness estimation** — permutation-only (defensible post-Eklund), but no parametric fallback for tiny-N and no AFNI/SPM parity. | Document the explicit "permutation-only" stance; a `3dFWHMx`-equivalent ACF/FWHM estimator if parity is wanted. |
| **N4** | Med | open | **FLAME has no outlier-deweighting** (FLAME1/FLAMEO) — a named-feature gap vs the FSL comparator. | Surface as a known divergence; the deweighting iteration extends the existing REML loop. |
| **N5** | Med | open | **No effect-size / CI outputs** anywhere (effect+SE+dof are returned, so CIs are one `t_crit` away). | A thin `confidence_interval(effect, se, dof, level)` + a standardized-effect helper. |
| **N6** | Low | open | Cluster-robust SE is **one-way only**; no FSL `-g` variance-group / PALM `-vg`. | Document the `-e` (have it) vs `-g` (don't) distinction; add variance-groups if heteroscedastic two-sample is a target. |
| **N7** | Low | open | No conjunction (min-statistic) / multi-contrast battery / one-/two-/paired-sample design helpers / FWE across contrasts. | Conveniences; partly intended for the `gramform`/`nwx` consumer layer. |

> **Out of scope (by design):** a Wilkinson **formula interface** and design/contrast
> builders are *intentionally* delegated to the downstream `gramform` / `nwx` DSL,
> not a suite defect. The mass-univariate `(V, N)` + shared-`(N, p)`-design API is
> the deliberate, field-correct shape.

## Suggested sequencing

- **Now (verified bugs, isolated, ~10–20 lines each + a regression test):** **B1**
  (`precision` cuSOLVER), **B2** (`spline_design` clamp), **B3** (`sandwich_cov`
  label densify — share a `_densify_labels` helper).
- **Quick win:** **P2** (default `low_rank=True` when `M ≪ N`).
- **High-leverage:** **P1** (`_corrfit`/`_crossed` analytic curvature), **D1** (shared
  `@stats_result` + uniform `re_var`/`cov_re` shape, subsumes **D4**), **N1** (GAM
  smooth-significance).
- **Polish:** **D5** (`resolve_family` export), **P3** (AGQ r≥3 guard), **M1**
  (`edf_total`), **M2** (FLAME oracle), **P6** (dtype-aware pivot).
- **Deferred / document:** **O1** (`glmm/` split), **N2** (surface nulls), **D3**
  (F-contrasts on R2+), **P4** (= #36, separate FR).

## Cross-references

- `docs/feature-requests/glmm-random-slope-robust-solver.md` — the #36 analytic-
  Laplace-gradient / autodiff-through-scan item (overlaps **P3**/**P4**).
- `docs/feature-requests/stats-modelling-suite-v3.md` — the v3 ledger this suite
  grew from.
- Provenance: seven-lens fan-out review, 2026-06-20; high-severity items verified
  on `feat`-merged `main`.
</content>
</invoke>
