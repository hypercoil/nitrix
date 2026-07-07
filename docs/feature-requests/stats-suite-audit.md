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
| **D1** | High | ✅ **done** | new `stats/_result.py` + 14 Result types | Pytree `tree_flatten`/`tree_unflatten` hand-rolled ~8× (add-a-field-touch-3-places footgun); and **inconsistent** — the `lme_fit` dispatch returned `NamedTuple`s (`LMEResult`/`NestedLMEResult`/`CrossedLMEResult`/`CorrLMEResult`/`GLSResult`) that flattened `tier`/`corr`/`df_resid` strings/ints as **dynamic leaves**, opposite to the careful static-aux split the dataclass results take. (design F1 + code-org) | **FIXED:** added `register_result(children=, aux=)` (`_result.py`) — one decorator synthesises both pytree methods and **asserts the named fields exactly partition the dataclass** (a later field not registered fails loudly at import). Applied to the 9 dataclass results; converted the 5 `lme` `NamedTuple` results to `@register_result` dataclasses with static-aux, so `tier`/`corr`/`df_resid`/`weights` are now static aux, not traced leaves (round-trip + leaf-count tests pin it). |
| **D2** | High | ✅ **done** | `lme/reml.py` + `lme/_corrfit.py` (5 result types) | 5-way `Union` return with the RE variance under a different name/shape per tier (`sigma_b_sq` / `cov_re` / `var_outer`+`var_inner` / `var_group`+`var_cross`); no tier-agnostic accessor. | **FIXED:** uniform `.cov_re` `(V, k, k)` + `.re_labels` (k strings) **properties** on all 5 (`REMLResult`/`LMEResult`/`NestedLMEResult`/`CrossedLMEResult`/`CorrLMEResult`). Single-factor tiers carry the genuine within-factor covariance; multi-factor (nested/crossed) are **block-diagonal** (per-factor variances on the diagonal, off-diagonals structurally zero), disambiguated by `re_labels`. Properties (not fields) → no pytree change. Contract test pins shapes/diagonals/labels across all tiers. |
| **D3** | High | ✅ **R2 done; R3/R4/+corr deferred** | `lme/reml.py` + `lme/_blockwoodbury.py` | `lme_t_contrast`/`lme_f_contrast` accepted only `REMLResult`, so contrasts failed on R2/R3/R4/+corr (graceful `TypeError`, not silent). | **FIXED (R2):** new `bw_inference` surfaces the block-Woodbury `fixed_cov`/`theta_cov`/`grad_m` (`grad_m_k = Σ_g C_g (∂G/∂θ_k) C_g^T` for G-params, `σ_e²·X^T V^{-2} X` for the residual); they ride on `LMEResult`, and the contrast fns accept `Union[REMLResult, LMEResult]` with the **same** Satterthwaite df — validated end-to-end vs a dense-numpy oracle to ~1e-6 (lmerTest's compiled chain won't build in this env), incl. the `F==t²` / `df2==df` `L=1` collapse. **Deferred (per scope decision):** R3/R4/+corr (need per-tier derivations) keep a sharpened error; KR stays R1-only. |
| **D4** | Med | ✅ **done** | `stats/glmm.py` `GLMMResult.re_var` (~108) | Silently shape-polymorphic `(V,)` / `(V,r)` / `(V,r,r)` by `z`/`structure` — a `vmap`/indexing footgun; contrast `LMEResult.cov_re` which is *always* `(V,r,r)`. (design F5 + slope review) | **FIXED (green-field, no back-compat):** `re_var` is now uniformly `(V, r, r)` across every tier (few/many/slope/laplace/agq) — scalar intercept `(V, 1, 1)`, diagonal slope a diagonal `(V, r, r)` (zero off-diagonals, asserted), unstructured the full `G` — matching `LMEResult.cov_re`. |
| **D5** | Low | ✅ **done** | `_family.py` / `glm.py` / `stats/__init__.py` | `resolve_family` reachable but not exported, while `resolve_link` is (3×). | **FIXED:** added to `glm.__all__` + imported / exported from `stats.__init__`. |
| **D6** | Low-Med | ✅ **done** | `glm.py`/`glmm/`/`lme/reml.py` | Dispatch axes (`method`/`structure`/`type`/`kind`/`test`/`dof`) were bare `str`; only `gam_fit`/`pca_fit`/`reml.Structure` used `Literal`. `tier` was a free `str`. | **FIXED:** `Literal` aliases for the multi-value dispatch axes — `predict.PredictType`, `sandwich_cov.HCKind`, `compare_models.CompareTest`, `glmm.GLMMStructure`/`GLMMMethod`, `lme.ContrastDof`, and `GLMMResult.tier` (`GLMMTier`, 5 values). The single-value lme result tiers (`'R2'`/`'R3'`/…) are already fixed by their result type. Annotation-only (runtime `raise ValueError` unchanged); 25 glm tests green. |
| **D7** | Low | ✅ **done** | `lme/_recov.py` (new) | Shared RE-covariance (log-Cholesky) helpers reached across module boundaries from `reml.py`, `_corrfit.py`, and `glmm.py` (4× in-function private-name imports). (code-org + design) | **FIXED:** `_tril_layout`/`_param_layout`/`_build_chol`/`cov_re_from_chol` lifted verbatim to `lme/_recov.py` (jnp-only, no deps). `_blockwoodbury` imports `_build_chol` from it; `reml`/`_corrfit`/`glmm` source the helpers from `_recov` — and glmm's 5 in-function imports collapse to one top-level import. |
| **D8** | Low-Med | ✅ **done** | `basis.py` `SmoothBasis` Protocol; `gam.py` | New basis type ⇒ edit the union *and* every `isinstance` branch — the open-set registry story `Family`/`CorrSpec` get right was absent. | **FIXED:** `SmoothBasis` `Protocol` (`design`/`dim`/`penalty_blocks()`/`eval_design()`); the 3 bases implement the methods, and gam.py dispatches through them — both `isinstance` chains (`_smooth_penalties`, `smooth_partial_effect`) are gone. A custom Protocol-conforming basis now fits via `gam_fit` + `smooth_partial_effect` with **no gam.py edit** (test pins the open-set property). |
| **D9** | Taste | ~mostly done | various | `n_iter` vs `n_outer`/`n_inner`/`n_mode`/`n_quad` naming drift; intercept policy differs (`X`-carries-own vs `intercept=` vs forbidden); `VarCompSpec.reml` is pure ceremony (`= cls(**kw)`); `low_rank` is an R1-only silent no-op. | **DONE:** dropped `VarCompSpec.reml` (11 call-sites → direct `VarCompSpec(...)`; `.flame` kept — it sets a real default); documented the intercept policy on the fitters that lacked it (`glm_fit`/`reml_fit`/`lme_fit` "carries its own intercept"; ordinal/gaulss/betareg/gam already noted theirs); `lme_fit.low_rank` now documented **R1-only** (silent no-op on R2/R3/R4/+corr). **Deferred (cosmetic):** the `n_iter` vs `n_outer`/`n_inner`/`n_mode`/`n_quad` rename across public signatures. |

## Code organisation

| ID | Sev | Status | Location | Issue | Recommendation |
|----|-----|--------|----------|-------|----------------|
| **O1** | High | ✅ **done** | `stats/glmm/` package | 6-solver monolith (1573 LOC: few-level / structured slope / many-level Schur / Laplace / Laplace-slope / AGQ) behind one dispatcher; banners already mark the seams. | **FIXED:** split into a `glmm/` package by **method family** — `__init__.py` (the `glmm_fit` dispatcher), `_base.py` (`GLMMResult` + constants), `_pql.py` (all PQL: few / diagonal-slope / unstructured-slope / many), `_laplace.py` (scalar + slope), `_agq.py` (borrows the mode-finder from `_laplace`). Pure relocation (byte-identical bodies, no cycle); new files made format-clean. |
| **O2** | High | ✅ **done** | `stats/_optimise.py` (moved) | The "one shared Newton" was housed *inside* `lme/` yet driven by `_ordinal` (non-mixed-model), and typed on `VarCompSpec` though it reads only primitive fields — forcing `_ordinal` to build a variance-components spec it had no use for. (Only `_ordinal` actually used it; `_betareg`/`_gaulss` have their own IRLS.) | **FIXED:** `damped_newton` moved to `stats/_optimise.py` (beside `_irls`/`_batching`), now taking primitive kwargs (`n_iter`/`damping`/`max_step`/`n_backtrack`) — no `VarCompSpec`. Mixed-model sites pass `**spec.newton_kwargs` (a new `VarCompSpec` property); `_ordinal` drops `VarCompSpec` entirely and passes `n_iter`/`ridge` directly (its `ridge` now also floors the information solve, as its docstring already claimed — default `1e-8` bit-preserved). |
| **O3** | Low | ✅ **done** | `betareg.py`/`gaulss.py`/`ordinal.py` | `_`-prefixed yet export public API (`beta_fit`/…), unlike peer public fitters `glm`/`gam`/`glmm`. | **FIXED (rename):** `git mv` to `betareg.py`/`gaulss.py`/`ordinal.py` (history preserved); the 3 imports in `stats/__init__.py` updated. `nitrix.stats.{betareg,gaulss,ordinal}` are now public module paths, matching `glm`/`gam`/`glmm`. Tests import from the public `nitrix.stats` namespace, so unaffected. |
| **O4** | Low | open | `tests/test_stats.py` | `covariance`/`pca`/`gaussian`/`_irls` fold into `test_stats.py` (no 1:1 test file) while newer fitters are 1:1. | Split `test_stats.py` as those grow. |

## Performance / hardware

| ID | Sev | Status | Location | Issue | Recommendation |
|----|-----|--------|----------|-------|----------------|
| **P1** | High → **none** | **deferred (measured)** | `lme/_corrfit.py` (~241/529), `lme/_crossed.py` (~126/172) | Flagged as "the largest compile graph / OOM risk" — but **the premise did not hold under measurement.** corr-GLS ar1 compiles in **3.4–4.2 s with no OOM** even at the agent's "large groups" sizes (G=120 T=15 n=1800; G=60 T=40 n=2400) — *comparable to* the analytic-curvature R1 sibling (4.2 s) and running faster. The cost is **graph-dominated**; the autodiff Hessian is only a small `theta_dim^2` (~3–5) multiplier, so an analytic curvature (sharing the same graph) would land at the same ~4 s. Same lesson as #36. **No measured impact → deferred** (feasible but not worth the substantial ar1 `∂R/∂rho` AI-REML derivation). The `_crossed` `q2` sequential-Cholesky note stands as a separate (latency-only) item if very large crossed factors ever arise. |
| **P2** | Med | ✅ **done** | `lme/reml.py` R1 (~744) | `low_rank=False` default forced the dense `(N,N)` `ZZ^T` eig where the FaST-LMM `(q,q)` form suffices for `q ≪ N`. | **FIXED:** `low_rank: Optional[bool] = None` auto-selects low-rank when `q < N` (the brain-scale case); explicit `True`/`False` override. Verified `auto == dense` to 1e-6 (incl. gappy/rank-deficient `Z`, absorbed by the null space); 81 LME tests green. |
| **P3** | Med | ✅ **done** | `stats/glmm.py` AGQ | `K = n_quad**r` tensor nodes differentiated through the mode scan — explodes at r≥3. | **FIXED:** `_AGQ_MAX_NODES = 128` cap in the dispatch raises a clear `ValueError` (admits r=2 n_quad≤11, r=3 n_quad≤5; blocks r≥4 / large r=3) pointing to a smaller `n_quad` or Laplace. |
| **P4** | Med | open | `stats/glmm.py` Laplace/Laplace-slope/AGQ | All three marginal GLMM fits take the autodiff-Hessian-through-mode-scan fork. See the **#36** FR (`glmm-random-slope-robust-solver.md`): the clean fix is implicit-diff of the mode (`custom_vjp`/IFT); deferred on ROI (cold-compile, amortised). | Tracked separately (#36). AGQ would benefit identically. |
| **P5** | Low | open | `gam.py` `_smooth_penalties` (~195) | Data-independent penalty eigendecomp (`eigh` of `k×k`) recomputed every `gam_fit` call — wasteful across CV/λ loops. | Cache `penalty_eigs` in the `SplineBasis` container at construction. |
| **P7** | High value | ✅ **done** | `stats/glmm.py` `glmm_fit`; `stats/basis.py` `re_smooth` | **`glmm_fit` was not `jax.jit`-traceable** — derived the level count as a concrete Python int from the data, so under `jit` it raised `ConcretizationTypeError`. Consumers couldn't fuse a pipeline containing it; perf-bench couldn't benchmark the flagship. Surfaced by **perf-bench**. See [`glmm-fit-jit-incompatible-static-group-count.md`](resolved/glmm-fit-jit-incompatible-static-group-count.md). | **FIXED:** optional static `n_groups: Optional[int] = None` (`None` → byte-identical eager `int(jnp.max(group))+1`; supplied → jit-traceable). The minimal `n_groups` change masked a 2nd blocker on the **few-level** path (`re_smooth` built its ridge as `jnp.eye(q)`, a tracer that `gam._smooth_penalties` `np.asarray`'s) — fixed by a host-constant `np.eye(q)`. All families × methods × intercept/slope now jit + match eager (acceptance + negative test). |
| **P6** | Low | ✅ **done** | `linalg/_smalllinalg.py` `_PIVOT_REL_FLOOR=1e-12` (~62); `covariance.py:134` | Pivot floor / `ridge=1e-8` defaults sit below fp32 eps (~1.2e-7) → inert in fp32 on the squared-condition (X^T X) Cholesky; `covariance.py:134` hard-codes `float32`. Suite quietly assumes x64. (engineering + hardware) | **FIXED:** `_pivot_rel_floor(dtype) = 1e2·finfo(dtype).eps` (fp32 ~1.2e-5, fp64 ~2.2e-14) used in `spd_chol`/`small_inv_logdet` — now active in fp32 (regression test pins it), fp64 bit-unchanged vs numpy; `_denom_factor` takes the data `dtype` (no more hard-coded `float32`); module docstring documents the x64 expectation for squared-condition designs. **Follow-on:** `residual.py` carried a verbatim copy of the rolled Cholesky (`_chol_lower`) with the same inert `1e-12` floor — deleted; `_solve_cholesky` now reuses `spd_chol` (dtype-aware floor, drift removed). |

## Mathematical correctness — test-gaps & doc (no bugs; all Low)

| ID | Sev | Status | Location | Issue | Recommendation |
|----|-----|--------|----------|-------|----------------|
| **M1** | Med | ✅ **done** (doc) | `glmm.py` Laplace/AGQ (`edf_total=float(p)`, `dispersion=1`) | Stubs: `edf_total` fixed-effect-only, `dispersion=1` wrong for a Gaussian Laplace fit. (engineering + math) | **DOCUMENTED:** `GLMMResult` now states `dispersion`/`edf_total` are placeholders for the laplace/agq tiers (not for AIC), and that `deviance` *is* the comparable -2·marginal-loglik. (Computing the real marginal edf is left as a future enhancement.) |
| **M2** | Med | ✅ **done** | `tests/test_lme.py` `_flame_hand_iter` | The independent hand-computed FLAME oracle is **defined but never asserted** (dead code); FLAME pinned only by recover-truth + self-consistency. | **FIXED:** wired into `test_flame_matches_hand_computed_reference` — per-voxel `assert_allclose` of `sigma_b_sq` *and* `gamma` against the oracle (verified rtol 1e-6; the two solvers agree to ~1e-10). |
| **M3** | Low | ✅ **done** | `_blockwoodbury.py` `diagonal=True` | Uncorrelated `(x‖g)` diagonal-G slope has **no** dense oracle (only the unstructured case is checked vs `_dense_r2`). | **FIXED:** `test_r2_diagonal_blockwoodbury_recovers_all_seeds` (8 seeds) pins the `structure='diagonal'` block-Woodbury fit against a dense 3-parameter diagonal-G profile-REML oracle (`_dense_r2_diagonal`). |
| **M4** | Low | ✅ **done** | `glmm.py` Laplace/AGQ | Fisher-curvature Laplace is only tested on logit/log (canonical, Fisher = observed Hessian); the non-canonical-link (probit/cloglog slope) path is unpinned. | **FIXED:** `test_laplace_slope_noncanonical_link_marginal_matches_numpy` (probit + cloglog) pins `_laplace_slope_nll` against an independent numpy Fisher-scoring computation at arbitrary `theta` (rtol 1e-6) — discriminating, since for a non-canonical link Fisher ≠ observed Hessian (a formula-level pin, free of optimiser noise). |
| **M5** | Low | ✅ **done** | `covariance.py` `partialcorr` | Off-diagonal not pinned to an external oracle (only `diag==1`). | Closed by **B1**'s `test_partialcorr_offdiagonal_matches_numpy`. |
| **M6** | Low | ✅ **done** | `gam.py` λ-selection | Absolute λ / EDF never compared to mgcv (only inner fit, EDF, FS-identity, and λ-responds). | **FIXED:** mgcv 1.9.4 / R 4.5.3 EDF anchors (`test_gam_edf_matches_mgcv_exact` / `_comparable_to_mgcv`). P-spline + cubic-regression bases coincide with nitrix → per-smooth EDF matches mgcv's REML fit to ~1e-4 (a tight absolute pin on Fellner-Schall λ-selection); thin-plate / cyclic use nitrix's own construction → comparable within ~0.5 df (documented). Absolute λ differs by mgcv's penalty-scaling convention (not asserted — EDF is the invariant). |
| **M7** | Low | ✅ **done** | `linalg/residual.py` JS docstring | "James-Stein dominates OLS in MSE for k≥3" overstates — it's a sound heuristic (plug-in σ²), not the strict theorem. | **FIXED:** both docstrings now frame it as the positive-part JS *heuristic* — a plug-in analogue of the known-variance ``k>=3`` dominance theorem, not the exact result. |

## Neuroimaging community gaps (capability, not defects)

| ID | Sev | Status | Issue | Recommendation |
|----|-----|--------|-------|----------------|
| **N1** | High value | ✅ **done** | No GAM smooth-term p-values. | **SHIPPED:** `smooth_significance(result, smooths)` → `SmoothTest` (`stat`/`edf`/`rank`/`p_value`, `(V, m)`), the Wood-2013 integer-rank test (QR-projected eigen-pseudo-inverse, χ² for fixed-dispersion / F otherwise), mask-truncated for `vmap`. **Validated to EXACT `mgcv:::testStat(type=1)` parity** (both F and χ² branches, stat/rank/p to 6 digits, R 4.5.3 / mgcv 1.9.4). Regression vs an independent numpy+scipy reference. The fractional-rank default (`type=0`, needs the weighted-χ² CDF / Davies) is a documented follow-up. |
| **N2** | High (surface/dMRI) | 🔶 **spin + Moran nulls SHIPPED (2026-07-07); mesh-TFCE + variogram/BrainSMASH open** — see [`spatial-null-models`](spatial-null-models.md) | **TFCE/clustering are lattice-only** — no spin test (Alexander-Bloch/Váša) or mesh/graph adjacency; blocks surface (CIFTI) / fixel (ModelArray) workflows. | A mesh/graph `connected_components` path (the `morphology` dep exists) + a spin-test permutation operator. **Evaluated (2026-06-21):** *not* technically blocked — `sparse.mesh.mesh_k_ring_adjacency` + `morphology._label` + `geometry.sphere.{geodesic,coords,conv}` all ship and are stable; the `geometry-suite` adds surface-**reconstruction** primitives N2 doesn't depend on. Deferred anyway on value-timing: N2's consumers are downstream of the geometry sprint producing the surfaces, it is the largest Wave-4 item (L), and one sprint lets its API co-settle with the surface workflow. |
| **N3** | Med | ✅ **documented** | **No RFT / ACF-FWHM smoothness estimation** — permutation-only (defensible post-Eklund), but no parametric fallback for tiny-N and no AFNI/SPM parity. | **DOCUMENTED:** the `inference` module docstring states the deliberate permutation-only stance (Eklund 2016: parametric RFT cluster inference is anti-conservative) — no ACF/FWHM estimator is provided or planned; a parametric tiny-N fallback is the consumer's. |
| **N4** | Med | ✅ **done** | **FLAME has no outlier-deweighting** (FLAME1/FLAMEO) — a named-feature gap vs the FSL comparator. | **FIXED:** `flame_two_level(robust=True)` runs a Huber-IRLS deweighting outer loop — an outlier subject's within-variance is inflated (`s_i^2/w_i`) from its studentized residual, down-weighting it in the precision-weighted group fit; the converged per-subject `weights` ride on `FLAMEResult`. Deterministic robust M-estimator (documented as *not* the full Woolrich-2008 Bayesian outlier-mixture MCMC). Tests: outlier down-weighted + group estimate recovered vs the outlier-pulled standard fit; clean-data robust==standard; `robust=False` weights==1 (estimate unchanged). |
| **N5** | Med | ✅ **done** | **No effect-size / CI outputs** anywhere (effect+SE+dof are returned, so CIs are one `t_crit` away). | **FIXED:** `confidence_interval(effect, se, dof, level=0.95)` and `standardized_effect(effect, scale)` (`stats/_effect.py`, exported). The per-element Student-t quantile (Newton on `betainc`, normal-seeded — so a per-voxel Satterthwaite `dof` works) is pinned vs `scipy.stats.t.ppf` to <1e-6. |
| **N6** | Low | ✅ **documented** | Cluster-robust SE is **one-way only**; no FSL `-g` variance-group / PALM `-vg`. | **DOCUMENTED:** `sandwich_cov(groups=)` docstring states it is one-way (`-e` exchangeability) clustering, *not* a variance-group model (`-g` / `-vg`) — a separate per-group residual variance is a different estimator, not provided. |
| **N7** | Low | ✅ **done (conjunction); design helpers downstream** | No conjunction (min-statistic) / multi-contrast battery / one-/two-/paired-sample design helpers / FWE across contrasts. | **FIXED:** `conjunction(stats)` (per-voxel minimum statistic, Nichols et al. 2005 — the *valid* conjunction null) + `conjunction_pvalue(p)` (max-p dual) in `inference.multiple_comparisons`. The one-/two-/paired-sample **design-matrix** helpers stay delegated to the `gramform`/`nwx` consumer layer (the formula/design interface is out of scope here, per the register's design-boundary note). |

> **Out of scope (by design):** a Wilkinson **formula interface** and design/contrast
> builders are *intentionally* delegated to the downstream `gramform` / `nwx` DSL,
> not a suite defect. The mass-univariate `(V, N)` + shared-`(N, p)`-design API is
> the deliberate, field-correct shape.

## Suggested sequencing

### Round 1 — ✅ complete (merged to `main`, merge `356c9d2`)

The originally-actionable sequence, all shipped + validated green per-file (376
stats+linalg tests, 0 failures):

- **Bugs:** **B1** (`precision` cuSOLVER), **B2** (`spline_design` clamp), **B3**
  (`sandwich_cov` label densify). **B4/C1** ("gappy labels bias REML") refuted.
- **Quick win:** **P2** (auto `low_rank` when `q < N`).
- **High-leverage:** ~~**P1**~~ (deferred — no measured impact), **D1** (shared
  `register_result` pytree) **+ D4** (uniform `re_var` `(V,r,r)`), **N1** (GAM
  smooth-significance, mgcv-validated).
- **Polish:** **D5** (`resolve_family` export), **P3** (AGQ r≥3 guard), **M1**
  (`edf_total` doc), **M2** (FLAME oracle wired), **P6** (dtype-aware pivot floor).

### Round 2 — remaining (this register), four waves by readiness

Effort **S** ≈ <½ day · **M** ≈ ½–1 day · **L** ≈ multi-day. ⚖️ = needs a design
decision (bring a proposal before coding).

- **Wave 1 — ✅ complete** (quick high-value + harden what shipped):
  **P7** ✅ (static `n_groups` + host-constant RE penalty → `glmm_fit`
  jit-traceable on every path), **M7** ✅ (JS docstring softened + residual
  Cholesky dedup), **M3** ✅ (diagonal-G dense oracle), **M4** ✅ (probit/cloglog
  Laplace-curvature pin), **M6** ✅ (mgcv EDF anchor per basis kind), **D9** ✅
  (dropped `VarCompSpec.reml`; intercept-policy + `low_rank` R1-only docs;
  `n_iter` rename deferred as cosmetic).
- **Wave 2 — ✅ complete** (code organisation, mechanical, behaviour-preserving):
  **O2** ✅ (`damped_newton` → `stats/_optimise.py`, decoupled from `VarCompSpec`),
  **D7** ✅ (RE-cov helpers → `lme/_recov.py`),
  **O1** ✅ (`glmm.py` → `glmm/` package),
  **O3** ✅ (renamed `_betareg`/`_gaulss`/`_ordinal` → public module paths).
- **Wave 3 — ✅ complete** (design contracts):
  **D2** ✅ (uniform lme `.cov_re (V,k,k)` + `.re_labels`; nested/crossed
  block-diagonal), **D3** ✅ R2 (F/t-contrasts on the R2 LMEResult via
  `bw_inference`; R3/R4/+corr deferred per scope decision),
  **D6** ✅ (`Literal` dispatch taxonomies), **D8** ✅ (`SmoothBasis` `Protocol`
  — open-set smooth registry).
- **Wave 4 — neuroimaging capabilities (features; largest):**
  **N5** ✅ (`confidence_interval` + `standardized_effect`),
  **N2** ⏸️ **deferred to post-`geometry-suite`** (not technically blocked — the
  mesh-adjacency / sphere-geodesic primitives ship in `sparse.mesh` /
  `geometry.sphere` — but its consumers are downstream of the geometry sprint
  producing the surfaces, and it is the largest item; let its API settle with
  the surface workflow),
  **N4** ✅ (FLAME outlier-deweighting / robust Huber-IRLS),
  **N3/N6** ✅ (documented the permutation-only / `-e`-not-`-g` stance),
  **N7** ✅ (conjunction min-statistic; design helpers stay downstream).
- **Deferred (tracked, not in waves):** **P4** (implicit-diff mode — separate FR
  #36), **P5** (cache `penalty_eigs`).

## Cross-references

- `docs/feature-requests/glmm-fit-jit-incompatible-static-group-count.md` — the
  perf-bench-surfaced **P7** item (optional static `n_groups` for jit-traceable
  `glmm_fit`).
- `docs/feature-requests/glmm-random-slope-robust-solver.md` — the #36 analytic-
  Laplace-gradient / autodiff-through-scan item (overlaps **P3**/**P4**).
- `docs/feature-requests/stats-modelling-suite-v3.md` — the v3 ledger this suite
  grew from.
- Provenance: seven-lens fan-out review, 2026-06-20; high-severity items verified
  on `feat`-merged `main`.
</content>
</invoke>
