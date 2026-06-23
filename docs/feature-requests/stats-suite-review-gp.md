# `feat/stats-gp` pre-merge review — findings register — `nitrix.stats`

> **Status (2026-06-22): open register.** Pre-merge review of the GP/HGP work on
> branch `feat/stats-gp` (PR1–PR10: `stats/gp.py`, `stats/hgp.py`, `stats/basis.py`,
> `stats/priors.py`, `linalg/kernel.py`) **plus** a suite-wide re-audit. Produced by a
> 13-reviewer fan-out across 7 lenses (mathematical correctness, engineering rigour,
> community needs & value, consumer/user ergonomics, suite performance, code
> organisation/abstraction/design, hardware-awareness / GPU) with **per-finding
> adversarial verification** — every concrete (correctness/engineering/perf/gpu) claim
> was independently re-read against the code and, where numeric, checked by running it.
> **101 findings kept, 2 refuted** (13 high / 34 medium / 54 low). This is a tracking
> ledger: update the **Status** column (`open` → `wip` → `done` / `deferred` /
> `refuted`) as items are worked. Sibling register: the 2026-06-20 standing-suite audit
> [`stats-suite-audit.md`](stats-suite-audit.md); the FR it reviews against is
> [`gaussian-process-models.md`](gaussian-process-models.md).

## Verdict — **ship-after-fixes**

The new GP/HGP work is **mathematically sound and architecturally well-integrated** — it
rides the suite's penalised-REML / Fellner–Schall spine rather than bolting on a foreign
paradigm, and its load-bearing numerics are verified against dense references to ~1e-13
(REML profile, K-block `log|S_λ|₊` decomposition, per-component EDF, corr-whitening
Jacobian, exact-engine == `lme.reml_fit`, multi-D HSGP == sklearn 2-D GPR). The
cuSOLVER-free / jit-vmap-clean invariants are **genuinely and verifiably respected on
every per-voxel device hot path** (HLO custom-call guards pass). **The defects are not in
the math; they are at the edges** — silent-failure traps on out-of-contract input and a
memory cliff where `block=` does not bound the dominant consumer during the ρ search.

**Two cross-cutting patterns** account for most of the high/medium load:

1. **Silent wrong-result on out-of-contract input** — the dominant correctness signature
   (MC2, MC3, ER1, plus AR1-without-`time`, `corr_raw_bounds` clamping). Cheap host-side
   validation closes most of it.
2. **`block=` under-bounds memory** — the knob is advertised to prevent an OOM it does not
   actually cap in three paths (PF1, PF2).

Clear the **Round 1 blockers** (§1) before merge; everything else is tracked follow-up.

---

## §0 — Record corrections (REFUTED — do not re-raise)

| ID | Claim | Why refuted |
|----|-------|-------------|
| **R-GAM** | `smooth_significance` builds the QR projector from the **unweighted** Gram `Xkᵀ Xk` for all families; mgcv uses weighted `XWX` for non-Gaussian | The claim correctly *describes* the nitrix code but **misreads the mgcv algorithm** it is measured against — `mgcv:::testStat(type=1)` parity is already pinned exact (prior audit **N1**). No bug. |
| **R-LME** | Variance functions (`var_power`/`var_ident`) have **no** numerical test — the heteroscedastic GLS path is unvalidated | **False** — the heteroscedastic GLS path *is* tested. |

---

## §1 — Round 1: pre-merge blockers (this PR)

The merge-gating set. All are confirmed (MC4 partial). Suggested order is cheapest-guard
first → correctness → memory; see §8 for rationale.

> **✅ Round 1 COMPLETE (2026-06-22)** — all 7 blockers fixed, tested, ruff/mypy-clean,
> committed on `feat/stats-gp`:
> **ER1·MC2·MC3** `18707fe` · **MC1** `d914cd9` · **MC4** `587fed9` (minimal `ValueError`
> guard; full log-φ marginal deferred) · **PF1** `07b3447` · **PF2** `b97c812` (core
> `(V,p,p)` influence transient removed; `(V,N)`/cov-return residual → Round 3). The Status
> cells below read `open` as authored; treat this banner as the live state.

| # | ID | Sev | Status | Location | Issue (verified) | Fix |
|---|----|-----|--------|----------|------------------|-----|
| 1 | **ER1** | High | open | `hgp.py:329, 471-491`; `glmm/__init__.py:227` | **Out-of-range / negative group labels silently dropped.** `jax.nn.one_hot` (hgp) and `segment_sum`/`b[group]` (glmm) map any label `≥ n_levels` or `< 0` to a zero row/segment, so observations silently fall out of the group structure → wrong-but-finite fit, no error. Exactly the path an explicit `n_levels`/`n_groups` invites. *Suite-wide pattern.* | Host-side, after resolving `L`/`L2`/`n_groups`: `if max(group) ≥ L or min(group) < 0: raise ValueError(...)` (and `group_inner` vs `L2`). One host sync already happens. Add negative/too-small-`n_levels` tests. |
| 2 | **MC2** | High | open | `linalg/kernel.py:364-368, 395-397` | **Integer `omega` silently corrupts kernel params via dtype coercion.** `rho = asarray(rho, dtype=omega.dtype)` truncates `rho`/`amplitude` to int → SE returns all-zeros, Matern returns all-NaN (`λ=1/0`) on valid public input. HSGP-internal path is safe (`sqrt_lambda` always float); the public `se_/matern_spectral_density` are not. | Promote `omega` to float before deriving the working dtype (`dt = promote_types(omega.dtype, float)`); cast `rho`/`amplitude` to `dt` **independently** of `omega`'s incoming dtype. |
| 3 | **MC3** | High | open | `inference/permutation.py:53-60` | **`sign_flips` couples distinct exchangeability blocks for non-contiguous labels.** `n_blocks = int(ids.max())+1`; negative/sparse labels wrap the column index (`blocks=[-1,-1,0,0]` → one column → two blocks aliased), giving fewer effective relabellings and an **invalid exchangeability null**. `permutations()` is safe (labels used only as a sort key). | Canonicalise: `ids = jnp.unique(blocks, return_inverse=True)[1]`. Apply in `permutations()` too for consistency. Add a non-contiguous-label test. |
| 4 | **MC1** | High | open | `basis.py:846-856` | **`gp_basis` identity penalty does not encode the Matern RKHS prior.** The RKHS penalty in `alpha` is `diag(1/lam)`, not the identity; the implied prior cov `D P⁻¹ Dᵀ` is **~91% off** the true Matern Gram `C_xx` (verified), violating the documented `λ↔σ_f` variance-component identity (the curve survives only via the single REML λ). | Set `penalty = diag(1/lam)` (keep `radial_transform = U_k`, or keep the `1/lam` whitening and still use `diag(1/lam)`). Assert `D P⁻¹ Dᵀ` reproduces `C_xx` to rel-err < 1e-2 in the `gp_basis` tests; fix the docstring at 846-847. |
| 5 | **MC4** | High | open (partial) | `glmm/_laplace.py:104-115, 293-300`; `glmm/_agq.py:107-115` | **Laplace/AGQ marginal hardcodes `dispersion=1.0`.** For non-fixed-dispersion families (gaussian/gamma/negbinomial) the free residual scale is pinned to 1, mis-specifying the marginal so the optimiser **folds the missing scale into G** (verified: Gaussian AGQ recovers G wrong, reports `dispersion=1.0`). PQL correctly branches on `has_fixed_dispersion`; only laplace/agq are affected. Deepens prior audit **M1** (which only *documented* the result-field placeholder — this is the marginal-objective misspecification). | Minimal: reject non-fixed-dispersion families on the laplace/agq paths with a `ValueError` pointing to `lme_fit`/`reml_fit`. Full: add `log φ` to θ. Add a gaussian-vs-`reml_fit` regression on both marginal methods. |
| 6 | **PF1** | High | open | `gp.py:991-1005, 614-639, 1208-1216, 1354-1364`; `hgp.py:543-560` | **ρ-search pooled-NLL vmaps over all V; `block=` ignored — OOM cliff for `engine='exact'`.** `block=` is threaded only into the *final* fit; every grid evaluation (`n_rho`×) holds `O(V·N²)` inverse/logdet intermediates — the `(V,N,N)` footprint the docstring promises to avoid. HGP's `(1+L1+L2)`-wider design is the acuter cliff. | Route the pooled-NLL vmap through `blocked_vmap(...).sum()` (a sum reduction — a drop-in) in `_pooled_nll`, `_pooled_nll_from_design`, the nd `_pooled`, and the corr `_nll_jit`, honouring the same `block=`. Peak → `O(block·p²)`. |
| 7 | **PF2** | High | open | `gam.py:639-666, 327-330` | **`block=` does not bound the `gam_fit` epilogue.** The full-V `(V,p,p)` influence einsum, `(V,p,p)` `cov_unscaled`, and `(V,N)` fitted/deviance materialise regardless of `block=` (~3.2 GB at V=1e6, p=20, fp64); the shared-λ path even replicates one identical `(p,p)` cov across all V. | Fold the EDF/deviance epilogue into the per-element fn (so `blocked_vmap` chunks it), or compute diag-based EDF via `einsum('vij,vji->vi')` without forming `(V,p,p)`; special-case the shared-λ path to a single `(p,p)` cov. |

---

## §2 — Mathematical correctness (21)

Cores correct & tightly anchored; defects at the edges, almost all *silent wrong-result on
out-of-contract input*. Blockers MC1–MC4 in §1.

| ID | Sev | Status | Location | Issue | Fix |
|----|-----|--------|----------|-------|-----|
| **MC5** | Med | open | `gp.py:762-763, 949-951, 1368-1378` | `corr_raw_bounds=(-2.5,2.5)` cannot represent **negative `cs`** correlation nor reach the documented range for `cs`/`car1` (sigmoid clamps to ~(0.076,0.924)); the estimate silently clamps. | Make bounds structure-aware (one-sided window per structure) or document the reachable window; add reachable-range tests. |
| **MC6** | Med | open | `lme/_corrfit.py:82-127` | AR1 with `time=None` silently mis-estimates ρ when rows are **not** in within-group time order (ρ=0.37 vs true 0.70 on shuffled rows). | Require explicit `time=`, or warn when groups are not monotone in time. |
| **MC7** | Med | open | `lme/_corrfit.py:270-277` | `log_lik` normalisation **inconsistent across LME tiers** — GLS includes the `(N-p)log(2π)` REML constant; every other tier omits it → not cross-tier comparable. | Pick one convention everywhere (include or omit the full constant). |
| MC8–MC21 | Low | open | — | See §7 (low-severity backlog): GP boundary-ρ diagnostic, isotropic-grid axis mismatch, `gp_predict(exact)` latent-ρ read, `n_basis=1` empty design, `linear_distance`/`_theta_kind` edge cases, `lognormal_prior` mode/median doc, D>1 spectral-density reference anchor, ordinal probit oracle, etc. | — |

---

## §3 — Engineering rigour (37)

Strong single-source-of-truth IRLS/penreml cores and best-in-class GP/HGP error messages;
recurring weakness is **missing input validation** turning user error into silent wrong
results or cryptic internal errors (suite-wide). Blocker ER1 in §1.

| ID | Sev | Status | Location | Issue | Fix |
|----|-----|--------|----------|-------|-----|
| **ER2** | Med | open | `hgp.py:419-420, 486-491` | Nested inner factor must be **globally** numbered; conventional per-outer numbering silently mis-pools, no check. | Validate, or accept `(outer,inner)` pairs and renumber internally. |
| **ER3** | Med | open | `glm.py:330` | GLM `rank` hardcoded to `p` → wrong `dof_resid` / inflated SE / phantom AIC params on rank-deficient designs (masked by ridge/pivot floor). | Detect effective rank, or validate full-rank host-side. |
| **ER4** | Med | open | `lme/reml.py:1112` | 1-D random covariate `z` shape `(N,)` misroutes (`r = shape[-1] = N`) → confusing deep `IndexError`. | Coerce `z[:, None]`, or raise naming the `(N,r)` contract. |
| **ER5** | Med | open | `glm.py:519-528` | Cluster-robust `sandwich_cov` not jittable (`jnp.unique` data-dependent shape + `int()` on a tracer) despite inviting jitted pipelines. | Document eager-only, or accept pre-densified `groups`+`n_groups`. |
| **ER6** | Low | open | `hgp.py:461-462`; `gp.py:869-870`; `basis.py:378-381, 1492-1494`; `cluster.py:47` | **Suite-wide dtype-leak pattern** under the fp64 invariant — integer `Y`/`x` propagation; `mrf_smooth` float32 design/penalty (confirmed); `cluster_size_map` hardcoded float32 (confirmed). Erodes TFCE fp64 exactness; risks non-integer extents past 2²⁴. Related to prior audit **P6** (pivot-floor x64). | Promote against the canonical float dtype consistently. |
| **ER7** | Low | open | tests (`test_hgp.py`, `test_gp.py:914-919`, `kernel.py:96-329`, `glm.py`, `test_inference.py:172-506`) | **Broad test-coverage gaps** (mostly confirmed/partial): GP HSGP `rank≥N`/tiny-N; hgp `block`/`map_rho`/`bounds`/explicit-`n_levels`; core kernel primitives untested; GLM HC1-3 / Gamma-Tweedie llf-AIC / weighted IRLS; inference driver `var_smooth`/`blocks`/`mask`/sign-flip Freedman–Lane. | Add targeted tests as the items above are fixed. |
| ER8–ER37 | Low | open | — | See §7: `n>p` validation, weights/Y shape checks, `dof≤0`/`cluster_thresh` guards, GPD tiny-count degeneracy, `n_perm` vs distinct-relabelling guard, `beta_fit` silent clip, `glasso` zero-diagonal divide, matrix-weight covariance on indefinite W, PCA silent truncation, etc. | — |

---

## §4 — Performance (11)

Hot-path numerics are p-space and vectorise cleanly; the systemic issue is `block=`
under-bounding the dominant memory consumer (PF1, PF2 in §1).

| ID | Sev | Status | Location | Issue | Fix |
|----|-----|--------|----------|-------|-----|
| **PF3** | Med | open | `hgp.py:553-560` | HGP ρ grid runs in a **host Python loop** with `n_rho` device→host syncs vs `gp.py`'s on-device `lax.map`. | Mirror `gp_fit`'s single-dispatch `lax.map`. |
| **PF4** | Med | open (partial) | `_family.py:289-410` | Custom `Family`/`Link` factory instances are **unequal & re-hash** (Callable fields compared by identity) → jit recompiles per fresh `negbinomial(α)`/`tweedie(p)`/`with_link`. | Value-based `__eq__`/`__hash__` keyed on `name` + scalar metadata. |
| PF5–PF11 | Low | open | `gp.py:403-419` (exact host-eigh redone per grid-ρ & per predict); `gam.py` penalty-eig recompute (≈ prior **P5**); FLAME per-IRLS re-dispatch (`lme/flame.py:226-238`); etc. | See §7. | — |

---

## §5 — Hardware-awareness / GPU (2)

cuSOLVER-free & jit/vmap-clean **verifiably respected on every per-voxel device hot path**
(small_inv_logdet / hand-Cholesky / Jacobi eig; host eigh off-device; corr grid compiled
once; HLO guards pass). Only:

| ID | Sev | Status | Location | Issue | Fix |
|----|-----|--------|----------|-------|-----|
| **HW1** | Med | open | `basis.py:378-381, 1492-1494`; `cluster.py:47` | float32 design/penalty under the fp64 invariant (folded into **ER6**). | Promote to canonical float dtype. |
| **HW2** | Low | open | `tests/test_inference.py:389-422` | The cuSOLVER-free HLO assertion covers only `enhancement='voxel'`; TFCE/cluster paths (with `connected_components` while-loops) are not HLO-scanned. | Parametrise the HLO test over enhancement modes. |

---

## §6 — Community value (7), Ergonomics (14), Design (9)

Subjective lenses (judgement calls; not adversarially refuted). **None merge-gating** — but
the high items are the highest-leverage follow-ups.

### Community needs & value
| ID | Sev | Status | Location | Gap | Direction |
|----|-----|--------|----------|-----|-----------|
| **CV1** | High | **deferred → tracked under prior `N2`** | `inference/tfce.py:67-77`; `cluster.py:72` | Cluster/TFCE inference is **lattice-only** — no surface-mesh adjacency, blocking the FreeSurfer/HCP vertex-wise workflow FSL randomise/PALM support. | **Duplicate of [`stats-suite-audit.md`](stats-suite-audit.md) N2** (deferred post-`geometry-suite`; mesh-adjacency/sphere-geodesic prims already ship). Not re-litigated here — see N2 for the standing plan. |
| **CV2** | High | ✅ Phase 1 done `e7ac76e` | `gp.py:771-772` | GP lengthscale estimation is **Gaussian-only**; non-Gaussian GP regression (binary activation, lesion counts — the FR's own headline) is reachable only with ρ *pinned*. | **DONE (Phase 1):** `gp_fit(family='binomial'\|'poisson')` estimates ρ by **PQL-REML** — relinearise to the per-element weighted Gaussian working problem, profile ρ with the same `_pooled_nll_one`/`_gp_fit_one` core (per-voxel `X^TWX` inside `blocked_vmap`). Gaussian path **byte-identical** (no marquee regression); `gp_predict(type='response')`; recovery + ρ-tracking + guarded mgcv tests. Design: [`gp-non-gaussian-lengthscale.md`](../design/gp-non-gaussian-lengthscale.md). **Phase 2 deferred:** LAML (mgcv-grade), multi-D/ARD, gamma/negbin, corr=. |
| **CV3** | Med | ✅ done `63d7707` | `__init__.py:11-13` | Top-level package docstring still advertises stats as covariance/spectral only, **hiding** the GLM/GAM/GP/LME/GLMM/inference surface. | One-paragraph rewrite — outsized adoption leverage, ~5 min. |
| **CV4** | Med | ✅ done `5be6354` | `multiple_comparisons.py:53-75` | FDR is **BH-only** — no dependence-aware BY or Storey q-value, notable given strong voxel correlation. | **DONE:** `fdr_by` (Benjamini-Yekutieli, arbitrary-dependence; == statsmodels), `storey_pi0` + `fdr_storey` (π₀-adaptive, higher power), unified `fdr(method=)` dispatcher. `fdr_bh` refactored onto a shared `_bh_stepup` (byte-identical). |
| CV5–CV7 | Low | open | — | posterior-uncertainty / plotting-IO ergonomics; see §7. | — |

### Consumer & user ergonomics
The new GP/HGP code is excellent on its own terms; the problem is **suite-wide result/signature
inconsistency the additions sit inside** (a user moving between models the suite *equates*
hits a different convention at each step).
| ID | Sev | Status | Location | Inconsistency | Direction |
|----|-----|--------|----------|---------------|-----------|
| **UX1** | High | ✅ done `63d7707` | `gp.py:122`; `glmm/_base.py:52`; `lme/reml.py:162` | Coefficient field forks **`coef`** (GLM/GAM/GP/HGP/Beta/Ordinal) vs **`beta_hat`** (LME/GLMM); `coef_mu`/`coef_scale` (GauLSS). Generic `result.coef @ contrast` silently breaks across families. | Converge on `coef` (the majority/neutral choice); add a deprecated `beta_hat`→`coef` alias property on LME/GLMM. New GP/HGP code already chose `coef`. |
| **UX2** | High | ✅ done `63d7707` | `gp.py:744`; `hgp.py:383` | Covariate arg is lowercase **`x`** (single covariate) in GP/HGP but uppercase **`X`** (full design) everywhere else; `gp_fit`'s `x` typed `Any` vs `hgp_fit`'s `Float[Array,'N']`. | Keep `x` (it genuinely is one covariate) but tighten the annotation to `Union[Float[Array,'N'], Float[Array,'N D']]` and add a one-line "`x` is NOT the full design — linear covariates go to `parametric=`" note at each docstring head. |
| **UX3** | Med | open | `gp.py:1564,1579` vs `glm.py:599,604` | Two parallel name-clashing IC families (`aic`/`bic` vs `gp_aic`/`gp_bic`); GAM supports neither. | Make `aic`/`bic` polymorphic over result types, or cross-reference. |
| **UX4** | Med | open | `hgp.py:381`; `gp.py:744` vs `basis.py:911`; `gp.py:122`, `lme/reml.py:175`, `ordinal.py:66` | `group` positional in `hgp_fit` but keyword-only in `lme_fit`/`glmm_fit`; reduced-rank count `rank` (fit) vs `n_basis` (basis); coef-cov named 3 ways (`cov_unscaled`/`fixed_cov`/`cov_coef`) with a hidden scaled-vs-unscaled distinction. | Converge names; document the scaled/unscaled distinction at each site. |
| UX5–UX14 | Low | open | — | predict-return asymmetry, duplicated `n_levels` docstring, `n_search`/`n_inner`, unimplemented `select=` Literal, `PriorFn`/spectral-density non-export, `PCAResult` NamedTuple-vs-dataclass; see §7. | — |

### Code organisation / abstraction / design
Right altitude; the headline factoring decision (NOT folding ρ into gam.py FS) is correct &
verified. Weaknesses are **duplication from the ship-PR-by-PR cadence** — no correctness
defects; one consolidation pass removes ~300 lines.
| ID | Sev | Status | Location | Issue | Direction |
|----|-----|--------|----------|-------|-----------|
| **DS1** | High | ✅ done `cadbea0` | `hgp.py:165-316` vs `gp.py:452-611` | Diagonal penalised-REML core written **twice** (gp single-block, hgp K-block) when single-block is the K=1 case (`_quantities`/`_fs_lambda`/`_reml_nll` ≅ `_mb_*`). Flagged deliberate debt in design doc PR4a/5c. | **DONE:** new `stats/_penreml.py` is the single source of truth (`mb_quantities`/`mb_fs`/`mb_reml_nll`/`reml_const`). hgp re-exports it under the `_mb_*` names; gp's single-block fns are thin K=1 delegations. Shared core uses the single-block float expressions so all gp paths are **bit-identical** to pre-refactor (hgp shifts ~1.6e-13); every test-referenced internal name preserved. |
| **DS2** | Med | open | `gp.py:683-736, 975-1037, 1095-1168, 1171-1285` | The design/penalty closure abstraction is wired only into the corr path; four other fit bodies re-spell the same scaffolding inline (5×). | Lift one `_run_gp_fit` driver. |
| **DS3** | Med | open | `gp.py:367-380` vs `kernel.py:445-455`; `gp.py:383-400`; `gp.py:236-245` vs `basis.py:1004-1008` | Kernel-name normalisation and Matern/RBF stationary covariances duplicated/mis-homed vs `linalg/kernel.py`; HSGP eigenfunction design evaluated twice (blocked by an import cycle). | A neutral `stats/_hsgp.py` resolves the cycle and de-dups the design eval; re-home covariances to `linalg/kernel.py`. |
| DS4–DS9 | Low | open | `gp.py:946,1323` (reaches into `lme._corr`/`_corrfit` privates); `gp.py:106-198` (mode-conditional `GPResult` fields across 4 engines); `hgp.py:684-688` (nested per-group predict unimplemented); etc. | See §7. | — |

---

## §7 — Low-severity backlog (54)

Tracked, non-blocking. Grouped by area; each is file:line-pinned in the full review
artifact `docs/design/stats-suite-review.md` §5.

- **GP core:** silent boundary-minimum ρ with no diagnostic (`gp.py:1413-1436`); `group` with `corr=None` silently ignored (`gp.py:942-945`); isotropic multi-D ρ grid centred on mean half-range mismatches anisotropic axes (`gp.py:1218-1268`); `gp_predict(exact)` reads ρ from `theta[0,2]` on a latent shared-ρ invariant (`gp.py:1500`); exact host-eigh redone per grid-ρ & per predict (`gp.py:403-419`).
- **Bases:** `hsgp_basis(n_basis=1, center=True)` → empty `(n,0)` design (`basis.py:968-969`); `hsgp_basis_nd` skips the resolution warning when `rho=None` (`basis.py:1374-1390`).
- **Kernels/priors:** `linear_distance` wrong for non-symmetric matrix θ (`kernel.py:178-189`); `_theta_kind` misclassifies batched diagonal θ at B==d (`kernel.py:82-88`); `lognormal_prior` docstring conflates mode vs median (`priors.py:100-122`); spectral densities not re-exported from `linalg/__init__.py`; no D>1 reference-value anchor.
- **GLM/GLMM/LME:** no `n>p` validation → saturated model reports p=0 (`glm.py:366-369`); weights/Y shape unchecked (`glm.py:283`); Tweedie/Gamma absolute llf-AIC convention vs statsmodels; diagonal many-level PQL slope has no Schur route + misleading `tier='few'` (`glmm/_pql.py:87-154`); AGQ/Laplace placeholder `edf_total=p`/`dispersion=1` invite AIC misuse (≈ prior **M1**); scalar Laplace omits `clip_eta`; optimiser damping leaks into inferential `theta_cov` (`lme/_varcomp.py:401-402`); robust FLAME re-dispatches per IRLS step with no early stop.
- **Inference:** `dof≤0` design → all-p=1 instead of raising (`randomise.py:237-258`); `cluster_thresh>0` unvalidated; GPD degenerates for tiny exceedance counts; no guard on `n_perm` exceeding distinct relabellings.
- **Multivariate/misc:** matrix-weight covariance misbehaves on non-symmetric/indefinite W (`covariance.py:104-114`); PCA silently truncates when `n_components > min(n,d)` (`pca.py:228`, confirmed — vmap shape-contract hazard); `glasso` divide-by-zero on zero/negative S diagonal; `beta_fit` silently clips out-of-(0,1) responses; ordinal probit path not oracle-anchored.
- **Ergonomics/design polish:** predict-return contract asymmetric across suite; duplicated `n_levels` docstring entry; `n_search`/`n_inner` naming; single-value `select=` Literal advertises an unimplemented mode; `PriorFn` not re-exported; `PCAResult` NamedTuple while peers are frozen dataclasses; `GPResult` mode-conditional fields across four engines; nested-HGP per-group prediction unimplemented (`hgp.py:684-688`).

---

## §8 — Suggested sequencing

Effort **S** ≈ <½ day · **M** ≈ ½–1 day · **L** ≈ multi-day.

### Round 1 — pre-merge blockers (this PR) — ✅ **COMPLETE**
Order = cheapest validation guard → correctness → memory (later items don't depend on
earlier, but this front-loads risk reduction). All ✅ done + tested + committed:

1. **ER1** (S) — group-label validation, `hgp` + `glmm` (host guard).
2. **MC2** (S) — integer-`omega` dtype promotion in `kernel.py`.
3. **MC3** (S) — `sign_flips` label canonicalisation.
4. **MC1** (S–M) — `gp_basis` RKHS penalty `diag(1/lam)` + `D P⁻¹ Dᵀ ≈ C_xx` assertion.
5. **MC4** (S) — reject non-fixed-dispersion families on laplace/agq with a clear `ValueError` (the partial-scoped minimal fix) + gaussian-vs-`reml_fit` regression.
6. **PF1** (M) — route ρ-search pooled-NLL through `blocked_vmap(...).sum()` (gp + hgp + nd + corr).
7. **PF2** (M) — bound the `gam_fit` epilogue by `block=` (diag-EDF einsum / fold into per-element fn).

Each lands with its regression test; full GP/HGP/suite sweep green before merge.

### Round 2 — high-value capability + consistency (follow-up PRs, not merge-gating)
> **Consistency batch ✅ done (`63d7707`):** CV3 + UX1 + UX2.  **DS1 ✅ done (`cadbea0`).**
> **CV4 ✅ done (`5be6354`). CV2 Phase 1 ✅ done (`e7ac76e`)** — non-Gaussian GP
> lengthscale (PQL-REML) for Binomial/Poisson; LAML/multi-D/gamma deferred to
> Phase 2. **Round 2 complete** (CV1 deferred = prior audit N2).
- **CV3** ✅ `63d7707` (S) — package docstring (quick adoption win).
- **UX1/UX2** ✅ `63d7707` (S–M) — `coef` alias on all 6 LME/GLMM result types + `x`-vs-`X` annotation/doc.
- **DS1** ✅ `cadbea0` (M) — shared `stats/_penreml.py` K-block core (gp calls with K=1; bit-identical).
- **CV2** ✅ Phase 1 `e7ac76e` (L) — non-Gaussian GP lengthscale (PQL-REML ρ, Binomial/Poisson); Phase 2 (LAML / multi-D / gamma) deferred.
- **CV4** ✅ `5be6354` (S) — BY / Storey FDR + unified `fdr(method=)` dispatcher.
- **CV1** — **deferred, tracked under prior `N2`** (post-`geometry-suite`).

### Round 3 — medium correctness/robustness hardening (mostly done)
> **Done (9):** **ER3·ER5** `d260bb4` (GLM effective-rank dof + jit-traceable cluster
> sandwich) · **MC6·ER2·ER4·ER6** `ee84fda` (AR1 time-order warning; nested
> global-numbering guard; 1-D `z` coercion; integer-Y / cluster-size / mrf dtype leaks) ·
> **PF3** `c52312b` (hgp ρ-search on-device `lax.map`) · **MC5** `6068576` (corr= wider
> default bounds + edge-clamp warning) · **MC7** `c2ac15b` (full `(N-p)log2π` REML constant
> across all LME tiers — matches statsmodels `llf`).
> **Remaining:** **DS2/DS3** (fit-driver + `_hsgp.py` de-dup — refactors) · **PF4**
> (`Family`/`Link` hashing — deferred; needs param-encoded names).

**MC5** ✅ (corr bounds), **MC6** ✅ (AR1 time order), **MC7** ✅ (log_lik normalisation),
**ER2** ✅ (nested global numbering), **ER3** ✅ (GLM rank), **ER4** ✅ (lme `z` shape),
**ER5** ✅ (sandwich jit), **ER6/HW1** ✅ (dtype-leak pattern), **PF3** ✅ (hgp `lax.map`),
**PF4** (`Family`/`Link` hashing), **DS2/DS3** (fit-driver + `_hsgp.py` de-dup).

### Round 4 — low/polish + test-coverage backlog
**ER7** (test gaps) + the §7 low-severity items, as bandwidth allows.

---

## §9 — Cross-references

- [`stats-suite-audit.md`](stats-suite-audit.md) — the 2026-06-20 standing-suite audit
  register. **CV1 = its N2** (surface-mesh TFCE, deferred post-geometry-suite); **MC4/§7
  glmm placeholder deepens its M1**; **ER6 relates to its P6** (x64 pivot floor).
- [`gaussian-process-models.md`](gaussian-process-models.md) — the GP/HGP FR this branch
  implements (HSGP-primary, (a)-scope).
- `docs/design/gaussian-process-implementation.md` — as-built decisions (§5–§5j).
- `docs/design/stats-suite-review.md` — the full consolidated review artifact (all 101
  findings, strengths, per-axis detail, file:line for every low item).
- Provenance: 13-reviewer / 53-agent adversarially-verified fan-out, 2026-06-22, on
  `feat/stats-gp`.
