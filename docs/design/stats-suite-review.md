# Nitrix Statistical Suite — Final Consolidated Review

**Scope:** New GP/HGP work (`stats/gp.py`, `stats/hgp.py`, `stats/basis.py`, `stats/priors.py`, `linalg/kernel.py`) plus suite-wide context. Worktree `/scratch/nitrix-gp` (branch `feat/stats-gp`). Read-only.

**Method:** 13 reviewers (10 module-deep + 3 suite-wide) across 7 axes, every concrete claim adversarially verified against the code (numerical checks where applicable), then synthesized. 101 findings kept, 2 refuted. Severity: 13 high / 34 medium / 54 low. By axis: engineering 37, correctness 21, ergonomics 14, performance 11, design 9, value 7, gpu 2.

---

## 1. Executive verdict

The new GP/HGP work is **mathematically sound, well-anchored, and architecturally well-integrated** — it reuses the suite's penalised-REML/Fellner-Schall spine rather than bolting on a foreign paradigm, and its core REML/HSGP/exact-engine numerics are verified against dense references to machine precision. The defects are not in the math; they are at the edges: silent-failure traps (out-of-range group labels dropped, integer-dtype coercion zeroing kernel params, a kriging penalty that does not encode its stated RKHS prior) and memory/scaling gaps where `block=` does not bound the dominant consumer during the rho search (a real brain-scale OOM cliff).

**Bottom line: ship-after-fixes** — clear the critical/high silent-correctness and OOM items below; the rest are tracked follow-ups.

---

## 2. Top action items before merge (critical & high, ranked)

1. **`gp_basis` identity penalty does not encode the Matern RKHS prior** → `stats/basis.py:846-856` → the implied prior covariance `D P⁻¹ Dᵀ` is ~91% off the true Matern Gram, so the documented `λ↔σ_f` variance-component identity is violated (the curve is salvaged only by the single REML λ). **Fix:** set `penalty = diag(1/lam)` (with `radial_transform = U_k`, or keep the `1/lam` whitening and still use `diag(1/lam)`); assert `D P⁻¹ Dᵀ` reproduces `C_xx` to rel-err <1e-2; fix the docstring claim. *(confirmed)*

2. **Out-of-range / negative group labels silently dropped** → `stats/hgp.py:329, 471-491` (and the same centralised bug in **`stats/glmm/__init__.py:227`**) → `jax.nn.one_hot` maps any label `≥ n_levels` or `< 0` to an all-zero row, so observations silently fall out of the group structure and return a wrong-but-finite fit with no error — exactly the path an explicit `n_levels` invites. **Fix:** after resolving `L`/`L2`/`n_groups`, validate host-side `0 ≤ min(group)` and `max(group) < L` (and `group_inner` vs `L2`), raising `ValueError`. *Suite-wide pattern — same silent one-hot/segment-sum drop in GLMM.* *(confirmed)*

3. **Integer `omega` silently corrupts kernel params** → `linalg/kernel.py:364-368, 395-397` → an integer `omega` array coerces `rho`/`amplitude` to int, returning all-zeros (SE) or all-NaN (Matern) on valid public input. **Fix:** promote `omega` to float before deriving the working dtype, and cast `rho`/`amplitude` independently of `omega`'s incoming dtype. *(confirmed)*

4. **rho-search vmaps over all V, `block=` ignored — OOM cliff for `engine='exact'`** → `stats/gp.py:991-1005, 614-639, 1208-1216, 1354-1364` and **`stats/hgp.py:543-560`** → `block=` is threaded only into the *final* fit; every pooled-NLL grid evaluation vmaps the full V, holding `O(V·N²)` inverse/logdet intermediates per grid point (the `(V,N,N)` footprint the docstring promises to avoid), `n_rho` times. The hierarchical design is `(1+L)`/`(1+L1+L2)` wider, making hgp the acuter cliff. **Fix:** route the pooled-NLL vmap through `blocked_vmap(...).sum()` (a sum reduction — a drop-in) so `block=` caps the search. *Suite-wide pattern: `block=` under-bounds memory — see also GAM (#5).* *(confirmed)*

5. **`block=` does not bound the `gam_fit` epilogue** → `gam.py:639-666` (and shared-λ path `gam.py:327-330`) → the full-V `(V,p,p)` influence einsum, `(V,p,p)` `cov_unscaled`, and `(V,N)` fitted/deviance all materialise regardless of `block=` (~3.2 GB at V=1e6, p=20, fp64); the shared-λ path even replicates one identical `(p,p)` cov across all V. **Fix:** fold the EDF/deviance epilogue into the per-element fn (or compute diag-based EDF via `einsum('vij,vji->vi')` without forming `(V,p,p)`); special-case the shared path to a single `(p,p)` cov. *(confirmed)*

6. **`sign_flips` silently couples distinct exchangeability blocks for non-contiguous labels** → `inference/permutation.py:53-60` → negative/sparse block labels wrap the column index (`blocks=[-1,-1,0,0]` aliases two blocks into one), giving fewer effective relabellings and an invalid exchangeability null, passed through unchecked by the driver. **Fix:** canonicalise with `jnp.unique(blocks, return_inverse=True)[1]` (apply in `permutations()` too). *(confirmed)*

7. **Laplace/AGQ marginal hardcodes `dispersion=1.0`** → `glmm/_laplace.py:104-115, 293-300`, `glmm/_agq.py:107-115` → for non-fixed-dispersion families (gaussian/gamma/negbinomial) the free residual scale is pinned to 1, mis-specifying the marginal so the optimiser folds the missing scale into G (verified: G recovered wrong). The PQL path correctly branches on `has_fixed_dispersion`. **Fix:** reject these families on the laplace/agq paths with a clear `ValueError` (point to `lme_fit`/`reml_fit`), or add log-φ to θ; add a gaussian-vs-`reml_fit` regression. *(partial — real, recheck framing)*

---

## 3. Per-axis findings

### Mathematical correctness
The GP/HGP cores are correct and tightly anchored (REML profile, K-block log-pdet decomposition, per-component EDF, corr-whitening Jacobian all match dense references to ~1e-13). The confirmed correctness defects are at the edges and almost all share one signature: **silent wrong-result on out-of-contract input.**
- **High:** `gp_basis` RKHS penalty wrong — `basis.py:846-856` (see #1).
- **Medium:** `corr_raw_bounds=(-2.5,2.5)` cannot represent negative `cs` correlation nor reach the documented range for `cs`/`car1` (sigmoid clamps to ~(0.076,0.924)); estimate silently clamps — `gp.py:762-763, 949-951, 1368-1378`. Make bounds structure-aware or document the one-sided window per structure; add reachable-range tests.
- **Medium:** AR1 with `time=None` silently mis-estimates ρ when rows aren't in within-group time order (ρ=0.37 vs true 0.70 on shuffled rows) — `lme/_corrfit.py:82-127`. Require explicit `time=` or warn on unsorted groups.
- **Medium:** `log_lik` normalisation inconsistent across LME tiers — GLS includes the `(N-p)log(2π)` REML constant, every other tier omits it, so values are not cross-tier comparable — `lme/_corrfit.py:270-277`. Pick one convention everywhere.

### Engineering rigour
Strong single-source-of-truth IRLS/penreml cores and exemplary error messages in the new GP code; the recurring weakness is **missing input validation that turns user error into silent wrong results or cryptic internal errors** (a suite-wide pattern spanning hgp/glmm/glm/lme/inference/pca).
- **High:** group-label range unchecked — `hgp.py:329`, `glmm/__init__.py:227` (see #2).
- **Medium:** nested inner factor must be *globally* numbered; conventional per-outer numbering silently mis-pools with no check — `hgp.py:419-420, 486-491`. Validate or accept `(outer,inner)` pairs and renumber internally.
- **Medium:** GLM `rank` hardcoded to `p` → wrong `dof_resid`/inflated SE/phantom AIC params on rank-deficient designs, masked by the ridge/pivot floor — `glm.py:330`. Detect effective rank or validate full-rank host-side.
- **Medium:** 1-D random covariate `z` shape `(N,)` misroutes (`r = shape[-1] = N`) into the wrong dispatch branch and crashes deep with a confusing `IndexError` — `lme/reml.py:1112`. Coerce `z[:, None]` or raise naming the `(N,r)` contract.
- **Medium:** cluster-robust `sandwich_cov` not jittable (`jnp.unique` data-dependent shape + `int()` on a tracer) despite positioning that invites jitted inference pipelines — `glm.py:519-528`. Document eager-only or accept pre-densified `groups`+`n_groups`.
- **Suite-wide pattern (low, recurring):** integer-dtype leaks under the fp64 invariant — integer `Y`/`x` propagation (`hgp.py:461-462`, `gp.py:869-870`), `mrf_smooth` float32 design/penalty (`basis.py:378-381, 1492-1494`, confirmed), `cluster_size_map` hardcoded float32 (`cluster.py:47`, confirmed). Promote against the canonical float dtype consistently.
- **Low (broad test-coverage gaps, mostly confirmed/partial):** GP HSGP `rank≥N` and tiny-N untested (`gp.py:914-919`); hgp `block`/`map_rho`/`bounds`/explicit-`n_levels` untested (`tests/test_hgp.py`); core kernel primitives have no direct tests (`kernel.py:96-329`, confirmed); GLM HC1-3 / Gamma-Tweedie llf-AIC / weighted IRLS untested (`glm.py`); inference driver `var_smooth`/`blocks`/`mask`/sign-flip Freedman-Lane untested (`tests/test_inference.py:172-506`, confirmed).

### Community needs & value
The headline capability — REML-lengthscale mass-univariate GP/HGP at ~1e6 voxels, cuSOLVER-free and jit/vmap-clean — is **real and appears unique** (brms/mvgam are full-Bayes single-series; mgcv is CPU; FSL/nilearn/statsmodels have no GP). The adoption gaps cluster in three places (all `unverified` judgement calls):
- **High:** cluster/TFCE inference is **lattice-only** — no surface-mesh adjacency, blocking the dominant FreeSurfer/HCP vertex-wise workflow that FSL randomise/PALM support — `inference/tfce.py:67-77`. Add a graph-adjacency/edge-list path (the sparse ELL format is already in-tree).
- **High:** GP lengthscale estimation is **Gaussian-only**; non-Gaussian GP regression (binary activation, lesion counts — the feature request's own headline) is reachable only with ρ *pinned* — `gp.py:771-772`. Wrap PIRLS around the diagonal-S(ρ) penalty for a PQL-REML ρ on Binomial/Poisson.
- **Medium:** top-level package docstring still advertises stats as covariance/spectral only, **hiding the entire GLM/GAM/GP/LME/GLMM/inference surface** — `__init__.py:11-13`. One-paragraph fix, outsized adoption leverage.
- **Medium:** FDR is BH-only (no dependence-aware BY or Storey q-value), notable given strong voxel correlation — `multiple_comparisons.py:53-75`.

### Consumer & user ergonomics
The new GP/HGP code is ergonomically excellent on its own terms (thorough docstrings, sane justified defaults, best-in-class validation messages). The dominant problem is **suite-wide result-object and signature inconsistency the GP/HGP additions sit inside** rather than cause — a user moving between models the suite explicitly equates hits a different name/convention at each step (all `unverified`):
- **High:** coefficient field forks `coef` (GLM/GAM/GP/HGP/Beta/Ordinal) vs `beta_hat` (LME/GLMM) — `gp.py:122`, `glmm/_base.py:52`, `lme/reml.py:162`. Converge or add deprecated alias properties.
- **High:** covariate arg is lowercase `x` (single covariate) in GP/HGP but uppercase `X` (full design) everywhere else, with `gp_fit`'s `x` typed `Any` vs `hgp_fit`'s `Float[Array,'N']` — `gp.py:744`. Tighten the annotation and add an explicit "x is NOT the full design" note.
- **Medium:** two parallel name-clashing IC families (`aic`/`bic` vs `gp_aic`/`gp_bic`), GAM supports neither — `gp.py:1564, 1579` vs `glm.py:599, 604`. Make `aic`/`bic` polymorphic or cross-reference.
- **Medium:** `group` positional in `hgp_fit` but keyword-only in `lme_fit`/`glmm_fit` — `hgp.py:381`; reduced-rank count is `rank` (fit layer) vs `n_basis` (basis layer) — `gp.py:744` vs `basis.py:911`; coefficient covariance named three ways (`cov_unscaled`/`fixed_cov`/`cov_coef`) with a hidden scaled-vs-unscaled distinction — `gp.py:122`, `lme/reml.py:175`, `ordinal.py:66`.

### Suite performance
Hot-path numerics are p-space and vectorise cleanly; the systemic issue is that **`block=` does not bound the dominant memory consumer in several paths** (the rho search and the GAM epilogue) — the brain-scale OOM cliff the knob was advertised to prevent.
- **High:** rho-search unblocked vmap — `gp.py:991-1005`, `hgp.py:543-560` (see #4).
- **High:** `gam_fit` epilogue / shared-λ replication unbounded by `block=` — `gam.py:639-666, 327-330` (see #5).
- **Medium:** hgp rho grid runs in a host Python loop with `n_rho` device→host syncs vs `gp.py`'s on-device `lax.map` — `hgp.py:553-560`. Mirror `gp_fit`'s single-dispatch `lax.map`.
- **Medium:** custom `Family`/`Link` factory instances are unequal & re-hash (Callable fields compared by identity), forcing jit recompiles per fresh `negbinomial(α)`/`tweedie(p)`/`with_link` — `_family.py:289-410`. Give value-based `__eq__`/`__hash__` keyed on `name`+scalar metadata. *(partial)*

### Code organisation / abstraction / design
The new work is at the right altitude and the headline factoring decision (NOT folding ρ into gam.py FS) is correct and verified. The weaknesses are **duplication and under-applied abstraction from the ship-PR-by-PR cadence** — no correctness defects; one consolidation pass removes ~300 lines (all `unverified`):
- **High:** diagonal penalised-REML core written twice (gp single-block, hgp K-block) when single-block is the K=1 case — `hgp.py:165-316` vs `gp.py:452-611`. Promote the K-block core to a shared `stats/_penreml.py`.
- **Medium:** the design/penalty closure abstraction is wired only into the corr path; four other fit bodies re-spell the same scaffolding inline (appears 5×) — `gp.py:683-736, 975-1037, 1095-1168, 1171-1285`. Lift one `_run_gp_fit` driver.
- **Medium:** kernel-name normalisation (`gp.py:367-380` vs `kernel.py:445-455`) and Matern/RBF stationary covariances (`gp.py:383-400`) are duplicated/mis-homed relative to `linalg/kernel.py`; HSGP eigenfunction design evaluated twice (`gp.py:236-245` vs `basis.py:1004-1008`), blocked by an import cycle a neutral `stats/_hsgp.py` would resolve.

### Hardware-awareness / GPU
The cuSOLVER-free and jit/vmap-clean invariants are **genuinely and verifiably respected on every per-voxel device hot path** (small_inv_logdet / hand-Cholesky / Jacobi eig; host eigh kept off-device; corr grid compiled once; HLO guard tests pass). Only two minor items:
- **Medium:** `mrf_smooth` produces float32 design/penalty under the fp64 invariant (`basis.py:378-381, 1492-1494`, confirmed) and `cluster_size_map` hardcodes float32, eroding TFCE fp64 exactness and risking non-integer extents past 2²⁴ (`cluster.py:47`, confirmed). *(folded into the dtype-leak pattern above)*
- **Low:** the cuSOLVER-free HLO assertion covers only `enhancement='voxel'`; TFCE and cluster paths (which add `connected_components` while-loops) are not HLO-scanned — `tests/test_inference.py:389-422`. Parametrise the test over enhancement modes.

---

## 4. Strengths / what's solid

- **REML correctness, verified to machine precision.** `gp._reml_nll` matches a dense `(N,N)` marginal-likelihood reference up to a constant; `log_mlik` matches with the full `(n,M₀)` constant to absolute precision; the K-block `log|S_λ|₊` decomposition and per-component EDF via `diag(V XᵀX)` match dense references exactly. The Fellner-Schall update was independently re-derived and matches `_fs_lambda`.
- **cuSOLVER-free / jit-vmap-clean is real, not aspirational.** Every small solve routes through `small_inv_logdet` (closed-form p≤2, rolled hand-Cholesky + trsm otherwise); host eigh stays off the per-voxel loop; the corr grid is compiled once with the moving design/penalty as traced args (avoiding the O(grid)-recompile OOM trap). HLO custom-call guards across GLM/GAM/LME/GLMM/inference/glasso.
- **Architecturally integrated, not bolted on.** The exact engine is verified equal to `lme.reml_fit` to machine precision; `corr=` rides `lme._corr` whitening verbatim; the multi-D HSGP is the correct Solin-Särkkä tensor-product construction (tracks sklearn 2-D GPR to corr 0.99). EDF accounting is consistent across GP and HGP with no double-counting.
- **Spectral densities and priors are correct in every tested regime.** 1-D closed forms and the general gammaln D-dim form reconstruct sklearn RBF/Matern (D=2 to ~1e-10, D=1 to ~1e-15); the three MAP-ρ priors carry the correct Jacobian and combine as `2·map_rho(ρ)`.
- **The wider suite is rigorously anchored.** LME REML engines exact vs dense `V⁻¹` (~1e-15); analytic AI-REML scores match autodiff to machine precision with verified-PSD information; Kenward-Roger anchored to an independent dense oracle. GLM families match statsmodels to 1e-6–1e-10. Shrinkage estimators bit-exact to scikit-learn; GLASSO verified by KKT. Permutation operators verified unbiased with a structurally-guaranteed `p_fwe ≥ 1/n_perm` floor.
- **Exemplary documentation and error ergonomics.** `docs/design/gaussian-process-implementation.md` is file:line-anchored with named parity anchors and honest deferred-debt disclosure; GP/HGP validation errors uniformly name the arg, repr the value, and state the fix.

---

## 5. Deferred / nice-to-have (low-severity, non-blocking)

- **GP core:** silent boundary-minimum ρ returned with no diagnostic (`gp.py:1413-1436`) — surface a boundary flag; `group` with `corr=None` silently ignored (`gp.py:942-945`); isotropic multi-D ρ grid centred on mean half-range mismatches anisotropic-scale axes (`gp.py:1218-1268`); `gp_predict(exact)` reads ρ from `theta[0,2]` relying on a latent shared-ρ invariant (`gp.py:1500`); exact-engine host eigh redone per grid ρ and per predict (`gp.py:403-419`).
- **Bases:** `hsgp_basis(n_basis=1, center=True)` yields an empty `(n,0)` design (`basis.py:968-969`) — require `n_basis≥2`; `hsgp_basis_nd` skips the resolution warning when `rho=None` (`basis.py:1374-1390`).
- **Kernels/priors:** `linear_distance` wrong for non-symmetric matrix θ (`kernel.py:178-189`); `_theta_kind` misclassifies batched diagonal θ at B==d (`kernel.py:82-88`); `lognormal_prior` docstring conflates mode vs median (`priors.py:100-122`); spectral densities not re-exported from `linalg/__init__.py` (`kernel.py:71-74`); no D>1 reference-value anchor (`kernel.py:408-419`).
- **GLM/GLMM/LME:** no `n>p` validation → saturated model reports p=0 (`glm.py:366-369`); weights/Y shape unchecked (`glm.py:283`); Tweedie/Gamma absolute llf-AIC convention divergence from statsmodels (`_family.py:343-406`, `glm.py:317-319`); diagonal many-level PQL slope has no Schur route + misleading `tier='few'` (`glmm/_pql.py:87-154`); AGQ/Laplace placeholder `edf_total=p`/`dispersion=1` invite silent AIC misuse (`glmm/_laplace.py:148`); scalar Laplace omits `clip_eta` (`glmm/_laplace.py:57-79`); optimiser damping leaks into inferential `theta_cov` (`lme/_varcomp.py:401-402`); robust FLAME re-dispatches per IRLS step with no early stop (`lme/flame.py:226-238`).
- **Inference:** `dof≤0` design yields all-p=1 instead of raising (`randomise.py:237-258`); `cluster_thresh>0` unvalidated (`randomise.py:216-223`); GPD degenerates for tiny exceedance counts (`randomise.py:421-437`); no guard on `n_perm` exceeding distinct relabellings (`permutation.py:37-95`).
- **Multivariate/misc:** matrix-weight covariance silently misbehaves on non-symmetric/indefinite W (`covariance.py:104-114`); PCA silently truncates when `n_components > min(n,d)` (`pca.py:228`, confirmed — a jit/vmap shape-contract hazard worth promoting if PCA is on a hot path); `glasso` divide-by-zero on a zero/negative S diagonal (`connectivity.py:180-183`); `beta_fit` silently clips out-of-(0,1) responses (`betareg.py:114-116`); ordinal probit path not oracle-anchored (`test_ordinal.py:61-69`).
- **Ergonomics/design polish:** predict-return contract asymmetric across suite (`gp.py:1439`); duplicated `n_levels` docstring entry (`hgp.py:433-441`); `n_search` vs `n_inner` naming (`gp.py:768`); single-value `select=` Literal advertises an unimplemented mode (`gp.py:835`); `PriorFn` not re-exported (`priors.py:42`); `PCAResult` is a NamedTuple while all other results are frozen dataclasses (`pca.py:74`); `gp.py` reaches into `lme._corr`/`_corrfit` private modules with no public contract (`gp.py:946, 1323`); `GPResult` carries mode-conditional fields across four engines (`gp.py:106-198`); nested-HGP per-group prediction unimplemented (`hgp.py:684-688`).

---

## Appendix — refuted claims (correctly filtered out)

- *"smooth_significance builds the QR projector from the unweighted Gram for all families"* (GAM) — **refuted**: correctly describes the code but misreads the mgcv algorithm it's measured against; no bug.
- *"Variance functions (var_power/var_ident) have no numerical test"* (LME) — **refuted**: the heteroscedastic GLS path is in fact tested.
