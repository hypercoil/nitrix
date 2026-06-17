# Statistical modelling suite v2 — completeness + regularised connectivity (ledger)

> **Status (2026-06-17): planned — agreed scope, not started.** Follow-up to the
> shipped v1 suite ([`stats-modelling-suite.md`](stats-modelling-suite.md):
> LME size-dispatch, GLM/GAM/GAMM, TFCE `randomise` — merged to `main`). v2
> collects (a) the items v1 deliberately deferred and (b) the long-deferred
> **regularised connectivity estimators** Ledoit-Wolf and graphical LASSO
> ([`ledoit-wolf-shrinkage.md`](ledoit-wolf-shrinkage.md),
> [`graphical-lasso.md`](graphical-lasso.md), SPEC §12.14). It reuses the v1
> substrate wholesale — the cuSOLVER-free tiny-SPD solve (`stats._smalllinalg`,
> incl. the rolled Cholesky), the chunked-`vmap` spine (`_blocked_vmap`), the
> `_varcomp` REML engine, the GLM/IRLS core, and the spline-basis machinery —
> so each item is an *extension*, not a new stack.

## §0. Framing

v1 shipped the spine; v2 fills it out along two axes:

1. **Completeness of the v1 surfaces** — the bases / statistics / enhancement
   modes that round out mgcv / FSL-`randomise` / FaST-LMM parity but were not
   on the critical path: the LME q-rank decomposition, the GAM thin-plate /
   cyclic / tensor-product bases and a shared-λ fast mode, and the `randomise`
   cluster-extent/mass enhancement, F-contrast, and tail-accelerated p-values.
2. **Regularised connectivity estimators** — Ledoit-Wolf / OAS analytic
   shrinkage and graphical LASSO, the small-sample covariance / sparse-precision
   estimators the fMRI connectome literature (and nilearn's *defaults*) are
   built on. v1's `stats.covariance` ships only the *raw* empirical estimators;
   these are the regularised counterparts.

Both axes are **score kernels** (arrays → covariance / statistic / p-value
arrays), per SPEC_UPDATE_v0.5 §1 — same classification as the v1 suite; no
new scalarisation, container, or CLI surface. The cuSOLVER-free discipline and
the differentiability expectations carry over unchanged.

---

## §1. Theme 1 — LME completeness

### §1.1 `reml_fit` q-rank `Z`-decomposition  *(deferred from v1 Workstream A)*

**What.** Replace the shared `safe_eigh(ZZ^T)` (`O(N^3)`, a full `N×N` eigh)
with the FaST-LMM **low-rank** formulation: a thin SVD of `Z` (`N×q`) or a
`q×q` eig of `Z^T Z` gives `ZZ^T = U_r diag(S^2) U_r^T` in `O(N q^2)`, with the
`N − r` null-space directions all sharing the residual variance. The rotated
fit splits into the `r`-dim range subspace (per-coordinate `d_i = sigma_b^2
S_i^2 + sigma_e^2`) and the null space, which enters only through its
sum-of-squares `||y||^2 − ||U_r^T y||^2` with multiplicity `N − r`.

**Why.** `O(N^3) -> O(N q^2)` is real at large `N` (cohort group analyses with
hundreds of subjects, small random-effect rank `q`). Secondary for the
benchmark shapes (small `N`) — the reason it was deferred — but the right
asymptotics for brain-scale cohorts.

**cuSOLVER note.** The decomposition is still one-off and routes through
`safe_eigh`/`safe_svd` (CPU-latch); `gesvd` is *untested* on the broken L4
stack and may fail like `syevd`, so keep the CPU-latch fallback and verify on
the target GPU (the win is compute, not GPU-availability — the per-voxel path
is already cuSOLVER-free). **Effort: M.**  **Oracle:** matches the current
full-eigh fit to the iterative tolerance; HLO carries no `N×N` intermediate
beyond `U_r` (`N×r`).

---

## §2. Theme 2 — GAM bases & smoothing completeness

v1 ships the P-spline (`bs='ps'`); v2 adds the bases needed for full mgcv
parity and for multi-covariate / periodic / heterogeneous-smoothness settings.
All return the same `(design, penalty)` `SplineBasis` contract and reuse the
Householder sum-to-zero constraint (cuSOLVER-free).

### §2.1 Thin-plate regression spline basis (`bs='tp'`, the mgcv default)

**What.** `thinplate_regression_basis(x, k, ...)` — the TPS basis (Wood 2003)
truncated by an eigendecomposition of the TPS penalty to a rank-`k` basis. The
mgcv **default** smoother; isotropic (rotation-invariant), so it is the natural
choice for multi-dimensional smooths `s(x, y)` (spatial / surface coordinates).

**Substrate.** The eigen-truncation is a one-off `eigh` per smooth at
construction (route via `safe_eigh` / host); the resulting `(design, penalty)`
slot straight into `gam_fit`. **Effort: M.** **Oracle:** `mgcv::gam(...,
bs='tp')` coefficients / EDF (rpy2 or pinned fixtures), and the
penalty-nullspace dimension.

### §2.2 Cyclic cubic spline basis (`bs='cc'`)

**What.** `cyclic_cubic_basis(x, k, *, bounds)` — a periodic cubic regression
spline with wrap-around continuity (`f`, `f'`, `f''` matched at the period
ends) and a cyclic penalty. For periodic covariates: cortical angle, phase,
time-of-day, gradient direction. **Effort: S-M.** **Oracle:** `mgcv` `bs='cc'`;
continuity of the fitted smooth and its first two derivatives at the seam.

### §2.3 Tensor-product smooths (`te()`)

**What.** `tensor_smooth([basis_x, basis_y, ...])` — an interaction smooth
`f(x_1, ..., x_d)` from marginal bases via the row-wise tensor product of
designs and a Kronecker-sum of per-margin penalties (so each margin has its own
smoothing parameter, selected by the existing Fellner-Schall loop — it already
supports `K > 1` penalty blocks). Enables spatially-varying coefficients and
covariate interactions. **Effort: M.** **Oracle:** `mgcv` `te(...)`.

### §2.4 Shared-λ GAM fast mode  *(deferred from v1 Workstream B)*

**What.** A `lambda_mode='shared'` (vs the v1 per-element default) that selects
**one** smoothing-parameter vector across all elements (Fellner-Schall on
pooled sufficient statistics), then runs a single fixed-λ penalised IRLS per
element. Removes the per-voxel outer loop — the dominant GAM cost — for the
common case of homogeneous smoothness across the brain. **Effort: S** (reuse
the FS update on pooled `X^T W X` / energy). **Oracle:** recovers a smooth at
large pooled `V`; the per-element default remains for heterogeneous smoothness.

---

## §3. Theme 3 — `randomise` completeness

### §3.1 Cluster-extent / cluster-mass enhancement  *(deferred from v1 Workstream C)*

**What.** Wire the existing `cluster.cluster_size_map` / `cluster_mass_map`
into `permutation_test` as `enhancement='cluster_extent' | 'cluster_mass'` with
a cluster-forming threshold `cluster_thresh` — the classic FSL cluster-extent /
cluster-mass FWE (max-cluster-statistic null). The maps already exist and are
validated; this is the driver branch + the threshold parameter. **Effort: S.**
**Oracle:** FSL `randomise -c` / `-C`; agreement of the max-cluster null.

### §3.2 F-contrast statistic

**What.** Support multi-row contrasts `C` (the F-statistic from `glm.f_contrast`)
in `permutation_test`, with the matching Freedman-Lane partitioning for a
multi-row effect (Winkler et al. 2014). Per-permutation F-map → enhance → FWE.
**Effort: M.** **Oracle:** observed F == `glm.f_contrast`; FSL F-test mode.

### §3.3 Tail-accelerated p-values (GPD)

**What.** Fit a generalized Pareto distribution to the upper tail of the null
max distribution (Winkler et al. 2016, *Faster permutation inference*) so
corrected p-values are smooth and resolvable **below** the `1/n_perm` discrete
floor at a fraction of the permutations. A `pvalue_method='gpd'` on the
`PermResult`, plus the few-permutation + GPD acceleration path. **Effort: M.**
**Oracle:** PALM's GPD tail; convergence of the GPD p to the empirical p as
`n_perm` grows.

---

## §4. Theme 4 — Regularised connectivity estimators

The small-sample covariance / sparse-precision estimators the connectome
literature defaults to. v1's `stats.covariance` ships the **raw** empirical
`cov` / `precision` / `partialcorr` (need `obs > c`, noisy at `c ≈ obs`); these
are the regularised counterparts that stay invertible and well-conditioned in
the small-sample regime. New module `stats/connectivity.py` (or fold into
`stats.covariance`).

### §4.1 Ledoit-Wolf / OAS analytic shrinkage  *(was [`ledoit-wolf-shrinkage.md`](ledoit-wolf-shrinkage.md))*

**What.** `ledoit_wolf(X) -> (cov, shrinkage)` and `shrunk_covariance(X, *,
method='ledoit_wolf' | 'oas')`: the closed-form convex blend `Σ̂ = (1−α)S + αμI`
(`μ = tr(S)/p`, `α` from the Ledoit-Wolf 2004 / OAS Chen 2010 formulas). Pure
JAX — trace / Frobenius reductions + one scalar `α`, **no solver, trivially
differentiable, GPU-resident, batched over edges/regions**. Feeds the existing
`precision` / `partialcorr` directly (`safe`/rolled-Cholesky inverse of `Σ̂`).
**This is the nilearn-default connectome path** (its `ConnectivityMeasure`
defaults to `LedoitWolf`), so it is the missing piece for a
nilearn-default-equivalent estimator. **Effort: S** (~20-line closed form).
**Oracle:** `sklearn.covariance.LedoitWolf` / `OAS` to ~1e-10.

### §4.2 Graphical LASSO  *(was [`graphical-lasso.md`](graphical-lasso.md), §12.14)*

**What.** `glasso(S, lam) -> Θ` (sparse precision), `glasso_path(S, lambdas)`,
`ebic_score(Θ, S, lam, gamma)`:

    Θ̂ = argmin_Θ  ⟨S, Θ⟩ − log det Θ + λ ‖Θ‖_{1,off}

via coordinate descent (Friedman/Hastie/Tibshirani 2008) — the per-row lasso
update working on the estimated covariance `W` directly (no per-iteration
factorisation), with the `log det` for EBIC from the **rolled Cholesky**
(cuSOLVER-free; `O(p^2)` graph, fine for connectome `p = 100–400` regions). The
sparse precision is the conditional-independence graph the fMRI literature has
defaulted to for ~15 years. **Differentiability:** unrolled-AD through a fixed
iteration count first (the v1 LME-Newton pattern), with the implicit-function
VJP at the KKT-stationary active set as the follow-up (the careful part).
**Effort: M-L.** **Oracle:** `sklearn.covariance.GraphicalLasso` coefficients /
support recovery; EBIC model selection on a known sparse precision.

---

## §5. Cross-cutting substrate reuse

Every v2 item lands on v1 infrastructure — the point of having built it:

- **`stats._smalllinalg.small_inv_logdet`** (closed-form `p≤2`; rolled
  Cholesky `p>2`, `O(p^2)` graph) — the cuSOLVER-free inverse / log-det for
  shrinkage inversion and the GLASSO `log det`.
- **`_blocked_vmap`** — the memory-bounded mass-univariate spine for batched
  shrinkage over edges and per-element GAM/LME.
- **`_varcomp`** REML engine — the q-rank LME and (conceptually) the GAM
  smoothness selection.
- **GLM / penalised-IRLS core + `SplineBasis` contract** — the new bases and
  the F-contrast randomise path.
- **`morphology.connected_components`** — already powering TFCE; the
  cluster-extent/mass enhancement is the same maps under FWE.
- **`linalg._eigsolve.eigsolve_top_k`** — the candidate for the GLASSO / TPRS
  eigen-truncation and the v1-deferred aCompCor consumer that would promote it
  to public (the eigsolve doc names this).

No new heavyweight dependency; all `jax`-only, differentiable where the maths
is smooth, cuSOLVER-free on the broken L4.

---

## §6. Proposed module layout (additions)

```
nitrix/
  linalg/_bspline_core.py  # NEW (Phase 0): uniform_bspline_weights + difference_penalty,
                           #   shared by bias._bspline AND stats.basis (v1 re-implemented these)
  stats/
    _batching.py    # NEW (Phase 0): _blocked_vmap moved out of lme/_varcomp; LME/GLM/GAM/randomise all use it
    _irls.py        # NEW (Phase 0): one penalised-IRLS core (glm._pirls_one + gam._penalised_irls collapse here)
    _family.py      # NEW (Phase 0): Family (stays frozen) + _FAMILIES registry; str|Family resolution
    _smalllinalg.py # Phase 0: + modified-Cholesky pivot floor (sqrt-of-negative guard)
    basis.py        # + thinplate_regression_basis, cyclic_cubic_basis, tensor_smooth (on _bspline_core)
    gam.py          # + lambda_mode='shared'; block= knob; shared REML scaffold
    lme/_varcomp.py # + q-rank Z low-rank rotation (internal; API unchanged)
    connectivity.py # NEW: ledoit_wolf, shrunk_covariance (oas), glasso, glasso_path, ebic_score
    inference/
      randomise.py  # + enhancement='cluster_extent'|'cluster_mass', F-contrast, pvalue_method='gpd';
                    #   block= knob; uncorrected-p on the RAW statistic; constant-voxel masking
      _gpd.py       # NEW: generalized-Pareto tail fit for p-values
```

---

## §7. Phasing & priority (revised after the §8.5 three-lens review)

**Phase 0 — v1 hardening & shared-core refactor (do FIRST).** Every feature
below depends on shared cores that v1 did not factor out; adding features first
would multiply the duplication the review found. Land these before any v2
feature (details + anchors in §8.5).

> **Status (2026-06-17): latent-bug subset SHIPPED** on
> `fix/stats-v2-phase0-hardening` — **H1** (modified-Cholesky pivot floor;
> well-conditioned solves bit-unchanged, singular/boundary now finite),
> **H6** (randomise: uncorrected p on the raw statistic; constant-voxel
> exclusion), and the **H7** latent items (dead `_blocked_vmap` line;
> `penalty_order < n_basis` guard; corrected `_default_theta_init` docstring),
> each with adversarial guards. Remaining Phase 0 = the shared-core refactors
> **H2** (`_bspline_core`), **H3** (`_irls`), **H4** (`Family` registry),
> **H5** (`_batching` + `block=` into glm/gam/randomise), and the rest of H8.

- **H1. Modified-Cholesky pivot floor** in `_smalllinalg` (+ `p≤2` `det`/`a`
  guards) — a correctness prerequisite for q-rank LME (§1.1) and GLASSO (§4.2).
- **H2. `linalg/_bspline_core.py`** — extract the B-spline weights + difference
  penalty, shared by `bias` and `stats.basis`; blocks the three new bases (§2)
  from re-duplicating yet again.
- **H3. `stats/_irls.py`** — one penalised-IRLS core (collapses
  `glm._pirls_one` + `gam._penalised_irls`; removes the redundant post-loop
  re-inversion); substrate for the new families and F-contrast randomise.
- **H4. `Family` registry** (`str | Family`); keep the frozen object.
- **H5. `stats/_batching.py`** — move `_blocked_vmap` out of `lme/_varcomp`;
  wire `block=` into `glm_fit` / `gam_fit` / `permutation_test` (the OOM gap).
- **H6. randomise correctness** — uncorrected p on the *raw* statistic (free;
  `stat_p` already computed), constant-voxel masking before the max statistic.
- **H7. Cleanups** — delete the dead `_blocked_vmap` line; guard
  `_difference_penalty` `order≥k`; `predict`/`compare_models` `Literal`; the
  `NamedTuple`-vs-dataclass result rule; cache penalty `rank` on `SplineBasis`.
- **H8. Adversarial tests** — near-singular Cholesky, σ²-boundary, constant /
  empty voxel, non-contiguous / unequal blocks, TFCE-at-zero.

**Phase 1 — quick wins (S), now higher-value per the perf review.** §2.4
shared-λ GAM (the *single biggest GAM lever*: removes the ~20× per-voxel outer
loop) + §4.1 Ledoit-Wolf/OAS (consumer waiting) + §3.1 cluster-extent/mass
enhancement (≈100× cheaper per permutation than TFCE — relieves the
randomise CRITICAL) + **TFCE perf** (`n_steps` default 100→~50; single-pass
two-sided).

**Phase 2 — mgcv-parity bases (M):** §2.1 thin-plate + §2.2 cyclic, then §2.3
tensor-product (⚠ breaks the disjoint-block `rank_k/λ_k` FS shortcut — needs the
general summed-penalty inverse; see §8.5).

**Phase 3 — GLASSO (M-L):** §4.2 — coordinate descent with the **row loop
rolled** (`lax.scan`, never Python-unrolled — the trap the rolled Cholesky
fixed), `log det` via `spd_inv_logdet_chol` only at converged path points;
unrolled-AD first, implicit-VJP follow-up.

**Phase 4 — randomise depth (M):** §3.2 F-contrast (per-permutation dispersion;
hoist the constant `C·cov·Cᵀ` out of the perm scan) + §3.3 GPD tail (exclude
the observed from the exceedances; lets `n_perm` drop, a linear runtime cut).

**Phase 5 — LME asymptotics (M):** §1.1 q-rank (after H1; gated on a large-`N`
consumer). **Deep perf (gated, high effort):** hierarchical / Pallas
connected-components for TFCE (exploit threshold-nesting monotonicity).

Each phase validates against its pinned oracle (sklearn, mgcv, FSL/PALM) kept
in `tests/` only (SPEC §5.2), with the no-large-intermediate HLO audit and the
cuSOLVER-free guard extended to every new mass-univariate op.

## §8. Open decisions

Resolved by the §8.5 review:

- **`connectivity.py` as a new module** — ✅ confirmed (covariance.py is the
  raw-empirical surface; shrinkage/GLASSO are a distinct regularised family).
- **GLASSO solver** — ✅ coordinate descent (no per-iter factorisation), row
  loop rolled; log-det only at converged path points.
- **GLASSO differentiability** — ✅ unrolled-AD first, implicit-VJP follow-up.
- **`Family`: dataclass vs Protocol** — ✅ keep the frozen dataclass
  (hashability is load-bearing for `vmap`/`custom_vjp`); add a registry, not a
  bare Protocol.
- **GAM REML unification** — ✅ unify the *driver/scaffold* (outer-loop ∘ inner
  IRLS ∘ dispersion ∘ chunking), keep FS vs AI-REML as pluggable *update rules*
  (they are genuinely different, not duplication).

Still open:

- **Scope cut** — whether §2.3 tensor-product and §1.1 q-rank wait for an
  explicit consumer (both are "right but not yet demanded").
- **Phase-0 breadth** — land *all* of H1–H8 before features, or only the
  feature-blocking subset (H1–H5) and fold H6–H8 into the relevant phases.

## §8.5 Three-lens review (2026-06-17) — findings & refinements

A fan-out audit of the shipped v1 code through three lenses (engineering
rigour & mathematical correctness; performance; clean abstraction). The
load-bearing statistics were **confirmed correct** (REML profile / AI-REML
score & average-information, t/F/χ² survival functions, Fellner-Schall, EDF =
influence trace, TFCE integral, Freedman-Lane, the permutation-0 in-scan
capture) and several pieces **confirmed well-designed and not to be touched**
(`_smalllinalg` — used consistently, no cuSOLVER leak; the `_varcomp` REML DRY
win; the `inference` subpackage boundary; the score-kernel/scalarisation
boundary; jaxtyping coverage). The actionable findings:

**Correctness (→ Phase 0 H1/H6, H8):**
- `_smalllinalg.py:77` — hand-Cholesky `sqrt(A[j,j]−s[j])` and the `p≤2`
  `det`/`log(a)` have **no positive floor**; a near-singular or σ²-boundary
  system (where callers' tiny ridge is insufficient) yields silent `NaN`. Add a
  modified-Cholesky pivot floor. *Prerequisite for q-rank and GLASSO.*
- `inference/randomise.py:279` — the **uncorrected** p-map is built on the
  TFCE-*enhanced* value, not the raw statistic (FSL uses the raw); `stat_p` is
  already computed and discarded → free fix.
- `randomise` SE-floor `1e-30` turns a **constant voxel** (σ²=0) into a `+∞`
  statistic that dominates the max-statistic null and deflates every FWE p →
  mask sub-floor-variance voxels before the max.
- `reml.py:188-197` `_default_theta_init` — a false FLAME claim and a `mean(λ)`
  scaling that underflows to the σ²-boundary for large-rank `Z`.
- `basis.py:118` `_difference_penalty` returns a silent all-zero penalty when
  `order ≥ n_basis` → validate and raise.

**Performance (→ Phase 0 H5, Phase 1, deep-perf):**
- `randomise` re-runs full two-sided TFCE on **every** permutation
  (`n_perm × n_steps × connected_components` ≈ 100k CC solves) — the suite's
  dominant cost. Levers: cluster-extent mode (≈100×), `n_steps` default,
  single-pass two-sided, hierarchical/Pallas CC.
- `_blocked_vmap` wired only into LME → GLM/GAM/randomise materialize full-`V`
  `(V,p,p)` (~3.6 GB at `V=1M,p=30`, OOM risk). → H5.
- GAM re-inverts `(p,p)` after both inner and outer loops (`gam.py:243`,
  `glm.py:356`) — carry the converged factor out (→ H3); warm-start fewer inner
  iters on later FS steps.

**Design (→ Phase 0 H2/H3/H4, H7):**
- B-spline weights/penalty re-implemented in `basis.py` instead of shared with
  `bias/_bspline.py` (the v1 plan named the lines) → H2.
- `glm._pirls_one` ≈ `gam._penalised_irls` (drifted: `_EPS` constant vs literal
  `1e-10`) → one `_irls.py` core (H3).
- `Family` has no registry; `family.name == 'gaussian'` string-sniffing is
  scattered → `_FAMILIES` (H4).
- GAM's Fellner-Schall is a standalone REML engine, not on `_varcomp`'s
  scaffold → unify the driver (see §8 resolved).
- Minors: dead `_blocked_vmap` line (`:339`); result-object `NamedTuple`-vs-
  dataclass inconsistency (`FLAMEResult`/`PermResult` could be `NamedTuple`);
  `predict(type=...)` shadows the builtin and is stringly-typed; `gam._assemble`
  host `np.linalg.matrix_rank` is an undocumented eager-construction boundary
  (cache `rank` on `SplineBasis`).

**Per-item v2 traps surfaced (folded into §7):** tensor-product smooths break
the disjoint-block `rank_k/λ_k` Fellner-Schall shortcut; the GPD tail must
exclude the observed from the exceedances; GLASSO's coordinate-descent row loop
must stay rolled and its log-det computed only at converged path points;
F-contrast in randomise needs per-permutation dispersion.

## §9. Cross-references

- v1 ledger (shipped): [`stats-modelling-suite.md`](stats-modelling-suite.md).
- Origin docs folded here: [`ledoit-wolf-shrinkage.md`](ledoit-wolf-shrinkage.md),
  [`graphical-lasso.md`](graphical-lasso.md) (§12.14),
  [`robust-statistics.md`](robust-statistics.md) (§12.7 — adjacent, *not* in v2
  scope; robust LME / M-estimators, a future v3 candidate).
- Substrate: `src/nitrix/stats/_smalllinalg.py`, `stats/lme/_varcomp.py`,
  `stats/glm.py`, `stats/basis.py`, `stats/inference/`,
  `src/nitrix/morphology/_label.py`, `src/nitrix/linalg/_eigsolve.py`.
- Design: [`../design/stats.md`](../design/stats.md),
  [`../design/lme.md`](../design/lme.md),
  [`../design/eigsolve-dispatcher.md`](../design/eigsolve-dispatcher.md).
- Governing spec: SPEC §1, §5.2; SPEC_UPDATE_v0.5 §1 (score-kernel boundary);
  SPEC_UPDATE_v0.3 §12.14 / §13 (GLASSO origin + acceptance protocol).
