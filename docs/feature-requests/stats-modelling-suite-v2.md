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
nitrix/stats/
  basis.py          # + thinplate_regression_basis, cyclic_cubic_basis, tensor_smooth
  gam.py            # + lambda_mode='shared'
  lme/_varcomp.py   # + q-rank Z low-rank rotation (internal; API unchanged)
  connectivity.py   # NEW: ledoit_wolf, shrunk_covariance (oas), glasso, glasso_path, ebic_score
  inference/
    randomise.py    # + enhancement='cluster_extent'|'cluster_mass', F-contrast, pvalue_method='gpd'
    _gpd.py         # NEW: generalized-Pareto tail fit for p-values
```

---

## §7. Phasing & priority (recommended)

Ordered by value/effort, cheapest high-value first:

1. **Quick wins (S):** §4.1 Ledoit-Wolf/OAS (nilearn-default parity, ~20 lines,
   a consumer is waiting per perf-bench) + §3.1 cluster-extent/mass enhancement
   (maps already shipped) + §2.4 shared-λ GAM mode.
2. **mgcv-parity bases (M):** §2.1 thin-plate (the mgcv default) + §2.2 cyclic;
   then §2.3 tensor-product.
3. **GLASSO (M-L):** §4.2 — solver first (unrolled-AD), implicit-VJP follow-up.
4. **randomise depth (M):** §3.2 F-contrast + §3.3 GPD tail.
5. **LME asymptotics (M):** §1.1 q-rank (gated on a large-`N` consumer).

Each phase validates against its pinned oracle (sklearn, mgcv, FSL/PALM) kept
in `tests/` only (SPEC §5.2), with the no-large-intermediate HLO audit and the
cuSOLVER-free guard extended to every new mass-univariate op.

## §8. Open decisions (to ratify before implementing)

- **`connectivity.py` vs fold into `covariance.py`** — recommend a new module
  (covariance.py is the raw-empirical surface; shrinkage/GLASSO are a distinct
  regularised family).
- **GLASSO solver** — coordinate descent (recommended; the reference algorithm,
  no per-iter factorisation) vs ADMM (proximal log-det via `linalg.symlog`, an
  eigendecomposition each step — cuSOLVER-heavier).
- **GLASSO differentiability** — unrolled-AD first vs implicit-VJP up front.
- **Scope cut** — whether §2.3 tensor-product and §1.1 q-rank wait for an
  explicit consumer (both are "right but not yet demanded").

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
