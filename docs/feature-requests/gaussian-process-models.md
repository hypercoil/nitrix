# Gaussian-process & hierarchical-GP models — `nitrix.stats.gp`

> **Status (2026-07-07): SHIPPED (`nitrix.stats.gp` + `nitrix.stats.hgp`).**
> The full headline scope is built, exported, and tested: kernel spectral
> densities (`se_spectral_density` / `matern_spectral_density`), the HSGP basis
> (`hsgp_basis` / `hsgp_basis_nd`, ARD), the standalone `gp_fit` / `GPResult` /
> `gp_predict` (HSGP + exact engines, MAP-`ρ` priors, non-Gaussian PQL-REML),
> AR1/CAR1/CS residual composition (`corr=`), the factor-smooth + hierarchical /
> nested `hgp_fit` / `HGPResult` / `hgp_predict`, and `gp_aic` / `gp_bic`.
> **Residual (deferred, minor):** the optional periodic kernel (1 of 5) and an
> optional perf-bench case. This doc is retained as the family ledger. **Primary
> reduced-rank engine: the Hilbert-space
> approximate GP (HSGP; Solin–Särkkä 2020, Riutort-Mayol/Bürkner 2023)** — a
> fixed Laplace-eigenfunction basis whose hyperparameters enter *only* as a
> diagonal spectral reweighting, so lengthscale `ρ` estimation stays `eigh`-free
> and inside the suite's fast paths (§3). The kriging `bs="gp"` basis
> (`gp_basis`) is retained for parity. Scope here is the **mixed-model sense** of
> "hierarchical GP" (multi-level GP ≡ GAMM with GP components); the
> hierarchical-Bayesian-regression / normative-modelling flavour (HBR, PCN
> toolkit) is explicitly **deferred** (see §6, "(b)"). Sibling context:
> [`stats-modelling-suite-v3.md`](stats-modelling-suite-v3.md) (the shipped
> GAMM surface this builds on) and [`stats-suite-audit.md`](stats-suite-audit.md).

## 1. Objective

Add Gaussian-process regression as a first-class member of the mass-univariate
stats suite, on the same `*_fit(Y, X, …) → *Result`-pytree spine as
`glm_fit` / `gam_fit` / `glmm_fit` / `reml_fit`, and extend it to the
**hierarchical** (multi-level) case where a population-level GP is partially
pooled with group-level GP deviations.

This is not a foreign paradigm bolted on — it is the next node on the suite's
own organising identity, **penalised regression ≡ variance-components REML ≡
mixed model**. A GP is that identity with a *kernel* prior covariance:

```
y_v = X β_v + f_v + ε_v ,   f_v ~ N(0, σ_f² K(θ)) ,   ε_v ~ N(0, σ_e² I)
Cov(y_v) = σ_f² K(θ) + σ_e² I
```

which is structurally identical to `reml_fit`'s `V = σ_b² ZZᵀ + σ_e² I` with
`K` in the role of `ZZᵀ`. The "H" in HGP is then exactly what a GAMM already
is — *add a grouped GP component*.

## 2. Findings — the substrate is already ~70% present

Verified against the live `src/nitrix` surface (2026-06-20).

| Capability a GP needs | Where it lives | State |
|---|---|---|
| **1-D reduced-rank GP smooth** (mgcv `bs="gp"`) | `stats/basis.py:724` `gp_basis` | **Ships.** Matérn-3/2 kriging, knot-Gram `eigh`, eigen-reparam to **identity penalty** (`:783–797`); rides `gam_fit` as a `SplineBasis(kind='gp')`. Kept as the parity basis; HSGP becomes primary (§3) |
| **Generic penalised-REML loop** (Fellner–Schall) | `stats/gam.py:393–426` | **Ships, and is generic over the penalty `Sₖ`** — a kernel penalty plugs in unchanged; per-element + shared-λ Gaussian fast path + non-Gaussian PIRLS |
| **Fixed-kernel GP solver, in disguise** | `stats/lme/reml.py` `reml_fit` | **Ships.** FaST-LMM spectral trick (`ZZᵀ = UΛUᵀ` once, profile β, Newton on log-variances). Swap `ZZᵀ→K` ⇒ a fixed-shape GP |
| Correlated-residual structures (AR1/CAR1/CS) | `stats/lme/_corrfit.py` `gls_fit`, `lme_fit(corr=)` (v3 §1.4) | **Ships.** First-class whitening in the *same* solver — lets a GP trend + autocorrelation be fit jointly (§5, R6) |
| ML covariance kernels | `linalg/kernel.py` | `rbf_kernel` (`:247`), `gaussian_kernel` (`:262`), linear/poly/sigmoid/cosine. **Matérn-3/2 inline** at `basis.py:716`. **Missing:** spectral densities `S_θ` for Matérn-1/2·5/2, periodic |
| Cholesky solve, log-det — **cuSOLVER-free** | `linalg/solve.py:cho_solve` (`:62`), `_smalllinalg.py:spd_inv_logdet_chol` (`:93`) | **Ships.** Exactly the GP marginal-likelihood primitives |
| Differentiable top-k eigensolve / CG / randomised SVD | `linalg/_eigsolve.py:eigsolve_top_k` (`:829`, impl-VJP), `krylov.py:cg` (`:38`), `decompose.py:randomized_svd` (`:70`) | **Ships.** For the kriging-legacy differentiable-`ρ` path (D3) / Nyström |
| Hierarchy scaffolding | `glmm.py`, `lme/_nested.py`, `_crossed.py`, `basis.py` `by_factor_smooth` / `varying_coefficient_smooth` / `re_smooth` / `REBasis` (v3 §2, §3.1 — shipped) | **Ships.** The "H" of HGP — factor-smooths + RE pooling + nested/crossed REML |
| Per-smooth effect + significance | `gam.py` `smooth_partial_effect`, `smooth_significance` (Wood 2013) | **Ships.** A reduced-rank GP term gets credible bands + a term test **for free** |
| Result pytree registration | `stats/_result.py:37` `register_result` | **Ships.** `GPResult` is a 20-line dataclass under this decorator |

**Net:** a single-covariate kriging GP smooth works *now*. Missing are (i) the
**HSGP basis** — the fixed Laplace-eigenfunction construction that is the
primary engine (§3/§5) — plus the kernel spectral densities `S_θ` feeding it,
(ii) **kernel-hyperparameter (lengthscale) estimation**, which the HSGP basis
makes `eigh`-free and tractable, and (iii) the standalone `gp_fit` / `GPResult`
and the HGP wrapper.

## 3. Lengthscale estimation — why HSGP, not kriging, is primary

Every fast path in the suite rests on a **fixed eigenbasis**: `reml_fit`
eigendecomposes `ZZᵀ` *once*; `gp_basis` eigendecomposes the knot Gram *once*
(`:783`); then all per-element work is `O(N)` elementwise on the eigenvalues.
That holds because with a fixed lengthscale only the **amplitude** varies — a
linear variance component. The crux: estimating the lengthscale `ρ` of a
**kriging** kernel changes `K`'s **eigenvectors**, not just its eigenvalues, so
the fixed-basis trick breaks — which is why `gp_basis` *fixes* `ρ = range/2`
(`basis.py:767`) and the analytic average-information `_varcomp` path cannot
take an arbitrary `K(θ)`. Profiling `ρ` over a grid (the mgcv `bs="gp"` route)
re-`eigh`s per `ρ` and is the "good-enough start" the GP community rightly
flags as poorly-behaved.

**The Hilbert-space approximate GP (HSGP) removes the crux by construction.**
Approximate a stationary GP on a bounded domain `[-L, L]^D` by the
Laplace–Dirichlet eigenfunctions `φ_j` (eigenvalues `λ_j`), which are
**independent of the kernel hyperparameters**:

```
f(x) ≈ Σ_j √(S_θ(√λ_j)) · φ_j(x) · β_j ,   β_j ~ N(0, 1)
```

The design `Φ = [φ_j(x)]` is built **once**; `(ρ, σ_f)` enter *only* through the
kernel's spectral density `S_θ(√λ_j)`, which rescales the coefficient prior
variances. So in the HSGP basis:

- `ρ` no longer moves any eigenvector — it is a smooth, **diagonal** reweighting
  of a fixed `Φ` (penalty `diag(1 / (σ_f² · S_θ(√λ_j)))`), with **no `eigh` in
  the inner loop**.
- The Fellner–Schall / `_varcomp` machinery generalises by one step: from "one
  scalar `λ` scaling a fixed `S`" to "`(σ_f², ρ)` shaping a fixed-`Φ` diagonal
  `S(ρ)`". The marginal likelihood is smooth and cheaply differentiable in `ρ`.
- Cost stays `O(V·N·m)` (`m` = #basis) with `Φ` shared under `vmap` — the
  suite's fast-path philosophy is **preserved under `ρ` estimation**, not worked
  around.

This is the principled reason to make **HSGP the primary reduced-rank engine**
and keep kriging `bs="gp"` only for parity: HSGP is the one construction whose
hyperparameter estimation is native to nitrix's fixed-eigenbasis design.

**Decision.** Estimate `ρ` *shared across voxels*, amplitude `σ_f²` and noise
`σ_e²` *per voxel* (matches the shared-design assumption; one `Φ` reused under
`vmap`). Resolutions, in priority order: **(D1) HSGP diagonal-`S(ρ)` profile or
gradient** — the default, `eigh`-free; **(D2)** kriging grid-`ρ` (legacy parity,
re-`eigh` per `ρ`); **(D3)** differentiate through `eigh` via `eigsolve_top_k`'s
impl-VJP (`_eigsolve.py:228`) only if a kriging path needs gradient `ρ`.

**Priors on `ρ` (the (a)-scope nod).** A genuine *posterior* over `ρ` is the
deferred (b) Bayesian scope (§6). But the well-behaved HSGP likelihood admits a
lightweight **penalised/MAP `ρ`** within (a): add a prior-as-penalty on `log ρ`
(e.g. a half-normal / inverse-gamma on the lengthscale) to the REML objective.
One extra term; it keeps the process from over-flexing and — crucially — stops
the trend from absorbing short-timescale autocorrelation that belongs in a
correlated residual (§5; §9 R6). Full Bayesian `ρ` priors remain (b).

## 4. The other invariant — `O(V·N)` memory, never `V·N²`

The suite's value proposition is that 100k–1M per-voxel fits stay in
`O(V·N)` memory with no per-voxel `N×N` materialisation — there is an explicit
HLO-budget regression test for it (`lme.md:128`,
`test_reml_max_tensor_size_within_budget`). A naïve per-voxel **full** GP is
`O(V·N²)` memory / `O(V·N³)` compute and would violate that contract outright.

**Reduced-rank (`m ≪ N`) is therefore near-mandatory here, not merely
convenient** — it keeps cost at `O(V·N·m)` *and* is the path with the most
reuse (HSGP basis + `gam_fit`). Because the HSGP design `Φ` is
hyperparameter-independent (§3), even `ρ` *estimation* stays within this
budget — no per-`ρ` decomposition is ever materialised, the one place the
kriging route would. The full-rank dense-Cholesky GP is offered only as a
small-`N`, shared-kernel specialisation (§5, Tier 2b). A GP-specific HLO-budget
test mirrors the REML one.

## 5. Design — proposed surface

Three tiers, increasing ambition and blast radius. Each is independently
shippable and additive (new families never touch existing code).

### Tier 1 — HSGP smooth term: multi-kernel, `ρ`-estimated (effort **S–M**)

Add the HSGP basis as the primary construction; keep kriging `gp_basis` for
parity. No `gam_fit` change for fixed `ρ` (the FS loop is already generic); the
one small generalisation is the `(σ_f², ρ)` diagonal-`S(ρ)` step (§3):

```python
# basis.py — HSGP basis (PRIMARY): fixed Laplace-eigenfunction design Φ;
# hyperparameters enter only via the spectral density S_θ(√λ_j).
def hsgp_basis(x, n_basis=20, *, kernel='matern52', rho=None, sigma=None,
               boundary=1.5, center=True): ...
#   kernel ∈ {'matern12'(exp), 'matern32', 'matern52', 'rbf'(SE), 'periodic'}
#   boundary = L / max|x|  (domain extension; see (m, L, ρ) caveat, §9 R4)
#   rho=None ⇒ estimated (diagonal-S(ρ) REML, §3);  float ⇒ fixed.
def hsgp_basis_nd(X, n_basis=..., *, kernel='matern52', rho=None, ard=False): ...
#   multi-D via the tensor product of per-axis Laplace eigenfunctions
#   (ARD: one ρ per axis). NATIVE to HSGP — no knot/inducing construction.

# kriging GP basis (parity / legacy): re-eigh of the knot Gram per ρ.
def gp_basis(x, n_basis=10, *, kernel='matern32', rho=None, ...): ...  # surface unchanged
```

Reuses `gam_fit`, `smooth_partial_effect`, `smooth_significance` **verbatim**
for fixed `ρ`; the `ρ`-estimation step is the §3 diagonal-`S(ρ)` profile/grad.
This is the highest-leverage move.

### Tier 2 — standalone `gp_fit` / `GPResult` (effort **M**)

```python
@register_result(
    children=('coef', 'cov_unscaled', 'theta', 'log_mlik', 'edf', 'dispersion'),
    aux=('kernel', 'n_obs', 'rank'),
)
@dataclass(frozen=True)
class GPResult:
    coef:         Float[Array, 'V m']      # reduced-rank posterior weights
    cov_unscaled: Float[Array, '... m m']  # posterior covariance (→ predictive var)
    theta:        Float[Array, 'V h']      # log-hyperparameters (σ_f², ρ…, σ_e²)
    log_mlik:     Float[Array, 'V']        # log marginal likelihood (→ aic/bic/compare_models)
    edf:          Float[Array, 'V']        # effective dof of the GP fit
    dispersion:   Float[Array, 'V']
    kernel:       str
    n_obs:        int
    rank:         int

def gp_fit(Y, x, *, parametric=None, kernel='matern52', rank=...,
           engine='hsgp', select='shared-rho', map_rho=None, corr=None,
           n_iter=..., ridge=1e-8, block=None) -> GPResult: ...
def gp_predict(result, basis, x_new) -> Tuple[mean, var]: ...
```

Default engine: **HSGP reduced-rank** (rides the Tier-1 `hsgp_basis`; fixed `Φ`,
β & predictive var from the existing penalised solve; `log_mlik` from the REML
criterion; `ρ` by the §3 diagonal-`S(ρ)` step, optionally MAP-penalised via
`map_rho=`). `select='shared-rho'` ⇒ one `Φ` shared under `vmap`, `ρ` estimated
jointly, `eigh`-free (D1). `engine='full-rank-shared'` routes the small-`N`
shared-kernel case through **`reml_fit` almost verbatim** (`K` for `ZZᵀ`,
amplitude+noise as the two variance components) — Tier 2b. `corr=` threads an
AR1/CAR1/CS residual through the same fit (§5 *Composability*). `log_mlik` feeds
the shipped `aic`/`bic`/`compare_models` so GP-vs-spline comparison is immediate.

### Tier 3 — hierarchical GP, mixed-model sense (effort **M–L**)

The natural realisation is mgcv's **factor-smooth** (`bs="fs"`): a *shared*
smoother with per-group deviations under **one** common hyperparameter set,
the deviations pooled as random effects. That is precisely a hierarchical GP
with shared kernel + group-level realisations — and it is GAMM-shaped, so it
composes the **already-shipped** `re_smooth` (v3 §2) + `by_factor_smooth`
(v3 §3.1) + `gam_fit`:

```python
def gp_factor_smooth(x, f, *, kernel='matern52', rho=None): ...
#   bs="fs" GP analogue: per-level hsgp_basis sharing ONE (σ_f², ρ) + an RE
#   pooling penalty (partial pooling). Distinct from by_factor_smooth (indep. λ).

def hgp_fit(Y, x, *, group, kernel='matern52', parametric=None, …) -> GPResult:
#   thin wrapper: population GP mean + per-group GP deviations (shared θ),
#   fit by the existing penalised-REML / PQL machinery.
```

Nested two-level HGP (`(gp | g1/g2)`) maps onto the shipped nested-LME
structure (`lme/_nested.py`); with HSGP each level shares the *same* fixed `Φ`,
so the per-level blocks differ only in their diagonal `S(ρ)` — cleaner than the
kriging case (which needed a shared eigenbasis assumption). Otherwise it is an
additive stack of `gp_factor_smooth` blocks (gam_fit already sums penalties). No
new optimiser — reuses the v3 saddle-free damped Newton.

### Composability — GP trend + correlated residuals in one fit

A headline advantage over the spline route. The suite already carries AR1 /
CAR1 / compound-symmetry residual structures as first-class **whitening** in the
same penalised-REML / GLS solver (`gls_fit(..., corr=)`, `lme_fit(corr=)`; v3
§1.4). So "HSGP trend + AR(1) correlated residual" is just two terms in one
fit — `ρ` joins the REML `θ` alongside the AR parameter — where mgcv must route
the correlation through `gamm()` + `nlme`'s PQL `corAR1` and often cannot pin
both. The identifiability caveat the GP literature raises holds and motivates
the design: trend and autocorrelation compete for the same short-timescale
variance, so a **well-behaved** basis (HSGP) plus, ideally, the §3 MAP prior on
`ρ` is what keeps them separable. This is why HSGP-over-kriging matters even
more here than in a trend-only model (§9 R6).

## 6. Scope & interpretation

"HGP" here = **(a) hierarchical / multi-level GP in the mixed-model sense** —
the §5 Tier-3 target, which maps cleanly onto the GAMM/LME hierarchy and the
suite's REML/PQL idiom.

**(b) hierarchical-Bayesian-regression (HBR / normative-modelling) GP** —
partial pooling of *hyperparameters* across sites/groups with a full Bayesian
posterior (PCN-toolkit style) — is a genuinely different estimator
(variational / MCMC / empirical-Bayes nesting), sits outside the frequentist
REML/PQL spine, and is **explicitly deferred**. It would share Tiers 1–2 but
fork at Tier 3; file as a separate proposal when prioritised.

**Relation to `{brms}` / `{mvgam}`.** Those tools fit HSGP-based GPs under full
Bayes (Stan/HMC): full posteriors, priors on `ρ`/`σ`, and — in `mvgam` —
explicit latent dynamic trend + autocorrelation decomposition for time series.
nitrix is frequentist REML, GPU, **mass-univariate at 10⁵–10⁶ units**, where HMC
is infeasible (a GP *per voxel/vertex*, not one rich series). The right reading
of the GP community's critique: adopt the part that is a basis/numerics
improvement — **HSGP** — into the at-scale regime (this revision); the full-Bayes
part — *posterior* priors on `ρ`, latent dynamic trends — stays (b) / `brms` /
`mvgam` territory by design. The (a) scope still gets a MAP `ρ` (§3) and
composable correlated residuals (§5) as the frequentist analogue.

## 7. Engineering plan & phasing

| Phase | Deliverable | Files | Validation anchor | Effort |
|---|---|---|---|---|
| **P1** | Kernel spectral densities `S_θ(√λ)`: Matérn-1/2·3/2·5/2, RBF/SE, periodic; ARD | `linalg/kernel.py` (+inline lift from `basis.py:716`) | `S_θ` vs analytic / `brms gp()` spectral density to ~1e-12 | XS–S |
| **P2** | Tier 1: **`hsgp_basis`** (primary) + `hsgp_basis_nd`; kriging `gp_basis(kernel=…)` kept for parity; diagonal-`S(ρ)` REML `ρ`-estimation | `stats/basis.py` | `brms gp()` HSGP partial effect; HSGP → full-GP posterior mean (`sklearn`/`GPy`) to ~1e-3 as `m→∞`; `ρ` recovery on synthetic | S–M |
| **P3** | Tier 2: `gp_fit`/`GPResult`/`gp_predict` (HSGP, shared-`ρ`, optional MAP `ρ`) | **new** `stats/gp.py` + 2 lines `stats/__init__.py` | `log_mlik` & `(σ_f, ρ, σ_e)` vs `sklearn`/`GPy`/`brms` to ~1e-4; finite-diff grad-through-`ρ`; HLO-budget test | M |
| **P4** | Tier 2b: full-rank shared-kernel via `reml_fit` (`K`↔`ZZᵀ`) | `stats/gp.py` (thin) | equals dense Cholesky GP to ~1e-7; agrees with P3 HSGP as `m→∞` | S |
| **P4b** | GP(HSGP) trend + `corr=` correlated residual in one fit (compose v3 §1.4) | `stats/gp.py` (wire `corr=` through) | trend/AR1 separation recovered on synthetic; vs dense GLS-REML ~1e-5 | S |
| **P5** | Tier 3: `gp_factor_smooth` + `hgp_fit` (factor-smooth / `fs`) | `stats/basis.py`, `stats/gp.py` | mgcv `s(x, g, bs="fs")`; per-group curve recovery corr > 0.97 (cf. v3 §3.1) | M |
| **P6** | Nested HGP `(gp | g1/g2)` | `stats/gp.py` (reuses `lme/_nested.py`) | dense REML reference ~1e-5 | M |
| **P7** *(opt.)* | perf-bench case | `nitrix-perf-bench/.../cases/gp_fit.py` | vs `sklearn`-looped; contract test on `op_qualname` | S |

Person-time, rough: Tier 1 (P1–P2) ~2 days; Tier 2 (P3–P4b) ~1 week; Tier 3
(P5–P6) ~1.5–2 weeks. **Full (a)-scope ≈ 3–4 weeks** incl. validation.

**Suite invariants every phase must hold** (the audit's standing bar):
cuSOLVER-free (route per-element solves through `spd_inv_logdet_chol` /
`sym_eig_jacobi`; HSGP needs **no** runtime `eigh` — `Φ` is closed-form, §3);
jit/`vmap`/grad-clean (every array field a `register_result` child, θ
hashable-free); `O(V·N·m)` memory with an HLO-budget test; mass-univariate
`(Y:(V,N), X:(N,p))` keyword-only signature; ruff/mypy clean.

## 8. Touch-point checklist (blast radius)

**Create:** `stats/gp.py` (~250–450 LOC); `tests/test_gp.py` (~150–250 LOC);
optional `nitrix-perf-bench/.../cases/gp_fit.py` + its contract test.
**Modify:** `stats/basis.py` (add `hsgp_basis` / `hsgp_basis_nd` — primary; keep
`gp_basis` for parity; add `gp_factor_smooth`); `linalg/kernel.py` (+spectral
densities `S_θ`); `stats/__init__.py` (import + `__all__`, 2–4 lines);
`tests/test_basis.py` (+HSGP cases); 1–2 lines in the `stats/__init__.py`
module docstring. **Not required:** no `pyproject.toml` entry points; no
top-level `nitrix` re-export (`nitrix/__init__.py` is docstring-only); no result
registry beyond the `@register_result` decorator. **Existing code touched:
effectively nil** — every change is additive.

## 9. Risks & open decisions

- **R1 (decided, §3).** Primary engine = **HSGP** (fixed `Φ`, `ρ` as a diagonal
  spectral reweighting, `eigh`-free). `ρ` shared-across-voxels by default;
  per-voxel `ρ` is an escape hatch. Kriging `gp_basis` retained for parity.
- **R2 (decided, §4).** Reduced-rank is the default engine; full-rank dense GP
  only for small-`N` shared-kernel (Tier 2b). Add the HLO-budget test.
- **R3 (resolved — was open).** Multi-D / ARD is **native to HSGP** (tensor
  product of per-axis Laplace eigenfunctions, one `ρ` per axis); the earlier
  "tensor-product of kriging margins first" plan is dropped — HSGP supersedes it.
- **R4 (open — HSGP accuracy).** HSGP accuracy couples `m` (#basis) and the
  boundary factor `L = boundary·max|x|` to `ρ`: small `ρ` needs larger `m`
  (Riutort-Mayol et al. give the guidance). Encode the `(m, L, ρ)` relationship
  as defaults + a validity check/warning; a constraint, not a blocker.
- **R5 (note).** Inference semantics already covered: HSGP terms inherit
  `smooth_significance` + credible bands; `GPResult.log_mlik` drives
  `compare_models`. No new inference surface required for (a)-scope.
- **R6 (note — identifiability).** Trend vs short-term autocorrelation compete
  for the same variance; the well-behaved HSGP basis + the §3 MAP `ρ` + a
  `corr=` residual (§5) are what keep them separable. Test the separation
  explicitly (P4b).

## 10. Cross-references

- `src/nitrix/stats/basis.py:724` (`gp_basis`, kriging parity basis), `:716`
  (`_matern32_kernel`), `:783–797` (eigen-reparam → identity penalty), `:767`
  (fixed `ρ` — the limitation HSGP removes).
- `src/nitrix/stats/gam.py:393–426` — the generic Fellner–Schall penalty loop.
- `src/nitrix/stats/lme/reml.py` — FaST-LMM `reml_fit` (the `K`↔`ZZᵀ` route);
  [`docs/design/lme.md`](../design/lme.md).
- `src/nitrix/stats/lme/_corrfit.py` `gls_fit` + v3 §1.4 (AR1/CAR1/CS `corr=`)
  — the correlated-residual composition (§5, R6).
- `src/nitrix/stats/_result.py:37` — `register_result` (the `GPResult` pattern).
- `src/nitrix/linalg/{kernel.py:247,262, solve.py:62, _smalllinalg.py:93,`
  `_eigsolve.py:829, krylov.py:38, decompose.py:70}` — GP linear-algebra prims.
- **HSGP literature.** Solin & Särkkä (2020), *Hilbert space methods for
  reduced-rank Gaussian process regression*, Stat. Comput. 30:419–446;
  Riutort-Mayol, Bürkner, Andersen, Solin, Vehtari (2023), *Practical Hilbert
  space approximate Bayesian GPs for probabilistic programming*, Stat. Comput.
  33:17. Reference implementations: `brms::gp()`, `mvgam`.
- [`stats-modelling-suite-v3.md`](stats-modelling-suite-v3.md) §2 (GAMM
  surface: `re_smooth`/`REBasis`), §3.1 (`by_factor_smooth`), §1.4 (`gls_fit`
  `corr=`) — the Tier-3 + composability substrate; the shared "penalised GLM ≡
  variance-components REML ≡ mixed model" framing.
- [`stats-suite-audit.md`](stats-suite-audit.md) — standing-suite bar.
- Deferred sibling (b): HBR / normative-modelling GP — to be filed separately.
