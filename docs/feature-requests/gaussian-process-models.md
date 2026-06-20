# Gaussian-process & hierarchical-GP models — `nitrix.stats.gp`

> **Status (2026-06-20): proposed — substrate ~70% shipped; the model
> wrappers + hyperparameter estimation + alternative/multi-D kernels are the
> net-new work.** A single-covariate GP *smooth* already fits today
> (`gp_basis` + `gam_fit`). Scope here is the **mixed-model sense** of
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
| **1-D reduced-rank GP smooth** (mgcv `bs="gp"`) | `stats/basis.py:724` `gp_basis` | **Ships.** Matérn-3/2 kriging, knot-Gram `eigh`, eigen-reparam to **identity penalty** (`:783–797`); rides `gam_fit` as a `SplineBasis(kind='gp')` |
| **Generic penalised-REML loop** (Fellner–Schall) | `stats/gam.py:393–426` | **Ships, and is generic over the penalty `Sₖ`** — a kernel penalty plugs in unchanged; per-element + shared-λ Gaussian fast path + non-Gaussian PIRLS |
| **Fixed-kernel GP solver, in disguise** | `stats/lme/reml.py` `reml_fit` | **Ships.** FaST-LMM spectral trick (`ZZᵀ = UΛUᵀ` once, profile β, Newton on log-variances). Swap `ZZᵀ→K` ⇒ a fixed-shape GP |
| ML covariance kernels | `linalg/kernel.py` | `rbf_kernel` (`:247`), `gaussian_kernel` (`:262`), linear/poly/sigmoid/cosine. **Matérn-3/2 inline** at `basis.py:716`. **Missing:** Matérn-5/2, exponential (Matérn-1/2), periodic |
| Cholesky solve, log-det — **cuSOLVER-free** | `linalg/solve.py:cho_solve` (`:62`), `_smalllinalg.py:spd_inv_logdet_chol` (`:93`) | **Ships.** Exactly the GP marginal-likelihood primitives |
| Differentiable top-k eigensolve / CG / randomised SVD | `linalg/_eigsolve.py:eigsolve_top_k` (`:829`, impl-VJP), `krylov.py:cg` (`:38`), `decompose.py:randomized_svd` (`:70`) | **Ships.** For reduced-rank / Nyström / differentiable-`ρ` paths |
| Hierarchy scaffolding | `glmm.py`, `lme/_nested.py`, `_crossed.py`, `basis.py` `by_factor_smooth` / `varying_coefficient_smooth` / `re_smooth` / `REBasis` (v3 §2, §3.1 — shipped) | **Ships.** The "H" of HGP — factor-smooths + RE pooling + nested/crossed REML |
| Per-smooth effect + significance | `gam.py` `smooth_partial_effect`, `smooth_significance` (Wood 2013) | **Ships.** A reduced-rank GP term gets credible bands + a term test **for free** |
| Result pytree registration | `stats/_result.py:37` `register_result` | **Ships.** `GPResult` is a 20-line dataclass under this decorator |

**Net:** a single-covariate GP smooth works *now*. Missing are (i) alternative
& multi-dimensional kernels, (ii) **kernel-hyperparameter (lengthscale)
estimation**, (iii) the standalone `gp_fit` / `GPResult` and the HGP wrapper.

## 3. The one hard part — lengthscale ↔ eigenbasis

Every fast path in the suite rests on a **fixed eigenbasis**: `reml_fit`
eigendecomposes `ZZᵀ` *once*; `gp_basis` eigendecomposes the knot Gram *once*
(`:783`); then all per-element work is `O(N)` elementwise on the eigenvalues.
That holds because with a fixed lengthscale only the **amplitude** varies — a
linear variance component. Estimating the lengthscale `ρ` changes `K`'s
**eigenvectors**, not just its eigenvalues, so the fixed-basis trick no longer
applies (and `gp_basis` accordingly *fixes* `ρ = range/2`, `basis.py:767`; the
analytic average-information `_varcomp` path is likewise hardcoded to linear
components and cannot take an arbitrary `K(θ)`).

Two clean resolutions, **both backed by primitives already in-repo**:

- **D1a — profile/grid `ρ`** (default): rebuild the reduced-rank basis per `ρ`
  on a short grid (or a 1-D outer optimise) and pick the best REML score.
  Embarrassingly parallel, vmap-clean, cheap when rank `k ≪ N`.
- **D1b — differentiate through `eigh`**: `eigsolve_top_k` already carries an
  implicit-VJP subspace kernel (`_eigsolve.py:228`), so gradient-based `ρ` is
  feasible where the grid is too coarse.

**Decision (recommended): estimate `ρ` *shared across voxels*, amplitude `σ_f²`
(λ) and noise `σ_e²` *per voxel*.** This matches the suite's shared-design
assumption (one kernel eigendecomposition reused under `vmap`), keeps the cost
model intact, and covers the dominant neuroimaging use (one group-level
covariate axis, V responses). Per-voxel `ρ` is a Tier-2 escape hatch, not the
default.

## 4. The other invariant — `O(V·N)` memory, never `V·N²`

The suite's value proposition is that 100k–1M per-voxel fits stay in
`O(V·N)` memory with no per-voxel `N×N` materialisation — there is an explicit
HLO-budget regression test for it (`lme.md:128`,
`test_reml_max_tensor_size_within_budget`). A naïve per-voxel **full** GP is
`O(V·N²)` memory / `O(V·N³)` compute and would violate that contract outright.

**Reduced-rank (`k ≪ N`) is therefore near-mandatory here, not merely
convenient** — it keeps cost at `O(V·N·k)` *and* is the path with the most
reuse (`gp_basis` + `gam_fit`). The full-rank dense-Cholesky GP is offered only
as a small-`N`, shared-kernel specialisation (§5, Tier 2b). A GP-specific
HLO-budget test mirrors the REML one.

## 5. Design — proposed surface

Three tiers, increasing ambition and blast radius. Each is independently
shippable and additive (new families never touch existing code).

### Tier 1 — multi-kernel, `ρ`-selected GP *smooth term* (effort **S–M**)

Extend the existing basis, no `gam_fit` change (the FS loop is already generic):

```python
# basis.py — extend in place, backward-compatible (default kernel/ρ unchanged)
def gp_basis(x, n_basis=10, *, kernel='matern32', rho=None,
             max_knots=100, bounds=None, center=True): ...
#   kernel ∈ {'matern12'(exponential), 'matern32', 'matern52', 'rbf'(SE), 'periodic'}
#   — the inline Matérn-3/2 at basis.py:716 is the 10-line-each template.

def gp_basis_nd(X, n_basis=..., *, kernel='matern52', rho=None, ard=False): ...
#   multi-D GP smooth: isotropic (or ARD) distance over D columns, knot/inducing
#   construction; OR compose existing tensor_product_basis over 1-D gp margins.

def gp_smooth(x, *, kernel='matern52', select='reml', grid=None): ...
#   ρ-selection wrapper (D1a): returns the basis at the REML-optimal ρ.
```

Reuses `gam_fit`, `smooth_partial_effect`, `smooth_significance` **verbatim**.
This is the highest-leverage move and ~80 % done already.

### Tier 2 — standalone `gp_fit` / `GPResult` (effort **M**)

```python
@register_result(
    children=('coef', 'cov_unscaled', 'theta', 'log_mlik', 'edf', 'dispersion'),
    aux=('kernel', 'n_obs', 'rank'),
)
@dataclass(frozen=True)
class GPResult:
    coef:         Float[Array, 'V k']      # reduced-rank posterior weights
    cov_unscaled: Float[Array, '... k k']  # posterior covariance (→ predictive var)
    theta:        Float[Array, 'V h']      # log-hyperparameters (σ_f², ρ…, σ_e²)
    log_mlik:     Float[Array, 'V']        # log marginal likelihood (→ aic/bic/compare_models)
    edf:          Float[Array, 'V']        # effective dof of the GP fit
    dispersion:   Float[Array, 'V']
    kernel:       str
    n_obs:        int
    rank:         int

def gp_fit(Y, x, *, parametric=None, kernel='matern52', rank=...,
           hyper='reml', select='shared-rho', n_iter=..., ridge=1e-8,
           block=None) -> GPResult: ...
def gp_predict(result, basis, x_new) -> Tuple[mean, var]: ...
```

Default engine: **reduced-rank** (rides the Tier-1 basis; β & predictive var
from the existing penalised solve; `log_mlik` from the REML criterion).
`hyper='shared-rho'` ⇒ one `eigh`/grid shared under `vmap` (D1a). A
`select='full-rank-shared'` flag routes the small-`N` shared-kernel case
through **`reml_fit` almost verbatim** (`K` for `ZZᵀ`, amplitude+noise as the
two variance components) — Tier 2b. `log_mlik` feeds the shipped
`aic`/`bic`/`compare_models` so GP-vs-spline model comparison is immediate.

### Tier 3 — hierarchical GP, mixed-model sense (effort **M–L**)

The natural realisation is mgcv's **factor-smooth** (`bs="fs"`): a *shared*
smoother with per-group deviations under **one** common hyperparameter set,
the deviations pooled as random effects. That is precisely a hierarchical GP
with shared kernel + group-level realisations — and it is GAMM-shaped, so it
composes the **already-shipped** `re_smooth` (v3 §2) + `by_factor_smooth`
(v3 §3.1) + `gam_fit`:

```python
def gp_factor_smooth(x, f, *, kernel='matern52', rho=None): ...
#   bs="fs" GP analogue: per-level gp_basis sharing ONE λ/ρ + an RE pooling
#   penalty (partial pooling). Distinct from by_factor_smooth (independent λ).

def hgp_fit(Y, x, *, group, kernel='matern52', parametric=None, …) -> GPResult:
#   thin wrapper: population GP mean + per-group GP deviations (shared θ),
#   fit by the existing penalised-REML / PQL machinery.
```

Nested two-level HGP (`(gp | g1/g2)`) maps onto the shipped nested-LME
structure (`lme/_nested.py`) when the per-level kernels share an eigenbasis;
otherwise it is an additive stack of `gp_factor_smooth` blocks (gam_fit already
sums penalties). No new optimiser — reuses the v3 saddle-free damped Newton.

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

## 7. Engineering plan & phasing

| Phase | Deliverable | Files | Validation anchor | Effort |
|---|---|---|---|---|
| **P1** | Kernels: Matérn-5/2, exponential, periodic; ARD distance | `linalg/kernel.py` (+inline lift from `basis.py:716`) | `sklearn`/`GPy` kernel-matrix parity to ~1e-12 | XS–S |
| **P2** | Tier 1: `gp_basis(kernel=…)`, `gp_basis_nd`, `gp_smooth` ρ-selection | `stats/basis.py` | mgcv `s(x, bs="gp")` partial effect; reduced-rank → full-GP posterior mean (`sklearn`) to ~1e-3 as `k→n` | S–M |
| **P3** | Tier 2: `gp_fit`/`GPResult`/`gp_predict` (reduced-rank, shared-ρ) | **new** `stats/gp.py` + 2 lines `stats/__init__.py` | `log_mlik` & θ vs `sklearn`/`GPy` (small-N full-rank) to ~1e-5; finite-diff grad-through-θ; HLO-budget test | M |
| **P4** | Tier 2b: full-rank shared-kernel via `reml_fit` (`K`↔`ZZᵀ`) | `stats/gp.py` (thin) | equals dense Cholesky GP to ~1e-7; agrees with P3 reduced-rank as `k→n` | S |
| **P5** | Tier 3: `gp_factor_smooth` + `hgp_fit` (factor-smooth / `fs`) | `stats/basis.py`, `stats/gp.py` | mgcv `s(x, g, bs="fs")`; per-group curve recovery corr > 0.97 (cf. v3 §3.1) | M |
| **P6** | Nested HGP `(gp | g1/g2)` | `stats/gp.py` (reuses `lme/_nested.py`) | dense REML reference ~1e-5 | M |
| **P7** *(opt.)* | perf-bench case | `nitrix-perf-bench/.../cases/gp_fit.py` | vs `sklearn`-looped; contract test on `op_qualname` | S |

Person-time, rough: Tier 1 (P1–P2) ~2 days; Tier 2 (P3–P4) ~1 week; Tier 3
(P5–P6) ~1.5–2 weeks. **Full (a)-scope ≈ 3–4 weeks** incl. validation.

**Suite invariants every phase must hold** (the audit's standing bar):
cuSOLVER-free (route per-element solves through `spd_inv_logdet_chol` /
`sym_eig_jacobi`; one-off knot-Gram `eigh` on host, as `gp_basis` already
does); jit/`vmap`/grad-clean (every array field a `register_result` child, θ
hashable-free); `O(V·N·k)` memory with an HLO-budget test; mass-univariate
`(Y:(V,N), X:(N,p))` keyword-only signature; ruff/mypy clean.

## 8. Touch-point checklist (blast radius)

**Create:** `stats/gp.py` (~250–450 LOC); `tests/test_gp.py` (~150–250 LOC);
optional `nitrix-perf-bench/.../cases/gp_fit.py` + its contract test.
**Modify:** `stats/basis.py` (extend `gp_basis`, add `gp_basis_nd` /
`gp_smooth` / `gp_factor_smooth`); `linalg/kernel.py` (+3 kernels);
`stats/__init__.py` (import + `__all__`, 2–4 lines); `tests/test_basis.py`
(+kernel cases); 1–2 lines in the `stats/__init__.py` module docstring.
**Not required:** no `pyproject.toml` entry points; no top-level `nitrix`
re-export (`nitrix/__init__.py` is docstring-only); no result registry beyond
the `@register_result` decorator. **Existing code touched: effectively nil** —
every change is additive.

## 9. Risks & open decisions

- **R1 (decided, §3).** `ρ` shared-across-voxels by default (D1a grid/profile);
  per-voxel `ρ` and differentiable-`ρ` (D1b) are escape hatches.
- **R2 (decided, §4).** Reduced-rank is the default engine; full-rank dense GP
  only for small-`N` shared-kernel (Tier 2b). Add the HLO-budget test.
- **R3 (open).** Multi-D scalability: Hilbert-space reduced-rank GP
  (Solin–Särkkä) vs tensor-product of 1-D `gp` margins (already composable via
  `tensor_product_basis`). Recommend tensor-product first (zero new math),
  Hilbert-space as a follow-up for true ARD.
- **R4 (open).** Periodic-kernel reduced-rank construction (eigenbasis of a
  periodic Gram) needs a numerical check at small `ρ`.
- **R5 (note).** Inference semantics already covered: reduced-rank GP terms
  inherit `smooth_significance` + credible bands; `GPResult.log_mlik` drives
  `compare_models`. No new inference surface required for (a)-scope.

## 10. Cross-references

- `src/nitrix/stats/basis.py:724` (`gp_basis`), `:716` (`_matern32_kernel`),
  `:783–797` (eigen-reparam → identity penalty), `:767` (fixed `ρ`).
- `src/nitrix/stats/gam.py:393–426` — the generic Fellner–Schall penalty loop.
- `src/nitrix/stats/lme/reml.py` — FaST-LMM `reml_fit` (the `K`↔`ZZᵀ` route);
  [`docs/design/lme.md`](../design/lme.md).
- `src/nitrix/stats/_result.py:37` — `register_result` (the `GPResult` pattern).
- `src/nitrix/linalg/{kernel.py:247,262, solve.py:62, _smalllinalg.py:93,`
  `_eigsolve.py:829, krylov.py:38, decompose.py:70}` — GP linear-algebra prims.
- [`stats-modelling-suite-v3.md`](stats-modelling-suite-v3.md) §2 (GAMM
  surface: `re_smooth`/`REBasis`), §3.1 (`by_factor_smooth`) — the Tier-3
  substrate; the shared "penalised GLM ≡ variance-components REML ≡ mixed
  model" framing.
- [`stats-suite-audit.md`](stats-suite-audit.md) — standing-suite bar.
- Deferred sibling (b): HBR / normative-modelling GP — to be filed separately.
