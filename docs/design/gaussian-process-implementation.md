# GP / HGP implementation plan — `nitrix.stats` (HSGP-primary)

> Forward-looking engineering plan for the GP feature on `feat/stats-gp`. The
> *what/why* lives in `docs/feature-requests/gaussian-process-models.md`; this is
> the *how*, grounded in the as-built code (file:line). Status: **PR1 + PR2
> shipped** (`linalg/kernel.py` spectral densities, `stats/basis.py` `hsgp_basis`,
> `stats/gp.py` `gp_fit`/`gp_predict`/`GPResult`; tests `test_hsgp.py`,
> `test_gp.py`, `test_gp_mgcv_parity.py`). PR3–5 remain plan-only.

## 0. Principle & ordering

HSGP is primary (the feature request §3 justification): a **fixed** Laplace-
eigenfunction design `Φ` whose hyperparameters enter only as a **diagonal**
spectral reweighting, so `ρ`-estimation is `eigh`-free and rides the suite's
fixed-eigenbasis fast paths. The build order follows the dependency chain, each
PR independently shippable and gam_fit-compatible:

| PR | Scope | Rides | New code |
|----|-------|-------|----------|
| **PR1** | Spectral densities + fixed-`ρ` `hsgp_basis` (1-D) | `gam_fit` **unchanged** | `linalg/kernel.py`, `stats/basis.py`, test |
| **PR2** | `gp_fit`/`GPResult`/`gp_predict` — HSGP, shared-`ρ` diagonal-`S(ρ)` REML, optional MAP-`ρ` | new `stats/gp.py` | + HLO-budget test |
| **PR3a** | Tier 2b full-rank `engine='exact'` (kernel-eigenfeature REML) | shared PR2 core (≡ `lme.reml_fit`) | thin |
| **PR3b** | `corr=` composition (structured residual: ar1/car1/cs) | `lme._corr.whiten`, `build_group_layout` | thin |
| **PR4a** | Tier 3 hierarchical `hgp_fit` (global + group smooths, GS model) | shared PR2 core, generalised to K penalty blocks | `stats/hgp.py` |
| **PR4b** | `gp_factor_smooth` fixed-`ρ` factor-smooth GP basis (gam_fit drop-in) | `gam_fit`, `hsgp_basis` | `basis.py` |
| **PR5** | multi-D `hsgp_basis_nd` (tensor-product, isotropic + ARD); `dim`-general spectral densities | `gam_fit`, `linalg/kernel.py` | `basis.py`, `kernel.py` |

This document specifies **PR1 fully** and PR2 to the design level; PR3–5 are
sketched (they don't constrain PR1/PR2). **PR2, PR3a/b, PR4a/b, PR5, PR6 are
shipped** (`gp_fit` HSGP + `engine='exact'` + `corr=`; `hgp_fit` hierarchical;
`gp_factor_smooth`; `hsgp_basis_nd` multi-D; lengthscale-prior regularisers). The
full **(a)** scope is complete.

## 1. Math spec (1-D HSGP)

**Domain / centering.** `c = (min x + max x)/2`; `S = max|x − c|`; `L = boundary·S`
(`boundary ≥ 1`, default `1.5`). Domain assumed `[c−L, c+L]`.

**Dirichlet-Laplacian eigenpairs on `[−L, L]`** (shifted by `c`), `j = 1..m`:
```
√λ_j = j·π / (2L)
φ_j(x) = √(1/L) · sin( √λ_j · (x − c + L) )
```

**Spectral densities `S_θ(ω)`** (1-D, lengthscale `ℓ = ρ`, amplitude `α²`;
matched to scikit-learn's `RBF`/`Matern` parameterisation). Closed forms for the
supported set (no special functions needed):
```
SE / RBF      :  S(ω) = α² · √(2π)·ℓ · exp(−½ ℓ²ω²)
Matérn ν=1/2  :  λ=1/ℓ      ;  S(ω) = α² · 2λ   / (λ²+ω²)
Matérn ν=3/2  :  λ=√3/ℓ     ;  S(ω) = α² · 4λ³  / (λ²+ω²)²
Matérn ν=5/2  :  λ=√5/ℓ     ;  S(ω) = α² · (16/3)λ⁵ / (λ²+ω²)³
```
(General-ν form via `gammaln` is a later add; not needed for {½,3⁄2,5⁄2,SE}.)

**Whitening → penalised-basis form.** Weights `s_j = S_θ(√λ_j)` with `α = 1`
(amplitude is carried by the smoothing parameter, below). Whitened column
`ψ_j(x) = √(s_j)·φ_j(x)`; design `Ψ = [ψ_j(x_i)]` (n×m); **penalty `= I_m`**.
Optional sum-to-zero constraint via `_householder_null` of the column sums
(identical to `gp_basis`, basis.py:799–804).

**GP equivalence.** `f ~ N(0, ΨΨᵀ) → α²K_θ` as `m→∞`. Under `gam_fit` the
penalty is `λ·I` with `λ = 1/σ_f²` ⇒ `β ~ N(0, σ_f²)` ⇒ `f ~ N(0, σ_f²ΨΨᵀ) ≈
σ_f²K_θ`. So **the FS-estimated smoothing parameter is the GP amplitude**, and a
fixed-`ρ` HSGP smooth needs *no* `gam_fit` change.

**`ρ`-estimation (PR2).** For variable `ρ`, only `diag(√s(ρ))` changes; `Φ` is
fixed. Equivalent placements: (A) put `ρ` in the design `Ψ(ρ)=Φ·diag(√s(ρ))`,
penalty `I`; (B) keep design `Φ`, penalty `diag(1/s_j(ρ))`. Form (B) is a fixed
design + a `ρ`-dependent **diagonal** penalty — the REML criterion is smooth in
`ρ` with no `eigh`. `gp_fit` profiles `β | (ρ,σ_f²,σ_e²)` and optimises
`(log σ_f², log σ_e², log ρ)` (1–3-D Newton / autodiff). **Shared `ρ` across
voxels** (one `Φ`), per-voxel `(σ_f², σ_e²)`. Optional MAP term: add a
prior-as-penalty on `log ρ` (half-normal / inverse-gamma) to the objective.

## 2. Data-structure design (no pytree change)

Reuse `SplineBasis` (basis.py:154) with `kind='hsgp'`, packing HSGP params into
existing fields — **no new dataclass fields, no `tree_flatten` edit**:

| Field | HSGP meaning |
|-------|--------------|
| `design` | `Ψ` post-constraint (n×k) |
| `penalty` | `I` post-constraint (k×k) |
| `kind` | `'hsgp'` |
| `constraint` | sum-to-zero `Z` or `None` |
| `knots` | `√λ_j` frequencies (m,) |
| `radial_transform` | `(m, 2)` = `[ √s_j , phase_j ]`, `phase_j = √λ_j·(L−c)` |
| `kernel_param` | `L` (for the `√(1/L)` factor) |
| `lo, hi` | data min/max (record) |
| `n_basis` | `m`; `degree=0`, `penalty_order=2` (unused) |

Re-evaluation — new `'hsgp'` branch in `_raw_features` (basis.py:335, mirrors the
`'gp'` branch at :358), using the stored phase so `c` need not be a field:
```python
if basis.kind == 'hsgp':
    u   = basis.knots                       # √λ            (m,)
    w   = basis.radial_transform[:, 0]      # √s            (m,)
    ph  = basis.radial_transform[:, 1]      # √λ·(L−c)      (m,)
    L   = basis.kernel_param
    phi = jnp.sqrt(1.0 / L) * jnp.sin(u[None, :] * x[:, None] + ph[None, :])
    return phi * w[None, :]                  # pre-constraint Ψ
```
`spline_design` then applies `constraint` (basis.py:380–383) exactly as today.
`gam_fit` consumes `.design`/`.penalty_blocks()` generically (confirmed: **no
`.kind` switch in gam.py**), so the `'hsgp'` basis is a drop-in smooth.

## 3. File-by-file (PR1)

**`src/nitrix/linalg/kernel.py`** — add + export:
```python
def matern_spectral_density(omega, *, rho, nu, amplitude=1.0): ...   # ν∈{0.5,1.5,2.5}
def se_spectral_density(omega, *, rho, amplitude=1.0): ...
def spectral_density(omega, *, kernel, rho, amplitude=1.0): ...      # str dispatcher
```
Pure JAX, elementwise on `omega`; closed forms from §1. `__all__` += the three.

**`src/nitrix/stats/basis.py`** — add:
```python
def hsgp_basis(x, n_basis=20, *, kernel='matern52', rho=None, amplitude=1.0,
               boundary=1.5, center=True) -> SplineBasis: ...   # kind='hsgp'
def _hsgp_eigen(x, n_basis, boundary): ...   # → (c, L, √λ, Φ-evaluator)
```
`rho=None ⇒ ρ = S = (hi−lo)/2` (a sane fixed default; estimation is PR2). Add the
`'hsgp'` branch to `_raw_features` (§2). Mirror `gp_basis`'s host-side
construction and constraint handling (basis.py:762–819).

**`src/nitrix/stats/__init__.py`** — `from .basis import (... hsgp_basis)`;
add `'hsgp_basis'` to `__all__` (next to `gp_basis`, line ~183); add one line to
the `basis` bullet in the module docstring.

**`tests/test_hsgp.py`** (new) — see §4.

## 4. Validation plan (PR1) — anchored, cuSOLVER-free, x64

`tests/test_hsgp.py`, `jax.config.update('jax_enable_x64', True)`:

1. **Spectral density vs analytic** — `S_θ(ω)` matches the §1 closed forms at
   sample `ω`; and **inverse-FT round-trip**: `k(r) = (1/π)∫₀^∞ S(ω)cos(ωr)dω`
   (numeric quad) matches `sklearn.gaussian_process.kernels.{Matern,RBF}`
   `k(r)` to ~1e-6 for ν∈{½,3⁄2,5⁄2}, SE. (Confirms the parameterisation lines
   up with the reference kernels.)
2. **Basis contract** — `hsgp_basis` returns identity penalty (pre-constraint),
   correct shapes `(n, k)`, finite design; `spline_design(basis, x_train)`
   reproduces `basis.design` to ~1e-10 (re-eval round-trip).
3. **HSGP → exact GP convergence** — fixed `(ρ, σ_f², σ_e²)`: posterior mean
   `m_hsgp(m) = Ψ(Ψᵀ Ψ + σ_e²I)⁻¹ Ψᵀ y` (with `α=σ_f`) converges to the dense
   `m_dense = K(K+σ_e²I)⁻¹y`, `K=σ_f²·Matern(ρ,ν)`: assert error **decreases**
   with `m` and `< 1e-2` at `m≈40` on a smooth target over a bounded domain.
4. **scikit-learn parity** — same setup vs
   `GaussianProcessRegressor(C(σ_f²)*Matern(ρ,ν)+White(σ_e²), optimizer=None)`
   predictive mean to ~1e-2 at `m` large.
5. **`gam_fit` integration** — noisy `sin`/`gp`-draw target: fitted smooth
   correlates `> 0.97` with truth; `edf` sane; runs under `jax.jit`.
6. **jit/vmap** — `gam_fit` over `V` voxels (shared design) agrees with the
   per-voxel loop to ~1e-6.

Reference libs confirmed present in the shared venv: numpy 2.4, scipy 1.17,
scikit-learn 1.9 (`gaussian_process`), jax 0.10.

**R parity (env surveyed 2026-06-21).** R 4.5.3 + `mgcv` / `nlme` at
`/scratch/nperf/renv/bin/Rscript` (**no `brms` / Stan**). mgcv
`s(x, bs="gp", m=c(3, rho))` is Matérn-3/2 kriging — the *same construction* as
`gp_basis` — so it anchors the **kriging parity basis exactly**; for the HSGP
basis it is only a *secondary* cross-check (different basis, same Matérn-GP
target, ~1e-2 on smooth data). The ideal HSGP-to-HSGP anchor (`brms::gp()`,
itself HSGP) needs Stan and is unavailable; the sklearn exact-GP anchor (above)
is the stronger HSGP correctness reference regardless. R tests guard on
`Rscript` availability (skip when absent). Placement: mgcv↔`gp_basis` parity is a
quick win; the mgcv↔`gp_fit` REML-range cross-check is natural in PR2.

## 5. PR2 — `stats/gp.py` (**shipped**)

```python
@register_result(children=('coef','cov_unscaled','theta','log_mlik','edf','dispersion'),
                 aux=('kernel','n_obs','rank'))
@dataclass(frozen=True)
class GPResult: ...        # fields per feature-request §5 Tier 2

def gp_fit(Y, x, *, parametric=None, kernel='matern52', rank=20,
           engine='hsgp', select='shared-rho', map_rho=None, corr=None,
           n_iter=..., ridge=1e-8, block=None) -> GPResult: ...
def gp_predict(result, basis, x_new) -> tuple[mean, var]: ...
```
Engine: build one fixed `Φ` (shared under `vmap`); per voxel, profile-REML over
`(log σ_f², log σ_e², log ρ)` with the diagonal-`S(ρ)` penalty (§1, form B);
`ρ` shared across voxels (estimate on pooled criterion or a representative
reduction), `(σ_f², σ_e²)` per voxel.

**As built (deltas from the sketch):**
- **Form chosen — fixed design, diagonal penalty (a tightened form B).** `gp_fit`
  uses `center=False` for the internal smooth, so the design `X = [1 | parametric |
  Φ]` is *exactly* `ρ`-independent (no sum-to-zero `Z` to recompute per `ρ`) and the
  penalty core is the pure diagonal `diag(1/s_j(ρ))` — the smooth/intercept
  confounding is handled by the explicit intercept + the GP shrinkage (the brms
  `gp()` convention), not a constraint. The Fellner–Schall trace collapses to the
  disjoint-penalty shortcut `tr(S_λ⁺S)=m/λ`.
- **`ρ` search — pooled-REML grid + parabolic refine.** A fixed log-spaced
  `log ρ` grid (`n_rho`, default 24) over the pooled `−2 l_R = Σ_v (n−M₀)log D_{p,v}
  + V(log|H| − log|S_λ|₊)`; each grid point runs the shared diagonal-penalty FS for
  `λ`, then a 3-point parabola sub-grid-refines the argmin. Host-driven outer loop
  (a host `argmin` separates the two device passes), so `gp_fit` is *not* one
  `jit` — the heavy per-element work (the `ρ`-search `vmap` and the final fit) is.
- **`gp_predict(result, x_new, *, parametric=None)`** — no `basis` argument: because
  the eigenbasis is `ρ`-independent and uncentred, `Φ(x_new)` is reconstructed from
  the recorded `(lo, hi, boundary, rank)` aux, so the result is self-contained.
- **`GPResult` aux** = `(kernel, engine, n_obs, rank, n_fixed, lo, hi, boundary)`
  (the domain descriptors feed `gp_predict`'s self-contained HSGP reconstruction);
  `theta` is `(V,3) = [log σ_f², log σ_e², log ρ]` (the `ρ` column is constant).
- **`map_rho`** is an optional `ρ→penalty` callable added to the pooled objective
  (MAP/prior-regularised lengthscale); `corr=` and `select='per-voxel'` raise
  `NotImplementedError` (PR3b / later).

## 5a. PR3a — `engine='exact'` (**shipped**)

The full-rank GP shares the **entire** PR2 penalised-REML core (`_gp_fit_one`,
`_quantities`, `_reml_nll`, the pooled-`ρ` grid + parabolic refine, `_assemble_gp_result`);
it differs only in the smooth design:

- **Design = kernel eigenfeatures** `Φ(ρ) = U_k diag(√λ_k)` from a host
  `numpy.linalg.eigh` of the kernel Gram `K_ρ` (closed-form Matérn-½/3⁄2/5⁄2 + RBF,
  matched to the same sklearn lengthscale convention as the spectral densities).
  `ΦΦᵀ` is the rank-`k` truncation of `K_ρ` — **exact when `rank=N`** (default for
  `engine='exact'`), the Karhunen-Loève / Nyström approximation for `rank<N`.
- **Unlike HSGP, `Φ` *moves* with `ρ`** (the eigenbasis is kernel-dependent), so the
  penalty is the plain identity (unit spectral weights), the cross-products are
  rebuilt per `ρ`, and the `ρ`-search is a **host loop** (one shared `eigh` per grid
  `ρ`, data-independent of `Y` ⇒ cuSOLVER-free; no `eigh` *in the jitted region*).
- **`gp_predict`** adds an `x_train` argument (required for `'exact'`): the
  out-of-sample features are the Nyström map `K(x*, x) U_k diag(1/√λ_k)`. HSGP stays
  self-contained (no `x_train`).
- **Equivalence to `lme.reml_fit`** (the "via `reml_fit`" intent): verified to
  **machine precision** — at fixed `ρ`, `(σ_f², σ_e², β)` reproduce
  `reml_fit(Y, X, Z=chol(K_ρ))` to `<2e-3` (`test_exact_matches_reml_fit_at_fixed_rho`).
  The exact engine *is* FaST-LMM variance-components REML, computed through the
  penalty form — so we reuse the one penalised-REML path rather than calling
  `reml_fit` (no whitening-space round-trip, uniform `GPResult`/`gp_predict`).
- **Validation:** exact-vs-exact sklearn GPR anchor (lengthscale + predictive mean,
  tighter than HSGP's); each kernel; full-rank vs KL-truncated; `x_train` guard.
- **Memory:** the exact engine is inherently `O(V·N²)` (`cov_unscaled` is `(V, p, p)`
  with `p ≈ N`) — the accepted exact-GP cost; the HLO-budget invariant is HSGP-only.

## 5b. PR3b — `corr=` structured residual (**shipped**)

Composes the GP smooth with a within-group correlated residual
`Cov(ε) = σ_e² R(ρ_c)` (`nlme`-style `ar1` / `car1` / `cs`), the
longitudinal-fMRI case (a smooth population trend over the covariate, autocorrelated
residuals within subject). Rides the **existing** `lme._corr` whitening verbatim:

- **Whitening reduction.** `W(ρ_c) R W^T = I` per group (the innovations form for
  `ar1`/`car1`, the rank-one transform for `cs`), so on whitened `(ỹ, X̃)` the
  residual is i.i.d. and the model is the PR2 penalised regression — the criterion
  is the shared profiled REML **plus the whitening Jacobian** `log|R(ρ_c)|`
  (`= 2·half_logdet`, returned by `CorrSpec.whiten`). Verified against a dense
  block-`R` marginal-likelihood reference to a constant offset
  (`test_corr_reml_matches_dense_up_to_constant`, `<1e-6`).
- **Joint `(ρ_GP, ρ_c)` grid.** `ρ_c` enters via the structure's unconstrained
  `raw_c` (gridded over `corr_raw_bounds`, default `(-2.5, 2.5)` ≈ `|ρ_c|<0.99`);
  the lengthscale axis is parabolically refined at the winning `raw_c`. Both engines
  compose (the design/penalty are built by `_hsgp_design_pen` / `_exact_design_pen`
  closures; whitening is engine-agnostic).
- **One compiled program.** All `Y` voxels are whitened at once by carrying the mass
  axis as the whitener's channel dim (`(G,T,V)`); the per-cell whitened
  cross-products + pooled REML are **`jit`-compiled once** and reused across every
  grid cell (the moving design / penalty / `raw_c` are traced args) — a naive
  per-cell `vmap` recompiled `O(n_rho·n_corr)` programs and exhausted the compiler
  (fixed; the same one-compile pattern was applied to the exact grid).
- **Output.** `GPResult` gains `corr_rho` (`(V,)`, the natural residual correlation;
  `0` for `iid`) and `corr` (aux). The fit's posterior is in whitened space, so the
  **latent** GP mean/variance — hence `gp_predict` — is unchanged (the residual
  structure is a nuisance on `ε`, not on `f`). `corr='iid'` reproduces `corr=None`
  exactly (`test_corr_iid_matches_no_corr`).

**Invariants (verified):** no runtime `eigh` (Φ closed-form), `O(V·(m+q)²)` working
memory, an HLO-budget test (`test_gp_final_fit_hlo_is_cusolver_free_and_N_free`)
asserting the final fit is cuSOLVER-free and carries **no `N`-sized tensor**,
ruff/mypy clean. **Validation:** the `p`-space profiled REML matches a dense
`(N,N)` marginal-likelihood reference to a constant offset (`< 1e-6` spread across
`(λ,ρ)`); `ρ̂` + predictive mean track sklearn exact GPR; mgcv cross-check in
`test_gp_mgcv_parity.py`. (brms/Stan absent — see §4 — so the sklearn exact-GP
anchor stands in for the HSGP-to-HSGP comparison.)

## 5c. PR4a — `hgp_fit` hierarchical GP (**shipped**, `stats/hgp.py`)

The multi-level / "(a)"-scope GP: a population smooth plus group-level smooth
*deviations* that share the kernel — the GP analogue of a random-slope mixed
model, and the "GS" hierarchical GAM (Pedersen et al. 2019). Partial pooling: a
sparse group is shrunk toward the population trend.

- **Model.** `y = β0 + f_pop(x) + Σ_g 1[g] f_g(x) + ε`, `f_pop ~ GP(0,σ²_pop K_ρ)`,
  `f_g ~ iid GP(0,σ²_grp K_ρ)`. Two GP variance components + noise + a single
  shared `ρ`.
- **Construction = two diagonal penalty blocks.** `X = [1 | Φ(x) | Φ(x)⊗onehot(g)]`
  (the factor-smooth interaction: group `g`'s columns are `Φ` on its own rows); the
  penalty is `blkdiag(λ_pop diag(1/s), λ_grp diag(1/s)⊗I_L)` — **fully diagonal,
  disjoint blocks**, so the FS trace stays `rank_k/λ_k` and the REML
  log-pseudo-determinant is a per-block sum (no `eigh`). `λ_pop`, `λ_grp` are the
  inverse GP amplitudes; the shared `ρ` is profiled by the same pooled-REML grid.
- **Generalised core.** `gp.py`'s single-block diagonal REML is generalised to **K
  blocks** in `hgp.py` (`_mb_quantities` / `_mb_fs` / `_mb_reml_nll`): the penalty
  is `Σ_k λ_k diag(d_blocks[k])`, FS updates each `λ_k`. (Kept in `hgp.py` so the
  PR2/PR3 single-block paths stay untouched; a future refactor can unify.)
- **`hgp_predict`** returns the population curve (`levels=None`) or per-group curves
  (`levels=…`, population + that group's deviation), reconstructing `Φ(x_new)` from
  the recorded domain (HSGP, `ρ`-independent — self-contained).
- **Validation:** the 2-block p-space REML matches a **dense** hierarchical
  marginal-likelihood reference (`M = I + σ²_pop·…+σ²_grp·…`) to a constant offset
  (`<1e-6` across `(λ_pop,λ_grp)`); recovers population + per-group curves (group
  curves track their own data > the bare population curve); `σ²_grp` collapses when
  groups don't truly differ (partial pooling). 6 tests; ruff/mypy clean.
- **Cost.** The factor-smooth is `(1+L)` smooths wide, so working size is
  `O(V·(M_0+(1+L)m)²)` — the inherent per-group-curve cost; bound with `block`.
  (Exploiting the disjoint-row block sparsity of the group design is a later
  optimisation.)

## 5d. PR4b — `gp_factor_smooth` fixed-`ρ` basis (**shipped**, `basis.py`)

The basis counterpart of `hgp_fit`: the factor-smooth GP block (mgcv `bs="fs"`
with a GP marginal) as a `gam_fit` drop-in — `hsgp_basis : gp_fit ::
gp_factor_smooth : hgp_fit`. It replicates the **whitened** HSGP design `Ψ` (form
A, fixed `ρ`) per group with an **identity** penalty (`_FactorGPBasis`, a
`SmoothBasis` like `REBasis`), so the single Fellner-Schall parameter on the block
is the shared group precision `1/σ²_grp`. Drop it into `gam_fit` next to a
population `hsgp_basis` of the same `ρ` for the GS model at fixed `ρ`, composable
with any other GAM term. `eval_design` takes a `(x, group)` tuple (the by-factor
convention). Verified: contract (width `L·m`, one identity block, tuple
round-trip), GS recovery via `gam_fit` (fitted vs truth corr `>0.99`, one shared
group `λ`), stable width under `n_levels`. 3 tests; no new mypy errors (the
`penalty` host-array follows the tolerated `REBasis` pattern).

## 5e. PR5 — `hsgp_basis_nd` multi-dimensional HSGP (**shipped**, `basis.py` + `kernel.py`)

A `D`-dimensional GP smooth (spatial smooth, or a smooth interaction of `D`
continuous covariates) — the tensor-product HSGP.

- **`dim`-general spectral densities** (`kernel.py`). `se_` / `matern_` /
  `spectral_density` gain a `dim` argument: the SE form `(2π)^{D/2} ρ^D
  exp(-½ρ²‖ω‖²)` and the Matérn form `C·(λ²+‖ω‖²)^{-(ν+D/2)}` with
  `C = 2^D π^{D/2} Γ(ν+D/2)(2ν)^ν / (Γ(ν)ρ^{2ν})` (via `gammaln`). The `dim=1`
  Matérn path is the **byte-identical** closed form (kept in a branch) so the
  heavily-tested 1-D code and its sklearn inverse-FT anchor are untouched; tests
  confirm the gammaln normaliser reproduces the closed forms at `D=1`.
- **`hsgp_basis_nd(X, n_basis, *, kernel, rho, …)`** (`basis.py`). `X` is `(n, D)`;
  the tensor-product Laplace eigenfunctions `∏_d φ_{j_d}(x_d)` (eigenvalue
  `Σ_d λ_{j_d}`) over the cartesian mode grid (`M = ∏_d m_d`), whitened by the
  spectral weight of the mode-frequency magnitude `‖ω‖=√(Σ_d λ_{j_d})`. **Isotropic**
  (scalar `rho` → the `D`-dim radial density) or **separable / ARD** (per-dim `rho`
  → product of 1-D densities). Identity penalty, optional sum-to-zero — a `gam_fit`
  drop-in (`_HSGPndBasis`, `eval_design` takes new `(g, D)` points).
- **Validation:** the `dim`-general densities reduce to the 1-D closed forms;
  2-D recovery via `gam_fit` (isotropic and ARD, fitted vs truth corr `>0.98`);
  **exact 2-D sklearn `GaussianProcessRegressor` parity** (predictive surface corr
  `>0.97`); mass-univariate over voxels; a 3-D smoke (`M=4³`); arg validation.
  8 tests; ruff/mypy clean (no new errors).
- **Cost / "perf-bench".** `M = ∏_d m_d` grows fast with `D` (keep `m` modest for
  `D≥2`); the design is `(n, M)` and `gam_fit` is `O(V·M²)` per voxel,
  cuSOLVER-free. Lengthscale *estimation* for the multi-D basis (an isotropic/ARD
  `gp_fit` extension) is the natural follow-up; the fixed-`ρ` basis already covers
  the headline spatial-smooth use case via `gam_fit`.

## 5f. PR6 — lengthscale-prior regularisers (**shipped**, `stats/priors.py`)

Closes the loop on the feature-request "Priors on `ρ`" nod: the curated set of
MAP-`ρ` penalties the `map_rho=` hook was built for. Each builder returns a
pure-JAX `ρ → −log p(ρ)` callable that drops into `gp_fit`/`hgp_fit`'s `map_rho=`
(the objective gains `2·map_rho` on the `−2 l_R` scale ⇒ MAP-`ρ`):

- `halfnormal_prior(sd)` — `ρ²/(2 sd²)`, caps **large** `ρ` (near-linear fits).
- `invgamma_prior(a, b)` — `(a+1)log ρ + b/ρ`, penalises **small** `ρ` (the
  PC-style regulariser stopping the lengthscale collapsing into noise-fitting).
- `lognormal_prior(mu, sd)` — `(log ρ−μ)²/(2 sd²) + log ρ`, centres on median
  `exp(μ)`.

Verified: the closed-form `−log p` values; the documented pull directions; jittable;
and that each shifts `ρ̂` the right way through `gp_fit` and `hgp_fit` (6 tests).
The `map_rho` mechanism itself was already threaded through every `gp_fit` path
(HSGP/exact/`corr`) and `hgp_fit` in PR2/PR3/PR4 — PR6 only adds the named library.
(Full *posterior* `ρ` priors remain scope (b) — Stan/`brms`.)

## 5g. PR7 — model selection (`gp_aic` / `gp_bic`; full REML `log_mlik`) (**shipped**)

Closes the one genuine Tier-2 oversight (the FR said `log_mlik` "feeds the shipped
`aic`/`bic`/`compare_models`", which it did not — those are GLM-specific).

- **`log_mlik` is now the *full* restricted log marginal likelihood.** The
  profiled-REML `−2 l_R` differs from the PR2 core by exactly the `(n, M_0)`
  constant `(n−M_0)(log 2π + 1 − log(n−M_0))` (measured: C₀ between the penalised
  and marginal forms is **0**). Adding it makes `log_mlik` match a dense REML
  reference *absolutely* (`<1e-5`), so AIC/BIC are valid even across models with
  different fixed-effect structure. It's constant in `(λ, ρ)`, so no `ρ̂` moves and
  the constant-offset anchor tests still hold (they check `ptp`, not the offset).
- **`gp_aic` / `gp_bic`** — `−2 l_R + 2k` / `−2 l_R + k log N` with `k` the effective
  dof (mgcv-style marginal IC). Duck-typed (a `Protocol`) over `GPResult` **and**
  `HGPResult` (no circular import); `HGPResult`'s `(V,2)` smooth-block edf gets the
  unpenalised `n_fixed` added back. Valid for same-fixed-effect GP-vs-GP /
  GP-vs-spline (kernel / rank) selection. 3 tests.

## 5h. PR8 — multi-D lengthscale estimation in `gp_fit` (**shipped**)

The FR-review follow-up: PR5 shipped the fixed-`ρ` `hsgp_basis_nd`; PR8 makes
`gp_fit` *estimate* the multi-D lengthscale (so spatial GPs need no supplied `ρ`).

- **`gp_fit(X, …)` accepts `(N, D)`** and routes to the tensor-product HSGP. The
  eigenbasis is `ρ`-independent, so the same pooled-REML core runs with the
  diagonal penalty as the only moving part — no new optimiser, just the penalty
  builder.
- **Isotropic** (`ard=False`, default): one shared `ρ` (the `D`-dim radial spectral
  density `S_D(‖ω‖)`), a 1-D grid as in 1-D `gp_fit`. **ARD** (`ard=True`): a
  per-axis `ρ_d` (separable density `∏_d S_1(ω_d;ρ_d)`) by **coordinate descent**
  over the axes (each axis a 1-D grid+parabola, a few cycles) — recovers the
  anisotropy (verified: longer `ρ` on the smooth axis than the wiggly one).
- **`GPResult` gains one aux field `nd_meta` = `(m_per, bounds, ard_rho)`** (a
  hashable tuple), so `gp_predict` rebuilds the tensor eigenbasis **self-contained**
  (no `x_train`); `theta[:,2]` carries the isotropic `ρ` or the ARD geometric mean.
- **Validation:** 2-D recovery (corr `>0.98`), ARD anisotropy direction, exact 2-D
  sklearn-GPR parity (`>0.97`), per-axis `rank`, predict, arg-validation
  (exact/`corr` reject multi-D). 4 tests; 1-D `gp_fit` suite unchanged; ruff/mypy
  clean. (ARD on a vector `ρ` is reported in `nd_meta`; a future per-axis MAP prior
  is the obvious extension.)

## 5i. PR9 — nested two-level HGP `(gp | g1/g2)` (**shipped**)

The FR P6 deliverable: a nested hierarchy on top of `hgp_fit`. Population +
**outer-group** + **inner-group (nested in outer)** GP deviations — three GP
variance components (`σ²_pop`, `σ²_outer`, `σ²_inner`) sharing `ρ`.

- **Falls straight out of the K-block core.** The hgp diagonal-penalty REML was
  already K-block (`_mb_*`); PR9 generalises the design/weights helper
  (`_block_weights(inv_s, n_fixed, level_counts)`) to any number of factor-smooth
  blocks. Nested = `level_counts = (L1, L2)` ⇒ K=3 blocks `[pop, Φ⊗oh(g1),
  Φ⊗oh(g2)]`; GS = `(L,)`. The fit, FS, REML are unchanged.
- **`hgp_fit(model='nested', group=g1, group_inner=g2)`.** `HGPResult` widens
  generically: `theta` is `(V, K+2)` (`(V,5)` nested), `edf` is `(V, K)`,
  `n_levels` is `(L1, L2)`. `gp_aic`/`gp_bic` work (the `_total_edf` sums the K
  blocks). `hgp_predict` returns the population curve (`levels=None`) for both
  models; per-group curves for the nested model are a documented follow-up.
- **Validation:** a dense **3-component** marginal-likelihood reference
  (`M = I + σ²_pop·…+σ²_outer·…+σ²_inner·…`) matches the p-space REML to a constant
  offset (`<1e-6`); the three variance components recover the simulated structure
  and the population trend (corr `>0.9`); `group_inner` is required. 3 tests; the
  GS suite + the renamed `_block_weights` reference unchanged; ruff/mypy clean.

## 6. Decisions (confirmed 2026-06-21)

1. **PR1 scope — fixed-`ρ` `hsgp_basis` only.** Spectral densities + the
   fixed/default-`ρ` basis that rides `gam_fit`; `ρ`-selection deferred to PR2.
2. **`ρ`-estimation home — dedicated `gp_fit` profile-REML.** Keep `gam_fit`'s
   hot path untouched and isolate the new `(σ_f², σ_e², ρ)` optimiser.
   **End-of-PR2 review — verdict: KEEP `gp_fit` separate; do *not* fold `ρ` into
   `gam_fit`'s Fellner–Schall.** As built, `gp_fit`'s `ρ`-search is a thin *outer*
   loop wrapping the **unmodified** diagonal-penalty FS (which still selects `λ`
   given a fixed penalty); `gam.py` is untouched and all reusable machinery
   (`small_inv_logdet`, the disjoint-penalty FS shortcut, the Gaussian
   cross-product fast path) is already shared. Folding `ρ` *into* the generic FS
   fails the no-regression bar for two reasons: (i) FS is a *multiplicative*
   fixed-point that scales a **fixed** penalty `S_k` by `λ_k` — but `ρ` *reshapes*
   the penalty itself (`diag(1/s(ρ))`), for which there is no closed-form
   multiplicative FS update, so it would need an interleaved Newton/grad-on-`ρ`
   step; (ii) that step plus a per-iteration penalty rebuild would land on **every**
   GAM smooth (including the `ρ`-free ones) — a real cost on the existing fast
   paths for no benefit to non-GP terms. The only genuinely GP-specific addition is
   the REML *value* `_reml_nll` (FS itself never computes a marginal likelihood);
   keeping it in `gp.py` is the right boundary. Net: the migration condition
   ("only if no perf regression") is not met → **no migration**.
3. **Periodic kernel — deferred.** Standard HSGP covers Matérn/SE; the periodic
   basis is a separate construction (own follow-up).
4. **Spectral-density home — `linalg/kernel.py`** (per the proposal).

## 7. Effort

PR1 ≈ ½–1 day (incl. tests). PR2 ≈ 1 week. PR3–5 ≈ 2–3 weeks. Full (a)-scope
≈ 3–4 weeks, matching the feature-request §7 estimate.
