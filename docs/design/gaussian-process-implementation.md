# GP / HGP implementation plan — `nitrix.stats` (HSGP-primary)

> Forward-looking engineering plan for the GP feature on `feat/stats-gp`. The
> *what/why* lives in `docs/feature-requests/gaussian-process-models.md`; this is
> the *how*, grounded in the as-built code (file:line). Status: **plan — no
> `src/` code written yet.**

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
| **PR3** | Tier 2b full-rank shared-kernel via `reml_fit`; `corr=` composition | `lme.reml_fit`, `gls_fit` | thin |
| **PR4** | Tier 3 `gp_factor_smooth` + `hgp_fit`; nested HGP | `re_smooth`/`by_factor_smooth`/`gam_fit`, `lme/_nested.py` | |
| **PR5** | multi-D `hsgp_basis_nd` (tensor product); perf-bench | | |

This document specifies **PR1 fully** and PR2 to the design level; PR3–5 are
sketched (they don't constrain PR1/PR2).

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

## 5. PR2 design sketch — `stats/gp.py` (no implementation yet)

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
reduction), `(σ_f², σ_e²)` per voxel. `log_mlik` → shipped
`aic`/`bic`/`compare_models`. **Invariants:** no runtime `eigh` (Φ closed-form),
`O(V·N·m)` memory + an HLO-budget test mirroring `test_reml_max_tensor_size…`,
ruff/mypy clean. Validation: `(σ_f, ρ, σ_e)` & `log_mlik` vs sklearn/GPy/brms
~1e-4; finite-diff grad-through-`ρ`.

## 6. Decisions (confirmed 2026-06-21)

1. **PR1 scope — fixed-`ρ` `hsgp_basis` only.** Spectral densities + the
   fixed/default-`ρ` basis that rides `gam_fit`; `ρ`-selection deferred to PR2.
2. **`ρ`-estimation home — dedicated `gp_fit` profile-REML.** Keep `gam_fit`'s
   hot path untouched and isolate the new `(σ_f², σ_e², ρ)` optimiser.
   **End-of-PR2 review (committed):** assess whether the diagonal-`S(ρ)` step can
   be folded into a *streamlined, generalised* Fellner–Schall (so any GAM smooth
   gains `ρ`) **with no performance regression** on the existing GAM fast paths;
   migrate only if that holds.
3. **Periodic kernel — deferred.** Standard HSGP covers Matérn/SE; the periodic
   basis is a separate construction (own follow-up).
4. **Spectral-density home — `linalg/kernel.py`** (per the proposal).

## 7. Effort

PR1 ≈ ½–1 day (incl. tests). PR2 ≈ 1 week. PR3–5 ≈ 2–3 weeks. Full (a)-scope
≈ 3–4 weeks, matching the feature-request §7 estimate.
