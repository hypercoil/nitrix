# Design: non-Gaussian GP lengthscale estimation (CV2)

> **Status (2026-06-22): proposed.** Design note for review register item
> [`stats-suite-review-gp.md`](../feature-requests/stats-suite-review-gp.md) **CV2**
> (Round 2, effort **L**). Closes the gap that `gp_fit` estimates the kernel
> lengthscale `rho` only for Gaussian responses, so the feature request's own
> headline outcomes — **binary task activation**, **lesion / spike counts** — are
> reachable today only with `rho` *pinned* (`hsgp_basis(rho=fixed) + gam_fit`).
> Bring a decision on §4 (criterion) and §7 (scope) before coding.

---

## 1. The gap

`gp_fit` (and `hgp_fit`) profile `rho` by a **Gaussian** pooled-REML grid: at each
`rho` the diagonal-`S(rho)` HSGP penalty gives a closed-form profiled restricted
likelihood (`gp._reml_nll`, the `D_p`/`log|H|`/`log|S|_+` form). That closed form
exists only because the Gaussian likelihood is quadratic in `beta`. For a
Binomial or Poisson response there is no closed-form profile, so:

- `gp_fit` is documented Gaussian-only (`gp.py:771-772`).
- A non-Gaussian GP smooth is reachable only with a **fixed** `rho`
  (`hsgp_basis(rho=...) + gam_fit` runs the penalised IRLS but cannot *choose*
  `rho`).

`brms::gp()` and mgcv's `s(x, bs="gp")` estimate the lengthscale under any
exponential family. This note adds the same to `gp_fit` for the common
neuroimaging cases.

## 2. What already exists (substrate, ~80% there)

- **Penalised IRLS** — `stats/_irls.py::fit_penalised_irls` solves
  `(X^T W X + S) beta = X^T W z` over the working response `z` and working weights
  `W = (mu_eta^2 / var(mu)) * prior_w` (`_working`), as a `lax.scan`. `gam_fit`
  already drives this with a multi-block penalty + Fellner-Schall `lambda`.
- **The diagonal-`S(rho)` HSGP penalty** — `gp._penalty_diag(sqrt_lambda, kernel,
  rho, n_fixed)` gives the `(d, log_pdet)` the GP smooth needs; identical object
  whatever the family.
- **Families** — `_family.Family` exposes `link`/`mu_eta`/`variance`/
  `unit_deviance`/`loglik`/`has_fixed_dispersion`/`clip_eta`. `resolve_family`
  maps `'binomial'`/`'poisson'`/… .
- **The shared K-block penalised-REML core** — `stats/_penreml.py` (DS1). Reused
  for the *Gaussian working problem* inside each IRLS step (see §4).
- **glmm precedent** — `glmm_fit` already wraps PQL / Laplace around `gam_fit`'s
  IRLS for the marginal; CV2 is the same move, profiling `rho` instead of a
  variance component.

The **only** net-new piece is a marginal-likelihood **criterion in `rho`** for the
non-Gaussian case (§4) — Fellner-Schall selects `lambda` but never forms a
marginal, so it cannot profile `rho`.

## 3. The model

Per element, a GP smooth of `x` under an exponential-family response:

```
g(mu_i) = eta_i = beta0 + x_i^par beta_par + f(x_i),   f ~ GP(0, sigma_f^2 K_rho)
y_i ~ ExpFamily(mu_i, phi)
```

HSGP reduces `f` to the fixed eigenbasis `Phi` with the diagonal penalty
`S(rho) = lambda diag(1/s(rho))`, `lambda = phi / sigma_f^2`. So the per-element
fit at fixed `rho` is *exactly* a penalised GLM (`gam_fit` with `hsgp_basis`),
and the open problem is choosing `rho` (and `lambda`).

## 4. The criterion — PQL-REML / Laplace (the design decision)

At a fixed `rho`, run penalised IRLS to convergence → working response `z`,
working weights `W`, coefficients `beta`. The Laplace approximation makes the
working problem a **Gaussian** penalised regression of `z` on `X` with known
weights `W`, i.e. `z ~ N(X beta, W^{-1})` with prior `beta ~ N(0, sigma_f^2 ...)`.
Two criteria sit on top of it:

**(a) Working-model PQL-REML (recommended for Phase 1).** Apply the Gaussian
restricted likelihood to the *whitened* working problem `(sqrt(W) z, sqrt(W) X)`:
this is the same `_penreml` REML evaluated on the working cross-products
`c = X^T W z`, `g = z^T W z`, `xtx = X^T W X`. Profiling this over the `rho` grid
(parabolic refine, exactly as Gaussian) gives a PQL-REML `rho`. Cheap, reuses
`_penreml` verbatim, and mirrors `glmm_fit`'s PQL. **Caveat:** the working
weights depend on `beta(rho)`, so this is the standard **PQL attenuation** —
biased for binary / low-count data (document it as `glmm_fit` does).

**(b) Laplace REML / LAML (mgcv `method="REML"`, Phase 2).** The proper
Laplace-approximate restricted marginal adds the exact-likelihood
log-determinant and the penalty-derivative correction that PQL drops:

```
-2 l_LAML(rho) = -2 sum_i loglik(y_i, mu_i, phi) + beta^T S(rho) beta
                 + log|X^T W X + S(rho)| - log|S(rho)|_+ + correction(dW/d.eta)
```

This removes most of the PQL bias and matches mgcv, at the cost of the
`dW/d eta` derivative term and a more careful profiled scale. Recommend landing
(a) first (closes the headline gap), then (b) behind the same API.

**`lambda` within each `rho`.** Keep Fellner-Schall on the working problem (as
`gam_fit` already does) so `lambda = phi/sigma_f^2` is selected per `rho`; the
outer `rho` grid then profiles only the lengthscale. (Joint `(rho, lambda)` grid
is the fallback if the nested FS interacts badly.)

**Dispersion.** `has_fixed_dispersion=True` (Binomial / Poisson) → `phi = 1`
known, the working-REML uses it directly (no profiled scale — note this differs
from the Gaussian path, which profiles `phi`). Free-dispersion families
(Gamma / negbinomial) estimate `phi` from the working Pearson statistic — but see
the **MC4** lesson: get the marginal scale right or defer those families.

## 5. API

```python
gp_fit(Y, x, *, family='gaussian', ...)          # 'gaussian' -> current path, unchanged
gp_fit(Y, x, family='binomial', prior_weights=…) # PQL-REML rho for binary
gp_fit(Y, x, family='poisson')                   # PQL-REML rho for counts
```

- `family: Union[str, Family] = 'gaussian'` via `resolve_family`. `'gaussian'`
  dispatches to the **existing** exact/closed-form path **byte-identically** (a
  hard requirement — no regression to the Gaussian fits).
- `prior_weights: Optional[(N,)]` for Binomial trial counts / exposure.
- `GPResult.theta` → `[log sigma_f^2, log phi, log rho]` (`phi` column constant `0`
  for fixed-dispersion families); `dispersion` carries `phi`; a new `family` aux
  field. `log_mlik` is the PQL-REML (Phase 1) / LAML (Phase 2) value — document
  that cross-family AIC/BIC compares only within the same approximation.
- `gp_predict` returns the **latent** `eta` mean/var unchanged; add
  `type: Literal['link','response'] = 'link'` to optionally map through
  `family.linkinv` (response-scale) — the latent variance does *not* transform
  trivially, so document `'response'` as the delta-method / point estimate only.

## 6. Engineering / invariants

- **cuSOLVER-free, jit/vmap-clean preserved.** The inner PIRLS is a `lax.scan`
  (reuse `fit_penalised_irls`); the per-voxel pass stays `vmap`/`blocked_vmap`
  over `V`; the `rho` grid stays a host loop with the per-cell working-REML
  `jit`-compiled **once** (the moving `rho`-penalty as a traced arg — the same
  jit-once discipline as the `corr=` grid, to avoid the O(grid)-recompile OOM).
- **`block=` bounds the search** — the working-REML pooled-NLL routes through
  `blocked_vmap(...).sum()` exactly as the Gaussian path (PF1), so non-Gaussian
  `gp_fit` inherits the memory bound.
- The Gaussian branch is untouched (the new family branch is purely additive),
  so all existing `gp_fit` parity (sklearn / dense-REML / mgcv) is preserved.

## 7. Scope & phasing

- **Phase 1 (the headline):** `family in {binomial, poisson}`, `engine='hsgp'`,
  isotropic 1-D `x`, PQL-REML `rho` (criterion (a)). Document PQL attenuation.
- **Phase 2:** multi-D / ARD `x`; Laplace REML / LAML (criterion (b)) for
  mgcv-grade `rho`; free-dispersion families (Gamma / negbinomial) with a correct
  marginal scale.
- **Deferred:** `corr=` structured residual under a non-Gaussian likelihood (the
  whitening interacts with the working weights — non-trivial); `engine='exact'`
  for non-Gaussian (the kernel-eigenfeature design is fine, but the host eigh per
  `rho` × PIRLS is costly); the hierarchical `hgp_fit` non-Gaussian extension
  (same recipe, K blocks — a follow-on once `gp_fit` lands).

## 8. Validation

- **Gaussian reduction:** `family='gaussian'` must reproduce the current `gp_fit`
  outputs (regression-pinned; the new branch must not touch the Gaussian path).
- **mgcv parity (guarded):** `rho`/EDF of `gam(y ~ s(x, bs="gp", k, m=c(3, rho)),
  family=binomial()/poisson(), method="REML")` vs `gp_fit(..., family=…)` —
  sub-`df` EDF and same-order `rho` on smooth simulated data (Phase 2 LAML should
  tighten this; Phase 1 PQL documented to differ by the known attenuation).
- **Recovery:** simulate `eta = sin(2 pi x)`, draw Binomial / Poisson `y`, check
  the recovered smooth correlates with the truth and the estimated `rho` tracks
  the generating scale across seeds.
- **Internal anchor:** at a *fixed* `rho`, the Phase-1 fit must equal
  `hsgp_basis(rho) + gam_fit` (the existing workaround) — the only new thing is
  the `rho` *search*, so pin the at-fixed-`rho` fit to the gam path.

## 9. Risks

- **PQL bias** for binary / low-count (Phase 1) — the documented price; Phase 2
  LAML is the mitigation. Bound it empirically vs mgcv before advertising
  Binomial as production-grade.
- **`rho`–`lambda` identifiability** under a non-Gaussian likelihood is weaker
  than Gaussian; the nested FS-in-`rho`-grid may need damping or a joint grid.
- **Convergence cost** — PIRLS (`n_inner`) × `rho` grid (`n_rho`) × `V`. The
  jit-once + `blocked_vmap` discipline keeps memory bounded, but compile/runtime
  is `~n_rho` × the Gaussian cost; keep `n_rho` modest and parabolic-refine.

## 10. Cross-references

- [`gaussian-process-implementation.md`](gaussian-process-implementation.md) — the
  as-built Gaussian `gp_fit` (the path this extends; the `_penreml` core, the
  jit-once grid discipline, PF1 `block=` bounding).
- `stats/glmm/_pql.py`, `stats/glmm/_laplace.py` — the PQL / Laplace marginal
  precedent (and the MC4 dispersion lesson).
- `stats/_irls.py`, `stats/gam.py` — the penalised-IRLS working loop reused at
  fixed `rho`.
