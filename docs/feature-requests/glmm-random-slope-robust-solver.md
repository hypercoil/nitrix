# GLMM random-slope robust solver — joint-Schur PQL + REML-EM — `nitrix.stats.glmm`

> **Status (2026-06-19): ✅ SHIPPED** (branch `feat/glmm-slope-followups`,
> `be5a277`). `_glmm_slope_structured_one` is now the monotone joint-Schur +
> REML-EM solver below. Validated: Gaussian unstructured == `lme_fit`'s REML
> (beta / G / sigma_e^2 to ~1e-8, converged by the **default** `n_outer=20` — the
> linear-convergence worry below did not materialise), and **clamp-insensitive**
> (identical G across `eta_bound` ∈ {20, 30, 60, ∞} for the seed that degenerated
> the old Newton at 30; a regression test pins it). The clamp reverts to pure
> overflow safety. The two performance items below (nested-REML waste, `block`
> default) are subsumed / still open as noted; the Laplace-gradient sibling item
> is tracked separately. *Original FR (now implemented) follows.*

**What.** Replace the solver *core* of the unstructured (correlated-`G`)
random-slope GLMM — `glmm.py::_glmm_slope_structured_one` — with a **joint-Schur
inner IRLS + monotone REML-EM outer**, removing the current path's sensitivity to
the IRLS linear-predictor clamp.

**Why.** The shipped solver wraps a *full* block-Woodbury REML fit (analytic
AI-Newton, `_blockwoodbury._fit_one`) inside the PQL outer loop. On the first,
poorly-scaled working response it can over-shoot the random-effect covariance
`G`, and the BLUP feedback then amplifies it — for a Poisson/Gamma random slope a
large `b_slope · x` blows up `exp(eta)`. The shipped fix (a per-family
`eta_bound` clamp, `bdfc55c`) is *load-bearing for the answer, not just for
overflow*: at clamp `20` the correlated-Poisson fit lands in the correct REML
basin (verified across seeds), at `30` it converges to a **degenerate** one
(random-intercept variance → 0). A monotone solver removes this knife-edge: EM
cannot over-shoot, so the fit is insensitive to the clamp (which reverts to pure
overflow safety), and the correlated path is robust regardless of the family's
dynamic range.

**Proposed approach.**

- **Inner (fixed `G`, `phi`): joint-Schur penalised IRLS for `(beta, b)`.** A
  direct `r`-dimensional lift of the *scalar* `_structured_solve` (which is
  proven robust — the scalar-intercept `tier='many'` path): per group
  `D_g = Z_g^T W Z_g + phi G^{-1}` (`r × r`), `B_g = X_g^T W Z_g` (`p × r`),
  Schur `S = X^T W X − Σ_g B_g D_g^{-1} B_g^T` onto the `p`-dim fixed block, then
  `b_g = D_g^{-1}(Z_g^T W z_work_g − B_g^T beta)`. All per-group `r × r` /
  `p × p` reductions via `segment_sum`; cuSOLVER-free (`small_inv_logdet`).
- **Outer: REML-EM update of `(G, phi)`.** `G = (1/q) Σ_g [b_g b_g^T + phi
  Cov(b_g)]` with the **β-uncertainty term** `Cov(b_g) = phi(D_g^{-1} +
  D_g^{-1} B_g^T S^{-1} B_g D_g^{-1})` (this is what makes it REML-EM, not
  ML-EM — and what makes it match `lme_fit`'s REML for the Gaussian case);
  `phi = 1` for fixed-dispersion families, else the Pearson estimate
  `Σ (y−mu)^2 / V / (n − edf)`. Monotone in the REML objective, so no
  over-shoot.

**Composition / blast radius.** Internal to `glmm.py`'s structured-slope path —
**no public-API or `GLMMResult` change** (same `tier='slope'`, same `re_var`
`(V, r, r)` / `blups` `(V, q, r)`). It *drops* the dependency on
`_blockwoodbury`'s `_fit_one` / `bw_score_and_ai` / `group_grams` (the rewrite is
self-contained segment-reduction algebra). The shipped `test_glmm_slope.py`
oracle (Gaussian slope == `lme_fit`; r=1 == scalar intercept; correlated
recovery) re-runs unchanged — it *is* the acceptance test.

**Risk.** Opposite profile to the algorithm's ease: the math is a
generalisation of code we already trust, but **EM converges linearly**, so
matching `lme_fit` (REML) to the current `1e-3` Gaussian-oracle tolerance may
need more outer iterations or a slightly looser tol; and the β-uncertainty trace
term must be exactly right or the Gaussian oracle drifts. The real cost is
re-validating the two Gaussian oracle tests, not writing the solver.

**Cheaper alternative (middle ground).** Keep the current iterated-REML core but
add a `G`-cap or step-damping on the first 1–2 outer iterations (~15 lines): kills
the over-shoot without the EM convergence-rate work, at the cost of being less
principled than EM. Worth doing first if the full rewrite is not yet justified.

**Effort.** M (~0.5–1 day), mostly validation.

## Performance findings (2026-06-19 review fan-out)

A three-lens review of the shipped branch surfaced two performance issues that
this rewrite (and one sibling item) should subsume:

- **The nested REML is the dominant cost, and the rewrite fixes it.** The current
  `_glmm_slope_structured_one` runs `n_outer` (≈20) PQL steps, and *each* calls
  `_fit_one` — a full block-Woodbury REML = `n_inner` (≈10) damped-Newton
  iterations with a 5-way backtracking line search — so ≈200 inner Newton bodies
  per voxel, re-converging a warm-started REML that has barely moved. The
  proposed **REML-EM outer is a single closed-form update per PQL step**, which
  removes this multiplicatively (≈3–5× fewer inner bodies). Interim mitigation if
  the rewrite is deferred: drop the *slope path's* effective `n_inner` to 1–3
  (warm-started, the IRREML identity only needs the variance components to
  *track*, not fully re-converge each step) — the public `n_inner=10` default is
  shared with the cheap many-level IRLS path and is overkill here.

- **`block=None` is a latent OOM for the slope/Laplace tiers.** These paths
  materialise per-observation `(N, r, r)` outer products (`segment_sum` to
  `(q, r, r)`); under the default `block=None` that is `(V, N, r, r)` live across
  all voxels at once (and, for Laplace, additionally taped by `jax.hessian`).
  The scalar paths tolerate `None` (their intermediates are `(N,)`/`(p, p)`).
  **Recommendation:** a non-`None` default `block` for the slope/Laplace tiers, or
  at least document that `block` is effectively required at brain scale.

**Sibling item — the Laplace slope's autodiff cost (separate from this rewrite).**
`_glmm_laplace_slope_one` lets `damped_newton` (`curvature=None`) take
`jax.grad`/`jax.hessian` *through* the `n_mode`-step mode-finding `lax.scan` —
the single largest cold-compile / autodiff-tape driver in the slope work. **The
obvious shortcut (`stop_gradient` on the mode / envelope theorem) is unsound
here:** the marginal's `-0.5 logdet H_g` term depends on the mode `b̂(θ)` through
the IRLS weights, so its θ-gradient does *not* vanish at the mode — only the
`ℓ_g(b̂) - ½b̂ᵀG⁻¹b̂` part does. Stopping the gradient would drop that term, change
the estimator to an approximate (PQL-like) Laplace, and break the
scipy-reference match. The correct optimisation is an **analytic Laplace
gradient** (implicit-function-theorem derivative of the mode feeding the logdet
term) supplied via the existing `curvature=` seam — a real derivation, scoped as
its own follow-up, not part of this rewrite. Until then the autodiff-through-scan
is the *correct* (if heavy) choice; it is exact when the mode is converged
(quadratic Newton, fine at `n_mode=20` for well-posed groups) and only biased for
ill-conditioned/empty groups with too few mode steps.

> **Investigated 2026-06-19 (measured, deferred).** Two things turned up that
> change the cost/benefit:
> 1. The cold-compile cost is dominated by the **gradient**-through-scan, not the
>    hessian: replacing `jax.hessian` with a cheap *fixed-mode* curvature (exact
>    `jax.grad`, mode `stop_gradient`'d **in the curvature only** — the gradient
>    stays exact, so the optimum is unchanged: verified to 5.5e-7) cut per-iteration
>    runtime ~2.7× but barely moved compile (4.65→4.41 s).
> 2. That fixed-mode curvature is not a true Newton, so it **converges ~2–3× slower**
>    (gradnorm 4.3e-3 at n_iter=60 vs 1e-8 for the real hessian) — the per-iteration
>    speedup is cancelled, and it under-converges at the default iteration count.
>
> So the only real win is the **exact** analytic hessian made cheap, which requires
> **implicit differentiation of the mode** (a `custom_vjp` whose backward solves the
> IFT cotangent through the scan-free per-group score `F`, `∂F/∂b = -H_g`): it keeps
> quadratic convergence *and* removes the scan from the autodiff tape. That is the
> right fix, but it is a focused, higher-risk piece (custom_vjp block-structure
> bookkeeping) whose payoff is mostly cold-compile time — which is **amortised** in a
> jitted mass-univariate run (compile once over all `V` voxels). Deferred on ROI:
> the autodiff-through-scan path is correct and the production cost is amortised.
> The AGQ path (`method='agq'`) shares the same mode-finder and would benefit
> identically if/when this lands.

**Live-code status.** Current solver: `src/nitrix/stats/glmm.py
::_glmm_slope_structured_one` (iterated full block-Woodbury REML in the PQL
loop). The load-bearing clamp is `_family.py::_ETA_MAX = 20.0` +
`Family.eta_bound`. The scalar template to generalise is the same file's
`_structured_solve` / `_dispersion`. No joint-Schur `r × r` slope inner; no
REML-EM update.

## Cross-references

- `src/nitrix/stats/glmm.py` — `_glmm_slope_structured_one` (target),
  `_structured_solve` + `_dispersion` (scalar template to lift), the Laplace
  slope (`_glmm_laplace_slope`, the accurate-but-slower alternative for the same
  hard regime).
- `src/nitrix/stats/lme/_blockwoodbury.py` — the REML the current path wraps
  (and that the rewrite removes from this hot loop).
- `src/nitrix/stats/_family.py` — `_ETA_MAX` / `eta_bound`, the clamp the rewrite
  would let revert to pure overflow safety.
- Memory `glmm-exp-link-irls-clamp` — the clamp-value / basin finding that
  motivates this.
