# GLMM random-slope robust solver — joint-Schur PQL + REML-EM — `nitrix.stats.glmm`

> **Status (2026-06-19): not started.** Hardening follow-up to the shipped
> non-Gaussian random-slope work (branch `feat/glmm-random-slopes`:
> `c8b44cd` the slope feature, `bdfc55c` the exp-link IRLS clamp). The
> correlated-slope PQL path *works* and is validated, but is clamp-sensitive;
> this FR tracks replacing its solver core with a monotone one. **Not a
> correctness blocker** — promotion gated on the correlated-Poisson/Gamma
> random-slope path becoming load-bearing for a real analysis.

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
