# Public BLUP `predict` for the mixed-effects fitters (LME / GLMM)

> **Status (2026-06-26): SHIPPED (R1/R2/R2+corr/GLMM); R4/R3 staged.**
> `stats.lme.lme_predict` / `stats.lme.ranef` + `stats.glmm_predict` /
> `stats.ranef` are public. **Population** level (`X @ beta_hat`) works for
> **every** tier. **Conditional** (BLUP) level: GLMM (modes always retained)
> and the LME **R1** (scalar intercept), **R2** (random slope) and **R2 + corr**
> (structured within-group residual) tiers via the opt-in
> `lme_fit(..., retain_blups=False)` (default off), which runs a post-fit
> mixed-model-equation BLUP pass (`b_g = (Z_g^T Σ_g^{-1} Z_g + G^{-1})^{-1}
> Z_g^T Σ_g^{-1} r_g`) — a **post-pass that never touches the inner REML
> solver**, so the default fit path is **byte-identical** (verified: R1/R2 and
> R2+corr `ar1`/`cs` default result arrays sha-match baseline) and `retain_blups`
> is **zero-cost when off**.
>
> The **R2 + corr** path reuses the *same* mixed-model equation on **whitened**
> data: with `Cov(eps_g) = σ_e² R_g(ρ)` and the per-group whitener `W_g`
> (`W_g R_g W_gᵀ = I` ⟹ `W_gᵀ W_g = R_g^{-1}`), `Z_gᵀ R_g^{-1} Z_g =
> (W_g Z_g)ᵀ(W_g Z_g)` and likewise for `Z_gᵀ R_g^{-1} r_g`, so the BLUP is the
> standard one on whitened `(Z_g, r_g)` — the per-group `r×r` solve is shared
> (`_blup._solve_blup_system`), the whitening (per-voxel ρ) lives with the fit's
> group layout in `_corrfit._blups_corr`. A uniform `ranef(result)` reads the
> modes across GLMM + the LME tiers; unseen / `None` groups fall back to the
> marginal mean. Tests: independent dense-`R(ρ)` MME-BLUP oracle for R2+corr
> (`ar1` + `cs`, scalar + random-slope), the homoscedastic MME oracle for R1 +
> R2, GLMM conditional, unseen-group fallback, opt-in contract
> (`tests/test_mixed_predict.py`).
>
> **Staged (not yet conditional):** the **crossed (R4)** and **nested (R3)**
> tiers — two grouping factors, so `V` is not block-diagonal by *either* factor
> and the modes are two arrays (a contract extension to `ranef` / `lme_predict`,
> deferred until a consumer needs crossed/nested *conditional* prediction);
> `retain_blups=True` there raises a clear `NotImplementedError` (population
> still works for both).
>
> **Original request (2026-06-25): nimox-estimators Tier-3 → nitrix.** The mixed-model
> sibling of [`nimox-stats-response-predict.md`](resolved/nimox-stats-response-predict.md),
> filed separately because its apply contract is materially different. The
> mass-univariate **mixed-effects** fitters — `lme_fit` (REML / FaST-LMM,
> dispatching `REMLResult` / `LMEResult` / nested / crossed tiers) and
> `glmm_fit` (PQL / Laplace, `GLMMResult`) — return rich fitted results
> (`beta_hat`, random-effect covariance, and for GLMM the per-level `blups`) but
> expose **no public predict**. nimox's `nimox.estimators` façade wants to wrap
> `LME` / `GLMM` as fit/transform estimators; unlike the fixed-effects
> regressions, prediction here needs the **group assignments** and a choice of
> marginal vs conditional level — a genuine contract, not a one-line
> link-inverse.

## The ask

A public conditional-prediction entry point per fitter, taking a new design
**plus** group structure and a prediction level:

```
lme_predict(result, X: Float[Array, 'N p'], *,
            Z: Optional[Float[Array, 'N r']] = None,
            group: Optional[Int[Array, 'N']] = None,
            level: Literal['population', 'conditional'] = 'population')
    -> Float[Array, 'V N']
    # population : eta = X @ beta_hat.T                      (marginal mean)
    # conditional: eta + Z-row . b_g  for SEEN groups g,     (BLUP, subject-
    #              0 (marginal fallback) for unseen groups    specific)

glmm_predict(result, X: Float[Array, 'N p'], *,
             z: Optional[Float[Array, 'N r']] = None,
             group: Optional[Int[Array, 'N']] = None,
             level: Literal['population', 'conditional'] = 'population',
             type: Literal['response', 'link'] = 'response')
    -> Float[Array, 'V N']
    # as lme_predict, then family.linkinv(eta) for type='response'
```

`GLMMResult` already carries `blups` `(V, q)` / `(V, q, r)` and `re_var`
`(V, r, r)`; `glmm_predict` is then mostly assembly + link-inverse. For `lme`
the BLUPs must be reachable consistently — see the contract note below.

## Why (the nimox consumer)

`nimox.estimators` (RFC §14, Tier-3) wants `LME` / `GLMM` estimators:
`fit(X, y, groups=...) -> Fitted`, `transform(X_new, groups_new) -> predictions`.
The fit half delegates to `lme_fit` / `glmm_fit`; the transform half has no
public delegate. Re-deriving BLUP prediction in nimox would mean reaching into
the per-tier result internals (`REMLResult` vs `LMEResult` vs nested / crossed)
and replicating the random-effect-mode algebra — exactly the convention
duplication the consumer FRs exist to avoid. The contract (population vs
conditional, unseen-group handling) belongs in nitrix, with the fitter, where it
can be correct across tiers.

This is the longitudinal / repeated-measures arm of the nimox normative-modelling
estimator set (subject random effects, multi-session designs) — high value, but
predict is the only thing missing.

## Notes / scope (the contract this FR is really about)

- **Cross-tier BLUP accessor.** `lme_fit` dispatches several result types
  (`REMLResult` scalar-intercept R1, `LMEResult` R2 block-Woodbury, nested,
  crossed). `lme_predict` needs a *uniform* way to read each tier's random-effect
  modes `b_g` per group; `GLMMResult.re_var` is already documented as uniform
  across tiers (D4) to avoid downstream branching — the predict path wants the
  same uniformity for the BLUPs themselves. If the modes are not all retained on
  the result today, surfacing them (or a small `ranef(result) -> (V, q[, r])`
  accessor) is the enabling primitive.
- **Unseen groups.** A new subject/site absent from the fit has no BLUP; the
  documented behaviour should be the **marginal fallback** (random mode = 0, i.e.
  the population prediction), so `transform` on held-out groups is well-defined.
  `group=None` => population level for all rows.
- **Group-index alignment.** `group` indexes the fitted levels (`0..q-1`); make
  the out-of-range / unseen convention explicit (the marginal fallback above).
- **Differentiability**: predict differentiable w.r.t. `X_new` (and `beta_hat`
  / BLUPs); the REML / PQL fit stays non-differentiable.
- Additive surface; no change to `lme_fit` / `glmm_fit` or the existing
  contrast-inference (`lme_t_contrast` / `lme_f_contrast`) paths.

## Acceptance

- `lme_predict` / `glmm_predict` public, with explicit `level` (population /
  conditional) and unseen-group (marginal-fallback) semantics; a uniform BLUP
  accessor across the lme tiers.
- **Self-consistency** under test: `level='conditional'` predict on the training
  `(X, group)` reproduces the fitted conditional means; `level='population'`
  equals `X @ beta_hat.T`; `glmm_predict` with the identity link / Gaussian
  family matches `lme_predict`.
- nimox `nimox.estimators` wraps `LME` / `GLMM` delegating `transform` to the
  new predict; fit/transform + serialise + vmap-fit substrate tests pass.

## Cross-references

- nimox `docs/feature-requests/nimox-estimators.md` §14 (Tier-3, rank 7) — the
  origin.
- [`nimox-stats-response-predict.md`](resolved/nimox-stats-response-predict.md) — the
  fixed-effects sibling (Beta / Ordinal / GauLSS / GAM); split from this because
  their predict is a one-line link-inverse, not a group-aware BLUP.
- `nitrix.stats.glmm` `GLMMResult.blups` / `re_var` — the BLUPs already on the
  GLMM result; `stats.lme` `REMLResult` / `LMEResult` — the per-tier results
  whose modes the uniform accessor must reach.
- [`stats-modelling-suite-v3.md`](stats-modelling-suite-v3.md) — GL(A)MM
  completeness / mixed-model inference, the standing stats-suite planning this
  composes with.
