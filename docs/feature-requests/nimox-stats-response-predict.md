# Public `*_predict` for the response-regression fitters (Beta / Ordinal / GauLSS / GAM)

> **Request (2026-06-25): nimox-estimators Tier-3 → nitrix.** A small,
> additive API-surface completion. `nitrix.stats.glm_fit` already pairs with a
> public `stats.predict(result, X_new)` (apply the fitted coefficients to a new
> design), which let nimox wrap a `GLM` fit/transform estimator. The sibling
> **response-regression fitters do not expose a predict**: `beta_fit`,
> `ordinal_fit`, `gaulss_fit`, `gam_fit` each return a rich fitted result
> (per-element coefficients + covariance) but there is no public
> `*_predict(result, X_new)`. nimox's estimator façade (`nimox.estimators`, the
> immutable sklearn-`fit`/`transform` analogue) needs the **apply** half to wrap
> these as fit/transform estimators — exactly the shape `glm.predict` already
> has. No new fit-math; the link-inverse / basis-evaluation pieces already exist
> internally (`gam`'s `smooth_partial_effect` proves the GAM building blocks).

## The ask

Expose a public `predict` for each fitter, mirroring `stats.predict`'s
`(result, X_new, *, type=...) -> predictions` shape. Per fitter:

```
beta_predict(result: BetaResult, X: Float[Array, 'N p'], *,
             type: Literal['response', 'link'] = 'response')
    -> Float[Array, 'V N']
    # eta = result.coef @ X.T;  response = expit(eta)  (the logit mean model)

gaulss_predict(result: GauLSSResult, X: Float[Array, 'N p'], *,
               scale_design: Optional[Float[Array, 'N q']] = None)
    -> tuple[Float[Array, 'V N'], Float[Array, 'V N']]
    # (mean, scale): mu = X @ coef_mu.T;  sigma = exp(Z @ coef_scale.T)
    # (Z defaults to an intercept column, matching gaulss_fit's default)

ordinal_predict(result: OrdinalResult, X: Float[Array, 'N p'], *,
                type: Literal['class_prob', 'cum_prob', 'class'] = 'class_prob')
    -> Float[Array, 'V N K'] | Float[Array, 'V N']
    # eta = X @ coef.T;  cum = F(thresholds - eta);  class_prob = diff(cum)
    # 'class_prob' -> (V, N, K) simplex; 'cum_prob' -> (V, N, K-1);
    # 'class' -> (V, N) argmax label

gam_predict(result: GAMResult, smooths: Sequence[Smooth],
            x_smooths: Sequence[...], *,
            parametric: Optional[Float[Array, 'N q']] = None,
            type: Literal['response', 'link'] = 'response')
    -> Float[Array, 'V N']
    # assemble [intercept | parametric | B_k(x_k)] at the NEW covariates using
    # the fitted knots/penalty (basis.eval_design + result.col_slices), then
    # eta = design @ coef.T;  response = family.linkinv(eta)
```

`glm.predict` is the template (and the byte-faithfulness bar for the Gaussian
identity-link case where GAM with no smooths == GLM).

## Why (the nimox consumer)

`nimox.estimators` wraps each nitrix fitter as a two-type immutable estimator —
`Fit(...).fit(X, y) -> Fitted`, `Fitted.transform(X_new) -> predictions`. The
fit half is delegated to the existing `*_fit`; the **transform half has no
public delegate**, so these four families cannot be wrapped without nimox
re-deriving the link-inverse / basis assembly against the result internals
(fragile — it would replicate `stats`-private conventions, the same anti-pattern
the histogram fit/apply and differentiable-registration FRs resolved). Exposing
`*_predict` lets each estimator delegate `transform` outright.

These four are the nimox-estimators **Tier-3** backlog (RFC §14): once
`*_predict` lands, `BetaRegression`, `OrdinalRegression`, `GaussianLocationScale`
and `AdditiveModel` (GAM) wrap immediately — normative-modelling and
bounded/ordinal/heteroscedastic response estimators that are otherwise blocked.

## Notes / scope

- **Output shapes differ** (the only wrinkle vs `glm.predict`'s `(V, N)`):
  `ordinal_predict` returns a per-observation class simplex `(V, N, K)` (or
  argmax `(V, N)`); `gaulss_predict` returns the **pair** `(mean, scale)` (the
  heteroscedastic point — the scale is the model's reason to exist).
- **GAM reuses the fitted knots.** `gam_predict` must evaluate each smooth basis
  at the new covariates with the *same* knot vector / penalty the fit used
  (`basis.eval_design(x_new)` + `result.col_slices`), so the prediction is
  consistent with the fitted smooth — `smooth_partial_effect` already does this
  for one block; `gam_predict` assembles the full design (intercept + parametric
  + all smooths). Watch the B3-style extrapolation edge (`spline_design` beyond
  the knot span) flagged in the stats-suite audit.
- **Differentiability** is unchanged: predict is differentiable w.r.t. `X_new`
  (and the fitted coefficients) through the link-inverse / basis eval; the fit's
  IRLS / Fisher-scoring search stays non-differentiable (matching `glm`).
- Pure additive surface — no change to any `*_fit` or result type.

## Acceptance

- `beta_predict` / `gaulss_predict` / `ordinal_predict` / `gam_predict` public,
  shaped like `stats.predict`; **self-consistency** under test (predict on the
  training `X` reproduces the fitted means — e.g. `gaulss_predict(...)[0]`
  matches `X @ coef_mu.T`; `gam_predict` with no smooths == `glm.predict`).
- nimox `nimox.estimators` wraps the four Tier-3 estimators delegating
  `transform` to the new `*_predict`; their fit/transform + serialise + vmap-fit
  substrate tests pass.

## Cross-references

- nimox `docs/feature-requests/nimox-estimators.md` §14 (the candidate backlog,
  Tier-3) — the origin; this is the predict half those rows are blocked on.
- `nitrix.stats.glm.predict` (`stats/glm.py`) — the template and parity bar.
- `nitrix.stats.gam.smooth_partial_effect` (`stats/gam.py`) — proves the GAM
  basis-eval + `col_slices` assembly pieces already exist.
- [`stats-modelling-suite-v3.md`](stats-modelling-suite-v3.md) (GAMM surfacing /
  inference completeness) and [`stats-suite-audit.md`](stats-suite-audit.md)
  (B2 `spline_design` extrapolation) — the standing stats-suite planning this
  composes with.
- Sibling nimox-consumer predict/apply split:
  [`nimox-histogram-match-fit-apply.md`](nimox-histogram-match-fit-apply.md)
  (same "expose the apply half so nimox wraps fit/transform" shape).
