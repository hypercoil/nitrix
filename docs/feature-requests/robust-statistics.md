# Robust statistics — `nitrix.stats.robust`

> **Status (2026-07-07): SHIPPED (`nitrix.stats.robust`).** :func:`mad`
> (median absolute deviation, normal-consistent), :func:`huber_regress`
> (monotone M-estimator), :func:`tukey_bisquare_regress` (redescender). Pure
> composition: differentiable fixed-iteration IRLS (`lax.fori_loop`, matching
> the GLM-IRLS convention) over the cuSOLVER-free `small_inv_logdet` ``(p, p)``
> weighted-least-squares solve; single response, `vmap` for the mass-univariate
> case. Provenance: `docs/feature-requests catalogue §12.7`.

**What.** M-estimator regression and the scale estimator that pairs with it.

**Proposed surface.**

```python
def huber_regress(X, y, *, delta): ...          # Huber-loss regression
def tukey_bisquare_regress(X, y, *, c): ...      # full redescender
def mad(x, axis): ...                             # median absolute deviation
```

**Composition.** M-estimators are iteratively reweighted least squares
(IRLS) on top of `linalg.residualise` (shipped):
`Xᵀ W(r) X β = Xᵀ W(r) y`, where `W` is the per-sample weight from the
influence function evaluated at the current residuals. Pure composition —
the IRLS loop is a `lax.while_loop` over the existing least-squares solve.

**Likely consumer.** Motion-corrupted fMRI regression, outlier-resistant
group analysis, robust mixed-effects via joint LME + IRLS.

**Effort.** S.

**Live-code status.** No `huber_regress` / `tukey_bisquare_regress` / `mad`.
`linalg.residualise` (the inner LS solve) and the `stats.lme` machinery
(`reml_fit`, `flame_two_level`) are shipped — the natural IRLS host.

## Cross-references

- `docs/feature-requests catalogue §12.7` — origin entry; `§13` — acceptance protocol.
- `src/nitrix/linalg/residual.py` — the LS solve IRLS reweights.
- [`docs/design/lme.md`](../design/lme.md) — the joint LME + IRLS consumer.
