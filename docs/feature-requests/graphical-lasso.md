# Graphical LASSO — `nitrix.stats.glasso`

> **Status (2026-06-02): not started — the empirical-covariance input
> (`stats.cov`) and dense `stats.precision` are shipped; the L1-penalised
> sparse estimator is not.** Brainstorm candidate; promotion gated by the
> §13 acceptance protocol. Provenance: `SPEC_UPDATE_v0.3.md §12.14`.

**What.** Sparse precision-matrix estimation (Friedman / Hastie / Tibshirani
2008):

```
Θ̂ = argmin_Θ  ⟨S, Θ⟩ − log det Θ + λ ‖Θ‖_{1,off}
```

with `S` the empirical covariance and `λ` the sparsity penalty.

**Proposed surface.**

```python
def glasso(S, lam, *, method='coord_descent'): ...   # sparse precision
def glasso_path(S, lambdas): ...                       # regularisation sweep
def ebic_score(theta, S, lam): ...                     # model selection
```

**Composition.** `S` is `stats.cov` (shipped). Two solver backends compose
with existing substrate:

- **Coordinate descent** (the GLASSO algorithm) — per-row partition +
  lasso-shrinkage update. Pure JAX; differentiable through the
  regularisation path via implicit-VJP at the KKT-stationary point (composes
  [`krylov-solvers.md`](krylov-solvers.md), §12.1).
- **ADMM** (Boyd 2011) — two proximal steps (log-det + soft-thresholding);
  `proximal_log_det` composes with `linalg.symlog` (shipped).

**Likely consumer.** Functional-connectivity sparse precision estimation
(partial correlations as conditional-independence graph edges),
graphical-model preprocessors for connectome analyses, ICA-followup
decompositions. The fMRI literature has defaulted to graphical LASSO for
~15 years; native support lets downstream pipelines avoid round-tripping
through `sklearn.covariance`.

**Effort.** M. Coordinate descent is the standard reference (~30 lines);
making it correctly differentiable (implicit-VJP at convergence) is the
careful part.

**Live-code status.** No `glasso` / `glasso_path` / `ebic_score`.
`stats/__init__` ships `cov` / `ccov` / `precision` / `partialcorr` /
`pcorr` (the empirical covariance and dense precision/partial-correlation
this sparsifies), and `linalg.symlog` (the ADMM `proximal_log_det` piece).

## Cross-references

- `SPEC_UPDATE_v0.3.md §12.14` — origin entry; `§13` — acceptance protocol.
- [`krylov-solvers.md`](krylov-solvers.md) — implicit-VJP inner solve.
- `src/nitrix/stats/covariance.py` — `cov` / `precision` / `partialcorr`.
- [`docs/design/stats.md`](../design/stats.md).
