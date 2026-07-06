# Ledoit-Wolf shrinkage covariance — `nitrix.stats.ledoit_wolf`

> **Status (2026-07-06): SHIPPED.** `stats.ledoit_wolf` / `stats.oas` — analytic
> shrinkage covariance (v2 §4.1; `stats/connectivity.py`, `test_connectivity.py`).
>
> **Original (2026-06-02): not started — low priority.** Brainstorm candidate
> (sibling of [`graphical-lasso.md`](graphical-lasso.md), §12.14; both
> regularise the empirical covariance/precision for the small-sample regime).
> Provenance: surfaced building the `nitrix-perf-bench` precision/partialcorr
> backfill against nilearn; ledger context in
> [`perf-bench-feedback.md`](../perf-bench-feedback.md).

**The gap (why this is worth a stub).** nilearn's `ConnectivityMeasure` — the
canonical fMRI connectome estimator — **defaults its covariance estimator to
`LedoitWolf(store_precision=False)`** (verified: `cov_estimator=None` resolves
to `LedoitWolf` in `fit`). So the community-standard `precision` /
`partial correlation` / `tangent` a downstream user gets *out of the box* are
built on a **shrunk** covariance, not the raw empirical inverse. nitrix.stats
ships only the **raw** estimators (`cov` / `precision` / `partialcorr`, which
need `obs > c` and are noisy when `c ≈ obs`) — there is **no shrinkage
estimator** in `nitrix.stats` (verified: no `ledoit`/`shrink`/`oas`). The
perf-bench had to pass `EmpiricalCovariance` explicitly to make nilearn match
nitrix; to instead match nilearn's *default* (the realistic comparison for a
connectome pipeline) requires round-tripping through `sklearn.covariance`.

**What.** Ledoit-Wolf (2004) analytic shrinkage: a convex blend of the sample
covariance `S` toward a scaled identity, with the shrinkage intensity `α`
estimated in **closed form** (no cross-validation):

```
Σ̂ = (1 − α) S + α μ I,   μ = tr(S)/p,   α = min(1, β² / δ²)
```

where `δ²` = ‖S − μI‖_F² and `β²` is the (clipped) average sample variance of
the entries of `S`. The shrunk `Σ̂` is well-conditioned and invertible even at
`c ≈ obs` / `c > obs`, so `precision`/`partialcorr` built on it are stable in
the small-sample regime the raw inverse cannot handle.

**Proposed surface.**

```python
def ledoit_wolf(X, *, assume_centered=False): ...   # -> (cov, shrinkage)
def shrunk_covariance(X, *, method='ledoit_wolf'):  # 'ledoit_wolf' | 'oas'
    ...
```

(OAS, Chen 2010, is the natural sibling — same closed form, a different `α`.)

**Composition.** Pure JAX — trace / Frobenius-norm reductions + one scalar `α`;
no solver, fully differentiable, GPU-resident. Feeds the existing
`precision` / `partialcorr` directly (`inv(shrunk_cov)`), and is the missing
piece for a nilearn-default-equivalent connectome path.

**Relationship to GLASSO (§12.14).** Both shrink the empirical estimate toward
a better-conditioned one. Ledoit-Wolf is **analytic + cheap (S, closed form)**
and shrinks the *covariance* toward identity; GLASSO is **iterative** and
imposes *sparsity* on the precision. Ledoit-Wolf is the lighter, default-path
estimator; GLASSO the heavier conditional-independence-graph one.

**Effort.** S — closed form, ~20 lines; the differentiability is trivial (no
implicit-VJP needed, unlike GLASSO).

**Live-code status.** `nitrix.stats.__all__` ships `cov` / `ccov` /
`precision` / `partialcorr` / `pcorr` (raw empirical) — no shrinkage estimator.

## Cross-references

- [`graphical-lasso.md`](graphical-lasso.md) — sibling regularised-precision
  estimator (§12.14); `SPEC §9` acceptance protocol.
- [`perf-bench-feedback.md`](../perf-bench-feedback.md) — the perf-bench-surfaced
  ledger (this gap surfaced matching nilearn's connectome estimators).
- `src/nitrix/stats/covariance.py` — `cov` / `precision` / `partialcorr`.
