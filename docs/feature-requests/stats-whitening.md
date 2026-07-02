# Whitening in `nitrix.stats` — findability wrapper + implementation-strategy research

> **Status (2026-06-30): PROPOSED — reframed (NOT a missing-estimator gap).**
> From the [`hypercoil-examples` migration](hypercoil-examples-migration.md)
> (`atlas/vmf.py::generalised_whitening`). **Duplication check (2026-06-30):** the
> whitening *estimator* already exists at the correct §6.5 seam —
> `nimox.estimators.whitening` (`Whitening` / `ShrunkWhitening` /
> `SparseConnectivity`) composes `nitrix.linalg.sympower(Σ, power=-0.5)`, and PCA
> whitening is `nimox.estimators.pca.PCA(whiten=True)`. We must **not** duplicate
> that container. Two narrower nitrix-side reasons nonetheless justify a small
> addition: **(A) findability** and **(B) an implementation-strategy research
> question** (a faster / more stable inverse-sqrt path than `sympower`-of-cov).
>
> **Correctness mandate — theory over legacy.** Clean-room from the
> matrix-fractional-power theory; the legacy is the recovery oracle. The
> eccentricity-monotonicity `cummax` "repair" is a heuristic to be theory-grounded
> or rejected (§4), not copied as default.

## 1. What already exists (do not duplicate)

- **Irreducible kernel (nitrix):** `linalg.sympower(Σ, power)` — the eigh-based
  SPD fractional power. ZCA matrix = `sympower(Σ, −0.5)`, inverse = `sympower(Σ,
  +0.5)`. **Partial sphering** is already expressible: `sympower(Σ, −s/2)`.
- **Estimator container (nimox):** `estimators.whitening.Whitening` /
  `ShrunkWhitening` / `SparseConnectivity` (immutable fit/transform eqx modules,
  shrinkage via `stats.ledoit_wolf`/`glasso`); PCA whitening via
  `estimators.pca.PCA(whiten=True)`. This is the correct §6.5 split and stays put.

So there is **no missing whitening capability**. What is missing is (A)/(B).

## 2. The two legitimate nitrix-side contributions

**(A) Findability — a named `whiten` in the stats vocabulary.** Today "whiten the
data" is folkloric knowledge — *compose `sympower` at `power=-0.5`, and remember
ZCA vs PCA, and floor the small eigenvalues*. A named `stats.whiten` /
`zca_whiten` makes the operation **discoverable and conventional** (SPEC §9's
*named member of a coherent vocabulary family* admission path — discoverability +
convention, distinct from the irreducible-content path), beside `stats.pca` /
`stats.ledoit_wolf`. It is a thin pure function (the eqx estimator stays nimox and
*re-uses it*, removing the convention from the consumer per §6.5 invariant 3).

**(B) Implementation strategy — is `sympower`-of-covariance the best inverse
sqrt?** The nimox path forms `Σ = cov(X)` then `eigh(Σ)`. For whitening
specifically that is worth re-examining on nitrix's two axes (perf + stability),
because forming the covariance is exactly the SPEC §2 tenet-11 conditioning
hazard (squaring the data squares the condition number) and `eigh` is the
cuSOLVER-fragile path nitrix is chartered to route around.

## 3. Implementation-strategy research (the substantive work)

Benchmark + analyse candidate inverse-sqrt / whitening paths; pick by stability
then GPU wall-clock (the latter delegated to `nitrix-perf-bench`):

- **`sympower(cov(X), −1/2)`** — the eigh-of-covariance baseline (what nimox
  uses). Reference for correctness.
- **Data-SVD-direct** — whiten from `X = U S Vᵀ` *without* forming the
  covariance: `Σ^{−1/2} = V diag(√(n−1)/S) Vᵀ` (ZCA), `S` the data singular
  values. Avoids the condition-number squaring of `cov` → strictly better fp32
  conditioning; one SVD vs cov+eigh.
- **Newton–Schulz inverse-sqrt iteration** — matrix-multiply-only fixed-point for
  `Σ^{−1/2}` (coupled iteration), **cuSOLVER-free**, GPU-friendly, differentiable
  by construction; converges for well-scaled SPD inputs (pre-scale by `tr Σ`).
  The hardware-aware, eigh-free pathway nitrix prefers.
- **Cholesky whitening** — `Σ = L Lᵀ`, `W = L⁻¹`: cheapest, but a *triangular*
  (non-ZCA) whitening — note as a distinct, documented variant, not a drop-in for
  the symmetric map.

The deliverable of (B) is a justified default (likely data-SVD or Newton–Schulz
for the fp32/GPU production path, with the `sympower` route as the reference
oracle), not necessarily a new public symbol beyond the (A) wrapper.

## 4. The suspect legacy bit (theory over legacy)

The legacy `cummax` eccentricity/ordering "repair" is rejected on the default
path: for `s ∈ [0,1]`, `λ ↦ λ^{−s/2}` is already monotone and `Σ^{−s/2}` is the
unique SPD solution — textbook whitening needs no repair. The `cummax` most
likely patches an artefact of the legacy *adaptive per-eigenvalue schedule*
(`sphering.py`), not whitening; if a non-global graded schedule is genuinely
wanted it is an explicit, separately-tested opt-in (and lives with its consumer),
never a silent perturbation of the fractional-power result. **Other edge cases to
own:** zero/near-zero eigenvalues (`rcond`/ridge floor, loud); fp32-negative
eigenvalues (clamp, loud); `n < p` rank deficiency.

## 5. Surface & boundaries

- `stats.whiten(x, *, reference=None, mode='zca'|'pca', sphering=1.0, rcond=None)`
  — a thin findable wrapper (`reference=None` ⇒ self-whiten); the fitted form is
  `whiten_fit`/`whiten_apply` (state = the whitening matrix, plain arrays) so
  nimox's `Whitening` re-uses *this* rather than re-deriving the convention.
- Pure `jax`/`jaxtyping`/`numpy`; differentiable (the SPD VJP; repeated
  eigenvalues documented). The eqx estimator + learnable `AdaptiveSphering`
  schedule stay in `nimox`.

## 6. Acceptance

- `stats.whiten(x, mode='zca')` equals `x @ sympower(cov(x), −0.5)` to tolerance
  (proves no behavioural divergence from the existing seam); `cov(whiten(x)) ≈ I`.
- `sphering=s` matches `Σ^{−s/2}`; `s=0` identity, `s=1` full — **without** the
  `cummax` repair.
- (B) the chosen production path is certified `≈ sympower` within the pinned
  tolerance and shown more stable (fp32 conditioning) and/or faster (GPU,
  cuSOLVER-free) — else default stays `sympower` and the FR ships (A) only.
- nimox `Whitening` re-points to the nitrix wrapper with its tests green (no
  convention drift).

## 7. Cross-references

- Ledger: [`hypercoil-examples-migration`](hypercoil-examples-migration.md).
- Existing seam (do not duplicate): `linalg.sympower` (`linalg/spd.py`),
  `nimox.estimators.whitening` / `pca`.
- Hardware-aware eigh-free motivation:
  [`jacobi-eigensolver-cusolver-free`](jacobi-eigensolver-cusolver-free.md),
  SPEC §2 tenet 11 (covariance-squaring fp32 hazard); perf delegated to
  `nitrix-perf-bench`.
- Provenance: `hypercoil-examples/atlas/vmf.py::generalised_whitening`.
