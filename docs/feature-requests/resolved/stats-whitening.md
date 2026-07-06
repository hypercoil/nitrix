# Whitening in `nitrix.stats` — findability wrapper + implementation-strategy research

> **Status (2026-07-06): SHIPPED (both (A) and (B)).** `nitrix.stats.whiten` /
> `whiten_fit` / `whiten_apply` / `whiten_inverse_apply` + `WhiteningState` land
> the (A) findable ZCA vocabulary on the §6.5 seam, and (B) resolves the
> inverse-sqrt question to the **cuSOLVER-free Newton-Schulz** driver
> (`nitrix.linalg.symsqrt(…, driver='newton_schulz')`, matmul-only, GPU-clean,
> gradient-stable at a repeated spectrum) as the full-sphering default — certified
> `≈ sympower(cov, -0.5)` to ~1e-9 and `cov(whiten(x)) ≈ I`. Partial sphering
> keeps the eigenvalue map (`sympower`); PCA whitening is **not** duplicated (it
> stays `nimox.estimators.pca.PCA(whiten=True)`, needing the eigenbasis). The
> nimox estimator re-point is filed as
> [`whitening-backend-nitrix-newton-schulz`](../../../../nimox/docs/feature-requests/whitening-backend-nitrix-newton-schulz.md).
> `WhiteningState` fields (`mean` / `matrix` / `inverse_matrix`) match
> `FittedWhitening` so the swap is mechanical. The `cummax` "repair" was rejected
> as predicted (§4).
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

## 5. Surface & boundaries — as shipped

- `stats.whiten(x, *, reference=None, sphering=1.0, eps=0.0, assume_centered=False)`
  — the thin findable wrapper (`reference=None` ⇒ self-whiten); fitted form
  `whiten_fit` → `WhiteningState(mean, matrix, inverse_matrix)` /
  `whiten_apply` / `whiten_inverse_apply`. The single-call is *defined as*
  `whiten_apply(x, whiten_fit(reference))` (§6.5 invariant 2, byte-identical).
- **`eps` ridge, not `rcond`.** The conditioner is a Tikhonov ridge on the
  covariance (`cov(l2=eps)`) — the SPD-preserving floor the matmul-only
  Newton-Schulz path wants (it cannot truncate small eigenvalues), and it
  matches nimox's `Whitening(eps=…)` for a mechanical swap. Eigenvalue
  *truncation* (`rcond`) is a PCA-whitening concept and stays with PCA (nimox).
- **`mode` dropped (ZCA only).** PCA whitening reintroduces the raw eigenbasis
  (there is no eigh-free PCA) and already lives at
  `nimox.estimators.pca.PCA(whiten=True)`, so `whiten` is the symmetric (ZCA)
  map only — no duplication, cuSOLVER-free by default.
- Pure `jax`/`jaxtyping`/`numpy`; jit/vmap/grad-clean. The eqx estimator +
  learnable `AdaptiveSphering` schedule stay in `nimox`.

## 6. Acceptance — met

- ✅ `stats.whiten(x)` equals `x_centred @ sympower(cov(x), −0.5)` to ~1e-9
  (no behavioural divergence from the existing seam); `cov(whiten(x)) ≈ I`.
- ✅ `sphering=s` matches `Σ^{−s/2}` (exact vs `sympower(−s/2)`); `s=0` centring
  only, `s=1` full — **without** the `cummax` repair.
- ✅ (B) the production path (Newton-Schulz) is certified `≈ sympower` within
  tolerance and is cuSOLVER-free / jit-clean (the eigh path falls back to eager
  CPU on the affected GPU stacks); grad-stable at a repeated spectrum. 20 stats
  tests + 9 `symsqrt` driver tests + op-matrix, green.
- ⏳ nimox `Whitening` re-point — filed as
  [`whitening-backend-nitrix-newton-schulz`](../../../../nimox/docs/feature-requests/whitening-backend-nitrix-newton-schulz.md)
  (nimox-side change; certify ≈ + tests green, no convention drift).

## 7. Cross-references

- Ledger: [`hypercoil-examples-migration`](../hypercoil-examples-migration.md).
- Existing seam (do not duplicate): `linalg.sympower` (`linalg/spd.py`),
  `nimox.estimators.whitening` / `pca`.
- Hardware-aware eigh-free motivation:
  [`jacobi-eigensolver-cusolver-free`](../jacobi-eigensolver-cusolver-free.md),
  SPEC §2 tenet 11 (covariance-squaring fp32 hazard); perf delegated to
  `nitrix-perf-bench`.
- Provenance: `hypercoil-examples/atlas/vmf.py::generalised_whitening`.
