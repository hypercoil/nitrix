# von Mises–Fisher directional statistics in `nitrix.stats`

> **Status (2026-06-30): PROPOSED — keystone gap.** The single most important
> item in the [`hypercoil-examples` migration](hypercoil-examples-migration.md):
> nitrix has a rich spherical **geometry** stack (`spherical_conv`,
> `spherical_geodesic_distance`, `spectral_sphere_embedding`,
> `spherical_parameterize`, …) and **zero spherical statistics**. Cortical
> surface signals live on Sᵖ⁻¹; the von Mises–Fisher (vMF) distribution is the
> canonical spherical law and vMF mixtures are the emission model for surface
> parcellation (Schaefer / HierarchBayesParcel lineage). Proposed home: a new
> `nitrix.stats.directional` submodule (vMF now; Watson / Bingham / Kent as the
> family grows), the spherical analogue of `stats.gaussian`.
>
> **Correctness mandate — and why this kernel is the worst legacy liability.**
> Clean-room from the directional-statistics literature (Mardia & Jupp; Wood
> 1994; Banerjee et al. 2005; DLMF §10), **not** a port. The legacy
> `hypercoil/init/vmf.py` is the recovery oracle, not the spec, and is
> known-suspect in two load-bearing places (below). Validate against
> `scipy.special.ive` / `mpmath.besseli` and Monte-Carlo (test-only).

## 1. The four kernels (theory)

For `x ∈ Sᵖ⁻¹`, `f(x; μ, κ) = C_p(κ) exp(κ μᵀx)`,
`C_p(κ) = κ^{p/2−1} / [(2π)^{p/2} I_{p/2−1}(κ)]`, so the log-normaliser is
`log C_p(κ) = (p/2−1) log κ − (p/2) log 2π − log I_ν(κ)` with order `ν = p/2 − 1`.

1. **`log_iv(nu, kappa)`** — log of the modified Bessel function of the first
   kind `log I_ν(κ)`. The normaliser of *everything* below. **This is the crux.**
2. **`vmf_log_prob(x, mu, kappa)`** — the vMF log-density (differentiable in
   `μ`, `κ`); a distributional score kernel (cf. `stats.gaussian.gaussian_nll`).
3. **`vmf_fit(x, *, weights=None) -> (mu, kappa)`** — MLE. `μ̂ = Σxᵢ / ‖Σxᵢ‖`;
   with `R̄ = ‖Σxᵢ‖ / n`, `κ̂` solves `A_p(κ) = R̄` where
   `A_p(κ) = I_{p/2}(κ) / I_{p/2−1}(κ)`. The §6.5 fit; `vmf_log_prob` is the apply.
4. **`vmf_sample(key, mu, kappa, shape) -> x`** — keyed sampler (Wood 1994).

## 2. The two known-suspect places (theory over legacy)

1. **`log_bessel` is a large-κ asymptotic only — wrong at small/moderate κ.**
   The legacy `log_bessel` (Fröhlich–Spencer large-κ form,
   `−½(log 2πκ + ½log(1+r²)) + κ√(1+r²) − ν·arcsinh(r)`, `r = ν/κ`) is the leading
   term of the uniform asymptotic, valid only for large κ, and its own docstring
   admits it is *unvalidated* ("we should run some Monte-Carlo… to assess this
   approximation"). At small/moderate κ it is materially wrong, which corrupts
   the normaliser → the log-prob → the κ-MLE → the whole emission model. JAX
   ships only `i0e`/`i1e` (orders 0, 1), so general-order `log I_ν` must be built.
   **Theory-correct approach:** a regime-split, accurate over the full
   `(ν ≥ −½, κ > 0)` range and validated to tight tolerance —
   - small/intermediate κ: the ascending series
     `I_ν(κ) = Σ_m (κ/2)^{2m+ν} / (m! Γ(m+ν+1))` in log-space (log-sum-exp);
   - large argument: the **uniform asymptotic expansion** (DLMF 10.41), keeping
     enough `U_k(t)/νᵏ` terms for the target tolerance (the legacy keeps only the
     leading term);
   - cross-check via the ratio `A_p(κ)` Perron continued fraction.
   Differentiable in `κ` (and `ν` static); golden-tested per dtype against
   `scipy.special.ive`/`mpmath`. This is *exactly* the irreducible, careful
   numerical content nitrix exists to own.
2. **The sampler has non-guaranteed acceptance (a correctness bug).** The legacy
   `random_VMF_angle` runs a **fixed `max_iter=5`** rejection loop and returns a
   `found` mask — i.e. it can silently return **invalid** (unaccepted) samples.
   Wood (1994) rejection sampling has guaranteed *eventual* acceptance; the
   faithful kernel uses a `lax.while_loop` per draw (or a vectorised scheme with
   a proven failure-probability bound and an explicit fallback), never a silent
   `found=False`. The tangent direction is uniform on Sᵖ⁻²; the μ-component `w`
   has density `∝ (1−w²)^{(p−3)/2} exp(κw)` sampled via the Beta((p−1)/2,(p−1)/2)
   envelope with `b = (−2κ + √(4κ² + (p−1)²)) / (p−1)`. Sampling is
   non-differentiable — documented as such.

   Also: the κ-MLE closed form `κ̂ ≈ R̄(p − R̄²)/(1 − R̄²)` (Banerjee) is itself an
   **approximation** — ship it with Newton refinement against the exact `A_p(κ)`
   and a documented error bound, not as if it were exact.

## 3. Surface & boundaries

- `nitrix.stats.directional`: `log_iv`, `vmf_log_prob`, `vmf_fit` (→ `VMFFit`
  `NamedTuple` of `(mu, kappa)`), `vmf_sample`. `log_prob` returns the unreduced
  per-point tensor + the flat `reduction` leaf (§5); `vmf_fit`/`log_prob` are the
  §6.5 fit/apply pair (state = `(mu, kappa)` plain arrays).
- Pure `jax`/`jaxtyping`/`numpy`. The `VonMisesFisher(numpyro.Distribution)`
  wrapper, `NormSphereParameter`, and the parcellation emission model stay in
  `nimox`. `scipy.special` / `mpmath` are **test-only** oracles.
- fp32/fp64-first (tenet 11): the Bessel/normaliser math is in the scientific
  core — fp32-accurate, no sub-fp32 path.

## 4. Acceptance

- `log_iv(ν, κ)` matches `scipy.special.ive(ν, κ)·exp(κ)` (i.e. `log ive + κ`)
  to tight tolerance across a `(ν, κ)` grid spanning small→large κ and large p
  (the surface-feature regime) — the explicit refutation of the legacy
  large-κ-only failure.
- `vmf_log_prob` integrates to 1 over the sphere (quadrature) and is `jax.grad`-
  finite in `κ`, `μ`.
- `vmf_fit` recovers planted `(μ, κ)` within CI on sampled data; κ̂ matches the
  exact `A_p(κ)=R̄` root, not just Banerjee.
- `vmf_sample` always returns unit-norm, valid samples (no `found=False` path);
  empirical `(μ̂, κ̂)` from samples ≈ the generating parameters.

## 5. Cross-references

- Ledger: [`hypercoil-examples-migration`](hypercoil-examples-migration.md).
- **Explicitly independent of** [`register-functional-alignment`](register-functional-alignment.md):
  ProMises uses the *matrix* vMF, whose normaliser never materialises — it does
  **not** depend on this family.
- Sibling score-kernel surface: `stats.gaussian` (`gaussian_nll`,
  `kl_diagonal_gaussian`). Geometry it complements: `geometry.sphere.*`.
- Provenance: `hypercoil/init/vmf.py` (`log_bessel`, `log_prob_vmf`,
  `random_VMF`/`random_VMF_angle`, the `VonMisesFisher` Distribution → `nimox`).
- References: Mardia & Jupp, *Directional Statistics*; Wood (1994); Banerjee et
  al. (2005) JMLR; DLMF §10.41.
