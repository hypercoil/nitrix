# von Mises–Fisher directional statistics in `nitrix.stats`

> **Status (2026-07-02): SHIPPED (`nitrix.stats.directional`).** The keystone
> item is built and validated — see §6. The single most important item in the
> [`hypercoil-examples` migration](hypercoil-examples-migration.md):
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

## 6. Status — SHIPPED (2026-07-02)

`nitrix.stats.directional` (`src/nitrix/stats/directional.py`): `log_iv`,
`vmf_log_prob`, `vmf_fit` (→ `VMFFit`), `vmf_sample`. Both legacy liabilities
are resolved from the theory, not ported:

- **`log_iv` is full-range**, not the legacy large-κ asymptotic. Because
  `ν = p/2−1` is fixed by the sphere dimension it is a **static** argument, so
  the regime is a compile-time branch: ascending series (DLMF 10.25.2) for
  `κ ≤ 120` and the large-argument asymptotic (DLMF 10.40.1) for `κ > 120` when
  `ν < 15`; the uniform (Debye) asymptotic (DLMF 10.41.3, terms `U₀…U₅`) for
  `ν ≥ 15`. The κ-split feeds each branch gradient-safe inputs (clamped into its
  valid region) so the unused branch cannot inject a NaN. Validated to
  **< 3.5e-9** against an `mpmath` oracle over `ν ∈ [0, 300]`,
  `κ ∈ [1e-3, 1e6]`; a dedicated test asserts the legacy leading-term form is
  materially wrong (> 1e-2) at small κ where `log_iv` is not.
- **`vmf_sample` has guaranteed acceptance** — a `jax.lax.while_loop` that
  resamples the not-yet-accepted draws until all are accepted (Wood 1994; `w`
  via the Beta envelope with `b = (−2κ + √(4κ² + (p−1)²))/(p−1)`, tangent uniform
  on `Sᵖ⁻²`). No fixed iteration cap, no `found=False` path: every returned
  sample is a valid unit vector (tested to 1e-6 across `p ∈ {2,3,10,64}`).
  Non-differentiable (documented).
- **`vmf_fit`** returns the *exact* `A_p(κ)=R̄` root — Banerjee closed-form warm
  start refined by Newton on the exact derivative
  `A_p'(κ) = 1 − A_p² − (p−1)/κ · A_p` (tested: residual < 1e-10, and the
  refinement provably moves off the Banerjee value). `μ̂` is the normalised
  (optionally weighted) resultant. The §6.5 fit; `vmf_log_prob` is the apply.
- `vmf_log_prob` integrates to 1 over the sphere (quadrature test) and is
  grad-finite in `μ`/`κ`; catalogued in the op-matrix (`log_iv` is pure
  elementwise → jit-clean on all backends, unlike the eigh-based register ops).

51 tests green; ruff + mypy clean. Downstream (`nimox`): the
`VonMisesFisher(numpyro.Distribution)` wrapper, `NormSphereParameter`, and the
parcellation emission model. `docs/op_matrix.json` render regen deferred (ungated
generated artifact; full run flakes on single-process XLA-CPU).

### Family growth — Watson SHIPPED (2026-07-06); Kent / Bingham next

**Watson** (the *axial* distribution, `x ≡ -x`) is built and validated on branch
`feat/stats-directional-watson`: `log_kummer_m` (the Kummer confluent-
hypergeometric normaliser `M(½, p/2, κ) = ₁F₁(½; p/2; κ)` — regime-split
series/large-argument asymptotic, Kummer's transform `M(a,b,z)=eᶻM(b−a,b,−z)` for
the girdle `κ<0`, validated **< 2e-13 vs mpmath** including `κ=0`), `watson_log_prob`
(surface-measure density, consistent with `vmf_log_prob`), and `watson_fit` /
`WatsonFit` (MLE = scatter eigenvector + bisection on the monotone `g(κ)=∂_κ log M
= r̄`, recovering **both** bipolar `κ>0` and girdle `κ<0`, likelihood-selected).
Two gradient-boundary fixes were load-bearing: the `|z|→z` double-`where` and an
**additive-`eps`** series floor (a clamping/denormal floor zeroes or underflows
`g(0)=1/p`, which then misroutes every girdle root-find). 18 tests + op-matrix.

**Kent (FB5) SHIPPED (2026-07-06)** — the S² *elliptical* vMF, density `∝
exp(κ γ₁ᵀx + β[(γ₂ᵀx)² − (γ₃ᵀx)²])`: `log_kent_normaliser` (the half-integer-
Bessel series `c(κ,β)=2π Σ_j (Γ(j+½)/j!) β^{2j} (2/κ)^{2j+½} I_{2j+½}(κ)`,
**reusing `log_iv`**; validated **< 7e-15 vs mpmath** and reducing to `1/C_3(κ)`
at β=0), `kent_log_prob` (surface-measure density), and `kent_fit` / `KentFit`
(Kent-1982 moment estimator — resultant mean + tangent-plane axis
diagonalisation; frame + κ recovered accurately, β carries the estimator's known
finite-κ downward bias, documented with an MLE-refinement note). 23 tests.

**Unnormalised energy SHIPPED (2026-07-06)** — `vmf_log_prob` /
`watson_log_prob` / `kent_log_prob` take **`normalize=False`**, returning the
bare exponent (natural-parameter · sufficient-statistic) with the normaliser
*dropped and never computed*. The tractable-in-high-d, per-site Gibbs/Markov-
random-field **clique potential** (the parcellation setting), score-kernel-clean
— field-energy composition/scalarisation stays downstream (SPEC §5).

**`fisher_bingham_energy` SHIPPED (2026-07-06)** — the *general* quadratic-
exponential sphere energy on any `S^{p-1}`: `E = κ γ₁ᵀx + Σⱼ βⱼ(γⱼᵀx)²`
(orthonormal frame + coefficient vector; the Fisher–Bingham family `exp(κμᵀx +
xᵀAx)`, `A = Γ diag(β) Γᵀ`). **Named for the family, not one member** (user
correction — it subsumes several). Validated to reduce, to machine precision, to
**vMF** (β=0), **Watson** (rank-1 quadratic), **Bingham** (κ=0, `= xᵀAx`), and the
**S² Kent** (p=3). Energy-only (the p-dim normaliser is intractable) — the
tractable-at-any-`p` MRF potential for high-dimensional directional fields.

> **Abstraction check (2026-07-06): the subsumption is mathematical, NOT an
> implementation-sharing directive — keep the specialised energies.** Empirically
> (jaxpr + compiled timing): `fisher_bingham_energy(β=0)` does **not** reduce to
> the vMF/Watson energy. It performs a full-frame `Γᵀx` matvec (**O(p²)**, one
> `dot_general`) and always evaluates the quadratic, whereas `vmf`/`watson_log_prob
> (normalize=False)` are **O(p)** single-direction dot products (0 `dot_general`;
> `μ` is a *vector*, not a frame). Since `κ`/`β` are runtime arrays, XLA cannot
> dead-code-eliminate back to the specialised path — measured **4.4× slower at
> p=64, 18.8× at p=256**, growing with `p`. Plus an API mismatch (vMF/Watson take a
> direction; delegating would force materialising an arbitrary orthonormal
> complement) and no shared numerical-stability code (the energies are plain
> polynomials; all stability lives in the *normalisers*, which this energy-only
> form doesn't touch). Only the **S² Kent** energy is ~equivalent (both project
> all axes; neutral). A guard-note is in the `fisher_bingham_energy` docstring so a
> future DRY refactor doesn't regress this.

**`watson_sample` + `bingham_sample` SHIPPED (2026-07-06)** — the
**Angular-Central-Gaussian rejection** (Kent–Ganeiber–Mardia 2018): a
`lax.while_loop` for **guaranteed acceptance** with **bounded ~25% efficiency
uniformly** in concentration and dimension (vs the naïve Beta/vMF envelopes,
which collapse to ~0% at high `κ`), and **normaliser-free**. Correctness is
*automatic*: accept = `r(s)/sup_s r`, independent of the envelope parameter `b`
(which only sets efficiency; `b` solves `Σ 1/(b+2aⱼ)=1` by bisection). Watson is
the rank-one Bingham. Validated: sampled `E[(μᵀx)²]` matches the exact Watson
second-moment oracle to <0.01 across `p` and bipolar/girdle `κ`; sample→fit
round-trips; Bingham mode structure + uniform limit. 24 tests.

Still to build — the two **research-grade** pieces (deferred over shipping
unvalidated/inefficient math; the correctness mandate):

1. **Kent sampler** — Kent is Fisher–Bingham (a linear term *plus* quadratic).
   The naïve vMF-envelope tilt has acceptance ~`exp(−β)` → unusable for eccentric
   Kent (`2β/κ → 1`). And the linear term breaks the pure-quadratic ACG's
   *automatic* sup-bound (the ratio then depends on `x` through two scalars, not
   one), so an efficient sampler needs the KGM **FB-specific** construction —
   research-tracked.
2. **Bingham normaliser + `bingham_log_prob` + `bingham_fit`** — the "very
   difficult" constant: the confluent hypergeometric of *matrix* argument. The
   **Kume–Wood saddlepoint** is the standard tractable route (its saddlepoint
   equation `Σ 1/(2(t̂−λⱼ))=1` mirrors the shipped ACG `b`), but it is an
   *approximation* needing careful derivation + Monte-Carlo validation before
   shipping. The Bingham **energy and sampler already ship**, so this is reduced
   to the normaliser-dependent density/MLE only — research-tracked.

The coordinate-kernel spatial-prior construction — **✅ SHIPPED (2026-07-06)** as
`register.CoordinateKernelPrior` / `EfficientProMises(spatial_prior=…)`: the
whole-brain-tractable ProMises anatomical prior (an RBF coordinate kernel projected
into the alignment subspace by random Fourier features, never forming the `(p,p)`
kernel). See [`register-functional-alignment`](register-functional-alignment.md) §6.

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
