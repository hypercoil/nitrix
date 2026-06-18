# Statistical modelling suite v3 — GL(A)MM completeness for the `nwx` DSL (ledger)

> **Status (2026-06-18): Tier-0 shipped + completed.** The blocking tranche is
> implemented, validated, and on `feat/stats-suite-v3`: **§2** GAMM surface
> (`REBasis` / `re_smooth`, a third `Smooth` variant); **§5.1** non-aggressive
> (AROMA) `partial_residualise` + a cuSOLVER-free Cholesky for `residualise`;
> **§1.3** mixed-model fixed-effect inference — both `lme_t_contrast`
> (Satterthwaite df; SE matches statsmodels to ~6e-4) **and `lme_f_contrast`**
> (multi-row Wald F with the Fai-Cornelius / lmerTest multivariate-Satterthwaite
> denominator df, collapsing exactly to the t-test at `L = 1`); **§1.1** the
> `lme_fit` R1/R2 dispatcher + the tier-R2 block-Woodbury REML for a correlated
> `(1 + x | g)` (matches statsmodels MixedLM random-slope exactly) **and the
> diagonal-`G` `(x || g)` `structure='diagonal'` path** (independent variance
> components; off-diagonal held at zero; matches a statsmodels diagonal free-mask
> fit). The F-contrast's denominator-df eigendirections use a new cuSOLVER-free
> small-symmetric Jacobi eigensolver (`stats._smalllinalg.sym_eig_jacobi`,
> jittable/`vmap`-clean). All cuSOLVER-free, ruff/mypy clean, per-component tests
> green. *(The original Tier-0 tranche was reconstructed on 2026-06-18 after a
> Code-Ocean host crash rewound the branch; source recovered from the session
> transcript + file-history and re-validated end-to-end. `lme_f_contrast` and the
> diagonal-`G` path close the last two Tier-0 surface gaps.)* Tier-1/Tier-2 (§1.2
> GLMM, §1.4 AR1/CAR1, §3.1 by-variable smooths, §4 families, §6.2 sandwich SEs,
> §1.1 R3/R4, §1.3 Kenward-Roger) remain proposed. Driver: the **`nwx`**
> neuroimaging Wilkinson-extension DSL (in `gramform`;
> `gramform/docs/nwx/spec.md`) emits an immutable `ModelSpec` IR that an engine
> lowers onto `nitrix` score kernels. `nwx`'s v1 scope guarantee — GLM / GAM /
> **GAMM** / (G)LMM + residualisation + multi-level node graph — surfaces the
> kernels below that v1 (shipped: [`stats-modelling-suite.md`](stats-modelling-suite.md))
> and v2 (shipped, merged to `main`: [`stats-modelling-suite-v2.md`](stats-modelling-suite-v2.md))
> did **not** deliver. Two items are *completeness-of-claimed-scope* fixes
> (§2 GAMM surfacing; §1.1 general random effects), not new ambitions — v1's
> own framing ("penalised GLM ≡ variance-components REML ≡ mixed model;
> a GAMM is just add random-effect components") promised them, but the public
> API does not yet let a caller express them.
>
> **Revised 2026-06-17 after an engineering review** (rigour / performance /
> abstraction; the as-submitted consumer version is in git history at commit
> `2962c19`). The revision (a) makes **performance-preserving dispatch** a
> first-class invariant — §0.1 — so generalising the suite never regresses the
> common fast paths; (b) splits §1.1 by linear-algebra complexity class (the
> single-grouping-factor block solve is cheap; crossed/multiple is not); (c)
> resolves the RE-block abstraction (a third `Smooth` variant, not a
> `SplineBasis` with dummy fields) and the API-shape question (additive
> `lme_fit` dispatcher, `reml_fit` unchanged); (d) re-tiers §4 by real effort.
> The incorporated review record is §13.

## §0. Framing

v1 built the spine (one variance-components REML core, one penalised-IRLS GLM
core, one `randomise`/TFCE engine). v2 filled out bases, randomise statistics,
and regularised connectivity. **v3 is the modelling-completeness tranche that a
real formula DSL exercises**: the moment `nwx` lets a neuroimager write
`(1 + age | subject)`, `s(subject, bs="re")`, a binomial mixed model, an AR(1)
longitudinal error structure, a Satterthwaite-corrected mixed-model contrast, or
a non-aggressive (AROMA) residualisation, it hits the edges of the shipped
surface.

Everything here is a **score kernel** (arrays → coefficients / statistics /
p-values), per the v1/v2 classification — no new scalarisation, container, or
CLI surface, and the cuSOLVER-free + differentiability discipline carries over
unchanged. Every item lands on existing substrate (`_varcomp` / `_lowrank` AI-
REML, `_irls` penalised-IRLS core, the `SplineBasis`/`(design, penalty)`
contract, `_smalllinalg` cuSOLVER-free solve, `_blocked_vmap`).

The items are tiered (§10) by whether they **block** `nwx`'s v1 guarantee.

### §0.1 Performance-preserving dispatch — the no-regression invariant  *(load-bearing)*

The single most important design constraint of v3, because v3 *generalises*
engines whose value is that they are fast on the common case:

> **Generalisation must not regress the common path.** Each general entry point
> **dispatches on its configuration to the cheapest *correct* solver**, and the
> existing validated fast paths remain the *realised* path for the
> configurations they already serve — guaranteed by a no-regression guard (the
> dispatched call for a shipped configuration compiles to the same HLO / same
> flop class as today's op).

This is not new to nitrix; v3 must simply honour the pattern the suite already
runs on:

- `stats._smalllinalg.small_inv_logdet` — static size-dispatch (`n==1`
  reciprocal, `n==2` closed form, `n>2` rolled Cholesky).
- `linalg._eigsolve.eigsolve_top_k` — `forward(method) ⟂ backward(format)` with
  an `auto` policy that picks the cheapest valid method.
- `stats.pca.pca_fit(solver='full'|'gram'|'randomized'|'auto')` — `auto` routes
  to the `(n, n)` Gram when `n < d`.
- `gam_fit` — the Gaussian **cross-product** fast path is selected automatically
  for the Gaussian family (no N in the inner loop), with the generic IRLS path
  for non-Gaussian; `lambda_mode='shared'` pools when smoothness is homogeneous.
- `reml_fit(low_rank=...)` — q-rank vs dense `ZZ^T` eig.

So the **LME dispatch ladder (§1.1)** and the **GAMM routing (§2)** are the
heart of v3's design: a new general `lme_fit` selects the *lowest-cost solver
that is exact for the given random-effects structure*, and the shipped
single-component FaST-LMM spectral path (dense + q-rank low-rank) stays the
realised path for `(1 | g)` — bit-for-bit, with its 50/50-GPU validation intact.
Each new general op ships with a guard asserting the shipped configuration still
lowers to the shipped op (no `(V, ·, ·)` intermediate it did not have before).

### §0.2 Scope boundary (nitrix vs the ecosystem)

Unchanged from v1 §1 and normative SPEC §1 / v0.5 §1:

- nitrix ships **score kernels**: arrays in → statistic arrays out. It does
  **not** ship the DSL, formula parsing, design-matrix assembly, data binding,
  the mass-axis `vmap` orchestration, file-format parsing, or any "loss".
- **`nwx`/`gramform` is the new named driver** (joining niffi / thrux / ilex).
  `nwx` produces a validated `ModelSpec`; an engine assembles `X`/`Z`/bases and
  calls these kernels. nitrix neither imports nor knows about `nwx` — it just
  needs to *expose* the kernels the engine dispatches to. This FR is the list
  of those kernels.
- **RNG / differentiability**: unchanged (keyed pure generators; fits
  differentiable, inference loop non-differentiable). Per-item differentiability
  caveats for the new inference are called out where they bite (dof estimates,
  sandwich weights are generally *not* differentiated through).

---

## §1. Theme 1 — LME / GLMM completeness

### §1.1 General random-effects covariance — the structure-dispatched LME  *(BLOCKING; tier split by cost)*

**What.** A general mixed-model entry that honours `nwx`'s `(…|…)` surface,
built as a **dispatcher over solver tiers** (§0.1) so the common cases keep their
shipped cost. The current `reml_fit(Y, X, Z)` (two components
`[log σ_b², log σ_e²]`, single scalar random effect, `lme/reml.py`) is **kept
unchanged** as the public single-component entry and as the realised fast path;
a new general entry sits *above* it:

```python
def lme_fit(Y, X, random=(), *, family=GAUSSIAN, structure='auto',
            corr=None, ...) -> LMEResult
#   random: tuple of grouping terms; each carries its (N, q_g) design Z_g and a
#           within-group covariance form ('scalar' (1|g), 'diagonal' (x||g),
#           'unstructured' (1+x|g) r×r, nested, crossed)
#   structure='auto' picks the lowest-cost EXACT solver for the random spec:
```

**The dispatch ladder (cheapest exact solver wins; this is the design):**

| Tier | Random spec | Solver | Per-voxel cost | Status |
|---|---|---|---|---|
| R0 | none (residual only) | OLS / GLS | `O(N p²)` | shipped (`glm_fit`) |
| R1 | one scalar factor `(1\|g)` | **FaST-LMM spectral REML** (dense; q-rank `low_rank=` when `q<N`) | `O(N p² + N)` / iter | **shipped (`reml_fit`) — unchanged** |
| R2 | one factor, correlated/diagonal `r×r` `(1+x\|g)`, `(x\|\|g)` | **block-diagonal per-group Woodbury** (`r×r` inner solve per group) | `O(Σ_g n_g r² + G r³ + p²)` | new, cheap, cuSOLVER-free |
| R3 | nested `(1\|g1/g2)` | hierarchical block solve (telescoping Woodbury) | structured, moderate | new |
| R4 | crossed `(1\|g1:g2)` / multiple factors | general AI-REML over the sparse mixed-model equations `(p + Σ q_g)` | `O((p+Σq_g)³)`-class — **expensive** | new, **HLO-gated** |

**Why the split matters (the review's headline finding).** v1/v2's cheapness is
*specific to one grouping factor*: `V = σ_b² ZZ^T + σ_e² I` is diagonalised once
by the `ZZ^T` eigenbasis (FaST-LMM), giving `O(N)` per-voxel work per iteration.
That trick **does not generalise** across grouping factors — for crossed
`(1|g1:g2)` or multiple terms, `V` has no shared eigenbasis and no block
structure, and you are into the `(p + Σ q_g) × (p + Σ q_g)` mixed-model
equations (lme4's sparse-Cholesky regime). At brain scale (`V = 500k` voxels ×
Newton iterations) a dense per-voxel `(p+Σq_g)³` solve is **prohibitive**, and
not trivially cuSOLVER-free. The consumer spec's "per-group covariance is tiny
`r ≤ 3` → cuSOLVER-free" is **correct for R1/R2 and wrong for R4** — so v3 ships
the cheap tiers first and gates R4 explicitly:

- **R2 is the genuine new win** and covers most of `nwx`'s `(…|…)` demand:
  a *single* grouping factor (even with a correlated `r×r` within-group Σ) keeps
  `V` block-diagonal across groups, so a per-group Woodbury with an `r×r` inner
  solve is exact and cheap — cuSOLVER-free (`r≤3` closed form / rolled Cholesky),
  mass-univariate-friendly, no `(V, N, N)` intermediate. This is the FaST-LMM
  idea generalised from "rotate by `ZZ^T`" to "exploit per-group block
  structure".
- **R4 (crossed/multiple)** is a different algorithm (sparse MME). It is in
  scope but tiered **Tier-2**, with its own no-large-intermediate HLO audit, and
  is *only* reached when `structure='auto'` cannot prove a cheaper tier applies.

**Substrate.** R1 = the shipped `_varcomp`/`_lowrank` AI-REML, untouched. R2 =
a new per-group block-Woodbury assembled from the same analytic score / average-
information machinery, `r×r` solves via `_smalllinalg` (the AI-REML derivations
already generalise to `K` thetas; the new part is the block-Woodbury `V⁻¹`/log-det
in place of the spectral diagonal). R3/R4 reuse the AI-REML θ-iteration with a
structured/sparse `V⁻¹`. **Effort: M (R2) · M (R3) · L (R4).** **Oracle:**
`lme4::lmer` (REML) variance components + fixed effects; balanced-design closed
forms; nested/crossed ANOVA. **Guard:** `lme_fit(..., random=((Z,'scalar'),))`
lowers to the same HLO as `reml_fit(Y,X,Z)` (R1 no-regression proof).

### §1.2 GLMM — random effects with non-Gaussian families  *(high value)*

**What.** Random effects under a binomial / Poisson / … family via
penalised-quasi-likelihood / Laplace-approximate REML: `glmm_fit(Y, X, random,
*, family, structure, ...)` = the §1.1 dispatcher wrapped in the IRLS loop
(working response → one structure-dispatched AI-REML step → repeat). It inherits
§1.1's tier dispatch, so a binomial `(1|g)` GLMM runs the R1/R2 cheap path, not
the R4 solver.

**Why.** v1's LME is Gaussian-only (`reml.py`); GAM does families but its
"random effect" is a ridge penalty under Gaussian-ish penalised IRLS. True
binomial/Poisson mixed models (lesion counts, binary outcomes per subject with
random intercepts) are unreachable. `nwx` exposes `{{ family=binomial }}` on a
formula carrying `(1|g)`.

**Substrate.** Compose the `_irls` penalised core (v2 H3) with the §1.1
structure-dispatched variance-component update. **Effort: L.** **Oracle:**
`lme4::glmer`; `mgcv::gam(family=, …, s(g, bs="re"))`. **Estimator:** PQL first
(cheap; documented bias for binary/low-count), Laplace as a follow-up
(§11). Differentiability: the fit is differentiable through the fixed PQL budget
(the v1 LME-Newton pattern).

### §1.3 Mixed-model fixed-effect inference — SE, contrasts, dof  *(BLOCKING; cheap)*

**What.** `REMLResult`/`LMEResult` currently exposes only `beta_hat`,
`theta_hat`, `log_lik` (`lme/reml.py`) — no fixed-effect standard errors, no
contrast test, no denominator dof. Add:

```python
def lme_t_contrast(result, contrast, *, dof='satterthwaite') -> LMEContrast
def lme_f_contrast(result, contrast, *, dof='kr')           -> LMEContrast
```

**Why.** `nwx` lets a user write `contrasts: age = age (t)` on a mixed model;
without an SE + valid dof there is no test. The GLM path has
`t_contrast`/`f_contrast` (`glm.py`); the LME path has nothing. v2 §8.5 flagged
"CI for LME fixed effects absent."

**Substrate (and why it is cheap).** The fixed-effect covariance
`(Xᵀ V⁻¹ X)⁻¹` is **already computed and discarded** inside the per-voxel solve
(`_varcomp._profile_beta`'s `A_inv`); Satterthwaite additionally reuses the
**average-information inverse** (`_score_and_info`'s `info`, the asymptotic
`cov(θ̂)`), also already computed. So §1.3 is "surface two already-formed
quantities + the dof algebra" — genuinely M-effort, cuSOLVER-free, per-voxel.
Kenward-Roger's bias-corrected vcov (second-derivative adjustment) is the heavy
part → **Tier-2**. Return a small frozen `LMEContrast` record (house style:
`GLMResult`/`GAMResult`/`REMLResult` are registered pytrees). **Differentiability:**
the effect/SE are differentiable; the *dof* estimate is a model-selection-style
scalar normally not differentiated through (document, don't promise a VJP).
**Effort: M (Satterthwaite) + M-L (Kenward-Roger, Tier-2).** **Oracle:**
`lmerTest` (Satterthwaite), `pbkrtest` (Kenward-Roger).

### §1.4 Error-correlation & heteroscedasticity structures  *(high value)*

**What.** Within-group residual correlation / non-constant variance (nlme
parity): `ar1(time|g)`, `car1`, `cs` (compound symmetry), and variance functions
`varIdent` / `varPower`. Surface: a `corr=` / `weights=` argument on `lme_fit`
shaping the residual `Σ_e(ρ)`.

**Why.** Longitudinal neuroimaging (repeated sessions/subject) needs an
AR(1)/CAR(1) within-subject error model; `nwx` exposes
`{{ correlation=ar1(session|subject) }}`. Currently residuals are `σ_e² I` only.

**Substrate.** A structured residual adds one (or few) correlation parameters to
the AI-REML θ-vector; this composes with the **R2 block structure** (the natural
home: within-group correlation presupposes a grouping factor), where each group's
`Σ_e(ρ)` inverse/log-det is closed-form — AR(1) tridiagonal, CS rank-1-plus-diagonal
— so it stays cuSOLVER-free and block-local. (Outside a grouping factor a global
`AR(1)` over `N` is also tridiagonal-closed-form but is the R0+corr path.)
**Effort: M-L.** **Oracle:** `nlme::lme(correlation=corAR1, weights=varIdent)`.

---

## §2. Theme 2 — GAMM surfacing (claimed in v1, not surfaced)  *(BLOCKING)*

**The discrepancy.** v1 scoped GAMM and `gam.py` *documents* it
(`gam.py:13-14`: *"a **GAMM** adds explicit random-effect blocks, which enter as
just more penalty components (a random effect is a ridge penalty)"*), and the
penalised-IRLS core genuinely can absorb such a block. **But there is no public
way to build or pass one:** `gam_fit` (`gam.py`) accepts only `smooths` /
`parametric` / `intercept` / `family` / `lambda_mode` — no random-effect entry;
and `basis._raw_features` (`basis.py:291`) recognises only
`kind ∈ {bspline, cyclic, tprs}`, with no `re`/`fs` constructor. So GAMM is a
**latent capability that never surfaced** — `nwx`'s `s(g, bs="re")` is parseable
but un-lowerable. This is a completeness gap against v1's own stated scope.

**What (the fix) — a third `Smooth` variant, not a dressed-up `SplineBasis`.**
v2 already generalised `gam._assemble` to accept `Smooth = Union[SplineBasis,
TensorBasis]`. A random effect is **not** a spline (no continuous covariate to
re-evaluate; `SplineBasis`'s `n_basis/degree/penalty_order/lo/hi/knots` and the
`kind`-dispatch in `spline_design` are all meaningless for it). So add a **third
variant** rather than a `SplineBasis` with dummy fields:

```python
@dataclass(frozen=True)
class REBasis:                 # mgcv bs="re"  — a (design, penalty) block + level metadata
    design: Float['n q']       # one-hot(g)  (random intercept)  |  one-hot(g)*by (random slope)
    penalty: Float['q q']      # identity (ridge); λ = 1/σ_b² selected by the FS loop
    levels: int

def re_smooth(g, *, by=None) -> REBasis                    # the constructor
def factor_smooth_basis(x, g, *, k, bs='ps') -> Smooth     # mgcv bs="fs" (per-level smooth, shared λ)
```

`gam._assemble` / `smooth_partial_effect` gain an explicit `REBasis` branch
(no spline re-evaluation). The single FS-selected `λ` **is** the random-effect
precision (`λ = 1/σ_b²`) — the v1 "penalised GLM ≡ variance-components REML"
identity, now actually reachable.

**Performance routing (the §0.1 invariant applied).** A random-effect block
*widens the per-voxel GAM system by `q` = #levels*, so `gam_fit`'s rolled
Cholesky becomes `O((p+q)³)` per voxel. This is **cheap for few-level factors**
(site / scanner / batch, `q ~ 10–50` — the common "smooth(age) + random
intercept per site") and the right home for `re_smooth`. For **many-level**
factors (random intercept per *subject*, `q ~ 100–1000`) the dense GAM design is
the wrong solver; that is the structured §1.1 LME (R1/R2 block-Woodbury). The
GAMM dispatcher therefore routes a many-level RE to the LME engine, not a wide
GAM design. (A GAM *smooth* + many-level structured RE — coupled — is the harder
"GAMM with structured RE"; Tier-2.) The constructor stays the same; the engine
chooses.

**Substrate.** `re_smooth` = one-hot design + identity penalty (trivial);
`factor_smooth_basis` = the P-spline design replicated per level with a
block-shared penalty. Both reuse the `(design, penalty)` contract + FS loop.
**Effort: S (`re`) + M (`fs`).** **Oracle:** `mgcv::gam(y ~ s(x) + s(g,
bs="re"))` vs `lme4` for the random-intercept equivalence; factor-smooth
recovery. **Guard:** a GAM with no RE block lowers to the shipped `gam_fit` HLO.

---

## §3. Theme 3 — GAM basis completeness (beyond v2's tp/cc/te)

v2 shipped `ps` (v1), `tp`, `cc/cp`, `te/ti`. `nwx`'s `bs=` surface and common
neuroimaging smooths need:

### §3.1 By-variable factor-smooth interactions (`s(x, by=f)`)  *(high value)*

**What.** A separate smooth of `x` per level of factor `f` (mgcv `by=`), and the
varying-coefficient case (`by=` continuous). Maps `nwx`'s `SmoothSpec.by`.

**Why.** "Does the age trajectory differ by diagnosis?" — `s(age, by=dx)` — is a
core neurodevelopmental question. The design widens by `#levels × k`; cheap for
the few-level factors `by=` typically carries (diagnosis groups). **Effort: M**
(replicate the marginal design per level; reuse FS). **Oracle:** `mgcv` `by=`.

### §3.2 Cubic-regression (`cr`), Gaussian-process (`gp`), Markov-random-field (`mrf`)  *(future)*

**What.** `cr` (knot-based cubic regression, cheap 1-D), `gp` (Gaussian-process
smooth), and especially `mrf` (Markov random field over a parcel/vertex
adjacency — the natural smoother on a cortical mesh, **penalty = the `nitrix.graph`
Laplacian**, a clean substrate reuse). **Why.** `mrf` lets `nwx` express
spatially-structured effects over a surface adjacency. **Effort: S (`cr`) /
M (`gp`) / M (`mrf`).** **Oracle:** `mgcv` `bs='cr'|'gp'|'mrf'`.

### §3.3 Shape-constrained & adaptive smooths  *(future)*

Monotone (`scam`-style) and spatially-adaptive smoothing parameters. **Effort:
M-L.** Lowest priority; named for completeness.

---

## §4. Theme 4 — GLM/GAM family completeness + distributional models

**What.** v1 ships `GAUSSIAN`, `BINOMIAL`, `POISSON` (`_family.py`) plus a
custom-`Family` hook. Add, in three effort classes (the review's correction —
they are *not* uniform):

- **`S` each — a frozen `Family` on the existing scalar IRLS core:** **Gamma**
  (positive continuous: RT, volumes), **negative binomial** (over-dispersed
  counts; +1 estimated dispersion), **Beta** (proportions: tissue fractions).
- **`M` each — fiddly density / extra structure on the scalar IRLS:** **Tweedie**
  (compound; variance `V(μ)=μ^p`, but the deviance / log-likelihood normalising
  constant needs a series evaluation and a profiled power `p`).
- **`M-L` — a second linear predictor (not the scalar IRLS):** **ordinal /
  multinomial** (cumulative / multiple linear predictors) and the
  **location-scale / distributional** families (`gaulss`-style: model `σ ~ …` as
  well as the mean, for `nwx`'s reserved `sigma ~ …` part).

**Why.** `{{ family=gamma|nb|tweedie|betar }}` are one-line requests in `nwx`.

**Substrate.** The `S`/`M` families slot into the `_FAMILIES` registry (v2 H4) +
IRLS core. Distributional / ordinal need the two-predictor penalised-IRLS
extension. **Oracle:** `statsmodels` GLM families; `mgcv` `gaulss`/`twlss`/`ocat`.

---

## §5. Theme 5 — Residualisation modes

### §5.1 Non-aggressive (partial / AROMA) residualisation  *(BLOCKING; cheap)*

**What.** `linalg.residualise` (`linalg/residual.py:159`) does **full
projection** only (`return_mode='residual'|'projection'`). Add the
signal-preserving partial scheme: jointly fit `Y ~ [signal | noise]` and remove
only the noise-*unique* fitted contribution — preserving variance shared between
signal and noise (ICA-AROMA non-aggressive denoising):

```python
def partial_residualise(Y, *, signal, noise, l2=0.0) -> Y_clean
#   beta = lstsq([signal | noise], Y);  Y_clean = Y - noise @ beta_noise
```

**Why.** `nwx`'s `~|` (non-aggressive default) + `signal()`/`noise()` role
markers lower onto exactly this; `~|!` (aggressive) reuses the shipped
`residualise`. Aggressive cleaning discards task-correlated variance — the very
failure mode AROMA's non-aggressive mode avoids.

**Substrate.** One joint least-squares solve (reuse the `residualise`
Cholesky/SVD path) + a column-subset reconstruction; cuSOLVER-free,
differentiable, batched. Algebraic identity: equals the full-projection special
case when `signal = ∅`. **Effort: S.** **Oracle:** `fsl_regfilt` non-aggressive.

### §5.2 Soft / shrunk residualisation  *(future)*

A ridge-shrunk variant (`l2 > 0`, or James-Stein shrinkage of the nuisance fit)
for ill-conditioned confound sets — composes the existing `residualise` `l2`
path with §5.1. **Effort: S.** **Oracle:** ridge-regression residual identity.

---

## §6. Theme 6 — Robust & sandwich inference

### §6.1 Robust regression / M-estimators  *(adjacent — promote existing doc)*

Already specified in [`robust-statistics.md`](robust-statistics.md) (SPEC §12.7;
v2 §9 names it a future v3 candidate): `huber_regress`, `tukey_bisquare_regress`,
`mad`, as IRLS over the shipped least-squares solve. **v3 names `nwx` as the
concrete driver** (`{{ estimator=robust }}`) — this FR elevates priority and
records the consumer, it does not re-specify. **Effort: S.** **Oracle:**
`statsmodels.RLM`.

### §6.2 Sandwich / cluster-robust standard errors  *(high value)*

**What.** Heteroscedasticity-consistent (`HC0`–`HC3`) and cluster-robust vcov for
the GLM `t`/`F` contrasts: `vcov = (XᵀX)⁻¹ (Σ_i w_i x_i x_iᵀ) (XᵀX)⁻¹`. Surface:
a `vcov=` option on `glm_fit` / the contrast functions (the §0.1 strategy-arg
pattern — default `vcov='model'` reproduces today's contrast exactly).

**Why.** `nwx` exposes `{{ vcov=hc3 }}` / `{{ vcov=cluster(subject) }}`. **Substrate.**
A reweighted cross-product around the existing unscaled covariance; per-voxel,
cuSOLVER-free. The sandwich weights are data-dependent constants (not
differentiated through). **Effort: M.** **Oracle:** `statsmodels`
`cov_type='HC3'|'cluster'`.

---

## §7. Theme 7 — Parametric (random-field-theory) inference  *(future)*

**What.** Gaussian random-field peak- and cluster-level FWE (the SPM/FSL
parametric correction) as an alternative to the v1/v2 permutation engine: RESEL
smoothness estimation from the residuals + the GRF p-value formulas. **Why.**
When permutation is too expensive or exchangeability is awkward, RFT is the
standard parametric route; complements `randomise`. `nwx` exposes
`{{ inference=parametric(correction=rft) }}`. **Effort: L.** **Oracle:** SPM
`spm_P_RF`; FSL `cluster` GRF mode.

---

## §8. Theme 8 — Model selection / cross-validation  *(future)*

**What.** GCV as an alternative smoothing-parameter criterion (vs the shipped
Fellner-Schall REML — a parallel outer objective `n·RSS/(n−edf)²`), and a thin
k-fold / LOO scaffold over the mass-univariate fit for predictive scoring
(CBPM / prediction regime). AIC/BIC/LRT already exist (`glm.compare_models`).
**Why.** `nwx`'s reserved predictive/`method=` extensions and `{{ select=gcv }}`.
**Effort: M.** **Oracle:** `mgcv` `method='GCV.Cp'`; sklearn CV scores.

---

## §9. Proposed module layout (additions)

```
nitrix/stats/
  lme/
    _varcomp.py     # R1 AI-REML (unchanged); + R2/R3 block-Woodbury V^-1/logdet (§1.1)
    _lowrank.py     # R1 q-rank (unchanged)
    reml.py         # reml_fit (UNCHANGED — the R1 public entry + fast path)
    lme_fit.py      # NEW: the structure-dispatching general entry (§0.1 ladder R0–R4) (§1.1);
                    #      lme_t_contrast / lme_f_contrast (§1.3)
    glmm.py         # NEW: PQL/Laplace GLMM = lme_fit dispatch wrapped in _irls (§1.2)
    _corr.py        # NEW: AR1/CAR1/CS residual structures + varIdent/varPower (§1.4)
  basis.py          # + REBasis variant + re_smooth / factor_smooth_basis (§2);
                    #   by= plumbing (§3.1); cr/gp/mrf (§3.2)
  gam.py            # + REBasis branch in _assemble / smooth_partial_effect;
                    #   many-level RE routes to the LME engine (§2 routing)
  _family.py        # + Gamma/NegBinomial/Beta (S); Tweedie (M); ordinal/distributional (M-L) (§4)
  glm.py            # + vcov= (HC0-HC3, cluster) on fit/contrasts, default 'model' (§6.2)
  robust.py         # NEW: M-estimators (§6.1 — per robust-statistics.md)
  inference/
    rft.py          # NEW: random-field-theory peak/cluster FWE (§7)
linalg/
  residual.py       # + partial_residualise (non-aggressive / AROMA) (§5.1)
```

`LMEResult` / `LMEContrast` are new registered-pytree result records (house
style). `reml_fit`'s `REMLResult` is kept; `lme_fit` returns `LMEResult` (a
superset carrying the per-voxel `(Xᵀ V⁻¹ X)⁻¹` and `cov(θ̂)` that §1.3 needs).

## §10. Phasing & priority (tiered by `nwx` v1 blocking)

**Tier 0 — BLOCKING for `nwx`'s v1 GL(A)MM guarantee, and cheap** (do first):

- **§2 GAMM surfacing** (`REBasis` + `re_smooth` + `gam_fit` branch) — S, closes
  a v1-claimed gap, unblocks the common few-level mixed model with **zero new
  solver work** (and routes many-level RE to §1.1).
- **§5.1 non-aggressive residualisation** — S, unblocks `~|` semantics.
- **§1.3 mixed-model inference (Satterthwaite)** — M, surfaces already-computed
  `(Xᵀ V⁻¹ X)⁻¹` + `cov(θ̂)`; cheap.
- **§1.1 R1→R2** — M, the structure-dispatched `lme_fit` with the cheap
  single-factor block-Woodbury (covers `(1|g)`, `(1+x|g)`, `(x||g)`), `reml_fit`
  fast path preserved by the no-regression guard.

**Tier 1 — high value (rounds out the GL(A)MM surface):**

- §1.1 R3 (nested); §1.2 GLMM (PQL); §1.4 AR1/CAR1; §3.1 by-variable smooths;
  §4 `S`-class families (Gamma/NB/Beta); §6.2 sandwich/cluster SEs.

**Tier 2 — future / heavier:**

- §1.1 R4 (crossed/multiple, HLO-gated); §1.3 Kenward-Roger; §1.2 Laplace;
  §3.2–§3.3 cr/gp/mrf/monotone/adaptive; §4 Tweedie + ordinal/distributional;
  §5.2 soft residualisation; §6.1 robust (promote `robust-statistics.md`); §7
  RFT; §8 GCV/CV.

Each tier validates against its pinned oracle (`lme4`/`lmerTest`/`pbkrtest`,
`nlme`, `mgcv`, `statsmodels`, SPM/FSL) kept in `tests/` (SPEC §5.2), **with the
§0.1 no-regression guard** (the shipped configuration still lowers to the shipped
op) plus the no-large-intermediate HLO audit and cuSOLVER-free guard on every new
mass-univariate op (per v1/v2 discipline).

## §11. Open decisions (mostly resolved by the review)

- **§1.1 component representation** — RESOLVED: ship the dispatch ladder; R2
  (single factor, diagonal *and* unstructured `r×r`, block-Woodbury) is the
  Tier-0 win; R4 (crossed/multiple) is Tier-2 and explicitly gated. Do not start
  with a dense general `V` — start with the structure that has a cheap exact
  solver.
- **§2 surface** — RESOLVED: a third `Smooth` variant `REBasis` (not a
  `SplineBasis` with dummy fields); `re_smooth(g, by=)` constructor. A `random=`
  keyword on `gam_fit` may be added as thin sugar over the same block if the
  engine prefers it.
- **API shape** — RESOLVED: **additive**. `reml_fit` is unchanged (R1 entry +
  validated fast path); the general `lme_fit` dispatches and *calls into* the
  same engines, so no break and no regression.
- **§1.3 dof** — Satterthwaite is the cheap per-voxel Tier-0 default; Kenward-Roger
  (heavier vcov correction) is Tier-2 / single-model.
- **GLMM estimator** — PQL first (cheap, documented bias for binary/low-count);
  Laplace Tier-2.

## §12. Cross-references

- v1 ledger (shipped): [`stats-modelling-suite.md`](stats-modelling-suite.md).
- v2 ledger (shipped, merged to `main`):
  [`stats-modelling-suite-v2.md`](stats-modelling-suite-v2.md).
- Robust statistics (promoted here as Tier 2, `nwx`-driven):
  [`robust-statistics.md`](robust-statistics.md) (SPEC §12.7).
- Driver spec: `gramform/docs/nwx/spec.md` (the `ModelSpec` IR these kernels
  lower from).
- Live substrate touched: `src/nitrix/stats/lme/{reml,_varcomp,_lowrank}.py`,
  `stats/gam.py` (`:13-14`), `stats/basis.py` (`:291`), `stats/_family.py`,
  `stats/glm.py`, `linalg/residual.py` (`:159`).
- Dispatch precedents (§0.1): `stats._smalllinalg.small_inv_logdet`,
  `linalg._eigsolve.eigsolve_top_k`, `stats.pca.pca_fit(solver=)`,
  `gam_fit` Gaussian cross-product path, `reml_fit(low_rank=)`.
- Design: [`../design/stats.md`](../design/stats.md),
  [`../design/lme.md`](../design/lme.md).
- Governing spec: SPEC §1, §5.2; SPEC_UPDATE_v0.5 §1 (score-kernel boundary).

## §13. Engineering-review record (incorporated 2026-06-17)

Three-lens review of the as-submitted spec (commit `2962c19`); findings folded
into the body above.

**Rigour.** Codebase claims verified accurate (two-component `reml_fit`;
`residualise` full-projection-only; no `gam_fit` RE entry; `basis` kind set;
the `gam.py:13-14` GAMM doc-vs-surface gap). Corrections made: (a) §1.1
conflated 2–3 linear-algebra complexity classes under one effort/cuSOLVER claim
— now the explicit R0–R4 ladder; (b) "already the `_lowrank` shape with K thetas"
was misleading (multi-factor `V` has no shared eigenbasis) — removed; (c) §4
"S each" was wrong for Tweedie (fiddly deviance) and ordinal/multinomial (second
predictor) — re-tiered into S/M/M-L; (d) differentiability of dof / sandwich
weights now noted per-item (not differentiated through).

**Performance.** Headline: the v1/v2 cheapness is FaST-LMM-specific to one
grouping factor; it does **not** generalise to crossed/multiple factors (the
`(p+Σq_g)³`-class sparse MME). The spec's "tiny `r≤3` → cuSOLVER-free" holds for
R1/R2 only. Resolved by the §1.1 dispatch ladder (cheap tiers first; R4 gated)
and the §2 routing (many-level RE → structured LME, not a wide GAM design).
§1.3/§5.1 confirmed genuinely cheap (reuse already-computed quantities).

**Abstraction.** (a) A random effect is not a `SplineBasis` — made a third
`Smooth` variant `REBasis` (follows the v2 `TensorBasis` precedent), not a
dummy-field spline. (b) Prefer additive API — `reml_fit` unchanged, new
`lme_fit` dispatcher — over a breaking signature change. (c) Strategy-arg seams
(`structure=`, `vcov=`, `corr=`, `family=`) confirmed consistent with the
suite's existing dispatch pattern.

**The load-bearing addition (§0.1).** Performance-preserving dispatch is now a
first-class invariant: every general entry routes to the cheapest exact solver
and the shipped fast paths stay the realised path for the configurations they
serve, proven by a no-regression HLO guard. This is the design spine that lets
v3 expand coverage without regressing the common case.
