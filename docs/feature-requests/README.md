# nitrix feature requests — index

Active feature requests live here, **one doc per proposal**. **Check this
index before filing a new one** — it is the duplicate-issue guard: if a
primitive or fix already has a doc, add to that doc rather than opening a
parallel one.

Four families of doc live here, each with a *context/ledger* doc that holds
the shared framing + history and indexes its atomised items:

| Family | Context/ledger doc | What it is |
|---|---|---|
| Consumer-pipeline substrate | [`ilex-pipeline-substrate.md`](ilex-pipeline-substrate.md) | Kernels a named downstream (ilex → thrux) is blocked or workaround-laden without |
| Consumer training-substrate | [`ilex-training-substrate.md`](ilex-training-substrate.md) | Augmentation / loss / model numerics ilex grew after 2026-06-02 (2026-06-08 audit) |
| Consumer forward-block kernels | [`nn-forward-block-kernels.md`](nn-forward-block-kernels.md) | The transformer / state-space forward blocks (attention, selective-scan) — the first NN-kernel tier (2026-06-23) |
| Statistical modelling suite | [`stats-modelling-suite.md`](stats-modelling-suite.md) | Mass-univariate GLM/GAM/GAMM, LME size-dispatch, TFCE `randomise` — perf-bench + ModelArray + niffi |
| `hypercoil-examples` migration | [`hypercoil-examples-migration.md`](hypercoil-examples-migration.md) | Theory-faithful kernels salvaged from the deprecated examples repo (functional alignment, vMF directional stats, whitening, DCBC, synthetic generators) — clean-room from the literature, **not** ports |
| Internal engineering backlog | [`internal-backlog.md`](internal-backlog.md) | Parked perf / Pallas / API-refinement items, each gated on a **Trigger** |
| Doc-drift / correctness fixes | [`perf-bench-feedback.md`](perf-bench-feedback.md) | Mechanical docstring fixes (file:line-pinned), *not* primitive proposals |
| docs/feature-requests catalogue §12 brainstorm catalogue | `docs/feature-requests catalogue §12` (origin) | Substrate-compatible *candidate* primitives; promotion gated by `§13` |

Status is verified against the live `src/nitrix` surface as of 2026-06-02
(interpolation-method dispatcher — `Lanczos` / `MultiLabel` /
`NearestNeighbour` on `resample` / `spatial_transform` — shipped 2026-06-07;
`CubicBSpline` (scipy `order=3` bit-exact) added 2026-06-08; see
`IMPLEMENTATION_PLAN.md §10.3`).

### Active vs. resolved

Fully **shipped / resolved** atomic FRs are archived under
[`resolved/`](resolved/) to keep the active backlog scannable — the doc is kept
as a provenance record, out of the working view. The family sections below still
list them (annotated *✅ SHIPPED / RESOLVED*, linking into `resolved/`) so each
section reads as a complete ledger and the duplicate-issue guard stays intact; a
single flat archive index is at the [bottom](#resolved-archived). What remains
**un-archived in this directory is genuinely open work**: the suite-scoping
ledgers/registers, the trigger-gated internal backlog, the `§12` brainstorm
candidates, and the actionable per-suite items. (Post-release this whole scheme
is expected to migrate to GitHub issues; the `resolved/` split is an
internal-dev convenience until then.)

## Consumer-pipeline substrate (ilex → thrux)

Context, scope boundary, and already-shipped record:
[`ilex-pipeline-substrate.md`](ilex-pipeline-substrate.md).

| Doc | Severity | Home |
|---|---|---|
| [connected-components](resolved/connected-components.md) | ENABLING (highest recurrence) | `morphology` |
| [pad-to-multiple](resolved/pad-to-multiple.md) | ENABLING | `numerics` |
| [crop-to-nonzero](resolved/crop-to-nonzero.md) | ENABLING | `numerics` |
| [cubic-resample](resolved/cubic-resample.md) | MISMATCH (parity) — ✅ **resolved**: `CubicBSpline` (scipy `order=3` bit-exact) (2026-06-08) | `geometry._interpolate` |
| [intensity-normalize-variants](resolved/intensity-normalize-variants.md) | CONVENIENCE | `numerics.normalize` |
| [sliding-window-weighting](resolved/sliding-window-weighting.md) | CONVENIENCE | `numerics` |
| [point-sample](resolved/point-sample.md) | CONVENIENCE — **partial**: capability via `spatial_transform(method=Linear(), mode='constant')`; flat-point-list wrapper unshipped (2026-06-07) | `geometry.grid` |
| [compute-vertex-normals](resolved/compute-vertex-normals.md) | CONVENIENCE | `sparse.mesh` |
| [upsample-nearest-nd](resolved/upsample-nearest-nd.md) | CONVENIENCE — ✅ **addressed**: `resample(method=NearestNeighbour())` (2026-06-07) | `geometry` |
| [spatial-transform-batched](resolved/spatial-transform-batched.md) | CONVENIENCE | `geometry.grid` |

## Consumer training-substrate (ilex → nitrix)

Context, boundary (post bitsjax–thrux–nitrix realignment), negative record,
and index: [`ilex-training-substrate.md`](ilex-training-substrate.md). From
the 2026-06-08 audit of the surfaces ilex grew **after** the 2026-06-02
survey — augmentation pipelines, the nimox loss library, and the numerics of
the nimox modules + newest models (`krakencoder`, `cortex_ode`, `surfnet`).

| Doc | Severity | Home |
|---|---|---|
| [lab2im-gmm-synthesis](resolved/lab2im-gmm-synthesis.md) | ENABLING | `augment` |
| [generative-bias-field](resolved/generative-bias-field.md) (simulated INU) | ENABLING | `augment` / `bias` |
| [intensity-augmentation-ops](resolved/intensity-augmentation-ops.md) | CONVENIENCE | `augment.intensity` |
| [gibbs-ringing](resolved/gibbs-ringing.md) | CONVENIENCE | `augment.intensity` |
| [geometric-augmentation-ops](resolved/geometric-augmentation-ops.md) | CONVENIENCE | `augment.geometric` |
| [dice-loss](resolved/dice-loss.md) | ENABLING | `metrics.dice` |
| [cross-entropy-focal](resolved/cross-entropy-focal.md) | CONVENIENCE (dedup ×3) | `metrics` |
| [contrastive-ssl-losses](resolved/contrastive-ssl-losses.md) | CONVENIENCE | `metrics` / `stats` |
| [field-regularisers](resolved/field-regularisers.md) | ENABLING | `register.regulariser` |
| [gaussian-kl-nll](resolved/gaussian-kl-nll.md) | CONVENIENCE | `stats` |
| [affine-matrix-algebra](resolved/affine-matrix-algebra.md) | ENABLING | `geometry.transform` |
| [pca-svd](resolved/pca-svd.md) | CONVENIENCE | `stats.pca` |
| [lp-normalize](resolved/lp-normalize.md) | CONVENIENCE | `numerics.normalize` |
| [mesh-laplacian-smoothing](resolved/mesh-laplacian-smoothing.md) | CONVENIENCE | `geometry.mesh` |

Expansions to existing docs (new drivers / scope from this audit):
[compute-vertex-normals](resolved/compute-vertex-normals.md) (+`cortex_ode`/`surfnet`),
[point-sample](resolved/point-sample.md) (arbitrary-point `grid_sample`; 3 dups; task
#138), [ode-integrators](ode-integrators.md) (per-vertex NODE; diffrax can't
follow into nitrix).

## Consumer forward-block kernels (nimox → nitrix)

Context, the new-family rationale (why attention/SSM weren't in the 2026-06-08
audit), the parity two-tier contract, and the priority-marked index:
[`nn-forward-block-kernels.md`](nn-forward-block-kernels.md). The **sequenced
implementation plan** is
[`nn-forward-kernels-suite.md`](nn-forward-kernels-suite.md) (namespace, golden
harness, per-op design, phasing). The first *neural-network forward-block*
kernel tier in nitrix — pure `(Array,…)->Array` contractions lifted out of the
ilex/nimox transformer and Mamba modules, where a hardware-aware (Pallas/Triton)
implementation pays off most. Consumer is `nimox` (being promoted to a standalone
repo in the ilex cycle; ilex axis ii). **All under one namespace `nitrix.nn`**
(`attention` / `ssm` / `norm` submodules); nitrix owns correctness + gross
memory, the perf suite (`bench/`) owns wall-clock parity at scale.

| Pri | Doc | Severity | Home |
|---|---|---|---|
| P0 | [attention-kernels](resolved/attention-kernels.md) (flash + windowed-bias + causal + cross) — ✅ **SHIPPED** (GPU-verified) | ENABLING | `nitrix.nn.attention` |
| P1 | [selective-scan](resolved/selective-scan.md) (Mamba/S6 fused scan) — ✅ **SHIPPED** (GPU-verified) | ENABLING | `nitrix.nn.ssm` |
| P3 | [attention-no-upcast-knob](attention-no-upcast-knob.md) (caller-controlled SDPA accumulation precision; nimox `diffusion_unet._Attention` consumer) | CONVENIENCE | `nitrix.nn.attention` |
| P2 | [affine-matrix-algebra](resolved/affine-matrix-algebra.md) / [spherical-parameterisation](spherical-parameterisation.md) / [field-regularisers](resolved/field-regularisers.md) *(existing — nimox-extraction blockers)* | ENABLING | `geometry` / `register` |
| P3 | [fused-norm-kernels](fused-norm-kernels.md) (fused LN/GN/IN, perf-only) | CONVENIENCE | `nitrix.nn.norm` |
| P3 | [nimox-mesh-loss-geometry](resolved/nimox-mesh-loss-geometry.md) (face normals / edge-face topology / seg-seg distance / chamfer NN — nimox mesh-loss consolidation; confirm equivalents to delegate, pairs with [mesh-spatial-acceleration](mesh-spatial-acceleration.md)) — ✅ **SHIPPED** (2026-06-25): delegation map confirmed; `sparse.face_normals` / `sparse.edge_face_adjacency` / `geometry.segment_segment_sq_dist` / `geometry.point_set_nearest_sq_dist` | CONVENIENCE (consolidation) | `geometry` / `sparse` |
| P3 | [nimox-differentiable-registration-layer](resolved/nimox-differentiable-registration-layer.md) (public implicit-diff registration returning a self-contained matrix — owns the centring conjugation nimox `AffineRegister(gradient='implicit')` hand-rolls; optional general-metric `implicit_minimize` path) — ✅ **SHIPPED** (2026-06-25): `register.register_implicit` (single-level) + `rigid_register_implicit` / `affine_register_implicit` (coarse-to-fine); forward driver hoisted behind a `LevelSolver` seam (byte-identical); SSD→`implicit_least_squares`, else→`implicit_minimize` | ENABLING (differentiable layer) | `register` / `linalg` |
| P3 | [nimox-histogram-match-fit-apply](resolved/nimox-histogram-match-fit-apply.md) (split `bias.histogram_match` into `fit(reference)->landmarks` / `apply(source, landmarks)` so nimox `HistogramMatch` carries ~9 floats not a reference volume) — ✅ **SHIPPED** (2026-06-25): `bias.histogram_match_fit` / `histogram_match_apply`; convenience is byte-faithful composition (§6.5) | CONVENIENCE (refinement) | `bias` |
| P3 | [nimox-stats-response-predict](resolved/nimox-stats-response-predict.md) (public `beta_predict` / `ordinal_predict` / `gaulss_predict` / `gam_predict` mirroring `stats.predict`, so nimox.estimators wraps the response regressions as fit/transform) — ✅ **SHIPPED** (2026-06-25): the four `*_predict`; `OrdinalResult.link` / `GAMResult.intercept` added as static aux (fit jaxpr byte-identical, zero fit-path cost) | CONVENIENCE (apply surface) | `stats.{betareg,ordinal,gaulss,gam}` |
| P3 | [nimox-stats-mixed-effects-predict](nimox-stats-mixed-effects-predict.md) (public BLUP `lme_predict` / `glmm_predict` — group-aware population/conditional prediction + a uniform cross-tier ranef accessor) — ✅ **SHIPPED (R1/R2/GLMM)** (2026-06-25): `lme_predict` / `ranef` / `glmm_predict`; population all tiers; conditional via opt-in `lme_fit(retain_blups=)` post-pass (default byte-identical, zero-cost). R4/R2+corr/R3 conditional staged (raise; population works) | ENABLING (apply surface) | `stats.{lme,glmm}` |
| P3 | [nimox-gp-fit-traceable-rho-search](resolved/nimox-gp-fit-traceable-rho-search.md) (make `gp_fit`'s REML lengthscale-search epilogue traceable — JAX-native `_parabolic_argmin` + traced `rho_hat`, dropping the host `np.asarray(nll_grid)` / `float()` at `gp.py:932-935`, so `gp_fit` `jit`/`vmap`s with `x` closed-over) — ✅ **SHIPPED (Tier-A)** (2026-06-25): `_parabolic_argmin_jax` + traced `rho_hat` + `jnp.log` in the result assembly; Gaussian HSGP `gp_fit` jit/vmaps (vmap==loop-of-eager); 53 GP tests green (fp-parity). **Follow-on OPEN (2026-06-25):** `hgp_fit` (the hierarchical GS model — nimox `HierarchicalGPRegressor` E8 consumer) has the **identical** Gaussian-HSGP rho-search, still eager-only (`hgp.py:71,515-518` reuse the old host `_parabolic_argmin`); the fix reuses the already-shipped `_parabolic_argmin_jax` (smaller than the original). Tier-B (traced-`x`) + non-Gaussian/exact paths deferred | ENABLING (jit/vmap the fit) | `stats.gp` |

## Statistical modelling suite (perf-bench → ModelArray → niffi)

Context, organising insight (penalised GLM ≡ variance-components REML ≡ mixed
model), scope boundary, locked decisions, phasing, and the atomised work-item
index: [`stats-modelling-suite.md`](stats-modelling-suite.md). One sprint
spanning three sibling requests that share one substrate.

| Workstream | Severity | Home | Driver |
|---|---|---|---|
| A — LME size-dispatch + cuSOLVER bypass | ✅ SHIPPED (merged to main) | `stats.lme._varcomp` | [lme-family-tiny-linalg-gpu-block-and-perf](lme-family-tiny-linalg-gpu-block-and-perf.md), [gpu-cusolver-first-call-handle-failure](resolved/gpu-cusolver-first-call-handle-failure.md) |
| B — mass-univariate GLM + GAM / GAMM | ✅ SHIPPED (merged to main) | `stats.{glm,gam,basis}` | ModelArray-parity request |
| C — TFCE `randomise` cluster correction | ✅ SHIPPED (merged to main) | `stats.inference` | niffi capability gap |

**v2 (shipped, `feat/stats-suite-v2`):**
[`stats-modelling-suite-v2.md`](stats-modelling-suite-v2.md) — deferred
completeness (LME q-rank; GAM thin-plate / cyclic / tensor-product bases +
shared-λ; randomise cluster-extent/mass + F-contrast + GPD tail) and the
regularised connectivity estimators **Ledoit-Wolf / OAS** and **graphical
LASSO**.

**v3 (proposed):** [`stats-modelling-suite-v3.md`](stats-modelling-suite-v3.md)
— GL(A)MM completeness driven by the **`nwx`** DSL (`gramform`): general
multi-/correlated random-effects REML, **GAMM surfacing** (the `re`/`fs` basis
v1 claimed but never exposed), mixed-model fixed-effect inference
(Satterthwaite / Kenward-Roger), AR1/CAR1 error structures, non-aggressive
(AROMA) residualisation, sandwich/cluster SEs, and extra GLM families.

**Audit register (2026-06-20):**
[`stats-suite-audit.md`](stats-suite-audit.md) — consolidated findings from a
seven-lens fan-out review of the *entire standing* stats suite (math correctness,
engineering rigour, neuroimaging use, organisation, design, performance,
hardware-awareness). Verdict: math sound, hardware-matched, no correctness bug;
a small set of **verified** bugs (B1 `precision` cuSOLVER, B2 `spline_design`
extrapolation, B3 `sandwich_cov` cluster count) + contract / perf / capability
items, each with a stable ID and Status column. Records the **empirical
refutation of the one Critical claim** (C1 "gappy labels bias REML" — bit-identical
fits; exact log|V| cancellation) so it is not re-raised.

**Spatial null models (family ledger, 2026-07-07):**
[`spatial-null-models.md`](spatial-null-models.md) — SA-preserving nulls for
map-correspondence inference (the audit's **N2**). The **spin / Moran /
BrainSMASH trio SHIPPED** (most-common / spectral / most-rigorous-parameterized)
on the shared `stats.inference.spatial_null_test` seam, plus the spin
**medial-wall + per-hemisphere + Váša bijective** refinements. Remaining: the
BrainSMASH large-mesh 'sampled' variant + parcel-level spin. Reference:
`neuromaps.nulls`.

**GP / HGP pre-merge review register (2026-06-22):**
[`stats-suite-review-gp.md`](stats-suite-review-gp.md) — consolidated findings from a
seven-lens, 13-reviewer / 53-agent **adversarially-verified** fan-out review of the
`feat/stats-gp` branch (PR1–PR10) *plus* a suite-wide re-audit. Verdict:
**ship-after-fixes** — math sound and cuSOLVER-free/jit-clean verified, but **7 merge-gating
blockers** (silent-wrong-result-on-bad-input traps + a `block=`-ignored ρ-search/GAM-epilogue
OOM cliff), each with a stable ID and Status column, sequenced into Round 1 (this PR) →
Round 4. Records the **2 refuted claims** so they are not re-raised, and cross-references the
standing audit (CV1 = its **N2** surface-TFCE; MC4 deepens its **M1**).

**GP / HGP models (proposed, 2026-06-20, rev. HSGP-primary):**
[`gaussian-process-models.md`](gaussian-process-models.md) — Gaussian-process
regression as a first-class family (`nitrix.stats.gp`) and its **hierarchical**
(multi-level, mixed-model sense) extension. Substrate is ~70 % shipped: a 1-D
reduced-rank kriging GP *smooth* (`gp_basis` + the generic Fellner–Schall loop)
fits today, `reml_fit`'s FaST-LMM is a fixed-kernel GP in disguise (`K`↔`ZZᵀ`),
and the cuSOLVER-free Cholesky+log-det / correlated-residual (`gls_fit corr=`)
prims exist. **Primary engine: the Hilbert-space approximate GP (HSGP)** — a
fixed Laplace-eigenfunction basis where the lengthscale enters only as a
diagonal spectral reweighting, so `ρ`-estimation stays `eigh`-free and inside
the suite's fast paths (it *dissolves* the kriging eigenbasis crux rather than
working around it), handles multi-D / ARD natively, and composes with AR1/CAR1
residuals for joint trend + autocorrelation fits. Net-new: the HSGP basis +
kernel spectral densities, and the `gp_fit`/`GPResult` + HGP factor-smooth
wrappers (Tier 3 composes the v3 §2/§3.1 GAMM surface). Blast radius on existing
code ≈ nil (additive); ~3–4 weeks to full (a)-scope. Full-Bayes priors on `ρ` /
the HBR / normative-modelling flavour (b) — the `brms`/`mvgam` territory — are
deferred to a separate proposal.

## Registration suite (`nitrix.register`)

Context/ledger + audit roadmap:
[`registration-suite.md`](registration-suite.md) — frames the suite
(rigid/affine, log-Demons / greedy-SyN, volreg, BBR), credits the shipped
substrate, indexes the atomised registration FRs below (the duplicate guard),
and specs the **genuinely-new gaps** from a six-lens audit (2026-06-22, 42→40
verified findings). **Numerics-only; image/file I/O → `thrux`.**

Atomised registration FRs (add to these, don't duplicate): metric-ADT /
transform-model ([`registration-typing-metric-adt`](resolved/registration-typing-metric-adt.md)),
matrix per-iter perf levers ([`registration-matrix-recipe-perf-levers`](registration-matrix-recipe-perf-levers.md)),
cold compile ([`registration-recipe-cold-compile`](resolved/registration-recipe-cold-compile.md)),
transparent differentiability ([`registration-recipe-transparent-differentiability`](registration-recipe-transparent-differentiability.md)),
early-stop while-loop ([`registration-early-stopping-while-loop`](resolved/registration-early-stopping-while-loop.md)),
v3 follow-ups ([`registration-suite-v3-followups`](registration-suite-v3-followups.md)),
affine small-grid ([`register-affine-small-grid-divergence`](resolved/register-affine-small-grid-divergence.md)),
demons 0/0 ([`register-demons-force-divide-by-zero`](resolved/register-demons-force-divide-by-zero.md)),
metric conventions ([`metrics-convention-vs-domain-tools`](resolved/metrics-convention-vs-domain-tools.md)),
Pallas ESM force ([`pallas-demons-esm-force`](pallas-demons-esm-force.md)),
Mosaic GPU kernels ([`mosaic-hopper-registration-kernels`](mosaic-hopper-registration-kernels.md)),
field regularisers ([`field-regularisers`](resolved/field-regularisers.md)),
**functional alignment / ProMises** ([`register-functional-alignment`](resolved/register-functional-alignment.md);
new capability — alignment in representation space; from the
[`hypercoil-examples` migration](hypercoil-examples-migration.md); its solver is
the linalg dependency [`linalg-orthogonal-procrustes`](resolved/linalg-orthogonal-procrustes.md)).

## Dynamical-systems / DE suite (`nitrix.numerics`)

Forward-looking scoping ledger:
[`dynamics-suite.md`](dynamics-suite.md) — frames the differentiable
**differential-equation integration substrate** that would unlock a subdiscipline
of neuroimaging dynamics (DCM, neural→haemodynamic forward models, TVB-style
whole-brain network simulation, brain digital twins), credits the shipped
substrate (the fixed-step ODE family + linalg/noise/signal plumbing), indexes the
atomised DE FRs below (the duplicate guard), and specs the **genuinely-new
integrator gaps** (DS-1 SDE family · DS-2 Brownian/keyed-noise contract · DS-3
exponential/local-linearization · DS-4 symplectic leapfrog · DS-5 DDE).
**Numerics-only**: integrators + plumbing are nitrix; every *model* and
*inversion framework* → downstream (nimox / ilex). Mostly **prospective** — each
DS item is gated on a named blocked consumer (SPEC §9).

Atomised DE FRs (add to these, don't duplicate): ODE family + adaptive/adjoint
roadmap ([`ode-integrators`](ode-integrators.md), the seed), fixed-point /
implicit combinators ([`fixed-point-combinators`](resolved/fixed-point-combinators.md)),
Krylov solvers ([`krylov-solvers`](krylov-solvers.md), the non-symmetric resolvent
for spectral DCM), matrix functions ([`matrix-functions`](matrix-functions.md)),
heat-kernel diffusion ([`heat-kernel-diffusion`](resolved/heat-kernel-diffusion.md), linear
network spreading).

## Internal engineering backlog

Ledger (framing, closed-by-design, resolved):
[`internal-backlog.md`](internal-backlog.md).

| B# | Doc | Kind |
|---|---|---|
| B2 | [perf-bench-sprint-surfaces](perf-bench-sprint-surfaces.md) | perf characterisation |
| B3 | [pallas-dispatch-edge-aggregate](pallas-dispatch-edge-aggregate.md) | Pallas dispatch (measured: not GPU-motivated) |
| B4 | [edge-aggregate-log-euclidean](edge-aggregate-log-euclidean.md) | semiring coverage (canonical entry) |
| B5 | [keops-genred-research](keops-genred-research.md) | research note |
| B6 | [pallas-gaussian-blur](pallas-gaussian-blur.md) | Pallas kernel |
| B7 | [pallas-trilinear-resample](pallas-trilinear-resample.md) | Pallas kernel (interim JAX gather **shipped** 2026-06-07; kernel parked) |
| B10 | [retune-pallas-log-matmul](retune-pallas-log-matmul.md) | kernel tuning (M) |
| B11 | [perfbench-migration](perfbench-migration.md) | tooling migration (in progress) |
| B12 | [iir-filter-gpu-backend](resolved/iir-filter-gpu-backend.md) | perf / API-default (IIR GPU backend, S+M) |
| B13 | [boundary-mode-parity](boundary-mode-parity.md) | API refinement (scipy/ITK boundary parity, M) |
| B14 | [spectral-embedding-gpu-solver](resolved/spectral-embedding-gpu-solver.md) | perf + robustness (lobpcg/eigh GPU solver, M) |
| B15 | [interpolation-backend-cpu-gpu-gap](interpolation-backend-cpu-gpu-gap.md) | perf characterisation (map_coordinates CPU/GPU) |
| B16 | [alternative-interp-backends-xla](alternative-interp-backends-xla.md) | research note (scipy/cupy interp backends in XLA) |
| B17 | [median-percentile-cpu-sort-cliff](median-percentile-cpu-sort-cliff.md) | perf characterisation (jnp.median/percentile CPU sort) |
| G1 | [spatial-transform-linear-extrap](spatial-transform-linear-extrap.md) | boundary-mode extension (S) |

## Doc-drift / correctness fixes

Ledger: [`perf-bench-feedback.md`](perf-bench-feedback.md). These are
mechanical docstring fixes, not primitive proposals. All below are ✅ resolved
and archived under [`resolved/`](resolved/).

| Doc | Site |
|---|---|
| [doc-lomb-scargle-normalisation](resolved/doc-lomb-scargle-normalisation.md) | `signal/lomb_scargle.py:154` |
| [doc-lomb-scargle-eigh-factorisation](resolved/doc-lomb-scargle-eigh-factorisation.md) | `signal/lomb_scargle.py:43–49` |
| [doc-lomb-scargle-cpu-eigh-caveat](resolved/doc-lomb-scargle-cpu-eigh-caveat.md) | `linalg/_solver.py:147` |
| [doc-tsconv-cross-correlation](resolved/doc-tsconv-cross-correlation.md) | `signal/tsconv.py:45` |
| [doc-lomb-scargle-interpolate-intended-use](resolved/doc-lomb-scargle-interpolate-intended-use.md) | `signal/lomb_scargle.py:~264–359` |
| [doc-op-matrix-inventory-gaps](resolved/doc-op-matrix-inventory-gaps.md) | `docs/op_matrix.json` (`ops`) |
| [doc-gaussian-kernel-gamma](resolved/doc-gaussian-kernel-gamma.md) | `linalg/kernel.py:38` |
| [doc-relaxed-modularity-newman-factor](resolved/doc-relaxed-modularity-newman-factor.md) | `graph/community.py:316,445,622` (landed as a behaviour fix) |
| [doc-iir-backend-default](resolved/doc-iir-backend-default.md) | `signal/_iir.py` (module docstring) |
| [doc-instantaneous-frequency-narrowband-caveat](resolved/doc-instantaneous-frequency-narrowband-caveat.md) | `signal/fourier.py` (`instantaneous_phase` / `_frequency`) |

## docs/feature-requests catalogue §12 brainstorm catalogue (candidate primitives)

`§` = origin entry in `SPEC §9` (the canonical origin record; the
`§13` acceptance protocol references items by number). **Status** is against
live code: *not started* / *partial* (some substrate shipped).

| § | Doc | Proposed module | Effort | Status |
|---|---|---|---|---|
| 12.1 | [krylov-solvers](krylov-solvers.md) | `linalg.krylov` | M | partial (`cg` shipped — registration) |
| 12.2 | [matrix-functions](matrix-functions.md) | `linalg.matrix_function` | S | partial (`sym*` + `matrix_exp` shipped) |
| 12.3 | [heat-kernel-diffusion](resolved/heat-kernel-diffusion.md) | `graph.heat_kernel` | S | ✅ shipped (`heat_kernel` exp/eigh; `diffusion_embedding`) |
| 12.4 | [sinkhorn-optimal-transport](sinkhorn-optimal-transport.md) | `transport` | M | not started |
| 12.5 | [discrete-exterior-calculus](discrete-exterior-calculus.md) | `geometry.dec` | M | partial (cotangent Laplacian shipped) |
| 12.6 | [mesh-curvature](resolved/mesh-curvature.md) | `geometry.curvature` | S | ✅ shipped (geometry suite: mean/gaussian/principal_curvatures) |
| 12.7 | [robust-statistics](robust-statistics.md) | `stats.robust` | S | not started |
| 12.8 | [fixed-point-combinators](resolved/fixed-point-combinators.md) | `numerics.fixed_point` | M | ✅ shipped (`fixed_point_solve` — registration) |
| 12.9 | [spherical-harmonic-transform](spherical-harmonic-transform.md) | `geometry.sphere.harmonics` | M | not started |
| 12.10 | [compensated-summation](compensated-summation.md) | `numerics.precision` | S | not started |
| 12.11 | [ode-integrators](ode-integrators.md) | `numerics.ode` | L | partial (`integrate_velocity_field` shipped) |
| 12.12 | [continuous-wavelet-transform](continuous-wavelet-transform.md) | `signal.cwt` | S | not started |
| 12.13 | [graph-wavelet-transform](graph-wavelet-transform.md) | `graph.wavelet` | S | not started (blocked on 12.2) |
| 12.14 | [graphical-lasso](resolved/graphical-lasso.md) | `stats.glasso` | M | ✅ shipped (`glasso` / `glasso_path`, v2 Phase 3) |
| 12.15 | [surface-resample-adap-bary](resolved/surface-resample-adap-bary.md) | `geometry.sphere.resample` | M | ✅ shipped (`surface_resample` `adap_bary_area` + `barycentric`) |
| 12.16 | [surface-boundary-map](resolved/surface-boundary-map.md) | `graph.parcellation.boundary` | S | ✅ shipped (`surface_boundary_map`) |
| 12.17 | [mesh-watershed](resolved/mesh-watershed.md) | `graph.parcellation.watershed` | M | ✅ shipped (`mesh_watershed`) |
| 12.18 | [clustering-primitives](clustering-primitives.md) | `numerics.cluster` | S/M | partial (`kmeans` shipped; `ward_linkage`/`nmf` deferred) |
| 12.19 | [normalised-cut](resolved/normalised-cut.md) | `graph.normalized_cut` | XS | ✅ shipped (`normalized_cut` = eigenmap + kmeans) |

**§12.20** (functional-parcellation strategy survey) is informational — a
strategy→primitive mapping table, not a primitive — and stays in
`docs/feature-requests catalogue §12.20`. The parcellation docs (12.16–12.19) link to it.

**Candidate not yet in docs/feature-requests catalogue §12:**
[ledoit-wolf-shrinkage](resolved/ledoit-wolf-shrinkage.md) (`stats.ledoit_wolf`, effort
S) — analytic shrinkage covariance; sibling of 12.14 glasso and nilearn's
*default* connectome estimator. **✅ shipped** (`stats.ledoit_wolf` / `stats.oas`,
v2 §4.1).

[linalg-orthogonal-procrustes](resolved/linalg-orthogonal-procrustes.md) (`linalg`,
effort S) — **✅ shipped**: orthogonal Procrustes + `subspace_angles`
(canonical/Grassmann angles) + `image_basis` (ranked range basis); the
SVD-of-cross-product subspace-geometry family. Lead item `orthogonal_procrustes`
is the solver under
[`register-functional-alignment`](resolved/register-functional-alignment.md); from the
[`hypercoil-examples` migration](hypercoil-examples-migration.md).

[clifford-geometric-algebra](clifford-geometric-algebra.md) (`algebra` /
`geometry.clifford`, effort M/L) — **PROPOSED, SPEC-review-gated (no code)** —
geometric-algebra vocabulary (multivector geometric product, rotor exp/log, the
sandwich transform, grade projection / involutions), the pure-function substrate
under equivariant geometric learning (GATr lineage). Distinct from the shipped
`geometry.algebra` (*transform* algebra of homogeneous matrices); GATr layers
stay downstream. Needs an explicit go-ahead + a lined-up consumer.

### Dependency edges (within the §12 catalogue)

```
12.1 krylov ──┬─> 12.8 fixed-point ──┬─> 12.18 clustering ──> 12.19 ncut
              │                       └─> 12.14 glasso (impl-VJP)
              └─> 12.11 ode (adjoint)
12.2 matrix-fn ─┬─> 12.3 heat-kernel
                └─> 12.13 graph-wavelet
12.16 boundary ──> 12.17 watershed
```

## Resolved (archived)

The 73 FRs below are fully shipped/resolved and live in [`resolved/`](resolved/). Listed here as one flat index; topical context and the shipped-status annotation remain in the family sections above.

- [`affine-matrix-algebra`](resolved/affine-matrix-algebra.md) — Affine matrix algebra (geometric convention) — `nitrix.geometry.affine`
- [`attention-kernels`](resolved/attention-kernels.md) — attention-kernels — scaled dot-product / flash attention
- [`augment-synthetic-connectivity`](resolved/augment-synthetic-connectivity.md) — Synthetic connectivity & time-series generators in `nitrix.augment`
- [`catmull-rom-interpolator`](resolved/catmull-rom-interpolator.md) — CatmullRom interpolator — `nitrix.geometry._interpolate`
- [`compute-vertex-normals`](resolved/compute-vertex-normals.md) — `compute_vertex_normals` — `nitrix.sparse.mesh`
- [`connected-components`](resolved/connected-components.md) — Connected-components / largest-component labelling — `nitrix.morphology`
- [`connectopy-symmetric-degree-normalisation`](resolved/connectopy-symmetric-degree-normalisation.md) — Connectopy normalises by the wrong degree on asymmetric graphs (symmetrises the Laplacian, no…
- [`contrastive-ssl-losses`](resolved/contrastive-ssl-losses.md) — Contrastive / self-supervised losses — `nitrix.metrics` / `nitrix.stats`
- [`crop-to-nonzero`](resolved/crop-to-nonzero.md) — `crop_to_nonzero` / bounding-box crop — `nitrix.numerics`
- [`cross-entropy-focal`](resolved/cross-entropy-focal.md) — Cross-entropy family + focal loss — `nitrix.metrics`
- [`cubic-resample`](resolved/cubic-resample.md) — Cubic (order-3) resample — `nitrix.geometry`
- [`dice-loss`](resolved/dice-loss.md) — Soft / binary Dice — `nitrix.metrics.dice`
- [`distance-transform-anisotropic-sampling`](resolved/distance-transform-anisotropic-sampling.md) — B20. `distance_transform` euclidean has no `sampling=` (anisotropic spacing)
- [`doc-gaussian-kernel-gamma`](resolved/doc-gaussian-kernel-gamma.md) — Doc-fix: `gaussian_kernel` sigma->gamma relation is wrong (½ factor)
- [`doc-iir-backend-default`](resolved/doc-iir-backend-default.md) — doc-drift: `_iir.py` module docstring says `backend='scan' (default)`
- [`doc-instantaneous-frequency-narrowband-caveat`](resolved/doc-instantaneous-frequency-narrowband-caveat.md) — FR (doc): instantaneous_phase / instantaneous_frequency — narrowband caveat
- [`doc-lomb-scargle-cpu-eigh-caveat`](resolved/doc-lomb-scargle-cpu-eigh-caveat.md) — Doc-fix: `lomb_scargle_interpolate` silently runs its eigh on CPU on cuSolver-broken stacks
- [`doc-lomb-scargle-eigh-factorisation`](resolved/doc-lomb-scargle-eigh-factorisation.md) — Doc-fix: lomb-scargle module docstring says "Cholesky"; code uses `eigh` + pseudo-inverse
- [`doc-lomb-scargle-interpolate-intended-use`](resolved/doc-lomb-scargle-interpolate-intended-use.md) — Doc-fix: `lomb_scargle_interpolate` — document the *intended use* (spectral bridge, not durab…
- [`doc-lomb-scargle-normalisation`](resolved/doc-lomb-scargle-normalisation.md) — Doc-fix: `lomb_scargle_periodogram` normalisation docstring is wrong
- [`doc-op-matrix-inventory-gaps`](resolved/doc-op-matrix-inventory-gaps.md) — Doc-fix: public ops missing from the op_matrix inventory
- [`doc-relaxed-modularity-newman-factor`](resolved/doc-relaxed-modularity-newman-factor.md) — Doc-fix: `relaxed_modularity` does not reduce to Newman modularity (½ factor)
- [`doc-tsconv-cross-correlation`](resolved/doc-tsconv-cross-correlation.md) — Doc-fix: `tsconv` is documented as "convolution" but implements cross-correlation
- [`field-regularisers`](resolved/field-regularisers.md) — Displacement-field regularisers — `nitrix.register.regulariser`
- [`fixed-point-combinators`](resolved/fixed-point-combinators.md) — Fixed-point combinators — `nitrix.numerics.fixed_point`
- [`gaussian-kl-nll`](resolved/gaussian-kl-nll.md) — Diagonal-Gaussian KL / NLL — `nitrix.stats`
- [`gaussian-smooth-traced-sigma`](resolved/gaussian-smooth-traced-sigma.md) — jit-safe traced sigma for `smoothing.gaussian`
- [`generative-bias-field`](resolved/generative-bias-field.md) — Generative bias field — simulated INU — `nitrix.augment`
- [`geometric-augmentation-ops`](resolved/geometric-augmentation-ops.md) — Geometric-augmentation ops — `nitrix.augment.geometric`
- [`gibbs-ringing`](resolved/gibbs-ringing.md) — Gibbs (truncation) ringing artefact — `nitrix.augment.intensity`
- [`glasso-roll-sweep-loop`](resolved/glasso-roll-sweep-loop.md) — Graphical-LASSO — roll the sweep loop (GPU-compile-hostile as shipped)
- [`glmm-fit-jit-incompatible-static-group-count`](resolved/glmm-fit-jit-incompatible-static-group-count.md) — `glmm_fit` is not `jax.jit`-traceable — data-dependent `int(jnp.max(group))` — `nitrix.stats.…
- [`glmm-random-slope-robust-solver`](resolved/glmm-random-slope-robust-solver.md) — GLMM random-slope robust solver — joint-Schur PQL + REML-EM — `nitrix.stats.glmm`
- [`gpu-cusolver-first-call-handle-failure`](resolved/gpu-cusolver-first-call-handle-failure.md) — GPU availability: cuSOLVER `gpusolverDnCreate` fails for a Cholesky/eigh-first program on the…
- [`graphical-lasso`](resolved/graphical-lasso.md) — Graphical LASSO — `nitrix.stats.glasso`
- [`heat-kernel-diffusion`](resolved/heat-kernel-diffusion.md) — Heat-kernel & diffusion-map embedding — `nitrix.graph.diffusion`
- [`iir-filter-gpu-backend`](resolved/iir-filter-gpu-backend.md) — B12. IIR `sosfilt`/`sosfiltfilt` GPU backend — default + missing associative path
- [`intensity-augmentation-ops`](resolved/intensity-augmentation-ops.md) — Intensity-augmentation ops — `nitrix.augment.intensity`
- [`intensity-normalize-variants`](resolved/intensity-normalize-variants.md) — Intensity-normalize variants — `nitrix.numerics.normalize`
- [`lab2im-gmm-synthesis`](resolved/lab2im-gmm-synthesis.md) — GMM label→image synthesis (lab2im) — `nitrix.augment`
- [`ledoit-wolf-shrinkage`](resolved/ledoit-wolf-shrinkage.md) — Ledoit-Wolf shrinkage covariance — `nitrix.stats.ledoit_wolf`
- [`linalg-orthogonal-procrustes`](resolved/linalg-orthogonal-procrustes.md) — Subspace geometry & orthogonal alignment in `nitrix.linalg`
- [`lp-normalize`](resolved/lp-normalize.md) — Lp / unit normalize + instance-norm statistics — `nitrix.numerics.normalize`
- [`mesh-curvature`](resolved/mesh-curvature.md) — Mesh curvature — `nitrix.geometry.curvature`
- [`mesh-laplacian-smoothing`](resolved/mesh-laplacian-smoothing.md) — Uniform 1-ring mesh Laplacian smoothing — `nitrix.sparse.mesh`
- [`mesh-watershed`](resolved/mesh-watershed.md) — Watershed segmentation on meshes — `nitrix.graph.parcellation.watershed`
- [`metrics-convention-vs-domain-tools`](resolved/metrics-convention-vs-domain-tools.md) — Convention check: `nitrix.metrics` similarity metrics vs domain-standard ITK/ANTs
- [`morphology-reduce-window-jitgrad`](resolved/morphology-reduce-window-jitgrad.md) — B19. `erode`/`dilate` flat-SE fast path breaks `jit(grad(...))`
- [`nimox-differentiable-registration-layer`](resolved/nimox-differentiable-registration-layer.md) — A public differentiable-registration layer (implicit-diff, self-contained matrix)
- [`nimox-gp-fit-traceable-rho-search`](resolved/nimox-gp-fit-traceable-rho-search.md) — Make `gp_fit`'s lengthscale-search epilogue traceable (jit / vmap the GP fit)
- [`nimox-histogram-match-fit-apply`](resolved/nimox-histogram-match-fit-apply.md) — Two-phase `histogram_match` (fit reference landmarks once, apply to many)
- [`nimox-mesh-loss-geometry`](resolved/nimox-mesh-loss-geometry.md) — nimox mesh-loss geometry → `nitrix.geometry` (consolidation handoff)
- [`nimox-stats-response-predict`](resolved/nimox-stats-response-predict.md) — Public `*_predict` for the response-regression fitters (Beta / Ordinal / GauLSS / GAM)
- [`normalised-cut`](resolved/normalised-cut.md) — Normalised-cut spectral clustering — `nitrix.graph.ncut`
- [`pad-to-multiple`](resolved/pad-to-multiple.md) — `pad_to_multiple` / `crop_to_multiple` (+ unpad) — `nitrix.numerics`
- [`pallas-attention-nonpot-fallback`](resolved/pallas-attention-nonpot-fallback.md) — Pallas attention: automatic fallback on non-power-of-2 token counts
- [`pca-svd`](resolved/pca-svd.md) — PCA fit / transform / inverse (SVD) — `nitrix.stats.pca`
- [`point-sample`](resolved/point-sample.md) — `point_sample` / `sample_volume_at_points` — `nitrix.geometry.grid`
- [`register-affine-small-grid-divergence`](resolved/register-affine-small-grid-divergence.md) — `affine_register` multi-level GN/LM **diverges at small grids** (v3 regression)
- [`register-demons-force-divide-by-zero`](resolved/register-demons-force-divide-by-zero.md) — Demons ESM force **0/0 → NaN** on uniform regions (real images always NaN)
- [`register-functional-alignment`](resolved/register-functional-alignment.md) — Functional alignment (Procrustes / ProMises) in `nitrix.register`
- [`registration-early-stopping-while-loop`](resolved/registration-early-stopping-while-loop.md) — Registration optimisers: try a `while_loop` early-exit forward (implicit backward already sup…
- [`registration-recipe-cold-compile`](resolved/registration-recipe-cold-compile.md) — Perf: registration recipes have a pathological cold compile (the Python-unrolled optimizer loop)
- [`registration-typing-metric-adt`](resolved/registration-typing-metric-adt.md) — Register v2: Metric ADT + TransformModel protocol + differentiable non-SSD metrics
- [`selective-scan`](resolved/selective-scan.md) — selective-scan — state-space-model fused scan (Mamba/S6)
- [`semiring-annihilator-field`](resolved/semiring-annihilator-field.md) — B8. Store the `(*)`-annihilator explicitly on `Semiring`
- [`sliding-window-weighting`](resolved/sliding-window-weighting.md) — Sliding-window weighting kernel + overlap-add stitch — `nitrix.numerics`
- [`spatial-transform-batched`](resolved/spatial-transform-batched.md) — `spatial_transform_batched` — `nitrix.geometry.grid`
- [`spectral-embedding-gpu-solver`](resolved/spectral-embedding-gpu-solver.md) — B14. Spectral-embedding solver on GPU: lobpcg lags eigsh; eigh-path wedges
- [`stats-whitening`](resolved/stats-whitening.md) — Whitening in `nitrix.stats` — findability wrapper + implementation-strategy research
- [`surface-boundary-map`](resolved/surface-boundary-map.md) — Surface-boundary / gradient mapping — `nitrix.graph.parcellation.boundary`
- [`surface-resample-adap-bary`](resolved/surface-resample-adap-bary.md) — Adaptive area-weighted barycentric resampling — `nitrix.geometry.sphere.resample`
- [`upsample-nearest-nd`](resolved/upsample-nearest-nd.md) — `upsample_nearest_nd` — `nitrix.numerics` / `geometry`
