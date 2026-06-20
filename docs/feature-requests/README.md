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
| Statistical modelling suite | [`stats-modelling-suite.md`](stats-modelling-suite.md) | Mass-univariate GLM/GAM/GAMM, LME size-dispatch, TFCE `randomise` — perf-bench + ModelArray + niffi |
| Internal engineering backlog | [`internal-backlog.md`](internal-backlog.md) | Parked perf / Pallas / API-refinement items, each gated on a **Trigger** |
| Doc-drift / correctness fixes | [`perf-bench-feedback.md`](perf-bench-feedback.md) | Mechanical docstring fixes (file:line-pinned), *not* primitive proposals |
| SPEC §12 brainstorm catalogue | `SPEC_UPDATE_v0.3.md §12` (origin) | Substrate-compatible *candidate* primitives; promotion gated by `§13` |

Status is verified against the live `src/nitrix` surface as of 2026-06-02
(interpolation-method dispatcher — `Lanczos` / `MultiLabel` /
`NearestNeighbour` on `resample` / `spatial_transform` — shipped 2026-06-07;
`CubicBSpline` (scipy `order=3` bit-exact) added 2026-06-08; see
`IMPLEMENTATION_PLAN.md §10.3`).

## Consumer-pipeline substrate (ilex → thrux)

Context, scope boundary, and already-shipped record:
[`ilex-pipeline-substrate.md`](ilex-pipeline-substrate.md).

| Doc | Severity | Home |
|---|---|---|
| [connected-components](connected-components.md) | ENABLING (highest recurrence) | `morphology` |
| [pad-to-multiple](pad-to-multiple.md) | ENABLING | `numerics` |
| [crop-to-nonzero](crop-to-nonzero.md) | ENABLING | `numerics` |
| [cubic-resample](cubic-resample.md) | MISMATCH (parity) — ✅ **resolved**: `CubicBSpline` (scipy `order=3` bit-exact) (2026-06-08) | `geometry._interpolate` |
| [intensity-normalize-variants](intensity-normalize-variants.md) | CONVENIENCE | `numerics.normalize` |
| [sliding-window-weighting](sliding-window-weighting.md) | CONVENIENCE | `numerics` |
| [point-sample](point-sample.md) | CONVENIENCE — **partial**: capability via `spatial_transform(method=Linear(), mode='constant')`; flat-point-list wrapper unshipped (2026-06-07) | `geometry.grid` |
| [compute-vertex-normals](compute-vertex-normals.md) | CONVENIENCE | `sparse.mesh` |
| [upsample-nearest-nd](upsample-nearest-nd.md) | CONVENIENCE — ✅ **addressed**: `resample(method=NearestNeighbour())` (2026-06-07) | `geometry` |
| [spatial-transform-batched](spatial-transform-batched.md) | CONVENIENCE | `geometry.grid` |

## Consumer training-substrate (ilex → nitrix)

Context, boundary (post bitsjax–thrux–nitrix realignment), negative record,
and index: [`ilex-training-substrate.md`](ilex-training-substrate.md). From
the 2026-06-08 audit of the surfaces ilex grew **after** the 2026-06-02
survey — augmentation pipelines, the nimox loss library, and the numerics of
the nimox modules + newest models (`krakencoder`, `cortex_ode`, `surfnet`).

| Doc | Severity | Home |
|---|---|---|
| [lab2im-gmm-synthesis](lab2im-gmm-synthesis.md) | ENABLING | `augment` |
| [generative-bias-field](generative-bias-field.md) (simulated INU) | ENABLING | `augment` / `bias` |
| [intensity-augmentation-ops](intensity-augmentation-ops.md) | CONVENIENCE | `augment.intensity` |
| [gibbs-ringing](gibbs-ringing.md) | CONVENIENCE | `augment.intensity` |
| [geometric-augmentation-ops](geometric-augmentation-ops.md) | CONVENIENCE | `augment.geometric` |
| [dice-loss](dice-loss.md) | ENABLING | `metrics.dice` |
| [cross-entropy-focal](cross-entropy-focal.md) | CONVENIENCE (dedup ×3) | `metrics` |
| [contrastive-ssl-losses](contrastive-ssl-losses.md) | CONVENIENCE | `metrics` / `stats` |
| [field-regularisers](field-regularisers.md) | ENABLING | `register.regulariser` |
| [gaussian-kl-nll](gaussian-kl-nll.md) | CONVENIENCE | `stats` |
| [affine-matrix-algebra](affine-matrix-algebra.md) | ENABLING | `geometry.transform` |
| [pca-svd](pca-svd.md) | CONVENIENCE | `stats.pca` |
| [lp-normalize](lp-normalize.md) | CONVENIENCE | `numerics.normalize` |
| [mesh-laplacian-smoothing](mesh-laplacian-smoothing.md) | CONVENIENCE | `geometry.mesh` |

Expansions to existing docs (new drivers / scope from this audit):
[compute-vertex-normals](compute-vertex-normals.md) (+`cortex_ode`/`surfnet`),
[point-sample](point-sample.md) (arbitrary-point `grid_sample`; 3 dups; task
#138), [ode-integrators](ode-integrators.md) (per-vertex NODE; diffrax can't
follow into nitrix).

## Statistical modelling suite (perf-bench → ModelArray → niffi)

Context, organising insight (penalised GLM ≡ variance-components REML ≡ mixed
model), scope boundary, locked decisions, phasing, and the atomised work-item
index: [`stats-modelling-suite.md`](stats-modelling-suite.md). One sprint
spanning three sibling requests that share one substrate.

| Workstream | Severity | Home | Driver |
|---|---|---|---|
| A — LME size-dispatch + cuSOLVER bypass | ✅ SHIPPED (merged to main) | `stats.lme._varcomp` | [lme-family-tiny-linalg-gpu-block-and-perf](lme-family-tiny-linalg-gpu-block-and-perf.md), [gpu-cusolver-first-call-handle-failure](gpu-cusolver-first-call-handle-failure.md) |
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
| B12 | [iir-filter-gpu-backend](iir-filter-gpu-backend.md) | perf / API-default (IIR GPU backend, S+M) |
| B13 | [boundary-mode-parity](boundary-mode-parity.md) | API refinement (scipy/ITK boundary parity, M) |
| B14 | [spectral-embedding-gpu-solver](spectral-embedding-gpu-solver.md) | perf + robustness (lobpcg/eigh GPU solver, M) |
| B15 | [interpolation-backend-cpu-gpu-gap](interpolation-backend-cpu-gpu-gap.md) | perf characterisation (map_coordinates CPU/GPU) |
| B16 | [alternative-interp-backends-xla](alternative-interp-backends-xla.md) | research note (scipy/cupy interp backends in XLA) |
| B17 | [median-percentile-cpu-sort-cliff](median-percentile-cpu-sort-cliff.md) | perf characterisation (jnp.median/percentile CPU sort) |
| G1 | [spatial-transform-linear-extrap](spatial-transform-linear-extrap.md) | boundary-mode extension (S) |

## Doc-drift / correctness fixes

Ledger: [`perf-bench-feedback.md`](perf-bench-feedback.md). These are
mechanical docstring fixes, not primitive proposals.

| Doc | Site |
|---|---|
| [doc-lomb-scargle-normalisation](doc-lomb-scargle-normalisation.md) | `signal/lomb_scargle.py:154` |
| [doc-lomb-scargle-eigh-factorisation](doc-lomb-scargle-eigh-factorisation.md) | `signal/lomb_scargle.py:43–49` |
| [doc-lomb-scargle-cpu-eigh-caveat](doc-lomb-scargle-cpu-eigh-caveat.md) | `linalg/_solver.py:147` |
| [doc-tsconv-cross-correlation](doc-tsconv-cross-correlation.md) | `signal/tsconv.py:45` |
| [doc-lomb-scargle-interpolate-intended-use](doc-lomb-scargle-interpolate-intended-use.md) | `signal/lomb_scargle.py:~264–359` |
| [doc-op-matrix-inventory-gaps](doc-op-matrix-inventory-gaps.md) | `docs/op_matrix.json` (`ops`) |
| [doc-gaussian-kernel-gamma](doc-gaussian-kernel-gamma.md) | `linalg/kernel.py:37` |
| [doc-relaxed-modularity-newman-factor](doc-relaxed-modularity-newman-factor.md) | `graph/community.py:245` |

## SPEC §12 brainstorm catalogue (candidate primitives)

`§` = origin entry in `SPEC_UPDATE_v0.3.md` (the canonical origin record; the
`§13` acceptance protocol references items by number). **Status** is against
live code: *not started* / *partial* (some substrate shipped).

| § | Doc | Proposed module | Effort | Status |
|---|---|---|---|---|
| 12.1 | [krylov-solvers](krylov-solvers.md) | `linalg.krylov` | M | partial (`cg` shipped — registration) |
| 12.2 | [matrix-functions](matrix-functions.md) | `linalg.matrix_function` | S | partial (`sym*` + `matrix_exp` shipped) |
| 12.3 | [heat-kernel-diffusion](heat-kernel-diffusion.md) | `graph.diffusion` | S | partial (`diffusion_embedding` shipped) |
| 12.4 | [sinkhorn-optimal-transport](sinkhorn-optimal-transport.md) | `transport` | M | not started |
| 12.5 | [discrete-exterior-calculus](discrete-exterior-calculus.md) | `geometry.dec` | M | partial (cotangent Laplacian shipped) |
| 12.6 | [mesh-curvature](mesh-curvature.md) | `geometry.curvature` | S | not started |
| 12.7 | [robust-statistics](robust-statistics.md) | `stats.robust` | S | not started |
| 12.8 | [fixed-point-combinators](fixed-point-combinators.md) | `numerics.fixed_point` | M | ✅ shipped (`fixed_point_solve` — registration) |
| 12.9 | [spherical-harmonic-transform](spherical-harmonic-transform.md) | `geometry.sphere.harmonics` | M | not started |
| 12.10 | [compensated-summation](compensated-summation.md) | `numerics.precision` | S | not started |
| 12.11 | [ode-integrators](ode-integrators.md) | `numerics.ode` | L | partial (`integrate_velocity_field` shipped) |
| 12.12 | [continuous-wavelet-transform](continuous-wavelet-transform.md) | `signal.cwt` | S | not started |
| 12.13 | [graph-wavelet-transform](graph-wavelet-transform.md) | `graph.wavelet` | S | not started (blocked on 12.2) |
| 12.14 | [graphical-lasso](graphical-lasso.md) | `stats.glasso` | M | not started — planned in [stats v2](stats-modelling-suite-v2.md) §4.2 |
| 12.15 | [surface-resample-adap-bary](surface-resample-adap-bary.md) | `geometry.sphere.resample` | M | partial (icosphere `BARYCENTRIC` shipped) |
| 12.16 | [surface-boundary-map](surface-boundary-map.md) | `graph.parcellation.boundary` | S | not started (composes shipped prims) |
| 12.17 | [mesh-watershed](mesh-watershed.md) | `graph.parcellation.watershed` | M | not started |
| 12.18 | [clustering-primitives](clustering-primitives.md) | `numerics.cluster` | S/M | not started |
| 12.19 | [normalised-cut](normalised-cut.md) | `graph.ncut` | XS | not started (blocked on 12.18) |

**§12.20** (functional-parcellation strategy survey) is informational — a
strategy→primitive mapping table, not a primitive — and stays in
`SPEC_UPDATE_v0.3.md §12.20`. The parcellation docs (12.16–12.19) link to it.

**Candidate not yet in SPEC §12:**
[ledoit-wolf-shrinkage](ledoit-wolf-shrinkage.md) (`stats.ledoit_wolf`, effort
S) — analytic shrinkage covariance; sibling of 12.14 glasso and nilearn's
*default* connectome estimator. Surfaced by perf-bench (nilearn defaults to
Ledoit-Wolf; nitrix has no shrinkage estimator). **Planned in
[stats v2](stats-modelling-suite-v2.md) §4.1** (a quick-win, consumer waiting).

### Dependency edges (within the §12 catalogue)

```
12.1 krylov ──┬─> 12.8 fixed-point ──┬─> 12.18 clustering ──> 12.19 ncut
              │                       └─> 12.14 glasso (impl-VJP)
              └─> 12.11 ode (adjoint)
12.2 matrix-fn ─┬─> 12.3 heat-kernel
                └─> 12.13 graph-wavelet
12.16 boundary ──> 12.17 watershed
```
