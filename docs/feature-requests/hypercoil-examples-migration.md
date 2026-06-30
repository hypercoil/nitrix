# `hypercoil-examples` → `nitrix` migration — context & ledger

> **Status (2026-06-30): scoping ledger.** The sibling `hypercoil-examples`
> repo (a deprecated experimental staging ground for the differentiable
> cortical-parcellation / functional-alignment project) is being wound down.
> This doc frames the salvage of its genuinely-valuable numerical kernels into
> `nitrix`, holds the **standing correctness mandate**, records the full
> inventory (migrate / already-covered / out), and indexes the atomised FRs (the
> duplicate-issue guard, per `README.md`). Numerics-only; everything module-,
> objective-, I/O-, or training-shaped stays downstream (`nimox` / `hypercoil`).

## 1. Standing correctness mandate — theory over the legacy code  [NORMATIVE]

**Every kernel below is a clean-room reimplementation from the primary
literature, not a port.** The `hypercoil-examples` implementation is a *reference
point and recovery oracle* — useful for fixing conventions and as a sanity check
— **not** a specification. The legacy code is experimental and is, at several
identified points, subtly incorrect or only valid in a regime (large-κ
approximations, self-flagged "can't make any promises" hacks, fixed-iteration
rejection samplers with non-guaranteed acceptance, unapplied Jacobian
corrections). **Where the legacy code and the theory disagree, the theory wins,
and the disagreement is documented** (a comment citing the deviation + a test
asserting the theory-correct property). A migration is "done" only when:

1. the kernel is **derived from / validated against an authoritative reference**
   (the paper, and an independent oracle — scipy / a quadrature / a Monte-Carlo
   check — in `tests/`, never a runtime dep);
2. its **edge cases and degeneracies** are enumerated and tested (rank
   deficiency, repeated singular values, small/large parameter regimes, the
   distribution boundaries, fp32 conditioning);
3. it satisfies nitrix's boundaries (pure `jax`/`jaxtyping`/`numpy`; no eqx /
   numpyro / scipy / BCOO-leak; §5 score-kernel and §6.5 fit/apply seams);
4. it carries a golden entry per `(op, dtype)` (SPEC §8).

This mandate is the reason the FRs below read as *design + research briefs*, not
"lift function X from file Y."

## 2. What this repo is (and is not)

~66 files, almost all Equinox modules, `optax` training loops, `numpyro`
objectives, and `nibabel`/`hyve` I/O + plotting. The salvageable kernels cluster
tightly around one theme — **differentiable cortical-surface parcellation &
functional alignment on the sphere** — and that cluster illuminates nitrix's
largest coherent gap: it has rich spherical **geometry** but no spherical
**statistics**, and no **feature-space (functional) alignment**.

## 3. Inventory

### 3.1 Migrate (atomised FRs below)

| Kernel(s) | Source (`hypercoil-examples/...`) | nitrix home | FR | Top correctness risk |
|---|---|---|---|---|
| **Functional alignment** (Procrustes / ProMises) | `atlas/promises.py` | `register` | [`register-functional-alignment`](register-functional-alignment.md) | the "empty" double-whitening is a self-flagged hack (PCA gauge ambiguity); cross-product orientation unverified |
| **Orthogonal Procrustes + subspace angles + image basis** | `atlas/totalangle.py`, `atlas/promises.py` | `linalg` | [`linalg-orthogonal-procrustes`](linalg-orthogonal-procrustes.md) | `arcsin/arccos` boundary sub-derivatives; reflection (`det<0`) handling |
| **von Mises–Fisher directional family** (`log_bessel`, `log_prob`, sampler, MLE) | `hypercoil/init/vmf.py` | `stats` | [`stats-directional-vmf`](stats-directional-vmf.md) | **`log_bessel` is a large-κ asymptotic only** — wrong at small/moderate κ; sampler has non-guaranteed acceptance |
| **Whitening** (ZCA/PCA/partial sphering) — *findability wrapper + impl research, not a missing estimator* | `atlas/vmf.py::generalised_whitening` | `stats` (thin wrapper) | [`stats-whitening`](stats-whitening.md) | **estimator already exists** (`nimox.estimators.whitening` over `linalg.sympower`) — open work is findability + a faster/more-stable inverse-sqrt path (data-SVD / Newton–Schulz vs eigh-of-cov); `cummax` repair rejected |
| **DCBC / DCCC parcellation-quality metric** | `atlas/dccc.py` | `metrics` | [`metrics-dccc`](metrics-dccc.md) | must reduce to published DCBC in the hard limit; the distance-kernel weighting is a hypercoil addition, not DCBC |
| **Synthetic connectivity generators** (band-limited mixture / low-rank-block / Markov) | `synthetic/scripts/{mix,sylo,corr,filter,denoise}.py` | `augment` | [`augment-synthetic-connectivity`](augment-synthetic-connectivity.md) | FFT bin-zeroing ≠ a proper band filter (leakage); data-dependent loops must become fixed-shape `scan`/`vmap` |

### 3.2 Already covered by nitrix (do **not** migrate)

| Legacy | Already in nitrix |
|---|---|
| `atlas/ellgat.py` ELL-GATv2 attention | `semiring.semiring_ell_edge_aggregate` + `ell_row_softmax` (+ `sparse.ell_add_self_loops`) — docstrings already cite Brody 2022 |
| `atlas/unet.py` scatter-mean pool + icosphere hierarchy | `sparse.mesh_coarsen_meanpool` / `mesh_pool_max` + `icosphere_cross_level_adjacency` / `icosphere_hierarchy` — docstrings cite SUGAR |
| `connectopy/direct/gradients.py` | `graph.laplacian_eigenmap` / `diffusion_embedding` |
| `connectopy/direct/community.py` | `graph.modularity_matrix` / `coaffiliation` / `relaxed_modularity` |
| `atlas/spatialinit.py` k-ring | `sparse.mesh_k_ring_adjacency` |
| `dccc.spherical_distance` | `geometry.spherical_geodesic_distance` (reuse; do not re-port) |

### 3.3 Out of scope → downstream (`nimox` / `hypercoil`)

`atlas/{energy,model}.py` (MRF energy / objective scalarisation — SPEC §5
forbidden), `atlas/{beta,sphering,encoders,multiencoder,autoencode}.py` (Equinox
modules + forbidden deps), `atlas/gatrefpyg.py` (torch/PyG oracle), the matrix-vMF
*distribution object* and the running-template *policy*, and all
experiment/training/IO/plotting scripts.

### 3.4 Deferred / borderline

`positional.py` (NeRF/Fourier features) and `selectransform.py::logistic_mixture_threshold`
(soft top-k) — small, admit only inside a coherent encoding/sparsification family;
arbitrary-ν icosphere subdivision (`atlas/icosphere.py`) — a real refinement gap
(nitrix only does power-of-2 resolutions) but **LGPL-2.1**, needs clean-room
reimplementation; KRLS-T (`atlas/krlst.py`) — novel online kernel regression,
needs a full eqx→pure-function + `NamedTuple`-state rewrite, niche.

## 4. Illuminated gaps (the larger picture)

1. **Directional statistics on the sphere** — the keystone. nitrix's surface
   stack is geometry-complete but probability-empty; a `stats` directional family
   (vMF now; Watson/Bingham/Kent later) is the missing substrate under every
   differentiable surface-parcellation/clustering consumer.
2. **Functional alignment / hyperalignment** — nitrix can register *images* but
   not *representations*. Resolved into `register` (see the atomised FR).
3. **A whitening + subspace-geometry `linalg` tier** — Procrustes, image basis,
   principal angles, generalised whitening round `linalg` out toward the
   subspace/alignment toolkit these consumers need.
4. **Spatial-autocorrelation nulls remain OPEN — not filled by this repo.**
   `spatialnulls/spatialnulls.py` is a *random-parcellation* generator
   (geometric-loss gradient descent), **not** a spin-test / variogram / Moran
   surrogate-map null. The gap (frequently wanted: `neuromaps`/BrainSMASH-style
   surrogates) must be sourced or built fresh elsewhere; recorded here so it is
   not mistakenly considered covered.

## 5. Sequencing

Smallest-irreducible-first, so later recipes compose earlier primitives:

1. **`linalg`: orthogonal Procrustes + subspace angles** — pure, oracle-testable,
   no boundary questions; unblocks #2.
2. **`register`: functional alignment (ProMises)** — composes (1) + ELL matvecs +
   the §6.5 seam.
3. **`stats`: vMF directional family** — the keystone; budget real effort for a
   **validated** `log_bessel` across the full κ range + a guaranteed-acceptance
   sampler. *(Independent of #1/#2 — ProMises uses the matrix-vMF, whose
   normaliser never materialises.)*
4. **`metrics`: DCBC** · **`augment`: synthetic generators** — independent,
   parallelisable. **`stats`: whitening** is lower-priority (the estimator
   already exists in nimox over `linalg.sympower`); its value is a findable
   wrapper + the inverse-sqrt implementation-strategy research, not a new
   capability.

## 6. Cross-references

- Atomised FRs: [`register-functional-alignment`](register-functional-alignment.md),
  [`linalg-orthogonal-procrustes`](linalg-orthogonal-procrustes.md),
  [`stats-directional-vmf`](stats-directional-vmf.md),
  [`stats-whitening`](stats-whitening.md), [`metrics-dccc`](metrics-dccc.md),
  [`augment-synthetic-connectivity`](augment-synthetic-connectivity.md).
- Ledgers this touches: [`registration-suite`](registration-suite.md) (§9
  functional alignment), the `stats` modelling suite, `augment`.
- Admission gate: SPEC §9; concern boundaries SPEC §5 / §6 / §6.5.
