# nitrix-moonshot — restricted-assistant filing ledger

**Status:** living ledger. Created 2026-07-08.

## Purpose

We have restricted access to a powerful coding assistant approved **only** for
problems that can be framed as strictly numerical — no biological/medical domain
content. Eligibility therefore hinges on hoisting a *pure numerical core* out of a
neuroimaging need, leaving the domain bridge as a separate downstream seam (the same
move by which `genred` generalises the streaming semiring, or a genus-repair kernel
is "mesh surgery" rather than "cortex").

The filings themselves live in the sibling workspace **`nitrix-moonshot/`** (its own
git repo), are **strictly domain-free and self-contained** (the assistant cannot see
nitrix), and are grep-gated against a denylist of domain/tool/purpose vocabulary. The
work is done there, outside nitrix, and **reconciled into nitrix afterward** against
the recommended `protocols.py` contract each filing ships.

**This ledger is the domain side of that seam.** It records, for each filing, the
neuroimaging anchor that motivated it (which must *never* appear in the filing), the
nitrix subsystem it reconciles into, and a relative assessment of difficulty, effort,
and ecosystem impact to guide which filings actually get spent on the assistant.

Scope note: this batch relaxes the SPEC §9 graduation requirement of a *concrete
blocked consumer* (these are forward-looking, community-value primitives) but keeps
every firm invariant binding — separation of concerns, no I/O, no PyTree/Equinox
modules, no `loss`/scalarisation namespace, runtime deps limited to
`jax`/`jaxtyping`/`numpy`, pure-functional + differentiable, fp32/fp64-first,
two-tier `pallas ≈ jax` parity, loud fallback.

## The batch — neuroimaging anchor and reconciliation target

Each filing's numerical framing is the domain-free statement sent to the assistant;
the **anchor** is the domain motivation, kept here.

### 01 · Certified genus-0 repair of a triangulated 2-manifold
- **Anchor.** Cortical surface topology correction. A surface reconstructed from a
  volumetric segmentation must be a topological sphere (genus 0), but voxelisation
  and segmentation errors introduce spurious handles/tunnels. The classical corrector
  (`recon-all` → `mris_fix_topology`; Fischl 2001 manifold surgery, Ségonne 2007 MAP
  retessellation) is slow and leaves residual defects, and it is unavoidable on the
  learned field→mesh route (fastcsr / SynthSR-style). This filing targets a
  *certified*, *minimal* corrector plus a regression corpus on which the classical
  method fails — the flag being that current tools cannot guarantee genus-0 +
  intersection-free + minimal perturbation.
- **Reconciles →** a host-side mesh-constructor seam (dynamic output shape, beside
  `geometry.marching_cubes` / `find_self_intersections`). Graduates GS-7.

### 02 · Vendor-solver-free differentiable symmetric eigensystems
- **Anchor.** The dead-cuSOLVER fragility that recurs across the suite. Laplace–
  Beltrami eigenmaps for spectral surface analysis, connectome graph-Laplacian
  spectral embedding / diffusion maps, PCA, the low-rank REML eigenproblem in
  mass-univariate stats, the LBO eigensolve inside spherical parameterisation, and
  the per-patch eig of filing 14 all need a symmetric/generalised eigensolver;
  cuSOLVER is flaky or absent on target GPUs and `linalg.safe_eigh` CPU-latches and
  cannot `vmap`.
- **Reconciles →** `linalg` solver/eig seam (beside `cg`/`safe_*`). Graduates B24;
  unblocks 03, 12, 14 and the sphere-param eigensolve.
- **⚠ Reconciliation constraint — adopt filing 14's spectral-function adjoint (X-1).** 02's
  thin-SVD adjoint carries an **unbounded** `1/σ` range-complement term (its open **G2**): the
  degeneracy routes broaden the *repeated*-σ coupling but **not** that term. **Do not fold 02's
  adjoint into `linalg` as-is.** See the cross-filing table below.

### 03 · Intrinsic geodesic distance on a triangulated 2-manifold
- **Anchor.** Geodesic (along-the-cortex) distance on the cortical surface, as
  opposed to Euclidean — for geodesic-aware surface smoothing, wavefront/geodesic
  parcellation, surface-based searchlight/representational-similarity analysis, and
  cortical distance matrices. Only the sphere case ships today.
- **Reconciles →** `geometry`/`sparse` mesh-operator seam (composes the shipped
  cotangent Laplacian + `cg`). Feeds GS-12 geodesic smoothing.

### 04 · Structured optimal transport (Gromov–Wasserstein, unbalanced, low-rank)
- **Anchor.** Embedding-free alignment. Gromov–Wasserstein matches two cortical
  surfaces or two connectomes by intrinsic geometry alone, with no shared coordinate
  space — cross-subject / cross-species functional or structural correspondence, and
  OT-based hyperalignment (the embedding-free complement to the ProMises
  hyperalignment in `register.functional_align`).
- **Reconciles →** `transport` seam (extends the shipped entropic `sinkhorn`).

### 05 · Non-negativity-constrained spherical deconvolution
- **Anchor.** Fibre orientation distribution estimation in diffusion MRI. Constrained
  spherical deconvolution (Tournier; MRtrix `dwi2fod`) deconvolves a response kernel
  from the diffusion-weighted signal on the sphere to yield the fODF for
  tractography — an entire signal-analysis pillar absent today, built directly on the
  shipped spherical-harmonic transform.
- **Reconciles →** fit/apply estimator seam (SPEC §6.5) over the SHT domain.

### 06 · Reproducible stochastic / symplectic / delay DE integrators
- **Anchor.** Whole-brain dynamical models and digital twins. Stochastic neural-mass
  / neural-field models and delay-coupled network models (The Virtual Brain — with
  conduction delays), stochastic DCM, and Hamiltonian/HMC posterior geometry all need
  keyed SDE, symplectic, and delay-DE integrators. The dynamics ledger calls the SDE
  family "the single biggest unlock".
- **Reconciles →** `numerics.ode` seam. Graduates DS-1/2/4/5; feeds filing 13.

### 07 · Embedding-constrained differentiable surface flow
- **Anchor.** Intensity-driven deformable white/pial surface placement (`recon-all` →
  `mris_place_surface`; Dale–Fischl–Sereno active surfaces). The surface must evolve
  under an image-derived force while never self-intersecting — precisely the failure
  mode of classical active-surface methods. Pairs with filing 01 on the
  surface-reconstruction vertical.
- **Reconciles →** a new/adaptive registration-adjacent seam. Graduates GS-11.

### 08 · Metric diffeomorphic matching by geodesic shooting (EPDiff)
- **Anchor.** Large-deformation diffeomorphic metric mapping (LDDMM; deformetrica,
  ANTs-class metric warps) — registration with a true geodesic distance and
  guaranteed invertibility, and the basis for diffeomorphometry / atlas building. The
  shipped `register` suite has greedy SyN/Demons, not a metric-geodesic warp.
- **Reconciles →** `register` recipe seam (reuses the velocity/displacement
  conventions).

### 09 · Differentiable persistent homology
- **Anchor.** Topological data analysis of brain data — persistent homology of
  structural/functional brain networks, cortical scalar-field topology, and fMRI
  dynamic state-space topology — with a differentiable persistence layer for
  topology-regularised learning. Also a principled descriptor of the very defects
  filing 01 repairs.
- **Reconciles →** a genuinely new seam (no `protocols.py`; return container
  described and adaptable).

### 10 · Confluent hypergeometric ₁F₁ of a symmetric matrix argument
- **Anchor.** Orientation statistics on the sphere / SO(3). The Bingham / matrix-
  Fisher normalising constant underlies axial and orientation distributions — fibre
  orientation dispersion (Bingham-NODDI), cortical orientation fields, and
  rotation/registration uncertainty. Completes the directional-statistics family
  (`stats.directional` ships vMF/Watson/Kent; the Bingham normaliser/fit was deferred
  because the cheap moment-map approximation is 30–38 % off).
- **Reconciles →** `stats.directional` named vocabulary family.

### 11 · Structured low-rank / sparse / independent decompositions
- **Anchor.** fMRI decomposition. Spatial/temporal independent component analysis is
  the workhorse of resting-state fMRI (group ICA; FSL MELODIC-style); robust PCA
  separates structured artefact from signal; CP/Tucker factorise multi-way
  subject × region × time × frequency connectivity arrays.
- **Reconciles →** `stats`/`linalg` decomposition seam (beside `pca_fit`/`nmf`/
  `whiten`; ICA as a fit/apply pair).

### 12 · Riemannian statistics on the SPD manifold
- **Anchor.** Tangent-space functional-connectivity analysis. Connectivity matrices
  are SPD; the affine-invariant / log-Euclidean geometry (Fréchet mean, tangent-space
  projection, parallel transport, tangent PCA) is the standard substrate for
  connectome classification/regression and SPD-manifold harmonisation (pyRiemann-class
  workflows).
- **Reconciles →** `linalg` SPD family (extends `symexp`/`symlog`/`symsqrt`/
  `tangent_project_spd`).

### 13 · Scalable Laplace-approximate inference for a differentiable forward operator
- **Anchor.** Effective-connectivity estimation by Dynamic Causal Modelling. DCM
  inverts a differential-equation generative model of coupled regions by variational
  Laplace; spectral DCM works in the frequency domain via the cross-spectral density
  (the transfer function / resolvent). Classical variational Laplace does not scale
  (dense Hessian, finite-difference sensitivities, nested ReML). The generic engine
  is hoisted (SPM `spm_nlsi_GN` / `spm_dcm_*` minus the neural model); the model
  assembly stays downstream. Also fills the non-symmetric-resolvent gap flagged in
  `krylov-solvers.md`.
- **Reconciles →** `stats` inference/inversion (beside `reml_fit` / `flame` /
  `levenberg_marquardt`). Resolves the dynamics-ledger RFC-1 in favour of
  "generic engine in nitrix, generative model downstream".

### 14 · Random-matrix-theory optimal singular-value shrinkage denoising
- **Anchor.** NORDIC / MP-PCA denoising of fMRI and diffusion MRI. RMT-optimal
  singular-value thresholding of local spatiotemporal (Casorati) patches denoises
  time-series (NORDIC; Veraart MP-PCA `dwidenoise`). The "acquisition-agnostic"
  improvement is to estimate the noise level from the data via the Marchenko–Pastur
  bulk rather than requiring a g-factor / noise map. This is the exact consumer named
  by the parked B24 eigensolver FR.
- **Reconciles →** `stats`/`numerics` (beside `randomized_svd`/`pca_fit`/
  `ledoit_wolf`); rides on filing 02's batched SVD.
- **⚠ Reconciliation constraint — 14 *supplies* the adjoint 02 is missing (X-1),** and 14's own
  review returned **5 blockers** (fabricated rank on correlated noise; all-NaN at `σ̂=0`; every
  shrinker discontinuous; fail-open non-convergence; `O(n_blocks)` reverse-mode memory). **Do not
  fold 14 into `stats`/`numerics` until those are closed** — the denoiser is exactly the consumer
  that would silently corrupt data. See the cross-filing table below.

---

## Cross-filing reconciliation constraints (tracked)

Findings that **cross filing boundaries**. Each is a constraint on the *order* or *content* of
what gets folded into nitrix — not a defect of any one filing, and therefore invisible in any
single filing's ADDENDUM. **This table is the tracking surface: check it before folding any
filing in.**

| id | constraint | status |
|---|---|---|
| **X-1** | **02's SVD adjoint must adopt 14's spectral-function formulation.** | **open** |

### X-1 · The spectral-function adjoint (02 ← 14)

**The finding.** Filing 02's thin-SVD adjoint divides by the singular value `σ` in its
range-complement term, guarded only against an exact zero. Its degeneracy routes broaden the
*repeated*-`σ` coupling but **not** this term, so a small-but-nonzero `σ` yields a
`1/σ`-amplified left-factor gradient with no floor available (02 **G2**, open).

**Filing 14 solved exactly this, structurally — and it is the stronger derivation.** 14 never
differentiates `U, Σ, V` at all. It writes the operation as the **spectral function**
`X ↦ X · g(XᴴX)` and differentiates *that* with a Daleckii–Krein divided-difference tangent.
**Both** singular terms then vanish from the derivation rather than being damped:

- the `1/(λᵢ − λⱼ)` coupling never appears — a **divided difference is finite at a repeat**;
- the `1/σ` complement **collapses in closed form** to `U·diag(g)·Uᴴ·dX·(I − VVᵀ)`, because
  `X·V_d = 0` on the discarded subspace.

This is not a tolerance trick; it is a different (and correct) object to differentiate.

**Why it must be tracked here and not in either ADDENDUM.** Both filings reconcile into nitrix,
02 into the `linalg` eig/SVD seam and 14 into `stats`/`numerics`. If each is folded in on its own
terms, **nitrix acquires two different answers to the same question**, and the weaker one lands in
the *lower-level, more widely-consumed* seam — `linalg` — where every future spectral consumer
inherits an unbounded `1/σ` gradient. The damage is done by the *order* of folding, which is
exactly what no single-filing review can see.

**The rule on reconciliation.**
1. Wherever a consumer differentiates a **spectral function** of the matrix — every shrinkage,
   filtering, or spectral-nonlinearity consumer, **including 02's own G2 motivating example** —
   `linalg` exposes the **Daleckii–Krein spectral-function adjoint**, not a factor-wise SVD adjoint.
   This is the default path.
2. The factor-wise adjoint is retained **only** for consumers that genuinely need `U`, `V`
   themselves (a subspace, a rotation, a sign-fixed basis). There the `1/σ` term is **irreducible**
   — it is a real property of the factor map, not an artefact — and 02's G2 stands on its own terms
   as a *documented domain restriction*, not a bug to be floored away.
3. Land the spectral-function adjoint **once**, in `linalg`, and have 14's shrinkage *consume* it.
   Neither filing should ship its own copy.

**Degeneracy is the common case here, not an edge case.** A noise bulk **is** a cluster of hundreds
of near-equal singular values, so the clustered-spectrum path is the one that runs on every block
of every real input — which is why this cannot be deferred as an edge-case hardening item.

---

## Score assessment (relative, within-set, 1 = lowest … 5 = highest)

Axes: **① low self-confidence** — *higher = less confident I could produce a robust,
certified implementation myself even with time and resources* (the signal for routing
to the restricted assistant); **② work** — relative effort (all are substantial);
**③ impact** — ecosystem leverage (perf/robustness, what it unblocks, community value).

| # | Filing | ① low self-confidence | ② work | ③ impact |
|---|---|:---:|:---:|:---:|
| 01 | Genus-0 topology repair | **5** | 5 | 4 |
| 02 | Vendor-solver-free eigensystems | 2 | 4 | **5** |
| 03 | Intrinsic mesh geodesics | 2 | 3 | 3 |
| 04 | Structured OT (Gromov–Wasserstein) | 3 | 4 | 3 |
| 05 | Constrained spherical deconvolution | 2 | 3 | 4 |
| 06 | SDE / symplectic / delay integrators | 2 | 4 | 4 |
| 07 | Embedding-constrained surface flow | 4 | 4 | 3 |
| 08 | EPDiff geodesic shooting | 4 | **5** | 3 |
| 09 | Differentiable persistent homology | 4 | 4 | 3 |
| 10 | Matrix-argument ₁F₁ / Fisher–Bingham | 4 | 3 | 2 |
| 11 | Structured decompositions (rPCA/CP/ICA) | 2 | 4 | 3 |
| 12 | Riemannian SPD statistics | 2 | 3 | 3 |
| 13 | Variational-Laplace inference + resolvent | 3 | 5 | 4 |
| 14 | RMT shrinkage denoising | 2 | 3 | 4 |

### Reasoning

- **① self-confidence.** The only likely *failure* to make robust myself is **01**
  ("certified genus-0 *and* minimal-perturbation *and* intersection-free on adversarial
  inputs" is an open problem). A real chance of a subtly-wrong or non-robust result
  (**4**): **07** (self-intersection invariant maintained *through* a differentiable
  flow), **08** (geodesic-shooting adjoint / EPDiff stability), **09** (𝔽₂ boundary
  reduction coupled to a differentiable readout at scale), **10** (<1e-9 matrix-argument
  ₁F₁ across all concentrations — the primitive nitrix already deferred as too hard).
  Probable-but-risky (**3**): **04** (differentiability through a non-convex GW fixed
  point), **13** (making a known method *scalable + differentiable + matrix-free*
  robustly). The rest (**2**) are established, well-documented algorithms I am fairly
  confident I could implement correctly given time — the score reflects the
  *certified-and-differentiable* bar, not "could I write a version".
- **③ impact.** **02** is unique in fanning out across subsystems (spectral embedding,
  LBO, PCA, low-rank REML, sphere-param) *and* unblocking filings 03/12/14 *and*
  removing the single biggest deployment fragility. **06/13/05/14/01** each open a lane
  (dynamics; model inversion + resolvent gap; the dMRI pillar; universal preprocessing;
  the surface-reconstruction vertical). The remainder are more vertical; **10** has the
  narrowest fan-out.

### Recommendations — routing the assistant

- **Route to the assistant first (high ① × meaningful ③):** **01** (the standout —
  hardest *and* flagship-impactful), then the frontier tier **08, 09, 07, 10**, then
  **13, 04**. This is where the restriction earns its keep.
- **Prefer building in-house (low ① despite high ③):** **02, 06, 14, 05, 12, 11, 03** —
  high value but implementable internally with confidence. The tension is sharpest for
  **02** and **06**: they are the highest-fan-out items *and* the ones I am most
  confident about, and they *unblock* several assistant-targeted filings (02 under
  03/12/14; 06 under 13). The pragmatic sequence is therefore to build **02** and **06**
  internally *first*, then spend the assistant on the frontier set that depends on them.
- **Cost caveat (②).** Among the top routing picks, **01, 08, 13** are the heaviest
  (5). If assistant throughput is the binding constraint, **01 → 09/10 → 07** gives the
  best hardness-per-unit-work before taking on the two remaining 5-work items.

## Provenance

- Filings workspace: `../../nitrix-moonshot/` (sibling repo; own git history, not
  pushed): `3a40a1a` initial 10-batch + shared contract; `348b54d` filings 11–12;
  `4a2424f` filings 13–14. Each `NN-slug/` holds `PROBLEM.md`, `validation.md`, and —
  where the result must fit an existing seam — `protocols.py` (a recommended,
  adaptable contract).
- Shared contract in the workspace: `CONVENTIONS.md` (numerical/execution/return/
  dependency/validation model) and `README.md` (framing contract + denylist + index).
- `genred` (accepted-for-build) is the exemplar of this hoisting pattern, not a filing;
  `geometry.spherical_parameterize` already shipped and is therefore excluded.
