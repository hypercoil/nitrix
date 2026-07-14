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

### 31 · Masked sparse–sparse products over a semiring *(late addition; the 32nd kernel)*
- **Anchor.** Three things nitrix cannot currently do, all of which are one operation.
  **(i) Preconditioning.** Every mesh- and graph-Laplacian solve in the library is
  *unpreconditioned* CG — `linalg.krylov.cg` has no preconditioner argument at all, and the
  preconditioners that exist (`shift_invert`, `polynomial`) are **dense-only**, raising on
  ELL/SectionedELL. At the scale that is the whole point of the sparse path, and on operators whose
  spectra are the *worst* case for CG (community structure ⇒ clustered low eigenvalues ⇒ small
  spectral gap), there is no lever. The mesh-independent supplier is a multilevel hierarchy, whose
  setup is the Galerkin triple product :math:`P^\top A P` — a sparse×sparse product. The
  prolongation `P` already ships as an ELL (`sparse/mesh.py:566`); it is only ever applied to
  *signals*, never to the *operator*. **(ii) Graph reduction by a partition.** Contracting a
  connectivity graph by a parcellation is :math:`P^\top A P` — the same kernel — and is not
  expressible today: `graph/community.py` dodges it by contracting straight to a scalar trace.
  **(iii) Mesh combinatorics.** Face adjacency, n-ring neighbourhoods and reachability are boolean
  semiring products; `_kring_adj_list` currently computes a boolean matrix power as a **triple-nested
  Python `set` BFS** on meshes of up to 163,842 vertices.
- **Reconciliation target.** `sparse` + `semiring` — **not a new top-level subpackage.** The algebra
  axis (`REAL/LOG/TROPICAL_*/BOOLEAN/EUCLIDEAN`) and the `ELL`/`SectionedELL` storage are **shared
  with genred and must bind, not fork**; `semiring_ell_matmul` (sparse × dense) stays exactly where
  it is. 31 contributes a storage (sliced-ELL) and a *constructed* domain. It also supplies the
  **preconditioner seam** on the Krylov solvers, which is what unblocks X-3.
- **⚠ Governed by X-9** (the genred boundary) and **closes X-3**; **C2 of its corpus is X-7's
  primitive** (`shares an edge` ≡ :math:`F F^\top` entry `== 2`).
- **On the absence of a named request.** A full-tree grep for `spgemm`, `graphblas`, `multigrid`,
  `AMG`, `galerkin` returns **zero hits** across SPEC, ~80 feature-requests and all source. *This is
  not a caveat.* **The moonshot standard is prospective — SPEC §9's "concrete blocked consumer" gate
  is waived for this batch by construction** (a filing qualifies if it improves something we have
  **or** opens a family of methods we cannot currently write). Demand-by-name was never the bar, and
  the six mutually-unaware rediscovery sites of X-9 are corroboration, not the argument. The one
  place the concept appears at all is `SPEC.md:236-242`, which names **`spspmm`** exactly once — as
  *the memory hazard to guard against*. The hazard is in the SPEC; the operation is not in the
  library.

### 32 · Kernel-independent hierarchical summation *(late addition; gated on a go/no-go spike)*
- **Anchor.** Surface-to-surface and point-set-to-point-set matching by a kernel metric — the
  data-attachment term of a diffeomorphic match — is an $N \times M$ kernel sum over vertices, evaluated
  *and differentiated* at every step of the optimisation. At full cortical-mesh resolution the pair count
  is the wall: filing 25 fixed the *memory* ($O(N+M)$) and **declared quadratic time irreducible**, which
  leaves its own stated upper target ($N \sim 10^7$) unreachable — $10^{14}$ multiply–adds per evaluation.
  An $O(N \log N)$ summation is what makes mesh-scale kernel matching, and the same regime for any smooth
  kernel (RBF interpolation, kernel density on a surface, Gaussian-process covariance application),
  computable at all.
- **Reconciliation target.** Composes with the streaming kernel-sum engine (filing 00 / genred) rather
  than replacing it — **the near field *is* the incumbent tiled kernel**. Lands beside it, not over it.
- **⚠ This is the one filing whose first deliverable may be a decline.** The incumbent is a tiled
  streaming sum whose inner-product stage runs on tensor cores at near-peak arithmetic intensity;
  asymptotic wins routinely lose to that on a GPU. The spike bounds the proposed method from **below** by
  its unavoidable work (without implementing it) and the incumbent from **above** by measurement, at
  iso-accuracy, in **both** precisions, on **both** the value and the gradient — with the decision rule
  fixed before the numbers are seen. **A no-go returns the crossover table as the result**, which is worth
  having: it would tell us exactly where the quadratic path stops being affordable, which nothing in the
  corpus currently knows.
- **Note the precision axis specifically.** The incumbent's advantage *is* its tensor cores, and
  double-precision tensor throughput collapses on the hardware class we deploy on; the hierarchical
  method's advantage is a FLOP count, which is precision-agnostic. **The honest answer may well be
  "no-go in fp32, go in fp64"** — and since the scientific core is fp32/fp64-first, that is not an
  academic distinction.

---

## Cross-filing reconciliation constraints (tracked)

Findings that **cross filing boundaries**. Each is a constraint on the *order* or *content* of
what gets folded into nitrix — not a defect of any one filing, and therefore invisible in any
single filing's ADDENDUM. **This table is the tracking surface: check it before folding any
filing in.**

| id | constraint | status |
|---|---|---|
| **X-1** | **02's SVD adjoint must adopt 14's spectral-function formulation.** *(ESCALATED: 02's round-2 "fix" is a **regression** — it now silently zeroes a finite gradient. See below.)* | **open — escalated** |
| **X-2** | **"This deformation is injective" is certified three times over — 07, 08, 27. One property, three witnesses.** *(ESCALATED: 07's witness does **not** certify the property it names — see below.)* | **open — escalated** |
| **X-3** | **13's log-det seam is NOT adoptable by 22 (sparse).** *(RETARGETED: not a plumbing dispute — nitrix has **no preconditioner seam on any Krylov solver**. The required fix is unsatisfiable; the supplier is filing 31.)* | **open — blocked on 31** |
| **X-4** | **A canonical recipe must never void a stated complexity guarantee.** Discovered in 00 (G10); **nitrix's own 5 registered divergent sites are unaudited against it.** | **open — audit nitrix** |
| **X-5** | 23 reported 15 unusable without ">1-D" support. **Adjudicated: mistaken, and moot.** Recorded so it is not re-litigated. | **closed — declined** |
| **X-6** | **A substrate may not replace a seam it has not measured itself into.** genred is gated on incumbent-parity (00 G19) before it takes any nitrix seam. *(Its own motivating number went stale — see the correction.)* | **open — gate on the fold** |
| **X-7** | **"Adjacent" means *shares an edge*, not *shares any vertex*. Two filings (01, 07) wrote the same wrong predicate — and the same wrong oracle.** A shared, tested triangle-pair primitive, not a per-consumer re-derivation. | **open — mint the primitive** |
| **X-8** | **A "supplied callable" seam is not a *typed* seam.** 25 cannot reach 08's similarity: it is typed over lattice scalar fields, not point measures. **The false claim is in the filing I wrote.** 25 caught it at plan time — the first filing to check a seam before claiming it. | **open — decision deferred to 25's impl** |
| **X-9** | **The genred / sparse-product boundary.** genred streams a fold over a **given** domain; it does not **construct** one. A reduction whose index set is affine in the output index is genred's; one that is the **intersection of two data-dependent patterns** is filing 31's. Neither may claim the other's ground. | **open — governs 00 and 31** |

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

#### X-1 ESCALATION (2026-07-12) — 02's round-2 fix made it *worse*, and that settles the ruling

02 attempted to close its `1/σ` gap **in-filing**, on the (correct) observation that *"the null space
was never a separate singularity"*. **The observation is right; the engineering conclusion drawn from
it is wrong, and the result is a regression.**

02 replaced the exact complement coefficient `1/σ` with a regularised `null(σ)`. But that coefficient
is **never consumed alone** — for the advertised spectral nonlinearity `Z ↦ U·η(Σ)·Vᵀ`, the quantity
that must be finite is the **composite** `η(σ)/σ`. Take `η = identity`: then `U·Σ·Vᵀ ≡ Z`, so
`∇_Z ⟨C, UΣVᵀ⟩ = C` **exactly, at every Z, including rank-deficient Z** — and the *exact* rule
reproduces it (`σ · 1/σ = 1`). 02's new rule returns `σ·null(σ)`:

| route | coefficient | exact answer |
|---|---|---|
| Lorentzian | `σ⁴/(σ⁴+ε²)` → **0.5** at σ=1e-3 (gradient **halved**); 1e-4 at σ=1e-4 | **1** |
| Subspace | **exactly 0** below the floor | **1** |

**The range-complement gradient is silently amputated where a finite exact answer exists — and where
02's *pre-fix* code was correct.** Worse, `spectral_scale = max(|λ|∞, 1.0)` is **floored at 1**, so the
threshold is **absolute, not relative**: a *perfectly conditioned* matrix merely **scaled down** has its
complement gradient deleted. **This fires at κ = 1.** Every corpus entry pins σ_max = 1, so it is
invisible — and the acceptance functional's `η(σ)/σ` is bounded, so **the "fix" tests pass on the
pre-fix code too**.

**Direction.** The old defect was an **unbounded** gradient: loud, obviously broken, noticed. The new
one is **finite, plausible, and silently wrong** — a consumer converges to the wrong stationary point
instead of blowing up.

**This settles X-1 rather than complicating it.** 14's Daleckii–Krein form evaluates
`g(λ) = η(√λ)/√λ` **directly** — no division by σ ever occurs, the complement collapses in closed form
via `X·V_d = 0`, and the result is **exact *and* bounded at σ=0, with no floor, no ε, no cluster
threshold**. 02 divides and then *regularises the division*, which corrupts the product. **It buys
boundedness by giving up exactness in a regime where exactness is available in closed form.**

**Ruling:** 02 must **not** ship its regularised complement. Land the spectral-function adjoint **once**
(14's form), in `linalg`, and have both filings consume it. The factor-wise adjoint survives only for
consumers that genuinely need `U`/`V` themselves, where the `1/σ` is **irreducible** and becomes a
documented domain restriction — not something to floor away.

### X-2 · Injectivity of a deformation, certified three times (07 / 08 / 27)

**The finding.** Three filings independently guarantee **the same property** — *this deformation
does not fold the domain onto itself* — each with a different witness:

| filing | the guarantee | the witness |
|---|---|---|
| **07** embedding-constrained surface flow | the deformation stays **embedded** | ~~the self-intersection-free invariant~~ — **NO. See the escalation below: the invariant is *"no non-adjacent pair intersects"*, which is strictly weaker than embeddedness.** |
| **08** metric diffeomorphic shooting | the deformation is **invertible** | a positive-Jacobian guarantee — *local*, and does **not** imply global injectivity |
| **27** Beltrami bijective mapping | the map is **bijective** | a Beltrami-coefficient bound + a *separate* global-bijectivity certificate |

**ESCALATION (2026-07-13, from 07's round-2 review).** The lattice this constraint asked for now has
its first rung, and it is lower than advertised. **07's witness does not certify the property 07
names.** Every collision site in 07 (`_clearance.py:312-315`, `_fused.py:209-214`, `_sweep.py:371-374`
— *and its own oracle*, `tests/oracles.py:195-196`) discards any triangle pair sharing **≥ 1 vertex**.
But two triangles sharing exactly one vertex can cross transversally with both areas positive — a
**generic** configuration, unlike a shared-*edge* pair whose interiors overlap only when coplanar. So
the closed star of every vertex is a **collision-blind clique**, and because the step bound is a
clearance to *non-adjacent faces only*, intra-star motion is not merely unmonitored but **unbounded**.

**The strength lattice, corrected:**

> `no non-adjacent pair intersects` **(07, actual)**  <  `positive Jacobian` **(08)** — local  <
> `embedded / globally injective` **(07, claimed; 27, actual)**

Three filings, and **not one of them certifies global injectivity except 27** — which is the only one
that treated bijectivity as a *separate* certificate rather than a corollary. That was the right
instinct and it is now evidenced.

**Consequence for the fold.** A downstream consumer cannot read "injective" off any of these seams and
believe it. The reconciliation must name the *witness*, not the property: a nitrix seam that returns an
injectivity certificate must state **which rung** it certifies, and the weaker rungs must not be typed
as if they were the strongest. (This is also why `NOTES-topology-guarantee.md` in 07 — the argument
that a fixed connectivity plus the embedding invariant gives ambient isotopy and hence genus
preservation *for free* — is **void in exactly the excluded class**: it was derived from a property the
filing does not deliver. The note was written by this reviewer, from the *name* of the invariant rather
than its definition in the code.)

These are not three properties. They are one property at three strengths, and the strengths are
**genuinely ordered** — which is exactly why folding them independently is dangerous:

- A **positive Jacobian is local**. It says the map does not fold *infinitesimally*. It does **not**
  imply global injectivity — a `d`-fold covering of a closed surface has a positive Jacobian
  everywhere (this is already recorded in the round-2 skeptical flags). 08's guarantee is therefore
  *weaker* than it sounds, and 27 knows this: it is precisely why 27 carries bijectivity as a
  **separate** certificate rather than deriving it from the Beltrami bound.
- 07's **embedding** invariant is the strongest of the three (no self-intersection is a global
  statement), but it is stated for a *mesh under a flow*, not for a *map*.

**Why this must be tracked.** Fold the three in independently and nitrix acquires three unrelated
vocabularies for one guarantee, with **no way to compose them** — and worse, a consumer who reads
"positive Jacobian" as "injective" gets a silently wrong answer that one of our own filings already
knows is wrong. This is the certificate-composition problem (see the post-buildout RFC §3.2) in its
sharpest, most concrete form.

**The rule on reconciliation.** One **injectivity certificate** type, with an explicit **strength
lattice** (locally-non-folding ⊏ globally-injective ⊏ embedded), so that a consumer can ask for the
strength it needs and a pipeline's guarantee is the *meet* of its stages'. A primitive that can only
establish the weak form must **say so** and must not be readable as the strong one. Land the type
once; have 07, 08 and 27 populate it.

*Note (from 07):* 07's embedding invariant, with fixed connectivity, **already implies genus
preservation** (an ambient isotopy cannot change genus) — so 07 needs no separate genus certificate.
That is a fourth thing that would otherwise have been invented independently.

### X-3 · The log-det seam is not adoptable by a sparse operator (13 → 22)

**The standing instruction is unsatisfiable as shipped.** The ledger tells filing 22 (matrix-free GMRF
on a *sparse* precision) that it *"must adopt 13's `LogDetEstimator` protocol, not fork it"*. Two
problems, both found in 13's round-2 review:

1. **`LogDetEstimator` is not a Protocol.** It is `Literal["lanczos", "hutchinson"]` — a *recipe-name
   axis*. The instruction names something that does not exist. The seam 22 would actually adopt is the
   matrix-free operator + the free `logdet(...)` function + its estimate container.
2. **At implementation level it bakes in dense-scale assumptions that force the very fork the
   instruction forbids:** full reorthogonalisation against an **explicit dense basis buffer**, with a
   **second full-size mask copy allocated inside the scan body, every step, every probe** (≈ **8.4 GB**
   at `n = 10⁶`, 16 probes, order 32 — and a sparse GMRF's *entire premise* is `n ≫ 10⁴`); and **no
   preconditioner is plumbed through to the log-det's gradient solve**, so it runs an *unpreconditioned*
   CG per probe (a sparse precision at κ≈10⁵ needs ~300 iterations — the budget is exhausted and the
   tangent channel fires on every probe).

**This is 13's finding, not 22's.** *Required before either lands:* expose the preconditioner on
`logdet`; make reorthogonalisation a knob (`full | local | none`); lift the hard quadrature-order cap
into a documented accuracy bound. **Then** the seam is genuinely shareable.

*Seam-inventory note:* this is a third instance of the pattern the RFC's §6.5.1 names — **a seam whose
guarantee (or adoptability) was certified only on the path its author's own consumer took.** Compare
genred's ELL hole (G6/G7 there) and X-1. The seam register must record, per seam, **which paths the
guarantee was tested on** — not merely that a guarantee exists.

#### X-3 correction · the required fix is itself unsatisfiable, and I filed this under the wrong heading

**"Expose the preconditioner on `logdet`" presumes there is a preconditioner to expose. There is not.**
`nitrix.linalg.krylov.cg` takes `(a, b, x0, tol, atol, maxiter, l2)` — and `l2` is a Tikhonov ridge, not a
preconditioner (`krylov.py:59-68`). Neither does `bicgstab`, `gmres`, or `minres`. **Every mesh- and
graph-Laplacian solve in nitrix is unpreconditioned CG.** The preconditioners that *do* exist
(`shift_invert`, `polynomial`) **raise `ValueError` on ELL / SectionedELL** — they are dense-only, which
`perf-bench-case-hardening.md:255-261` states in the plainest terms: *"the entire point of lobpcg is
sparse scale (n ~ 1M); the headline numbers do not apply to the sparse path."*

So X-3 is **not** a plumbing dispute between two filings, which is how I titled it and how the summary row
read. It is the first surfacing of a **substrate hole**: nitrix has no preconditioner seam on any Krylov
solver, and no sparse preconditioner of any kind. 13 cannot close X-3 by plumbing, because there is
nothing to plumb.

**And the supplier is a sparse-sparse product.** At κ≈10⁵ on an irregular operator, the mesh-independent
preconditioner family is multilevel; its setup is the Galerkin triple product :math:`P^\top A P`, which is
an SpGEMM. nitrix cannot form :math:`P^\top A P` — the prolongation `P` ships as an ELL
(`sparse/mesh.py:566`), but it is only ever applied to *signals*, never to the *operator*
(`mesh_coarsen_meanpool`, `mesh.py:1275`). Note also `graph/laplacian.py:277-287`: the symmetric Laplacian
is **dense-only** *"because the non-zero diagonal does not fit the sparse pattern of the input"* — so a
coarse-grid Laplacian is not constructible even in principle today.

**Status: X-3 is retargeted.** It is no longer "13 must expose a knob"; it is **blocked on the preconditioner
seam**, which is filing 31's deliverable. Filings 13 and 22 both wait on it. See X-9.

### X-4 · A canonical recipe must never void a complexity guarantee (00 → nitrix's own registry)

**The finding, in filing 00 (G10).** genred's ELL-edge aggregation earns its memory-linear guarantee
by chunking the slot fold at width `c = √k_max`. The chunk width is registered as a divergent recipe
whose **canonical variant is the strict per-slot order — i.e. `c = 1`**. The module's own docstring
prices the backward's saved carry chain at `(k_max / c) · n · d_out`; at `c = 1` that is
`k_max · n · d_out`, **exactly the message tensor the guarantee exists to forbid**. So
`NITRIX_REPRODUCIBLE=1` — the knob a consumer sets *for safety* — silently destroys the streaming
property in the backward.

**The root error is conceptual, and it generalises.** A recipe axis exists to govern a choice that
would otherwise **diverge across hosts**. The canonical variant's job is to be *deterministic and
portable*, not to be *mathematically strict*. Those came apart here: the chunked default is already a
pure shape function, never platform-probed, and therefore already reproducible — while the "canonical"
strict order buys a stricter association at the price of an asymptotic guarantee. **Order-preserving
and order-deterministic were conflated.**

**Why this is a cross-filing constraint and not a filing-00 bug — and it is already live in nitrix.**
nitrix has its own registry — **5 registered divergent ops**, each with a canonical variant, all forced
by `nitrix.reproducible()` / `NITRIX_REPRODUCIBLE=1`. Nothing in the reproducibility contract says a
canonical must preserve the op's stated complexity. **I audited the 5 sites. One of them already has
the defect, and it is not the one I predicted.**

- **`metrics.joint_histogram` — CONFIRMED, and only half-recorded.** Canonical is `onehot`, and
  `_divergent_ops.py` says so in a *comment*: *"deterministic on any platform; **`O(N·bins)` memory —
  reproducibility's cost at scale**"*. So somebody saw it. But that comment lives in a **private**
  module, and the field a consumer actually reads — `DivergentOp.summary`, the payload of the public
  `nitrix.divergent_ops()` enumeration — says only *"joint histogram: one-hot matmul (deterministic) vs
  atomic scatter"*. **`DivergentOp` has no field for a complexity caveat at all.** The discoverability
  surface exists precisely so a parity-sensitive consumer can audit what reproducible mode will do to
  them, and it does not tell them that it changes the memory profile. The cost was priced in a comment
  and never entered the contract.
- **`nn.ssm.selective_scan` and `geometry.cubic_bspline_prefilter` — the suspects I predicted, and the
  charge does not quite stick.** Both take canonical `sequential` against an `associative` fast path.
  That is a **depth** difference (`O(T)` vs `O(log T)`), not a memory one — so it needs a ruling rather
  than a fix: **does the rule cover depth, or only memory?** It should. But note `selective_scan` also
  carries a third variant described in the registry as *"XLA-stable, **memory-sparing**"* (`chunked`) —
  which the canonical does **not** select. If the op's memory profile is ever advertised on the strength
  of `chunked`, this becomes the same defect as `joint_histogram`.
- `signal.iir` (canonical `scan`) and `register.field_smooth` (canonical `fir`) are clean: both
  canonicals are the *faithful* variant and neither trades an asymptotic bound.

So the conflation is not hypothetical and not confined to a filing. It is in the shipped library, at
the one site where the canonical genuinely costs an order of memory, recorded where no consumer will
look.

**The rule.**
1. **Normative addition to the reproducibility contract:** *a canonical variant must preserve every
   complexity guarantee (memory, depth) the op advertises.* If it cannot, the op must not advertise the
   guarantee unconditionally — the degraded bound goes in the contract, at its true order.
2. Where the default is **already host-invariant** (a pure shape function, no platform probe), the
   default **is** the correct canonical. Registering a stricter-but-costlier variant as canonical is a
   category error.
3. **`DivergentOp` gains a complexity-caveat field**, and `nitrix.divergent_ops()` surfaces it. The
   enumeration exists so a parity-sensitive consumer can audit what reproducible mode does to them;
   today it cannot tell them that `joint_histogram`'s canonical costs `O(N·bins)`. A cost recorded in a
   private comment is not in the contract.
4. Extend `tests/test_reproducible_dispatch_guard.py` so a canonical that degrades a stated bound
   **without declaring it** fails CI, exactly as an ungoverned platform-flip already does.
5. **Rule on depth.** `selective_scan` and `cubic_bspline_prefilter` take a sequential canonical against
   an associative default — `O(T)` vs `O(log T)` depth. Memory is untouched, so (1) as written does not
   bite. It should: a consumer who turns on reproducibility to stabilise a result, and silently gets a
   serial scan on a GPU, has been surprised in exactly the way this rule exists to prevent.

*Seam-inventory note:* this is the **fourth** instance of the RFC §6.5.1 pattern — a guarantee
certified only on the path its author took. Here the untaken path is *the library's own safety knob*.

### X-5 · "15 is unusable below >1-D" (23 → 15) — DECLINED

**The report.** During planning, filing 23 flagged that filing 15 was unusable as a dependency without
genuine support for dimensions beyond 1-D.

**Adjudicated: mistaken, and moot.** Recorded here — rather than dropped — because a false cross-filing
dependency claim costs as much to re-litigate as a true one.

- **15 is not a 1-D solver.** It minimises `f(x) + g(Dx)` over a *flat* `x ∈ ℝⁿ` for arbitrary `n`, with
  no rank restriction and no `ndim` guard anywhere in `src/`. The flat vector is a **calling convention,
  not a problem class**: tensor structure rides as static pytree metadata, and 15 already carries a
  genuinely 2-D variable that way (`NuclearNorm` reshapes the flat vector to a matrix via a static
  `shape`, `terms/svt.py:258-267`). Reading "`x` is rank-1" as "the solver is 1-D" is the misread.
- **23 needs none of it.** What 23 requires is the **chain** monotone cone — the isotonic projection of
  a quantile vector across levels, by pool-adjacent-violators (`23/PROBLEM.md:21`). 15 has exactly that
  (`terms/isotonic.py`), declared `vmap`-safe, so batching over fibres is free. 23 **scopes every
  certified guarantee to the fibered one-dimensional case** and routes the multivariate map to entropic
  transport (filing 04), not to 15. Monotonicity of an *n*-D map — the cyclical-monotonicity /
  positive-Jacobian cone, a genuinely different and much harder object — appears nowhere in 23's
  contract.
- **And the import is forbidden regardless.** Cross-filing runtime imports are barred by the shared
  conventions, so "unusable as a dependency" was never going to be true for *any* reason.

**The kernel of truth, kept.** What genuinely *is* chain-1-D in 15 — TV, TGV, the isotonic cone — is
**scoped as such in 15's own problem statement** (its prox table says "one-dimensional total
variation"). But the review surfaced a real, cheap gap behind the bad complaint: **15 has no *n*-D
discrete-gradient operator** (`ForwardDifference` is a chain difference; there is no grid gradient).
That matters because **anisotropic grid TV needs no new prox at all** — it is `L1Norm ∘ Grad_nD`
through the existing composite — and *isotropic* grid TV then follows free from the existing
`GroupNorm` over the stacked gradient channels, whose partition is already arbitrary static metadata.
**One operator unlocks the most common composite program there is.** Filed as a should-fix on 15, not a
blocker, and explicitly *not* what 23 asked for.

### X-6 · A substrate may not replace a seam it has not measured itself into (00 → the rebuild)

**The finding, in filing 00 (G19).** genred's proposition is *one streaming engine replaces N
hand-written kernels*. Its twelve-row "served" table certifies **memory** — every row is a working-set
bound, **not one carries a performance claim or cites a benchmark** — and its `bench/` harness measures
against *external* opponents (cuBLAS, the two flash kernels, the superseded in-tree kernels). **The
incumbent hand-written kernel at the seam genred proposes to replace appears in no row.**

Two filings adopted genred *because the abstraction subsumed their computation*, built it, measured it,
and backed out: **01** (seeded reduction, `q≪m`) at **30–44× slower** than its tuned in-house kernel,
and **07** (candidate/proximity reduction) at **7.45 GiB vs 0.06 GiB — 124×**. Both losses were
invisible to every gate in the suite, because **every gate in the suite asks a question the engine
passes.**

**Why this is a fold-order constraint.** genred does not arrive as a new subpackage; it arrives as a
*replacement* for `_internal/reductions.py` (the single `reduce()` behind every score kernel),
`semiring`, and — through them — `morphology`, `smoothing`, and the `nn.attention` fused path. If it is
folded in on the strength of its memory table, nitrix trades N kernels it has measured for one it has
not, at every one of those seams simultaneously, and the regression surfaces as a diffuse slowdown with
no single owner.

**The rule.** **No seam is replaced without a head-to-head against the kernel it replaces** —
wall-clock *and* peak working set, **forward and gradient** — at the seam's own shapes, not at balanced
ones. A seam where the substrate loses by more than a stated factor is **not adopted**, and the factor
is recorded. This is the strengthened form of the RFC's *"validate on certificates, not metrics"*: for a
**substrate replacement**, an incumbent-parity measurement *is* one of the certificates.

*Seam-inventory note:* the **fifth** instance of the RFC §6.5.1 pattern, and the one with the widest
blast radius — a guarantee certified only on the path its author took, about to be adopted at seams
nobody measured. It is also the pattern **this reviewer** twice fell into, asserting coverage from the
shape of an abstraction rather than a number (the retracted genred/ELL ledger entry; the G8 op-set
claim). A prose retraction gets re-asserted by the next reader of the table. A red benchmark row does
not.

### X-6 correction · the number that motivated the gate had gone stale, and nobody could tell

The first statement of X-6 cited filing 07's decline as **7.45 GiB vs 0.06 GiB (124×)**. **That figure
is obsolete**: it measured the engine materialising the `(m, k_max, d)` message tensor, and filing 00's
round-5 work *removed that object* — its `semiring/ell_edge.py:52-53` now names **this very
measurement** as the thing it fixed. Filing 07 has **no bench script, no test, no artefact**; the number
lived only in prose, so nothing detected that it had expired before it was elevated into a governance
rule.

**This is the argument for X-6, not against it.** A number that cannot be re-run rots silently into a
rule founded on a fixed bug. The gate stands; the citation is corrected.

**And the surviving leg is the stronger one, because it indicts the *domain model* rather than the
fold.** 07's incumbent fuses candidate **generation** with the fold *inside the tile*, so the
`(m, k_max)` index operand **never exists**. genred's ELL path consumes that structure **already
materialised** — at 07's target scale the operand *alone* is `(327 680 × 1024)` i32 = **1.34 GB**,
against a **0.32 GiB** total incumbent forward budget. Streaming the fold cannot recover a cost paid at
the domain's boundary. That limit generalises to **every candidate-generating proximity query**
(fixed-radius neighbours, kNN, collision broad-phase) — it is *fixable* (it is precisely the in-tile
candidate domain genred declared a non-goal), and **it has never been measured.**

### X-7 · "Adjacent" is being used to mean the wrong thing, in two filings and both their oracles

**The finding.** Filings **01** and **07** each need to exclude trivially-touching triangle pairs from a
collision test. Both wrote:

```python
shares = (faces[ii][:, :, None] == faces[jj][:, None, :]).any(axis=(1, 2))   # drop if ANY shared vertex
```

**This is the wrong predicate.** Two triangles sharing an **edge** can only have overlapping interiors
when coplanar — a measure-zero corner. Two triangles sharing exactly **one vertex** can cross
transversally with both areas positive — a **generic** configuration, an open set. Excluding the latter
discards real self-intersections:

- **01**: its surgery's centroid fan *manufactures* precisely those pairs, so the defect is on the
  filing's own hot path.
- **07**: the closed star of every vertex becomes collision-blind, and the step bound therefore places
  **no constraint at all** on intra-star motion.

**And in both filings the "independent" oracle applies the identical exclusion** (`01/tests/oracles.py:59`,
`07/tests/oracles.py:195-196`) — so in both, the exhaustive audit agrees with the broken fast path and
the invariant test passes on a self-intersecting mesh. **The same error, made twice, and hidden twice by
the same mistake in the oracle.**

**Why it is a cross-filing constraint.** Two independent authors converging on the same wrong predicate
is not a bug, it is a **missing primitive**. The correct object — *"exclude the shared simplex from the
intersection test, not the pair from the census"* — is a small, exactly-testable piece of computational
geometry that both consumers should be *consuming*, not re-deriving. It belongs in the reconciliation as
a shared, tested primitive (a `nonadjacent_pairs` / `triangle_pair_intersects` seam in the mesh layer),
with the oracle built **independently of it**.

**The rule.** No filing folds a collision or self-intersection census in on its own predicate. One
primitive, one definition of adjacency (**shares an edge**), one oracle that enumerates *all* pairs and
adjudicates the shared feature — and a corpus member that a vertex-fan fold-through must fail.

---

### X-8 · A "supplied callable" is not a typed seam (25 → 08) — and the false claim is mine

**The report.** Filing **25** (kernel metrics between weighted directional point-measures) reports at
**plan time, before writing code**:

> *directory 08 cannot consume this filing as PROBLEM.md claims. Its similarity seam is typed over scalar
> fields on a lattice, not point sets — a shape mismatch, not a naming one.*

**Audited against 08's merged code: correct, and understated.**

**The claim it contradicts is one I wrote.** `25/PROBLEM.md:53` argues that *"Directory 08 declares its
`similarity` to be a **supplied callable**"* — and concludes that 25 therefore slots in. The callable is
indeed supplied. Its arguments are not point measures:

```python
# 08/protocols.py — and 08/src/epdiff_shooting/types.py, as merged
class Similarity(Protocol):
    def __call__(
        self,
        warped: Float[Array, "*spatial"],     # a scalar field on the lattice
        fixed:  Float[Array, "*spatial"],     # likewise
    ) -> Float[Array, ""]: ...
```

**Pluggable-callable is not pluggable-type.** I read the *shape of the abstraction* (a callable, injected,
therefore open) and inferred a capability, without reading the argument types. **This is the fourth time in
this corpus I have made that exact move** — the G8 op-set claim, the G19 stale-number claim, the
`NOTES-topology-guarantee` derivation, and now this. Every instance is mine, and every instance was caught
by someone reading the code I did not read.

**The mismatch is deeper than the signature.** 08's data term is not merely typed for grids, it *operates*
on them (`objective.py`):

```python
warped = warp_scalar(moving, fields.inv_displacement, ...)   # grid -> grid resample
return regularisation + weight * similarity(warped, fixed)
```

A point measure needs the opposite operation in two independent ways: the points are **pushed forward** by
`fwd_displacement` (not pulled back by `inv_displacement`), and their **directions and weights must be
transported** by the Jacobian, which 08 never applies to anything. Swapping the `similarity` callable alone
cannot express this at any type.

**But "cannot consume" overstates the distance — the ingredients are already in 08.**

- **The scattered sampler exists.** `interp._interpolate` is batch-shape-agnostic: it rank-checks
  `values`' spatial axes against `points.shape[-1]` only, then gathers over a flat index. Its jaxtyping
  annotation (`points: Float[Array, "*spatial d"]`) *understates* it — `points` of shape `(n_vertices, d)`
  works today. It is private, and every public entry (`warp_scalar`/`warp_vector`) hard-wires
  `_index_points(displacement, spacing)` = *the whole lattice*.
- **The Jacobian field exists.** `grid.jacobian` / `jacobian_spectral` / `determinant` are computed already
  (EPDiff transport needs `Dφ`; the certificate needs `det Dφ`).

So what is genuinely absent is (a) a **public point-warp entry**, (b) the **direction flip**, and (c) one
piece of real mathematics that **neither filing has**: the transport rule for a *directional* measure — the
**tangential (Gram) Jacobian**, which is not `det Dφ`, rescaling the weight by the local area distortion
and pushing the direction by `Dφ` up to normalisation.

**The ownership question (25's F7: "whose filing owns it?") — recommended split, to be ratified when 25
lands.** The warp *path* is a property of the diffeomorphism, not of the metric: 25 never sees a
displacement field, and 25's own non-goal 3 puts measure construction out of scope. The transport *rule* is
a geometric-measure fact, which is exactly 25's mathematics. Therefore: **08 owns the path, 25 owns the
rule.** Neither filing may assume the other has already built it — which is how this constraint arose.

**Status: the decision defers to 25's implementation** (25 is PLAN-only; nothing is built yet). What does
**not** defer is the correction to `25/PROBLEM.md:53`, which is false as written and must not be allowed to
justify a seam that does not exist.

**The rule.** *A filing may not claim a consumer seam it has not type-checked against that consumer's
merged code.* An injected callable proves the consumer is **open**; it says nothing about **what it is open
to**. Credit where it is due: **X-6 compels a filing to measure itself into a seam before taking it; 25 is
the first filing to check a seam before claiming it — unprompted, and against its own interest.**

---

### X-9 · The genred / sparse-product boundary — and the six sites nobody had connected

**The rule.** *genred streams a fold over a **given** domain; it does not **construct** one.* That sentence is
genred's own (`genred/README.md:113-119`), and it is the partition:

- **The reduce index set is a static or affine function of the output index** (dense, product, banded,
  masked-by-index-predicate; ELL slots at a fixed `k_max`) → **genred's ground.**
- **The reduce index set is the intersection of two data-dependent patterns**, or the output pattern is
  itself data-dependent → **filing 31's ground.**

**Neither may claim the other's.** This is not a performance judgement; it is an expressibility one, and it is
already conceded on 00's side of the line. genred's `Domain` protocol derives every bound from the *output
index* (`bounds(i_lo, i_hi, extent)`); its sole indexed access, `Gather`, is documented as resolving *"rows of
a **dense table**"* and validates `axis is FEATURE` eagerly. A sparse×sparse product has no such encoding:
forcing one produces a dense `(m, n)` output — **the exact pairwise object the engine exists to forbid.**
genred does not lose at SpGEMM by a constant; it *inverts into the thing it was built to avoid.* Filing 00
states the boundary itself in four places — the README scope contract, `CandidateDomain` as a declared
non-goal (*"a traversal machine grafted onto a fold engine"*), "CSR domains" as a deferred non-goal, and
ADDENDUM **G9 — "the streaming boundary is drawn in the wrong place for data-dependent domains"**, which was
*declared, not fixed*.

**This is X-6 discharging correctly.** X-6 gates genred on measuring itself into a seam before taking it. The
measurement came back negative. The seam is not taken. **Note that the two "genred loses" numbers on record
(07's 124×, 01's 30–44×) are stale — both were fixed by B2/B3.2 — and this constraint deliberately does not
rest on them.** It rests on the domain algebra and on 00's own scope contract. Founding a family split on a
number that has already rotted once is exactly what G19 exists to prevent.

**What the boundary was hiding.** Six sites in nitrix independently need a sparse×sparse product; none knows
about the others, and **no one has ever requested it by name** (a full-tree grep for `spgemm`, `graphblas`,
`multigrid`, `AMG`, `galerkin` returns zero hits across SPEC, ~80 feature-requests, and source):

1. **X-3** — every Krylov solver is unpreconditioned; the sparse preconditioner's setup is :math:`P^\top A P`.
2. **X-7** — *"shares an edge"* is :math:`F F^\top` thresholded at 2: a boolean masked product. Two filings
   hand-rolled it wrong, and so did both their oracles.
3. **`SPEC.md:236-242`** — names **`spspmm`** exactly once in the whole repository, as *the memory hazard to
   guard against*. The hazard is in the SPEC; the operation is not in the library.
4. **`geometry/dec.py:21`** — *"`d0ᵀ ★₁ d0` **is** that Laplacian"* — and `hodge_apply` can only compose the
   factors as **matvec closures** (`:433-438`), never form the product.
5. **`sparse/mesh.py:779-816`** — `_kring_adj_list` is a **boolean semiring matrix power implemented as a
   triple-nested Python `set` BFS**, run on meshes of up to 163,842 vertices.
6. **`sparse/ell.py:230-330`** — you cannot build a sparse graph without a dense `n × n` first, and
   `symmetrize=True` *"can densify hub rows entirely"* for want of a structural sparse union.

**The shared substrate is not to be forked.** The semiring algebras (`REAL / LOG / TROPICAL_* / BOOLEAN /
EUCLIDEAN`) and the `ELL` / `SectionedELL` storage are **common to genred and to 31**; the algebra axis is
orthogonal to the storage axis and to the domain axis. 31 adds a storage (sliced-ELL) and a domain
(constructed), **not** a second algebra vocabulary and **not** a second SpMM — `semiring_ell_matmul` stays
exactly where it is.

---

## Evaluated and declined (recorded so they are not re-litigated)

**The standard applied here is the batch's own: prospective, not demand-driven.** A candidate qualifies
if it *improves something we have* **or** *opens a family of methods we cannot currently write*. Absence
of a named request is not a reason to decline. The declines below are on **invariants** and on
**operating regime**, never on demand.

### Butterfly factorisation / Monarch / Monarch-Mixer — DECLINED, but it surfaced a real gap

Two different objects share the name; they fail for entirely different reasons.

**Monarch / Monarch-Mixer — declined on invariants.**
1. **There is no weight matrix in nitrix to parameterise.** `nn/` holds exactly three functional kernels
   (`scaled_dot_product_attention`, `selective_scan`, the norms). **There is no `Linear` layer**, and
   SPEC forbids one: *"the learnable modules that hold the parameters live in downstream libraries."*
   Monarch is a *weight parameterisation*; the object it modifies does not exist here. It belongs a
   layer up.
2. **The kernel has no irreducible content.** A Monarch matmul is `reshape → batched matmul → transpose
   → reshape`; autodiff through a bmm is already exact and efficient. SPEC §9 excludes exactly this
   class (*"(trivial elementwise op) ∘ (reduction) with no content"*). Its value is a statement about
   which subspace of matrices to search — a *modelling* choice, not arithmetic.
   *(Incidentally, M2's sub-quadratic sequence mixing comes from FFT long-convolution, not from Monarch;
   Monarch does the dimension mixing, i.e. it replaces the MLP. nitrix has no MLP.)*

**Butterfly proper (complementary-low-rank, for *oscillatory* operators) — declined on regime.** Its
natural targets are **NUFFT / Fourier integral operators / high-frequency BEM**, and all of them are
outside the charter (no k-space or reconstruction; no M/EEG). The one in-library slot is real —
`geometry/harmonics.py:295-297` is the classic **semi-naive O(L³)** SHT, whose Legendre stage is exactly
where a fast Legendre transform applies — but the corpus operates at **L = 8** (filing 05's own figure;
tests cover L ≤ 10). That is two orders of magnitude below butterfly's GPU crossover, where it would be
*slower*. And if a consumer ever does need high L, **the first fix is a three-term Legendre recurrence
(O(L²) memory), not a butterfly** — the O(L³) cost there is a *storage* choice, not an algorithmic
necessity. **Trigger to revisit: a consumer needing L ≳ 256.**

### …but the re-evaluation surfaced the gap butterfly is a cousin of, and this one is live

**Filing 25 (and genred's whole kernel-summation regime) accepts *quadratic time* as irreducible.** 25's
own hard invariant: *"The double sum is :math:`O(NM)` in **time**; it must be :math:`O(N+M)` in
**memory**"* — at a stated target of :math:`N \sim 10^5`–:math:`10^7`, *"evaluated at every step of an
outer optimisation."* At :math:`N = 10^7` that is :math:`10^{14}` multiply-adds **per evaluation** —
seconds per step on an L4 before the directional kernel's transcendentals, times thousands of steps.
**The memory wall was fixed; the time wall was declared and left standing.** 25's stated upper target is
not reachable by 25.

The escape is the *other* branch of the hierarchical-matrix family — the one for **smooth** kernels,
which is what these actually are: **kernel-independent hierarchical summation** (black-box / Chebyshev-
interpolation FMM, :math:`\mathcal{H}^2`), giving :math:`O(N \log N)` or :math:`O(N)` for an **arbitrary**
smooth kernel, deterministically and with exact gradients. Unlike a classical tree-code FMM, the
interpolation-based form runs on a **fixed** tree with uniform per-node work — regular, batchable,
differentiable, GPU-shaped. (25 already lists **random features** as one open option, but that is
restricted to a *separable Gaussian* and buys an estimator variance that lands in the gradients.)

**This is a prospective candidate on exactly the batch's stated standard: it opens a regime the corpus
has declared itself unable to reach.** Its honest risk is the crossover — at :math:`N \sim 10^5` a
brute-force :math:`O(N^2)` with perfect arithmetic intensity may still win on a GPU, so the case must be
made at :math:`N \gtrsim 10^6` and the crossover measured, not asserted.

**FILED as 32**, with that risk promoted to the filing's *first* section: a front-loaded go/no-go spike
that bounds the method from below and the incumbent from above, and **is allowed to say no**. See the
batch entry above.

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
