# nitrix-moonshot round 2 — candidate problems and frontier kernels

**Status:** living ledger. Created 2026-07-09; filings 15–28 minted 2026-07-10.

## Purpose and method

Round 1 (`nitrix-moonshot-ledger.md`) filed 14 strictly-numerical problems for the
restricted assistant. Assistant credit turns out to be the non-binding constraint, so
this ledger opens a second, deliberately more forward-looking candidate set: **eight
frontier-and-evergreen problems in brain mapping**, and the **numerical kernels that
would unblock them or qualitatively change how they are approached**.

Method: a coverage audit of the shipped surface (15 subpackages) plus the 14 round-1
filings, followed by eleven parallel literature sweeps (SOTA + deliberately
adversarial "what is overlooked, what is popular but numerically weak"). Each sweep
was asked to state the bottleneck *in pure-mathematical terms* and to name a
hoistable kernel with an oracle.

**Scope exclusions honoured throughout:** no M/EEG/fNIRS (MNE's domain), no image
reconstruction / k-space, no pulse-sequence design. Everything below lives in the
image domain or above it.

**Eligibility, unchanged from round 1.** A candidate is admissible iff a pure
numerical core can be hoisted, leaving the biological bridge as a separate seam. The
SPEC §9 *concrete blocked consumer* gate stays relaxed; every firm invariant stays
binding (separation of concerns; no I/O; no PyTree/Equinox modules; no `loss`
namespace or scalarisation; runtime deps `jax`/`jaxtyping`/`numpy` only; pure,
differentiable, fp32/fp64-first; two-tier `pallas ≈ jax` parity with loud fallback).

**Deliberately under-specified.** Round-1 filings were tightly specified. These are
not: each kernel below is stated as a mathematical contract plus an oracle, and
leaves the *recipe* to the solver. That is the point — several of these are open
enough that prescribing the method would be prescribing the wrong one.

## Coverage audit — what is genuinely absent

Confirmed absent from `src/nitrix/` and from the round-1 batch: eikonal /
Hamilton–Jacobi solvers of any kind; Kalman filtering/smoothing; proximal operators,
ADMM, or any nonsmooth-convex machinery; phase unwrapping; integer/min-cost-flow
programming; NNLS; selected inversion (Takahashi); stochastic Lanczos quadrature;
SPDE/GMRF assembly on a mesh; DPSS/multitaper; Burer–Monteiro; parallel transport,
Fréchet means or geodesic regression on non-SPD manifolds; quasiconformal/Beltrami
machinery; polynomial-system solving; monotone/isotonic projection.

Present and load-bearing for what follows: `fixed_point_solve`, `implicit_minimize`,
`implicit_least_squares` (a **smooth** implicit-diff seam — the nonsmooth one does
not exist); `cg`/`gmres`/`minres`/`bicgstab`; `matrix_function`/`chebyshev_apply`;
`mesh_cotangent_laplacian`/`mesh_mass_matrix`; `hodge_decompose` (DEC); `sinkhorn`;
`semiring_*` + `genred`; `symlog`/`symexp`/`tangent_project_spd`.

---

## The eight problems

Each entry gives the **anchor** (domain motivation — stays in this repo, never in a
filing), **why it is stuck**, the **kernels** it implies, the **overlooked bet** the
literature sweep surfaced, and what it **builds on** from round 1.

### P1 · Tractography as a differentiable global inverse problem

- **Anchor.** Whole-brain structural connectomics. The field's own verdict is that
  streamline tractography recovers more invalid bundles than valid ones
  (Maier-Hein 2017, `10.1038/s41467-017-01285-x`) and that orientation-only data
  fundamentally under-determines the tracts (Thomas 2014,
  `10.1073/pnas.1405672111`). The convex filtering stage (SIFT2; COMMIT2,
  `10.1126/sciadv.aba8245`) is a large sparse non-negative least-squares problem
  with a group-lasso bundle prior — mature, and *not* the bottleneck. The bottleneck
  is that its dictionary `A` is produced by an upstream, non-differentiable,
  orientation-only streamline generator.
- **Why it is stuck.** Nobody has fused the front-propagation forward model with the
  connectivity inverse problem into one differentiable program. Geodesic
  tractography, the obvious candidate, is *popular but numerically weak*: inverse-DTI
  geodesics cut corners because the metric rewards straightness — an intrinsic
  pathology, not a discretisation artefact.
- **Kernels.** **K1** (anisotropic/Finsler Hamilton–Jacobi solver with exact discrete
  adjoint). Also serves cortical-ribbon geodesic distance and — via a
  volume-preservation constraint — equivolume laminar coordinates (Waehnert
  `10.1016/j.neuroimage.2013.03.078`), which are currently ad-hoc geometric
  constructions rather than a certified PDE solve.
- **Overlooked bet.** Curvature-penalised **sub-Riemannian front propagation on the
  orientation lift `R³ × S²`** (Duits, Mirebeau, Bekkers; arXiv:1612.06137). Lifting
  kills the shortcut pathology *by construction* and preserves crossings. It is
  under-used because the 5-D lift breaks the `d ≤ 3` guarantee of Selling/Voronoi
  stencil reduction — which is exactly the open numerical problem worth owning.
- **Builds on.** Round-1 filing 03 (intrinsic mesh geodesics) — K1 is its
  strongly-anisotropic, Finsler, volumetric generalisation.

### P2 · Microstructure inversion that does not hide its degeneracy in a prior

- **Anchor.** Biophysical model inversion in diffusion MRI (the "Standard Model":
  NODDI, SANDI, SMT). The two-branch degeneracy (Jelescu 2016, `10.1002/nbm.3450`)
  is exact and algebraic: the LEMONADE moment expansion reduces it to a *quadratic*
  (Novikov 2018, `10.1016/j.neuroimage.2018.03.006`). At SNR≈50 the two basins'
  cost differs by <10%, so goodness-of-fit cannot select between them.
- **Why it is stuck.** With single-diffusion-encoding data the ambiguity is
  intrinsic, and every fast "solution" removes it *by assumption*: NODDI fixes `D_a`;
  AMICO convexifies onto a fixed dictionary; ML/SBI impose a training prior. Gyori
  2022 (`10.1002/mrm.29014`) shows ML precision *masks* the prior-induced bias;
  Hermans 2022 (arXiv:2110.06581) shows amortised SBI posteriors are routinely
  overconfident. Extra encoding (DDE, LTE+PTE) provably collapses the branches — but
  that is a hardware answer, not a numerical one.
- **Kernels.** **K2** (certified enumeration of all real roots of a parametric
  polynomial system) and **K3** (profile-likelihood + Rician-exact Fisher/CRLB
  identifiability trace).
- **Overlooked bet.** The degeneracy is a *root-count phenomenon of a polynomial
  system*, so it is addressable with polynomial-system tools. Polyhedral-homotopy
  continuation returns **all** isolated roots; α-theory/Krawczyk interval
  certification proves each encloses a true root **and that none was missed**. You
  then *enumerate* the branch set and disambiguate with one extra invariant, instead
  of guessing a branch or leaning on a prior. Nobody treats the per-voxel moment
  system as a certified all-root problem. Separately, profile likelihood (Raue 2009,
  `10.1093/bioinformatics/btp358`) and MBAM model reduction (Transtrum 2014,
  `10.1103/PhysRevLett.113.098701`) would *diagnose* structural vs practical
  non-identifiability per voxel — and are essentially unused in this field.
- **Builds on.** Round-1 filing 10 (matrix-argument ₁F₁) supplies the orientation
  normalisers; filing 05 (spherical deconvolution) the rotational invariants.

### P3 · Ill-posed non-negative spectral inversion

- **Anchor.** One numerical core, four literatures: myelin-water T2 spectra
  (Whittall & MacKay), PET spectral analysis (Cunningham & Jones), DSC/DCE/ASL
  residue-function deconvolution, and 2-D diffusion–relaxation correlation spectra.
  All are `min_{x≥0} ½‖Ax − b‖² + λ‖Lx‖²` with an exponential kernel.
- **Why it is stuck.** The kernel is exponentially ill-conditioned (numerical rank
  ≈ `log(1/ε)`), and there is now a **sharp minimax law**: resolving `p`
  closely-spaced exponentials at super-resolution factor `SRF` requires noise
  `ε ≲ SRF^{-(2p-1)}` (Batenkov 2021, `10.1093/imaiai/iaaa005`). The number of
  recoverable components is *logarithmic in SNR* — a theorem, not folklore. Meanwhile
  each voxel is an independent tiny constrained QP with data-dependent control flow
  (Lawson–Hanson pivots), which is SIMD-hostile at 10⁶–10⁷ voxels.
- **Kernels.** **K4** (batched, differentiable, certified non-negative Fredholm
  inversion), riding on the substrate kernel **KA**.
- **Overlooked bet.** The **Butler–Reeds–Dawson uniform-penalty dual** (1981,
  `10.1137/0718025`; Venkataramanan 2002, `10.1109/78.995059`) converts the
  `n`-dimensional non-negativity-constrained primal into a **smooth, low-dimensional,
  unconstrained dual**. It is precisely the object one wants to differentiate — smooth,
  implicit-function-theorem-friendly, GPU-friendly — and it is barely used outside
  NMR. Note also the skeptical flag: oSVD/cSVD residue deconvolution and
  unconstrained spectral-analysis NNLS are *widely used but not numerically sound*
  (non-physical oscillatory residues, noise-absorbing spurious peaks).
- **Builds on.** KA; round-1 filing 11 (structured decompositions) for the SVD
  compression of tensor-product kernels.

### P4 · Field-to-source inversion with an integer obstruction

- **Anchor.** Quantitative susceptibility mapping and background-field removal —
  image-domain throughout (a field map derived from phase images; no k-space, no
  sequence). Two coupled sub-problems: unwrap the phase, then deconvolve the dipole.
- **Why it is stuck.** *Unwrapping is integer cohomology.* The wrapped gradient is a
  1-cochain `g`; the residues are the curl 2-cochain `r = dg/2π`; since `d² = 0`, `r`
  is a cocycle and the obstruction lives in `H¹`. In 2-D the node–arc incidence matrix
  is totally unimodular, so the LP relaxation is integral and Costantini's min-cost
  flow (`10.1109/36.673674`) is *globally optimal*. **In 3-D residues organise into
  divergence-free loops and total unimodularity is lost** — the fast tools shipped in
  imaging (ROMEO, SEGUE, PRELUDE) are greedy MST/region-growing heuristics with no
  certificate. Minimum-L0 unwrapping is NP-hard (Chen & Zebker).
  *Dipole inversion is a Fourier multiplier vanishing on a cone.* The symbol
  `D(k) = 1/3 − k_z²/‖k‖²` attains zero on the 54.7° magic-angle double cone, so
  `0 ∈ spectrum` and the pseudo-inverse is unbounded; the near-nullspace is exactly
  the double-cone streaking artefact. Ill-posedness is established only
  *qualitatively* — there is **no published sharp modulus of continuity, and no
  provable lower bound on the regularisation `λ` needed** to suppress streaking to a
  given tolerance.
- **Kernels.** **K5** (integer-consistent coboundary correction of a wrapped
  1-cochain) and **K6** (regularised inversion of a cone-degenerate multiplier).
- **Overlooked bet.** Two, both real. (i) Project out the exact part of `g` with one
  Poisson solve, then **integer-correct only the small coexact + harmonic residual**
  near the residue loops — collapsing a huge integer program to a sparse local flow.
  This is the correct generalisation of Costantini to arbitrary complexes. (ii)
  Unwrapping mod 2π *is* **ℤ-synchronisation**; Singer's eigenvector/SDP relaxation
  (`10.1016/j.acha.2010.02.001`) supplies both a noise-robust global initialiser and,
  via duality, an *a-posteriori optimality certificate* that greedy path-followers
  cannot offer.
- **Builds on.** Shipped `hodge_decompose`/DEC and filing 09's 𝔽₂ boundary-matrix
  reduction — both are chain-complex algebra over a different ring. K6 is a special
  case of KA plus a stability constant.

### P5 · Population pooling: random-effects meta-analysis, lifespan charts, harmonisation

- **Anchor.** Three population-tier problems that turn out to share two kernels:
  coordinate-based meta-analysis (ALE/MKDA and their fragile nulls), lifespan
  normative modelling (`10.1038/s41586-022-04554-y`, N≈10⁵), and multi-site
  harmonisation (ComBat and its critics).
- **Why it is stuck.** The principled model of coordinate-based meta-analysis is a
  **log-Gaussian Cox process** (Kang 2011, `10.1198/jasa.2011.ap09735`; Samartsidis
  2019, `10.1111/rssc.12295`) — and it is underused *because inference is hard*: the
  reference 3-D implementation costs ~30 GPU-hours, and the version that reached
  general use (CBMR) dropped the random field entirely for a fixed spline basis.
  Normative charts need extreme conditional quantiles whose sampling variance
  `≈ τ(1−τ)/(n f(Q_τ)²)` diverges as the density vanishes; beyond sample support the
  centile is *model-dictated, not data-identified*, and there is no distribution-free
  certificate. ComBat corrects only the first two moments and treats the measurement
  as truth.
- **Kernels.** **K7** (matrix-free differentiable Gaussian–Markov random field on a
  simplicial complex, with an exponential-family Laplace outer loop) and **K8**
  (certified monotone conditional transport).
- **Overlooked bet — the unification.** ComBat's estimator is *exactly* the
  Gaussian-affine corner of **conditional optimal transport**: find `T_s(·|x)`
  pushing each site's conditional onto the `W₂` barycentre while fixing the
  `X`-marginal. And in 1-D, `T_s(·|x) = F̄⁻¹(·|x) ∘ F_s(·|x)` is an increasing
  rearrangement — *the same object* as the non-crossing conditional-quantile problem
  of normative modelling (Chernozhukov 2010, `10.3982/ECTA7880`). Harmonisation,
  distributional regression, and monotone transport are one kernel; it is tractable,
  certified (PAVA/rearrangement), and missing from every GPU stack. Be skeptical of
  the deep-generative harmonisers: unfalsifiable "signal preservation", no
  guarantees, not reusable primitives.
- **Builds on.** Filing 02 (eigensolvers), filing 13 (stochastic log-det), filing 04
  (OT/Sinkhorn barycentres). K7 is the sparse-precision complement to filing 13's
  dense matrix-free inversion.

### P6 · Whole-brain latent dynamics at cohort scale

- **Anchor.** Latent state-space models of fMRI: hidden Markov states, switching
  linear dynamical systems, latent SDEs, and the frequency-domain machinery of
  spectral DCM.
- **Why it is stuck.** Less than one might think — and the sweep corrected a prior of
  mine. Span-parallel Bayesian smoothing exists (Särkkä & García-Fernández 2021,
  `10.1109/TAC.2020.2976316`), *and* the square-root/QR-factored associative elements
  exist for linear and nonlinear parallel smoothers (Yaghoobi et al., SIAM SISC 2025,
  `10.1137/23M156121X`) — eigensolver-free by construction, which matters given the
  dead-cuSOLVER environment. Parallel HMM forward–backward over semirings also exists
  (arXiv:2102.05743). What does *not* exist is one kernel subsuming them.
- **Kernels.** **K9** (square-root monoid prefix scan).
- **Overlooked bet.** Composition of Markov kernels is associative — the Markov-category
  fact. Aji–McEliece's Generalized Distributive Law unifies the *sequential* semiring
  message passing; Särkkä gives a non-commutative monoid on Gaussian elements; Hassan
  gives semiring matrix-product scans; the information/precision form is a Möbius
  monoid composing by matrix multiplication. All four are "scan the monoid of
  composable conditional kernels," **but no one has written down that one monoid scan
  subsuming Gaussian + finite-state + general chain factors in square-root form.**
  That is an open, publishable gap — and it is a `genred` generalisation, since
  forward–backward *is* a semiring reduction. The real scale wall is memory
  `O(T·D²)` and an `O(D³)` combine, which argues for structured (diagonal-plus-low-rank)
  elements.
- **Builds on.** Filing 00 (`genred`) directly; filing 06 (integrators); filing 13
  (variational Laplace).

### P7 · Correspondence-free shape statistics and longitudinal diffeomorphometry

- **Anchor.** Comparing white-matter bundles and cortical shapes *without* point
  correspondence, and doing statistics on the resulting deformations: atlas building
  (Fréchet mean), geodesic regression, and parallel transport of subject-specific
  change to a template (the deformetrica-class workflow).
- **Why it is stuck.** The attachment metric is a kernel Gram double sum over
  `O(NM)` pairs, needed at every shooting iteration. **There is no mature JAX-native
  KeOps** — the upstream issue is still open — so JAX users either materialise the
  `N×M` Gram (OOM past `N ≈ 10⁴`) or hand-roll tiling.
- **Kernels.** **K10** (directional-kernel metrics on geometric measures) and **K11**
  (pole-ladder parallel transport, Fréchet mean, geodesic regression from `exp`/`log`
  alone), both riding on the substrate kernel **KB**.
- **Overlooked bet.** The three shape representations differ *only* in the directional
  kernel `k_t`: currents use `u·v` (signed — antiparallel sheets cancel and become
  metrically invisible); varifolds use `(u·v)²` (orientation-invariant, but
  curvature-blind); normal cycles lift to the unit normal bundle and see curvature.
  Curve bundles are exactly where currents cancel and varifolds are blind at
  crossings — so **normal cycles for bundles** are the under-exploited case. On
  transport: pole ladder is 3rd-order per rung and *exact on symmetric spaces*
  (spheres, affine-invariant SPD); Schild's is 2nd-order; the fanning scheme is only
  `O(1/n)`. Guigui & Pennec (arXiv:2007.07585) settle this; the field has not.
- **Builds on.** Filing 00 (`genred` *is* the missing tiled reduction engine — this is
  the single most direct round-1 → round-2 dependency in the batch), filing 08
  (EPDiff shooting), filing 12 (SPD `exp`/`log` as the closed-form transport oracle).

### P8 · Cross-subject correspondence with certificates rather than by fiat

- **Anchor.** Two faces of the same problem. *Geometric:* areal surface alignment
  (MSM, spherical demons) drives a spherical map by cortical features under a strain
  penalty. *Functional:* individualised ("precision") parcellation (MS-HBM,
  `10.1093/cercor/bhab101`) estimates each subject's own areal topography.
- **Why it is stuck.** Both fix by fiat what they claim to infer.
  *Geometric:* on a **closed** genus-0 surface there is no boundary to pin, so
  `det ∇f > 0` everywhere certifies only that the map is a `d`-fold covering — local
  injectivity never implies global bijectivity. Every spherical method delivers a
  diffeomorphism "in practice"; MSM's strain penalty is *soft* (foldovers discouraged,
  not prevented). The triple {certified global bijectivity, hard distortion bound,
  data attachment} is open.
  *Functional:* the likelihood is invariant to relabelling parcels (`S_k` gauge), and
  MS-HBM **fixes the gauge by anchoring every subject to a group template** —
  correspondence is *assumed*, not inferred, so idiosyncratic topography with no
  template counterpart is unrepresentable. Its contiguity-weight annealing is a
  homotopy hack around local minima, mean-field VB systematically underestimates
  posterior variance, and DP priors are inconsistent for the number of clusters
  (Miller & Harrison, arXiv:1301.2708).
- **Kernels.** **K12** (Beltrami-chart quasiconformal map with a separate
  global-bijectivity certificate) and **K13** (Burer–Monteiro co-assignment SDP with a
  dual certificate + explicit OT gauge).
- **Overlooked bet.** *Geometric:* the Beltrami coefficient `μ` lives in the **open
  unit disk**, which is a global smooth chart of the local-injectivity cone. Compose
  with any diffeomorphism `ℝ² ≅ D` and you have a **barrier-free, unconstrained**
  parameterisation; the map is recovered from `μ` by one sparse linear-Beltrami solve,
  which is differentiable. Load-bearing caveat: that solve is a *projection*, so the
  realised `μ` differs from the prescribed one — `‖μ_prescribed‖ < 1` does **not**
  guarantee `‖μ_realised‖ < 1`; you must re-measure and re-clamp. Global bijectivity
  stays a separate, exact-predicate check (per-face `det > 0`, simplex-overlap test,
  and on the sphere a degree-1 signed-area check).
  *Functional:* the Burer–Monteiro factorisation `X = YYᵀ` of the Peng–Wei k-means SDP
  is a single object that is simultaneously differentiable, GPU-friendly, carries the
  SDP's dual certificate, and **is** the permutation-invariant co-assignment matrix.
  The `S_k` gauge is then solved *explicitly* as an entropic-OT match between two
  solutions' factors, instead of being hidden in a template.
- **Builds on.** Filing 07 (embedding-constrained flow), filing 01 (exact predicates,
  self-intersection), filing 04 (Sinkhorn), filing 02 (eigensolvers).

---

## The kernels

Two are **substrate** (they underpin several of the rest and should exist first).
Thirteen are **frontier**. Each statement is deliberately a contract plus an oracle,
not a recipe. All are stated domain-free and are filing-ready seeds.

### Substrate

**KA · Certified differentiable nonsmooth composite solver.**
Given proper closed convex `f`, `g` and a linear map `D`, solve
`min_x f(x) + g(Dx)` to a **certified duality gap ≤ ε**, returning (i) a primal–dual
pair with an explicit Fenchel upper bound on suboptimality; (ii) the identified
active set together with a boolean **strict-complementarity certificate**; and (iii)
the solution-map derivative `∂x*/∂θ` from the support-restricted KKT linearisation — a
*true* Jacobian when the certificate holds, otherwise a flagged element of the
conservative Jacobian. Forward map: first-order warm start → gap-safe screening to fix
the support → semismooth Newton on the reduced smooth problem.
*Why it matters.* This is the sharpest single finding of the sweep:
**"autodiff returns something" ≠ "the derivative is correct."** `cvxpylayers`/`jaxopt`
return a heuristic element of a set at degenerate solutions and **no library flags
it**. Semismooth-Newton ALM (Li–Sun–Toh, arXiv:1607.05428) reaches machine-accuracy
solutions — a prerequisite for both meaningful certificates and stable derivatives,
since first-order methods stall where the support still flickers — yet it is absent
from every batched-GPU autodiff stack for engineering reasons, not mathematical ones.
*Oracle.* KKT residual and gap-bounds-suboptimality vs high-accuracy SCS/OSQP; exact
prox subroutines vs Condat's taut-string (1-D TV), PAVA (isotonic), analytic
block-soft-threshold, SVT; derivative vs finite differences **only where the support
is verified stable**; a deliberately degenerate instance must fire the flag *and* show
naive autodiff diverging from FD.
*Serves:* K4, K6, K13. *Score:* ①3 ②5 ③5.

**KB · Memory-linear tiled kernel-Gram reduction with a hand-written VJP.**
Given `{(x_i, u_i, a_i)}` and `{(y_j, v_j, b_j)}` with `x ∈ ℝ^d`, `u, v ∈ S^{d-1}`, a
positive-definite spatial kernel `k_e` and a selectable directional kernel `k_t`,
compute the Gram double sum and its gradients **without materialising the `N×M`
matrix**, via an online tiled reduction with a `custom_vjp`.
*Why it matters.* This is the JAX-KeOps gap, and `genred` (round-1 filing 00) is
already this engine. Everything in P7 — currents, varifolds, normal cycles,
MMD-vs-Sinkhorn attachment, Nyström/random-feature compression — is a thin layer on
top. Largely an *integration* task, not a research one.
*Serves:* K10, K11, and any RKHS attachment. *Score:* ①2 ②3 ③5.

### Frontier

**K1 · Anisotropic/Finsler Hamilton–Jacobi front with an exact discrete adjoint.**
Given a field of dual metrics `F*(x,·)` on a `d`-dimensional grid (SPD Riemannian, or
Randers `M(x), ω(x)`), solve the causal monotone upwind discretisation of
`F*(x, ∇u) = 1`, `u|_Γ = 0` on a Selling/Voronoi-reduced adaptive stencil, returning
the arrival-time field, the geodesic flow, and — via one reverse-order linear-transport
back-substitution over the accepted ordering — the exact discrete adjoint
`∂J(u)/∂(M, ω)`. Forward `O(N log N)`, backward `O(N)`.
*The open quadruple:* GPU-parallel **and** differentiable **and** certified-monotone
under strong anisotropy **and** Finsler. Every *pair* exists; the quadruple does not.
Selling's guarantee stops at `d = 3`, so the `R³ × S²` lift is heuristic — that is the
prize.
*Oracle.* Constant metric → exact anisotropic norm `‖x − x₀‖_M`; Poincaré half-plane
(stresses `κ → ∞`); sphere → great circles; Zermelo navigation with constant drift →
analytic asymmetric fronts; Reeds–Shepp → analytic sub-Riemannian geodesics; adjoint
vs finite differences to machine precision on all of the above.
*Score:* ①4 ②5 ③4.

**K2 · Certified enumeration of all real roots of a parametric polynomial system.**
Given `F(x; θ) = 0`, `x ∈ ℝⁿ`, return the **complete** set of isolated real roots in a
box with a **completeness certificate** up to the mixed-volume (BKK) bound:
polyhedral-homotopy continuation from a generic start system, path-tracked in `ℂ`, then
a-posteriori interval certification of each endpoint. Output: the real root set, a
per-root validity certificate, and a "no root missed" flag. Roots differentiable in `θ`
by the implicit function theorem where the Jacobian is nonsingular.
*Honest caveat.* Exactness holds only at the truncation order. Under noise the
statistic lies *off* the image variety, so the exact system has no real solution and
the right object is *all critical points* of a weighted **minimum-distance** criterion
against the polynomial moment map — still yielding the certified global optimum and all
competing minima, which multi-start NLLS cannot guarantee. **Correction (2026-07-10,
raised by the assistant):** the count invariant there is the **Euclidean-distance
degree**, not the maximum-likelihood degree, and a **Rice/noncentral-χ likelihood is
inadmissible** — its stationarity is `ν = m·I₁(mν/σ²)/I₀(mν/σ²)`, transcendental in the
Bessel ratio, with no polynomial critical system. Magnitude data enter through the
*statistic* (noise-floor-corrected second moment, `E[m²] = ν² + dσ²`), never the
objective; the exact likelihood geometry is **K3/filing 18's** concern, and the two
compose — 17 certifies the complete candidate set, 18 adjudicates it.
*Oracle.* Root count = mixed volume / Bézout (exact system) and ED degree (critical
system); agreement with Gröbner bases on random dense systems; Krawczyk/Smale-α
certification; `‖F(x*)‖ ≤ ε` at fp64; a published ML degree of a *discrete rational*
model as an independent count oracle only.
*Score:* ①5 ②5 ③2. **The hardest item in the batch.**

**K3 · Profile-likelihood + Fisher/CRLB identifiability trace for batched NLLS.**
Given residuals `r(θ)` and a noise model (Gaussian or Rician), compute (a) the profile
objective `P_i(c) = min_{θ_{j≠i}} L(θ)` s.t. `θ_i = c`, returning the threshold-crossing
confidence interval and a **structural / practical / identifiable** flag from the
profile shape (flat / one-sided / bounded); and (b) the Fisher information
`JᵀΣ⁻¹J`, its eigenspectrum, condition number, and `diag(FIM⁻¹)`.
*Oracle.* Linear-Gaussian closed form (profile is an exact parabola, CRLB = OLS
covariance); profile CIs vs MCMC credible intervals on toy nonlinear models; the
smallest-eigenvalue FIM eigenvector aligns with the geodesic/MBAM sloppy direction;
CRLB ≤ empirical variance of repeated fits; rank-deficient FIM ⟺ flat profile.
*Score:* ①1 ②2 ③3. Cheap, exact, and conspicuously absent.

**K4 · Batched differentiable non-negative Fredholm inversion.**
Given a fixed exponential-kernel matrix `A_{ij} = e^{-t_i s_j}` (or an ND
tensor-product kernel), a smoothing operator `L`, and a *batch* of right-hand sides,
solve `min_{x≥0} ½‖Ax − b_k‖² + λ‖Lx‖²` with `λ` from a discrepancy rule, returning
each minimiser **and** `∂x*/∂(b, A, λ)` by implicit differentiation of the BRD-dual (or
KKT/active-set) system, batched with fixed-iteration branch-free inner solves.
*Oracle.* Known-spectrum recovery; the `SRF^{2p-1}` minimax breakdown law; parity with
CONTIN/BRD and, in 2-D, with SVD-compressed BRD; implicit-diff Jacobian vs FD away
from kinks; estimator variance vs the Cramér–Rao bound.
*Score:* ①2 ②3 ③4.

**K5 · Integer-consistent coboundary correction of a wrapped 1-cochain.**
Given an oriented cell complex with edge weights `w` and a 1-cochain `g` known modulo
`2π`, return the integer 1-cochain `m ∈ ℤ^E` minimising `Σ_e w_e |g_e − 2π m_e|`
subject to `d(g − 2π m) = 0` — equivalently a minimum-cost integer flow routing the
residue 2-cochain `r = dg/2π`. Suggested structure: remove the exact part by one
Poisson solve, integer-correct only the coexact + harmonic residual, optionally
initialise by ℤ-synchronisation spectral relaxation and report the duality gap.
*Oracle.* In 2-D the incidence matrix is totally unimodular, so the LP optimum is
integral and certifies the IP; exact recovery under the Itoh band-limit condition;
parity with Costantini/SNAPHU on planted-residue lattices; invariance of the cost under
adding any exact cochain.
*Do not over-promise:* 3-D loses total unimodularity; "globally optimal 3-D unwrapping"
is not on offer, a certificate-when-it-closes is.
*Score:* ①4 ②4 ③2.

**K6 · Regularised inversion of a Fourier multiplier degenerate on a cone.**
Given a real symmetric multiplier `m(k)` vanishing on a cone and data `ŷ`, solve
`min_χ ½‖m·χ̂ − ŷ‖² + λR(χ)`, `R ∈ {‖∇χ‖²₂, TV, TGV}`, via splitting whose `χ`-update
is the diagonal Fourier solve `χ̂ = (m ŷ + ρ(·))/(m² + ρ)`. Ship the closed-form `L2`
map and — the actual research content — a **stability estimate**: an explicit modulus
of continuity and a provable lower bound on `λ` to suppress the near-nullspace to a
given tolerance. No such constant is published.
*Oracle.* Analytic single-dipole/uniform-ball source; a multi-orientation ("COSMOS")
oracle that fills the cone and is well-posed; residual energy must localise on the
cone; `L2` error scales `O(λ^{-1/2})`; the symbol preconditioner `(m² + λ)⁻¹` gives
grid-independent CG iteration counts.
*Score:* ①3 ②3 ③2.

**K7 · Matrix-free differentiable Gaussian–Markov random field on a simplicial complex.**
Given sparse SPD `Q = ∏(κ²M + K)` assembled from the mass and stiffness matrices of a
piecewise-linear basis (optionally a fractional power), provide matrix-free,
differentiable operators for: (i) sampling `x ~ 𝒩(0, Q⁻¹)` by applying a
Chebyshev/rational approximation of `Q^{-1/2}` to white noise; (ii) `log det Q` and
`∂_θ log det Q` by stochastic Lanczos quadrature with a preconditioner and a
**randomized-truncation debiasing of the gradient**; (iii) `diag(Q⁻¹)` by exact
Takahashi selected inversion or a probing + variance-reduced Hutchinson estimator with
an error certificate. Plus the exponential-family Laplace outer loop: Newton to the
mode of `nᵀg − Aᵀe^g − ½(g−μ)ᵀQ(g−μ)`, each step a sparse solve of `(Q + diag(w))δ = r`.
No dense factorisation, no vendor sparse-direct solver.
*Why it matters.* `diag(Q⁻¹)` is the genuinely hard object — it needs *off-diagonal*
inverse entries even for diagonal output, and selected inversion is sequential over the
elimination tree with no GPU/autodiff story. And the literature reports log-det *values*
with error bars but not the **gradient bias that actually moves the fitted
hyperparameters**.
*Oracle.* Flat torus (Fourier eigenpairs → closed-form covariance, marginal variance,
log-det); sphere (`ℓ(ℓ+1)` eigenvalues → exact angular power spectrum, and the
definitive check that the SPDE field is **not** the geodesic Matérn — which is not even
PD for `ν > ½`); dense Cholesky reference at small `V`.
*Score:* ①3 ②5 ③5.

**K8 · Certified monotone conditional transport.**
(i) Given fitted conditional quantiles `q̂(τ_j, x)`, return the increasing
rearrangement — equivalently the isotonic (PAVA, `O(m)`) projection onto the monotone
cone — guaranteeing no crossing; and/or fit a monotone warp `w_φ` (sinh-arcsinh or
I-spline, `w' ≥ 0`) with analytic inverse and Jacobian. (ii) Fibered conditional-OT
map: partition coordinates into aligned `Y` and preserved `X`; in 1-D
`T_s(·|x) = F̄⁻¹(·|x) ∘ F_s(·|x)` with `F̄⁻¹` the barycentric quantile function
(closed form); in general, a log-domain Sinkhorn plan with a fixed `X`-marginal.
*Oracle.* Rearrangement idempotent and `L²`-non-expansive; PAVA parity; `w ∘ w⁻¹ = id`
to machine ε; Gaussian case vs the closed-form Bures barycentre; `T_s = id` when all
site conditionals coincide; invariance of the `X`-marginal (the covariate-preservation
check).
*Score:* ①1 ②2 ③4. The cheapest high-impact item here.

**K9 · Square-root monoid prefix scan.**
Given a chain of `T` local factors in an associative monoid — instantiated either as
affine-Gaussian conditionals under Gaussian-message composition, or as nonnegative
matrices over a semiring `(⊕, ⊗)` — compute all prefix and suffix products by a
work-`O(T)`, span-`O(log T)` parallel scan, returning each marginal. Every element and
combine is held in factored form (Cholesky/QR triangular, or information-square-root),
so each operation is a QR, Cholesky or triangular solve — never an explicit inverse or
eigendecomposition. Differentiable, batched over sequences, with an optional
structured-rank (diagonal-plus-low-rank) element to break the `O(T·D²)` memory and
`O(D³)` combine walls.
*Oracle.* Exactness vs sequential RTS / forward–backward / Viterbi to round-off (fp32
characterised, not assumed); smoother marginals vs a dense Cholesky solve of the
`(TD)×(TD)` block-tridiagonal precision; sum-product vs brute-force marginalisation;
max-plus vs exhaustive best-path; a small-`Q`/small-`R`/large-`T` stress sweep where the
square-root form must retain PSD and the covariance form must fail.
*Score:* ①3 ②4 ③4.

**K10 · Directional-kernel metrics on geometric measures.**
On top of **KB**: with `k_t(u,v)` selectable as `u·v` (currents), `(u·v)²` (varifolds),
a Gaussian-on-sphere (oriented varifolds), or a lift to the unit normal bundle (normal
cycles), compute the metric and its gradients w.r.t. positions, directions and weights.
*Oracle.* Dense `O(NM)` reference at small `N`; `D ≥ 0` and `D = 0 ⟺` measures equal for
a `c₀`-universal kernel; closed form for two Diracs; the **symmetry contract** — `u·v`
flips sign under `u → −u`, `(u·v)²` does not (this single test guards the
currents-vs-varifold distinction); the analytic cancellation identity
`2|ξ|²(1 − e^{−|x−y|²/σ²})`.
*Caveat to carry:* a kernel attachment is a low-pass filter at scale `σ`, not a true
distance below `σ`. "Correspondence-free" is not "assumption-free."
*Score:* ①2 ②3 ③3.

**K11 · Pole-ladder transport, Fréchet mean, geodesic regression from `exp`/`log`.**
Given only `exp` and `log` on a manifold, transport `w ∈ T_pM` along a geodesic by `n`
iterated pole-ladder rungs (3rd-order per rung, `O(1/n²)` global, **exact on symmetric
spaces**); build the Fréchet mean by Riemannian gradient descent; fit geodesic
regression by least squares on the tangent bundle.
*Oracle.* Sphere `S^n` — closed-form transport; verify the `O(1/n²)` log-log slope.
SPD with the affine-invariant metric (symmetric, `∇R = 0`) — pole ladder exact up to
geodesic-solver error. Euclidean — must be the identity. Cross-check that Schild's
ladder agrees to 2nd order and the fanning scheme degrades to `O(1/n)`, pinning the
certified order gap.
*Caveat:* ladder rates assume accurate geodesics; in an infinite-dimensional
deformation group `exp`/`log` are themselves approximate, so certified order claims
must be scoped to finite-dimensional oracle manifolds.
*Score:* ①2 ②3 ③3.

**K12 · Beltrami-chart quasiconformal map with a separate bijectivity certificate.**
Given a source triangulation and a target domain (planar region or `S²`), and a
per-face complex field `μ` constrained to the open unit disk by a **smooth
unconstrained reparameterisation**, reconstruct the unique piecewise-linear
quasiconformal map with Beltrami coefficient `μ` by one sparse generalised-Laplacian
(linear-Beltrami) solve, differentiably in `μ`. Orientation-preserving on every face by
construction, giving a barrier-free chart of the local-injectivity cone over which any
smooth data/distortion energy is minimised. Global bijectivity is certified as a
**separate, explicit step**: exact-predicate per-face `det > 0`, a BVH-accelerated exact
simplex-overlap test, and on the sphere a degree-1 (signed-area-sum) check.
*Load-bearing caveat:* the linear-Beltrami solve is a *projection*, so realised `μ` ≠
prescribed `μ`; re-measure and re-clamp. The honest guarantee is *certified local
injectivity + measured distortion bound + a differentiable map*, with global
bijectivity as a separate certified check — precisely the boundary the whole literature
runs into on closed surfaces.
*Oracle.* Prescribe analytic `μ` from a known affine/Möbius map and compare to closed
form; per-face SVD to check the dilatation `K = (1+|μ|)/(1−|μ|) = σ_max/σ_min`;
exact-predicate flipped-count = 0; brute-force segment intersection on small meshes;
FD vs autodiff vs the Beltrami-holomorphic-flow analytic gradient.
*Score:* ①4 ②4 ③3.

**K13 · Burer–Monteiro co-assignment with a dual certificate and an explicit gauge.**
Given a symmetric affinity `W` on a graph with adjacency constraint `A` and target
part-count `k`, compute the co-assignment matrix solving the Peng–Wei relaxation
`max ⟨W, X⟩` over `{X ⪰ 0, X1 = 1, X ≥ 0, tr X = k}`, parameterised low-rank as
`X = YYᵀ`, `r ≪ V` — a differentiable map `(W, A) ↦ X` whose stationary points are
certifiably near-global once `r ≳ √(2m)` (Boumal–Voroninski–Bandeira). `X` *is* the
permutation-invariant partition summary; a hard partition is a rounding of it;
cross-instance correspondence is a **separate** entropic-OT match between two
solutions' factors. Report the dual-certificate gap.
*Oracle.* Planted stochastic block model at the exact-recovery threshold
`(α+β)/2 − √(αβ) > 1` — must recover the phase transition, succeeding above and failing
below; stochastic-ball mixtures where Peng–Wei is provably tight (the dual certificate
must fire and `X` be integral); brute-force optima on ≤20 nodes; algebraic invariants
(`X ⪰ 0`, row-stochastic, `tr X = k`, exact relabelling invariance).
*Caveat:* a certificate certifies optimality *of the objective*, not correctness of the
model. Cortex has smooth gradients and no planted blocks; honesty comes from reporting
the certificate gap *and* the planted-model fit, never the certificate alone.
*Score:* ①3 ②4 ③3.

---

## Score assessment

Axes as in round 1. **① low self-confidence** — higher = *less* confident I could
produce a robust, certified implementation myself even given time and resources (the
routing signal). **② work** — relative effort. **③ impact** — ecosystem leverage
(performance, robustness, what it unblocks, community value).

| # | Kernel | Serves | ① low self-confidence | ② work | ③ impact |
|---|---|---|:---:|:---:|:---:|
| KA | Certified differentiable nonsmooth solver | P3, P4, P8 | 3 | **5** | **5** |
| ~~KB~~ | Tiled kernel-Gram reduction — **dissolved into K10** (see audit) | P7 | — | — | — |
| K1 | Anisotropic/Finsler HJ + adjoint | P1 | 4 | **5** | 4 |
| K2 | Certified all-real-root polynomial system | P2 | **5** | **5** | 2 |
| K3 | Profile-likelihood + Rician CRLB | P2 | 1 | 2 | 3 |
| K4 | Batched differentiable non-negative Fredholm | P3 | 2 | 3 | 4 |
| K5 | Integer coboundary correction (unwrapping) | P4 | 4 | 4 | 2 |
| K6 | Cone-degenerate multiplier inversion | P4 | 3 | 3 | 2 |
| K7 | Matrix-free differentiable GMRF on a complex | P5 | 3 | **5** | **5** |
| K8 | Certified monotone conditional transport | P5 | 1 | 2 | 4 |
| K9 | Square-root monoid prefix scan | P6 | 3 | 4 | 4 |
| K10 | Directional-kernel geometric-measure metrics | P7 | 2 | 3 | 3 |
| K11 | Pole-ladder transport / Fréchet / geodesic reg. | P7 | 2 | 3 | 3 |
| K12 | Beltrami chart + bijectivity certificate | P8 | 4 | 4 | 3 |
| K13 | Burer–Monteiro co-assignment + gauge | P8 | 3 | 4 | 3 |

### Reasoning

- **① self-confidence.** The one I would most likely *fail* to make robust is **K2**:
  polyhedral homotopy path-tracking with α-theory certification and a provable
  no-root-missed flag is a numerical-algebraic-geometry discipline, and I would very
  likely ship a fast, uncertified multi-start that silently misses roots — which is
  exactly the failure the kernel exists to prevent. Real risk of a subtly-wrong result
  (**4**): **K1** (the certified-monotone-under-strong-anisotropy quadruple, where
  Selling's guarantee expires at `d = 3`), **K5** (3-D residue loops, loss of total
  unimodularity), **K12** (the prescribed-vs-realised `μ` projection gap, plus the
  degree-1 covering trap that the whole literature walks into). Probable-but-risky
  (**3**): **KA** (getting the conservative-Jacobian flag *right* rather than
  plausible), **K7** (certified `diag(Q⁻¹)` and debiased log-det gradients), **K9**,
  **K6**, **K13**. The rest I am confident about; the score reflects the
  *certified-and-differentiable* bar, not "could I write a version".
- **③ impact.** **KA** and **K7** fan out furthest: KA is the missing nonsmooth twin of
  the shipped `implicit_minimize` seam and underpins three problems; K7 is the sparse
  complement to filing 13, and would make random-effects meta-analysis, spatial
  Bayesian GLMs, and GP priors on surfaces affordable in one stroke. **KB** is high
  impact for low novelty — `genred` already is the engine, so this is integration that
  unlocks an entire shape-statistics lane. **K8** is the best value in the batch:
  trivial mathematics, certified, and it collapses harmonisation and normative charts
  into one primitive. **K2** has the narrowest fan-out and the highest difficulty.

### Recommendations — routing

- **Build in-house first (substrate, and they gate the rest).** **KB** (mostly wiring
  `genred` to a directional-kernel front-end) and **KA** (the nonsmooth implicit-diff
  seam). KA in particular is a *correctness* fix for a class of silent errors, not a
  feature. Then the cheap certified wins: **K8**, **K3**, **K4**.
- **Route to the assistant (high ① × meaningful ③).** In order: **K1** (the flagship —
  hardest well-scoped item with real fan-out), **K7** (heaviest but highest impact),
  **K12**, **K5**, then **K2** if credit genuinely is not binding. **K2** is the
  purest expression of the restriction earning its keep: strictly numerical, entirely
  domain-free, and a discipline neither of us practises.
- **Sequencing note.** `KA → K4/K6`, `KB → K10/K11`, `K7 → P5 entirely`. Round-1
  filings 00 and 02 gate a surprising amount of round 2 (00 → KB → K10/K11 → K9;
  02 → K7, K13), which strengthens the round-1 recommendation to build those two
  internally and early.

## Overlap and synergy audit vs round 1

Checked against the *text* of the 14 round-1 `PROBLEM.md` files, in particular their
declared **non-goals** — which turn out to be the most informative surface, because a
declared non-goal is a seam someone deliberately left open.

### One kernel dissolved

**KB is not a filing.** The memory-linear tiled kernel-Gram reduction *is* an
`einred`/`genred` formula plus its backward atoms — filing 00 already builds that
engine, so KB would have re-filed it. The requirement (never materialise the `N×M`
Gram; ship a hand-written VJP) is instead folded into **K10** as a stated invariant,
where it belongs. Round 2 therefore mints **14** filings, not 15.

### Round-2 kernels that would *collide* if filed naively

| Kernel | Collides with | Resolution |
|---|---|---|
| K4 non-negative Fredholm | 05 spherical deconvolution | 05's forward operator is the diagonal zonal case; K4 is stated for a **general** linear operator. 05's `apply` becomes a K4 instance. |
| K6 cone-degenerate inversion | KA | K6's splitting *is* a KA instance. K6's filing is scoped to the **stability estimate** and the symbol preconditioner, and consumes KA. |
| K9 square-root monoid scan | 00 genred | genred is a *reduction*; K9 is a *prefix scan* over a non-commutative monoid. K9 consumes genred's algebra abstraction rather than restating it. |
| K11 manifold transport | 12 SPD statistics | 12 ships Fréchet mean + parallel transport **for SPD in closed form**. K11 is the generic `exp`/`log`-only algorithm; **SPD is K11's oracle, not its subject**. |
| K5 integer coboundary | 09 persistence | Same chain complex, different coefficient ring (`ℤ` vs `𝔽₂`) and different objective (min-cost flow vs pairing). Shared sparse boundary-operator representation. |
| K7 matrix-free GMRF | 13 variational Laplace | 13 estimates log-dets for **dense matrix-free** operators; K7 for **sparse `Q`** plus selected inversion. Must adopt 13's `LogDetEstimator` protocol, not fork it. |
| K1 anisotropic HJ | 03 heat-method geodesics | 03 is a mesh + elliptic-solve approximation; K1 is a grid + causal-upwind exact solve. Disjoint methods; 03's exact polyhedral oracle validates K1's isotropic limit. |

### Conversely — round-1 filings that gain from round 2

This is the stronger direction, and it lands squarely on filings' own declared
non-goals:

- **05 ← K4 + KA.** 05 explicitly declares *"No hard-constraint QP guarantee… the
  solver enforces non-negativity as a certified soft floor… interior-point/active-set
  QP with exact feasibility is out of scope."* K4 is precisely that missing exact
  constraint, and KA supplies its certificate and its correct derivative.
- **04 ← K8 + KA.** 04 declares *"No barycentre, multi-marginal, or continuous
  transport in this filing."* K8 needs and supplies the Wasserstein barycentre (in
  closed form in the 1-D fibered case). KA supplies a rigorous stopping certificate
  for the Sinkhorn fixed point in place of an iteration budget.
- **03 ← K1.** 03 declares *"Not an exact geodesic solver"* and *"No cut-locus /
  geodesic-path extraction. Only the distance field is returned, not the minimising
  paths."* K1 returns arrival time **and** the geodesic flow, exactly, anisotropically.
- **12 ← K11.** 12 declares *"No manifold-valued regression or classification — those
  are built by composing these primitives and are out of scope."* K11 is that
  composition, done once and generically.
- **08 ← K10.** 08 declares *"`similarity` is a supplied callable."* K10 is the
  supplied callable for curves and surfaces — a correspondence-free data attachment,
  which is the case 08 cannot otherwise serve.
- **11 ← KA.** Principal component pursuit is a nonsmooth composite (nuclear + ℓ¹)
  solved by ADMM with singular-value thresholding. KA gives it a duality-gap
  certificate and a *correct* Jacobian, rather than whatever autodiff returns at a
  rank-boundary.
- **13 ← K3 + K7.** K3's profile likelihood is the identifiability audit of exactly the
  fits 13 produces — it checks the Gaussian approximation that 13's `LaplaceResult.cov`
  assumes. K7 is 13's sparse-precision complement.
- **06 ← K9.** 06's integrators produce the trajectories; K9 infers the latent chain
  behind them in `O(log T)` span.
- **02 ← consumed by K7, K13.** Two more consumers for the eigensolver, strengthening
  the round-1 case to build it first.
- **01 ← reused by K12, K10.** 01's exact orientation predicates and broadphase are the
  certification substrate K12's global-bijectivity check needs; reuse hardens both.
- **00 ← two new consumers.** K10 adds a directional kernel-Gram atom; K9 adds a scan
  seam over the same algebra abstraction. `genred` gains its second and third external
  consumer, which is the best available evidence its abstraction is the right one.

**Net.** Every round-2 kernel either fills a round-1 non-goal, supplies a round-1
callable seam, or consumes a round-1 primitive. No kernel is orphaned, and the two
round-1 filings flagged as "build in-house first" (00, 02) gate the most round-2 work —
which is now a stronger recommendation than it was when made.

## Honest caveats carried forward

The sweeps were asked to be adversarial, and several findings are warnings rather than
opportunities:

- **Geodesic tractography is popular but numerically weak.** The shortcut pathology is
  intrinsic. Do not adopt it without curvature lifting.
- **FIM/FSM "GPU eikonal" solvers carry no convergence certificate** under strong
  anisotropy. Treat their outputs as uncertified.
- **oSVD/cSVD perfusion deconvolution and unconstrained spectral-analysis NNLS are
  widely used and not numerically sound** (non-physical oscillatory residues).
- **`det ∇f > 0` never implies global bijectivity on a closed surface.** It implies a
  `d`-fold covering.
- **A partition certificate certifies the objective, not the model.** SBM/stochastic-ball
  assumptions are the weak link on a cortex with smooth gradients.
- **The SPDE field on a manifold is Matérn only asymptotically.** The geodesic Matérn it
  evokes is not positive definite for `ν > ½` on `S²`.
- **Autodiff through a nonsmooth solver returns something, and that something is not
  the derivative at degeneracy.** No current library flags this.
- **ML/SBI microstructure fits do not resolve the branch degeneracy; they relocate it
  into the prior**, and high precision masks the resulting bias.

## Provenance

Derived from eleven parallel adversarial literature sweeps (2026-07-09) covering:
anisotropic eikonal & tractography-as-inverse-problem; inverse-Laplace/multi-exponential;
QSM dipole inversion & phase unwrapping; SPDE Gaussian fields on manifolds; parallel
Bayesian smoothing & latent dynamics; currents/varifolds & diffeomorphic statistics;
bounded-distortion bijective mapping; nonsmooth convex optimisation & implicit
differentiation; individualised parcellation with certificates; meta-analysis /
normative modelling / harmonisation; microstructure identifiability.

**Filings minted.** The fourteen kernels were filed as `nitrix-moonshot/15-…` through
`28-…` on branch `round-2-filings` (commit `49b5f93`), each a domain-free `PROBLEM.md`
+ `validation.md`, plus `protocols.py` in the eleven cases that target an existing seam
(17, 20 and 21 open new seams and carry none). Gates: 39 files, denylist-clean, uniform
section structure, 11/11 protocols parse. DOI strings are exempt from the grep gate and
stripped before it runs — a DOI carries no domain information, and dropping a canonical
citation to satisfy a substring match would be the worse defect.

Round-2 filings are **deliberately less specified** than round 1: each fixes a contract
and an oracle and leaves the recipe open, because for several of these problems
prescribing the method would be prescribing the wrong one.

The anchors in this ledger stay here, and never appear in a filing.
