# nitrix — Specification update (v0.2 → v0.3)

> **Status.** Scope-record addendum to ``SPEC_UPDATE_v0.2.md``.
> Apply on top of v0.2.  All v0.2 changes stand; this patch
> records the consumer-driven scope drift accumulated since v0.2
> and proposes a brainstorm of further candidate primitives for
> consideration as the substrate stabilises.

---

## §0. Meta-observation: consumer-driven plan drift

Implementation has run well ahead of the schedule declared in
``IMPLEMENTATION_PLAN.md``.  Several surfaces beyond the Phase 1–4
declared scope landed under pressure from concrete downstream
consumers (the JOSA feedback batch, sealed; the ilex feedback
batch, drained as of FA1–FA3).  Cataloguing the drift here so
the plan / SPEC pair remains a faithful description of what is
shipped, not what was originally scoped.

**Important invariant: separation of concerns has held.**  Every
shipped deviation respects the substrate's organising principles:

- The semiring / ELL substrate remains the locus of "structured
  reduction" (no parallel scatter API; no message-passing class).
- Geometric ops compose with the substrate rather than reimplementing
  it (``mesh_cotangent_laplacian`` returns ELL; ``icosphere_*``
  returns ELL; ``mesh_pool_max`` is sugar over ``semiring_ell_matmul``
  under ``TROPICAL_MAX_PLUS``).
- Morphology stays algebraically grounded (``max_pool_with_indices_nd``
  is the TROPICAL_MAX_PLUS-with-argmax variant of the same gather +
  reduction pattern as ``dilate`` / ``erode``).
- Stats / LME exposes ``NamedTuple`` of arrays (no PyTree modules,
  no learnable-state objects).

**Expect further drift.**  ``ilex`` continues to vendor models
(surface CNNs, encoder-decoder UNets, attention-based segmenters,
optimal-transport registrators, …); each new port will likely
surface one or two primitives ``nitrix`` should host.  The
deviation protocol in ``IMPLEMENTATION_PLAN.md §2.4`` should be
expected to fire repeatedly, not as a one-off.  The acceptance
test for future drift is the **separation-of-concerns invariant
above**, not "did we declare this in advance."

---

## §10.A Deviation log — shipped scope additions

This subsection extends ``IMPLEMENTATION_PLAN.md §10`` (deviation
log).  Each entry below should be merged into that log at the
next plan refresh.

### §10.A.1 `nitrix.semiring.semiring_ell_edge_aggregate`

**Plan reference.**  Phase 2.A did not name this primitive.

**What shipped.**  An edge-functional extension of
``semiring_ell_matmul``: a user-supplied
``edge_fn(h_i, h_j, w, ij) -> e`` runs at every (vertex, neighbour)
pair, and the per-row results reduce under a semiring (REAL,
TROPICAL_MAX_PLUS, TROPICAL_MIN_PLUS).  LOG / EUCLIDEAN /
BOOLEAN raise ``NotImplementedError``.

**Why it respects separation of concerns.**  The primitive is
gather-then-vmap-then-reduce — three operations already in the
substrate.  No new "message-passing" class; the user's closure
**is** the message function.  Differentiability through
``edge_fn`` parameters flows naturally via JAX (no custom_vjp
needed).

**Driving consumer.**  ilex/Topofit (FA2).  Covers GCN, GAT,
EdgeConv/DGCNN, MoNet, ChebNet with one
primitive + a user callable.

**Sealed-spec status.**  Add to SPEC §3.1 "Public surface
(illustrative signatures)" as the third illustrative signature
after ``semiring_matmul`` and ``semiring_ell_matmul``.

### §10.A.2 `nitrix.sparse.IcosphereHierarchy` + cross-level constructors

**Plan reference.**  Phase 2.A.9 named ``mesh_k_ring_adjacency``
and ``mesh_cotangent_laplacian``; did not name a hierarchy
container or cross-level helpers.

**What shipped.**

- ``IcosphereHierarchy`` — frozen dataclass holding
  ``meshes: tuple[Mesh, ...]`` and per-level ``parents`` tables.
- ``icosphere_hierarchy(max_level)`` — construction.
- ``icosphere_cross_level_adjacency(hier, L, L+1)`` — coarse-to-fine
  ELL for pooling.
- ``icosphere_bary_upsampler(hier, L, L+1)`` — fine-from-coarse
  ELL with barycentric weights for continuous upsampling.

Both cross-level helpers require consecutive levels and reject
non-consecutive pairs.

**Why it respects separation of concerns.**  All cross-level
operators are ELLs; downstream pool / upsample uses the standard
``semiring_ell_matmul`` path.  The hierarchy is host-side NumPy
bookkeeping; the resulting ELLs are JAX-native.

**Driving consumer.**  ilex/Topofit (FA2 — mesh-hierarchy
section).  Generalises to any surface-based encoder-decoder.

**Sealed-spec status.**  Add to SPEC §3.2 ``nitrix.sparse``
illustrative signatures.

### §10.A.3 `nitrix.morphology.pooling`

**Plan reference.**  Phase 4 (morphology) named ``dilate`` /
``erode`` / ``open`` / ``close`` / ``distance_transform`` /
``median_filter``; did not name pooling primitives with index
tracking.

**What shipped.**

- ``max_pool_with_indices_nd(x, pool_size, spatial_rank)`` →
  ``(pooled, indices)`` where ``indices`` is the global flat
  C-order argmax position.  N-D, channel-first.
- ``max_unpool_nd(x, indices, output_shape, spatial_rank)`` —
  scatter-based inverse.

**Why it respects separation of concerns.**  Max-pool is the
strided / argmax-tracking variant of ``dilate`` (both are
TROPICAL_MAX_PLUS window reductions).  Co-locating it with
``dilate`` / ``erode`` keeps the algebraic origin visible.
Cross-framework parity contract is documented as
**argmax-of-output agreement, not raw-logit allclose**.

**Driving consumer.**  ilex/PGlandsSeg (FA3).  Generalises to
V-Net, SegNet, attention-pooling networks, encoder-decoder
UNets with localisation-preserving upsampling.

**Sealed-spec status.**  Add to SPEC §3.4 ``nitrix.morphology``
public surface.

### §10.A.4 `nitrix.stats.lme` — promoted from STRETCH to shipped

**Plan reference.**  SPEC §3.5 declared LME as ``[STRETCH]`` with
"namespace reserved; no implementation required" (SPEC.md:440).
``IMPLEMENTATION_PLAN.md:679`` declared "stub module with
``NotImplementedError``."

**What shipped.**

- ``reml_fit(Y, X, Z, ...)`` — voxelwise REML via the FaST-LMM
  spectral rotation trick.  Validated against
  ``statsmodels.MixedLM`` to ~5e-3 (the convergence floor of
  both solvers).
- ``flame_two_level(beta, var_within, X_group, ...)`` — single-
  parameter REML (the FSL FLAME equivalent).
- Memory regime: shared-design vmap over voxels with no
  ``V * N^2`` intermediate; scales to V=1M voxels at fMRI shapes.

**Why it respects separation of concerns.**  Returns
``NamedTuple`` of arrays (no PyTree module; no learnable-state
object); routes ``residualise`` and ``safe_eigh`` from
``nitrix.linalg``; no LME-specific kernel.  The Phase 5 "namespace
reserved" commitment was a hedge against having to write a
custom solver; consumer pressure made the FaST-LMM /
single-parameter REML scope tractable to ship inside the existing
substrate.

**Driving consumer.**  hypercoil group-analysis pipelines
(JOSA feedback batch; sealed).  Continues to be needed for
fMRI group-level inference.

**Sealed-spec status.**  Promote SPEC §3.5 from ``[STRETCH]`` to
``[CORE]``; remove the "stub" language from
``IMPLEMENTATION_PLAN.md §3.B.10`` (or equivalent reference).
Document the design in ``docs/design/lme.md`` (already done).

---

## §11 Plan / SPEC hygiene actions

Items the next plan refresh should pick up alongside the §10.A
entries:

1. **MIGRATION.md refresh.**  Currently tagged "draft v0";
   predates the geometry / graph / morphology / smoothing /
   sparse namespaces.  Refresh against the current ``src/`` tree.
2. **Tutorial notebooks (Phase 5 commitment).**  The plan promises
   "writing a custom semiring", "using ELL for mesh ops",
   "choosing between gaussian and bilateral_gaussian".  None
   landed yet.  Notebook gestation should track the perf-bench
   work (B2 in ``docs/feature-requests/internal-backlog.md``).
3. **G0 / G1 / G2 gate outcomes** in ``§10`` deviation log.
   The plan's gate machinery says outcomes are logged in §10;
   the section is currently empty.  Backfill with the actual
   gate decisions (G0 ELL substrate baseline shipped; G1
   custom-VJP coverage shipped; G2 sparse-substrate consumer
   adoption ongoing).

---

## §12 Brainstorm: candidate primitives for future scope

Items below are **not commitments** — they're substrate-
compatible primitives that would land cleanly if consumer
pressure arrived.  Each entry names the substrate composition
(why it fits), the likely consumer (which class of model would
surface it), and the rough effort.  The acceptance protocol
in ``§13`` governs promotion to sprint scope.

> **Live tracking (2026-06-02).**  Each candidate below (§12.1–§12.19)
> now has an individual tracking doc under
> ``docs/feature-requests/`` (one doc per item, indexed by
> ``docs/feature-requests/README.md``), carrying its current status
> against the live ``src/nitrix`` surface — several items already have
> partial substrate shipped.  This §12 catalogue remains the canonical
> origin record (the §13 protocol references these by number); the
> feature-request docs are the live, per-item tracking surface.  §12.20
> is informational and stays here.

### §12.1 Iterative Krylov solvers — `nitrix.linalg.krylov`

**Composition.**  Implicit-operator pattern already proven via
``laplacian_eigenmap`` LOBPCG path: the operator is matrix-vector-
product-only (dense, ELL, or SectionedELL); implicit-VJP closes the
backward.  Generalise to:

- ``cg(A, b, x0, *, tol, max_iter)`` — symmetric positive-definite
  solver.
- ``minres(A, b, x0, ...)`` — symmetric indefinite.
- ``lsqr(A, b, x0, ...)`` — rectangular least-squares.
- ``bicgstab(A, b, x0, ...)`` — nonsymmetric.

**Likely consumer.**  Implicit smoothing (``(I - αL)x = b``),
gradient-flow integration, surface harmonic-coordinate solves,
neural-ODE adjoint passes.

**Effort.**  M.  CG is xs (it's 30 lines on top of a matvec);
the implicit-VJP pattern is already validated.  Other solvers
follow the same template.

### §12.2 Matrix functions — `nitrix.linalg.matrix_function`

**Composition.**  ``linalg.symlog`` / ``symsqrt`` / ``sympower``
already implement matrix log / sqrt / power via ``eigh``.
Generalise to ``matrix_function(A, fn)`` (apply ``fn`` to
eigenvalues; reassemble) plus three named specialisations:

- ``matrix_exp(A)`` — needed for heat-kernel diffusion.
- ``matrix_polynomial(A, coeffs)`` — Chebyshev polynomial of the
  Laplacian; useful for ChebNet, SGWT, low-rank approximations.
- ``frechet_derivative(A, fn, E)`` — directional derivative of a
  matrix function (used in matrix-exp gradient).

**Likely consumer.**  Heat-kernel signatures, ChebNet, spectral
graph wavelets, group-level fMRI dynamic-connectivity factorisations.

**Effort.**  S.

### §12.3 Heat-kernel & diffusion-map embedding — `nitrix.graph.diffusion`

**Composition.**  Composes ``matrix_function`` + ``laplacian`` +
``laplacian_eigenmap``: heat-kernel ``K_t = exp(-tL)`` is a matrix
function of the Laplacian; truncated eigendecomposition gives a
finite-rank approximation.  Diffusion-map embedding is the
weighted eigenmap normalised by the heat-kernel mass.

**Likely consumer.**  Surface-based connectivity embeddings,
manifold-learning preprocessors for fMRI, vertex descriptors for
mesh-correspondence problems (heat-kernel signatures).

**Effort.**  S — depends on §12.2.

### §12.4 Sinkhorn / optimal transport — `nitrix.transport`

**Composition.**  Sinkhorn iteration is *the* flagship use case
for the LOG semiring: the log-domain matmul against a cost
matrix plus alternating row / column normalisation converges to
the entropic OT plan.  Fits the streaming-kernel substrate
exactly (KeOps's original demo).

- ``sinkhorn(cost, mu, nu, *, eps, n_iter)`` — entropic OT plan.
- ``wasserstein_distance(...)`` — derived distance.
- ``barycentric_map(plan, x_source, y_target)`` — pushforward map.

**Likely consumer.**  Shape matching (mesh correspondence),
distribution alignment in fMRI ICA, transport-based segmentation,
domain-adaptation losses for medical-image pretraining.

**Effort.**  M.  Reuses the LOG semiring path (already declared);
the implicit-differentiation story for the OT plan is well-
established (Cuturi 2013; Feydy 2020).

### §12.5 Discrete differential geometry — `nitrix.geometry.dec`

**Composition.**  Generalises ``mesh_cotangent_laplacian`` to
the full discrete-exterior-calculus stack: vertex-edge incidence
(``d_0``), edge-face incidence (``d_1``), Hodge stars (``star_0``,
``star_1``, ``star_2``).  Cotangent Laplacian becomes
``d_0.T @ star_1 @ d_0``.  Every operator is an ELL.

- Differential operators: ``mesh_gradient(mesh) -> ELL``,
  ``mesh_divergence(mesh) -> ELL``, ``mesh_curl(mesh) -> ELL``.
- Hodge stars: ``mesh_star_k(mesh, k) -> ELL``.
- Hodge decomposition: ``hodge_decompose(omega, mesh)`` →
  ``(exact, coexact, harmonic)``.

**Likely consumer.**  Vector-field smoothing on cortical
surfaces, surface flow / advection modelling, Hodge-decomposition-
based shape descriptors.

**Effort.**  M.

### §12.6 Mesh curvature — `nitrix.geometry.curvature`

**Composition.**  Mean curvature is the cotangent Laplacian
applied to vertex positions (already feasible via existing
primitives).  Gaussian curvature is the angle-defect formula
(per-face arithmetic).  Principal curvatures come from the
Weingarten map eigendecomposition (per-vertex 2x2 eigh).

- ``mesh_mean_curvature(mesh)`` — pointwise.
- ``mesh_gaussian_curvature(mesh)`` — pointwise.
- ``mesh_principal_curvatures(mesh)`` → ``(k1, k2, e1, e2)``.

**Likely consumer.**  Cortical surface analysis (gyrification
indices), shape feature extraction, surface-based registration
regularisation.

**Effort.**  S.

### §12.7 Robust statistics — `nitrix.stats.robust`

**Composition.**  M-estimators are iteratively reweighted least-
squares (IRLS) on top of ``residualise`` (already shipped):
``X.T @ W(r) @ X β = X.T @ W(r) @ y`` where ``W`` is the per-
sample weight from the influence function evaluated at the
current residuals.  Pure composition.

- ``huber_regress(X, y, *, delta)`` — Huber-loss regression.
- ``tukey_bisquare_regress(X, y, *, c)`` — full redescender.
- ``mad(x, axis)`` — median absolute deviation (the standard
  scale estimator paired with M-estimators).

**Likely consumer.**  Motion-corrupted fMRI regression,
outlier-resistant group analysis, robust mixed-effects via
joint LME + IRLS.

**Effort.**  S.

### §12.8 Fixed-point combinators — `nitrix.numerics.fixed_point`

**Composition.**  Deep-equilibrium-style ``fixed_point_solve(f,
x0, *, tol, max_iter)`` returns the fixed point of ``f`` plus an
implicit-VJP (Jacobian of ``f`` at convergence, solved via the
existing Krylov solver §12.1).  Generalises the
``scaling_and_squaring`` pattern in ``integrate_velocity_field``.

**Likely consumer.**  Deep-equilibrium models for surface
registration, iterative ICA fixed-point solvers, implicit
filters that solve a per-sample optimisation.

**Effort.**  M.  Depends on §12.1.

### §12.9 Spherical harmonic transforms — `nitrix.geometry.sphere.harmonics`

**Composition.**  Extends the existing ``nitrix.geometry.sphere``
substrate (which currently covers icosphere + ``spherical_conv``)
with classical spherical-harmonic synthesis and analysis at
arbitrary band-limits.  Real and complex SHs.  Forward and
inverse transforms via Driscoll-Healy quadrature.

- ``sht_forward(f, *, band_limit)`` — spatial → SH coefficients.
- ``sht_inverse(coeffs, *, n_lat, n_lon)`` — SH coefficients →
  spatial.
- ``sht_rotation_matrix(R, band_limit)`` — Wigner-D-matrix
  rotation in SH basis.

**Likely consumer.**  Surface-based CNNs at non-icosphere
sampling, SH-equivariant networks (sphere-domain transformers),
fibre-orientation distribution modelling in dMRI.

**Effort.**  M.

### §12.10 Compensated summation / mixed-precision — `nitrix.numerics.precision`

**Composition.**  Pure-numerics utilities that any reduction in
the substrate can drop in.

- ``kahan_sum(x, axis)`` — compensated summation.
- ``neumaier_sum(x, axis)`` — Kahan's improved variant.
- ``stochastic_round(x, dtype)`` — for FP8 / FP16 accumulation.
- ``pairwise_sum(x, axis, blocksize)`` — log-depth tree reduction
  with controlled accumulation order.

**Likely consumer.**  Long-time-series fMRI covariance
accumulation at FP32 (cross-volume drift), Blackwell FP8 paths
when they arrive, reproducible reductions for golden-corpus
tests.

**Effort.**  S.

### §12.11 ODE integrators — `nitrix.numerics.ode`

**Composition.**  Generalises ``integrate_velocity_field``
(scaling-and-squaring) to:

- Explicit RK methods: ``rk4``, ``dormand_prince`` (adaptive).
- Symplectic integrators: ``leapfrog``, ``implicit_midpoint`` for
  Hamiltonian flows.
- Differentiable via the adjoint-equation pattern (Chen et al.
  2018), which is itself a fixed-point + Krylov solve (§12.1 +
  §12.8).

**Likely consumer.**  Neural-ODE-style continuous-time models,
advection-diffusion forward models for deformable registration,
Hamiltonian-Monte-Carlo posterior samplers for fMRI.

**Effort.**  L.

### §12.12 Continuous wavelet transform — `nitrix.signal.cwt`

**Composition.**  Extends the existing ``nitrix.signal`` family
(``lomb_scargle_*``, ``analytic_signal``, ``tsconv``) with
continuous-wavelet analyses at user-specified mother wavelets
(Ricker, Morlet, Paul).

**Likely consumer.**  fMRI / EEG time-frequency analysis,
non-stationary signal characterisation, scalogram features for
downstream classifiers.

**Effort.**  S.

### §12.13 Graph wavelet transforms (SGWT) — `nitrix.graph.wavelet`

**Composition.**  Hammond et al.'s Spectral Graph Wavelet
Transform — Chebyshev-polynomial approximations of band-pass
filters in the Laplacian eigenspectrum.  Direct composition of
``laplacian`` + ``matrix_polynomial`` (§12.2).

**Likely consumer.**  Multiscale features on cortical-surface
graphs, surface-domain wavelet shrinkage for denoising, SGWT-
based feature engineering for connectome analyses.

**Effort.**  S — depends on §12.2.

### §12.14 Graphical LASSO — `nitrix.stats.glasso`

**Composition.**  Sparse precision-matrix estimation
(Friedman / Hastie / Tibshirani 2008):

```
Θ̂ = argmin_Θ ⟨S, Θ⟩ − log det Θ + λ ‖Θ‖_{1,off}
```

with ``S`` the empirical covariance (``nitrix.stats.cov`` already
ships) and ``λ`` the sparsity penalty.  Two natural solver
backends compose with existing substrate:

- **Coordinate descent** (the GLASSO algorithm).  Per-row
  partition + lasso-shrinkage update.  Pure JAX; differentiable
  through the regularisation path via implicit-VJP at the
  KKT-stationary point (composes with §12.1 Krylov).
- **ADMM** (Boyd 2011 splitting).  Two proximal steps:
  log-det + soft-thresholding.  Each ``proximal_log_det``
  composes with ``linalg.symlog`` (already shipped).

Companion utilities: ``glasso_path(S, lambdas)`` for
regularisation sweeps; ``ebic_score(Θ, S, λ)`` for model
selection.

**Likely consumer.**  Functional-connectivity sparse precision
estimation (partial correlations as conditional-independence
graph edges), graphical-model preprocessors for connectome
analyses, ICA-followup decompositions.  fMRI literature has
defaulted to graphical LASSO for ~15 years; native support
would let downstream pipelines avoid round-tripping through
``sklearn.covariance``.

**Effort.**  M.  Coordinate-descent is the standard reference
implementation (Friedman 2008 ~30 lines); making it correctly
differentiable (the right thing usually being implicit-VJP at
convergence) is the careful part.

### §12.15 Adaptive area-weighted barycentric resampling — `nitrix.geometry.sphere.resample`

**Composition.**  Connectome Workbench's ``ADAP_BARY_AREA``
method for cross-mesh resampling (``wb_command -metric-resample``
/ ``-surface-resample``):

- Standard ``BARYCENTRIC`` is the existing
  ``mesh_bary_upsample`` path — per-target-vertex barycentric
  weights against the source triangle containing the projected
  target location.  Already shipped for icosphere hierarchies
  (``icosphere_bary_upsampler``); the gap is the
  arbitrary-triangulation case (native-mesh → standard mesh,
  e.g. subject ``surf.gii`` → ``fs_LR_32k``).
- **``ADAP_BARY_AREA``** additionally weights each source-vertex
  contribution by the source-triangle area, preserving total
  areal measurement under resampling (critical for vertex-area
  scalars, cortical-thickness maps, and any "amount-of-
  cortex" quantity).  The "adaptive" piece adjusts contribution
  width when source / target tessellation densities differ
  (a small target vertex catching from a large source triangle
  should pull from a wider neighbourhood than the strict bary
  weights would suggest).

Primitive shape:

```python
def surface_resample(
    source_mesh: Mesh, source_vals: Float[Array, '... n_source d'],
    target_mesh: Mesh,
    *, method: Literal['barycentric', 'adap_bary_area'] = 'adap_bary_area',
) -> Float[Array, '... n_target d']:
    ...
```

Returns an ELL (the resampling operator) at construction time
plus the resampled values; the ELL can be reused across
features for the same source/target pair.  Host-side
construction (point-in-triangle, area integrals); JAX-native
application via ``semiring_ell_matmul``.

**Likely consumer.**  Any port that consumes Connectome
Workbench outputs (the bulk of HCP / UK Biobank surface data),
``fs_LR`` ↔ ``fsaverage`` round-tripping, individualised-
parcellation pipelines that resample from subject-native to
group surface.

**Effort.**  M.  Point-in-triangle on the sphere is the
non-trivial part (use spherical-triangle bary coordinates,
not the planar formula); the adaptivity rule is a closed-form
weight adjustment from Workbench's source.

### §12.16 Surface-boundary / gradient mapping — `nitrix.graph.parcellation.boundary`

**Composition.**  The Wig / Cohen / Gordon / Schaefer
lineage for functional-parcellation boundary detection
(Cohen 2008; Wig 2014; Gordon 2016; Schaefer 2018):

1. Per-vertex connectivity profile ``C[v, :] = corr(t[v], t[*])``
   over all cortical vertices — already feasible via
   ``nitrix.stats.corr`` applied to the ``(n_vertices, T)``
   vertex time-series matrix.
2. Per-edge dissimilarity between adjacent vertices' profiles:
   ``d(v, u) = 1 − η²(C[v, :], C[u, :])``  (or
   ``1 − corr(C[v, :], C[u, :])`` depending on choice).
3. Per-vertex boundary map ``B[v] = aggregate_{u ∈ N(v)} d(v, u)``
   (mean or max).

This is exactly the shape of ``semiring_ell_edge_aggregate``
under REAL or TROPICAL_MAX_PLUS with
``edge_fn(h_i, h_j, w, ij) = (1 − corr(h_i, h_j))[None]``.
No new primitive needed; ship a named wrapper for
discoverability:

```python
def surface_boundary_map(
    connectivity_profiles: Float[Array, 'n_vertices d_profile'],
    adjacency: ELL,
    *,
    similarity: Literal['eta_squared', 'pearson'] = 'eta_squared',
    aggregate: Literal['mean', 'max'] = 'mean',
) -> Float[Array, 'n_vertices']:
    ...
```

The companion ``eta_squared(x, y)`` helper composes with
``stats``.  Memory caveat at ``ico_7``: the connectivity profile
is ``163842``-dim per vertex; consumers will need to tile or
chunk.  The wrapper accepts arbitrary chunking via
``jax.lax.map`` over the row axis (documented pattern, no API
surface).

**Likely consumer.**  Individualised parcellation pipelines
(MyConnectome-style, Gordon-individual), gradient-based
parcellation regeneration on new datasets, surface-based
boundary visualisation for QA.

**Effort.**  S.  Three lines on top of ``edge_aggregate`` plus
the ``eta_squared`` helper.  Documentation surface is the bulk
of the work.

### §12.17 Watershed segmentation on meshes — `nitrix.graph.parcellation.watershed`

**Composition.**  Priority-flood watershed on a vertex-valued
scalar field with arbitrary mesh adjacency.  The
boundary-mapping output (§12.16) is the natural input: local
minima of ``−B`` are catchment-basin seeds; flooding from
sorted-by-value pixels yields the basin labels.

Primitive shape:

```python
def mesh_watershed(
    field: Float[Array, 'n_vertices'],
    adjacency: ELL,
    *,
    min_basin_size: int = 1,
    h_min: float = 0.0,
) -> Int[Array, 'n_vertices']:
    '''Label each vertex by its watershed basin.'''
```

Implementation is fundamentally serial (priority-queue flooding),
so this is a **host-side NumPy** primitive that returns a JAX
int array.  Same pattern as the existing
``mesh_k_ring_adjacency`` (host-side BFS, JAX-array output).
Composition with the substrate is via the input ``adjacency``
ELL — the algorithm works on any mesh adjacency, not just
icosphere.

**Likely consumer.**  Gordon / Schaefer parcellation
regeneration; any boundary-map → discrete-parcels step;
medical-image post-processing where a marker-based segmentation
is wanted on a surface.

**Effort.**  M.  Priority-flood is well-understood (Barnes 2014
linear-time variant); the engineering surface is making it
correct on irregular triangulations with degenerate basins.

### §12.18 Clustering primitives — `nitrix.numerics.cluster`

**Composition.**  A small family of clustering primitives that
several parcellation strategies share:

- ``kmeans(X, k, *, init, max_iter, similarity)`` — Lloyd's
  algorithm with cosine / correlation / euclidean similarity.
  Pure JAX (vmap over centroid update + nearest-centroid
  assignment); differentiable via implicit-VJP at convergence
  (composition with §12.8 fixed-point combinators).
- ``ward_linkage(X, *, k)`` — agglomerative clustering with
  Ward variance-minimisation criterion.  Host-side
  (heap-priority-queue), JAX-array output.
- ``nmf(X, k, *, max_iter)`` — Lee-Seung multiplicative-updates
  nonnegative matrix factorisation.  Pure JAX; differentiable
  via implicit-VJP at the KKT stationary point.

These compose with the existing ``stats.corr`` /
``stats.cov`` / ``linalg.linear_kernel`` for the
similarity-matrix construction step.

**Likely consumer.**  Yeo 7/17 networks (k-means on
connectivity profiles), Bellec MIST (Ward linkage on stability
matrices), NMF parcellations (Eickhoff group), ICA-followup
clustering.  This is the load-bearing missing piece for the
"functional-parcellation by clustering" family.

**Effort.**  S for ``kmeans``; S for ``nmf``; M for
``ward_linkage`` (the linkage matrix bookkeeping).

### §12.19 Normalised-cut spectral clustering — `nitrix.graph.ncut`

**Composition.**  Shi-Malik 1997 / Craddock 2012-style NCut as a
thin wrapper:

1. ``laplacian_eigenmap`` (already shipped, with implicit-VJP
   through both ``eigh`` and ``lobpcg``) gives the top-``k``
   eigenvectors of the normalised Laplacian.
2. ``kmeans(eigvecs, k)`` (§12.18) discretises into parcels.

Primitive shape: a 5-line composition wrapper.

**Likely consumer.**  Craddock 2012 functional parcellation,
NCut-based ROI generation for connectome edges.

**Effort.**  XS — depends on §12.18.

### §12.20 Functional-parcellation strategy survey (informational)

For onboarding-doc / future-reference: the parcellation
methods we should expect ilex to surface, mapped to the
primitives that support them:

| Strategy | Primary primitive(s) needed | Status |
|---|---|---|
| Yeo 7 / 17 networks (k-means on conn profiles) | §12.18 kmeans | brainstorm |
| Craddock NCut | §12.19 ncut → §12.18 kmeans | brainstorm |
| Power ROIs (meta-analytic peaks) | no primitive (coords only) | n/a |
| Gordon individual / group | §12.16 boundary_map + §12.17 watershed | brainstorm |
| Schaefer Local-Global | §12.16 + gradient-weighted MRF | brainstorm |
| Bellec MIST | §12.18 ward_linkage + stability bootstrap | brainstorm |
| Glasser HCP multimodal | supervised classifier — **out of scope** (model, not primitive) | n/a |
| Eickhoff NMF parcellations | §12.18 nmf | brainstorm |
| Salehi individualised VAE | model, not primitive — **out of scope** | n/a |

The pattern: the primitive surface is small (5–6 reductions /
clustering operators) and substrate-aligned (every operator
composes with semiring/ELL/laplacian); the *strategies* are
recipes on top.  This is the right level of abstraction for
nitrix: ship the primitives, let the strategies live in
``ilex`` / ``nimox`` as recipe documents that compose them.

---

## §13 Acceptance protocol for §12 brainstorm items

A §12 candidate graduates to sprint scope when **all** the
following hold:

1. **Concrete consumer.**  A named downstream port (ilex model,
   hypercoil pipeline, nimox vendor) is blocked or
   workaround-laden without it.  "We might want X someday"
   does not qualify — keep it in ``docs/feature-requests/internal-backlog.md``.
2. **Substrate composition is verified.**  Implementor has
   sketched the composition with existing primitives in plain
   English (no new kernel, no parallel API).  If the
   composition requires a new primitive, that primitive
   itself goes through §13 first.
3. **Separation-of-concerns invariant holds.**  See §0: every
   addition should pull its weight in the existing structure,
   not create a parallel structure.
4. **Effort fits the time-budget.**  An ``XS`` / ``S`` item can
   land alongside a consumer-ask sprint; ``M`` items need a
   dedicated sprint slot; ``L`` items need a SPEC-level review
   first.

When a candidate graduates, the entry moves from §12 to the next
SPEC update's §10.A (deviation log) at the same time as the
implementation lands.  This keeps the brainstorm honest: items
in §12 are aspirational; items in §10.A are shipped.

---

## §14 Out-of-scope reminders

The drift policy in §0 does **not** license:

- New top-level subpackages without a clear substrate-composition
  story (no ``nitrix.image.classifier``, no ``nitrix.model.X``).
- PyTree-of-arrays / module classes; nitrix primitives return
  named tuples or arrays.  ``LMEResult`` is a ``NamedTuple``;
  ``IcosphereHierarchy`` is a frozen dataclass holding tuples.
  Neither is a learnable module.
- A "message-passing" base class.  ``edge_aggregate`` is the
  primitive; the user's closure is the message.
- Hardware-specific code paths beyond the ``pallas-cuda`` / ``jax``
  pair declared in v0.2 §1.1.
- Replacement of an existing primitive when an extension would
  do.  Prefer adding a kwarg over forking a function.

These mirror the non-goals in SPEC §1 and the
"What deviation does *not* license" in
``IMPLEMENTATION_PLAN.md §2.5``.  Both stand.
