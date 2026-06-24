# nitrix docs

Working notes for the in-flight implementation.  This directory captures
the **why** behind major decisions made during the Phase 0 / Phase 2
substrate build-out, plus links to the frozen benchmark artefacts that
back those decisions.

The intended endpoint is the *explanation* component of a Diataxis-style
documentation set: understanding-oriented essays distinct from the
reference docs (generated from docstrings), how-to guides (recipe-style
solutions), and tutorials (guided onboarding).  Until that structure
solidifies, this directory is a working scratchpad.  Pieces here should
be readable on their own, link liberally to the SPEC and to the
benchmark reports, and survive being plucked into the eventual
``docs/explanation/`` tree without major rewriting.

## Quick reference

- [`op_matrix.md`](op_matrix.md) — auto-generated status board for
  every public op: which JAX transformations (`jit`, `grad`,
  `vmap`, `jit∘grad`) work, what algorithmic invariants the
  current implementation declares, and the wall-time ratio
  against the natural reference where benched.  Regenerate via
  `python tools/op_matrix.py`; design rationale in
  [`design/op-matrix.md`](design/op-matrix.md).

## Index

### Phase 1: namespace consolidation

- [`design/linalg.md`](design/linalg.md) -- the four ``linalg``
  subpackages: ``matrix`` (sym↔vec bijection with hand-rolled
  custom-VJPs, Toeplitz, eigenspace reconditioning),
  ``residual`` (Cholesky-default OLS / WLS / ridge,
  ~9× faster than the legacy SVD path), ``kernel`` (squared-L2
  via the identity formula to eliminate the legacy ``O(n m d)``
  memory blowup), ``spd`` (SPEC §4.5 stability rewrite with
  eigenvalue-threshold clipping), plus the shared
  ``_solver.safe_eigh`` helper.
- [`design/stats.md`](design/stats.md) -- ``covariance`` (real and
  **complex** -- the "silently wrong on complex inputs" failure
  mode from the legacy is fixed and regression-tested; the
  JIT-trap in the weighted-cov dispatch is eliminated),
  ``fourier`` (analytic signal / Hilbert / envelope / inst freq
  / phase; vectorised mask construction).
- [`design/signal-and-numerics.md`](design/signal-and-numerics.md) --
  ``signal.linear_interpolate`` via parallel ``associative_scan``
  (replaces sequential ``lax.scan``), ``signal.lomb_scargle_*``
  via joint-GLM regression (fixes the visible-boundary-
  discontinuity failure mode of the legacy independent-per-
  frequency LS), the shared-Gram memory-safety pattern at fMRI
  scale, ``signal.polynomial_detrend``, ``signal.tsconv``,
  plus ``numerics.tensor_ops`` and ``numerics.normalize``.

### Design rationale

- [`design/backend-selection.md`](design/backend-selection.md) -- the
  three-level backend resolution, ``NitrixBackendFallback`` warning
  category, ``NITRIX_STRICT_BACKEND`` / ``NITRIX_SILENCE_FALLBACK`` env
  vars, and the per-signature deduplication rule.
- [`design/semiring-protocols.md`](design/semiring-protocols.md) -- the
  ``Monoid`` / ``Semigroup`` Protocol pair, why the accumulator is a
  pytree, the relaxed-``Semiring`` vs ``StrictSemiring`` split, and how
  user-defined algebras plug in.
- [`design/streaming-kernel.md`](design/streaming-kernel.md) -- what the
  "KeOps-style streaming kernel" claim from SPEC §4.1 actually means in
  our code, why we deliberately avoid tensor cores, and how we verified
  the ``O(M·N)`` peak-HBM claim empirically.
- [`design/ell-on-triton.md`](design/ell-on-triton.md) -- why
  ``semiring_ell_matmul`` runs on JAX unconditionally at first GA, what
  the Triton lowering gap is, and what would change the policy.
- [`design/op-matrix.md`](design/op-matrix.md) -- the auto-
  generated op-status board (``docs/op_matrix.md``) and its
  generator tool (``tools/op_matrix.py``).  Schema, host-
  snapshot caveat, "what the probes cannot catch."
- [`design/perf-audit-2025-05.md`](design/perf-audit-2025-05.md)
  -- audit of un-benched ops vs natural references (numpy /
  scipy / sklearn / statsmodels).  Headline: nitrix wins 100-800×
  at fMRI-realistic scales; the single "gap" against
  ``scipy.ndimage.distance_transform_edt`` is a feature-coverage
  issue (we ship chamfer DT, scipy ships Euclidean DT) not a
  speed issue.  Recommendation: doc-only Tier 1 (done); ship
  ``distance_transform_edt`` as a 1.x follow-up when a consumer
  asks.
- [`design/lme.md`](design/lme.md) -- voxelwise linear mixed-
  effects models: ``reml_fit`` (general variance-components REML
  via the FaST-LMM spectral rotation trick) and
  ``flame_two_level`` (the FSL FLAME equivalent, single-
  parameter REML for fMRI group analysis).  Memory regime:
  shared-design vmap over voxels with no V*N^2 intermediate;
  scales to V=1M voxels at fMRI shapes.  Validated against
  ``statsmodels.MixedLM`` to ~5e-3 (the convergence floor of
  both solvers).
- [`design/sparse-specialisations.md`](design/sparse-specialisations.md)
  -- ``nitrix.sparse.grid`` (regular-grid stencils with scipy-parity
  boundary modes; ``grid_laplacian`` / ``grid_identity`` /
  ``regular_grid_stencil``) and ``nitrix.sparse.mesh`` (icosphere
  via recursive subdivision; ``mesh_k_ring_adjacency`` via BFS;
  ``mesh_cotangent_laplacian`` for surface processing).  All return
  ``ELL`` for direct composition with the substrate.
- [`design/mesh-graph-conv.md`](design/mesh-graph-conv.md) --
  ``semiring_ell_edge_aggregate`` (the edge-functional extension
  of ``semiring_ell_matmul`` that covers GCN / GAT / EdgeConv /
  MoNet / ChebNet with one ``edge_fn`` callable + a semiring),
  ``morphology.pooling`` (``max_pool_with_indices_nd`` /
  ``max_unpool_nd`` for encoder-decoder pipelines, with the
  argmax-agreement parity caveat documented), and the
  ``sparse.mesh`` convenience wrappers (``mesh_pool_max`` /
  ``mesh_unpool_max`` / ``mesh_bary_upsample``).  Closes the
  ilex mesh-graph consumer asks FA2–FA5 (see ``SPEC.md §4.2``; §11 provenance).
- [`design/backward-kernels.md`](design/backward-kernels.md) -- the
  per-algebra ``jax.custom_vjp`` story, the per-algebra differentiability
  vocabulary, and the G1 finite-difference gate.
- [`design/convolution.md`](design/convolution.md) -- explicit im2col +
  ``semiring_matmul`` as the JAX-path strategy, the NaN-safe patch
  extractor, comparison to cuDNN, and the path to a future Pallas
  implicit-GEMM kernel.
- [`design/morphology.md`](design/morphology.md) -- ``dilate`` /
  ``erode`` / ``open`` / ``close`` / ``distance_transform`` as
  TROPICAL_* specialisations of ``semiring_conv``; ``median_filter``
  as the prototype "almost a semiring" gather-based op;
  n-D coverage including 4D fMRI shape (which cuDNN doesn't
  support).
- [`design/bias-field.md`](design/bias-field.md) -- ``nitrix.bias``:
  N4 (Tustison) bias-field correction validated to SimpleITK/ANTs parity,
  the separable regular-grid B-spline approximator and N3/N4 Wiener
  histogram-sharpening primitives it is built from, and the
  ``bias_field_correction`` dispatcher's unbiased least-squares / P-spline
  estimators.  Covers the parity-vs-correctness tradeoff, the
  smooth-shading-vs-sharp-anatomy framing, the bias-variance / ridge
  ("regularisation is denoising") finding, why multi-resolution is
  load-bearing, "no Pallas needed", and the Tier C/D extensions.  New
  subsystem added under the IMPLEMENTATION_PLAN §2 deviation protocol.
- [`design/smoothing.md`](design/smoothing.md) -- ``gaussian``
  (separable, scipy-parity), ``bilateral_gaussian`` (marquee edge-
  preserving capability: the **bounded bilateral** via
  ``semiring_ell_matmul`` over a feature-distance-weighted adjacency,
  with a factored ``FeatureMetric``, validity mask, and ``n_iters``),
  ``susan_emulator`` (composition with documented deltas from FSL
  SUSAN), plus the sectioned-ELL story for variable-degree adjacencies.
- [`design/bounded-bilateral.md`](design/bounded-bilateral.md) -- the
  bounded bilateral design and the **permutohedral retirement**: why
  conceding bounded support dissolves the lattice's obstacles, the
  ``FeatureMetric`` ADT (diagonal / low-rank / full), validity masking
  as a correctness fix, and fixed-affinity iteration.
- [`design/geometry.md`](design/geometry.md) -- ``grid``
  (voxelmorph-style deformable-registration primitives written from
  scratch), ``sphere`` with ``spherical_conv`` re-backed on
  ``semiring_ell_matmul`` (third validation of the substrate bet),
  ``coords`` (centre-of-mass, displacement, compactness penalty),
  plus the "rename for clarity" pass that retired the legacy
  ``cmass_*`` / ``diffuse`` / ``vec_int`` / ``rescale`` names.
  Updated for the JOSA-feedback sprint with ``mode`` pass-through
  to ``spatial_transform`` / ``integrate_velocity_field``, batched
  warp via shape relaxation, and ``jacobian_(det_)displacement``
  with explicit-formula determinants for ``d <= 3``.
- [`design/sphere-grid.md`](design/sphere-grid.md) --
  parameterised-sphere topology: pole-flip + longitudinal-wrap
  padding via ``sphere_grid_pad_2d``, and why composition with
  ``padding='VALID'`` kernels is the right design over threading
  another boundary mode through every kernel.
- [`design/graph.md`](design/graph.md) -- ``laplacian``,
  ``community`` (with sparse-factored ``relaxed_modularity``),
  ``connectopy`` (``eigh`` + ``lobpcg`` paths, multi-format
  support, source-device preservation under the cuSolver
  workaround).  Modularity primitives moved from ``laplacian`` to
  ``community`` against the spec's letter to keep ``laplacian``
  pure.  LOBPCG differentiability story in
  [`design/lobpcg-implicit-vjp.md`](design/lobpcg-implicit-vjp.md).
- [`design/lobpcg-implicit-vjp.md`](design/lobpcg-implicit-vjp.md)
  -- implicit-VJP wrapper around ``lobpcg_standard`` for **all
  three** operator formats (dense, flat ELL, SectionedELL):
  Hellmann-Feynman + F-matrix subspace-projector approximation,
  near-degeneracy clamp, sparsity-pattern-projected backward for
  ELL and per-section row_groups-gather-projected backward for
  SectionedELL.  Closes the last non-differentiable graph path.
- [`design/eigsolve-dispatcher.md`](design/eigsolve-dispatcher.md) --
  **(agreed design, not yet implemented)** lifting the extremal
  top-k eigensolver dispatch out of ``graph.connectopy`` into a
  dedicated ``nitrix.linalg.eigsolve`` facility, and extending
  ELL / SectionedELL coverage to ``shift_invert`` / ``poly``.  Keyed
  on the insight that the solver *forward* (method) and the gradient
  *backward* (operand format) are orthogonal, so the missing sparse
  cells fall out of the refactor.  ``eigh`` is folded in only in its
  extremal role; ``safe_eigh`` stays the full-spectrum primitive.
- [`design/permutohedral-g2.md`](design/permutohedral-g2.md) --
  the G2 tripwire outcome for ``permutohedral_lattice`` (**retired**;
  see ``SPEC.md §4.4``; superseded by the bounded bilateral, see
  ``bounded-bilateral.md``).  Retained as the historical record of the
  structural obstacles (hash-table representation, neighbour lookups
  during blur, simplex-identity gradient discontinuity) that bounded
  support dissolves.
- [`design/testing-strategy.md`](design/testing-strategy.md) -- how
  backend parity, identity propagation, numerical stability, finite-
  difference grad, and fallback observability are organised in the
  test suite, plus the JIT-the-loss trick that makes finite-diff
  affordable.

### Benchmark artefacts

See [`benchmarks/`](benchmarks/) for the index.  The raw markdown
reports live alongside the bench scripts in [`/bench`](../bench/) and
are regenerated by running those scripts.

## Conventions

- Each design doc opens with a one-paragraph TL;DR so future-you can
  skim and decide whether to read further.
- Where a decision was contested, the rejected alternatives are
  preserved (under "What we considered and didn't pick").
- Cross-references to SPEC sections are by ``§``; cross-references to
  source files are by relative path.
- Benchmark numbers are quoted with a date and hardware; assume any
  number older than the latest pinned JAX is stale.
