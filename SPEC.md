# nitrix — Specification (v1, consolidated)

> **Status.** Architectural specification, consolidated against the shipped
> code (2026-06-24). This single document supersedes and folds in the former
> `SPEC_UPDATE.md` / `SPEC_UPDATE_v0.2…v0.5.md` (their binding content is now
> inline; see the **Provenance map**, §11, which maps old section numbers here
> so existing in-code `SPEC_UPDATE §x` citations still resolve).
> **Scope.** *What* nitrix is, what belongs in it, what does not, and the
> contract it offers upstream libraries (`thrux`, `bitsjax`, `nimox`) and
> downstream consumers (`ilex`, `entense`). The firm concern boundaries (§1
> non-goals, §6) are load-bearing and take precedence over any convenience.

---

## 1. Charter

nitrix is the **lowest-level numerical substrate** of the diffprog neuroimaging
ecosystem: a pure-numeric, all-JAX library where every public symbol takes JAX
arrays (and, where relevant, a `jax.random` key or a static shape/param spec) and
returns JAX arrays. nitrix has no knowledge of image containers, sidecar
metadata, BIDS, filesystems, training loops, or PyTree modules. Those concerns
belong to libraries that *depend on* nitrix.

The long-term vision is a substrate of differentiable numerical primitives
sufficient to build software in the class of FSL / FreeSurfer / ANTs / AFNI on
GPU-accelerated hardware. In practice the surface has grown consumer-first
(ilex / JOSA ports surface a primitive → it lands here), and that is expected to
continue — governed not by "declared in advance" but by the separation-of-
concerns invariant (§6, §9).

### 1.1 Hardware scope

- **In scope:** NVIDIA GPUs of **Ampere generation and newer** (A100, A40, RTX
  30/40/50xx, L4/L40, H100/H200, B100/B200) via the **Pallas Triton** backend,
  and a **pure-JAX CPU fallback** (functionally correct; CPU performance is not a
  goal, but is exercised in CI as the correctness floor).
- **Out of scope at GA:** TPU / `pallas-tpu` (no dev access; the streaming-kernel
  design is architecturally compatible — revisit post-1.0 if access appears),
  AMD ROCm, Apple Metal, Intel GPUs (all fall back to JAX-CPU), and the
  Mosaic-GPU backend (Hopper-only would exclude most academic-lab hardware).
- **Accepted risk:** Pallas Triton is maintained best-effort by JAX, not as the
  primary Pallas target. The JAX fallback is the contractual floor; the §3.2
  fallback machinery covers Triton regressions, and releases pin a minimum JAX
  version (§8).

### Non-goals (firm boundary)

- No NIfTI / GIfTI / CIfTI I/O — that is `thrux`.
- No transform / pipeline / dataset abstractions — that is `bitsjax` / `entense`.
- No Equinox / PyTree modules — those are `nimox`. nitrix returns `NamedTuple`s
  or frozen dataclasses of arrays, never module objects.
- No template / atlas registration as user-facing API (low-level primitives only;
  atlas data structures live in `thrux`).
- No `loss` namespace and no objective scalarisation — that is `nimox` (§5).

---

## 2. Design tenets

1. **Pure functional.** Every public symbol is a function over arrays. *Clarified
   (formerly "(Array,…)→Array"):* a symbol may also be a **keyed pure generator**
   — `(shape | params, key) -> Array`, where the `jax.random` key is itself a
   typed `Array` — so `augment.*` generators and randomized solvers
   (`stats.pca(solver='randomized')`) are in scope. Pure in the JAX sense:
   deterministic given `(inputs, key)`, no hidden state. RNG *policy* (which key,
   how to split, schedules) stays with the caller. No PyTrees-as-modules in the
   public API; PyTree-shaped *config* records are acceptable keyword args.
2. **All differentiable.** Subgradients are explicit where appropriate; custom
   VJPs are registered where numerical stability or efficiency requires it.
   Non-differentiable outputs (hard labels, arg-max) are documented as such.
3. **JAX + Pallas-Triton on NVIDIA, with JAX fallback.** Hardware-aware Pallas
   Triton kernels for the marquee ops on Ampere+; a pure-JAX fallback is always
   present and exercised in CI. Backend selection is deterministic and
   user-overridable (§3). CPU is functionally supported; CPU perf is not a goal.
4. **Typed at boundaries.** `jaxtyping` annotations on all public functions. No
   bare `Array | NDArray` unions.
5. **No transitive heavyweight deps.** nitrix may import only `jax`, `jaxtyping`,
   `numpy`. `scipy`, `nibabel`, `numpyro`, `equinox`, etc. are forbidden at
   runtime import; test/reference deps are scoped to `tests/`.
6. **Stable kernels, breakable APIs.** Until 1.0 the API is mutable, but kernel
   output is numerically reproducible across releases (pinned references, §8).
7. **Loud fallbacks.** A silent performance regression is a bug. When a resolved
   backend cannot run (Triton can't tile a shape/algebra, hardware unsupported,
   …) nitrix falls back to JAX and emits a structured `NitrixBackendFallback`
   warning, deduplicated per `(function, shape, dtype, backend)` per process.
   `NITRIX_SILENCE_FALLBACK=1` suppresses; `NITRIX_STRICT_BACKEND=1` escalates to
   error.
8. **Reproducibility via a golden corpus.** Each kernel × dtype (× backend ×
   driver) has a checked-in reference under `tests/golden/`; tolerances are
   pinned in `tests/tolerance.toml` (and, for the driver axis, in the divergent-op
   registry, §3.3). A tolerance change is a public-API change (CHANGELOG entry).
9. **Reproducible dispatch (the `driver` axis).** Where nitrix auto-selects among
   *numerically divergent* implementations of one operation (e.g. sequential vs
   parallel scan, FIR vs recursive Gaussian, deterministic vs atomic histogram),
   the default is hardware-aware, but (a) it is overridable per call by a
   consistently-named `driver=` keyword (distinct from `backend=`), and (b) a
   single library-level mode — `nitrix.reproducible()` / `NITRIX_REPRODUCIBLE=1`
   — forces the **canonical** variant at *every* such site (and a deterministic
   reduction where the default is run-to-run nondeterministic), trading peak
   performance for cross-platform / cross-run stability up to each site's
   registered tolerance. Every such site is a registered, golden-tested contract
   (`nitrix.divergent_ops()`); the divergence is documented, never silent. The
   promise is *same algorithm / reduction order + determinism*, **not**
   bit-identity across hardware (FMA / transcendental / fusion residuals remain,
   bounded by the tolerance). Mechanically-equivalent dispatch (bit/ULP-identical,
   tested) is exempt. See §3.3 and `docs/feature-requests/reproducible-dispatch.md`.
10. **Two-tier parity.** An op may ship a bit-faithful JAX **reference** (the
    oracle consumers may pin) plus a fused **Pallas** path certified only
    `pallas ≈ jax` within the pinned `[op.dtype.pallas-cuda]` tolerance. The
    `custom_vjp` lives on the fused path; the reference is autodiff-native.

---

## 3. Dispatch & backends

### 3.1 Two orthogonal axes

- **`backend=`** — the *execution engine*: `'pallas-cuda'` vs `'jax'`. Resolves
  via `nitrix._internal.backend.resolve_backend`.
- **`driver=`** — the *numerical variant* of the same math (the scipy/torch
  convention). Resolves via `nitrix._internal.config.resolve_driver`. Distinct
  from `backend=` (engine), `method=` (algorithm *family*, e.g. interpolator or
  eigensolver), and `representation=` (e.g. registration group/algebra).

Keep them distinct: `backend` picks *where/which kernel* runs; `driver` picks
*which numerically-divergent recipe*.

### 3.2 Backend resolution (§7.2) + fallback observability

Three-level: explicit `backend=` keyword → `NITRIX_BACKEND` env → auto-detect.
Auto-detect resolves to `'pallas-cuda'` only when an Ampere+ NVIDIA GPU is
visible (compute-capability probed once at import), else `'jax'`. Under
reproducibility mode (tenet 9) auto-detect prefers `'jax'` (the reference), so a
fused-kernel op reproduces across platforms; an explicit `backend=`/env still
overrides. An explicit `backend='pallas-cuda'` on unsupported hardware raises
`NitrixBackendError`. Fallback is loud (tenet 7). `Backend = Literal['auto',
'pallas-cuda', 'jax']` (no `pallas-tpu`). Pallas kernels live in `_kernels/`;
nitrix pins a minimum JAX version and files a release-blocking issue (never a
silent disable) if a Triton change breaks a kernel between pins.

### 3.3 Driver resolution + reproducibility mode

`resolve_driver(driver, *, op, fast)` returns the concrete variant: an explicit
`driver` wins; else the registered **canonical** under reproducibility mode; else
`fast()` (the hardware-/shape-aware pick, evaluated lazily). Reproducibility mode
is a trace-time `contextvars` flag seeded from `NITRIX_REPRODUCIBLE`, toggled by
`nitrix.reproducible()` (context manager, nestable; `reproducible(False)` carves
a fast region) / `set_reproducible()`. Every divergent op is registered in the
central manifest (`nitrix._internal._divergent_ops`, eager-imported so
`nitrix.divergent_ops()` is complete at `import nitrix`) with
`{op, canonical, fast, driver_values, tolerance}`. Public surface at the package
root: `reproducible`, `reproducible_enabled`, `set_reproducible`, `divergent_ops`,
`DivergentOp`. **Registered divergent ops (5):** `signal.iir` (fft/scan/
associative; canon `scan`), `nn.ssm.selective_scan` (sequential/associative/
chunked; canon `sequential`), `geometry.cubic_bspline_prefilter` (sequential/
associative; canon `sequential`), `register.field_smooth` (fir/recursive; canon
`fir`), `metrics.joint_histogram` (onehot/scatter — the determinism case; canon
`onehot`). A completeness guard (§8) fails CI if a new platform-flip is added
ungoverned.

### 3.4 Pallas surface

Pallas kernels are an implementation detail behind the public API; kernel files
are private (`_kernels/cuda/`: attention, selective_scan, norm, semiring_matmul,
semiring_ell_matmul, demons_force, lncc_force). Each registers a `custom_vjp`
whose backward is a paired Pallas kernel or a JAX fallback. Wall-clock parity
versus heavy external references (torch / triton / cuda / ANTs / scipy) is **not**
nitrix's concern — it is delegated to the sibling perf suite (`bench/`,
`nitrix-perf-bench`). nitrix owns correctness and gross-memory behaviour.

---

## 4. Subsystems (the shipped surface)

The code is the surface-of-record; this section states each subsystem's intent,
its key surface, and any contract/boundary. "[CORE]" marks the marquee substrate.

### 4.1 `nitrix.semiring` — KeOps-style streaming reductions  [CORE]

Arbitrary-algebra reductions over matmul, convolution, and ELL-sparse adjacency
contraction, with the K-loop folding rank-1 combines into the accumulator (the
`(BM,BK,BN)` value tensor is never materialised). Algebras are a `(Monoid,
Semigroup, name)` triple over pytree carry state. **Typing:** `Semiring` (relaxed,
no associativity guarantee — the default) and `StrictSemiring <: Semiring` (asserts
associativity/distributivity); ops needing free reassociation document the
requirement. **Built-ins:** `REAL`, `LOG`, `TROPICAL_MAX_PLUS`,
`TROPICAL_MIN_PLUS`, `BOOLEAN` (strict), `EUCLIDEAN` (relaxed, non-associative).
**Surface:** `semiring_matmul`, `semiring_conv`, `semiring_ell_matmul`,
`semiring_ell_rmatvec`, `semiring_ell_edge_aggregate` (user `edge_fn` per
(vertex, neighbour) reduced under REAL/TROPICAL_*), `ell_row_softmax`, each with a
`reference_*` JAX oracle. No tensor-core `dot` (the algebra is general); no BCOO.
User-defined algebras are forward-only by default (wrap in `custom_vjp` to
differentiate).

### 4.2 `nitrix.sparse` — geometry-aware sparsity  [CORE]

**ELL is the primary format** (`ELL`: `(values, indices)` + row count +
algebra-identity padding) — brain adjacencies are naturally fixed-degree.
**`SectionedELL`** (CORE, not stretch) buckets variable-degree rows by
`ceil(log2(k))` and scatters back, preventing silent OOM on ragged graphs
(`sectioned_ell_from_ragged`, `sectioned_semiring_ell_matmul/_rmatvec`). Plus
regular-grid sparsity (`grid_identity`, `grid_laplacian`, `regular_grid_stencil`)
and a mesh layer atop ELL: `Mesh`, `icosphere`, `IcosphereHierarchy`
(`icosphere_hierarchy`, `icosphere_cross_level_adjacency` for pooling,
`icosphere_bary_upsampler` for continuous upsampling — all cross-level operators
are ELLs), `mesh_cotangent_laplacian`, `mesh_k_ring_adjacency`,
`mesh_laplacian_smooth`, `mesh_mass_matrix`, `mesh_pool_max`/`mesh_unpool_max`,
`mesh_bary_upsample`, `mesh_coarsen_meanpool`, `vertex_areas`, `face_areas`,
`compute_vertex_normals`. No `jax.experimental.sparse` BCOO.

### 4.3 `nitrix.morphology` — built atop semiring  [CORE]

Binary/grayscale `dilate`/`erode`/`open`/`close` and `distance_transform`(`_edt`)
as TROPICAL_MIN/MAX_PLUS specialisations of the semiring conv; plus gather-backed
ops outside the semiring (state unbounded in the K-loop): `median_filter`
(`gather → median`), `connected_components` / `largest_connected_component`
(label-propagation fixed point), and `max_pool_with_indices_nd` /`max_unpool_nd`
(strided argmax variant of dilate; cross-framework parity is argmax-of-output,
not raw-logit).

### 4.4 `nitrix.smoothing` — edge-preserving / baseline  [CORE]

- `gaussian` — separable Gaussian, the unconditional baseline (the FIR vs
  Young–van Vliet recursive engine is the `driver` axis, §3.3).
- `bilateral_gaussian` — the **bounded** high-dimensional bilateral (the marquee
  edge-preserving capability): one gather + one weighted reduction via
  `semiring_ell_matmul` over a bounded neighbourhood (`int` k-NN / index array /
  `ELL`), with values/features decoupled, a factored `FeatureMetric` (`M=LLᵀ`),
  optional validity `mask`, and fixed-affinity `n_iters` (bounded dense-CRF
  mean-field). Statically shaped, jit/vmap/grad-clean, smooth gradients
  everywhere.
- `FeatureMetric` ADT (`nitrix.smoothing.metric`): `DiagonalMetric(sigma)`,
  `FactorMetric(L)` (low-rank when `k<d_f`), `block_diagonal_metric`,
  `metric_from_spd`; registered pytrees, `L` differentiable end-to-end. Data-
  driven metric *fitting* is a consumer concern (built from `stats`/`linalg`).
- `susan_emulator` — `bilateral_gaussian` + `median_filter` composition
  (documents its deltas from FSL SUSAN); `brute_force_knn`,
  `spatial_cube_neighbourhood` helpers.
- **`permutohedral_lattice` is RETIRED** (not deferred): bounded support dissolves
  the lattice/hash/splat-blur-slice machinery and its gradient discontinuity.
  Bounded bilateral supersedes its role; the symbol does not exist and the
  namespace is not reserved.

### 4.5 `nitrix.linalg`

Matrix utilities (`sym2vec`/`vec2sym`/`squareform`/`symmetric`/`toeplitz`(`_2d`)/
`delete_diagonal`/`fill_diagonal`/`recondition_eigenspaces`); confound regression
(`residualise`, `partial_residualise`); solvers (`solve`, `cho_solve`, `cg`, and
the cuSolver-safe `safe_*` family that probe-and-latch to CPU on a dead solver
pool); `matrix_exp`/`matrix_log`; `randomized_svd`; nonlinear least squares /
optimisation (`gauss_newton`, `levenberg_marquardt`, `implicit_least_squares`,
`implicit_minimize`, `OptimizeResult`); parameterised kernels (`gaussian_kernel`,
`rbf_kernel`, `linear_kernel`, `polynomial_kernel`, `cosine_kernel`,
`sigmoid_kernel`, `matern_spectral_density`, `se_spectral_density`,
`parameterised_norm`, `linear_distance`); SPD-manifold ops (`symexp`, `symlog`,
`symsqrt`, `symmap`, `sympower`, `tangent_project_spd`, `cone_project_spd`,
`mean_euclidean`, `mean_log_euclidean`).

### 4.6 `nitrix.stats`

Mass-univariate modelling and second-order statistics. Package-level:
covariance/correlation family (`cov`, `corr`, `partialcov`, `partialcorr`,
`precision`, …); GLM (`glm_fit`, `predict`, `t_contrast`, `f_contrast`,
`GLMResult`, `Family`, `Link`); additive models (`gam_fit`, `GAMResult`); Gaussian
processes (`gp_fit`/`gp_predict`, `hgp_fit`/`hgp_predict`, HSGP); generalised
mixed models (`glmm_fit`); non-Gaussian GLMs (`beta_fit`, `gaulss_fit`,
`ordinal_fit`); `pca_fit`/`transform`/`inverse` (eigh-only, cuSolver-safe; keyed
`solver='randomized'`); bases (`bspline_basis`, `hsgp_basis`, `gp_basis`,
`gp_factor_smooth`); shrinkage / sparse precision (`ledoit_wolf`, `oas`, `glasso`,
`glasso_path`); effect size (`confidence_interval`, `standardized_effect`).
Submodules: **`stats.lme`** [CORE, promoted from STRETCH] — voxelwise `reml_fit`
(FaST-LMM spectral rotation, vmap over voxels with no `V·N²` intermediate; ~5e-3
vs statsmodels), `lme_fit`, `flame_two_level` (FSL FLAME); **`stats.inference`** —
`permutation_test`, `tfce`, `cluster_size_map`, `fdr`, `bonferroni`;
**`stats.gaussian`** — distributional score kernels (`gaussian_nll`,
`kl_diagonal_gaussian`, §5). All return `NamedTuple`s of arrays, never modules.

### 4.7 `nitrix.signal`

1-D / time-series DSP: Butterworth IIR (`butterworth_sos`, `iir_filter`,
`sosfilt`, `sosfiltfilt` — the fft/scan/associative engine is the `driver` axis,
§3.3) and the `bandpass`/`bandstop`/`lowpass`/`highpass` wrappers; Hilbert/analytic
(`hilbert_transform`, `analytic_signal`, `envelope`, `env_inst`,
`instantaneous_phase`/`_frequency`); `tsconv` (basis/polynomial/time-series conv);
Lomb–Scargle for non-uniform sampling (`lomb_scargle_interpolate`,
`lomb_scargle_periodogram`); `linear_interpolate`; `polynomial_detrend`;
`product_filter`/`product_filtfilt`; `sample_windows` (uses `jax.random`, not
`numpyro`).

### 4.8 `nitrix.geometry`

Grids & warps (`identity_grid`, `spatial_transform`(`_batched`), `resample`,
`sample_at_points`, `integrate_velocity_field`, `jacobian_displacement`/`_det`,
`spatial_gradient`, `downsample`/`upsample`/`gaussian_pyramid`); the `Interpolator`
ADT (`Linear`, `NearestNeighbour`, `Lanczos`, `CubicBSpline` — whose recursive
prefilter is the `driver` axis, §3.3 — `CatmullRomCubic`, `MultiLabel`) dispatched
by `method=`; affine/Lie chart (`rigid_exp`/`rigid_log`, `affine_exp`,
`params_to_affine_matrix`/`affine_matrix_to_params`, `angles_to_rotation_matrix`/
inverse, `fit_affine`, `apply_affine`, `affine_grid`, `make_square_affine`,
`invert_affine`, `compose_affine`); transform algebra (`compose_displacement`/
`compose_velocity`, `invert_displacement`, `field_log`, `transform_mean`/
`velocity_mean`, `fuse_transforms`, `transform_geodesic`); sphere/surface
(`spherical_conv` re-backed by `semiring_ell_*`, `spherical_geodesic_distance`,
`spectral_sphere_embedding`, `spherical_parameterize`, `surface_resample`,
`marching_cubes`, `inflate_surface`, `cartesian_to_latlong`/inverse,
`signed_spherical_areas`, `is_bijective_sphere_map`, `sphere_grid_pad/unpad_2d`).

### 4.9 `nitrix.graph`

Laplacian (multi-format: dense / ELL / SectionedELL) and `laplacian_matvec`;
degree vectors; modularity (`modularity_matrix`(`_matvec`), `coaffiliation`,
`girvan_newman_null`, `relaxed_modularity`); spectral embedding
(`laplacian_eigenmap`, `diffusion_embedding`); surface analysis
(`surface_boundary_map` on `edge_aggregate` + `eta_squared`, `mesh_watershed`
priority-flood).

### 4.10 `nitrix.metrics` — differentiable comparison kernels

**Comparison score kernels, not "losses"** (§5): similarity (`ssd`, `ncc`, `lncc`
+ `lncc_grad`(`_center`), `joint_histogram` — the onehot/scatter `driver` site —
`mutual_information` + `mi_grad`/`nmi_grad`, `correlation_ratio`); overlap (`dice`,
`jaccard`); stable-from-logits classification (`bce_with_logits`,
`cross_entropy_with_logits`, `focal_loss`); contrastive (`info_nce` — layout-
agnostic, positive of `za[i]` is `zb[i]` — `dino_cross_entropy`,
`ibot_cross_entropy`, `koleo`); plus `winsorize`, `match_histogram`. Each returns
an **unreduced** tensor by default and may expose only the flat leaf
`reduction ∈ {'none','sum','mean'}` plus the domain-mask weighted mean (§5).

### 4.11 `nitrix.register` — pairwise registration recipes

Linear (`rigid_register` SE(3) GN/LM, `affine_register` 12-DOF via `matrix_exp`,
`volreg` batched realignment, `bbr_register` boundary-based) and dense
(`diffeomorphic_demons_register` log-domain SVF, `greedy_syn_register` symmetric
SyN, `syn_pipeline`); composable ADTs (`Force`/`LNCCForce`/`DemonsForce`/
`MIForce`/`MetricForce`/`SumForce`, `Metric`/`SSD`/`LNCC`/`MI`/`CorrelationRatio`,
`Objective`/`MetricObjective`/`BoundaryObjective`, `TransformModel`/`Rigid`/
`Affine`, `CoordinateSpace`/`IndexSpace`/`WorldSpace`, `Convergence`,
`RegistrationSpec`/`Result`); field regularisers (`gradient_smoothness`,
`bending_energy`, `jacobian_folding_penalty`) as score kernels (§5). The velocity-
field Gaussian regulariser is the `register.field_smooth` `driver` site (§3.3).

### 4.12 `nitrix.numerics`

ODE integrators (`euler`, `rk4`, `odeint` — `lax.scan`, diffrax-free) and
`fixed_point_solve` (implicit-VJP); normalisation family (`intensity_normalize`,
`zscore_normalize`, `robust_zscore_normalize`, `psc_normalize`, `demean`,
`l2_normalize`, `lp_normalize`, `instance_norm`, `percentile_rescale(mask=)`);
shape/window math (`pad_to_multiple`, `crop_to_multiple`, `nonzero_bounding_box`,
`gaussian_window`, `overlap_add`); array utilities (`apply_mask`, `conform_mask`,
`broadcast_ignoring`, `complex_decompose`/`recompose`, `fold_axis`/`unfold_axes`,
`orient_and_conform`, `promote_to_rank`).

### 4.13 `nitrix.nn` — functional forward-block kernels

`scaled_dot_product_attention` (dense/windowed-bias/causal/cross, optional
`qk_norm`, fused flash path + fully-fused backward incl. learnable `d_bias`),
`selective_scan` (Mamba/S6 — the sequential/associative/chunked `driver` site,
fused chunked-cumsum path), and `layer_norm`/`group_norm`/`instance_norm` (with
the curse-of-depth `out_scale` hook). Two-tier parity (tenet 10): a bit-faithful
JAX reference + a fused Pallas path certified `pallas ≈ jax`. The fused norm
kernel was measured and deliberately **not** built (XLA wins; see
`bench/PERF_LAYER_NORM.md`).

### 4.14 `nitrix.augment` — keyed augmentation kernels

A numeric category (pure deterministic, keyed generators per tenet 1), ratified
with a substrate-composition story (§6/§9): geometric ops *compose* existing
substrate (`random_resized_crop`→`spatial_transform`, `random_affine_matrix`→
`params_to_affine_matrix`, `random_svf_displacement`→`integrate_velocity_field`),
while intensity/synthesis atoms are irreducible leaves (`gamma_contrast`,
`random_histogram_shift`, `gibbs_ringing`, `gaussian_noise`, `rician_noise`,
`gmm_label_to_image`, `simulate_bias_field`) plus `random_flip`/`random_crop`.
Augmentation *policy* (specs, registries, compose, multi-crop fan-out) stays in
ilex/bitsjax.

### 4.15 `nitrix.bias` — intensity inhomogeneity

`n4_bias_field_correction` (ITK/ANTs N4 parity), the `bias_field_correction`
dispatcher (`method='n4'|'least_squares'|'psplines'`), `bspline_approximate`
(separable cubic MBA scattered-data fit), `sharpen_histogram` (N3/N4 sharpening),
`histogram_match` (Nyúl–Udupa).

---

## 5. Score-kernel ↔ scalarisation boundary  [NORMATIVE]

The line between nitrix and `nimox`, drawn at the seam so there is one vocabulary,
not two rival ones:

- **Score kernel (nitrix)** — a pure function comparing/transforming arrays with
  *irreducible numerical content* (stable-from-logits rewrite, soft overlap,
  distance/similarity, distributional closed form, field regulariser). Canonical
  output is the **unreduced** tensor (`reduction='none'`), value → value.
- **Scalarisation (nimox)** — a higher-order combinator that *wraps* a score into
  a single training scalar (reductions, term weighting, `scheme` composition),
  function → function.

Invariants:
1. nitrix score kernels return unreduced tensors by default and MAY expose only
   the **flat, non-compositional** leaf `reduction ∈ {'none','sum','mean'}` (the
   innermost element a nimox `inner=` composition calls). nitrix does **not** own
   compositional / norm / max / softmax-self-weighted scalarisation.
2. The **one** weighted reduction nitrix owns is the **domain-mask weighted mean**
   `Σ(w·x)/Σw` — a property of *measurement* (foreground/validity masks make a
   score over background numerically meaningless), categorically distinct from
   *objective* weighting (class/term/hard-example weights = nimox).
3. Objective *structure* (view-pair layout, masked-token selection, EMA/centre
   bookkeeping, multi-term weighting) is recipe → nimox. A kernel takes structure
   as an explicit arg or exposes a structure-free core.
4. **No `loss` namespace in nitrix.** "Loss" (signed, scalarised, weighted
   objective) is a nimox concept; nitrix hosts score kernels in `metrics`,
   `stats`, and `register` regularisers.

Implementation: one `nitrix._internal.reductions.reduce(values, *, axis, weight,
reduction)` (the `'mean'`+`weight` branch is the §5.2 domain-mask mean) backs all
score kernels — no per-module `_reduce` copies.

---

## 6. Dependency contract & concern boundaries

### 6.1 nitrix may import
`jax` / `jax.numpy` / `jax.experimental.pallas`, `jaxtyping`, `numpy` (type
aliases / static host math only).

### 6.2 nitrix may **not** import
`equinox`, `quax`, `numpyro`, `scipy`, `sklearn`, `pingouin` (test-only),
`nibabel`, `templateflow`, `lytemaps`, `hypercoil`, `ilex`, `entense`, `thrux`,
`bitsjax`, `nimox`, `conveyant`, `gramform`, `paranox`, or stdlib beyond typing
needs.

### 6.3 Upstream contract
`thrux` wraps nitrix kernels in container-aware raise/lower pairs; `bitsjax`
packages ops as tensorbids operators / resolver steps; `nimox` wraps primitives
in Equinox modules and owns scalarisation (§5). `ilex`/`entense` import those, not
nitrix directly (allowed but discouraged for pure-tensor internals).

### 6.4 Structural rules (firm)
- **No new top-level subpackage without a substrate-composition story** (no
  classifiers, no models, no application packages). `augment` qualifies as a
  numeric category (§4.14); it is not a precedent for model packages.
- **No PyTree-of-arrays / module classes** — `NamedTuple` or frozen dataclass
  only.
- **No "message-passing" base class.** Graph/mesh ops are reductions over ELL.
- **No hardware-specific code paths** beyond the `pallas-cuda` / `jax` pair.
- **Prefer a keyword over forking a function.**

---

## 7. Migration (historical)

The v0→v1 source-by-source port (from `hypercoil` / `ilex` / legacy `nitrix`) is
complete; the detailed action list lives in `MIGRATION.md` and git history and is
no longer normative. Concern boundaries (§6) govern any future port.

---

## 8. Testing & validation

- pytest with pinned numerical references (scipy / statsmodels / sklearn /
  pingouin) under `tests/`, never runtime deps.
- **Golden corpus** (`tests/golden/*.npz`, `tests/tolerance.toml`, loader
  `tests/_golden.py`, regen `tools/regen_golden.py`): reference-vs-golden per
  `(op, dtype)`, loosened per `(op, dtype, backend)` for the fused path.
- **Backend parity:** `pallas-cuda ≈ jax` to the pinned tolerance; both paths run
  in CI (the Pallas path needs a GPU runner; CPU is the correctness floor under
  `JAX_PLATFORMS=cpu`). No TPU tests (two axes, not three).
- **Driver cross-variant contract:** every registered divergent op asserts
  `variant ≈ canonical` within the registry tolerance, read directly from
  `divergent_ops()` (`tests/test_reproducible_dispatch_contract.py`).
- **Completeness guards:** the op-matrix guard (`tools/op_matrix.py` +
  `tests/test_op_matrix_completeness.py`) and the reproducible-dispatch guard
  (`tests/test_reproducible_dispatch_guard.py`, which fails CI on a new ungoverned
  `default_backend_is_gpu()` flip or an unregistered/orphan driver op).
- **Hypothesis** property tests for the marquee ops (semiring identity/associativity,
  morphological idempotence, interpolation invariants, …).

---

## 9. Primitive admission & evolution

The surface grows consumer-first; admission is gated, not ad-hoc.

**Admission rule.** A symbol is admitted iff it has **irreducible numerical /
structural content** *or* is a named member of a **coherent vocabulary family**
(discoverability + convention). Excluded when it is *(trivial elementwise op) ∘
(reduction)* with no content — e.g. `mse`/`l1` stay out (they are
`scalarise(square|abs(a−b))` in nimox); `gamma_contrast`, the noise generators,
and the `augment.geometric` family stay in.

**Graduation gate** (formerly `SPEC_UPDATE_v0.3 §13`). A candidate primitive
(catalogued live in `docs/feature-requests/`, where the former `SPEC_UPDATE_v0.3
§12` brainstorm catalogue now lives) graduates to sprint scope only when **all**
hold: (1) a concrete named consumer is
blocked/workaround-laden without it; (2) substrate composition is verified (no new
kernel, no parallel API); (3) the separation-of-concerns invariant holds; (4)
effort fits the time budget (XS/S rides a consumer sprint; M needs a slot; L needs
SPEC review). Shipped deviations are logged with consumer + composition story; the
acceptance test is the invariant, not advance declaration.

---

## 10. Status & open questions

**Genuinely open:**
1. **Ampere ELL Triton-vs-XLA gate.** If Triton gather on Ampere underperforms
   `jnp.take_along_axis` + reduction by > 2×, the ELL kernel ships JAX-default with
   Triton opt-in. (Benchmark before committing Triton as the ELL default.)
2. **TPU support** — architecturally compatible, blocked on dev access; post-1.0.
3. **Kernel-registry exposure** — should `linalg.kernel` expose raw kernels for
   `thrux` to wrap, or only high-level ops? (Blocks the thrux contract.)
4. **Tensor-core fast path** — is a `backend='tensor_core'` real-semiring
   specialisation worth maintaining, or stay pure-Pallas?
5. **SPEC §2 tenet text** — the reproducible-dispatch tenet (tenet 9) is drafted
   here; fold the final wording into any downstream SPEC mirror as needed.

**Resolved** (historical, see git history): semiring representation (Monoid +
Semigroup, pytree carry); sparse format (ELL + SectionedELL, no BCOO); permutohedral
(retired for bounded bilateral); morphology placement (own subpackage); LME scope
(voxelwise CORE); the score-kernel ↔ scalarisation boundary (§5); `augment`
ratification; backwards compat (no legacy users — break freely).

---

## 11. Provenance map (consolidation)

This document folds in the former incremental specs; in-code `SPEC_UPDATE §x`
citations resolve here:

| Former location | Now |
|---|---|
| `SPEC_UPDATE` §2.7 loud fallbacks | §2 tenet 7, §3.2 |
| `SPEC_UPDATE` §2.8 golden corpus | §2 tenet 8, §8 |
| `SPEC_UPDATE` §3.1 strict/relaxed semiring + differentiability | §4.1 |
| `SPEC_UPDATE` §3.2 sectioned ELL | §4.2 |
| `SPEC_UPDATE` §3.3 smoothing tiers | §4.4 |
| `SPEC_UPDATE` §3.4 morphology split (gather-backed median) | §4.3 |
| `SPEC_UPDATE` §7.2 backend selection | §3.2 |
| `SPEC_UPDATE_v0.2` §1.1 hardware scope / §2.3 / §7.2 Ampere | §1.1, §2 tenet 3, §3.2 |
| `SPEC_UPDATE_v0.3` §10.A edge-aggregate / icosphere hierarchy / pooling / LME→CORE | §4.1, §4.2, §4.3, §4.6 |
| `SPEC_UPDATE_v0.3` §12 candidate catalogue / §13 gate / §14 out-of-scope | `docs/feature-requests/`, §9, §6.4 |
| `SPEC_UPDATE_v0.4` §3.3 permutohedral retired / FeatureMetric | §4.4 |
| `SPEC_UPDATE_v0.5` §1 score-kernel ↔ scalarisation | §5 |
| `SPEC_UPDATE_v0.5` §2 keyed generators / §3 augment / §3.1 admission rule | §2 tenet 1, §4.14, §9 |
| reproducible-dispatch principle (2026-06-24) | §2 tenet 9, §3.3, §8 |
