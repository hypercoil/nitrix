# nitrix — Specification (draft v0)

> **Status.** Architectural spec — first pass. Hand-off target: Claude app, downstream
> implementation planning.
> **Scope.** This document describes *what* nitrix is, what belongs in it, what does not,
> and the contract it offers upstream libraries (`thrux`, `bitsjax`, `nimox`) and downstream
> consumers (`ilex`, `entense`). Implementation pseudo-code is intentionally absent.

---

## 1. Charter

nitrix is the **lowest-level numerical substrate** of the diffprog neuroimaging ecosystem.
It is a pure-numeric, all-JAX library: every public symbol takes JAX arrays and returns JAX
arrays. nitrix has no knowledge of image containers, sidecar metadata, BIDS, filesystems,
training loops, or PyTree modules. Those concerns belong to other libraries that *depend on
nitrix*.

The long-term vision is a substrate of differentiable numerical primitives sufficient to
build software in the class of FSL / FreeSurfer / ANTs / AFNI on GPU-accelerated hardware.
The shorter-term vision is the union of:

1. **Marquee items** (§3): keops-like semiring ops; sparsity for brain geometries;
   multi-criterion smoothing; LME (stretch); essentials for the rest of the ecosystem.
2. **Foundational primitives** (§4) needed by `ilex` models and `entense` / `bitsjax`
   transforms.

### Non-goals

- No NIfTI / GIfTI / CIfTI I/O — that is `thrux`.
- No transform / pipeline / dataset abstractions — that is `bitsjax` and `entense`.
- No Equinox modules — those are `nimox`.
- No template / atlas registration as user-facing API (low-level primitives only — atlas
  data structures and template-aware operations live in `thrux`).

---

## 2. Design tenets

1. **Pure functional.** Every public symbol is a function `(Array, …) -> Array`. No state,
   no PyTrees in the public API. (PyTree-shaped configs are acceptable as keyword args.)
2. **All differentiable.** Subgradients are explicit where appropriate; custom VJPs are
   registered where numerical stability or efficiency requires it (precedent:
   `linalg/matrix.py` `sym2vec_*` rules in the existing codebase).
3. **JAX + Pallas, with fallbacks.** Hardware-aware Pallas kernels (NVIDIA / TPU) for the
   marquee ops; pure JAX fallbacks always present. Backend selection deterministic and
   user-overridable.
4. **Typed at boundaries.** `jaxtyping.Array` / `Float[Array, "..."]` annotations on all
   public functions. No bare `Array | NDArray` unions.
5. **No transitive heavyweight deps.** `nitrix` may import only `jax`, `jaxtyping`,
   `numpy`. `scipy`, `nibabel`, `numpyro`, `equinox`, etc. are NOT permitted at runtime
   import. Test deps are scoped to `tests/`.
6. **Stable kernels, breakable APIs.** Until 1.0, the API is mutable, but kernel output
   must be numerically reproducible across releases (test against pinned references).

---

## 3. Marquee subsystems

Each marquee item has its own subpackage with a clear surface. Status flags below describe
*intended* state at first GA; current legacy / partial state lives in §6 (Migration).

### 3.1 `nitrix.semiring` — KeOps-style reductions  [CORE]

Arbitrary-algebra reductions over matmul, convolution, and ELL-sparse adjacency contractions.
The full target surface includes the strict semirings (sum-product, tropical max-plus /
min-plus, log-sum-exp, Boolean) **and** "semiring-analogous" algebras whose `(*)` is not
strictly associative — most importantly the **Euclidean** algebra
(`sqrt ∘ Σ ∘ (a − b)²`) and its relatives. These power both linear-algebraic reductions
*and* distance-driven operations (k-NN, neighbourhood smoothing, geodesic propagation).
This generality is a deliberate design bet: a single performant kernel substrate covers
matmul, convolution, distance, graph algebra, and morphology.

#### Algebra representation

The algebra is decomposed into a small pair of Protocols (informed by the
`semiring_gemm.py` brainstorm; see §11):

- **`Semigroup`** — a single `combine(a, b)` broadcasting binary op (the `(*)` step).
- **`Monoid[S]`** — `(init, update, merge, finalize)` over a pytree state `S`. The pytree
  state is critical: it lets numerically-stable online reductions (e.g. log-sum-exp's
  `(running_max, sum_exp)` carry; Welford-style variance; running-norm Euclidean) thread
  state through the K loop without materialising intermediates.
- **`Semiring`** — the frozen `(monoid, semigroup, name)` triple.

Pre-built algebras: `real`, `tropical_max_plus`, `tropical_min_plus`, `log`, `euclidean`,
`boolean`. User-defined Monoids and Semigroups compose freely.

#### Public surface (illustrative signatures)

```python
def semiring_matmul(
    A: Num[Array, "... m k"],
    B: Num[Array, "... k n"],
    *, semiring: Semiring = REAL,
    backend: Backend = "auto",
) -> Num[Array, "... m n"]
```

```python
def semiring_conv(
    x: Num[Array, "... *spatial c_in"],
    k: Num[Array, "... *kspatial c_in c_out"],
    *, semiring: Semiring = REAL,
    stride: ..., padding: ..., dilation: ...,
    backend: Backend = "auto",
) -> Num[Array, "... *spatial c_out"]
```

```python
def semiring_ell_matmul(           # the central op for brain-geometry workloads
    values: Num[Array, "m k_max"], # ELL values (m rows × at most k_max neighbours)
    indices: Int[Array, "m k_max"],
    B: Num[Array, "n n_cols"],
    *, semiring: Semiring = REAL,
    n_rows: int,                   # outer dim of the implicit M×N sparse matrix
    backend: Backend = "auto",
) -> Num[Array, "m n_cols"]
```

#### Kernel strategy

- **KeOps-style streaming.** The K-block iteration folds rank-1 outer combines directly
  into the (BM, BN) accumulator; the (BM, BK, BN) value tensor is never materialised. This
  keeps peak on-chip memory at `O(BM·BN + BM·BK + BK·BN)` and makes algebras whose `(*)`
  is non-multiplicative (Euclidean, tropical, log) practical at scale.
- **No tensor-core / `dot` primitive.** Tensor cores assume `(*) = ×`. We issue plain
  CUDA-core / TPU SIMD ops so the same kernel codegens across all algebras. For the real
  semiring on hardware that can use tensor cores, an optional `backend="tensor_core"`
  specialisation falls back to `jnp.matmul`; this is a thin fast path, not the primary
  surface.
- **Pytree accumulator.** Monoid state is a pytree; `lax.fori_loop` over K threads the
  state through. `finalize` applies once at the end (e.g. Euclidean's `sqrt`, log's
  `m + log(s)`).
- **No BCOO.** ELL is the primary sparse format (see §3.2). The ELL kernel walks the
  per-row neighbour list via gather + the same Monoid/Semigroup glue — Pallas-friendly,
  no jaxlib-sparse adversarial path.

#### Backends

`pallas-cuda` (default on NVIDIA) and `pallas-tpu` (default on TPU); JAX fallback (built
on `lax.fori_loop` + the same `reference_semiring_gemm`-style algebra plumbing) for CPU
and for shapes / algebras the Pallas builder cannot tile cleanly. Backend selection per
§7.2.

#### Why this is a foundation, not a feature

The same machinery underpins: graph path algebras (tropical), softmax / attention-style
reductions (log), k-NN and bilateral search (Euclidean as the inner step of a
nearest-neighbour scan), morphological opening / closing (tropical conv), spherical
convolution on mesh sparsity (real or weighted on ELL adjacency), and binary connectivity
analysis (Boolean). Subsequent §3 items (sparse, smoothing, morphology) are largely
*specialisations* of this surface, not parallel implementations.

### 3.2 `nitrix.sparse` — geometry-aware sparsity  [CORE]

Sparsity primitives for the structures we actually care about in neuroimaging. **ELL is
the primary format** — brain-geometry adjacencies (mesh k-rings, deformed icospheres,
distance-thresholded neighbourhoods, atlas parcel members) are naturally
fixed-degree-per-row, so ELL captures them losslessly with zero padding overhead in the
common case and a single padded-row dimension in the worst.

Submodules:

- `sparse.ell` — ELL format primitives (construction, reshape, gather/scatter, padding
  with semiring identity, batch broadcasting). The format is a thin pair of arrays
  `(values: [m, k_max], indices: [m, k_max])` plus a row-count and an algebra-identity for
  pad positions. **No jax-sparse BCOO.** The historical BCOO-based path in
  `hypercoil/functional/sparse.py` has been a persistent friction surface against the
  XLA / Pallas boundary; we implement on plain dense arrays + gather primitives so the
  kernels integrate cleanly.
- `sparse.grid` — regular-grid sparsity (low-bandwidth band matrices, stencil ops). The
  thin specialisation of ELL where every row has the same neighbour offsets.
- `sparse.mesh` — icosphere / deformed-icosphere mesh sparsity built atop `sparse.ell`:
  k-ring adjacency, sparse Laplacians, geodesic neighbourhoods.

The semiring kernels in §3.1 operate directly on ELL representations
(`semiring_ell_matmul`, `semiring_ell_conv`). Treat ELL + semiring as a single
co-designed pair: they are the substrate for both linear ops on graph adjacency *and*
distance-driven ops over k-NN graphs.

### 3.3 `nitrix.smoothing` — multi-criterion / hyper-spatial  [CORE]

Edge-preserving, multi-channel, multi-domain smoothing centred on the **permutohedral
lattice**. Multi-domain means the feature space can mix space, intensity, gradient
direction, time, etc. — permutohedral handles arbitrary `d_f` in expected linear time and
subsumes the bilateral / trilateral / cross-bilateral special cases that SUSAN, Gaussian,
and friends individually cover.

```python
def permutohedral_lattice(
    values: Float[Array, "n d_v"],
    features: Float[Array, "n d_f"],
    *, sigma_features: Float[Array, "d_f"],
) -> Float[Array, "n d_v"]
```

Plus a baseline `gaussian` (for cases where edge preservation is not wanted) and a
`bilateral` thin wrapper (the canonical `d_f = d_space + d_intensity` configuration).

SUSAN is intentionally **not** part of the public surface: its USAN/edge-preservation
behaviour is recovered by feeding intensity (and, optionally, intensity gradient) into the
permutohedral feature space. Skipping SUSAN cuts implementation scope without giving up
capability.

### 3.4 `nitrix.morphology` — built atop semiring  [CORE]

Binary and grayscale erode / dilate / open / close, distance transforms. Implemented as
specialisations of `semiring.conv` with `TROPICAL_MIN` / `TROPICAL_MAX`. Listed separately
because it is a major user-facing surface, not because it has independent implementation.

### 3.5 `nitrix.stats.lme` — voxelwise LME  [STRETCH]

Efficient voxelwise linear mixed-effects fits. Out of scope for first GA; the spec reserves
the namespace and documents the API shape so downstream consumers can plan around it.

```python
def voxelwise_lme(
    Y: Float[Array, "n_obs *voxels"],
    X: Float[Array, "n_obs p"],       # fixed effects design
    Z: Float[Array, "n_obs q"],       # random effects design
    groups: Int[Array, "n_obs"],
    *, method: Literal["reml", "ml"] = "reml",
) -> LMEResult                          # NamedTuple of arrays — NOT a PyTree module
```

Open: solver choice (closed-form on small q vs iterative); whether to expose Henderson's
mixed-model equations as a separate primitive.

---

## 4. Foundational primitives  [CORE]

These are the lower-glamour but high-traffic operations that ilex models, entense
transforms, nimox modules, and bitsjax resolvers actually call. Most exist (in some form)
in the current nitrix or in hypercoil; the migration map (§6) details origins.

### 4.1 `nitrix.linalg`

- `matrix.py` — symmetric / `sym2vec` / `vec2sym` / `squareform` / `toeplitz` /
  `toeplitz_2d` / `delete_diagonal` / `fill_diagonal` / `diag_embed` /
  `recondition_eigenspaces`. (Existing nitrix is healthy here.)
- `spd.py` — SPD manifold: `symexp`, `symlog`, `symmap`, `symsqrt`; tangent-space project /
  unproject (`BatchTangentProject` numerics, not the module). **Stability rewrite required**
  — current hypercoil implementation is flagged numerically unstable.
- `kernel.py` — parameterised kernels (linear, polynomial, Gaussian, RBF, Laplace) with
  single-dispatch over input matrix type. Includes initialisation helpers (Laplace,
  Toeplitz) folded in from legacy `hypercoil/init/`.
- `residual.py` — `residualise` (L2-regularised least squares). Existing; keep, fix the
  off-diagonal-weight gap from `covariance.py` while we are here.

### 4.2 `nitrix.stats`

- `covariance.py` — `cov`, `corr`, `partialcov`, `partialcorr`, `pairedcov`, `pairedcorr`,
  `conditionalcov`, `conditionalcorr`, `precision`, `corrnorm`. **Bug fix required**:
  non-diagonal weight matrices currently silently produce wrong answers post-JIT
  (`covariance.py:719–726`). Either implement properly or raise unambiguously.
- `fourier.py` — `product_filter`, `product_filtfilt`, `analytic_signal`,
  `hilbert_transform`, `envelope`, `instantaneous_phase`, `instantaneous_frequency`.
  (Existing; keep.)
- `lme.py` — STRETCH (§3.5).

### 4.3 `nitrix.signal`

- `window.py` — `sample_windows` (existing). **Drop `numpyro` dependency** — use
  `jax.random` directly for multinomial sampling.
- `filter.py` — FIR / IIR / frequency-domain filters. Pure-numeric implementations,
  separated from the module-shaped `nn/freqfilter.py` and `nn/iirfilter.py` in hypercoil.
- `tsconv.py` — basis / polynomial / time-series convolutions (port of
  `hypercoil/functional/tsconv.py`).
- `interpolate.py` — spectral / linear / hybrid interpolation for missing data (extract
  numerics from `hypercoil/functional/interpolate.py`).
- `normalize.py` — `intensity_normalize` (min / p99 / clip) and friends. Migrates out of
  `ilex/models/synthstrip/preprocessing.py`.

### 4.4 `nitrix.geometry`

- `grid.py` — `cmass_regular_grid`, `identity_grid`, `spatial_transform`, `vec_int`,
  `rescale` (migrates `ilex/models/voxelmorph/_numerical.py`; folds in existing nitrix
  `geom.py` grid bits).
- `sphere.py` — icosphere generation, spherical geodesics, sphere-to-normals / latlong,
  spherical convolution. Existing `geom.py` + `hypercoil/functional/sphere.py`. Spherical
  conv is **re-backed** by `semiring.conv` over mesh sparsity (§3.1–3.2) — drop the legacy
  O(N²) inner loop.
- `coords.py` — coordinate utilities (`cmass_coor`, `cmass_reference_displacement_*`,
  spherical ↔ Cartesian).
- `metrictensor.py` — metric-tensor primitives (port from hypercoil).

### 4.5 `nitrix.graph`

- `laplacian.py` — graph / modularity Laplacian, modularity matrix, Girvan–Newman null,
  coaffiliation.
- `connectopy.py` — eigenmaps, diffusion maps (extract from `hypercoil/functional/`,
  decouple from brainspace).
- `community.py` — community / relaxed-modularity numerics.

### 4.6 `nitrix.numerics` — uncategorised array ops

- `tensor_ops.py` — `transpose`, `reshape_to`, `transpose_tf_conv_kernel`,
  `broadcast_bias`, etc. The pure-array half of `ilex/core/adapters.py` (the adapter
  *registry* stays in ilex).

### 4.7 `nitrix._internal`

Internal utilities (axis manipulation, mask helpers, complex-number decompose / recompose,
docstring formatters). Existing; healthy.

---

## 5. Dependency contract

### 5.1 nitrix may import

- `jax`, `jax.numpy`, `jax.experimental.pallas`
- `jaxtyping`
- `numpy` (for type aliases only)

### 5.2 nitrix may **not** import

- `equinox`, `quax` — modules are upstream concerns
- `numpyro` — currently violated by `window.py`; fix on migration
- `scipy`, `sklearn`, `pingouin` — test-only
- `nibabel`, `templateflow`, `lytemaps` — container-level, lives in `thrux`
- `hypercoil`, `ilex`, `entense`, `thrux`, `bitsjax`, `nimox`, `conveyant`, `gramform`,
  `paranox`
- the standard library beyond what is needed for typing

### 5.3 Upstream contract

- `thrux` imports nitrix and wraps its kernels in container-aware raise / lower pairs.
- `bitsjax` imports nitrix (and thrux) and packages ops as tensorbids operators / resolver
  steps.
- `nimox` imports nitrix and wraps primitives in Equinox PyTree modules.
- `ilex` and `entense` import the above, not nitrix directly (except where a model's
  internal numerics are pure-tensor; this is allowed but discouraged — prefer going through
  nimox / bitsjax).

---

## 6. Migration map  (module-level)

The detailed source-by-source action list lives in `MIGRATION.md`. Summary by destination:

| Destination subpkg | Sources |
|---|---|
| `nitrix.semiring` | NEW — design from the `semiring_gemm.py` brainstorm (§11); no legacy port |
| `nitrix.sparse` | NEW. **Do not port** `hypercoil/functional/sparse.py` — historical BCOO friction. Re-implement on plain JAX gather + dense arrays. ELL primary; grid / mesh as ELL specialisations |
| `nitrix.smoothing` | NEW. Gaussian baseline can fold in the kernel from existing `geom.py`; permutohedral is clean-room. SUSAN dropped |
| `nitrix.morphology` | NEW — built atop semiring |
| `nitrix.linalg` | existing nitrix `matrix.py`, `residual.py` + hypercoil `functional/{matrix, kernel, symmap, semidefinite, metrictensor}` + `init/{laplace, toeplitz, semidefinite}` |
| `nitrix.stats` | existing `covariance.py`, `fourier.py` + hypercoil `functional/cov` consolidation; LME is NEW (stretch) |
| `nitrix.signal` | existing `window.py` (de-numpyro'd) + hypercoil `functional/{tsconv, interpolate, fourier-bits}` + ilex `models/synthstrip/preprocessing.py` `intensity_normalize` |
| `nitrix.geometry` | existing `geom.py` (split) + hypercoil `functional/{sphere, cmass, metrictensor}` + ilex `models/voxelmorph/_numerical.py` |
| `nitrix.graph` | hypercoil `functional/{graph, connectopy, cmass}` |
| `nitrix.numerics` | ilex `core/adapters.py` pure-array half (≈ lines 150–250) |
| entense `instance.py` impls (`polynomial_detrend_p`, `confound_regression_p`) | merge into `nitrix.signal.filter` / `nitrix.linalg.residual` |

---

## 7. Performance & kernels

### 7.1 Pallas surface

Pallas kernels are an **implementation detail** behind the public API. The user-facing
function (`semiring_matmul`, `permutohedral_lattice`, etc.) chooses a backend; the kernel
file is private (`_kernels/`).

### 7.2 Backend selection

Three-level resolution: explicit `backend=` keyword → env var (`NITRIX_BACKEND`) →
auto-detect from `jax.default_backend()`. Auto-detect prefers `pallas-cuda` on NVIDIA,
`pallas-tpu` on TPU, `jax` fallback otherwise.

### 7.3 Differentiability

Every kernel has a JAX-side gradient. Pallas kernels register a `jax.custom_vjp` whose
backward is either (a) a paired Pallas kernel, or (b) a JAX fallback. Tests assert
forward / backward numerical agreement across backends (tolerance pinned per dtype).

---

## 8. Testing & validation

- pytest, pinned numerical references (pingouin, scipy.ndimage, sklearn, communities) live
  in `tests/` and are not runtime deps.
- Add hypothesis-based property tests for the marquee ops (associativity / identity for
  semirings; idempotence for morphological close-after-open at large kernels; etc.).
- Add backend-parity tests: same op via `pallas-cuda` and `jax` fallback must agree to
  pinned tolerance.
- Add the **JIT-trap regression** test for `covariance` with non-diagonal weights — this is
  a known bug, not a feature.

---

## 9. Open questions

Resolved (during this drafting pass):

- ~~Semiring representation~~ — adopt the Monoid + Semigroup Protocol pair with pytree
  accumulator state (per `semiring_gemm.py` brainstorm).
- ~~Sparse format unification~~ — ELL primary, dense JAX gather under the hood; no BCOO.
- ~~Backwards compat~~ — no legacy users; we break freely.

Deferred:

1. **Morphology placement.** Independent subpackage (current spec) or buried inside
   `semiring` as thin convenience wrappers?
2. **LME scope.** Voxelwise-independent (cheapest, plenty useful) vs voxelwise-with-
   spatial-regularisation (much harder, much more useful)?
3. **Kernel registry exposure.** Should `linalg.kernel` expose raw kernels for `thrux` to
   wrap, or only high-level ops?
4. **`numerics.reshape` vs `numerics.tensor_ops`.** Submodule split-out granularity inside
   the small "uncategorised" area.
5. **lytemaps.** Does nitrix subsume lytemaps's JAX-compilable bits, or does lytemaps
   remain orthogonal (high-level wrappers around nitrix)? Recommended: orthogonal for now;
   revisit when nitrix.geometry.sphere matures.
6. **Tensor-core fast path.** For the real semiring on hardware that can use tensor cores,
   does the `backend="tensor_core"` specialisation pay off enough to maintain? Or stay
   pure-Pallas everywhere for simplicity?

---

## 10. Success criteria (first GA)

- All §4 foundational primitives implemented, tested, JAX + Pallas where applicable.
- `semiring.{matmul, conv, ell_matmul}` shipped with `real`, `tropical_max_plus`,
  `tropical_min_plus`, `log`, `euclidean`, `boolean` built-in algebras, and a documented
  user-extension path (custom `Monoid` / `Semigroup`).
- KeOps-style streaming kernel passes parity tests against `reference_semiring_gemm` and
  against naive broadcast formulations, with identity propagation (e.g. `-inf` in
  tropical / log) and numerical stability (log with large magnitudes) regressions covered.
- `sparse.{ell, grid, mesh}` shipped; `geometry.sphere.spherical_conv` re-backed by
  `semiring_ell_conv`. No `jax.experimental.sparse` BCOO dependency.
- `smoothing.gaussian`, `smoothing.bilateral`, `smoothing.permutohedral_lattice` shipped
  and tested against reference implementations. SUSAN intentionally absent.
- `morphology.{binary, grayscale}` shipped atop tropical-semiring conv.
- All known bugs (covariance non-diag weights; `numpyro` import; 2D-only `spatial_conv`)
  resolved before any of the above land.
- Downstream blockers for `ilex`, `entense`, `thrux`, `bitsjax`, `nimox` cleared.
- `lme` namespace reserved; no implementation required.

---

## 11. Source artefacts referenced

- **Semiring brainstorm (prior session, untested stubs, does not match house style):**
  - `_refstubs/semiring_gemm.py` — `Monoid` / `Semigroup` / `Semiring` Protocols,
    pre-built algebras (`real`, `tropical_*`, `log`, `euclidean`), KeOps-style Pallas
    kernel builder, pure-JAX reference implementation.
  - `_refstubs/test_semiring_gemm.py` — parity tests against naive broadcast and the
    pure-JAX reference; identity-propagation and numerical-stability regressions.
  Treat as design input, not as a port target. Reimplement in the diffprog house style;
  preserve the Protocol shape, pytree-state pattern, and KeOps streaming idea.
