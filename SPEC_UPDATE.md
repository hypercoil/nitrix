# nitrix — Specification update (draft v0 → v0.1)

> **Status.** Patch to SPEC.md draft v0 integrating review and clarifying-question
> responses. Apply in-place; sections not listed are unchanged.
> **Summary of changes.** Strict vs. relaxed semiring split; differentiability story
> pinned to a fixed vocabulary; permutohedral demoted to *target* with explicit tripwire
> and two unconditional fallbacks; sectioned-ELL promoted to CORE; median filter added as
> a gather-based op outside the semiring abstraction; backend-fallback observability and
> golden-corpus reproducibility pinned; Open Questions re-ranked.

---

## §2. Design tenets — additions

Add as 2.7 and 2.8:

7. **Loud fallbacks.** Backend fallback (Pallas → JAX, requested algebra → relaxed
   variant, unsupported dtype, etc.) is observable: a structured warning is emitted on
   first occurrence per `(function, shape-signature, dtype, backend)` tuple, suppressible
   per process via `NITRIX_SILENCE_FALLBACK=1`. Silent perf regressions are a bug.

8. **Reproducibility via golden corpus.** §2.6 ("Stable kernels") is operationalised as:
   each kernel × dtype × algebra triple has a checked-in reference array under
   `tests/golden/`. Tolerance is a `(dtype, backend)` matrix pinned per release. Changing
   a tolerance entry is a public API change and goes in CHANGELOG. The JIT-trap covariance
   regression and the tropical/log identity-propagation regressions are members of this
   corpus from day one.

---

## §3.1 `nitrix.semiring` — algebra representation, replace the "Algebra representation"
subsection with:

#### Algebra representation

We expose **two** Protocols, with a strict-subtype relationship:

- **`Semiring`** — `(monoid, binary_op, name, identity)`. The relaxed type. Makes no
  associativity guarantee on `binary_op`. Closed under the operations downstream
  consumers most commonly want. This is the default user-facing type and what 90% of
  user code annotates.
- **`StrictSemiring <: Semiring`** — additionally asserts associativity and (where the
  algebra requires it) distributivity of `binary_op` over `monoid.combine`. The
  strict-subtype boundary is checked structurally at construction (a `strict=True` flag
  on the constructor; users opt in).

Public functions whose correctness depends on free reassociation of the K-loop (e.g.
hierarchical / tree-reduction kernels, multi-stage reductions across sharded devices)
annotate `semiring: StrictSemiring`. The streaming `semiring_matmul` / `semiring_conv` /
`semiring_ell_matmul` kernels accept the relaxed `Semiring` since they fix the
reduction order.

Pre-built algebras and their strictness:

| Algebra | Type | Associative `binary_op` | Differentiable |
|---|---|---|---|
| `REAL` | `StrictSemiring` | yes | yes |
| `LOG` | `StrictSemiring` | yes | yes |
| `TROPICAL_MAX_PLUS` | `StrictSemiring` | yes | subgradient |
| `TROPICAL_MIN_PLUS` | `StrictSemiring` | yes | subgradient |
| `BOOLEAN` | `StrictSemiring` | yes | no (documented) |
| `EUCLIDEAN` | `Semiring` | no (rank-1 fold) | yes |

The relaxed/strict split replaces the earlier flat `Semiring` Protocol. Consumer
functions document their requirement at the signature site; the type system carries it.

---

## §3.1 `nitrix.semiring` — differentiability subsection, NEW

Insert after the "Kernel strategy" subsection:

#### Differentiability vocabulary

Built-in algebras ship with hand-derived backward kernels in the same algebra family.
The backward is implemented as a **paired streaming kernel**: same `(BM, BK, BN)`
tiling, same Pallas / JAX backend selection, same pytree-state pattern. The vocabulary
is fixed:

| Algebra | Forward | Backward kernel |
|---|---|---|
| `REAL` | inner product | transpose-matmul (same kernel, swapped operands) |
| `LOG` | log-sum-exp | softmax-weighted gradient (recompute softmax in K loop; no materialised softmax tensor) |
| `TROPICAL_MAX_PLUS` | max over sum | argmax-gather (subgradient; one-hot through the max) |
| `TROPICAL_MIN_PLUS` | min over sum | argmin-gather (subgradient) |
| `EUCLIDEAN` | √∑(a−b)² | normalised-difference gradient with √ singularity guard |
| `BOOLEAN` | OR over AND | not differentiable; backward raises if reached |

Each backward is registered via `jax.custom_vjp`. Numerical-stability tests for the
backward live in `tests/golden/` alongside the forward; for `LOG` and `EUCLIDEAN` these
explicitly cover the large-magnitude and near-zero regimes respectively.

**User-defined algebras.** Users supplying a custom `Semiring` are forward-only by
default. To make a custom algebra differentiable, the user wraps the `semiring_*` call
in their own `jax.custom_vjp` and supplies a backward. The spec does *not* attempt
keops-style symbolic autodiff over the formula DAG; the implementation cost is too high
for the marginal user-base who would benefit.

---

## §3.2 `nitrix.sparse` — replace the ELL caveat paragraph

ELL is the primary format. It captures fixed-degree-per-row geometries (mesh k-rings,
deformed icospheres, regular-grid stencils) losslessly with zero padding overhead.
**Variable-degree geometries are handled by sectioned ELL** (`sparse.ell.sectioned`,
see below), not by padding everything to the worst-case row.

##### Degree-variance regime

The naive `[m, k_max]` ELL layout is only memory-efficient when `k_max ≈ median(k)`.
Two cases break this assumption and both are explicit nitrix targets:

- **Distance-thresholded neighbourhoods** in irregular point clouds: worst-case `k_max`
  can be 10–100× the median row.
- **Atlas parcel members**: parcel sizes vary 1–2 orders of magnitude across the brain.

For these, padding to `k_max` reintroduces the memory cliff that motivated dropping
BCOO. **Sectioned ELL** addresses this: rows are bucketed by `ceil(log2(k))` (or a
user-supplied bucketing), and the ELL kernel runs once per bucket with the bucket's
local `k_max`. Outputs are scattered back to the original row ordering. This is a
Python-level wrapper around `semiring_ell_matmul`, ~150 lines, no new kernel code.

##### Submodules

- `sparse.ell` — ELL format primitives (construction, reshape, gather/scatter, padding
  with semiring identity, batch broadcasting). The format is a thin pair of arrays
  `(values: [m, k_max], indices: [m, k_max])` plus a row-count and an algebra-identity
  for pad positions.
- `sparse.ell.sectioned` — bucketed-row ELL for variable-degree adjacencies. **CORE,
  not stretch:** the implementation is mechanical and the failure mode it prevents is
  silent OOM on the variable-degree workloads nitrix explicitly targets.
- `sparse.grid` — regular-grid sparsity (low-bandwidth band matrices, stencil ops). A
  thin specialisation of ELL where every row has the same neighbour offsets.
- `sparse.mesh` — icosphere / deformed-icosphere mesh sparsity built atop `sparse.ell`:
  k-ring adjacency, sparse Laplacians, geodesic neighbourhoods.

No `jax.experimental.sparse` BCOO dependency anywhere.

---

## §3.3 `nitrix.smoothing` — replace in full

Edge-preserving, multi-channel, multi-domain smoothing for neuroimaging workloads where
the feature space mixes space, intensity, gradient direction, additional imaging
modalities (e.g. T1 + T2 + diffusion-derived scalars), and functionally-derived signals.

Three tiers, all of which ship at first GA:

##### `gaussian` — unconditional baseline

Separable Gaussian. Pure JAX. Ships unconditionally. For cases where edge preservation
is not wanted.

##### `bilateral_gaussian` — unconditional, direct N-body

Direct N-body bilateral with arbitrary feature dimensionality `d_f`. For each output
voxel, gather the feature-space neighbourhood and weight by a Gaussian over `d_f`
features. **Quadratic in spatial neighbourhood size, linear in `d_f`.** Practical for
neighbourhoods up to ~7³ voxels and `d_f ≤ 5`, which covers the multi-modal /
tissue-class use case (spatial + 1–3 intensity/modality channels). Implemented as a
`semiring_ell_matmul` over a distance-thresholded sectioned-ELL adjacency with the
Gaussian-weighted real semiring. Ships unconditionally; **this is the marquee capability
delivered at GA regardless of permutohedral risk.**

```python
def bilateral_gaussian(
    values: Float[Array, "n d_v"],
    features: Float[Array, "n d_f"],
    *, sigma_features: Float[Array, "d_f"],
    neighbourhood: int | Int[Array, "n k_max"],  # k or explicit adjacency
) -> Float[Array, "n d_v"]
```

##### `permutohedral_lattice` — target, with tripwire

Permutohedral lattice filtering: linear-time high-dimensional Gaussian filtering via
the simplicial-grid splat / blur / slice algorithm. The asymptotic win for larger
spatial extents and larger `d_f` (≥ 6).

**Pallas-pure is explicitly relaxed as a requirement.** The lattice's hash-table and
gather-heavy access pattern is precisely the regime Pallas docs flag as awkward. A
JAX-only implementation, or a JAX-with-some-Pallas-helpers hybrid, is acceptable
provided overall wall-clock and compile-time targets are met. Maintaining five
architecture-specific Pallas variants is not in scope; we will not pay that cost for
this one kernel.

**v1 stop criterion.** `permutohedral_lattice` ships at first GA if and only if all of:

1. **Parity.** Output agrees with the Adams 2010 reference at PSNR > 40 dB on the
   bilateral test set, all dtypes in the support matrix.
2. **Performance.** End-to-end smoothing of a 256³ volume with `d_f = 5` features
   completes in < 10× the wall time of an equivalent-σ separable `gaussian` on the same
   hardware.
3. **Compilation overhead.** First-call compilation < 30 s on the reference NVIDIA and
   TPU configurations; cached-recompile < 1 s.
4. **Gradient.** Backward passes finite-difference checks at the pinned per-dtype rtol;
   the gradient is not approximate.

(Numbers above are placeholders pending benchmarking on the reference hardware; pin
before GA.)

Failing any of (1)–(4), the namespace is reserved but the symbol raises
`NotImplementedError` pointing to `bilateral_gaussian` for the `d_f ≤ 5` case, and the
team revisits at 1.x. No interim "partial" shipping.

##### SUSAN emulator

```python
def susan_emulator(
    image: Float[Array, "..."], *,
    sigma_space: float,
    sigma_intensity: float,
    use_median: bool = True,
    bthresh: float | None = None,
) -> Float[Array, "..."]
```

Convenience wrapper. Internally composes `bilateral_gaussian` (or
`permutohedral_lattice` if shipped and `d_f` warrants) with `morphology.median_filter`
(§3.4) for the impulse-noise fallback that FSL's SUSAN handles via local median.
Documents the behavioural differences from FSL SUSAN: the brightness-similarity
weighting is recovered by the bilateral path; the median fallback is recovered by the
median_filter composition; the auto-flat-kernel-at-small-extents behaviour is *not*
replicated and is documented as such.

---

## §3.4 `nitrix.morphology` — replace in full

Binary and grayscale erode / dilate / open / close, distance transforms, and median
filtering.

##### Semiring-backed: erode / dilate / open / close, distance transforms

Specialisations of `semiring_conv` / `semiring_ell_conv` with `TROPICAL_MIN_PLUS` /
`TROPICAL_MAX_PLUS`. Distance transforms via the standard two-pass min-plus algorithm
on a regular-grid structuring element. Implementation cost low: thin wrappers, no new
kernel.

##### Gather-backed: median filter

```python
def median_filter(
    x: Float[Array, "... *spatial"],
    *, neighbourhood: int | Int[Array, "n k"],
) -> Float[Array, "... *spatial"]
```

**Not a semiring op.** True median requires materialising the full neighbourhood at
each output position (state size unbounded in the streaming K loop). For the
neighbourhood sizes nitrix targets (3³ = 27 voxels, mesh k-rings of O(10s)), the
materialisation is fine: implement as `gather → jnp.median`, no streaming kernel. Ships
in `morphology` (not `semiring`) precisely because it does not fit the semiring
abstraction.

The `nitrix.semiring` module documents this exclusion: median is the canonical example
of an op whose state size is unbounded in K and therefore outside the semiring
substrate.

(Approximate-median variants — quantile sketches, bucketed histograms — are not in
scope for first GA. The exact median over a small neighbourhood is what users want;
they can ask for an approximate-quantile primitive if the small-neighbourhood
assumption breaks.)

---

## §4.3 `nitrix.signal` — minor

`normalize.py` moves to `nitrix.numerics.normalize`. `intensity_normalize` is a
data-prep op, not DSP; it has no business living next to `filter`, `tsconv`,
`interpolate`. Update §6 migration map accordingly.

---

## §7.2 Backend selection — replace

Three-level resolution unchanged: explicit `backend=` keyword → env var
(`NITRIX_BACKEND`) → auto-detect from `jax.default_backend()`. Auto-detect prefers
`pallas-cuda` on NVIDIA, `pallas-tpu` on TPU, `jax` fallback otherwise.

**Fallback observability.** When the resolved backend cannot satisfy a call (Pallas
tiling fails for the given shape × algebra; user-requested backend unavailable; etc.),
nitrix falls back to the next-best backend in `[pallas-cuda | pallas-tpu, jax]`, emits
a structured warning via the standard `warnings` module with category
`NitrixBackendFallback`, and proceeds. Warnings are deduplicated per `(function,
shape-signature, dtype, backend)` per process. Set `NITRIX_SILENCE_FALLBACK=1` to
suppress. Set `NITRIX_STRICT_BACKEND=1` to convert fallback to error.

**Pallas API churn policy.** `jax.experimental.pallas` is itself experimental. nitrix
pins a minimum `jax` version per release. If a Pallas-side change breaks a kernel
between minimum-pin updates, the affected kernel falls back to JAX with the standard
warning; a release-blocking issue is filed; no kernel is silently disabled. The "stable
kernel output" tenet (§2.6) holds via the JAX fallback path during such windows.

---

## §9 Open questions — re-rank

Replace the flat deferred list with:

**Blocking implementation (resolve before code lands):**

1. Kernel registry exposure. Should `linalg.kernel` expose raw kernels for `thrux` to
   wrap, or only high-level ops? Affects the thrux contract.

**Blocking 1.0 but not 0.1:**

2. LME scope: voxelwise-independent vs voxelwise-with-spatial-regularisation. The
   returned `LMEResult` shape will likely need to differ between the two; commit before
   1.0 so downstream destructuring is stable.
3. lytemaps integration: subsume the JAX-compilable bits or stay orthogonal? Recommend
   orthogonal; revisit when `geometry.sphere` matures.

**Resolved by this update:**

- ~~Morphology placement~~ — independent subpackage (§3.4), with median filter as a
  gather-based op outside the semiring substrate.
- ~~Semiring representation: relaxed vs strict~~ — both, with `StrictSemiring <:
  Semiring`. See §3.1.
- ~~Permutohedral scope~~ — target with tripwire and unconditional fallbacks. See §3.3.

**Dropped as out of scope:**

- ~~Tensor-core fast path for real-semiring matmul~~. Users wanting tensor cores on
  real-semiring contraction call `jnp.matmul` directly. nitrix's `semiring_matmul`
  serves the algebras `jnp.matmul` can't accelerate. Documented in §3.1.
- ~~`numerics.reshape` vs `numerics.tensor_ops` granularity~~. Cosmetic; revisit if
  the namespace grows past ~6 functions.

---

## §10 Success criteria — additions

Add to the existing list:

- `bilateral_gaussian` shipped with `d_f` up to 8 supported, parity tests against a
  reference direct-N-body NumPy implementation.
- `permutohedral_lattice` *either* shipped meeting all four §3.3 tripwire criteria *or*
  the symbol raises `NotImplementedError` pointing at `bilateral_gaussian`. No
  intermediate state.
- `morphology.median_filter` shipped, parity against `scipy.ndimage.median_filter`
  within the pinned tolerance for the standard structuring elements.
- `morphology.susan_emulator` shipped, with documented behavioural deltas from FSL
  SUSAN in the docstring.
- `sparse.ell.sectioned` shipped, with golden tests demonstrating zero-memory-cliff
  behaviour on a degree-variance test case (median k = 12, max k = 240).
- Each built-in `StrictSemiring` ships with a paired backward kernel passing
  finite-difference checks at the pinned per-dtype rtol. `EUCLIDEAN` (relaxed) ships
  with backward; `BOOLEAN` ships forward-only with a documented gradient raise.
- Backend-fallback warning infrastructure (§7.2) wired into the test suite: a test
  asserts that the warning fires under a forced shape × algebra combination that
  Pallas cannot tile, and does not fire on the happy path.
- Golden corpus established under `tests/golden/` covering every (kernel, dtype,
  algebra, backend) cell shipped at GA; tolerance matrix pinned in `tests/tolerance.toml`.

---

## §6 Migration map — diff

Two rows change:

| Destination subpkg | Change |
|---|---|
| `nitrix.smoothing` | Add: `bilateral_gaussian` as primary capability, permutohedral as target. SUSAN emulator as convenience composing bilateral + median |
| `nitrix.morphology` | Add: `median_filter` as gather-based op (not semiring) |
| `nitrix.numerics` | Add: `normalize.py` (moved from `nitrix.signal`) |

---

## Notes for implementation planning

- The §3.1 strict/relaxed split is the only API-shape change that affects users writing
  custom algebras. Document migration in the 0.x → 1.0 changelog.
- §3.3's three-tier smoothing means the marquee capability ("smoothing in richer
  feature spaces") lands at GA *via `bilateral_gaussian`* regardless of permutohedral
  outcome. Permutohedral is then a performance upgrade, not a capability gate.
- §3.4's split of morphology between semiring-backed and gather-backed is the
  prototype for handling the next "almost a semiring" op that comes along: don't force
  it.
- Sectioned-ELL (§3.2) is the smallest CORE addition relative to v0; ~150 lines of
  Python over the existing ELL kernel. The cost of *not* shipping it is silent OOMs on
  workloads the charter explicitly names.
