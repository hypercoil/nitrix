# nitrix — Specification update (v0.3 → v0.4)

> **Status.** Smoothing-scope addendum to ``SPEC_UPDATE_v0.3.md``.
> Apply on top of v0.3.  All v0.3 changes stand; this patch retires
> the permutohedral lattice and re-bases the §3.3 marquee smoother on
> a **bounded bilateral** filter, generalising the already-shipped
> ``bilateral_gaussian``.  Sections not listed are unchanged.
> **Summary of changes.** Permutohedral lattice retired (not merely
> deferred); the tripwire of SPEC_UPDATE §3.3 / SPEC_UPDATE_v0.2 §3.3
> is moot and withdrawn.  ``bilateral_gaussian`` generalised from a
> diagonal per-feature bandwidth to a factored metric ``M = L Lᵀ``
> (the ``FeatureMetric`` ADT), gains a validity ``mask`` for ragged /
> padded neighbourhoods, and gains fixed-affinity iteration
> (``n_iters``).  No new top-level symbol: the bounded bilateral *is*
> ``bilateral_gaussian``.

---

## §0. Rationale: bounded support dissolves the permutohedral obstacles

The permutohedral lattice (Adams, Baek, Davis 2010) was reserved in
SPEC §3.3 as the marquee high-`d_f` smoother and demoted to a
*tripwire-gated target* in SPEC_UPDATE §3.3.  The G2 assessment
(`docs/design/permutohedral-g2.md`) found the pure-JAX path structurally
unfit: a dynamic-membership hash table for sparse lattice vertices,
neighbour materialisation during the blur, and a splat whose simplex
identity is piecewise-constant in `features` (a gradient discontinuity
at simplex boundaries).  Every one of those obstacles exists **only**
because permutohedral targets *unbounded* support in high `d_f` — that
is what the lattice machinery buys.

Our use cases accept **bounded-hop** neighbourhoods (a radius-`r` grid
box; a mesh k-ring; later, a geodesic ball).  Conceding bounded support
removes the reason the lattice existed: there is no lattice, no hash
table, no splat/blur/slice, no piecewise-constant simplex selection —
just a dense gather over a static neighbour index and a smooth weighted
sum.  nitrix already shipped exactly that operator as
``bilateral_gaussian`` on the ELL semiring substrate, and the grid /
k-ring neighbour builders already exist (``spatial_cube_neighbourhood``,
``sparse.grid.regular_grid_stencil``, ``sparse.mesh.mesh_k_ring_adjacency``).
This update generalises that operator to fill the role permutohedral was
reserved for, and retires the lattice.

See ``docs/design/bounded-bilateral.md`` for the full design.

---

## §3.3 `nitrix.smoothing` — replace in full

Two tiers ship; both are unconditional.

##### `gaussian` — unconditional baseline

Unchanged from SPEC §3.3 / SPEC_UPDATE §3.3.

##### `bilateral_gaussian` — the bounded bilateral (marquee capability)

A true high-dimensional bilateral filter over a **bounded** neighbourhood
and a feature-space metric.  One gather plus one weighted reduction via
``semiring_ell_matmul`` (REAL); statically shaped, ``jit`` / ``vmap`` /
``grad`` clean, GPU-native.  The weight is a smooth ``exp`` of a quadratic
form over a *fixed* neighbourhood, so the gradient is smooth everywhere
(no sort, no scatter, no simplex branch) — and it carries no separability
artefact, computing the joint kernel directly.

```python
def bilateral_gaussian(
    values: Float[Array, "n d_v"],          # decoupled from features
    features: Float[Array, "n d_f"],
    *,
    metric: FeatureMetric,                  # M = L Lᵀ, supplied factored
    neighbourhood: int | Int[Array, "n k_max"] | ELL,
    mask: Bool[Array, "n k_max"] | None = None,
    n_iters: int = 1,
    backend: Backend = "auto",
) -> Float[Array, "n d_v"]
```

Three orthogonal choices parameterise a call:

- **Values vs. features are decoupled** (already true pre-v0.4): `values`
  (e.g. a BOLD time series) and `features` (the multimodal signature set)
  are separate arguments with separate column counts.  The "filter the
  features by themselves" case is `values is features` at the call site.
- **The metric** `M = L Lᵀ` is supplied factored as a `FeatureMetric`
  (new; see below).  In the large-bandwidth limit on a channel, that
  channel drops out and the filter degrades toward a spatial-only
  Gaussian — the property the diagonal `sigma_features` argument
  provided in v0.3, now a special case.
- **The neighbourhood** is bounded: an `int` `k` (feature-space k-NN), an
  explicit `(n, k_max)` index array, or an `ELL` adjacency (grid box,
  mesh k-ring, geodesic ball).  Padded / ragged neighbourhoods carry a
  validity `mask` so padding contributes nothing — which also removes the
  double-counting a naive padded gather incurs at grid boundaries and at
  low-degree mesh vertices (e.g. the 12 pentagonal vertices of an
  icosphere).  When the neighbourhood is an `ELL`, the mask is derived
  from its padding identity unless given explicitly.

`n_iters > 1` grows effective support cheaply: the affinity graph
(features, neighbours, weights) is held fixed and only the values
diffuse, so the normalised weights are built once and re-applied —
`n_iters` mean-field updates of a bounded dense CRF (Krähenbühl &
Koltun 2011).

**Breaking change from v0.3.** The diagonal-only `sigma_features:
Float[Array, "d_f"]` keyword is replaced by `metric: FeatureMetric`.
The exact v0.3 behaviour is `metric=DiagonalMetric(sigma_features)`.
Per SPEC §9 ("no legacy users; we break freely") no shim is provided.

##### `FeatureMetric` — new ADT (`nitrix.smoothing.metric`)

`M ⪰ 0` is supplied factored as `M = L Lᵀ`, `L ∈ R^{d_f×k}`, `k ≤ d_f`.
The bilateral kernel only ever needs the projection `z = Lᵀ d` and the
squared norm `q = zᵀz = dᵀ M d`, so the metric is a tiny ADT exposing a
single `project(deltas) -> projection` method.  Both concrete records are
registered JAX pytrees whose array field is the single leaf, so a metric
flows through `jit` / `vmap` and `L` is differentiable end-to-end (a
learnable parameter, if a consumer wants one).

- `DiagonalMetric(sigma)` — `M = diag(1/σ²)`; one bandwidth per feature.
  The cheap, interpretable default; recovers the v0.3 bilateral.
- `FactorMetric(factor)` — a general factor `L: (d_f, k)`.  `k < d_f`
  gives a **low-rank** metric (correlated channels projected into a
  `k`-dim discriminative subspace; weight cost `O(k)` not `O(d_f)`);
  `k = d_f` gives a general anisotropic metric.

Pure constructors compose the tiers without leaving the substrate:
`block_diagonal_metric(blocks)` (independent per-modality bandwidths —
e.g. a tissue-intensity block weighted separately from a
functional-correlation block) and `metric_from_spd(M)` (Cholesky factor
of an explicit SPD metric).

**Out of scope.** Data-driven *fitting* of `L` (population PCA,
supervised metric learning) is a modelling concern for a consumer, built
from `nitrix.stats.covariance` and `nitrix.linalg`; nitrix supplies only
the pure metric mechanism.

##### `permutohedral_lattice` — RETIRED

The symbol and its stub are removed.  The bounded bilateral above
supersedes its intended role for the feature dimensionalities we target,
and — via a low-rank `FactorMetric` — beyond.  The G2 tripwire
(SPEC_UPDATE §3.3 criteria 1–4; SPEC_UPDATE_v0.2 §3.3 criterion 2) is
withdrawn as moot.  The namespace is **not** reserved: there is no
deprecation stub, since there are no consumers to redirect.

##### SUSAN emulator

Unchanged in surface (`sigma_space` / `sigma_intensity` floats).
Internally it now builds a `DiagonalMetric` and threads the
spatial-cube validity mask, so boundary voxels average only over their
in-bounds neighbours.

---

## §10 / §7.2 Success-criteria & plan amendments

- SPEC §10 / SPEC_UPDATE §3.3 exit lines reading "`permutohedral_lattice`
  *either* shipped meeting all four tripwire criteria *or* raises
  `NotImplementedError`" are **closed as retired**: neither branch
  applies; the symbol does not exist.
- IMPLEMENTATION_PLAN gate **G2 (Permutohedral tripwire)** resolves to
  **retired** (see `docs/design/bounded-bilateral.md`); the Phase 4
  checklist item is satisfied by the bounded bilateral.
- `bilateral_gaussian` remains the marquee §3.3 capability, now with the
  factored metric, validity mask, and fixed-affinity iteration.
