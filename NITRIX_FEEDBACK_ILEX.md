# nitrix gap analysis — rolling

This file accumulates feedback about `nitrix` gaps, mismatches, and missing primitives encountered while porting models into ilex. It is structured to grow over time: each finding gets its own dated section, names the driving port, and proposes a concrete fix.

Companion / history: `NITRIX_FEEDBACK_JOSA.md` is the previous, JOSA-driven feedback batch that has already been sent upstream and is sealed.

**Source.** `/Users/rastko.ciric/dev/diffprog/nitrix` (read-only from this side; recommendations live here and graduate to upstream patches via the nitrix repo).

## How to use this doc

When a port surfaces a nitrix gap:

1. Add a `### YYYY-MM-DD — <short title> (<driving port>)` section below.
2. Describe what nitrix does today, what the consumer needs, and what the diff costs at the call site (workaround complexity).
3. Propose a concrete fix recipe — API change, default override, new primitive, or "document the gotcha".
4. Mark **severity**: `BLOCKING` (port fails parity without a workaround), `MISMATCH` (parity passes but a non-obvious override is required), or `CONVENIENCE` (the port works fine, but a helper would reduce friction for the next consumer).
5. Once the fix lands upstream, move the entry into the "Resolved" section at the bottom with a one-line note pointing at the resolving commit / PR.

## Open findings

### 2026-05-14 — `gaussian` truncate + boundary-mode defaults (ExVivoNorm)

> **RESOLVED (2026-05-20)** — see "Resolved findings" below.

**Severity.** MISMATCH (port passes parity at atol=1e-3 after a two-keyword override; defaults silently miscompute by ~6 in boundary cells before the override).

**What surfaced this.** Porting `exvivo.norm.lh.h5`, whose wrapper graph applies `neurite.layers.GaussianBlur(sigma=1)` to the inner UNet output before exponentiation to a multiplicative bias field. The first parity run produced max abs diff = 6.21 between JAX bundle and TF reference; the worst voxel was exactly at the grid corner `(0, _, 0, _)`.

**What's actually different.**

* **Kernel half-width formula.** `neurite.layers.GaussianBlur` uses `windowsize = round(sigma * 3) * 2 + 1` (truncate-at-3-sigma → window 7 at σ=1). `nitrix.smoothing.gaussian` defaults to `truncate=4` (→ window 9 at σ=1). Same sigma, different bandwidth.
* **Boundary mode.** neurite's `separable_conv` uses TF `padding='SAME'`, which **zero-pads** at the boundary. `nitrix.smoothing.gaussian` defaults to `mode='reflect'`. For a per-voxel exp() post-processing like ExVivoNorm's, this produces ~0.7 logit-space differences in the boundary cells, which become ~6× magnitude differences after exp().

**Workaround.** Pass `truncate=3.0, mode='constant'` explicitly. With both overrides, parity drops to ~1.5e-3 per-voxel diff (consistent with the standard TF→JAX accumulation envelope). This is what `ilex.models.exvivo_norm` does today; see `src/ilex/models/exvivo_norm/model.py` for the call site.

**Proposed fix.** Two minimal docstring additions, no API change required:

1. In the `truncate` parameter docstring, add: *"Set `truncate=3` to match the neurite / SynthMorph / SynthSeg / FreeSurfer-augmentation lineage, which uses `round(sigma * 3) * 2 + 1` taps. nitrix's default `truncate=4` matches scipy's `gaussian_filter`."*
2. In the `mode` parameter docstring, add: *"Use `mode='constant'` to match TF Conv2D / Conv3D `padding='SAME'` semantics (zero-pad at the boundary). nitrix's default `mode='reflect'` matches scipy."*

The defaults can stay (scipy-compatible is a defensible choice and easier for non-neurite users) — what's missing is the cross-reference. Without it, every neurite-trained model that consumes `nitrix.gaussian` will silently miscompute at the boundary.

A stretch option: a `mode='tf_same'` alias that resolves to `'constant'` would make the intent more discoverable from the consumer side. Low priority.

### 2026-05-14 — ELL-backed icosphere conv primitives (Topofit)

> **RESOLVED (2026-05-20)** — see "Resolved findings" below.

**Severity.** CONVENIENCE (Topofit ports without these; nitrix already has the foundational sparse-mesh + ELL surface that makes them additive). **High blast-radius**: lifting these primitives would let nitrix host every standard mesh-graph-conv variant (GCN, GAT, EdgeConv / DGCNN, MoNet, ChebNet, ...) on top of a single ELL backend, so any future surface-based learned model lands as a thin ilex / nimox wrapper. Topofit is the first ilex port to exercise mesh-graph-conv; the next surface-domain model (anything from the SphereMorph family, future surface-based segmenters, surface-attention transformers) will rediscover the same primitives.

**Current state in nitrix (great foundation).** The pieces topofit can lean on directly:

* `nitrix.sparse.ELL` — frozen-dataclass `(values, indices, shape, identity)`.
* `nitrix.sparse.mesh.icosphere(n_iterations)` — builds the canonical subdivided icosphere as a `Mesh(vertices, faces)`.
* `nitrix.sparse.mesh.mesh_k_ring_adjacency(mesh, k)` — returns an `ELL` whose per-row neighbour-list IS the icosphere's k-ring adjacency. Topofit's per-level `(adj_edges_a, adj_edges_b, adj_weights)` is exactly this object pre-flattened.
* `nitrix.semiring.semiring_ell_matmul(values, indices, B, *, semiring)` — the standard ELL × dense matmul under any semiring, with the canonical contract `C[i, j] = (+)_p values[i, p] (*) B[indices[i, p], j]`.
* `nitrix.semiring.algebras` — SumMonoid / ProductSemigroup / LogSumExpMonoid / MaxMonoid / etc., composable into custom semirings.

The existing surface already covers **spectral / fixed-kernel mesh ops** cleanly: a graph Laplacian smooth, a heat-kernel diffusion, a Gaussian smoothing on the icosphere — all "stored-values × dense-rhs" matmuls. The gap is **edge-functional** mesh convs.

**The edge-functional gap (the topofit-driven proposal).** Topofit's `DynamicGraphConv` is the DGCNN / EdgeConv (Wang et al. 2019) pattern:

```python
for each (i, p) where indices[i, p] = j:
    edge_msg[i, p] = mlp(concat([x[i], x[j] - x[i]]))      # learned per-edge function
    edge_msg[i, p] *= values[i, p]                         # stored geometric weight
out[i, :] = sum_p edge_msg[i, p]                           # ELL-aggregate
```

The standard `semiring_ell_matmul` doesn't reach this because `edge_msg` is COMPUTED at forward time from a learned function of `(x[i], x[j])` rather than STORED in the matmul's left operand. But the AGGREGATION half (steps 3 + 4) IS exactly `semiring_ell_matmul` under sum-product, given the per-edge messages. So the missing piece is **a primitive that builds per-edge messages from a callable, then aggregates them via ELL** — leaving the existing semiring-aggregate path unchanged.

**Concrete API proposal.** A new module `nitrix.semiring.ell_edge` with one core primitive:

```python
def semiring_ell_edge_aggregate(
    edge_fn: Callable[[Array, Array, Array, Array], Array],
    ell: ELL,
    x: Float[Array, '*batch n_vertices in_features'],
    *,
    semiring: Semiring = REAL,
    n_cols: Optional[int] = None,
    backend: Backend = 'auto',
) -> Float[Array, '*batch n_vertices out_features']:
    """Per-edge functional aggregation over an ELL adjacency.

    For each (i, p) in the ELL pattern where j = ell.indices[i, p]:
      e[i, p] = edge_fn(x[i], x[j], ell.values[i, p], (i, j))
    Then aggregate per row under ``semiring``:
      out[i] = (+)_p e[i, p]

    The ``edge_fn`` signature is intentionally permissive: source-
    feature, neighbour-feature, edge-value (optional), and the
    (i, j) pair (optional). Implementations only need to forward
    the args the user's fn declares.
    """
```

With this primitive, the standard variants compose as one-liners:

| Variant | `edge_fn` body |
|---|---|
| **Spectral / GCN** (Kipf-Welling style) | `lambda h_i, h_j, w, _: w * (W @ h_j)` (W is the shared linear projection) |
| **EdgeConv / DGCNN** (topofit) | `lambda h_i, h_j, w, _: w * mlp(concat([h_i, h_j - h_i]))` |
| **MoNet** (Monti et al. 2017) | `lambda h_i, h_j, w, ij: gaussian_kernel(coord[i] - coord[j]) * (W @ h_j)` (pulls geometry per-edge) |
| **GAT** (Velickovic et al. 2018) | precompute α via softmax over `attention_fn(h_i, h_j)`, then `lambda h_i, h_j, _, ij: α[ij] * (W @ h_j)` (this needs an attention-pre-pass; see below) |
| **ChebNet** | iterate `semiring_ell_matmul` with Chebyshev polynomial of the Laplacian — no edge_fn needed; the existing primitive covers this directly |

**Sub-proposal: ELL-attention helper.** GAT and friends need a per-edge softmax across rows. Topofit doesn't use this, but the API is so close it would be a one-line addition:

```python
def ell_row_softmax(
    edge_scores: Float[Array, '*batch n_vertices k_max'],
    ell_indices: Int[Array, '*batch n_vertices k_max'],
    n_cols: int,
) -> Float[Array, '*batch n_vertices k_max']:
    """Row-wise softmax over the ELL pattern. The denominator
    sums over each row's k_max neighbour slots; the padding
    semiring-identity entries naturally drop out (they're 0 in
    REAL after `exp(-inf) = 0`)."""
```

This makes `GAT` a 4-line construction: compute attention scores, row-softmax, multiply scores into edge messages, aggregate with `semiring_ell_edge_aggregate(sum-product)`.

**Mesh hierarchy primitives (the cross-level pool / unpool / upsample).** Topofit's icosphere cascade also needs three primitives that aren't a strict fit for the existing ELL-aggregate path because they cross different vertex sets:

1. **`mesh_pool_max`** — for each coarse vertex `i`, take the max of fine-vertex features in its neighbourhood. The upstream's "EMG" (extended mapping graph) representation expresses this as a rectangular `(n_coarse, k_max)` index array `emg_a` indexing into fine-vertex features, then `max` along axis -2. This is mechanically an ELL-aggregate under the **MaxMonoid** semiring with values = constant identity (so values[i, p] = 0 → SUM-PRODUCT degenerates to MAX over neighbour features), AND the row/col index sets differ between input and output. Specifically: it's a `semiring_ell_matmul` with a "selector" left operand (a 0/1 mask) and the MAX monoid; the result picks the max of selected dense rows.

   Already feasible with current nitrix surface if you frame the EMG as an ELL on the cross-level adjacency with `values = jnp.zeros(...)` and `semiring = MAX`. Document this composition as a `mesh_pool_max(coarse_ell, fine_features)` recipe in the SPEC; a thin wrapper would smooth ergonomics.

2. **`mesh_unpool_max`** — the symmetric coarse-to-fine version. Same recipe with the cross-level ELL flipped.

3. **`mesh_bary_upsample`** — for each fine-level vertex i: `out[i] = sum_k weights[i, k] * coords[sources[i, k]]`. This IS standard `semiring_ell_matmul` with `values = bary_weights`, `indices = bary_sources`, `B = coarse_coords`, `semiring = REAL`. **Zero new code needed** — just packaging the bary data as an ELL and calling the existing primitive. A `mesh_bary_upsample(bary_ell, coords)` convenience wrapper would make this discoverable.

**Mesh hierarchy construction.** `nitrix.sparse.mesh.icosphere(n)` builds a single-level icosphere. Topofit needs the **full hierarchy** (ico-0 through ico-7) plus the cross-level edge sets. A natural extension:

```python
def icosphere_hierarchy(n_levels: int) -> List[Mesh]:
    """Return a list of icospheres at subdivision levels 0..n_levels.
    Each consecutive pair shares the cross-level adjacency via the
    midpoint-subdivision rule (each fine vertex either coincides
    with a coarse vertex or sits at the midpoint of a coarse edge)."""

def icosphere_cross_level_adjacency(
    hierarchy: List[Mesh],
    coarse_level: int,
    fine_level: int,
) -> ELL:
    """ELL whose row-i (coarse vertex) is the fine vertices that
    project to it under the subdivision rule. Returns the pool /
    unpool edge set; topofit's `mapping-N-to-M-indices` is this
    object."""

def icosphere_bary_upsampler(
    hierarchy: List[Mesh],
    coarse_level: int,
    fine_level: int,
) -> ELL:
    """ELL whose row-i (fine vertex) is the 3 coarse-vertex sources
    that the fine vertex's coordinate barycentrically interpolates
    from, with the weights stored in ``values``. Topofit's
    `bary-N-sources` + `bary-N-bary` is exactly this object."""
```

The functions are pure-NumPy host-side because the subdivision logic is combinatorial; the resulting ELLs are then JAX-compatible. This mirrors the structure of the existing `nitrix.sparse.mesh.mesh_k_ring_adjacency` which is also host-built.

**Pulling it together: what a topofit-equivalent forward looks like with the proposed surface.**

```python
hierarchy = icosphere_hierarchy(n_levels=7)
kring_per_level = [mesh_k_ring_adjacency(m, k=1) for m in hierarchy]
pool_per_level = [icosphere_cross_level_adjacency(hierarchy, L-1, L) for L in 1..7]
bary_per_level = [icosphere_bary_upsampler(hierarchy, L-1, L) for L in 1..7]

# Per DynamicGraphConv layer in topofit's DeformationBlock:
def dgc(h, kring, mlp_weights):
    def edge_fn(h_i, h_j, w, ij):
        msg = w * (mlp_weights @ jnp.concatenate([h_i, h_j - h_i]))
        return jax.nn.leaky_relu(msg, 0.3)
    return semiring_ell_edge_aggregate(edge_fn, kring, h, semiring=REAL)

# Mesh pool / unpool:
def pool(h, fine_to_coarse_ell):
    return semiring_ell_matmul(
        jnp.zeros_like(fine_to_coarse_ell.values),    # max-monoid identity
        fine_to_coarse_ell.indices, h,
        semiring=MAX_PLUS,
    )

# Barycentric upsample (zero new code):
def bary_up(coords, bary_ell):
    return semiring_ell_matmul(
        bary_ell.values, bary_ell.indices, coords, semiring=REAL,
    )
```

This collapses topofit's ~400 lines of mesh-domain JAX into ~30 lines on top of nitrix's surface. The same primitives serve any future mesh-graph-conv network.

**Backend story.** `semiring_ell_matmul` already has a Pallas backend (per `nitrix/semiring/ell.py`); `semiring_ell_edge_aggregate` should inherit it. The edge_fn callable is unrolled per-`(i, p)` block via vmap; if the edge_fn is a fixed-shape per-edge MLP (the topofit case), the inner kernel is a tiled matmul that maps to the same Pallas dispatch. For dynamic edge_fns (e.g. attention with row-softmax) a slower JAX fallback is fine — Pallas-fast is a stretch goal.

**Migration path for topofit specifically.** Once `semiring_ell_edge_aggregate` lands, the JAX port at `ilex/src/ilex/models/topofit/_graph_conv.py:DynamicGraphConv` reduces to ~15 lines (the per-edge MLP plus a `semiring_ell_edge_aggregate(edge_fn, kring_ell, x)` call). The `topo_*` placeholder tuples in `model.py` consolidate into one `mesh_hierarchy: List[ELL]` field. The save/load path through `default_pytorch_registry` is unchanged — the rename adapter just emits `mesh_hierarchy.<level>.values` / `.indices` keys instead of the current 70-tensor topology surface.

### 2026-05-14 — `MaxUnpool3d` and indices-based pooling primitives (PGlandsSeg)

> **RESOLVED (2026-05-20)** — see "Resolved findings" below.

**Severity.** CONVENIENCE (the port itself works; this is a primitive worth lifting for future consumers).

**What surfaced this.** Porting `mri_pglands_seg`. The architecture uses ``torch.nn.MaxPool3d(return_indices=True)`` paired with ``torch.nn.MaxUnpool3d`` for encoder->decoder coupling — the encoder records the per-window argmax positions during pooling, and the matching decoder unpool scatters values back into the pre-pool grid at those same positions. JAX has neither primitive: `lax.reduce_window` supports max but doesn't return indices; `jax.scipy.ndimage` doesn't have an unpool. We vendor `maxpool3d_with_indices` and `maxunpool3d` in `ilex/src/ilex/models/pglands_seg/_unet.py` (~80 lines, tested against torch parity on fresh random inputs at bit-exact agreement).

**Why this is a nitrix-shaped primitive.** Indices-based pooling shows up in:

* V-Net and several U-Net variants beyond pglands (anywhere the architect wants "remember where the max came from").
* Attention-pooling networks where the saved indices feed an attention readout.
* General medical-imaging encoder-decoder networks where the upsample path needs to preserve high-confidence spatial localisation (this is the pglands rationale).

The primitive pair is generic — no model-specific knowledge required. Channel-first, spatial-rank-agnostic (1D / 2D / 3D); the only parameter is `pool_size`. A clean signature looks like:

```python
def max_pool_with_indices_nd(
    x: Float[Array, '... C *spatial'],
    *,
    pool_size: int | Tuple[int, ...],
    spatial_rank: int,
) -> Tuple[Float[Array, '... C *pooled'], Int[Array, '... C *pooled']]:
    ...

def max_unpool_nd(
    x: Float[Array, '... C *pooled'],
    indices: Int[Array, '... C *pooled'],
    *,
    output_shape: Tuple[int, ...],
) -> Float[Array, '... C *spatial']:
    ...
```

**Suggested home.** A new `nitrix.pooling` subpackage (or `nitrix.morphology` if the SPEC prefers grouping with morphological dilation/erosion -- both are window-reduction patterns, although the argmax+scatter angle is closer to "remember-and-restore" than the morphological algebra). Either works.

**JOSA-adjacent note: float-noise fragility of MaxUnpool3d across frameworks.** The PGlandsSeg port surfaces a *consumer*-side concern worth flagging here even though it's not a nitrix gap directly: argmax-based pooling primitives are fragile to cross-framework float-noise. When the encoder accumulates ~1e-3 max abs difference between JAX and torch (typical for a 4-level Conv3D cascade), the per-window argmax positions can flip at the 0.02-0.03% of voxels where two neighbouring values are nearly equal. The matching unpool then scatters values to slightly different positions, producing O(10) per-voxel raw-logit differences. At the semantic level (per-voxel class assignment via argmax over the channel axis) this is harmless — the inter-class ordering is preserved by the magnitude shifts. Documented this in PGlandsSeg's DEVNOTES.md as "the load-bearing parity check is argmax agreement, not raw-logit allclose". Future consumers of any nitrix `max_pool_with_indices_nd` primitive will hit the same trap when verifying parity against a torch / TF reference; a docstring note recommending the argmax-agreement check would save them the diagnostic time.

### 2026-05-14 — TF `load_weights(by_name=True)` silently fails on nested Functional `.h5` (ExVivoNorm)

> **RESOLVED (2026-05-20)** — N/A for nitrix; see "Resolved findings" below.

**Severity.** N/A for nitrix specifically — this is a TF / Keras footgun, not a nitrix gap. Filed here because **the same trap surfaces during any "load FreeSurfer-Keras-bundled .h5 into a hand-rebuilt skeleton" workflow**, which is the staging pattern several ilex ports use; nitrix consumers staging weights from these formats need to know about it.

**What surfaced this.** Porting `exvivo.norm.lh.h5`. The .h5 was saved from a wrapper Keras model whose top-level structure is `InputLayer → Functional("inorm") → GaussianBlur → Lambda → Multiply`. The "inorm" inner Functional contains the trainable UNet; `model.weights` are stored in the .h5 at `model_weights/inorm/<inner_layer>/kernel:0`.

When you hand-rebuild ONLY the inner Functional (without the outer wrapper) and call `load_weights(by_name=True)`, **the load silently does nothing** — every layer retains its random init, with no error. The model runs forward at plausible magnitude (it's a randomly-initialised UNet, output looks "in range") and the parity test reports massive disagreement with the JAX bundle that DID load the real weights.

The fix is to rebuild via `tf.keras.Model.from_config(...)` (which preserves the nested-Functional structure the saved .h5 expects) and wrap it in an outer `Model(inputs=inp, outputs=inner(inp))` before calling `load_weights`. See `make_safetensors_exvivo/extract_tf.py` for the implementation.

**Proposed action.** Add a heads-up to `ilex/SKILL.md` Stage 3.1 ("Extract the canonical artefact"), Common failure modes — make it failure mode #12 alongside the Keras `skip_mismatch=True` trap (failure mode #11). The relationship is structurally similar: both produce "loaded random weights but ran without error", both manifest as "JAX and TF disagree by O(10x)+ with no obvious topological clue". The diagnostic is the same: dump `layer.kernel.numpy().sum()` after load and compare to the .h5 directly.

This isn't a nitrix concern at all (nitrix has no Keras surface), but I'm noting it here so the next port that hits it can find the fix faster.

### 2026-05-18 — SUGAR exercises the implemented topofit primitives + surfaces three deltas (SUGAR)

> **RESOLVED (2026-05-20)** — all three deltas addressed; FreeSurfer I/O
> deliberately kept out of nitrix. See "Resolved findings" below.

**Status of the topofit-proposed primitives.** All three primitives the topofit feedback
entry proposed (`semiring_ell_edge_aggregate`, `icosphere_hierarchy` / `_cross_level_adjacency`
/ `_bary_upsampler`, `mesh_pool_max` / `mesh_unpool_max` / `mesh_bary_upsample`) **are
implemented in nitrix today** -- the topofit entry's "Resolved" annotation just hasn't been
moved to the bottom of this file yet (this entry is the trigger for that bookkeeping).

**Severity.** CONVENIENCE (SUGAR ports without these via port-local vendoring; the deltas
described below are blast-radius reducers, not blockers). High blast-radius: SUGAR is the
**second** surface-domain port to need ELL-mesh-graph-conv primitives (topofit was the first),
which is exactly the second-consumer signal the topofit entry asked for. Both ports' edge-
functional aggregations and pool / unpool / bary surface land cleanly on the
**already-implemented** `semiring_ell_edge_aggregate` + `icosphere_*` API; SUGAR adds three
concrete refinements the topofit entry didn't anticipate.

**What SUGAR is.** Spherical Ultrafast Graph Attention Registration (arXiv 2307.00511; DeepPrep
pins this for cortical-surface registration via its `pBFSLab/SUGAR` fork). 16 published
checkpoints: a per-hemisphere rigid net + 4 per-hemisphere non-rigid nets across `fsaverage{3,4,5,6}`.
All share the same `GatUNet` architecture parameterised on the finest level:

* Encoder: 8 `ResEncodingBlock`s with mean-pool to the next coarser fsaverage level.
* Bottom: one `ResEncodingBlock` at `fsaverage0` (12 vertices).
* Decoder: mirror, with mid-edge unpooling (each fine vertex either coincides with a coarse
  vertex or is the mean of two coarse parents) + skip-cat.
* Output head: one more `ResEncodingBlock` at the finest level producing per-vertex Euler
  angles (the rigid variant means-reduces across vertices to a single `(1, 3)` global Euler).
* Each `ResEncodingBlock` is `(conv1 -> LeakyReLU -> conv2 + residual)` where `conv{1,2}` is
  literally `torch_geometric.nn.GATv2Conv(share_weights=True, edge_dim=27, num_heads=H)`.

**Confirmation of the topofit proposal.** ~80% of SUGAR's compute is exactly the topofit
proposal's surface:

* `semiring_ell_edge_aggregate` — the GATv2 forward fits the edge-functional contract.
* `ell_row_softmax` — required for the GAT attention normalisation (topofit didn't actually
  consume this; SUGAR is the first real consumer).
* `icosphere_bary_upsampler` — matches `IcosahedronUnPooling` exactly (mid-edge mean from
  two parent indices stored in `utils/auxi_data/fsaverage{i}_upsample_neighbors.npz`).
* `icosphere_hierarchy` — needs to span 7 subdivision levels (fsaverage0..6: 12, 42, 162, 642,
  2562, 10242, 40962 vertices), matching the topofit proposal's signature.

The ELL row-degree is bounded (`<=7` including self-loop; 6 for non-pole vertices), so the
ELL layout is friendly: `k_max = 7`, no degenerate padding overhead.

**Delta 1 — `semiring_ell_edge_aggregate` needs an edge-attribute argument.**

The topofit proposal's `edge_fn` signature is
`Callable[[h_i, h_j, edge_value, (i, j)], message]`. SUGAR's GATv2Conv with `edge_dim=27`
needs a per-edge tensor: each edge carries a `(F_e,)` Fourier embedding of the midpoint
coordinate `xyz[i] + xyz[j] / 2`, which gets folded into the attention score and the
message via a learned `lin_edge: Linear(F_e, H * F_out)`. That's a per-edge tensor distinct
from the scalar `edge_value` the topofit proposal exposed.

Two ways to fix:

A. **Generalise `edge_fn`'s 3rd argument** to accept either a scalar ELL value or an
   arbitrary per-edge tensor (`Float[Array, 'k_max F_e]'` per ELL row). The implementation
   carries an extra `edge_attr: Optional[Float[Array, '*batch n_vertices k_max F_e']]`
   argument on `semiring_ell_edge_aggregate`. When `edge_attr` is None, the third
   `edge_fn` arg is `ell.values[i, p]` as before. When set, it's `edge_attr[i, p, :]`.

B. **A separate primitive** like `semiring_ell_edge_aggregate_with_edge_attr` to keep the
   simpler topofit signature intact.

Option A reads cleaner in practice — the GATv2 case becomes one extra kwarg at the call site,
and the topofit's edge_value-only case stays an explicit `edge_attr=None`. Existing callers
need no change.

**Delta 2 — `mesh_coarsen_meanpool` as a sibling of `mesh_bary_upsample`.**

Topofit proposed `mesh_pool_max` (max-monoid over a cross-level ELL). SUGAR's
`IcosahedronPooling` is **mean over a fine-vertex neighbourhood with self-loops**: for each
coarse vertex `i`, take the mean of its child fine-vertex features and its own previous-level
feature. This is `scatter_mean` over a `(parent, child)`-edge ELL with self-loops included
on each parent row.

The topofit proposal correctly noted that max-pool over an ELL maps to `semiring_ell_matmul`
under the MAX monoid. The same composition trick works for mean-pool under the
`SumMonoid / count` semiring -- or more precisely, two `semiring_ell_matmul` calls: one to
sum over neighbours, another (with a values-only-ones ELL) to count them, then divide. But the
two-call recipe is awkward to discover; a one-liner helper

```python
def mesh_coarsen_meanpool(
    coarsen_ell: ELL,             # rows = coarse vertices, cols = fine + self
    fine_features: Float[Array, '*batch n_fine F']
) -> Float[Array, '*batch n_coarse F']:
    ...
```

would let SUGAR / topofit / any future surface-coarsening consumer call one function. The
existing `mesh_pool_max` proposal becomes its sibling.

**Delta 3 — FreeSurfer fsaverage topology source.**

The topofit proposal's `icosphere_hierarchy(n_levels)` returns a math-canonical subdivided
icosphere (the same one `nitrix.sparse.mesh.icosphere` already builds). SUGAR's pre-trained
checkpoints were trained against **FreeSurfer's actual `fsaverage{0..6}.sphere` topology**
(loaded via `nibabel.freesurfer.read_geometry`), which has different vertex ordering and
slightly different vertex coordinates from the math icosphere. The trained ELL-GAT
weights only round-trip if the topology at inference time matches the topology at training time.

So `icosphere_hierarchy` needs either:

A. An optional `topology_source: Literal['math', 'freesurfer']` kwarg (with `'freesurfer'`
   sourcing the per-level vertices / faces from a fsaverage subjects-dir path).
B. A dedicated `freesurfer_fsaverage_hierarchy(subjects_dir)` parallel to
   `icosphere_hierarchy`.

The clean separation is (B): FreeSurfer is a specific upstream choice with its own
distribution mechanism (the `.sphere` binaries under `$SUBJECTS_DIR/fsaverage*/surf/`) and
its own per-vertex sulc-normalisation training-set statistics. nitrix can stay
math-canonical; ports that need the FreeSurfer topology pass the right hatch in.

The matching pool / bary primitives for the FreeSurfer hierarchy should also branch on the
hatch:

* The mid-edge `_upsample_neighbors.npz` (precomputed by SUGAR's authors, dumped from the
  actual FreeSurfer subdivision) gives the bary upsampler's `(fine_vertex,
  parent_a, parent_b)` triples directly.
* The IcosahedronPooling neighborhood is the same triples flipped (`(parent_x, child_set)`),
  with self-loops added on each parent row.

**Port-local vendoring path (what SUGAR ships today).** Pending the upstream lift, the
SUGAR port at `ilex/src/ilex/models/sugar/_mesh.py` vendors:

* `FsaverageHierarchy` -- the per-level FreeSurfer mesh + cross-level adjacency tuples,
  loaded from a bundled `auxi_data/` subdirectory (the `.sphere` + `.npz` files).
* `GATv2Aggregate` -- the SUGAR-specific GATv2 forward, written directly against
  `nitrix.sparse.ELL` + `nitrix.semiring.semiring_ell_matmul` for the sum-aggregate path,
  with port-local attention pre-pass (scores -> row-softmax -> messages).
* `IcosahedronPooling` / `IcosahedronUnPooling` -- pure-JAX scatter_mean / mid-edge mean,
  both consuming the cross-level adjacency from `FsaverageHierarchy`.

This vendoring is intentional and tracked alongside the topofit vendoring. When the deltas
above land in nitrix, both ports' port-local primitives lift to the shared surface and the
adapters reduce by ~150 LOC each.

**Backend story.** Same as topofit. The SUGAR forward at `fsaverage6` (40962 vertices) is
the largest mesh-domain workload in ilex; a Pallas-backed `semiring_ell_edge_aggregate` would
benefit SUGAR most. Fallback JAX is acceptable at inference time.

### 2026-05-21 — GATv2 `add_self_loops(fill_value='mean')` is not covered by the ELL-edge surface (SUGAR)

> **RESOLVED (2026-05-22)** — see "Resolved findings" below.

**Severity.** MISMATCH (a GAT/GATv2 port built on the ELL-edge surface *passes structurally and runs at plausible magnitude* but silently miscomputes — the canonical `ell_edge.py` GATv2 recipe omits the step, and the SUGAR feedback's "k_max=7 including self-loop" assumption was wrong). This bit the SUGAR JAX port for real: max abs diff was ~0.20 on a ~0.18-magnitude signal until the step was added, after which parity dropped to float32 precision (~1e-6) against the upstream `torch_geometric.nn.GATv2Conv` oracle.

**What surfaced this.** Wiring a genuine torch oracle for the SUGAR parity fixtures (reconstructing the real upstream `GatUNet` and running it forward) instead of the prior NaN placeholder. The oracle disagreed with the ilex port everywhere. Root cause: `torch_geometric.nn.GATv2Conv` defaults to `add_self_loops=True`, and at **forward time** it does `remove_self_loops(edge_index, edge_attr)` then `add_self_loops(edge_index, edge_attr, fill_value='mean')`. So every vertex gets an `(i, i)` self-edge, and — crucially — that self-edge's `edge_attr` is filled with the **per-destination mean of the vertex's incoming edge attributes** (`scatter(edge_attr, edge_index[1], reduce='mean')`), *not* a geometric edge feature. Verified bit-exact: the ilex GATv2 algorithm matches PyG to 0.0 *with* the self-loop + mean-fill and diverges by 1.4 (single layer) *without*.

**What's actually different / why the current surface misses it.**

* `ell_edge.py`'s own worked GATv2 example (the `edge_fn` + `ell_row_softmax` recipe in the module docstring) builds the attention from the bare mesh adjacency and **never adds a self-loop**. A consumer following it verbatim reproduces the bug.
* The geometric mesh adjacency (`mesh_k_ring_adjacency`, or SUGAR's `get_network_index` built from triangle faces) contains **no self-loops** — they're a GATv2 *forward-time* augmentation, not part of the graph. The 2026-05-18 SUGAR entry's "`k_max=7` including self-loop; 6 for non-pole vertices" was therefore incorrect: the geometric k-ring is 6 (non-pole) with no self slot; PyG manufactures the 7th (self) slot per forward.
* Even a consumer who knows to add a self slot to the ELL still has to fill its `edge_attr` with the GATv2 `'mean'` semantics. The scalar `ell.values` padding signal doesn't help here — the self-loop is a *real* edge whose vector attribute is a reduction over the row's other attributes.

**Workaround (what SUGAR ships today).** The ilex port is still port-local (it doesn't yet route through `semiring_ell_edge_aggregate` — see the 2026-05-18 Delta 1 entry). The fix lives in `ilex/src/ilex/models/sugar/_gat.py::GATv2Conv._add_self_loops`: append `(i, i)` per vertex and set the loop attribute to `segment_sum(edge_attr, dst) / max(count, 1)`. ~20 lines, jit-safe (SUGAR graphs are self-loop-free so the PyG `remove` step is a documented no-op).

**Proposed fix.** When SUGAR's GATv2 conv migrates onto the ELL-edge surface (Delta 1), it needs a self-loop helper so the next GAT consumer doesn't rediscover this. Either:

A. **A helper** `ell_add_self_loops(ell, edge_attr, *, fill='mean') -> (ell, edge_attr)` that adds the self slot to the ELL pattern and fills the corresponding `edge_attr` slot with the per-row reduction of the other slots' attributes (`'mean'` matches PyG; expose `'zero'` / `'add'` for completeness). This mirrors `torch_geometric.utils.add_self_loops` at the ELL level.

B. **At minimum, document the gotcha** in the `ell_edge.py` GATv2 example: add the self-loop + mean-fill step explicitly, with a one-line note that PyG's `GATv2Conv` does this by default and omitting it silently miscomputes. The gaussian truncate/mode finding set the precedent that a default-semantics cross-reference is worth a docstring even when the API stays put.

Option A is the discoverable fix (GAT is the headline ELL-edge consumer); B is the floor. Both are additive — no change to existing `semiring_ell_edge_aggregate` callers.

## Resolved findings

Commit references land on merge; until then each entry points at the resolving
surface (functions / files) and the deviation-log date in
`IMPLEMENTATION_PLAN.md §10.3`.

- **2026-05-14 — `gaussian` truncate + boundary-mode defaults (ExVivoNorm).**
  RESOLVED. `nitrix.smoothing.gaussian`'s `truncate` and `mode` docstrings now
  carry the neurite/SynthMorph/SynthSeg (`truncate=3`) and TF `padding='SAME'`
  (`mode='constant'`) cross-references. Defaults unchanged (scipy-compatible).

- **2026-05-14 — ELL-backed icosphere conv primitives (Topofit).** RESOLVED. The
  full stack shipped: `nitrix.semiring.semiring_ell_edge_aggregate` (edge-functional
  aggregation), `nitrix.sparse.{icosphere_hierarchy, icosphere_cross_level_adjacency,
  icosphere_bary_upsampler}`, and the `mesh_pool_max` / `mesh_unpool_max` /
  `mesh_bary_upsample` wrappers. See SPEC_UPDATE_v0.3 §10.A.1–§10.A.2.

- **2026-05-14 — `MaxUnpool3d` and indices-based pooling (PGlandsSeg).** RESOLVED.
  `nitrix.morphology.pooling.max_pool_with_indices_nd` / `max_unpool_nd` shipped
  (N-D, channel-first), with the documented argmax-agreement parity contract. See
  SPEC_UPDATE_v0.3 §10.A.3.

- **2026-05-14 — TF `load_weights(by_name=True)` footgun (ExVivoNorm).** N/A for
  nitrix (a TF/Keras staging footgun, not a nitrix gap). Recorded here for
  downstream porters; no nitrix change.

- **2026-05-18 — SUGAR deltas (SUGAR).** RESOLVED (2026-05-20 sprint;
  `IMPLEMENTATION_PLAN.md §10.3`):
  - *Delta 1* — `semiring_ell_edge_aggregate` gained an `edge_attr=` kwarg; when
    set, `edge_fn` receives a 5th per-edge vector arg `a` **in addition to** the
    scalar `w` (refines the proposal's Option A, which would have displaced `w`,
    the padding signal). The first real consumer of `ell_row_softmax` (now shipped)
    is SUGAR's GATv2 attention.
  - *Delta 2* — `mesh_coarsen_meanpool` shipped as the mean sibling of
    `mesh_pool_max`; `icosphere_cross_level_adjacency` now carries a 1.0/0.0
    validity indicator so the mean is `sum(v·x)/sum(v)`.
  - *Delta 3* — **FreeSurfer I/O rejected inside nitrix** (SPEC §5.2 / non-negotiable
    §2.2.1: no `nibabel`, no `$SUBJECTS_DIR` reads). Instead,
    `icosphere_hierarchy_from_levels(meshes, parents)` lets the consumer (or `thrux`)
    read the `.sphere` / `.npz` files and hand nitrix plain arrays; every cross-level
    operator then runs on FreeSurfer topology with no branching. This is cleaner than
    the proposal's "branch the pool/bary primitives on a hatch."
  - *Bonus* — `nitrix.sparse.ell_mask` for masking incomplete geometries (medial
    wall / grey matter) by the correct per-semiring identity (see the masking note
    in `IMPLEMENTATION_PLAN.md §10.3`).

- **2026-05-21 — GATv2 self-loops not covered by the ELL-edge surface (SUGAR).**
  RESOLVED (2026-05-22; `IMPLEMENTATION_PLAN.md §10.3`).
  `nitrix.sparse.ell_add_self_loops(ell, edge_attr=None, *, fill='mean'|'add'|'zero',
  self_value=1.0)` appends a per-row `(i, i)` slot and, for per-edge attributes,
  fills the self-edge from the row's valid edges. Justified by the literature —
  graph attention's neighbourhood includes node `i` (Velickovic et al. 2018), and
  GCN renormalisation adds `Â = A + I` (Kipf & Welling 2017) — **not** by parity
  with a particular GNN library. The `ell_edge.py` GATv2 worked example (which had
  omitted the self-loop) is corrected. An aggregation-side convenience wrapper was
  deliberately **not** shipped: self-loops are architecture-specific (EdgeConv /
  MoNet / plain GCN omit them), so a bundled default would re-create the silent-
  default footgun this finding is about — revisit on demonstrated demand, or host
  the GAT composition downstream (a `nimox` ELLGAT module).
