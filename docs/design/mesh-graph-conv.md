# mesh-graph convolutions — the edge-functional substrate

> **TL;DR.**  ``nitrix.semiring.semiring_ell_edge_aggregate`` adds
> *edge-functional* aggregation to the ELL substrate: a user-supplied
> ``edge_fn(h_i, h_j, w, ij) -> e`` runs at every (vertex, neighbour)
> pair and the per-row results are reduced under a semiring.
> This single primitive covers GCN, GAT, MoNet, EdgeConv/DGCNN,
> and ChebNet — without exposing a separate "message-passing" API.
> Companion encoder-decoder primitives ship in
> ``nitrix.morphology.pooling`` (``max_pool_with_indices_nd`` /
> ``max_unpool_nd``) and convenience wrappers in ``nitrix.sparse.mesh``
> (``mesh_pool_max`` / ``mesh_unpool_max`` / ``mesh_bary_upsample``).
> Together they close the consumer ask in
> ``NITRIX_FEEDBACK_ILEX.md`` (FA2–FA4) for a coherent mesh-graph CNN
> stack on top of the substrate.

## Motivation

The ilex consumer feedback (2026-05-14) identified that
``nitrix.semiring`` already supports *linear* graph convolution
(``semiring_ell_matmul``) but lacks the *edge-functional* extension
needed for the modern graph-conv family.  Every architecture in that
family — GCN, GAT, EdgeConv/DGCNN, MoNet, ChebNet — has the same
shape:

```
h_i' = Reduce_{j ∈ N(i)}  φ(h_i, h_j, w_ij, (i, j))
```

What varies between the architectures is:

- **φ** (the *edge function*): a linear projection of ``h_j`` (GCN), a
  concat-then-MLP of ``(h_i, h_j - h_i)`` (EdgeConv), a softmax-
  weighted message (GAT), a learned kernel of edge attributes
  (MoNet).
- **Reduce** (the semiring): sum (REAL, the GCN/EdgeConv/GAT case),
  max (TROPICAL_MAX_PLUS, the symmetric-pool case),
  min (TROPICAL_MIN_PLUS, the morphological-erode case).
- The adjacency ELL itself (``ell.values``, ``ell.indices``).

So the primitive is one function with one user callable.  No
``message_passing.MessagePassing`` base class, no scatter API, no
"propagate / aggregate / update" trichotomy.  Just gather-the-
neighbours, apply-the-user-fn, reduce.

## The primitive

```python
semiring_ell_edge_aggregate(edge_fn, ell, x, *, semiring) -> jnp.ndarray
```

with

```python
def edge_fn(h_i, h_j, w, ij) -> e:
    # h_i: (d_in,)  vertex-i feature
    # h_j: (d_in,)  vertex-j feature (a neighbour)
    # w:   scalar   stored edge weight (ell.values[i, p])
    # ij:  (2,)     index pair, jnp.int32
    # return: (d_out,) -- the edge message
    ...
```

Output: ``(*batch, n_rows, d_out)``.  The ``ij`` argument is always
passed even though most edge_fns ignore it; consistency wins over
brevity when the alternative is keyword-shaped overloads.

### Implementation pattern

```python
# Gather neighbour features: (*batch, n_rows, k_max, d_in)
x_neigh = x[..., ell.indices, :]
# Broadcast source features to (n_rows, k_max, d_in)
x_src = jnp.broadcast_to(x[..., :, None, :], x_neigh.shape)
# Build (n_rows, k_max, 2) index grid
ij_grid = jnp.stack(jnp.meshgrid(
    jnp.arange(n_rows), jnp.arange(k_max), indexing='ij',
), axis=-1)
ij_grid = jnp.stack([ij_grid[..., 0], ell.indices], axis=-1)
# Apply edge_fn at every (i, p) pair via nested vmap
e = jax.vmap(jax.vmap(edge_fn))(
    x_src, x_neigh, ell.values, ij_grid,
)
# Reduce along the neighbour axis under the semiring
return _reduce(e, semiring)
```

The inner vmap is over ``p`` (the neighbour slot); the outer vmap
is over ``i`` (the row).  Leading batch dimensions on ``x`` are
handled by a further vmap with
``in_axes=(0, 0, None, None)`` — the source *and* neighbour
features are batched but the ELL structure (values, ij_grid) is
not.  Getting these axes wrong was the load-bearing bug during
implementation: ``in_axes=(0, None, None, None)`` raised a clear
shape-mismatch error because ``x_neigh`` is batched too (since
it's ``x[..., indices, :]``).

### Semiring coverage

At first GA we ship three of the six algebras:

- **REAL** → ``jnp.sum`` along the neighbour axis.  Covers GCN /
  GAT / EdgeConv / MoNet / ChebNet (their reductions are all sums).
- **TROPICAL_MAX_PLUS** → ``jnp.max``.  Covers symmetric mesh
  max-pool and tropical morphology composed with a learned φ.
- **TROPICAL_MIN_PLUS** → ``jnp.min``.  The dual.

**LOG** and **EUCLIDEAN** raise ``NotImplementedError`` with a
specific message naming the algebra.  Rationale: the meaningful
form of an edge-functional LOG aggregate is
``logsumexp(edge_fn(...))``, but the edge_fn output shape is
arbitrary — the user is in a better position than us to decide
how the log-domain reduction should compose with their per-edge
message.  We'd rather raise than guess.  **BOOLEAN** is excluded
for the same reason (most user edge_fns produce floats; the
boolean-cast semantics would be a footgun).

When a consumer arrives with a concrete LOG use case we'll ship
the right specialisation then.

### Padding semantics

ELL pads short rows with the algebraic identity (``0`` for REAL,
``-inf`` for TROPICAL_MAX_PLUS, ``+inf`` for TROPICAL_MIN_PLUS).
The pad column has ``values[i, pad_col] = 0`` for the REAL case.

For an edge_fn that **multiplies by ``w``** (the common case for
GCN-style ``w * (W @ h_j)``), padding is automatically absorbed:
``w == 0`` makes the message zero, which is the REAL identity.
Tropical cases need ``w == -inf`` / ``+inf`` at pads — which is
what the ELL identity already provides for those algebras.

For an edge_fn that *ignores* ``w`` (a pure-MLP-of-h-only
message), the user takes responsibility for masking — typically
by gating the result on ``w``.  We surface this in the docstring
rather than enforcing it via a wrapper, because there's no
universal "absorb pad" operation when the edge_fn output type
isn't a linear function of the stored value.

This is the same contract as ``semiring_ell_matmul``: the
substrate stores values, runs the user-named algebra, and trusts
the algebra's identity to make padding inert.  ``edge_aggregate``
extends that contract by making the *user* responsible for
ensuring their edge_fn respects the identity.

### Differentiability

``edge_fn`` is a plain Python callable that closes over its
parameters.  ``jax.grad`` through ``semiring_ell_edge_aggregate``
flows naturally back to those parameters (via the closure) and
to ``x`` (via ``x_neigh`` / ``x_src``).  No custom_vjp needed:
the gather + vmap + reduce composition is fully differentiable.

ELL ``values`` and ``indices`` are non-differentiable by
construction — they describe the graph topology and edge
weights, not learnable parameters.  (Architectures that *learn*
edge weights, e.g. GAT, do so by computing them inside ``edge_fn``
from ``h_i`` / ``h_j``, not by treating ``ell.values`` as a
learnable tensor.)

## Encoder-decoder pooling: ``morphology.pooling``

### Why morphology

Max-pool is the symmetric variant of TROPICAL_MAX_PLUS
morphological dilation: same gather pattern, same reduction
algebra, different fan-out / stride.  Erode is the dual.  So
``max_pool_with_indices_nd`` *belongs* in
``nitrix.morphology`` next to ``dilate`` and ``erode`` —
co-locating it with the algebraically-related ops keeps the
substrate's organising principle visible.  (We considered a
top-level ``nitrix.pool`` subpackage; rejected for adding a new
top-level surface without adding a new conceptual layer.)

### `max_pool_with_indices_nd(x, pool_size, spatial_rank)`

Returns ``(pooled, indices)`` where ``indices`` is the
**global flat C-order index** of the argmax in the unpooled
spatial volume.  Why flat-and-global instead of per-window:
``max_unpool_nd`` needs a single integer to scatter back to, and
storing the per-window index forces every downstream consumer to
do the same arithmetic.  Doing it once at pool time amortises the
cost across the encoder-decoder lifecycle.

Implementation: reshape spatial axes to expose per-window
sub-axes (``(B, C, n_h, ph, n_w, pw, ...)``), transpose so all
within-window axes are contiguous at the end, flatten to a single
window axis, take ``argmax`` / ``max``, then convert the
within-window flat-index back to the *global* C-order index via
the suffix-product trick:

```
global_flat = sum_d (window_start_d + within_window_d) * spatial_stride_d
```

Channel-first layout, N-D (we exercise 2-D and 3-D in tests;
4-D fMRI shapes are reachable but not currently benched).

### `max_unpool_nd(x, indices, output_shape, spatial_rank)`

Scatters values back: ``jnp.zeros(target).at[idx].set(values)``,
vmapped per (batch, channel) so the scatter doesn't fight with
JAX's array-at-array indexing rules.

### Cross-framework parity caveat

A specific test (``test_unpool_argmax_agreement_is_load_bearing``)
encodes the contract we expect downstream consumers to honour:
**parity with PyTorch / Keras max_unpool implementations is on
*argmax-of-output-agreement*, not raw-logit allclose.**

The reason: ties in the pre-pool tensor (which the test
deliberately probes with a 1e-8 perturbation) cause different
frameworks to pick different argmax indices for the same max
value.  Those tie-breaks are an implementation detail of the
underlying argmax; they cascade through unpool into different
zero-vs-nonzero patterns in the output, which still produce the
*same* per-pixel argmax-of-channels (the load-bearing readout in
segmentation networks).  Documenting this up front prevents the
"my port gets 1e-3 raw-logit difference, is that a bug?" support
load.

## Mesh wrappers: ``sparse.mesh``

``mesh_pool_max`` / ``mesh_unpool_max`` / ``mesh_bary_upsample``
are five-line convenience wrappers, not new primitives:

```python
def mesh_pool_max(cross_level_adjacency, fine_features):
    # TROPICAL_MAX_PLUS with zero-valued ELL: max-plus degenerates to max.
    zero_ell = ELL(
        values=jnp.zeros_like(cross_level_adjacency.values),
        indices=cross_level_adjacency.indices,
        n_cols=cross_level_adjacency.n_cols,
        identity=-jnp.inf,
    )
    return semiring_ell_matmul(
        zero_ell.values, zero_ell.indices, fine_features,
        semiring=TROPICAL_MAX_PLUS, n_cols=zero_ell.n_cols,
    )

def mesh_bary_upsample(bary_ell, coarse_coords):
    # Direct REAL matmul on barycentric weights.
    return semiring_ell_matmul(
        bary_ell.values, bary_ell.indices, coarse_coords,
        semiring=REAL, n_cols=bary_ell.n_cols,
    )
```

The wrappers exist for **discoverability**: a consumer skimming
``nitrix.sparse.mesh`` for "how do I pool on an icosphere
hierarchy" finds a named function instead of needing to know that
TROPICAL_MAX_PLUS with zero values does the right thing.

We deliberately did **not** ship:

- ``mesh_pool_avg`` — the consumer can pass a row-stochastic ELL
  to ``semiring_ell_matmul`` directly; naming a wrapper for it
  is bikeshedding.
- ``mesh_unpool_avg`` — same.
- A separate "graph pool" API distinct from this — pooling on a
  mesh is a graph operation; conflating it with the morphology
  pool API would have wasted a clear analogy.

## Sprint B preview: icosphere hierarchy

This sprint shipped the *primitive*; the next sprint (FA2-adjacent,
deferred) will ship the *construction* helpers:

- **``icosphere_hierarchy(n_subdivisions) -> list[Mesh]``** — a
  sequence ``[ico_0, ico_1, ..., ico_n]`` with the parent-child
  relationships preserved.  Today's ``icosphere(n)`` builds the
  finest level directly; the hierarchy variant exposes the
  intermediate levels.
- **``icosphere_cross_level_adjacency(parent, child) -> ELL``** —
  the ``n_child × n_parent`` ELL that records "which fine-vertex
  derives from which coarse-vertex (or edge midpoint thereof)".
  Feed this to ``mesh_pool_max`` / ``mesh_unpool_max``.
- **``icosphere_bary_upsampler(parent, child) -> ELL``** — same
  shape, but with barycentric weights for *continuous*
  upsampling of vertex-valued fields.  Feed this to
  ``mesh_bary_upsample``.

The reason these are deferred: ilex's near-term consumer asks
required the *primitives* to be in place first.  The construction
helpers depend on subdivision-level bookkeeping that the current
``icosphere`` implementation doesn't expose; rewriting that
internally is a separable change.

## What we considered and didn't pick

**A ``message_passing.MessagePassing`` base class** (PyG style).
Rejected: the OOP class adds a new conceptual layer
(``propagate`` / ``message`` / ``update`` / ``aggregate``) that
buys nothing over the closure-based ``edge_fn`` once you have
``vmap``.  The PyG class exists because PyTorch's eager mode
needs explicit message-tensor materialisation; JAX's
``vmap`` + ``jit`` does the same thing without the boilerplate.

**A separate ``ell_row_softmax`` primitive for GAT-style
attention.**  Rejected: it's three lines of composition
(``softmax`` over the neighbour axis after a per-edge score).
The consumer feedback explicitly agreed this didn't merit a
named function.

**Folding ``max_pool_with_indices_nd`` into
``semiring_pool`` as a generic semiring-pool primitive.**
Tempting but rejected: the *indices* are the load-bearing output
for the encoder-decoder use case, and they're only meaningful
under the argmax semiring.  Generalising would push the indices
into a sum-type return value that most callers would have to
``None``-check.  The morphology API stays direct.

**A ``mode='tf_same'`` alias on Gaussian smoothing.**  Rejected
per FA1: the alias would add a third name for what
``mode='constant'`` already does.  We added a cross-reference
note in the docstring instead — same discoverability, no new
surface area.

## Tests

- ``tests/test_ell_edge_aggregate.py`` (10 tests): GCN vs hand-
  roll at 1e-13; DGCNN end-to-end; TROPICAL_MAX/MIN_PLUS;
  padding-absorption; differentiability through edge_fn closure
  and through ``x``; batched-vs-unbatched parity; the
  unsupported-semiring raise path; the ``ij`` argument is
  correctly threaded.
- ``tests/test_pooling.py`` (10 tests): 2-D / 3-D pool / unpool
  parity; argmax indices resolve to the correct value;
  anisotropic ``pool_size``; non-divisible input raises; pool↔unpool
  round-trip preserves argmax positions; differentiability
  through both directions; the cross-framework parity
  disclaimer.
- ``tests/test_sparse_specialisations.py`` (4 new tests):
  ``mesh_pool_max`` takes the window-max; ``mesh_bary_upsample``
  is the weighted sum; ``mesh_unpool_max`` inverts pool at
  single-source mapping; ``mesh_bary_upsample`` is
  differentiable.
