# nimox mesh-loss geometry → `nitrix.geometry` (consolidation handoff)

> **Status (2026-06-25): RESOLVED — delegation map confirmed + gaps hosted.**
> All four primitives now have a public nitrix home (see the **Confirmed
> delegation map** below). The homes were **corrected** from this doc's
> original guess: the two topological primitives land in `nitrix.sparse.mesh`
> (beside `compute_vertex_normals` / `mesh_k_ring_adjacency`, the existing
> mesh-primitive home), and the two differentiable distance kernels land in a
> new pure-JAX `nitrix.geometry._mesh_distance` (re-exported from
> `nitrix.geometry`) — **not** in the host-side, non-differentiable
> `geometry._triangle_distance`, which is a different (point→triangle-mesh,
> NumPy) op. nimox re-points its four helpers and keeps its loss wrappers +
> parity tests. Shipped on `feat/nimox-mesh-loss-geometry`.
>
> **Original status (2026-06-25): handoff / request (nimox axis-iv → nitrix).** The
> nimox training-engine mesh-loss family (`nimox.loss.functional.registration`,
> landed on nimox `main`) hosts several **pure, model-agnostic mesh-geometry
> kernels** that are not loss-specific and almost certainly belong in (or
> already exist under) `nitrix.geometry`. This asks the geometry ledger to
> **confirm the existing equivalents so nimox can delegate** (mirroring the
> `metrics` → nitrix delegation it already did via
> [`classification-surface-metrics`]), and to **host the genuine gaps**. The
> nimox loss-shaped wrappers stay in nimox; only the geometry substrate moves.
>
> Per `README.md` / [`geometry-suite.md`](../geometry-suite.md) this is filed as a
> **consumer pointer into the existing ledger**, not a parallel kernel spec —
> dedupe against `surface.py` / `topology.py` / `_triangle_distance.py` /
> `intersection.py`, which look like the natural homes.

## What nimox currently hosts (and the candidate nitrix home)

| nimox primitive (`loss/functional/registration.py`) | What it is | Candidate nitrix home |
|---|---|---|
| `_face_normals(verts, faces)` | unit triangle normals via edge cross-product | `geometry.surface` / `differential` |
| `edge_face_adjacency(faces)` | host-side shared-edge → face-pair topology (static) | `geometry.topology` |
| `_seg_seg_sq_dist(...)` | Ericson clamped segment–segment squared distance | `geometry._triangle_distance` |
| `_nearest_sq_dist(...)` (chunked) | point-set nearest-neighbour squared distance (the chamfer core) | `geometry._triangle_distance` + the broad-phase ([`mesh-spatial-acceleration.md`](../mesh-spatial-acceleration.md)) |

The **loss reductions** — `chamfer_surface_distance`, `mesh_edge_smoothness`,
`mesh_normal_consistency`, `mesh_self_intersection_penalty` — stay in
`nimox.loss` (they encode the *loss* shape / unscalarised convention) and would
**delegate** the geometry above, exactly as the seg/surface metrics delegate.

## Confirmed delegation map (SHIPPED)

| nimox primitive | nitrix delegate | Notes |
|---|---|---|
| `_face_normals(verts, faces)` | **`nitrix.sparse.face_normals(vertices, faces) -> (F, 3)`** | New. Unit per-face normal `(v1−v0)×(v2−v0)/‖·‖`. **Not** the un-normalised, area-weighted cross product `compute_vertex_normals` accumulates internally — that one is an area weight, this one is unit length. Zero-area face → zero (no NaN). Pure JAX, grad-through-vertices. |
| `edge_face_adjacency(faces)` | **`nitrix.sparse.edge_face_adjacency(faces) -> (n_pairs, 2)`** | New. Host-side/static (NumPy over `faces`), sibling of `mesh_k_ring_adjacency`. Emits `(face_a, face_b)` pairs (`a < b`) for each edge with **exactly two** incident faces; boundary (1) and non-manifold (≥3) edges yield no pair. The face-pair topology a normal-consistency penalty consumes. |
| `_seg_seg_sq_dist(...)` | **`nitrix.geometry.segment_segment_sq_dist(p1, q1, p2, q2) -> (*batch,)`** | New, in `geometry._mesh_distance`. Branchless, vectorised Ericson `ClosestPtSegmentSegment` (RTCD §5.1.9), squared distance, degeneracy-safe. Pure JAX, differentiable w.r.t. all four endpoints. |
| `_nearest_sq_dist(...)` (chamfer core) | **`nitrix.geometry.point_set_nearest_sq_dist(queries, refs, *, chunk_size=None) -> (n,)`** | New, in `geometry._mesh_distance`. Per-query nearest **squared** distance to a point set (the *unreduced* chamfer core, value→value — the symmetric mean reduction stays in `nimox.loss`, §5). Pure JAX, differentiable. Dense `O(n·m)` broad phase (chunkable); the exact spatial index stays in [`mesh-spatial-acceleration.md`](../mesh-spatial-acceleration.md). |

**Home correction.** This doc's table originally guessed
`geometry.surface`/`topology`/`_triangle_distance`. The actual homes: the two
topological primitives go to `sparse.mesh` (where `Mesh`,
`compute_vertex_normals`, `mesh_k_ring_adjacency` already live), and the two
differentiable distance kernels go to a new pure-JAX `geometry._mesh_distance`.
The existing `geometry._triangle_distance.nearest_surface_distance` is **not** a
delegate for the chamfer core: it is host-side NumPy, returns an unsigned
(rooted) point→**triangle-mesh** distance, and is non-differentiable — a
different op for `cortical_thickness` / `mesh_to_sdf`.

## Asks

1. **Confirm existing equivalents.** For each row, point nimox at the nitrix
   function to delegate to (or confirm none exists yet). Face normals,
   edge–face topology, and point/segment/triangle distance all look like things
   `nitrix.geometry` either has or should.
2. **Host the gaps** as small pure kernels under the right `geometry` module,
   on the existing array conventions (so they `jit`/`vmap`/`grad` straight
   through).
3. **Acceleration is already tracked, not duplicated here.** The chamfer
   nearest-neighbour and the self-intersection candidate generation are the
   `O(n·m)` / `O(F²)` broad-phase that [`mesh-spatial-acceleration.md`](../mesh-spatial-acceleration.md)
   already owns (AI-C5, the exactness-gated spatial index). nimox ships a
   chunked brute-force interim and a precomputed-`candidate_pairs` self-
   intersection surrogate until that lands; this FR only asks for the **dense
   geometry kernels**, not the index.

## Acceptance

- A confirmed delegation map: nimox `registration.py` re-points its four
  geometry helpers at `nitrix.geometry`, keeping the loss wrappers + their
  parity tests green (the gate-reviewed mesh tests: chamfer / edge / normal-
  consistency / self-intersection, brute-force + analytic oracles).
- No numerical change (fp-parity with the current nimox implementations).

## Cross-references

- [`geometry-suite.md`](../geometry-suite.md) — the geometry ledger (dedupe here).
- [`mesh-spatial-acceleration.md`](../mesh-spatial-acceleration.md) — the broad-phase
  acceleration for the chamfer / self-intersection queries (separate, gated).
- nimox `docs/feature-requests/training-engine-impl-plan.md` (M3 mesh family) +
  the gate-review *community* lens that flagged the consolidation opportunity.
