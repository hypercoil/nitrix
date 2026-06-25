# nimox mesh-loss geometry → `nitrix.geometry` (consolidation handoff)

> **Status (2026-06-25): handoff / request (nimox axis-iv → nitrix).** The
> nimox training-engine mesh-loss family (`nimox.loss.functional.registration`,
> landed on nimox `main`) hosts several **pure, model-agnostic mesh-geometry
> kernels** that are not loss-specific and almost certainly belong in (or
> already exist under) `nitrix.geometry`. This asks the geometry ledger to
> **confirm the existing equivalents so nimox can delegate** (mirroring the
> `metrics` → nitrix delegation it already did via
> [`classification-surface-metrics`]), and to **host the genuine gaps**. The
> nimox loss-shaped wrappers stay in nimox; only the geometry substrate moves.
>
> Per `README.md` / [`geometry-suite.md`](geometry-suite.md) this is filed as a
> **consumer pointer into the existing ledger**, not a parallel kernel spec —
> dedupe against `surface.py` / `topology.py` / `_triangle_distance.py` /
> `intersection.py`, which look like the natural homes.

## What nimox currently hosts (and the candidate nitrix home)

| nimox primitive (`loss/functional/registration.py`) | What it is | Candidate nitrix home |
|---|---|---|
| `_face_normals(verts, faces)` | unit triangle normals via edge cross-product | `geometry.surface` / `differential` |
| `edge_face_adjacency(faces)` | host-side shared-edge → face-pair topology (static) | `geometry.topology` |
| `_seg_seg_sq_dist(...)` | Ericson clamped segment–segment squared distance | `geometry._triangle_distance` |
| `_nearest_sq_dist(...)` (chunked) | point-set nearest-neighbour squared distance (the chamfer core) | `geometry._triangle_distance` + the broad-phase ([`mesh-spatial-acceleration.md`](mesh-spatial-acceleration.md)) |

The **loss reductions** — `chamfer_surface_distance`, `mesh_edge_smoothness`,
`mesh_normal_consistency`, `mesh_self_intersection_penalty` — stay in
`nimox.loss` (they encode the *loss* shape / unscalarised convention) and would
**delegate** the geometry above, exactly as the seg/surface metrics delegate.

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
   `O(n·m)` / `O(F²)` broad-phase that [`mesh-spatial-acceleration.md`](mesh-spatial-acceleration.md)
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

- [`geometry-suite.md`](geometry-suite.md) — the geometry ledger (dedupe here).
- [`mesh-spatial-acceleration.md`](mesh-spatial-acceleration.md) — the broad-phase
  acceleration for the chamfer / self-intersection queries (separate, gated).
- nimox `docs/feature-requests/training-engine-impl-plan.md` (M3 mesh family) +
  the gate-review *community* lens that flagged the consolidation opportunity.
