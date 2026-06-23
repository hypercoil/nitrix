# C5 — Exact spatial broad-phase for mesh point queries → `geometry._triangle_distance` / `geometry.sphere`

> **Status (2026-06-22): design doc / FR for a deferred, correctness-sensitive
> perf item.** Audit item **AI-C5** (`geometry-suite-audit.md`): the host-side
> point-to-triangle and spherical point-in-triangle searches are brute-force
> `O(n_query · n_faces)` and a wall at recon-all / HCP resolution. Accelerating
> them is *not* a mechanical vectorisation like the rest of Tier C — a spatial
> index that prunes wrongly returns a **silently incorrect** SDF / distance /
> resample, so this is gated behind a design that fixes the **exactness
> guarantee** first. Parent: [`geometry-suite-audit.md`](geometry-suite-audit.md)
> (AI-C5, MIN-06); plan [`../design/geometry-suite.md`](../design/geometry-suite.md).
> The brute-force paths are **exact and shipped today**; this is a
> performance-only change behind the same public functions.

## 1. Why this needs a design doc

The other Tier-C items (C1–C4, C7, C8) are *parity-by-construction*: they compute
the same quantity with numpy instead of a Python loop, so equivalence is obvious.
C5 is different on two axes:

- **Correctness risk is high.** A spatial index replaces "test all triangles"
  with "test a pruned candidate set". If the pruning ever drops the true nearest
  (or containing) triangle, the result is wrong with **no error** — a too-small
  SDF magnitude, a wrong cortical thickness, a mis-routed resample weight — that
  passes a smoke test and corrupts everything downstream (myelin maps,
  registration targets, morphometry). The brute-force is exact; any accelerator
  must **prove** it returns the identical answer.
- **Scope is substantial.** Two *different* geometries need two *different*
  structures: Euclidean point-to-triangle (3-D grid, for `mesh_to_sdf` and
  symmetric `cortical_thickness`) and spherical point-in-triangle (`S²`
  bucketing, for `surface_resample`). Each needs an exact-query algorithm plus a
  vectorised implementation under the no-`scipy.spatial` constraint.

So we specify the exactness invariant, the two structures, and the parity/perf
validation up front, rather than discovering a pruning bug in production data.

## 2. The problem

Three host-side queries are brute-force `O(n_query · n_faces)`:

| Call site | Query | Cost driver |
|---|---|---|
| `_triangle_distance.nearest_surface_distance` | `n` points → nearest unsigned distance to the mesh surface | `n × n_faces` (chunked) |
| `isosurface.mesh_to_sdf` | every voxel of an `X·Y·Z` grid (via the above) | `n_voxels × n_faces` |
| `sphere._spherical_barycentric` (in `surface_resample`) | each target vertex → the source triangle containing it | `n_query × n_source_faces` |

At real scale: a `256³` SDF (~1.6·10⁷ voxels) against a ~250k-face white surface
is ~4·10¹² point-triangle tests; an fsaverage↔fs_LR resample at ico6/7 is
`n_q·n_f` in the 10¹⁰–10¹² range. These are **HOST-CTOR/QA** (run once, not
per-iteration), so not merge-unsafe — but slow enough that a user reaches for
FreeSurfer/Workbench instead, which defeats the suite's purpose at full
resolution.

The **correspondence** `cortical_thickness` (the DIFF-JAX `‖pial − white‖` path)
is already O(n) and scalable — this FR does **not** touch it; it is the
recommended route where applicable. C5 is for the cases that genuinely need a
nearest-surface / point-location query.

## 3. Scope boundary

In scope (SPEC §5): a clean-room uniform-grid spatial index + **exact**
nearest-triangle / point-location query, host-side, vectorised, returning the
**bit-identical** result of the brute-force it replaces. No public API change —
it sits *behind* `nearest_surface_distance` / `mesh_to_sdf` / `surface_resample`.

Out of scope: a KD-tree / BVH library (`scipy.spatial` is a banned runtime dep;
no new C deps); GPU spatial structures (dynamic ragged candidate gathering does
not lower cleanly to XLA — the mesh tier's host-construct→array pattern applies);
approximate-nearest (the guarantee is *exact*); differentiability (the brute-force
is host-side / non-differentiable; the accelerator preserves that contract).

## 4. Proposed surface (internal — no public API change)

```python
# geometry._triangle_distance  (Euclidean; behind nearest_surface_distance)
def _triangle_grid_index(verts, faces, *, cell=None): ...   # build the grid
def nearest_surface_distance(query, mesh, *, method='auto', ...): ...
#   method: 'auto' (grid for large n_query·n_faces, else brute) | 'grid' | 'brute'

# geometry.sphere  (spherical; behind surface_resample)
def _spherical_barycentric(verts, faces, query, *, index=None): ...
#   optional prebuilt spherical bucket index; falls back to brute per-query miss
```

The public functions (`nearest_surface_distance`, `mesh_to_sdf`,
`surface_resample`) keep their signatures; `method='auto'` selects the grid when
the problem is large enough to amortise index construction and falls back to
brute-force otherwise (and for exactness — §5). A `method='brute'` escape stays
for parity testing and pathological meshes.

## 5. Design — exactness is the invariant

### 5.1 Euclidean point-to-triangle (3-D uniform grid + expanding shell)

1. **Build** a uniform grid with cell size `h ~ k · median_edge` (`k≈1–2`); bin
   each triangle's AABB into the cells it overlaps → a CSR `cell → triangle`
   table, using the **vectorised binning already written for C4**
   (`np.repeat` + linear-index decode + combined cell key). A triangle near any
   point is therefore in a cell near that point (its AABB overlaps the local
   neighbourhood) — the property the exactness proof relies on.
2. **Query** by expanding Chebyshev shells in lockstep over all queries:
   - Each query is assigned its cell. Maintain a running `d_min` per query.
   - For ring radius `R = 0, 1, 2, …`: for queries **not yet finalised**, gather
     candidate triangles in cells within Chebyshev radius `R`, update `d_min`
     against them, and **finalise** a query once `d_min ≤ R · h`.
   - **Exactness bound:** a triangle whose AABB occupies only cells at Chebyshev
     distance `> R` from the query's cell has every point at `L∞` distance
     `≥ R · h` (the query lies inside its own cell), hence Euclidean
     `≥ R · h`. So once `d_min ≤ R · h`, no unsearched triangle can be closer —
     the query's answer is final and **exact**.
   - **Fallback (the safety net):** any query still unfinalised at `R = R_max`
     (a far query in a sparse region) falls back to a **brute-force scan over all
     triangles** — guaranteeing exactness unconditionally; the fallback set is
     small in practice (surface-proximal queries finalise at small `R`).
3. **Vectorisation:** shells are processed for all live queries together; the
   per-shell ragged candidate gather is done with the CSR + segment reductions
   (group by query, `minimum.at` over candidate distances). `_closest_point_dist2`
   (the shipped Ericson kernel) is reused unchanged for the per-candidate test.

### 5.2 Spherical point-in-triangle (`S²` bucketing)

1. **Build** a coarse spherical bucketing of the *source* faces: assign each
   source face to the buckets its corners fall in, where buckets are either an
   equirectangular lat/long grid (reusing the `sphere_grid` pole topology) or the
   vertices of a coarse icosphere (each face → its nearest coarse vertices). The
   bucket size is chosen **larger than the maximum angular triangle extent**, so
   the triangle containing any query lies in the query's bucket or an adjacent
   one.
2. **Query** each target vertex against the faces in its bucket + the
   1-ring of adjacent buckets (the `max-edge < bucket` choice makes this
   sufficient), using the existing min-signed-edge-distance containment test.
3. **Fallback:** if the local buckets yield no containing triangle (a clipped
   barycentric / oversized triangle / pole seam), that query falls back to a
   brute-force search over all source faces — exact containment guaranteed.

### 5.3 The non-negotiable: parity, not "close enough"

Both structures are validated by **bit-exact parity** against the brute-force on
real meshes (§7), not by a tolerance. The `method='brute'` path is retained
precisely so the grid path can be diffed against it in tests and, if a user ever
distrusts the index on a pathological mesh, at the call site.

## 6. Substrate reuse

| Need | Reuse |
|---|---|
| Vectorised AABB→cell binning, combined cell key | the C4 `_candidate_pairs` pattern (`intersection.py`) |
| Per-candidate point-to-triangle distance | `_triangle_distance._closest_point_dist2` (Ericson, unchanged) |
| Spherical containment + barycentric | `sphere._spherical_barycentric` core (unchanged inner test) |
| Spherical bucketing topology | `sphere_grid` (equirectangular pole/seam) or `icosphere` (coarse-vertex buckets) |
| CSR group-by / segment reductions | numpy `argsort` + `bincount` + `np.minimum.at` (the Tier-C idiom) |

Net-new: the grid-index builder, the expanding-shell exact query, the spherical
bucket index. No new runtime dependency; all host-side numpy.

## 7. Validation / test plan

1. **Exact parity (the keystone).** On the icosphere, an analytic sphere SDF, a
   real fsaverage white/pial pair, and fs5↔fs4 spheres: `method='grid'` returns
   distances / containing-triangles / resample weights **identical** to
   `method='brute'` (float-exact, modulo deterministic tie-breaks on exactly
   equidistant triangles — which must be made deterministic and matched).
2. **Fallback coverage.** A constructed case where a query finalises only via the
   `R_max` brute fallback (a far point / sparse region) still returns the exact
   answer; a spherical query whose containing triangle spans a bucket boundary is
   found.
3. **Analytic oracle.** Sphere SDF still matches `‖x‖ − r`; `mesh_to_sdf` round-
   trips with `marching_cubes`; `surface_resample` identity + integral
   conservation hold (unchanged from the brute path).
4. **Perf demonstration** (the point of the change): a recorded ico6 / mid-grid
   wall-clock improvement vs brute — reported via `log`, not asserted as a flaky
   timing gate (no `@perf` marker exists in the repo; a perf-harness is its own
   infra task).

## 8. Risks

| Risk | Severity | Mitigation |
|---|---|---|
| Pruning drops the true nearest/containing triangle → silent wrong result | **High** | the expanding-shell exactness bound + the unconditional brute-force fallback; **bit-exact parity tests** vs brute on real meshes gate the merge |
| Cell-size heuristic degenerate on highly non-uniform meshes (huge AABB triangles span many cells) | Med | `cell` is overridable; a triangle spanning many cells still binned correctly (just more memberships); the fallback covers the pathological tail |
| Expanding-shell vectorisation complexity / off-by-one in the bound | Med | derive the bound explicitly (`≥ R·h`, §5.1); test against brute including the boundary cases; keep `method='brute'` as the reference |
| Grid memory for a huge bounding box / tiny `h` | Low–Med | CSR (sparse) membership, not a dense grid; cap `h` from the median edge; document the memory model |
| Tie-breaking differs from brute on equidistant triangles | Low | make both paths choose the lowest face index on ties so parity is bit-exact |
| Spherical pole/seam misses a containing triangle | Med | bucket > max angular edge + adjacent-bucket search + per-query brute fallback |

## 9. Phasing

- **C5a — Euclidean grid + exact expanding-shell nearest-triangle** in
  `_triangle_distance`; wire `method=` into `nearest_surface_distance` (and hence
  `mesh_to_sdf` + symmetric `cortical_thickness`). Exit: bit-exact parity vs
  brute on a real white surface + analytic sphere SDF; the fallback path tested.
  **DONE 2026-06-23** (branch `perf/mesh-spatial-acceleration`): `method=`
  `'auto'`/`'grid'`/`'brute'`; the exactness bound (`d_min <= R*cell`) +
  unconditional brute fallback at `r_max` hold; **bit-exact (`np.array_equal`)
  vs brute** on icosphere + real fsaverage (near/far/fallback) + analytic
  sphere; real `cortical_thickness` (corr 0.94 vs FS) and `mesh_to_sdf`
  consumers stay green. Note: it accelerates near-surface queries (thickness,
  SDF near-band); far SDF-grid voxels fall back to brute (exact, no worse) — a
  narrow-band / sweep SDF for the far field is a separate future enhancement.
- **C5b — spherical bucket index** for `surface_resample._spherical_barycentric`.
  Exit: bit-exact parity vs brute on fs5↔fs4; bucket-boundary + fallback tested.
- **C5c — validation + perf record.** The §7 matrix; a logged perf comparison.

Each phase keeps `method='brute'` as the reference and commits independently with
its parity test.

## 10. Governance

§13 four-gate: **consumer** — recon-all / HCP at full surface/grid resolution
(the named blocked use; the brute paths are the current bottleneck);
**composition** — entirely on shipped primitives (the C4 binning idiom + the
Ericson kernel + the spherical containment test); **SoC** — an *internal*
accelerator behind unchanged public functions (no new API surface, so not a §14
kwarg-vs-fork concern beyond the internal `method=`); **effort** — M–L. Because
it changes *no* public behaviour (only speed) and is gated by **bit-exact parity
tests**, it does not need a SPEC-level review like GS-2/GS-11 — but it does need
this design fixed first given the correctness stakes. On completion, record the
AI-C5 closure in `geometry-suite-audit.md` and an as-built note (chosen `h`
heuristic, fallback rate on real data, measured speed-up) in
[`../design/geometry-suite.md`](../design/geometry-suite.md) §8.

## 11. Cross-references

- Audit ledger: [`geometry-suite-audit.md`](geometry-suite-audit.md) (AI-C5,
  MIN-06); design plan [`../design/geometry-suite.md`](../design/geometry-suite.md).
- Substrate: `geometry/_triangle_distance.py` (`_closest_point_dist2`,
  `nearest_surface_distance`), `geometry/isosurface.py` (`mesh_to_sdf`),
  `geometry/sphere.py` (`_spherical_barycentric`, `surface_resample`),
  `geometry/intersection.py` (the C4 vectorised binning idiom),
  `geometry/sphere_grid.py` (equirectangular topology).
- Governance: `SPEC.md §5` (no `scipy.spatial` runtime dep),
  `SPEC_UPDATE_v0.3.md §13`.
