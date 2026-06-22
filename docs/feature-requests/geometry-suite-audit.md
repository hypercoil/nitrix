# Geometry suite — pre-merge audit findings & action ledger

> **Status (2026-06-22): pre-merge audit record.** A 7-axis static review of the
> cortical-surface geometry suite (Phases 0–5) + the sparse substrate, run as a
> read-only multi-agent fan-out (mathematical correctness, engineering rigour,
> community value, consumer/user ergonomics, performance, code organisation,
> hardware/GPU). This document is the durable record: the verdict, the complete
> indexed finding set (47), the consolidated action items mapped to remediation
> tiers, and what is intentional / deferred. Parent:
> [`geometry-suite.md`](geometry-suite.md); plan:
> [`../design/geometry-suite.md`](../design/geometry-suite.md). Branch
> `feat/geometry-suite-phase0` (not pushed).

## 1. Verdict

**Merge-readiness: READY-WITH-MINOR-FIXES. Zero blockers.**

The mathematics was verified correct formula-by-formula (cotangent stiffness,
Meyer mixed-Voronoi mass, Gauss–Bonnet angle defect, Van Oosterom–Strackee
signed solid angle, spectral shift-invert recovery `λ=c(1−v)`, the
`adap_bary_area` partition-of-unity conservation proof, Cauchy–Green strain,
Ericson point-to-triangle, marching-tetrahedra dedup, `mesh_to_sdf`
winding-number sign). No wrong-result defect. The cuSolver-free discipline holds
(matrix-free CG, closed-form 2×2 strain eigensolve, LOBPCG shift-invert — the
suite survives this box's dead cuSolver pool *by design*). The
DIFF-JAX/HOST-CTOR/HOST-QA execution-class discipline is honoured (no HOST-QA op
inside a jitted loop; self-intersection cleanup is strictly post-hoc). The
`apply_operator` seam is genuinely consumed (algorithms do not branch on storage
format). Dependency direction (`geometry`/`graph` → `sparse`/`semiring`) holds.

**Counts:** 0 blocker · 10 major · 25 minor · 12 nit = 47 findings. The 10 majors
consolidate to 5 distinct themes (several axes flagged the same issue).

| Axis | Verdict |
|---|---|
| Mathematical correctness | Strong — no blocker/major numerical defect; only doc/contract nuances. |
| Engineering rigour | High — pervasive guards + validation; gaps in jit/grad coverage of `spherical_parameterize` and some degenerate-input guards. |
| Community needs & value | High value; two real gaps (medial-wall ROI mask; recon-all-clinical GS-7 hole) + host-side perf cliffs. |
| Consumer/user ergonomics | Coherent API; friction in enum typing + the `deform_to_sdf` boundary-kwarg inconsistency. |
| Suite performance | Hot paths jitted correctly; **construction-time host-side Python loops are the systemic risk at ico7** (all HOST-CTOR/QA, not merge-unsafe). |
| Code organisation & design | Sound seams; debt = three cotangent impls, duplicated VOS formula, an unused mass-matrix seam. |
| Hardware/GPU | Boundary drawn correctly; cuSolver-free holds; over-stated jit/diff claims on `spherical_parameterize`. |

## 2. Consolidated action-item index

Each action item (AI-*) consolidates one or more raw findings (§7). Tier A =
quick wins (this branch); Tier B = substantive fixes (this branch); Tier C =
deferred perf follow-up; INT = intentional/documented (doc-only or no action).

| AI | Action | Tier | Sev | Source findings | Files |
|---|---|---|---|---|---|
| **A1** | `spherical_parameterize` orientation flip → jit-safe `jnp.where`; clarify DIFF-JAX contract (refine loop jittable; spectral init host-side); add jit+grad test on the refine path | A | major | M-04, M-05, MIN-25 | geometry/sphere.py |
| **A2** | Fix stale "diagonal as LAST column" comment → column 0; assert diagonal location in `spectral_sphere_embedding` instead of hard-coding `[:,0]` | A | minor | MIN-19 | sparse/mesh.py, geometry/sphere.py |
| **A3** | Bare-`str` enum kwargs → `Literal[...]` suite-wide | A | minor | MIN-08 | surface.py, sphere.py, parcellation.py |
| **A4** | `deform_to_sdf`: expose `mode`/`cval`; inline comment that interpolation stays `Linear` (diff holds); grad-w.r.t.-vertices test | A | major | M-03, NIT-06, NIT-09 | geometry/surface.py |
| **A5** | `gaussian_curvature`: `arccos(clip)` → `arctan2(‖cross‖, dot)` (smooth gradient at slivers; identical forward) | A | minor | MIN-16 | geometry/surface.py |
| **A6** | Remove dead antipodal `jnp.where` branch in `_geodesic_pair`; fix comment | A | nit | NIT-04 | geometry/sphere.py |
| **A7** | Docstring tightenings (see §3) | A | minor/nit | MIN-01, MIN-07, MIN-20, MIN-21, NIT-02, NIT-08, NIT-10, MIN-13, MIN-09 | surface.py, sphere.py |
| **A8** | As-built record: marching-tetrahedra vs locked asymptotic-decider; GS-7 gate-only limitation note | A | minor | MIN-05, M-02 | design doc §5, isosurface.py, topology.py |
| **A9** | Reconcile design-doc/FR `spherical_parameterize` signature (`distance_weight`→`conformal_weight` + new kwargs) | A | minor | MIN-10 | design doc, FR |
| **A10** | VOS solid-angle: cross-reference comment between the two impls (full extraction → C10) | A | minor | MIN-02 | sphere.py, isosurface.py |
| **B1** | `surface_smooth` medial-wall **ROI mask** (Neumann at boundary) + masked test; extend ROI to `surface_boundary_map`/`mesh_watershed` | B | major | M-01 | surface.py, parcellation.py |
| **B2** | `spherical_parameterize`: build cotangent Laplacian **once**, pass into `spectral_sphere_embedding` (was built twice/call) | B | major | M-06(part), M-08(part) | sphere.py |
| **B3** | Empty/degenerate-mesh guards on curvature/area entry points + test | B | minor | MIN-12 | surface.py |
| **B4** | Hand-built obtuse-triangle `vertex_areas` partition test | B | nit | NIT-07 | tests |
| **C1** | Vectorise `mesh_cotangent_laplacian` assembly (np.add.at on (i,j,w) triples) | C | major | M-06 | sparse/mesh.py |
| **C2** | Vectorise `surface_resample adap_bary_area` scatter (np.add.at; bincount pack) | C | major | M-07, MIN-04, MIN-15, MIN-17, MIN-24 | sphere.py |
| **C3** | Vectorise watershed `_neighbour_lists`/`is_min`; saddle-heap merge | C | major | M-10 | parcellation.py |
| **C4** | Vectorise self-intersection broad phase; reuse binning across iters | C | major | M-09, MIN-18, NIT-01 | intersection.py |
| **C5** | Broad-phase (uniform-grid/bucket) for `mesh_to_sdf`/`nearest_surface_distance`/spherical search | C | minor | MIN-06 | _triangle_distance.py, isosurface.py, sphere.py |
| **C6** | `spherical_parameterize` refine: fuse areas/energy passes; bound line-search halvings | C | major | M-08 | sphere.py |
| **C7** | `marching_cubes`: `np.unique(axis=0)` → 1-D combined-key unique | C | minor | MIN-22 | isosurface.py |
| **C8** | `spherical_conv` int-`k`: tile top-k via `lax.map` or hard-raise above a threshold | C | minor | MIN-23 | sphere.py |
| **C9** | Default algorithm-layer callers to `format='auto'` for irregular real meshes | C | nit | NIT-12 | sphere.py, surface.py |
| **C10** | Extract one shared VOS solid-angle core (apex-parameterised) | C | minor | MIN-02 | sphere.py, isosurface.py |
| **C11** | Extract one shared `_face_cotangents` kernel (collapse 3 cot impls to 1) | C | minor | MIN-03 | sparse/mesh.py, surface.py |
| **INT-1** | GS-7 genus-0 corrector deferred = D1 locked decision (gate-only; template+`deform_to_sdf` is the supported genus-0 path) — documented, not built | INT | major | M-02 | topology.py |
| **INT-2** | `mesh_mass_matrix` operator-seam unused — demote §2.4 claim to reality (consumed as the `vertex_areas` vector) | INT/A7 | minor | MIN-01 | mesh.py |
| **INT-3** | `surface_resample` returns `(operator, field)` — documented operator-reuse contract; keep | INT | nit | NIT-03 | sphere.py |
| **INT-4** | `ribbon_map` `sigma` ignored for `weighting='pv'` — documented "gaussian only"; keep | INT | minor | MIN-11 | surface.py |
| **INT-5** | `gaussian_curvature` closed-mesh 2π defect; `principal_curvatures` disc clamp; coplanar self-intersection miss — documented restrictions | INT | nit | NIT-05, NIT-11, MIN-14 | surface.py, intersection.py |

## 3. Tier A — quick wins (this branch)

Low-risk, high-clarity. **A1** (jit-safe `jnp.where` orientation flip) resolves
the single most-corroborated finding (4 axes) cheaply: it removes the host sync
so the `radial` init + refine loop are jittable in isolation, and the
contract is restated precisely (the `spectral` init remains host-side and is
correctly labelled non-differentiable). **A5** swaps `arccos(clip)` for
`arctan2(‖cross‖, dot)` — identical forward value, smooth gradient at the
near-collinear slivers real cortex produces. **A7 docstring set:** `ribbon_map`
("midpoint column mean; exact for constant/linear-along-column, midpoint-accurate
otherwise"; + the `-ribbon-constrained` voxel-membership divergence); `surface_resample`
("conservation is w.r.t. the *supplied* `source_area`/`target_area`"; + an
up-sampling-hole transparency note); `strain_distortion` ("isometry → (1,1) only
to ~1e-6 from the discriminant floor"); `mesh_mass_matrix` (demote the
operator-seam claim — the lumped mass is consumed as the `vertex_areas` vector);
`spectral_sphere_embedding` (on a wedged dense-solver stack the LOBPCG
orthonormalisation runs on CPU via the solver-device fallback); `mesh_watershed`
(plateau/tie boundary assignment is arbitrary-but-stable, index-ordered); the
operator-layer (array-in) vs algorithm-layer (Mesh-in) convention.

## 4. Tier B — substantive fixes (this branch)

- **B1 — `surface_smooth` ROI / medial-wall mask** (the top community gap; the
  plan §4 P1.3 promised it). Add `roi: Bool[Array,'n_vertices'] | None`; zero the
  cotangent coupling on edges crossing the ROI boundary (Neumann at the medial
  wall) before the CG solve so values do not bleed across non-cortex. Ship the
  promised masked-medial-wall real-mesh test. Apply the same ROI restriction to
  `surface_boundary_map` / `mesh_watershed`.
- **B2 — build the cotangent Laplacian once per `spherical_parameterize` call.**
  Today the host-side assembly runs twice (the refine energy + the spectral init
  via `spectral_sphere_embedding`). Let `spectral_sphere_embedding` accept a
  prebuilt operator (or compute once and reuse) — halves the worst host cost on
  the hardest primitive without touching the (deferred) vectorisation.
- **B3 — empty/degenerate-mesh guards** on the curvature/area entry points (an
  empty `marching_cubes` output currently yields meaningless zeros, not a clear
  error) + a regression test.
- **B4 — the planned hand-built obtuse-triangle test** for the `vertex_areas`
  mixed-Voronoi partition (the obtuse branch is load-bearing on real cortex and
  only validated indirectly today).

## 5. Tier C — deferred to a perf follow-up

The systemic theme: **host-side construction is written element-at-a-time in
Python** and is a wall at ico7 (~163k vertices). All are correctly HOST-CTOR/QA
(never inside a jitted loop) so **none is merge-unsafe**, and all share the same
remedy — numpy vectorisation (`np.add.at` on stacked triples; combined-key
`np.unique`; bincount packing) + uniform-grid/bucket broad-phases. C1–C11 above.
This is a sizable, separately-testable effort (each needs its own parity test
against the current Python path at ico5 and a `@perf`-marked ico6/ico7 bound), so
it is deferred rather than rushed under the current memory constraints. Until
then, the practical vertex/voxel ceilings are documented in the affected
docstrings (A7).

## 6. Strengths (recorded)

- Mathematics verified correct formula-by-formula; no blocker/major numerical
  defect — incl. the subtle `adap_bary_area` conservation proof and the spectral
  `λ=c(1−v)` recovery.
- cuSolver-free by design — survives this box's dead cuSolver pool (CG, closed-form
  2×2 strain eigensolve, LOBPCG shift-invert), not by accident.
- DIFF-JAX / HOST-CTOR / HOST-QA discipline genuinely observed (§2.5 rule held;
  self-intersection cleanup strictly post-hoc).
- The `apply_operator` seam is actually consumed by the new algorithm layer
  (`surface_smooth`, `spherical_parameterize`, `surface_resample`) — no
  isinstance-branching on storage format.
- Pervasive deliberate numerical guards + specific guiding `ValueError`s; fp64
  host assembly for cotangent/area.
- Coherent, well-documented API tracking community vocabulary
  (`mris_inflate`→`inflate_surface`, `mris_sphere`→`spherical_parameterize`,
  `-ribbon-constrained`→`ribbon_map`, `BARYCENTRIC`/`ADAP_BARY_AREA`→`method=`).
- DRY where it matters: `_triangle_distance` shared by `mesh_to_sdf` and
  `cortical_thickness`.

## 7. Appendix — complete finding index (47)

Stable IDs; `→AI` maps each to its consolidated action item.

| ID | Sev | Axis | Title | →AI |
|---|---|---|---|---|
| M-01 | major | community | `surface_smooth` no medial-wall/ROI mask | B1 |
| M-02 | major | community | recon-all-clinical GS-7 corrector deferred, no substitute | INT-1 |
| M-03 | major | ergonomics | `deform_to_sdf` hardcodes `mode='nearest'`, no control | A4 |
| M-04 | major | engineering | `spherical_parameterize` DIFF-JAX claim vs host control flow, untested jit/grad | A1 |
| M-05 | major | hardware | `spherical_parameterize` not jittable/end-to-end-diff (host sync + host init) | A1 |
| M-06 | major | performance | `mesh_cotangent_laplacian` Python-dict assembly; built 2×/`spherical_parameterize` | B2 + C1 |
| M-07 | major | performance | `surface_resample adap_bary_area` nested Python loops/dicts | C2 |
| M-08 | major | performance | `spherical_parameterize` recomputes signed areas ~4×/iter; host Laplacian | B2 + C6 |
| M-09 | major | performance | self-intersection broad phase Python triple loop + set | C4 |
| M-10 | major | performance | watershed neighbour-lists/minima/flood Python loops | C3 |
| MIN-01 | minor | design | `mesh_mass_matrix` seam exported/tested but unused by algorithms | A7/INT-2 |
| MIN-02 | minor | design | VOS solid-angle duplicated (sphere.py vs isosurface.py) | A10 + C10 |
| MIN-03 | minor | design | three cotangent implementations coexist | C11 |
| MIN-04 | minor | design | `adap_bary_area` Python dict assembly (perf-transparency) | C2 |
| MIN-05 | minor | community | marching-tetrahedra shipped, not the locked asymptotic-decider | A8 |
| MIN-06 | minor | community | host-side O(n·F) distance/SDF/resample perf cliff | A7(doc) + C5 |
| MIN-07 | minor | community | `surface_resample` up-sampling silently mixes regimes | A7(doc) |
| MIN-08 | minor | ergonomics | enum kwargs typed bare `str` not `Literal` | A3 |
| MIN-09 | minor | ergonomics | Mesh-in vs (vertices,faces)-in convention split | A7(doc) |
| MIN-10 | minor | ergonomics | `spherical_parameterize` kwarg names drifted from doc/FR | A9 |
| MIN-11 | minor | ergonomics | `ribbon_map` `sigma` ignored for `weighting='pv'` | INT-4 |
| MIN-12 | minor | engineering | curvature/area unguarded on empty/single-vertex meshes | B3 |
| MIN-13 | minor | engineering | watershed tie-break non-determinism (index-ordered) | A7(doc) |
| MIN-14 | minor | engineering | coplanar self-intersection silently skipped | INT-5 |
| MIN-15 | minor | engineering | `adap_bary_area` Python loop untested at ico7 | C2 |
| MIN-16 | minor | hardware | `gaussian_curvature` `arccos` gradient blows up at slivers | A5 |
| MIN-17 | minor | hardware | `adap_bary_area` pure-Python assembly perf cliff | C2 |
| MIN-18 | minor | hardware | self-intersection broad phase 4-deep Python loop | C4 |
| MIN-19 | minor | math | contradictory cotangent-diagonal-column docs (code correct) | A2 |
| MIN-20 | minor | math | `ribbon_map` midpoint quadrature is a column mean, not endpoint-inclusive | A7(doc) |
| MIN-21 | minor | math | `adap_bary_area` conservation exact only for the supplied area vectors | A7(doc) |
| MIN-22 | minor | performance | `marching_cubes` `np.unique(axis=0)` scaling | C7 |
| MIN-23 | minor | performance | `_spherical_knn_indices` dense (n,n) matrix (documented) | A7(doc) + C8 |
| MIN-24 | minor | performance | `compute_vertex_normals` recomputed per iter (correct, noted) | INT (none) |
| MIN-25 | minor | performance | `is_bijective_sphere_map`/orientation host syncs | A1 |
| NIT-01 | nit | design | self-intersection broad-phase Python loops | C4 |
| NIT-02 | nit | community | `ribbon_map` no cortical-ribbon voxel masking | A7(doc) |
| NIT-03 | nit | ergonomics | `surface_resample` returns (operator, field) order | INT-3 |
| NIT-04 | nit | engineering | `_geodesic_pair` dead antipodal-wrap branch | A6 |
| NIT-05 | nit | engineering | `principal_curvatures` disc clamp masks invalid curvature | INT-5 |
| NIT-06 | nit | engineering | `deform_to_sdf` `mode='nearest'` vs diff claim | A4 |
| NIT-07 | nit | engineering | no direct obtuse-triangle `vertex_areas` test | B4 |
| NIT-08 | nit | hardware | `spectral_sphere_embedding` runs on CPU on broken cuSolver | A7(doc) |
| NIT-09 | nit | hardware | `deform_to_sdf` `mode='nearest'` reads like NN interpolation | A4 |
| NIT-10 | nit | math | `strain_distortion` disc floor splits equal stretches ~1e-6 | A7(doc) |
| NIT-11 | nit | math | `gaussian_curvature` no boundary guard (closed-mesh) | INT-5 |
| NIT-12 | nit | performance | auto-section opt-in; default `'ell'` pads high-valence | C9 |

## 8. Cross-references

- Parent: [`geometry-suite.md`](geometry-suite.md); plan +as-built:
  [`../design/geometry-suite.md`](../design/geometry-suite.md); SPEC-review FRs:
  [`spherical-parameterisation.md`](spherical-parameterisation.md),
  [`place-surface.md`](place-surface.md).
- Governance: `SPEC_UPDATE_v0.3 §13` (acceptance), `§14` (kwarg-not-fork),
  `IMPLEMENTATION_PLAN.md §10` (graduations at merge).
