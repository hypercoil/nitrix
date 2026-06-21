# Geometry suite — implementation plan

> **Status (2026-06-21): pre-build implementation plan.** The *how* for the
> cortical-surface geometry suite. The *what / why / ledger of items* lives in
> the feature request
> [`docs/feature-requests/geometry-suite.md`](../feature-requests/geometry-suite.md)
> (GS-1…14 + adopted §12 items); read it first. This doc fixes the build order,
> the module seams, the per-task contracts (signatures, differentiability
> class, composition), the test matrix (icosphere analytic oracle **and** real
> FreeSurfer / fs_LR mesh), and the governance / graduation records. No code is
> written yet.

## 0. How to read this

- **§1** — the inherited non-negotiables (the contract every task holds to).
- **§2** — the architectural decisions, locked. Read before any task.
- **§3–§8** — Phases 0–5: per-task contract, composition, differentiability
  class, tests, effort, exit criterion.
- **§9** — the cross-cutting test strategy (the real-mesh harness).
- **§10** — governance: the §13 gate status per item and the §12→§10.A
  graduation records to write at merge.
- **§11** — dependency graph & sequencing. **§12** — risk register.
  **§13** — cross-references.

This plan is **prescriptive about contracts** (what "done" means per task) and
**permissive about sequencing within a phase** (tasks in a phase are mostly
parallel; the cross-phase edges in §11 are the real constraints), mirroring
`IMPLEMENTATION_PLAN.md`'s discipline.

## 1. Inherited non-negotiables

From SPEC §2 / §5 and `IMPLEMENTATION_PLAN.md §2.2`, holding for every task:

1. **Dependency contract.** Runtime imports stay `jax` / `jaxtyping` / `numpy`
   only. No `scipy` / `nibabel` / `nilearn` / `templateflow` at runtime — those
   are **test-only** (§9). Host-side combinatorial construction that emits plain
   arrays / `ELL` is in scope (the `icosphere` / `mesh_k_ring_adjacency`
   idiom); IO and container/atlas resolution are **not** (→ `thrux`).
2. **Pure-functional API.** Every public symbol is `(arrays, …) -> arrays` (or
   a frozen, pytree-registered container of arrays). No modules, no mutable
   state.
3. **JAX-fallback floor.** The mesh tier lowers onto `semiring_ell_matmul`,
   which is **XLA/JAX, not Pallas** at GA (Triton cannot lower `lax.gather`,
   per [`ell-on-triton.md`](ell-on-triton.md)). Plan for XLA-lowered
   gather/scatter + `fori_loop`; do **not** promise custom kernels for this
   tier (§2.6).
4. **Golden corpus + real-mesh.** Pinned numerical references in `tests/`; **and**
   a real-FS-mesh check per Phase-0/1 primitive (§9).
5. **Loud fallbacks.** Any divergence from a reference (e.g. GS-12 vs
   Workbench), any non-differentiable path, any host-side stage is announced in
   the docstring; silent degradation is a defect.
6. **Backward-compatible.** No break to a shipped symbol; extend via a kwarg,
   never a fork (SPEC_UPDATE_v0.3 §14). New behaviour is opt-in.

## 2. Architectural decisions (locked)

### 2.1 The `sparse.mesh` ↔ `geometry.surface` seam

One organising rule, to end the "areas in `sparse.mesh` but curvature in
`geometry.curvature`" inconsistency in the ledger:

> **`sparse.mesh`** = the **operator / measure layer** — assembly of ELL /
> SectionedELL operators (cotangent Laplacian, k-ring, mass) and the
> per-element measures those operators need (`face_areas`, `vertex_areas`,
> `compute_vertex_normals`). **`geometry.surface`** (new) = surface
> **algorithms** that compose those operators (curvature, distortion, geodesic
> smoothing, thickness, inflation, SDF-march, ribbon map).

`geometry` may depend on `sparse`; never the reverse. `Mesh` stays in
`sparse.mesh` (moving it is needless churn; algorithms import it). New modules
to create (all confirmed absent): `geometry/surface.py`, `geometry/topology.py`,
`geometry/isosurface.py`.

### 2.2 The `{ELL, SectionedELL}` apply-seam

A single dispatch so the algorithm layer never branches on operator storage,
mirroring the `linalg._eigsolve` / `Interpolator` "factor the independent axes"
discipline:

```python
# nitrix.sparse  (new, thin)
def apply_operator(op: ELL | SectionedELL, x: Num[Array, '... n d'], *,
                   semiring: Semiring = REAL) -> Num[Array, '... m d']: ...
```

Dispatch: `ELL → semiring_ell_matmul`, `SectionedELL →
sectioned_semiring_ell_matmul`. Curvature, distortion, GS-12, DEC all call
`apply_operator`, so the icosphere (plain `ELL`) and a `recon-all` white
surface (`SectionedELL`) flow through identical algorithm code.

### 2.3 PyTree registration (B22)

Mirror `smoothing/metric.py`'s `@jax.tree_util.register_pytree_node_class` +
`tree_flatten`/`tree_unflatten`. **Arrays are children; hashable scalars are
aux.**

| Type | children | aux (static, hashable) | rationale |
|---|---|---|---|
| `ELL` | `(values, indices)` | `(n_cols, identity)` | `grad` populates `values`, zeros `indices` — consistent with the existing `eigsolve` `custom_vjp`. Indices as a child (not aux) supports `vmap` over a stack of same-pattern ELLs and avoids forcing a large unhashable int array into the jit cache key. |
| `Mesh` | `(vertices, faces)` | `()` | Diff w.r.t. `vertices`; `faces` as a child carries through with zero cotangent. Faces-as-aux would force a huge unhashable int array static. |
| `SectionedELL` | per-section `ELL` children | section structure (`row_groups` as a hashable tuple) + `(n_rows, n_cols, identity)` | nested but mechanical once `ELL` is registered. |
| `IcosphereHierarchy` | — | — | **Consciously declined.** `parents` are host-side NumPy construction metadata (np arrays → unhashable aux); the hierarchy is consumed to *build* ELLs/meshes on host *before* tracing, so it never legitimately crosses a `jit` boundary. |

Risk to check at build (from B22): the `eigsolve` `custom_vjp` already unpacks
internally — confirm no double-handling and that `jax.grad(f)(ell)` and the
internal VJP compose to the same cotangent. Sweep for any `tree_map` over a
structure holding an `ELL` (the silent-leaf trap).

### 2.4 Mass-matrix contract (GS-5)

The cotangent operator stays **unnormalised stiffness `L`**; the mass `M` is a
*separate* symbol (`mesh_mass_matrix`), not a normalisation flag baked into the
Laplacian. Algorithms form `M⁻¹L` (curvature) or `M + tL` (smoothing) at the
algorithm layer via the apply-seam. Ship the **lumped** (diagonal) `M` first
(`vertex_areas` on the diagonal → trivially invertible). The consistent
(non-lumped) FEM mass is a later opt-in (`lumped=False`) if a consumer needs it.
`scheme='voronoi'` (Meyer mixed, obtuse-safe) is the default; `'barycentric'`
is the cheap fallback.

### 2.5 Differentiability & execution classes

Every public symbol is tagged with one of three classes, stated in its
docstring (non-negotiable 5):

- **DIFF-JAX** — jittable, differentiable kernel. (areas, mass, curvature,
  distortion, GS-12, GS-10, correspondence-thickness, ribbon-map, inflate /
  parameterise forward.)
- **HOST-CTOR** — host-side construction emitting `Mesh`/`ELL` (data-dependent
  shape). (marching cubes, euler/genus, watershed, ADAP_BARY assembly,
  SectionedELL emission.)
- **HOST-QA** — host-side analysis / query, not in a differentiable path.
  (symmetric thickness, self-intersection, BVH/uniform-hash queries.)

The load-bearing rule: a **HOST-CTOR/QA op must never be called inside a jitted
`fori_loop`** (the GS-10/GS-8 correction). Optimisers regularise with DIFF-JAX
ops only; HOST cleanups run before/after.

### 2.6 GPU posture

The mesh tier is XLA, not Pallas (§1.3). That is the substrate bet and it is
fast (XLA-lowered gather/scatter + `fori_loop` on GPU). Do not open a Pallas
kernel for any mesh op in this suite; if a profile later demands one, it goes
through the `internal-backlog.md` Trigger process, not this plan.

### 2.7 Real-FS-mesh validation harness (test-only)

See §9. The principle: real-mesh IO is test-only and *is itself* a test of the
`icosphere_hierarchy_from_levels` array-handoff contract.

---

## 3. Phase 0 — substrate & enablers

*Goal: make the rest correct (mass), differentiable (pytree), efficient on real
meshes (SectionedELL), and continuously validated (fixture). All low risk.*

### P0.1 — PyTree lift (B22) · DIFF-JAX enabler · Effort S
- Register `ELL`, `SectionedELL`, `Mesh` per §2.3; decline `IcosphereHierarchy`
  (document the decline in its docstring).
- **Tests:** `tree_flatten`/`unflatten` round-trip each type; `jit` and `vmap`
  with each as an argument; `vmap` over a stack of same-pattern `ELL`s;
  `jax.grad` w.r.t. an `ELL` returns an `ELL`-structured cotangent (values
  populated, indices zero) agreeing with the `eigsolve` `custom_vjp`; a
  `tree_map` over `{params, ell}` touches `ell.values` (the trap is gone).
- **Exit:** a jitted `f(mesh) -> mesh` works; the perf-bench unpack/repack
  workaround (`_eigenmap.py`) can be deleted.

### P0.2 — areas + lumped mass (GS-5) · DIFF-JAX · Effort S
- `face_areas`, `vertex_areas(scheme=...)`, `mesh_mass_matrix(scheme=, lumped=)`
  per §2.4. Home `sparse.mesh`.
- **Tests:** icosphere — `Σ vertex_areas == 4πr²` and `== Σ face_areas` to tol;
  barycentric ≈ voronoi on near-equilateral; a hand-built **obtuse** triangle —
  the mixed rule gives the correct Voronoi/barycentric partition (the branch
  the icosphere never exercises). Real mesh — `Σ vertex_areas ≈ Σ FS ?h.area`
  and per-vertex Spearman vs `?h.area` above a pinned floor.
- **Exit:** curvature and GS-12 can form `M`; this lands before either.

### P0.3 — SectionedELL operator emission (D3) · HOST-CTOR · Effort M
- Sectioned path for `mesh_cotangent_laplacian`, `mesh_k_ring_adjacency`, and
  the mass operator on arbitrary meshes: `format={'ell','sectioned','auto'}`
  (icosphere → `ell`; irregular valence → `sectioned`). `auto` sections when
  `max_degree / median_degree` exceeds a pinned threshold; uses
  `sectioned_ell_from_ragged`'s `bucket_by`.
- **Tests:** real fsaverage5 / fs_LR valence **histogram recorded** in the test
  (sets the bucket boundaries from data, not a guess); sectioned-vs-plain-ELL
  bit-parity on the same logical operator (via the apply-seam); padding-waste
  reduction measured and asserted < a pinned ratio on the real mesh.
- **Exit:** arbitrary-mesh operators avoid the high-valence padding cliff.

### P0.4 — `{ELL, SectionedELL}` apply-seam · DIFF-JAX · Effort S
- `sparse.apply_operator` per §2.2.
- **Tests:** identical results for an operator expressed as `ELL` vs
  `SectionedELL`; differentiable through both; batched leading axes.
- **Exit:** the algorithm layer (§4+) is storage-format-agnostic.

### P0.5 — real-FS-mesh test fixture · test infra · Effort S
- `tests/_real_meshes.py`: lazy-fetch fsaverage5 (~10 242, CI default) +
  one fs_LR_32k surface (topology-mismatch / scale) via nilearn / templateflow;
  `pytest.importorskip` + offline-skip; cache; yield `(vertices, faces)` +
  overlays (`curv`/`sulc`/`thickness`/`area`) when present. fsaverage (163 842)
  marked `@pytest.mark.perf`. Add nilearn / templateflow / nibabel to the test
  extras (`pyproject [test]` / `noxfile`), never runtime.
- **Exit:** every subsequent primitive has a real-mesh target available.

---

## 4. Phase 1 — measurement

*Goal: high-value, low-risk composition on the Phase-0 substrate; the features
the learned registrars (`sugar`/`josa`) and morphometry consume.*

### P1.1 — mesh curvature (12.6) · DIFF-JAX · Effort S · Home `geometry.surface`
```python
def mean_curvature(mesh, *, area_scheme='voronoi') -> Float[Array, 'n_vertices']
def gaussian_curvature(mesh) -> Float[Array, 'n_vertices']
def principal_curvatures(mesh) -> Float[Array, 'n_vertices 2']
```
- Mean `H = ½‖M⁻¹L·v‖` with sign from alignment with `compute_vertex_normals`;
  Gaussian via angle-defect `(2π − Σθ_i)/A_mixed`; principal `κ = H ± √(H²−K)`.
- **Tests:** unit sphere — `H = 1/r`, `K = 1/r²` exactly (analytic oracle); a
  saddle — sign of `K < 0`. Real white surface — **Gauss–Bonnet**:
  `Σ K·A ≈ 2πχ = 4π` for genus-0 (a beautiful global real-mesh check); mean
  curvature sign correlates with FS `?h.curv` (sulci < 0, gyri > 0) above a
  pinned floor.

#### P1.1 as-built — sign convention & the FS `?h.curv` correlation (decision record)

Shipped 2026-06-21 (`geometry/surface.py`; `tests/test_surface_curvature.py`).
Two decisions, recorded here (not in the docstring, which stays
consumer-agnostic per house style):

- **Sign convention: convex-positive.** `mean_curvature` returns
  `H = sign(H_vec·n)·‖H_vec‖`, `H_vec = ½ M⁻¹L v`, `n` the outward vertex
  normal — so `H > 0` on convex regions (a sphere, gyral crowns), `< 0` in
  sulcal fundi. This is the **opposite sign to FreeSurfer `?h.curv`** (which is
  positive in sulci). Chosen because it follows directly from the
  outward-normal / mean-curvature-vector definition with no extra flip;
  consumers wanting FS sign negate. Differentiability forced a JAX-native
  per-face cotangent apply (`_cotangent_apply`) rather than the host-side ELL
  `mesh_cotangent_laplacian` (whose weights are numpy-constructed and so not
  differentiable w.r.t. vertices); the two agree to ~1e-4 (regression-tested).

- **Validation is class/correlation, not bit-parity** (the metrics↔ITK
  posture). Sphere oracle is exact (`H = 1.0000`, Gauss–Bonnet `= 4π`);
  real-cortex Gauss–Bonnet `= 4π` (genus-0). Against FS `?h.curv` on
  fsaverage5 white the (anti-)correlation is **−0.90**, and the gap from unity
  is **fully accounted for** (measured, not assumed):
  1. **FS smooths `curv`** (`mris_curvature` averages by default) — the
     dominant cause. Applying matched light smoothing (~2 one-ring passes) to
     nitrix `H` lifts the correlation to **−0.95**, then it decays with
     over-smoothing (the peak-then-decay signature of "the reference is a
     lightly-smoothed version of what we compute").
  2. **Different discrete estimators** — nitrix uses the cotangent/Meyer
     mean-curvature normal; FS fits a local quadric (second fundamental form).
     Both converge to continuous `H`; they differ pointwise on a discrete mesh.
  3. **The fsaverage5 overlay is group-averaged + resolution-resampled** —
     `curv` is a cross-subject average carried to ico5, not computed on the
     exact 10 242-vertex tessellation (same artefact class as the `area`
     overlay summing to 0.75× geometric).
  4. **It is white-surface-specific** — measuring `H` on pial drops to −0.74,
     confirming `?h.curv` is the white-surface curvature.
  5. Minor: coarse ico5 + a few high-curvature outliers (Spearman −0.91 ≥
     Pearson −0.90). The test threshold (−0.8) is deliberately conservative
     for robustness across nilearn versions.

### P1.2 — areal & strain distortion (GS-6) · DIFF-JAX · Effort S · `geometry.surface`
```python
def areal_distortion(source: Mesh, warped: Mesh) -> Float[Array, 'n_vertices']
def strain_distortion(source: Mesh, warped: Mesh) -> Float[Array, 'n_faces 2']
```
- Areal `log2(A_warped / A_orig)` per vertex (GS-5 areas); strain = principal
  stretches from the per-triangle deformation gradient (`eigh` of the
  Cauchy–Green / first-fundamental-form ratio).
- **Tests:** rigid warp → 0 distortion; uniform scale `s` → `log2(s²)` areal,
  `(s, s)` strain; real fsaverage→fs_LR resample distribution sane (no infs at
  the medial wall under masking).

### P1.3 — geodesic smoothing (GS-12, D2) · DIFF-JAX · Effort M · `geometry.surface`
```python
def surface_smooth(mesh, values: Float[Array, '... n_vertices'], *, fwhm: float,
                   ) -> Float[Array, '... n_vertices']
```
- Backward-Euler `(M + tL)x = M·x₀`, `t = fwhm²/(16 ln 2)`, solved with
  `linalg.krylov.cg` on the `(M + tL)` operator through the apply-seam (SPD).
  Docstring states the documented divergence from `wb_command -metric-smoothing`
  (D2); medial wall handled via `ell_mask`.
- **Tests:** mass conservation (`Σ M·x` preserved for a closed mesh); a delta
  on a near-flat patch → approximately Gaussian profile with the requested
  FWHM (the FWHM↔t calibration check); constants preserved; differentiable
  (grad through `cg`); real-mesh smoke with masked medial wall.

---

## 5. Phase 2 — field↔mesh + geometry-light movers

*Goal: unlock the `fastcsr`/`synthdist` field route and the geometry-light pial
strategy. Mix of HOST-CTOR (field↔mesh) and DIFF-JAX (movers).*

### P2.1 — euler/genus defect gate (GS-7 partial, D1) · HOST-CTOR · Effort XS · `geometry.topology`
```python
def euler_characteristic(mesh) -> int      # V - E + F
def genus(mesh) -> int                      # (2 - χ)/2, closed orientable
```
- The friction-detector that signals when the genus-0 escape hatch is unsafe.
- **Tests:** icosphere χ=2, genus 0; a torus mesh genus 1; real white surface
  reports genus 0 (or flags otherwise).

### P2.2 — marching cubes (GS-3) · HOST-CTOR · Effort M · `geometry.isosurface`
```python
def marching_cubes(volume, *, level=0.0, spacing=(1.,1.,1.)) -> Mesh
```
- **Asymptotic-decider** variant (§0 correction 5). Host-side table; output
  vertex count data-dependent → emits `Mesh`.
- **Tests:** analytic sphere SDF → mesh area ≈ 4πr², genus 0 (via P2.1),
  manifold (no non-manifold edges — the variant's whole point); real synthdist
  SDF smoke.

### P2.3 — point-to-triangle distance + mesh→SDF (GS-4) · HOST-CTOR · Effort M · `geometry.surface`
```python
def mesh_to_sdf(mesh, shape, *, spacing=(1.,1.,1.)) -> Float[Array, 'X Y Z']
```
- New clean-room point-to-triangle (unsigned) distance with a uniform-grid
  broad-phase (no `scipy.spatial`); sign via winding number / normal test.
- **Tests:** analytic sphere — `mesh_to_sdf` matches `‖x‖ − r`; sign correct
  inside/outside; round-trip with P2.2 (`sdf → mc → sdf` agrees to voxel tol).

### P2.4 — SDF normal-march (GS-10) · DIFF-JAX · Effort M · `geometry.surface`
```python
def deform_to_sdf(mesh, sdf, *, n_iterations=50, step=0.2, smooth_weight=0.1,
                  spacing=(1.,1.,1.)) -> Mesh
```
- Jitted `fori_loop`: sample sdf + `spatial_gradient` at vertices
  (`sample_at_points`), step along the normal scaled by the local sdf value,
  apply a Laplacian-smoothing fraction + step clamp. **In-loop regularisation
  only — no GS-8 guard** (§2.5).
- **Tests:** a sphere mesh → a larger sphere's SDF preserves vertex
  correspondence and genus; differentiable (grad through the loop); real
  topofit-white → synthdist-pial smoke (the geometry-light pial step).

### P2.5 — cortical thickness (GS-9) · DIFF-JAX (corr.) / HOST-QA (sym.) · Effort S · `geometry.surface`
```python
def cortical_thickness(white, pial, *, method='symmetric') -> Float[Array, 'n_vertices']
```
- `'correspondence'` = `‖pial − white‖` (DIFF-JAX, exact when P2.4 made pial);
  `'symmetric'` = host-side nearest-point (uniform-hash, HOST-QA).
- **Tests:** parallel offset surfaces → constant thickness (both modes); real
  white/pial vs FS `?h.thickness` correlation.

### P2.6 — self-intersection (GS-8) · HOST-QA · Effort M · `geometry.surface`
```python
def find_self_intersections(mesh) -> Int[Array, 'n_pairs 2']
def remove_self_intersections(mesh, *, n_iterations=10) -> Mesh
```
- Uniform-hash broad-phase + Möller narrow-phase; removal = local Laplacian
  relaxation. **Post-hoc cleanup, never in-loop** (§2.5).
- **Tests:** synthetic self-intersecting mesh detected; relaxation reduces the
  pair count; real warped-sphere smoke.

---

## 6. Phase 3 — hard continuous optimisers (SPEC-review-gated)

*Goal: the two genuinely hard optimisers. Per §13.4, GS-2 needs a SPEC-level
review before it lands.*

### P3.1 — inflation + sulc (GS-1) · DIFF-JAX · Effort M · `geometry.surface`
```python
def inflate_surface(mesh, *, n_iterations=100, spring_weight=0.5,
                    metric_weight=1.0) -> tuple[Float[Array,'n_vertices 3'],
                                                Float[Array,'n_vertices']]
```
- `fori_loop` GD on `E = w_s·E_smooth + w_m·E_metric` (umbrella-Laplacian
  smoothing + 1-ring edge-length preservation); sulc = integrated signed normal
  displacement.
- **Tests:** folded synthetic surface inflates toward a sphere with bounded
  metric distortion; real white surface — sulc correlates with FS `?h.sulc`.

### P3.2 — spherical parameterisation (GS-2) · DIFF-JAX forward · **Effort L, SPEC review** · `geometry.sphere`
```python
def spherical_parameterize(mesh, *, n_iterations=200, area_weight=1.0,
                           distance_weight=1.0) -> Float[Array, 'n_vertices 3']
```
- **Recommended formulation (the SPEC-review topic):** conformal / Tutte-style
  *initialisation* (not bare centroid-normalise), then GD on the
  distance/area-distortion energy with a **signed-area fold barrier** (forbid
  negative triangles) and a robust line-search / step schedule. The risk is
  *quality* (a bijective, fold-free embedding), not differentiability.
- **Tests:** an already-inflated sphere ≈ identity; real inflated white surface
  → **all signed triangle areas > 0** (bijectivity), distortion vs the FS
  `?h.sphere` within a pinned band.

---

## 7. Phase 4 — HCP back-end

### P4.1 — ADAP_BARY_AREA arbitrary-mesh resample (12.15) · HOST-CTOR + DIFF-JAX apply · Effort M · `geometry.sphere`
```python
def surface_resample(source_mesh, source_vals, target_mesh, *,
                     method='adap_bary_area') -> tuple[ELL, Array]
```
- Host-side spherical point-in-triangle search (clean-room) → barycentric ELL;
  JAX application via the apply-seam. The fsaverage↔fs_LR bridge.
- **Tests:** identity when `source == target`; fsaverage↔fs_LR round-trip error
  bounded; area-weighting conserves the integral.

### P4.2 — ribbon map (GS-14) · DIFF-JAX · Effort M · `geometry.surface`
```python
def ribbon_map(volume, white, pial, *, n_samples=10, weighting='pv',
               spacing=(1.,1.,1.)) -> Float[Array, '... n_vertices']
```
- Sample `n_samples` along each white→pial segment, `sample_at_points`,
  weighted reduce. The HCP myelin (T1w/T2w) map.
- **Tests:** constant volume → constant map; linear ramp → exact mean; real
  T1w/T2w → myelin-map smoke.

### P4.3 — josa boundary mode (GS-13) · **verify-first** · Effort S · `geometry`
- **Likely already shipped** (§0 correction 6). First write a composition test:
  `sphere_grid_pad_2d` + `spatial_transform(mode='nearest')` on an
  equirectangular SVF. If it matches the reference, close with the test + a
  docstring cross-ref; only if it fails does a code change apply.

---

## 8. Phase 5 — parcellation + optional / research-tracked

### P5.1 — boundary map (12.16) → watershed (12.17) · S + M · `graph.parcellation`
- `surface_boundary_map` = named wrapper on `semiring_ell_edge_aggregate` +
  `stats.corr` (eta²), profiles tiled at ico7 via `lax.map`; `mesh_watershed` =
  host-side priority-flood (Barnes 2014) on the boundary field.
- **Tests:** synthetic two-region profile → a boundary at the seam; watershed
  recovers the two basins; real connectivity smoke.

### P5.2 — place_surface (GS-11) · DIFF-JAX · **Effort L, SPEC review, optional** · `register.surface`
- Same `fori_loop` skeleton as P2.4 with an image-gradient / target-intensity
  external force. Optional — learned models replace it; ship for classic
  `recon-all` parity / hard clinical cases.

### P5.3 — research-tracked (on a named consumer, §13.1)
- **GS-7 corrector** (Fischl 2001 / Ségonne 2007) — the D1 seam is kept open;
  candidate for `bitsjax` if it grows a heavy solver.
- **DEC stack** (12.5) and **spherical-harmonic transform** (12.9) —
  composition-ready but pulled only when a concrete consumer arrives.

---

## 9. Cross-cutting test strategy

Two oracles per Phase-0/1 primitive (non-negotiable 4):

1. **Icosphere analytic oracle.** Exact closed forms (sphere `H=1/r`,
   `K=1/r²`; area `4πr²`; rigid → zero distortion; Gauss–Bonnet `Σκ̃ = 4π`).
   The icosphere's uniform valence makes these crisp.
2. **Real-FS-mesh check.** Via the P0.5 fixture: (a) the op runs on
   irregular-valence, obtuse-triangle, medial-wall geometry at mm scale in
   fp32 without NaN/inf; (b) where FS ships an overlay, a **class/correlation**
   comparison (not bit-parity, per the metrics↔ITK precedent):

   | primitive | FS oracle | check |
   |---|---|---|
   | `vertex_areas` (P0.2) | `?h.area` | sum + per-vertex Spearman |
   | `mean_curvature` (P1.1) | `?h.curv` | sign (sulci/gyri) + correlation |
   | `inflate_surface` sulc (P3.1) | `?h.sulc` | correlation |
   | `cortical_thickness` (P2.5) | `?h.thickness` | correlation |

The fixture is itself a test of the `icosphere_hierarchy_from_levels`
array-handoff contract — if the suite breaks on real external topology, that
boundary is where we find out. fp32 conditioning on real mm-scale coordinates
(cf. `lme-lowrank-eigh-fp32-conditioning`) is explicitly probed by running the
P0.2/P1.1 ops on the un-normalised real white surface, not just the unit sphere.

## 10. Governance & graduation

Per SPEC_UPDATE_v0.3 §13, each item carries the four-gate. Named consumer
(`recon-all-clinical` / HCP / `sugar` / `josa`) and composition sketch are
satisfied by the FR; the operative gates are SoC and effort:

| Item | §12 origin | §13 gate | Graduation record |
|---|---|---|---|
| GS-5, areas/mass | beyond §12 | XS/S — lands in P0 | §10.A at merge |
| mesh-curvature | 12.6 | S — P1 | §12.6 → §10.A |
| GS-6 distortion | beyond §12 | S — P1 | §10.A |
| GS-12 smoothing | 12.3-adjacent | M — P1 | §10.A (cites `cg`, D2) |
| GS-3 / GS-4 / GS-9 / GS-10 | beyond §12 | M — P2 | §10.A |
| euler/genus | 12-adjacent | XS — P2 | §10.A |
| GS-8 | beyond §12 | M — P2 | §10.A (HOST-QA) |
| GS-1 inflate | beyond §12 | M — P3 | §10.A |
| **GS-2 parameterise** | beyond §12 | **L — SPEC review (§13.4)** | SPEC update first |
| ADAP_BARY / GS-14 | 12.15 / beyond | M — P4 | §12.15 → §10.A |
| GS-13 | beyond §12 | verify-first | §10.A (likely no-code) |
| boundary/watershed | 12.16 / 12.17 | S/M — P5 | §12.16/17 → §10.A |
| **GS-11 place_surface** | beyond §12 | **L — SPEC review, optional** | SPEC update first |
| GS-7 corrector / DEC / SHT | 12.5 / 12.9 | research-tracked | on a named consumer |

Each graduation is logged in `IMPLEMENTATION_PLAN.md §10` at merge with
type / trigger / shape / deferred-work, exactly as the registration suite did
for `cg` / `matrix_exp` / `fixed_point_solve`. PyTree lift (B22) is logged as a
substrate enabler, not a §12 graduation.

## 11. Dependency graph & sequencing

```
P0.1 pytree ─┬─────────────────────────> (all DIFF optimisers: P2.4, P3.1, P3.2, P4.2)
P0.2 areas/mass ─┬─> P1.1 curvature ─> sugar/josa features
                 ├─> P1.2 distortion
                 └─> P1.3 smoothing (also needs linalg.krylov.cg ✓ shipped)
P0.3 sectioned ─┬─> P0.4 apply-seam ─> (P1.x, P2.3, P1.3 on arbitrary meshes)
P0.5 fixture ───────> (real-mesh exit criterion for every P0/P1 task)

P2.2 marching cubes ─> P2.1 euler/genus gate ─[GS-7 slot]─> P3.1 inflate
P2.3 mesh->SDF ─> P2.4 SDF-march ─> P2.5 thickness (corr.)
P2.6 self-intersection (post-hoc; not on any in-loop path)

P3.1 inflate ─> P3.2 parameterise ─> sugar/josa inputs
P4.1 resample   P4.2 ribbon   P4.3 josa-boundary (verify)
P5.1 boundary ─> P5.1 watershed     P5.2 place_surface (opt)
```

The only hard cross-phase edges: **P0.2 → P1.1/P1.3** (mass), **P0.1 → all
optimisers** (pytree), **P0.3/P0.4 → arbitrary-mesh ops**, **P2.2 → P2.1**,
**P2.3 → P2.4 → P2.5**, **P3.1 → P3.2**. Within a phase, tasks parallelise.

## 12. Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| GS-2 fails to produce a fold-free bijection | High | conformal/Tutte init + signed-area barrier + robust line-search; SPEC review before build; bijectivity is the test gate (all signed areas > 0). |
| fp32 conditioning on real mm-scale folded geometry | Med | probe P0.2/P1.1 on un-normalised real surfaces; fp64 host path for area/cotangent assembly (already done in `mesh_cotangent_laplacian`). |
| SectionedELL bucket boundaries mis-set | Med | set from the **measured** real valence histogram (P0.3), not a guess; parity test vs plain ELL. |
| Host-side perf cliff at ico7 (marching cubes / watershed / BVH) | Med | uniform-grid acceleration; mark heavy paths `@perf`; these are construction-time, not per-iteration. |
| Friction forces the GS-7 corrector (D1 pivot) | Low–Med | seam kept open (P2.1 gate + `Mesh` in/out); pivot is additive, not a rewrite. |
| GS-13 not actually shipped | Low | verify-first composition test before any code (P4.3). |
| nilearn/templateflow unavailable in CI | Low | `importorskip` + offline-skip keep the core suite green; real-mesh checks are additive, not gating for networkless runs. |

## 13. Cross-references

- **Feature request (what/why/ledger):**
  [`../feature-requests/geometry-suite.md`](../feature-requests/geometry-suite.md).
- **Substrate enabler:**
  [`../feature-requests/register-sparse-dataclasses-as-pytrees.md`](../feature-requests/register-sparse-dataclasses-as-pytrees.md)
  (B22).
- **Live substrate:** `src/nitrix/sparse/{ell,ell_sectioned,mesh}.py`,
  `src/nitrix/geometry/{sphere,sphere_grid,grid,differential,coords}.py`,
  `src/nitrix/linalg/krylov.py` (`cg`), `src/nitrix/graph/connectopy.py`.
- **Design context:** [`geometry.md`](geometry.md), [`sphere-grid.md`](sphere-grid.md),
  [`mesh-graph-conv.md`](mesh-graph-conv.md),
  [`sparse-specialisations.md`](sparse-specialisations.md),
  [`ell-on-triton.md`](ell-on-triton.md), [`testing-strategy.md`](testing-strategy.md).
- **Governance:** `SPEC.md §5` (dep contract), `SPEC_UPDATE_v0.3.md §13`
  (acceptance) / §14 (kwarg-not-fork), `IMPLEMENTATION_PLAN.md §2.2`
  (non-negotiables) / §10 (deviation log).
- **Dependency FRs adopted by the suite:** `mesh-curvature.md` (12.6),
  `discrete-exterior-calculus.md` (12.5), `heat-kernel-diffusion.md` (12.3),
  `surface-resample-adap-bary.md` (12.15), `surface-boundary-map.md` (12.16),
  `mesh-watershed.md` (12.17), `spherical-harmonic-transform.md` (12.9),
  `field-regularisers.md` (shipped), `point-sample.md` (shipped),
  `distance-transform-anisotropic-sampling.md` (distinct from GS-4).
