# Cortical-surface geometry suite — `nitrix.geometry` / `nitrix.sparse.mesh`

> **Status (2026-06-17): context/ledger doc.** Frames a primitive family,
> credits the shipped substrate, indexes the already-filed atomised items
> that belong to it, and specs the **genuinely-new numerical gaps**. No new
> kernel is proposed here that an existing FR already owns — those are
> referenced, not duplicated (this index is the duplicate-issue guard, per
> `README.md`). Driver: ilex assembling `recon-all`, `recon-all-clinical`,
> and the HCP minimal-preprocessing pipelines on top of its curated synth\*
> + learned-surface model suite.

## 0. 2026-06-21 — correction pass & build decisions

> A pre-build review (recorded in full as the implementation plan at
> [`docs/design/geometry-suite.md`](../design/geometry-suite.md)) found this
> ledger sound in scope but **wrong or under-specified on a few load-bearing
> numerics**, and resolved three scope forks. The corrections are folded into
> the item entries below; this section is the summary of record.

**Math / sequencing corrections (folded into §3 / §5 / §7):**

1. **The mass matrix is missing and load-bearing.** `mesh_cotangent_laplacian`
   ships the *unnormalised stiffness* `L` only; there is **no** vertex-area /
   mass-matrix symbol in `src/nitrix`. So mean curvature is **not**
   `L @ vertices` (that is the *integrated* mean-curvature normal
   `2·H·A_mixed·n̂`); recovering `H` needs `M⁻¹`. **GS-5 (areas + a lumped mass
   matrix) is a Phase-0 prerequisite for the correctness of both
   `mesh-curvature` and GS-12**, not a Phase-1 convenience. The §7 graph gains
   the edge **GS-5 → GS-12**.
2. **GS-12 is an implicit sparse *solve*, not a dense exponential.** Use
   backward-Euler `(M + tL) x = M·x₀` via the shipped `linalg.krylov.cg` on the
   ELL operator; dense `matrix_exp` does not scale to ico7 (163 842 vertices).
   See decision **D2** for the parity stance.
3. **GS-4's dependency was mis-stated.** It needs a **point-to-triangle**
   (unsigned) distance + a sign test — a new clean-room host-side primitive,
   *not* the morphology mask-EDT in
   [distance-transform-anisotropic-sampling](resolved/distance-transform-anisotropic-sampling.md).
4. **GS-10 cannot call GS-8 in-loop.** GS-8 (BVH / Möller) is host-side and
   un-jittable; GS-10's `fori_loop` uses only jittable regularisation
   (Laplacian fraction + step clamp). GS-8 is a **post-hoc** cleanup pass.
5. **GS-3 variant matters.** Specify the **asymptotic-decider** marching-cubes
   variant — plain Lorensen–Cline's face-saddle ambiguity produces
   non-manifold output that *fights* the genus-0 goal. Output vertex count is
   data-dependent → host-side construction emitting a `Mesh`.
6. **GS-13 is probably already shipped.** `spatial_transform` already forwards
   `mode='nearest'` (the geometry JOSA sprint); the equirectangular path
   composes `sphere_grid_pad_2d` + `spatial_transform(mode='nearest')`. Verify
   with a composition test before writing any code.

**Substrate prerequisites elevated to Phase 0 (the first pass assumed these
present or trivial):**

- **PyTree lift** of `ELL` / `SectionedELL` / `Mesh`
  ([register-sparse-dataclasses-as-pytrees](register-sparse-dataclasses-as-pytrees.md),
  B22) — every differentiable optimiser (GS-1/2/10/11) and the GS-12 solve
  needs these across `jit` / `grad` / `scan`. **Consciously decline**
  `IcosphereHierarchy` (host-side NumPy `parents` → unhashable aux; a
  construction artefact, not a traced operand).
- **`SectionedELL` operator emission** for arbitrary (non-icosphere) meshes
  (decision **D3**) + a **format-agnostic apply-seam** over {`ELL`,
  `SectionedELL`} so the measurement layer never branches on storage.
- A **real-FS-mesh validation fixture** in `tests/` (principle below).

**Locked decisions (2026-06-21):**

- **D1 — Topology: geometry-light this pass, pivot-ready.** Ship marching cubes
  (asymptotic-decider) + `euler_characteristic` / `genus` as the **defect
  gate**; rely on the `topofit`-white → GS-10 SDF-march geometry-light path
  that inherits genus-0. **Defer** the full GS-7 corrector to a research track
  but keep the seam `GS-3 → [GS-7 slot] → GS-1` open (corrector consumes /
  emits `Mesh`). Pivot if friction develops.
- **D2 — GS-12 smoothing: heat-diffusion native, divergence documented.** Ship
  the backward-Euler heat smoother; document that it is *not*
  `wb_command -metric-smoothing` (a geodesic-distance-weighted Gaussian) — the
  same posture as the metrics↔ITK convention. Workbench parity, if ever needed,
  arrives behind a `method=` kwarg (§14), not a fork.
- **D3 — `SectionedELL` for arbitrary meshes.** The operator / measure layer
  gains a sectioned path for irregular-valence `recon-all` surfaces; the
  icosphere (uniform valence ≈ 6) stays plain `ELL`.

**Real-mesh validation principle.** Test on real FreeSurfer / fs_LR meshes
**early**, IO-safe: the IO lives in `tests/` only (nilearn / templateflow /
nibabel are test-only per SPEC §6.2 + §8), never in `nitrix`. The test plays
the "consumer reads files → hands nitrix arrays" role, so it doubles as a live
test of the `icosphere_hierarchy_from_levels` array-handoff seam. **Every
Phase-0/1 primitive must pass on both an icosphere (analytic oracle) and a real
FS mesh** — compared to FS's own per-vertex overlays (`?h.area`, `?h.curv`,
`?h.sulc`, `?h.thickness`) where one exists (class / correlation, not
bit-parity).

**Governance.** GS-1…14 are candidates **beyond the current §12 set**; each
needs the §13 four-gate (named consumer ✓, composition sketch ✓, SoC, effort).
Per **§13.4 the Effort-L items — GS-2 (parameterise) and GS-11 (place_surface)
— need a SPEC-level review before slotting.** Record each §12→§10.A graduation
in `IMPLEMENTATION_PLAN.md` at merge.

## 1. Why this doc exists

ilex now curates the *neural* half of cortical reconstruction end-to-end —
contrast-agnostic segmentation / super-resolution / skull-strip
(`synthseg`, `synthsr`, `synthstrip`), volumetric registration
(`synthmorph`, `voxelmorph`), and a near-complete learned **surface** stack:
white-surface reconstruction (`topofit`), level-set surface regression
(`fastcsr`), signed-distance surface prediction (`synthdist`, the
recon-all-clinical core), and spherical registration (`sugar` ≈ MSMSulc,
`josa` ≈ MSMAll). What those models do **not** supply is the classical
mesh-geometry numerics that turn their outputs into a pipeline: extracting a
mesh from a predicted field, making it a topological sphere, inflating it,
parameterising it on `S²`, measuring it, and resampling it to a standard
mesh.

A triage of the three target pipelines against the live `nitrix` surface
shows the gap is **smaller than it looks** — the load-bearing operators
(the cotangent Laplace–Beltrami operator, uniform smoothing, vertex
normals, the icosphere hierarchy, geodesic distance, barycentric upsampling,
the full volumetric registration suite, bias correction) are **already
shipped**. This doc is the ledger that (a) records that, (b) pulls the
scattered surface-relevant FRs under one roof, and (c) names the handful of
numerical primitives still missing, each mapped to the pipeline stage that
needs it.

## 2. Scope boundary

**In scope — numerical primitives only**, consistent with the SPEC §6
dependency contract (`numpy` + `jax`; **no `nibabel`, no filesystem, no
container/atlas resolution**). Host-side combinatorial *construction* that
emits plain arrays / `ELL` is in scope and already idiomatic here — the
icosphere subdivision, k-ring BFS, and cotangent assembly in
`sparse/mesh.py` are all host-side NumPy that hand off to JAX.

**Out of scope — IO and disk/container formats → `thrux`:**

- FreeSurfer surface / overlay binaries (`.white`, `.pial`, `.sphere`,
  `.inflated`, `.curv`, `.sulc`, `.thickness`, `.annot`), GIFTI
  (`.surf.gii`, `.func.gii`, `.label.gii`), CIFTI dense files.
- `$SUBJECTS_DIR` resolution, `fsaverage` / `fs_LR` topology *loading*,
  any `nibabel` / `surfa` container round-trip.

The seam is already established: `sparse.mesh.icosphere_hierarchy_from_levels`
takes the *plain arrays* a consumer reads from `fsaverage{0..6}.sphere` and
builds the same `IcosphereHierarchy` the math-canonical path builds —
nitrix never sources the topology. Every new surface primitive here follows
that contract: **the consumer reads files and the surfa-side surface↔sphere
projection wrapper; nitrix supplies the array math.** See
[`ilex-pipeline-substrate.md`](ilex-pipeline-substrate.md) for the same
boundary stated from the consumer side.

**Edge case — `bitsjax`-flagged numerics are still in scope here.** A few
items (marching cubes, the SDF kernel) were historically pencilled for
`bitsjax` only because the *reference* implementation (nighres) drags in
JCC/OpenJDK. The kernels themselves are pure array math with no IO, so they
belong in this numerical ledger regardless of which package ultimately hosts
them; the **Home** field flags the open question per item.

## 3. Already shipped — the substrate this suite builds on

Verified against `src/nitrix` as of 2026-06-17. **These are done; new items
compose on them.**

| Capability | Symbol(s) | Module |
|---|---|---|
| Discrete Laplace–Beltrami (cotangent) | `mesh_cotangent_laplacian` | `sparse.mesh` |
| Uniform / umbrella Laplacian smoothing | `mesh_laplacian_smooth` | `sparse.mesh` |
| Per-vertex normals | `compute_vertex_normals` | `sparse.mesh` |
| Triangle-mesh container + icosphere | `Mesh`, `icosphere` | `sparse.mesh` |
| Icosphere hierarchy (+ external-topology ctor) | `icosphere_hierarchy`, `icosphere_hierarchy_from_levels` | `sparse.mesh` |
| k-ring adjacency (`ELL`) | `mesh_k_ring_adjacency` | `sparse.mesh` |
| Mesh pool / unpool / bary-upsample / meanpool | `mesh_pool_max`, `mesh_unpool_max`, `mesh_bary_upsample`, `mesh_coarsen_meanpool` | `sparse.mesh` |
| Cross-level adjacency / bary upsampler | `icosphere_cross_level_adjacency`, `icosphere_bary_upsampler` | `sparse.mesh` |
| Sphere coordinate conversions | `cartesian_to_latlong`, `latlong_to_cartesian` | `geometry.sphere` |
| Geodesic distance on the 2-sphere | `spherical_geodesic_distance` | `geometry.sphere` |
| Spherical mesh convolution (`O(n·k)`) | `spherical_conv` | `geometry.sphere` |
| Equirectangular sphere-grid padding (pole-flip / wrap) | `sphere_grid_pad_2d`, `sphere_grid_unpad_2d` | `geometry.sphere_grid` |
| Grid sampling / warp / resample (+ interpolators) | `spatial_transform`, `resample`, `sample_at_points`, `Linear`/`NearestNeighbour`/`Lanczos`/`CubicBSpline`/`MultiLabel` | `geometry.grid`, `geometry._interpolate` |
| Velocity-field integration (scaling-and-squaring) | `integrate_velocity_field` | `geometry.grid` |
| Displacement Jacobian / det (volumetric) | `jacobian_displacement`, `jacobian_det_displacement` | `geometry.grid` |
| Image spatial gradient (∇I; central/sobel/scharr) | `spatial_gradient` | `geometry.differential` |
| Gaussian pyramid / up / downsample | `gaussian_pyramid`, `downsample`, `upsample` | `geometry.pyramid` |
| Rigid / affine Lie chart + closed-form point-set fit | `rigid_exp/log`, `affine_exp`, `fit_affine`, `compose_affine`, `invert_affine` | `geometry.transform`, `geometry.affine` |
| Displacement / velocity algebra | `compose_displacement`, `invert_displacement`, `field_log`, `compose_velocity`, `transform_mean` | `geometry.deformation`, `geometry.algebra` |
| Volumetric registration recipes | diffeomorphic / SVF / SyN / demons / **BBR** | `register.*` (`_bbr`, `_svf`, `_syn`, `diffeomorphic`, `recipes`) |
| Bias-field correction (incl. B-spline) | `bias.correction`, `bias._bspline` | `bias` |
| Connected components / labelling | `morphology._label` | `morphology` |

Consequences worth stating plainly:

- The **Laplace–Beltrami *stiffness* substrate is here, but the *mass* matrix
  is not.** `mesh_cotangent_laplacian @ vertices` gives the **integrated**
  mean-curvature normal `2·H·A_mixed·n̂`, *not* the mean curvature `H` — that
  needs the mixed-Voronoi mass `M⁻¹` (GS-5), which is **not shipped**. The
  [mesh-curvature](resolved/mesh-curvature.md) FR therefore depends on GS-5; it cannot
  just "name" an existing op. See §0 correction 1.
- The **HCP PreFreeSurfer volumetric front-end is largely shipped**: rigid /
  affine / BBR / SyN registration + B-spline bias correction cover ACPC,
  T1w↔T2w (BBR), brain-mask-via-template, and the MNI affine+nonlinear warp.
  EPI/readout distortion is the one volumetric gap (a 1-D displacement field
  along the phase-encode axis — ilex also has `gdc_net`/`fd_net`/`syn_disco`
  on the learned side).
- The **`josa` spherical-diffeomorphism path is ~90% shipped**:
  `sphere_grid_pad_2d` + `integrate_velocity_field` give the padded 2D-grid
  SVF warp; the residual is a boundary-mode keyword (item **GS-13**).

## 4. Atomised items already filed (belong to this suite)

These exist under `docs/feature-requests/`; this ledger adopts them and
records pipeline relevance + current status. **Add to the linked doc, not
here.**

| Item | FR | §12 | Status (2026-06-17) | Pipeline role |
|---|---|---|---|---|
| Mesh curvature (mean / Gaussian / principal) | [mesh-curvature](resolved/mesh-curvature.md) | 12.6 | not started — substrate shipped | curv/sulc features for `sugar`/`josa`; gyrification |
| Discrete exterior calculus | [discrete-exterior-calculus](discrete-exterior-calculus.md) | 12.5 | partial (cotangent LBO shipped) | unifying operator for smoothing / parameterisation |
| Vertex normals | [compute-vertex-normals](resolved/compute-vertex-normals.md) | — | ✅ shipped (`compute_vertex_normals`) | normals for deformation / mapping |
| Uniform Laplacian smoothing | [mesh-laplacian-smoothing](resolved/mesh-laplacian-smoothing.md) | — | ✅ shipped (`mesh_laplacian_smooth`) | `mris_smooth` analogue |
| Adaptive area-weighted bary resample | [surface-resample-adap-bary](resolved/surface-resample-adap-bary.md) | 12.15 | partial (icosphere bary shipped; arbitrary-mesh `ADAP_BARY_AREA` missing) | **HCP** `fs_LR_32k` downsampling; `fsaverage`↔`fs_LR` |
| Heat-kernel diffusion | [heat-kernel-diffusion](resolved/heat-kernel-diffusion.md) | 12.3 | partial (`diffusion_embedding` shipped) | geodesic surface smoothing substrate (**GS-12**) |
| Spherical harmonic transform | [spherical-harmonic-transform](spherical-harmonic-transform.md) | 12.9 | not started | spectral surface analysis; atlas bases |
| Surface boundary map | [surface-boundary-map](resolved/surface-boundary-map.md) | 12.16 | not started (composes shipped prims) | functional parcellation (aparc-adjacent) |
| Mesh watershed | [mesh-watershed](resolved/mesh-watershed.md) | 12.17 | not started | parcellation from boundary map |
| Field regularisers (bending / strain energy) | [field-regularisers](resolved/field-regularisers.md) | — | filed (ENABLING) | inflation & distortion energies (**GS-1**, **GS-6**) |
| Distance transform (anisotropic) | [distance-transform-anisotropic-sampling](resolved/distance-transform-anisotropic-sampling.md) | — | filed | SDF generation substrate (**GS-4**) |
| Point sampling | [point-sample](resolved/point-sample.md) | — | partial (`sample_at_points` core) | surface→volume sampling (**GS-11**) |

## 5. New gaps — numerical primitives not yet owned by any FR

The heart of this doc. Each item follows the house format (What / Proposed
surface / Composition / Likely consumer / Home / Effort / Live-code status).
IDs are `GS-n`. Effort scale matches the catalogue (XS/S/M/L/XL).

---

### GS-1 — Surface inflation (`mris_inflate`) → `geometry.surface.inflate`

**What.** Inflate a folded genus-0 white surface toward a smooth
near-spherical surface, minimising metric (inter-vertex distance) distortion
against a smoothing term. Emits **sulcal depth** as the integrated signed
normal displacement — the `?h.sulc` feature `sugar`/`josa` consume.

**Proposed surface.**

```python
def inflate_surface(
    mesh: Mesh, *, n_iterations: int = 100,
    spring_weight: float = 0.5, metric_weight: float = 1.0,
) -> tuple[Float[Array, 'n_vertices 3'], Float[Array, 'n_vertices']]:
    """Return (inflated_vertices, sulcal_depth)."""
```

**Composition.** Gradient descent (`jax.lax.fori_loop`) on
`E = w_s·E_smooth + w_m·E_metric`. `E_smooth` is the umbrella-Laplacian step
already in `mesh_laplacian_smooth`; `E_metric` penalises change in 1-ring
edge lengths vs. the original surface (reuses the `mesh_k_ring_adjacency`
edge set). Fully differentiable, JAX-native.

**Likely consumer.** Every surface pipeline — the mandatory precursor to
spherical mapping, and the sole source of `sulc`. `sugar`/`josa` are dead
without it (they consume the inflated-surface curvature features).

**Home.** New `geometry.surface` (sits beside `geometry.sphere`; the
folded-surface energies are distinct from the `S²`-grid helpers). **Effort
M.** **Live-code status.** No `inflate`/`sulc` symbol; energies compose on
shipped smoothing + adjacency. See [field-regularisers](resolved/field-regularisers.md).

---

### GS-2 — Spherical parameterisation (`mris_sphere`) → `geometry.sphere.parameterize`

**What.** Map an inflated genus-0 surface onto the unit sphere minimising
**areal + metric distortion**, with an oriented-area (no-fold) term keeping
the embedding a homeomorphism. This is the energy-minimising *embedding* —
**distinct from `cartesian_to_latlong`**, which only re-coordinatises an
already-spherical mesh.

**Proposed surface.**

```python
def spherical_parameterize(
    mesh: Mesh, *, n_iterations: int = 200,
    area_weight: float = 1.0, distance_weight: float = 1.0,
) -> Float[Array, 'n_vertices 3']:  # vertices on the unit sphere
```

**Composition.** Project to sphere (centroid + normalise), then gradient
descent on a distance/area-distortion energy with a per-triangle signed-area
penalty to forbid negative (folded) triangles. Reuses
`compute_vertex_normals`, the cotangent weights, and per-vertex/face areas
(**GS-5**). Differentiable; JAX `fori_loop`.

**Likely consumer.** `sugar`/`josa` inputs; any `fsaverage`/`fs_LR`
registration. **Home.** `geometry.sphere`. **Effort L** (the hardest of the
continuous optimisers — fold-prevention makes the energy non-convex; needs a
robust line-search / step schedule). **Live-code status.** Absent;
coordinate conversion + geodesic distance shipped, the embedding optimiser is
not. **Full SPEC-review design doc (§13.4):**
[`spherical-parameterisation.md`](spherical-parameterisation.md) — supports both
families: a one-shot **spectral** embedding (FastSurfer/recon-surf: generalised
LBO `(L, M)` eigfns 1–3 → normalise; the fast default, verified against the
recon-surf source) with a Tutte+stereographic guaranteed-bijective fallback, and
the iterative conformal+area energy (collapse-trap defence) under Riemannian
descent with a fold-safe line-search + Möbius normalisation. Phased GS-2a–d +
test plan.

---

### GS-3 — Marching cubes (isosurface) → `geometry.isosurface` *(home: nitrix vs bitsjax — open)*

**What.** Extract a triangle mesh at the zero level-set of a signed-distance
/ level-set volume. The mesh-extraction step `fastcsr` and `synthdist` (and
hence **recon-all-clinical**) need but do not perform — the networks emit
fields, not meshes.

**Proposed surface.**

```python
def marching_cubes(
    volume: Float[Array, 'X Y Z'], *, level: float = 0.0,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Mesh:
```

**Composition.** **Asymptotic-decider** 256-case edge-table lookup + linear
edge interpolation (plain Lorensen–Cline's face-saddle ambiguity yields
non-manifold output that fights the genus-0 goal — §0 correction 5). Host-side
table construction (like the icosphere midpoint table); the per-cube gather /
interpolate is vectorised, but the **output vertex count is data-dependent**,
so the whole op is a host-side construction emitting a `Mesh` (the icosphere
idiom), not a fixed-shape JAX kernel. No IO. Pairs with the GS-7 defect gate
(`euler_characteristic` / `genus`) and keeps the `GS-3 → [GS-7 slot] → GS-1`
seam open (decision D1).

**Likely consumer.** `fastcsr`, `synthdist`, any level-set surface model;
volumetric label → surface for QA. **Home.** Numerical and IO-free, so
nitrix-eligible (`geometry.isosurface`); historically pencilled for
`bitsjax` only to dodge the nighres/OpenJDK reference dep — that reason does
not apply to a clean-room kernel. **Flag for a home decision.** **Effort M.**
**Live-code status.** Absent.

---

### GS-4 — Signed-distance field generation (mesh → SDF) → `geometry.surface.sdf`

**What.** Convert a triangle mesh to a signed-distance volume (negative
inside). The inverse of GS-3; the training-target generator behind
`synthdist`/`fastcsr` and a validation tool for the field↔mesh round-trip.

**Proposed surface.**

```python
def mesh_to_sdf(
    mesh: Mesh, shape: tuple[int, int, int], *,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Float[Array, 'X Y Z']:
```

**Composition.** Unsigned **point-to-triangle** distance (a new clean-room
host-side primitive, uniform-grid accelerated — *not* the morphology mask-EDT
in [distance-transform-anisotropic-sampling](resolved/distance-transform-anisotropic-sampling.md),
which measures distance to a binary mask, §0 correction 3) + sign from
generalised winding number / normal test. `scipy.spatial` is banned at runtime
(SPEC §6.2), so the broad-phase is a clean-room uniform-grid hash, not
`cKDTree`. **Home.** `geometry.surface`. **Effort M.** **Live-code status.**
Absent (needs the new triangle-distance primitive).

---

### GS-5 — Per-vertex / per-face surface area → `sparse.mesh.vertex_areas`

**What.** Mixed Voronoi (barycentric on obtuse triangles) per-vertex area,
and per-face area. The area measure under every area-weighted operation:
`ADAP_BARY_AREA` resampling, areal distortion, anatomical-stats surface area,
mass-matrix-normalised LBO.

**Proposed surface.**

```python
def face_areas(mesh: Mesh) -> Float[Array, 'n_faces']: ...
def vertex_areas(mesh: Mesh, *, scheme: Literal['barycentric', 'voronoi'] = 'voronoi') -> Float[Array, 'n_vertices']: ...
def mesh_mass_matrix(mesh: Mesh, *, scheme: Literal['barycentric', 'voronoi'] = 'voronoi', lumped: bool = True) -> ELL: ...  # diagonal lumped M (== vertex_areas)
```

**Composition.** Cross-product face areas (the magnitude already computed
inside `compute_vertex_normals`) scattered to vertices; the Voronoi / mixed
rule per Meyer et al. — the **obtuse-triangle branch is load-bearing on real
surfaces** (barycentric is the safe fallback). The lumped mass matrix `M` is
the diagonal of `vertex_areas`; ship it first (diagonal → trivially
invertible, which both `M⁻¹L` curvature and the GS-12 backward-Euler solve
need). Pure JAX. **Home.** `sparse.mesh` (the operator / measure layer).
**Effort S.** **Live-code status.** Absent — no area or mass symbol exists; the
cotangent Laplacian is unnormalised stiffness only. **Prerequisite (Phase 0):**
`mesh-curvature` and GS-12 are *incorrect* without `M`; this ships before
either (§0 correction 1).

---

### GS-6 — Areal & strain distortion of a surface warp → `sparse.mesh.distortion`

**What.** The surface analogue of `jacobian_det_displacement`: per-vertex
**areal distortion** `log2(area_warped / area_orig)` and per-triangle
**strain** (principal stretches λ₁, λ₂ — eigenvalues of the Cauchy–Green /
deformation-gradient tensor). The MSM regulariser readout and the
`?h.jacobian` QA metric.

**Proposed surface.**

```python
def areal_distortion(source: Mesh, warped: Mesh) -> Float[Array, 'n_vertices']: ...
def strain_distortion(source: Mesh, warped: Mesh) -> Float[Array, 'n_faces 2']: ...
```

**Composition.** GS-5 areas for the areal term; per-triangle 2×2 first
fundamental forms + `eigh` for the strain tensor. Pure JAX. **Home.**
`sparse.mesh`. **Effort S.** **Live-code status.** Volumetric Jacobian
shipped; no surface/sphere distortion. Pairs with
[field-regularisers](resolved/field-regularisers.md) (strain energy as a *loss*).

---

### GS-7 — Genus-0 topology correction (`mris_fix_topology`) → `geometry.topology` *(combinatorial)*

**What.** Force a tessellated/marching-cubes surface to a topological sphere:
locate topological defects (where the spherical map is non-homeomorphic /
triangles overlap) and retessellate each defect region (Fischl 2001 manifold
surgery; Ségonne 2007 MAP retessellation). **The single hardest classical
primitive** and the one that *cannot* be skipped on the `fastcsr`/`synthdist`
field→mesh route (it is sidestepped only by template-deformation models like
`topofit`, which inherit genus-0 from a fixed icosphere).

**Proposed surface.**

```python
def euler_characteristic(mesh: Mesh) -> int: ...        # cheap: V - E + F
def genus(mesh: Mesh) -> int: ...
def correct_topology(mesh: Mesh, *, intensity: Float[Array, 'X Y Z'] | None = None) -> Mesh: ...
```

**Composition.** `euler_characteristic`/`genus` are trivial combinatorics on
`faces` (ship first — cheap correctness gate). The corrector is host-side
combinatorial (defect localisation + greedy/MAP retessellation), in the same
NumPy-construction idiom as `icosphere`/`mesh_k_ring_adjacency`, emitting a
new `Mesh`. **Home.** New `geometry.topology` (or `bitsjax` if it grows a
heavy solver — flag). **Effort L–XL** (the corrector; `euler`/`genus` are
XS). **Live-code status.** Absent. **Note for consumers:** prefer the
`topofit`-template route to avoid this entirely; reserve the corrector for
level-set models.

**Decision D1 (2026-06-21).** Ship `euler_characteristic` / `genus` now as the
**defect gate** (the signal that tells a consumer the genus-0 escape hatch is
unsafe); **defer the corrector** to a research track. The geometry-light path
bypasses it. Keep the seam `GS-3 → [GS-7 slot] → GS-1` open so a corrector can
slot in later (consume / emit `Mesh`) without re-architecting. Pivot if
friction develops.

---

### GS-8 — Self-intersection detection / removal (`mris_remove_intersection`) → `geometry.surface.intersection`

**What.** Detect and resolve triangle–triangle self-intersections. The
mandatory post-step for `topofit` (`--rsi`), `josa`'s warped spherical mesh,
and any deformable fit (GS-10/GS-11).

**Proposed surface.**

```python
def find_self_intersections(mesh: Mesh) -> Int[Array, 'n_pairs 2']: ...   # face-pair indices
def remove_self_intersections(mesh: Mesh, *, n_iterations: int = 10) -> Mesh: ...
```

**Composition.** Spatial-hash / BVH broad-phase + Möller triangle-triangle
narrow-phase (host-side); removal is local Laplacian relaxation of offending
vertices (reuses `mesh_laplacian_smooth`). **Home.** `geometry.surface`.
**Effort M.** **Live-code status.** Absent. **Use as a post-hoc cleanup, not
an in-loop guard** — the host-side broad/narrow phase cannot run inside a
jitted `fori_loop`, so GS-10 must not call it per-iteration (§0 correction 4).

---

### GS-9 — Cortical thickness → `geometry.surface.thickness`

**What.** Symmetric white↔pial distance (mean of white→pial and pial→white
nearest-point distance; Fischl & Dale 2000), plus the
correspondence-preserving variant when white/pial share vertex indexing.

**Proposed surface.**

```python
def cortical_thickness(
    white: Mesh, pial: Mesh, *,
    method: Literal['symmetric', 'correspondence'] = 'symmetric',
) -> Float[Array, 'n_vertices']:
```

**Composition.** `correspondence` mode is a per-vertex `‖pial − white‖`
(trivial, exact when GS-10 produces pial from white). `symmetric` mode needs
nearest-point-on-mesh queries (a KD-tree / BVH host build; JAX gather for the
distances). **Home.** `geometry.surface`. **Effort S.** **Live-code
status.** Absent.

---

### GS-10 — SDF-guided normal-march deformation → `geometry.surface.deform_to_sdf`

**What.** The recommended *geometry-light* pial step (and a general
field-driven surface mover): advance a genus-0 mesh's vertices along their
normals toward the zero level-set of a target SDF, regularised by smoothness
and a self-intersection guard. Deforming `topofit`'s white surface outward to
`synthdist`'s pial SDF this way **preserves vertex correspondence and
inherits genus-0** — no second marching-cubes + topology-correction pass, and
GS-9 `correspondence` thickness falls out for free.

**Proposed surface.**

```python
def deform_to_sdf(
    mesh: Mesh, sdf: Float[Array, 'X Y Z'], *,
    n_iterations: int = 50, step: float = 0.2, smooth_weight: float = 0.1,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Mesh:
```

**Composition.** `fori_loop`: sample the SDF (and its `spatial_gradient`) at
vertices via `sample_at_points`, step along the normal (`compute_vertex_normals`)
scaled by the local SDF value, apply a Laplacian smoothing fraction
(`mesh_laplacian_smooth`) and a step clamp. **The loop uses only jittable
regularisation — no in-loop GS-8 guard** (host-side, un-jittable; §0 correction
4); run GS-8 as a post-hoc cleanup if needed. Fully differentiable — usable
inside a learned-refinement loss. **Home.** `geometry.surface` (or a `register`
surface recipe). **Effort M.** **Live-code status.** Absent; every in-loop
ingredient is shipped.

---

### GS-11 — Intensity-driven deformable surface fit (`mris_place_surface`) → `register.surface`

**What.** The classical active-surface fallback for when no learned
boundary model is trusted: evolve vertices under internal forces (spring +
non-self-intersection) and an external **target-intensity / image-gradient**
force toward the gray/white (white surface) or gray/CSF (pial) boundary.

**Proposed surface.**

```python
def place_surface(
    mesh: Mesh, image: Float[Array, 'X Y Z'], *,
    target_intensity: float, n_iterations: int = 100,
    smooth_weight: float = 0.2, step: float = 0.1,
) -> Mesh:
```

**Composition.** Same `fori_loop` skeleton as GS-10 with the external force
from `spatial_gradient`/target-intensity instead of an SDF. Reuses the
`register` convergence / line-search machinery (`register._converge`).
**Home.** `register.surface` (it is a registration-style optimiser).
**Effort L.** **Live-code status.** Absent. *Optional* — learned models
(`topofit`/`synthdist`) replace it in the default pipeline; ship for
classic-`recon-all` parity and hard clinical cases. **SPEC-review design
doc:** [`place-surface.md`](place-surface.md) (the Effort-L §13.4 review
artifact; build gated on a concrete consumer).

---

### GS-12 — Geodesic (along-surface) Gaussian smoothing → `smoothing.metric` / `geometry.surface.smooth`

**What.** Smooth a per-vertex scalar along the surface (`wb_command
-metric-smoothing`), not through Euclidean space — the correct smoother for
`sulc`/`curv`/myelin/thickness maps and any surface GLM input.

**Proposed surface.**

```python
def surface_smooth(
    mesh: Mesh, values: Float[Array, '... n_vertices'], *, fwhm: float,
) -> Float[Array, '... n_vertices']:
```

**Composition.** Backward-Euler heat diffusion: solve `(M + tL) x = M·x₀` (one
implicit step, or a few), with `t = FWHM² / (16 ln 2)` and `M` the GS-5 lumped
mass. Solved via the **shipped `linalg.krylov.cg`** on the `(M + tL)` ELL /
SectionedELL operator (SPD), through the apply-seam — *not* a dense `exp(-tL)`,
which does not scale to ico7 (§0 correction 2). Depends on **GS-5** (`M`).
**Decision D2:** ship as the nitrix-native geodesic smoother and **document the
divergence** from `wb_command -metric-smoothing` (a geodesic-distance-weighted
Gaussian); a Workbench-parity variant, if ever needed, arrives behind a
`method=` kwarg. **Home.** `geometry.surface` (`surface_smooth`). **Effort M.**
**Live-code status.** Volumetric Gaussian shipped; no surface / geodesic
smoother; `cg` + (once GS-5 lands) `M` are the ingredients.

---

### GS-13 — Spherical SVF boundary mode (`josa`) → `geometry.sphere_grid` / `geometry.grid`

**What.** The one residual blocker for the `josa` spherical-diffeomorphism
port: `spatial_transform` on the equirectangular sphere grid needs a
`mode='nearest'` boundary option so the pole/seam handling composes with
`sphere_grid_pad_2d` without re-vendoring a sampler. Tracked historically in
the JOSA nitrix-feedback thread.

**Proposed surface.** A boundary-mode keyword on the existing sampler — no
new function. Composes with [boundary-mode-parity](boundary-mode-parity.md)
and [spatial-transform-linear-extrap](spatial-transform-linear-extrap.md).

**Composition.** `sphere_grid_pad_2d` + `integrate_velocity_field` +
`spatial_transform` already provide the padded 2D-grid SVF warp; this is the
boundary keyword. **Home.** `geometry.grid` / `geometry.sphere_grid`.
**Effort S.** **Live-code status.** **Likely already satisfied** —
`spatial_transform` already forwards `mode='nearest'` (the geometry JOSA
sprint), so the equirectangular path composes `sphere_grid_pad_2d` +
`spatial_transform(mode='nearest')`. **Verify with a composition test before
writing code** (§0 correction 6); if it passes, close GS-13 with the test +
a docstring note rather than new code.

---

### GS-14 — Ribbon-constrained volume→surface mapping (HCP myelin) → `geometry.surface.ribbon_map`

**What.** Map a volume onto the surface by integrating voxels in the cortical
column between white and pial with partial-volume / Gaussian weighting
(`wb_command -volume-to-surface-mapping -ribbon-constrained`). The basis of
HCP **myelin maps** (T1w/T2w ratio sampled on the surface).

**Proposed surface.**

```python
def ribbon_map(
    volume: Float[Array, 'X Y Z'], white: Mesh, pial: Mesh, *,
    n_samples: int = 10, weighting: Literal['pv', 'gaussian'] = 'pv',
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Float[Array, '... n_vertices']:
```

**Composition.** Sample `n_samples` points per vertex along the
white→pial segment (interpolated coordinates), `sample_at_points` the volume,
weighted-reduce. Pure JAX. **Home.** `geometry.surface`. **Effort M.**
**Live-code status.** Absent; sampling primitive shipped.

---

## 6. Pipeline coverage matrices

Legend: ✅ shipped · 📋 filed FR (§4) · 🆕 new gap (§5) · 🚫 out of scope
(→ thrux/ilex).

**`recon-all` (classic surface stream)**

| Stage | Coverage |
|---|---|
| WM / pial boundary | ✅ `topofit` / `fastcsr` / `synthdist` (ilex models) |
| Volume→mesh (tessellation / marching cubes) | 🆕 GS-3 (needed for level-set route; `topofit` bypasses) |
| Genus-0 topology correction | 🆕 GS-7 (needed for level-set route; template models bypass) |
| Smoothing | ✅ `mesh_laplacian_smooth` |
| Inflation + sulc | 🆕 GS-1 |
| Spherical parameterisation | 🆕 GS-2 |
| Curvature / sulc features | 📋 mesh-curvature (substrate ✅) |
| Atlas registration | ✅ `sugar` / `josa` (ilex); regulariser readout 🆕 GS-6 |
| Surface I/O | 🚫 thrux |
| Thickness / area / Jacobian | 🆕 GS-9 / GS-5 / GS-6 |
| Self-intersection cleanup | 🆕 GS-8 |
| Orchestration (`recon-all` driver) | 🚫 ilex |

**`recon-all-clinical`** — superset of the field→mesh route: `synthseg` +
`synthsr` + `synthdist` (✅ ilex) then **GS-3 → GS-7 → GS-1 → GS-2 →
registration → GS-9/5/6**. The intensity-driven deformable fit (GS-11) is the
*only* classical stage it removes; marching cubes (GS-3) + topology
correction (GS-7) become **non-optional** (no template escape hatch).

**HCP minimal preprocessing**

| Stage | Coverage |
|---|---|
| PreFS: ACPC / brain mask / MNI affine+warp | ✅ `register` (rigid/affine/SyN) |
| PreFS: T1w↔T2w | ✅ `register._bbr` |
| PreFS: bias field (sqrt T1·T2) | ✅ `bias` |
| PreFS: gradient / EPI distortion | 🆕 1-D PE-axis displacement (+ ilex `gdc_net`/`fd_net`) |
| FreeSurfer (T2w-refined) | as `recon-all` above |
| PostFS: midthickness / inflated | ✅ vertex average / 🆕 GS-1 |
| PostFS: myelin (ribbon map) | 🆕 GS-14 |
| PostFS: MSMSulc / MSMAll | ✅ `sugar` / `josa` (ilex); 🆕 GS-6 distortion regulariser |
| PostFS: `fs_LR_32k` resample | 📋 surface-resample-adap-bary (arbitrary-mesh `ADAP_BARY_AREA`) |
| PostFS: geodesic smoothing | 🆕 GS-12 |
| PostFS: GIFTI/CIFTI assembly | 🚫 thrux |

## 7. Phasing & dependency graph

```
Phase 0  substrate & enablers (NEW — corrects/unblocks everything below)
  B22 pytree lift (ELL/SectionedELL/Mesh; decline IcosphereHierarchy)
  GS-5 vertex/face area + lumped mass matrix M
  SectionedELL operator emission (arbitrary meshes) + {ELL,SectionedELL} apply-seam
  real-FS-mesh test fixture (nilearn/templateflow, test-only IO)

Phase 1  measure (compose on substrate) — unblock features + QA
  GS-5 ──┬─> GS-6 areal/strain distortion
         ├─> (mesh-curvature 12.6: H = M⁻¹L v) ──> sugar/josa features
         └─> GS-12 geodesic smoothing  ((M+tL)x=Mx₀ via linalg.krylov.cg)

Phase 2  field <-> mesh + geometry-light movers
  GS-3 marching cubes (asymptotic-decider) ──> euler/genus defect gate ──[GS-7 slot]
  GS-4 mesh->SDF (point-to-triangle distance, NEW prim)
  GS-10 SDF normal-march (jittable; no in-loop GS-8) ──> GS-9 thickness (corr.)
  GS-8 self-intersection (post-hoc cleanup, not in-loop)

Phase 3  hard continuous optimisers (SPEC-review-gated, §13.4)
  GS-1 inflation+sulc ──> GS-2 spherical parameterisation (L) ──> sugar/josa inputs

Phase 4  HCP back-end
  surface-resample-adap-bary (12.15)   GS-14 ribbon map   GS-13 josa boundary (verify shipped)

Phase 5  parcellation + optional/research-tracked
  surface-boundary-map (12.16) ──> mesh-watershed (12.17)
  GS-11 intensity-driven deformable fit (optional; L, SPEC-gated)
  GS-7 corrector / DEC (12.5) / SHT (12.9): research-tracked on a named consumer
```

Rationale: Phase 0 is pure substrate that makes the rest *correct* (mass
matrix) and *differentiable* (pytree lift) and *efficient on real meshes*
(SectionedELL) — low risk, unblocks everything. Phase 1 is composition on that
substrate (low risk, high immediate value — features for the learned registrars
+ morphometry). Phase 2 unlocks the `fastcsr`/`synthdist` field route and the
geometry-light pial strategy. Phase 3 isolates the two genuinely hard
continuous optimisers behind the §13.4 SPEC-review gate. Phase 4 finishes HCP.
Phase 5 adds parcellation and quarantines the research-grade corrector. The
**geometry-light path** (`topofit` white → GS-10 SDF-march to pial → GS-1 →
GS-2 → `sugar`/`josa`) never invokes GS-3/GS-7 and is the recommended default
(decision D1). Full per-task detail, signatures, and the test matrix live in
the implementation plan — [`docs/design/geometry-suite.md`](../design/geometry-suite.md).

## 8. Cross-references

- **Implementation plan.** [`docs/design/geometry-suite.md`](../design/geometry-suite.md)
  — the concrete phased build (per-task signatures, contracts, differentiability
  classes, the test matrix, governance/graduation records, and the risk
  register). This ledger is the *what/why*; that doc is the *how*.
- **Acceptance.** `docs/feature-requests catalogue §12` (brainstorm catalogue — the §4
  items 12.5/12.6/12.9/12.15/12.16/12.17 are this suite's surface members) /
  `§13` (acceptance protocol). New items GS-1/2/3/4/7/8/9/10/13/14 are
  candidate additions beyond the current §12 set.
- **Dependency contract.** `SPEC.md §5` (no `nibabel`/filesystem; container
  IO → `thrux`).
- **Design context.** `docs/design/geometry.md`, `docs/design/sphere-grid.md`,
  `docs/design/mesh-graph-conv.md`, `docs/design/registration.md`.
- **Live substrate.** `src/nitrix/sparse/mesh.py`,
  `src/nitrix/geometry/{sphere,sphere_grid,grid,differential,transform,affine}.py`,
  `src/nitrix/register/`, `src/nitrix/bias/`.
- **Consumer side.** [`ilex-pipeline-substrate.md`](ilex-pipeline-substrate.md)
  (the surfa-side projection scope boundary) and the existing atomised FRs
  indexed in §4.
