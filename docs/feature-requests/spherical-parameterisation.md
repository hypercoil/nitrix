# GS-2 — Spherical parameterisation (`mris_sphere`) → `geometry.sphere.spherical_parameterize`

> **Status (2026-06-22): SPEC-review design doc / FR.** GS-2 is the geometry
> suite's hardest continuous optimiser (Effort **L**), which `SPEC_UPDATE_v0.3
> §13.4` flags for **SPEC-level review before it lands**. This document is that
> review artifact: it fixes the rationale, the mathematical design and its
> components, the phased implementation process, the test plan, and the risk
> register, so the build is a transcription of an agreed design rather than an
> exploration. Parent ledger: [`geometry-suite.md`](geometry-suite.md) §5 GS-2;
> implementation plan: [`../design/geometry-suite.md`](../design/geometry-suite.md)
> §6 P3.2. Phases 0–2 of the suite are built and green; this is the keystone of
> Phase 3.

## 1. Why this needs a design doc

Every other geometry-suite primitive either (a) has a closed-form / exact
answer (areas, curvature, distortion, marching tetrahedra) or (b) is a
*convex-ish* relaxation that converges from any start (geodesic smoothing's SPD
solve, the SDF normal-march toward a fixed target). GS-2 is neither:

- The decision variables live on a **nonlinear constraint manifold** — the unit
  sphere `(S²)ⁿ`, not free `ℝ³`.
- The map must be a **homeomorphism** (bijective, fold-free): every spherical
  triangle must keep positive oriented area. A step that flips one triangle
  silently destroys the result, and naive distortion descent *does* flip
  triangles.
- The natural conformal energy has a **trivial degenerate minimiser** (every
  vertex collapsing to a single point — energy 0). Avoiding it is not optional;
  it is the load-bearing subtlety.
- The energy with a fold barrier is **non-convex**, so the *initialisation* and
  the *step control* — not the energy alone — determine success.

These are exactly the properties §13.4 means by "needs a SPEC-level review
first." The cost of getting it wrong is a primitive that passes a sphere
smoke-test and then produces folded garbage on a real inflated cortex. So we
specify the four components precisely and validate bijectivity as a hard
invariant at every step, not just the final distortion.

## 2. The problem

Given an **inflated, genus-0** triangle mesh `M = (V, F)` (the output of GS-1
`inflate_surface`, topologically a sphere), find vertex positions
`Φ = {φ_i} ⊂ S²` (the unit sphere) such that the piecewise-linear map
`V → Φ` is a **bijection** `M → S²` and **minimises geometric distortion**
(area + metric, with a conformal/angle term as the well-behaved core).

This is the discrete realisation of the uniformisation theorem (every genus-0
surface is conformally equivalent to the round sphere); it is FreeSurfer's
`mris_sphere`, and the spherical domain every surface registrar
(`sugar`≈MSMSulc, `josa`≈MSMAll) and every `fsaverage`/`fs_LR` resampling acts
on.

**Distinct from `cartesian_to_latlong`** (which only re-coordinatises an
*already* spherical mesh) and from `deform_to_sdf` (which moves toward a *fixed*
target field): here the target is the sphere itself and the objective is the
intrinsic distortion of the embedding.

## 3. Scope boundary

In scope (SPEC §5): the array math producing `Φ` from plain `(V, F)` arrays.
Out of scope: reading `?h.inflated`/`?h.sphere` binaries, `$SUBJECTS_DIR`
(→ `thrux`/consumer). The validation harness reads `fsaverage` surfaces
**test-side only** (the established `tests/_real_meshes.py` seam), including
FreeSurfer's own `?h.sphere` as a distortion oracle.

## 4. Proposed surface

```python
# geometry.sphere  (the embedding optimiser sits beside the S²-grid helpers)
def spherical_parameterize(
    mesh: Mesh,
    *,
    init: Literal['tutte', 'radial'] = 'tutte',
    n_iterations: int = 200,
    conformal_weight: float = 1.0,
    area_weight: float = 1.0,
    metric_weight: float = 0.0,
    step: float = 0.1,
    radius: float = 1.0,
) -> Float[Array, 'n_vertices 3']:   # vertices on the sphere of given radius
    ...

# Supporting primitives (also independently useful; GS-2a / GS-2b).
def signed_spherical_areas(
    vertices: Float[Array, 'n_vertices 3'], faces: Int[Array, 'n_faces 3'],
) -> Float[Array, 'n_faces']: ...          # signed solid-angle per triangle

def is_bijective_sphere_map(
    vertices: Float[Array, 'n_vertices 3'], faces: Int[Array, 'n_faces 3'],
    *, tol: float = 1e-3,
) -> bool: ...                              # all areas > 0 AND sum ≈ 4π

def tutte_embedding(
    mesh: Mesh, *, weights: Literal['uniform', 'cotangent'] = 'uniform',
) -> Float[Array, 'n_vertices 2']: ...      # guaranteed fold-free planar disk
```

`spherical_parameterize` returns vertices on `S²` with the **same `faces`**
(the parameterisation never re-tessellates — correspondence is preserved, like
`deform_to_sdf`). The default `init='tutte'` is the robust, bijectivity-
guaranteed path; `init='radial'` is the fast path for near-convex inputs.

## 5. Design — the four components

### 5.1 Initialisation — guaranteed-bijective by construction

A fold-free *start* is the single most important ingredient: from a bijective
init the optimiser only has to *reduce distortion*, never *recover* bijectivity
(which a from-scratch non-convex solve cannot guarantee).

**Primary: Tutte embedding → inverse stereographic projection.**
1. Cut one face `f₀` (or one vertex's star); `M ∖ f₀` is a topological **disk**.
2. Fix `f₀`'s boundary vertices to a convex polygon in the plane.
3. Solve the **uniform-weight** (combinatorial) Laplace system
   `L_uniform · x_interior = b` for the interior 2-D positions (one CG solve
   per coordinate via the shipped `linalg.krylov.cg`). **Tutte's theorem**
   (1963): for a 3-connected planar graph with the outer face fixed convex, the
   *uniform-weight* barycentric embedding is **guaranteed fold-free**. (Cotangent
   weights do *not* carry this guarantee — they can flip on non-Delaunay meshes
   — so the init uses uniform weights deliberately; cotangent weights enter only
   later, in the energy.)
4. **Inverse-stereographic-project** the planar embedding onto `S²` (a
   homeomorphism), so the spherical map inherits bijectivity. `f₀` maps to the
   neighbourhood of the projection pole.

The pole region carries large area distortion — irrelevant: the init only has
to be bijective; §5.2–5.4 remove the distortion. This is the standard robust
spherical-parameterisation start (Gu et al. 2004; Choi et al. 2015).

**Fast path: radial projection.** For an *already near-convex* inflated surface,
`φ_i = radius · (v_i − c) / ‖v_i − c‖` (centroid `c`) is often already
bijective. `spherical_parameterize(init='radial')` uses it, then
**`is_bijective_sphere_map` gates it**: if any triangle is flipped, fall back to
Tutte (a loud, automatic fallback — never silently optimise from a folded
start).

### 5.2 The energy — and the collapse trap

Per-triangle / per-edge terms, all differentiable (JAX autodiff), with weights
from the **original** mesh `M`:

- **Conformal (Dirichlet) core** `E_conf = ½ Σ_(i,j) w_ij ‖φ_i − φ_j‖²` with
  `w_ij` the **cotangent** weights of `M` (the shipped
  `mesh_cotangent_laplacian` / `_cotangent_apply`). Minimised by the harmonic /
  conformal map — the smooth, well-behaved core.
- **Areal** `E_area = Σ_f A⁰_f · log²(a_f / A⁰_f)` where `a_f` is the spherical
  triangle area (GS-2a `signed_spherical_areas`) and `A⁰_f` the original face
  area (shipped `face_areas`), normalised so `Σ A⁰ = Σ a = 4π·radius²`.
- **Metric/isometric** (optional, `metric_weight`) `E_metric = Σ_(i,j) (‖φ_i −
  φ_j‖ − ℓ⁰_ij)²` on 1-ring edges — FreeSurfer's distance-preservation term.

**The collapse trap (load-bearing).** `min E_conf` over maps to `S²` has the
trivial global minimiser "all `φ_i` → one point" (energy 0). Any pure-conformal
descent will drift toward it. Two defences, both required:

1. **The area term** `E_area → ∞` as any `a_f → 0`, penalising the global
   shrink.
2. **Möbius / centroid normalisation** each iteration (§5.4) — removes the
   conformal gauge along which collapse happens.

This is exactly the conformal-energy-minimisation (CEM, Choi et al. 2015)
insight (`E_conf − Area` has no collapse minimiser); we realise it through the
area term + the normalisation step rather than a bespoke energy, so the same
machinery covers the area/metric-controlled (non-purely-conformal) objective
FreeSurfer/MSM want.

### 5.3 Riemannian optimisation on `S²` with a fold-safe line-search

Vertices are constrained to the sphere; we do **projected (Riemannian) gradient
descent on `(S²)ⁿ`**:

1. Euclidean gradient `g = ∇_Φ E` (JAX `grad` — the energy is differentiable).
2. **Tangent projection**: at each vertex remove the radial component,
   `g⊥_i = g_i − (g_i · φ̂_i) φ̂_i` (move along the sphere, not off it).
3. **Step + retraction**: `φ_i ← normalize(φ_i − α g⊥_i) · radius` (renormalise
   back to `S²` — the retraction).
4. **Fold-safe backtracking line-search** (the "robust line-search"): accept the
   step `α` only if it (a) decreases `E` and (b) keeps **every**
   `signed_spherical_areas > 0`; otherwise halve `α` and retry
   (`jax.lax.while_loop`). This makes bijectivity an **invariant of the
   iteration**, not just a hoped-for property of the optimum — the optimiser can
   never leave the fold-free region.

A log-barrier `−μ Σ log(a_f)` on the signed areas can be added to *repel* from
the fold boundary (smoother than rejection alone); the line-search remains the
hard guarantee. Outer loop is `lax.fori_loop` (jittable).

### 5.4 Gauge fixing — Möbius / centroid normalisation

Spherical conformal maps are unique only up to the 3-parameter Möbius group of
`S²`; left free, the optimiser drifts along this gauge and (with the conformal
term) collapses. After each step, apply the **area-weighted centroid
normalisation** (Choi et al.): translate/rescale on the sphere so the
area-weighted centroid of `Φ` returns to the origin, then renormalise to
`S²`. This fixes the translational Möbius gauge and is the concrete mechanism
that blocks collapse. (The remaining rotational gauge is harmless — a global
rotation of a valid spherical map is still valid; optionally anchor one vertex /
align principal axes for reproducibility.)

## 6. Substrate reuse — almost everything is shipped

| Need | Shipped primitive |
|---|---|
| Cotangent weights / Dirichlet energy | `sparse.mesh_cotangent_laplacian`, `geometry.surface._cotangent_apply` |
| Tutte linear solve | `linalg.krylov.cg` (SPD, matrix-free, GPU-native) |
| Uniform Laplacian (Tutte weights) | `sparse.mesh_k_ring_adjacency(binary=False)` / a uniform variant |
| Original areas / areal distortion | `sparse.face_areas`, `geometry.surface.areal_distortion` |
| Signed spherical area / bijectivity | **GS-2a (new)** — the `_solid_angle` formula already written for `isosurface.mesh_to_sdf` (Van Oosterom–Strackee) is exactly the signed solid angle; lift it to a shared helper |
| Energy gradient | `jax.grad` (the energy is pure JAX) |
| Fold-safe line-search / outer loop | `jax.lax.while_loop` / `fori_loop` |
| Coordinate conversions / geodesics | `geometry.sphere.{cartesian_to_latlong, spherical_geodesic_distance}` |
| Validation oracle | `fsaverage` `sphere_left` (test-side, via `tests/_real_meshes.fsaverage_surface`) |

Net-new numerics: GS-2a (signed spherical area + bijectivity check — small,
reuses the solid-angle formula), GS-2b (Tutte+stereographic init), GS-2c (the
energy + Riemannian descent + fold-safe line-search + normalisation).

## 7. Differentiability & hardware

- **Forward is JAX / jittable** (DIFF-JAX-forward): energy, gradient, tangent
  projection, retraction, line-search (`while_loop`), normalisation, outer loop
  (`fori_loop`) are all pure JAX → runs on GPU, no host round-trip. The
  *initialisation* Tutte CG solve is also JAX; the face-cut bookkeeping is a
  small host-side step (the icosphere idiom).
- **Differentiating *through* the optimiser** (grad of `Φ` w.r.t. input `V`) is
  **out of scope for v1**: spherical parameterisation is a preprocessing
  artifact, not usually backpropagated through. If a learned consumer needs it,
  the principled route is implicit differentiation at the energy minimiser (IFT
  via the energy Hessian) rather than unrolling — a documented follow-up, not a
  v1 requirement.
- GPU posture matches the rest of the mesh tier: XLA-lowered gather/scatter +
  loops, no custom Pallas kernel.

## 8. Implementation process (phased, each independently testable)

Build bottom-up so bijectivity is verifiable before the hard optimiser exists:

- **GS-2a — signed spherical area + bijectivity measures.** `signed_spherical_areas`
  (signed solid angle per triangle), `is_bijective_sphere_map` (all areas `> 0`
  **and** `Σ areas ≈ 4π` — the degree-1 cover test). Lift `_solid_angle` from
  `isosurface` into `geometry/_triangle_distance.py` (or a small shared helper)
  so both use one implementation. *Effort XS.* Gate for everything below.
- **GS-2b — Tutte + stereographic initialisation.** `tutte_embedding`
  (uniform-weight disk solve via `cg`) + inverse stereographic projection;
  `init='radial'` fast path with the `is_bijective_sphere_map` auto-fallback.
  *Effort M.* Exit: a **provably bijective** spherical map of a real inflated
  white surface (every signed area `> 0`).
- **GS-2c — the energy + Riemannian optimiser.** `E_conf`/`E_area`/`E_metric`,
  tangent-projected gradient descent, the **fold-safe backtracking line-search**,
  and the **Möbius/centroid normalisation**. *Effort L* (the core). Exit:
  starting from the GS-2b init, distortion decreases monotonically, bijectivity
  holds at **every** iteration, and it converges.
- **GS-2d — assembly + validation harness.** `spherical_parameterize` wiring +
  the §9 test matrix (icosphere identity, bijectivity invariant, real-mesh vs
  FS `?h.sphere`).

Each phase commits independently (per the frequent-commit practice), bijectivity
asserted from GS-2a onward.

## 9. Validation / test plan

Two oracle classes, mirroring the rest of the suite:

1. **Analytic / invariant.**
   - **Identity**: an already-spherical input (the icosphere) → output ≈ input
     up to a Möbius/rotation gauge; distortion ≈ 0; bijective.
   - **Bijectivity invariant** (the hard one): `is_bijective_sphere_map` holds
     on the output, and — asserted *throughout* GS-2c — at every iteration
     (`signed_spherical_areas > 0` always; `Σ = 4π`). A test that perturbs
     toward a fold confirms the line-search rejects it.
   - **No-collapse**: output area-spread is bounded (not collapsed to a point) —
     `min(signed_spherical_areas) > 0` and the area histogram is sane.
   - **Monotone descent**: total energy decreases over iterations.
2. **Real cortical surface.** On `fsaverage` inflated/white (genus-0):
   - the result is **bijective** (every signed area `> 0`, `Σ = 4π`) — the
     property that matters most in the wild;
   - **areal & metric distortion** (shipped `areal_distortion` /
     `strain_distortion`) are low and comparable to FreeSurfer's own
     `?h.sphere` distribution (class/correlation, not bit-parity — the
     metrics↔ITK posture);
   - vertex correspondence preserved (faces unchanged).

## 10. Risk register

| Risk | Severity | Mitigation |
|---|---|---|
| **Collapse to a point** (conformal trivial minimiser) | High | area term + Möbius/centroid normalisation each iteration (§5.2/5.4); no-collapse test |
| **Folds during optimisation** | High | bijective Tutte init + fold-safe line-search (bijectivity is a loop invariant) + optional log-barrier |
| **Non-convergence / slow at ico7 (160k verts)** | Med | Tutte init puts us in a good basin; CG for the linear solve; document iteration budget; coarse-to-fine over the icosphere hierarchy is a lever if needed |
| **fp32 conditioning** | Low | the domain is the *unit* sphere (well-scaled) once initialised; areas/energies are O(1) |
| **Gauge drift / non-reproducibility** | Low | centroid normalisation fixes translation; optional axis-anchor fixes rotation |
| **Tutte 3-connectivity assumption violated** (degenerate mesh) | Low | document the precondition (a clean genus-0 manifold mesh); `is_bijective_sphere_map` catches a bad init and the radial→Tutte fallback is loud |
| Differentiate-through demanded by a consumer | Low | documented out-of-scope for v1; implicit-diff follow-up if a named consumer arrives |

## 11. Governance & graduation

§13 four-gate: **consumer** — `sugar`/`josa` and every `fsaverage`/`fs_LR`
registration require the spherical map (✓ named); **composition** — this doc
sketches it on the shipped substrate (✓); **SoC** — lives in `geometry.sphere`,
composes existing primitives, adds no parallel structure (✓); **effort** — L,
hence this **SPEC-level review** (this document). On approval, GS-2 graduates
into Phase 3 of the implementation plan; record the §12→§10.A graduation in
`IMPLEMENTATION_PLAN.md §10` at merge (GS-2a/b/c/d as the sub-entries), and
backfill the as-built decision record (the chosen init, energy weights, and the
real-mesh distortion vs FS `?h.sphere`) into
[`../design/geometry-suite.md`](../design/geometry-suite.md) §6, mirroring the
P1.1 curvature record.

## 12. Cross-references & literature

- **Suite docs.** [`geometry-suite.md`](geometry-suite.md) §5 GS-2 (parent),
  [`../design/geometry-suite.md`](../design/geometry-suite.md) §6 P3.2,
  [`field-regularisers.md`](field-regularisers.md) (strain/bending energies),
  GS-1 inflation (the upstream producer of the inflated input).
- **Substrate.** `sparse/mesh.py` (cotangent), `linalg/krylov.py` (`cg`),
  `geometry/surface.py` (`_cotangent_apply`, `areal_distortion`,
  `strain_distortion`), `geometry/isosurface.py` (`_solid_angle`),
  `geometry/sphere.py`, `tests/_real_meshes.py`.
- **Governance.** `SPEC_UPDATE_v0.3.md §13` (acceptance, §13.4 L-review),
  `SPEC.md §5` (dep contract).
- **Literature.** Tutte 1963 (barycentric embedding / fold-free guarantee);
  Gu, Wang, Chan, Thompson, Yau 2004 (genus-zero surface conformal mapping);
  Choi, Lam, Lui 2015 (FLASH / conformal-energy-minimisation, the collapse-free
  spherical conformal map + Möbius normalisation); Smith & Schaefer 2015
  (bijective parameterisation / flip-preventing energies); Fischl, Sereno, Dale
  1999 (`mris_sphere`, the FreeSurfer metric/area relaxation); Kazhdan, Solomon,
  Ben-Chen 2012 (mean-curvature-flow spherical conformal — an alternative init).
