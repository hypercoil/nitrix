# GS-11 — Intensity-driven deformable surface fit (`mris_place_surface`) → `register.surface.place_surface`

> **Status (2026-06-22): SPEC-review design doc / FR.** GS-11 is an Effort **L**
> optimiser, which `SPEC_UPDATE_v0.3 §13.4` flags for **SPEC-level review before
> it lands** — and, unlike GS-2, it is **explicitly optional**: learned boundary
> models (`topofit` / `fastcsr` / `synthdist`, ilex) replace it in the default
> pipeline, so this document must justify *whether* to build it as much as
> *how*. This is that review artifact: rationale (incl. the optional-justification),
> the mathematical design and its components, the phased implementation process,
> the test plan, and the risk register, so a build is a transcription of an
> agreed design. Parent ledger: [`geometry-suite.md`](geometry-suite.md) §5
> GS-11; implementation plan:
> [`../design/geometry-suite.md`](../design/geometry-suite.md) §8 P5.2. Phases
> 0–5 of the suite (non-optional scope) are built and green; this is the one
> remaining buildable item and is **gated, not scheduled**.

## 1. Why this needs a design doc

Two reasons, and the first is unusual for this suite.

**(a) It is optional — the doc must justify building it at all.** The default
nitrix cortical pipeline is *geometry-light*: a learned model
(`topofit`/`fastcsr`/`synthdist`) emits the white/pial boundary and the suite's
`deform_to_sdf` (GS-10) snaps a template mesh onto it. `place_surface` is the
**classical active-surface fallback** for the cases that path does not cover:

- **No trusted learned model** for the contrast/population at hand (a new
  scanner, a paediatric or lesioned brain, a non-human or ex-vivo sample) — the
  intensity-driven fit needs only the image, not a trained network.
- **Classic `recon-all` parity** — reproducing the FreeSurfer
  `mris_place_surface` white/pial surfaces exactly, for method comparison or for
  a pipeline that must match a published FreeSurfer result.
- **`recon-all-clinical` hard cases** — when `synthdist`'s distance field is
  unreliable in a region, an intensity-driven refinement is the salvage step.

So GS-11 is shipped as a *capability*, not a default. The doc's job is to draw
that scope line cleanly (it is a `register`-style optimiser, not a default
`geometry` mover) and to keep it from accreting into the happy path.

**(b) The optimiser is genuinely hard.** Like GS-2 it is non-convex with a
load-bearing geometric invariant — but here the invariant is **non-self-
intersection**, and the architecture forbids the obvious enforcement:

- The surface evolves under **competing forces** — an *external* image force
  pulling toward a boundary and *internal* forces (smoothness, curvature)
  resisting it. The balance, not either term alone, determines the result.
- A deformable surface **self-intersects** when a sheet folds through another
  (the GS-8 failure). The classical fix is an explicit collision test +
  intersection removal **inside** the evolution loop — but nitrix's mesh tier is
  jitted (`fori_loop`/`scan`) and `find_self_intersections` is **host-side
  numpy** (BVH/Möller, the HOST-QA class), which **cannot run inside a jitted
  loop** (the suite's §2.5 rule and the GS-8/GS-10 correction). The in-loop
  defence must therefore be a *differentiable, jittable* surrogate, with the
  exact host-side cleanup confined to **between** jitted chunks. Getting that
  split right is the keystone subtlety (§5.2).
- The external force depends on **image intensity contrast**, which nitrix does
  not normalise (that is a consumer/`synthsr` step). The force model must be
  robust to, or explicit about, that dependency (§5.4).

These are exactly the "passes a synthetic smoke-test, then produces a folded /
mis-placed surface on a real volume" risks §13.4 means to catch before a build.

## 2. The problem

Given an intensity **image** `I : ℝ³ → ℝ` (sampled on a grid) and an **initial
genus-0 surface** `M = (V, F)` near the target boundary (e.g. a smoothed initial
white-matter tessellation, or an existing white surface when placing the pial),
evolve the vertices to `V'` so the piecewise-linear surface `(V', F)` sits on the
target tissue boundary — the **gray/white** interface (white surface) or the
**gray/CSF** interface (pial surface) — while remaining a **non-self-intersecting,
smooth, genus-0** manifold in vertex correspondence with `M` (faces unchanged).

This is FreeSurfer's `mris_place_surface` (the modern replacement for
`mris_make_surfaces`): the deformable-model / active-surface step of classical
cortical reconstruction (Dale–Fischl–Sereno 1999). It is **distinct** from
`deform_to_sdf` (GS-10), which snaps a mesh to the zero-set of a *given* signed
distance field: there the target is explicit and smooth; here the target is
*implicit in the image intensity* and must be inferred per vertex from the local
intensity profile, with no precomputed field.

## 3. Scope boundary

In scope (SPEC §5): the array math producing `V'` from plain `(image, V, F)`
arrays + scalar parameters. **Out of scope:**

- Reading MGZ/NIfTI volumes or surface binaries, `$SUBJECTS_DIR` (→
  `thrux`/consumer).
- **Intensity normalisation / bias correction** (the `target_intensity` model
  assumes a normalised image, e.g. FreeSurfer's `brain.finalsurfs` at
  white≈110; producing that is `register._bbr`/`bias`/`synthsr`/consumer work).
  The `force='gradient'` mode (§5.1) is the contrast-agnostic escape when
  normalisation is unavailable.
- The `recon-all` **orchestration** (which surface, in which order, with which
  masks) → ilex.

The validation harness reads images/surfaces **test-side only** (the established
`tests/_real_meshes.py` seam) and uses synthetic analytic volumes for the exact
oracles (§9).

## 4. Proposed surface

```python
# register.surface  (a registration-style optimiser: external image force +
# internal regularisers + a convergence loop -- hence register, not geometry)
def place_surface(
    mesh: Mesh,
    image: Float[Array, 'X Y Z'],
    *,
    boundary: Literal['white', 'pial'] = 'white',
    force: Literal['intensity', 'gradient', 'combined'] = 'intensity',
    target_intensity: float | None = None,   # required for 'intensity'/'combined'
    inside_brighter: bool = True,             # gray/white contrast polarity
    n_iterations: int = 100,
    intensity_weight: float = 1.0,
    gradient_weight: float = 0.5,             # 'combined' only
    smooth_weight: float = 0.2,               # tangential Laplacian fraction
    curvature_weight: float = 0.1,            # bending resistance
    step: float = 0.1,
    max_thickness: float | None = None,       # pial: cap expansion from white
    cleanup_every: int | None = 25,           # host-side de-intersection cadence
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    convergence: Convergence | None = None,   # reuse register._converge
) -> Mesh:                                     # same faces (correspondence kept)
    ...
```

Returns a `Mesh` with the **same `faces`** (no re-tessellation; correspondence
preserved, like `deform_to_sdf`). `boundary='pial'` expects `mesh` to be the
*white* surface and expands it outward to the gray/CSF boundary, optionally
capped by `max_thickness` (the cortical-thickness prior). `cleanup_every=None`
disables the host-side de-intersection pass (pure jitted march); an `int` runs
`remove_self_intersections` every that-many iterations **between** jitted chunks
(§5.2). `convergence=None` runs a fixed `n_iterations` `lax.scan` (reproducible,
reverse-differentiable); a `Convergence` early-exits via `while_loop` (no reverse
rule — §7).

## 5. Design — the components

### 5.1 The external force — an intensity isosurface, sought along the normal

The external force moves each vertex **along its (outward) normal** `n_i`
(`compute_vertex_normals`) — tangential image motion would only shear the
parameterisation. Three force models:

- **`force='intensity'` (the `mris_place_surface` model).** Place the vertex on
  the `I = target_intensity` isosurface. Sample `I` and its directional
  derivative along the normal (`spatial_gradient` · `n_i`) at the current vertex
  (via `sample_at_points` at `v_i / spacing`), then take a **damped 1-D Newton
  step** toward the level set:

  ```
  d_i = (target_intensity − I(v_i)) / clamp(∇I(v_i)·n_i, ε)
  v_i ← v_i + intensity_weight · clamp(d_i, ±step_max) · n_i
  ```

  `inside_brighter` fixes the contrast polarity (gray/white: WM brighter →
  outward motion when `I < target` inside). This is exactly the GS-10
  isosurface-seek with the intensity residual `I − target` playing the role of
  the SDF value, and the directional derivative replacing the unit SDF gradient.
- **`force='gradient'` (contrast-agnostic).** Move toward the **maximum
  intensity-gradient magnitude** along the normal — the tissue *edge* — without a
  target value: a short normal-line search of `|∇I|` (sample at a few `±n_i`
  offsets, step toward the max). Robust when the image is not intensity-
  normalised; the escape hatch §3 promises.
- **`force='combined'`.** `intensity_weight · f_intensity + gradient_weight ·
  f_gradient` — FreeSurfer uses both (the intensity term places, the gradient
  term sharpens to the edge).

All three are pure JAX (`sample_at_points` + `spatial_gradient` are shipped and
differentiable), so the external force is **DIFF-JAX** and GPU-native.

### 5.2 Internal forces + the non-self-intersection trap (load-bearing)

Internal forces regularise the evolution; **all must be jittable** (they live
inside the loop):

- **Tangential smoothing** — a `smooth_weight` fraction of `mesh_laplacian_smooth`
  applied to the *tangential* component only (project out the normal so smoothing
  does not undo the external placement). Unfolds spikes; the primary stability
  term.
- **Curvature / bending resistance** — `curvature_weight · ` a mean-curvature
  penalty (`mean_curvature`, shipped) that resists high-bending excursions, the
  precursors of self-intersection.
- **Area-positivity barrier** — keep every triangle's area positive and bounded
  (reuse the `signed`-area / face-area machinery): a jittable hinge that repels
  *local* fold-over, the most common intersection cause.

**The trap.** None of the above prevents a *global* self-intersection (two
distant sheets passing through each other). The classical fix is an in-loop
collision test + removal — but `find_self_intersections` / `remove_self_intersections`
are **host-side numpy (HOST-QA)** and the suite's §2.5 rule forbids calling them
inside a jitted `fori_loop`/`scan`. The resolution (mirroring the GS-10
decision, "in-loop regularisation only — no in-loop GS-8 guard"):

> **In-loop: jittable surrogates only** (tangential smoothing + curvature +
> area-positivity + bounded step + a fold-safe step rejection). **Between jitted
> chunks: the exact host-side cleanup.** The march runs in segments of
> `cleanup_every` iterations; after each segment, if `cleanup_every` is set, the
> host-side `remove_self_intersections` (GS-8) repairs any intersection that
> slipped past the surrogates, and the next jitted segment resumes from the
> repaired mesh.

This is the architecturally honest split: the hot loop stays on the GPU; the
serial host-side geometry cleanup is amortised across chunks and never blocks
the jit. `cleanup_every=None` opts out entirely (when the surrogates suffice,
e.g. a well-behaved target and small steps).

### 5.3 The optimiser — bounded normal march with fold-safe stepping

Each iteration: external force (§5.1) + internal forces (§5.2) → a bounded
vertex update, with a **fold-safe step**: reduce the step (or reject the
per-vertex move) if it would make a triangle area non-positive — the
spherical-parameterise line-search pattern, reused so non-degeneracy is a loop
invariant of the *jitted* part (global intersection is handled by §5.2's
cleanup, not here). The outer loop reuses **`register._converge.run_iterations`**
— `lax.scan` for a fixed budget (the default; reproducible, `vmap`-batchable,
reverse-differentiable) or `lax.while_loop` with a `Convergence` (early-exit on a
small mean displacement; no reverse rule). For the pial (`boundary='pial'`), the
external force is biased outward and the per-vertex displacement from the white
surface is capped at `max_thickness` (the cortical-thickness prior that stops the
pial leaking across a sulcus into the opposite bank).

### 5.4 The intensity-model assumption (stated, not hidden)

`force='intensity'` assumes a **normalised** image (a known `target_intensity` —
e.g. WM≈110 in FreeSurfer's `brain.finalsurfs`, or `0.5` for a probability map).
nitrix does not normalise; the docstring states this loudly and points to the
contrast-agnostic `force='gradient'` mode for un-normalised inputs (the loud-
fallback non-negotiable). `inside_brighter` makes the contrast polarity explicit
rather than guessed.

## 6. Substrate reuse — most of it is shipped

| Need | Shipped primitive |
|---|---|
| The `fori_loop`/`scan` normal-march skeleton | `geometry.surface.deform_to_sdf` (GS-10) — the direct template |
| Sample image + directional derivative at vertices | `geometry.sample_at_points`, `geometry.spatial_gradient` |
| Outward vertex normals | `sparse.compute_vertex_normals` |
| Tangential smoothing | `sparse.mesh_laplacian_smooth` |
| Curvature / bending term | `geometry.surface.mean_curvature` |
| Area-positivity barrier | `sparse.face_areas` / the signed-area machinery (GS-2a) |
| **Host-side de-intersection (between chunks)** | `geometry.intersection.{find,remove}_self_intersections` (GS-8) |
| Convergence / early-exit loop | `register._converge.run_iterations` + `Convergence` |
| External-force vocabulary / combinator precedent | `register._force.Force` / `SumForce` (the design analogue for `intensity`+`gradient`) |
| Validation oracles | synthetic analytic volumes; `tests/_real_meshes` for a real T1-like smoke |

Net-new numerics: the **external intensity/gradient force** (§5.1) and the
**in-loop area-positivity + tangential-smoothing + fold-safe stepping** wiring
(§5.2/5.3) — small additions over the `deform_to_sdf` skeleton. No new kernel,
no new solver, no Pallas.

## 7. Differentiability & hardware

- **Forward is JAX / jittable** (DIFF-JAX-forward) for the marching segments:
  forces, updates, fold-safe step, `scan`/`fori_loop` are pure JAX → GPU-native.
  The **host-side de-intersection** (§5.2) is, by construction, *not* jitted and
  *not* differentiable — it runs between chunks (HOST-QA), exactly like the rest
  of GS-8. With `cleanup_every=None` the whole forward is a single jitted march.
- **Differentiating *through* `place_surface`** (grad w.r.t. `image` or initial
  `V`) is **out of scope for v1**: it is a preprocessing/fallback artifact, and
  the host-side cleanup breaks the tape anyway. The `convergence=None` `scan`
  path *is* reverse-differentiable for a `cleanup_every=None` run if a consumer
  later needs a short differentiable refinement; the early-exit `while_loop` path
  is not (the `_converge` no-reverse caveat). Implicit-diff at the force
  equilibrium is the principled follow-up if a named consumer arrives.
- GPU posture matches the mesh tier: XLA-lowered gather/scatter + loops, no
  custom Pallas kernel.

## 8. Implementation process (phased, each independently testable)

- **GS-11a — external force.** `force='intensity'` (the Newton-on-normal
  isosurface step) and `force='gradient'` (edge-seeking), `'combined'` as their
  `SumForce`-style weighting. *Effort M.* Exit: on a synthetic volume with a
  known intensity boundary (a ball at intensity `c`), a sphere mesh placed with
  `target_intensity` between background and `c` converges onto the analytic
  boundary radius.
- **GS-11b — internal forces + the de-intersection split.** Tangential
  smoothing, curvature term, area-positivity barrier, fold-safe step; the
  `cleanup_every` host-side `remove_self_intersections` between jitted chunks.
  *Effort M.* Exit: a deliberately spiky/near-folding start evolves without a
  *persistent* self-intersection (cleanup removes transients), area stays
  positive, and `cleanup_every=None` vs an `int` is a measurable
  intersection-count difference.
- **GS-11c — assembly (white + pial) + convergence.** `place_surface` wiring,
  `boundary='white'|'pial'`, `max_thickness` cap, `register._converge`
  integration. *Effort M.* Exit: white then pial on a synthetic two-shell volume
  give non-crossing surfaces with a sane thickness map (`cortical_thickness`).
- **GS-11d — validation harness.** The §9 matrix incl. the real T1-like smoke.

Each phase commits independently (the frequent-commit practice), with the
area-positivity / non-intersection invariant asserted from GS-11b on.

## 9. Validation / test plan

Two oracle classes, mirroring the suite:

1. **Analytic / invariant.**
   - **Ball boundary**: an intensity ball (value `c` inside radius `r`, `0`
     outside, smoothed); a sphere mesh of radius `r₀ ≠ r` placed with
     `target_intensity = c/2` converges to radius `r` (mean radial error ≪ a
     voxel). The exact placement oracle.
   - **Gradient mode**: the same, with `force='gradient'` and no
     `target_intensity`, lands on the edge (max `|∇I|`) — contrast-agnostic.
   - **Non-intersection invariant** (the hard one): from a perturbed start,
     `find_self_intersections` returns empty on the result with `cleanup_every`
     set; area positivity holds throughout the jitted segments.
   - **Correspondence / genus**: faces unchanged; Euler characteristic preserved.
   - **Differentiable** (the `cleanup_every=None`, `convergence=None` path):
     `jax.grad` w.r.t. the image is finite (the short-refinement use case).
2. **Real cortical surface.** A real white surface (`tests/_real_meshes`) + a
   synthetic intensity volume built from it (interior bright, exterior dark): a
   slightly-shrunk copy placed with `force='intensity'` recovers the white
   surface (low vertex error), stays non-self-intersecting, and — placing a pial
   outward with `max_thickness` — yields a `cortical_thickness` distribution in
   the physiological band (≈1–4.5 mm) with white and pial not crossing. Class /
   distribution checks, not bit-parity (the metrics↔ITK posture); exact
   `mris_place_surface` parity on a real FreeSurfer volume is a consumer-side
   integration test, not a unit oracle.

## 10. Risk register

| Risk | Severity | Mitigation |
|---|---|---|
| **In-loop self-intersection can't use the host-side guard** (§2.5) | High | jittable surrogates in-loop (tangential smooth + curvature + area-positivity + fold-safe step) **plus** host-side `remove_self_intersections` between chunks (`cleanup_every`); the non-intersection test gates GS-11b |
| **Intensity-normalisation dependency** (`target_intensity` meaningless on a raw image) | High | state it loudly (§5.4); ship `force='gradient'` contrast-agnostic mode; `inside_brighter` makes polarity explicit |
| **Pial leaks across a sulcus** into the opposite bank | Med | `max_thickness` cap on white→pial displacement + the outward bias + the non-intersection cleanup |
| **Force balance mis-set** → over-smoothed (misses the boundary) or under-smoothed (spiky) | Med | separate `intensity_weight`/`smooth_weight`/`curvature_weight` knobs with documented defaults from the ball oracle; the synthetic boundary test calibrates them |
| **Non-convergence / cost at ico7** | Med | bounded step + `register._converge` early-exit; coarse-to-fine over the icosphere hierarchy is a lever; mark heavy real-volume tests `@perf` |
| **Host-side cleanup cost** dominates (called too often) | Low | `cleanup_every` amortises it across chunks; `None` opts out when surrogates suffice; cleanup is O(chunks), not O(iterations) |
| **It is optional and may rarely be used** → maintenance burden | Med | scope it to `register.surface` (off the default path), keep the surface minimal, and gate the build on a concrete consumer asking (this FR's premise) |
| Differentiate-through demanded | Low | documented out-of-scope v1; `scan`/`cleanup_every=None` path is reverse-diff for a short refinement; implicit-diff follow-up |

## 11. Governance & graduation

§13 four-gate: **consumer** — classic `recon-all` parity, `recon-all-clinical`
hard cases, and any no-trusted-learned-model setting (✓ named, but *optional* —
not on the default path); **composition** — this doc sketches it on the shipped
substrate (✓, almost entirely `deform_to_sdf` + `register._converge` + GS-8);
**SoC** — home **`register.surface`** (an external-image-driven optimiser with a
convergence loop — a `register` citizen, deliberately *not* a default
`geometry.surface` mover, which keeps it off the happy path); **effort** — L,
hence this **SPEC-level review** (this document) *and* the optional flag.

**Recommendation: build only on a concrete consumer request.** Phases 0–5 give
the full geometry-light + HCP pipeline without it; `place_surface` should
graduate into Phase 5 (P5.2) *when* a parity / no-model use case is on the table,
not speculatively. On approval + a consumer, record the §12-adjacent → §10.A
graduation in `IMPLEMENTATION_PLAN.md §10` at merge (GS-11a–d as sub-entries) and
backfill the as-built decision record (the chosen force model, force weights, the
`cleanup_every` cadence, and the real-volume thickness/non-intersection results)
into [`../design/geometry-suite.md`](../design/geometry-suite.md) §8, mirroring
the P1.1 / P3.2 / P5.1 records.

## 12. Cross-references & literature

- **Suite docs.** [`geometry-suite.md`](geometry-suite.md) §5 GS-11 (parent),
  [`../design/geometry-suite.md`](../design/geometry-suite.md) §8 P5.2,
  [`spherical-parameterisation.md`](spherical-parameterisation.md) (the sibling
  Effort-L SPEC-review FR this mirrors), GS-10 `deform_to_sdf` (the skeleton),
  GS-8 self-intersection (the cleanup), GS-9 `cortical_thickness` (the pial
  validator).
- **Substrate.** `geometry/surface.py` (`deform_to_sdf`, `mean_curvature`,
  `cortical_thickness`), `geometry/grid.py` (`sample_at_points`),
  `geometry/differential.py` (`spatial_gradient`), `sparse/mesh.py`
  (`compute_vertex_normals`, `mesh_laplacian_smooth`, `face_areas`),
  `geometry/intersection.py` (`remove_self_intersections`),
  `register/_converge.py` (`run_iterations`, `Convergence`), `register/_force.py`
  (the `Force`/`SumForce` design analogue), `tests/_real_meshes.py`.
- **Governance.** `SPEC_UPDATE_v0.3.md §13` (acceptance, §13.4 L-review),
  `SPEC.md §5` (dep contract), `SPEC_UPDATE_v0.3.md §14` (kwarg-not-fork — the
  `boundary`/`force` modes are kwargs on one entry point).
- **Method lineage.** Dale, Fischl, Sereno 1999 (cortical surface
  reconstruction — the deformable-surface white/pial fit); Fischl, Sereno, Dale
  1999 (the surface-based stream); Fischl et al. 2002 (atlas-constrained
  surface deformation); FreeSurfer `mris_place_surface` (the modern
  `mris_make_surfaces` replacement; the target-intensity + gradient force model
  of record). Active-contour / deformable-model roots: Kass, Witkin,
  Terzopoulos 1988 (snakes); McInerney & Terzopoulos 1996 (deformable models in
  medical image analysis). MacDonald et al. 2000 (ASP / CLASP — coupled
  white/pial with an explicit non-self-intersection constraint, the design
  precedent for §5.2). The learned alternatives that make GS-11 optional:
  Hoopes et al. 2022 (TopoFit); Ma et al. 2021/2022 (CortexODE); Cruz et al.
  2021 (DeepCSR); Henschel et al. 2020 (FastSurfer / `fastcsr`).
