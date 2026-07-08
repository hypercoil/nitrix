# Reducing unnecessary eagerness in the geometry suite — JIT-boundary hygiene, execution-class contract & a device-construction (cupy) cost/benefit

> **Status (2026-07-08): PROPOSED — triage + design only, no code.** A suite-wide
> pass over every `nitrix.geometry` / `nitrix.graph` op that executes host-side
> (runtime `numpy`, eager, or degrades under `jax.jit`), folding two overlapping
> concerns into one sprint:
> 1. **Host-side construction audit** — which host paths are *irreducible* vs a
>    missed device opportunity, and an honest per-op *execution-class* docstring
>    contract (W1–W2).
> 2. **JIT-boundary hygiene** — "mixed-mode" ops that fuse a host construction
>    step with an otherwise-jittable data-path, forcing a consumer to flag the
>    *whole* op eager and split its JIT region (running compilable code at eager
>    speed). Fix: lift the construction out behind a plan-passthrough / fit-apply
>    seam (W3).
>
> The device-construction (cupy) dimension (W4) is **SPEC-gated** — it amends the
> §5/§6.1/§6.2 dependency contract — and must clear a perf-bench-demonstrated
> *multiplicative* win before any code. Provenance: user triage directives
> 2026-07-08. Related: [`mesh-spatial-acceleration.md`](mesh-spatial-acceleration.md)
> (C5), [`alternative-interp-backends-xla.md`](alternative-interp-backends-xla.md)
> (B16), [`morphology-gpu-scaling.md`](morphology-gpu-scaling.md),
> [`geometry-suite-audit.md`](geometry-suite-audit.md) (MIN-06, M-05, A1).

## Headline finding

The motivating framing — *"host-side construction ⟹ the XLA path is
infeasible"* — is **true for only one of four categories**. The geometry/graph
ops that leave the device sort cleanly, and each category has a *different*
right lever:

| Category | What it means | XLA genuinely infeasible? | Right lever |
|---|---|---|---|
| **A. host-only (dynamic output shape)** | output *shape* is a function of input *values* (crossing count, intersecting-pair count, resampling nnz) | **Yes** — pad-to-worst-case intractable | keep host; **cupy** is the only device option (W4). *If it already returns an operator (`surface_resample`), the apply-seam is separate & jittable.* |
| **B. host-by-implementation** | numpy by choice; output shape is **static** | **No** — a `jnp` port is jittable + differentiable + GPU-native | **`jnp` port** (zero dep; strictly beats cupy here) — W2 |
| **C. host-built plan, device data-path** | build-once operator/plan from **concrete connectivity or a static band-limit**; apply-seam is pure JAX | **No** — this is the *"static host math"* SPEC §6.1 sanctions | keep the build; **expose the plan** so the data-path is jittable end-to-end (W3) |
| **D. eager convenience** | an `'auto'`/`.max()+1` read unavailable under `jit` | n/a — pass the explicit static value | already documented; nothing to do |

SPEC anchor: **§6.1 authorises `numpy` for "type aliases / static host math
only."** C and D are that clause working as intended; B is a missed `jnp`
opportunity; only A is an irreducible host obstruction — and the only place an
optional device-construction dependency could earn its keep.

**The JIT-boundary problem is a property of Category C specifically**, and it
splits into two sub-cases that need different fixes:
- **C-mesh** — the plan derives from an *array-valued* mesh (connectivity /
  vertex metric). `static_argnames` cannot rescue this (arrays aren't static
  args), so a consumer genuinely cannot `jit` through the op → the region splits.
  **This is the real mixed-mode problem.** Fix = plan passthrough (W3).
- **C-static** — the plan derives from a *static scalar* (SHT `band_limit`). The
  op is *already* jittable via `static_argnames` (the plan folds into the trace
  as a constant), but that rebuilds the `O(L²)` Legendre table on the host at
  **every trace**. The same fit-state passthrough applies (the plan factors out
  cleanly), so this takes the **identical (b) seam** as C-mesh — it is simply
  lower-urgency (uniformity + rebuild-elimination, not newly enabling jit).

## Full triage

Line numbers as-of-audit (2026-07-08). "Diff?" = differentiable through the op
w.r.t. its *data* argument on the device path (never w.r.t. a concrete mesh's
vertices when the metric is host-baked). "Seam" = does it already return/accept
the built operator (the good pattern) or fuse build+apply (mixed-mode)?

### geometry — field ↔ mesh / distance

| Op | Cat | Host-side mechanism (trace-unknowable) | Diff? | Seam | Docstring |
|---|---|---|---|---|---|
| `marching_cubes` | **A** | # iso-crossing triangles = f(volume vs level) | no | host-only | **adequate** |
| `mesh_to_sdf` | **B** | *none* — output shape is the static `shape` arg | no | — | partial |
| `nearest_surface_distance` (grid) | **A** | ragged triangle→cell bin length | no | host-only | partial |
| `nearest_surface_distance` (brute) | **B** | *none* — static dense `O(n·F)` | no | — | partial |
| `cortical_thickness` (`symmetric`) | **A** | inherits nearest-triangle search | no | (delegates) | **adequate** |
| `surface_smooth` | **C-mesh** | cotangent ELL `k_max`; **build+CG fused in one call** | **yes** | **mixed-mode** ← W3 | partial |
| `find_self_intersections` | **A** | # intersecting pairs = f(coords), `O(F²)` | no | host-only | **adequate** |
| `remove_self_intersections` | **A** | inherits pair count; per-iter host↔device bounce | no | host-only | **adequate** |

### geometry — sphere / harmonics

| Op | Cat | Host-side mechanism | Diff? | Seam | Docstring |
|---|---|---|---|---|---|
| `spectral_sphere_embedding` | **C-mesh** | cotangent operator (connectivity sparsity + vertex weights) | no (init) | mixed-mode (low pri) | **adequate** |
| `spherical_parameterize` | **C-mesh** | same cotangent plan; refine loop device-jittable | no (init) | mixed-mode (low pri) | **excellent** |
| `surface_resample` | **A** | resampling-ELL nnz & `k_max` = f(geometry) | **yes** (apply) | **returns ELL (good)** | good |
| `parcel_centroids` | **D** | `n_parcels = int(parcellation.max())+1` when `None` | yes | (scalar) | **adequate** |
| `spherical_conv` | — | *not host-side*; `O(n²)` device compute only | yes | — | adequate |
| `sht_forward` / `sht_inverse` | **C-static** | Legendre plan from **static band-limit only** | yes | already jittable; uncached | **adequate** |
| `real_sht_forward` / `real_sht_inverse` | **C-static** | inherit static plan | yes | as above | adequate |
| `sht_rotation_matrix` / `sht_rotate` | **C-static** | Wigner-y generators from static degree | **yes** (in `R`, `coeffs`) | as above | good |
| `sht_grid` | **C-static** | nodes from static band-limit | n/a | **is** the plan handle | marginal |

### geometry — DEC / topology / intersection

| Op | Cat | Host-side mechanism | Diff? | Seam | Docstring |
|---|---|---|---|---|---|
| `mesh_gradient` / `mesh_curl` | **C-mesh** | # unique edges from concrete `faces`; `±1` incidence | n/a (connectivity) | **returns ELL (good)** | **adequate** |
| `mesh_star_k` | **C-mesh** | k=1: `E` + cotangent (host). **k=0/k=2 pure `jnp`** | **k=0/k=2 yes** | **returns ELL (good)** | **inadequate** — W2 |
| `mesh_divergence` | **C-mesh** | `E` + `k_max` from connectivity | no | **returns ELL (good)** | adequate |
| `hodge_decompose` | **C-mesh** | edge topology + host `star1`; **build+CG fused** | **yes** (in `omega`) | **mixed-mode** ← W3 | **adequate** |
| `euler_characteristic` / `genus` | **D** | returns Python `int`; unique-edge count | n/a (invariant) | — | **adequate** |
| `spatial_gradient` | — | *not host-side*; pure `jnp` | yes | — | adequate |

### graph — parcellation / components + grid

| Op | Cat | Host-side mechanism | Diff? | Seam | Docstring |
|---|---|---|---|---|---|
| `surface_boundary_map` | — | *not host-side*; ELL edge-aggregate | yes | — | adequate |
| `connected_components` | — | *not host-side*; device label-propagation (exemplar) | n/a | — | **adequate** |
| `mesh_watershed` | **A** | priority-flood pop order + basin count; scalar-Python | no | host-only | adequate (silent on jit) |
| `integrate_velocity_field` (`n_steps='auto'`) | **D** | `n_steps = f(max|v|)` | yes | (scalar) | **adequate** (×3) |

## W1 — execution-class docstring contract (ask #1)

One greppable tag line per host-touching op, **honest per category** (not a
uniform "XLA infeasible" that is false for B/C/D):

- **A** → `Execution class: host-only. Output shape is data-dependent (<the
  quantity>); no static-shape XLA lowering, pad-to-worst-case intractable.`
- **B** → `Execution class: host-by-implementation. Output shape is static; a
  device (jnp) port is feasible (this FR, W2). Not currently jittable.`
- **C** → `Execution class: host-built plan + device data-path. The <operator>
  is built once from concrete <connectivity | band-limit> (static host math,
  SPEC §6.1); the data-path is pure JAX, differentiable w.r.t. <arg>. Supply
  <plan-arg> to run the whole call under jit (W3).`
- **D** → `Execution class: eager convenience. Pass <arg> explicitly for jit.`

Add/correct only where the triage marks *partial/inadequate/marginal*
(`mesh_to_sdf`, `nearest_surface_distance`, `surface_smooth`, `sht_grid`,
`mesh_star_k`, `mesh_watershed`). The `adequate`/`excellent` docstrings
(`spherical_parameterize`, `marching_cubes`, `hodge_decompose`, the SHT module
header) are the template — no churn. **Pure docs; proceed on approval un-gated.**

## W2 — two latent findings beyond docs

1. **`mesh_star_k` — a wrong differentiability claim + dead eager work.** The DEC
   module docstring's blanket *"not differentiable w.r.t. vertex coordinates"* is
   **false for k=0 (`⋆₀`) and k=2 (`⋆₂`)** (pure `jnp`, differentiable). And
   `mesh_star_k` unconditionally runs `np.asarray(mesh.vertices/faces)` (≈`dec.py:208`)
   even for k=0/k=2, doing dead host work and defeating jit on the two
   device-clean branches. Fix = correct the split + hoist the host conversion
   into the k=1 branch only. Small, correctness-adjacent.

2. **`mesh_to_sdf` and brute `nearest_surface_distance` are Category B, not A.**
   No shape barrier — a `jnp` port (with `lax.map` over voxel/query chunks) is
   jittable, GPU-native, static-shaped, **zero new dependency**. Strictly better
   than cupy for these two. First perf move if their host cost is measured to
   matter (delegated to `nitrix-perf-bench`); filed here, not committed pending a
   bench signal.

## W3 — plan-passthrough seams: un-splitting the JIT region (ask: mixed-mode ops)

**The problem.** A "mixed-mode" op fuses a host construction (from an
array-valued mesh) with an otherwise-jittable data-path in a *single* call that
returns the result. A consumer library drawing JIT boundaries must therefore
flag the entire op eager — splitting the region and running the compilable
data-path (a CG solve, an FFT, a contraction) at eager speed. The offenders are
the **C-mesh** ops that return a *result* rather than an *operator*:

| Mixed-mode op | Host-built plan to lift out | Device data-path protected |
|---|---|---|
| **`surface_smooth`** | cotangent stiffness ELL + lumped-mass diagonal | matrix-free `cg` solve (the hot part) |
| **`hodge_decompose`** | edge topology + `⋆₁` gather/scatter closures | two Poisson `cg` solves |
| `spherical_parameterize` | cotangent operator | fold-safe refine loop (low pri — init/preproc) |
| `spectral_sphere_embedding` | cotangent operator | LOBPCG eigensolve (low pri — init/preproc) |

**The pattern already exists in two forms nitrix has committed to:**
- *In stats (the escape hatch the concern references):* `stats.gaulss` takes
  `scale_design=None`; `stats.basis` lifts host-built operators out as
  `Optional[Array] = None` passthroughs (`knots`, `radial_transform`,
  `constraint`) — supply the precomputed array and the body is pure-`jnp`.
- *In geometry, done even more cleanly:* the DEC constructors
  (`mesh_gradient(mesh) -> ELL`, `mesh_curl`, `mesh_star_k`, `mesh_divergence`)
  and `surface_resample` are **pure constructors that return the operator**. The
  consumer builds once (host) and applies inside their *own* JIT region. **These
  have no mixed-mode problem and are the target pattern** — the mixed-mode ops
  should look like them.

**Two approaches; (b) is the default — lean to it wherever ambiguous** (it gives
consumers a *stronger, unconditional* jit guarantee and cannot be mis-called):

- **(b) Explicit constructor + apply pair** (the DEC model; SPEC §6.5 fit/apply,
  already **normative** in nitrix) — `surface_smooth_operator(mesh) -> (K, M)`
  plus `surface_smooth(values, K, M, ...)`, with the single-call convenience
  *defined as* `apply(values, *build(mesh))` so the paths can't drift. *Pro:* the
  apply half is **unconditionally** jittable — a clean, discoverable contract
  matching DEC/`surface_resample`. *Con:* a second public symbol per op. **The
  default.**
- **(a) Optional passthrough arg** — `surface_smooth(mesh, values, *,
  operator=None)`; supplying `operator` makes the body pure-`jnp`/jittable. Falls
  back to this **only** where a second public symbol is genuinely unwarranted
  (a rarely-jitted op with no natural operator name). *Con:* "conditionally
  jittable" is a subtle contract — a naive `jit(f)` still fails unless the caller
  knows to pass `operator`; that weaker guarantee is why (b) is preferred.

**The SHT family takes (b), not a special case.** Its `_plan` factors cleanly
into fit-state: promote the private `_plan` (nodes, weights, `(L+1,2L+1,n_lat)`
Legendre table, `m_to_fft`) to a public `SHTPlan` pytree (a superset of the
existing `SHTGrid`), expose `sht_plan(band_limit, *, grid) -> SHTPlan` as the
fit, and take `sht_forward(f, plan)` / `sht_inverse(coeffs, plan)` as the apply
(`band_limit` is implicit in the plan's shapes — `sht_inverse` already infers
it). This dissolves the uncached-rebuild wart (the consumer holds the plan)
without any caching machinery, and unifies SHT with the rest of the suite.
`sht_rotation_matrix` / `sht_rotate` generalise identically (the Wigner
generators are the plan). SHT is *already* jittable via `static_argnames`, so it
is lowest-urgency within W3 — but it takes the same seam.

**Per-op priority:** `surface_smooth` and `hodge_decompose` first (they are *not*
jittable at all today — highest consumer value); the SHT family next (uniformity
+ rebuild-elimination); `spherical_parameterize` / `spectral_sphere_embedding`
last — init/preprocessing consumers rarely jit, do only if asked. All **zero-dep**
and in the same sprint as W1/W2.

## W4 — device-side dynamic construction (cupy): ask #2

**cupy is a SPEC amendment, not an engineering nod.** Runtime imports are
contractually `jax` / `jaxtyping` / `numpy` only (SPEC §5, §6.1, §6.2). There is
**no cupy in the tree today** (every `cupy` string in `src/` is a benchmark
reference in a comment). Admitting it — even as an optional extra — is the same
class of decision as the `transport` subpackage: it needs an explicit SPEC
argument.

**cupy's niche is narrow.** Where shapes are static, a `jnp` port strictly wins
(jittable, differentiable, no dep). cupy's *only* unique capability is **eager,
dynamic-shape execution on the GPU** — the Category-A ops, where you cannot have
static shapes yet want to avoid the device→host→device round-trip. Candidate set:

| Candidate (Cat A, vectorised numpy) | cupy win | Caveat |
|---|---|---|
| `marching_cubes` | 256³ volume stays on-GPU; `unique`/mask reductions on-GPU | `cupy.unique` coverage; still non-diff |
| `surface_resample` (ELL build) | ico7 verts on-GPU; barycentric + ELL pack on-GPU | bucket broad-phase is scalar-Python → not portable (overlaps C5) |
| `nearest_surface_distance` (grid) | on-GPU binning + shell search | overlaps C5 exactness gate |
| `find_self_intersections` (broad) | on-GPU AABB/cell membership | within-cell pairing loop scalar-Python → not portable |

**Where cupy does *not* help:** scalar-Python cores (`mesh_watershed` heapq +
union-find, self-intersection within-cell loops, `genus`) — cupy is an array API,
it cannot accelerate Python control flow; and every B/C op, where a `jnp` port
(B) or the amortised host plan (C) already dominates.

**Prior art owns pieces of this** and this FR defers to it: B16
(`alternative-interp-backends-xla`) is the parked dlpack/zero-copy cupyx-in-XLA
research note; C5 (`mesh-spatial-acceleration`) is the exactness-gated broad-phase
for the distance/resample ops; `morphology-gpu-scaling` benched JAX vs `cupyx`
label propagation. The novel slice here is the *geometry-construction* framing +
the taxonomy that says **which** ops could even use cupy.

## Recommendation (sequencing)

1. **One zero-dep sprint (un-gated on approval):** W1 (docstring contract) + W2.1
   (`mesh_star_k` fix) + W3 ((b)-style constructor/apply seams: `surface_smooth`
   and `hodge_decompose` first, then the SHT family via a public `SHTPlan`
   fit-state). All pure-JAX / docs, no SPEC impact; W3 is the highest-value
   consumer-facing win (un-splits real JIT regions).
2. **Still zero-dep, on a bench signal:** W2.2 — `jnp`-port `mesh_to_sdf` and the
   brute distance path.
3. **SPEC-gated + bench-gated:** a cupy path admitted *only* as a **profile-gated
   optional backend behind the existing `backend=` axis** (auto-off; loud
   `NitrixBackendFallback` to numpy when absent, per SPEC §7), scoped to the
   Category-A vectorised ops, and **only if** `nitrix-perf-bench` shows a
   *multiplicative* win at ico7 / 256³ that a `jnp` port cannot reach. Rides on
   the B16 dlpack verdict. If the win isn't multiplicative, cupy stays out.

## Non-goals

- No change to the numerics of any op (host and device paths stay bit-faithful
  within `tolerance.toml`); the single-call convenience must equal
  `apply(x, build(mesh))` so the two paths cannot drift.
- No cupy as a **hard** dependency, ever — at most an optional extra behind a
  loud fallback, and only after the SPEC argument lands.
- Making the *metric fills* differentiable w.r.t. vertices (host-baked
  cotangents) is a separate concern (M-05 / A1 in the geometry audit), not this
  FR. W3 makes the data-path jittable; it does not add vertex gradients.

## Cross-references

- [`geometry-suite-audit.md`](geometry-suite-audit.md) — MIN-06 (host `O(n·F)`
  cliff), M-05 / A1 (`spherical_parameterize` jittability), the systemic
  "host-side construction" theme this FR formalises.
- [`mesh-spatial-acceleration.md`](mesh-spatial-acceleration.md) — C5, the
  exactness-gated broad-phase for `nearest_surface_distance` / `surface_resample`.
- [`alternative-interp-backends-xla.md`](alternative-interp-backends-xla.md) —
  B16, the cupyx-in-XLA / dlpack zero-copy feasibility note W4 defers to.
- [`morphology-gpu-scaling.md`](morphology-gpu-scaling.md) — JAX-vs-`cupyx`
  scaling precedent.
- SPEC.md §5, §6.1–6.2 (dependency contract), §6.5 (fit/apply seam W3 follows),
  §7 (loud fallbacks), §3 (the `backend=` axis a cupy path would ride on).
- Precedent for the W3 passthrough: `stats.gaulss` (`scale_design=`),
  `stats.basis` (`knots` / `radial_transform` / `constraint` optional args), and
  the DEC constructors / `surface_resample` (return-the-operator seam).
