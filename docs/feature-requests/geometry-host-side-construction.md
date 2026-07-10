# Reducing unnecessary eagerness in the geometry suite — JIT-boundary hygiene, execution-class contract & a device-construction (cupy) cost/benefit

> **Status (2026-07-10): PARTIALLY IMPLEMENTED — the zero-dep sprint shipped;
> the dep-gated / bench-gated remainder is deferred by design.** A suite-wide
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
> **Sprint 1 (zero-dep) is DONE** on branch `feat/geometry-eagerness-reduction`
> (5 commits, not merged/pushed): W1 execution-class docstrings, W2.1 `mesh_star_k`
> fix, and all three W3 (b)-style seams (`surface_smooth`, `hodge_decompose`, the
> SHT family via a public `SHTPlan`). The device-construction (cupy) dimension
> (W4) remains **SPEC-gated** — it amends the §5/§6.1/§6.2 dependency contract —
> and must clear a perf-bench-demonstrated *multiplicative* win before any code;
> W2.2 (`jnp`-port `mesh_to_sdf` / brute distance) is bench-signal-gated; the
> two low-priority sphere-param seams are "only if a consumer asks". Provenance:
> user triage directives 2026-07-08, sprint 2026-07-10. Related:
> [`mesh-spatial-acceleration.md`](mesh-spatial-acceleration.md)
> (C5), [`alternative-interp-backends-xla.md`](alternative-interp-backends-xla.md)
> (B16), [`morphology-gpu-scaling.md`](morphology-gpu-scaling.md),
> [`geometry-suite-audit.md`](geometry-suite-audit.md) (MIN-06, M-05, A1).

## Implementation status (2026-07-10)

Branch `feat/geometry-eagerness-reduction`, off `main` after the FR commit
`cd5d2d7`. Not merged, not pushed. Every step verified per-file green (ruff +
mypy-strict on the geometry sources; final combined regression 56 passed across
`test_dec` / `test_surface_smoothing` / `test_harmonics` /
`test_op_matrix_completeness`).

| Work item | State | Commit | Notes |
|---|---|---|---|
| **W1** execution-class docstrings | ✅ shipped | `3b08105` | `mesh_to_sdf` (B), `nearest_surface_distance` grid+brute (A/B), `mesh_watershed` (A), plus the C/D tags on the seam ops |
| **W2.1** `mesh_star_k` k=0/k=2 pure-JAX | ✅ shipped | `b94a8c1` | host `np.asarray` hoisted into the k=1 branch; module + method docstrings corrected (k=0/k=2 jittable + differentiable) |
| **W3** `surface_smooth` seam | ✅ shipped | `2ac7acd` | `surface_smooth_operator` + `surface_smooth_apply`; convenience *defined as* apply∘operator (byte-identical) |
| **W3** `hodge_decompose` seam | ✅ shipped | `f812156` | `hodge_operator` + `hodge_apply`; static vertex count rides in `gradient.n_cols` (no custom pytree) |
| **W3** SHT family seam | ✅ shipped | `6058316` | private `_plan` → public `SHTPlan` + `sht_plan`; `sht_forward(f, plan)` / `sht_inverse(coeffs, plan)`; `n_lon = plan.longitude.shape[0]` |
| **W2.2** `jnp`-port `mesh_to_sdf` / brute distance | ⏸ deferred | — | bench-signal-gated (delegated to `nitrix-perf-bench`) |
| **W3** `spherical_parameterize` / `spectral_sphere_embedding` seams | ⏸ deferred | — | init/preproc; "only if a consumer asks" |
| **W4** cupy optional backend | ⏸ SPEC-gated | — | needs the SPEC argument *and* a multiplicative perf-bench win |

Op-matrix bookkeeping: the three new *apply* ops enter `CATALOGUE`
(`surface_smooth_apply`, `hodge_apply`, the plan-threaded SHT overrides); the
three *operator constructors* (`surface_smooth_operator`, `hodge_operator`,
`sht_plan`) join the EXCLUDE allowlist alongside the existing `mesh_gradient`
family, per the return-the-operator convention. No op-matrix regen was run
(the completeness guard reads `CATALOGUE` at import).

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
| `mesh_to_sdf` | **B** | *none* — output shape is the static `shape` arg | no | — | ✅ W1 |
| `nearest_surface_distance` (grid) | **A** | ragged triangle→cell bin length | no | host-only | ✅ W1 |
| `nearest_surface_distance` (brute) | **B** | *none* — static dense `O(n·F)` | no | — | ✅ W1 |
| `cortical_thickness` (`symmetric`) | **A** | inherits nearest-triangle search | no | (delegates) | **adequate** |
| `surface_smooth` | **C-mesh** | cotangent ELL `k_max`; build lifted to `surface_smooth_operator` | **yes** | ✅ **seam (W3)** | ✅ W1 |
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
| `sht_forward` / `sht_inverse` | **C-static** | Legendre plan from **static band-limit only** | yes | ✅ **`SHTPlan` seam (W3)** | ✅ adequate |
| `real_sht_forward` / `real_sht_inverse` | **C-static** | inherit static plan | yes | ✅ **`SHTPlan` seam (W3)** | adequate |
| `sht_rotation_matrix` / `sht_rotate` | **C-static** | Wigner-y generators from static degree | **yes** (in `R`, `coeffs`) | static-degree (not migrated; already jittable) | good |
| `sht_grid` | **C-static** | nodes from static band-limit | n/a | **is** the plan handle | ✅ W1 (→ `sht_plan`) |

### geometry — DEC / topology / intersection

| Op | Cat | Host-side mechanism | Diff? | Seam | Docstring |
|---|---|---|---|---|---|
| `mesh_gradient` / `mesh_curl` | **C-mesh** | # unique edges from concrete `faces`; `±1` incidence | n/a (connectivity) | **returns ELL (good)** | **adequate** |
| `mesh_star_k` | **C-mesh** | k=1: `E` + cotangent (host). **k=0/k=2 pure `jnp`** | **k=0/k=2 yes** | **returns ELL (good)** | ✅ W2.1 + W1 |
| `mesh_divergence` | **C-mesh** | `E` + `k_max` from connectivity | no | **returns ELL (good)** | adequate |
| `hodge_decompose` | **C-mesh** | edge topology + host `star1`; build lifted to `hodge_operator` | **yes** (in `omega`) | ✅ **seam (W3)** | **adequate** |
| `euler_characteristic` / `genus` | **D** | returns Python `int`; unique-edge count | n/a (invariant) | — | **adequate** |
| `spatial_gradient` | — | *not host-side*; pure `jnp` | yes | — | adequate |

### graph — parcellation / components + grid

| Op | Cat | Host-side mechanism | Diff? | Seam | Docstring |
|---|---|---|---|---|---|
| `surface_boundary_map` | — | *not host-side*; ELL edge-aggregate | yes | — | adequate |
| `connected_components` | — | *not host-side*; device label-propagation (exemplar) | n/a | — | **adequate** |
| `mesh_watershed` | **A** | priority-flood pop order + basin count; scalar-Python | no | host-only | ✅ W1 (now explicit) |
| `integrate_velocity_field` (`n_steps='auto'`) | **D** | `n_steps = f(max|v|)` | yes | (scalar) | **adequate** (×3) |

## W1 — execution-class docstring contract (ask #1) — ✅ SHIPPED (`3b08105`)

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

Added/corrected only where the triage marked *partial/inadequate/marginal*
(`mesh_to_sdf`, `nearest_surface_distance`, `surface_smooth`, `sht_grid`,
`mesh_star_k`, `mesh_watershed`). The `adequate`/`excellent` docstrings
(`spherical_parameterize`, `marching_cubes`, `hodge_decompose`, the SHT module
header) were the template — no churn.

## W2 — two latent findings beyond docs

1. **`mesh_star_k` — a wrong differentiability claim + dead eager work. ✅ FIXED
   (`b94a8c1`).** The DEC module docstring's blanket *"not differentiable w.r.t.
   vertex coordinates"* was **false for k=0 (`⋆₀`) and k=2 (`⋆₂`)** (pure `jnp`,
   differentiable). And `mesh_star_k` unconditionally ran
   `np.asarray(mesh.vertices/faces)` even for k=0/k=2, doing dead host work and
   defeating jit on the two device-clean branches. Fixed: the host conversion is
   hoisted into the k=1 branch only; k=0/k=2 are now jittable through the mesh
   and differentiable w.r.t. the vertices (pinned by
   `test_star_k_metric_stars_are_jittable_and_differentiable`), with the k=1
   host-only path guarded by `test_star1_is_host_only`; module + method
   docstrings corrected.

2. **`mesh_to_sdf` and brute `nearest_surface_distance` are Category B, not A.
   ⏸ DEFERRED (bench-signal-gated).**
   No shape barrier — a `jnp` port (with `lax.map` over voxel/query chunks) is
   jittable, GPU-native, static-shaped, **zero new dependency**. Strictly better
   than cupy for these two. First perf move if their host cost is measured to
   matter (delegated to `nitrix-perf-bench`); filed here, not committed pending a
   bench signal.

## W3 — plan-passthrough seams: un-splitting the JIT region (ask: mixed-mode ops) — ✅ SHIPPED (`2ac7acd`, `f812156`, `6058316`)

**Shipped seams (all (b)-style, convenience *defined as* apply∘operator):**
`surface_smooth_operator` + `surface_smooth_apply`; `hodge_operator` +
`hodge_apply` (returning the `HodgeOperator` NamedTuple `(gradient, curl, star1)`,
static vertex count in `gradient.n_cols`); and the SHT transform family on a
public `SHTPlan` + `sht_plan` fit (`sht_forward(f, plan)` / `sht_inverse(coeffs,
plan)` / the real variants). Each apply half is **unconditionally jittable** with
the operator held as a plain pytree arg — no custom pytree needed. **Remaining
(deferred):** the `spherical_parameterize` / `spectral_sphere_embedding` seams
(low-pri init/preproc) and the `sht_rotation_matrix` / `sht_rotate` migration
(already jittable via `static_argnames`; not consumer-blocking).

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

**The SHT family takes (b), not a special case.** *(Shipped `6058316` — this is
the design record.)* Its `_plan` factors cleanly into fit-state: the private
`_plan` (nodes, weights, `(L+1,2L+1,n_lat)` Legendre table, `m_to_fft`) was
promoted to a public `SHTPlan` pytree, `sht_plan(band_limit, *, grid) -> SHTPlan`
is the fit, and `sht_forward(f, plan)` / `sht_inverse(coeffs, plan)` are the apply
(`band_limit` is implicit in the plan's shapes — `sht_inverse` already infers
it). This dissolves the uncached-rebuild wart (the consumer holds the plan)
without any caching machinery, and unifies SHT with the rest of the suite.
`sht_rotation_matrix` / `sht_rotate` generalise identically (the Wigner
generators are the plan). SHT is *already* jittable via `static_argnames`, so it
is lowest-urgency within W3 — but it takes the same seam.

**Per-op priority (as executed):** `surface_smooth` and `hodge_decompose` first
(they were *not* jittable at all before — highest consumer value); the SHT
family next (uniformity + rebuild-elimination) — **all three shipped**;
`spherical_parameterize` / `spectral_sphere_embedding` last — init/preprocessing
consumers rarely jit, deferred until a consumer asks. All **zero-dep** and in the
same sprint as W1/W2.

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

1. **One zero-dep sprint (un-gated on approval): ✅ DONE
   (`feat/geometry-eagerness-reduction`, not merged/pushed).** W1 (docstring
   contract) + W2.1 (`mesh_star_k` fix) + W3 ((b)-style constructor/apply seams:
   `surface_smooth` and `hodge_decompose` first, then the SHT family via a public
   `SHTPlan` fit-state). All pure-JAX / docs, no SPEC impact; W3 was the
   highest-value consumer-facing win (un-splits real JIT regions).
2. **Still zero-dep, on a bench signal: ⏸ deferred.** W2.2 — `jnp`-port
   `mesh_to_sdf` and the brute distance path.
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
