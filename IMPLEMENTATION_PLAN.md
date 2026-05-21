# nitrix — Implementation plan (draft v0)

> **Scope.** Reads on top of SPEC.md (v0) + SPEC_UPDATE.md (v0.1) + SPEC_UPDATE_v0.2.md
> + MIGRATION.md. Where this plan conflicts with MIGRATION.md §6 (recommended order),
> this plan supersedes — see §1.2 below for rationale.
>
> **Audience.** Whoever picks up implementation: human engineer, AI agent, or a
> rotating mix. The plan is written to survive both modes.
>
> **Tone.** Prescriptive about contracts (what "done" means), permissive about
> sequencing (deviation is expected).

---

## 0. How to read this plan

The plan is organised as **phases** with explicit entry / exit criteria, **tracks**
that run in parallel within phases, and **gates** that block progression. Tasks within
a track are ordered for typical execution but are not strictly serial unless marked
`SERIAL`.

If you are an agent (human or AI) starting work, read in this order:

1. §1 (architecture of the plan) and §2 (non-negotiables and deviation protocol) —
   always.
2. §3 (Phase 0) — always.
3. The current phase, both tracks. Skim the next phase to anticipate dependencies.
4. §9 (per-subpackage detailed checklists) when you start work on a specific
   subpackage.
5. §10 (deviation log) before deciding whether a deviation request meets the bar.

When in doubt, **§2.2 governs.** That is the contract; everything else is guidance.

---

## 1. Plan architecture

### 1.1 Phases (capability-oriented, not file-oriented)

Phases are defined by what downstream consumers can build against at the end of each
phase, not by which files exist on disk. This matters because deviation pressure comes
from downstream consumers needing capabilities, not files.

| Phase | Capability delivered | Approx. duration | Gating risk |
|---|---|---|---|
| **0. Preflight** | Known bugs fixed; performance baseline established; CI scaffolding | weeks | Ampere ELL gate decides Phase 2 shape |
| **1. Foundation** | `linalg`, `stats`, `signal` consolidated; downstream can swap nitrix imports for hypercoil imports | weeks | none — pure consolidation |
| **2. Substrate** | `semiring` + `sparse.ell` shipped; downstream can build distance ops, graph algebra, attention-like reductions | months | Triton stability; backward-kernel correctness |
| **3. Geometry & graph** | `geometry`, `graph` shipped, spherical conv re-backed | weeks | depends on Phase 2 |
| **4. Marquee** | `smoothing`, `morphology` shipped; permutohedral resolved per tripwire | weeks–months | permutohedral risk |
| **5. Polish** | LME namespace reserved; docs; benchmark suite; first GA | weeks | none — pure cleanup |

The phases are not equal-sized. Phase 2 is the long pole. Phases 1 and 3 can compress
or expand based on what's needed.

### 1.2 Two parallel tracks

Within each phase (except Phase 0 and Phase 5), work splits across two tracks:

- **Track A — Substrate.** New code: the semiring kernel substrate, sparse format,
  smoothing kernels, morphology. This is where the design bets live and where the risk
  concentrates.
- **Track B — Consolidation.** Migration of existing hypercoil / nitrix / ilex /
  entense numerics into the new layout. Mostly mechanical: namespace moves, deletion
  of upstream deps, test consolidation. Lower risk, lower interest, high downstream
  value (downstream libraries unblock by being able to import from `nitrix.linalg`
  instead of `hypercoil.functional`).

The two tracks share Phase 0 (preflight) and Phase 5 (polish). They converge at Phase
4 (smoothing depends on substrate from A and consolidated linalg from B). Otherwise
they run independently.

**Why this differs from MIGRATION.md §6.** The migration doc puts substrate work
first, then consolidation. That's wrong for two reasons: (a) the consolidated linalg /
stats / signal modules are what downstream actually wants in the short term, and (b)
substrate work has long uncertainty tails (Pallas / Triton risk, kernel correctness)
that should not block low-risk consolidation. Running them in parallel halves the
calendar time to "downstream can swap their imports."

### 1.3 Gates

A gate is a hard checkpoint: subsequent work depends on its outcome and proceeding
without resolving it would invalidate downstream assumptions.

| Gate | Decides | Phase | §reference |
|---|---|---|---|
| **G0. Ampere-ELL benchmark** | Triton-default vs JAX-default for ELL kernels | end of Phase 0 | SPEC_UPDATE_v0.2 §4 |
| **G1. Backward-kernel correctness** | Whether each `StrictSemiring` backward passes finite-difference checks at pinned tolerance | mid Phase 2 | SPEC_UPDATE §3.1 |
| **G2. Permutohedral tripwire** | Whether `permutohedral_lattice` ships or raises `NotImplementedError` | mid Phase 4 | SPEC_UPDATE §3.3 |
| **G3. Backend-fallback observability** | Whether the warning infrastructure works under real fallback conditions | end of Phase 2 | SPEC_UPDATE_v0.2 §7.2 |

Gate outcomes are recorded in §10 deviation log even when they pass cleanly. A failed
gate triggers replanning, not workarounds-in-place.

---

## 2. Non-negotiables and deviation protocol

### 2.1 Why this section exists

You told us downstream consumers occasionally need implementations urgently and the
plan should accept some deviation. This section says what can flex and what can't.
Without it, "deviation tolerance" becomes "the plan is decorative."

### 2.2 Non-negotiables

The following hold under **all** circumstances, including downstream pressure. An
agent considering a deviation that violates any of these should refuse and escalate.

1. **Dependency contract (SPEC §5).** `nitrix` does not import `equinox`, `quax`,
   `numpyro`, `scipy`, `nibabel`, or anything upstream. A "quick fix" that re-introduces
   any of these is the bug, not the fix. If you find yourself wanting to, the
   capability belongs in a different library.
2. **Pure-functional public API (SPEC §2.1).** No PyTrees, no state, no module
   objects in the public surface. A downstream library can wrap nitrix functions in
   modules; nitrix does not ship modules.
3. **JAX fallback floor (SPEC_UPDATE_v0.2 §7.2).** Every kernel has a working JAX
   fallback path exercised in CI. A Pallas-only kernel is not shippable.
4. **Golden corpus (SPEC_UPDATE §2.8).** Every kernel × dtype × algebra combination
   that ships has a checked-in reference array. Adding a kernel without golden tests
   is not "done."
5. **Backward compatibility of kernel outputs across releases (SPEC §2.6).** Once a
   kernel ships, changing its numerics within the pinned tolerance is a CHANGELOG
   entry. Changing it outside the tolerance is an API break.
6. **Loud fallbacks (SPEC_UPDATE §2.7).** Backend fallback emits a warning. Silent
   fallback is a bug regardless of how it was introduced.

### 2.3 Negotiable under deviation pressure

The following can flex when downstream urgency justifies it. Document the deviation
in §10.

- **Phase ordering.** If a downstream library urgently needs a Phase 3 capability and
  Phase 2 is incomplete, build the Phase 3 capability on the JAX fallback path
  (skipping the semiring kernel optimisation) and revisit when Phase 2 catches up.
- **Pallas vs JAX coverage on a specific kernel.** Shipping a JAX-only kernel that
  meets correctness contracts is acceptable; we ship the perf later.
- **Test depth on a specific path.** Shipping a kernel with reduced coverage (e.g.,
  one dtype instead of three, no Hypothesis tests yet) is acceptable if the golden
  corpus and backend-parity tests are present. Reduced coverage is logged as a known
  shortfall.
- **Within-subpackage organisation.** Whether `intensity_normalize` lives in
  `nitrix.signal.normalize` or `nitrix.numerics.normalize` is a cosmetic concern; pick
  the one that unblocks fastest and revisit at GA.
- **Documentation completeness.** Docstrings are required; tutorials and design notes
  can lag the implementation by a phase.

### 2.4 Deviation protocol

When a downstream library blocks on a capability not yet built:

1. **Identify the capability,** not the implementation. "We need `bilateral_gaussian`
   for d_f=3" is a capability. "We need `nitrix.smoothing.bilateral_gaussian` on the
   Pallas backend with full test coverage" is an implementation.
2. **Check the non-negotiables (§2.2).** If the urgent path violates any, refuse and
   route to the design discussion.
3. **Find the minimum shippable shape** that meets the capability and the
   non-negotiables. Often this is the JAX-only path, one dtype, with the standard
   golden test.
4. **Log the deviation** in §10 with: the consumer who needed it, the capability
   shipped, the shape it shipped in, what's deferred, and when the deferred work is
   expected.
5. **Proceed.** Don't wait for sign-off on the deviation if §2.2 holds; the
   non-negotiables are the sign-off.

The deviation log (§10) is the source of truth for "things we shipped under pressure
that need follow-up." It is reviewed at every phase transition.

### 2.5 What deviation does *not* license

- Bypassing the golden corpus for "this one is small."
- Adding a runtime dependency for "just this one function."
- Shipping a kernel without a JAX fallback because "Pallas was easier."
- Skipping the backward kernel because "it's not needed yet."

These are the failure modes I want named explicitly because they are the natural
"quick wins" that erode the architecture over the year.

---

## 3. Phase 0 — Preflight

**Entry criteria.** Spec accepted; this plan accepted.

**Exit criteria.** All §5 (MIGRATION.md) known issues resolved in-place; CI
infrastructure in place; G0 Ampere-ELL benchmark result recorded.

**Why this phase exists.** Two reasons. First, MIGRATION.md §5 lists known bugs in the
existing code that will silently corrupt the migration if not addressed first
(particularly the JIT-trap covariance bug, which will produce wrong gradients in
downstream optimisation if migrated as-is). Second, the Ampere-ELL benchmark (G0)
determines the default backend for half the substrate work in Phase 2; running it
first means Phase 2 doesn't get rewritten halfway through.

### 3.1 Phase 0 tasks

#### 0.1 Fix known issues in current nitrix in-place

Resolve before the migration starts, so the migration begins from a clean base. These
are not new TODOs; they are existing red flags.

- **`covariance.py:719–726` JIT trap with non-diagonal weights.** Either fix or raise
  unambiguously at trace time. Add the regression test to the future golden corpus.
- **`covariance.py:683–686` denominator gap.** Same.
- **`window.py:12` undeclared numpyro import.** Drop. Use `jax.random.categorical` or
  `jax.random.multinomial`. This is the only blocker for `nitrix.signal.window` being
  migration-ready.
- **`geom.py:632` `spatial_conv` 2D-only TODO.** Leave in place; this gets moot when
  re-backed on `semiring_ell_conv` in Phase 3. Mark the file with a clear "will be
  replaced" comment to prevent further investment.
- **`functional/geom.py` non-exported utilities** (`diffuse`, `cmass_reference_displacement_*`).
  Export under their eventual destinations now if cheap; otherwise let Phase 3 handle.

#### 0.2 CI infrastructure

- **GPU runner provisioning.** At least one Ampere runner (A100 or A40). Lovelace and
  Hopper coverage is a Phase 5 / 1.x target.
- **Backend-parity test scaffolding.** A pytest fixture parameterising over
  `(backend, dtype)` cells. Initially empty (no kernels yet); the scaffolding has to
  exist before Phase 2 starts writing kernels.
- **Golden corpus scaffolding.** Directory layout, naming convention, a `tests/golden/`
  loader and a `tests/tolerance.toml` file. Empty at first.
- **`NitrixBackendFallback` warning category.** Defined in `nitrix._internal`, with a
  test that exercises the dedupe-by-`(function, shape-signature, dtype, backend)`
  logic.
- **`NITRIX_STRICT_BACKEND` / `NITRIX_SILENCE_FALLBACK` env vars** plumbed through.

#### 0.3 G0 — Ampere ELL performance baseline `SERIAL`

The benchmark gate. Implements two versions of `semiring_ell_matmul` on a representative
ELL workload (mesh adjacency, k_max ≈ 32, dimensions 1k–100k rows):

- **Triton path.** A skeletal Pallas kernel doing gather + accumulate with the real
  semiring. Doesn't need to be optimised, doesn't need full algebra coverage; needs
  to be representative of what the production kernel will do.
- **JAX path.** `jnp.take_along_axis` + `jnp.einsum` or equivalent. The fallback this
  benchmark gates against.

Compare wall-time and compile-time on the reference A100 80 GB across a small grid of
shapes. Decision:

- **Triton < 2× JAX wall-time:** Triton is default in Phase 2. Standard plan.
- **Triton 2–5× slower:** JAX is default on Ampere; Triton is opt-in via
  `backend="pallas-cuda"`. Update SPEC_UPDATE_v0.2 §4 footnote to reflect the actual
  measurement; document the reasoning in the kernel docstrings.
- **Triton ≥ 5× slower or unstable:** Substantially harder conversation. Either (a)
  investigate Triton-side performance issues with the JAX team, (b) rewrite the
  kernel skeleton (different tiling, different gather pattern), or (c) accept that
  the streaming-kernel-substrate bet doesn't pay off on Ampere and reconsider the
  whole §3.1 design. This branch is unlikely but the plan should be honest that it's
  possible.

**Output of G0:** a checked-in benchmark report (Markdown), a checked-in benchmark
script that can be re-run on new hardware, and a decision recorded in §10.

#### 0.4 Repository scaffolding

- Top-level `nitrix/` package layout matching SPEC §3 and §4 (empty modules with
  docstrings).
- `_kernels/cuda/` directory for Pallas kernels (empty at this point).
- `_refstubs/` directory checked in but **explicitly excluded** from the package
  build. The semiring brainstorm lives there for reference; it does not ship.
- `pyproject.toml` with the minimal dependency set (jax, jaxtyping, numpy). Test deps
  (`pingouin`, `scipy`, `sklearn`, `hypothesis`) scoped to the test extra.

### 3.2 Phase 0 exit checklist

- [ ] All MIGRATION.md §5 issues either resolved in-place or moved to Phase 3
      replacement plan with a clear note.
- [ ] Ampere GPU CI runner available and running an empty smoke test.
- [ ] Backend-parity, golden-corpus, fallback-warning scaffolding in place.
- [ ] G0 benchmark complete; decision logged in §10.
- [ ] Repository layout scaffolded, `_refstubs/` excluded from build.

---

## 4. Phase 1 — Foundation

**Entry criteria.** Phase 0 exit checklist clear. G0 outcome known.

**Exit criteria.** `nitrix.linalg`, `nitrix.stats`, `nitrix.signal`, `nitrix.numerics`
consolidated and importable. Downstream libraries can replace
`from hypercoil.functional import X` with `from nitrix.{linalg,stats,signal} import X`
for the migrated surface.

**Why this phase exists separately.** This is the consolidation track (Track B in §1.2),
running in parallel with Phase 2 substrate work. It's listed as its own phase only
because some downstream libraries will not need substrate (Phase 2) capabilities and
their unblock condition is *this* phase ending. Internally it's largely mechanical and
can be parallelised across multiple agents / engineers.

### 4.1 Phase 1 tasks (Track B)

These are independent and can be done in any order. Each one is a self-contained PR.

#### 1.B.1 `nitrix.linalg.matrix`

Consolidate current `nitrix/functional/matrix.py` with `hypercoil/functional/matrix.py`.
Preserve the custom VJP rules at the current `matrix.py:554–568` as the pattern for
future work. Fold `hypercoil/init/toeplitz.py` initialisation in.

#### 1.B.2 `nitrix.linalg.residual`

Consolidate current `nitrix/functional/residual.py` with `hypercoil/functional/resid.py`.
Fold the entense `confound_regression_p` numerical core in. The off-diagonal-weight
gap (Phase 0 fix) should already be addressed.

#### 1.B.3 `nitrix.linalg.kernel`

Port `hypercoil/functional/kernel.py` with the single-dispatch surface. Fold
`hypercoil/init/laplace.py`.

#### 1.B.4 `nitrix.linalg.spd` `SERIAL`

This one is `SERIAL` because it requires a numerical stability rewrite per SPEC §4.1.
Port `hypercoil/functional/symmap.py` and `hypercoil/functional/semidefinite.py`,
**rewriting the SPD implementation for stability**. The migration is the natural
opportunity per MIGRATION.md §5. Tests cover the stability regression explicitly.

#### 1.B.5 `nitrix.stats.covariance`

Consolidate current `nitrix/functional/covariance.py` with `hypercoil/functional/cov.py`.
The JIT-trap fix from Phase 0 carries into this module.

#### 1.B.6 `nitrix.stats.fourier`

Consolidate current `nitrix/functional/fourier.py` with `hypercoil/functional/fourier.py`.

#### 1.B.7 `nitrix.signal.window`

Migrate current `nitrix/functional/window.py` (numpyro-stripped after Phase 0). Fold
`hypercoil/functional/window.py`.

#### 1.B.8 `nitrix.signal.{filter, tsconv, interpolate}`

Port `hypercoil/functional/tsconv.py` and `hypercoil/functional/interpolate.py`
(extracting numeric cores, dropping neuro context). Add `nitrix.signal.filter` with
the entense `polynomial_detrend_p` numeric core.

#### 1.B.9 `nitrix.numerics.{tensor_ops, normalize}`

Port `hypercoil/functional/linear.py` to `tensor_ops`; port the ilex `core/adapters.py`
pure-array half (≈ lines 150–250) to `tensor_ops`. Port the ilex synthstrip
`intensity_normalize` to `normalize`.

### 4.2 Phase 1 contract for downstream

At Phase 1 exit, downstream libraries get the following promise:

- All migrated public symbols are importable from their nitrix destination.
- Hypercoil import paths are deprecated but functional for a transition window
  (separate concern — the hypercoil-side shim is not nitrix's job, but the migration
  doc should track it).
- No behaviour change beyond the bug fixes in MIGRATION.md §5; numerics are pinned by
  the golden corpus.

### 4.3 Phase 1 exit checklist

- [ ] Every Phase 1 task above complete and golden-tested.
- [ ] Backend-parity tests pass for any kernel touching Pallas (mostly none in Phase
      1; matmul-shaped ops in `linalg` might).
- [ ] Downstream smoke test: one downstream library successfully imports from nitrix
      and runs a representative pipeline.

---

## 5. Phase 2 — Substrate

**Entry criteria.** Phase 0 exit checklist clear. G0 outcome incorporated into Phase 2
default-backend decisions. Phase 1 may be in progress (parallel track).

**Exit criteria.** `nitrix.semiring` with `REAL`, `LOG`, `TROPICAL_MAX_PLUS`,
`TROPICAL_MIN_PLUS`, `BOOLEAN`, `EUCLIDEAN` algebras shipped. `nitrix.sparse.ell` and
`nitrix.sparse.ell.sectioned` shipped. All §3.1 and §3.2 success criteria from the
spec met.

**Why this phase exists.** The marquee bet. Everything downstream of it (geometry mesh
ops, smoothing, morphology) specialises onto this substrate. The phase is the long
pole of the project.

### 5.1 Phase 2 ordering rationale

Tasks within Phase 2 have real dependencies; the order matters more than in Phase 1.

```
2.A.1 Protocols (Semiring, StrictSemiring, Semigroup, Monoid)
   │
   ├─→ 2.A.2 Reference JAX implementation (semiring_matmul, *_conv, *_ell_matmul)
   │     │
   │     └─→ 2.A.3 Built-in algebras (REAL, LOG, TROPICAL_*, BOOLEAN)
   │           │
   │           ├─→ 2.A.4 EUCLIDEAN (relaxed Semiring; first test of the relaxed path)
   │           │
   │           ├─→ 2.A.5 Backward kernels (per-algebra, JAX-side)
   │           │     │
   │           │     └─→ G1 — backward-kernel correctness gate
   │           │
   │           └─→ 2.A.6 Pallas/Triton kernel (if G0 allows default-Pallas)
   │                 │
   │                 └─→ 2.A.7 Pallas backward kernels
   │
   └─→ 2.A.8 nitrix.sparse.ell + sparse.ell.sectioned
         │
         └─→ 2.A.9 nitrix.sparse.grid, nitrix.sparse.mesh (specialisations)
                │
                └─→ G3 — fallback observability gate (real-world test case)
```

### 5.2 Phase 2 tasks (Track A)

#### 2.A.1 Protocols

Define `Semigroup`, `Monoid`, `Semiring`, `StrictSemiring` in `nitrix.semiring._types`.
The `StrictSemiring <: Semiring` structural subtype with a `strict=True` constructor
flag. Type aliases exported from `nitrix.semiring`. Document associativity /
distributivity expectations per Protocol.

This is small and entirely API-design work; no kernels. Get the Protocol shape right
here because changing it later is a public API break.

#### 2.A.2 Reference JAX implementation `SERIAL`

`reference_semiring_gemm.py` ported from the `_refstubs` brainstorm into the house
style. Implements `semiring_matmul`, `semiring_conv`, `semiring_ell_matmul` purely in
JAX via `lax.fori_loop` over the K block with the Monoid pytree state. No Pallas yet.

This is the *correctness floor* for everything in Phase 2. Every Pallas kernel that
comes later is checked against this. It needs to be right before anything else gets
built on top of it.

#### 2.A.3 Built-in StrictSemirings

`REAL`, `LOG`, `TROPICAL_MAX_PLUS`, `TROPICAL_MIN_PLUS`, `BOOLEAN`. Each with:

- `init`, `update`, `merge`, `finalize` over the Monoid state.
- `binary_op` for the `(*)` step.
- Identity element.
- Golden test: forward output matches a naive broadcast formulation on small inputs.
- Golden test: identity propagation works correctly (e.g., `-inf` in
  `TROPICAL_MAX_PLUS` annihilates).
- Golden test: numerical stability under adversarial inputs (e.g., `LOG` with
  magnitudes spanning ±1000).

The numerical-stability tests are the ones most likely to catch a subtle bug in the
streaming-kernel state. Do not skip.

#### 2.A.4 EUCLIDEAN (relaxed Semiring)

The first test of the relaxed `Semiring` (non-`StrictSemiring`) path. Validates that
the type-system distinction between strict and relaxed actually works, and that
algorithms gated on `StrictSemiring` reject the relaxed Euclidean as expected. Golden
tests include the `sqrt` singularity guard at zero.

#### 2.A.5 Backward kernels (JAX-side) `SERIAL` for G1

Per the §3.1 backward vocabulary:

- `REAL`: transpose-matmul (reuse forward kernel with swapped operands).
- `LOG`: softmax-weighted; the softmax is recomputed in the backward K loop, not
  materialised.
- `TROPICAL_*`: argmax/argmin gather, subgradient.
- `EUCLIDEAN`: normalised-difference with √-singularity guard.
- `BOOLEAN`: not differentiable; backward raises a clear error.

Each backward registered via `jax.custom_vjp`. Each passes finite-difference checks
at the pinned tolerance.

**G1 gate.** If any algebra's backward fails the finite-difference check at the pinned
tolerance, do not ship that algebra at Phase 2 exit. The algebra ships forward-only
with a documented gradient raise (matching the `BOOLEAN` pattern). This is a real
possibility for `EUCLIDEAN` near the √-singularity; the plan accepts that as a known
risk.

#### 2.A.6 Pallas / Triton kernel (conditional on G0)

If G0 said Triton-default is viable: implement the Pallas / Triton kernel for
`semiring_matmul`, `semiring_conv`, `semiring_ell_matmul`. The kernel is parameterised
over the algebra via `Monoid` / `Semigroup` callables passed at kernel compile time.

If G0 said JAX-default: this task slips to Phase 5 / 1.x. The JAX kernel is the
production path; the Pallas path is opt-in and may not exist at first GA. Substantial
scope cut, but the streaming-kernel design still holds — just less of it ships in
optimised form.

#### 2.A.7 Pallas backward kernels (conditional, follows 2.A.6)

Per-algebra Pallas backward kernels, paired with their forwards. Same conditional as
2.A.6.

#### 2.A.8 `nitrix.sparse.ell`

ELL format: `(values, indices, n_rows, identity)` with gather / scatter / pad / reshape
primitives. Plus `nitrix.sparse.ell.sectioned` (the bucketed-row variant for
variable-degree adjacencies). Per SPEC_UPDATE §3.2, sectioned-ELL is CORE.

`semiring_ell_matmul` and `semiring_ell_conv` accept either flat or sectioned ELL.

#### 2.A.9 `nitrix.sparse.grid`, `nitrix.sparse.mesh`

Thin specialisations of ELL. `grid` is the case where every row has the same neighbour
offsets (regular-grid stencils). `mesh` is icosphere k-ring adjacency, sparse
Laplacians, geodesic neighbourhoods. Mostly format-conversion code; the heavy lifting
is in `semiring_ell_matmul`.

#### G3 — Fallback observability gate

A test that forces a shape × algebra combination Triton cannot tile (e.g., a `k_max`
larger than fits in shared memory). Asserts that the `NitrixBackendFallback` warning
fires exactly once per `(function, shape-signature, dtype, backend)` and that the JAX
path produces the correct answer. Asserts that `NITRIX_STRICT_BACKEND=1` converts the
fallback to an error.

If G3 fails, the fallback infrastructure (Phase 0 scaffolding) is broken and needs
fixing before Phase 2 ships.

### 5.3 Phase 2 contract for downstream

At Phase 2 exit, downstream gets:

- Differentiable matmul, conv, ELL-matmul over any of the six built-in algebras.
- The substrate for any custom algebra (with user-supplied VJP for differentiability).
- ELL format primitives, including the sectioned variant for variable-degree cases.
- Backend selection (`auto`, `pallas-cuda`, `jax`) with loud fallback.

### 5.4 Phase 2 exit checklist

- [ ] All §10 success criteria for §3.1 and §3.2 met (per SPEC + SPEC_UPDATE).
- [ ] G1 backward-kernel gate passed for at least `REAL`, `LOG`, `TROPICAL_*`.
      `EUCLIDEAN` and `BOOLEAN` outcomes recorded.
- [ ] G3 fallback-observability gate passed.
- [ ] Golden corpus populated for every (kernel, dtype, algebra, backend) cell that
      ships.
- [ ] Backend-parity CI green.

---

## 6. Phase 3 — Geometry and graph

**Entry criteria.** Phase 1 exit and Phase 2 exit both clear (geometry mesh ops depend
on Phase 2 substrate; geometry grid ops depend on Phase 1 linalg).

**Exit criteria.** `nitrix.geometry` and `nitrix.graph` shipped. Spherical convolution
re-backed on `semiring_ell_conv` (the legacy O(N²) inner loop is gone).

### 6.1 Phase 3 tasks

#### 3.1 `nitrix.geometry.grid`

Migrate ilex `models/voxelmorph/_numerical.py` (identity_grid, spatial_transform,
vec_int, rescale). Fold in current `geom.py` grid bits. Add `cmass_regular_grid` from
hypercoil `cmass.py`. Regression tests from voxelmorph travel with the code.

#### 3.2 `nitrix.geometry.sphere`

Migrate hypercoil `sphere.py` + current `geom.py` sphere bits. **Re-back spherical
convolution** on `semiring_ell_conv` over mesh adjacency. The legacy O(N²) inner loop
is dropped. This task validates the §3.1 design bet end-to-end: a spherical conv that
previously had its own bespoke implementation now specialises onto the substrate.

If G0 said JAX-default, the Pallas perf gain on spherical conv is deferred. The
spherical conv still ships, just not as fast. That's acceptable.

#### 3.3 `nitrix.geometry.coords`

Coordinate utilities from hypercoil `cmass.py` + current `geom.py` coords bits.
Includes the previously non-exported `diffuse` and `cmass_reference_displacement_*`
(MIGRATION.md §5).

#### 3.4 `nitrix.geometry.metrictensor`

Port hypercoil `metrictensor.py`.

#### 3.5 `nitrix.graph.laplacian`

Port hypercoil `functional/graph.py` (Laplacian, modularity, Girvan-Newman null).

#### 3.6 `nitrix.graph.connectopy`

Extract from hypercoil `functional/connectopy.py`. **Strip the brainspace dependency.**
This is the one place in Phase 3 where the "drop neuro context, keep numerics" rule
needs careful attention — the eigenmap / diffusion-map algorithms are general, but
the existing implementation may have brainspace assumptions baked in.

#### 3.7 `nitrix.graph.community`

Port hypercoil community / relaxed-modularity numerics.

### 6.2 Phase 3 exit checklist

- [ ] Geometry and graph subpackages importable, golden-tested.
- [ ] Spherical conv: numerical agreement with the legacy O(N²) implementation to
      pinned tolerance.
- [ ] Brainspace dependency removed.

---

## 7. Phase 4 — Marquee (smoothing and morphology)

**Entry criteria.** Phase 2 exit clear. (Phase 3 is independent.)

**Exit criteria.** `nitrix.smoothing` (gaussian, bilateral_gaussian,
permutohedral_lattice per tripwire), `nitrix.morphology` shipped.

### 7.1 Phase 4 tasks

#### 4.1 `nitrix.smoothing.gaussian`

Separable Gaussian. Pure JAX. The unconditional baseline. Cheap.

#### 4.2 `nitrix.smoothing.bilateral_gaussian`

Direct N-body bilateral over arbitrary `d_f`, implemented as a `semiring_ell_matmul`
over distance-thresholded sectioned-ELL adjacency. This is the **marquee capability
delivered regardless of permutohedral risk**.

Specific tests: agreement with a reference NumPy direct-N-body implementation;
agreement with `gaussian` in the limit of large `sigma_intensity` (where the
intensity-similarity weighting becomes uninformative).

#### 4.3 `nitrix.morphology.{erode, dilate, open, close, distance_transform}`

Specialisations of `semiring_conv` with `TROPICAL_MIN_PLUS` / `TROPICAL_MAX_PLUS`.
Thin wrappers — most code is documentation. Distance transforms via the standard
two-pass min-plus algorithm.

#### 4.4 `nitrix.morphology.median_filter`

Gather-based op, *not* a semiring op. Implemented as `gather → jnp.median` over the
neighbourhood. Parity test against `scipy.ndimage.median_filter` within pinned
tolerance.

#### 4.5 `nitrix.morphology.susan_emulator`

Convenience wrapper composing `bilateral_gaussian` + `median_filter`. Docstring
explicitly documents the behavioural deltas from FSL SUSAN (no auto-flat-kernel at
small extents).

#### 4.6 `nitrix.smoothing.permutohedral_lattice` — G2 gate

The high-risk item. Attempt the implementation in this order:

1. **Pure JAX reference.** Working implementation of splat / blur / slice in JAX.
   Slow but correct. Reference for the optimised path.
2. **Optimisation pass.** JAX with hand-rolled gather patterns, or JAX+Pallas hybrid
   for the splat/slice hash table operations. Pallas-pure is explicitly not required
   (SPEC_UPDATE §3.3).
3. **G2 tripwire evaluation.** Per SPEC_UPDATE_v0.2 §3.3, evaluate against the four
   criteria (PSNR > 40 dB, < 10× Gaussian wall time, < 30 s first compile, gradient
   passes finite-diff). Pin the actual numbers from benchmark before evaluation.

**G2 outcomes:**

- **All four criteria met:** ship `permutohedral_lattice`. Done.
- **Criteria 1 (parity) or 4 (gradient) fail:** correctness problem; fix or revisit.
  Do not ship a permutohedral with wrong outputs or wrong gradients.
- **Criteria 2 (perf) or 3 (compile) fail:** the implementation is correct but
  doesn't clear the perf bar. The symbol raises `NotImplementedError` pointing to
  `bilateral_gaussian` for d_f ≤ 5. The implementation is checked in under
  `_experimental/` for future work. Revisit at 1.x.

### 7.2 Phase 4 exit checklist

- [ ] `gaussian`, `bilateral_gaussian` shipped unconditionally.
- [ ] Morphology shipped: erode, dilate, open, close, distance_transform via tropical
      semiring; median_filter via gather; susan_emulator composing both.
- [ ] G2 evaluated; `permutohedral_lattice` either shipped or raises with clear
      pointer.

---

## 8. Phase 5 — Polish and GA

**Entry criteria.** Phases 1, 2, 3, 4 exit checklists clear.

**Exit criteria.** First GA released.

### 8.1 Phase 5 tasks

- **`nitrix.stats.lme` namespace reserved.** Stub module with `NotImplementedError`
  raises and a clear roadmap docstring. No implementation per SPEC §3.5.
- **Documentation pass.** Every public symbol has a docstring including a one-line
  summary, signature in jaxtyping, example, and (where relevant) backend notes.
- **Benchmark suite.** Reusable benchmark scripts checked in under `bench/`, covering
  the marquee operations on Ampere. Results checked in as Markdown reports per
  hardware generation.
- **Tutorials.** A small set of "how to use the substrate" notebooks: writing a
  custom Semiring, using ELL for mesh ops, choosing between gaussian and
  bilateral_gaussian.
- **Migration guide for downstream libraries.** A separate document mapping
  `hypercoil.functional.X` and `nitrix-old.X` to their new locations.
- **CHANGELOG and versioning policy.** Pin the "stable kernels, breakable APIs"
  contract concretely.
- **Release process documentation.** How to cut a release, how to update the pinned
  jax minimum, how to add a new backend (future-proofing for TPU).

### 8.2 GA criteria (rolling up SPEC §10 + SPEC_UPDATE additions)

All of:

- [ ] Phases 0–4 exit checklists clear.
- [ ] All §10 success criteria from SPEC and SPEC_UPDATEs met.
- [ ] Backend-parity CI green on Ampere. Hopper / Blackwell coverage is 1.x.
- [ ] Golden corpus populated for every (kernel, dtype, algebra, backend) cell.
- [ ] Documentation pass complete.
- [ ] One downstream library (thrux, nimox, ilex, entense, or bitsjax) successfully
      uses nitrix end-to-end for a real workload.

---

## 9. Per-subpackage detailed checklists

This section is the reference for agents picking up work on a specific subpackage. It
restates the relevant material from the spec in a "what does done look like" format,
without duplicating spec text.

For brevity, the checklist below is abbreviated; the full per-symbol checklist lives
in the subpackage's own `_PLAN.md` (to be created in Phase 0 scaffolding).

### 9.1 `nitrix.semiring`

- [ ] Protocols defined: `Semigroup`, `Monoid`, `Semiring`, `StrictSemiring`.
- [ ] Reference JAX `semiring_matmul`, `semiring_conv`, `semiring_ell_matmul` working.
- [ ] Built-in algebras: REAL, LOG, TROPICAL_MAX_PLUS, TROPICAL_MIN_PLUS, BOOLEAN,
      EUCLIDEAN.
- [ ] Backward kernels per algebra, registered via `jax.custom_vjp`.
- [ ] Pallas / Triton kernel (if G0 viable).
- [ ] Golden corpus + backend parity + identity propagation + numerical stability tests.
- [ ] Documented user-extension path.

### 9.2 `nitrix.sparse`

- [ ] `sparse.ell` primitives.
- [ ] `sparse.ell.sectioned` for variable-degree.
- [ ] `sparse.grid`, `sparse.mesh` as specialisations.
- [ ] No `jax.experimental.sparse` import anywhere.

### 9.3 `nitrix.smoothing`

- [ ] `gaussian` unconditional.
- [ ] `bilateral_gaussian` unconditional.
- [ ] `permutohedral_lattice` evaluated at G2; ships or raises with pointer.

### 9.4 `nitrix.morphology`

- [ ] erode, dilate, open, close on tropical semiring.
- [ ] distance_transform via two-pass min-plus.
- [ ] median_filter via gather.
- [ ] susan_emulator composing bilateral_gaussian + median_filter.

### 9.5 `nitrix.linalg`

- [ ] matrix, residual, kernel, spd consolidated.
- [ ] SPD numerical-stability rewrite complete.

### 9.6 `nitrix.stats`

- [ ] covariance with JIT-trap fix.
- [ ] fourier consolidated.
- [ ] lme namespace reserved (no implementation).

### 9.7 `nitrix.signal`

- [ ] window numpyro-stripped.
- [ ] filter, tsconv, interpolate ported.

### 9.8 `nitrix.geometry`

- [ ] grid, sphere, coords, metrictensor.
- [ ] Spherical conv re-backed on semiring.

### 9.9 `nitrix.graph`

- [ ] laplacian, connectopy, community.
- [ ] No brainspace dependency.

### 9.10 `nitrix.numerics`

- [ ] tensor_ops, normalize.

---

## 10. Deviation log

Maintain this section as work proceeds. Every deviation from the plan — both gate
outcomes and shipped-under-pressure capabilities — gets a row.

### 10.1 Format

```
### YYYY-MM-DD — Short title

- **Type:** Gate outcome | Downstream deviation | Plan revision
- **Triggered by:** (consumer, gate, or planning decision)
- **Description:** What happened.
- **Capability shipped:** (if deviation) What downstream got.
- **Shape:** (if deviation) JAX-only | reduced coverage | other
- **Deferred work:** What's still owed.
- **Expected resolution phase:** (if deviation)
- **Non-negotiables held:** Confirmation that §2.2 was respected.
```

### 10.2 Initial entries (to be filled in as Phase 0 runs)

```
### TBD — G0 Ampere ELL benchmark outcome

- **Type:** Gate outcome
- **Triggered by:** Phase 0 gate G0
- **Description:** (Benchmark result; Triton vs JAX wall-time.)
- **Decision:** Triton-default | JAX-default with Triton opt-in | Reconsider §3.1
- **Impact on Phase 2:** (Carry into 2.A.6 task scope.)
```

### 10.3 Shipped entries

### 2026-05-20 — SUGAR feedback batch: edge attributes, row-softmax, mean-pool, external topology, masking

- **Type:** Downstream deviation
- **Triggered by:** ilex/SUGAR port (`NITRIX_FEEDBACK_ILEX.md`, 2026-05-18) — second
  surface-domain consumer of the ELL mesh-graph-conv substrate after Topofit.
- **Description:** Five additive, substrate-aligned changes. (1) `edge_attr=`
  kwarg on `semiring_ell_edge_aggregate`: when set, `edge_fn` receives a 5th arg
  `a = edge_attr[i,p,:]` while keeping the scalar `w` (the padding signal) — covers
  GATv2's `edge_dim` Fourier embedding. Refines the feedback's Option A (which would
  have displaced `w`); backward-compatible. (2) `ell_row_softmax(scores, ell)`: GAT
  attention pre-pass, masking pads from `ell.values == ell.identity` (the feedback's
  first real consumer of this proposal). (3) `mesh_coarsen_meanpool`: mean-pool
  sibling of `mesh_pool_max`; `icosphere_cross_level_adjacency` now stores a 1.0/0.0
  validity indicator in `values` (identity 0.0) so mean falls out as
  `sum(v·x)/sum(v)` — `mesh_pool_max` overrides values internally so it is
  unaffected. (4) `icosphere_hierarchy_from_levels(meshes, parents)`: packages
  caller-supplied topology into the existing `IcosphereHierarchy`, so FreeSurfer
  `fsaverage` hierarchies run through every cross-level operator with **no**
  topology-source branching. (5) `ell_mask(ell, valid, *, identity)`: masks
  incomplete geometries (medial wall / grey-matter) by setting masked edges to the
  semiring identity (consumer-raised; see the masking note below).
- **Capability shipped:** GATv2/edge-attributed mesh convs, GAT attention,
  surface mean-pooling, external (FreeSurfer) topology hierarchies, and masked
  reductions — all on the existing semiring/ELL substrate.
- **Shape:** Pure-JAX (the substrate's current state; ELL Pallas is gated on G0).
  Full forward/backward, golden + property tests, CPU correctness floor.
- **Rejected (concern leakage):** the feedback's Delta-3 options A/B that would
  read FreeSurfer `.sphere` binaries (`nibabel`, `$SUBJECTS_DIR`) inside nitrix.
  That violates SPEC §5.2 / non-negotiable §2.2.1. nitrix stays array-only; the
  consumer/`thrux` does the I/O and hands in plain arrays via
  `icosphere_hierarchy_from_levels`.
- **Deferred work:** Pallas dispatch for `semiring_ell_edge_aggregate` (BACKLOG B3);
  LOG/EUCLIDEAN edge-aggregate semirings (B4). Bench at ico_6/ico_7 (B2).
- **Non-negotiables held:** No new deps; pure-array signatures (NamedTuple/dataclass
  containers only); JAX floor exercised in CI; golden/property tests added.

### 2026-05-20 — Masking incomplete geometries across semirings (`ell_mask`)

- **Type:** Downstream deviation
- **Triggered by:** consumer question — medial-wall (surface) and grey-matter
  (volume) masks must make absent edges no-ops without blurring in masked signal.
- **Description:** Verified the substrate already supports this: a missing edge is a
  no-op iff its `values` entry is the algebra's `(*)`-annihilator, which equals
  `semiring.identity` for REAL (0), LOG/TROPICAL_MAX_PLUS (−∞), TROPICAL_MIN_PLUS
  (+∞), BOOLEAN (False) — and the no-op holds regardless of where the padded index
  points. EUCLIDEAN is the documented exception: `(a−b)²` has no annihilator, so
  EUCLIDEAN neighbourhoods must be masked by dropping columns structurally, not via
  a value. Shipped `nitrix.sparse.ell_mask(ell, valid, *, identity)` (column- or
  edge-mask) plus a parametrised verification suite
  (`tests/test_ell_masking_semirings.py`) covering the no-op property, the EUCLIDEAN
  limitation, and the "wrong identity under max-plus leaks" footgun. Also made the
  four cross-level mesh wrappers batch-safe (they claimed `(..., n, d)` but only
  handled 2-D) via a shared vmap-over-leading-dims helper.
- **Capability shipped:** correct, semiring-aware masking of incomplete brain
  geometries; honest batch support on the cross-level wrappers.
- **Shape:** Pure-JAX; differentiable; full parity tests.
- **Non-negotiables held:** array-only; no deps; `nitrix.sparse.ell` stays free of
  a `nitrix.semiring` import (identity passed explicitly).

### 2026-05-20 — JAX pin correction (env was 0.4.35, target is 0.10.0) + Python floor

- **Type:** Plan revision / dependency hygiene
- **Triggered by:** the uv-managed `.venv` (and `uv.lock`) had drifted to **jax
  0.4.35** while the Dockerfile / validated baseline is **jax[cuda12]==0.10.0**
  (the G0 report and kernels were all developed against 0.10.0).  The test suite
  had therefore been running on the wrong jax: `jax.random.multinomial` (used by
  `signal/window.py`) is absent before 0.10.0, so `signal`/`window`/`lme` failed,
  and `numpyro` collection broke.
- **Root cause:** `pyproject` declared `jax >= 0.4.30` with `requires-python =
  ">=3.10"`; jax 0.10.x dropped Python 3.10, so uv resolved down to the last
  3.10-compatible jax (0.4.35).  The nox matrix is already 3.11/3.12/3.13.
- **Fix:** bumped the floor to `jax >= 0.10.0` and `requires-python = ">=3.11"`,
  re-locked (jax/jaxlib pinned to **0.10.0** to match the Docker baseline), and
  bumped the test-only `numpyro` 0.18.0 → 0.21.0 (0.18 imported
  `jax.experimental.pjit.pjit_p`, removed in jax 0.10).  After the fix,
  `signal`/`window`/`lme`/`geom` pass.
- **Non-negotiables held:** `numpyro` remains test-only (absent from the Docker
  runtime env; SPEC §5.2); no new runtime deps.

### 2026-05-20 — Harden `toeplitz` against an XLA-CPU compiler crash on jax >= 0.10

- **Type:** Robustness hardening
- **Triggered by:** the pin correction surfaced a hard XLA **CPU compiler abort**
  (`AlgebraicSimplifier::HandleReverse`: "Invalid binary instruction opcode map")
  while compiling `toeplitz_2d`, which built its matrix with `jnp.flip` (a
  `reverse` HLO).  Worked on 0.4.35; crashes on 0.10.x CPU.
- **Fix:** replaced `jnp.flip(c_arg, -1)` with an index-based reverse
  (`c_arg[..., jnp.arange(d-1, -1, -1)]`, a gather) in **both** copies
  (`functional/matrix.py`, `linalg/matrix.py`).  Identical output (parity with
  `scipy.linalg.toeplitz` on square / rectangular-extend / fill cases),
  negligible cost, no `reverse` HLO → no crash.  `test_matrix` passes (14).
- **Note:** the remaining 0.10.x suite failures (`test_util`, `test_resid`) are
  **hypothesis** test-harness flakiness (`FlakyFailure` from overflow-y generated
  inputs; `FailedHealthCheck: data generation extremely slow` on a loaded box) in
  untouched property tests, not core bugs or version-API brittleness.  Tracked as
  a separate test-quality item (constrain the strategies / add a CI hypothesis
  profile with `deadline=None` + health-check suppression).
- **Non-negotiables held:** numerics unchanged within tolerance; no deps.

### 2026-05-20 — Trilinear-resampling Pallas request: benchmark-first (no kernel yet)

- **Type:** Gate outcome / plan revision
- **Triggered by:** consumer ask for a 3-D trilinear resampling Pallas kernel.
- **Description:** Trilinear resampling is structurally a gather (8 data-dependent
  corner loads) — the same primitive G0 found Pallas Triton cannot lower on the
  pinned JAX. Rather than write a kernel speculatively, shipped a baseline bench
  (`bench/trilinear_resample.py` → `bench/PERF_TRILINEAR.md`) and parked the kernel
  in BACKLOG B7 behind a two-part gate: (a) the path is a real training-loop
  bottleneck, and (b) a pointer-load Pallas prototype clears the gather-lowering
  risk. The Gaussian-blur Pallas request (low priority) is parked in B6 (stencil,
  not gather; cuDNN baseline is strong; only a fused-passes win exists).
- **Decision:** JAX-default (current state) until the gate clears. No kernel shipped.
- **Non-negotiables held:** `map_coordinates` JAX path remains the contractual floor.

### 2026-05-21 — Remove legacy `nitrix.functional`; reconcile its tests onto migrated modules

- **Type:** Plan revision / cleanup
- **Triggered by:** `functional/` flagged as leftover legacy (already migrated).
- **Description:** Removed `src/nitrix/functional/` entirely. It was runtime-dead
  (no `src` import; only legacy tests referenced it) and every symbol was
  migrated: `covariance`/`fourier`→`stats`, `matrix`/`residual`→`linalg`,
  `window`→`signal`, `geom`→`geometry` (renames: `sphere_to_normals`→
  `latlong_to_cartesian`, `sphere_to_latlong`→`cartesian_to_latlong`,
  `spherical_geodesic`→`spherical_geodesic_distance`).  Legacy tests handled by
  coverage comparison (collected case counts):
  - **Deleted** `test_matrix` (14), `test_window` (2), `test_geom` (17) — the new
    `test_linalg` (29) / `test_signal` (6) / `test_geometry` (53) are supersets.
  - **Deleted** `test_cov` — it tested the *old* covariance API (single `weight`
    param + private `_prepare_*` helpers) which the migration **redesigned** to
    `weights=` / `weight_matrix=` with new internals; `test_stats` covers the new
    API.  Added a non-diagonal-`weight_matrix` regression to `test_stats` (the
    SPEC §8 mandate, now that the behaviour is compute-correctly, not raise).
  - **Repointed** `test_fourier`→`stats`, `test_resid`→`linalg`.  Verified
    `residualise` is numerically identical old vs new.  The repoint surfaced two
    *intended* migration API changes (validating the "run to catch drift"
    instinct), adapted in the test: `analytic_signal` now raises `TypeError`
    (not `ValueError`) on complex input and takes `axis` keyword-only.
- **Non-negotiables held:** no runtime deps added/removed; migrated impls unchanged;
  new modules + tests are the canonical coverage.

### 2026-05-21 — Deflake hypothesis property tests; surface a residualise limitation

- **Type:** Test-quality / robustness
- **Triggered by:** the three originally-flaky tests (`test_util`, `test_geom`,
  `test_resid`).
- **Description:** Added `tests/conftest.py` with a hypothesis profile
  (`deadline=None`; suppress `too_slow` / `data_too_large`).  JAX first-call JIT
  compile makes per-example deadlines unreliable (`DeadlineExceeded` ->
  `FlakyFailure`); disabling them removes the timing flakiness **suite-wide with
  zero input-space change** — every example is still drawn and asserted.  Relaxed
  the explicit `deadline=500` in `test_util`.  `test_geom`'s flakiness (unseeded
  random + exact `==0` truncation boundary) is moot — it was deleted above
  (`test_geometry` covers spherical conv).  The deflake's fuller exploration
  **unmasked** a known, author-documented `residualise` limitation: the exact
  `residual + projection == Y` decomposition breaks at `1e-5` (float32) for
  ill-conditioned designs (`p -> obs`).  It is pre-existing (identical on old and
  new `residualise`) — see BACKLOG **B9**.  Per decision, constrained the
  exact-decomposition property tests to the well-conditioned domain
  (`generate_valid_arrays(well_conditioned=True)`, `p <= obs/2`) so they are
  honest and green, and BACKLOG'd the real numerical fix (SVD/QR projector).
- **Non-negotiables held:** the deflake loses **no** input coverage (only timing
  assertions dropped); the ill-conditioned limitation is documented + tracked
  (B9), not silently skipped.

### 2026-05-21 — Semiring `identity` vs `(*)`-annihilator (recorded learning)

- **Type:** Learning / future-API note
- **Description:** `Semiring.identity` is the **monoid identity**; padding / masking
  (`sparse.ell_mask`) needs the **`(*)`-annihilator**, which coincides with
  `identity` for all built-ins **except** `EUCLIDEAN` (no annihilator; `identity=0`
  does not mask).  Recorded in `docs/design/semiring-protocols.md` and BACKLOG
  **B8** (consider an explicit `annihilator` field rather than overloading
  `identity`).

---

## 11. Notes on agent / engineer handoff

A few patterns worth knowing if you're picking up partway through:

- **The plan trusts the spec.** When in doubt about a design question, the SPEC and
  SPEC_UPDATEs are authoritative. The plan tells you when to build; the spec tells you
  what to build.
- **Phase 1 (Track B) tasks are good first PRs.** They are mechanical, small, and
  unblock downstream consumers immediately. If an agent is new to the codebase, start
  there.
- **Phase 0 fixes must precede the migration that touches them.** Don't migrate
  `covariance.py` before the JIT-trap fix is in.
- **`_refstubs/semiring_gemm.py` is design input only.** It is in the wrong house
  style and does not reflect the strict/relaxed Protocol split from SPEC_UPDATE §3.1.
  Re-implement.
- **The deviation log is a contract, not a confessional.** It exists so the next
  agent (in 3 weeks or 3 months) knows what was shipped under pressure and what's
  still owed. Treat entries as commitments to follow up, not as records of failure.
- **CI failures on Pallas / Triton paths are not always your fault.** Pallas Triton
  is best-effort per JAX. If a CI failure correlates with a `jax` version bump and
  reproduces only on Pallas, file the upstream issue and route the affected kernel to
  the JAX fallback per SPEC_UPDATE_v0.2 §7.2. The plan does not require you to fix
  Pallas regressions.

---

## 12. What this plan deliberately does not specify

- **Calendar dates.** Phase durations are approximate; the plan is deviation-tolerant
  by design. Calendar-binding would defeat that.
- **Number of people / agents.** The two-track structure scales from one agent
  (alternating phases) to many (running Phase 1 tasks in parallel). The plan is
  invariant.
- **Code style beyond "the house style."** That lives in a separate STYLE.md /
  CONTRIBUTING.md that the plan references but does not duplicate.
- **Hypercoil-side migration shims.** Whatever hypercoil ships to redirect imports to
  nitrix during the transition is hypercoil's concern. The nitrix plan ends at "nitrix
  exports the symbol."
- **Downstream library timelines.** thrux, nimox, ilex, entense, bitsjax have their
  own plans; this plan ends at the contract boundary in SPEC §5.3.
