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
| **4. Marquee** | `smoothing`, `morphology` shipped; permutohedral retired (SPEC_UPDATE_v0.4), bounded bilateral supersedes it | weeks–months | — (permutohedral risk eliminated) |
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
| **G2. Permutohedral tripwire** | RESOLVED → **retired**. Bounded support dissolves the lattice's obstacles; the bounded bilateral (`bilateral_gaussian` + factored metric / mask / `n_iters`) supersedes it | closed | SPEC_UPDATE_v0.4 §3.3; `docs/design/bounded-bilateral.md` |
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

**Exit criteria.** `nitrix.smoothing` (gaussian, bilateral_gaussian —
the bounded bilateral), `nitrix.morphology` shipped. Permutohedral
retired (SPEC_UPDATE_v0.4); no symbol ships.

### 7.1 Phase 4 tasks

#### 4.1 `nitrix.smoothing.gaussian`

Separable Gaussian. Pure JAX. The unconditional baseline. Cheap.

#### 4.2 `nitrix.smoothing.bilateral_gaussian` — the bounded bilateral

Direct N-body bilateral over arbitrary `d_f`, implemented as a `semiring_ell_matmul`
over a bounded feature-space adjacency. This is the **marquee capability**, and (per
SPEC_UPDATE_v0.4) the bounded bilateral that supersedes permutohedral. It takes a
factored feature metric (`FeatureMetric`: `DiagonalMetric` / `FactorMetric`, with
`block_diagonal_metric` / `metric_from_spd` constructors), a validity `mask` for
ragged / padded neighbourhoods (grid box, mesh k-ring, geodesic ball), and
fixed-affinity iteration (`n_iters`).

Specific tests: agreement with a reference NumPy direct-N-body implementation;
`DiagonalMetric` ≡ `FactorMetric(diag(1/σ))`; low-rank `FactorMetric` matches the
dense quadratic form; the mask zeroes padding (removing the double-count at mesh
pentagons / grid boundaries); `n_iters` matches a manual re-apply; finite-diff
gradient w.r.t. the metric factor; agreement with `gaussian` in the large-bandwidth
limit (where the intensity-similarity weighting becomes uninformative).

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

#### 4.6 ~~`nitrix.smoothing.permutohedral_lattice` — G2 gate~~ — RETIRED

**Resolved (SPEC_UPDATE_v0.4): retired, not shipped.** The G2 assessment
(`docs/design/permutohedral-g2.md`) found the lattice's pure-JAX obstacles —
dynamic-membership hash table, blur-time neighbour materialisation, simplex-identity
gradient discontinuity — to be consequences of targeting *unbounded* support in high
`d_f`. Our use cases accept bounded-hop neighbourhoods, which dissolves all three.
The bounded bilateral (§4.2: `bilateral_gaussian` with a factored metric, validity
mask, and `n_iters`) fills the role permutohedral was reserved for — for the feature
dimensionalities we target, and via a low-rank `FactorMetric` beyond — at one gather
plus one smooth weighted reduction. The symbol and its stub are removed; the
namespace is not reserved (no consumers to redirect). See
`docs/design/bounded-bilateral.md`.

### 7.2 Phase 4 exit checklist

- [ ] `gaussian`, `bilateral_gaussian` (the bounded bilateral) shipped
      unconditionally.
- [ ] Morphology shipped: erode, dilate, open, close, distance_transform via tropical
      semiring; median_filter via gather; susan_emulator composing both.
- [x] G2 resolved → permutohedral retired (SPEC_UPDATE_v0.4); bounded bilateral
      supersedes it.

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
- [ ] `bilateral_gaussian` (the bounded bilateral) unconditional: factored
      `FeatureMetric`, validity mask, `n_iters`.
- [x] `permutohedral_lattice` retired (SPEC_UPDATE_v0.4); bounded bilateral
      supersedes it.

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

### 2026-06-09 — ilex-backlog implementation sprint (Phases 1–5) + review-driven boundary

- **Type:** Downstream deviation (consumer-driven scope) + Plan revision (boundary)
- **Triggered by:** The 2026-06-02 / 2026-06-08 ilex audits
  (`docs/feature-requests/ilex-{pipeline,training}-substrate.md`); follow-up
  code review (`docs/review-ilex-backlog-impl.md`).
- **Description:** Implemented the full ilex backlog across `numerics`,
  `metrics`, `stats`, `geometry`, `morphology`, `register`, `sparse`, and a
  new `augment` subpackage. The review surfaced that the
  **score-kernel ↔ scalarisation boundary** and the **keyed-generator**
  symbol category were never specified; both are now drawn.
- **Capability shipped:** See the per-symbol table in
  **`SPEC_UPDATE_v0.5.md §10.A`** (the detailed record for this sprint).
- **Shape:** JAX-only throughout. Dense factorisations (PCA, affine
  decompose) route through `linalg._solver.safe_*` for the cuSolver-dead GPU.
- **Deferred work:** Pallas kernels; adaptive/symplectic ODE + adjoint;
  sparse-`X` PCA; 2-D affine-param decomposition (tracked).
- **Non-negotiables held:** §2.2 respected — pure functions, NamedTuple /
  frozen containers, no equinox/scipy runtime imports, jaxtyping throughout.
  The boundary (`SPEC_UPDATE_v0.5 §1`), keyed-generator clarification (§2),
  and `augment` ratification (§3) keep the additions inside the substrate's
  organising principles rather than creating a parallel structure.

### 2026-06-08 — Registration suite (rigid/affine + diffeomorphic) + three §12 graduations

- **Type:** Plan revision + §12 → §10.A graduations
- **Triggered by:** Building a JAX registration suite backed by nitrix
  ahead of the `entense` backend (one rigid-body / affine registrator in
  the `3dvolreg`/AIR lineage; one diffeomorphic in the SyN/demons family).
  Design + phased plan in `docs/design/registration.md`.
- **Description:** Shipped on branch `registration-suite` in phases
  R0–R2. **R0** (shared image substrate): `geometry.spatial_gradient`,
  `geometry.{gaussian_pyramid,downsample,upsample}`, new `nitrix.metrics`
  (`ssd`/`ncc`/`lncc`/`joint_histogram`/`mutual_information`/
  `correlation_ratio`, all FD-grad-checked), `linalg.{solve,cho_solve}`,
  shared `_internal.separable.correlate1d`. **R1** (rigid+affine):
  `linalg.matrix_exp`, `linalg.cg`, `linalg.optimize.{gauss_newton,
  levenberg_marquardt}` (matrix-free), `geometry.transform`
  (`rigid_exp`/`rigid_log`/`affine_exp`/`apply_affine`/`affine_grid`),
  recipes `register.{rigid_register,affine_register}`. **R2**
  (diffeomorphic): `numerics.fixed_point_solve`,
  `geometry.{compose_displacement,compose_velocity,invert_displacement}`,
  recipe `register.diffeomorphic_demons_register` (log-Demons).
- **§12 graduations (per the §13 protocol — registration is the named
  blocked consumer with a verified substrate composition):**
  - **§12.1 `cg`** → `linalg.krylov.cg` (matrix-free SPD; the
    wedge-resilient on-device path for the GN/LM normal equations).
  - **§12.2 `matrix_exp`** → `linalg.matrix_exp` (general matrix
    exponential for the affine generator; `matrix_log` / `matrix_polynomial`
    / `frechet_derivative` remain on the §12.2 backlog).
  - **§12.8 `fixed_point_solve`** → `numerics.fixed_point_solve`
    (Picard + implicit-VJP; backs `invert_displacement`).
- **Capability shipped:** End-to-end rigid/affine and diffeomorphic
  registration as pure functions returning `NamedTuple`s (the `reml_fit`
  precedent — distinct from the SPEC §1 "no template/atlas registration
  API" non-goal, which is about atlas *data structures*). Validated by
  synthetic-transform / synthetic-warp recovery (2-D/3-D), the
  diffeomorphism gate (no negative `det J`), and FD gradient checks.
- **Shape:** JAX-only (no Pallas kernels added; the recipes compose
  existing ops). Dev/test on an L4 whose cuSolver pool is dead, hence the
  SPD-vs-general split: general dense solves / `matrix_exp` route through
  the CPU-fallback `_solver.safe_*` wrappers, the SPD optimiser path is
  matrix-free (`cg`/BFGS) and stays on-device.
- **Deferred work:** R3 polish (implicit-diff differentiable-layer
  wrapper, benchmarks, tutorials); `matrix_log`; LNCC-driven / symmetric
  Demons; real-data parity vs AFNI `3dvolreg` / FSL `mcflirt` / ANTs SyN.
- **Non-negotiables held:** Pure-functional surface (NamedTuples / frozen
  records, no PyTree modules); JAX fallback floor; jaxtyping; ruff; mypy;
  `custom_vjp` where stability/efficiency needs it (`fixed_point_solve`'s
  implicit-VJP; `implicit_least_squares`). `matrix_exp` is pure-matmul
  Taylor scaling-and-squaring (GPU-native, no cuSolver). No new runtime
  deps; no atlas/template structures or I/O.

### 2026-05-20 — SUGAR feedback batch: edge attributes, row-softmax, mean-pool, external topology, masking

- **Type:** Downstream deviation
- **Triggered by:** ilex/SUGAR port (2026-05-18) — second
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
- **Deferred work:** Pallas dispatch for `semiring_ell_edge_aggregate` (internal-backlog B3);
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
  in internal-backlog B7 behind a two-part gate: (a) the path is a real training-loop
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
  new `residualise`) — see internal-backlog **B9**.  Per decision, constrained the
  exact-decomposition property tests to the well-conditioned domain
  (`generate_valid_arrays(well_conditioned=True)`, `p <= obs/2`) so they are
  honest and green, and parked the real numerical fix (SVD/QR projector) in the backlog.
- **Non-negotiables held:** the deflake loses **no** input coverage (only timing
  assertions dropped); the ill-conditioned limitation is documented + tracked
  (B9), not silently skipped.

### 2026-05-21 — Semiring `identity` vs `(*)`-annihilator (recorded learning)

- **Type:** Learning / future-API note
- **Description:** `Semiring.identity` is the **monoid identity**; padding / masking
  (`sparse.ell_mask`) needs the **`(*)`-annihilator**, which coincides with
  `identity` for all built-ins **except** `EUCLIDEAN` (no annihilator; `identity=0`
  does not mask).  Recorded in `docs/design/semiring-protocols.md` and internal-backlog
  **B8** (consider an explicit `annihilator` field rather than overloading
  `identity`).

### 2026-05-22 — Static-typing rigor pass: strict mypy gate + jaxtyping-native, shed `Tensor`

- **Type:** Plan revision (new gate)
- **Triggered by:** review found many `nitrix` functions un-/under-typed; aligns the
  surface to the `thrux` static-typing standard.
- **Description:** Lifted `[tool.mypy]` to the thrux bar (`disallow_untyped_defs`,
  `disallow_incomplete_defs`, `warn_unused_ignores` on top of the existing strict
  base) and added a `typecheck` nox session running `mypy src/nitrix` (now in
  `nox.options.sessions`).  The base config already existed but was never run --
  136 latent errors under the old settings, 332 under the new.  Drove
  `mypy src/nitrix` to **0 errors across 65 files**: every def annotated;
  jaxtyping-native array types throughout
  (`Float/Num/Int/Bool/Shaped/Complex[Array, '...']`); the legacy
  `Tensor = Union[jax.Array, NDArray]` alias removed (confined to
  `_internal/util.py` + one re-export).  Contracts leaned on protocols: a `TypeIs`
  guard (`graph._is_sparse`) narrows `Array | ELL | SectionedELL` in both branches
  across laplacian / connectopy; `Monoid` / `Semigroup` / `Semiring[S]` generics
  threaded through the algebra surface.  No `Any`-silencing and no new
  `# type: ignore`; type loss across untyped JAX boundaries
  (`jit` / `vmap` / `custom_vjp` / `fori_loop` / `pallas_call` / `jnp.linalg.*`) is
  restored with zero-runtime-cost `typing.cast`.  Full per-file test suite green
  (one regression caught + fixed: `cast()` to a two-variadic jaxtyping shape fails
  at runtime, since cast targets are runtime-evaluated -- use `'...'`).  Resolved
  the ruff<->jaxtyping mismatch by ignoring `F722` / `F821` (ruff reads jaxtyping
  shape strings as forward-ref annotations); mypy's `[name-defined]` remains the
  real undefined-name backstop.
- **Deferred work:** pre-existing, non-jaxtyping ruff debt remains (`I001` unsorted
  imports, `F401` dead imports, `F841`, `E702`, `E402`; ~212, 86 auto-fixable) --
  a separate cleanup, untouched here.  `typing_extensions` (for `TypeIs`) is relied
  on as a guaranteed-transitive dep via jaxtyping; declaring it directly is
  optional.
- **Non-negotiables held:** typing-only changes (no numerics / control-flow edits);
  every cluster's tests re-run green; no silent `Any` / ignore escapes.

### 2026-05-22 — ELL self-loops for graph-attention convs (`ell_add_self_loops`)

- **Type:** Downstream deviation
- **Triggered by:** ilex/SUGAR feedback (2026-05-21) --
  a GATv2 port built on the ELL-edge surface ran at plausible magnitude but
  miscomputed because the surface had no self-loop step.
- **Description:** Graph attention attends each vertex to *itself* -- the
  neighbourhood in Velickovic et al. (2018) explicitly includes node `i` -- and the
  GCN renormalisation trick (Kipf & Welling 2017) adds the self-connection
  `A_hat = A + I`.  The geometric mesh adjacency (`mesh_k_ring_adjacency`) is
  self-loop-free, so a literature-correct GAT / GCN-renorm conv must add the
  self-edge before aggregating.  Shipped `nitrix.sparse.ell_add_self_loops(ell,
  edge_attr=None, *, fill='mean'|'add'|'zero', self_value=1.0)`: appends a per-row
  `(i, i)` slot (sibling of `ell_pad` / `ell_mask`); for per-edge attributes it
  fills the self-edge from the row's **valid** (non-pad) edges -- `'mean'` (the
  natural default when no intrinsic self-feature exists), `'add'`, or `'zero'`.
  Also corrected the GATv2 worked example in `semiring/ell_edge.py`, which had
  omitted the self-loop.  Framed throughout via the literature, **not** parity with
  any particular GNN library.  Pure-JAX, jit-safe, differentiable, additive
  (mypy-clean under the new gate; 5 new tests in `tests/test_ell.py`).
- **Capability shipped:** literature-correct self-attention / renormalisation on the
  ELL-edge surface, so a GAT / GCN consumer adds one explicit call instead of
  re-vendoring the self-loop + masked-mean-fill (the bit consumers get wrong).
- **Deferred work:** an aggregation-side convenience wrapper bundling
  add-self-loops + aggregate was **deliberately not added** -- self-loops are
  architecture-specific (EdgeConv / DGCNN, MoNet, plain GCN omit them), so bundling
  would promote a non-universal default and re-create the silent-default footgun
  this finding is about.  Revisit only on demonstrated demand, or host the GAT
  composition downstream (a `nimox` ELLGAT module).
- **Non-negotiables held:** additive (no change to `semiring_ell_edge_aggregate`'s
  signature or any existing caller); no framework dependency or concern leakage
  (no `torch_geometric`, no PyG-named API); docs grounded in the literature.

### 2026-05-22 — `residualise` rank-deficient robustness verified + documented (internal-backlog B9)

- **Type:** Robustness / test-quality
- **Triggered by:** internal-backlog **B9** -- the 2026-05-21 deflake surfaced that
  `linalg.residualise` loses the exact `residual + projection == Y`
  decomposition for ill-conditioned designs; the property tests were capped to
  the well-conditioned regime to stay green.
- **Description:** Root-caused: the default `method='cholesky'` (Cholesky of
  the Gram `X Xᵀ`) returns **NaN** for rank-deficient `X` (`p > obs` / collinear
  columns) -- the singular Gram has no Cholesky factor -- while the
  already-shipped `method='svd'` path (`jnp.linalg.lstsq`) is exact there.  So
  the fix is verification + documentation, not new numerics.  (1)
  `tests/test_resid.py` now exercises the SVD path across the *full* `p -> obs`
  and `p > obs` regime (`test_residual_decomposition_svd_robust`) and pins the
  cholesky-NaN / svd-finite contract (`test_svd_robust_where_cholesky_degenerates`);
  `lstsq` was confirmed to vmap over batch dims.  (2) The `method` docstring now
  documents the **min-norm** least-squares semantics, the unique-projection
  guarantee (why the decomposition is stable even though the coefficients are
  not), the cholesky NaN failure mode, and the `lstsq(rcond=None)` cutoff
  pitfall (prefer `l2 > 0` for deterministic shrinkage of weak directions).
- **Decision:** default stays `cholesky` -- ≈2× faster on the common,
  well-conditioned case (fMRI confound regression is `obs >> p`); `svd` is the
  documented robust opt-in.  Making `svd` the default was **considered and
  rejected** on the perf-vs-common-case trade-off (a silent 2× regression for
  every caller to fix a regime with an explicit escape hatch).
- **Non-negotiables held:** no numerics change to either solve path; default
  behaviour unchanged; the well-conditioned cholesky property tests stay as the
  fast-path guard while the new svd tests cover the wide regime.

### 2026-06-02 — Non-primitive sprint: doc-drift fixes, intensity-normalize variants, `spatial_transform_batched`, B8 annihilator

- **Type:** Plan revision / consolidation (no new primitives) — work picked
  while the perf-bench suite (B11) catches up to the existing primitive bank,
  scoped deliberately to *fix / enhance / refactor existing functionality*.
- **Triggered by:** feature-request triage across `docs/feature-requests/`;
  the doc-drift findings (`perf-bench-feedback.md`) and consumer-convenience
  enhancements (`ilex-pipeline-substrate.md`) that touch shipped surfaces
  rather than adding primitives.
- **Description.** Three tiers landed:
  - **Tier 1 (doc-correctness fixes, shipped functions, no behaviour change):**
    (a) `lomb_scargle_periodogram` normalisation docstring corrected — it
    returns the classic Scargle `P_raw/var` (≡ `scipy.signal.lombscargle(
    normalize=False)/var`), **not** scipy `normalize=True` (which differs by a
    factor `N/2`); a scipy-parity regression test pins both relations.
    (b) lomb-scargle module "Memory regime" prose corrected from
    Cholesky/triangular-solve to the eigh + truncated-pseudo-inverse path the
    code ships. (c) `lomb_scargle_interpolate` gained an intended-use note
    (spectral bridge for AR/IIR filtering, **not** durable per-frame
    imputation; censored-frame values not bit-reproducible across precisions)
    and the `safe_eigh` CPU-routing caveat. (d) `tsconv` documented as
    cross-correlation (kernel not flipped). (e) `resample` / `spatial_transform`
    flagged as linear-only (order 0/1; no cubic B-spline) — the nnUNet/`hd_bet`
    parity deviation (`cubic-resample.md`).
  - **Tier 2 (enhancements to existing surfaces, no new primitive):**
    `numerics.normalize` gained `percentile_rescale` (the Synth* min–p99–clip
    recipe; distinct from `intensity_normalize` in denominator and clip-order)
    and a `nonzero_mask=` option on `zscore_normalize` (BraTS/nnUNet per-channel
    foreground z-score, background left at 0). `geometry.grid` gained
    `spatial_transform_batched` (single-leading-axis `vmap` that broadcasts a
    shared image or deformation — the case `spatial_transform` deliberately
    refuses). Closes `intensity-normalize-variants.md` and
    `spatial-transform-batched.md` (JOSA §3).
  - **Tier 3 (API refinement — B8):** `Semiring` gained an explicit
    `annihilator` field (`None` for EUCLIDEAN; `= identity` for the other
    built-ins), and `ell_mask` gained a `semiring=` path that reads
    `semiring.annihilator` and raises clearly when `None` (EUCLIDEAN), instead
    of overloading `identity`. The legacy `ell_mask(identity=...)` form is
    retained but emits a `DeprecationWarning` (user-requested migration nudge).
    `sparse.ell` stays free of a *runtime* `nitrix.semiring` import (the
    `Semiring` annotation is `TYPE_CHECKING`-only; the field is read
    duck-typed). Resolves internal-backlog **B8**; the masking footgun in
    `docs/design/semiring-protocols.md` is now machine-checked. Tier-3 items
    **B4** (LOG/EUCLIDEAN `edge_aggregate`) and **G1**
    (`spatial_transform(mode='linear_extrap')`) were **considered and held**
    per their own §13/Trigger discipline (no concrete consumer yet).
- **Capability shipped:** correct docs on the lomb-scargle / tsconv / resample
  surfaces; the two missing intensity-normalisation recipes; a batched-warp
  convenience; and a machine-checked, annihilator-aware masking path.
- **Shape:** Pure-JAX; differentiable where applicable; CPU correctness floor
  exercised (`JAX_PLATFORMS=cpu`). New/updated tests in `test_signal_interpolate`,
  `test_numerics`, `test_geometry`, `test_ell_masking_semirings`.
- **Verification:** full suite **627 passed / 26 skipped / 1 pre-existing
  failure** (`test_lme.py::test_reml_voxelwise_per_voxel_match_unbatched`, a
  float32 batched-vs-unbatched tolerance miss confirmed identical on a clean
  checkout — environment drift, not this work). `mypy src/nitrix` adds **zero**
  net-new errors over the pre-existing 8-error drift baseline (newer mypy/jax
  resolved without a `uv.lock`; `redundant-cast`/`unused-ignore` in untouched
  files). `ruff check` adds zero net-new per-file errors.
- **Non-negotiables held:** no new runtime deps (the `Semiring` annotation in
  `sparse.ell` is `TYPE_CHECKING`-only, preserving the array-only sparse layer);
  pure-functional surfaces; golden/parity/regression tests added; the deprecated
  `identity=` path stays functional (no silent break).

### 2026-06-02 — Op-matrix completeness: catalogue all public ops + anti-rot guard

- **Type:** Tooling / coverage completion.
- **Triggered by:** `docs/feature-requests/doc-op-matrix-inventory-gaps.md` — the
  perf-bench coverage join is keyed on the op-matrix inventory, and a full
  `__all__`-vs-catalogue diff found the matrix covered only **59 of ~163**
  public functions (the request named 3; the real deficit was ~64%).
- **Description.** Expanded `tools/op_matrix.py` from 59 to **137 cataloged
  ops** (78 new `OpInfo` entries, core ops only). Scope agreed with the
  requester: catalogue membership signals perf-bench to add a baseline, so the
  **EXCLUDE allowlist** (26 callables) deliberately omits legacy aliases (7
  geometry + 4 stats `"Alias for X"` wrappers), `reference_*` impls, `*_matvec`
  closures, shape/layout helpers, metric constructors, and the
  `spatial_transform_batched` thin wrapper; the 20 type/container/constant
  symbols are auto-dropped by a "callable & not a class" filter. Container-arg
  ops (ELL/Mesh — not registered pytrees) are probed through their
  differentiable array via `fn_override` closures; host-side constructors are
  `skip_jit`. Added a **completeness-guard test**
  (`tests/test_op_matrix_completeness.py`) that fails CI if any public op is
  neither cataloged nor on the EXCLUDE allowlist — the durable fix so the
  inventory cannot silently rot again. Regenerated `docs/op_matrix.{json,md}`
  (CPU host snapshot).
- **Honest red cells (not forced green):** `bias.n4_bias_field_correction` /
  `bias_field_correction` fail `grad` / `jit(grad)` — the iterative N4 fit uses
  `lax.while_loop` (no reverse-mode rule), exactly as the request flagged; the
  *intended* status is forward-only and is now recorded. Final matrix: jit
  122/137 (15 `skip_jit` constructors are n/a), grad 119/121, vmap 94/94.
- **Incidental fixes surfaced by the pass:** (a) `signal.tsconv` was imported
  into `nitrix.signal` but missing from its `__all__` — added. (b) the cataloged
  `smoothing.bilateral_gaussian` fixture still used the pre-v0.4 `sigma_features`
  kwarg (removed when the bounded bilateral took a factored `metric`) — updated
  to `DiagonalMetric`. (c) `linalg.squareform`'s square→vec direction branches
  on a `jnp.allclose` symmetry check (not jit-safe); the jit-safe vec→square
  direction is probed with a note (the docstring already steers callers to
  `sym2vec`/`vec2sym`).
- **Verification:** completeness guard green (3 tests); generator runs clean (0
  fixture failures); `signal/__init__.py` ruff-clean. `tools/` and `tests/` are
  outside the `mypy src/nitrix` gate.
- **Non-negotiables held:** no new runtime deps; capability-only matrix (perf
  stays in nitrix-perf-bench); failures recorded honestly, not suppressed.

### 2026-06-07 — Resampling interpolation-method dispatcher (Lanczos / MultiLabel / NN)

- **Type:** Primitive extension (additive kwarg; SPEC_UPDATE_v0.3 §14
  "prefer adding a kwarg over forking a function" — not a §12 → §10.A
  graduation, not a new subpackage).
- **Triggered by:** standard neuroimaging-workflow interpolation strategies
  absent from the linear-only default (segmentation warping, high-fidelity
  resampling); the consumer-side asks tracked in `upsample-nearest-nd.md`,
  `point-sample.md`, and the ANTs-output / `hd_bet` survey context.
- **Description:** `geometry.resample` became the **common dispatcher** over an
  `Interpolator` ADT (new `geometry/_interpolate.py`); `spatial_transform` /
  `spatial_transform_batched` share the same sampler via a `method=` kwarg
  (default `Linear()`, behaviour-preserving). Three orthogonal axes —
  coords(task) ⟂ kernel(method) ⟂ backend(execution) — mirroring the
  `linalg._eigsolve` factoring. Four methods: `Linear` (default),
  `NearestNeighbour` (order-0; value-differentiable, coord-flat), `Lanczos`
  (order-3 default windowed sinc; separable, partition-of-unity-normalised,
  fully differentiable; ANTs `LanczosWindowedSinc` *class*, not bit-exact
  ITK), `MultiLabel` (per-label indicator resample + arg-max over a **static**
  label set; non-differentiable; anti-aliased label resampling). The records
  are frozen, hashable, **static** config (no array leaves → not pytrees;
  `SolverSpec`-style), with differentiability exposed as record properties.
- **Engine / B7:** the gather kernels lower onto one explicit separable gather
  (the B7 "8-corner gather"), with all five `map_coordinates` boundary modes
  reproduced at the integer-tap level (parity to ~1e-15). The B7 perf win is
  **GPU-only** (measured A10G ~1.5–1.7×; *inverts* to ~1.3× slower on CPU,
  where `map_coordinates` is tighter and CPU interpolation is the B15
  throughput-sensitive path), so `Linear`/`NearestNeighbour` select the engine
  **per platform** (`default_backend_is_gpu`, the `signal._iir` precedent);
  `map_coordinates` is retained as the CPU engine and the parity oracle.
  `resample` resamples the separable grid axis-by-axis for wide kernels
  (`Lanczos`: O(N·T·ndim) vs the dense O(N·T**ndim) = 6³ corners), gated by a
  `prefers_separable_resample` flag so low-tap kernels keep the CPU
  `map_coordinates` win. A Pallas pointer-load gather stays reserved behind the
  `backend` axis (gated on the B7 trigger); the external scipy/cupyx backend
  track (B15/B16) is explicitly out of scope.
- **Verification:** behaviour-preserving (all prior geometry tests unchanged —
  no golden re-pin needed, parity ≤ 1e-15); +~30 dispatcher tests
  (per-mode gather-vs-`map_coordinates` parity, platform-engine agreement,
  Lanczos fidelity / partition-of-unity / hand-computed-weight oracle,
  MultiLabel label-set / anti-aliasing / non-differentiability); `mypy
  src/nitrix/geometry` clean; op-matrix regenerated with the four method
  variants (all jit/grad/vmap green; differentiability nuance in the notes).
- **Non-negotiables held:** pure-functional immutable records + protocols;
  no new runtime deps; differentiability recorded honestly (non-diff paths
  return zeros, don't error); capability-only matrix (perf stays in
  nitrix-perf-bench).
- **Design doc:** `docs/design/geometry.md` ("The resampling-method
  dispatcher").

### 2026-06-08 — `CubicBSpline` interpolator (order-3 spline; closes `cubic-resample`)

- **Type:** Primitive extension (one more `Interpolator` record; the
  dispatcher's open extension point). Resolves the
  `cubic-resample.md` parity gap.
- **Triggered by:** order-3 B-spline preprocessing (nnUNet / `hd_bet`
  style) used in some ilex pipelines.
- **Description:** `CubicBSpline` is a two-step B-spline interpolator, not
  a plain cubic convolution: (1) a recursive **prefilter** (cubic pole
  `√3 − 2`) converts samples to interpolating coefficients, separably per
  spatial axis; (2) a 4-tap cubic-basis gather (partition of unity) of the
  coefficients. Differentiable in values and coordinates (prefilter and
  gather are both smooth/linear). **Bit-exact** with
  `scipy.ndimage.map_coordinates(order=3, mode='mirror')` — interior and
  boundary, ~1e-15. To stay self-consistent (the interpolation property
  only holds when prefilter and gather share a boundary, and only the
  mirror init is implemented) it **forces the mirror boundary and ignores
  `mode` / `cval`** (the `MultiLabel`-ignores-`cval` precedent) -- but, per
  the loud-fallbacks tenet, **announces** the override with a
  `CubicBSplineBoundaryWarning` when an explicit non-mirror `mode` /
  non-zero `cval` is supplied (the bare default stays silent).  A
  mode-aware prefilter for `nearest`/`reflect` parity is the one remaining
  follow-up (`boundary-mode-parity.md`).
- **Engine:** the prefilter is a first-order linear recurrence → sequential
  `lax.scan` on CPU, parallel `lax.associative_scan` (O(log N) depth) on
  GPU, the `signal._iir` platform split; both exact, agree to a ULP.
  `_iir`'s *FFT* engine is deliberately not reused — the mild pole makes
  the prefilter a ~25-tap short FIR, too short for FFT overhead to amortise
  (associative_scan is the right parallel form for a recurrence). Reuses
  the existing `_separable_gather` / `_separable_resample` machinery via a
  `_prepare` hook on the separable-kernel protocol (identity for the direct
  kernels, prefilter for the spline).
- **Verification:** scipy `order=3` parity in 1-/2-/3-D (interior +
  boundary); grid-point identity (interpolation property); constant
  preservation; resample-separable == scattered-gather; CPU-scan ==
  GPU-associative-scan (forced-branch); `mode`-inert; differentiability;
  jit/vmap. `mypy src/nitrix/geometry` clean; op-matrix `resample[cubic]`
  variant added.
- **Design doc:** `docs/design/geometry.md` ("CubicBSpline: the prefilter,
  and why not FFT").

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
