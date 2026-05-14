# nitrix — Specification update (v0.1 → v0.2)

> **Status.** Hardware-scope addendum to SPEC_UPDATE.md (v0.1). Apply on top of v0.1.
> All v0.1 changes stand; this patch adds hardware-scope constraints and removes TPU
> surface that v0.1 inherited from v0.

---

## §1.1 Hardware scope — NEW section, insert after §1 Charter

nitrix targets **NVIDIA GPUs, Ampere generation and newer**, with the JAX-on-CPU
fallback supported for tests, small workloads, and CI without GPU runners. This scope
is set by realistic dev / test access and by the target user population — academic and
research labs, where Ampere (A100, A40, RTX 30xx) and Lovelace (RTX 40xx, L40) are the
dominant hardware, with growing Hopper (H100, H200) and emerging Blackwell (B100, B200,
RTX 50xx) deployments in well-funded clusters.

### In scope

- **NVIDIA Ampere, Lovelace, Hopper, Blackwell** via the Pallas Triton backend.
  Blackwell support tracks Pallas/Triton support; we do not maintain a separate
  Blackwell code path.
- **CPU** via the pure JAX fallback. Functionally correct, performance not a goal:
  CPU exists for tests, CI, and small-scale exploratory use.

### Explicitly out of scope at first GA

- **TPU.** No dev access; no Mosaic / `pallas-tpu` kernels written or tested. The
  `Backend` enum does not include `pallas-tpu`. Adding TPU support is a 1.x question
  contingent on dev environment access; no architectural blocker exists, since the
  KeOps-style streaming kernel and the Monoid / Semigroup Protocol shape are
  hardware-agnostic.
- **AMD ROCm, Apple Metal, Intel GPUs.** Out of scope; users on these platforms get
  the JAX-CPU fallback. Revisit only if Pallas adds first-class support.
- **Mosaic GPU backend** (the Hopper+-only Pallas lowering path). Targeting Hopper+
  only would cut out the majority of academic-lab hardware. Triton covers Ampere+ at
  the cost of being maintained "on a best-effort basis" by the JAX team — see §7.2
  Pallas API churn policy.

### Known acceptance: Triton best-effort maintenance

The Pallas Triton backend is, per JAX documentation, maintained on a best-effort basis
and is not the primary recommendation for new Pallas users (Mosaic GPU is). We accept
this risk because the alternative (Hopper-only) is incompatible with the target user
base. Mitigation: the JAX fallback path is the contractual floor; every kernel
preserves correctness under JAX-only execution, and the §7.2 fallback machinery covers
Triton regressions without user-visible breakage beyond a perf warning.

---

## §2 Design tenets — replace tenet 3

3. **JAX + Pallas-Triton on NVIDIA, with JAX fallback.** Hardware-aware Pallas kernels
   target the Pallas Triton backend on NVIDIA Ampere and newer; pure JAX fallbacks
   always present and exercised in CI. Backend selection is deterministic and
   user-overridable. CPU execution is functionally supported via the JAX fallback;
   performance is not optimised for CPU.

(All other tenets, including the v0.1 additions for loud fallbacks and reproducibility
via golden corpus, unchanged.)

---

## §3.1 `nitrix.semiring` — kernel strategy, replace the "Backends" subsection

#### Backends

- **`pallas-cuda`** — default on NVIDIA Ampere+. Pallas Triton backend. Kernel files
  live in `_kernels/cuda/`.
- **`jax`** — pure-JAX fallback built on `lax.fori_loop` + the `reference_semiring_gemm`-style
  algebra plumbing. Exercised in CI on CPU runners; correctness floor for the library.
  Used automatically when no NVIDIA GPU is present, when the requested shape × algebra
  combination cannot be tiled by the Triton kernel, or when the user explicitly
  requests `backend="jax"`.

The `Backend` literal is `Literal["auto", "pallas-cuda", "jax"]`. No `pallas-tpu`.

(The "No tensor-core / `dot` primitive" and "KeOps-style streaming" and "Pytree
accumulator" subsections of §3.1 kernel strategy are unchanged from v0.1.)

---

## §3.3 `nitrix.smoothing` — permutohedral tripwire, replace criterion 2

The v0.1 tripwire's criterion (2) referenced "the same hardware" without pinning it.
Pin to the reference NVIDIA configuration:

2. **Performance.** End-to-end smoothing of a 256³ volume with `d_f = 5` features
   completes in < 10× the wall time of an equivalent-σ separable `gaussian` on the
   reference NVIDIA configuration (A100 80 GB at first GA; revisit when Hopper/Blackwell
   become the lab baseline). No TPU criterion.

(Criteria 1, 3, 4 unchanged. The fallback-to-`bilateral_gaussian` resolution rule
unchanged.)

---

## §7.2 Backend selection — replace in full

Three-level resolution: explicit `backend=` keyword → env var (`NITRIX_BACKEND`) →
auto-detect from `jax.default_backend()`. Auto-detect resolves as:

- NVIDIA GPU present and Ampere or newer → `pallas-cuda`.
- Otherwise (CPU; NVIDIA pre-Ampere; non-NVIDIA accelerator) → `jax`.

The compute-capability check happens once at library import. Explicit
`backend="pallas-cuda"` on incompatible hardware raises (unless
`NITRIX_STRICT_BACKEND=0`, default; with `NITRIX_STRICT_BACKEND=1` it would have
errored at auto-detect too).

**Fallback observability.** When the resolved backend cannot satisfy a call (Triton
tiling fails for the given shape × algebra; requested algebra is not associative and
the kernel needs strict; etc.), nitrix falls back to `jax`, emits a structured warning
via `warnings.warn` with category `NitrixBackendFallback`, and proceeds. Warnings are
deduplicated per `(function, shape-signature, dtype, backend)` per process.

- `NITRIX_SILENCE_FALLBACK=1` suppresses the warnings.
- `NITRIX_STRICT_BACKEND=1` converts fallback to error (useful in CI; off by default).

**Pallas Triton churn policy.** The Pallas Triton backend is maintained best-effort by
the JAX team and is not the primary Pallas target. nitrix pins a minimum `jax` version
per release and exercises the full kernel surface against that pin in CI. When a
Triton-side change breaks a kernel between pin updates: (a) the kernel falls back to
JAX with the standard warning, (b) a release-blocking issue is filed, (c) no kernel is
silently disabled. The "stable kernel output" tenet (§2.6) holds via the JAX fallback
path during such windows. Releases will not unpin `jax` while a known Triton regression
is in flight.

This explicit dependence on a best-effort backend is a known risk; see §1.1.

---

## §4 Foundational primitives — performance footnote

Add to §4 preamble:

The §3.1 claim that one streaming kernel substrate covers matmul, convolution,
distance, graph algebra, and morphology depends on the Triton backend performing well
on the gather-heavy ELL access pattern over the supported NVIDIA architectures.
Ampere's gather throughput is the conservative case (older than Lovelace/Hopper,
shipping in academic clusters today); Ampere benchmarks for `semiring_ell_matmul` are
the GA performance baseline. If Triton's gather lowering on Ampere underperforms a
hand-rolled `jnp.take_along_axis` + reduction by more than 2×, the kernel ships with
the JAX path as default on Ampere and the Triton path as opt-in, pending kernel work.

---

## §8 Testing & validation — replace the backend-parity bullet

The v0 / v0.1 spec said "same op via `pallas-cuda` and `jax` fallback must agree to
pinned tolerance." Replace with:

- **Backend-parity tests.** Same op via `pallas-cuda` and `jax` must agree to the
  pinned tolerance matrix per `(dtype, op)` cell. CI runs both paths on every PR; the
  `pallas-cuda` path requires a GPU runner (Ampere in first-GA CI; expanding to
  Lovelace and Hopper as runner availability allows). PRs that cannot run GPU CI
  (community contributors without runner access) are tagged for maintainer-side GPU
  verification before merge.
- **CPU correctness floor.** Every kernel passes its golden-corpus test under
  `JAX_PLATFORMS=cpu`. CPU performance is not a CI gate.
- **No TPU tests.** The test matrix has two backend axes (`pallas-cuda`, `jax`), not
  three.

---

## §10 Success criteria — additions

Add to the existing list:

- Backend-fallback warning fires correctly on a forced shape × algebra combination
  that Triton cannot tile on Ampere; does not fire on the happy path. (This was in
  v0.1; the new addendum is:)
- CI runs the full backend-parity test suite on at least one Ampere GPU runner. Hopper
  and Blackwell coverage is a 1.x target as runners come online.
- `Backend` literal type is `Literal["auto", "pallas-cuda", "jax"]` and documentation
  explicitly enumerates supported NVIDIA generations (Ampere, Lovelace, Hopper,
  Blackwell). Pre-Ampere NVIDIA falls back to JAX with a warning at import time.

---

## §6 Migration map — no changes from v0.1

(TPU-related migration items did not exist; no rows to remove.)

---

## §9 Open questions — additions

Add under "Blocking 1.0 but not 0.1":

- **TPU support.** Architecturally compatible (the streaming kernel and Protocol shape
  are hardware-agnostic), but requires dev/test access we don't currently have. Revisit
  if access becomes available; otherwise the namespace remains NVIDIA-only at 1.0 with
  clear documentation.

Add under "Blocking implementation":

- **Ampere ELL performance baseline.** Benchmark `semiring_ell_matmul` Triton vs.
  `jnp.take_along_axis`-based JAX on A100 before committing to Triton as default for
  ELL ops. If the gap is < 2×, ship JAX-default with Triton opt-in (per §4 footnote).

---

## Notes for implementation planning

- The hardware scope is narrow enough that the single most likely portability surprise
  is a Triton regression affecting Ampere but not Hopper, or vice versa. The fallback
  machinery (§7.2) and the CPU correctness floor (§8) are what protect us; both need
  to be in place before any kernel ships.
- Blackwell is in the supported list because Triton supports it, not because we have a
  Blackwell code path. If Triton introduces Blackwell-specific intrinsics we want to
  use, that's a 1.x conversation.
- The §4 Ampere-ELL-benchmark gate is the load-bearing test for whether the v0.1
  "single performant kernel substrate" claim survives contact with real hardware. It
  needs to happen early in implementation, not late.
