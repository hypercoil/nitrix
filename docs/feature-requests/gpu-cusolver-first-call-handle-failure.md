# GPU availability: cuSOLVER `gpusolverDnCreate` fails for a Cholesky/eigh-first program on the L4 (`flame_two_level` skips) — cause unknown

> **Status (2026-06-12): GPU-availability finding — OBSERVATIONAL ONLY.** On the
> benchmark L4 (jax 0.10.0), `stats.lme.flame_two_level` skips on GPU with a
> cuSOLVER handle-creation error. We characterised the *behaviour* fairly
> thoroughly but **did not establish a cause** — read every statement below as
> "what we observed on this one machine," not a mechanism. A candidate one-line
> workaround (a cuBLAS warm-up) removed the symptom in every trial we ran, but
> **it must be proven robust by repeated measurement before it is relied on**
> (see *Verification demands* — the most important section here).

## What we observe

`flame_two_level(...)` on GPU raises, then the harness records a graceful
`gpu_solver_unavailable` skip:

```
jaxlib/gpu/solver_handle_pool.cc:37:
  operation gpusolverDnCreate(&handle) failed: cuSolver internal error
```

Every observation below is a *fresh process* (the perf-bench runner is one
subprocess per attempt), `x64`, single L4, jax 0.10.0. No stack trace precedes
the error; stderr shows nothing else.

**It is order-dependent across cuSOLVER routines.** Run each as the *first*
cuSOLVER routine in a fresh process:

| first cuSOLVER routine in the process | result |
|---|---|
| `potrf` (Cholesky) — bare `vmap(cholesky)`; every V incl. `V=4`; shapes 1×1…4×4; ones / random-SPD / diag content | **FAIL** |
| `syevd` (eigh) — bare `vmap(eigh)` | **FAIL** |
| `getrf` (LU) — bare `vmap(solve)` | **OK** |

The failure is independent of matrix **size, shape, and content** (it fails
before touching data) and of **batch size / memory** (fails at `V=4`).

**It is removed by some prior GPU work, but not all.** Insert a warm-up before
the first `potrf`:

| warm-up before the first `potrf` | `potrf` then |
|---|---|
| a `getrf` (a `jnp.linalg.solve`) | **OK** |
| a matmul (cuBLAS) — even `2×2`, synced or not | **OK** |
| `reml_fit` (which internally runs a `2×2` `getrf`) | **OK** |
| an elementwise op (`sin`) + `block_until_ready` | **FAIL** |
| a `device_put` + `block_until_ready` | **FAIL** |
| nothing (`potrf` is first) | **FAIL** |
| `CUDA_LAUNCH_BLOCKING=1`, no warm-up | **FAIL** |

Once any one cuSOLVER handle is created in the process it is pooled and every
routine reuses it (a bare `cholesky` that fails in a fresh process **succeeds**
if a `getrf` — or a `reml_fit`, or a matmul — ran earlier in the same process).
Synchronisation is not the factor (an *un-synced* matmul also clears it);
forcing synchronous launches (`CUDA_LAUNCH_BLOCKING=1`) does not help.

**Whole-op consequence.** `flame_two_level`'s only cuSOLVER routine is `potrf`,
and its compiled program apparently dispatches no cuBLAS gemm before it → it is
"first-`potrf`" → it skips at every V. `reml_fit` runs a `2×2`
`jnp.linalg.solve` (`getrf`) in its Newton step, so a handle is created before
its own `potrf`/`syevd` → `reml_fit` runs `ok` on GPU (its `ok` store rows are
correct). Confirmed end-to-end: a `2×2` matmul inserted before
`flame_two_level` makes the **real op** run `ok` on GPU (`V=8192`, finite
output).

## What we do NOT know

We did not find the cause. The table above is correlation between "what ran
first" and "did the handle create." Plausible but **unverified** stories
include CUDA context / library init order, allocator state, cuBLAS-before-
cuSOLVER load ordering, or a driver/runtime/jaxlib-version interaction specific
to this L4 image. We have **not** bisected jax/jaxlib/CUDA versions, tried other
GPUs, or inspected the cuSOLVER internals. A different GPU or library build may
not show this at all — so "`flame_two_level` is GPU-blocked" is partly a
property of *this environment*, not solely of the code. Please do not restate
the observations above as a mechanism in downstream notes.

## Candidate workaround (provisional)

A single cuBLAS matmul early in the process (at first `flame_two_level` call, or
at module import) removed the symptom in every trial, including on the real op.
It is one line and changes no math. **It is a symptom-suppressor for an effect
we don't understand — not a fix.** Do not ship it on the strength of these
observations alone.

## Verification demands (the important part)

This is an opaque setting; a change that "works in my trials" can be fragile to
process state, device, driver, concurrency, or jax version. Any change here —
the warm-up, or anything else touching this path — must be shown **robust by
repeated measurement**, never reasoned to be correct:

- **Repeat at scale.** Run ≥50 fresh processes, not one. An intermittent
  init failure passes a single run and fails in CI. Require a *stable* pass rate
  across the whole matrix before calling it resolved; treat one green run as
  **no evidence**.
- **Vary conditions.** V from 4 to brain-scale; cold and warmed GPU; with and
  without concurrent GPU work; through the full perf-bench subprocess path, not
  just a REPL.
- **Test the real ops end-to-end** (`flame_two_level`, `reml_fit`), not only
  bare `cholesky`.
- **Verify correctness, not just "it ran."** A handle that creates is not proof
  the op is right — diff the compiled HLO `custom_call_target`s before/after,
  and confirm the GPU result is finite and matches the CPU/oracle output.
- If the workaround is adopted, **guard it** so a future refactor can't silently
  drop the warm-up and regress GPU availability, and add a test that would catch
  the regression.

## Relationship to the perf rewrite

Companion FR
[`lme-family-tiny-linalg-gpu-block-and-perf`](lme-family-tiny-linalg-gpu-block-and-perf.md)
proposes replacing the tiny per-voxel Cholesky with closed-form algebra for
CPU-steady/compile wins. Because that path makes **no cuSOLVER call at all**, it
also sidesteps *this* bug for the `p ∈ {1, 2}` hot path — the most robust
resolution, since it removes the dependency instead of working around an effect
we can't explain. The two are separable: a *validated* warm-up restores GPU
availability for the existing code today; the rewrite removes the cuSOLVER
dependency and also wins on CPU. Fix-risk noted there: on this machine,
stripping `reml_fit`'s `getrf` (its `2×2` solve) while leaving its `syevd`
(eigh) in place would reintroduce this skip.
