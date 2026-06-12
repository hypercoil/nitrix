# LME family: the per-voxel 1×1 Cholesky blocks the GPU and inflates CPU + compile (`flame_two_level` + `reml_fit`)

> **Status (2026-06-12): performance + GPU-availability finding (with a silent
> regression angle).** Surfaced wiring **FSL FLAME** (`flameo`) as the fair
> external baseline for `stats.lme.flame_two_level` in `nitrix-perf-bench`. Both
> LME ops `skip` on GPU with a cuSOLVER error, and CPU compile scales linearly
> in the voxel count `V`. A one-line-of-reasoning fix — don't call
> `jnp.linalg.cholesky` on the tiny `(p, p)` fixed-effect system inside the
> per-voxel `vmap` — **unblocks the GPU, gives 3–6× CPU steady, and flattens
> compile** (prototype measured on an L4 below). Applies to **both**
> `flame_two_level` and `reml_fit`; `reml_fit` has the same root cause plus
> three further tiny-linalg liabilities.

## Symptom

Three coupled symptoms, all traced to the same place:

1. **GPU-blocked.** `flame_two_level` and `reml_fit` both `skip` on GPU (the
   harness records `gpu_solver_unavailable`). The whole point of these ops is a
   *batched, device-resident* fit over `V` voxels — and it never runs on the
   device. (`reml_fit`'s case currently has stale `ok` GPU rows in the
   perf-bench store; it fails live now — a GPU regression that went unnoticed
   precisely because the cuSOLVER path fails silently into a CPU-only skip.)
2. **Compile scales linearly in `V`** (XLA:CPU). `flame_two_level`:
   `compile = 1.5 s @ V=1024 → 10.2 s @ 65536 → 21.1 s @ 131072` (≈ linear at
   the top), where a clean `vmap` over a batch axis should compile in ~constant
   time.
3. **Execution is super-linear and OOMs at scale.** Steady grows slightly
   faster than `V` (cache/bandwidth), and at `V=262144`, `N=60` on a 15 GB box
   the fit **times out** (the batch's autodiff intermediates exceed memory).

## Root cause

The per-voxel fit (`_flame_fit_one_voxel` / `_reml_fit_diagonal_one_voxel`),
`vmap`'d over `V`, solves the profiled fixed-effect system with a Cholesky:

```python
# flame.py:109-115  (also 224-230;  reml.py:181-190, 315-321)
XtVinvX = Xw.T @ X_group               # (p, p)
XtVinvX = XtVinvX + ridge * jnp.eye(p)
XtVinvy = Xw.T @ beta                  # (p,)
L = jnp.linalg.cholesky(XtVinvX)       # <-- cuSOLVER potrf on GPU
z = jsla.solve_triangular(L, XtVinvy, lower=True)
gamma = jsla.solve_triangular(L.T, z, lower=False)
log_det_XtVinvX = 2.0 * jnp.sum(jnp.log(jnp.diagonal(L)))
```

The dominant design (the FLAME one-sample group mean, the benchmark, and most
real use) is the **intercept**, `X = ones(N, 1)` ⇒ **`p = 1`**. So this is a
**1×1 Cholesky** — mathematically a `sqrt` — dressed as a batched LAPACK /
cuSOLVER custom-call.

- **GPU:** `jnp.linalg.cholesky` lowers to cuSOLVER `potrfBatched`, whose handle
  creation (`gpusolverDnCreate`) fails on this L4 → the whole op skips. (A 1×1
  `sqrt` would not touch cuSOLVER.)
- **CPU compile / runtime:** the score and Hessian are taken by **second-order
  autodiff** (`value_and_grad` + `grad(grad)` in flame; `jax.grad` +
  `jax.hessian` in reml) *through* the Cholesky, re-traced across the
  20–30-iteration Newton `scan` **and** a 4-step backtracking sub-`scan`
  (~7 `nll` evaluations per Newton step). Differentiating through `cholesky`
  pulls in its backward rule (triangular solves + symmetrisation), nested for
  the second derivative — a large batched-linalg+autodiff graph. Its XLA:CPU
  lowering acquires a term that scales with `V` (compile), and it materialises
  many per-voxel intermediates (the memory super-linearity / OOM).

## Repros (all validated on the L4, 2026-06-12)

**(a) The Cholesky is the GPU blocker.** `jnp.linalg.cholesky` in a `vmap` over
`(V, 1, 1)` fails at *every* `V`; the scalar equivalent runs:

```python
jax.vmap(jnp.linalg.cholesky)(ones((V,1,1)))   # FAIL: gpusolverDnCreate  (V=256..8192)
jnp.sqrt(ones((V,1,1)))                          # OK
flame_two_level(...)  # V=1024 GPU -> FAIL gpusolverDnCreate
reml_fit(...)         # V=1024,8192 GPU -> FAIL gpusolverDnCreate (store rows are STALE)
```

**(b) The linear compile is the combined graph, not `vmap` or `cholesky`
alone.** Compile time (cold, XLA:CPU) across `V`:

| V | trivial `vmap` | bare `cholesky`-`vmap` | `flame_two_level` |
|---|---|---|---|
| 8192 | 0.09 s | 0.59 s | 1.76 s |
| 65536 | 0.05 s | 0.04 s | 10.25 s |
| 131072 | 0.07 s | 0.04 s | 21.12 s |

Neither a plain `vmap` nor a bare batched Cholesky compiles linearly in `V` —
only the full second-order-autodiff-through-Cholesky graph does.

## Fix

Replace the tiny `(p, p)` Cholesky+`solve_triangular` with **explicit
closed-form / unrolled small-matrix algebra** (no cuSOLVER/LAPACK custom-call).
For `p = 1` it is scalar; for the small `p` these ops ever use, an unrolled
solve (or a hand-written `p×p` Cholesky) with the closed-form log-det:

```python
# p = 1 specialisation (the FLAME / one-sample group-mean case):
x       = X_group[:, 0]
XtVinvX = jnp.sum(x * x * inv_d) + ridge
XtVinvy = jnp.sum(x * beta * inv_d)
gamma   = XtVinvy / XtVinvX
log_det_XtVinvX = jnp.log(XtVinvX)
```

### Dispatch on `p`: specialise the hot path, keep a capable fallback

The fix is **not** "delete the Cholesky and assume `p = 1`" — it is a
**size-dispatch** on the (statically known) fixed-effect width `p`:

- **`p ∈ {1, 2}` — the ~80 % hot path.** The dominant designs (FLAME one-sample
  group mean ⇒ `p = 1`; a mean + single covariate / two-group contrast ⇒
  `p = 2`) get specific closed-form dispatches: the scalar form above for
  `p = 1`, and the explicit symmetric `2×2` for `p = 2`
  (`det = a·c − b²`; `inv = [[c, −b], [−b, a]] / det`; `log_det = log(det)`).
  These are pure elementwise algebra — **no cuSOLVER/LAPACK custom-call** — so
  they are exactly the branch that unblocks the GPU and flattens compile, and
  they cover the vast majority of real calls (and the entire benchmark).
- **`p > 2` — the ~20 % branch.** General fixed-effect designs still need a real
  factorisation, and that path **must fall back to a more capable engine** —
  *not* the same cuSOLVER `potrf` custom-call that silently fails the handle
  creation here. Route it to a backend that actually runs and is numerically
  robust for arbitrary `p`: a CPU-side LAPACK Cholesky/LU fallback, or a more
  robust symmetric factorisation (`LDLᵀ` / `eigh`) on a path that does not hit
  the broken `gpusolverDnCreate` handle. The point is that the minority of
  general-`p` callers keep both **correctness** and **availability**, instead of
  inheriting today's blanket skip.

Statically dispatching on `p` (it is a compile-time shape, so the branch is a
Python-level `if`, not a traced `lax.cond`) means the common case never
constructs the heavy graph at all, and the general case degrades to a working
engine rather than the failing one.

### Prototype result (flame, Cholesky → scalar; Newton/autodiff structure unchanged)

Measured on the L4 (`flame_two_level` vs the prototype; accuracy vs the
closed-form REML oracle):

| V | | CPU compile | CPU steady | GPU |
|---|---|---|---|---|
| 1024 | orig | 1.50 s | 98 ms | **BLOCKED** |
| | **fast** | **0.41 s** | **32 ms** | **1.4 ms** |
| 8192 | orig | 1.56 s | 934 ms | **BLOCKED** |
| | **fast** | **0.55 s** | **153 ms** | **2.7 ms** |
| 65536 | orig | 10.22 s | 9510 ms | **BLOCKED** |
| | **fast** | **2.11 s** | **1713 ms** | **13.6 ms** |

Accuracy is **identical** (max abs error vs oracle 1.9e-4 for both). Removing
just the 1×1 Cholesky:

- **Unblocks the GPU** — the batched fit runs on-device (1.4–13.6 ms), where it
  is **sublinear** in `V` (1.4 → 2.7 ms for 8× the voxels): the compounding
  device win these ops were designed for. At `V=65536` the GPU fit is **~700×**
  the original CPU path (13.6 ms vs 9.5 s).
- **3–6× faster CPU steady**, **3–5× faster + flatter compile**.

This single change turns `flame_two_level` from a GPU-blocked op with a modest,
non-compounding ~2× CPU edge over the looped tools (FSL `flameo` /
`statsmodels`) into a device-resident batched fit that is multiplicatively
faster — i.e. it is *why* the op currently looks like a weak GPU story.

## `reml_fit`: same fix, plus three more tiny-linalg liabilities

`reml_fit` shares the Cholesky (root cause above) and adds:

1. **`jax.hessian` of the `K=2` nll** (`reml.py:231`) — a *full* second-order
   autodiff, even heavier than flame's scalar `grad(grad)`. The AI-REML score
   and Fisher information for the diagonalised variance-components model are
   **closed-form**; using them removes the second-order autodiff entirely.
2. **`jnp.linalg.solve(hess_damped, grad)` on `(2, 2)`** (`reml.py:239`) — a
   `2×2` LU custom-call where the closed-form inverse is three flops. (It does
   *not* block the GPU — `vmap`'d `(2,2)` `solve` runs — but it is avoidable
   custom-call overhead.)
3. **`safe_eigh(ZZ^T)` on the full `N×N`** (`reml.py:448`) — `ZZ^T` has rank
   `q` (the random-effect column count), so this is an `O(N³)` eigendecomposition
   where `Z`'s SVD (or a `q×q` eig) is `O(N q²)`, and it is the part `safe_eigh`
   routes to CPU. Secondary (one-off, small `N` in the benchmark), but real at
   large `N`.

## Additional optimisation (both ops): analytic score & Hessian

Beyond removing the Cholesky, replacing the second-order autodiff with the
closed-form REML score/Hessian (single variance component for FLAME; AI-REML for
`reml_fit`) shrinks the graph further (compile) and cuts the op count
(runtime/memory). The prototype above **kept** the autodiff and still got the
wins; analytic derivatives are additive, and would also flatten the residual
compile growth.

## Memory / large-`V`

The `vmap` materialises all `V` voxels' intermediates at once, so peak memory
grows with `V × (per-voxel buffer count)`. The fixes above cut the per-voxel
buffer count (no Cholesky-backward, no second-order autodiff tape); chunking the
voxel batch (`lax.map` over blocks, or a `scan` over voxel-blocks with `vmap`
inside) would additionally **bound** peak memory as a tunable knob, fixing the
`V=262144` OOM/timeout on memory-constrained hosts.

## Cross-references

- `src/nitrix/stats/lme/flame.py`: `_flame_neg_loglik` 92–120 (cholesky
  113–115), `_flame_newton_step` 123–186 (autodiff 140–152), `vmap` 315.
- `src/nitrix/stats/lme/reml.py`: `_neg_reml_loglik_diagonal` 153–198 (cholesky
  188–190), `_newton_scoring_step` 201–277 (grad/hessian 225/231, solve 239),
  `reml_fit` `safe_eigh` 448, `vmap` 467.
- Same root cause as the registration-side cuSOLVER notes; see
  [`perf-bench-feedback.md`](perf-bench-feedback.md) — the perf-bench ledger.
