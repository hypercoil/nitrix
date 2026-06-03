# B14. Spectral-embedding solver on GPU: lobpcg lags eigsh; eigh-path wedges

> **Status (2026-06-03): parked (perf + robustness) — two measured
> spectral-embedding solver issues on the cuSOLVER-broken L4.** Not a
> commitment — gated on the **Trigger** below. Effort **M**. Provenance:
> surfaced building the `nitrix-perf-bench` `laplacian_eigenmap` case; ledger
> context in [`internal-backlog.md`](internal-backlog.md), evidence in
> [`perf-bench-feedback.md`](perf-bench-feedback.md).

`laplacian_eigenmap` / `diffusion_embedding` (`graph/connectopy.py`) offer two
solvers (`'eigh'`, `'lobpcg'`). On GPU, measured against `scipy`/`cupy`
`eigsh`, two issues:

**1. `lobpcg` is the GPU-robust path but markedly slow.** The matrix-free
`lobpcg_standard` path **runs genuinely on the GPU at every size** (dodging the
broken dense cuSOLVER) — it is the *only* reliable GPU spectral-embedding path
(see issue 2) — but it is **8–17× slower than `cupyx.scipy.sparse.linalg.eigsh`
on the GPU** and **5–9× slower than `scipy.sparse.linalg.eigsh` on the CPU**.
The implicit-VJP differentiability is the reason lobpcg is the default (eigsh /
ARPACK-Lanczos are not differentiable), so this is a real tradeoff — but the
gap is large. **Direction:** a faster top-k eigensolver for the *non*-
differentiable case (a Lanczos/ARPACK-style path, or a tuned lobpcg —
block size / preconditioner / iteration budget), selectable when gradients are
not needed.

**2. The `eigh` path wedges on GPU instead of reliably falling back to CPU.**
`solver='eigh'` (and `'auto'` on dense input) routes through `safe_eigh`, whose
CPU-vs-GPU decision is **cached at import** from a one-shot `eigh_device()`
cuSOLVER probe. When cuSOLVER is healthy at import but wedges later (it is
intermittent on this L4), the eigh path issues a GPU `eigh` that fails
**`gpu_solver_unavailable`** mid-run — a hard error, not the intended graceful
CPU fallback. (perf-bench: `nitrix-jax-eigh` is `gpu_solver_unavailable` at
n=512/1024/2048 on jax-cuda12, while it runs fine on the CPU platform.)
**Direction:** make `safe_eigh`'s fallback *adaptive* — catch the cuSolver
error at call time and retry on CPU (or re-probe), so `solver='eigh'`/`'auto'`
is GPU-safe regardless of the import-time device state.

**Trigger.** A consumer running spectral embedding / diffusion maps on GPU at
scale (connectome gradients, large parcellations) where wall-time or the
eigh-wedge matters; or the next `nitrix-perf-bench` sweep.

**Notes (evidence; NVIDIA L4, jax 0.10.0,
`../nitrix-perf-bench/reports/PERF_LAPLACIAN_EIGENMAP.md`).** At n=1024, k=8:
GPU `lobpcg` **552 ms** vs `cupy eigsh` **45 ms** (12×) vs `scipy eigsh` (CPU)
**58 ms** (9.5×); the dense `eigh` path is `gpu_solver_unavailable` on GPU. All
solvers agree on the eigenvalues (lobpcg to its iterative tolerance). Re-bench
on the target arch — and note issue 2 only bites on cuSOLVER-broken stacks.

**Effort.** M — issue 1 is a new/tuned solver path (no API-surface change if
auto-selected when `grad` is not required); issue 2 is an adaptive fallback in
`linalg._solver.safe_eigh` (catch + retry on CPU).

**Cross-refs.** `../nitrix-perf-bench/reports/PERF_LAPLACIAN_EIGENMAP.md`;
`src/nitrix/graph/connectopy.py` (`_auto_solver`, the eigh / lobpcg paths);
`src/nitrix/linalg/_solver.py` (`safe_eigh`, `eigh_device`); nitrix-perf-bench
`85f0ea8`.

## Cross-references

- [`internal-backlog.md`](internal-backlog.md) — the engineering-backlog
  ledger.
- [`perf-bench-feedback.md`](perf-bench-feedback.md) — the perf-bench-surfaced
  ledger. (See also [`boundary-mode-parity.md`](boundary-mode-parity.md) B13,
  [`iir-filter-gpu-backend.md`](iir-filter-gpu-backend.md) B12.)
