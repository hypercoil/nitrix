# Doc-fix: `lomb_scargle_interpolate` silently runs its eigh on CPU on cuSolver-broken stacks

> **Status (2026-06-02): RESOLVED.** Added a "Device placement" Notes
> paragraph to `lomb_scargle_interpolate` describing the `safe_eigh`
> CPU-routing on cuSolver-broken stacks (correct results, host round-trip;
> GPU-residency not guaranteed) and the contrast with the GPU-resident
> matrix-function ops. See `IMPLEMENTATION_PLAN.md §10.3` (2026-06-02 entry).
> Provenance: surfaced building a `nitrix-perf-bench` case; ledger context in
> [`perf-bench-feedback.md`](../perf-bench-feedback.md).

`_lomb_scargle_solve_shared_mask` calls `safe_eigh`
(`src/nitrix/linalg/_solver.py:147`), which `device_put`s the Gram to
`eigh_device()` and runs `jnp.linalg.eigh` there. On this L4 / driver-580 /
CUDA-12 stack `eigh_device()` probes to **`cpu:0`** (dense cuSolver `eigh`
fails at d≥256 — `gpusolverDnCreate failed`; confirmed at K=499). Because
`device_put` to CPU is an XLA placement hint (it is **not** traced away
under `jit`), `lomb_scargle_interpolate` invoked on GPU data runs its
eigendecomposition **on the host** with GPU→CPU→GPU round-trips — i.e. it is
*not* a GPU-resident op on affected stacks, and its K×K Gram (K up to
~`2·n_freq+1` ≈ 499 at fMRI `n_obs`=500) is exactly in the broken range.
This is correct behaviour (it is what `safe_eigh` is *for*), but it is a real
portability/perf caveat the docstring does not mention.

**Fix.** Add a Notes line: on stacks where dense cuSolver `eigh` is
unavailable, the Gram solve is routed to CPU (correct results, with host
transfer cost); GPU-residency of the solve is not guaranteed. (Contrast: the
matrix-function ops `symlog`/`symsqrt`/`sympower` *consume* a raw eigh that
XLA lowers off cuSolver, so they stay GPU-resident — a useful pattern to
note.)

## Cross-references

- [`perf-bench-feedback.md`](../perf-bench-feedback.md) — the doc-drift ledger.
- [`doc-lomb-scargle-eigh-factorisation.md`](doc-lomb-scargle-eigh-factorisation.md)
  — the `eigh`-vs-Cholesky docstring drift on the same solve.
- `src/nitrix/linalg/_solver.py:147` — `safe_eigh`.
