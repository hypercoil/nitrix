# B12. IIR `sosfilt`/`sosfiltfilt` GPU backend — default + missing associative path

> **Status (2026-06-05): RESOLVED.** Both deficits closed, and then some -- the
> GPU path now *beats* cupy. Three commits on `perf/iir-gpu-backend`: (1)
> backend-aware default (`'auto'` -> scan on CPU, parallel on GPU); (2)
> parallel zero-phase `sosfiltfilt` via a transposed-DF2 associative engine
> that threads `zi`; (3) the **FFT-convolution engine** (`backend='fft'`, the
> GPU default) -- an IIR filter is LTI so its output is exactly convolution
> with the (host, geometrically-decaying) impulse response, made `O(T log T)`
> and parallel. On the L4: `sosfilt` 147 ms -> **0.95 ms** (cupy 1.78 ms),
> `sosfiltfilt` 299 ms -> **1.77 ms** (cupy 9.4 ms) at ch=1024/obs=4096, exact,
> with compile 10.5 s -> 0.29 s and HBM 704 MB -> 201 MB. See
> `docs/design/signal-and-numerics.md` ("Why an IIR filter still has a parallel
> FFT path"). The original parked write-up follows for the record.
>
> **Status (2026-06-03): parked (perf / API-default) — two measured GPU
> deficits in the recursive filter ops.** Not a commitment — gated on the
> **Trigger** below. Effort **S + M**, the first is a default/doc change, the
> second adds a kernel path. Provenance: surfaced building the
> `nitrix-perf-bench` `sosfilt`/`sosfiltfilt` cases; ledger context in
> [`internal-backlog.md`](../internal-backlog.md), evidence in
> [`perf-bench-feedback.md`](../perf-bench-feedback.md).

The IIR filters are the suite's first *recursive* ops, and the recurrence depth
dominates wall-time on the GPU. Two concrete, measured deficits:

**1. `sosfilt`'s default `backend='scan'` is a poor GPU default.** The default
sequential `lax.scan` (O(T) depth) is **8.3× slower on the L4 than
`backend='associative'`** (the parallel-prefix `associative_scan`, O(log T)
depth) — and slower than scipy on the *CPU*. The optimal backend is
**platform-dependent** (scan wins on CPU, associative on GPU), so a fixed
`'scan'` default is wrong for half the platforms. **Direction:** make the
default platform-aware (select `associative` when the data is on a GPU device),
or at minimum surface the guidance prominently in the docstring. Effort **S**.

**2. `sosfiltfilt` has no associative path.** The zero-phase forward-backward
filter is `lax.scan`-only, so it cannot escape the O(T) sequential penalty on
the GPU: it runs **299 ms** at ch=1024/obs=4096, **4.6× slower than scipy on
CPU** and **35× behind cupy**. **Direction:** add an associative-scan path for
`sosfiltfilt` (the forward and backward passes are each linear recurrences; the
steady-state `zi` init + odd padding need carrying through the parallel
formulation). Effort **M**.

**Trigger.** A consumer running IIR band-pass (fMRI `sosfilt`/`sosfiltfilt`) on
GPU at scale where wall-time matters; or the next `nitrix-perf-bench` sweep
promoting this to a kernel/default-tuning task. Item 1 (the default) is cheap
enough to do opportunistically when the file is next touched.

**Notes (evidence; NVIDIA L4, jax 0.10.0,
`../nitrix-perf-bench/reports/PERF_SOSFILT.md` + `PERF_SOSFILTFILT.md`).** At
ch=1024, obs=4096, order-4 band-pass: GPU `sosfilt` scan **168 ms** vs
associative **20 ms** (8.3×); scipy-CPU **27 ms** (so the GPU scan loses to CPU
scipy). CPU `sosfilt` is the reverse — scan **66 ms** vs associative **601 ms**
(associative does more total work with no parallelism payoff). `sosfiltfilt`
GPU **299 ms** vs scipy-CPU **65 ms** vs cupy **8.4 ms**. **Deeper gap (lower
priority):** even nitrix's best (associative, **20 ms**) trails cupy's CUDA
recurrence kernel (**1.8 ms**) by ~11× — a kernel-quality gap beyond the
backend choice; correctness is not at issue (both backends match
`scipy.signal` to ~1e-9). Re-bench on the target arch before acting.

**Effort.** S (item 1, default/doc) + M (item 2, associative `sosfiltfilt`). No
API *surface* change for item 1 if the default is auto-selected; item 2 is a
new internal kernel path, same signature.

**Cross-refs.** `../nitrix-perf-bench/reports/PERF_SOSFILT.md`,
`PERF_SOSFILTFILT.md`; `src/nitrix/signal/_iir.py` (`_sosfilt_scan` /
`_sosfilt_associative` / `sosfiltfilt`); nitrix-perf-bench `87c99d4`.

## Cross-references

- [`internal-backlog.md`](../internal-backlog.md) — the engineering-backlog
  ledger.
- [`perf-bench-feedback.md`](../perf-bench-feedback.md) — the perf-bench-surfaced
  ledger.
