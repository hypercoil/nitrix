# B10. Re-tune the Pallas LOG (logsumexp) `semiring_matmul` kernel

> **Status (2026-06-02): parked (kernel tuning) — LOG is the outlier among
> the built-in algebras.** Not a commitment — gated on the **Trigger**
> below. Effort **M**, no API change. Provenance: migrated from the retired
> top-level `BACKLOG.md` (B-numbering preserved); ledger context in
> [`internal-backlog.md`](internal-backlog.md).

The Pallas/Triton LOG-semiring kernel is markedly under-tuned vs both the
pure-JAX reference and a naive materialise-then-reduce — uniquely among the
built-in algebras.

**Trigger.** A consumer running the LOG matmul (logsumexp contraction —
softmax/attention aggregation) at scale where wall-time matters; or the next
`nitrix-perf-bench` sweep promoting this to a kernel-tuning sprint.

**Notes (evidence; A10G, jax 0.10.0,
`../nitrix-perf-bench/reports/PERF_SEMIRING_MATMUL.md`).** LOG Pallas vs JAX
reference: **1.22× at 256³ (slower)**, 0.59× at 512³. REAL / TROPICAL_MAX /
EUCLIDEAN Pallas hit 0.07–0.20× on the same shapes — tiling is fine in
general; LOG is the outlier. A naive materialise-then-`logsumexp` reaches
0.11× at 512³, so the throughput is achievable. Suspect: the
online-logsumexp `(m, s)` recurrence in `_LogSumExpMonoid` (flash-attention
style) serialises the K-loop / carries register pressure the `sum`/`max`
monoids don't. Direction: tile/unroll/occupancy pass on the LOG kernel,
following online-softmax literature; gate it benchmark-first (ship only if
it beats JAX). **Don't over-read the naive win** — its speed is steady-state
only; it pays a pathological cold compile (~45–585 s/point, ~580 s for 512³)
and elevated peak-HBM (68–85 MB vs streaming 2.6–23 MB). Pallas keeps O(M·N)
streaming memory + ~2 s compile, so it stays default; the task is narrow —
close the LOG-kernel throughput gap, not adopt materialisation. **Arch
caveat:** numbers are A10G; re-bench on target before acting.

**Effort.** M — single-kernel tuning + re-bench; no API change.

**Cross-refs.** `../nitrix-perf-bench/reports/PERF_SEMIRING_MATMUL.md`;
`docs/design/streaming-kernel.md`; `bench/G0_ELL_REPORT.md`;
`src/nitrix/semiring/algebras.py` `_LogSumExpMonoid`.

## Cross-references

- [`internal-backlog.md`](internal-backlog.md) — the engineering-backlog
  ledger.
