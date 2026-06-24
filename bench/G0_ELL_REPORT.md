# G0 — Ampere ELL policy + JAX-path baseline

> SPEC reference: IMPLEMENTATION_PLAN §3.1; SPEC §10.
> This is a *policy* document plus the JAX-path baseline; the
> Pallas-vs-JAX wall-time comparison is deferred until Pallas
> Triton lowers the `gather` primitive on the pinned JAX.

## Policy

`semiring_ell_matmul` runs on the **JAX backend unconditionally** on Ampere+; `backend="pallas-cuda"` resolves to JAX and emits one `NitrixBackendFallback` warning per `(shape, dtype, algebra)` signature.

Pallas-falls-back probe (current JAX pin): yes.

## Host

- Device: NVIDIA A10G (gpu)
- Platform: Linux-6.1.161-183.298.amzn2023.x86_64-x86_64-with-glibc2.39
- JAX: 0.10.0

## JAX-path baseline (post-warm-up steady state)

| m | k_max | ncol | algebra | JAX wall-time | JAX compile | effective Mevents/s |
|---|------:|-----:|---------|--------------:|------------:|--------------------:|
| 1024 | 16 | 32 | real | 257.8 µs | 92.77 ms | 2033.3 |
| 1024 | 16 | 32 | log | 279.5 µs | 136.06 ms | 1875.5 |
| 1024 | 16 | 32 | tropical_max_plus | 256.3 µs | 77.98 ms | 2045.4 |
| 4096 | 32 | 32 | real | 346.4 µs | 94.11 ms | 12106.6 |
| 4096 | 32 | 32 | log | 465.9 µs | 148.06 ms | 9002.5 |
| 4096 | 32 | 32 | tropical_max_plus | 355.6 µs | 79.31 ms | 11796.4 |

## Decision (per IMPLEMENTATION_PLAN §3.1)

**JAX-default on Ampere+, unconditional for first GA.**  The
reason is structural (Pallas Triton lacks gather lowering),
not a performance-margin call.  Revisit when either: (a) JAX
lands gather in the Triton lowering, or (b) we accept a
shape-specialised Pallas variant that avoids gather (e.g.,
small fixed `k_max` with fully Python-unrolled per-row
loads, gated by tile size).

## Why ELL is JAX-only at GA

The ELL streaming kernel reads, for each output row, the
rows of `B` at the column indices `indices[i, :]`.  In
JAX this lowers to `jnp.take` / `lax.gather` over `B`, which
Triton-on-Pallas does not currently lower.  Two workarounds
were tried and rejected:

- **Per-row `pl.ds` ref loads inside a Python-unrolled loop
  over BM rows.**  The loaded rows must be stacked back into
  the `(BM, BN)` tile shape via concatenation along axis 0;
  Triton supports only axis-(-1) concatenate.
- **`jnp.take` inside the kernel body.**  Lowers to `gather`
  in HLO; same lowering gap.

The right fix is upstream (Pallas Triton landing gather) or
a separate hand-rolled kernel layout (Hopper Mosaic GPU,
which we explicitly exclude per §1.1).  Both are 1.x
conversations.
