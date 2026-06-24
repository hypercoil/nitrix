# The streaming kernel substrate

> **TL;DR.**  Our matmul kernel folds rank-1 outer combines directly
> into the ``(BM, BN)`` accumulator inside a ``lax.fori_loop`` over
> ``K``, so the ``(BM, BK, BN)`` value tensor is never materialised.
> Peak HBM is bounded by the ``A + B + output`` I/O floor, not by
> anything resembling ``M * K * N``.  We deliberately avoid
> tensor-core / ``dot`` primitives so the same kernel codegens for
> every algebra (REAL, LOG, TROPICAL_*, EUCLIDEAN, BOOLEAN).  Measured
> on Ampere: streaming JAX and Pallas paths peak at the I/O floor;
> naive ``(A[:,:,None] * B[None,:,:]).sum(axis=1)`` would allocate
> 100× more.

## What "streaming" means in our code

The Pallas Triton kernel in
[`_kernels/cuda/semiring_matmul.py`](../../src/nitrix/_kernels/cuda/semiring_matmul.py)
is the canonical example.  For a given ``(BM, BK, BN)`` tile, the body
looks like this (pseudocode):

```python
acc = monoid.init((BM, BN), dtype)
for kk in range(K):
    a_col = A_ref[:, kk]            # (BM, 1)
    b_row = B_ref[kk, :]            # (1, BN)
    value = binary_op.combine(a_col, b_row)   # (BM, BN)
    acc   = monoid.update(acc, value)         # (BM, BN)
out = monoid.finalize(acc)          # (BM, BN)
```

Three properties:

1. The per-step ``value`` is ``(BM, BN)``, never ``(BM, BK, BN)``.
   That's the difference between O(BM·BN) and O(BM·BK·BN) registers
   per CTA.
2. ``acc`` lives in registers / SMEM the entire time; it is written
   to HBM exactly once at the end via ``pl.store``.
3. ``monoid.update`` can carry a pytree state (LSE's ``(m, s)``).
   The accumulator footprint is then a small constant multiple of
   ``(BM, BN)``, not ``(BM, BK, BN)``.

The reference JAX implementation in
[`semiring/_reference.py`](../../src/nitrix/semiring/_reference.py)
uses ``lax.fori_loop`` and the same pattern.  XLA fuses the body so
the same streaming property holds: the JAX path also avoids the
``(M, K, N)`` materialisation, even though the user wrote a Python
``for``-like loop.

## Why no tensor cores

Tensor-core / ``dot`` primitives (``mma.sync``, cuDNN's
``IMPLICIT_PRECOMP_GEMM``, ``jnp.matmul`` with TF32) all assume
``(*) == ×`` and ``(+) == +`` -- they're hardwired in silicon to
real-multiply-add.  We want the same kernel to lower for
``TROPICAL_MAX_PLUS`` (``(+, *) -> (max, +)``), for ``LOG``
(``(logsumexp, +)``), for ``EUCLIDEAN`` (``(+, (a-b)²)`` plus a
``sqrt`` projection), and so on.  Issuing plain CUDA-core SIMD ops
costs us ~8× peak throughput on Ampere TF32 hardware, but it's the
*only* approach that supports the algebra surface in SPEC §4.1.

Users who specifically want tensor-core throughput on the real
semiring should call ``jnp.matmul`` directly -- ``nitrix.semiring``'s
advantage is the *other* five algebras and the differentiable
substrate, not the real-semiring fast path.  This is called out in
the ``semiring_matmul`` docstring.

## Verifying the memory claim

The streaming-kernel design is *the* load-bearing perf claim from
SPEC §4.1 ("peak on-chip memory at O(BM·BN + BM·BK + BK·BN)").  We
verify it empirically with
[`bench/mem_streaming_kernel.py`](../../bench/mem_streaming_kernel.py),
which reads ``jax.devices()[0].memory_stats()['peak_bytes_in_use']``
before and after each call.

Headline numbers on Ampere A10G, fp32, ``(M=512, K=256, N=512)``:

| variant | peak HBM | naive analytical | ratio |
|---|---:|---:|---:|
| streaming JAX (REAL) | 2.5 MB | 256 MB | 0.010× |
| streaming JAX (LOG) | 5.5 MB | 256 MB | 0.021× |
| Pallas (REAL) | 2.5 MB | 256 MB | 0.010× |
| Pallas (LOG) | 5.5 MB | 256 MB | 0.021× |
| naive (analytical) | 256 MB | -- | 1.000× |

The streaming paths stay within ~2.5× of the I/O floor (``A + B +
output ≈ 2 MB``) -- the extra is JAX pool slack and the JIT'd
loop's small intermediate state.  The naive ``(A[:,:,None] *
B[None,:,:]).sum(axis=1)`` formulation, which XLA cannot fuse without
materialising the inner tensor, is reported analytically because
XLA's compile time for that fusion blows past 60 s on shapes
``≥ (512, 256, 512)``.  See the report at
[`/bench/MEM_STREAMING_KERNEL.md`](../../bench/MEM_STREAMING_KERNEL.md).

The ``Δ`` column in that report is 0 across all rows: a steady-state
call after warm-up does not push the HBM HWM further, so there is no
per-call leak.

## When this would break

The streaming property holds as long as XLA fuses the ``fori_loop``
body.  Practical conditions under which we'd lose it:

- A change to the JAX optimiser that disables fusion across the
  ``while`` lowering of ``fori_loop`` -- unlikely but possible.
- A user supplying a ``Monoid`` whose ``update`` materialises
  intermediates that escape the loop (a poorly-written custom
  algebra).  No general guard against this; the user owns the
  invariant.
- Very large ``BM·BN`` exceeding SMEM on a given CTA.  Our tile
  selector in ``_pick_blocks`` falls back to smaller tiles before
  failing; if it can't find a viable tile it raises
  ``PallasNotTileable`` and the dispatcher routes to the JAX path.

The mem-streaming bench is the regression detector: if a future
change causes either the JAX or the Pallas peak to scale with
``M·K·N``, that test surfaces it immediately.

## Cross-references

- SPEC §4.1 "Kernel strategy"; the original design intent.
- ``src/nitrix/_kernels/cuda/semiring_matmul.py`` -- Pallas kernel.
- ``src/nitrix/semiring/_reference.py`` -- JAX reference.
- ``bench/mem_streaming_kernel.py`` -- HBM measurement script.
- ``bench/MEM_STREAMING_KERNEL.md`` -- frozen report.
- [`semiring-protocols.md`](semiring-protocols.md) -- the pytree-state
  pattern.
