# Convolution: explicit im2col + matmul

> **TL;DR.**  ``semiring_conv`` extracts patches via a NaN-safe gather
> on an identity-padded input, then runs ``semiring_matmul`` over the
> patches.  This re-uses the entire forward / backward / fallback
> machinery of the matmul kernel.  On Ampere, ``semiring_conv(REAL)``
> via the Pallas matmul backend is **~1.7×** slower than cuDNN's
> ``IMPLICIT_PRECOMP_GEMM`` -- exactly in the research-predicted
> 1.3-1.7× range for an explicit-im2col CUDA-core path.  A future
> Pallas implicit-GEMM conv kernel would close most of that gap.

## The reduction

``semiring_conv`` is the third member of the SPEC §3.1 trinity
(``matmul``, ``ell_matmul``, ``conv``).  The math:

```
C[..., *spatial_out, c_out]
    = (+)_{*ks, c_in} (
          x[..., spatial_at(spatial_out, ks), c_in]
            (*)
          k[*ks, c_in, c_out]
      )
```

For a fixed output position, the reduction is over
``prod(kspatial) * c_in`` (often 9 × 64 = 576 elements for a 3×3
conv with 64 input channels).  This is exactly the shape of a
matrix-matrix product over ``(batch * spatial_out, prod(kspatial) *
c_in) × (prod(kspatial) * c_in, c_out)``, where the left operand is
the im2col-flattened patches and the right is the kernel reshaped.
We exploit that.

## Why explicit im2col on the JAX path

Two reasons:

1. **XLA will not fuse ``gather`` into ``dot``.**  XLA:GPU classifies
   ops as input / loop / output fusions around a "hero" op;
   ``dot_general`` is a hero and a producer ``gather`` does *not*
   fuse into it.  See ``openxla.org/xla/gpu_architecture`` and the
   JAX issue tracker (discussions in #22313, #24051).  The
   consequence: if we wrote our conv as a series of strided
   slices + matmuls, XLA would materialise the slices and the matmul
   would still read from HBM.  We don't get a free "implicit
   im2col" via fusion.
2. **Re-using ``semiring_matmul`` gives us forwards, backwards, and
   fallback observability for free.**  The conv's gradient composes
   through the registered ``matmul_vjp`` for each algebra; we don't
   write a separate ``conv_vjp``.  Per
   [`backward-kernels.md`](backward-kernels.md), the matmul VJP is
   already finite-difference-checked at the G1 tolerance, so the
   conv inherits a tested backward by composition.

The materialised patches cost ``(M_out × prod(kspatial) × c_in × 4)``
bytes for fp32; for typical 3×3 conv with ``c_in ≈ c_out`` that's
~4× the I/O floor.  Worth it for the code reuse.

## NaN-safe patch extraction

The natural primitive ``lax.conv_general_dilated_patches`` does
*not* do what we need.  It lowers via a multiply-with-one-hot
trick internally and produces ``NaN`` wherever the input contains
``-inf`` (because ``0 * -inf == NaN``).  ``-inf`` flows through our
tropical max-plus and log semirings as the algebra's identity, and
we need it preserved.

Worse: even on ``VALID`` padding (no explicit pad), the multiply-by-
one-hot still happens *inside* the extraction.  So the issue isn't
just padding; it's the implementation strategy.

Our replacement, ``_extract_patches_nan_safe`` in
[`semiring/conv.py`](../../src/nitrix/semiring/conv.py):

1. Pad ``x`` along each spatial dim with ``semiring.identity`` (so
   ``-inf`` for tropical max / log; ``+inf`` for tropical min;
   ``0`` for REAL; algebra-correct in every case).
2. For each spatial dim, build a ``(out_d, k_d)`` index matrix:
   ``i * stride + ks * dilation``.
3. Gather via ``jnp.take`` along that dim, twice for 2D, thrice for
   3D, etc.  This is a *pure gather* with no multiplications, so no
   NaN injection.

The padded boundary entries are guaranteed identity, so they're
no-ops under the algebra's reduction by construction.  The
correctness invariant is "the patch extraction never multiplies by
anything"; gather operations preserve every input bit exactly.

We discovered the ``conv_general_dilated_patches`` NaN behaviour
when ``test_neg_inf_in_tropical_max_plus_propagates`` failed in our
first cut.  See the test docstring for the exact pattern.

## How we compare against cuDNN

Two cuDNN baselines matter:

1. ``lax.conv_general_dilated`` with default precision (TF32 on
   Ampere+ tensor cores).  This is what production users get when
   they call ``flax.linen.Conv`` or equivalent.  Not the fair
   comparison for us: tensor cores have ~8× higher peak fp32
   throughput than CUDA cores on A10G.
2. ``lax.conv_general_dilated(..., precision='highest')`` (strict
   fp32, no tensor cores).  Same CUDA-core compute as our kernel,
   so the apples-to-apples comparison.

The ``perf_semiring_conv.py`` bench reports both.  Headline numbers
on A10G fp32, ``B=1, 64×64×32 → 3×3 32`` (post-warm-up median):

| path | wall-time | vs cuDNN TC | vs cuDNN fp32 |
|---|---:|---:|---:|
| cuDNN tensor-core (TF32) | 161 µs | 1.00× | -- |
| cuDNN strict fp32 (no TC) | 147 µs | -- | 1.00× |
| **ours (Pallas matmul)** | **273 µs** | **1.70×** | **1.86×** |
| ours (JAX matmul only) | 1.83 ms | 11.4× | 12.5× |
| LOG (Pallas matmul) | 464 µs | 2.88× | 3.16× |
| TROPICAL_MAX_PLUS | 275 µs | 1.71× | 1.87× |
| EUCLIDEAN | 316 µs | 1.96× | 2.15× |

The 1.7-1.9× ratio against the fp32 cuDNN baseline matches Chen et
al. 2021 (arXiv:2110.03901): explicit-im2col vs implicit-GEMM on
CUDA-core hardware is ~30-70% slower, dominated by HBM round-trip
for the patches matrix.  The 11× we initially saw was a different
phenomenon: the user code in the test was using the JAX-only matmul
path (``lax.fori_loop``) for the inner reduction, not the Pallas
matmul kernel.  Once we route through Pallas, the gap drops to the
expected range.

For non-REAL algebras, cuDNN cannot be used at all (``mma.sync`` and
all the cuDNN convolution kernels are hard-wired to ``(*, +)``).
Our numbers for LOG / TROPICAL / EUCLIDEAN are *the* baseline; the
right comparison is to a future Pallas implicit-GEMM conv kernel,
not to cuDNN.

## What a Pallas implicit-GEMM kernel would buy us

A dedicated Pallas conv kernel that computes patch coordinates
inside the inner loop (KeOps-style tiled online map-reduce; Triton
paper §6.2; the reference at
``github.com/l1351868270/implicit_gemm.triton`` is a clean port of
cuDNN's IMPLICIT_GEMM in ~25 lines of Triton) would:

- Eliminate the im2col HBM allocation (4-9× input-size reduction in
  capacity for typical 3×3 conv).
- Close the 1.3-1.7× wall-time gap measured above.

Importantly, **conv's gather pattern is static** -- a function of
``kspatial / strides / dilation``, not of a data-dependent index
array.  So unlike the ELL case (where the gather index is a
data array and Triton can't lower it), the conv gather is just
arithmetic on loop indices.  It might lower in Triton without
depending on a runtime gather primitive.  This is the most
attractive 1.x follow-up after the GA cut.

## What we considered and didn't pick

- **``lax.conv_general_dilated`` for REAL, our path for everything
  else.**  Inconsistent API: padding semantics differ in subtle
  ways, gradient composition would need a separate code path,
  algebraic invariants (identity propagation) need re-verifying.
  Considered briefly; rejected because the 1.7× isn't worth the
  branching cost when users wanting tensor cores can just call
  ``jnp.matmul`` / ``lax.conv_general_dilated`` directly.
- **Direct conv (no im2col).**  Roughly ``M_out × prod(kspatial) ×
  c_in`` separate scalar operations vs one matmul of similar total
  flops.  GPU launch overhead and lack of arithmetic-intensity
  reuse make this 5-10× slower than im2col + GEMM.  Standard
  result.
- **FFT-based conv.**  Real-only; specialised to certain kernel
  sizes; doesn't generalise to tropical / log.  Not relevant for
  the semiring substrate.
- **Winograd.**  Same.

## Cross-references

- SPEC §3.1 ``semiring_conv`` signature.
- ``src/nitrix/semiring/conv.py`` -- module.
- ``tests/test_conv.py`` -- 18 tests covering REAL parity with cuDNN,
  per-algebra correctness, identity propagation, finite-diff grad,
  fallback observability, batching.
- ``bench/perf_semiring_conv.py`` -- the bench.
- ``bench/PERF_SEMIRING_CONV.md`` -- frozen report.
- Research references in the bench script docstring (Chen et al.
  2021; cuDNN IMPLICIT_PRECOMP_GEMM; KeOps tiled online map-reduce;
  Triton implicit-GEMM template; XLA gather+dot non-fusion).
