# ELL on Triton: the gather gap

> **TL;DR.**  ``semiring_ell_matmul`` runs on the JAX backend
> unconditionally on Ampere+ at first GA.  The reason is structural,
> not perf-marginal: the Pallas Triton backend in the pinned JAX
> version does not lower the ``gather`` primitive (or axis-0
> ``concatenate`` of arbitrary fan-out), both of which the ELL
> streaming kernel needs.  We file this as the G0 outcome from
> IMPLEMENTATION_PLAN §3.1 and revisit when JAX lands gather in the
> Triton lowering.

## What the ELL kernel would need

The ELL forward reduction is::

    C[i, j] = (+)_p ( values[i, p] (*) B[indices[i, p], j] )

For a tile of ``BM`` output rows and ``BN`` output columns, the K
loop reads, on every step ``p``:

- ``values_block[i, p]`` -- contiguous, fine.
- ``B[indices[i, p], j]`` for ``j in [j0, j0 + BN)`` -- **a per-row
  gather of ``B``** with a different row index per ``i``.

That second access is the problem.  In JAX it lowers to ``lax.gather``
(or equivalently, ``jnp.take`` along an axis with a vector index),
which the Pallas Triton backend does not lower as of the pinned JAX
version (it raises
``NotImplementedError: Unimplemented primitive in Pallas GPU lowering:
gather``).

## What we tried

Three workarounds inside the kernel:

1. **``B_blk[idx_p]`` (advanced indexing).**  Lowers to ``gather`` in
   HLO; same Triton lowering failure.
2. **``jnp.take(B_blk, idx_p, axis=0)``.**  Same.
3. **Python-unrolled per-row ref loads with ``pl.ds(idx_mp, 1)`` and
   ``jnp.stack`` along axis 0.**  Lowers to per-row loads (which work)
   plus axis-0 concatenate (which Triton lowers only for the last
   axis: ``NotImplementedError: Only concatenate along the last
   dimension is supported``).  Pairwise concatenation via a binary
   tree hits the same limit.

The only workable approaches require either an upstream Pallas
Triton fix (gather, or axis-flexible concat) or a layout flip that
swaps which axis is the per-row gather; the layout flip would force
us to materialise the entire output tile column-major and re-do the
stride math, which is significantly more code than the JAX path is
worth at first GA.

## The policy

Per the G0 decision rule in IMPLEMENTATION_PLAN §3.1 ("Triton ≥ 5×
slower or unstable: JAX-default; Pallas is opt-in"), the GA shape
for ELL is:

- ``backend="jax"`` -- works.
- ``backend="pallas-cuda"`` -- resolved by the dispatcher, then
  immediately falls back to JAX with one
  ``NitrixBackendFallback`` warning per ``(shape, dtype, algebra)``.
- ``backend="auto"`` -- on Ampere+, resolves to ``pallas-cuda`` and
  then falls back, same as above.  This is the *intended* behaviour:
  the user gets a warning surfacing the policy.

The Pallas kernel stub at
[`_kernels/cuda/semiring_ell_matmul.py`](../../src/nitrix/_kernels/cuda/semiring_ell_matmul.py)
raises ``PallasELLNotTileable`` unconditionally with a pointer at
the bench report.  When Pallas Triton grows a usable gather we
swap the stub for a real kernel; the public dispatcher does not
change.

## What the JAX path gets us

The JAX reference at
[`semiring/_reference.py:reference_semiring_ell_matmul`](../../src/nitrix/semiring/_reference.py)
walks the K dim via ``lax.fori_loop`` with ``B[idx_p]`` for the
gather.  It is correct, differentiable (the backward in
[`semiring/_backward.py`](../../src/nitrix/semiring/_backward.py)
composes a per-algebra rule with ``scatter_add`` for the ``B``
gradient), and reasonably performant on Ampere.

The G0 bench at
[`bench/g0_ampere_ell.py`](../../bench/g0_ampere_ell.py) records the
JAX-path baseline.  Headline numbers (A10G, fp32, mesh-adjacency
``(m=4096, k_max=32, ncol=32)``):

| algebra | wall-time | effective Mevents/s |
|---|---:|---:|
| REAL | 346 µs | 12 100 |
| LOG | 466 µs | 9 000 |
| TROPICAL_MAX_PLUS | 356 µs | 11 800 |

These are the numbers downstream consumers should plan around.
The report at [`/bench/G0_ELL_REPORT.md`](../../bench/G0_ELL_REPORT.md)
includes a ``confirm_pallas_falls_back()`` probe that re-verifies
the structural fallback on each run; if a future JAX release lands
gather lowering, the probe flips to ``no`` and we know to revisit.

## What changes the policy

Any of:

1. Pallas Triton lands a viable ``gather`` lowering (or axis-0
   concat).  We swap the stub for a real kernel; the bench probe
   flips, the policy doc is rewritten.
2. We accept a Hopper-only Mosaic GPU code path -- excluded at first
   GA per SPEC_UPDATE_v0.2 §1.1 because it would cut out the bulk of
   academic-lab hardware.  Revisit at 1.x.
3. We write a shape-specialised Pallas variant that avoids gather --
   e.g., small fixed ``k_max`` with the kspatial × c_in axis fully
   Python-unrolled at trace time and a static row-permutation
   computed offline.  Worth investigating for the mesh-icosphere
   case where ``k_max ≈ 6``; out of scope for first GA.

## What we considered and didn't pick

- **Materialising patches to dense and running ``semiring_matmul``.**
  Would route through the working Pallas matmul kernel, but the
  patches matrix is ``M × N``; for any reasonable ELL workload (e.g.,
  100k mesh vertices, 8k feature columns) that's 3 GB.  Defeats the
  point of ELL.
- **Routing through ``jax.experimental.sparse`` BCOO.**  Explicitly
  excluded by SPEC §3.2: BCOO has been a persistent friction surface
  against the XLA / Pallas boundary; the entire reason ELL is the
  primary format is to avoid that machinery.
- **CPU Pallas backend.**  Pallas does have a CPU lowering, but the
  spec restricts us to NVIDIA Ampere+ for Pallas (SPEC_UPDATE_v0.2
  §1.1).  CPU users get the JAX path.

## Cross-references

- IMPLEMENTATION_PLAN §3.1 (G0 decision rule), §5.2 (Phase 2.A.8).
- SPEC §3.2 (ELL format), SPEC_UPDATE §3.2 (sectioned ELL -- a future
  addition layered on top of this kernel).
- ``bench/g0_ampere_ell.py`` -- the G0 baseline script.
- ``bench/G0_ELL_REPORT.md`` -- the frozen report.
- ``tests/test_ell.py::test_ell_pallas_falls_back_with_warning`` --
  asserts the policy is in effect.
- [`backend-selection.md`](backend-selection.md) -- the fallback
  machinery.
