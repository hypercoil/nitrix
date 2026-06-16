# Mosaic GPU (Hopper/Blackwell) registration kernels — `nitrix._kernels`

> **Status (2026-06-13): PROPOSED, hardware-blocked (cannot develop/test on the
> current box).** Two registration force kernels are *blocked or non-winning on
> the pinned JAX `cuda12==0.10.1` **Triton** Pallas backend*, but are plausibly
> expressible — and winning — on the **Mosaic GPU** backend
> (`jax.experimental.pallas.mosaic_gpu`), which targets **sm_90+ (Hopper H100/
> H200, Blackwell B100/B200)**. The development box is an **NVIDIA L4 (sm_89,
> Ada)** — Mosaic GPU does not run there, so this is filed rather than built:
> we do not work on a kernel we cannot test. Revisit when a Hopper/Blackwell
> backend is in CI. Provenance: the v4 registration perf campaign (LNCC → 22.6×,
> MI → 8.6× vs ANTs-CPU; see `registration-suite-v4-force-kernels.md`).

## Why Mosaic might lift the Triton blockers

The Triton backend (what `_kernels/cuda/*.py` use today) has two limits the
campaign hit, both of which Mosaic GPU's lower-level, SMEM-ref programming model
(explicit async copies/TMA, warp specialization, shared-memory scratch with
atomics) is expected to relax:

1. **No slicing of intermediate arrays** (`pl.ds` works on *refs* only; a static
   slice of a loaded/intermediate array is an unimplemented Triton lowering).
   This blocks the in-tile **separable / integral-image** windowed sum.
2. **Atomics + shared-memory scratch friction.** `plgpu.atomic_add` exists and
   a *global*-atomic kernel works, but the **per-block SMEM-partial** pattern
   (block-local atomics into a shared-memory histogram, then one merge) hit an
   indexer-lowering error on the merge. Mosaic's first-class SMEM refs are the
   natural home for it.

## Candidate 1 — LNCC sliding-window: separable in-tile variant

The shipped L2b kernel (`_kernels/cuda/lncc_force.py`) already **wins** on
Triton (1.2× @128³ → 4× @256³) via the ITK scanning-window (scan x, carry 5
running sums, drop/add x-planes — ref-slices only, no intermediate slicing).
That is sufficient. The Mosaic angle is only a *possible further* win: an in-tile
**separable** windowed sum (cumsum-and-difference, or three 1-D sweeps) is O(N)
and would not carry the running-sum state, but it needs intermediate-array
slicing — blocked on Triton, plausibly fine on Mosaic. **Low priority** (L2b
already wins; this is an incremental).

## Candidate 2 — MI 5c: per-block SMEM-partial joint-histogram (the real one)

The MI bottleneck is the joint-histogram **scatter** (92% of `mi_grad`). On
Triton: `bincount` ties the XLA `scatter_add`, the histogram-as-matmul is 2.6×
*slower* (dense one-hot materialization), and a **naive global-atomic** Pallas
kernel merely *matches* XLA (the 32²=1024-entry table is L2-cached, so global
atomics are not DRAM-bound). The only variant that could beat XLA is the classic
**per-block SMEM-partial histogram** — block-local SMEM atomics + one global
merge — which on Triton hit the merge-indexer friction above and, for an
L2-cached 1 KB table, has a small ceiling anyway. On Mosaic + Hopper:
warp-specialized histograms with first-class SMEM and larger/persistent partials
are the established fast path, and a larger-bin (e.g. 64²) high-fidelity Mattes
MI would benefit more. **Build behind a size/bin gate; expect the win to grow
with the bin count and the per-iteration call rate (deformable MI / fMRIPrep).**

## Plumbing already in place

- The backend dispatch (`_internal/backend.py`: `resolve_backend`,
  `auto_backend`, the Ampere+ probe) has the structure for a third resolved arm
  — add `'pallas-mosaic'` gated on an sm_90+ capability probe, beside
  `'pallas-cuda'` (Triton) and `'jax'`. `Force.backend` already threads through.
- The kernels live in `_kernels/cuda/`; a Mosaic variant could be
  `_kernels/cuda/<name>_mosaic.py` selected by the resolved backend.

## Gate (do not build until)

- A Hopper/Blackwell (sm_90+) box is available in development/CI, AND
- a profile on that box shows the candidate op dominates per-iteration HBM/time
  for a bottlenecked consumer (the standing `perf-wins-must-certify-at-scale`
  discipline) — Mosaic is lower-level and only worth the porting cost where it
  measurably beats the Triton/XLA path.

## Cross-references

- [`registration-suite-v4-force-kernels.md`](../design/registration-suite-v4-force-kernels.md)
  — the Phase-5 kernel design + as-shipped status (5a ESM, 5b LNCC, 5c MI).
- `src/nitrix/_kernels/cuda/lncc_force.py` (L2b, the winning Triton sliding
  window), `demons_force.py` (5a).
- `src/nitrix/metrics/information.py` (`mi_grad`, the scatter 5c would fuse),
  `register/_force.py` (`MIForce`, `sample_stride`).
- `src/nitrix/_internal/backend.py` — where a `'pallas-mosaic'` arm would land.
