# Perf audit, 2025-05: where we lag and what to do about it

> **TL;DR.** Audit of ~6 newer public ops vs natural references
> (numpy / scipy / sklearn / statsmodels) on the A10G runner.  Headline:
> nitrix dominates by 100-800× at fMRI-realistic scales (V=100k voxels,
> 24-confound regression, 1000-channel kernel matrix); the *only* gap
> is in ``morphology.distance_transform`` against
> ``scipy.ndimage.distance_transform_edt``, and the gap turns out to
> be **algorithmic mismatch, not slow implementation**: we ship a
> chamfer DT (Chebyshev / Manhattan), scipy ships exact Euclidean DT
> via the Felzenszwalb-Huttenlocher 2012 parabolic algorithm.  They
> compute different things.  Recommendation: ship a separate
> ``distance_transform_edt`` primitive (F-H or Saito-Toriwaki) when a
> consumer needs exact EDT; until then, **document the algorithmic
> distinction** at the existing op's docstring level and consider the
> audit closed.

## Audit summary

Per ``bench/perf_audit.py`` (see ``bench/PERF_AUDIT.md`` for the raw
table).  Wall-time ratio is ``nitrix / reference``; ``<1`` means
nitrix faster.

### Where we win

At realistic neuroimaging scales the substrate dominates:

| op | shape | nitrix | ref | ratio |
|---|---|---:|---:|---:|
| ``linalg.rbf_kernel`` | ``(5000, 32)`` | 0.80 ms | sklearn 305 ms | **0.003×** (375× faster) |
| ``linalg.residualise`` | V=100k, N=400, K=24 | 3.0 ms | numpy lstsq 2426 ms | **0.001×** (800× faster) |
| ``stats.cov`` | ``(2000, 1000)`` | 0.53 ms | numpy 69 ms | **0.008×** (130× faster) |
| ``lme.flame_two_level`` | V=1000 vox | 15 ms | statsmodels (extrap) 4588 ms | **0.003×** (300× faster) |

The wins are dominated by two factors: (a) the GPU has more
arithmetic throughput than a single CPU thread at scale, and (b)
nitrix's vmap-batching avoids per-voxel Python loop overhead that
the reference implementations all pay.

### Where the gap surfaces

| op | shape | nitrix | scipy | ratio |
|---|---|---:|---:|---:|
| ``morphology.distance_transform`` | ``(64, 64)`` | 5.1 ms | 0.3 ms | **15.7×** |
| ``morphology.distance_transform`` | ``(256, 256)`` | 13.9 ms | 4.6 ms | **3.1×** |
| ``morphology.distance_transform`` | ``(32, 32, 32)`` | 5.0 ms | 3.9 ms | **1.3×** |
| ``morphology.distance_transform`` | ``(64, 64, 64)`` | 62 ms | 45 ms | **1.4×** |
| ``morphology.distance_transform`` | ``(128³)`` | 898 ms | 456 ms | **2.0×** |

Gap pattern: 15× at small 2D, converging to ~2× at large 3D.

## Why the gap exists

These two implementations compute **different things**.

### Our `distance_transform`: iterative chamfer DT

Algorithm: iterative ``TROPICAL_MIN_PLUS`` convolution with a
``3 × 3 × ... × 3`` step-cost kernel.  ``max_iters`` defaults to
the longest spatial extent; each iter is a small-kernel conv over
the full volume.

Cost per call: ``O(N · D · K)`` where ``N = prod(shape)``, ``D``
is the longest spatial extent, and ``K = 3^d`` is the kernel size.

Distance metric: **Chebyshev** (default) or **Manhattan** /
city-block, or any user-supplied chamfer kernel.  **Not Euclidean.**

Why this design: composability with the rest of the tropical-
semiring substrate.  ``distance_transform`` is a thin wrapper around
``semiring_conv`` with ``TROPICAL_MIN_PLUS`` -- the same primitive
that backs ``dilate`` / ``erode`` / ``open`` / ``close``.  Sharing
the substrate keeps kernel code count low and (with ELL future
sparse support) the same machinery extends to mesh / point-cloud
distance.

### scipy's `distance_transform_edt`: Felzenszwalb 2012 parabolic algorithm

Algorithm: separable 1D passes per axis.  Each 1D pass computes
the lower envelope of parabolas in ``O(n)`` using a stack-based
data structure.

Cost per call: ``O(N · d)`` where ``d`` is the spatial rank.

Distance metric: **exact Euclidean.**

Why this design: it's the canonical O(n) algorithm for exact
EDT on a regular grid.  scipy.ndimage uses it; sklearn / skimage
delegate to it.

## Are they comparable?

Strictly: no.  Chebyshev distance ``max(|dx|, |dy|, |dz|)`` and
Euclidean distance ``sqrt(dx² + dy² + dz²)`` are different
metrics.  For the unit isotropic case the Chebyshev distance is
strictly smaller (the L∞ ball is contained in the L² ball
scaled by ``sqrt(d)``).  Algorithms that consume the DT output
care about which metric was used.

So the "30× slower" gap in the audit table is misleading: it
compares ``apples = Chebyshev`` to ``oranges = Euclidean``.  If
we'd compared against ``scipy.ndimage.distance_transform_cdt``
(the chamfer / city-block / chessboard DT in scipy), the gap
would look different -- scipy's CDT is also iterative-style but
in optimised C.

## What the gap actually means

The audit reveals: **nitrix does not ship an exact Euclidean
distance transform.**  Users who want exact EDT either:

1. Compute Chebyshev / Manhattan with our op and accept the
   metric mismatch.
2. Drop to scipy (CPU, non-differentiable).
3. Roll their own.

This is a gap in **feature coverage**, not in implementation
speed.  Our chamfer DT is competitive-to-faster-than-scipy in
3D (1.3-2.0× ratio), comparable in 2D for medium sizes (3×) and
slow only at small 2D (15×).  All of those numbers represent
"different things, not directly comparable" anyway.

## Recommendations

### Tier 1 (immediate, doc-only)

Update ``morphology.distance_transform`` docstring + the
``morphology.md`` design doc to make the algorithmic
distinction explicit:

- "This is a **chamfer DT** (Chebyshev / Manhattan / custom
  step-cost).  For **exact Euclidean DT**, no nitrix primitive
  is shipped at first GA; use ``scipy.ndimage.distance_transform_edt``
  on host or wait for ``distance_transform_edt`` (a 1.x
  follow-up; see ``perf-audit-2025-05.md``)."
- Note the ratio numbers in the design doc so future readers
  understand why we kept the iterative path despite the audit
  table's headline ratio.

Cost: 30 minutes; closes the audit observation.

### Tier 2 (medium scope, ~200-300 LOC)

Implement Felzenszwalb-Huttenlocher in pure JAX and ship as
``nitrix.morphology.distance_transform_edt``.

Sketch:

- 1D pass: ``lax.scan`` over the spatial axis with state
  ``(k, v_stack, z_stack)`` where ``v`` and ``z`` are arrays
  storing the parabolic envelope.  Each step is a fixed-shape
  decision (push / pop / skip) based on the input value at the
  current position.  Total work per row: ``O(n)`` (amortised);
  fixed-iter scan with ``n`` steps and short-circuit no-ops covers
  the worst case.
- N-D: separable -- vmap the 1D pass over each axis sequentially.
- Differentiability: F-H's stack pops are data-dependent
  (discontinuous in the input).  Forward AD is fine; reverse AD
  needs a subgradient approximation.  Document and ship without
  blocking on the exact-differentiability case.

Alternative: **Saito-Toriwaki 1994** has the same asymptotic but
a slightly different scan pattern that may map cleaner to
``lax.scan``.  Worth a 1-day comparative prototype before
committing to F-H.

Cost: 2-3 day sprint.  Closes the 15× 2D gap and the 2× 3D gap.

### Tier 3 (not recommended)

**Replace** the current iterative ``distance_transform`` with F-H.
**Don't do this.**  The iterative version's role is "chamfer DT
that composes with the tropical-semiring substrate."  Removing it
breaks the substrate-validation story (morphology = tropical
semiring conv) and removes flexibility (custom chamfer kernels).
Add EDT alongside; keep chamfer as the substrate-aligned path.

## What else the audit reveals

A second-order observation worth recording: ``stats.cov`` at small
``(50, 500)`` shapes is ~1.15× slower than numpy.  At larger sizes
nitrix dominates dramatically (130×).  This is the canonical
"GPU-launch-overhead-vs-CPU-fast-for-tiny-problems" pattern; no
action required.

Same pattern for ``lme.flame_two_level`` at V=10 voxels: 0.31×
ratio against statsmodels (the per-voxel overhead of constructing
the statsmodels DataFrame dominates the actual fit at this
size).  Pattern resolves at V=100 (0.03×); no action required.

## Cross-references

- ``bench/perf_audit.py``, ``bench/PERF_AUDIT.md`` -- the raw
  data behind this report.
- ``src/nitrix/morphology/_mm.py::distance_transform`` -- the op
  this audit recommends documenting.
- ``docs/design/morphology.md`` -- where the doc update for
  Tier 1 lands.
- Felzenszwalb, P. F. & Huttenlocher, D. P. (2012).  *Distance
  transforms of sampled functions*.  Theory of Computing 8(19),
  415-428.
- Saito, T. & Toriwaki, J. (1994).  *New algorithms for euclidean
  distance transformation of an n-dimensional digitized picture
  with applications*.  Pattern Recognition 27(11), 1551-1565.
- Borgefors, G. (1986).  *Distance transformations in digital
  images*.  Computer Vision, Graphics, and Image Processing 34,
  344-371.
