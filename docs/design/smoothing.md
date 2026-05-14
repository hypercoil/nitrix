# Smoothing: gaussian, bilateral, susan

> **TL;DR.**  Three smoothers ship at first GA: ``gaussian`` (separable
> n-D, pure JAX, unconditional baseline), ``bilateral_gaussian`` (the
> marquee edge-preserving capability, direct N-body via
> ``semiring_ell_matmul``), and ``susan_emulator`` (convenience
> wrapper composing bilateral + median).  Sectioned ELL ships as a
> sibling in ``nitrix.sparse`` for variable-degree adjacencies.  All
> are pure-JAX; bilateral and susan re-use the existing semiring
> substrate with no new kernel code.

## Gaussian: separable, n-D, scipy-parity

The simplest of the three: convolve along each spatial axis
independently with a 1D Gaussian kernel.  Implementation in
[`smoothing/gaussian.py`](../../src/nitrix/smoothing/gaussian.py)
is ~120 lines.

Two scipy-compatibility quirks worth documenting because they cost
us a debugging round:

1. **Kernel half-width is ``int(truncate * sigma + 0.5)``, not
   ``ceil(truncate * sigma)``.**  Off by one for ``sigma = 0.8,
   truncate = 4`` (where ``int(3.2 + 0.5) = 3`` but
   ``ceil(3.2) = 4``).  We match scipy.
2. **scipy ``mode="reflect"`` is numpy ``mode="symmetric"``.**  scipy
   includes the boundary in the mirror (``a, b | b, a, ...``); numpy
   excludes it (``a, b | a, b, ...``).  We follow scipy convention by
   default; numpy's "reflect" is exposed as ``mode="mirror"`` for
   users who want it explicitly.

For the actual convolution we use ``lax.conv_general_dilated`` rather
than ``semiring_conv`` -- gaussian is REAL-only and benefits from
tensor cores on Ampere+ (TF32 by default).  The separability makes
n-D efficient: ``ndim`` passes of 1D conv, each ~``O(n)``.

### `kernel_size` override (JOSA J.2b)

The ``truncate * sigma`` heuristic produces *odd-only* kernel
sizes (always ``2 * half + 1``).  The JOSA consumer's
``NegativeJacobianFiltering`` needs an explicit **2×2**
Gaussian-weighted average at ``sigma = 0.7``, which the heuristic
cannot reach for any value of ``truncate``.  We exposed an
explicit ``kernel_size`` parameter that overrides the heuristic:

- **Odd** ``kernel_size`` (e.g. ``3, 5, 7``): symmetric kernel,
  output on the same pixel grid as input.
- **Even** ``kernel_size`` (e.g. ``2, 4``): kernel taps at
  half-integer offsets (e.g. ``-0.5, +0.5`` for
  ``kernel_size=2``); **the output is half-pixel-shifted along
  that axis**.  Documented at the public API.  Used for
  Gaussian-weighted local averages where the half-pixel shift is
  intentional (or doesn't matter because the subsequent step
  consumes the smoothed value, not its alignment).

The default ``kernel_size=None`` falls back to the
``truncate * sigma`` heuristic exactly -- regression-tested
against the prior behaviour, so existing consumers see no
change.

The implementation: ``_gaussian_1d_kernel`` builds taps at
half-integer offsets for even sizes via ``arange(-half+0.5, half,
1)`` rather than the standard ``arange(-half, half+1)``.  The
companion ``_conv_1d_along_axis`` switches to asymmetric padding
``(K//2 - 1, K//2)`` for even kernels to keep the output the
same shape as the input.

## Bilateral: the marquee Phase 4 capability

Per SPEC_UPDATE §3.3 the marquee edge-preserving smoother, delivered
at first GA regardless of permutohedral risk.  Mathematically::

    out[i, :] = (1 / Z[i]) * sum_p w[i, p] * values[idx[i, p], :]
    w[i, p]   = exp(-0.5 * sum_d ((feat[i, d] - feat[idx[i, p], d]) / sigma[d])**2)
    Z[i]      = sum_p w[i, p]

The implementation in
[`smoothing/bilateral.py`](../../src/nitrix/smoothing/bilateral.py)
gathers neighbour features, computes Gaussian weights, normalises
per row, and reduces via ``semiring_ell_matmul`` with the REAL
semiring.  All of the heavy lifting is in the existing kernel; the
bilateral file is ~50 lines plus ``brute_force_knn``.

The reduction *is* an ELL matmul: ``weights`` are the per-row values,
``indices`` are the per-row neighbour-list, ``values`` are the dense
operand.  Normalising the weights to sum to 1 (rather than carrying
``Z`` as a separate accumulator state) was the simplest correct
choice; for further-optimised variants where ``Z`` and the numerator
share a streaming pass, we'd need a custom Monoid -- not worth it for
the first cut.

### What "feature space" means here

The bilateral filter weights by *feature* distance, not by *value*
distance.  Features typically include spatial coordinates and one
or more intensity / modality channels; ``values`` is whatever signal
you want to smooth (often intensity itself, but it could be
multi-channel features, segmentation logits, etc.).

For neuroimaging the natural feature space is ``(x, y, z, intensity)``
with ``d_f = 4``.  The spec promises support up to ``d_f ≤ 5`` and
spatial neighbourhoods up to ~7³ voxels, which covers
multi-modal smoothing (e.g., T1 + T2 + diffusion-derived scalar) cleanly.

### The neighbourhood argument

Two cases, per the spec signature:

- **int ``k``**: brute-force k-NN in the rescaled feature space
  (rescaling by ``sigma_features`` so the kNN metric matches the
  weight metric).  Materialises the ``(n, n)`` distance matrix;
  quadratic memory.  Practical for ``n ≲ 10k``.
- **explicit ``(n, k_max)`` adjacency**: use as-is.  This is the
  path for ``n`` larger than brute-force can handle -- the caller
  pre-computes the adjacency via a spatial index (KD-tree, grid
  hashing, atlas parcellation, etc.).

For ``susan_emulator``, we hand-build a spatial-cube adjacency
(``spatial_cube_neighbourhood``) so the bilateral feature space is
spatial + intensity but the adjacency is *fixed* to a spatial cube
around each voxel.  This is the standard SUSAN-style use case and
avoids the O(n²) k-NN.

## Edge preservation in practice

The headline behavioural test: a step image ``[0, 0, ..., 0, 5, 5,
..., 5]`` smoothed with ``sigma_space=2.0, sigma_intensity=0.5``
retains its full 0 → 5 contrast at the step boundary, while the
gaussian smoother flattens it to ~2 → ~3.  Concretely:

| position | raw | gaussian (σ=2) | bilateral (σ_space=2, σ_int=0.5) |
|---:|---:|---:|---:|
| 7 | 0.0 | 2.00 | 0.00 |
| 8 | 5.0 | 3.00 | 5.00 |

The bilateral degrades cleanly to gaussian as ``sigma_intensity``
grows: with ``sigma_intensity = 1e6`` the intensity contribution to
weights vanishes and the result equals a spatial-only Gaussian to
~1e-12 (verified in
``test_bilateral_approaches_gaussian_at_large_sigma_intensity``).

## SUSAN: composition with documented deltas

``susan_emulator`` is the most user-facing op of the smoothing module.
It accepts a raw n-D image (no flatten-to-points dance) and returns a
smoothed image of the same shape.  Internally it:

1. Optionally applies ``median_filter`` (impulse-noise suppression).
2. Builds ``features = (spatial coords, intensity)`` with
   ``sigma_features = [sigma_space] * ndim + [sigma_intensity]``.
3. Builds a spatial-cube neighbourhood (``spatial_cube_neighbourhood``).
4. Calls ``bilateral_gaussian`` with the explicit adjacency.
5. Reshapes back.

Documented behavioural deltas from FSL SUSAN (in the docstring):

- The brightness-similarity weighting that FSL SUSAN does explicitly
  is recovered by including intensity in the bilateral feature space.
- The impulse-noise median fallback is exposed as ``use_median=True``
  (default), applied *before* the bilateral pass.  FSL SUSAN
  integrates this as a single op; we expose the chain because the
  diffprog convention is to compose small primitives.
- The "auto-flat-kernel at small spatial extents" behaviour of FSL
  SUSAN is *not* replicated.  ``sigma_space`` controls the spatial
  weighting directly.
- ``bthresh`` (bilateral threshold) is accepted for API compatibility
  but is currently advisory.  Hard-cutoff variants are easy to add if
  the diagnostic value materialises.

## Sectioned ELL: the variable-degree story

Per SPEC_UPDATE §3.2, sectioned ELL is CORE.  The motivation:

- Distance-thresholded neighbourhoods in irregular point clouds:
  ``k_max`` can be 10-100× ``median(k)``.
- Atlas parcel adjacencies: parcel sizes vary 1-2 orders of magnitude.

The naive ELL layout pads every row to the global ``k_max``, so
those workloads pay catastrophic memory cost.  Sectioned ELL
bucketises rows by ``ceil(log2(degree))`` and runs the kernel once
per bucket with the bucket's *local* ``k_max``.

The implementation in
[`sparse/ell_sectioned.py`](../../src/nitrix/sparse/ell_sectioned.py)
is ~200 lines, no new kernel code:

- **Construction** (``sectioned_ell_from_ragged``) is *host-side*:
  walk the per-row neighbour lists, assign each row to a bucket,
  pad bucket-locally to ``2^bucket``, store original row indices for
  scatter-back.  Not JIT-compatible because the per-bucket shapes
  are data-dependent.  Construction is typically done once per
  adjacency (e.g., once per mesh) and reused across many calls.
- **Matmul** (``sectioned_semiring_ell_matmul``) is a Python-level
  loop over buckets, each running the existing
  ``semiring_ell_matmul``.  Scatter-back uses
  ``out.at[row_indices].set(bucket_out)``.

Storage win: in our test case (degrees ``[1, 3, 8, 2, 5]``) sectioned
ELL stores 23 entries vs the flat ELL's 40 -- a ~2× reduction.  For
real workloads with worst-case degree 10-100× the median, the
reduction is closer to the same factor.

Currently used by ``bilateral_gaussian`` only via the explicit-
adjacency path (we don't auto-section the user's neighbourhood).
Auto-sectioning is a future ergonomic improvement -- the user-facing
API would gain a ``sectioned=True`` flag.

## What we considered and didn't pick

- **A SectionedELL-as-the-only-format API.**  Considered; rejected
  because the uniform-degree case (e.g., a fixed k-NN neighbourhood)
  fits a flat ELL perfectly and adding bucketing overhead is wasteful.
  Both formats live in ``nitrix.sparse``.
- **Permutohedral lattice for bilateral.**  SPEC_UPDATE §3.3 lists it
  as a target with a four-criterion tripwire (parity, perf,
  compile-time, gradient).  We deliberately deferred until the
  bilateral marquee shipped via direct N-body, because bilateral is
  the *capability* promise and permutohedral is the *performance*
  optimisation.  Revisit at 1.x per the tripwire.
- **Streaming Z accumulator in the bilateral matmul.**  We compute
  ``Z`` as a separate ``weights.sum`` pass before normalising.  A
  custom Monoid that carries ``(weighted_sum, Z)`` in its state could
  fuse the two passes.  Not worth the code complexity at first cut;
  the bilateral perf is dominated by the gather + Gaussian-weight
  computation anyway, not by the post-reduction division.
- **Re-exporting ``susan_emulator`` from ``nitrix.morphology``.**
  Considered for discoverability (SUSAN is a noise-reduction op);
  rejected because the re-export creates a circular import
  (``smoothing.susan`` imports ``morphology.median_filter``).  Users
  import SUSAN from ``nitrix.smoothing``, where the spec puts it.

## Cross-references

- SPEC §3.3, SPEC_UPDATE §3.3 -- the smoothing surface.
- SPEC_UPDATE §3.2 -- the sectioned ELL motivation.
- ``src/nitrix/smoothing/`` -- gaussian, bilateral, susan.
- ``src/nitrix/sparse/ell_sectioned.py`` -- sectioned ELL.
- ``tests/test_smoothing.py``, ``tests/test_ell_sectioned.py``.
- [`semiring-protocols.md`](semiring-protocols.md),
  [`ell-on-triton.md`](ell-on-triton.md) -- the substrate that
  bilateral and sectioned ELL specialise onto.
- [`morphology.md`](morphology.md) -- the sibling Phase 4 module.
