# N4 bias-field correction on the accelerator

> **TL;DR.**  ``nitrix.bias.n4_bias_field_correction`` is the Tustison
> (2010) N4 algorithm, GPU-accelerated in pure JAX and numerically
> faithful to ITK / ANTs.  Two reusable primitives carry the weight:
> ``sharpen_histogram`` (the N3 Wiener log-histogram deconvolution -- 1-D
> FFTs) and ``bspline_approximate`` (the Lee--Wolberg--Shin multilevel
> B-spline approximation, **specialised to the regular image grid** so the
> scattered-data fit and reconstruction collapse to separable dense
> per-axis matrix contractions -- no gather, no scatter).  The iteration is
> a ``lax.while_loop`` with ITK's coefficient-of-variation convergence
> test; the multi-resolution hierarchy is a static outer loop doubling the
> B-spline mesh.  Validated to **correlation 1.000000, ~1e-4 relative
> RMSE** against SimpleITK's ``N4BiasFieldCorrectionImageFilter`` on a
> phantom (checked-in golden array + live re-derivation).  No Pallas: the
> hot ops are dense small-matrix ``dot``s and 1-D FFTs, which XLA already
> schedules well.

## Why this lives in nitrix

N4 is exactly the "software in the class of FSL / FreeSurfer / ANTs /
AFNI" the SPEC charter names.  It is a *pure-numeric* algorithm -- array
in, array out, no containers, no sidecar metadata -- so it sits inside the
nitrix contract (SPEC §1, §2.1).  The container-aware wrapper (load a
NIfTI, resample to a mask, write the corrected image) belongs upstream in
``thrux``; nitrix owns the numerics.

It is a *new subsystem* not enumerated in the original SPEC §3/§4.  It was
added under the deviation protocol (IMPLEMENTATION_PLAN §2) in response to
a direct downstream request.  The two sub-primitives are exposed
(``bias.bspline_approximate``, ``bias.sharpen_histogram``) rather than
buried, because both have independent value and that matches nitrix's
"reusable substrate, not feature silo" philosophy (cf. how morphology and
smoothing are specialisations of the semiring substrate).

## The algorithm, and what N4 changed from N3

In the log domain the model is ``v = u + f`` where ``v`` is the observed
log intensity, ``u`` the (unknown) bias-free log intensity, and ``f`` the
smooth log bias field.  N3 (Sled 1998) and N4 (Tustison 2010) share the
same per-iteration *sharpening*: assume the bias spreads the intensity
histogram by a Gaussian point-spread, deconvolve to recover a sharper
intensity distribution, and remap each voxel to its conditional
expectation.  N4's two changes over N3:

1. **B-spline approximation** of the residual field instead of N3's
   smoothing spline (the Lee--Wolberg--Shin scattered-data approximator,
   ITK's ``BSplineScatteredDataPointSetToImageFilter``).
2. A **multi-resolution** hierarchy: fit successively finer B-spline
   meshes, each to the residual the coarser fit left behind.

The driver (``n4.py``):

```
log_input = log(image)             # over the mask
field = 0
for level in range(n_fitting_levels):           # mesh doubles each level
    while not converged and iter < max_iter[level]:
        sharpened  = sharpen_histogram(log_input - field)   # N3 deconv
        residual   = (log_input - field) - sharpened
        field     += bspline_fit_then_reconstruct(residual) # B-spline smooth
        cv         = std(exp(Δfield)) / mean(exp(Δfield))   # ITK's metric
corrected = image / exp(field)
```

Defaults mirror ITK exactly: 4 fitting levels × 50 iterations, 4 control
points per axis at the coarsest level, 200 histogram bins, FWHM 0.15,
Wiener noise 0.01, spline order 3, convergence threshold 1e-3.

## The key idea: the regular-grid B-spline collapses to separable matmuls

The general Lee--Wolberg--Shin MBA is a *scatter* over arbitrary point
positions: each data point splats a weighted contribution onto the
``(order+1)^d`` control points around it.  That is the friction surface
the SPEC §3.2 warns against (scatter/gather adversarial to XLA / Pallas).

But N4's data points are *image voxels*, which lie on a **regular grid**.
On a regular grid the tensor-product B-spline weight from a voxel to a
control point depends only on the voxel's position relative to the control
lattice -- and it **factorises across axes**.  So:

- **Reconstruction** (control lattice -> voxel grid) is a sequence of
  small dense matrix contractions, one per axis, with banded matrices
  ``R_d`` of shape ``(n_vox_d, n_control_d)`` whose rows are the
  ``order+1`` non-zero basis weights.  This is identical in spirit to how
  a 1-D B-spline curve library evaluates ``curve = Phi @ control_points``
  (e.g. `splinex`), generalised to a separable n-D tensor product.
- **The fit** (voxel grid -> control lattice) is the exact adjoint *with
  the Lee--Wolberg--Shin ``w^2`` weighting*.  The per-point normaliser
  ``Σ_j w_j^2`` factorises as ``Π_d (Σ_a R_d[i,a]^2)`` (a per-axis
  profile), and the accumulation onto control points is a per-axis
  contraction with ``R_d^2`` (denominator) and ``R_d^3`` (numerator):

  ```
  φ[k] = Σ_c w_{c→k}^3 z_c / (Σ_j w_{c→j}^2)   ÷   Σ_c w_{c→k}^2
       = (⊗_d R_d^3ᵀ)·(m·z / normaliser)        ÷   (⊗_d R_d^2ᵀ)·m
  ```

Both directions are dense, differentiable, and lower to XLA ``dot`` (the
control grids are tiny -- ≤ 11 per axis at the default finest level -- so
the matrices are tiny).  No gather, no scatter, no
``jax.experimental.sparse``.  A mask enters as the per-voxel confidence
weight ``m``; zero-weight voxels drop out of both sums, and unsupported
control points reconstruct to zero.

### Single-level MBA is biased; the multilevel hierarchy fixes it

A single-level MBA does **not** reproduce even a constant on dense data --
the ``w^2`` averaging carries a systematic bias (~14% on a constant, and
it does not vanish with more control points).  This is inherent to the
Lee--Wolberg--Shin estimator, not a bug, and ITK's
``BSplineScatteredData`` shares it.  N4's multi-resolution hierarchy is
precisely the remedy: refitting the *residual* on a doubling mesh drives
the error down geometrically.  Measured (1-D constant, mesh doubling from
1):

| level | control pts | field max-error |
|---|---|---|
| 0 | 4  | 0.43 |
| 1 | 5  | 0.21 |
| 2 | 7  | 0.097 |
| 3 | 11 | 0.040 |
| 4 | 19 | 0.016 |
| 5 | 35 | 0.0056 |

This is the Lee--Wolberg--Shin multilevel-B-spline convergence theorem,
and it is why we test *multilevel* convergence rather than single-level
reproduction (``tests/test_bias.py::test_multilevel_mba_converges``).

### Equivalence to ITK's control-point-lattice accumulation

ITK accumulates a B-spline *control-point lattice* and, between levels,
refines it by B-spline subdivision.  We accumulate the reconstructed
*field* at full resolution and fit each level's residual on a finer grid.
Because reconstruction is **linear** in the control points,
``reconstruct(L_coarse + ΔL) = reconstruct(L_coarse) + reconstruct(ΔL)``,
so accumulating fields is equivalent to accumulating (and refining) the
lattice -- the reconstructed fields agree up to float.  The field form is
cleaner on the accelerator (no explicit subdivision step; the coarse field
is already a voxel array) and is what makes the multi-resolution loop a
plain static Python loop.

## Histogram sharpening (the N3 Wiener step)

``sharpen_histogram`` builds a 1-D log-intensity histogram (triangular /
Parzen binning), treats it as the true density ``U`` blurred by a Gaussian
``G`` of the given FWHM, and recovers ``U`` by Wiener deconvolution
``Uf = Vf · conj(Gf) / (|Gf|^2 + Z)``.  It then forms the conditional
expectation map ``E[u|v] = ((c·U) * G)(v) / ((U) * G)(v)`` (``c`` = bin
centre) and remaps each voxel by linear interpolation.  All of this is 1-D
FFTs on a static power-of-two buffer, so it JITs cleanly and is cheap.
The bin count, FWHM, Wiener noise, power-of-two zero-pad and Parzen split
mirror ITK's ``SharpenImage`` for parity.

## Why no Pallas

The two hot operations are (a) dense per-axis matrix contractions on tiny
control lattices and (b) 1-D FFTs of length 512.  Both lower to XLA
primitives (``dot`` -- tensor-core eligible -- and ``fft``) that XLA
already schedules near-optimally.  There is no materialised
``(BM, BK, BN)`` tensor to stream, no gather to fuse: the structure that
makes the semiring substrate want a custom kernel is simply absent here.
Per the house "benchmark-first, don't build Pallas speculatively" policy
(BACKLOG B6 / B7), N4 ships pure-JAX; a kernel would be revisited only if a
consumer benchmark showed a wall.  Measured wall time (A10G, jax 0.10.0,
post-compile): ~72 ms / call at 64³, ~533 ms / call at 128³ -- against
ANTs N4's seconds-to-tens-of-seconds on CPU.

## Differentiability

The B-spline fit and reconstruction are differentiable end-to-end (dense
contractions; autodiff flows through w.r.t. both the data and the control
points).  The histogram sharpening is **not** differentiable through the
bin assignment (piecewise-constant ``floor``).  Per the consumer ask,
efficient end-to-end differentiability is a plus, not a requirement, at
this time; a future soft-binning variant of ``sharpen_histogram`` would
close the gap if a consumer needs gradients through the full pipeline.

## Validation

Three layers (``tests/test_bias.py``):

1. **Unit** -- reconstruction vs ``scipy.interpolate.BSpline`` (uniform
   knots, 2.5e-7); partition of unity; multilevel convergence (1-D and
   3-D); mask isolation; differentiability; sharpening vs an independent
   NumPy Wiener reference; JIT parity.
2. **Phantom** -- a concentric-shell tissue phantom with a known smooth
   multiplicative bias: N4 recovers the field (correlation > 0.99, < 2%
   scaled RMSE) and flattens within-tissue intensity variation by ~20-80×.
3. **Golden parity** -- a checked-in SimpleITK
   ``N4BiasFieldCorrectionImageFilter`` output
   (``tests/artefacts/bias/n4_sitk_golden.npz``) the corrected image and
   bias field must match to **correlation 1.000000, ~1e-4 relative RMSE,
   < 1% max pointwise** -- runs with no SimpleITK present.  A second test,
   guarded by ``importorskip('SimpleITK')``, re-derives the reference live.

## What we considered and didn't pick

- **General scattered-data MBA (arbitrary point positions).**  More
  general, exactly ITK's code path, but the scatter is the bottleneck and
  the regular-grid specialisation is exact for the N4 use case (image
  voxels *are* a regular grid).  We kept the general MBA *formula* (the
  ``w^2`` weighting, so we match ITK numerically) but realised it as
  separable matmuls.
- **Least-squares B-spline fit** (reproduces constants exactly, no MBA
  bias).  Diverges from ITK's numerics (would fail golden parity) and the
  masked case is non-separable (needs a CG solve).  The multilevel MBA
  reaches the same place while matching ANTs; correctness without parity
  was not the brief.
- **Clamped / endpoint-interpolating knots** (the choice a curve-fitting
  library like `splinex` makes).  ITK/ANTs N4 uses *uniform* (non-clamped)
  B-splines with ``n_control = n_spans + order``; matching that is what
  gives golden parity.
- **A Pallas kernel.**  See "Why no Pallas".

## Cross-references

- ``src/nitrix/bias/{_bspline,_sharpen,n4}.py`` -- implementation.
- ``tests/test_bias.py`` -- the three validation layers.
- SPEC §1 (charter: "class of ANTs"), §2.1 (pure-numeric contract),
  §3.2 (the no-scatter/BCOO stance the separable form honours).
- IMPLEMENTATION_PLAN §2 (deviation protocol -- new subsystem).
- BACKLOG B6 / B7 (benchmark-first Pallas policy).
- Tustison et al. 2010, "N4ITK: Improved N3 Bias Correction", IEEE TMI.
- Lee, Wolberg, Shin 1997, "Scattered Data Interpolation with Multilevel
  B-Splines", IEEE TVCG.
