# N4 bias-field correction on the accelerator

> **TL;DR.**  ``nitrix.bias.n4_bias_field_correction`` is the Tustison
> (2010) N4 algorithm, GPU-accelerated in pure JAX and numerically
> faithful to ITK / ANTs (correlation 1.000000, ~1e-4 relative RMSE vs
> SimpleITK on a phantom; checked-in golden array + live re-derivation).
> Two reusable primitives carry the weight: ``sharpen_histogram`` (the N3
> Wiener log-histogram deconvolution -- 1-D FFTs) and
> ``bspline_approximate`` (B-spline scattered-data approximation
> **specialised to the regular image grid** so the fit and reconstruction
> collapse to separable dense per-axis matrix contractions -- no gather, no
> scatter).  ``bias_field_correction(method=...)`` is a dispatcher over the
> same N3/N4 iteration with a selectable field estimator: ``'n4'`` (the MBA
> above; parity), ``'least_squares'`` and ``'psplines'`` -- *unbiased*
> estimators for internal use where ANTs bit-compatibility is not required.
> ``n4_bias_field_correction`` stays deliberately pure-N4 (a non-N4
> estimator would mislabel the output).  No Pallas: the hot ops are dense
> small-matrix ``dot``s and 1-D FFTs, which XLA already schedules well.

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

## Parity vs correctness: the field-fit estimators

### The neuroimaging stakes: smooth shading vs sharp anatomy

A bias field (intensity non-uniformity, INU) is a *smooth, low-frequency*
multiplicative shading -- receive-coil sensitivity rolloff, dielectric / B1
effects -- typically a gentle 10--40 % drift across the head over many
centimetres.  What must survive correction is everything *sharp*: the
grey/white boundary, the cortical ribbon, deep-grey nuclei against the
internal capsule, a subtle lesion.  Those anatomical and physiological
contrasts live at millimetre scales and carry the signal every downstream
step (tissue classification, surface placement, registration, volumetry)
relies on.

The failure mode that matters is **contrast washout**: if the estimator
mistakes a piece of *anatomy* for bias, dividing the estimate out erases
real contrast.  If, say, the GM/WM intensity step near a control point
leaks into the field, the "corrected" image flattens grey and white toward
each other -- destroying exactly the separability a segmentation depends
on.  N3/N4 are trusted precisely because they are conservative here.  Two
mechanisms encode the prior "bias is smooth, anatomy is not":

1. *The histogram model.*  Each tissue should form a tight intensity mode;
   the bias *spreads* those modes.  ``sharpen_histogram`` attributes the
   spread to the field and the modes to anatomy -- a genuine tissue
   boundary is a *separate mode*, so it is preserved, while within-tissue
   shading is what gets removed.
2. *The B-spline smoothness scale.*  The control-point spacing is the
   explicit "what counts as smooth" knob: structure coarser than the mesh
   is treated as bias (removed), structure finer than it as anatomy (kept).
   A coarse mesh (N4 starts at one span) cannot represent a sharp edge, so
   anatomy is protected by construction.

The bias-variance tradeoff below is, in these terms, *how readily an
estimator calls sharp residual structure "bias"*:

- **MBA (N4)** is doubly conservative -- the coarse mesh forbids sharp
  fields, and the ``w^2`` local averaging smears any residual
  tissue-boundary structure before it can enter the lattice.  It
  under-fits, which is the *safe* direction (leave a little shading rather
  than eat anatomy); the multi-resolution hierarchy then recovers the
  smooth bias it left behind.  Hence its robustness on brains with many
  boundaries.
- **Unregularised least-squares** drops the averaging safety margin: the
  mesh still forbids *very* sharp fields, but LS spends every control-point
  degree of freedom fitting the residual -- including coherent
  tissue-boundary structure -- and the accumulation loop compounds it.
  This is the corr-0.49, contrast-washing failure quantified below.
- **Ridge / P-spline least-squares** reinstate the smoothness prior
  *explicitly* (penalise field magnitude / curvature) instead of as a
  side effect of averaging: unbiased, yet refusing to absorb sharp anatomy
  -- MBA's protection recovered as an explicit dial (raise ``ridge`` /
  ``penalty`` to keep more anatomy, lower it to chase a more wrinkled
  field).  Mesh spacing and penalty together set the anatomy/bias frequency
  boundary.

Practical guidance follows directly: when in doubt, *under*-correct.  A
residual shading is cosmetic; a washed-out GM/WM boundary is lost data.
That asymmetry is why both the parity default (MBA) and the higher default
``ridge`` for the LS / P-spline loop lean conservative.

### Estimators and the bias-variance mechanics

N4 is a *specific* algorithm.  Its B-spline fit is the Lee--Wolberg--Shin
MBA, which is **biased** on dense data (a single-level fit of a constant is
off by ~14 %; §"single-level MBA is biased").  The bias is harmless for
*parity* -- it is exactly what ITK/ANTs do, and the multi-resolution
hierarchy drives it out -- but it raises the question: would an *unbiased*
fit be better?  We built two and made them selectable, while keeping N4
itself untouched.

**Two entry points, one iteration.**  ``n4_bias_field_correction`` is N4,
full stop (MBA fit, ITK parity).  ``bias_field_correction(method=...)``
dispatches over the *same* N3/N4 iteration (``_core.apply_bias_field_correction``)
with a pluggable per-level fit: ``'n4'`` -> MBA, ``'least_squares'`` ->
weighted LS, ``'psplines'`` -> penalised LS.  We deliberately did **not**
add a ``field_fit=`` switch to ``n4_bias_field_correction``: an LS/P-spline
estimator is a *different* algorithm, and labelling its output "N4" would
mislead.

**The estimators.**  All solve, per fitting level, a control lattice
``phi`` from the residual ``z`` over the mask ``W``:

- **MBA** -- ``phi_k = (sum_c w_{c,k}^3 z_c / sum_j w_{c,j}^2) / sum_c
  w_{c,k}^2``.  Each control point is a *local* ``w^2``-weighted average of
  nearby data -- inherently low-variance (a denoiser), at the cost of bias.
- **least-squares** -- ``(R^T W R + rho I) phi = R^T W z``.  Unbiased; the
  ridge ``rho`` is the variance control.
- **psplines** -- ``(R^T W R + rho I + lambda P) phi = R^T W z`` with ``P``
  a tensor-product difference penalty (Eilers--Marx).  ``lambda`` is a
  roughness knob independent of grid resolution.

All three assemble the Gram ``R^T W R`` *without* materialising ``R`` (the
separable multilinear contraction of §"the key idea"), and the lattice is
tiny, so the dense regularised solve is cheap and is done **once per level**
(the Gram does not depend on the iteration's data) and reused across the
sharpening iterations.  The solve routes through ``linalg._solver.safe_inv``
(CPU fallback on the cuSolver-broken runner, like ``safe_eigh``).

**The load-bearing lesson: regularisation is denoising, not a stabiliser.**
The per-iteration residual ``z = log_uncorrected - sharpened`` is *noisy*
and only weakly correlated with the true bias (corr ~0.36 on the phantom).
The fit has to *extract* the smooth bias from it.  MBA does this for free
(local averaging).  An **unregularised** LS fit instead *overfits* the
residual's non-bias structure -- even at a coarse grid -- and the
field-accumulation loop amplifies it:

| fit of the first residual (corr to true bias) | value |
|---|---|
| residual itself | 0.36 |
| MBA (coarse) | **0.94** |
| LS, ``ridge = 1e-4`` | 0.49 |
| LS, ``ridge = 1e-1`` | **0.97** |

So ``ridge`` (a 0th-order Tikhonov term) -- or the P-spline ``penalty``
(2nd-order) -- is the LS analogue of MBA's ``w^2`` averaging: the
bias-variance knob.  ``bias_field_correction`` therefore defaults ``ridge``
*high* (``1e-1``), whereas the standalone ``bspline_approximate`` (for
*clean* data) defaults it low (``1e-4``).  This was the single biggest
correctness trap: the "obvious" small ridge makes LS *worse* than N4.

**Multi-resolution is required, not optional.**  A single fine-grid fit
(the textbook P-spline setup) *fails* in the loop (corr ~0.1): it captures
non-bias structure from iteration one.  The coarse-to-fine schedule lets
the smooth bias be captured first and only adds detail to the residual of
that.  All three methods use it.

**Where each wins.**  On phantoms, LS / P-splines (with sensible ``ridge``)
are *competitive* with N4 -- better on simple tissue (2-class: LS scaled
RMSE 7e-4 vs N4 1.3e-3), modestly worse on complex tissue with many
boundaries (3-class: LS ~2e-2 vs N4 6e-3), where MBA's local averaging is
more robust to boundary structure in the residual.  They are *alternatives*
with different bias-variance and a tunable knob, **not** a strict upgrade,
and -- unlike MBA -- the fit is differentiable w.r.t. the data (the solve is
linear in the right-hand side).  The optimal ``ridge`` / ``penalty`` is
data-dependent; principled selection is a noted extension (below).

## Extensions (not built)

- **Tier C -- matrix-free scaling + differentiable fit.**  The dense Gram
  inverse is ``O(N_ctrl^2)`` memory; fine ``N_ctrl`` should switch to a
  matrix-free preconditioned conjugate gradient (the matvec is
  reconstruct -> mask -> adjoint, all separable; the unmasked separable
  Kronecker inverse is a near-ideal preconditioner).  Pair with a
  ``custom_vjp`` implicit-diff backward (mirroring ``graph/_lobpcg_diff``)
  for efficient end-to-end gradients.
- **Tier D -- automatic regularisation.**  ``ridge`` / ``penalty`` are the
  P-spline smoothing parameter under different names; GCV or REML selection
  (the standard P-spline machinery) would remove the manual tuning that the
  bias-variance finding above makes necessary.

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
- **Least-squares / P-spline fit as the N4 *default*.**  We built them
  (``method='least_squares'`` / ``'psplines'`` on ``bias_field_correction``)
  but did *not* make either the default or fold them into
  ``n4_bias_field_correction``: they diverge from ITK numerics (would fail
  golden parity) and are competitive-not-dominant (see above).  Shipped as
  opt-in unbiased alternatives, with N4/MBA the parity default.
- **Clamped / endpoint-interpolating knots** (the choice a curve-fitting
  library like `splinex` makes).  ITK/ANTs N4 uses *uniform* (non-clamped)
  B-splines with ``n_control = n_spans + order``; matching that is what
  gives golden parity.
- **A Pallas kernel.**  See "Why no Pallas".

## Cross-references

- ``src/nitrix/bias/{_bspline,_sharpen,n4,_core,correction}.py`` --
  implementation (``n4`` = pure N4; ``correction`` = the dispatcher;
  ``_core`` = the shared iteration).
- ``src/nitrix/linalg/_solver.py`` -- ``safe_inv`` (cuSolver CPU fallback).
- ``tests/test_bias.py`` -- the validation layers (unit / phantom / golden,
  plus the LS / P-spline estimator and dispatcher tests).
- SPEC §1 (charter: "class of ANTs"), §2.1 (pure-numeric contract),
  §3.2 (the no-scatter/BCOO stance the separable form honours).
- IMPLEMENTATION_PLAN §2 (deviation protocol -- new subsystem).
- BACKLOG B6 / B7 (benchmark-first Pallas policy).
- Tustison et al. 2010, "N4ITK: Improved N3 Bias Correction", IEEE TMI.
- Lee, Wolberg, Shin 1997, "Scattered Data Interpolation with Multilevel
  B-Splines", IEEE TVCG.
