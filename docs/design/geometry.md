# Geometry: grid, sphere, coords

> **TL;DR.**  ``nitrix.geometry`` ships three submodules:
> ``grid`` (voxelmorph-style deformable-registration primitives:
> ``identity_grid``, ``spatial_transform``, ``integrate_velocity_field``,
> ``resample``, ``center_of_mass_grid``), ``sphere`` (lat/long
> conversion, geodesic distance, and ``spherical_conv`` re-backed on
> ``semiring_ell_matmul`` -- the third validation of the substrate
> bet), and ``coords`` (centre-of-mass on point clouds, displacement
> regularisers, compactness penalty).  All function names were
> rewritten for clarity; legacy aliases kept for migration and will
> be removed at v0.1 cleanup.

## The renaming pass

This sprint did a "rename for clarity" pass on the legacy
``functional/geom.py`` API before moving anything to the new
location.  Legacy aliases are kept for one minor version, then
removed.

| Legacy name | New name | Why |
|---|---|---|
| ``sphere_to_normals`` | ``latlong_to_cartesian`` | "Normals" is ambiguous; "Cartesian" is the standard. |
| ``sphere_to_latlong`` | ``cartesian_to_latlong`` | Direction of the conversion was unclear. |
| ``spherical_geodesic`` | ``spherical_geodesic_distance`` | Makes it explicit it's a *distance*, not a coordinate transform. |
| ``cmass_regular_grid`` | ``center_of_mass_grid`` | scipy-aligned naming; "cmass" was project jargon. |
| ``cmass_coor`` | ``center_of_mass_points`` | "coor" was unclear; "points" makes the point-cloud semantics explicit. |
| ``cmass_reference_displacement_grid`` | ``displacement_from_reference_grid`` | "Cmass reference displacement" reads backwards in English. |
| ``cmass_reference_displacement_coor`` | ``displacement_from_reference_points`` | Same. |
| ``diffuse`` | ``compactness_penalty`` | "diffuse" sounds like an operation (verb); "penalty" makes the scalar-output semantics explicit. |
| ``vec_int`` | ``integrate_velocity_field`` | "vec int" was opaque; this is the standard scaling-and-squaring SVF integration. |
| ``rescale`` | ``resample`` | "rescale" was ambiguous with intensity rescaling; "resample" is the medical-imaging convention. |
| ``kernel_gaussian`` | dropped | No longer needed; users wanting a Gaussian kernel call ``smoothing.gaussian`` directly, or compute ``exp(-x²/2σ²)`` inline. |
| ``spatial_conv`` / ``euclidean_conv`` | dropped | Legacy ``O(N²)`` all-pairs implementations replaced by ``spherical_conv`` (re-backed on ``semiring_ell_matmul``). |
| ``cmass`` (deprecated) | dropped | Already deprecated; cleanup. |

The shape convention was also rationalised.  ``cmass_coor`` had a
``(*, D, L)`` (channel-leading) coordinate layout that didn't match
the rest of the library.  The new ``center_of_mass_points`` takes
``coords: (..., n_points, n_dim)``, matching ``bilateral_gaussian``,
``brute_force_knn``, etc.  The internal einsum becomes a plain
``weight @ coords`` -- shorter and clearer.

## grid: voxelmorph primitives, written from scratch

Per IMPLEMENTATION_PLAN §6.1 task 3.1, the voxelmorph numerics live
in ``geometry.grid``.  We don't have the original voxelmorph code
checked out alongside this repo, so the primitives are written
fresh based on the standard reference implementations:

- ``identity_grid(spatial_shape)`` -- a coordinate grid where
  ``grid[i]`` equals ``i``.  Adding a displacement field gives an
  absolute deformation consumable by ``spatial_transform``.
- ``spatial_transform(image, deformation, cval=0.0)`` -- warps an
  image by an *absolute* coordinate map (not a relative
  displacement; users compose with ``identity_grid + delta`` to get
  a translation).  Linear interpolation via
  ``jax.scipy.ndimage.map_coordinates`` with constant-fill
  out-of-bounds.  Channel-last layout.
- ``integrate_velocity_field(velocity, n_steps=7)`` -- scaling-and-
  squaring SVF integration (the diffeomorphic exponential map).
  Standard voxelmorph trick: scale ``v`` by ``1 / 2^n_steps`` to a
  step small enough that ``id + v`` is approximately
  diffeomorphic, then double the integration ``n_steps`` times via
  ``phi <- phi + phi(id + phi)``.
- ``resample(image, target_shape, cval=0.0)`` -- evenly-distributed
  bilinear resize, ``align_corners=True`` convention.  Linear ramps
  are exact under round-trip; arbitrary content is blurred (this is
  bilinear, not band-limited).
- ``center_of_mass_grid(weight, axes=None, na_value=None)`` -- legacy
  ``cmass_regular_grid`` with a name aligned with scipy convention.

## sphere: the third substrate-validation

``spherical_conv`` is the marquee Phase 3 task per
IMPLEMENTATION_PLAN §6.1 ("validates the §3.1 design bet end-to-
end").  The legacy implementation in ``functional/geom.py`` had a
clear comment:

> This is implemented in the least clever of all possible ways, but
> it works.

It computed pairwise geodesic distances between all ``n`` points
in chunks (controlled by ``max_bin``), then did an explicit
``weight @ data`` against the entire ``n × n`` distance-Gaussian
matrix.  ``O(n²)`` time, ``O(n²)`` working memory.

The re-backed version:

1. Compute a per-point k-NN adjacency by spherical geodesic.  For
   small ``n`` we use brute-force ``lax.top_k``; for large ``n``
   the user passes the adjacency explicitly (often from the
   icosphere's natural k-ring).
2. For each point, compute geodesic distances *only to its k
   neighbours* and weight by a Gaussian.  ``O(n · k)`` per pass.
3. Normalise weights per row so each output is a convex
   combination of neighbour data.
4. Reduce via ``semiring_ell_matmul`` with the REAL semiring.

That last step is the substrate handle.  We don't write any new
kernel code; the per-point gather and weighted-sum is exactly the
ELL matmul we already have.  ``test_spherical_conv_matches_all_pairs_at_large_k``
verifies bit-equality between the new path with ``neighbourhood=n``
and the all-pairs reference -- the design bet pays off.

This is the third such validation.  Morphology specialises onto
``semiring_conv``; bilateral smoothing specialises onto
``semiring_ell_matmul``; spherical conv now also specialises onto
``semiring_ell_matmul``.  Across these three marquee user surfaces,
*zero new kernel code* was written -- the entire user-facing
behaviour falls out of the existing substrate plus a thin Python
layer.

## coords: the regularisation primitives

``center_of_mass_points`` and friends are common regularisation
hooks: pulling the centre-of-mass of a parameterised weight toward
a fixed reference, or penalising the spatial spread of a weight
around its own centre.

The math:

- ``center_of_mass_points``: ``cm[r, :] = (weight @ coords)[r, :] /
  weight.sum(-1)[r]``.  Reduces to a single ``matmul`` plus a
  divide.
- ``compactness_penalty``: per-region, the weighted mean of
  ``|coords - cm[r]|`` over points (Euclidean or spherical
  distance).  Zero for a delta-weight; grows with spread.  Useful
  as a soft compactness regulariser for atlas / parcellation
  learning.

The ``radius`` parameter on ``center_of_mass_points`` projects the
CM back onto a sphere of that radius -- the centre-of-mass of points
on a sphere generally lies *inside* the sphere, so this projection
recovers a geometrically-meaningful "centre on the sphere".

## What we considered and didn't pick

- **Migrating ``girvan_newman_null``, ``modularity_matrix`` etc.**
  These graph primitives live in ``functional/geom.py`` for
  historical reasons but per SPEC §4.5 they belong in
  ``nitrix.graph``.  Deferred to the Phase 3 graph sprint.
- **Keeping ``spatial_conv`` and ``euclidean_conv`` as separate
  surfaces.**  Both were generic over the metric; the legacy
  implementation took a callable ``metric`` arg.  The new
  ``spherical_conv`` is metric-specific.  For "convolution on an
  irregular point cloud with arbitrary metric" the right answer is
  ``bilateral_gaussian`` (Phase 4) plus a custom metric -- or, for
  REAL semiring, just call ``semiring_ell_matmul`` directly.  We
  do not need a separate top-level surface; the legacy ones are
  dropped.
- **An auto-iterating icosphere generator.**  Useful for users who
  want a regular spherical grid; out of scope for first GA but
  small enough to add later if it doesn't fit naturally elsewhere
  (e.g., as a helper in ``thrux``).
- **Per-channel sigma in ``spherical_conv``.**  Trivial to add but
  not currently in scope; users wanting per-channel smoothing call
  it multiple times.

## The JOSA-feedback sprint additions

Driven by the ``ilex.models.josa`` port (a FreeSurfer
``mris_register_josa`` reimplementation that is the densest
consumer of ``geometry`` / ``smoothing`` / ``morphology`` to
date).  The driving JOSA consumer gap report has since been
retired; the resolved design is recorded here and in
``IMPLEMENTATION_PLAN.md §10.3``.

### `spatial_transform(mode=...)` and `integrate_velocity_field(mode=...)`

The original ``spatial_transform`` hard-coded
``map_coordinates(mode='constant', cval=cval)``.  That is the
right semantics for image sampling but **wrong** for flow-field
warping: when ``integrate_velocity_field`` samples beyond the
support of its previous step's flow, ``constant`` + ``cval=0``
silently substitutes a zero displacement, while voxelmorph
clamps to the edge.  The bug surface is ``O(n_steps)`` voxels
of divergence at every boundary cell; the consumer verified
``max abs diff = 15.18`` against the TF reference on a JOSA SVF.

The fix is a one-keyword pass-through: forward ``mode`` from
``spatial_transform`` to ``jax.scipy.ndimage.map_coordinates``
(which already supports ``'constant' / 'nearest' / 'wrap' /
'mirror' / 'reflect'``).  ``integrate_velocity_field`` **flips
its default to ``mode='nearest'``** because every voxelmorph
consumer needs edge-replicate; the ``constant`` default would
just embed the bug.  This is the only semantics-changing
default change in the sprint and is documented at the public
docstring.

We did *not* ship a ``'linear_extrap'`` mode (the consumer
initially proposed it, then walked back to edge-replicate after
re-reading the TF reference): no current consumer needs it, and
``map_coordinates`` doesn't expose it natively.

### `spatial_transform` now accepts leading batch dims natively

Previously every consumer hit ``vmap(spatial_transform)`` to
batch a warp.  We relaxed the shape contract: the trailing
``(*spatial, c)`` / ``(*spatial, ndim)`` axes are the core; any
leading axes are batch and are vmapped internally.  The two
inputs must agree on the leading-axes shape (no broadcast
attempt -- silent broadcast of warp-and-image is too easy to
get wrong).  Same primitive, no new symbol, one less paper-cut
per consumer.

### `jacobian_displacement` and `jacobian_det_displacement`

Per-point Jacobian and determinant of a deformation
``φ = id + u``.  Central differences along each spatial axis;
boundary handling via ``boundary_mode={'nearest', 'wrap',
'mirror'}`` (default ``'nearest'`` -- the voxelmorph QA
convention).  Anisotropic ``spacing`` is per-axis-configurable
for the common medical-imaging case of non-isotropic voxels.

For the determinant we ship explicit closed-form formulas for
``d ≤ 3`` (avoiding both the ``O(d^3)`` LU factorisation cost
and the precision drift of an iterative det); ``d > 3`` falls
back to ``jnp.linalg.det``.  The 2D case uses the standard
``ad - bc``; the 3D case uses the Rule of Sarrus, summing six
products.  These specialisations are not just micro-optimisation
-- they bypass cuSolver, which we already know has ABI issues on
this Ampere stack (see ``[lobpcg-implicit-vjp.md`` for the same
workaround story).

The HLO audit confirms efficient lowering of the boundary
patches (``field.at[edge].set(field[edge])``): on the largest
shape we measured (``(64, 64, 64, 3)``), the max single tensor
is exactly the size of the Jacobian field, no inflation; the
``.at[].set(...)`` calls compile to ``dynamic-update-slice``
(O(slice_size) per update, six per spatial axis) rather than
full-tensor copies.

### `gaussian(kernel_size=...)`

The ``truncate * sigma`` heuristic that scipy uses produces
*odd-only* kernel sizes (``2 * half + 1``).  The consumer's
``NegativeJacobianFiltering`` step wants an explicit **2×2**
Gaussian-weighted average at ``sigma=0.7``; no ``truncate`` value
reaches that.  We added an explicit ``kernel_size`` override:
odd values stay on-grid; even values produce taps at half-integer
offsets, which shifts the output by half a pixel along that
axis.  Documented at the public API.  The default behaviour
(``kernel_size=None``) is unchanged -- bit-for-bit regression-
tested against the prior scipy-parity path.

### `sphere_grid_pad_2d` (new submodule)

See [`sphere-grid.md`](sphere-grid.md) -- broken out into its
own document because the parameterised-sphere topology is
distinct enough from the rest of ``geometry`` that conflating
them would confuse readers.

### What we explicitly didn't ship

- ``dilate(padding='periodic')``: superseded by composition
  (``sphere_grid_pad_2d`` + ``padding='VALID'`` is 3 lines, more
  general, and reuses the J.1a primitive).  Threading another
  boundary mode through ``lax.conv_general_dilated`` would only
  cover the toroidal case anyway -- not the pole-flip topology
  that the JOSA consumer actually needs.
- A separate ``spatial_transform_batched`` symbol: the
  shape-relaxation approach above is strictly more general at
  the cost of slightly more permissive input validation, which
  we mitigated with a strict assertion on leading-axes shape
  agreement.

## The resampling-method dispatcher

``resample`` and ``spatial_transform`` started linear-only (a wrap of
``map_coordinates(order=1)``).  Standard neuroimaging workflows need a
few more interpolation strategies -- ``Lanczos`` windowed sinc (high-
fidelity, differentiable), ANTs-style ``MultiLabel`` (anti-aliased
segmentation resampling), and plain ``NearestNeighbour`` -- so ``resample``
became the **common dispatcher** over them, with ``spatial_transform``
sharing the same sampler.  See ``docs/design/`` history and
``IMPLEMENTATION_PLAN.md §10.3`` for the as-built record.

### Three orthogonal axes

The design mirrors the ``linalg._eigsolve`` dispatcher's "factor the
independent axes, don't hand-write the cross product" discipline.  A
sample decomposes into three orthogonal axes::

    sample = backend(execution) o kernel(method) o coords(task)

- **coords (task)** -- where the sample positions come from:
  ``resample``'s align-corners resize grid, ``spatial_transform``'s
  deformation field.  Both delegate to one shared
  ``_sample_at_coords`` / ``_resample_on_grid`` seam.
- **kernel (method)** -- how neighbours are weighted: an immutable
  ``Interpolator`` record passed as ``method=``.  Adding a method is
  adding one frozen record; the grid functions are untouched.
- **backend (execution)** -- ``jax`` today; a Pallas pointer-load
  gather is reserved behind the same seam (the parked B7 kernel).

The ``Interpolator`` ADT follows the ``smoothing.metric.FeatureMetric``
precedent (a frozen-record ADT exposing one pure method) -- but the
records carry **static** config (spline order, the label set), so unlike
``FeatureMetric`` they are *not* pytrees: plain frozen, hashable
dataclasses that ride ``jit`` static args / ``custom_vjp`` non-diff slots
(the ``SolverSpec`` pattern).  Boundary handling (``mode`` / ``cval``)
stays a *call* keyword, not a record field.

### The four methods

| method | engine | differentiable | parity |
|---|---|---|---|
| ``Linear`` (default) | gather / ``map_coordinates`` | values + coords | the prior behaviour |
| ``NearestNeighbour`` | gather / ``map_coordinates`` (order 0) | values only (**coord-flat**) | scipy / ANTs ``NearestNeighbor`` |
| ``Lanczos(order=3)`` | explicit separable gather | values + coords | ANTs ``LanczosWindowedSinc`` *class* (not bit-exact ITK) |
| ``MultiLabel(labels=...)`` | per-label inner resample + arg-max | **none** (hard arg-max) | ANTs ``MultiLabel`` |

Differentiability is a *property of the record*
(``differentiable_in_values`` / ``_in_coords``), so the contract is
self-describing and the op-matrix probe reports it honestly.  None of
the paths *error* under ``jax.grad``; the non-differentiable ones return
zero cotangents.

### The explicit separable gather (and where it runs)

All gather kernels lower onto one engine: each axis contributes a few
integer taps and per-tap weights, the boundary mode folds the taps to
in-range indices (the five modes reproduce ``map_coordinates`` exactly
-- ``constant`` masks out-of-range taps to ``cval``, ``nearest`` clamps,
``wrap`` is ``i mod n``, ``mirror`` folds with period ``2(n-1)``,
``reflect`` with period ``2n``), and the ``T**ndim`` corner values are
contracted against the outer product of the weights.  Linear is the
2-tap case (the "explicit 8-corner gather" of B7); Lanczos the ``2a``-tap
case.

This gather is the perf win of B7 -- but **only on GPU**.  The measured
A10G win (~1.5--1.7x over ``map_coordinates``) *inverts* on CPU
(~1.3x slower), where the XLA ``map_coordinates`` lowering is tighter and
CPU interpolation is already the throughput-sensitive path (B15).  So
``Linear`` / ``NearestNeighbour`` select the engine **per platform** --
explicit gather on GPU, ``map_coordinates`` on CPU -- a trace-time branch
(``default_backend_is_gpu``) exactly mirroring ``signal._iir``'s
scan-vs-associative-scan choice.  The two are parity-equal to a ULP, so
it is a pure perf selection; ``_map_coordinates_sample`` doubles as the
CPU engine and the parity **oracle** the tests pin the gather against.
``Lanczos`` has no ``map_coordinates`` equivalent and always uses the
gather.

For ``resample`` the grid is separable (an outer product of per-axis 1-D
coordinate vectors), so wide kernels resample **axis-by-axis** --
``O(N . T . ndim)`` instead of the dense ``O(N . T**ndim)`` corner gather
(``Lanczos`` order 3 in 3-D is ``6**3 = 216`` corners; the 1-D passes are
the only tractable form).  A ``prefers_separable_resample`` flag gates
this: ``True`` for ``Lanczos``, ``False`` for the low-tap ``Linear`` /
``NearestNeighbour`` (whose dense path keeps the CPU ``map_coordinates``
win).  ``spatial_transform``'s coordinates are scattered (non-separable),
so it always uses the dense gather.

### MultiLabel: substrate composition, two constraints

``MultiLabel`` is pure composition over the existing sampler: resample
each label's binary indicator with the ``inner`` kernel, then arg-max
the per-label scores (a memory-bounded running-arg-max fold -- ``O(K)``
inner resamples, never a ``K``-channel stack).  Two constraints fall out
and are documented at the API:

- **Static label set.**  ``jnp.unique`` is dynamically shaped and breaks
  ``jit``; the label tuple is a constructor argument.  Out-of-support
  samples (and ties) resolve to ``labels[0]`` -- list the background
  first.
- **Non-differentiable.**  The arg-max is a hard selection; grad is
  zero.  The inner kernel's *support width* sets the anti-aliasing:
  ``Linear`` (2-tap) refines ``NN`` at junctions but coincides with it on
  a 2x decimation and drops sub-pixel structures; a wider ``Lanczos``
  inner area-weights over a larger neighbourhood and preserves thin
  labels (the role the ANTs Gaussian inner plays -- which we do not ship
  as a separate kernel).

### Governance and what we didn't pick

This is an **extension of an existing primitive** (``resample`` gains a
kwarg), which SPEC_UPDATE_v0.3 §14 explicitly blesses ("prefer adding a
kwarg over forking a function") -- not a new subpackage, not a §12 -> §10.A
graduation.  It is deviation-logged in ``IMPLEMENTATION_PLAN.md §10.3``.

- **A flat ``method: Literal`` + per-method kwargs** (the
  ``bias_field_correction`` idiom): rejected because the methods carry
  divergent config (Lanczos ``order``, MultiLabel ``labels`` / ``inner``)
  and the differentiability contract rides better on the type than on a
  string.
- **A Gaussian inner / a separate cubic B-spline method**: out of scope
  (the ask was Lanczos / MultiLabel / NN).  ``Lanczos`` covers the wide-
  inner role; the order-3 B-spline parity gap stays open
  (``cubic-resample.md``).
- **The external scipy / cupyx backend track (B15/B16)**: a separate,
  heavier, parked research effort; the in-XLA gather (B7) is the one that
  belongs here.  The Pallas pointer-load kernel is reserved behind the
  ``backend`` axis (gated on the B7 trigger).

## Cross-references

- SPEC §4.4 (geometry primitives) and §6.1 task 3 (the migration
  plan).
- ``src/nitrix/geometry/{grid,sphere,sphere_grid,coords}.py`` and
  ``src/nitrix/geometry/_interpolate.py`` (the ``Interpolator`` ADT,
  the explicit separable gather, the per-platform engine selection).
- ``tests/test_geometry.py`` -- including:
  - all-pairs reference parity check for ``spherical_conv``;
  - ``spatial_transform`` mode + batch tests for the J.0 / J.2a fix;
  - ``sphere_grid_pad_2d`` topology tests for the J.1a primitive;
  - ``jacobian_det_displacement`` zero / compression / folding /
    anisotropic-spacing tests for J.1b;
  - the interpolation-method dispatcher: per-mode gather-vs-``map_coordinates``
    parity, platform-engine agreement, Lanczos fidelity / partition-of-unity /
    hand-computed oracle, and MultiLabel label-set / anti-aliasing /
    non-differentiability tests.
- [`semiring-protocols.md`](semiring-protocols.md),
  [`ell-on-triton.md`](ell-on-triton.md) -- the substrate that
  ``spherical_conv`` lowers onto.
- [`morphology.md`](morphology.md), [`smoothing.md`](smoothing.md) --
  the prior validations of the same substrate bet.
- [`sphere-grid.md`](sphere-grid.md) -- the parameterised-sphere
  topology rationale (J.1a).
- ``IMPLEMENTATION_PLAN.md §10.3`` -- the deviation-log record of this
  JOSA-driven sprint (the original ``NITRIX_FEEDBACK_JOSA.md`` gap report
  has been retired).
