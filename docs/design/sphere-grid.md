# sphere_grid: parameterised-sphere topology

> **TL;DR.** The 2D equirectangular parameterisation of a sphere
> lives on a regular grid but has non-Euclidean boundary topology:
> the longitudinal axis is periodic, and the latitudinal axis is
> pole-bounded with a "go over the pole" rule that reflects
> vertically AND rolls longitudinally by ``W / 2``.  We ship
> ``nitrix.geometry.sphere_grid_pad_2d`` (plus its
> ``sphere_grid_unpad_2d`` inverse) as the canonical primitive for
> this padding scheme, and rely on **composition** (pad + VALID-
> mode kernel + unpad) to extend any conv / morphology / smoothing
> primitive to spherical-grid topology.  The alternative -- adding
> ``padding='spherical_grid'`` modes to every kernel -- was
> rejected because no single boundary mode in
> ``lax.conv_general_dilated`` captures the pole-flip
> longitudinal-roll topology.

## The topology

For an equirectangular parameterisation of a sphere, latitude θ
∈ [−π/2, π/2] maps to image rows ``0 ≤ h < H`` (with ``h = 0``
and ``h = H − 1`` the pole rows), and longitude φ ∈ [0, 2π) maps
to image columns ``0 ≤ w < W``.

The grid's boundary behaviour:

- **Longitudinal axis (``W``)**: φ = 2π is the same line as φ = 0,
  so the columns wrap circularly.  Padding by ``w_pad`` on the left
  takes the last ``w_pad`` columns; padding on the right takes the
  first ``w_pad``.
- **Latitudinal axis (``H``)**: there is nothing "above" the north
  pole or "below" the south pole.  The spatial neighbour of pixel
  ``(0, j)`` going up across the north pole is ``(1, j + W / 2)``
  -- you cross the pole and land on the opposite longitude.  The
  same rule applies at the south pole.

So a "go over the pole" pad is:

1. Take the rows just inside the pole (``rows 1 : pad + 1`` for
   north, ``rows H − pad − 1 : H − 1`` for south).
2. Flip them vertically (the row closest to the pole goes
   outermost).
3. Roll longitudinally by ``W / 2`` (you crossed the pole; you
   are now on the opposite longitude).

The pole rows themselves (``row 0`` and ``row H − 1``) are **not**
used in the reflection -- they are compressed points on the
sphere, and reflecting them would duplicate them.  This is the
load-bearing topological detail; the JOSA reference paper flagged
it because the wrong version (reflecting the pole row too)
produces a one-pixel artefact at the seam.

The pole roll requires ``W % 2 == 0``; we reject odd ``W`` at the
API rather than silently shifting by half a pixel.

## Sign flip for flow fields

A 2D flow field stored on the equirectangular grid has two
channels: latitudinal flow and longitudinal flow.  When you cross
a pole, longitude reverses direction (you were going east; now
you are going west).  So the **longitudinal-flow channel must
sign-flip** in the pole-pad rows.  The latitudinal-flow channel
does **not** (latitude is continuous across the pole crossing,
just bounded).

Our API exposes this via ``pole_negate_channel`` -- an explicit
index into a named channel axis, defaulting to ``None`` (no sign
flip; scalar image).  We deliberately rejected the consumer's
``is_flow: bool`` proposal: it bundles "this tensor is a flow"
(true / false) with "negate channel 0" (an implicit choice of
which channel).  A tensor with the latitudinal flow in channel 0
would get wrongly negated.  One concept per parameter is cheaper
to reason about, even if the call site is two words longer.

## Why a new submodule

``nitrix.geometry.sphere`` ships **mesh-based** spherical
primitives -- vertex coordinates, k-ring adjacency, geodesic
distance, ``spherical_conv`` on a mesh.  The parameterised-grid
case ("an image at (θ, φ) coordinates on a regular grid") is a
different storage model: regular grid, Euclidean indexing, but
non-Euclidean boundary topology.  Mixing the two under one module
would conflate two mental models that are best kept distinct --
when you reach for ``sphere``, you should know whether you're
working with vertex lists or image grids.

Composition between the two is out of scope here -- it's a
``surfa.SphericalMapBarycentric``-style operation that the
consumer applies upstream of ``nitrix``.

## Composition pattern (instead of new boundary modes)

For sphere-grid convolution / morphology / smoothing, the
canonical pattern is:

```python
padded = sphere_grid_pad_2d(image, pad=kernel_radius)
out_padded = some_valid_padded_kernel(padded)  # e.g. dilate(..., padding='VALID')
out = sphere_grid_unpad_2d(out_padded, pad=kernel_radius)
```

This is **three lines** and works for any kernel that supports
VALID padding -- which is everything in ``nitrix.morphology``,
``nitrix.smoothing``, and ``nitrix.semiring.semiring_conv``.  It
also generalises trivially to other non-Euclidean topologies
(toroidal: ``jnp.pad(mode='wrap')``; reflection-padded: scipy's
modes); each topology gets its own pad primitive and composes
the same way.

We considered the alternative: threading
``padding='spherical_grid'`` through every kernel via a new
boundary-mode argument.  Rejected because:

1. **No single ``conv_general_dilated`` boundary mode captures
   the pole-flip topology.** We'd have to pre-pad anyway, just
   inside the kernel function.
2. **The composition is more general.** Future custom topologies
   would need yet another mode argument; with composition they
   only need a new pad primitive.
3. **It only saves 2 lines of user code.** Compared to the cost
   of new mode arguments on every kernel and their cross-product
   with backend dispatch (Pallas / JAX), it's not worth it.

The morphology and smoothing public docstrings now point at the
composition pattern explicitly; the J.6 entry of the JOSA
consumer report ("``dilate(padding='periodic')``") is satisfied
by the composition.

## Differentiability

``sphere_grid_pad_2d`` is pure ``jnp.concatenate`` / ``jnp.roll``
/ flip / ``.at[].set()`` -- fully ``jax.jit`` and ``jax.grad``
friendly.  The gradient w.r.t. the input is identity-on-the-body
plus the (linear) contributions of the pad cells back to their
source rows.  Tested in ``test_sphere_grid_pad_2d_differentiable``.

## Cross-references

- ``src/nitrix/geometry/sphere_grid.py`` -- the implementation.
- ``tests/test_geometry.py`` -- topology tests (longitudinal
  wrap, pole-skip, full roundtrip, channel-last layout, pole
  negate channel, odd-width rejection, differentiability).
- ``src/nitrix/morphology/_mm.py::dilate`` -- the docstring
  pointer for the composition pattern.
- [`geometry.md`](geometry.md) -- the parent design doc.
- the JOSA consumer request (the ``NITRIX_FEEDBACK_JOSA.md`` gap report
  has been retired; see ``IMPLEMENTATION_PLAN.md §10.3``).
