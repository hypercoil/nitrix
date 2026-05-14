# Mathematical morphology via the semiring substrate

> **TL;DR.**  ``dilate`` / ``erode`` / ``open`` / ``close`` /
> ``distance_transform`` are thin wrappers over ``semiring_conv``
> with ``TROPICAL_MAX_PLUS`` and ``TROPICAL_MIN_PLUS`` -- ~200 lines
> of glue, zero new kernel code.  ``median_filter`` is a separate
> gather-based op (the canonical example of a neighbourhood
> reduction whose state is unbounded in K).  All ops are bit-exact
> with ``scipy.ndimage`` on the 2D / 3D / 4D test cases.
> Single-channel API per SPEC_UPDATE §3.4 (the natural shape for
> neuroimaging volumes; users with channels ``vmap`` externally).

## Why morphology is the first marquee user surface

We invested heavily in the streaming kernel substrate (SPEC §3.1)
specifically so that downstream user-facing ops could be specialisations
rather than parallel implementations.  Morphology is the cleanest test
of that thesis: every grayscale op is exactly a ``semiring_conv`` with
the right algebra.

| op | algebra | structuring element semantics |
|---|---|---|
| ``dilate`` | ``TROPICAL_MAX_PLUS`` | ``out[i] = max_p (x[i+p] + se[p])`` |
| ``erode`` | ``TROPICAL_MIN_PLUS`` | ``out[i] = min_p (x[i+p] - se[p])`` (we pass ``-se`` so the algebra's ``+`` produces the conventional ``-``) |
| ``open`` | -- | ``dilate ∘ erode`` |
| ``close`` | -- | ``erode ∘ dilate`` |
| ``distance_transform`` | ``TROPICAL_MIN_PLUS`` iterated | Step-cost SE; ``max(spatial_extent)`` iterations to converge |

The ``_mm.py`` implementation is 200 lines, most of which is
structuring-element-shape normalisation and the iterative loop for
distance transforms.  No new Pallas kernel.  No new gradient rule
(the TROPICAL backwards from
[`backward-kernels.md`](backward-kernels.md) compose through).

## API choice: single-channel ``(..., *spatial)``

SPEC_UPDATE §3.4 specifies the user-facing surface takes
``(..., *spatial)`` -- no explicit channel dim.  The reasoning:

- ``scipy.ndimage`` convention is rank-agnostic.  Users porting from
  scipy expect ``ndi.grey_dilation(volume, size=3)`` to "just work"
  on any number of dims.
- The natural neuroimaging shape is 3D volume or 4D volume-and-time,
  with channels (when present) layered on as an outer batch
  concern (subjects, conditions, etc.).
- Channel-aware morphology (treating each channel independently) is a
  ``jax.vmap`` away.  Joint-channel morphology (where the SE has a
  c_in × c_out structure) doesn't have an obvious physical meaning
  for the operations morphology supports.

Internally each op adds a trivial ``c_in = c_out = 1`` dim before
calling ``semiring_conv`` and squeezes it back.  The single
``_conv_wrap`` helper centralises this and keeps the public API surface
clean.

## Distance transform: iterative vs Felzenszwalb-Huttenlocher

The standard distance-transform algorithm on a regular grid has two
implementations:

- **Iterative min-plus conv** with a step-cost structuring element.
  Each iteration propagates the distance one grid step.
  ``max(spatial_shape)`` iterations suffice for convergence.  O(N²)
  in total work for the longest axis ``N``.  Maps trivially onto our
  ``semiring_conv``; we ship this.
- **Felzenszwalb-Huttenlocher** (the "lower envelope" algorithm).
  O(N) per pass, ``ndim`` passes total.  Asymptotically faster but
  involves per-axis sorting and parabola intersection -- significantly
  more code, doesn't fit the semiring substrate.  Deferred to 1.x;
  the function ``distance_transform`` accepts ``metric="euclidean"``
  in its signature only to raise a clear error pointing at this
  deferral.

For the Chebyshev (chessboard) and city-block (taxicab) metrics, the
iterative path *is* the right answer and is bit-exact with
``scipy.ndimage.distance_transform_cdt``.  Verified on 2D and 3D
test cases.

For Euclidean DT, the iterative path with an approximate-Euclidean
SE (``[√2, 1, √2]`` neighbours) converges to an approximate solution;
exact Euclidean DT requires Felzenszwalb-Huttenlocher.  We don't
ship the approximate version because the API contract of "metric =
euclidean" should mean exact, not approximate.

## Median filter: deliberately not a semiring op

SPEC_UPDATE §3.4 makes this point sharply: the true median requires
materialising the full neighbourhood at each output position, because
the state size for a streaming reduction is *unbounded in K*.  For
the small neighbourhoods morphology targets (3×3 = 9 voxels, mesh
k-rings of O(10s)) the materialisation is fine; for general K it
isn't.

We implement ``median_filter`` as ``gather → jnp.nanmedian``:

- Pad the spatial dims with ``NaN`` for SAME-mode boundary handling.
  ``nanmedian`` then skips boundary positions, which is the
  morphological analogue of "ignore values outside the image" without
  needing an algebra identity (median has none in the semiring sense).
- An optional ``structuring_element`` Boolean mask gets folded in via
  ``where(mask, patches, nan)``: excluded positions become NaN and
  the median skips them.

The gather is the same NaN-safe pattern from
[`convolution.md`](convolution.md) (one ``jnp.take`` per spatial dim,
no multiplication, so ``-inf`` and ``+inf`` flow through cleanly).
For morphology we use ``NaN`` padding instead of an algebra identity
because the median *isn't* one of our semiring algebras.

The result: ``median_filter`` lives in ``morphology`` alongside the
semiring-backed ops, but the implementation strategy is different.
This is the prototype for handling "almost a semiring" ops, per
SPEC_UPDATE §3.4 "the split of morphology between semiring-backed
and gather-backed is the prototype for handling the next 'almost a
semiring' op that comes along: don't force it."

## n-D coverage

The semiring conv substrate is rank-polymorphic (the patches
extractor loops over ``spatial_rank``; the inner matmul is rank-2).
Tests exercise:

- **1D** (timeseries): smoke-tested in the conv suite.
- **2D** (images): bit-exact with ``lax.conv_general_dilated`` and
  ``scipy.ndimage.grey_dilation``.
- **3D** (volumes): bit-exact with ``lax.conv_general_dilated`` and
  ``scipy.ndimage`` for dilate / erode / DT.  This is the most
  common neuroimaging shape.
- **4D** (volume + time, fMRI-shaped): bit-exact with
  ``scipy.ndimage.correlate`` for conv.  **Note: cuDNN does not
  support 4 spatial dims**; ``lax.conv_general_dilated`` crashes
  when asked to do 4D conv.  We learned this when writing the 4D
  test and switched the reference to ``scipy.ndimage.correlate``,
  which is generic over rank.

The cuDNN limitation is worth flagging for downstream consumers:
**``semiring_conv`` is more general than ``lax.conv_general_dilated``
for rank**.  This is one of the concrete capabilities the substrate
buys us beyond the algebra surface.

## ``susan_emulator``: a stub with a pointer

Per SPEC_UPDATE §3.3, ``susan_emulator`` is the convenience wrapper
composing ``smoothing.bilateral_gaussian`` (the brightness-similarity
half) with ``morphology.median_filter`` (the impulse-noise half).  The
former is Phase 4 smoothing work not yet landed; until it lands,
``susan_emulator`` raises ``NotImplementedError`` with a pointer at
the alternatives.  This is the same "reserve the namespace; raise
with a clear pointer" pattern as ``permutohedral_lattice`` (SPEC §3.3).

## Differentiability

All semiring-backed morphology ops are differentiable.  Per algebra:

- **``dilate``**: subgradient is one-hot through the argmax neighbour
  per output cell.  Verified in ``test_dilate_gradient_routes_to_argmax``:
  each output cell routes 1 unit of gradient to the position attaining
  the max; the gradient is non-negative and sums to the number of
  output cells.
- **``erode``**: same with argmin.
- **``open``** / **``close``**: composition of the above; gradient
  flows through both passes via the chain rule.
- **``distance_transform``**: differentiable iteratively (each
  ``TROPICAL_MIN_PLUS`` pass has a gradient), but the gradient
  semantics are not particularly useful (you're effectively learning
  the shortest path from each interior point to the boundary).  Not
  blocked, not promoted.
- **``median_filter``**: differentiable via ``jnp.nanmedian``'s
  registered VJP, which routes the gradient to the element attaining
  the median.  This is what the user expects for an N-th order
  statistic.

The gradient story comes "for free" from the substrate: we did not
write a single ``custom_vjp`` for morphology.

## What we considered and didn't pick

- **A multi-channel ``morphology.dilate`` API with ``c_in / c_out``.**
  Considered for symmetry with ``semiring_conv``; rejected because
  the single-channel API matches scipy convention, makes the
  structuring element semantics unambiguous (the SE has the same
  rank as the spatial axes, period), and ``vmap`` is the right
  composition primitive for multi-channel use.
- **Binary morphology via the BOOLEAN semiring.**  Considered; the
  algebra works (``BOOLEAN`` conv would give us OR-of-AND) but the
  BOOLEAN backward raises (not differentiable), and most users
  doing "binary" morphology in neuroimaging are operating on
  threshold-derived masks they want to keep as floats for
  downstream pipeline differentiability.  Currently: pass a bool
  array to ``dilate`` and it gets promoted to float automatically;
  the result is bool-compatible.  We can add a dedicated
  ``binary_dilate`` etc. if the use case materialises.
- **Approximate-Euclidean DT.**  See above; would weaken the API
  contract.  Better to defer to F-H or to a dedicated
  ``distance_transform_approx_euclidean`` if it turns out to be
  needed.

## Cross-references

- SPEC §3.4 ``morphology`` surface; SPEC_UPDATE §3.4 (median filter
  carve-out, semiring vs gather split).
- ``src/nitrix/morphology/`` -- the module.
- ``tests/test_morphology.py`` -- 19 tests including 4D fMRI shape.
- [`convolution.md`](convolution.md) -- the underlying ``semiring_conv``.
- [`semiring-protocols.md`](semiring-protocols.md) -- ``TROPICAL_*``
  algebra definitions.
