# Mathematical morphology via the semiring substrate

> **TL;DR.**  ``dilate`` / ``erode`` / ``open`` / ``close`` are thin
> wrappers over ``semiring_conv`` with ``TROPICAL_MAX_PLUS`` and
> ``TROPICAL_MIN_PLUS`` -- ~200 lines of glue, zero new kernel code.
> ``distance_transform`` defaults to **exact Euclidean** DT, computed as a
> separable sequence of per-axis ``TROPICAL_MIN_PLUS`` *matmuls* against the
> squared-distance matrix (``semiring_matmul``) -- still zero new kernel code,
> and on the L4 it matches ``cupyx`` EDT at 64³ and beats scipy on CPU; the
> iterative chamfer conv is the opt-in non-Euclidean path.  ``median_filter``
> is a separate gather-based op (the canonical example of a neighbourhood
> reduction whose state is unbounded in K).  All ops are bit-exact / exact-to-
> round-off with ``scipy.ndimage`` on the 2D / 3D / 4D test cases.
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
| ``dilate`` | ``TROPICAL_MAX_PLUS`` (flat: `reduce_window` max) | ``out[i] = max_p (x[i+p] + se[p])`` |
| ``erode`` | ``TROPICAL_MIN_PLUS`` (flat: `reduce_window` min) | ``out[i] = min_p (x[i+p] - se[p])`` (we pass ``-se`` so the algebra's ``+`` produces the conventional ``-``) |
| ``open`` | -- | ``dilate ∘ erode`` |
| ``close`` | -- | ``erode ∘ dilate`` |
| ``distance_transform`` (euclidean, **default**) | ``TROPICAL_MIN_PLUS`` matmul | per-axis ``out[p] = min_q (g[q] + (q-p)²)`` against the squared-distance matrix; exact EDT |
| ``distance_transform`` (chamfer, opt-in) | ``TROPICAL_MIN_PLUS`` iterated conv | Step-cost SE; ``max(spatial_extent)`` iterations to converge |

The ``_mm.py`` implementation is ~250 lines, most of which is
structuring-element-shape normalisation, the iterative chamfer loop, and the
separable min-plus-matmul EDT driver.  No new Pallas kernel (the EDT reuses the
``semiring_matmul`` kernel; the chamfer path reuses ``semiring_conv``).  No new
gradient rule (the TROPICAL backwards from
[`backward-kernels.md`](backward-kernels.md) compose through both paths).

**Flat-structuring-element fast path.** The substrate thesis still holds, but a
*flat* (all-zero) structuring element -- the common `dilate(size=3)` /
`erode(size=3)` case -- is just a sliding-window max / min, which
``lax.reduce_window`` lowers to a single fused pooling-like kernel.  Routing the
flat case there instead of through ``semiring_conv``'s im2col-patches + matmul
is bit-exact (``SAME`` padding pads with the algebra identity ``-inf`` / ``+inf``)
and, on the L4 at 256², flips `erode`/`dilate` from ~1.3× *slower* than cupy to
~1.3× *faster*, with peak HBM 72 MB → 0.79 MB (no materialised patch tensor).
The non-flat (explicit per-position offsets) path keeps the semiring engine.
``median_filter`` stays on the gather path -- profiling shows the sort, not the
gather, dominates it, so the only lever is a hardcoded sorting network, deferred
pending a fidelity oracle (see its docstring).

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

## Distance transform: exact Euclidean (default) vs chamfer (opt-in)

``distance_transform`` ships two engines, selected by ``metric``:

- **Euclidean (default) -- exact, as a separable min-plus matmul.** This is
  the key result: the 1D squared-EDT along an axis,
  ``out[p] = min_q (g[q] + (q - p)²)``, is *exactly* the tropical (min, +)
  contraction of the per-position cost ``g`` against the squared-distance
  matrix ``D2[q, p] = (q - p)²``.  Reshaping the off-axis dims into a batch of
  lines, each axis pass is one ``semiring_matmul((lines, n), (n, n))`` with
  ``TROPICAL_MIN_PLUS``; squared Euclidean distance separates over axes, so
  composing the passes and taking ``sqrt`` at the end is exact.  Seeds are the
  zero positions of the mask; a finite ``_EDT_BIG`` sentinel marks "no seed"
  (mapped to ``+inf`` at the end if still unreached).  Matches
  ``scipy.ndimage.distance_transform_edt`` to fp32 round-off.
- **Chamfer (opt-in) -- iterative min-plus conv** with a step-cost
  structuring element (``metric="chebyshev"`` / ``"city_block"`` or a custom
  ``structuring_element``).  Each iteration propagates the distance one grid
  step; ``max(spatial_shape)`` iterations converge.  Bit-exact with
  ``scipy.ndimage.distance_transform_cdt`` on 2D / 3D test cases.

The Euclidean default is the principle of least surprise (scipy / cupy / ITK
all mean Euclidean by "distance transform"), and the matmul formulation makes
it both exact *and* fast without leaving the semiring substrate.

### Why min-plus matmul, not Felzenszwalb-Huttenlocher

The classic O(N·d) exact-EDT algorithm is **Felzenszwalb-Huttenlocher** (the
per-axis "lower envelope of parabolas").  We implemented it first -- it is
correct and exact -- but it is **control-flow-bound on the GPU**: the
envelope is a data-dependent stack (push/pop at a dynamic index), and a
pure-JAX rendering (nested ``lax.while_loop`` + dynamic-index scatter under
``vmap``) lowers to a sequence of small kernels that ran **~18 ms at 64³ on
the L4 -- ~80× behind cupy** (though already ~29× faster than the old chamfer
default *on CPU*, where the control-flow penalty is mild).

The min-plus matmul does O(N·n) work per axis (more arithmetic than F-H's
O(N)), but it is **dense, branch-free, and reuses the tuned semiring streaming
kernel** -- so the GPU eats it: **0.24 ms at 64³ (parity with cupy)**, 7 ms at
256³, and it *beats* scipy on CPU (1.8-2× at 3D shapes).  There is no realistic
size where F-H's lower asymptotic work wins back its control-flow constant
before the matmul, so F-H was dropped.  This is the morphology-thesis lesson in
miniature: the right move was not a bespoke EDT kernel but expressing the EDT
in the *existing* semiring algebra.

### Perf characteristics (resolving the 2025-05 audit)

The 2025-05 perf audit (``docs/design/perf-audit-2025-05.md``) measured the
*old* iterative chamfer default against ``scipy.ndimage.distance_transform_edt``
and -- correctly noting the two computed **different metrics** -- positioned the
gap as a "feature coverage gap" (no exact EDT shipped), recommending a separate
``distance_transform_edt`` primitive.  That gap is now **closed**: exact EDT is
the default (with ``distance_transform_edt`` as a scipy-named alias), at parity
with cupy on GPU and faster than scipy on CPU.  See the perf-bench
``PERF_DISTANCE_TRANSFORM`` report for the per-shape numbers.

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
the alternatives.  This is the standard "reserve the namespace; raise
with a clear pointer until the dependency lands" pattern.

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
- **``distance_transform``**: differentiable on both engines -- the
  euclidean default through the ``semiring_matmul`` (min, +) VJP, the chamfer
  path through each ``TROPICAL_MIN_PLUS`` conv pass.  The gradient w.r.t. a
  *binary* mask is structurally ~zero (an infinitesimal mask perturbation does
  not move a seed); the useful gradient is w.r.t. a soft/real-valued cost
  field, where it routes along the shortest path to the boundary.  Not
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
- **Approximate-Euclidean DT** (an ``[√2, 1, √2]`` chamfer SE).  Rejected:
  ``metric="euclidean"`` should mean exact, not approximate.  Moot now that
  the exact min-plus-matmul EDT is both exact *and* fast (see above) -- there
  is no accuracy-for-speed trade left to make.

## Cross-references

- SPEC §3.4 ``morphology`` surface; SPEC_UPDATE §3.4 (median filter
  carve-out, semiring vs gather split).
- ``src/nitrix/morphology/`` -- the module.
- ``tests/test_morphology.py`` -- 19 tests including 4D fMRI shape.
- [`convolution.md`](convolution.md) -- the underlying ``semiring_conv``.
- [`semiring-protocols.md`](semiring-protocols.md) -- ``TROPICAL_*``
  algebra definitions.
