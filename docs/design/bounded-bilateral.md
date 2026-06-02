# Bounded bilateral smoothing (and permutohedral retirement)

> **TL;DR.**  The marquee high-dimensional smoother is a **bounded
> bilateral** filter: a true joint bilateral over a bounded-hop
> neighbourhood (grid box, mesh k-ring, later a geodesic ball) with a
> factored feature metric ``M = L Lᵀ`` and decoupled value vs. feature
> channels.  It is one gather plus one weighted reduction on the ELL
> semiring substrate — statically shaped, ``jit`` / ``vmap`` / ``grad``
> clean, GPU-native, and free of any separability artefact.  It *is*
> ``nitrix.smoothing.bilateral_gaussian``, generalised in
> SPEC_UPDATE_v0.4.  The permutohedral lattice is **retired**: bounded
> support dissolves every obstacle that motivated it, and the bounded
> bilateral fills its intended role for the feature dimensionalities we
> target (and, via a low-rank metric, beyond).  This note supersedes
> [`permutohedral-g2.md`](permutohedral-g2.md).

## Why bounded support retires permutohedral

The G2 assessment ([`permutohedral-g2.md`](permutohedral-g2.md)) found
the Adams-2010 lattice structurally unfit for pure JAX: a
dynamic-membership hash table for sparse lattice vertices, neighbour
materialisation during the blur, and a splat whose simplex identity is
piecewise-constant in ``features`` (a gradient discontinuity at simplex
boundaries).  Each obstacle is a consequence of targeting **unbounded**
support in high ``d_f`` — the lattice machinery is precisely what buys
``O(d_f)`` splat cost at unbounded range.

Our use cases accept **bounded-hop** neighbourhoods.  Conceding that
removes the reason the lattice existed.  With a fixed-arity neighbour
index there is no lattice, no hash table, no splat/blur/slice, and no
piecewise-constant simplex selection — only a dense gather over a static
index and a smooth weighted sum.  That operator was already shipped as
``bilateral_gaussian``; SPEC_UPDATE_v0.4 generalises it and removes the
permutohedral stub.

## The operator

Given values ``X ∈ R^{n×d_v}`` and features ``F ∈ R^{n×d_f}`` (decoupled
— the motivating case is ``X`` a BOLD time series, ``F`` a multimodal
signature set):

```
Y_i = ( Σ_{j∈N(i)} w_ij · X_j ) / ( Σ_{j∈N(i)} w_ij )
w_ij = exp( −½ (f_i − f_j)ᵀ M (f_i − f_j) )
```

- ``N(i)`` is a **bounded** neighbourhood supplied as a static neighbour
  source: an ``int`` k (feature-space k-NN), an explicit ``(n, k_max)``
  index array, or an ``ELL`` adjacency.  A boolean ``mask`` nulls padding
  / ragged slots.
- ``M = L Lᵀ`` is a feature metric supplied factored (see below).

The implementation gathers ``F[neighbours]``, forms ``z = (Δf) @ L`` and
``q = Σ z²``, weights ``w = exp(−½ q) · mask``, row-normalises, and
reduces with ``semiring_ell_matmul`` under REAL.  ``n_iters`` re-applies
the (fixed) weights to diffuse values further — a bounded dense-CRF
mean-field update (Krähenbühl & Koltun 2011); the weights are built once
because the affinity is iteration-invariant.

### Why this is GPU-trivial and correct

- **One gather + dense math + one reduction.**  Static shape, immutable,
  smooth gradient everywhere; differentiable w.r.t. ``X``, ``F``, and the
  metric factor ``L`` alike.
- **No separability artefact.**  The joint kernel ``exp(−½ dᵀM d)`` is
  computed directly, unlike a separable approximation that accumulates
  cross-dimension error as ``d_f`` grows.  So the bounded bilateral is
  not merely faster than a separable filter at high ``d_f`` — it is
  *more correct*.
- **Validity masking is a correctness fix, not just ergonomics.**  Ragged
  neighbourhoods are padded by repeating a real neighbour
  (``mesh_k_ring_adjacency`` pads with the first neighbour; the grid box
  clamps out-of-bounds taps to the edge).  Without a mask those duplicate
  indices receive a full Gaussian weight and are **double-counted** — at
  every grid boundary voxel and at every low-degree mesh vertex (e.g. the
  12 pentagonal vertices of an icosphere).  The mask zeroes them.

## The metric ``M`` (the ``FeatureMetric`` ADT)

``M ⪰ 0`` is supplied factored as ``M = L Lᵀ``, ``L ∈ R^{d_f×k}``,
``k ≤ d_f``.  The kernel only needs the projection ``z = Lᵀ d`` and
``q = zᵀz``, so the metric is a small ADT
([`smoothing/metric.py`](../../src/nitrix/smoothing/metric.py)) exposing
``project(deltas)``.  Three tiers, identical kernel:

1. **`DiagonalMetric(sigma)`** — ``M = diag(1/σ²)``; one bandwidth per
   feature.  The cheap, interpretable default; recovers the classic
   per-channel bilateral (the v0.3 ``sigma_features`` argument).
2. **`FactorMetric(L)`, low rank (`k < d_f`)** — correlated channels
   (e.g. DMN sub-network correlations) projected into a ``k≈3–4``
   discriminative subspace before the norm, so the weight costs ``O(k)``
   rather than ``O(d_f)``.  ``L`` can be fit by a consumer from a
   population covariance (``U√Λ``) or learned end-to-end.
3. **`FactorMetric(L)`, full (`k = d_f`)** — general anisotropic metric;
   ``L = chol(M)`` via ``metric_from_spd``.

``block_diagonal_metric(blocks)`` assembles a block-diagonal ``L`` —
independent per-modality bandwidths (a tissue-intensity block weighted
separately from a functional-correlation block), the recommended default
when channels group into distinct modalities.

Both records are registered JAX pytrees (single array leaf), so a metric
traces cleanly and ``L`` is differentiable.  **Data-driven fitting of
``L``** (population PCA, supervised metric learning) is deliberately out
of scope — a modelling concern for a consumer, built from
``nitrix.stats.covariance`` and ``nitrix.linalg``; nitrix ships only the
pure mechanism.

## Why the guided filter was not adopted

A high-dimensional **guided filter** (local linear model
``Y = aᵀI + b`` with a ridge-regularised local covariance inversion) is
GPU-friendly but fits one linear map per spatial window and applies it
uniformly to every site in the window — it produces no per-neighbour
gating weight.  Where two classes interdigitate inside a window it fits
one compromise model to the mixture rather than excluding off-class
neighbours, which is exactly the failure the bounded bilateral avoids.
The one salvageable idea — exploiting feature correlation via a low-rank
projection — is retained, but applied to the bilateral *metric* ``L``,
not to a guided-filter regression.

## Cost and scaling

Per pass: ``O(n · k_max · k)`` for weights + ``O(n · k_max · d_v)`` for
the reduction, where ``k`` is the metric's projected dimension (``≤
d_f``).  ``k_max`` is the cost knob (grid ``r=2, d=3`` → 125; icosphere
2-ring → ~19).  Larger effective support via iteration costs ``O(n_iters
· …)`` — linear in ``n_iters``, versus the super-linear blow-up of
enlarging ``k_max``.

## Domains and the unchanged kernel

The kernel is domain-agnostic; only the host-built neighbour source
differs, and all three already exist in ``nitrix``:

- **Regular grid**, radius-``r`` box — ``spatial_cube_neighbourhood``
  (with a validity mask) and ``sparse.grid.regular_grid_stencil``.
- **Icosphere / mesh k-ring** — ``sparse.mesh.mesh_k_ring_adjacency``
  (ELL; the validity is its padding identity).
- **Geodesic ball** (stretch) — a host-side builder thresholding a
  surface geodesic at radius ``ρ``, emitting the *same* ``(indices,
  mask)`` pair (a larger, padded ``k_max``).  The kernel does not change.

## Pointers

- [`smoothing/bilateral.py`](../../src/nitrix/smoothing/bilateral.py) — the operator.
- [`smoothing/metric.py`](../../src/nitrix/smoothing/metric.py) — the ``FeatureMetric`` ADT.
- [`smoothing/susan.py`](../../src/nitrix/smoothing/susan.py) — SUSAN emulator + grid box.
- [`sparse/mesh.py`](../../src/nitrix/sparse/mesh.py), [`sparse/grid.py`](../../src/nitrix/sparse/grid.py) — neighbour sources (ELL).
- [`permutohedral-g2.md`](permutohedral-g2.md) — the retired lattice's G2 assessment (superseded).
- [`smoothing.md`](smoothing.md) — the smoothing module design.
- SPEC_UPDATE_v0.4 §3.3 — the spec amendment retiring permutohedral.
- Krähenbühl & Koltun 2011 — dense-CRF mean-field reading of the iterate.
- Adams, Baek, Davis 2010 — the permutohedral algorithm retired here.
