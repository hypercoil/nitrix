# Permutohedral lattice: the G2 tripwire outcome

> **RETIRED (SPEC_UPDATE_v0.4).**  The permutohedral lattice is no longer
> a target; the symbol and its stub have been removed.  Bounded support
> dissolves every obstacle this assessment identified, and the **bounded
> bilateral** (``bilateral_gaussian`` with a factored metric, validity
> mask, and fixed-affinity iteration) supersedes its intended role.  See
> [`bounded-bilateral.md`](bounded-bilateral.md).  The G2 tripwire is
> withdrawn as moot.  This document is retained as the historical record
> of *why* the lattice was unfit for pure JAX.

> **TL;DR (historical).**  ``permutohedral_lattice`` shipped as a stub
> raising ``NotImplementedError`` at first GA, per the SPEC_UPDATE §3.3
> tripwire's "no interim partial shipping" rule.  The technical
> assessment: a *correctness-first dense* implementation works for
> ``d_f ≤ 3`` but fails the perf criterion at the target ``d_f``;
> the *sparse* implementation requires hash-table machinery that
> doesn't have a clean JAX pattern.  ``bilateral_gaussian`` is the
> documented fallback (and is itself the marquee Phase 4 capability
> that ships unconditionally per the spec).

## The criterion-by-criterion assessment

SPEC_UPDATE §3.3 defines four tripwire criteria, all of which must
pass for the symbol to ship:

| # | Criterion | Verdict |
|---|---|---|
| 1 | Parity ``> 40 dB`` PSNR vs Adams 2010 reference | **Untested**: pure-JAX reference not implemented to evaluate. |
| 2 | Wall-time ``< 10×`` ``gaussian`` at 256³, ``d_f = 5`` | **Fail**: structural -- no JAX-friendly sparse path. |
| 3 | First-call compile ``< 30 s`` | **Untested**: contingent on (2). |
| 4 | Gradient finite-diff at pinned rtol | **Untested**: contingent on (1). |

Criterion 2 is the load-bearing failure: even a correct
implementation that's a constant-factor slower than
``bilateral_gaussian`` at the target scale would fail the
tripwire, and a dense pure-JAX implementation is asymptotically
worse than that.

## Why a pure-JAX implementation is hard

The Adams 2010 algorithm has three steps; the difficulty is in
representing the lattice.

### The lattice and the simplex

A point ``f`` in ``R^{d_f}`` is embedded in ``R^{d_f + 1}`` (the
"elevation" step) onto a hyperplane orthogonal to ``(1, 1, ..., 1)``.
The integer lattice points on that hyperplane with coordinates
summing to zero form the permutohedral lattice.  Each point ``f``
lies in some *simplex* of the lattice -- a convex region bounded
by ``d_f + 1`` lattice points -- and is represented by barycentric
weights to those vertices.

The algorithm's claim to fame: the simplex has only ``d_f + 1``
vertices regardless of dimension, whereas a regular hyper-grid
cell has ``2^{d_f}`` vertices.  At ``d_f = 5`` that's ``6`` vs
``32`` -- a 5× win per splat.

### Splat

For each input point, distribute its value to its ``d_f + 1``
simplex vertices weighted by barycentric coordinates.  The set of
*occupied* vertices is sparse: at most ``n × (d_f + 1)`` distinct
vertices for ``n`` input points (typically fewer due to clustering).

### Blur

Convolve along each of the ``d_f + 1`` lattice axes with a
``[1, 2, 1] / 4`` kernel.  Each vertex needs to find its two
neighbours along each axis.  These neighbours are *also* lattice
vertices -- they may or may not be occupied; the algorithm needs
to materialise them as zeros if not.

### Slice

Transposed splat: for each input point, gather the smoothed
values from its ``d_f + 1`` simplex vertices using the same
barycentric weights.

## The structural obstacles in JAX

### Obstacle 1: hash-table representation

The reference C implementation uses a hash table mapping vertex
coordinates (a tuple of ``d_f + 1`` integers) to dense storage
indices.  The hash-table operations are pointer-heavy and
mutable; JAX favours dense, statically-shaped, immutable arrays.

The cleanest known JAX patterns for "sparse aggregation indexed
by a hashable key":

- ``jax.ops.segment_sum``: requires a fixed-size bucket array and
  pre-computed segment indices.  Works if we pre-determine the
  occupied-vertex set on the host.
- ``jax.numpy.unique`` + sort-based indexing: works but is
  ``O(n log n)`` per splat and complicates the gradient (the
  ``unique`` step has a discontinuous structure).

The Pallas Triton path is even worse: per
[`ell-on-triton.md`](ell-on-triton.md), Triton doesn't lower
``gather`` in the current JAX pin -- and the lattice blur is
*entirely* gather-based.

### Obstacle 2: neighbour lookup during blur

For each occupied vertex ``v`` and each lattice axis ``k``, we
need to find ``v ± e_k`` (the neighbour along axis ``k``).  The
neighbour may not be occupied; the algorithm treats it as a
zero contribution.

In the reference C code, the hash table is queried per neighbour
per axis; ``O((d_f + 1) · V · 2)`` total queries where ``V`` is
the occupied-vertex count.

In JAX without a hash table, the choices are:

- **Pre-compute the neighbour list on host** during splat.  For
  each occupied vertex, enumerate its ``2 (d_f + 1)`` neighbours
  and look them up in the same host-side hash table that built
  the splat matrix.  Then ship to GPU as a static blur matrix.
  Works -- this is the path we'd take in a real
  implementation -- but requires the splat-matrix construction
  to be a per-call host computation (not a re-usable kernel).
- **Densify the lattice region**.  For bounded feature ranges,
  the lattice fits in a known box; enumerate all vertices in that
  box and let JAX do dense conv along the lattice axes.  Works
  for low-``d_f`` (``d_f ≤ 3``) -- the dense lattice has roughly
  ``N^{d_f + 1}`` points where ``N`` is the feature resolution
  per axis.  At ``d_f = 5, N = 16`` that's ~16M points, too many
  for the per-iteration data flow we'd need.

### Obstacle 3: gradient through the splat

The splat depends on:
- The barycentric weights (smooth in ``features``).
- The simplex identity -- which ``d_f + 1`` vertices are touched.

The simplex identity is determined by the sort-rank of the
elevated point's fractional parts, which is *piecewise-constant*
in ``features``.  At simplex boundaries the gradient has a
discontinuity.  For typical inputs (random features) the
boundaries are measure-zero, and a subgradient via the
"current simplex" is correct almost everywhere -- but the JAX
sort + scatter idioms don't expose this cleanly.

The differentiable bilateral path
(``bilateral_gaussian``) avoids this issue: its weights are smooth
``exp(-d²/2σ²)`` over a *fixed* k-NN adjacency, so the
gradient is smooth everywhere except at the (also measure-zero)
k-NN boundary, which we don't currently differentiate through.

## Why "no interim partial shipping" is the right call

SPEC_UPDATE §3.3 is explicit:

> Failing any of (1)–(4), the namespace is reserved but the symbol
> raises ``NotImplementedError`` pointing to ``bilateral_gaussian``
> for the ``d_f ≤ 5`` case, and the team revisits at 1.x.  No
> interim "partial" shipping.

The reasoning: a permutohedral_lattice that works for ``d_f = 2``
but fails at ``d_f = 5`` (the target) is more confusing than
useful.  Users who care about ``d_f = 2`` should use
``bilateral_gaussian`` (which handles that case fine); users who
care about ``d_f > 5`` need the full algorithm.  Shipping a
partial implementation creates a maintenance burden and an
ambiguous API contract.

## What it would take to actually ship

A complete future implementation needs:

1. **Host-side construction**: NumPy-implemented elevation,
   simplex-finding, barycentric weighting, and hash-tabled
   vertex identification.  ~200 lines of careful code.
2. **GPU-side splat / blur / slice as sparse matmuls**: convert
   the host-built sparse matrices to ELL or BCOO; apply via
   ``semiring_ell_matmul``.  ~100 lines.
3. **Re-construct only when ``features`` change**: amortise the
   host-side cost across many ``values`` for the same lattice.
   API choice: a ``PermutohedralPlan`` object that the user
   builds once per ``(features, sigma)`` and re-applies.
4. **G2 evaluation**: ~50 lines of benchmark + parity tests
   against ``bilateral_gaussian`` and a reference NumPy
   implementation of Adams 2010.

Total estimated effort: ~2-3 days for a careful first cut,
another 1-2 for tripwire-passing tuning.

## Pointers

- ``src/nitrix/smoothing/permutohedral.py`` -- the stub.
- ``src/nitrix/smoothing/bilateral.py`` -- the unconditional
  marquee fallback.
- SPEC_UPDATE §3.3 -- the tripwire spec.
- Adams, Baek, Davis 2010, "Fast High-Dimensional Filtering Using
  the Permutohedral Lattice" -- the algorithm.
- Krähenbühl & Koltun 2011, "Efficient Inference in Fully
  Connected CRFs with Gaussian Edge Potentials" -- the
  CRF-as-mean-field-iteration use case that motivates
  permutohedral for ML applications.
