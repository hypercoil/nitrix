# sparse.grid + sparse.mesh — ELL specialisations

> **TL;DR.**  ``nitrix.sparse.grid`` ships regular-grid stencil
> constructors (general n-D ``regular_grid_stencil`` plus the
> ``grid_laplacian`` / ``grid_identity`` convenience builders).
> ``nitrix.sparse.mesh`` ships triangle-mesh constructors
> (``icosphere`` for canonical spherical meshes, ``mesh_k_ring_adjacency``
> for k-hop neighbour structure, ``mesh_cotangent_laplacian`` for
> the discrete Laplace-Beltrami operator).  All return ``ELL`` so
> they compose directly with ``semiring_ell_matmul``,
> ``laplacian_eigenmap``, ``relaxed_modularity``, etc.  Construction
> is host-side NumPy (indexing-heavy; no advantage to JAX); the
> resulting ELL is plain JAX from there.

## Motivation

Per IMPLEMENTATION_PLAN §2.A.9: ``grid`` and ``mesh`` are "thin
specialisations of ELL.  Mostly format-conversion code; the heavy
lifting is in ``semiring_ell_matmul``."  The bet: once these are
in place, downstream consumers wanting (a) finite-difference
operators on regular voxel grids or (b) Laplacian-eigenmap-style
embeddings on spherical meshes get the full force of the
substrate without writing any kernel code.

## `sparse.grid`: regular-grid stencils

### The structure

A regular grid (size ``(d_1, d_2, ..., d_n)``) with a fixed
neighbour stencil (offsets ``{o_k}``) has a special structure:
*every voxel uses the same offset pattern relative to its
coordinate*.  So ``indices[v, k] = linearise(coord(v) + o_k)``;
``values[v, k] = w_k`` independent of ``v``.

This is the shift-invariant case.  Storage cost is the same as
general ELL (``O(n_voxels * n_offsets)``) -- we *could* store just
the offsets and recompute indices on the fly, but that breaks the
``semiring_ell_matmul`` API.  The decision: keep the ELL surface
uniform; pay the memory.  At 256³ × 7 = 1.2 GB indices for a 3-D
Laplacian on a high-res volume, this is real -- but voxelwise
analyses at that scale typically work on the masked brain (~50k
voxels), and the indices are reusable across batches.

### Boundary handling

Three modes, all matched to ``scipy.ndimage`` conventions:

- **``'replicate'``** (= scipy ``'nearest'``): out-of-bounds
  index clamped to the edge.  The default; zero-flux boundary
  for the Laplacian.
- **``'periodic'``** (= scipy ``'wrap'``): modular arithmetic;
  toroidal topology.
- **``'reflect'``** (= scipy ``'reflect'``, half-sample
  symmetric, ``d c b a | a b c d | d c b a``): mirror with edge
  repetition.  Note: scipy's ``'mirror'`` (without edge
  repetition) is *not* exposed; the algorithm below covers
  ``scipy.ndimage.laplace(mode='reflect')`` exactly.

The boundary handling lives in ``_linearise_offsets`` -- a pure
NumPy routine that takes ``(n_voxels, n_dim)`` coordinates plus
``(n_offsets, n_dim)`` offsets, applies the boundary fold, and
returns the linearised ``(n_voxels, n_offsets)`` index array.

Verified by parity tests against ``scipy.ndimage.laplace``:

- ``test_grid_laplacian_2d_matches_scipy_replicate`` -- 1.7e-15 at fp64.
- ``test_grid_laplacian_2d_matches_scipy_periodic`` -- ditto.
- ``test_grid_laplacian_2d_matches_scipy_reflect`` -- ditto.

### Anisotropic spacing

``grid_laplacian`` accepts per-axis ``spacing`` for anisotropic
voxel grids (the typical fMRI / dMRI case of ``(3, 3, 4)`` mm
voxels): the per-axis weight is ``1/h_d^2``.  Tested with
``test_grid_laplacian_anisotropic_spacing``.

## `sparse.mesh`: icosphere + adjacency + cotangent Laplacian

### Icosphere construction

Recursive subdivision of the base icosahedron (12 vertices, 20
faces).  Each iteration replaces every triangle with 4 smaller
triangles via midpoint insertion + unit-sphere projection.
Vertex count: ``10 * 4^n + 2`` -- ``n=0`` gives the icosahedron;
``n=7`` gives 163842 vertices (FreeSurfer's ``ico7``).

Implementation is plain Python (recursive midpoint dictionary
for vertex deduplication, list-of-faces accumulation).  Could be
vectorised but the construction is O(n_faces) and you call it
once per mesh; the simple form is readable and fast enough.

### k-ring adjacency

BFS expansion of the 1-ring (edge adjacency) to k-hops.  Returns
an ``ELL`` with binary entries (``binary=True``, default) or
row-stochastic entries (``binary=False`` for random-walk style
operators).

Two checks the tests exercise:

- ``test_icosahedron_one_ring_degree_is_five``: every vertex of
  the base icosahedron has exactly 5 neighbours (the
  topology-defining property).
- ``test_icosphere_subdivided_one_ring_max_degree``: after
  subdivision, the original 12 vertices keep degree 5; the new
  (edge-midpoint) vertices have degree 6.  So ``max degree = 6``,
  ``min degree = 5``.  This is the topological signature of an
  icosphere -- regression-safety against an indexing bug in the
  subdivision step.

### Cotangent Laplacian

The discrete Laplace-Beltrami operator on a triangle mesh:

``(L u)[i] = sum_{j ∈ N(i)} w_ij (u[j] - u[i])``

where ``w_ij = (cot α_ij + cot β_ij) / 2`` and ``α_ij``, ``β_ij``
are the angles opposite edge ``(i, j)`` in the two incident
triangles (or one for boundary edges).

The standard reference is Pinkall & Polthier 1993.  This is what
"discrete Laplace-Beltrami" means in computer graphics and
geometry processing.  Used in:

- Spectral surface analysis (eigenfunctions of L are the surface
  harmonics).
- Heat kernel smoothing on surfaces (``e^{-tL}``).
- Surface registration via harmonic coordinates.
- Functional-MRI surface-based statistical analysis (atlas
  alignment).

### Layout decision: diagonal in column 0

Each row has a diagonal entry (``L_ii = +sum_j w_ij``) plus up to
``max_degree`` off-diagonal entries (``-w_ij``).  We place the
diagonal in column 0 for predictable extraction.  Pad slots use
the first neighbour's index with value 0.

Verified:

- ``test_mesh_cotangent_laplacian_sends_constants_to_zero``: row
  sums are zero (constants in the null space).
- ``test_mesh_cotangent_laplacian_is_psd_smoke``: ``L[v, 0]``
  always indexes ``v`` itself with a positive value.

### Differentiability

The cotangent weights are computed at construction time from the
vertex positions.  If the consumer wants gradients w.r.t. vertex
positions (e.g., for mesh deformation pipelines), they'd
re-construct the L each call -- the dominant cost is the host-
side BFS, not the cotangent math.  The current API stores the
constructed ELL as a fixed object; for end-to-end
differentiability we'd need a JAX-side reconstruction (deferred).

For consumers wanting gradients w.r.t. the **signal** (the
operand of the L-matvec), the ``semiring_ell_matmul`` backward
handles it directly:

- ``test_mesh_cotangent_differentiable`` -- ``jax.grad`` over
  ``trace(L u)^2`` returns finite gradients.

## What we considered and didn't pick

- **Storing only the offsets for grid stencils, recomputing
  indices in the matvec.** Saves ``O(n_voxels * n_offsets)``
  memory.  Rejected because the matvec then needs custom kernel
  code -- breaks the "thin specialisation of ELL" contract.
  If a future consumer hits the memory wall, a separate
  ``GridELL`` class with a custom matvec is the right adapter.
- **JIT-time icosphere construction.** Possible but ugly: the
  midpoint deduplication needs dynamic-shape intermediates that
  don't pattern-match cleanly on JAX.  The host-side build is
  ``O(n_faces)``, called once per mesh; not a bottleneck.
- **Per-mesh diffusion-distance neighbourhood**: would
  generalise ``mesh_k_ring_adjacency`` to include vertices
  within a *geodesic* distance threshold rather than a
  combinatorial k.  Deferred; the k-ring covers the FreeSurfer-
  surface use cases.
- **Mean-value Laplacian / Beltrami-volumetric Laplacian / other
  Laplacian variants.** Cotangent is the standard; adding others
  would be one-function per variant once a consumer asks.

## Cross-references

- ``src/nitrix/sparse/{grid,mesh}.py``.
- ``tests/test_sparse_specialisations.py`` -- 20 tests including
  scipy.ndimage parity for grid stencils and icosphere topology
  invariants.
- ``src/nitrix/sparse/ell.py`` -- the underlying ``ELL`` format.
- [`ell-on-triton.md`](ell-on-triton.md) -- the substrate that
  ``semiring_ell_matmul`` lowers onto when applied to these
  operators.
- [`graph.md`](graph.md) -- the ``connectopy`` / ``laplacian``
  consumers that the mesh constructors feed.
- Pinkall, U. & Polthier, K. (1993). *Computing discrete minimal
  surfaces and their conjugates*. -- the cotangent Laplacian
  reference.
- Loop, C. T. (1987). *Smooth subdivision surfaces based on
  triangles*. -- the icosphere subdivision scheme.
