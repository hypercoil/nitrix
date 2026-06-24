# Graph: Laplacians, community, connectopy

> **TL;DR.**  Three submodules: ``laplacian`` (graph Laplacians +
> sparse-friendly matvec), ``community`` (modularity-related
> operators with both dense and sparse-factored paths), and
> ``connectopy`` (eigenmap + diffusion-map embeddings with dense
> ``eigh`` and sparse ``lobpcg`` solvers).  Most operations accept
> dense, ELL, and SectionedELL inputs; the eigensolver dispatches
> on format.  LOBPCG is forward-only at first GA (JAX's
> ``lax.while_loop`` is not reverse-mode differentiable); the
> implicit-VJP derivation lives in [`lobpcg-implicit-vjp.md`](lobpcg-implicit-vjp.md)
> as a stretch TODO.

## Why split modularity into ``community.py``

SPEC §4.9 lists ``modularity_matrix``, ``girvan_newman_null``, and
``coaffiliation`` under ``laplacian.py``.  We pushed back: the
*algebraic* shape of these operators overlaps with Laplacians (both
are symmetric, both encode the graph structure linearly) but their
*role* is community detection.  Putting them in ``community.py``
alongside ``relaxed_modularity`` keeps ``laplacian.py`` pure --
just the graph Laplacian variants and the shared matvec
machinery -- and removes a circular-import risk in any future
refactor where the eigensolvers want to import Laplacian
machinery but not community machinery.

The module is otherwise faithful to SPEC §4.9: same surface
functions, same default arguments.

## Multi-format support: where the math admits

The library uses three adjacency representations:

- **Dense** ``(..., n, n)``: the default.  Easy to read, but
  ``O(n²)`` memory.
- **ELL** ``(values, indices)`` with fixed per-row degree
  ``k_max``: lossless for fixed-degree graphs (regular meshes,
  fixed-k k-NN), padded for variable degree.
- **SectionedELL**: variable-degree bucketed by
  ``ceil(log2(degree))``.

Each graph operation supports the maximal set of formats that the
math allows:

| Op | Dense | ELL | SectionedELL |
|---|:-:|:-:|:-:|
| ``degree_vector`` | ✓ | ✓ | ✓ |
| ``laplacian`` (dense matrix) | ✓ | -- | -- |
| ``laplacian_matvec`` | ✓ | ✓ | ✓ |
| ``girvan_newman_null`` | ✓ | (rank-1; output is dense) | -- |
| ``modularity_matrix`` (dense) | ✓ | -- | -- |
| ``modularity_matrix_matvec`` | ✓ | ✓ | ✓ |
| ``coaffiliation`` | ✓ | -- | -- |
| ``relaxed_modularity`` | ✓ | ✓ | ✓ |
| ``laplacian_eigenmap`` | ✓ (eigh) | ✓ (lobpcg) | ✓ (lobpcg) |
| ``diffusion_embedding`` | ✓ (eigh) | ✓ (lobpcg) | ✓ (lobpcg) |

The matrix-producing operations (``laplacian``, ``modularity_matrix``,
``coaffiliation``) are dense-only by definition: the normalised
symmetric Laplacian's diagonal is non-zero, and the modularity
matrix is ``A - rank-1`` which is dense regardless of ``A``'s
format.  When the user wants the *operator* (e.g., for an
eigensolver), ``..._matvec`` versions apply the implicit operator
to a vector without materialising the dense intermediate.

The sparse-aware paths use the registered VJPs of
``semiring_ell_matmul`` and ``sectioned_semiring_ell_matmul``, so
gradients flow through ``relaxed_modularity`` on sparse adjacencies
without extra work.

## ``relaxed_modularity``: the factored sparse path

The dense path is faithful to Newman 2006:

    Q = sum_{i,j} (A_ij - gamma * k_i k_j / 2m) * (CC^T)_{ij}

For large sparse ``A`` this is expensive (the modularity matrix
``B`` is ``n²`` dense even when ``A`` isn't).  The factored form
sidesteps it:

    Q = sum_{i,j} A_ij (CC^T)_{ij} - (gamma / 2m) * (C^T k) · (C^T k)
      = trace(A · CC^T) - (gamma / 2m) * |C^T k|²

Both terms are sparse-friendly:

- ``trace(A · CC^T)``: ``AC`` is one ``semiring_ell_matmul``
  call, ``(n, k_comm)`` output; then ``trace = sum(C * AC)``.
  Total cost ``O(nnz · k_comm + n · k_comm)``.
- ``|C^T k|²``: ``C^T k`` is a single ``(k_comm,)``-shaped
  inner-product reduction; squared norm is ``k_comm`` flops.

For a 100k-node mesh with ``k_comm = 16``, this is the difference
between "runs in a few ms" and "OOM / hours".  The factored path
is verified to match the dense path bit-for-bit at fp64 on test
graphs.

The ``exclude_diag`` semantics survive the factorisation via a
diagonal-correction subtraction (see the implementation
docstring); for graphs without self-loops the correction is zero
and the path is exact.

## Connectopy: ``eigh`` vs ``lobpcg``

Three load-bearing reasons to support both:

1. **Scale**.  ``jnp.linalg.eigh`` is ``O(n³)`` time / ``O(n²)``
   memory.  At ``n = 10k`` that's seconds and ~1 GB on GPU; at
   ``n = 100k`` it's infeasible.  LOBPCG is matrix-free
   (``O((nnz + n·k²) · iters)``) and scales to ``n = 1M+``.
2. **Format**.  ``eigh`` requires a dense matrix.  ELL and
   SectionedELL inputs *must* go through ``lobpcg`` because
   materialising the dense Laplacian defeats the point.
3. **Spectrum coverage**.  ``eigh`` returns the full spectrum;
   ``lobpcg`` returns only the top ``k``.  For diagnostic /
   spectral-gap analysis the full spectrum can be useful, hence
   the explicit ``solver="eigh"`` option even on graphs where
   ``lobpcg`` would scale better.

The ``solver="auto"`` default routes by input format: dense ->
``eigh``, sparse -> ``lobpcg``.

### The LOBPCG "smallest eigenvalues" trick

LOBPCG natively returns the *largest* eigenvalues of a symmetric
operator.  We want the *smallest* non-trivial eigenvalues of the
normalised Laplacian ``L_sym = I - M`` where ``M = D^(-1/2) A
D^(-1/2)`` is the symmetric affinity.  The trick:

    lambda_L = 1 - lambda_M

So the smallest ``k+1`` eigenvalues of ``L_sym`` correspond to the
*largest* ``k+1`` eigenvalues of ``M``.  We compute the latter via
LOBPCG, drop the trivial ``lambda_M = 1`` (constant eigenvector),
and transform back.  Cleaner than shift-and-invert iterations.

For diffusion embedding the eigenvalues of the diffusion operator
``M_alpha`` are *also* in ``(-1, 1]`` with largest at 1, so the
LOBPCG fit is direct: largest ``k+1``, skip trivial.

### The LOBPCG constraint

``lobpcg_standard`` requires ``5 · search_dim < n``.  For the
default ``skip_trivial=True``, ``search_dim = n_components + 1``.
On small graphs this fails (``n=16, n_components=2`` needs
``search_dim=3`` and would need ``n > 15`` -- borderline).  The
docstring documents the constraint; for tiny graphs the user
should use ``solver="eigh"`` instead.

### Device preservation under the cuSolver workaround

JAX's ``eigh`` and ``lobpcg_standard`` both internally call
cuSolver-backed QR / Cholesky on GPU.  We've seen environments
where cuSolver is broken (ABI mismatch between the linked cuSolver
library and the driver -- e.g. CUDA 12.8 libs against a CUDA 13.0
driver).  ``_eigh_device`` probes once per process; if GPU eigh
fails it routes both eigh *and* lobpcg to CPU.

For inputs that lived on GPU, we then move them to CPU, run the
solver, and move the result back.  This is transparent to the
user -- you don't see a surprise CPU array on a GPU-only pipeline
-- but is documented for the perf-conscious (the round trip
adds latency).

A ``test_connectopy_returns_outputs_on_source_device`` test pins
this invariant.

## LOBPCG differentiability

``lobpcg_standard`` is **not** reverse-mode differentiable as
shipped: the iterative solver uses ``lax.while_loop`` with a
data-dependent termination criterion, which JAX cannot lower
backward.  ``jax.grad`` over a function that goes through the
``lobpcg`` path raises a native JAX error about while-loop
dynamism.

The fix is implicit differentiation through the eigenvalue
equation; the math sketch is in
[`lobpcg-implicit-vjp.md`](lobpcg-implicit-vjp.md).  Tracked as a
stretch TODO because the use case (analysis primitives, not
training operators) is typically forward-only.  Users who need
gradients on small dense graphs use ``solver="eigh"``, which
ships with a registered VJP.

## What we considered and didn't pick

- **Re-using ``jax.experimental.sparse`` BCOO**.  The SPEC §4.2
  explicitly forbids this; ``semiring_ell_matmul`` is the
  sanctioned sparse path.  Internally, ``laplacian_matvec`` and
  the eigensolver matvec closures route through it.
- **Auto-densification for small graphs**.  Considered making
  ``solver="auto"`` switch to ``eigh`` for any input below some
  ``n`` threshold (regardless of format).  Rejected: the user's
  format choice *is* their solver hint.  An ELL input with
  ``n=100`` says "I'm treating this as sparse"; we honour that.
  Users wanting dense eigh for a tiny ELL densify themselves.
- **Eigenvector sign normalisation**.  Eigenvectors are defined
  up to sign; tests assume the cluster-recovery property is
  symmetric in sign.  We don't impose a sign convention; users
  needing one apply it themselves.
- **Approximate Euclidean DT-style fast diffusion**.  The
  diffusion-map literature has parametric approximations (e.g.,
  the "diffusion wavelets" hierarchy) that compute embeddings
  faster but with approximation error.  Out of scope; the LOBPCG
  path gives exact top-``k`` already.

## Cross-references

- SPEC §4.9 -- the graph subsurface.
- SPEC §6.1 tasks 3.5-3.7 -- the migration plan.
- ``src/nitrix/graph/{laplacian, community, connectopy}.py``.
- ``tests/test_graph.py`` -- 34 tests including the dense-vs-sparse
  agreement checks and the device-preservation invariant.
- [`lobpcg-implicit-vjp.md`](lobpcg-implicit-vjp.md) -- the
  differentiability TODO.
- [`ell-on-triton.md`](ell-on-triton.md) -- the underlying ELL
  story.
- [`semiring-protocols.md`](semiring-protocols.md) -- the kernel
  substrate the sparse paths lower onto.
