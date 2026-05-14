# LOBPCG: the implicit-VJP path

> **Status.**  Reverse-mode AD through ``laplacian_eigenmap`` /
> ``diffusion_embedding`` with ``solver="lobpcg"`` is **shipped for
> dense and flat-ELL** operands via an implicit-VJP wrapper around
> ``jax.experimental.sparse.linalg.lobpcg_standard``.  The wrapper
> implements the subspace-projector approximation (option 1
> below): exact for losses that depend only on eigenvalues or on
> in-subspace functionals of the eigenvectors, biased for losses
> that depend on the orthogonal complement of the returned top-
> ``k`` subspace.  SectionedELL remains forward-only -- convert to
> flat ELL for gradients -- because we don't have a sparsity-
> projected backward for the bucketed format yet.  This document
> records the math, the implementation choices, the validation,
> and what would be needed to ship the exact (Krylov-corrected)
> path.

## The math

Forward: given symmetric ``A: (n, n)`` and initial guess ``X_0: (n,
k)``, lobpcg returns ``(U: (n, k), Λ: (k,))`` such that approximately
``A U = U diag(Λ)`` with ``U^T U = I_k`` (the top-``k`` eigenpairs).
For partial decompositions (``k < n``), the columns of ``U`` span a
``k``-dimensional invariant subspace of ``A``.

The implicit differentiation strategy: differentiate the *eigenvalue
equation* directly rather than the iterative solver.

### Eigenvalue gradient (Hellmann-Feynman, exact)

For each eigenpair ``(λ_i, u_i)`` of symmetric ``A`` with ``u_i^T u_i
= 1``, the eigenvalue gradient is:

    ∂λ_i / ∂A = u_i u_i^T

This is exact and requires no full-spectrum information.  Stack
across ``i`` and weight by upstream cotangents to get the eigenvalue
contribution to ``∂L/∂A``:

    (∂L/∂A)_{eigenvalue} = sum_i (∂L/∂λ_i) u_i u_i^T = U diag(g_λ) U^T

where ``g_λ = ∂L/∂Λ`` is the cotangent on the eigenvalues.

### Eigenvector gradient (the hard part)

The eigenvector gradient is governed by the standard "F-matrix"
formula.  For the *full* symmetric eigendecomposition ``A = V diag(λ)
V^T`` with ``V: (n, n)`` and ``λ`` the eigenvalues:

    (∂L/∂A) = V diag(g_λ) V^T + V (F ⊙ S) V^T

where ``F[i, j] = 1 / (λ_j - λ_i)`` for ``i ≠ j`` and ``0`` on the
diagonal, ``S = (V^T g_V - g_V^T V) / 2`` is the *antisymmetric*
part of the projected eigenvector cotangent, and ``g_V = ∂L/∂V``.
This formula is what ``jnp.linalg.eigh``'s registered VJP uses.

The antisymmetrization is load-bearing: only the antisymmetric part
of ``V^T g_V`` contributes, because the eigenvectors are defined up
to sign and only the rotation between them is observable.

For a *partial* decomposition with only the top ``k`` eigenpairs in
``(U, Λ)``, we don't have the remaining ``n - k`` eigenvalues to
build the full ``F`` matrix.  Two options:

#### Option 1: Subspace-projector approximation (shipped)

Drop the eigenvector gradient contribution from the orthogonal
complement of the top-``k`` subspace.  Within the top-``k`` subspace,
build ``F: (k, k)`` using only the known eigenvalues:

    F_kk[i, j] = 1 / (Λ_j - Λ_i) for i ≠ j in 1..k, else 0

Then:

    (∂L/∂A) ≈ U diag(g_λ) U^T + U (S_kk ⊙ F_kk) U^T

with ``S_kk = (U^T g_U - g_U^T U) / 2``.  The error is in the
contribution of ``(I - U U^T) g_U`` -- the part of the eigenvector
cotangent that lies outside the top-``k`` subspace.  For *purely*
in-subspace losses (functionals of the eigenvalues; quadratic forms
over the embedding; spectral-clustering objectives like
``trace(U^T M U @ T)``), this term is zero and the approximation is
exact.  For losses that depend on the full eigenvectors at high
frequencies (the discarded part of the spectrum), the approximation
is biased -- typically biased *low* in magnitude, since the dropped
contribution adds to the gradient rather than cancelling.

This is what KeOps and most "differentiable spectral methods"
papers use.  It is the right default.

The shipped wrapper in ``src/nitrix/graph/_lobpcg_diff.py``
implements exactly this.

#### Option 2: Iterative correction via the orthogonal-complement Hessian

Solve the orthogonal-projection equation:

    (A - Λ_i I) (∂u_i / ∂A) = (I - U U^T) (∂A) u_i

via a Krylov solver (CG / GMRES) for each eigenvector.  This recovers
the missing contribution exactly but costs ``O(k * matvec_cost *
krylov_iter)`` per backward pass.  Reasonable for ``k`` small and
``matvec_cost`` low (sparse adjacency); expensive otherwise.

Not shipped at first GA -- option 1 is sufficient for the analysis
use cases nitrix targets (Laplacian-eigenmap, diffusion-map
embeddings, modularity-style spectral clustering).  Revisit if a
downstream consumer needs exact gradients through eigenvectors for
training-style optimisation.

## Implementation details

### Public surface

``src/nitrix/graph/_lobpcg_diff.py`` ships two ``jax.custom_vjp``
wrappers:

- ``lobpcg_top_k_dense(M, X0, n_iters, tol, eps_clamp)`` -- ``M``
  dense ``(n, n)``, diff w.r.t. ``M``.  ``dM`` is the dense ``(n,
  n)`` matrix from the formula above, symmetric by construction
  (``diag(g_λ)`` is symmetric, ``S ⊙ F`` is symmetric because both
  factors are antisymmetric in their ``(i, j)`` indexing).
- ``lobpcg_top_k_ell(values, indices, X0, n_cols, n_iters, tol,
  eps_clamp)`` -- ``M`` stored as ELL ``(values, indices)``; diff
  w.r.t. ``values`` only.  The dense gradient ``U K U^T`` (with
  ``K`` the ``k x k`` core matrix) is **projected onto the existing
  sparsity pattern**: ``g_values[i, p] = (U K U^T)[i, indices[i,
  p]] = (U K)[i, :] @ U[indices[i, p], :]``.  Cost ``O(nnz * k + n
  * k^2)``, consistent with the forward.

Both share the ``_subspace_vjp_kernel`` helper that builds ``K =
diag(g_λ) + S ⊙ F`` -- the only expensive piece, ``O(k^2)``.

### Near-degeneracy clamp

``eps_clamp`` (default ``1e-8``) floors the F-matrix denominator:
pairs ``(i, j)`` with ``|Λ_i - Λ_j| < eps_clamp`` get ``F[i, j] =
0`` rather than ``1 / 0``.  The eigenvector gradient at those pairs
is undefined anyway -- the rotation within a degenerate subspace is
not observable -- so dropping the contribution is correct.

For the connectopy use cases:

- Laplacian eigenmaps of well-mixed graphs have distinct
  eigenvalues; the clamp never fires.
- Symmetry-induced degenerate pairs (e.g., a regular ring's matched
  ``cos / sin`` eigenvectors) trigger the clamp; the gradient is
  zero within the degenerate pair, which is the correct
  "operationally undefined" answer.
- Disconnected graphs have a fully-degenerate zero subspace; the
  ``skip_trivial`` flag pre-removes one trivial eigenvector but
  multiple connected components still produce a multi-dimensional
  zero subspace.  Use ``eps_clamp`` slightly above the eigenvalue
  spread of that subspace to suppress the rotation gradient.

### Sign / order conventions

``lobpcg_standard`` returns eigenvalues **largest-first** with
matching eigenvector columns.  The custom_vjp preserves this; the
backward formula is invariant to ordering (just relies on the
returned ``(Λ, U)``).  Sign flips between runs are also handled --
the F-matrix term involves ``U^T g_U`` which transforms covariantly
under ``U -> U S`` for diagonal sign ``S``.

### Integration with connectopy

``src/nitrix/graph/connectopy.py`` builds the symmetric affinity
operator ``M = D^(-1/2) A D^(-1/2)`` (for ``alpha = 0``) or its
Coifman-Lafon variant (for ``alpha > 0``) **explicitly** -- not as
a matvec closure -- via ``_build_affinity_operator``.  Dense
returns a dense ``M``; ELL returns an ELL ``M`` with the same
indices and rescaled values.  Then ``_lobpcg_top_k`` dispatches
on the operator's type:

- Dense ``M`` -> ``lobpcg_top_k_dense`` (diff path).
- ELL ``M`` -> ``lobpcg_top_k_ell`` (diff path, sparsity-projected
  gradient).
- SectionedELL ``M`` -> closure-based ``lobpcg_standard`` call,
  forward-only; ``jax.grad`` raises the native while-loop error.

The skip-trivial / sort / ``mu -> 1 - mu`` post-processing happens
in plain JAX so the chain rule flows through naturally.  For dense
adjacency ``A`` the chain rule via ``M = D^(-1/2) A D^(-1/2)``
returns gradients in ``A``-space; for ELL ``A`` it returns
gradients in ``A.values``-space (the only differentiated leaf).

## Validation

### Correctness

All in ``tests/test_graph.py``:

- ``test_lobpcg_dense_eigenvalue_grad_matches_eigh`` -- eigenvalue
  gradient matches ``jnp.linalg.eigh``'s VJP to ``1e-12`` at fp64.
- ``test_lobpcg_dense_in_subspace_loss_matches_analytical`` -- for
  the loss ``trace(U^T M U @ T)`` where ``T: (k, k)``, the analytical
  gradient ``U diag(diag(T)) U^T`` is matched to ``1e-12``; the F-
  matrix correction exactly cancels the off-diagonal of ``T``.
- ``test_lobpcg_dense_gradient_is_symmetric`` -- the gradient lives
  in the symmetric subspace, matching the symmetric-eigh convention.
- ``test_lobpcg_dense_degenerate_eigenvalues_clamp`` -- near-
  identity ``M`` (fully-degenerate spectrum, slightly perturbed)
  produces a finite, ``~zero`` gradient under the clamp.
- ``test_lobpcg_ell_grad_matches_dense_projected`` -- the ELL
  gradient equals the dense gradient gathered at the ELL indices.
- ``test_laplacian_eigenmap_lobpcg_{dense,ell}_differentiable`` --
  the public surface routes through the diff wrapper end-to-end.
- ``test_lobpcg_sectioned_ell_not_differentiable`` -- SectionedELL
  remains forward-only as documented.

### Performance and XLA shape preservation

``bench/perf_lobpcg.py`` measures forward, forward+backward, and
the **pure backward kernel in isolation** (no LOBPCG iteration in
the path).  The motivating concern: XLA fusion passes can defeat
a hand-written sparsity projection by materialising an ``O(n^2)``
intermediate at compile time.  The bench audits the compiled HLO
to detect this directly.

Frozen result on A10G fp32 (``bench/PERF_LOBPCG.md``):

- **No ``(n, n)`` tensor in the compiled ELL backward at any
  measured ``n`` up to ``65536``.**  The audit greps the compiled
  HLO for tensors with two axes equal to ``n`` and reports the
  count; this is zero for every ELL row.
- **Pure ELL backward wall-time is roughly constant at
  ``~165 µs`` from ``n = 256`` (1.3k nnz) to ``n = 65536`` (327k
  nnz).**  At these sizes the cost is kernel-launch limited, not
  compute limited; the gather + ``(nnz * k)`` einsum is small
  relative to the dispatch overhead.
- **Pure dense backward scales as expected ``O(n^2)``** -- 148 µs
  at ``n = 256``, 2.4 ms at ``n = 16384``.  The gradient *is*
  ``(n, n)`` for the dense path so this is correct, not a
  regression.
- **End-to-end LOBPCG wall-time is dominated by the iteration**,
  not the backward.  The ``bwd_only`` column (grad - forward)
  stays under 600 µs for ELL across all sizes.

The audit is automated: ``_audit_hlo`` in the bench script grepss
the compiled HLO for ``f32[..., n, n, ...]``-shaped intermediates
and reports their presence.  If a future XLA upgrade introduces
a pessimisation, the bench fails loudly.

## What we considered and didn't pick

### A unified `lobpcg_top_k(A, ...)` dispatch

Tempting: one entry point, internal type-dispatch on ``A``.  Rejected
because ``jax.custom_vjp`` applications have a *fixed* diff-arg
structure -- the dense path differentiates w.r.t. ``M``, the ELL path
w.r.t. ``M.values`` only.  A unified function with conditional
custom_vjp registration is awkward; separate wrappers are the clean
JAX pattern.  The connectopy-level dispatcher (``_lobpcg_top_k``)
provides the unified user-facing surface.

### Registering `ELL` as a pytree

Would allow a single ``custom_vjp`` over ``A: ELL`` with ``values``
as the diff leaf and ``indices / n_cols / identity`` as aux.  Cleaner
than carrying ``values`` and ``indices`` separately.  Defered for
two reasons:

1. JAX 0.10's ``custom_vjp`` machinery has rough edges with
   user-pytreed inputs (the ``nondiff_argnums`` semantics for pytree
   aux is unclear); the separate-arg form is just easier to reason
   about.
2. ``ELL`` is currently a frozen dataclass used widely as a value
   carrier; making it a pytree changes its semantics under
   ``jax.tree_util.tree_map`` -- a refactor that should land
   coordinated with sparse format consolidation, not bundled into
   this VJP.

### Wrapping `lobpcg_standard` directly

Skip the connectopy layer and ship a generic differentiable
``lobpcg_top_k(matvec, X0, ...)``.  Rejected because ``matvec`` is a
closure -- not hashable for ``nondiff_argnums``, and not
differentiable in any well-defined sense (closures capture state
that's hidden from JAX).  The shipped surface takes *concrete*
operands so the gradient targets are unambiguous.

## Open questions

- **Degenerate eigenvalues at the user's loss boundary.**  The clamp
  drops the gradient pair, but the user's loss may compose multiple
  pairs in a way that's well-defined even when individual pairs
  blow up.  We don't currently expose a "soft clamp" knob; if a
  consumer hits this, a follow-up adds a configurable smoothing
  (Tikhonov-style: ``F[i, j] = (λ_j - λ_i) / ((λ_j - λ_i)^2 +
  eps^2)``).
- **SectionedELL-aware backward.**  The forward-only constraint is
  load-bearing: the bucketed-row format's gradient projection needs
  per-section handling of ``U[indices[i, p], :]`` gathers.  Doable
  -- the math is the same as flat ELL, just over multiple
  sections -- but the implementation cost (and test surface) wasn't
  worth blocking first GA.  Track as a follow-up.
- **Krylov-corrected option 2.**  Specced above; not shipped.
  Trigger criterion for shipping: a downstream consumer demonstrably
  needs exact gradients through eigenvectors for a non-spectral-
  clustering loss.

## Cross-references

- ``src/nitrix/graph/_lobpcg_diff.py`` -- the implementation.
- ``src/nitrix/graph/connectopy.py`` -- the call site;
  ``_lobpcg_top_k`` is the dispatcher.
- ``tests/test_graph.py`` -- the validation suite (sections
  "LOBPCG implicit VJP" and adjacent).
- ``jax.experimental.sparse.linalg.lobpcg_standard`` -- the
  underlying solver.
- ``jnp.linalg.eigh`` -- the dense path that's already differentiable
  (used as ground truth in tests).
- Magnus & Neudecker (1988) "Matrix Differential Calculus" -- the
  canonical reference for the eigenvector-derivative formula.
