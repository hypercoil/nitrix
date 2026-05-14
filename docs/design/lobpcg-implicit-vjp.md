# LOBPCG: the implicit-VJP TODO

> **Status.**  ``jax.experimental.sparse.linalg.lobpcg_standard`` is
> not natively differentiable (the iterative ``lax.while_loop`` has a
> data-dependent termination criterion).  Reverse-mode AD through the
> connectopy ``lobpcg`` path raises a native JAX error; the docstring
> directs users to ``solver="eigh"`` for differentiable use on small
> dense graphs.  This document records the math for the eventual
> implicit-VJP implementation so future-me (or future-you) doesn't
> have to re-derive it.

## The math

Forward: given symmetric ``A: (n, n)`` and initial guess ``X_0: (n,
k)``, lobpcg returns ``(U: (n, k), Λ: (k,))`` such that approximately
``A U = U diag(Λ)`` with ``U^T U = I_k`` (the top-``k`` eigenpairs).
For partial decompositions (``k < n``), the rows of ``U`` span a
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
V^T`` with ``V: (n, n)``:

    (∂L/∂A) = V diag(g_λ) V^T + V (F ⊙ (V^T g_V)) V^T

where ``F[i, j] = 1 / (λ_j - λ_i)`` for ``i ≠ j`` and ``0`` on the
diagonal, and ``g_V = ∂L/∂V``.  This formula is what
``jnp.linalg.eigh``'s registered VJP uses.

For a *partial* decomposition with only the top ``k`` eigenpairs in
``(U, Λ)``, we don't have the remaining ``n - k`` eigenvalues to
build the full ``F`` matrix.  Two options:

#### Option 1: Subspace-projector approximation

Drop the eigenvector gradient contribution from the orthogonal
complement of the top-``k`` subspace.  Within the top-``k`` subspace,
build ``F: (k, k)`` using only the known eigenvalues:

    F_kk[i, j] = 1 / (Λ_j - Λ_i) for i ≠ j in 1..k, else 0

Then:

    (∂L/∂A) ≈ U diag(g_λ) U^T + U (F_kk ⊙ (U^T g_U)) U^T

The error is in the contribution of ``(I - U U^T) g_U`` -- the part
of the eigenvector cotangent that lies outside the top-``k``
subspace.  For *purely* in-subspace losses (e.g., functionals of the
eigenvalues, or quadratic forms over the embedding), this term is
zero and the approximation is exact.  For losses that depend on the
full eigenvectors at high frequencies (the discarded part of the
spectrum), the approximation is biased.

This is what KeOps and most "differentiable spectral methods"
papers use.  It's the right default.

#### Option 2: Iterative correction via the orthogonal-complement Hessian

Solve the orthogonal-projection equation:

    (A - Λ_i I) (∂u_i / ∂A) = (I - U U^T) (∂A) u_i

via a Krylov solver (CG / GMRES) for each eigenvector.  This recovers
the missing contribution exactly but costs ``O(k * matvec_cost *
krylov_iter)`` per backward pass.  Reasonable for ``k`` small and
``matvec_cost`` low (sparse adjacency); expensive otherwise.

#### Recommendation for first cut

Implement option 1 (subspace projector) with a docstring note about
the approximation.  Option 2 is a follow-up if a downstream consumer
needs exact gradients through eigenvectors.

## The custom_vjp shape

The wrapping function needs to handle a callable ``matvec`` argument
(which is unhashable for ``nondiff_argnums``).  The right pattern is
to take ``A`` directly (dense, ELL, or SectionedELL) and dispatch on
its type inside the wrapped function, building the matvec there:

```python
@jax.custom_vjp
def lobpcg_top_k(A, X0, iters, tol):
    matvec = _make_matvec(A)
    return lobpcg_standard(matvec, X0, m=iters, tol=tol)

def fwd(A, X0, iters, tol):
    eigvals, eigvecs, _ = lobpcg_top_k(A, X0, iters, tol)
    return (eigvals, eigvecs), (A, eigvals, eigvecs)

def bwd(res, g):
    A, Λ, U = res
    g_λ, g_U = g
    # Hellmann-Feynman + F-matrix subspace approximation
    F = jnp.where(
        jnp.arange(k)[:, None] != jnp.arange(k)[None, :],
        1.0 / (Λ[None, :] - Λ[:, None]),
        0.0,
    )
    UtgU = U.T @ g_U
    dA = U @ jnp.diag(g_λ) @ U.T + U @ (F * UtgU) @ U.T
    return (dA, None, None, None)

lobpcg_top_k.defvjp(fwd, bwd)
```

The catch: when ``A`` is ``ELL`` / ``SectionedELL``, the
``custom_vjp`` machinery sees ``A`` as a Python object with array
fields.  We'd register ``ELL`` as a pytree (via ``jax.tree_util.register_pytree_node``) so the gradient flows back through the
``.values`` field.  ``indices`` and ``n_cols`` would be marked as
auxiliary (non-diff) data.

Plus a separate type-dispatch on ``A`` inside the bwd to construct
the gradient correctly for each sparse format -- the dense
``U @ jnp.diag(g_λ) @ U.T`` becomes a "scatter back the U-weighted
contributions into the ELL pattern of A" for sparse adjacencies.

## Open questions

- **Degenerate eigenvalues.**  When ``λ_i ≈ λ_j`` for some ``i ≠ j``,
  ``F`` blows up.  The standard fix is to clamp the denominator
  (``|λ_j - λ_i| < eps`` -> set ``F[i, j] = 0`` and warn) but the
  user is then operating in a regime where eigenvectors aren't
  individually meaningful.  Document and clamp.
- **Sparse-aware backward.**  For ELL ``A``, the dense formula
  ``U @ diag(g_λ) @ U.T`` is ``O(n^2)`` and defeats the sparse
  forward.  The right gradient is the *projection* of this onto the
  sparsity pattern of ``A``: ``∂L/∂A.values[i, p] = sum_k g_λ_k *
  U[i, k] * U[A.indices[i, p], k]``.  That's
  ``O(nnz * k)`` -- consistent with the forward cost.

## Cross-references

- ``src/nitrix/graph/connectopy.py`` -- the call site.
- ``jax.experimental.sparse.linalg.lobpcg_standard`` -- the
  underlying solver.
- ``jnp.linalg.eigh`` -- the dense path that's already differentiable
  (for comparison and unit testing of any partial-subspace VJP).
- Magnus & Neudecker (1988) "Matrix Differential Calculus" -- the
  canonical reference for the eigenvector-derivative formula.
