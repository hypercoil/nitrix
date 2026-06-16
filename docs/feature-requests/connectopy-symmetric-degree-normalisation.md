# Connectopy normalises by the wrong degree on asymmetric graphs (symmetrises the Laplacian, not the adjacency)

> **RESOLVED 2026-06-16** (branch `fix/connectopy-adjacency-symmetrisation`).
> Correctness bug surfaced reviewing the connectopic-mapping spectral path
> (perf-bench `laplacian_eigenmap` cases). For an **asymmetric** affinity ``A``
> (the default top-k-per-row sparsified graph is asymmetric — see
> ``ell_from_dense``'s own warning), ``laplacian_eigenmap`` /
> ``diffusion_embedding`` normalise by the **row-sum out-degree** of ``A`` and
> only *then* symmetrise the operator. That symmetrises the **Laplacian**, not
> the **adjacency**, and normalises by the wrong diagonal.

## The bug

`graph/connectopy._build_affinity_operator` built

```
M = D_A^{-1/2} A D_A^{-1/2},   D_A = diag(rowsum(A))   # out-degree
```

then symmetrised the *operator* (densely, or at the matvec via
``½(A x + Aᵀ x)`` for sparse), giving

```
½(M + Mᵀ) = D_A^{-1/2} · ½(A + Aᵀ) · D_A^{-1/2} = D_A^{-1/2} W D_A^{-1/2},
            W = ½(A + Aᵀ).
```

The **correct** symmetric normalised affinity of the symmetrised graph ``W`` is

```
M* = D_W^{-1/2} W D_W^{-1/2},   D_W = diag(rowsum(W)) = ½(out-degree + in-degree).
```

These differ whenever ``A`` is asymmetric, because ``D_A`` (the out-degree of
``A``) is not the degree of ``W``. Consequences:

- ``M*`` has the degree-weighted constant ``D_W^{1/2}·1`` as its trivial
  eigenvector with eigenvalue **exactly 1**. The buggy ``M`` does not, so the
  "trivial" eigenpair that ``skip_trivial`` discards is itself wrong, and the
  remaining embedding eigenvectors are normalised by the wrong diagonal.

## Repro (measured)

Asymmetric top-k-per-row affinity, ``n=60``, ``k=6``
(``‖A−Aᵀ‖₁ / ‖A‖₁ = 1.79``):

| quantity | buggy (out-degree) | correct (symmetric degree) |
|---|---|---|
| largest affinity eigenvalue | **1.0433** | 1.0000 (proper) |
| embedding subspace vs textbook (principal angles) | **8.8° / 18.5° / 35.8°** | 0° |

`laplacian_eigenmap` reproduced the buggy operator exactly (pre-fix); post-fix
it matches the textbook operator to machine precision (`[0, 0, 0.01]°`). It is
a **no-op for symmetric ``A``** (out == in), i.e. zero change for ordinary
(symmetric) connectivity matrices.

## Fix

Normalise by the **symmetric degree** ``d_W = ½(out + in)`` everywhere a degree
is taken — both the ``alpha == 0`` Laplacian path and the ``alpha > 0``
Coifman–Lafon density degrees (the ``d^α`` and the ``d₂`` of ``K``). The
operator symmetrisation (dense ``½(M+Mᵀ)`` / sparse matvec ``½(Ax+Aᵀx)``) is
**unchanged**, so pairing it with ``d_W`` yields exactly ``D_W^{-1/2} W
D_W^{-1/2}``.

New `graph.laplacian` primitives: ``in_degree_vector`` (column sum / in-degree)
and ``symmetric_degree_vector`` (``½(out + in)``).

### Reconciliation with the `promise_symmetry` opt-in

`promise_symmetry` is the user asserting ``A == Aᵀ`` to skip the adjoint matvec
(~2x cheaper, the explicit performance opt-in). The degree choice must be
governed by the **same** flag, so the normalisation always matches the matvec:

| path | matvec | normalisation degree |
|---|---|---|
| dense (flag ignored) | always symmetrised `½(M+Mᵀ)` | symmetric `d_W` |
| sparse, `promise_symmetry=False` (default) | symmetrised `½(Ax+Aᵀx)` | symmetric `d_W` |
| sparse, `promise_symmetry=True` | bare `Ax` | plain out-degree |

On a *valid* (genuinely symmetric) `promise_symmetry=True` input, out-degree
`== d_W`, so this is **identical** to the default path — it is not a correctness
difference there; it just **skips the in-degree adjoint** along with the
symmetrising matvec, honouring the opt-in fully. The correctness fix is entirely
in the default `promise_symmetry=False` path (where asymmetric inputs are
actually handled); `promise_symmetry=True` on an asymmetric input remains a
caller contract violation, exactly as before. Cost note: the in-degree is a
*one-time* build-time adjoint (`O(nnz)`), dwarfed by the iterative solve's
per-iteration matvecs — so even on the default path the correctness fix is
effectively free.

### Retaining the efficient sparse (ELL / SectionedELL) path

The hard requirement was *not* to materialise a symmetric sparse pattern. The
in-degree is the **additive ELL adjoint applied to the ones vector** —
``Aᵀ·1`` — one ``O(nnz)`` pass on the *existing* stored pattern
(``semiring_ell_rmatvec`` for ELL, ``sectioned_semiring_ell_rmatvec`` for
SectionedELL; the same adjoint the solver already uses for its symmetric
matvec). So the only added cost is one ones-vector adjoint per build; the
sparsity pattern, the eigensolver, and the implicit-VJP differentiability are
all untouched.

## Tests

`tests/test_graph.py`:
- `test_connectopy_normalises_by_symmetric_adjacency_degree` — asymmetric-input
  parity: ``laplacian_eigenmap`` matches the textbook symmetric-degree operator
  (trivial eigenvalue 1, eigenvalues + embedding-subspace agreement) across
  dense, ELL, and SectionedELL.
- `test_connectopy_promise_symmetry_noop_on_symmetric_input` —
  ``promise_symmetry=True`` equals the default path on a genuinely symmetric
  input.
- The existing dense-vs-sparse parity tests (symmetric inputs) still pass
  unchanged (the fix is a no-op there).

## Cross-references

- `src/nitrix/graph/connectopy.py` (`_build_affinity_operator`),
  `src/nitrix/graph/laplacian.py` (`degree_vector` /
  new `in_degree_vector` / `symmetric_degree_vector`).
- `src/nitrix/sparse/ell.py` (`ell_from_dense` asymmetry warning),
  `semiring.semiring_ell_rmatvec` / `sparse.sectioned_semiring_ell_rmatvec`.
