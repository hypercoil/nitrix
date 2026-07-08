# Discrete differential geometry — `nitrix.geometry.dec`

> **Status (2026-07-08): SHIPPED (`nitrix.geometry.dec`).** The full DEC
> operator stack on triangle meshes: `mesh_gradient` (d_0) / `mesh_curl` (d_1)
> exterior derivatives, `mesh_star_k` (Hodge stars 0/1/2), `mesh_divergence`
> (codifferential), and `hodge_decompose` (Helmholtz–Hodge, two matrix-free
> `cg` Poisson solves). Every operator an `ELL`, built host-side like the
> cotangent Laplacian. Validated against the DEC identities: `d_1 d_0 == 0`
> exactly, `d_0^T star_1 d_0 == mesh_cotangent_laplacian`, and the Hodge parts
> sum + are star_1-orthogonal + the sphere harmonic part vanishes (~1e-8);
> `hodge_decompose` is differentiable in the 1-form. Provenance:
> `docs/feature-requests catalogue §12.5`.

**What.** Generalise the cotangent Laplacian to the full discrete-exterior-
calculus stack: incidence operators, Hodge stars, Hodge decomposition.
Every operator is an `ELL`.

**Proposed surface.**

```python
def mesh_gradient(mesh) -> ELL: ...      # d_0  (vertex→edge)
def mesh_divergence(mesh) -> ELL: ...
def mesh_curl(mesh) -> ELL: ...          # d_1  (edge→face)
def mesh_star_k(mesh, k) -> ELL: ...      # Hodge stars
def hodge_decompose(omega, mesh): ...     # -> (exact, coexact, harmonic)
```

The cotangent Laplacian becomes `d_0.T @ star_1 @ d_0`.

**Composition.** Generalises `sparse.mesh_cotangent_laplacian`; every
operator is an `ELL`, so application reuses `semiring_ell_matmul`. Vertex-
edge (`d_0`) and edge-face (`d_1`) incidence + Hodge stars (`star_0/1/2`)
are host-side constructions returning `ELL`s.

**Likely consumer.** Vector-field smoothing on cortical surfaces, surface
flow / advection modelling, Hodge-decomposition shape descriptors.

**Effort.** M.

**Live-code status.** `sparse/__init__` ships `mesh_cotangent_laplacian`,
`mesh_k_ring_adjacency`, the `Mesh` container, and the `ELL` algebra it
would build on. No incidence / Hodge-star / `hodge_decompose` symbols.

## Cross-references

- `docs/feature-requests catalogue §12.5` — origin entry; `§13` — acceptance protocol.
- [`mesh-curvature.md`](resolved/mesh-curvature.md) — sibling geometry primitive.
- `src/nitrix/sparse/mesh.py` — the cotangent Laplacian this generalises.
- [`docs/design/mesh-graph-conv.md`](../design/mesh-graph-conv.md).
