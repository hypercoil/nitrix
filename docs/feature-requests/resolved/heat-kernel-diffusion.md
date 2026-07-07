# Heat-kernel & diffusion-map embedding — `nitrix.graph.diffusion`

> **Status (2026-07-07): SHIPPED.** `graph.heat_kernel(A, t, *,
> normalisation='symmetric', method='exp'|'eigh')` returns the dense heat
> kernel `K_t = exp(-tL)` (composing the shipped `linalg.matrix_exp` +
> `graph.laplacian`); `method='eigh'` gives the symmetric-exact
> `Q exp(-tΛ) Qᵀ` via the cuSolver-robust `safe_eigh`. Homed in `graph`
> (adjacency-in, so it owns the Laplacian assembly + conditioning), **not** a
> new `graph.diffusion` module. For a **mesh** heat kernel (HKS on the
> cotangent Laplacian) compose `linalg.matrix_exp(-t * L)` directly — the mesh
> heat-diffusion *smoother* (implicit-Euler step on a signal) is
> `geometry.surface_smooth`. 7 tests; GPU-verified.
>
> **Original (2026-06-02): partial — `diffusion_embedding` (Coifman & Lafon
> 2006) is shipped; the explicit heat-kernel operator `K_t = exp(−tL)` is
> not.** Brainstorm candidate; promotion gated by the §13 acceptance
> protocol. Provenance: `docs/feature-requests catalogue §12.3`.

**What.** The heat-kernel matrix function of the Laplacian, plus
heat-kernel-mass-normalised diffusion-map embedding.

**Composition.** `K_t = exp(−tL)` is a matrix function of the Laplacian
(composes [`matrix-functions.md`](../matrix-functions.md) §12.2 `matrix_exp` +
`graph.laplacian`); a truncated eigendecomposition gives a finite-rank
approximation. Diffusion-map embedding is the weighted eigenmap normalised
by the heat-kernel mass.

**Likely consumer.** Surface-based connectivity embeddings, manifold-
learning preprocessors for fMRI, vertex descriptors for mesh-correspondence
(heat-kernel signatures).

**Effort.** S — depends on §12.2.

**Live-code status.** `graph/connectopy.py` already ships
`diffusion_embedding` (Coifman & Lafon 2006: top-`k` eigenspace of the
density-normalised diffusion operator scaled by `λ^t`, `eigh`/`lobpcg`
backends, differentiable) and `laplacian_eigenmap`. The *embedding* is
therefore done; what is missing is the explicit `K_t = exp(−tL)`
heat-kernel **operator** (blocked on `matrix_exp`, §12.2) for heat-kernel
signatures / diffusion-distance use cases that need the kernel itself, not
just its leading eigenspace.

## Cross-references

- `docs/feature-requests catalogue §12.3` — origin entry; `§13` — acceptance protocol.
- [`matrix-functions.md`](../matrix-functions.md) — `matrix_exp` dependency.
- `src/nitrix/graph/connectopy.py` — the shipped `diffusion_embedding`.
