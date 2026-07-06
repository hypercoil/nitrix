# Uniform 1-ring mesh Laplacian smoothing — `nitrix.sparse.mesh`

> **Status (2026-06-09): SHIPPED.** `sparse/mesh.py` adds
> `mesh_laplacian_smooth(vertices, faces, *, lam, iterations)` — the uniform
> (combinatorial) 1-ring step `v ← (1−λ)v + λ·mean(neighbours)`, neighbours
> from triangle edges via `segment`-style scatter-add (shared edges cancel
> in the mean on a closed manifold, so no de-dup needed), iterated with
> `lax.fori_loop`. `lam=0` is identity; differentiable. Landed in
> `sparse.mesh` (next to `Mesh` / `compute_vertex_normals`) rather than a new
> `geometry.mesh`. Model-numeric item from the 2026-06-08 ilex audit
> ([`ilex-training-substrate.md`](../ilex-training-substrate.md)); distinct from
> `graph.laplacian` (graph operator) and the cotangent Laplacian.

**What.** One uniform (combinatorial) Laplacian smoothing step on a triangle
mesh: `v ← (1−λ)·v + λ·mean(1-ring neighbours)`, with the 1-ring incidence
derived from the face list. Iterated, this is Taubin/Laplacian surface
smoothing. `ilex/models/cortex_ode/_pipeline_helpers.py:133`
(`laplacian_smooth_uniform`) — currently numpy (`np.add.at`); the JAX form is
a `segment_sum` / `.at[].add` over the edge incidence.

**Driver.** `cortex_ode` (surface post-processing); `surfnet`; a natural
companion to the other surface primitives the new ODE/diffeomorphic surface
models need.

**API sketch.**

```python
def laplacian_smooth(
    vertices: Float[Array, 'V 3'],
    faces: Int[Array, 'F 3'],
    *,
    lam: float = 0.5,
    iterations: int = 1,
) -> Float[Array, 'V 3']:
    """Uniform 1-ring Laplacian smoothing (combinatorial weights)."""
```

**Pure / XLA note.** Edge incidence → `jax.ops.segment_sum` (or
`.at[idx].add`) for the neighbour mean + a per-vertex valence divide;
`lax.fori_loop` for the iteration. jit-clean with a static `iterations`. The
neighbour structure is exactly the ELL/mesh adjacency
`nitrix.sparse.mesh.Mesh` already models — build on it rather than
re-deriving incidence.

**Relation to existing nitrix.** Distinct from:
`graph.laplacian` (a graph operator *matrix*, not a per-vertex geometric
step); `geometry.dec` cotangent Laplacian (area/angle-weighted, for DEC —
this is the cheaper combinatorial sibling). Pairs with
[`compute-vertex-normals`](compute-vertex-normals.md) and
[`mesh-curvature`](../mesh-curvature.md) as the per-vertex mesh-geometry family.

**Home.** `nitrix.geometry` mesh surface (or `nitrix.sparse.mesh`, next to
the `Mesh` dataclass and `compute_vertex_normals`).

## Cross-references

- [`ilex-training-substrate.md`](../ilex-training-substrate.md) — survey context.
- [`compute-vertex-normals.md`](compute-vertex-normals.md),
  [`mesh-curvature.md`](../mesh-curvature.md) — the per-vertex mesh family.
- `src/nitrix/sparse/mesh.py` — the `Mesh` adjacency to build on.
