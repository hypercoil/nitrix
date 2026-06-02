# Mesh curvature — `nitrix.geometry.curvature`

> **Status (2026-06-02): not started.** Brainstorm candidate; promotion
> gated by the §13 acceptance protocol. Provenance:
> `SPEC_UPDATE_v0.3.md §12.6`.

**What.** Pointwise curvature scalars on a triangulated surface.

**Proposed surface.**

```python
def mesh_mean_curvature(mesh): ...           # pointwise
def mesh_gaussian_curvature(mesh): ...        # angle-defect formula
def mesh_principal_curvatures(mesh): ...      # -> (k1, k2, e1, e2)
```

**Composition.** Mean curvature is the cotangent Laplacian applied to
vertex positions (feasible today via `sparse.mesh_cotangent_laplacian`).
Gaussian curvature is the per-face angle-defect formula. Principal
curvatures come from the Weingarten-map eigendecomposition (per-vertex 2×2
`eigh`).

**Likely consumer.** Cortical surface analysis (gyrification indices),
shape-feature extraction, surface-based registration regularisation.

**Effort.** S.

**Live-code status.** No curvature symbols. The building blocks are shipped:
`sparse.mesh_cotangent_laplacian` (mean curvature) and the `Mesh` container
(face/vertex arithmetic for the angle-defect and Weingarten paths).

## Cross-references

- `SPEC_UPDATE_v0.3.md §12.6` — origin entry; `§13` — acceptance protocol.
- [`discrete-exterior-calculus.md`](discrete-exterior-calculus.md) — sibling
  geometry primitive (shares the cotangent-Laplacian substrate).
- `src/nitrix/sparse/mesh.py`.
