# Displacement-field regularisers — `nitrix.register.regulariser`

> **Status (2026-06-08): not started — ENABLING.** Loss-numeric item from the
> 2026-06-08 ilex audit
> ([`ilex-training-substrate.md`](ilex-training-substrate.md)). A real gap:
> `nitrix.register`'s diffeomorphic recipe regularises by **Gaussian
> smoothing** (the Green's function of the fluid/diffusion operator) — there
> is **no explicit penalty** on the field for a learned/optimised warp.

**What.** Explicit smoothness/invertibility penalties on a displacement (or
velocity) field, the regularisers learned registration (VoxelMorph /
SynthMorph) and the nimox registration losses use:

1. **`gradient_smoothness`** — diffusion penalty: per-voxel Σ over spatial
   axes of squared first-order finite differences of the field, summed over
   channels (VoxelMorph `Grad('l2')`). `ilex/nimox/loss/functional/
   registration.py:59` (`grad_smoothness_3d`); the 2D analogue
   (`:259`, `grad_smoothness_2d`) is an unimplemented stub — ship one n-D
   primitive instead.
2. **`bending_energy`** — second-order (thin-plate) penalty Σ of squared
   second differences / mixed partials. The standard alternative to (1);
   natural sibling.
3. **`jacobian_folding_penalty`** — a monotone penalty on `J = det(I +
   ∇u) ≤ 0` (folding / non-diffeomorphism). `registration.py:289`
   (`jacobian_folding_penalty_2d`, stub). The **determinant is already in
   nitrix** (`geometry.jacobian_det_displacement`, documented for folding
   detection); only the smooth penalty wrapper (e.g. `relu(−log J)` /
   `(J<0)` smooth) is new.

**Drivers.** `voxelmorph`, `synthmorph`, `synthmorph_deform` registration
training; the nimox `registration` loss family; the diffeomorphic
`cortex_ode` / `surfnet` flows.

**API sketch.**

```python
def gradient_smoothness(field: Float[Array, '*spatial ndim'], *,
                        reduction='mean') -> Array: ...
def bending_energy(field: Float[Array, '*spatial ndim'], *,
                   reduction='mean') -> Array: ...
def jacobian_folding_penalty(displacement: Float[Array, '*spatial ndim'], *,
                             reduction='mean') -> Array: ...
```

**Pure / XLA note / reuse.** All n-D, `jnp` finite-difference stencils;
jit-clean. `gradient_smoothness` should build on the existing
`nitrix.geometry.spatial_gradient` (already used inside the demons recipe);
`jacobian_folding_penalty` on `geometry.jacobian_det_displacement`. No new
heavy substrate — these are thin, careful wrappers over shipped primitives.

**Home.** `nitrix.register.regulariser` (a new module beside `recipes.py` /
`diffeomorphic.py`), so the optimisation recipes and the standalone penalties
share a home. The penalties are independently useful as training losses.

## Cross-references

- [`ilex-training-substrate.md`](ilex-training-substrate.md) — survey context.
- `src/nitrix/register/diffeomorphic.py` — the Gaussian-smoothing
  regularisation these explicit penalties complement.
- `src/nitrix/geometry/{differential.py,grid.py}` — `spatial_gradient` and
  `jacobian_det_displacement`, the substrate to build on.
