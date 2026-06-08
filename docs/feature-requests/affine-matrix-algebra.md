# Affine matrix algebra (geometric convention) — `nitrix.geometry.transform`

> **Status (2026-06-08): not started — ENABLING.** Model-numeric item from
> the 2026-06-08 ilex audit
> ([`ilex-training-substrate.md`](ilex-training-substrate.md)). The source
> file (`ilex/nimox/modules/affine.py`) is **already pure numerics** — no
> equinox, every symbol `(Array,…)->Array` — and its own docstring names
> "Long-term home: nitrix". A clean, high-value, low-risk lift.

**What.** The decompositional affine algebra in the **Euler-angle /
scale / shear** convention (SynthMorph / VoxelMorph / lab2im), complementary
to nitrix's existing **Lie-algebra** `transform.py` (`rigid_exp`/`affine_exp`
via `matrix_exp` + axis-angle). Different parameterisation, both needed:

- **`params_to_affine_matrix` / `affine_matrix_to_params`** — compose/decompose
  a 3D affine `T@R@S@E` from/to a 12-vector (translation, rotation, scale,
  shear). Decompose via Cholesky of `MᵀM` + determinant sign fix.
  `affine.py:199` / `:250`.
- **`angles_to_rotation_matrix` / `rotation_matrix_to_angles`** — Euler
  (intrinsic `X@Y@Z`) ↔ rotation, with the gimbal-lock branch.
  `affine.py:126` / `:159` (+ helper `_divide_no_nan`, `:191`).
- **`fit_affine`** — closed-form (weighted) least-squares affine between
  corresponding point sets via the normal equations. `affine.py:96`. The
  missing closed-form sibling of nitrix's gradient-descent `register`.
- **`make_square_affine` / `invert_affine` / `compose_affine`** —
  `(N,N+1)`↔`(N+1,N+1)` homogenisation, affine inverse, composition.
  `affine.py:61`–`:89`. Promote nitrix's private `_homogeneous`
  (`transform.py:126`) to public and add the rectangular/compose variants.

**Drivers.** lab2im `sample_affine_matrix` (`lab2im.py:162`, the augmentation
in [`geometric-augmentation-ops.md`](geometric-augmentation-ops.md));
`synthmorph` / `voxelmorph` affine fit/compose/decompose; nitrix's own
`register` would use `fit_affine` for closed-form initialisation.

**API sketch.**

```python
def params_to_affine_matrix(params: Float[Array, '... 12']
                            ) -> Float[Array, '... 3 4']: ...
def affine_matrix_to_params(affine: Float[Array, '... 3 4']
                            ) -> Float[Array, '... 12']: ...
def angles_to_rotation_matrix(angles) -> Array: ...
def rotation_matrix_to_angles(r) -> Array: ...
def fit_affine(source, target, *, weights=None) -> Array: ...   # (...,N,N+1)
def compose_affine(*affines) -> Array: ...
def invert_affine(affine) -> Array: ...
```

**Pure / XLA note.** `jnp.linalg.{cholesky, det, inv}` + matmul; the only
non-array control flow is static shape validation (`params.shape[-1]` check)
— trace-safe. jit-clean. On GPU, the small `cholesky`/`inv` should route
through `linalg._solver.safe_*` (the cuSolver-pool fallback noted in the dev
env).

**Home.** `nitrix.geometry.transform` (beside the Lie-algebra API), or a new
`nitrix.linalg.affine` if a cleaner separation from the grid utilities is
wanted. Either way, keep both parameterisations — exp-map and
Euler/scale/shear — explicitly named.

## Cross-references

- [`ilex-training-substrate.md`](ilex-training-substrate.md) — survey context.
- [`geometric-augmentation-ops.md`](geometric-augmentation-ops.md) — the
  `random_affine_matrix` consumer.
- `src/nitrix/geometry/transform.py` — the existing Lie-algebra affine API
  (`rigid_exp`/`affine_exp`/`apply_affine`/`affine_grid`) this complements.
