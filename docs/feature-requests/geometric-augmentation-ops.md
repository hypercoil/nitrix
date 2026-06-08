# Geometric-augmentation ops — `nitrix.augment.geometric`

> **Status (2026-06-08): not started — CONVENIENCE.** Training-substrate
> item from the 2026-06-08 ilex audit
> ([`ilex-training-substrate.md`](ilex-training-substrate.md)). Spatial
> augmentation index/coordinate math; pure keyed `(Array,…)->Array`.

**What.** The spatial ("paired" / "image-only") augmentation primitives from
the FM recipe and the lab2im deformation half:

1. **`random_flip`** — per-axis Bernoulli flip, N-D (ilex hard-codes 3
   axes). `ilex/train/augment/geometric.py:36` (`random_flip_3d`).
2. **`random_crop`** — random-offset, fixed-size crop via
   `lax.dynamic_slice` (traced start, static size; jit-clean).
   `geometric.py:69` (`random_crop_3d`).
3. **`random_resized_crop`** — sample per-axis crop extents
   `~ U(scale_range)·shape`, random offset, build a target grid mapped back
   into the crop window, trilinear (or nearest) resample → fixed
   `target_shape` (DINOv2 `RandomResizedCrop`, 3D). `geometric.py:120`
   (`random_resized_crop_3d`).
4. **`random_affine_matrix`** / **`random_svf_displacement`** — sample a
   random affine (rotation/scale/shear/translation) and a random
   diffeomorphic displacement (low-res velocity → upsample → integrate).
   `ilex/train/augment/lab2im.py:162` / `:188`.

**Drivers.** FM pretraining (3DINO multi-crop, `compose.py`); the lab2im
spatial corruption stage (`labels_to_image`).

**API sketch** (deterministic where possible; keyed where the draw is
intrinsic):

```python
def random_flip(x, key, *, axes=None, p=0.5) -> Array: ...      # N-D
def random_crop(x, key, *, size) -> Array: ...                  # dynamic_slice
def random_resized_crop(x, key, *, size, scale, method=Linear()) -> Array: ...
def random_affine_matrix(key, *, max_rotation, max_scale, max_shear,
                         max_translation) -> Float[Array, 'd d+1']: ...
def random_svf_displacement(shape, key, *, max_std, grid_fraction,
                            n_steps=5) -> Float[Array, '*spatial ndim']: ...
```

**Composition / reuse — do not re-implement.**

- `random_resized_crop`, `random_svf_displacement`, and label-map warping
  should be built **on existing nitrix geometry**, not fresh:
  `nitrix.geometry.resample` / `spatial_transform` (the interpolation
  dispatcher — `Linear`/`NearestNeighbour`), `identity_grid`, `affine_grid`,
  and `integrate_velocity_field` (scaling-and-squaring — already called by
  `sample_svf_displacement`).
- `random_affine_matrix` depends on the Euler/scale/shear
  `params_to_affine_matrix` in
  [`affine-matrix-algebra.md`](affine-matrix-algebra.md).
- The ilex `deform_label_map` (`lab2im.py:219`, nearest-interp affine-about-
  centre + displacement on an integer label map) is **already composable**
  from `affine_grid` (about centre) `+ displacement` →
  `spatial_transform(method=NearestNeighbour())`. At most a thin convenience;
  not new numerics.

**XLA note.** Static output shapes throughout (`size`/`target_shape` are
static; `dynamic_slice` start is traced) ⇒ jit-clean. Channels convention:
ilex is channels-first `(C,…)`; nitrix is channels-last / channel-free — the
nitrix versions should be N-D and layout-agnostic, leaving the transpose to
the caller.

**Home.** `nitrix.augment.geometric` (or distribute: `random_crop`/
`random_flip` → `nitrix.numerics`; the affine/SVF generators →
`nitrix.geometry.deformation`). See the ledger namespace open question.

## Cross-references

- [`ilex-training-substrate.md`](ilex-training-substrate.md) — survey context.
- [`affine-matrix-algebra.md`](affine-matrix-algebra.md) — the affine
  param↔matrix substrate `random_affine_matrix` needs.
- `src/nitrix/geometry/{grid.py,transform.py,deformation.py}` — the resample
  / affine / SVF substrate to build on.
