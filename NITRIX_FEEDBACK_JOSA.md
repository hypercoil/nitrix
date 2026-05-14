# nitrix gap analysis — driven by the JOSA port

**Source.** `/Users/rastko.ciric/dev/diffprog/nitrix` (read-only for the purposes of this doc; recommendations live here, not as patches).

**Driving consumer.** `ilex.models.josa` (the FreeSurfer `mris_register_josa` port). JOSA is the densest consumer of `nitrix.geometry`/`nitrix.smoothing`/`nitrix.morphology` in `ilex` so far, so it surfaces gaps that the prior Tier A ports (VoxelMorph 3D MSE, SynthStrip, SynthSeg, SynthMorph, SynthSR) did not hit. The ports of those prior models reach into `nitrix` only via the 3D-warp + SDT-distance-transform paths; JOSA additionally needs **2D spherical topology**, a **diffeomorphic Jacobian filter**, **atlas-as-state**, and **bidirectional warp** semantics.

## Summary

| Need (JOSA-driven) | Current state in nitrix | Verdict |
|---|---|---|
| 2D spherical padding (top/bottom = flip + W/2 roll; left/right = circular; optional sign-flip for flow channels) | Not present | **Missing primitive.** Should land in `nitrix.geometry.sphere` (or a new `nitrix.geometry.sphere_grid`). |
| `SpatialTransformer` / `integrate_velocity_field` with **edge-replicate** out-of-bounds (the TF `fill_value=None` mode that voxelmorph uses) | `grid.spatial_transform` uses `jsp_ndi.map_coordinates(..., mode='constant', cval=cval)` — constant-fill only | **BLOCKING for JOSA.** `cval=0` mode is correct when sampling beyond the support of an image; it is NOT correct when sampling a flow field beyond its support. Voxelmorph's `fill_value=None` clamps coords to `[0, max-1]` per axis (edge replication, NOT linear extrapolation), which maps to `scipy.ndimage.map_coordinates(mode='nearest')`. Verified `max abs diff = 15.18` for one integrate-velocity-field call on JOSA's reference SVF. See §2 below for the proposed fix recipe. |
| `spatial_transform` over channel-last with **leading batch dim** | Documented as single-sample; `vmap` is the prescribed escape hatch | **Workable, mild friction.** Vmap-ing every per-sample primitive is fine in pure JAX but produces tracer-heavy graphs; a `batched_spatial_transform` convenience that internally vmaps would lower the surface area for every consuming model class. Not blocking. |
| 2D Jacobian determinant of a displacement field (central differences) | Not present | **Missing primitive.** This is the diffeomorphism-validity check used by JOSA's NegativeJacobianFiltering; also by any post-hoc QA on a learned warp. Should land in `nitrix.geometry.grid` or a new `nitrix.geometry.deform` module. |
| `nitrix.smoothing.gaussian` for very small sigma + small kernels (NJF uses `filter_shape=(2,2), sigma=0.7`) | Present, but with `truncate * sigma + 0.5` kernel-half-width rounding that yields a half-width of 0 (i.e. a single-tap kernel = identity) at sigma=0.7 if the user wants a fixed `(2, 2)` window | **Behavioural mismatch.** JOSA expects an *explicit* 2×2 window with a Gaussian-weighted 4-tap kernel, not a separable-and-truncate Gaussian. The `truncate` API only specifies the spatial extent and not the explicit window shape; we cannot reach the upstream semantic from the current call surface. |
| Grayscale dilation with **periodic / spherical** boundary handling | `nitrix.morphology.dilate` exists with `padding='SAME'` (algebra-identity pad = -inf) | **Mismatch.** JOSA's mask dilation in NJF should ideally use the same spherical wrap as the conv chain; falling back to algebra-identity pad will spuriously fail to expand the negative-jacobian mask across the longitudinal seam. (Workaround: pad the mask spherically, then `nitrix.morphology.dilate` with VALID, then unpad. We do this manually.) |
| **Inside-`lax.while_loop` reuse** of nitrix primitives that allocate persistent state | Primitives are pure; should compose with `lax.while_loop` directly | **Confirmed clean.** No gap. (Verified mentally; if any of `gaussian`, `dilate`, `spatial_transform` allocated through some non-functional registry this would have surfaced.) |
| 2D linear interpolation with arbitrary fill-value mode (`'edge'`, `'reflect'`, periodic) | `map_coordinates` exposes mode= argument internally but the public `spatial_transform` only forwards `cval` (constant) | **Surface gap.** Forwarding `mode` from `spatial_transform` to `map_coordinates` (and adding a thin `mode='periodic'` axis-by-axis wrap) would close the linear-extrap + spherical-warp + periodic-image use cases all at once. |
| Trained atlas as a 2D image tensor that is **broadcast** as a synthetic per-sample channel | `nitrix` does not host module-level abstractions (`SPEC.md` §1 non-goals: "No Equinox modules — those are `nimox`.") | **Out of scope for nitrix.** This is a `nimox` / Equinox module concern (the parameter lives in an `eqx.Module` field). |

## Discussion / detail

### 1. Spherical padding (`pad_2d_image_spherically`) — **propose to add**

JOSA pads its 256×512 spherical map by 16 each side to 288×544 so all subsequent Conv2D ops can use VALID padding internally. The spherical-pad scheme:

* **Top / bottom (poles).** Take the next `pad_size` rows starting from row 1 (skipping the pole row itself, since reflecting the pole would duplicate it), flip vertically, and roll horizontally by `W/2`. This is the natural "go over the pole" topology — the spatial neighbour of pixel `(0, j)` "above" the north pole is `(1, j + W/2)`.
* **Left / right (longitudinal).** Plain circular wrap.
* **Sign flip on poles, for flow channels only.** When the tensor represents a 2D flow field, the horizontal-flow channel reverses sign across a pole (because longitude reverses). Without this, a coherent flow at a pole would discontinuously sign-flip across the pole boundary.

Upstream signature: `pad_2d_image_spherically(img, pad_size, input_no_batch_dim, is_flow)`. The pole-flip / longitudinal-wrap topology is **the** sphere-grid pad scheme; it's reusable for any future surface-grid CNN (FreeSurfer's surface-based registration, surface-based saliency maps, surface diffusion models, etc).

**Suggested home.** `nitrix.geometry.sphere` already exists for spherical-mesh primitives (`spherical_geodesic_distance`, `spherical_conv`); the SPEC §4.4 splits `geometry` between `grid.py` (regular-grid voxel ops) and `sphere.py` (spherical mesh ops). The 2D parameterised-sphere case (regular grid AND spherical topology) bridges the two — `sphere_grid_pad` could live in `nitrix.geometry.sphere` next to the existing sphere primitives, or in a new `geometry.sphere_grid` module. Given that the dominant use is "2D conv on a parameterised sphere", `sphere.py` is the natural home and the symbol name `sphere_grid_pad_2d` (vs. the mesh-based sphere ops) keeps the distinction clear.

**API proposal:**
```python
def sphere_grid_pad_2d(
    image: Float[Array, "*leading H W *trailing"],
    pad_size: int,
    *,
    height_axis: int = -3,        # or detect from leading-axes pattern
    width_axis: int = -2,
    is_flow: bool = False,        # sign-flip top/bottom pad if image represents 2D flow
) -> Float[Array, "*leading (H+2p) (W+2p) *trailing"]
```

The companion `sphere_grid_unpad_2d` is a trivial slice and could just live next to it.

### 2. Edge-replicate sampling in `spatial_transform` (**HIGH PRIORITY** — empirically blocking)

**Severity: BLOCKING for JOSA.** Verified by running `nitrix.geometry.grid.integrate_velocity_field` on JOSA's TF reference SVF and comparing to the TF reference's `vxm.layers.VecInt(method='ss', int_steps=7)` output. `max abs diff = 15.18` over a single pixel, `mean abs diff = 0.09` — the divergence is local to grid-boundary regions where the integration step samples the flow at coordinates that land outside the support of the previous step's field. The catastrophic cliff comes from `cval=0`: at the boundary, voxelmorph clamps to the edge cell, while nitrix substitutes zero.

I initially described this gap as "linear extrapolation" — **that was wrong, and the correction matters for the fix**. What voxelmorph actually does (per `neurite.tf.utils.utils.interpn`, lines 137-191):

* Clamp the float location to `[0, max_loc[d]]` along every axis BEFORE computing the floor / ceil corners AND the interpolation weights.
* The interpolation weights are computed from the *clamped* location, not the original. So when the original is OOB, the floor and ceil corners coincide and the weight degenerates to 1 on the boundary cell.
* The single `fill_value=None` branch (the default; what JOSA / SynthMorph / VxmDense all use) realises **edge-replication**, NOT linear extrapolation.

The good news: this maps cleanly to `scipy.ndimage.map_coordinates(..., mode='nearest')` — "the input is extended by replicating the last pixel". `jax.scipy.ndimage.map_coordinates` already supports this mode. The fix is purely a one-keyword pass-through.

**Concrete proposed change.** In `nitrix.geometry.grid._gather_coords_linear`:

```python
def _gather_coords_linear(image, coords, *, cval, mode='constant'):
    # ...
    def sample_one_channel(img_ch):
        return jsp_ndi.map_coordinates(
            img_ch, coords_flat, order=1, mode=mode, cval=cval,
        )
    # ...
```

And in `spatial_transform` / `integrate_velocity_field`:

```python
def spatial_transform(
    image, deformation, *,
    cval: float = 0.0,
    mode: Literal['constant', 'nearest', 'wrap', 'mirror', 'reflect'] = 'constant',
) -> Array:
    # ...
    return _gather_coords_linear(image, deformation, cval=cval, mode=mode)

def integrate_velocity_field(velocity, *, n_steps=7, mode='nearest'):
    # ...
    for _ in range(n_steps):
        phi = phi + spatial_transform(phi, id_grid + phi, mode=mode)
    return phi
```

**Recommended defaults.** For `integrate_velocity_field` flip the default to `mode='nearest'` (edge-replicate) because EVERY voxelmorph-style consumer needs this: the diffeomorphic SS integration step is **defined** as composing the flow with `id + phi` and reading the flow at the displaced coordinates — at the boundary, those coordinates land OOB and edge-replication is the canonical convention. The current `cval=0` default silently miscomputes the integrated flow at the boundary cells; for a 288×544 grid with a smooth flow the worst-case error reaches `~15 voxels` per integration step, dominating any reasonable downstream loss.

For `spatial_transform` the default can stay `mode='constant'` (the prior contract — sampling a foreground image outside its support is genuinely a "fill the void" operation, where 0 is a fine default), but voxelmorph-style flow-warp consumers should be encouraged to pass `mode='nearest'` explicitly.

If there's appetite for it later: a separate `mode='linear_extrap'` (per the original suggestion — pad-by-1 + edge-replicate the image + adjust coords by 1) would be a genuine linear extrapolator. That isn't what voxelmorph uses, so it's not required for parity, but the implementation cost is small and the use case is real (any sampler used inside a learned generative model where you want gradient continuity at the boundary). Low priority.

**JOSA-side workaround until this lands:** the JOSA module vendors a local `_warp_with_edge_replicate` that calls `jax.scipy.ndimage.map_coordinates(..., mode='nearest')` directly + a `_integrate_velocity_field` reimplementation that uses it. Same algorithm as nitrix, one-keyword difference. See `ilex/src/ilex/models/josa/_warp.py`. When nitrix lands the `mode='nearest'` option the JOSA wrapper can revert to nitrix in one PR.

### 3. Batched `spatial_transform` convenience

JOSA's TF `SpatialTransformer` takes `(batch, H, W, C)`-shaped inputs natively. Every consumer that wraps `nitrix.geometry.grid.spatial_transform` in a model has to `vmap` over leading dims. The cost is trivial — at most one wrapping line in each consumer — but it's a paper cut every consumer hits. If consumers wrap differently (one model uses `jax.vmap`, another uses `jax.tree_util.tree_map`, etc.) the cross-model surface fragments.

**Proposed convenience:**
```python
def spatial_transform_batched(
    image: Float[Array, "*leading H W ... C"],
    deformation: Float[Array, "*leading H W ... ndim"],
    *,
    cval: float = 0.0,
    mode: str = 'constant',
) -> Float[Array, "*leading H W ... C"]
```
Internally `vmap`-s over leading axes that the two inputs share. Strictly redundant with `vmap(spatial_transform)` but reduces 5+ call sites to 1.

### 4. 2D Jacobian determinant of a displacement field — **propose to add**

JOSA's NJF needs `J = det(I + ∇u)` for a 2D displacement `u`. This is a central-difference op on the channels-last `(H, W, 2)` field:

```python
def jacobian_2d_det(displacement, *, mode='same', is_replace_nan=False):
    # ∂u_x/∂x = (u[i, j+1, 0] - u[i, j-1, 0]) / 2, etc.
    # J = (1 + du_x_dx)(1 + du_y_dy) - du_x_dy * du_y_dx
```

This is the basic diffeomorphism-validity check. Useful for:
* JOSA's NegativeJacobianFiltering 
* QA / regularisation on any learned warp (post-training validity)
* Folding detection
* Future SynthMorph deformable hypernet ports

**Suggested home.** `nitrix.geometry.grid` next to `integrate_velocity_field` and `spatial_transform`. An N-D variant (`jacobian_det_displacement`) would also be useful; the 2D version is just the degenerate case.

### 5. `nitrix.smoothing.gaussian` with explicit kernel-window shape

JOSA's NJF replaces negative-jacobian voxels with Gaussian-filtered values from a `(2, 2)` window at `sigma=0.7`. The upstream `gaussian_filter_2d` in spheremorph explicitly takes `(filter_shape, sigma)` and emits a 2×2 kernel weighted by the Gaussian; sigma scales the falloff, filter_shape pins the support.

`nitrix.smoothing.gaussian` takes only `sigma` and uses `int(truncate * sigma + 0.5)` for the half-width. At `sigma=0.7, truncate=4` we get half=3, kernel width 7 — too wide. At any `truncate <= 1.43` we get half=0, kernel width 1 — identity. There is no `truncate` value that yields exactly the 2-tap kernel the upstream uses. (And the kernel runs per-axis separable, so the *effective* 2D kernel is a 2-tap separable, not a 2×2 dense.)

**Three options:**
1. **Add an explicit `kernel_size` argument** that overrides `truncate * sigma` rounding — pinning the window directly to the user-specified size. This matches scipy/skimage when those users want pixel-exact control.
2. **Add a `gaussian_dense_2d(filter_shape, sigma)` companion** that builds the dense 2D Gaussian kernel (non-separable) for the small-kernel case. Slower than separable but exact.
3. **Document the constraint** and require consumers to apply their own ad-hoc filter. (We do this for JOSA — see the workaround in `josa/_njf.py`.)

Option 1 is the lowest-friction; it preserves separability and adds one parameter. The upstream behaviour is recoverable with `kernel_size=(2, 2), sigma=0.7`.

### 6. Morphology dilation with periodic boundaries

Less critical than the smoothing gap. JOSA's NJF dilates the negative-jacobian mask by a 2×2 structuring element. If the longitudinal seam has a negative jacobian, the dilation should wrap; with `nitrix.morphology.dilate(padding='SAME')` and tropical-max-plus algebra-identity padding (`-inf`), the dilation will fail to cross the seam.

For JOSA the workaround is: pre-pad the mask spherically by 1, dilate with `padding='VALID'`, unpad. Doable but adds 2 lines.

**Suggested add.** A `padding: Literal['SAME', 'VALID', 'periodic']` option on `dilate`/`erode` would expose the wrap-around case for free. The `lax.conv_general_dilated` underlying implementation supports periodic via explicit `roll` + slice; doable but non-trivial.

### 7. `integrate_velocity_field` — works as-is

`nitrix.geometry.grid.integrate_velocity_field(velocity, n_steps=7)` matches the voxelmorph `VecInt(method='ss', int_steps=7)` semantic exactly:
* Same scaling factor (`v / 2^n_steps`).
* Same composition rule (`phi <- phi + spatial_transform(phi, id + phi)`).
* Same number of doublings (default 7, JOSA also uses 7).

No gap. Used as-is for JOSA.

### 8. `identity_grid` — works as-is

Used directly. `dtype` parameter is helpful (we use `float32` to match the TF reference); the `(D, H, W, ndim)` channel-last layout matches what `spatial_transform` consumes.

## Out-of-scope (correctly so)

JOSA also needs:

* **Pre-trained atlas as a learnable model parameter.** This is a `nimox` / Equinox-module concern (an `eqx.field` on the JOSA module). Not a nitrix concern.
* **TF-checkpoint key renaming for the JOSA layer naming.** This is an `ilex.core.adapters` concern. Not a nitrix concern.
* **Surface-to-spherical-image parameterisation** (`surfa.SphericalMapBarycentric`). This is upstream of ilex / nitrix — the user runs FreeSurfer surface preprocessing and passes the 256×512 spherical image. Not a nitrix concern.

## Priority ranking (nitrix-side)

If you want to act on this list in priority order based on *blast radius* (how many future consumers would benefit):

1. **`sphere_grid_pad_2d`** (high blast: every surface-based learned model needs this).
2. **`spatial_transform(mode='linear_extrap')`** (high blast: every voxelmorph-style flow-warp).
3. **`jacobian_det_displacement`** (medium blast: every deformable-registration QA + folding detector).
4. **`gaussian(kernel_size=...)`** (medium blast: small-window Gaussian convolution use cases).
5. **`spatial_transform_batched`** convenience (low blast: 1-line saving per consumer).
6. **`dilate(padding='periodic')`** (low blast: only relevant for periodic boundary domains).
