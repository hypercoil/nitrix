# ilex pipeline-substrate feature requests

Numerical primitives `nitrix` should host so that the **ilex vendored-model
pipelines can migrate to `thrux`**. Driven by a 2026-06-02 survey of all 32
vendored models in `ilex/src/ilex/models/` (DEVNOTES `Port scope` / `What v0
means` sections, `card.yaml`, any `pipeline.py` / `preprocessing.py`), cross-read
against `ilex/UPSTREAM.md` (the migration ledger) and the current `nitrix`
surface.

**Framing.** The ports are almost all *forward-only v0s*: the network is parity-
locked, the upstream's image-domain pre/post-processing is deferred. The
ecosystem split (per ilex `UPSTREAM.md` / `DESIGN.md` §2) routes those deferred
steps as **numerics → `nitrix` (phase 1)**, modules → `nimox` (phase 2), operator
chains → `bitsjax`, container/affine/IO boundary + raise/lower → `thrux`. A
pipeline migrates to `thrux` when `thrux` can orchestrate an affine-/container-
aware operator chain *whose kernels exist in `nitrix`*. This file is the phase-1
kernel shopping list.

**What is already done (do not re-request).** The mesh/surface and warp/registration
substrate the surface ports needed is shipped: `semiring_ell_edge_aggregate(+edge_attr)`,
`ell_row_softmax`, `ell_add_self_loops`, `ell_mask`, `icosphere_hierarchy(_from_levels)`
/ `_cross_level_adjacency` / `_bary_upsampler`, `mesh_pool_max` / `_unpool_max` /
`_coarsen_meanpool` / `_bary_upsample`, `max_pool_with_indices_nd` / `max_unpool_nd`,
`geometry.{spatial_transform(mode='nearest'), integrate_velocity_field,
jacobian_det_displacement, sphere_grid_pad_2d}`, `smoothing.gaussian(kernel_size=)`,
and `bias.histogram_match` (Nyul–Udupa). The resolution history for that whole
tier lives in `IMPLEMENTATION_PLAN.md §10.3` (shipped-deviation log) and
`SPEC_UPDATE_v0.3 §10.A`. The remaining gap is the *mundane, ubiquitous volumetric
pre/post-processing* every CNN/UNet/ViT v0 punted on — that is what is collected
below.

> Provenance note: active feature requests live here from now on. The pre-directory
> ledgers (`NITRIX_FEEDBACK_ILEX.md`, `NITRIX_FEEDBACK_JOSA.md`, top-level
> `BACKLOG.md`) were swept on 2026-06-02 — their genuinely-open items migrated to
> this directory (`internal-backlog.md` for the parked nitrix engineering backlog;
> this file for the consumer-pipeline substrate), verified against the live code,
> so those three files can be deleted.

## Scope boundary (what is NOT nitrix)

So the phase-1 lift does not over-reach:

- **`conform` (resample + reorient + crop in vox2world / world space)** → `thrux`
  (container/affine-aware), built *on* the `nitrix` resample kernel. The array
  resample is `nitrix`; the affine bookkeeping is not.
- **Affine fit / compose / decompose** (SynthMorph barycenter + weighted-LS) →
  already in `nimox.modules.affine`.
- **Surface↔sphere parameterisation** (`surfa.SphericalMapBarycentric`),
  FreeSurfer `.sphere` / `.mgz` I/O, atlas label-LUT remap → consumer / `thrux`;
  explicitly rejected from `nitrix` (SPEC §5.2).
- **Sliding-window inference orchestration** (tiling, scheduling) → `nimox.inference`
  / `thrux`; only the per-window weighting kernel + overlap-add reduction is
  `nitrix`-shaped (item F below).

## Open

### 2026-06-02 — Volumetric pre/post-processing substrate tier (ilex → thrux)

The recurring pre/post steps across the volumetric segmentation / strip / SR /
enhancement / foundation ports. Severity is rated by the migration goal:
**ENABLING** = a pipeline cannot move to `thrux` without it; **CONVENIENCE** =
works inline today, a shared primitive removes duplication.

**A. Connected-components / largest-component labelling — ENABLING (highest recurrence).**
`nitrix.morphology` ships `dilate`/`erode`/`open`/`close`/`distance_transform`/
`median_filter` but **no connected-components** (verified: nothing matches
`connected_comp|label_components` in `src/nitrix/`). It is the single most common
omitted *post*-processing step:

- `exvivo_strip`, `synthstrip` — largest-CC to clean the brain mask after the
  SDT/SDF→mask threshold.
- `synthseg`, `wmh_synthseg`, `supersynth`, `hd_bet` — morphological cleanup /
  hole-fill / largest-CC on the label map.

API sketch (N-D, channel-free, label image out; plus a thin largest-CC helper):

```python
def connected_components(
    mask: Bool[Array, '*spatial'],
    *,
    connectivity: int = 1,            # 1 = faces; ndim = full (incl. diagonals)
) -> Int[Array, '*spatial']:          # 0 = background, 1..K = component ids
    ...

def largest_connected_component(
    mask: Bool[Array, '*spatial'], *, connectivity: int = 1,
) -> Bool[Array, '*spatial']:
    ...
```

XLA note: static-shape label propagation (iterated `max`-relabel to fixed point
via `lax.while_loop`, or Playne–Equivalence) keeps it jit-able; the component
*count* stays data-independent (a fixed label-image, not a ragged list).
Home: `nitrix.morphology`.

**B. `pad_to_multiple` / `crop_to_multiple` (+ unpad) — ENABLING (near-universal pre-step).**
Almost every volumetric net requires the grid be a multiple of its pooling factor:
`synthstrip` ×64, `synthseg`/`wmh_synthseg`/`brain_ldm(_vae)` ×16/×8, `synthdist`
×32, `bme_x` ×8. Each port inlines an ad-hoc pad. The *array* pad + the recorded
crop-back slice are pure-`nitrix`; the affine-origin update is `thrux`'s half.

```python
def pad_to_multiple(
    x: Float[Array, '*spatial c'],
    multiple: int | Sequence[int],
    *,
    spatial_rank: int,
    mode: BoundaryMode = 'constant',
    cval: float = 0.0,
) -> Tuple[Float[Array, '*padded c'], Tuple[Tuple[int, int], ...]]:
    '''Returns the padded array and the per-axis (lo, hi) pad widths so the
    caller (or thrux) can unpad / update the affine.'''
    ...
```

Home: `nitrix.numerics` (sibling to `tensor_ops`).

**C. `crop_to_nonzero` / bounding-box crop — ENABLING (nnUNet / template pre-step).**
`hd_bet`, `fastcsr` (nnUNet `crop_to_nonzero`), `pglands_seg` (MNI-template crop),
`brainiac`. The output shape is data-dependent, so the *index math* (the bbox of
`mask`/`x>thr`) is the `nitrix` piece; the actual slice + affine update is `thrux`.
A pure index-computing primitive keeps `nitrix` jit-clean:

```python
def nonzero_bounding_box(
    x: Float[Array, '*spatial'] | Bool[Array, '*spatial'],
    *,
    threshold: float = 0.0,
) -> Tuple[Int[Array, 'ndim'], Int[Array, 'ndim']]:   # (lo, hi) per axis
    ...
```

Home: `nitrix.numerics`.

**D. Cubic (order-3) resample — MISMATCH (parity-relevant gap).**
`geometry.spatial_transform` / `resample` are **linear-only** — they wrap
`jax.scipy.ndimage.map_coordinates`, which supports `order` 0/1 only (the
documented FreeSurfer-port deviation, ilex SKILL FM #17). `hd_bet`'s nnUNet
preprocessing resamples with order-3 spline; the linear-only path is a documented
parity deviation, not a match. If bit-parity with nnUNet preprocessing is wanted,
`nitrix` needs a cubic-spline resampler (separable B-spline prefilter + cubic
sampling) rather than the `map_coordinates` order cap. Lower priority than A–C
(linear is "good enough" for most consumers); flag the deviation in the
`resample` docstring at minimum.
Home: `nitrix.geometry.grid`.

**E. Intensity-normalize variants — CONVENIENCE (close, but not exact).**
`numerics.normalize` ships `zscore_normalize`, `robust_zscore_normalize`,
`intensity_normalize`, `psc_normalize`, `demean` — but two recurring upstream
recipes are not exactly reachable:

- **min–p99–clip** (`x → clip((x - min) / p99, 0, 1)`): `synthstrip`, `synthdist`,
  `synthsr` (after CT-clip). The existing two-sided percentile path is not this
  strict-min/p99 form.
- **per-channel *nonzero*-masked z-score**: `brats_segresnet`, `brainsegfounder`
  (BraTS multi-channel). Needs the mask = `x != 0` per channel, distinct from the
  global/robust z-score.

Add these as variants/kwargs on the existing `normalize` surface (e.g. a
`percentile_rescale(x, *, lo=0.0, hi=99.0, clip=True)` and a `nonzero_mask=` /
`axis=` channel-wise option) rather than new top-level names.
Home: `nitrix.numerics.normalize`.

**F. Sliding-window weighting kernel + overlap-add stitch — CONVENIENCE (borderline).**
`hd_bet`, `fastcsr`, `wholebrain_unest`, `brats_segresnet`, `synthstrip`, `bme_x`
all tile a large volume and blend overlapping patch logits with a Gaussian-weighted
window. The *orchestration* (tile scheduling, model dispatch) belongs in
`nimox.inference` / `thrux`, but the two numeric pieces are `nitrix`-shaped and
reused by every consumer: (1) the separable Gaussian patch-weight window, and (2)
the weighted overlap-add accumulation `out += w * patch; norm += w` then `out /=
norm`. Ship those as kernels; leave the loop to the orchestrator.
Home: `nitrix.numerics` (window) + the reduction; orchestration stays out.

### 2026-06-02 — Residual phase-1 primitives (mesh / UNet ports)

Small primitives the surface and neurite-UNet ports still vendor; all flagged
"pending" in ilex `UPSTREAM.md` ("Numerical patterns delegated to nitrix →
Pending primitives"). Lower blast-radius than the volumetric tier above.

**A. `point_sample` / `sample_volume_at_points` — CONVENIENCE.**
Trilinear interpolation of a 3-D volume at floating-point vertex coordinates with
**zero-fill** out-of-bounds (distinct from `spatial_transform`'s edge-replicate
`mode='nearest'`). Driver: `topofit` (sampling the image-UNet feature volume at
deforming mesh vertices); future voxel-features-on-vertices consumers. Home:
`nitrix.geometry.grid` (a `mode='zero'` point sampler, or an explicit
`sample_at_points(volume, points, *, mode)`).

**B. `compute_vertex_normals` — CONVENIENCE.**
Per-vertex unit normals from triangle faces (face cross-products → scatter-add to
vertices → normalise). Driver: `topofit`. Home: `nitrix.sparse.mesh` (next to the
`Mesh` dataclass).

**C. `upsample_nearest_nd` — CONVENIENCE.**
Nearest-neighbour spatial upsample; inlined by ~7 neurite-family UNet decoders
(`synthseg`/`synthsr`/`fsm_seg`/`synthstrip`/`exvivo_*`/`voxelmorph`). Pure-array
resize → `nitrix`-shaped. Home: `nitrix.numerics` / `geometry`.

**D. `spatial_transform_batched` — CONVENIENCE (low blast).**
A leading-batch convenience that internally `vmap`s `spatial_transform`; saves one
wrap line per consuming model and keeps the cross-model surface uniform. (Originally
a JOSA-port convenience request.) Home: `nitrix.geometry.grid`.

**E. `LOG` / `EUCLIDEAN` support in `semiring_ell_edge_aggregate` — CONVENIENCE.**
Currently REAL / TROPICAL_MAX_PLUS / TROPICAL_MIN_PLUS only (LOG/EUCLIDEAN raise
`NotImplementedError`); adding them would let log-domain and distance-based mesh
aggregations (attention-softmax-in-log, geodesic propagation) compose on the same
primitive. **Canonical entry: `internal-backlog.md` B4** (parked with trigger).

## Resolved

_(none yet — reference the resolving `nitrix` commit on fix.)_

The large mesh/surface/warp tier these ports needed is already shipped; its
resolution history lives in `IMPLEMENTATION_PLAN.md §10.3` and
`SPEC_UPDATE_v0.3 §10.A` (and git history), not here.
