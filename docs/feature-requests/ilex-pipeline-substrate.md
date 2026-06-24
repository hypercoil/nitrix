# ilex pipeline-substrate — survey context & index

> **This doc is now the survey *context + index*.** Each individual
> primitive request has been atomised into its own tracking doc (one doc per
> proposal, to reduce duplicate-issue risk); this file keeps the shared
> survey framing, the scope boundary, the already-shipped record, and the
> index of the atomised items. See [`README.md`](README.md) for the
> directory-wide index.

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
aware operator chain *whose kernels exist in `nitrix`*. The atomised items below
are the phase-1 kernel shopping list.

**What is already done (do not re-request).** The mesh/surface and warp/registration
substrate the surface ports needed is shipped: `semiring_ell_edge_aggregate(+edge_attr)`,
`ell_row_softmax`, `ell_add_self_loops`, `ell_mask`, `icosphere_hierarchy(_from_levels)`
/ `_cross_level_adjacency` / `_bary_upsampler`, `mesh_pool_max` / `_unpool_max` /
`_coarsen_meanpool` / `_bary_upsample`, `max_pool_with_indices_nd` / `max_unpool_nd`,
`geometry.{spatial_transform(mode='nearest'), integrate_velocity_field,
jacobian_det_displacement, sphere_grid_pad_2d}`, `smoothing.gaussian(kernel_size=)`,
and `bias.histogram_match` (Nyul–Udupa). The resolution history for that whole
tier lives in `IMPLEMENTATION_PLAN.md §10.3` (shipped-deviation log) and
`SPEC §9`. The remaining gap is the *mundane, ubiquitous volumetric
pre/post-processing* every CNN/UNet/ViT v0 punted on — that is what the atomised
items collect.

> Provenance note: active feature requests live here from now on. The pre-directory
> ledgers (`NITRIX_FEEDBACK_ILEX.md`, `NITRIX_FEEDBACK_JOSA.md`, top-level
> `BACKLOG.md`) were swept on 2026-06-02 — their genuinely-open items migrated to
> this directory, verified against the live code, so those three files could be
> deleted.

## Scope boundary (what is NOT nitrix)

So the phase-1 lift does not over-reach:

- **`conform` (resample + reorient + crop in vox2world / world space)** → `thrux`
  (container/affine-aware), built *on* the `nitrix` resample kernel. The array
  resample is `nitrix`; the affine bookkeeping is not.
- **Affine fit / compose / decompose** (SynthMorph barycenter + weighted-LS) →
  already in `nimox.modules.affine`.
- **Surface↔sphere parameterisation** (`surfa.SphericalMapBarycentric`),
  FreeSurfer `.sphere` / `.mgz` I/O, atlas label-LUT remap → consumer / `thrux`;
  explicitly rejected from `nitrix` (SPEC §6.2).
- **Sliding-window inference orchestration** (tiling, scheduling) → `nimox.inference`
  / `thrux`; only the per-window weighting kernel + overlap-add reduction is
  `nitrix`-shaped ([`sliding-window-weighting.md`](sliding-window-weighting.md)).

## Atomised items

### Volumetric pre/post-processing substrate tier (ilex → thrux)

Severity by migration goal: **ENABLING** = a pipeline cannot move to `thrux`
without it; **CONVENIENCE** = works inline today, a shared primitive removes
duplication; **MISMATCH** = a documented parity deviation.

| Item | Doc | Severity | Home |
|---|---|---|---|
| Connected-components / largest-CC | [connected-components](connected-components.md) | ENABLING (highest recurrence) | `morphology` |
| `pad_to_multiple` / `crop_to_multiple` | [pad-to-multiple](pad-to-multiple.md) | ENABLING | `numerics` |
| `crop_to_nonzero` / bbox | [crop-to-nonzero](crop-to-nonzero.md) | ENABLING | `numerics` |
| Cubic (order-3) resample | [cubic-resample](cubic-resample.md) | MISMATCH | `geometry.grid` |
| Intensity-normalize variants | [intensity-normalize-variants](intensity-normalize-variants.md) | CONVENIENCE | `numerics.normalize` |
| Sliding-window weighting + overlap-add | [sliding-window-weighting](sliding-window-weighting.md) | CONVENIENCE | `numerics` |

### Residual phase-1 primitives (mesh / UNet ports)

Small primitives the surface and neurite-UNet ports still vendor; all flagged
"pending" in ilex `UPSTREAM.md`. Lower blast-radius than the volumetric tier.

| Item | Doc | Severity | Home |
|---|---|---|---|
| `point_sample` (zero-fill) | [point-sample](point-sample.md) | CONVENIENCE | `geometry.grid` |
| `compute_vertex_normals` | [compute-vertex-normals](compute-vertex-normals.md) | CONVENIENCE | `sparse.mesh` |
| `upsample_nearest_nd` | [upsample-nearest-nd](upsample-nearest-nd.md) | CONVENIENCE | `numerics` / `geometry` |
| `spatial_transform_batched` | [spatial-transform-batched](spatial-transform-batched.md) | CONVENIENCE (low blast) | `geometry.grid` |
| `LOG` / `EUCLIDEAN` `edge_aggregate` | [edge-aggregate-log-euclidean](edge-aggregate-log-euclidean.md) (B4, canonical) | CONVENIENCE | `semiring` |

## Resolved

Shipped 2026-06-02 (see `IMPLEMENTATION_PLAN.md §10.3`, 2026-06-02 entry):

- **Intensity-normalize variants** ([intensity-normalize-variants](intensity-normalize-variants.md))
  — `percentile_rescale` (min–p99–clip) + `zscore_normalize(nonzero_mask=)`.
- **`spatial_transform_batched`** ([spatial-transform-batched](spatial-transform-batched.md))
  — leading-batch `vmap` with shared-operand broadcast.
- **Cubic resample — docstring deviation flagged** ([cubic-resample](cubic-resample.md));
  the full order-3 B-spline path stays deferred (a new sampling path, not a fix).

The large mesh/surface/warp tier these ports needed is already shipped; its
resolution history lives in `IMPLEMENTATION_PLAN.md §10.3` and
`SPEC §9` (and git history), not here.
