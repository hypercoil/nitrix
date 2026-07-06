# Adaptive area-weighted barycentric resampling — `nitrix.geometry.sphere.resample`

> **Status (2026-07-06): SHIPPED.** `geometry.surface_resample` exposes both
> `method='barycentric'` and `method='adap_bary_area'` — the adaptive
> area-weighted case (the shipped **default**, `geometry/sphere.py`) is
> complete, alongside the icosphere `mesh_bary_upsample` /
> `icosphere_bary_upsampler`. Provenance:
> `docs/feature-requests catalogue §12.15`.

**What.** Connectome Workbench's `ADAP_BARY_AREA` cross-mesh resampling
(`wb_command -metric-resample` / `-surface-resample`), for the
arbitrary-triangulation case (native `surf.gii` → `fs_LR_32k`).

**Proposed surface.**

```python
def surface_resample(
    source_mesh: Mesh, source_vals: Float[Array, '... n_source d'],
    target_mesh: Mesh,
    *, method: Literal['barycentric', 'adap_bary_area'] = 'adap_bary_area',
) -> Float[Array, '... n_target d']: ...
```

Returns an `ELL` (the resampling operator, reusable across features for the
same source/target pair) plus the resampled values. Host-side construction
(point-in-triangle, area integrals); JAX-native application via
`semiring_ell_matmul`.

**Composition.** Standard `BARYCENTRIC` is the existing `mesh_bary_upsample`
path (per-target-vertex barycentric weights against the containing source
triangle), already shipped for icosphere hierarchies
(`icosphere_bary_upsampler`). The two gaps:

- the **arbitrary-triangulation** case (native mesh → standard mesh), and
- **`ADAP_BARY_AREA`** — weight each source-vertex contribution by source-
  triangle area (preserving total areal measurement under resampling —
  critical for vertex-area scalars, cortical-thickness maps), with the
  *adaptive* width adjustment when source/target tessellation densities
  differ.

**Likely consumer.** Any port consuming Connectome Workbench outputs (the
bulk of HCP / UK Biobank surface data), `fs_LR` ↔ `fsaverage`
round-tripping, individualised-parcellation pipelines resampling
subject-native → group surface.

**Effort.** M. Point-in-triangle on the sphere is the non-trivial part
(spherical-triangle bary coordinates, not the planar formula); the
adaptivity rule is a closed-form weight adjustment from Workbench's source.

**Live-code status.** `sparse/__init__` ships `mesh_bary_upsample`,
`icosphere_bary_upsampler`, `icosphere_cross_level_adjacency`, and the `ELL`
application path — the icosphere-hierarchy `BARYCENTRIC` case. No
arbitrary-mesh `surface_resample` and no area-weighted / adaptive variant.

## Cross-references

- `docs/feature-requests catalogue §12.15` — origin entry; `§13` — acceptance protocol.
- `src/nitrix/sparse/mesh.py` — `mesh_bary_upsample` / `icosphere_bary_upsampler`.
- [`ilex-pipeline-substrate.md`](../ilex-pipeline-substrate.md) — the
  surface↔sphere parameterisation scope boundary (the `surfa`-side projection
  stays with the consumer; the array resample is `nitrix`).
