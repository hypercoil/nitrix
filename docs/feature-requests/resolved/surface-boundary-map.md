# Surface-boundary / gradient mapping — `nitrix.graph.parcellation.boundary`

> **Status (2026-07-06): SHIPPED** (via the geometry suite).
> `graph.parcellation.surface_boundary_map` — the named wrapper on
> `edge_aggregate` + `eta_squared` — tested in `test_parcellation.py`.
> Provenance: `docs/feature-requests catalogue §12.16`.

**What.** The Cohen / Wig / Gordon / Schaefer functional-parcellation
boundary-detection lineage (Cohen 2008; Wig 2014; Gordon 2016; Schaefer
2018): per-vertex connectivity profiles → adjacent-profile dissimilarity →
per-vertex boundary map.

**Proposed surface.**

```python
def surface_boundary_map(
    connectivity_profiles: Float[Array, 'n_vertices d_profile'],
    adjacency: ELL,
    *,
    similarity: Literal['eta_squared', 'pearson'] = 'eta_squared',
    aggregate: Literal['mean', 'max'] = 'mean',
) -> Float[Array, 'n_vertices']: ...

def eta_squared(x, y): ...   # companion similarity helper (composes stats)
```

**Composition.** Exactly the shape of `semiring_ell_edge_aggregate` under
`REAL` / `TROPICAL_MAX_PLUS` with
`edge_fn(h_i, h_j, w, ij) = (1 − corr(h_i, h_j))[None]`. **No new
primitive** — ship a named wrapper for discoverability plus the
`eta_squared` helper. Connectivity profiles come from `stats.corr` on the
`(n_vertices, T)` vertex time-series matrix.

Memory caveat at `ico_7`: the profile is `163842`-dim per vertex; consumers
tile via `jax.lax.map` over the row axis (documented pattern, no API
surface).

**Likely consumer.** Individualised parcellation pipelines
(MyConnectome-style, Gordon-individual), gradient-based parcellation
regeneration, surface-based boundary visualisation for QA. Feeds
[`mesh-watershed.md`](mesh-watershed.md) (§12.17) — local minima of `−B` are
catchment-basin seeds.

**Effort.** S. Three lines on top of `edge_aggregate` plus the `eta_squared`
helper; documentation surface is the bulk of the work.

**Live-code status.** No `surface_boundary_map` / `eta_squared`. Every
ingredient is shipped: `semiring_ell_edge_aggregate`, `semiring.REAL` /
`TROPICAL_MAX_PLUS`, `stats.corr`. The `graph` module already hosts the
adjacent parcellation-substrate functions (`laplacian`, `modularity_matrix`,
`diffusion_embedding`).

## Cross-references

- `docs/feature-requests catalogue §12.16` — origin entry; `§13` — acceptance protocol;
  `§12.20` — the functional-parcellation strategy survey this enables
  (Gordon / Schaefer rows).
- [`mesh-watershed.md`](mesh-watershed.md) — the natural downstream
  (boundary map → discrete parcels).
- `src/nitrix/semiring/ell_edge.py` — `semiring_ell_edge_aggregate`.
