# Normalised-cut spectral clustering — `nitrix.graph.ncut`

> **Status (2026-07-07): SHIPPED** as `nitrix.graph.normalized_cut`
> (Shi–Malik / Ng–Jordan–Weiss). The thin composition `laplacian_eigenmap`
> (shipped) + `numerics.kmeans` (shipped §12.18); default `similarity='cosine'`
> gives the NJW row-normalisation for free. Recovers planted communities;
> GPU-verified. Provenance: `docs/feature-requests catalogue §12.19`.

**What.** Shi–Malik 1997 / Craddock 2012-style normalised-cut spectral
clustering — a thin composition wrapper.

**Composition.** A ~5-line composition:

1. `laplacian_eigenmap` (shipped, with implicit-VJP through both `eigh` and
   `lobpcg`) gives the top-`k` eigenvectors of the normalised Laplacian.
2. `kmeans(eigvecs, k)` ([`clustering-primitives.md`](../clustering-primitives.md),
   §12.18) discretises into parcels.

**Likely consumer.** Craddock 2012 functional parcellation, NCut-based ROI
generation for connectome edges.

**Effort.** XS — depends on §12.18.

**Live-code status.** No `graph.ncut`. The eigenvector half is shipped
(`graph.laplacian_eigenmap`); the only missing ingredient is `kmeans`,
tracked in [`clustering-primitives.md`](../clustering-primitives.md).

## Cross-references

- `docs/feature-requests catalogue §12.19` — origin entry; `§13` — acceptance protocol;
  `§12.20` — strategy survey (Craddock NCut row).
- [`clustering-primitives.md`](../clustering-primitives.md) — the `kmeans`
  dependency.
- `src/nitrix/graph/connectopy.py` — `laplacian_eigenmap`.
