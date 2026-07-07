# Spatial null models (spin / Moran / variogram) — `nitrix.stats.inference` + `nitrix.geometry.sphere`

> **Status (2026-07-07): the spin / Moran / BrainSMASH trio is SHIPPED** (on
> the shared surrogate→test seam) — the most-common (spin), a spectral
> (Moran), and the most-rigorous parameterized (BrainSMASH variogram) nulls.
> The spin refinements are **complete** (medial-wall, per-hemisphere, Váša
> bijective, parcel-level) and the BrainSMASH **'sampled'** large-mesh (KNN,
> memory-lean) variant is now shipped. Remaining: the mesh/graph-TFCE
> clustering half of N2. Family ledger for
> the spatial-autocorrelation-preserving null models used to test the
> correspondence of two brain maps. Origin: the **N2** item of the
> [`stats-suite-audit`](stats-suite-audit.md) ("no spin test (Alexander-Bloch/
> Váša)"), deferred *post-geometry-suite* — now unblocked. The gap was also
> recorded in the [`hypercoil-examples-migration`](hypercoil-examples-migration.md)
> (ProMises is *not* a spin/variogram/Moran null). Reference implementation:
> **`neuromaps.nulls`**.

## Why

Correlating two brain maps with an ordinary p-value is anticonservative: both
maps carry strong spatial autocorrelation (SA), so even unrelated maps
correlate. A spatial null builds a surrogate distribution that **preserves each
map's SA** and tests the observed statistic against it.

## Design — distribute by numerical kind (the parcellation precedent, SPEC §6.4)

A spatial null = a **surrogate generator** (method-specific) + a **null test**
(shared). Generators live with their numerical kind; the test lives in
`stats.inference`:

| Method | Generator kind | Home |
|---|---|---|
| **Spin** (rotate sphere → reassign nearest vertex) | geometric | `geometry.sphere` |
| **Moran spectral randomization** (randomise Laplacian-eigenmap coefficients) | spectral | composes shipped `graph.laplacian_eigenmap` |
| **Variogram / BrainSMASH (Burt 2020)** (distance-matched smoothing of a permuted map) | generative | geometry distance + smoothing |
| **The null test** (observed stat vs surrogate null → p) | inference | `stats.inference` |

Seam to hoist once ≥2 generators exist: `spatial_null_test(x, y, surrogates,
stat) -> (observed, p)` — generator pluggable, test shared (mirrors the shipped
`permutation_test` p-value machinery).

## Shipped — the spin test (2026-07-07)

- `geometry.sphere.random_rotation(key, n)` — uniform SO(3) via **Shoemake's
  quaternion** method (factorisation-free → cuSolver-independent; an Euler-angle
  draw is *not* uniform on the group).
- `geometry.sphere.spin_surrogates(coords, x, rotations)` — rotate + nearest-
  vertex reassignment (Alexander-Bloch / Vázquez-Rodríguez); `lax.map`-streamed
  (peak memory `O(V²)`, not `O(nV²)`).
- `stats.inference.spin_test(x, y, coords, *, key, n_spin, two_sided)` — observed
  Pearson correlation, rotated null, add-one two-sided permutation p.

## Shipped — Moran spectral randomization + the shared seam (2026-07-07)

- `graph.moran_surrogates(A, x, n, key)` — Wagner & Dray (2015) MSR: project the
  mean-centred map onto the Moran eigenvectors of the doubly-centred weight
  matrix (via the cuSolver-robust `safe_eigh`), sign-flip the coefficients,
  reconstruct. Preserves mean / variance / the Moran's-`I` spectrum; **needs no
  coordinates** (surface *or* volume graph).
- `stats.inference.spatial_null_test(x, y, surrogates, *, two_sided)` +
  `SpatialNullResult` — the **generator-agnostic test** (observed / null / p).
  `spin_test` and `moran_test` are thin wrappers over it; `SpinTestResult`
  aliases `SpatialNullResult`.
- `stats.inference.moran_test(x, y, adjacency, *, key, n_surrogates, two_sided)`.

## Shipped — BrainSMASH variogram-matched generative null (2026-07-07)

The most general / rigorous parameterized null (Burt 2020): reproduces the map's
empirical **variogram** from a distance matrix, so it works for surface *and*
volume without a sphere or a graph.

- `stats.inference.variogram(x, D, *, n_bins)` — binned empirical semivariogram.
- `stats.inference.brainsmash_surrogates(x, D, n, key, *, deltas, kernel,
  resample)` — permute → multi-bandwidth distance-kernel smooth → NNLS-match the
  target variogram (projected-gradient, **cuSolver-free**) → rank-resample to
  the target marginal (default).
- `stats.inference.brainsmash_test`.

Fully `jit`/`vmap`-clean on **any** backend (no `eigh`), unlike Moran.

31 tests total (spin 9 + Moran/seam 6 CPU-effective + BrainSMASH 8, plus the seam
mechanics); spin + BrainSMASH GPU-verified incl. `jit`, Moran eager-verified on
GPU (`safe_eigh`'s device fallback is eager-only → `jit`-eigh needs a healthy
cuSolver).

## Roadmap (want at least the most-common + a rigorous parameterized model)

1. ✅ **Moran spectral randomization (MSR)** — SHIPPED (`graph.moran_surrogates`
   + `stats.inference.moran_test`; sign-flip the doubly-centred weight-matrix
   eigen-coefficients). Needs no coords; surface *or* volume.
2. ✅ **Variogram / BrainSMASH (Burt 2020)** — SHIPPED (`variogram` +
   `brainsmash_surrogates` + `brainsmash_test`). The **'sampled'** large-mesh
   subset variant is also shipped (`brainsmash_surrogates_sampled` +
   `brainsmash_test_sampled`): memory-lean :math:`O(nk)` surrogates from each
   vertex's `k` nearest neighbours (index + distance arrays), variogram
   estimated over a random seed subset — no dense :math:`n \times n` matrix.
3. ✅ **Spin refinements — COMPLETE.** Medial-wall (the shared
   `spatial_null_test` statistic is non-finite-aware, dropping `NaN` pairwise
   over each spin's finite support); per-hemisphere (`spin_surrogates` /
   `spin_test(..., hemisphere=)` — independent rotation with the Alexander-Bloch
   mirror reflection + within-hemisphere reassignment); Váša **bijective**
   (`assignment='bijective'` — greedy one-to-one, each surrogate an exact
   permutation); and **parcel-level** (`geometry.parcel_centroids` → spin on
   the parcel centroids).
4. ✅ **The shared `spatial_null_test` seam** — SHIPPED
   (`stats.inference.spatial_null_test` + `SpatialNullResult`); `spin_test` /
   `moran_test` are thin wrappers, and a new generator plugs straight in.

## Cross-references

- [`stats-suite-audit.md`](stats-suite-audit.md) — the N2 origin item.
- [`hypercoil-examples-migration.md`](hypercoil-examples-migration.md) — where the
  gap was recorded (ProMises is not a spatial null).
- `neuromaps.nulls` — the reference (Markello & Misić 2021, *Comparing spatial
  null models for brain maps*).
