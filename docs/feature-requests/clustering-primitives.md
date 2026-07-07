# Clustering primitives — `nitrix.numerics.cluster`

> **Status (2026-07-07): PARTIAL — `kmeans` SHIPPED; `ward_linkage` / `nmf`
> deferred to the parcellation sprint.** `numerics.cluster.kmeans` /
> `kmeans_fit` / `kmeans_predict` (Lloyd; euclidean/cosine/correlation) on the
> §6.5 fit/apply seam (`KMeansState`, `similarity` as static aux; `n_init`
> restarts). Homed in **`numerics`** (geometric clustering) per the SPEC §6.4
> distribute-by-numerical-kind decision — **no `parcellation` namespace**
> (application assembly stays downstream); probabilistic mixtures (GMM) will go
> to `stats`. This was the load-bearing piece: it unblocked
> [`normalised-cut`](resolved/normalised-cut.md) (SHIPPED). `ward_linkage` (host-side
> agglomerative) and `nmf` reuse the same seam + `similarity` and are the
> sprint follow-ups. Provenance: `docs/feature-requests catalogue §12.18`.

**What.** A small family of clustering primitives several parcellation
strategies share.

**Proposed surface.**

```python
def kmeans(X, k, *, init, max_iter, similarity): ...   # Lloyd; cosine/corr/euclid
def ward_linkage(X, *, k): ...                          # agglomerative (Ward)
def nmf(X, k, *, max_iter): ...                         # Lee–Seung NMF
```

**Composition.**

- `kmeans` — Lloyd's algorithm (vmap over centroid update +
  nearest-centroid assignment). Pure JAX; differentiable via implicit-VJP at
  convergence (composes [`fixed-point-combinators.md`](resolved/fixed-point-combinators.md),
  §12.8).
- `ward_linkage` — host-side (heap priority queue), JAX-array output.
- `nmf` — Lee–Seung multiplicative updates. Pure JAX; differentiable via
  implicit-VJP at the KKT stationary point.

Similarity-matrix construction composes the shipped `stats.corr` /
`stats.cov` / `linalg.linear_kernel`.

**Likely consumer.** Yeo 7/17 networks (k-means on connectivity profiles),
Bellec MIST (Ward linkage on stability matrices), NMF parcellations
(Eickhoff group), ICA-followup clustering. Directly unblocks
[`normalised-cut.md`](resolved/normalised-cut.md) (§12.19), which is `kmeans` on
Laplacian eigenvectors.

**Effort.** S for `kmeans`; S for `nmf`; M for `ward_linkage` (linkage-matrix
bookkeeping).

**Live-code status.** No `kmeans` / `ward_linkage` / `nmf` and no
`nitrix.numerics.cluster` namespace. The similarity-matrix substrate is
shipped (`stats.corr` / `cov`, `linalg.linear_kernel` / `cosine_kernel`).

## Cross-references

- `docs/feature-requests catalogue §12.18` — origin entry; `§13` — acceptance protocol;
  `§12.20` — strategy survey (Yeo / Bellec / Eickhoff rows).
- [`normalised-cut.md`](resolved/normalised-cut.md) — downstream (kmeans on eigvecs).
- [`fixed-point-combinators.md`](resolved/fixed-point-combinators.md) — convergence
  gradients.
- `src/nitrix/linalg/kernel.py` — the similarity kernels.
