# Clustering primitives — `nitrix.numerics.cluster`

> **Status (2026-06-02): not started — the load-bearing missing piece for
> the "functional-parcellation by clustering" family.** Brainstorm
> candidate; promotion gated by the §13 acceptance protocol. Provenance:
> `SPEC_UPDATE_v0.3.md §12.18`.

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
  convergence (composes [`fixed-point-combinators.md`](fixed-point-combinators.md),
  §12.8).
- `ward_linkage` — host-side (heap priority queue), JAX-array output.
- `nmf` — Lee–Seung multiplicative updates. Pure JAX; differentiable via
  implicit-VJP at the KKT stationary point.

Similarity-matrix construction composes the shipped `stats.corr` /
`stats.cov` / `linalg.linear_kernel`.

**Likely consumer.** Yeo 7/17 networks (k-means on connectivity profiles),
Bellec MIST (Ward linkage on stability matrices), NMF parcellations
(Eickhoff group), ICA-followup clustering. Directly unblocks
[`normalised-cut.md`](normalised-cut.md) (§12.19), which is `kmeans` on
Laplacian eigenvectors.

**Effort.** S for `kmeans`; S for `nmf`; M for `ward_linkage` (linkage-matrix
bookkeeping).

**Live-code status.** No `kmeans` / `ward_linkage` / `nmf` and no
`nitrix.numerics.cluster` namespace. The similarity-matrix substrate is
shipped (`stats.corr` / `cov`, `linalg.linear_kernel` / `cosine_kernel`).

## Cross-references

- `SPEC_UPDATE_v0.3.md §12.18` — origin entry; `§13` — acceptance protocol;
  `§12.20` — strategy survey (Yeo / Bellec / Eickhoff rows).
- [`normalised-cut.md`](normalised-cut.md) — downstream (kmeans on eigvecs).
- [`fixed-point-combinators.md`](fixed-point-combinators.md) — convergence
  gradients.
- `src/nitrix/linalg/kernel.py` — the similarity kernels.
