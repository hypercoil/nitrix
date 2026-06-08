# PCA fit / transform / inverse (SVD) — `nitrix.stats.pca`

> **Status (2026-06-08): SHIPPED.** `stats/pca.py` adds `pca_fit`
> (`PCAResult` NamedTuple of `components`/`mean`/`explained_variance`),
> `pca_transform`, `pca_inverse_transform`. The basis is the **covariance
> eigendecomposition via `linalg._solver.safe_eigh`** — deliberately *not*
> `jnp.linalg.svd`, which is dead on the cuSolver-affected GPU here (verified:
> `pca_fit` runs on GPU via the transparent CPU fallback). Deterministic
> `svd_flip` sign convention; `transform`/`inverse` take explicit
> `(components, mean)` so a pre-fit basis (loaded weights) applies without a
> local fit. Model-numeric item from the 2026-06-08 ilex audit
> ([`ilex-training-substrate.md`](ilex-training-substrate.md)).

**What.** Principal-component analysis as a small pure-numeric triple:

- **`pca_fit(X) -> (components, mean, explained_variance)`** — center, SVD,
  return the top components. The genuinely-missing primitive.
- **`pca_transform` / `pca_inverse_transform`** — the centered affine
  projection `(x − mean) @ componentsᵀ` and its inverse. `krakencoder` ships
  these as a pre-fit module (`ilex/models/krakencoder/_pca.py:111`), so it
  only *applies* PCA — but the fit is the reusable substrate gap, and the
  apply/inverse are 2-line affines worth co-locating.

**Drivers.** `krakencoder` (connectome PCA stack, `_pca.py`); any future
dimensionality-reduction front-end. General enough to sit beside
`stats.covariance` (PCA is the eigendecomposition of the covariance, which
nitrix already computes).

**API sketch.**

```python
def pca_fit(X: Float[Array, 'n d'], *, n_components: int | None = None
            ) -> tuple[Float[Array, 'k d'], Float[Array, 'd'], Float[Array, 'k']]:
    """Returns (components, mean, explained_variance) via centered SVD."""

def pca_transform(X, components, mean) -> Array: ...
def pca_inverse_transform(Z, components, mean) -> Array: ...
```

**Pure / XLA note.** `jnp.linalg.svd` (or eigh on the Gram for `d ≪ n`) +
matmul; jit-clean for static `n_components`. On GPU, route the
factorisation through `linalg._solver.safe_*` (cuSolver-pool fallback). The
`KrakencoderPCAStack` dict-of-modules container stays in nimox; this is the
kernel it would call.

**Home.** `nitrix.stats.pca` (new module beside `covariance.py`). Returns a
plain tuple / NamedTuple of arrays — not a PyTree module (SPEC §2).

## Cross-references

- [`ilex-training-substrate.md`](ilex-training-substrate.md) — survey context.
- `src/nitrix/stats/covariance.py` — PCA is the eigendecomposition of this;
  shared substrate.
- [`ledoit-wolf-shrinkage.md`](ledoit-wolf-shrinkage.md),
  [`graphical-lasso.md`](graphical-lasso.md) — the other covariance-family
  estimators.
