# `pairedcorr` forms the full `cov(X)` / `cov(Y)` just to read their diagonals — ~2× redundant matmul

> **Status (2026-06-10): perf finding, micro-optimisation candidate.**
> Surfaced by `nitrix-perf-bench` benching the paired / conditional stats family
> across scale on an L4 (nitrix-jax vs numpy CPU floor + cupy GPU twin,
> `jax-cpu` / `jax-cuda12`; see that repo's `pairedcorr` / `pairedcov` cases +
> `tools/scaling_report.py`). Not a regression — the result is correct; it just
> does ~3× the matmul FLOPs it needs to.

## The redundancy

`pairedcorr` (`src/nitrix/stats/covariance.py:313–328`) normalises the
cross-covariance by the geometric mean of the two blocks' per-variable
variances, and gets those variances by forming the **full** covariance matrices
and slicing their diagonals:

```python
sigma_xy = pairedcov(X, Y, **kwargs)                                  # (c, d)
sigma_xx_diag = jnp.diagonal(cov(X, **kwargs), axis1=-2, axis2=-1)    # (c,)
sigma_yy_diag = jnp.diagonal(cov(Y, **kwargs), axis1=-2, axis2=-1)    # (d,)
norm = jnp.sqrt(sigma_xx_diag[..., :, None] * sigma_yy_diag[..., None, :])
return sigma_xy / (norm + jnp.finfo(norm.dtype).eps)
```

`cov(X)` is a full `(c, c)` matmul — `O(c² · obs)` — but only its **diagonal**
(the `c` per-variable variances) is used; likewise `cov(Y)` is a full `(d, d)`
matmul for `d` variances. XLA does **not** rewrite `diag(Xc @ Xcᵀ)` into the
`O(c · obs)` row-wise sum-of-squares — it materialises the full product and then
slices — so the op runs **three** `O(c²·obs)`-class matmuls (`cov(X)`, `cov(Y)`,
and the genuinely-needed cross-cov) where **one** plus two cheap `O(c·obs)`
variance reductions would do.

## Measured (L4; `c = d`, `obs` to whole-brain-parcel scale)

The numpy / cupy reference computes the variances directly
(`(Xc*Xc).sum(-1)/(n−1)`), so the gap below **is** the redundant full-cov work.
`min` steady-state, post-warm-up:

| size (`c=d`, `obs`) | nitrix-jax GPU | cupy GPU (direct) | gap | nitrix-jax CPU | numpy CPU (direct) | gap |
|---|---|---|---|---|---|---|
| 128, 1024  | 0.13 ms | 0.35 ms | nitrix 2.8× ahead* | 0.75 ms | 0.54 ms | numpy 1.4× |
| 256, 2048  | 0.21 ms | 0.37 ms | nitrix 1.7× ahead* | 7.34 ms | 2.54 ms | numpy 2.9× |
| 512, 4096  | 0.81 ms | 0.38 ms | **cupy 2.1×** | 35.7 ms | 20.6 ms | **numpy 1.7×** |
| 1024, 4096 | 2.37 ms | 1.18 ms | **cupy 2.0×** | 153 ms  | 71.4 ms | **numpy 2.1×** |
| 2048, 8192 | 19.2 ms | 9.21 ms | **cupy 2.1×** | 1189 ms | 523 ms  | **numpy 2.3×** |

\* At tiny `c` nitrix's fixed launch overhead wins regardless; the redundant
matmul only dominates once the `O(c²·obs)` term is the cost, from `c ≳ 512`,
where the gap settles at a clean **~2×** on both CPU and GPU.

**Control:** `pairedcov` — the *same* cross-cov but with **no** normaliser —
sits at parity with both refs (GPU 0.97–1.39× vs cupy, CPU 0.92–1.23× vs numpy)
across the identical tier. So the ~2× is isolated to the corr normaliser's two
full covariance matmuls, not the shared cross-cov path.

## Fix

Read the diagonals directly instead of forming the matrices — the per-variable
variance is `O(c · obs)`:

```python
# unweighted / vector-weighted: the diagonal of cov without the matmul
Xc = X - _weighted_mean(X, ...)
var_x = (Xc * Xc * w).sum(-1) / fact      # diag(cov(X)), O(c·obs)
```

i.e. a small `_variance`/`_cov_diag` helper sharing `cov`'s centring +
`_denom_factor` (so `ddof` / `bias` / `weights` stay consistent), used in place
of `jnp.diagonal(cov(...))`. The full-matrix `weight_matrix` (`W`) path can keep
its current form or use `einsum('ci,ij,cj->c', Xc, W, Xc)`; the common
unweighted / vector-weighted path is where the ~2× lives. Result is identical
(verified to ~1e-16 in the perf-bench oracle).

`pairedcov` needs no change (no normaliser). `conditionalcorr` is **not**
affected: its `_corrnorm` reads the diagonal of the conditional covariance that
the op already had to form `(c, c)`, so there is no extra matmul there.

## How it is measured

perf-bench cases `pairedcorr` / `pairedcov` (vs an fp64 numpy oracle + a cupy
GPU twin, both computing the variances directly), brain-parcel size tier
(`c = d` to 2048, `obs` to 8192) + `tools/scaling_report.py` (the scale-gaming
defence — `pairedcorr` is flagged a scale risk: the direct-variance baseline is
~2× ahead at the largest size, while `pairedcov` is not).

## Cross-references

- `src/nitrix/stats/covariance.py:313–328` (`pairedcorr`); `:266–276` (`corr`,
  which legitimately needs the full `(c,c)` cov it normalises — not the same
  case); `:149–189` (`_cov_core` / `_denom_factor`, the centring + ddof a
  `_cov_diag` helper would share).
- [`perf-bench-feedback.md`](perf-bench-feedback.md) — the perf-bench finding
  ledger.
