# Make `gp_fit`'s lengthscale-search epilogue traceable (jit / vmap the GP fit)

> **Status (2026-06-25): SHIPPED (Tier-A).** The Gaussian HSGP rho-search
> epilogue is traceable: a JAX-native `_parabolic_argmin_jax` (grid argmin +
> 3-point parabolic refine, `jnp.where` fallbacks → a **traced** scalar)
> replaces the host `_parabolic_argmin(np.asarray(...))`, `rho_hat` stays a
> traced scalar (dropped the `float()`), and `_assemble_gp_result` uses
> `jnp.log` so it flows through. `gp_fit` (Gaussian HSGP) now runs under
> `jax.jit` / `jax.vmap` with the covariate closed over —
> `vmap(lambda Y: gp_fit(Y, x, ...))(Y_stack)` returns per-dataset `GPResult`s
> that **fp-match a loop of eager fits** (`tests/test_gp_traceable.py`); the 53
> existing GP fit/predict tests stay green (the eager path now runs the new
> code — a tracing change, not a numerics change). **Deferred:** Tier-B (traced
> `x`, a result-schema change) and the non-Gaussian PQL / exact / varying-coef
> host searches (still eager; lower priority — Gaussian HSGP is the default and
> the nimox estimator's primary path). nimox `GaussianProcessRegressor` can flip
> its eager-only witness to a vmap-fit proof.
>
> **Original request (2026-06-25): nimox-estimators consumer → nitrix.** `gp_fit` is
> currently **eager-only**: its REML lengthscale (``rho``) selection computes the
> per-grid NLL in pure JAX (``lax.map``) but then drops to the host for the
> sub-grid refinement — `_parabolic_argmin(np.asarray(log_rho_grid),
> np.asarray(nll_grid))` followed by `rho_hat = float(np.exp(log_rho_hat))`
> (`stats/gp.py:932-935`). The `np.asarray(nll_grid)` on the **traced** grid and
> the `float(...)` cast make `gp_fit` raise under `jax.jit` / `jax.vmap`
> (`TracerArrayConversionError`). Fine for one-shot fits (and nimox ships the GP
> estimator today on that basis — `gp_fit` already vectorises over the `V`
> elements internally), but it **bites the moment someone wants the fit inside a
> larger jitted / vmapped workflow** — a normative-model training loop, vmap over
> CV folds / bootstrap resamples / multiple datasets sharing a covariate. nimox's
> `GaussianProcessRegressor` documents the constraint and a test witnesses it; this
> FR asks whether nitrix can lift it, because the blocker looks small and
> self-contained.

## The diagnosis (Gaussian HSGP path — the common one)

```python
nll_grid = lax.map(_pooled_nll, log_rho_grid)        # (n_rho,)  PURE JAX already
log_rho_hat = _parabolic_argmin(
    np.asarray(log_rho_grid), np.asarray(nll_grid)   # <-- host: traced -> numpy
)
rho_hat = float(np.exp(log_rho_hat))                 # <-- host: concretise
```

`_parabolic_argmin` is a **fixed-shape, data-independent** computation: take the
grid argmin, clamp it off the boundary, fit a parabola through the three
bracketing points, return its vertex (with grid-point fallbacks at a boundary
minimum / non-convex bracket). There is no dynamic shape and no control flow that
can't be a `jnp.where` — it is exactly the kind of thing that rewrites to pure
JAX 1:1.

## The ask (Tier A — small, high-value)

Make the rho-search **epilogue** traceable:

1. Rewrite `_parabolic_argmin` JAX-native: `i = jnp.argmin(nll)`; clamp to
   `[1, n-2]`; gather the three bracketing `(log_rho, nll)` points by
   `jnp.take` / `lax.dynamic_slice` on the clamped index; compute the vertex
   closed-form; replace the Python `if` boundary / non-convex / degenerate
   branches with `jnp.where`. Returns a **traced scalar** `log_rho_hat`.
2. Keep `rho_hat = jnp.exp(log_rho_hat)` a **traced scalar** (drop the `float()`),
   threading it into the existing `pen_fn(rho_hat)` / `_gp_fit_one` epilogue —
   which already accept a traced `rho` (that is exactly what `_pooled_nll` passes
   inside the `lax.map`), so the downstream final fit is unchanged.

This alone makes `gp_fit` `jit` / `vmap`-able **whenever the covariate domain is
concrete** — i.e. when `x` is closed-over / static and only the responses `Y`
vary. That is precisely the workflow nimox hits: `eqx.filter_vmap(lambda Y:
gp_fit(Y, x, ...))(Y_stack)` (vmap over datasets / folds / resamples sharing one
covariate grid), and `jax.jit` of a fit with `x` captured. The domain `(lo, hi)`,
basis, and `xtx` are all computed from the concrete `x` at trace time, so only the
(traced, `Y`-dependent) `nll_grid` needs the traceable argmin.

## Notes / scope

- **Why `x`-closed-over is enough.** The blocker nimox actually trips is the
  `np.asarray(nll_grid)` on the `Y`-dependent grid; the domain construction
  (`lo = float(np.min(x))`, `stats/gp.py:829-831`) reads the **concrete** `x` and
  is fine when `x` is not itself a traced argument. So Tier A unblocks the
  high-value cases without touching the result schema.
- **Tier B (optional, more invasive).** Full `jit(gp_fit)` with `x` as a *traced
  argument* additionally needs the domain `lo` / `hi` — today stored as **static**
  `GPResult` aux (Python floats, consumed by `gp_predict` to rebuild the HSGP
  eigenbasis) — to become traced array leaves. That is a result-schema change;
  flag it as a follow-on only if a traced-covariate fit is wanted (nimox does not
  need it now).
- **The other rho searches.** The non-Gaussian PQL path (`stats/gp.py:~1114-1130`)
  and the `exact` / `corr` paths run their own `float()`-heavy host searches; the
  same JAX-native `_parabolic_argmin` + traced-`rho_hat` treatment applies, lower
  priority (Gaussian HSGP is the default and the nimox estimator's primary path).
- **Parity bar.** The refactor must be fp-faithful to the current eager result
  (same `rho_hat` to tolerance on the existing GP fit tests); it is a tracing
  change, not a numerics change. `hgp_fit` shares the lengthscale-search shape and
  would benefit from the same lift (note if it reuses `_parabolic_argmin`).

## Acceptance

- `gp_fit` (Gaussian HSGP) runs under `jax.jit` and `jax.vmap` with `x`
  closed-over — `eqx.filter_vmap(lambda Y: gp_fit(Y, x, ...))(Y_stack)` returns
  per-dataset `GPResult`s, fp-matching a Python-loop of eager fits.
- The existing GP fit/predict tests stay green (no numerical change).
- nimox `GaussianProcessRegressor` gains `vmap`-fit; its eager-only witness test
  flips to a `vmap`-fit substrate proof.

## Cross-references

- nimox `docs/feature-requests/nimox-estimators.md` §11 (E5 GP) / §14 (Tier-1
  rank-2) — the consumer; `src/nimox/estimators/gp.py` documents the constraint
  and `tests/test_gp.py::test_fit_is_eager_only` witnesses it.
- `nitrix.stats.gp.gp_fit` / `_parabolic_argmin` (`stats/gp.py:932-935`, the
  helper just below `gp_predict`) — the site.
- [`gaussian-process-models.md`](gaussian-process-models.md) /
  [`stats-suite-review-gp.md`](stats-suite-review-gp.md) — the GP suite's own
  planning / review register (the HSGP "rho-estimation stays inside the fast
  paths" design intent this completes).
