# Make `gp_fit`'s lengthscale-search epilogue traceable (jit / vmap the GP fit)

> **Status (2026-07-06): SHIPPED (Tier-A + the `hgp_fit` follow-on).** Both the
> `gp_fit` and the hierarchical `hgp_fit` Gaussian-HSGP rho-searches are traceable
> and tested (`test_gp_traceable.py` / `test_hgp_traceable.py`); the Tier-B
> traced-`x` / non-Gaussian-exact paths stay deferred (lower priority). The
> Gaussian HSGP rho-search
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

## Follow-on (2026-06-25): `hgp_fit` — the same fix, and the helper already exists

> **Status (2026-06-26): SHIPPED.** `hgp_fit` (Gaussian HSGP) now imports
> `_parabolic_argmin_jax`, keeps `rho_hat = jnp.exp(log_rho_hat)` traced, and
> uses `jnp.log` for the `theta` log-rho column — so the shared Gaussian-HSGP
> path runs under `jax.jit` / `jax.vmap` with the covariate **and grouping**
> closed over. One adjacent host concretisation also had to move to NumPy: the
> label-range **validation** (`int(jnp.min(group))`, hgp.py:391, plus the nested
> `group_inner` twin) was accidentally device-side — its own comment says
> "host-side", and gp_fit's analogous domain validation is already NumPy — so it
> now reads the concrete closed-over `group` via `np.asarray` (static, like the
> covariate domain). **No numerics change** — the eager result is **byte-identical**
> to baseline (coef/theta/log_mlik max |Δ| = 0.0). `vmap(lambda Y: hgp_fit(Y, x,
> group, ...))(Y_stack)` fp-matches a loop of eager fits
> (`tests/test_hgp_traceable.py`, jit + vmap); the 17 existing `hgp_fit` tests
> stay green. nimox `HierarchicalGPRegressor` (E8) can flip its eager-only
> witness to a vmap-fit proof. (The non-Gaussian PQL / exact host searches in
> `gp_fit` remain eager — same deferred tail as the original Tier-A.)

The **hierarchical** GP `hgp_fit` (the GS factor-smooth / multi-site normative
model) has the **identical** Gaussian-HSGP rho-search and is **still eager-only**:

```python
# stats/hgp.py:71
from .gp import _parabolic_argmin                       # <-- the OLD host helper

# stats/hgp.py:515-518
log_rho_grid_j = jnp.asarray(log_rho_grid, dtype=Y.dtype)
nll_grid = jax.lax.map(_pooled_nll, log_rho_grid_j)     # (n_rho,)  PURE JAX already
log_rho_hat = _parabolic_argmin(log_rho_grid, np.asarray(nll_grid))  # <-- host
rho_hat = float(np.exp(log_rho_hat))                    # <-- host: concretise
# stats/hgp.py:539
log_rho_col = jnp.full_like(sigma_e2, np.log(rho_hat))  # <-- np.log on a float
```

This is **lower-effort than the original Tier-A** because the JAX-native
`_parabolic_argmin_jax` you shipped for `gp_fit` already exists — `hgp_fit` just
needs to import/use it instead of the host `_parabolic_argmin`, keep
`rho_hat = jnp.exp(log_rho_hat)` traced (drop the `float()`), and use `jnp.log`
at line 539. The downstream `_blocks(rho_hat)` already accepts a traced `rho`
(that is exactly what `_pooled_nll` passes inside the `lax.map` at hgp.py:501),
and the domain `(lo, hi)` / `group` factor are read from the **concrete**
covariate / grouping (lines 427-446), so — as with `gp_fit` — this unblocks
`jit` / `vmap` whenever `x` and `group` are closed over (the
vmap-over-datasets-sharing-a-covariate-and-grouping case).

**Concrete consumer.** nimox's `HierarchicalGPRegressor` (E8) ships today on the
eager `hgp_fit` with an eager-only witness test; lifting this flips it to a
vmap-fit proof (mirroring what the `gp_fit` fix did for
`GaussianProcessRegressor`). This is a sharper ask than the original passing
mention ("`hgp_fit` shares the lengthscale-search shape"): it is the **same
Gaussian-HSGP path** (not one of the deferred non-Gaussian PQL / exact host
searches), with the helper already written — so it is the natural next
increment, not the lower-priority tail. fp-parity bar as before (a tracing
change, not a numerics change); the existing `hgp_fit` tests must stay green.

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
- [`gaussian-process-models.md`](../gaussian-process-models.md) /
  [`stats-suite-review-gp.md`](../stats-suite-review-gp.md) — the GP suite's own
  planning / review register (the HSGP "rho-estimation stays inside the fast
  paths" design intent this completes).
