# `glmm_fit` is not `jax.jit`-traceable — data-dependent `int(jnp.max(group))` — `nitrix.stats.glmm`

> **Status (2026-06-20): ✅ SHIPPED** (branch `feat/stats-audit-round2`, audit
> item **P7**). `glmm_fit` now takes an optional static `n_groups: Optional[int]
> = None`; supplied, it traces under `jax.jit` for **all** families × methods ×
> intercept/slope (acceptance test
> `test_glmm_fit_jit_traceable_with_static_n_groups`, jitted == eager), and a
> negative test pins that omitting it still raises `ConcretizationTypeError`.
>
> **Note — the minimal fix was necessary but not sufficient.** The `n_groups`
> concretization short-circuited first, masking a *second* blocker on the
> **few-level** (`gam_fit` / `re_smooth`) path: `re_smooth` built its identity
> ridge as `jnp.eye(q)`, a *tracer* under `jit`, and `gam.py`'s
> `_smooth_penalties` `np.asarray`'s every penalty at trace time
> (`TracerArrayConversionError`). Fixed by building the RE penalty as a host
> constant `np.eye(q)` — consistent with the spline difference-penalties, which
> are already host constants (the penalty depends only on the static level count,
> not on group *values*). The many / slope / laplace / agq paths needed only the
> `n_groups` change. *Original FR (now implemented) follows.*

**What.** `glmm_fit` cannot be traced inside `jax.jit`. The very first thing it
does after validation is derive the level count as a **concrete Python int**
from the data:

```python
# glmm.py  (main:1486 / docs/stats-suite-audit HEAD:1500), in glmm_fit(...)
n_groups = int(jnp.max(group)) + 1
```

`n_groups` is then used as a **static** value to set output shapes throughout —
`re_smooth(group, n_levels=n_groups)`, `segment_sum(..., num_segments=n_groups)`,
`jnp.zeros((n_groups, r))`, and as the `n_groups` argument threaded into every
tier dispatcher (`_glmm_laplace_slope`, `_glmm_agq_slope`,
`_glmm_slope_structured_one`, the few/many intercept paths). Under `jax.jit`,
`group` is a tracer (even when the caller passes a *concrete* array — a captured
constant still traces through `jnp.max`), so `int(jnp.max(group))` raises:

```
jax.errors.ConcretizationTypeError: Abstract tracer value encountered where a
concrete value is expected: traced array with shape int32[]
  operation a:i32[] = reduce_max[axes=(0,)] b
    from .../nitrix/stats/glmm.py:1486 (glmm_fit)
```

**Minimal repro** (pin `a5b7e80`; reproduces identically on `main` and HEAD):

```python
import jax, jax.numpy as jnp, numpy as np
from nitrix.stats.glmm import glmm_fit

rng = np.random.default_rng(0)
N, q, V = 120, 6, 4
group = jnp.asarray(np.repeat(np.arange(q), N // q))
X = jnp.asarray(np.c_[np.ones(N), rng.standard_normal(N)])
Y = jnp.asarray(rng.standard_normal((V, N)))
call = lambda y: glmm_fit(y, X, group=group, family='gaussian').beta_hat

call(Y)            # EAGER: OK -> beta_hat (4, 2)
jax.jit(call)(Y)   # JIT:   ConcretizationTypeError at n_groups = int(jnp.max(group))
```

**Why it matters.**

1. **Consumers cannot fuse a pipeline containing `glmm_fit`.** The natural
   brain-scale usage is one jitted, fully-fused batched fit over all voxels; that
   is impossible today. The op runs *only* eagerly (op-by-op dispatch of the
   outer scaffolding, with just the inner `lax.scan` / `lax.fori_loop` kernels
   compiled).
2. **`glmm_fit` is the only marquee op perf-bench cannot benchmark.** The runner
   measures every `jax` baseline as `jax.jit(fn)` (the canonical "fully-fused XLA
   program" measurement). All `glmm_fit` rows come back `compile_error`, so the
   flagship has **no steady-state / scaling data** — the small-fast-scale
   extrapolation methodology (theory exponent × empirical power-law fit →
   brain-scale projection) has nothing to fit. A measurement-side eager
   workaround was rejected on purpose: eager op-dispatch overhead dominates the
   small scales and would corrupt the power-law fit (inflate the small-`n`
   constant), defeating the extrapolation.
3. **Parity regression vs the LME sibling.** `lme_fit` / `reml_fit` are
   jit-safe because the caller supplies the level structure (`q`, `n_per`)
   explicitly. `glmm_fit` regressed that contract by deriving the level count
   internally from the data.

**Proposed fix (minimal, back-compatible).** Add an optional **static** level
count to the signature:

```python
def glmm_fit(Y, X, *, group, z=None, ..., n_groups: Optional[int] = None, ...):
    ...
    if n_groups is None:
        n_groups = int(jnp.max(group)) + 1   # eager path, unchanged
    # else: use the caller-supplied static int  -> fully jit-traceable
```

`n_groups` is *already* a static `int` threaded through every internal helper, so
this is literally "expose the existing internal static value as an optional
parameter." When omitted, behaviour is byte-identical to today (eager). When
supplied (a Python `int`, the level count the caller already knows), `glmm_fit`
traces cleanly under `jax.jit` for **all** families / structures / methods. This
restores parity with `lme_fit`/`reml_fit` and unblocks both the consumer
fuse-the-whole-fit use case and the perf-bench flagship.

**Composition / blast radius.** Signature-only additive change — a new optional
keyword with a default that preserves current behaviour. No `GLMMResult` change,
no algorithm change, no change to any tier's math. **Drift-safe** for perf-bench:
an additive optional kwarg is backward-compatible (the drift gate compares the
behaviour digest + signature; an added default-valued kwarg does not change
existing-call behaviour). A second, more defensive guard could also validate
`group < n_groups` only outside `jit` (when `n_groups is None`).

**Acceptance test.** `jax.jit(lambda y: glmm_fit(y, X, group=group,
n_groups=q, family=fam, method=m).beta_hat)(Y)` traces and runs for `fam ∈
{gaussian, binomial, poisson}` × `m ∈ {pql, laplace, agq}` × intercept/slope,
and the jitted result matches the eager result to tolerance. The perf-bench
GLMM case (`glmm_fit`) then measures (all paths leave `compile_error`).

**Related.** [`glmm-random-slope-robust-solver.md`](glmm-random-slope-robust-solver.md)
(shipped; same op, solver core) and the v3 GLMM ledger
[`stats-modelling-suite-v3.md`](../stats-modelling-suite-v3.md). Index this under
the **Statistical modelling suite** family. Perf-bench finding; no consumer
divergence beyond the inability to jit.
