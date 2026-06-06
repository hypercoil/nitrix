# B19. `erode`/`dilate` flat-SE fast path breaks `jit(grad(...))`

> **Status (2026-06-06): open — capability regression report from
> `nitrix-perf-bench`.** Surfaced by regenerating `docs/op_matrix.json` against
> the post-`perf/morphology-micro-wins` tree: the `jit_of_grad` cell for
> `dilate`, `erode`, `open`, and `close` flipped **`pass` → `ValueError`**.
> Authored perf-bench-side (COVERAGE_MANDATE §5); nitrix disposes.

## TL;DR

The flat-box `lax.reduce_window` fast path shipped in `eba7c43` (`erode`/`dilate`
via fused sliding-window min/max) is **not safe under `jit(grad(...))`**. Eager
`grad` works; `jit(grad)` fails at trace time with:

```
ValueError: Linearization failed to produce known values for all output
primals. This is typically caused by attempting to differentiate [through a
primitive whose JVP/transpose is not implemented for these abstract values].
```

This bites the **default** path (`structuring_element=None`), i.e. every caller
who takes the fast path. It is the cross-cutting cost the perf-bench
case-hardening report (B18, Win 3) warned about in the abstract, now observed:
the fast path improves the benched metric (steady-state speed on the easy
branch) while silently dropping a capability the op-matrix advertises.

## Controlled attribution (warranted)

Same environment, same nitrix HEAD, same JAX 0.10.0 — only the dispatch path
differs (default flat box → `reduce_window`; explicit flat SE → old semiring
`TROPICAL_MAX_PLUS` path, per B18 Win 3):

| path | `grad` | `jit(grad)` |
|---|---|---|
| `dilate(x, size=3)` — default (`reduce_window`) | pass | **ValueError (linearization failed)** |
| `dilate(x, structuring_element=ones((3,3)))` — semiring | pass | pass |

Because the semiring path (the pre-`eba7c43` behaviour) still differentiates
under `jit`, the failure is attributable to the `reduce_window` lowering, not to
the JAX version or the autodiff harness. The error is raised during
trace/linearization (before lowering), so it is **platform-independent** (it
will fail identically on GPU; the matrix was generated on CPU). `erode` shares
the path; `open`/`close` compose `erode`+`dilate`, so they inherit the failure.

Repro:

```python
import jax, jax.numpy as jnp
from nitrix.morphology import dilate
x = jax.random.normal(jax.random.key(0), (10, 10))
jax.grad(lambda z: jnp.sum(dilate(z, size=3) ** 2))(x)            # ok
jax.jit(jax.grad(lambda z: jnp.sum(dilate(z, size=3) ** 2)))(x)   # ValueError
```

## Why it matters

`jit(grad(...))` is the double-transform training loops actually run (an inner
`grad` compiled by an outer `jit`). Plain eager `grad` passing is not a
substitute: a morphology op inside any `@jax.jit` training step is on the broken
path. The op-matrix `grad` column being green is misleading for these four ops
unless read alongside `jit_of_grad`.

## Suggested directions (nitrix disposes)

1. **Custom VJP for the `reduce_window` path** that is linearization-clean under
   `jit` — likely the real fix; the semiring path already proves a working
   gradient exists for the same op.
2. **Route through the semiring path when a gradient is being traced** (e.g.
   detect the transform / expose a `differentiable=` escape hatch), keeping the
   fast path for forward-only inference where it wins.
3. **At minimum, document the limitation** on `erode`/`dilate`/`open`/`close`
   (fast path is forward + eager-grad only; use an explicit `structuring_element`
   for `jit(grad)`) so the contract is honest until (1) lands — mirroring the
   ships-with-a-case discipline.

## Perf-bench follow-through

The hardened `erode`/`dilate` cases (B18 Win 3) will add a **`jit(grad)`
capability row** alongside the disk-SE / border-parity / window-scaling perf
rows, so this regression is gated, not just timed. A perf win on the flat-box
branch must not land while its `jit(grad)` is red.

## Cross-references

- [`perf-bench-case-hardening.md`](perf-bench-case-hardening.md) (B18, Win 3 —
  the morphology fast-path seams).
- Shipped optimisation: `eba7c43` (ENH morphology: flat-SE erode/dilate via
  fused reduce_window), `3db2ddb` (DOC reduce_window fast path).
- [`internal-backlog.md`](internal-backlog.md) — ledger index (B19 pointer).
