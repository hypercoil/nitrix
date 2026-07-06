# B19. `erode`/`dilate` flat-SE fast path breaks `jit(grad(...))`

> **Status (2026-06-07): RESOLVED.** Fixed by making the `reduce_window` window
> init a *concrete* scalar sourced from the algebra identity (suggested
> direction 1's goal, reached without a `custom_vjp` — see *Resolution* below).
> The `jit_of_grad` cells for `dilate`/`erode`/`open`/`close` are back to
> `pass` in `docs/op_matrix.json`, gated by new tests. Folded in a uniform
> integer/boolean → `float32` promotion contract on the flat path.
>
> _Original (2026-06-06): open — capability regression report from
> `nitrix-perf-bench`.** Surfaced by regenerating `docs/op_matrix.json` against
> the post-`perf/morphology-micro-wins` tree: the `jit_of_grad` cell for
> `dilate`, `erode`, `open`, and `close` flipped **`pass` → `ValueError`**.
> Authored perf-bench-side (COVERAGE_MANDATE §5); nitrix disposes._

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

## Resolution (2026-06-07)

**Root cause (confirmed):** the fast path passed the window init as
`jnp.asarray(identity, dtype)`. That is concrete in eager mode but becomes a
constant *tracer* under `jit`. JAX's `_get_monoid_window_reducer` only routes a
generic `lax.max` / `lax.min` reducer to the differentiable specialised
primitives (`reduce_window_max_p` / `reduce_window_min_p`) when the init is
`core.is_concrete` *and* equals the dtype's max/min identity. A traced init
fails the concreteness test, so dispatch fell back to the generic
`reduce_window_p` — which has **no transpose rule**, hence the linearization
failure under `jit(grad)` while eager `grad` (which never needs transpose) stayed
green.

**Fix (beats both suggested directions without their cost):** neither a bespoke
`custom_vjp` (direction 1) nor mode-dependent dispatch (direction 2) is needed.
The minimal, principled fix is to *stop defeating JAX's own monoid detection*:

1. Source the identity from the same `Semiring` the non-flat path already uses
   (`TROPICAL_MAX_PLUS.identity = -inf` / `TROPICAL_MIN_PLUS.identity = +inf`,
   concrete Python floats), and pass it through **`np.asarray`** (NumPy — stays
   concrete under trace) rather than `jnp.asarray` (a tracer under `jit`). This
   routes to JAX's *maintained* differentiable pooling primitive — so the "no
   `custom_vjp` for morphology" property of the substrate is preserved.
2. Lift integer / boolean inputs to `float32` at the op boundary (`_to_float`),
   giving a uniform `float-in → float-out` contract across *both* paths. This
   also fixes a latent `-inf → int` overflow (integer input) and a bool-boundary
   bug that the flat path would otherwise hit.

The semiring fallback (direction 2) was rejected as the *primary* fix: it would
forfeit the fast path's win exactly inside training loops (`jit(grad)`), which is
where the speed matters most.

**Landed in `src/nitrix/morphology/_mm.py`:** new `_to_float` helper;
`_windowed_reduce` takes a concrete `identity` and casts via `np.asarray`;
`dilate` / `erode` promote inputs and pass the algebra identity. mypy clean (a
strict improvement — HEAD had a pre-existing `no-any-return`), `ruff check`
clean.

**Gated by** (`tests/test_morphology.py`):

- `test_flat_path_jit_of_grad_is_finite` — `jit(grad)` of `dilate`/`erode`/
  `open`/`close` returns a finite gradient of the right shape (the direct
  regression gate).
- `test_flat_path_matches_semiring_forward_and_grad` — flat path vs explicit
  zero-SE semiring path agree on forward, `grad`, and `jit(grad)`.
- `test_flat_path_promotes_int_bool_to_float` — int/bool promote to floating,
  floating dtypes pass through, output matches scipy `grey_dilation` /
  `grey_erosion`.

`docs/op_matrix.json` / `.md` regenerated: the four `jit_of_grad` cells flip
`ValueError → pass` (MD "jit(grad) pass" summary 116 → 120). Design rationale
folded into [`../design/morphology.md`](../../design/morphology.md) (flat-path
section: the concrete-init requirement + the float-promotion contract).

## Perf-bench follow-through

The hardened `erode`/`dilate` cases (B18 Win 3) will add a **`jit(grad)`
capability row** alongside the disk-SE / border-parity / window-scaling perf
rows, so this regression is gated, not just timed. A perf win on the flat-box
branch must not land while its `jit(grad)` is red.

## Cross-references

- [`perf-bench-case-hardening.md`](../perf-bench-case-hardening.md) (B18, Win 3 —
  the morphology fast-path seams).
- Shipped optimisation: `eba7c43` (ENH morphology: flat-SE erode/dilate via
  fused reduce_window), `3db2ddb` (DOC reduce_window fast path).
- [`internal-backlog.md`](../internal-backlog.md) — ledger index (B19 pointer).
