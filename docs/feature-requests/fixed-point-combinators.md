# Fixed-point combinators — `nitrix.numerics.fixed_point`

> **Status (2026-06-08): SHIPPED** — `numerics.fixed_point_solve` (Picard
> iteration with an implicit-VJP backward, the IFT adjoint solved as its own
> fixed point; O(1) memory in the iteration count). Graduated by the
> registration suite — backs `geometry.invert_displacement`
> (`s_inv = -s∘(id + s_inv)`) — per the §13 protocol; see
> `docs/design/registration.md` and `IMPLEMENTATION_PLAN.md §10.3`.
> Provenance: `docs/feature-requests catalogue §12.8`.

**What.** A deep-equilibrium-style fixed-point solver with an implicit-VJP
backward.

**Proposed surface.**

```python
def fixed_point_solve(f, x0, *, tol, max_iter): ...
```

Returns the fixed point of `f` plus an implicit-VJP — the Jacobian of `f`
at convergence, solved via the Krylov solver
([`krylov-solvers.md`](krylov-solvers.md), §12.1).

**Composition.** Generalises the `scaling_and_squaring` pattern in
`geometry.integrate_velocity_field`. The backward is the standard
implicit-function-theorem solve, reusing the §12.1 inner linear solve — the
same implicit-VJP machinery already validated for LOBPCG.

**Likely consumer.** Deep-equilibrium models for surface registration,
iterative ICA fixed-point solvers, implicit filters that solve a per-sample
optimisation. Underpins the convergence-point gradients in
[`graphical-lasso.md`](graphical-lasso.md) and
[`clustering-primitives.md`](clustering-primitives.md).

**Effort.** M. Depends on §12.1.

**Live-code status.** No `nitrix.numerics.fixed_point` /
`fixed_point_solve`. `geometry.integrate_velocity_field`
(scaling-and-squaring) is the one shipped fixed-point-shaped primitive; the
general combinator + implicit-VJP is not extracted.

## Cross-references

- `docs/feature-requests catalogue §12.8` — origin entry; `§13` — acceptance protocol.
- [`krylov-solvers.md`](krylov-solvers.md) — the inner-solve dependency.
- `src/nitrix/geometry/grid.py` — `integrate_velocity_field`, the special
  case this generalises.
