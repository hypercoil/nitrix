# ODE integrators — `nitrix.numerics.ode`

> **Status (2026-06-02): partial — one integrator (`integrate_velocity_field`,
> scaling-and-squaring) is shipped; the general RK / symplectic family is
> not.** Brainstorm candidate; promotion gated by the §13 acceptance
> protocol. Provenance: `SPEC_UPDATE_v0.3.md §12.11`.

**What.** General-purpose ODE integration, differentiable via the adjoint
equation.

**Proposed surface.**

```python
# Explicit Runge–Kutta
def rk4(f, y0, t): ...
def dormand_prince(f, y0, t, *, rtol, atol): ...   # adaptive

# Symplectic (Hamiltonian flows)
def leapfrog(f, y0, t): ...
def implicit_midpoint(f, y0, t): ...
```

**Composition.** Generalises `geometry.integrate_velocity_field`
(scaling-and-squaring). Differentiable via the adjoint-equation pattern
(Chen et al. 2018), which is itself a fixed-point + Krylov solve — composes
[`fixed-point-combinators.md`](fixed-point-combinators.md) (§12.8) +
[`krylov-solvers.md`](krylov-solvers.md) (§12.1).

**Likely consumer.** Neural-ODE-style continuous-time models,
advection-diffusion forward models for deformable registration,
Hamiltonian-Monte-Carlo posterior samplers for fMRI.

**Effort.** L.

**Live-code status.** `geometry.integrate_velocity_field` (the
scaling-and-squaring integrator for stationary velocity fields) is shipped;
no general `rk4` / `dormand_prince` / `leapfrog` / `implicit_midpoint` and
no `nitrix.numerics.ode` namespace.

## Cross-references

- `SPEC_UPDATE_v0.3.md §12.11` — origin entry; `§13` — acceptance protocol.
- [`fixed-point-combinators.md`](fixed-point-combinators.md) and
  [`krylov-solvers.md`](krylov-solvers.md) — adjoint-pass dependencies.
- `src/nitrix/geometry/grid.py` — `integrate_velocity_field`, the special
  case this generalises.
