# ODE integrators — `nitrix.numerics.ode`

> **Status (2026-06-09): fixed-step family SHIPPED.** `numerics/ode.py` adds
> `euler`, `rk4`, and an `odeint(f, y0, t, *, method)` dispatcher for
> `dy/dt = f(t, y)` over a `lax.scan` (one step per time interval), returning
> the state at each time point and **differentiable straight through the
> solver**. Pure JAX — the portable, diffrax-free substrate the per-vertex
> neural-ODE models (`cortex_ode` / `surfnet`) need. Verified vs the
> exp-decay closed form, RK4≪Euler accuracy, energy conservation, and grad
> correctness. Still roadmap: adaptive (Dormand–Prince), symplectic
> (leapfrog / implicit-midpoint), and the memory-efficient adjoint backward.
> Provenance: `docs/feature-requests catalogue §12.11`.

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

**Concrete consumer (2026-06-08 ilex audit).** `cortex_ode` and `surfnet`
are **per-vertex neural-ODE** surface models: they integrate
`dx/dt = f(t, x, args)` over a vertex set. Today they use `diffrax`
(`ilex/nimox/modules/ode.py:107` `integrate_vertex_flow`) — which **cannot
follow into nitrix** (off the allowlist; SPEC §6.2). This is the demand
signal for a pure-`jax.lax.scan` fixed-step `rk4`/`euler` (and adaptive
`dormand_prince`) under `nitrix.numerics.ode`: the nimox `SurfaceNeuralODE`
module would then call the nitrix integrator instead of diffrax. Note this is
distinct from `integrate_velocity_field` (a *stationary voxel-grid* SVF via
scaling-and-squaring); the surface NODE is a *general per-vertex* flow.

**Effort.** L.

**Live-code status.** `geometry.integrate_velocity_field` (the
scaling-and-squaring integrator for stationary velocity fields) is shipped;
no general `rk4` / `dormand_prince` / `leapfrog` / `implicit_midpoint` and
no `nitrix.numerics.ode` namespace.

## Cross-references

- `docs/feature-requests catalogue §12.11` — origin entry; `§13` — acceptance protocol.
- [`fixed-point-combinators.md`](fixed-point-combinators.md) and
  [`krylov-solvers.md`](krylov-solvers.md) — adjoint-pass dependencies.
- `src/nitrix/geometry/grid.py` — `integrate_velocity_field`, the special
  case this generalises.
