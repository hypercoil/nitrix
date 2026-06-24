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

## ilex axis-iv backend-swap follow-up (2026-06-24)

> **Context.** ilex axis-iv (training engine) examined the two diffrax consumers
> (`cortex_ode`, `surfnet`) against their upstreams (`m-qiang/CortexODE`,
> `MLDataAnalytics/SurfNet`). Both integrate `dx/dt = f(t, x, args)` with a
> **fixed-step Euler** forward (CortexODE config default `solver='euler'`,
> options `midpoint`/`rk4`; SurfNet `euler`), `t ∈ [0, 1]`, step `0.1–0.2`. The
> shipped `nitrix.numerics.ode.{euler, rk4, odeint}` already covers the forward
> + the **default differentiable backward** (straight-through the `lax.scan`,
> which is exactly diffrax's default `RecursiveCheckpointAdjoint` semantics).
> This is the green light to retire diffrax (it is now an optional `surf`/`ode`
> extra in ilex/nimox as a stopgap). Three small asks complete the swap:

**1. `midpoint` solver — complete the CortexODE method set.** CortexODE's config
exposes `[euler, midpoint, rk4]`; nitrix ships `euler`/`rk4` but not `midpoint`
(explicit-midpoint / RK2). ~10 LOC tableau under the existing `odeint(method=…)`
dispatcher. Needed so a brought CortexODE checkpoint trained/eval'd with
`solver='midpoint'` reproduces.

**2. Forward-parity acceptance gate vs diffrax (the BYOW safety net).** The
load-bearing requirement: a user **bringing pre-trained CortexODE/SurfNet
weights** for inference must get a **bit-identical** deformed surface after the
backend swap. So add an acceptance test asserting
`nitrix.numerics.ode.odeint(method='euler'|'midpoint'|'rk4')` matches
`diffrax.diffeqsolve(Euler()|Midpoint()|…)` **to fp tolerance (target exact)** on
a representative time-independent per-vertex field over `[0,1]`. Euler is
`xₙ₊₁ = xₙ + h·f(xₙ)` — the same arithmetic, so bit-exact is achievable; this
gate is what makes the swap safe for brought weights. (diffrax may live in the
nitrix *test* deps only, as the reference oracle.)

**3. Roadmap — memory-efficient checkpointed adjoint.** The shipped
straight-through backprop stores per-step activations; fine for the 5–20-step
surface models, but the roadmapped checkpointed backward (recompute `f` in the
backward — the `RecursiveCheckpointAdjoint` memory profile) is the optimisation
for deep/high-vertex integration. Not blocking the swap.

**Deferred extension (Option 2) — the continuous adjoint.** Both upstreams
*train* with torchdiffeq `odeint_adjoint` (the O(1)-memory continuous adjoint,
Chen et al. 2018). This is **explicitly deferred**, not required for the swap:
inference is forward-only (gate #2 covers it), and default fine-tuning uses the
straight-through backprop, which is the **exact** gradient of the discretised
forward (arguably more correct than the continuous adjoint, which has its own
integration error and converges to straight-through as `h→0`). The continuous
adjoint is needed *only* to bit-reproduce the upstream's adjoint *training run*
— a niche, already-framework-fragile reproducibility goal. Implement it (the
augmented adjoint ODE + `jax.custom_vjp`, composing
[`fixed-point-combinators.md`](fixed-point-combinators.md) +
[`krylov-solvers.md`](krylov-solvers.md)) only if a concrete torch-training-parity
need lands; until then the nimox backend raises a loud error for
`adjoint='backsolve'` rather than silently substituting a different gradient.

**Effort (this follow-up):** S (midpoint + the parity gate); the continuous
adjoint is the only L item and is deferred.

## Cross-references

- `docs/feature-requests catalogue §12.11` — origin entry; `§13` — acceptance protocol.
- [`ode-backend-nitrix.md`](../../../nimox/docs/feature-requests/ode-backend-nitrix.md)
  (nimox) — the consumer-side rewire of `integrate_vertex_flow` onto this.
- [`fixed-point-combinators.md`](fixed-point-combinators.md) and
  [`krylov-solvers.md`](krylov-solvers.md) — adjoint-pass dependencies.
- `src/nitrix/geometry/grid.py` — `integrate_velocity_field`, the special
  case this generalises.
