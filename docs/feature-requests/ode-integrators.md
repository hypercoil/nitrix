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
[`fixed-point-combinators.md`](resolved/fixed-point-combinators.md) (§12.8) +
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

> **Status (2026-06-25): asks #1 + #2 SHIPPED; the `args=` contract resolved;
> adaptive steppers gated.** `nitrix.numerics.ode` now ships
> `euler`/`midpoint`/`rk4`/`odeint` (explicit `midpoint` = RK2 added, exported,
> dispatched). The forward-parity gate is in `tests/test_ode.py` as a
> **round-off-exact** assertion against the hand-unrolled textbook tableau
> (`rtol=1e-12`, ~thousands× tighter than any step error — it pins the
> *arithmetic*, not accuracy), plus a `diffrax`-oracle test behind
> `importorskip` for when diffrax becomes a test dep. **Correction to the
> "bit-exact" framing below:** an XLA-compiled `scan` may contract a `y+dt·f`
> multiply–add into a single FMA, so the result is *not* bit-for-bit identical
> to an eager loop — it agrees to ~1 ULP. Genuine bit-identity is only
> expected between two computations compiled *the same way* (nitrix `odeint`
> vs a diffrax solve, both under XLA); that is the importorskip oracle, and it
> is the honest statement of the BYOW guarantee (round-off-exact, FMA-stable
> across same-compiler runs). The **`args=` mismatch** in the nimox consumer
> doc is resolved in favour of nitrix's existing contract: the integrator
> takes a bare `f(t, y)` and the caller closes over extras
> (`odeint(lambda t, y: field(t, y, args), …)`); there is deliberately no
> `args=` kwarg (closed-over arrays `jit`/`grad` identically). The adaptive
> names `dopri5`/`dopri8`/`tsit5` now **raise a clear `ValueError`** from
> `odeint` (not in the fixed-step set; roadmapped separately). Ask #3
> (checkpointed adjoint) and the Option-2 continuous adjoint remain deferred.
> The nimox `ode-backend-nitrix.md` should drop its `args=args` call form for
> the close-over form, and is otherwise unblocked.

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

**1. `midpoint` solver — complete the CortexODE method set. ✅ SHIPPED.**
CortexODE's config exposes `[euler, midpoint, rk4]`; nitrix now ships all three
(explicit-midpoint / RK2 added as a ~10 LOC tableau under the existing
`odeint(method=…)` dispatcher + exported from `nitrix.numerics`). A brought
CortexODE checkpoint trained/eval'd with `solver='midpoint'` reproduces.

**2. Forward-parity acceptance gate vs diffrax (the BYOW safety net). ✅
SHIPPED.** The load-bearing requirement: a user **bringing pre-trained
CortexODE/SurfNet weights** for inference must get a deformed surface that
reproduces after the backend swap. The gate asserts
`nitrix.numerics.ode.odeint(method='euler'|'midpoint'|'rk4')` matches the
textbook tableau, unrolled by hand, on a representative time-independent
per-vertex field over `[0,1]`, to **floating-point round-off** (`rtol=1e-12`).
⚠️ The original "bit-exact is achievable" claim was over-stated: even Euler
(`xₙ₊₁ = xₙ + h·f(xₙ)`) is *not* bit-for-bit identical to an eager loop, because
XLA may fold `h·f + xₙ` into a single FMA inside the compiled `scan` (a ~1-ULP
difference). Bit-identity is well-defined only between same-compiler
computations — nitrix `odeint` vs a diffrax solve, both under XLA — which is the
`diffrax`-oracle test (`importorskip`, target `atol=1e-12`, active once diffrax
is a nitrix *test* dep). Round-off-exactness against the textbook tableau is the
honest, portable BYOW guarantee and is what the shipped gate enforces.

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
[`fixed-point-combinators.md`](resolved/fixed-point-combinators.md) +
[`krylov-solvers.md`](krylov-solvers.md)) only if a concrete torch-training-parity
need lands; until then the nimox backend raises a loud error for
`adjoint='backsolve'` rather than silently substituting a different gradient.

**Effort (this follow-up):** S (midpoint + the parity gate) — **done
2026-06-25**; the continuous adjoint is the only L item and is deferred.

## Adaptive steppers — `dopri5` / `dopri8` / `tsit5` (separate FR, deferred)

ilex's `SolveConfig` enumerates the diffrax solver catalogue, so the names
`dopri5` / `dopri8` / `tsit5` appear in user configs — but **no concrete
consumer integrates with them**: both upstreams (`CortexODE`, `SurfNet`) run
fixed-step Euler. They are config-surface completeness, not a model
requirement, and the swap is **not** blocked by their absence. Architecturally
they are a different control-flow shape from the shipped fixed-step family: an
embedded-RK error estimate (`dopri5` = Dormand–Prince 5(4), `tsit5` =
Tsitouras 5(4), `dopri8` = 8(7)) + a PI step-size controller driving a
`lax.while_loop` with a **dynamic** step count — not the static `lax.scan` (and
the "one step per `t`-interval, static trajectory shape" contract) the
fixed-step methods rely on. `odeint(method='dopri5'|'dopri8'|'tsit5')` now
**raises a clear `ValueError`** pointing at this gap rather than silently
falling back. Build only when a model actually needs adaptivity, as its own FR;
suggested order `tsit5` (diffrax's own default, best for non-stiff) → `dopri5`
(the canonical default) → `dopri8`. Effort **L**.

## Cross-references

- `docs/feature-requests catalogue §12.11` — origin entry; `§13` — acceptance protocol.
- [`ode-backend-nitrix.md`](../../../nimox/docs/feature-requests/ode-backend-nitrix.md)
  (nimox) — the consumer-side rewire of `integrate_vertex_flow` onto this.
- [`fixed-point-combinators.md`](resolved/fixed-point-combinators.md) and
  [`krylov-solvers.md`](krylov-solvers.md) — adjoint-pass dependencies.
- `src/nitrix/geometry/grid.py` — `integrate_velocity_field`, the special
  case this generalises.
