# Dynamical-systems / differential-equation suite — `nitrix.numerics`

> **Status (2026-06-25): forward-looking scoping ledger.** Frames a primitive
> family — the differentiable **differential-equation integration substrate** —
> credits the shipped substrate (the fixed-step ODE family + the linear-algebra,
> noise, and signal plumbing the rest composes on), indexes the already-filed
> items that belong to it, and specs the **genuinely-new numerical gaps** that
> would unlock a whole subdiscipline of neuroimaging dynamics: dynamic causal
> modelling (DCM), neural→haemodynamic forward models, The-Virtual-Brain-style
> whole-brain network simulation, and brain digital twins. No item here
> duplicates an existing FR — those are referenced, not re-opened (this index is
> the duplicate-issue guard, per `README.md`). **Numerics-only scope**: the
> *integrators and the plumbing* are nitrix; every *model* (Balloon-Windkessel,
> Jansen–Rit, Wong–Wang, Hopf, Epileptor, …) and every *inversion framework*
> (variational Laplace, NUTS, SBI) is **out of scope → downstream** (nimox
> modules / an ilex dynamics suite). Driver: ilex/nimox standing up a
> differentiable, GPU-parallel dynamical-systems modelling stack on a JAX
> substrate, the way `vbjax` and differentiable-DCM efforts re-base SPM/TVB.

## 0. Governance & locked decisions

> Unlike the geometry / stats / registration suites — each of which graduated
> against a **concrete, already-blocked** ilex/nimox consumer — most of this
> suite is **prospective**. The fixed-step ODE family has real consumers today
> (`cortex_ode` / `surfnet`, see [`ode-integrators.md`](ode-integrators.md)); the
> *new* DE families below do **not yet** have a named blocked consumer. This doc
> is therefore the **design-of-record that pre-stages the substrate** so that
> when a consumer files (an ilex DCM port, a nimox TVB / digital-twin module),
> the build path, the API contracts, and the hard design decisions are already
> settled. Per the SPEC §9 graduation gate, **no item here graduates to sprint
> scope until its named consumer is blocked/workaround-laden without it.** This
> is a roadmap, not a build list.

**Locked decisions (2026-06-25):**

- **D1 — Separation of concerns is the load-bearing boundary.** nitrix ships the
  **integrators + the numerical plumbing** (the noise contract, the resolvent /
  matrix-exponential solves, the post-simulation analysis primitives). It ships
  **no models**: the neural-mass equations, the connectome coupling operator, the
  haemodynamic / lead-field observation operators, and the estimation frameworks
  all live downstream (SPEC §1: "no … training loops, no PyTree modules … no
  models"; SPEC §4: no new top-level model subpackage). This mirrors the existing
  split exactly — `integrate_velocity_field` is the nitrix primitive, the
  registration *recipes* are consumers; `ode.odeint` is the primitive,
  `cortex_ode`/`surfnet` are consumers.

- **D2 — The noise contract is *close-over-the-key*** — the SDE analog of the
  `args` contract already settled for the ODE family. Brownian increments
  `dW ~ √dt·𝒩(0,1)` are a **deterministic** function of `(key, step_index)` via a
  folded-in `jax.random` key; the integrator takes a bare `key` and the caller
  closes over any noise parameters. **No stateful RNG, no new `nitrix.random`
  package** — reuse `jax.random` with the keyed-only convention already
  established in `augment`. Determinism in the key is what makes the path
  reproducible *and* what makes pathwise gradients correct.

- **D3 — SDE differentiation is discretise-then-optimize** (straight-through the
  fixed-step scan with the **Brownian path held fixed** via D2's reparameterised
  increments), exactly mirroring the ODE straight-through decision. The
  continuous **stochastic** adjoint (Li et al. 2020, the Brownian-tree /
  Lévy-area machinery) is the niche deferred analog of the deferred ODE
  continuous adjoint — **fail-loud**, built only on a concrete torch-training-
  parity need.

- **D4 — Adaptive stepping stays deferred, and the stochastic case *reinforces*
  it.** Naive step-size adaptation breaks Brownian-path consistency; correct
  adaptive SDE integration needs a Virtual Brownian Tree (Lévy-area)
  construction. The field standard for neural-mass/whole-brain models is
  **fixed-step stochastic Heun** precisely for this reason. So the existing
  adaptive-ODE deferral ([`ode-integrators.md`](ode-integrators.md)) is the right
  call across *both* ODE and SDE for this domain; adaptive embedded-RK is **not**
  on this suite's critical path.

- **D5 — Integrators home in `nitrix.numerics`; no new top-level subpackage.**
  Exponential / local-linearization / symplectic steppers extend `numerics.ode`
  (they integrate ODEs); the stochastic and delay families get `numerics.sde` /
  `numerics.dde` submodules (the `numerics.ode` idiom), **not** a top-level
  `dynamics` package — SPEC §4 forbids a model package and the substrate-
  composition story keeps these under `numerics`.

- **D6 — The thesis is differentiability + `vmap`.** The reason to re-base
  DCM/TVB on this substrate rather than reuse SPM/TVB is structural and not
  available upstream: (1) the simulator becomes **differentiable end-to-end** —
  gradient-based personalisation, variational inversion, and hybrid neural-
  physical training fall out of `jax.grad` through the scan; (2) **GPU + `vmap`**
  over parameter sweeps and cohorts. Every primitive below preserves both.

**RFC-1 (2026-06-25) — where the *inversion* boundary actually falls (tentative;
not a locked decision).**

> Unlike D1–D6, this is an **open proposal recorded to be argued with**, not a
> settled rule — it will sharpen, and may move, when the first real DCM / TVB
> consumer lands. D1 puts "models + inversion frameworks" downstream, but
> *"has an inversion / a solver"* cannot by itself be the test: `glm_fit`,
> `lme_fit`, `n4`, and the registration recipes all bundle a solver **and** a
> model fit, and all correctly live **in** nitrix. RFC-1 proposes the finer line.

- **The discriminator — closed parametric form vs. open model program.** nitrix
  hosts inference for model families whose **form it can close over**: `Xβ + link`
  (GLM), the mixed-model form (LME), the B-spline log-bias field (N4),
  `transform-group + cost-from-a-menu` (register). The caller supplies *arrays*;
  the form is finite and universal, and there is one canonical computation for
  everyone. DCM has **no single form to own** — the generative model is a
  user-authored coupled ODE and the **model *space* is the scientific result**
  (which edges exist, compared by evidence). A discipline of *authoring-and-
  comparing* forms is downstream; a fixed form nitrix can own is nitrix-eligible.

- **Corollary test — self-contained vs. framework-relative output.** GLM
  coefficients, a bias field, and a transform are complete on their own. DCM's
  posterior `(μ, Σ)` and free energy `F` are only interpretable inside model
  comparison (`F` is comparative) and hierarchy (`Σ` feeds PEB). Output that is
  meaningful only within a surrounding framework belongs *with* that framework.

- **Proposed layering — the DRY answer.** The fit decomposes by *layer*, each
  living in exactly **one** place, so no consumer reimplements the core scaffold:
  - **A — kernels** (integrate; Gauss–Newton / LM + the local-linearization
    ascent step; ReML precision update; `logdet`; the spectral resolvent solve):
    **nitrix, already shipped** (`linalg.optimize`, `linalg.matrix_exp`,
    `numerics.ode.local_linearization`, `linalg._smalllinalg`, `stats.lme.reml`).
  - **B — the form-agnostic inversion loop**: variational-Laplace over a
    *supplied* `g(θ)` + Gaussian prior → `(μ, Σ, F)`, baking in **no** model
    structure (model, priors, precision components are all inputs). **Tentatively
    nitrix** — the Bayesian sibling of the already-resident `reml_fit` /
    `gauss_newton`, e.g. `nitrix.stats.variational_laplace`. This is the single
    shared "numerical core scaffold"; it must live in **one** place, whichever
    side of the line it ultimately lands.
  - **C — the DCM-ness**: the specific generative model + connection-matrix priors
    + BMR / BMS / PEB / BMA + the posterior/evidence ensemble — **downstream
    (nimox)**.
  - **D — interface adapters**: a `thrux` graph node / an `entense` differentiable
    step / a nimox `fit · predict · score` estimator — **per-library, and *meant*
    to differ**; that divergence is the reason those libraries exist.

- **Consequence for duplication — none.** Dependency DAG
  `nitrix ← nimox ← {thrux, entense}`: layers A/B are shared from nitrix, C lives
  once in nimox, and thrux/entense **wrap** the nimox estimator rather than
  re-deriving VL. The verbatim-duplication failure mode arises only if Layer B
  lives *nowhere* (each consumer assembling it inline) — so the fix is to give the
  generic loop a single home, **not** to push the DCM recipe (Layer C) into nitrix.

- **Open / expected to sharpen.** (1) Whether Layer B is nitrix or nimox — turns
  on whether "variational free energy" reads as a *numerical* objective (like
  `½‖r‖²`) or a *modeling* commitment. (2) Whether the generic VL loop stays
  genuinely form-agnostic once spectral DCM's **complex non-symmetric resolvent**
  `(iωI − J)⁻¹` and stochastic DCM's **generalized-coordinate filtering** are in
  view, or whether those fork it. (3) Whether `gauss_newton` should grow a full
  Gaussian-prior form (prior precision matrix + mean), or that stays a Layer-B
  concern. **Revisit when the first consumer files** — do not treat RFC-1 as
  authoritative until then.

## 1. Why this doc exists

The neuroimaging dynamical-systems landscape — DCM, neural→haemodynamic forward
modelling, TVB-style whole-brain network simulation, brain digital twins — maps
almost perfectly onto a **taxonomy of differential-equation classes**, and the
capability boundary falls on that taxonomy: deterministic capabilities need ODE
/ exponential integrators (mostly shipped or assembly), the stochastic
network-model family needs the **SDE** family (the single biggest unlock), and
delay-coupled whole-brain models additionally need **DDE** support. nitrix's job
is the bottom layer of all of it.

A triage of the landscape against the live surface shows the gap is **specific
and smaller than it looks**: the fixed-step ODE family is shipped; the
exponential / local-linearization integrator (the classic SPM `spm_int` scheme)
is **mostly assembly** over the already-shipped `matrix_exp` + `cg`; linear
network-diffusion spreading is **nearly free** (`graph` Laplacian + `matrix_exp`
/ heat-kernel); and the post-simulation analysis layer (Kuramoto order
parameter, phase-FC, metastability) is **already shipped** in `signal`
(`analytic_signal` / `instantaneous_phase`). What is genuinely missing is the
**stochastic integrator family** (and, for full TVB parity, delays). This doc is
the ledger that (a) records what is shipped, (b) pulls the scattered DE-relevant
FRs under one roof, and (c) names the handful of integrator primitives still
missing, each mapped to the modelling capability that needs it.

## 2. Scope boundary

**In scope — numerical primitives only**, consistent with the SPEC §6 dependency
contract (`numpy` + `jax`; no `nibabel`, no filesystem, no container/atlas
resolution, no PyTree modules). Kept in scope and specced below:

- The **integrators**: the SDE family, the exponential / local-linearization
  ODE step, symplectic steppers, and delay (DDE) integration.
- The **noise plumbing**: the keyed Brownian-increment contract (D2) and the
  colored-noise composition (an OU process / filtered white noise via the
  shipped `signal` IIR path).
- The **inversion substrate** that is genuinely numerical: the resolvent /
  transfer-function solves `(sI − J)⁻¹` (linalg), the matrix-exponential action
  `expm(A)·v` (Krylov), the symplectic leapfrog step an HMC sampler is built on.

**Out of scope — models and frameworks → downstream (nimox / ilex):**

- **Every dynamical model**: Balloon–Windkessel haemodynamics, the bilinear DCM
  neural state equation, Jansen–Rit / canonical-microcircuit neural masses,
  reduced Wong–Wang (DMF), the Hopf / Stuart–Landau oscillator, Kuramoto, the
  Epileptor, Wilson–Cowan, neural fields, compartmental PK models, the Bloch
  equations. These are *model code* (parameters + the RHS), not integrators.
- **The coupling and observation operators**: the connectome matvec `G·Σ W·S(x)`
  (a *use* of the shipped sparse/graph matvec, assembled by the model), the BOLD
  observation equation, the EEG/MEG lead-field projection.
- **The inversion frameworks**: DCM's variational-Laplace / free-energy scheme,
  TVB's parameter-sweep + FCD/metastability harness, the NUTS *sampler* proper
  (warmup, dual-averaging step adaptation — the *leapfrog primitive* is in scope,
  the full algorithm is consumer/stats, à la blackjax), simulation-based
  inference (SBI) posterior networks (NN, consumer).

**Out of scope — the usual SPEC boundaries:**

- No I/O, no NIfTI/GIfTI/CIfTI, no time-series file formats → `thrux`.
- No transform / pipeline / dataset abstractions → `bitsjax` / `entense`.
- No Equinox / PyTree modules → `nimox`. The integrators take a bare callable
  `f(t, y)` (and a `key`); the model that *is* a PyTree module lives in nimox and
  closes its parameters over that callable (D1/D2).

## 3. Already shipped — the substrate this suite builds on

Verified against `src/nitrix` as of 2026-06-25. **These are done; new items
compose on them.**

| Capability | Symbol(s) | Module |
|---|---|---|
| Fixed-step ODE family (differentiable, `lax.scan`) | `euler`, `midpoint`, `rk4`, `odeint` | `numerics.ode` |
| Implicit-solve / fixed-point (Picard / Anderson, implicit-VJP) | `fixed_point_solve` | `numerics.fixed_point` |
| Matrix exponential / logarithm | `matrix_exp`, `matrix_log` | `linalg` |
| SPD matrix functions (exp/log/sqrt/power on the cone) | `symexp`, `symlog`, `symsqrt`, `sympower` | `linalg` |
| Matrix-free / dense SPD solve + CG | `solve`, `cho_solve`, `cg` | `linalg` |
| Extremal eigensolver (Jacobian spectra; resolvent via eig) | `eigsolve_top_k` | `linalg` |
| Graph Laplacian (dense / ELL / SectionedELL) | `graph.laplacian` family | `graph` |
| Sparse connectome matvec | `ELL` / `SectionedELL` apply | `sparse` |
| Analytic signal / instantaneous phase & frequency / envelope | `analytic_signal`, `instantaneous_phase`, `instantaneous_frequency`, `envelope` | `signal` |
| Butterworth IIR / zero-phase (post-sim + colored noise) | `iir_filter`, `sosfiltfilt`, `bandpass`/… | `signal` |
| Time-series convolution (HRF / input-function conv) | `tsconv` | `signal` |
| Velocity-field integration (the shipped special-case ODE) | `integrate_velocity_field` | `geometry.grid` |
| Vector-field sampling (tractography / advection RHS) | `sample_at_points`, `Interpolator` ADT | `geometry` |
| Observation-noise models + inference | `glm_fit`/`glmm_fit`/`lme_fit`, `CorrSpec` (`ar1`/`car1`), `permutation_test`, `tfce` | `stats`, `stats.inference` |
| Parameter priors / sparse connectivity | `gp_fit`, `hsgp_basis`, `glasso` | `stats` |
| Keyed stochastic-kernel convention (the D2 precedent) | `augment.*` (keyed, stateless) | `augment` |

Consequences worth stating plainly:

- **The exponential-integrator core is already here.** `matrix_exp` + `cg` (and,
  for the action form, the Krylov work in §4) are exactly the kernels SPM's
  local-linearization scheme (`spm_int`) needs — DS-3 is *assembly*, not a new
  kernel.
- **Linear network-diffusion spreading is nearly free.** The Raj-2012
  neurodegeneration model `ẋ = −βLx` has the closed form `x(t) = expm(−βLt)·x₀`,
  and both the `graph` Laplacian and `matrix_exp` (and the heat-kernel path,
  §4) are shipped.
- **The post-simulation analysis layer is shipped.** TVB's read-outs — the
  Kuramoto order parameter, phase-based FC, metastability — are computed from
  `signal.analytic_signal` / `instantaneous_phase`; FCD is windowed correlation
  (`stats.corr`). Only the *forward integrator* (SDE, optionally DDE) is missing.
- **The differentiable-backward story already holds for the ODE family** and
  extends to the SDE family under D3 (fixed Brownian path). The continuous
  adjoint is deferred for both (§4).

## 4. Atomised items already filed (belong to this suite)

These exist under `docs/feature-requests/`; this ledger adopts them and records
DE-suite relevance + current status. **Add to the linked doc, not here.**

| Item | FR | Status (2026-06-25) | Suite role |
|---|---|---|---|
| Fixed-step ODE + adaptive/adjoint roadmap | [`ode-integrators.md`](ode-integrators.md) | euler/midpoint/rk4 ✅ shipped; adaptive dopri5/8/tsit5 + continuous adjoint deferred | the **seed** of this suite; owns DS-6 (adaptive) and DS-7 (adjoint) |
| Fixed-point / implicit combinators | [`fixed-point-combinators.md`](resolved/fixed-point-combinators.md) | ✅ `fixed_point_solve` shipped | backbone for an **implicit ODE step** and the **continuous-adjoint** fixed-point solve |
| Krylov solvers (CG ✅; MINRES/GMRES/BiCGStab deferred) | [`krylov-solvers.md`](krylov-solvers.md) | CG shipped; non-symmetric family deferred | the **non-symmetric resolvent** `(iωI − J)⁻¹` for spectral DCM transfer functions |
| Matrix functions | [`matrix-functions.md`](matrix-functions.md) | `matrix_exp`/`matrix_log` ✅; `expm_multiply` action open | exponential integrator (DS-3) + large-graph `expm(A)·v` (network diffusion) |
| Heat-kernel diffusion | [`heat-kernel-diffusion.md`](heat-kernel-diffusion.md) | partial (`diffusion_embedding` shipped) | `exp(−tL)` on the connectome = **linear network-diffusion spreading** |
| Gaussian-process models | [`gaussian-process-models.md`](gaussian-process-models.md) | ✅ `gp_fit`/HSGP shipped | **priors over model parameters** for Bayesian inversion |
| Graphical lasso | [`graphical-lasso.md`](resolved/graphical-lasso.md) | ✅ `glasso` shipped | sparse **effective-connectivity** priors / structure |
| Continuous / graph wavelet transform | [`continuous-wavelet-transform.md`](continuous-wavelet-transform.md), [`graph-wavelet-transform.md`](graph-wavelet-transform.md) | filed | time-frequency analysis of **simulated** output (post-sim) |

## 5. New gaps — integrator primitives not yet owned by any FR

The heart of this doc. Each item follows the house format (What / Proposed
surface / Composition / Likely consumer / Home / Effort / Live-code status).
IDs are `DS-n`. Effort scale matches the catalogue (XS/S/M/L/XL).

---

### DS-1 — Stochastic differential-equation integrators (SDE family) → `numerics.sde`

**What.** Fixed-step integrators for `dy = f(t,y)·dt + g(t,y)·dW` — **the single
highest-leverage addition**, the gate to the entire stochastic network-model
family (TVB, stochastic DCM, the Hopf resting-state model, digital twins). Ship
**Euler–Maruyama** (Itô, order-½ strong) and **stochastic Heun** (the TVB
workhorse: a predictor–corrector, Stratonovich-consistent for general noise and
order-1 strong for additive noise), with **Milstein** (Itô correction term) as
an optional add for state-dependent noise.

**Proposed surface.**

```python
Drift = Callable[[Array, Array], Array]        # f(t, y)
Diffusion = Callable[[Array, Array], Array]    # g(t, y); shape per `noise`

def sde_euler_maruyama(drift, diffusion, y0, t, key, *, noise='diagonal'): ...
def sde_heun(drift, diffusion, y0, t, key, *, noise='diagonal'): ...   # TVB default
def sdeint(drift, diffusion, y0, t, key, *, method='heun', noise='diagonal'): ...
#   noise ∈ {'scalar', 'diagonal', 'general'}; t = sample times; key folded per step (D2)
```

**Composition.** A `lax.scan` over the `t`-intervals (the ODE family's shape),
with the per-step Brownian increment drawn via DS-2's keyed contract; the Heun
corrector reuses the drift/diffusion evaluations. Differentiable straight-through
the scan with the path held fixed (D3). **Home.** `numerics.sde`. **Effort L**
(new noise contract + gradient semantics + Itô/Stratonovich correctness tests).
**Live-code status.** Absent — nitrix has **zero** SDE/Brownian machinery.
**SPEC-review gated (Effort-L, §9): build on a named consumer** (a nimox TVB /
Hopf / digital-twin module, or a stochastic-DCM port). **Acceptance.**
Euler–Maruyama matches the Itô closed form for geometric Brownian motion (weak +
strong order); Heun recovers an OU process mean/variance; additive-noise Heun is
Stratonovich≡Itô; `jax.grad` through the solve is correct against a
fixed-path finite-difference oracle.

---

### DS-2 — Keyed Brownian-increment contract + colored noise → `numerics.sde`

**What.** The noise plumbing DS-1 stands on: reproducible, differentiable
Brownian increments as a deterministic function of a `jax.random` key (D2), plus
the composition for **colored** (temporally-correlated) noise that TVB and
biophysical models use.

**Proposed surface.**

```python
def brownian_increments(key, t, shape, *, dtype=float32) -> Array:
    """√Δt · 𝒩(0,1) per `t`-interval; deterministic in `key` (fold-in step index)."""
# colored noise is *composition*, not a new primitive: integrate an OU process as
# its own SDE state, or filter white increments through signal.iir_filter.
```

**Composition.** `jax.random.fold_in` per step index over `brownian_increments`;
colored noise = an Ornstein–Uhlenbeck augmentation (an extra SDE state) **or**
white increments through the shipped `signal` IIR — no new kernel. **Home.**
`numerics.sde`. **Effort S** (small surface, but **load-bearing** — it is the
contract that makes DS-1 reproducible and differentiable). **Live-code status.**
Absent; the keyed-stateless convention is precedented in `augment`. **Acceptance.**
Same key → byte-identical path; `fold_in` decorrelation across steps; OU
autocorrelation matches `exp(−Δt/τ)`; gradient flows through a closed-over noise
parameter.

---

### DS-3 — Exponential / local-linearization integrator → `numerics.ode`

**What.** The integrator that unlocks the **classic SPM DCM/Balloon** scheme and
makes **stiff** linear-ish systems exact. For a semilinear system `y' = A·y +
N(t,y)` the exponential-Euler step integrates the linear part exactly via
`expm`; the **local-linearization** (SPM `spm_int`) variant linearizes the full
RHS each step and applies `expm` of the augmented Jacobian — exact for linear
systems, first-order for nonlinear, and stable where explicit RK struggles
(the Balloon haemodynamic model is mildly stiff).

**Proposed surface.**

```python
def exponential_euler(linear, nonlinear, y0, t): ...    # y' = linear·y + nonlinear(t,y)
def local_linearization(f, y0, t): ...                  # SPM spm_int: per-step expm(J·Δt)
```

**Composition.** `matrix_exp` (shipped) for the propagator and `jax.jacfwd`
for the per-step Jacobian; the action form `expm(A)·v` routes through the Krylov
work in §4 for large systems (avoids dense `expm` on a full connectome).
Differentiable. **Home.** `numerics.ode` (it integrates ODEs). **Effort M**
(**mostly assembly** over shipped primitives; the work is stiff-stability +
accuracy testing). **Live-code status.** Core kernels (`matrix_exp`, `cg`)
shipped; the integrator is not assembled. **Acceptance.** Exact for a linear
test system to round-off; first-order LL convergence on a nonlinear oracle;
stable on a stiff Balloon-like system where explicit Euler diverges at the same
step.

---

### DS-4 — Symplectic integrators (leapfrog / velocity-Verlet) → `numerics.ode`

**What.** Energy-conserving integrators for separable Hamiltonian flows. **The
consumer the ODE FR called "no consumer" — HMC** (and the No-U-Turn sampler) for
Bayesian inversion of digital twins (the Virtual Epileptic Patient inverts
per-region excitability with NUTS). Also velocity-Verlet for second-order
mechanical neural-mass schemes.

**Proposed surface.**

```python
def leapfrog(grad_potential, q0, p0, t, *, mass=None): ...   # velocity-Verlet; returns (q, p) trajectory
```

**Composition.** The half-kick / drift / half-kick three-stage step; reuses
`jax.grad` for the potential gradient and the existing `lax.scan` trajectory
shape. **Home.** `numerics.ode`. **Effort S–M.** **Live-code status.** Absent;
roadmap-only in [`ode-integrators.md`](ode-integrators.md) (now with a consumer).
**Note.** The leapfrog *primitive* is in scope; the full NUTS sampler (warmup,
dual-averaging, tree-doubling) is **consumer/stats** (à la blackjax), not nitrix.
**Acceptance.** Bounded energy drift over a long harmonic-oscillator trajectory
(vs. the secular drift of RK4 at the same step); time-reversibility to round-off.

---

### DS-5 — Delay-differential-equation integration (history buffer) → `numerics.dde`

**What.** Fixed-step integration of `dy/dt = f(t, y(t), y(t−τ))` for systems with
**conduction delays** — the one genuinely-new integrator *shape* and the residual
gap for full TVB parity (`τ_ij = distance_ij / conduction_speed`). Also the
corticothalamic (delay-PDE) EEG-spectrum models.

**Proposed surface.**

```python
def dde_integrate(f, history, t, *, max_delay, method='heun', key=None): ...
#   f(t, y, y_delayed); `history` provides y(t<t0); ring buffer holds the past `max_delay` window
```

**Composition.** A `lax.scan` over a **ring-buffer** carry holding the last
`ceil(max_delay/Δt)` states + interpolation (`numerics`/`signal`
`linear_interpolate`) at `t−τ`; with an optional `key`, the delayed-stochastic
(D2) variant for noisy TVB. The delay structure is static (shapes known at trace
time) so the buffer is a fixed-shape carry. **Home.** `numerics.dde`. **Effort
L** (new control-flow shape; buffer sizing + interpolation correctness). **SPEC-
review gated (§9): build on a concrete TVB-with-delays consumer** — do **not**
speculatively build. **Live-code status.** Absent. **Acceptance.** Recovers the
analytic solution of a linear scalar DDE (`y' = −a·y(t−τ)`) including its
oscillatory/stability regimes; reduces to the ODE integrator at `τ→0`.

---

## 6. Capability coverage matrix

Each named neuroimaging-dynamics capability, its governing DE class, the
primitive it needs, and coverage. Legend: ✅ shipped · 🆕 new gap (DS-n) ·
📋 filed FR (§4) · 🚫 model/framework → downstream (nimox/ilex).

| Capability | DE class | Integrator / numerical need | Model & inversion (downstream) |
|---|---|---|---|
| **Neural→haemodynamic (Balloon–Windkessel)** | nonlinear ODE (mildly stiff) | ✅ `rk4` / 🆕 DS-3 (LL, the SPM scheme) | 🚫 Balloon eqs + BOLD obs |
| **DCM-fMRI (bilinear)** | ODE + Balloon | ✅ ODE; `jax.grad` Jacobian (free); ✅ `gauss_newton`/`cho_solve` | 🚫 bilinear `ż=(A+ΣuB)z+Cu`; variational Laplace |
| **Stochastic DCM** | SDE | 🆕 DS-1 / DS-2 | 🚫 state-noise model; generalised filtering |
| **Spectral DCM (resting-state)** | linearized ODE → frequency | ✅ `eigsolve`/`solve`; 📋 non-symmetric resolvent `(iωI−J)⁻¹` (krylov) | 🚫 cross-spectral-density fit |
| **DCM-EEG/MEG (neural mass)** | coupled ODE; steady-state spectra | ✅ ODE; 📋 transfer functions | 🚫 Jansen–Rit / canonical microcircuit |
| **TVB whole-brain network** | **SDE** (+ optional delays) | 🆕 DS-1 (Heun) + DS-2; 🆕 DS-5 (delays); ✅ sparse/graph coupling matvec; ✅ `signal` phase/FC read-out | 🚫 mass models + connectome coupling + BOLD/lead-field |
| **Hopf / Stuart–Landau resting-state** | coupled SDE | 🆕 DS-1 / DS-2; ✅ coupling matvec | 🚫 `ż=(a+iω−|z|²)z+GΣW(z_j−z_i)+βη`; FC/FCD fit |
| **Brain digital twin / Virtual Epileptic Patient** | SDE forward + Bayesian inversion | 🆕 DS-1/DS-2; 🆕 DS-4 (leapfrog→HMC); differentiable sim (✅) | 🚫 Epileptor model; NUTS / SBI |
| **Network-diffusion neurodegeneration** | linear ODE | ✅ `graph` Laplacian + `matrix_exp`; 📋 heat-kernel `exp(−tL)`; nonlinear FKPP via ✅ ODE | 🚫 spreading model + staging |
| **Tractography (streamline)** | ODE along a vector field | ✅ `euler`/`rk4` + ✅ `Interpolator`/`sample_at_points` | 🚫 fODF-peak field + seeding/stopping |
| **Neural fields (Amari / Wilson–Cowan field)** | integro-differential / PDE on mesh | ✅ ODE-in-time + ✅ graph/geodesic conv for the spatial integral | 🚫 field model |
| **PET / DCE pharmacokinetics** | linear compartmental ODE / convolution | ✅ ODE / DS-3 / ✅ `tsconv` | 🚫 Tofts / 2TCM + TAC fit |
| **Bloch equations (MR physics)** | ODE | ✅ `rk4` / 🆕 DS-4 (Verlet) | 🚫 sequence model |
| **LDDMM geodesic shooting** | ODE on momentum (EPDiff) | ✅ ODE + ✅ `integrate_velocity_field` | 🚫 shooting recipe (register-adjacent) |

Read-out: deterministic capabilities are largely **reachable today or by
assembly** (DCM-Balloon via DS-3, tractography by composition, network-diffusion
nearly free); the **stochastic network-model family is gated on DS-1/DS-2**; full
TVB parity additionally wants DS-5; digital-twin inversion brings DS-4 a real
consumer.

## 7. Phasing & dependency graph

```
Phase 0  substrate & contracts (NEW — unblocks everything stochastic)
  DS-2 keyed Brownian-increment contract + colored-noise composition
  (confirm matrix_exp / cg / graph-Laplacian are integrator-ready — they are)

Phase 1  deterministic enablers (compose on shipped substrate — low risk, high value)
  DS-3 exponential / local-linearization integrator  ──> DCM-Balloon forward
  (network-diffusion: graph-Laplacian + matrix_exp / heat-kernel — already composable)
  (tractography: ode + Interpolator — already composable; consumer-side recipe)

Phase 2  stochastic core (the big unlock — SPEC-review-gated on a named consumer)
  DS-2 ──> DS-1 SDE family (Euler–Maruyama, stochastic Heun, ±Milstein)
           ──> TVB / Hopf / stochastic-DCM / digital-twin forward models (downstream)

Phase 3  inversion substrate
  DS-4 symplectic leapfrog ──> HMC/NUTS digital-twin inversion (sampler downstream)
  non-symmetric resolvent (krylov, §4) ──> spectral-DCM transfer functions
  continuous adjoint (ode-integrators.md / fixed-point-combinators.md): deferred, fail-loud

Phase 4  delays (full TVB parity — SPEC-review-gated on a TVB-with-delays consumer)
  DS-5 DDE history-buffer integration  ──> conduction-delay whole-brain models
```

Rationale: Phase 0 is the cheap, load-bearing contract (DS-2) that makes all of
Phase 2 reproducible and differentiable. Phase 1 is deterministic composition on
shipped substrate — DS-3 is mostly assembly and de-risks the lowest-risk first
consumer (DCM-Balloon, all linalg present), while network-diffusion and
tractography need no new integrator at all. Phase 2 is the genuine unlock and is
correctly the most gated (Effort-L, named consumer). Phase 3 turns the
forward simulator into an inversion substrate (and finally gives symplectic a
consumer). Phase 4 quarantines the one new integrator *shape* (delays) behind a
concrete TVB consumer. The recommended **first consumer** is a deterministic
**DCM-Balloon** port (exercises DS-3, lowest risk); the **SDE family** lands when
a TVB/Hopf/digital-twin module is filed.

## 8. Cross-references

- **Seed FR.** [`ode-integrators.md`](ode-integrators.md) — the shipped fixed-step
  family + the deferred adaptive (DS-6) and adjoint (DS-7) roadmap this suite
  extends.
- **Inversion / solve dependencies.** [`fixed-point-combinators.md`](resolved/fixed-point-combinators.md)
  (implicit step + continuous-adjoint solve), [`krylov-solvers.md`](krylov-solvers.md)
  (non-symmetric resolvent for spectral DCM), [`matrix-functions.md`](matrix-functions.md)
  (`expm` action), [`heat-kernel-diffusion.md`](heat-kernel-diffusion.md)
  (`exp(−tL)` network diffusion).
- **Prior / structure dependencies.** [`gaussian-process-models.md`](gaussian-process-models.md),
  [`graphical-lasso.md`](resolved/graphical-lasso.md).
- **Governance.** SPEC §1 (charter / non-goals), §4 (no model subpackage), §9
  (consumer-first graduation gate — every DS item is gated on a named blocked
  consumer; the Effort-L items DS-1/DS-5 + the continuous adjoint need SPEC-level
  review before slotting). Record each graduation in `IMPLEMENTATION_PLAN.md` at
  merge.
- **Implementation plan (the "how").** A companion `docs/design/dynamics-suite.md`
  — per-task signatures, the Itô/Stratonovich + path-fixed-gradient contracts,
  the differentiability classes, and the test matrix — is **not yet written**;
  author it when the first consumer files (the geometry suite's
  ledger→`docs/design/geometry-suite.md` precedent).
- **Live substrate.** `src/nitrix/numerics/{ode,fixed_point}.py`,
  `src/nitrix/linalg/`, `src/nitrix/graph/`, `src/nitrix/signal/`,
  `src/nitrix/geometry/grid.py`.
