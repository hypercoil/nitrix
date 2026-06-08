# Registration substrate & recipes

> **TL;DR.**  Two representative registration algorithms, *backed by
> nitrix*, in preparation for the ``entense`` backend: a **rigid +
> affine** intensity registrator (Gauss–Newton / Levenberg–Marquardt on
> SE(3)/affine — the ``3dvolreg`` / AIR lineage) and a **diffeomorphic
> log-Demons** registrator (stationary velocity field; ESM force;
> Gaussian fluid+diffusion regularisation; scaling-and-squaring exp).
> Both are *optimisation on a Lie group with an ``exp`` map* — and nitrix
> already owns one ``exp`` (``integrate_velocity_field``).  nitrix ships
> the **kernels** (distributed into existing modules + two small new
> namespaces ``metrics`` and ``register``); the registration
> **algorithms** are thin, pure-functional **recipes** over them
> (``NamedTuple`` outputs — the ``reml_fit`` precedent, *not* the
> template/atlas API that SPEC §1 keeps in ``thrux``).

This is a design-plus-implementation-plan document, written so the work
survives a hand-off to a fresh machine.  It reads on top of ``SPEC.md``
(+ updates), ``IMPLEMENTATION_PLAN.md`` (phases / tracks / gates /
deviation protocol), and ``docs/design/geometry.md`` (the SVF stack the
diffeomorphic recipe lowers onto).

## 1. Why these two algorithms

**Rigid (+affine) = Gauss–Newton / Levenberg–Marquardt with se(3)/affine
parametrisation.**  ``3dvolreg`` (Cox & Jesmanowicz 1999) is *iterated
linearised weighted least squares* — Gauss–Newton on a first-order Taylor
expansion of the image in the motion parameters.  Recast with a Lie-group
update it is the inverse-compositional Lucas–Kanade / Baker–Matthews
scheme: second-order convergence, a trivial ``6×6`` (rigid) / ``12×12``
(affine) SPD solve per iteration, fully vectorised, GPU-native, and
analytically differentiable.  ``mcflirt``'s Powell search over the
correlation ratio is robust but not JAX-elegant (no gradients, sequential
line minimisations); GN/LM recovers its robustness through the *metric*
(swap SSD → LNCC / MI / CR) and the multi-resolution pyramid while
staying differentiable.

**Diffeomorphic = log-domain Demons (SVF).**  The log-Demons algorithm
(Vercauteren et al. 2009) maps **1:1 onto kernels nitrix already owns**:
``φ = exp(v)`` with ``v`` a stationary velocity field, ``exp`` via
scaling-and-squaring (== ``integrate_velocity_field``).  The optimisation
is operator splitting: a closed-form per-voxel **ESM demons force** (a
Gauss–Newton step on SSD with Vercauteren's symmetric Hessian
approximation — *no inner linear solve*) alternating with **Gaussian
smoothing** as the regulariser's Green's function (fluid = smooth the
update; diffusion = smooth the accumulated velocity).  SyN (Avants 2008)
is the quality benchmark but heavier (symmetric geodesic shooting,
time-dependent velocity, ODE adjoint); the *same primitive set* upgrades
Demons toward it — a symmetric forward+inverse formulation with an
**LNCC** metric is greedy-SyN, and full geodesic SyN then only adds the
ODE-adjoint layer (§12.11) as a later increment.

## 2. Algorithm outlines & constituent kernels

### 2.1 Rigid / affine (GN/LM)

Transform ``T(p) = exp(Σ pᵢ Gᵢ)`` (rigid: ``{Gᵢ}`` the 6 se(3)
generators; affine: the 12 gl generators), ``p`` the parameter vector.
Residual ``r(x;p) = M(T(p)·x) − F(x)``.  Steepest-descent images
``∂r/∂pⱼ = ∇M · (Gⱼ x)`` (image gradient ∘ warp-parameter Jacobian).
Gauss–Newton normal equations ``(JᵀWJ) δp = −JᵀW r`` — an SPD solve; LM
adds ``λ·diag(JᵀWJ)``.  Manifold update ``T ← exp(δξ)·T``.  Coarse-to-fine
over a Gaussian pyramid; warm-start each level.

| Kernel | Status | Home |
|---|---|---|
| image spatial gradient ``∇M`` | ❌→**R0** | ``geometry.spatial_gradient`` |
| Gaussian pyramid | ❌→**R0** | ``geometry.gaussian_pyramid`` |
| similarity + gradient (SSD/NCC/LNCC/MI/CR) | ❌→**R0** | ``nitrix.metrics`` |
| SPD solve (normal equations) | ⚠️→**R0** | ``linalg.cho_solve`` |
| se(3)/affine generators + ``exp``/``log`` | ❌→R1 | ``geometry.transform`` (+ ``linalg.matrix_exp``) |
| general ``matrix_exp`` / ``matrix_log`` (+ Fréchet VJP) | ❌→R1 | ``linalg`` (graduates §12.2) |
| ``apply_affine`` / ``affine_grid`` | ⚠️→R1 | ``geometry.transform`` |
| GN / LM optimiser (small-residual NLLS) | ❌→R1 | ``linalg.optimize`` |
| resample / interpolation | ✅ | ``geometry`` (``spatial_transform``, ``Interpolator``) |

### 2.2 Diffeomorphic (log-Demons, SVF)

Initialise ``v = 0``.  Per pyramid level, repeat: warp ``M∘exp(v)``;
ESM force ``u = (F − M∘φ)·J / (|J|² + (F − M∘φ)²/σ²)`` with
``J = ½(∇F + ∇(M∘φ))``; **fluid** ``u ← G_σf(u)``; **log-update**
``v ← v ∘ exp(u) ≈ v + u + ½[v,u]`` (BCH); **diffusion** ``v ← G_σd(v)``;
optional symmetrise (forward + inverse force).  Monitor ``det J`` for
folding.

| Kernel | Status | Home |
|---|---|---|
| image spatial gradient | ❌→R0 (shared) | ``geometry.spatial_gradient`` |
| SVF exponential | ✅ | ``geometry.integrate_velocity_field`` |
| warp / resample | ✅ | ``geometry.spatial_transform`` |
| Gaussian fluid+diffusion | ✅ | ``smoothing.gaussian`` |
| folding QA | ✅ | ``geometry.jacobian_det_displacement`` |
| LNCC metric + gradient | ❌→R0 (shared) | ``nitrix.metrics`` |
| Gaussian pyramid | ❌→R0 (shared) | ``geometry.gaussian_pyramid`` |
| field composition (BCH) | ⚠️→R2 | ``geometry.compose_velocity`` |
| ``invert_displacement`` (symmetric variant) | ❌→R2 | ``geometry`` (composes §12.8) |
| ESM demons force | recipe | ``nitrix.register`` |

The demons force itself is closed-form arithmetic over the kernels above
— a *recipe*, not a kernel.

## 3. Module layout (final)

- ``nitrix.geometry``: ``spatial_gradient``; ``gaussian_pyramid`` /
  ``downsample`` / ``upsample``; ``geometry.transform`` (se(3)/affine
  generators, ``rigid_exp/log``, ``affine_exp/log``, ``apply_affine`` /
  ``affine_grid``); ``compose_displacement`` / ``compose_velocity`` /
  ``invert_displacement``.
- ``nitrix.linalg``: ``matrix_exp`` / ``matrix_log`` (+ ``frechet_derivative``,
  graduates §12.2); ``solve`` / ``cho_solve``; ``linalg.optimize``
  (``gauss_newton`` / ``levenberg_marquardt``); ``linalg.krylov.cg``
  (graduates §12.1).
- ``nitrix.numerics``: ``fixed_point.fixed_point_solve`` (+ implicit-VJP,
  graduates §12.8).
- ``nitrix.metrics`` *(new)*: ``ssd``, ``ncc`` (global), ``lncc``
  (local), ``mutual_information`` (+ ``joint_histogram``),
  ``correlation_ratio``.  Substrate-composition story per term: LNCC =
  local sums (separable box / ``semiring_conv`` REAL); NCC = ``stats.corr``
  shape; MI / CR = soft (Parzen) histogram scatter-add.
- ``nitrix.register`` *(new — pure-functional recipes)*:
  ``rigid_register`` / ``affine_register`` /
  ``diffeomorphic_demons_register`` returning ``NamedTuple`` results, a
  ``RegistrationSpec`` frozen-record config (the ``SolverSpec`` /
  ``Interpolator`` ADT precedent — static config on ``jit`` static slots),
  and a shared coarse-to-fine driver.

## 4. Phases, gates, validation

Phases follow ``IMPLEMENTATION_PLAN`` idiom (capability-oriented;
non-negotiables in §2.2 hold throughout: pure-functional surface, JAX
fallback floor, golden corpus, jaxtyping, ruff, ``custom_vjp`` where
stability/efficiency needs it).

> **Status (2026-06-08): R0 + R1 SHIPPED** on branch
> ``registration-suite``.  Developed/tested on an L4 GPU whose cuSolver
> handle pool is dead (``eigh`` / ``cholesky`` / ``solve`` / ``inv`` /
> ``svd`` all fail; only ``triangular_solve`` + conv/elementwise work) --
> hence the SPD/general split below: general dense solves and ``matrix_exp``
> route through the CPU-fallback ``_solver.safe_*`` wrappers, while the
> SPD optimiser path is matrix-free (``cg`` / BFGS) and stays on-device.

- **R0 — shared image substrate.** ✅ SHIPPED.  ``geometry.spatial_gradient``;
  ``geometry.gaussian_pyramid`` / ``downsample`` / ``upsample``;
  ``nitrix.metrics`` (SSD / NCC / LNCC / MI / CR + gradients);
  ``linalg.cho_solve`` / ``solve``.  Shared separable engine
  ``_internal.separable.correlate1d`` (de-dups gradient + LNCC box-sum).
  **Gate G-R0:** finite-difference gradient checks on every metric
  (incl. soft-histogram MI / CR) -- passing.
- **R1 — rigid + affine.** ✅ SHIPPED.  ``linalg.matrix_exp`` (general,
  via ``safe_expm``; ``matrix_log`` / Fréchet still backlog);
  ``linalg.cg`` (matrix-free SPD, graduates §12.1);
  ``linalg.optimize.{gauss_newton, levenberg_marquardt}`` (matrix-free,
  ``jvp``/``vjp`` + ``cg``); ``geometry.transform`` (closed-form rigid
  exp/log, ``matrix_exp``-based affine, ``apply_affine`` / ``affine_grid``);
  recipes ``register.{rigid_register, affine_register}`` (coarse-to-fine;
  SSD->GN/LM, LNCC/MI/CR->BFGS).
  **Gate (met):** exp round-trip + grad; **synthetic-recovery** of known
  2-D/3-D rigid and 2-D affine transforms (ncc > 0.97-0.99).  Still TODO:
  parity vs AFNI ``3dvolreg`` / FSL ``mcflirt`` / ANTs on real data.
  *Bug found & fixed by the 3-D oracle:* ``_so3_exp`` ``sqrt(0)`` NaN in
  reverse-mode froze the 3-D optimiser at identity (regression-tested).
- **R2 — diffeomorphic.** ✅ SHIPPED.  ``numerics.fixed_point_solve``
  (Picard + implicit-VJP, graduates §12.8);
  ``geometry.{compose_displacement, compose_velocity, invert_displacement}``;
  recipe ``register.diffeomorphic_demons_register`` (stationary velocity
  field; closed-form ESM force; Gaussian fluid+diffusion; scaling-and-
  squaring exp; coarse-to-fine).  ``DemonsSpec`` / ``DiffeomorphicResult``.
  **Gate (met):** **diffeomorphism** -- synthetic-warp recovery (2-D ncc
  0.965→0.999, 3-D) with min ``det J > 0`` (no folding); identity +
  validation.  Still TODO: LNCC-driven / symmetric-forces variants;
  parity vs ANTs SyN / demons on real data.
- **R3 — differentiable layer + polish.** ✅ SHIPPED.
  ``linalg.implicit_least_squares`` (IFT differentiation of an argmin
  through the optimum -- GN-Hessian adjoint via ``cg``, O(1) memory),
  the differentiable-layer entry point; the unrolled path is the
  always-available alternative (both verified w.r.t. images).  §12.1 /
  §12.2 / §12.8 graduations recorded in ``IMPLEMENTATION_PLAN §10.3`` and
  the feature-request docs flipped to shipped.  ``bench/perf_registration.py``
  (GPU-native rigid-SSD + Demons, fwd + grad); usage in §6.1.  *Deferred
  upgrades:* ``matrix_log``; LNCC-driven / symmetric Demons; real-data
  parity vs AFNI ``3dvolreg`` / FSL ``mcflirt`` / ANTs SyN.

## 5. Decisions on record (2026-06-08)

1. **Driver home = nitrix recipe layer** (``nitrix.register``), pure
   functions → ``NamedTuple`` (LME / ``reml_fit`` precedent).  Distinct
   from the SPEC §1 "no template/atlas registration API" non-goal, which
   is about atlas *data structures* (those stay in ``thrux``).
2. **Diffeomorphic representative = log-Demons (SVF).**  SyN is the
   upgrade path, not the first cut; full geodesic SyN deferred (needs
   §12.11 + §12.8).
3. **Metrics include cross-modal:** SSD, LNCC, MI (Mattes/Parzen
   soft-histogram), correlation ratio.  LNCC is the diffeomorphic
   workhorse; MI / CR primarily serve rigid/affine cross-modal.
4. **Rigid + affine** from the start; affine via general
   ``linalg.matrix_exp`` (graduating §12.2).

## 6. Differentiability (the ``entense``-backend requirement)

Two paths, both shipped: **unrolled** (every primitive is differentiable,
so ``jax.grad`` through a recipe works out of the box — verified for the
SSD rigid recipe and the Demons recipe w.r.t. the input images) and
**implicit** (``linalg.implicit_least_squares`` differentiates an argmin
through the optimum by the implicit-function theorem — solving the
Gauss-Newton-Hessian adjoint ``(JᵀJ) w = x̄`` with ``cg`` and pushing
``-w`` through ``∂_data(Jᵀr)``, O(1) memory in the iteration count;
``numerics.fixed_point_solve`` is the analogous combinator for the
velocity/inverse fixed points).  This is what lets ``entense`` use a
registrator as a differentiable layer or a loss.  (The BFGS metric path
— LNCC/MI/CR — is not differentiable through the solve; use the SSD/LM
path or ``implicit_least_squares`` when a backward is needed.)

## 6.1 Usage

```python
from nitrix.register import (
    rigid_register, affine_register, RegistrationSpec,
    diffeomorphic_demons_register, DemonsSpec,
)

# Rigid (motion correction; SSD + Gauss-Newton/LM, coarse-to-fine).
res = rigid_register(moving, fixed,
                     spec=RegistrationSpec(levels=3, iterations=30))
res.matrix      # (ndim+1, ndim+1) homogeneous transform (fixed -> moving)
res.warped      # moving resampled onto the fixed grid

# Cross-modal rigid (correlation ratio / MI via BFGS).
affine_register(moving, fixed,
                spec=RegistrationSpec(metric='cr', levels=3))

# Diffeomorphic (log-Demons; guaranteed diffeomorphism).
d = diffeomorphic_demons_register(moving, fixed,
                                  spec=DemonsSpec(levels=3, iterations=80))
d.velocity, d.displacement, d.warped
assert float(d.jacobian_det.min()) > 0.0   # no folding

# Differentiable layer (exact gradients at the optimum, no unrolling):
from nitrix.linalg import implicit_least_squares
# residual_fn((moving, fixed), theta) = (warp(moving, theta) - fixed).ravel()
```

## 7. Out of scope (scope discipline)

Atlas/template data structures and template-aware ops (→ ``thrux``); any
I/O; PyTree / module wrappers (→ ``nimox`` / ``entense``);
Powell/derivative-free optimisers (GN/LM + autodiff supersede); full
geodesic-shooting SyN in v1 (defer to the ODE-adjoint increment).

## Cross-references

- ``docs/design/geometry.md`` — the SVF stack (``integrate_velocity_field``,
  ``spatial_transform``, ``jacobian_det_displacement``, ``resample``,
  ``Interpolator``) the diffeomorphic recipe lowers onto.
- ``docs/feature-requests/{matrix-functions,krylov-solvers,fixed-point-combinators,ode-integrators}.md``
  — §12 candidates that registration (the named blocked consumer) graduates.
- ``src/nitrix/{geometry,linalg,numerics,metrics,register}/`` — the homes above.
- ``IMPLEMENTATION_PLAN.md`` §2.2 (non-negotiables), §10.A (deviation log
  where the §12 graduations are recorded as they land).
