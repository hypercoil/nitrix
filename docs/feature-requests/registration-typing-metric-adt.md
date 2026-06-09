# Register v2: Metric ADT + TransformModel protocol + differentiable non-SSD metrics

> **Status (2026-06-09): SHIPPED (A+B+D).** The engineering-rigor /
> extensibility follow-up to the P0 perf round
> (`registration-recipe-cold-compile.md`, shipped `ddc2e10`). Replaced
> the stringly-typed metric/transform dispatch in the registration
> recipes with house-style ADTs (the `Interpolator` Protocol +
> frozen-dataclass precedent), and made the cross-modal metrics
> differentiable as a layer. Deliberately structured to *enable* the
> coming registration suite (BBR, greedy SyN, full SyN / LDDMM), not
> clash with it.
>
> Landed as three commits on branch `perf/registration-roll-loops`:
> **B** `TransformModel` (`Rigid`/`Affine`) — `01b3b80`; **A** `Metric`
> ADT (`SSD`/`LNCC`/`MI`/`CorrelationRatio`) — `3e4da91`; **D**
> `linalg.implicit_minimize` (general scalar-objective IFT, exact Hessian
> via matrix-free `cg`; `O(M + P)` memory, O(1) in the iteration count)
> + the non-SSD differentiable-layer tests. All public `nitrix.register`
> exports; clean-break metric API. 77 registration tests green; mypy
> (111 files) + ruff clean. C (closed-form steepest-descent) and E
> (adaptive `matrix_exp`) remain deferred. Bench re-run still pending per
> the hold.

## Locked decisions (user, 2026-06-09)

1. **Scope = A + B + D** (below). C and E are opt-in / deferred.
2. **Clean break** on the metric API: `metric='ssd'` → `metric=SSD()`;
   `lncc_radius` / `bins` leave `RegistrationSpec` (they migrate onto the
   metric records). Existing recipe tests get updated; no string-compat
   shim.
3. **Public exports**: the `Metric` / `TransformModel` implementers
   (`SSD` / `LNCC` / `MI` / `CorrelationRatio`; `Rigid` / `Affine`) are
   top-level `nitrix.register` exports (the public `Interpolator`
   precedent), so `entense` and other consumers construct them directly.

## A. `Metric` ADT — replace the stringly-typed metric dispatch

**Problem (cited).** `RegistrationSpec.metric: str` (`register/_core.py:87`)
is dispatched by `if`-chains in `_metric_cost`
(`_core.py:148-160`, which raises `ValueError` at trace time on an
unknown metric) and the `use_lsq = spec.metric == 'ssd' and …` branch
(`_core.py:174`). Per-metric parameters are flattened onto the spec as a
god-config (`lncc_radius` `_core.py:94`, `bins` `_core.py:95`), which
balloons as metrics are added.

**Design.** A `Metric` Protocol (mirroring `geometry._interpolate.Interpolator`),
with frozen-dataclass implementers carrying their own parameters, each
wrapping the existing `nitrix.metrics` kernels — **no metric math moves**
(`ssd` / `lncc` / `mutual_information` / `correlation_ratio` stay in
`nitrix.metrics`); this is encapsulation + cost-orientation only:

- `is_least_squares: ClassVar[bool]` — `SSD` True, others False → selects
  the GN/LM least-squares path vs the scalar BFGS path (formalises the
  current `use_lsq` branch).
- `cost(self, warped, fixed) -> scalar` — lower-is-better (the
  minimisation objective; e.g. `1 - lncc`, `-mutual_information`).
- `residual(self, warped, fixed) -> (M,)` on the least-squares members
  (the vector the GN/LM path consumes).
- Implementers: `SSD()`, `LNCC(radius=4)`, `MI(bins=32,
  normalized=False)`, `CorrelationRatio(bins=32)`.
- `RegistrationSpec.metric: Metric = SSD()` (was `str`); drop
  `lncc_radius` / `bins`.

**Extensibility / honesty.** Removes the `if`-chain; a new *dense
image-pair* metric (e.g. a MIND / self-similarity descriptor) is one
implementer. **BBR is *not* a drop-in `Metric` here** — its cost is over
boundary-point samples along surface normals (`cost(T, moving, surface)`,
no `fixed` image) — so it is a *sibling objective*, not
`Metric.cost(warped, fixed)`. What BBR reuses is the **optimiser +
`TransformModel` (B) + the point sampler**; it follows the same Protocol
*shape*. The longer-term generalisation is an `Objective` protocol
(θ ↦ cost/residual over closed-over data) of which the image-pair
`Metric` is one constructor — noted, not built until BBR lands.

## B. `TransformModel` protocol — unify rigid/affine, decouple the driver

**Problem (cited).** `ExpFn = Callable[..., …]` is loose
(`_core.py:49`); `n_params` is recomputed ad-hoc per recipe
(`recipes.py:71`, `recipes.py:100`); and the driver hard-codes the
parameter *layout* in the warm-start rescale —
`jnp.concatenate([params[:-ndim], params[-ndim:] * ratio])`
(`_core.py:259`) assumes "translation is the last `ndim` params," which
breaks for any future parametrisation ordered differently.

**Design.** A `TransformModel` Protocol with frozen `Rigid()` /
`Affine()` implementers wrapping `geometry.rigid_exp` / `affine_exp`:

- `exp(self, params, *, ndim) -> matrix`
- `n_params(self, ndim) -> int`
- `rescale_to_grid(self, params, ratio) -> params` (the model owns the
  translation rescale; the driver stops slicing `params[-ndim:]`)
- optional `generators(ndim)` — the hook that enables C.

The recipes pass a `model` instead of the `exp_fn` + `n_params` + `ndim`
triplet. **BBR uses `Rigid()`**; greedy SyN's dense velocity field is a
*different* family (large-`P`, the diffeomorphic driver, not this
matrix-transform model), so `TransformModel` scopes the matrix-transform
recipes + BBR cleanly. A future 7-DOF similarity model = one implementer.

## D. Differentiable non-SSD metrics (capability gap)

**Problem (cited).** The BFGS path (LNCC / MI / CR) is not differentiable
through the solve (`_core.py:200`; design §6 admits it), so only SSD/LM
registration is a differentiable layer.

**Design.** With the `Metric` ADT, a scalar-cost implicit-function
backward — stationarity `∇_θ cost(θ*) = 0`, Hessian `∇²_θ cost` via the
metric's own autodiff, adjoint solved matrix-free with `cg` — makes the
LNCC/MI/CR registration differentiable w.r.t. the images too. This
generalises `linalg.implicit_least_squares` (which assumes a
least-squares residual + Gauss-Newton Hessian) to a general scalar
objective; likely a sibling `linalg.implicit_minimize` (or a shared
core). The entense differentiable-layer story then spans all metrics, not
just SSD.

## C. (opt-in micro-opt, after B) closed-form steepest-descent
With B exposing generators, the assembled Jacobian column
`J_:,j = ∇M·(Gⱼx)` is analytic — faster + a smaller graph than the
`jacfwd`-through-the-warp of the P0 assembled path. Registration-specific
(tracks the generator set), so opt-in (`jacobian='analytic'`) and only if
profiling warrants. **Deferred.**

## E. (minor) adaptive `matrix_exp` squaring
Key `n_squarings` on `‖A‖` (near-identity affine generators → fewer
squarings), trimming the affine graph. **Deferred / bundle-or-drop.**

## Sequencing
**B → A → D.** B is smaller, unblocks C, and removes the layout coupling;
A is the bigger surface change and the BBR-readiness centrepiece; D is the
capability add on top of A. Public recipe call sites
(`rigid_register(m, f, spec=…)`) stay stable; the one real public break is
the `RegistrationSpec` metric field (string → `Metric` ADT), handled as a
clean break per the locked decision.

## Validation (gates for the round)
- Existing synthetic-recovery recipe tests + the R0 finite-difference
  metric-gradient gates stay green (re-pointed at the ADT constructors).
- New: `is_least_squares` routing test (SSD → LM residual path; LNCC/MI/CR
  → BFGS scalar path) and the `TransformModel.rescale_to_grid` round-trip.
- D: grad-through-LNCC/MI/CR registration == finite-difference, and ==
  the unrolled gradient at convergence (the `implicit_least_squares`
  test pattern, generalised).
- ruff + ruff-format + mypy clean throughout (the standing gate).

## Cross-references
- P0 perf round (shipped): `registration-recipe-cold-compile.md`,
  commit `ddc2e10`.
- House ADT precedent: `geometry._interpolate.Interpolator` (Protocol) +
  `Linear` / `NearestNeighbour` / … (frozen dataclasses, `ClassVar`
  capability flags).
- Durable design: `docs/design/registration.md` (§3 module layout, §6
  differentiability).
- Sites: `src/nitrix/register/{_core,recipes,diffeomorphic}.py`,
  `src/nitrix/metrics/`, `src/nitrix/linalg/optimize.py` (D).
