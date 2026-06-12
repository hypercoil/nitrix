# v4 Phase 2 — dense-field algebra↔group drivers + `field_log` (implementation design)

> **Status (2026-06-12): implementation-ready design, no code yet.** The detailed
> design for Phase 2 of [`registration-suite-v4.md`](registration-suite-v4.md):
> the dense-field **algebra↔group** split, the **group (greedy) driver**, and the
> **`field_log`** recovery bridge. Concrete signatures, algorithms, numerics, and
> the test plan, ready to implement against the current tree. Folds in the review
> findings for this phase (greedy is a *sibling driver*, not a seam-implementer;
> the per-step Jacobian bound; total-field smoothing retained; greedy≠SVF so the
> gate is recovery-tolerance; `invert_displacement` robustness is a hard
> predecessor). **Corrects** `registration-suite-v4.md §2.5`: greedy SyN needs
> **no per-iteration inversion** (the single inversion is at finalisation, exactly
> as the current recipe), so the gather reduction is a clean ~3–3.5×, not 4–7/iter.

## 1. Scope & deliverables

| # | Deliverable | Home (new/changed) |
|---|---|---|
| 2.1 | `field_log` (group→algebra log) + `_diffeo_sqrt` helper | `geometry/deformation.py` (new fns), `geometry/__init__.py` (export) |
| 2.2 | Group driver level functions (`group_single_sided_level`, `group_symmetric_level`) + `_group_regularise` / `_step_clamp_diffeo` | `register/_svf.py` (new fns) |
| 2.3 | Algebra driver = the **existing** `single_sided_level` / `symmetric_level`, unchanged (the exact parity oracle) | `register/_svf.py` (untouched numerics) |
| 2.4 | `representation` spec field; group/algebra dispatch + finalisation branch; on-demand velocity recovery | `register/diffeomorphic.py`, `register/_syn.py`, specs |
| 2.5 | `convergence` early-exit for the SVF path (followup A3) + `n_steps` reconciliation (C5) | `register/_svf.py`, specs |

**Public surface delta:** one new exported function (`geometry.field_log`); two new
spec fields (`representation`, `convergence`) on `DemonsSpec` / `SyNSpec`
(back-compatible defaults). `DiffeomorphicResult` / `SyNResult` shapes are
**unchanged** (velocity fields are recovered via `field_log` in group mode). The
existing recipe behaviour is reproduced exactly by `representation='algebra'`.

**Out of this phase (in v4 but elsewhere):** the forces themselves (Phase 1), the
recursive Gaussian (Phase 1d), `_normalise_step` RMS (0d), `invert_displacement`
robustness (0e — a *predecessor*, consumed here), the SVF-preamble refactor
(Phase 6 C4 — strictly after this).

## 2. The representation model (precise)

A dense deformation is an element of the diffeomorphism group `Diff`. Two carried
states, voxel-unit channel-last fields `(*spatial, ndim)`:

- **Group element** — displacement `s`, `φ = id + s`. Product = `compose_displacement`,
  inverse = `invert_displacement`. The *greedy* domain.
- **Algebra element** — stationary velocity `v` (a vector field). `exp = integrate_velocity_field`,
  `log = field_log` (§3). The *log-domain / SVF* domain.

`exp` is a local diffeomorphism but **not surjective**: a greedy-composed `φ`
(many small non-commuting warps) is in general **not** `exp(v)` for any single
stationary `v`. Consequences, load-bearing for the design:

- The **algebra driver** stays on the SVF submanifold → `v` is exact.
- The **group driver** explores all of `Diff` → `v = field_log(φ)` is the
  *best stationary-velocity fit* (exact iff `φ ∈ image(exp)`), recovered on demand.
- The two drivers solve **different variational problems** (greedy is not a
  one-parameter subgroup), so their fixed points differ → the cross-driver gate is
  *synthetic-recovery-to-tolerance*, never field-wise equality (§7).

This mirrors the shipped matrix transform-algebra (`matrix_exp`/`matrix_log`,
`transform_mean` in the algebra): `field_log` is the dense `matrix_log`.

## 3. `field_log` — the recovery bridge (group → algebra)

> Full numerical spec (convergence, the log-approximation order, the round-trip-vs-
> generator distinction, the fit residual): [`field-log.md`](field-log.md). Summary here.

`field_log(s) → v` with `integrate_velocity_field(v) ≈ id+s`, by **inverse
scaling-and-squaring**: take `n_sqrt` diffeomorphism square roots until near
identity, then scale.

```
v = field_log(s):
    w = s
    for _ in range(n_sqrt):          # w ← φ^{1/2} of current w (Python loop, static n_sqrt)
        w = _diffeo_sqrt(w)
    return w * (2.0 ** n_sqrt)
```

After `k` roots, `w = displacement(φ^{1/2^k})`; at `k = n_sqrt`, `φ^{1/2ⁿ} ≈ id + v/2ⁿ`,
so `v ≈ 2ⁿ·w`. Exact-on-SVF round-trip: `integrate_velocity_field(field_log(s), n_steps=n_sqrt) == id+s`
(both are "halve, square n times"), the parity oracle.

### 3.1 `_diffeo_sqrt` — the square root of a diffeomorphism (damped fixed point)

We need `w` with `compose_displacement(w, w) = s` (i.e. `(id+w)∘(id+w) = id+s`).
**The naïve Picard `w ← s − warp(w, id+w)` does not converge** — near identity its
linearisation is `w ← s − w` (Jacobian `−I`, oscillates). Use the **½-damped**
iteration, whose near-identity Jacobian is `0`:

```
w* = _diffeo_sqrt(s):  fixed point of   g(s, w) = w + ½·(s − compose_displacement(w, w))
                       init w0 = s/2,   via numerics.fixed_point_solve
```

At the fixed point `compose_displacement(w,w) = s`. Near identity
`compose_displacement(w,w) ≈ 2w`, so `g ≈ w + ½(s − 2w) = ½s` — converges in ~one
step from `w0 = s/2`; the early (larger-deformation) roots take more. Uses the
existing `numerics.fixed_point_solve` (`f(params=s, x=w)`), so it is **IFT-
differentiable** (O(1) memory) and consistent with `invert_displacement`.

### 3.2 Signature, numerics, contract

```python
def field_log(
    displacement: Float[Array, '*spatial ndim'],
    *,
    n_sqrt: int = 6,
    sqrt_tol: float = 1e-6,
    sqrt_max_iter: int = 50,
    mode: BoundaryMode = 'nearest',
) -> Float[Array, '*spatial ndim']:
    """Stationary-velocity logarithm of a displacement field (inverse
    scaling-and-squaring; the dense analogue of ``linalg.matrix_log``)."""
```

- `n_sqrt = 6` (matches the integration `n_steps` default and `matrix_log`'s
  `n_sqrt`); differentiable; GPU-native (compose + fixed point, no `safe_inv`,
  unlike `matrix_log` — so `field_log` is jit/grad-clean even on the wedged box).
- **Cost:** `n_sqrt` square roots × (~10–30 damped iters) × 1 gather ≈ ~120 gathers,
  paid **once** at finalisation — <1% over a recipe that ran hundreds of iterations.
- **Convergence regime (measured):** the ½-damped sqrt is **super-linear** — plain
  Picard reaches a `~1e-10` residual in under ten steps even for a large deformation,
  so no acceleration is used on the default path. Only a *pathological* `‖∇s‖ → 1`
  field (still invertible, `det J > 0`, but stiff) would slow Picard; **mitigation**
  is the opt-in Anderson on `fixed_point_solve` (validated to converge where Picard
  stalls on a genuinely stiff fixed point) plus a **reported residual**
  (`‖compose(w,w) − s‖`). Since the velocity is a *secondary* output, an
  approximate-with-residual recovery is acceptable; the displacement (primary output)
  never depends on `field_log`.
- **Feeds** `geometry.velocity_mean` / `transform_mean` (template construction):
  a group-solved warp recovers a velocity that the v3 barycentre machinery consumes.

## 4. The drivers (shared scaffold, two level closures, one finalisation branch)

### 4.1 The scaffold is reused unchanged

`_svf.svf_coarse_to_fine` is **untouched**. It carries `state: tuple[Array, ...]`
of `n_fields` voxel-unit `(*spatial, ndim)` fields and prolongs by
`upsample(s, shape) * ratio`. That prolongation is **representation-agnostic** — a
velocity and a displacement are both voxel-unit vector fields and prolong
identically — so the *only* differences between algebra and group mode are (a) the
per-level **`level_solve` closure** and (b) the recipe's **finalisation**. This is
what makes the group driver an honest *sibling* (a new closure + finalisation
branch), not a fork of the scaffold and not a fake "implementer behind the seam."

### 4.2 Algebra driver — unchanged (the exact oracle)

`single_sided_level` / `symmetric_level` (current code) **are** the algebra driver:
carry `v`, warp by `exp(v) = integrate_velocity_field(v, n_steps)`, `_regularise`
(fluid-smooth update → `v += u` → diffusion-smooth `v`), finalise by `exp`. Left
numerically identical so `representation='algebra'` reproduces today's recipe
byte-for-byte (the anchor oracle).

### 4.3 Group driver — new

State is the displacement(s). Per iteration: warp by `φ = id+s` **directly** (one
gather, no `exp`), force, fluid-smooth the increment, **compose** it onto `φ`
(compositive update), diffusion-smooth the **total** field. No per-iteration `exp`,
no per-iteration inversion.

**Single-sided (greedy Demons) — `group_single_sided_level`:**

```
per scan step, state s:
    warped = spatial_transform(moving, id + s, mode)          # 1 gather
    u      = mask/restrict( bound.update(warped) )            # the Phase-1 force, bound once to `fixed`
    δ      = _group_regularise(u, s, σ_fluid, σ_diffusion, step, rel_spacing, ndim)
    s      = compose_displacement(s, δ)                       # 1 gather  →  φ ← φ ∘ (id+δ)
    s      = _smooth_vector(s, σ_diffusion, ndim)             # total-field (diffusion) regulariser
    return s, bound.cost(warped)
```

- **2 gathers/iter** (warp + compose) vs algebra's **7** → ~3.5×.
- The compositive update `s ← compose_displacement(s, δ)` realises `φ_old ∘ (id+δ)`
  (`compose_displacement(outer=s, inner=δ) = (id+s)∘(id+δ)`) — the diffeomorphic
  (compositive) demons update `φ ← φ∘exp(u)`, with `(id+δ) ≈ exp(δ)` to first order.
  It agrees with the algebra update `v ← v+u` to first order in the step (BCH
  `exp(v+u) ≈ exp(v)∘exp(u)`), validated by the first-order-agreement test (§7).

**Symmetric (greedy SyN) — `group_symmetric_level`:**

```
per scan step, state (s_fwd, s_inv):
    a = spatial_transform(moving, id + s_fwd, mode)           # 1 gather  (moving → midpoint)
    b = spatial_transform(fixed,  id + s_inv, mode)           # 1 gather  (fixed  → midpoint)
    u_fwd = mask( force.bind(b).update(a) ); u_inv = mask( force.bind(a).update(b) )
    s_fwd = compose_displacement(s_fwd, _group_regularise(u_fwd, s_fwd, ...))   # 1 gather
    s_inv = compose_displacement(s_inv, _group_regularise(u_inv, s_inv, ...))   # 1 gather
    s_fwd, s_inv = _smooth_vector(s_fwd, σ_diff), _smooth_vector(s_inv, σ_diff)
    return (s_fwd, s_inv), force.bind(b).cost(a)
```

- **4 gathers/iter** (2 warps + 2 composes) vs algebra's **12** → 3×.
- **No per-iteration inversion.** The two half-warps are driven toward the midpoint
  by the symmetric force, exactly as the current algebra SyN drives `v_fwd`/`v_inv`;
  inverse-consistency is realised **once at finalisation** (`compose(s_fwd,
  invert(s_inv))`), not per step. (This corrects the plan's §2.5 over-cost.)

**`_group_regularise(u, s, …)`** — the group analogue of `_svf._regularise`:

```
δ = _normalise_step(u, step)                # RMS-clamped trust region (0d)
δ = _smooth_vector(δ, σ_fluid, ndim)        # fluid (update) smoothing — the regulariser Green's fn
δ = _step_clamp_diffeo(δ, ndim)             # §4.4 per-step diffeomorphism guard
return δ                                     # caller composes δ onto s; diffusion smooth of total s is separate
```

Note the regulariser **split is retained** (the review's D3): fluid smooths the
*increment* `δ`; diffusion smooths the *total* `s` after composition (in the level
fn above). Setting `σ_diffusion = 0` recovers ANTs `SyN[grad, σ_fluid, 0]`
(total-field off); the current recipe default `σ_diffusion = 1.5` is preserved.

### 4.4 `_step_clamp_diffeo` — the per-step diffeomorphism guard

Each increment must keep `id + δ` a diffeomorphism: `det(I + ∇δ) > 0` everywhere
(a clamp on the *Jacobian*, not just `‖δ‖` — a magnitude clamp can still fold).

```
_step_clamp_diffeo(δ, ndim, *, det_floor=0.1, max_halvings=3):
    for _ in range(max_halvings):                          # static, tiny (Python loop)
        d = jacobian_det_displacement(δ)                   # det(I+∇δ), closed-form for ndim≤3
        if jnp.min(d) > det_floor: break                   # (as lax.cond / jnp.where, jit-safe)
        δ = δ * 0.5
    return δ
```

- Reuses `geometry.jacobian_det_displacement` (closed-form Sarrus for `ndim ≤ 3`).
  Cost: `ndim` central-diff passes + the det — cheap vs the warps; the halving
  rarely fires when `_normalise_step` + fluid smoothing are reasonable.
- Mirrors the 0b IC **backtracking** pattern (bounded step-halving, no per-step
  re-solve) for consistency across the suite.
- The *total*-field `det J > 0` QA (the result's `jacobian_det`) is necessary but
  not sufficient (an intermediate compose can fold while the total stays positive),
  so the guard is **per step** — this is the §2.3-plan obligation made concrete.

## 5. Recipe wiring & finalisation

### 5.1 Spec fields

```python
@dataclass(frozen=True)
class DemonsSpec / SyNSpec:
    ...
    representation: Literal['group', 'algebra'] = 'group'   # NEW; group is the v4 default
    convergence: Optional[Convergence] = None               # NEW (A3); single-pair early-exit
    # n_steps reconciled (C5): one documented convention; integrate_velocity_field
    # finalisation and the algebra-mode warp use the SAME n_steps as the spec.
```

`representation='group'` is the default for **both** recipes (the v4 perf/CPU-timeout
win); `'algebra'` is the opt-in exact-SVF / oracle / geodesic-base path. Naming: in
group mode the "log-domain" of *log-Demons* is the recoverable velocity via
`field_log`; `representation='algebra'` is the literal log-domain solve.

### 5.2 Dispatch & finalisation (per recipe)

Each recipe builds the pyramids / forces / masks as today (unchanged), then branches:

```
if spec.representation == 'algebra':
    level_solve = <existing algebra closure>     # single_sided_level / symmetric_level
    state, hist = svf_coarse_to_fine(..., level_solve=level_solve)
    # finalise as today:
    Demons:  residual = integrate_velocity_field(v, n_steps); velocity = v
    SyN:     s_fwd, s_inv = exp(v_fwd), exp(v_inv); residual = compose(s_fwd, invert(s_inv))
             forward_velocity, inverse_velocity = v_fwd, v_inv
else:  # 'group' (default)
    level_solve = <new group closure>            # group_single_sided_level / group_symmetric_level
    state, hist = svf_coarse_to_fine(..., level_solve=level_solve)   # state carries displacements
    # finalise from the group state directly (no exp):
    Demons:  residual = s; velocity = field_log(s)                    # velocity recovered on demand
    SyN:     residual = compose(s_fwd, invert(s_inv))                 # ONE inversion (0e), as algebra
             forward_velocity, inverse_velocity = field_log(s_fwd), field_log(s_inv)
```

`finalize_with_init` (compose residual with the warm-start init, warp original
`moving`, `jacobian_det`) is **unchanged** and shared by both modes. The result
`NamedTuple`s keep their fields; only *how* `velocity` is produced differs.

### 5.3 A3 early-exit (folded here)

A `while_loop` variant of each level fn (group and algebra) using the shared
windowed-slope criterion (hoisted to `register/_converge.py`, Phase 6 C7 — landed
here as the shared helper since A3 is its first SVF consumer), gated by
`spec.convergence`. Single-pair only (breaks `vmap`), matching the matrix IC path;
the fixed `lax.scan` stays the default. Re-derive clamp-vs-scale (a convergence gate
makes ANTS scale-to-step viable) at this site.

## 6. Concrete new/changed surface (signatures)

```python
# geometry/deformation.py  (+ export field_log from geometry/__init__.py)
def field_log(displacement, *, n_sqrt=6, sqrt_tol=1e-6, sqrt_max_iter=50,
              mode='nearest') -> Float[Array, '*spatial ndim']: ...
def _diffeo_sqrt(s, *, tol, max_iter, mode) -> Float[Array, '*spatial ndim']: ...   # damped fixed point

# register/_svf.py  (new; mirror the existing level-fn signatures)
def group_single_sided_level(moving, fixed, s, *, force, ndim, iterations,
        boundary_mode, sigma_fluid, sigma_diffusion, step, rel_spacing,
        mask=None) -> tuple[Array, Array]: ...                      # (s, costs)
def group_symmetric_level(moving, fixed, s_fwd, s_inv, *, force, ndim, iterations,
        boundary_mode, sigma_fluid, sigma_diffusion, step, rel_spacing,
        mask=None) -> tuple[Array, Array, Array]: ...               # (s_fwd, s_inv, costs)
def _group_regularise(u, s, *, step, sigma_fluid, sigma_diffusion, ndim) -> Array: ...
def _step_clamp_diffeo(delta, ndim, *, det_floor=0.1, max_halvings=3) -> Array: ...
```

Note the group level fns drop the algebra-only `n_steps` and `bch_order` arguments
(no integration, no log-domain BCH) — the signature delta itself documents that
these are a different driver, not a re-parametrised one.

## 7. Test plan & parity oracles

Named explicitly, because greedy has **no** exact cross-driver oracle:

1. **Algebra byte/ULP oracle.** `register(..., representation='algebra') ==`
   current shipped output (Demons + SyN). Anchors correctness exactly.
2. **`field_log` round-trip (exact on SVF).** Random smooth `v`:
   `field_log(integrate_velocity_field(v, n)) ≈ v` and the displacement round-trip
   `integrate_velocity_field(field_log(s), n) ≈ s` for `s = exp(v)` — to integration
   tolerance (`n_sqrt = n`).
3. **`field_log` differentiability.** `jax.grad` through `field_log` finite and
   FD-matched (the `fixed_point_solve` IFT path).
4. **`_diffeo_sqrt` correctness + convergence.** `compose_displacement(w, w) ≈ s`
   residual below `sqrt_tol`; converges from `w0 = s/2` across small→moderate `s`;
   stress a near-`‖∇s‖=1` field and assert the residual is reported (not silently
   wrong).
5. **First-order agreement (compose direction).** One `group` step vs one `algebra`
   step from the same state agree to `O(ε²)` — validates the compositive update
   direction/order.
6. **Recovery gate (the main cross-driver test).** Group-mode SyN/Demons recover a
   synthetic ground-truth warp to NCC tolerance, within `Δ` of algebra mode.
   Explicitly a *recovery* gate, **not** field-wise equality (greedy ≠ SVF fixed
   point) — the named discipline exception.
7. **Per-step + total diffeomorphism.** `det J > 0` on the total `φ`; and an
   adversarial sharp-force case where the magnitude clamp alone would fold but
   `_step_clamp_diffeo` keeps every step `det > det_floor` (no intermediate fold).
8. **Inversion residual (depends on 0e).** Large-deformation SyN:
   `(id+s_fwd)∘(id+s_inv)⁻¹` assembly with the inversion residual asserted bounded.
9. **Velocity recovery feeds the barycentre.** `field_log` of a group result runs
   through `velocity_mean` without error; `exp(field_log(φ)) ≈ φ` to the documented
   projection residual.
10. **Scaling case.** Group SyN/Demons at 256³ single + a cohort tier: gathers/iter,
    wall-clock, HBM-per-voxel; certify the per-iteration gather reduction and the
    ≤-linear target (the latter **gated on the Phase-0 roofline** confirming §0.4's
    bandwidth hypothesis, per the plan).

## 8. Dependencies & sequencing

- **Hard predecessor: 0e** (`invert_displacement` robustness + residual). Consumed
  twice: the final SyN inversion *and* `_diffeo_sqrt`'s large-deformation regime
  (reuse the Anderson/Newton acceleration on `fixed_point_solve`). Land 0e first.
- **Consumes Phase 1 forces** unchanged — the group driver calls `force.bind` /
  `BoundForce.update` / `.cost` exactly as the algebra driver does (the `Force`
  protocol is untouched). 0d (`_normalise_step` RMS) is used by `_group_regularise`.
- **Independent of** Phase 3 (matrix) and Phase 4 (preprocessing).
- **Predecessor of** Phase 6 C4 (the SVF-preamble refactor must factor the *final*
  two-driver shape) and Phase 5a/5b (the fused force/stencil kernels feed the group
  driver's surviving per-iteration ops).

## 9. Risks & mitigations

| Risk | Mitigation |
|---|---|
| `field_log` slow/under-converged on a strongly-deforming greedy `φ` | 0e acceleration; report the residual; velocity is a *secondary* output (the displacement never depends on it) |
| Group ≠ algebra fixed point misread as a bug | Named recovery-tolerance gate (#6), not equality; documented in the recipe + `field_log` docstrings |
| Per-step det-guard backtracking adds variable cost | Bounded (`max_halvings=3`); fires rarely under a sane `_normalise_step`; cheap `jacobian_det_displacement` |
| Compose direction/sign wrong | First-order-agreement test (#5) + the recovery gate (#6) catch it |
| `representation='group'` default surprises a velocity-output consumer | Velocity still returned (via `field_log`); the projection residual documented; `representation='algebra'` restores exact-velocity |

## 10. Corrected perf accounting (supersedes plan §2.5)

| Recipe | algebra gathers/iter | group gathers/iter | reduction | per-iteration inversion? |
|---|---|---|---|---|
| Demons (single-sided) | 7 (`exp` n_steps=6 + warp) | **2** (warp + compose) | ~3.5× | none |
| SyN (symmetric) | 12 (2× `exp` n_steps=5 + 2 warps) | **4** (2 warps + 2 composes) | 3× | none |

`field_log` adds a **one-time** finalisation cost (~120 gathers), negligible vs the
hundreds of optimiser iterations. The gather reduction is the per-iteration headline;
the *total* speedup also depends on the surviving Gaussian-regulariser and force
costs, which Phase 1c (LNCC integral image) and Phase 1d (recursive Gaussian) attack
— so Phase 2 is necessary but not alone sufficient for the ≤-linear-at-256³ target.

## 11. Cross-references

- [`registration-suite-v4.md`](registration-suite-v4.md) §2 (the plan this details),
  §5 (sequencing), §6 (economic verdict).
- `src/nitrix/geometry/{deformation,grid,pyramid}.py` (`compose_displacement`,
  `invert_displacement`, `integrate_velocity_field`, `jacobian_det_displacement`,
  `upsample`); `src/nitrix/numerics/fixed_point.py` (the IFT solver `field_log` reuses);
  `src/nitrix/geometry/algebra.py` (`velocity_mean` — the `field_log` consumer;
  `matrix_log` — the structural template); `src/nitrix/register/{_svf,diffeomorphic,_syn}.py`.
- `docs/feature-requests/registration-suite-v3-followups.md` (A3 early-exit, B5
  inversion robustness, C4/C7 refactors).
