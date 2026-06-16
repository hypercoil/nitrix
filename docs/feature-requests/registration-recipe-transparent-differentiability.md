# Transparently-differentiable registration recipes (implicit-VJP wrapper)

> **Status: proposed (filed 2026-06-12, during v4 3b).** A follow-up to the 3b
> differentiability contract. Today a `jax.grad` through a recipe's *early-exit*
> forward (the `lax.while_loop`, the single-pair default) raises a loud,
> actionable error (`register/_converge._early_exit_barrier`) pointing at the
> implicit-function entry points. This proposes making the recipes themselves
> *transparently* reverse-differentiable so `jax.grad(rigid_register)` "just
> works" -- regardless of early-exit -- via the implicit adjoint at the optimum,
> dropping the barrier.

## Motivation

The barrier (option A, shipped in 3b) is correct but forces a context switch: a
differentiable-layer consumer must rewrite their registration through
`linalg.implicit_least_squares` / `implicit_minimize`, or opt out of early-exit
with `convergence=None`. Option B is the ergonomic alternative — the recipe is
differentiable as-is, and the gradient is the **at-convergence implicit
gradient** (the GN-Hessian adjoint), which is the *correct* gradient for a
converged registration and is trajectory-independent (so it is unaffected by the
early-exit trip count, exactly as `numerics.fixed_point_solve`'s `custom_vjp`
makes `invert_displacement` / `field_log` differentiable despite their forward
`while_loop`).

## Design

Wrap **only the matrix-producing solve** `(_solve: (moving, fixed[, init]) ->
matrix*)` in a `jax.custom_vjp` (or `lax.custom_root`); the finalisation
(`warped = warp(moving, matrix*)`, params, jacobian) stays ordinary autodiff, so
`warped`'s gradient flows through both `matrix*` (implicit) and `moving` (direct):

- **forward:** the existing coarse-to-fine optimiser (IC early-exit included) ->
  `matrix*`; save `(matrix*, moving, fixed)` (+ the finest reference level).
- **backward:** the implicit-LSQ adjoint at the finest level. Residual
  `r(θ) = warp(moving, model.exp(θ_about_matrix*)) − fixed`; the cotangent on
  `matrix*` maps to a cotangent on `θ` (via `matrix = exp(θ)`); solve the GN
  normal equations `(JᵀJ) w = θ̄` (matrix-free CG, the `implicit_least_squares`
  machinery); push `w` back through `∂r/∂(moving, fixed)` for both image grads.

## Why it is not a thin wrapper (the real work)

1. **Cannot reuse `implicit_least_squares` directly** — its *forward* is its own
   solver, not the recipe's coarse-to-fine early-exit IC. Factor its adjoint
   (`(JᵀJ)⁻¹` CG solve + the `∂r/∂data` push) into a reusable
   `implicit_lstsq_adjoint(residual_fn, params*, data, cotangent)` that B's
   `custom_vjp.bwd` calls, while B's `fwd` keeps the recipe's solver.
2. **The forward Hessian is the wrong one.** The IC path stores the *constant
   template* (Baker–Matthews) Hessian; the adjoint needs the *forward-linearised*
   GN Hessian of the cost at `θ*`. The backward must build it from the residual
   fn (autodiff/CG), not reuse the stored template `H⁻¹`.
3. **Parameterisation + multilevel + both images.** Adjoint at the finest level
   only (coarse levels are just initialisation); matrix↔θ cotangent conversion;
   gradients to `moving` *and* `fixed`.
4. **Coverage.** Rigid + affine are moderate. The **SVF recipes** (Demons / SyN)
   are a *field-valued* argmin — a harder implicit problem (`implicit_minimize`
   over a velocity/displacement field, with the regulariser in the objective);
   the at-convergence optimality is the regularised-similarity stationarity, not
   a small dense Hessian. This is a separate, larger sub-task (the barrier covers
   it uniformly for free; B does not).

## Test plan

- `jax.grad(rigid_register)` / `affine_register` (default early-exit) is finite,
  FD-matched, and equals `jax.grad` of the `convergence=None` (unrolled scan)
  recipe **at convergence** (the existing implicit-vs-unrolled oracle in
  `test_registration_r3`).
- Gradient flows to both `moving` and `fixed`.
- The barrier is removed only once B covers a path; until then, an uncovered path
  (e.g. SVF) keeps the barrier (no silent break).

## Scope decision (2026-06-12)

Filed rather than built inline: B is a focused feature with correctness
subtleties (the adjoint-Hessian distinction, multilevel, field-valued SVF), not a
tweak. 3b ships **A** (the barrier + `'auto'` early-exit default + C3/C6); B
supersedes the barrier for the matrix recipes when implemented.

## Cross-references

- `src/nitrix/register/_converge.py` (`_early_exit_barrier`), `_inverse_compositional.py`
  (the IC solve + stored template Hessian), `recipes.py` (the recipe boundary).
- `src/nitrix/linalg/` (`implicit_least_squares` / `implicit_minimize` — the adjoint
  machinery to factor), `src/nitrix/numerics/fixed_point.py` (the IFT-`custom_vjp`
  template). `docs/design/registration-suite-v4.md` §Phase 3 (3b).
