# A public differentiable-registration layer (implicit-diff, self-contained matrix)

> **Status (2026-06-25): request (nimox-estimators E3 → nitrix).** nitrix
> already ships the pieces for a differentiable registration *layer* — the
> implicit-function entry points `linalg.implicit_least_squares` /
> `implicit_minimize` (IFT-exact, O(1) memory, the path `register/_converge.py`
> steers gradient users to), the `TransformModel.exp` charts, and
> `geometry.affine_grid` / `spatial_transform`. What is **not** public is a
> registration entry point that *composes* them and returns a transform in the
> **centre-baked-in matrix convention** `apply_transform` consumes. nimox's
> `AffineRegister(gradient='implicit')` (`nimox.estimators`) had to hand-roll
> that composition and replicate the centring convention. This asks nitrix to
> own it, so the differentiable layer is a nitrix call.

## What nimox currently hand-rolls

To make registration differentiable through `fit` *correctly* (implicit, not by
unrolling the coarse-to-fine recipe), nimox builds, in `registration.py`:

```python
def warp(mv, theta):                       # affine_exp is about-origin
    return spatial_transform(mv[..., None],
        affine_grid(model.exp(theta, ndim=ndim), fixed.shape, center=center),
        method=Linear())[..., 0]
def residual(data, theta):
    mv, fx = data; return (warp(mv, theta) - fx).reshape(-1)
theta_star = implicit_least_squares(residual, (moving, fixed), theta0, n_iters=...)
matrix = _self_contained(model.exp(theta_star, ndim=ndim), center, ndim)  # <-- convention bridge
```

The `_self_contained` step conjugates the about-origin `affine_exp(θ*)` matrix
to the centre-baked-in form `apply_transform` applies with `center=0` — i.e.
nimox is **replicating a nitrix-internal convention** (`_conjugate_about` in
`register/_space.py`) to make the implicit result interoperate with the recipe
result. That convention knowledge belongs in nitrix.

## The ask

A public differentiable registration that returns a self-contained
`RegistrationResult` (same `.matrix` convention as `rigid_register` /
`affine_register`), IFT-differentiable w.r.t. the images. Either shape works:

- a sibling entry point — `affine_register_implicit(moving, fixed, *, model,
  metric=SSD(), n_iters=...) -> RegistrationResult`; or
- a mode on the existing recipes — `affine_register(..., differentiable=True)`
  that routes the single-level solve through the implicit path instead of the
  coarse-to-fine scan.

It should:

1. **Return the centre-baked-in matrix** (the conjugation lives here, not in the
   consumer), so `apply_transform(image, result)` just works.
2. **Differentiate at the optimum** via `implicit_least_squares` (SSD) — and,
   ideally, expose the **general-metric** `implicit_minimize` path too, so a
   differentiable LNCC/MI registration layer is available (nimox cannot reach
   that today without the same hand-rolled objective).
3. Carry `cost_history` (the final cost) like the recipes, for the fitted-object
   diagnostics nimox surfaces.

## Why (and why now)

- **Convention ownership.** The centring conjugation is nitrix's convention; a
  consumer replicating it is fragile (a future `apply_transform` change silently
  breaks nimox's implicit path).
- **Reach.** nimox's hand-roll is SSD-only (the `implicit_least_squares`
  residual). The `implicit_minimize` general-metric path — a differentiable
  LNCC/MI registration layer — needs the objective built against nitrix's warp +
  metric, which nitrix is positioned to do correctly and nimox is not.
- **It is the §9 seam resolution.** nimox's estimators RFC resolves "is classical
  registration a nimox or ilex concern" as *fully nimox-side, with the
  differentiable layer delegated to nitrix's implicit-diff abstraction.* This FR
  is the nitrix half of that delegation.

## Acceptance

- A public differentiable registration returning a self-contained
  `RegistrationResult`; synthetic-warp **recovery** parity with the recipe path
  and a finite **IFT-exact gradient** (vs finite-difference) under test.
- nimox `AffineRegister(gradient='implicit')` / `RigidRegister(...)` delegate to
  it and **delete** `_self_contained` + the hand-rolled residual; the E3 tests
  (recovery + grad-through-fit) stay green.
- Optional stretch: the `implicit_minimize` general-metric variant, with a
  differentiable-LNCC recovery test.

## Cross-references

- nimox `docs/feature-requests/nimox-estimators.md` §9 (the resolved ilex seam)
  + §13 Q6 (the origin) + §2 (the "grad-through-fit" row this realises).
- `nitrix.linalg.implicit_least_squares` / `implicit_minimize`
  (`linalg/optimize.py`) — the IFT entry points to wrap.
- `register/_space.py` `_conjugate_about` / `register/recipes.py`
  `apply_transform` — the matrix convention to own.
- [`affine-matrix-algebra`](affine-matrix-algebra.md) — the sibling
  nimox-extraction matrix-convention request (same `register`/`geometry` home).
