# `geometry.field_log` — stationary-velocity logarithm of a deformation (design)

> **Status (2026-06-12): implementation-ready design, no code yet.** The deep
> numerical spec for the dense **field logarithm** introduced in
> [`registration-suite-v4-phase2.md`](registration-suite-v4-phase2.md) §3 — the
> group→algebra bridge that lets a greedy (group-solved) deformation recover its
> stationary velocity. A reusable `geometry` primitive, the dense analogue of
> `linalg.matrix_log`; the Phase-2 driver is its first consumer but it stands
> alone (template/barycentre construction, motion analysis, any SVF parameterisation
> of a given warp).

## 1. Problem & contract

Given a displacement field `s` (the diffeomorphism `φ = id + s`), return a
stationary velocity field `v` with `exp(v) = φ`, where `exp = integrate_velocity_field`
(scaling-and-squaring). `field_log` is the inverse of `integrate_velocity_field`,
computed by **inverse scaling-and-squaring**.

Two distinct accuracy notions — keep them separate, they drive the design:

- **Round-trip fidelity** — `integrate_velocity_field(field_log(s)) == φ`. The design
  makes this **exact** (to the square-root solver tolerance), unconditionally. This is
  what most consumers need: a velocity that *reproduces the given warp*.
- **Generating-velocity fidelity** — if `φ = exp(v_true)`, how close is the returned
  `v` to `v_true`. This is the **log-approximation error** (§3), `O(‖v‖²/2^{n_sqrt})`,
  and is *secondary* (matters only when the velocity is interpreted as the true
  generator, e.g. geodesic/momentum analysis).

These differ because `exp` is **not surjective**: a general (e.g. greedy-composed)
`φ` is not `exp` of *any* stationary velocity, so `field_log` returns the **best SVF
fit** — exact round-trip by construction, with a reported **fit residual** off the
SVF submanifold.

## 2. Algorithm — inverse scaling-and-squaring

`exp` does: `φ₀ = id + v/2ⁿ`; square `n` times (`φ_{i+1} = φ_i ∘ φ_i`). Invert it:

```
v = field_log(s):
    w = s
    for _ in range(n_sqrt):          # static Python loop
        w = _diffeo_sqrt(w)          # w ← (id+w)^{1/2} − id   (one square root)
    return w * (2.0 ** n_sqrt)
```

After `k` roots `w = displacement(φ^{1/2^k})`; at `k = n_sqrt`, `v = 2^{n_sqrt}·w`.

**Why the round-trip is exact regardless of the log approximation:** `v/2^{n_sqrt} = w_{n_sqrt}`
*by definition*, and `id + w_{n_sqrt} = φ^{1/2ⁿ}` (each `_diffeo_sqrt` is solved to
tolerance), so squaring it `2ⁿ` times reconstructs `φ` exactly. The only
approximation is in reading `v` as `2ⁿ·w_n` rather than the true generator (§3).

### 2.1 `_diffeo_sqrt` — square root of a diffeomorphism (damped fixed point)

We need `w` with `compose_displacement(w, w) = s` (i.e. `(id+w)∘(id+w) = id+s`).

**The naïve fixed point fails.** `w = s − warp(w, id+w)` linearises near identity to
`w ← s − w` (Jacobian `−I`) — it *oscillates*, never converges. Use the **½-damped**
iteration:

```
w* = _diffeo_sqrt(s):   fixed point of   g(s, w) = w + ½·(s − compose_displacement(w, w))
                        init  w₀ = s/2,  via numerics.fixed_point_solve
```

**Convergence.** Let `C(w) = compose_displacement(w, w) = w + warp(w, id+w)`. Near
identity `C(w) ≈ 2w`, so `∂g/∂w = I − ½·∂C/∂w ≈ I − ½·2I = 0` — locally super-linear;
from `w₀ = s/2`, `g(s, s/2) ≈ s/2` converges in ~one step for small `s`.
**Measured (implementation):** plain Picard on this damped iteration reaches a
square-root residual `~1e-10` in **under ten steps** even for a large deformation
(`max‖∇φ‖ ≈ 0.2` after a sizeable velocity) — so **no acceleration is needed**; the
damping *is* the accelerator. Anderson was tried and is marginally *worse* here (its
least-squares over the near-collinear, fast-converging residual history is noisier),
so `field_log` uses plain Picard. The Anderson opt-in on `fixed_point_solve` remains
available for a *pathologically* stiff `‖∇s‖ → 1` field (it is validated to converge
where Picard stalls on a genuinely stiff fixed point), with a smaller damping
`α < ½` for the first root as a further fallback — but neither is on the default path.

**Differentiability.** `_diffeo_sqrt` is one `numerics.fixed_point_solve` (IFT-VJP,
O(1) memory in the iteration count); the `n_sqrt` Python loop composes them. So
`field_log` is fully differentiable, **GPU-native** (compose + fixed point, **no
`safe_inv`** — unlike `matrix_log` — so it is jit/grad-clean even on the wedged-cuSolver
dev box).

## 3. The log approximation (order, accuracy, the BCH option)

`v = 2ⁿ·w_n` is the **first-order** near-identity log: `log(id+w) ≈ w`. The flow of a
small velocity gives `displacement(exp(u)) = u + ½(u·∇)u + O(u³)`, so inverting,
`v = w − ½(w·∇)w + O(w³)` per square root. Hence

```
v = field_log(s) = v_true + O(‖v‖² / 2^{n_sqrt})      (generating-velocity error)
```

— each extra square root halves it. `n_sqrt = 6` → error `~‖v‖²/64`; for the few-voxel
velocities registration produces this is small, and **it does not affect the round-trip**
(§2). Choose `n_sqrt = 6` (matches `integrate_velocity_field`'s `n_steps` default and
`matrix_log`'s `n_sqrt`, so `exp∘log` and `log∘exp` are consistent at the default).

**Optional higher-order (off by default).** A one-term BCH correction
`v ≈ 2ⁿ(w_n − ½(w_n·∇)w_n)` gives `O(‖v‖³/2^{2n})` *generating-velocity* accuracy with
fewer square roots — **but it breaks the exact round-trip** (then `v/2ⁿ ≠ w_n`). Since
the registration consumers want the velocity that *reproduces* `φ` (round-trip
fidelity), the **default is first-order** `v = 2ⁿw_n`. Expose the BCH correction only
behind an explicit flag for the rare generating-velocity use (geodesic/momentum), with
the round-trip/generator trade-off documented.

## 4. The fit residual (off the SVF submanifold)

For a general greedy `φ ∉ image(exp)`, `field_log` returns the best SVF fit; report

```
residual = ‖ integrate_velocity_field(v, n_steps=n_sqrt) − s ‖ / (‖s‖ + ε)
```

(zero to solver tolerance on the SVF submanifold; > 0 measuring the projection error
otherwise). Surface it on the recipe result / as an optional return, so a consumer
interpreting `v` as the true generator does so with eyes open — not a silent
approximation.

## 5. Signature & parameters

```python
def field_log(
    displacement: Float[Array, '*spatial ndim'],
    *,
    n_sqrt: int = 6,
    sqrt_tol: float = 1e-6,
    sqrt_max_iter: int = 50,
    correction: Literal['first_order', 'bch'] = 'first_order',
    mode: BoundaryMode = 'nearest',
) -> Float[Array, '*spatial ndim']:
    """Stationary-velocity logarithm of `id + displacement` by inverse
    scaling-and-squaring (the dense analogue of `linalg.matrix_log`)."""
```

- `mode='nearest'` (edge-replicate, the flow-field convention shared by
  `integrate_velocity_field` / `invert_displacement`).
- Home: `geometry/deformation.py` (beside `compose_displacement` / `invert_displacement`
  / and exported next to `integrate_velocity_field`). Export from `geometry/__init__.py`.

## 6. Cost

`n_sqrt` square roots × (~10–30 damped iters, the first root dominating) × 1
`compose_displacement` gather ≈ **~120 gathers**, paid **once** (e.g. at recipe
finalisation), negligible vs an optimiser that ran hundreds of iterations. No
`safe_inv`, no factorisation.

## 7. Relation to alternatives

- **`linalg.matrix_log`** — the structural sibling (inverse scaling-and-squaring with
  square roots). `matrix_log` uses Denman–Beavers + `safe_inv`; `field_log`'s square
  root is the *diffeomorphism* sqrt (a fixed point, no solve), so it stays GPU-native.
  `field_log` is to `integrate_velocity_field` exactly as `matrix_log` is to
  `matrix_exp`.
- **BCH-based log-domain demons** (Vercauteren 2009) computes the velocity update via
  the BCH series rather than an explicit field log. We use inverse scaling-and-squaring
  instead because it is the **exact inverse of our forward integrator**, giving the
  exact round-trip (§2) the group-driver recovery needs; BCH would not round-trip
  exactly against `integrate_velocity_field`.

## 8. Test plan

1. **Round-trip exact (SVF submanifold).** Random smooth `v`:
   `field_log(integrate_velocity_field(v, n)) ≈ v` and
   `integrate_velocity_field(field_log(s), n) ≈ id+s` for `s = exp(v)`, `n_sqrt = n`,
   to integration tolerance.
2. **`_diffeo_sqrt` correctness.** `compose_displacement(w, w) ≈ s` residual `< sqrt_tol`
   in well under ten Picard steps (the damping is super-linear); converges from
   `w₀ = s/2` across small→large `s`. A *pathological* near-`‖∇s‖=1` field falls back to
   the opt-in acceleration or **reports** the residual (no silent wrong answer).
2b. **Naïve-fixed-point regression guard.** A unit test asserting the *damped* form
   converges where the un-damped `w = s − warp(w,id+w)` does not — locks the damping in.
3. **Log-approximation accuracy law.** Recovered-vs-true generating velocity error
   shrinks `~‖v‖²/2^{n_sqrt}` (halves per extra root); the `bch` option improves it but
   degrades the round-trip — assert both directions of the trade-off.
4. **Differentiability.** `jax.grad`/`jax.jacfwd` through `field_log` finite and FD-matched.
5. **Fit residual off-manifold.** A deliberately non-SVF (greedy-composed) `φ`: the
   reported residual is > 0 and the round-trip recovers `φ` to tolerance despite `v ≠ v_true`.
6. **Consumer.** `field_log` output flows through `geometry.velocity_mean` /
   `transform`-algebra without error (the template-construction path).
7. **Anisotropy / boundary.** `mode='nearest'` parity with `integrate_velocity_field`'s
   boundary; behaviour on a 2-D and 3-D field.

## 9. Risks

| Risk | Mitigation |
|---|---|
| First square root slow / non-convergent at a *pathological* `‖∇s‖→1` | the damped Picard is super-linear and suffices for the realistic regime (measured `~1e-10` in <10 steps); the opt-in Anderson + a smaller first-root `α` are the fallbacks; report the residual; the *displacement* output never depends on `field_log` |
| Round-trip vs generator confusion | The two are documented separately (§1, §3); default first-order = exact round-trip; BCH behind a flag with the trade-off stated |
| `n_sqrt` too small for a large warp | the accuracy law (#3) sets it; default 6 matches `exp`/`matrix_log` |

## 10. Cross-references

- [`registration-suite-v4-phase2.md`](registration-suite-v4-phase2.md) §3 (the driver
  consumer), [`registration-suite-v4.md`](registration-suite-v4.md) §2.2.
- `src/nitrix/geometry/{deformation,grid}.py` (`compose_displacement`,
  `invert_displacement`, `integrate_velocity_field`), `numerics/fixed_point.py` (the IFT
  solver), `geometry/algebra.py` (`velocity_mean` consumer; `matrix_log` structural
  template), `linalg/matrix_function.py` (`matrix_log` / `matrix_exp`).
