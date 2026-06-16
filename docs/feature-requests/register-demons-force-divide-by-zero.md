# Demons ESM force **0/0 → NaN** on uniform regions (real images always NaN)

> **RESOLVED 2026-06-12 (v4 `63d69e7`), regression-tested 2026-06-16.** Guarded
> the ESM denominator with a gradient-safe double-``where`` (`_DEMONS_DENOM_EPS
> = 1e-8`): force is zeroed where ``|j|² + α²·diff² ≈ 0`` (no gradient *and* no
> mismatch -> no information), keeping both the forward warp and its velocity
> finite on a matched uniform region.  Regression guard:
> ``tests/test_demons_recipe.py::test_demons_finite_on_uniform_background``
> (exact-zero background, 1 iteration -> finite warp + velocity).  Sibling
> paths checked: ``lncc_grad`` (and so ``LNCCForce``) divides by ``var_m·var_f
> + eps`` so a uniform window yields a finite zero force; ``MetricForce`` relies
> on each metric's own ``eps`` floor.

> **Status (2026-06-11): correctness bug, perf-bench finding.** Surfaced
> building real-anatomy registration benchmarks (MNI152 T1): the diffeomorphic
> demons recipe returns an **all-NaN** warp on *any* image with a uniform
> background — i.e. every real scan. Reproduces at **one iteration**, nilearn-
> free, on a blob over a zero background.

## The bug

`DemonsForce.update` (`src/nitrix/register/_force.py:253–257`) computes the ESM
force with an **unguarded** denominator:

```python
diff  = self.fixed - warped
grad  = spatial_gradient(warped, spacing=_grad_spacing(self.rel_spacing))
j     = 0.5 * (self.grad_fixed + grad)
denom = jnp.sum(j * j, axis=-1) + (self.alpha**2) * diff * diff   # line 256
return _to_voxel((diff / denom)[..., None] * j, self.rel_spacing)  # line 257
```

In a **uniform region where the images already match** — background voxels where
`F == M∘φ` (so `diff == 0`) *and* there is no gradient (so `j == 0`) — the
denominator is `0 + 0 = 0` and the numerator `diff` is `0`, so `diff / denom`
is **`0 / 0 = NaN`**. The `NaN` then propagates (`NaN * j`, the Gaussian smooth,
the SVF integration) and the **entire** velocity field / warp becomes `NaN`.

Real medical images are mostly uniform background (air/skull-stripped zero), so
this fires on essentially every real volume; the synthetic gaussian-noise test
inputs have no uniform region, which is why it was not caught.

## Minimal repro (nilearn-free, 1 iteration)

```python
import numpy as np, jax.numpy as jnp, scipy.ndimage as spnd
from nitrix.register import diffeomorphic_demons_register, DemonsSpec

def blob(shift):
    v = np.zeros((32, 32, 32), np.float32); v[12:20, 12:20, 12:20] = 1.0
    return np.roll(spnd.gaussian_filter(v, 1.5), shift, axis=0)

fixed, moving = blob(0), blob(2)            # a blob on a UNIFORM zero background
r = diffeomorphic_demons_register(jnp.asarray(moving), jnp.asarray(fixed),
                                  spec=DemonsSpec(levels=1, iterations=1))
print(np.isfinite(np.asarray(r.warped)).all())   # False  -- all NaN
# Adding a tiny noise floor (breaking the uniform region) -> True.
```

## Fix

Guard the division so a `0/0` region yields **zero force** (which is the
correct demons update there — no gradient and no intensity mismatch means no
information):

```python
# epsilon floor:
denom = jnp.sum(j * j, axis=-1) + (self.alpha**2) * diff * diff + eps
# or an explicit guard (force = 0 where denom is ~0):
scale = jnp.where(denom > eps, diff / denom, 0.0)
return _to_voxel(scale[..., None] * j, self.rel_spacing)
```

This is the standard demons treatment (Vercauteren et al. zero the update where
`|J|² + (F−M)²` is below a threshold). `MetricForce` (the `jax.grad`-of-cost
path) and the LNCC force should be checked for the same uniform-region 0/0.

## perf-bench handling

The real-anatomy harness adds a small **background noise floor** to its inputs
(realistic — real scans have acquisition noise — and it sidesteps the 0/0), so
demons benches on real anatomy meanwhile; the synthetic cases are unaffected
(no uniform region). This doc tracks the underlying op fix.

## Cross-references

- `src/nitrix/register/_force.py:253–257` (`DemonsForce.update`); the formula
  doc at `diffeomorphic.py:13–14, 76`.
- [`perf-bench-feedback.md`](perf-bench-feedback.md) — the perf-bench ledger.
