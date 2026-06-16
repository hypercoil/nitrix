# `affine_register` multi-level GN/LM **diverges at small grids** (v3 regression)

> **RESOLVED 2026-06-12 (v4 `63d69e7`), regression-tested 2026-06-16.** Two
> perf-neutral fixes: (1) a geometric trust region in the IC loop
> (``_inverse_compositional._trust_scale``) clamps only a step whose induced
> grid displacement exceeds the grid extent (a normal step is byte-unchanged, so
> the well-conditioned path keeps single-step Gauss-Newton convergence); (2) a
> loud affine-only pyramid-depth cap (``recipes._cap_levels`` +
> ``AffinePyramidDepthWarning``) shortens the pyramid so the coarsest level stays
> ``>= 16`` vox/axis, where the 12-DOF affine Hessian is reliable (rigid is
> untouched).  Recovery restored (24³/28³ +0.88/+0.86, was +0.18/-0.04).
> Regression guard:
> ``tests/test_register_recipes.py::test_affine_small_grid_stays_bounded`` (28³,
> default pyramid -> warns, params bounded, ncc recovers).  **perf-bench can now
> un-``xfail`` its 28³ affine recovery test.**

> **Status (2026-06-11): v3 regression, perf-bench finding.** Surfaced
> re-benching the registration suite against `registration-suite-v3`
> (`356c768`). The same `nitrix-perf-bench` affine recovery test passed against
> the pre-v3 pin (`c865f67`); v3 introduced the divergence. Narrow (small grids
> only) but the params *blow up*, so it is flagged for the v3 loop. **Not** a
> realistic-size problem (≥32³ is fine).

## The bug

`affine_register` with the **default** optimiser (`'auto'` → GN/LM) and a
**multi-level** pyramid **diverges** when the grid is small enough that the
coarse pyramid level is tiny (≲14³). The affine parameters explode (rotation
generators → ~20) and the warped output ends up *anti*-correlated with the
fixed image. Measured recovery (`warp_pair` smoothed-noise, a ~4°/few-voxel
rigid+affine warp, `RegistrationSpec(levels=2, iterations=15)`):

| grid | ncc before → after | |
|---|---|---|
| 24³ | 0.790 → **+0.179** | diverge |
| 28³ | 0.785 → **−0.034** | diverge (params explode at L3×30 → `[19.8, −6.6, 3.9, …]`) |
| **32³** | 0.770 → **+0.881** | ok |
| 48³ | 0.718 → +0.923 | ok |
| 64³ | 0.665 → +0.941 | ok |
| MNI152 99³ (real T1) | 0.940 → +0.990 | ok |

The threshold sits between 28³ and 32³ — i.e. when the coarsest level (the
grid ÷ `pyramid_factor^(levels−1)`) falls to ≈14³ or below.

## It is isolated to the affine GN/LM path (V4b/V4c)

Same input, same `(levels=2, iterations=15)`, 28³:

- **`rigid_register`** → ncc **+0.939** (fine). Rigid uses the V4a
  inverse-compositional fast path; affine uses the **V4b** affine IC fast path
  + **V4c** closed-form forward-warp Jacobian.
- affine **single-level** (`levels=1, iterations=30`) → **+0.859** (fine).
- affine **`optimizer='bfgs'`** → **+0.949** (fine) — BFGS does not use the
  closed-form Jacobian.
- affine default GN / LM (`'gn'` / `'lm'`) → **−0.034** (diverges).

So the instability is in the **affine multi-level GN/LM step at small coarse
grids**, not in rigid (whose IC path is fine) and not in the BFGS path. The
candidate is the **V4c closed-form forward-warp Jacobian** (or the V4b affine IC
update) being ill-conditioned / sign-unstable when the coarse-level grid is
very small, with the across-level affine parameter rescale compounding it into a
blow-up.

## Repro

```python
import numpy as np, jax.numpy as jnp, scipy.ndimage as spnd
from scipy.spatial.transform import Rotation
from nitrix.register import affine_register, RegistrationSpec

rng = np.random.default_rng(0)
fixed = spnd.gaussian_filter(rng.standard_normal((28, 28, 28)).astype('f4'), 2.0)
rot = Rotation.from_euler('xyz', [3, 2, 1.5], degrees=True).as_matrix()
c = (np.array(fixed.shape) - 1) / 2
moving = spnd.affine_transform(fixed, rot, offset=c - rot @ c + [1.5, -1, .8],
                               order=1, mode='nearest').astype('f4')
res = affine_register(jnp.asarray(moving), jnp.asarray(fixed),
                      spec=RegistrationSpec(levels=2, iterations=15))
print(np.asarray(res.params))   # explodes; warped anti-correlates with fixed
# levels=1, or optimizer='bfgs', or grid >= 32^3 -> recovers fine.
```

## Impact / perf-bench handling

Low at realistic brain sizes (≥32³ recovers; the affine size-tier bench at
96³–192³ is unaffected). But it is a robustness regression vs v2 and the params
*diverge* rather than just under-converge. `nitrix-perf-bench` `xfail`s its 28³
affine recovery test referencing this doc until it is resolved.

## Cross-references

- `src/nitrix/register/recipes.py` (`affine_register`); the V4b/V4c levers
  (`9e53019` affine inverse-compositional fast path, `aaa0f4a` closed-form
  forward warp Jacobian); rigid's working V4a path (`fbd2597`).
- [`perf-bench-feedback.md`](perf-bench-feedback.md) — the perf-bench ledger.
