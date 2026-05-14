# nitrix — Migration map (draft v0)

> **Purpose.** Module-level mapping from existing sources into the nitrix layout proposed
> in `SPEC.md`. Source repos surveyed: this repo's current `src/nitrix/`,
> `../hypercoil/src/hypercoil/`, `../ilex/src/ilex/`, `../entense/src/entense/`.
>
> **Legend (action column).**
>
> - **keep** — already where it belongs; minimal change.
> - **fix** — keep but address known bugs (see §4 below).
> - **port** — move to the destination module, no semantic change beyond namespace.
> - **port-split** — move and split into multiple destination modules.
> - **rewrite** — new implementation; old code is *inspiration only* (numerical stability,
>   adversarial format, etc.).
> - **fold** — fold into an existing destination module rather than carrying as a unit.
> - **extract** — extract the numeric core; drop the surrounding domain context.
> - **drop** — do not migrate.

---

## 1. From this repo (`./src/nitrix/`)

| Source | Destination | Action | Notes |
|---|---|---|---|
| `_internal/util.py` | `nitrix._internal.util` | keep | Axis / mask / complex helpers; healthy |
| `_internal/docutil.py` | `nitrix._internal.docutil` | keep | Docstring formatters |
| `_internal/testutil.py` | `nitrix._internal.testutil` | keep | |
| `functional/covariance.py` | `nitrix.stats.covariance` | fix | Address `:719–726` non-diagonal-weight JIT trap and `:683–686` denominator gap before further migration |
| `functional/matrix.py` | `nitrix.linalg.matrix` | keep | Custom VJP rules at `:554–568` are the pattern to emulate elsewhere |
| `functional/residual.py` | `nitrix.linalg.residual` | keep | Healthy `lstsq`-backed residualise |
| `functional/fourier.py` | `nitrix.stats.fourier` | keep | |
| `functional/window.py` | `nitrix.signal.window` | fix | Drop `numpyro.distributions.Multinomial` import (`:12`); use `jax.random` |
| `functional/geom.py` | `nitrix.geometry.{grid,sphere,coords,metrictensor}` + `nitrix.smoothing.gaussian` | port-split | Spatial / spherical conv to be *re-backed* by `nitrix.semiring`, not by the legacy O(N²) inner loop (`:632` TODO); split the modularity / centre-of-mass / Gaussian-kernel halves into their natural destinations |

## 2. From `hypercoil`

### 2.1 `hypercoil/functional/`

| Source | Destination | Action | Notes |
|---|---|---|---|
| `sparse.py` | `nitrix.sparse.ell` (inspiration) | rewrite | **Do not port the BCOO path.** Reimplement on plain dense arrays + `jnp.take`/`lax.gather`. Preserve the top-k-shared-pattern *idea*, drop the format |
| `kernel.py` | `nitrix.linalg.kernel` | port | Singledispatch over kernel type — preserve |
| `matrix.py` | `nitrix.linalg.matrix` | fold | Merge with existing; reconcile overlaps |
| `graph.py` | `nitrix.graph.laplacian` | port | Laplacian / modularity matrix / Girvan–Newman null |
| `cov.py` | `nitrix.stats.covariance` | fold | Consolidate with existing covariance.py |
| `sphere.py` | `nitrix.geometry.sphere` | port | Icosphere generation, spherical geodesics. **Re-back `spatial_conv` on `semiring_ell_conv`** instead of the existing block loop |
| `fourier.py` | `nitrix.stats.fourier` | fold | Consolidate with existing fourier.py |
| `linear.py` | `nitrix.numerics.tensor_ops` | port | Compartmentalised linear maps, normalisations |
| `symmap.py` | `nitrix.linalg.spd` | port | Symmetric map transformations |
| `semidefinite.py` | `nitrix.linalg.spd` | rewrite | Existing implementation flagged numerically unstable in docstring; stabilise before migrating |
| `connectopy.py` | `nitrix.graph.connectopy` | extract | Keep the eigenmap / diffusion-map numerics; drop the brainspace dependency |
| `cmass.py` | `nitrix.geometry.{grid,coords}` | port-split | Centre-of-mass to `grid`; coordinate utilities to `coords` |
| `tsconv.py` | `nitrix.signal.tsconv` | port | Time-series / basis convolution kernels |
| `metrictensor.py` | `nitrix.geometry.metrictensor` | port | Geometry primitives |
| `interpolate.py` | `nitrix.signal.interpolate` | extract | Drop neuro context; keep spectral / linear / hybrid algorithms |
| `activation.py` | — | drop | `jax.nn` has these |
| `crosshair.py`, `crosssim.py` | — | drop | Connectivity-specific; belongs higher up |
| `window.py` | `nitrix.signal.window` | fold | Consolidate with existing window.py |
| `resid.py` | `nitrix.linalg.residual` | fold | Consolidate with existing residual.py |

### 2.2 `hypercoil/init/` (numeric halves only)

| Source | Destination | Action | Notes |
|---|---|---|---|
| `laplace.py` | `nitrix.linalg.kernel` | port | Laplace kernel init; merge into the kernel module |
| `toeplitz.py` | `nitrix.linalg.matrix` | fold | `toeplitz` already exists in current nitrix matrix module |
| `semidefinite.py` | `nitrix.linalg.spd` | rewrite | Depends on `functional/semidefinite.py` (rewrite first) |

### 2.3 NOT for nitrix (route to thrux / bitsjax / nimox / drop)

These appear in the hypercoil salvage list but do **not** belong in nitrix:
`functional/{cov-as-module wrappers}`, all of `nn/`, all of `loss/`, all of `init/atlas*`,
all of `neuro/`, all of `formula/`, `engine/`. The `nn/` and `loss/` items are nimox /
bitsjax candidates; `init/atlas*` and `neuro/` are thrux candidates; `formula/` is split
across thrux (imops) and bitsjax (nnops). See those libraries' MIGRATION.md.

## 3. From `ilex`

| Source | Destination | Action | Notes |
|---|---|---|---|
| `models/voxelmorph/_numerical.py` | `nitrix.geometry.grid` | port | `identity_grid`, `spatial_transform`, `vec_int`, `rescale` are general deformable-registration primitives, not voxelmorph-specific. Regression tests in `models/voxelmorph/tests/regression/test_pytorch_parity.py` travel with the code |
| `models/synthstrip/preprocessing.py::intensity_normalize` | `nitrix.signal.normalize` | port | Pure-numeric intensity normalisation. Tests in `models/synthstrip/tests/test_preprocessing.py` (the intensity_normalize slice) travel with it |
| `core/adapters.py` (≈ lines 150–250 — pure-array transforms) | `nitrix.numerics.tensor_ops` | port-split | `reshape_to`, `transpose`, `transpose_tf_conv_kernel`, `broadcast_bias` are pure array → array. The adapter registry, rules, and StateDict integration stay in ilex (`core/`) |

## 4. From `entense`

| Source | Destination | Action | Notes |
|---|---|---|---|
| `instance.py::polynomial_detrend_p` impl | `nitrix.signal.filter` | port | Pure tensor-level detrending; XLA-fusable |
| `instance.py::confound_regression_p` impl | `nitrix.linalg.residual` | fold | The numerical core is residualisation; consolidate. The orchestration / dim-handling wrapper stays in thrux/entense |
| `instance.py::polynomial_detrend_p` *Primitive wrapping* | — | (stays in entense / thrux) | Only the bare-tensor `_impl_f` migrates |

## 5. Known issues to resolve during migration

These are not new TODOs — they are existing red flags that the survey flagged. The
migration is the natural opportunity to fix them.

| Issue | Site | Fix-during action |
|---|---|---|
| Non-diagonal weight matrices in `cov` silently wrong after JIT | `functional/covariance.py:719–726` | Either implement properly or raise unambiguously at trace time |
| Undeclared hard dep on `numpyro.distributions.Multinomial` | `functional/window.py:12` | Drop the import; use `jax.random.categorical` / `jax.random.multinomial` |
| `spatial_conv` 2D-only | `functional/geom.py:632` | Will be moot once re-backed by `semiring_ell_conv` |
| `diffuse()` and `cmass_reference_displacement_*` not exported | `functional/geom.py` | Export under `nitrix.geometry.coords` after the split, with tests |
| `SPD` numerical instability | `hypercoil/functional/semidefinite.py` | Rewrite (per spec) on migration, not after |
| `connectopy` brainspace dep | `hypercoil/functional/connectopy.py` | Strip on extract |

## 6. Recommended migration order

1. **Fixes first.** Resolve §5 issues in-place in the current nitrix where possible —
   covariance, window — so the migration starts from a clean base.
2. **Semiring core.** Build `nitrix.semiring` from the brainstorm (§11 of SPEC). This is
   the substrate everything downstream specialises onto, so land it before re-backing
   spherical conv, mesh ops, or morphology.
3. **Sparse format.** `nitrix.sparse.ell` clean-room implementation; `grid` and `mesh` as
   thin specialisations.
4. **Geometry split.** Split current `geom.py` and migrate `hypercoil/functional/sphere`
   and `cmass` in; re-back `spherical_conv` on semiring + ELL.
5. **Stats + signal + linalg consolidation.** Fold existing nitrix + hypercoil sources
   into `nitrix.{stats, signal, linalg}` per §1–§2.
6. **Smoothing + morphology.** Permutohedral lattice and morphology come last; both
   depend on semiring and sparse landing first.
7. **LME namespace.** Reserve and stub; no implementation in this pass.
