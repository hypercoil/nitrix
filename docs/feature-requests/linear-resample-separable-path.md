# Perf: `Linear` / `NearestNeighbour` resample uses the dense corner-gather; a separable resize path would be faster

> **Status (2026-06-09): OPEN — perf-optimisation candidate.** Surfaced
> building the `nitrix-perf-bench` interpolation-kernel coverage (resample
> dispatch branches, Phase 3). A measured perf observation, not a bug: the
> output is correct (matches scipy `map_coordinates` to fp32 round-off) — it is
> just slower than it needs to be for an axis-aligned resize.

## Observation

On `nitrix.geometry.resample` (align-corners resize), benchmarked across the
dispatcher's kernels on an L4, output `64³ → 128³` (steady-state min):

| kernel | nitrix | taps | dispatch path |
|---|---|---|---|
| NearestNeighbour | 0.10 ms | 1 | dense gather (`prefers_separable_resample=False`) |
| **Lanczos(3)** | **0.15 ms** | 6/axis | **separable 1-D** (`True`) |
| **CubicBSpline** | **0.61 ms** | 4/axis | **separable 1-D** (`True`) |
| **Linear** | **1.02 ms** | 2/axis | **dense `2^ndim`-corner gather** (`False`) |

**Linear is the slowest kernel for a resize** — slower than the higher-order
Lanczos and CubicBSpline — which is counter-intuitive (it has the fewest taps).

## Cause (verified)

`resample` routes through `_resample_on_grid` (`geometry/grid.py`), which picks
the **separable 1-D** path when `method.prefers_separable_resample` is `True`,
else the **dense meshgrid corner-gather**. `Linear` and `NearestNeighbour` set
that flag `False` (`geometry/_interpolate.py:628,686`), so a linear resize
materialises the full `(ndim, *out)` coordinate grid and gathers `2^ndim`
corners per output voxel — heavier (and less XLA-friendly) than the separable
1-D passes `Lanczos` / `CubicBSpline` use. Nearest stays cheap only because it
is a single tap.

## Why it is fixable

An **axis-aligned resize is separable for linear (and nearest)** interpolation,
exactly as it is for the higher-order kernels — the per-axis 1-D weight passes
compose to the same result. So `Linear` resize could take the separable path
(set `prefers_separable_resample=True`, or have `_resample_on_grid` treat the
separable kernels uniformly) and would likely match or beat CubicBSpline.

**Scope caveat:** this is **resample-specific**. `Linear`'s dense corner-gather
is genuinely required for `spatial_transform` (arbitrary, non-separable
deformation coordinates), so the flag/path change must apply only to the
axis-aligned resize, not to general warping.

## Cross-references

- [`interpolation-backend-cpu-gpu-gap.md`](interpolation-backend-cpu-gpu-gap.md) — the per-platform Linear/NN gather engine (`_separable_gather` / `_gather_sample`).
- [`pallas-trilinear-resample.md`](pallas-trilinear-resample.md) — the Pallas pointer-load kernel for the dense gather.
- `nitrix.geometry.resample`, `_resample_on_grid` (`geometry/grid.py`), `Linear` / `NearestNeighbour` (`geometry/_interpolate.py`).
