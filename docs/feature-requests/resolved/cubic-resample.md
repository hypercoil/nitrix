# Cubic (order-3) resample — `nitrix.geometry`

> **Status (2026-06-08): ✅ RESOLVED — `CubicBSpline` shipped.** The
> separable B-spline prefilter + cubic sampler is implemented as a
> `CubicBSpline` `Interpolator` record
> (`geometry/_interpolate.py`): `resample(img, shape,
> method=CubicBSpline())` / `spatial_transform(..., method=CubicBSpline())`.
> **Bit-exact** with `scipy.ndimage.map_coordinates(order=3, mode='mirror')`
> (interior *and* boundary, ~1e-15) — the nnUNet / `hd_bet` order-3 parity
> the ilex pipelines need. See `IMPLEMENTATION_PLAN.md §10.3` (2026-06-08)
> and `docs/design/geometry.md`. Provenance: 2026-06-02 ilex vendored-model
> survey ([`ilex-pipeline-substrate.md`](../ilex-pipeline-substrate.md),
> volumetric item D); ilex SKILL FM #17.

**What (as built).** A `CubicBSpline` interpolation method, the order-3
spline path consumers like nnUNet / `hd_bet` preprocessing use. Unlike a
plain cubic *convolution*, it is the two-step B-spline operation: (1) a
recursive **prefilter** (the cubic pole `√3−2`) converts samples to
interpolating coefficients, and (2) a 4-tap cubic-basis gather. Without the
prefilter a cubic basis only *approximates* (blurs); with it the result
passes through the samples (the interpolation property). Differentiable in
values and coordinates. The prefilter forces the **mirror** boundary (both
prefilter and gather) so it stays self-consistent; it ignores the `mode` /
`cval` call args, and **announces** that override with a
`CubicBSplineBoundaryWarning` when a non-mirror `mode` / non-zero `cval` is
explicitly supplied (the bare default is silent). A mode-aware prefilter for
`nearest`/`reflect`/… parity is the one remaining follow-up — see
`boundary-mode-parity.md`.

**Engine note.** The prefilter is a first-order linear recurrence: sequential
`lax.scan` on CPU, parallel `lax.associative_scan` (O(log N) depth) on GPU —
the same platform split as `signal._iir`, both exact. An FFT convolution (the
*other* `_iir` engine) is deliberately **not** used: the cubic pole is mild,
so the prefilter's impulse response is a ~25-tap short FIR, too short for the
transform overhead to amortise.

**What it needs.** A separable B-spline prefilter + cubic sampling. With the
dispatcher in place this now slots in cleanly as a **new `Interpolator`
record** (e.g. `CubicBSpline(order=3)`) in `geometry/_interpolate.py` — the
prefilter is a per-axis recursive IIR pass, the sampler a 4-tap separable
gather (the existing `_separable_gather` / `_separable_resample` machinery
takes any tap rule). No new top-level surface; just another `method=`.

**Priority.** Lower than the ENABLING items A–C (linear is "good enough" for
most consumers). The deviation is flagged in the `resample` docstring.

**Home.** `nitrix.geometry._interpolate` (a new `Interpolator` record).

## Cross-references

- [`ilex-pipeline-substrate.md`](../ilex-pipeline-substrate.md) — survey
  context + the volumetric tier.
- [`pallas-trilinear-resample.md`](../pallas-trilinear-resample.md) (B7) — the
  *linear* resampling perf track; explicitly a **separate** concern from
  this order-3 parity gap.
- `src/nitrix/geometry/_interpolate.py` — the `Interpolator` dispatcher a
  cubic method would extend; `src/nitrix/geometry/grid.py` —
  `spatial_transform` / `resample`.
