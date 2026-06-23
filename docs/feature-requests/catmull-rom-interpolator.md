# CatmullRom interpolator — `nitrix.geometry._interpolate`

> **Status (2026-06-23): SHIPPED.** `CatmullRomCubic` added to the
> `Interpolator` dispatch set (`geometry._interpolate`), re-exported from
> `nitrix.geometry` and `geometry.grid`, and wired into the `method=` dispatch
> of `resample` / `spatial_transform`. Bit-exact with the ilex
> `CatmullRomCubic` reference and the FD-Net STN `_hermite` polynomial (fp32
> ULP at every fractional offset — the load-bearing port-parity test). 16 new
> tests (analytic weights, interpolation property 1-D/2-D/3-D, upstream-Hermite
> parity, partition-of-unity, linear-ramp reproduction, overshoot, resample ==
> spatial_transform, differentiability, jit/vmap, record contract); op-matrix
> entry `resample[catmull]`; full geometry suite 131 passed. **Consumer action:
> `nimox._interpolate` can drop the vendored `CatmullRomCubic` and import
> `from nitrix.geometry import CatmullRomCubic`.**
>
> **Correction carried to the consumer.** The vendored ilex docstring claims
> the cardinal-cubic basis is *not* a partition of unity ("weights sum to 1
> only at the knots and the midpoint"). That is **wrong** — the four
> tension-(-1/2) weights sum to exactly 1 for *all* `t` (verified
> algebraically and numerically), so the kernel reproduces constants **and**
> linear ramps exactly (Keys third-order accuracy). The real caveat is that it
> is not a *convex* combination (small negative side lobes ~-0.07 → can
> overshoot/ring near edges), unlike `Linear`. The nitrix docstring states this
> correctly; the ilex vendored copy should be corrected when it is retired.
>
> _Original request (open, consumer request, ilex → nitrix). Verified against
> `nitrix main@6449cfa`: the `Interpolator` set was `Linear` /
> `NearestNeighbour` / `Lanczos` / `CubicBSpline` (`_interpolate.py`) — no
> interpolating cubic-Hermite (Catmull-Rom) kernel. Surfaced while migrating
> ilex's geometry cluster onto nitrix; this was the one interpolation primitive
> with no nitrix equivalent, so it **blocked** the `nimox._interpolate`
> migration (the others — warp/affine — landed)._

**What.** Add a **Catmull-Rom** cubic interpolator to the `Interpolator`
dispatch set, matching ilex's vendored
`ilex/nimox/modules/_interpolate.py:186` (`CatmullRomCubic`, a separable
cubic-Hermite gather over the existing `_separable_gather` substrate).

**Why it's distinct from the shipped kernels.** Catmull-Rom is an
**interpolating** C¹ cubic Hermite spline (the curve passes through the
samples; tangents are centred finite differences). nitrix's existing cubics
are different functions:
- `CubicBSpline` is **approximating**, not interpolating (it needs the IIR
  prefilter to interpolate, and that prefilter is itself backend-dependent —
  see [`cubic-bspline-prefilter-backend-parity`](cubic-bspline-prefilter-backend-parity.md));
- `Lanczos` is a windowed-sinc, a different kernel and support.

So no shipped kernel reproduces a Catmull-Rom resample bit-for-bit; a consumer
pinned to Catmull-Rom (matching an upstream port) cannot switch to `CubicBSpline`
or `Lanczos` without changing numerics.

**Consumer / driver.** `ilex/models/fd_net` (the forward distortion path) uses
`CatmullRomCubic` and is parity-locked against its upstream. Until nitrix ships
the kernel, `nimox._interpolate` stays vendored (the warp + affine geometry
migrations delegated cleanly; this is the residual).

**Fit.** Should be cheap: a Catmull-Rom is a 4-tap separable cubic-Hermite
gather — the same `_separable_gather` engine `Lanczos` already rides, with a
fixed 4-tap weight rule (the Catmull-Rom basis) and `order`-3 tap extent. No
prefilter (it interpolates directly), so unlike `CubicBSpline` it has **no
backend-dependent scan** — a clean, backend-stable addition.

**Ask.** A `CatmullRom` (or `CubicHermite`) `Interpolator` record with the
standard tension-½ Catmull-Rom weights, slotted into the `method=` dispatch of
`spatial_transform` / `resample`, with the usual golden-corpus parity pin
against the reference gather.

**Effort.** S — a weight rule on the existing separable-gather substrate.

## Cross-references

- `ilex/nimox/modules/_interpolate.py` (`CatmullRomCubic` — the reference impl
  to match), `ilex/models/fd_net` (consumer).
- [`cubic-resample`](cubic-resample.md) — where `CubicBSpline` (order-3) was
  added; the sibling cubic, but approximating not interpolating.
- `src/nitrix/geometry/_interpolate.py` (`Lanczos`, `CubicBSpline`,
  `_separable_gather` — the substrate the new kernel rides).
