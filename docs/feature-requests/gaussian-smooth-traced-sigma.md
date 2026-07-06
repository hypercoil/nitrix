# jit-safe traced sigma for `smoothing.gaussian`

> **Home:** `smoothing.gaussian`.
> **Severity:** ENABLING for the ilex training-substrate augmentation
> family (blocks per-view sampled sigma inside a JIT-traced augmentation
> function).

## The problem

`nitrix.smoothing.gaussian(volume, sigma=sigma, spatial_rank=3)` calls
`_normalise_sigma(sigma, spatial_rank)` at line ~332 of
`src/nitrix/smoothing/gaussian.py`, which iterates and casts to Python
float:

```python
out = tuple(float(s) for s in sigma)
```

When `sigma` is a JAX tracer (a `jnp.uniform(...)` sample inside a
`jit`/`vmap`-traced augmentation function), this raises
`TypeError: iteration over a 0-d array` (or ``ConcretizationTypeError`` in a
`jit` trace). The op is designed against **host-static** sigma — the
FIR kernel *shape* (kernel_size / half-width) is derived from sigma and
must be known at trace time.

The docstring notes this intentionally
(`_normalise_sigma` / `gaussian_kernel_1d` — "The design uses host (static)
floats — it is not traced."), so this is by design for the current
implementation. This FR is asking for a **traced-sigma variant**.

## Why it matters

The ilex 3DINO augmentation pipeline (and any SSL augmentation stack that
mirrors upstream ``DataAugmentationDINO3d``) samples the smoothing sigma
uniformly per view (upstream ``RandGaussianSmooth`` with
``sigma_x=(0.25, 1.5)``). Fixing sigma to a single Python value forfeits
this diversity — a strictly weaker augmentation than the reference. In
our stage-18 recipe scan (`ilex/train/pipelines/dino3d.py`) the smoothing
op is one of the intensity-augmentation levers we need turned on to
match upstream; the host-static requirement blocks it under the traced
per-view sampling that every other op supports.

We hit this concretely while reconfiguring the `scaling-train` view
function (2026-07-05) to match upstream augmentation probabilities; the
workaround was to disable smoothing entirely
(`gaussian_smooth_prob = 0.0` in the ilex `ViewConfig` defaults). The
sharpening op (`_maybe_gaussian_sharpen`) has the same constraint via
the same underlying kernel.

## Proposal (sketch)

Provide a `smoothing.gaussian(..., kernel_size=K, sigma=<traced>)` mode
where `kernel_size` is host-static (a Python `int`) but `sigma` is a
JAX tracer. Kernel taps at each half-width position `k` in `[-h..+h]`
are computed as ``exp(-0.5 * (k / sigma) ** 2)`` then normalised — all
tracer-safe. The kernel *shape* stays static; only the *weights* vary
with sigma.

Caller obligation: pick a `kernel_size` large enough to cover the
maximum sigma of interest (e.g. `kernel_size = 2 * ceil(3 * sigma_max) + 1`).
Under-sizing silently truncates the Gaussian tail — the docstring should
call this out.

Two possible surfaces:

1. **Extend the existing signature.** Allow `sigma` to be traced when
   `kernel_size` is provided. Keep the current host-static path when
   `kernel_size is None`.
2. **New variant `gaussian_fixed_kernel(...)`.** Explicit sibling that
   makes the invariant obvious at the call site.

(1) preserves the API but stretches the contract; (2) is clearer but
adds a public surface. Prefer (1) if the trace-time check is cheap.

## Trigger

Any ilex/nimox training pipeline that wants **per-view sampled
smoothing sigma** under jit/vmap. Already blocking the stage-18 3DINO
pipeline (see ilex `train/pipelines/dino3d.py`, `scaling-train/src/`
`scaling_train/config.py` — `ViewConfig.gaussian_smooth_prob = 0.0`
comment).

## Cross-references

- Existing FR context: [`ilex-training-substrate.md`](ilex-training-substrate.md).
- Related FR (perf, orthogonal concern): [`pallas-gaussian-blur.md`](pallas-gaussian-blur.md).
- Docstring convention: `src/nitrix/smoothing/gaussian.py` — the
  "host (static) floats — it is not traced" note (line ~242) should be
  updated to describe the traced-sigma path once it exists.

## Non-goals

- **Not** a request to remove the host-static path — it stays the
  default. Traced sigma is opt-in via ``kernel_size``.
- **Not** a request for a traced ``kernel_size`` — kernel shape must
  stay static so that the FIR conv can be traced. Only sigma varies.
- **Not** a Pallas kernel — orthogonal to the perf story in
  ``pallas-gaussian-blur.md``.
