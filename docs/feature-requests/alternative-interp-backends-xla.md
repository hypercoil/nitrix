# B16. Research: scipy / cupy interpolation backends spliced into the XLA graph

> **Status (2026-06-04): parked (research note).** Not a commitment — the
> deliverable is a design doc + a differentiability/zero-copy feasibility
> verdict, not a primitive. Provenance: prompted by the
> [`interpolation-backend-cpu-gpu-gap`](interpolation-backend-cpu-gpu-gap.md)
> (B15) measurements; ledger context in
> [`internal-backlog.md`](internal-backlog.md).

B15 measured that `jax.scipy.ndimage.map_coordinates` is 5–9× slower than
`scipy.ndimage` on CPU and ~1.8× slower than `cupyx.scipy.ndimage` for a
single-shot GPU warp. The obvious lever is to call the optimised external
kernel (scipy on CPU, cupyx on GPU) instead of the XLA lowering — **without**
losing jit-composability or differentiability. This note scopes whether that is
achievable and how.

The hard constraint: nitrix's interpolation ops must remain (1) `jit`-able, (2)
differentiable (they sit inside registration losses), and (3) free of hidden
host↔device round-trips. A naive `numpy`/`cupy` call satisfies none. The
question is which XLA integration mechanism preserves all three.

## Candidate mechanisms (to evaluate)

1. **`jax.pure_callback` + custom VJP.** Wrap the external kernel in a
   `pure_callback`; supply the gradient explicitly via `jax.custom_vjp`. For
   **linear** interpolation the VJP is analytic — the cotangent w.r.t. the
   sampled image is the *adjoint scatter* (`map_coordinates`'s transpose,
   `scipy.ndimage` has no direct adjoint but it is `index_add` at the same
   floor/frac weights), and the cotangent w.r.t. the coordinates is the
   image-gradient dotted with the upstream cotangent. So differentiability is
   recoverable without autodiffing through the C kernel. **Open question:**
   `pure_callback` on the **CPU** backend is cheap (no transfer), but on the
   **GPU** backend it forces a device→host→device bounce — which would erase the
   win. So `pure_callback` is plausibly the **CPU-only** answer (the
   higher-value half per B15); it is *not* the GPU answer.

2. **`dlpack` zero-copy + cupy (GPU).** For the GPU path, hand the on-device
   buffer to cupy via `jax.dlpack.to_dlpack` / `from_dlpack`, call
   `cupyx.scipy.ndimage.map_coordinates`, hand it back — all device-resident.
   **Must verify there are no hidden host↔device round-trips**: confirm (a)
   `to/from_dlpack` is genuinely zero-copy (shared buffer, no H2D/D2H), and (b)
   cupy does not stage through host internally for `map_coordinates`. Caveat:
   dlpack hand-off across a `jit` boundary needs `io_callback`/`pure_callback`
   plumbing too, and stream-ordering between the JAX and cupy CUDA streams must
   be synchronised or the result races. This is the subtle part — a wrong
   stream sync gives intermittently-wrong output that fidelity tests on a quiet
   device would *not* catch.

3. **XLA FFI / custom_call (`jax.ffi`, jax ≥ 0.4.3x).** The principled path:
   register the scipy/cupyx kernel (or a thin C/CUDA shim) as an XLA custom
   call, so it lives *inside* the compiled graph — no callback bounce, no
   separate stream, fusible-adjacent. More work (FFI handler + build), but it is
   the only option that is both GPU-native and graph-resident. Differentiability
   still needs an explicit `custom_vjp` (the FFI call is opaque to autodiff).

4. **Pallas kernel (in-XLA, no external dep).** Cross-ref
   [`pallas-trilinear-resample`](pallas-trilinear-resample.md) (B7): write the
   gather as a Pallas kernel. Sidesteps the host/stream/dlpack issues entirely
   (it *is* XLA), at the cost of hand-writing + tuning the kernel. The
   "no external backend" alternative to 1–3.

## Suggested deliverable

A short design doc that, per backend (CPU/GPU), picks one mechanism and answers:
- differentiability: is the analytic linear-interp VJP correct and cheap?
- zero-copy: for the cupy path, a measured check that dlpack hand-off does **no**
  H2D/D2H (e.g. `nvidia-smi`/nsys trace, or a buffer-pointer identity check),
  plus the stream-sync story;
- graph residency: does the op stay inside `jit` without forcing a barrier;
- the throughput actually recovered (re-run the B15 cases).

A reasonable hypothesis to test: **`pure_callback`+scipy for CPU** (cheap, no
transfer, big win) and **either dlpack+cupy or an FFI custom_call for GPU**
(only the single-shot `spatial_transform` needs it; `integrate_velocity_field`
already wins on GPU via fusion, so a backend swap could *regress* it — measure
before swapping).

**Trigger.** B15 graduates (a consumer is blocked on CPU interpolation
throughput), or a JAX-version bump makes the FFI path materially easier.

**Effort.** M (CPU `pure_callback` + analytic VJP, the high-value slice) → L
(GPU FFI/dlpack with the stream-sync + zero-copy verification).

## Cross-references

- [`interpolation-backend-cpu-gpu-gap`](interpolation-backend-cpu-gpu-gap.md)
  (B15) — the measurements that motivate this.
- [`pallas-trilinear-resample`](pallas-trilinear-resample.md) (B7),
  [`keops-genred-research`](keops-genred-research.md) (B5) — sibling
  in-XLA-kernel research.
- `src/nitrix/geometry/grid.py` (`_gather_coords_linear`).
