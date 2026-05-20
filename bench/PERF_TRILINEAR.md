# Trilinear resampling — baseline + Pallas decision

> Consumer ask: a Pallas kernel for 3D trilinear resampling
> (``geometry.spatial_transform`` / ``integrate_velocity_field``).
> This report is the "benchmark first" step before any kernel work.

## Host

- Device: NVIDIA A10G (gpu)
- Platform: Linux-6.1.161-183.298.amzn2023.x86_64-x86_64-with-glibc2.39
- JAX: 0.10.0

## Steady-state (post-warm-up median)

`fwd` = shipped ``spatial_transform`` (``map_coordinates`` order=1).
`fwd+bwd` = ``value_and_grad`` of a scalar loss wrt the field
(the registration-training cost). `explicit` = pure-JAX 8-corner
gather (forward), the closest pure-XLA analogue of a fused Pallas
kernel.

| shape | voxels | fwd | fwd Mvox/s | fwd+bwd | fwd+bwd Mvox/s | explicit fwd | explicit Mvox/s |
|---|------:|----:|----:|----:|----:|----:|----:|
| 64x64x64 | 262144 | 157.1 µs | 1668 | 275.9 µs | 950 | 129.4 µs | 2025 |
| 128x128x128 | 2097152 | 512.5 µs | 4092 | 1.69 ms | 1241 | 328.1 µs | 6393 |
| 192x192x192 | 7077888 | 1.38 ms | 5124 | 4.81 ms | 1472 | 813.0 µs | 8706 |
| 256x256x256 | 16777216 | 3.14 ms | 5338 | 13.01 ms | 1289 | 1.80 ms | 9330 |

## Reading

- Trilinear resampling is gather-bound: the shipped
  ``map_coordinates`` path and the explicit 8-corner gather both
  lower to ``lax.gather`` over the volume.  Their throughput is
  the practical ceiling for an XLA implementation on this device.
- A Pallas Triton kernel would need to express the 8 corner loads
  as data-dependent ``pl.load`` / pointer arithmetic.  The G0 gate
  (``bench/G0_ELL_REPORT.md``) found Triton on the pinned JAX does
  not lower the ``gather`` HLO primitive; a pointer-load
  formulation *may* sidestep that, but it is unproven and is the
  same upstream risk surface that kept ELL on the JAX path.

## Decision

See the inline summary printed by this script and the writeup in
the feedback / backlog.  The kernel is gated on (a) these numbers
showing the path is a real bottleneck in a consumer training loop,
and (b) a pointer-load Pallas prototype clearing the gather-
lowering risk -- otherwise ship JAX-default (the current state),
with the kernel revisited when upstream Pallas grows gather.
