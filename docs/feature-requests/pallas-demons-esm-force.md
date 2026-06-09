# Pallas kernel for the Demons ESM-force inner step

> **Status (2026-06-09): proposed, profile-gated.** Surfaced by
> `nitrix-perf-bench` while benching `register.diffeomorphic_demons_register`
> across scale on an L4 (see that repo's `reports/REGISTRATION_SCALING.md`).
> A commitment-free follow-up: build it only once a profile confirms the
> gradient+force kernels dominate HBM traffic (Trigger below). Unlike
> [`pallas-trilinear-resample.md`](pallas-trilinear-resample.md), the proposed
> target is **gather-free**, so it does *not* inherit the ELL
> gather-lowering blocker that parks that kernel.

**Observation (the symptom).** Post the `lax.scan` loop-roll, the Demons
recipe's GPU steady time is **memory-bandwidth-bound at scale**: its GPU/CPU
speedup *peaks* ~43× (48–96³) then **erodes to ~28× (160³)**, and GPU steady
grows **super-linearly** (96³→160³: 80→648 ms = 8.1× for 4.6× the voxels) — the
most bandwidth-bound of the three registration recipes (rigid/affine, being
compute-bound on the assembled small-P normal equations + `matrix_exp`, scale
cleanly and are *not* Pallas candidates). The cost is the per-iteration traffic
over several 3-component (d=3) spatial fields at low arithmetic intensity.

**Target candidate — the ESM force, fused with the warped-image gradient.**
Each `lax.scan` step of `register/diffeomorphic.py::_demons_level` is:

1. `integrate_velocity_field` (scaling-squaring: `n_steps` self-composition
   **gathers**)
2. `spatial_transform` (the warp **gather**)
3. `spatial_gradient(warped)` (**stencil** → a d-vector field)
4. ESM force — `diff = F − M∘φ`; `j = ½(∇F + ∇warped)`;
   `denom = |j|² + α²·diff²`; `u = (diff/denom)·j` (**elementwise** over d)
5. `_smooth_vector(u, σ_fluid)` (**separable Gaussian conv**)
6. `v += u` (or BCH `compose_velocity`)
7. `_smooth_vector(v, σ_diffusion)` (**separable Gaussian conv**)

XLA fuses the pure-elementwise chain (4) but **breaks fusion at the stencil (3)
and the convolutions (5,7)**, so `∇warped`, `j`, `denom`, `u` each round-trip
HBM every iteration. **The prime candidate is fusing 3+4**: a tiled Pallas
kernel that loads `warped` (with a 1-voxel halo for the stencil), `fixed`, and
the precomputed `∇F`, computes `∇warped` in-tile, does the channel reduction for
`denom`, and writes **only** `u` (d components). This collapses ~4 full-field
intermediates into one read-set → one write — pure stencil + elementwise +
channel-reduction, Pallas's sweet spot.

**Why it dodges the gather blocker.** Steps 3+4 use **no** HLO `gather`
primitive (a fixed-offset stencil is `pl.load` with static slices, not a
data-dependent gather), so this kernel avoids the Triton gather-lowering risk on
the pinned JAX that keeps `pallas-trilinear-resample` parked. The gather-heavy
steps (1,2) are a *weaker* target — irregular access, and XLA's gather is
already competitive — and the separable Gaussians (5,7) are a well-trodden conv
path with modest expected gains. So the force kernel is both the
highest-bandwidth and the most tractable piece.

**Trigger (build only when both hold).** (a) A profile (HLO op-level HBM
traffic / a roofline on the L4 or A10G) confirms the gradient+force kernels
dominate the per-iteration traffic — the hypothesis here is from the timing
*shape*, not yet a profile; (b) a downstream registration/training loop shows
Demons is a real bottleneck. A kernel must register a `custom_vjp` (the
backward of the fused stencil+force) and keep the pure-JAX path as the floor /
parity oracle.

**How it would be measured.** perf-bench already reserves a `nitrix-pallas`
baseline slot alongside `nitrix-jax`; a Pallas force kernel drops in there and
is measured directly against the XLA path on the existing
`diffeomorphic_demons` scale tier (96–160³), so the win (or its absence) is a
recorded number, not a guess.

**Baseline (L4, jax 0.10.0; perf-bench `reports/REGISTRATION_SCALING.md`).**
Demons GPU steady (`levels=2, iters=20`): 48³ 9.7 ms, 96³ 80 ms, 128³ 295 ms,
160³ 648 ms; per-voxel HBM ~3 KB (vs rigid/affine ~1.8) at the clean small
sizes; speedup vs CPU 43× → 28× across 48³→160³.

## Cross-references

- [`pallas-trilinear-resample.md`](pallas-trilinear-resample.md) — the parked
  gather kernel (the ELL gather-lowering blocker this one avoids).
- [`pallas-gaussian-blur.md`](pallas-gaussian-blur.md) — the separable-Gaussian
  kernel (steps 5,7 here would reuse it).
- [`registration-recipe-cold-compile.md`](registration-recipe-cold-compile.md)
  — the (now-resolved) loop-roll finding that exposed this steady-state regime.
- `src/nitrix/register/diffeomorphic.py` (`_demons_level` — the inner `lax.scan`
  step); `src/nitrix/geometry/differential.py` (`spatial_gradient`).
