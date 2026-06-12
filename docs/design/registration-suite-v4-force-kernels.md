# v4 Phase 5 — fused force kernels (Pallas) (implementation design)

> **Status (2026-06-12): implementation-ready design, profile-gated build.** The
> three custom force kernels of [`registration-suite-v4.md`](registration-suite-v4.md)
> Phase 5 — fused **ESM** (5a), **LNCC windowed-statistics** (5b), and **MI
> histogram+gradient** (5c). Grounded in the §3.2 HBM analysis: each force has a
> *different* bottleneck shape, so each warrants a *different* kernel and a
> different priority. **Build discipline:** none is built speculatively — each is
> gated on a recorded roofline confirming the op dominates per-iteration HBM **and**
> a bottlenecked consumer (`pallas-demons-esm-force` Trigger). All land in the
> existing custom-kernel home (`src/nitrix/_kernels/`, beside the semiring kernels)
> and the perf-bench `nitrix-pallas` baseline slot.

## 0. Common framework (all three kernels)

- **Parity oracle = the pure-JAX path.** Every kernel ships behind a dispatch
  (`backend='pallas'` vs the default JAX path) and is validated ULP/tolerance-equal to
  the JAX force it replaces (`DemonsForce` / `LNCCForce` / `MIForce`). The JAX path is
  the floor and the correctness oracle — the kernel is *pure perf*, never a new
  numeric.
- **`custom_vjp` or JAX-backward fallback.** Each kernel registers a `custom_vjp` for
  the differentiable-layer path; where the backward is heavy (5b, 5c) the **forward**
  kernel ships first with the backward routed through the pure-JAX autodiff path (a
  correct, slower gradient) until the backward kernel earns its build.
- **Gather-free where possible (the parked-blocker dodge).** 5a/5b use only
  **fixed-offset stencils** (`pl.load` with static slices + halos), **not** the
  data-dependent `gather` HLO that parks `pallas-trilinear-resample` on the pinned JAX.
  5c uses a **scatter** (histogram) — a distinct primitive whose pinned-JAX Pallas
  support (atomics / shared-memory accumulation) must be validated first (§3.0).
- **Measured, not asserted.** Each drops into the perf-bench `nitrix-pallas` slot and
  is scored against the JAX path on the existing scale tier (96³–256³ single + cohort),
  with the HBM-per-voxel and the crossover recorded — per `perf-wins-must-certify-at-scale`.
- **Pinned stack:** JAX `cuda12==0.10.1` (the dev env); `_kernels/cuda/` already hosts
  Pallas/Triton semiring kernels — reuse that build/dispatch scaffold.

---

## 1. Kernel 5a — fused ESM stencil+force (Demons)

**Bottleneck (§3.2):** ~5–6 full-field round-trips — XLA fuses the elementwise force
chain but **breaks fusion at the `∇warped` stencil**, so `∇warped`, `j`, `denom`, `u`
each round-trip HBM every iteration. **Highest tractability; build first.**

**Fuse steps 3+4** (`spatial_gradient(warped)` + the ESM force) into one tiled kernel.

- **Inputs (per call):** `warped (M)`, `fixed (M)`, `grad_fixed = ∇F (M, ndim)`
  (precomputed once in `bind`), `alpha`, `eps` (the 0a guard).
- **Tiling:** spatial tiles (e.g. `8×8×8` for 3-D) with a **1-voxel halo** for the
  central-difference stencil. Per tile:
  1. `pl.load` `warped` tile **+ 1-voxel halo**, `fixed` tile, `grad_fixed` tile.
  2. compute `∇warped` in-tile (central difference, uses the halo) → `j = ½(∇F + ∇warped)`.
  3. `diff = fixed − warped`; `denom = Σ_d j_d² + α²·diff²`;
     `u = where(denom > eps, diff/denom, 0) · j` (the **0a guard fused in**).
  4. `pl.store` **only** `u (M, ndim)`.
- **I/O:** read `warped(+halo) + fixed + ∇F` ≈ `M·(ndim+2)`; write `u` = `M·ndim`. The
  ~4 intermediate full-field round-trips collapse to one read-set + one write.
- **Backward (`custom_vjp`):** `u = f(warped, fixed; ∇F)`. `∂u/∂warped` needs the
  **adjoint of the central-difference stencil** (the transpose stencil) composed with
  the elementwise Jacobian of the force — a second tiled stencil kernel (`fixed`/`∇F`
  constant). Ship the forward kernel first with the backward via the JAX path; add the
  backward kernel when the differentiable demons-layer is bottlenecked.
- **Parity:** ULP-equal to `DemonsForce.update` (incl. the 0a guard) on random + flat
  inputs.
- **Gate:** the §3.2 profile (gradient+force dominate per-iteration HBM) + a bottlenecked
  consumer. Reuse the recursive Gaussian (Phase 1d) for the smooth steps 5/7 (not fused
  here — separable convs are a separate, well-trodden path).

---

## 2. Kernel 5b — fused LNCC windowed-statistics (SyN)

**Bottleneck (§3.2):** the **worst** — `lncc_grad` runs ~27 separable-conv passes
(~5× ESM), each a full-field round-trip XLA cannot fuse across axes. **Largest HBM to
recover; the brain-scale endgame — build *after* Phase 1c** (the integral-image
algorithmic win may already suffice below 256³; this kernel closes the 256³ tail).

Structure: LNCC-grad is two **stencil-reduce** stages over a window of radius `r`,
separated by an elementwise stage. Fuse each stage into one halo'd tiled pass.

- **Inputs:** `warped (M)`, `fixed (M)`, `radius r`, `eps`. (Symmetric SyN re-binds per
  step, so no fixed-only hoist here — that is the single-sided L-ii win; this kernel
  recomputes both.)
- **Pass 1 — windowed statistics → (p, q, means).** Spatial tiles with a **radius-`r`
  halo**. Per tile, accumulate the five windowed sums over the `(2r+1)^ndim` window
  (`Σm, Σf, Σm², Σf², Σmf`) — in-tile via a local integral image (O(tile), the Phase-1c
  trick at tile scale) or direct accumulation; then `cross = Σmf − ΣmΣf/n`,
  `var_m, var_f`, `D = var_m·var_f + eps`, `p = 2cross/D`, `q = −2cross²·var_f/D²`,
  `mbar = Σm/n`, `fbar = Σf/n`. Store `p, q, mbar, fbar` (4×`M`).
- **Pass 2 — re-sum + assemble + fuse the warped gradient.** Tiles with a radius-`r`
  halo on `p, q` (and a 1-voxel halo for `∇warped`). Per tile: `box(p)`, `box(p·fbar)`,
  `box(q)`, `box(q·mbar)`; `scalar = fixed·box(p) − box(p·fbar) + warped·box(q) −
  box(q·mbar)` (= `lncc_grad`); compute `∇warped` in-tile and write the force
  `u = scalar·∇warped (M, ndim)`.
- **I/O:** ~2 halo'd passes vs ~27 conv round-trips. Halo overhead = `((T+2r)/T)^ndim`;
  for `r≤4`, `T≈8–16` it is a small multiplier on a massive pass-count reduction.
- **Backward:** LNCC's force is itself a gradient; the differentiable-layer backward
  (`∂u/∂warped`) is second-order and heavy → ship forward-only, backward via the JAX
  `lncc_grad` autodiff path; a backward kernel is a later, separately-gated build.
- **Parity:** interior-ULP-equal to `lncc_grad·∇warped` (the documented box-filter
  boundary divergence carries over — pin interior equality, assert the boundary
  contract).
- **Gate:** after Phase 1c; a 256³ SyN profile where LNCC dominates per-iteration HBM.
  Risk: the radius-`r` halo + two passes are the most complex of the three — prototype
  Pass 1 alone (the 5-sum accumulation) and measure before committing Pass 2.

---

## 3. Kernel 5c — fused MI histogram+gradient (Mattes MI)

**Bottleneck (§3.2):** *different shape* — not field round-trips but the joint-histogram
**scatter** (N→B² with atomics contention) + the per-voxel tiny-table gather. The
Phase-1a closed form already removes the autodiff **tape** (the dominant cost), so this
kernel addresses only the **residual forward-scatter contention at 256³** — **lowest
priority**, built only if §1a's 256³ scatter measurement shows it dominates.

### 3.0 Feasibility gate (do first)

Validate that the pinned-JAX Pallas supports the **shared-memory histogram** pattern
(in-tile accumulation + a reduction/atomic into the global B² table). If atomics /
shared-memory scatter are unavailable on `cuda12==0.10.1`'s Pallas, fall back to the
XLA `scatter_add` for Pass 1 (still tape-free via the closed form) and skip 5c — the
closed form alone is the bulk of the win.

### 3.1 Structure (two passes + a tiny host step)

- **Pass 1 — partial histograms.** Tiles over voxels; per tile soft-bin `warped`/`fixed`
  (`_soft_bin`), accumulate a **per-tile B² partial histogram in shared memory** (B=32 →
  1024 floats, fits), then reduce/atomic-add partials into the global `P (B²)`. Dodges
  global-atomic contention (the §3.2 256³ hotspot: 16.7M adds into 1024 buckets).
- **Host/small step.** Normalise `P`; `P_m = P.sum(1)`; the forward-difference table
  `D[a,b] = W[a+1,b] − W[a,b]` with `W = where(P>0, logP − logPm, 0)` (matching the cost,
  Phase 1a §2.1). Tiny (B²); resident.
- **Pass 2 — per-voxel gradient + fused warped gradient.** Tiles over voxels; soft-bin
  `warped`/`fixed`; gather `D[k,l]`, `D[k,l+1]` from the resident `D`; combine
  `g = (s_m/N)·[(1−t_f)D[k,l] + t_f·D[k,l+1]]` (zeroed outside `[lo_m,hi_m]`); compute
  `∇warped` in-tile; write `u = g·∇warped (M, ndim)`.
- **I/O:** Pass 1 reads `2M` writes `B²`; Pass 2 reads `2M` + the resident `D` writes
  `M·ndim`. The `D`-gather is from a tiny cache-resident table (not a field gather), so
  it does **not** hit the trilinear gather-lowering blocker.
- **Backward:** route through the JAX closed-form `mi_grad` autodiff path (the MI
  differentiable-layer is a niche; a backward kernel is not scoped).
- **Parity:** tolerance-equal to `mi_grad·∇warped` (Phase 1a), incl. the empty-bin
  convention.
- **Gate:** the Phase-1a §6 256³ scatter measurement showing the forward scatter
  dominates after the tape is gone; the §3.0 feasibility check. **Alternative lever**
  (cheaper, no kernel): ANTs-style **histogram subsampling** — build `P` from a random
  voxel subset, apply the gradient densely — cutting the scatter volume directly;
  evaluate this before committing 5c.

---

## 4. Priority, sequencing, and the build gate

| Kernel | Bottleneck | Priority | Predecessor | Gate |
|---|---|---|---|---|
| 5a ESM | stencil+force round-trips | **1st** | — | profile: ∇warped+force dominate HBM |
| 5b LNCC | ~27 conv passes (worst HBM) | 2nd | Phase 1c (integral image) | 256³ SyN profile; prototype Pass 1 first |
| 5c MI | forward scatter contention | 3rd (if needed) | Phase 1a (closed form) | §3.0 feasibility + 1a 256³ scatter measurement; consider subsampling instead |

- All three are **after the algorithmic levers** (Phase 1c/1d, Phase 2): those cut the
  *count* of ops the kernels would otherwise fuse, and may make a kernel unnecessary
  below brain scale. The kernels are the **256³ endgame**, not the first move.
- Each is independently gated and independently shippable (forward-first, backward later).
- The §5d trilinear gather (the universal warp kernel) remains parked on the pinned-JAX
  gather lowering — orthogonal to these (which are gather-free / scatter), and the higher-
  ROI gather-*count* reductions (Phase 2/3a) precede it.

## 5. Test plan (per kernel)

1. **Parity** (forward): ULP/tolerance-equal to the JAX force on random, flat, and
   boundary inputs (5a incl. the 0a guard; 5b interior-equal + boundary contract; 5c
   incl. empty-bin convention).
2. **Backward**: where a `custom_vjp` kernel exists, FD-matched; else assert the JAX
   backward fallback is wired and correct.
3. **Determinism / batch**: stable across tile sizes; correct under the cohort `vmap`.
4. **Scaling case**: perf-bench `nitrix-pallas` vs `nitrix-jax` on 96³–256³ single +
   cohort — record HBM-per-voxel, the crossover, and the win (or its absence, which is
   also a recorded number, not a guess).

## 6. Cross-references

- [`registration-suite-v4.md`](registration-suite-v4.md) §3.2 (the HBM analysis driving
  the kernel choice), Phase 5; `pallas-demons-esm-force.md` (5a, the Trigger),
  `pallas-trilinear-resample.md` (5d, the parked gather), `perf-wins-must-certify-at-scale.md`.
- [`registration-suite-v4-phase1-mi-force.md`](registration-suite-v4-phase1-mi-force.md)
  (the closed form 5c accelerates the scatter of).
- `src/nitrix/_kernels/cuda/` (the existing Pallas/Triton home + dispatch scaffold),
  `register/_force.py` (the JAX oracles), `metrics/{intensity,information}.py`,
  `geometry/differential.py` (`spatial_gradient` — the stencil 5a/5b fuse).
