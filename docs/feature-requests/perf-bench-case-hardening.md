# B18. perf-bench case-hardening report — where the GPU-deficit wins can be gamed

> **Status (2026-06-06): open — post-implementation report *to*
> `nitrix-perf-bench`.** Supersedes the 2026-06-05 planning-era index (same
> number): every win below is now **shipped**, so this rewrite is grounded in
> the actual dispatch branches and accuracy seams rather than a plan. Authored
> nitrix-side and kept at arm's length on purpose (COVERAGE_MANDATE §5 — the
> suite owns its cases; nitrix proposes hard-path cases, the suite maintainer
> disposes; nitrix never takes a perf-bench dependency).
>
> **Who should read this:** the perf agent maintaining `nitrix-perf-bench`.
> Each section ends with concrete case specs (sizes, dtypes, accuracy gates)
> that you can drop in. Treat the priority ordering at the end as the
> highest-leverage subset if you only land a few.

## TL;DR

I optimised the five ops the perf audit flagged as 1–2 orders off cupy/scipy:
`distance_transform`, `sosfilt`, `sosfiltfilt`, `erode`/`dilate`, and the
spectral-embedding eigensolver (`laplacian_eigenmap` / `diffusion_embedding`).
**Every win lives on a dispatch branch.** The fast path handles the *common,
easy* shape (flat box SE, Euclidean metric, GPU-default FFT backend, dense
operator, well-separated spectrum); the *hard* branch beside it is either slow,
absent, or accuracy-dependent. The current perf-bench case for each op exercises
**only the easy branch** — so a green case today does **not** mean the op is
fast on the workloads users actually run. This report names, per win, the seam
and the realistic case that closes it.

The single most important cross-cutting point: **for iterative / truncated ops
(eigensolver, FFT-IIR), wall-time without a *pinned accuracy target* is
meaningless.** Half my eigensolver "speedup" is an early-stop at lower accuracy;
it is only a win if the suite gates the accuracy it was measured at.

---

## Part 1 — Cross-cutting gaming vectors (apply to every op)

These are the JAX-vs-cupy/scipy harness seams. If any is unguarded, *all* the
numbers below are suspect regardless of per-op coverage.

1. **Async dispatch — the timer must block.** JAX dispatches asynchronously: a
   kernel call returns a future immediately. A timing loop without
   `jax.block_until_ready(out)` (or `out.block_until_ready()`) measures
   *dispatch latency*, not compute — every op looks ~instant and a 100× "win"
   is just a deeper queue. **Guard:** assert the harness blocks on every timed
   JAX output; spot-check by timing a deliberately huge matmul and confirming
   the number scales with size.

2. **Compile vs steady-state — measure and report both.** JAX JIT-compiles on
   first call (often 100×–1000× the steady-state time). Timing only the first
   call measures the compiler; timing only warmed-up calls hides a real
   first-call latency that bites short-lived processes (a CLI invocation, a
   per-subject pipeline step). **Guard:** report compile time and steady-state
   as separate numbers; never average them.

3. **Constant-folding / DCE.** If inputs are compile-time constants or the
   output is discarded, XLA can fold or dead-code-eliminate the whole
   computation, timing nothing. **Guard:** feed inputs as traced args (not
   literals), consume the output (return/reduce it), and sanity-check that the
   measured time is non-trivially above dispatch overhead.

4. **Dtype parity.** nitrix defaults to fp32; scipy/cupy frequently run fp64.
   An fp32-vs-fp64 comparison is a silent ~2× (and a different accuracy class).
   **Guard:** pin both sides to the same dtype, and add an explicit fp64 row
   wherever the baseline is fp64 by default (notably the eigensolver and IIR).

5. **Accuracy must be pinned for iterative/truncated ops.** The eigensolver
   (lobpcg / shift-invert / polynomial) and the IIR FFT engine trade accuracy
   for speed via a tolerance (`lobpcg_tol`) or truncation (`impulse_atol`). A
   loose correctness gate lets a fast-but-wrong config ship green. **Guard:**
   gate these at a *tight, stated* accuracy (e.g. eigenvalue rel-err ≤ 1e-5 or
   subspace angle ≤ 1e-4; IIR max-abs-err ≤ 1e-8 vs an fp64 `lfilter` oracle)
   and print the accuracy *next to* the time. A time without its accuracy is
   not a comparable number.

6. **Default-vs-pinned dispatch.** Several wins are the *default* path
   (`backend='auto'` → FFT on GPU; `metric='euclidean'`; flat-box morphology).
   A case that pins a non-default kwarg (e.g. `backend='scan'`) measures a path
   users don't hit on that platform — a default-only regression then hides, and
   a default-only improvement shows "no movement." **Guard:** the headline row
   for each op must call it the way a user does (no engine kwarg), per platform.

7. **Broken-cuSolver artifacts inflate the eigensolver win.** On this L4 the
   dense `eigh` path intermittently fails with `gpu_solver_unavailable`
   (cuSolver handle wedge). When the *baseline* (`nitrix-jax-eigh`, and any
   cuSolver-backed comparison) errors out, the iterative path looks infinitely
   better — but that is an environment defect, not an algorithmic win. **Guard:**
   record stack health per run; do not credit a speedup against a baseline that
   is *failing* rather than *running*. Re-bench on a healthy cuSolver stack.

8. **Device-transfer accounting.** If nitrix keeps data resident on the GPU but
   the baseline pays a host↔device copy (or vice-versa), the transfer dominates
   small sizes. **Guard:** account transfers identically on both sides; state
   whether the timed region includes them.

---

## Part 2 — Per-win seams and the cases that close them

### Win 1 — `distance_transform` (Euclidean EDT via min-plus matmul)

**What shipped.** `metric='euclidean'` is now the default and is **exact**: the
separable squared-EDT `out[p] = min_q (g[q] + (q−p)²)` is expressed as a tropical
`(min,+)` matmul `(lines, n) @ D2(n, n)` against `D2[q,p]=(q−p)²`, run per axis
on the semiring kernel. This replaced the old control-flow-bound chamfer default
(~18 ms → ~0.24 ms at the bench size; parity with cupy, exact). Chamfer
(`metric='chebyshev'/'city_block'` or a custom `structuring_element`) is retained
but opt-in and approximate.

**Where it can be gamed / what's uncovered:**

- **The matmul is O(n²) per axis; Felzenszwalb separable EDT is O(n).** The win
  is **size-dependent**: at small/moderate side length (the doc validates
  parity at 64³) the matmul's constant factor and GPU-friendliness win; as the
  side length grows the n² term overtakes the O(n) reference and a compile/memory
  cliff appears (the D2 matrix is n×n). A case pinned at one small size is
  gameable — it certifies "fast EDT" while the asymptotic is *worse*. **Case:**
  sweep side length to **256³** (and 512² in 2D) so the crossover is visible; if
  cupy/scipy pull ahead there, that is the honest picture.
- **Anisotropic voxel spacing is unsupported, not just untested.** The euclidean
  engine bakes unit spacing into `D2=(q−p)²` — there is **no `sampling=`
  parameter**. Real medical volumes are anisotropic (e.g. 1×1×3 mm MRI). scipy's
  `distance_transform_edt(sampling=...)` handles this; nitrix would be wrong
  unless the caller pre-rescales. **Case:** an anisotropic-spacing input checked
  against `scipy.ndimage.distance_transform_edt(sampling=(...))`; expect it to
  surface the gap (drives a feature request, not just a perf row).
- **Batched EDT silently leaks across the batch axis.** `_distance_transform_edt`
  treats **every** array axis as spatial (scipy convention). Passing a stack of
  N 2D masks as an (N,H,W) array computes a *3D* transform — distances bleed
  between images. There is no batch dimension. **Case:** a "batch of 2D images"
  input, asserting the result equals a per-image loop; this catches the trap and
  measures the (currently absent) batched path that ML pipelines actually want.
- **Degenerate masks.** All-zero / all-foreground masks hit trivial paths.
  **Case:** ensure the benched mask has a realistic interior/boundary ratio so
  the number reflects genuine work, not a short-circuit.
- **Metric default-vs-pinned + tolerance.** The historical case used a loose
  `atol≈1.0 voxel` — a chamfer-era crutch that now *hides an exact-EDT
  regression*. **Case:** tighten the euclidean case to `atol≈1e-4`; add separate
  `chebyshev`/`city_block` rows vs `scipy.ndimage.distance_transform_cdt` so the
  approximate engine is measured as approximate.

### Win 2 — `sosfilt` / `sosfiltfilt` (FFT-convolution IIR on GPU)

**What shipped.** `backend='auto'` resolves to **`'fft'` on GPU**, `'scan'` on
CPU. An IIR filter is LTI, so its output is exactly convolution with the impulse
response; the FFT engine truncates the (geometrically decaying) impulse at
`impulse_atol` (default 1e-12, effectively exact) and convolves via rFFT —
parallel, latency-free on the GPU. Measured ~1.9× over cupy on `sosfilt`, ~5.3×
on `sosfiltfilt`. Edge transients (`zi`) are recovered as `x[0]·g` (zero-input
response).

**Where it can be gamed / what's uncovered:**

- **The current case pins `backend='scan'` — it never measures the GPU
  default.** The FFT engine (what every GPU user gets) is unbenched. **Case:** a
  no-kwarg `backend='auto'` headline row per platform.
- **Sharp filters silently fall back to the recurrence.** A filter whose impulse
  hasn't decayed below `impulse_atol` within `2¹⁵=32768` taps (poles near
  |z|=1: high-Q resonators, narrow notches) falls back to `associative` (GPU) /
  `scan` (CPU) **with a warning** — the FFT win does *not* apply. A bench using
  only well-damped filters always hits the fast path and over-reports. **Case:**
  include a **high-Q line-noise notch** (e.g. 60 Hz, Q≈35, `iirnotch`) and a
  near-unstable filter (pole radius 0.999): these are exactly the EEG/MEG and
  audio filters practitioners run, and they exercise the realistic fallback
  cost + assert the fallback fires (catch the warning).
- **FFT cost scales with impulse length → with filter order and pole radius.** A
  single low-order well-damped section has a short impulse (cheap FFT); an
  order-8 cascade near the passband edge has a long one (up to the 32768 cap).
  **Case:** sweep order (2/4/8) and pole radius; obs ≥ 32768 (where FFT wins)
  *and* a short-signal row (where `scan` is competitive — verifies the `auto`
  switch picks correctly).
- **Truncation accuracy rides on `impulse_atol`.** Loosening it shrinks the FFT
  and speeds it up at the cost of geometrically-bounded edge error. **Case:** a
  fidelity guard at the default `impulse_atol=1e-12` against an fp64 `lfilter`
  oracle (max-abs-err ≤ 1e-8); optionally a second row at a loose atol to *show*
  the perf↔fidelity tradeoff rather than let it hide.
- **fp32 FFT precision.** rFFT/irFFT in fp32 loses precision vs scipy's fp64
  `lfilter`; the `zi`/`g` host constants are fp64 but the transform is fp32 on
  fp32 input. **Case:** an fp32-vs-fp64 accuracy row.
- **`sosfiltfilt` edge handling.** Zero-phase forward-backward only supports
  `padtype='odd'` (scipy-matched); the doubled pass + odd-extension padding is
  where transient bugs hide, and the associative path's rel-to-tol was already
  ~30× the scan's. **Case:** order-8 + long-series *fidelity* guard on the
  forward-backward result (not just timing) vs `scipy.signal.sosfiltfilt`.
- **Multichannel.** Real EEG/MEG filtering is 64–306 channels at once; nitrix
  vectorises over channels while a naive baseline loops. **Case:** a multichannel
  row — it both shows a genuine nitrix win and stops a per-channel baseline from
  looking artificially slow if the harness loops it.

### Win 3 — `erode` / `dilate` (flat-box reduce_window fast path)

**What shipped.** For a **flat box** structuring element (`structuring_element=
None`, the default), dilation lowers to a fused `lax.reduce_window` sliding-window
max and erosion to a sliding-window min — XLA lowers it like a pooling op, far
cheaper than the `semiring_conv` im2col + tropical-matmul path. SAME padding
uses the algebra identity (±inf), matching the semiring path bit-for-bit.
Measured ~1.3× over cupy.

**Where it can be gamed / what's uncovered — this is the win the user flagged
("custom structuring elements"):**

- **The fast path fires *only* for `structuring_element=None`.** **Any explicit
  SE — even a flat custom shape — routes through the slow semiring path.** That
  means:
  - **Custom-shaped flat SEs (disk, cross, ring) get no speedup.** A disk/ball
    footprint is the *default* in skimage and the common choice in scipy
    (`disk(r)`, `ball(r)`); passing it as `structuring_element=<binary mask>`
    bypasses `reduce_window` entirely. A bench that only uses flat boxes
    (`structuring_element=None`) certifies "fast morphology" while the
    footprint users actually pick is on the slow branch. **Case:** an
    `erode`/`dilate` row with a **binary disk** SE vs `scipy.ndimage`
    `grey_erosion(footprint=disk(r))` — this is the single most important
    morphology gap.
  - **Non-flat (grayscale) SEs route to semiring too.** Rolling-ball background
    subtraction and grayscale morphology use additive (non-flat) SEs. **Case:**
    a non-flat `structure=` row vs scipy `grey_erosion(structure=...)`.
- **Border parity is not scipy's.** SAME + ±inf is nitrix's convention; scipy's
  default border (`reflect` for grey, `border_value` for binary) differs. A
  timing win with mismatched edges is a *silent correctness regression* at the
  boundary (B13). **Case:** pin and oracle-match the border mode; a fast path
  that quietly changes the border must fail correctness, not just pass timing.
- **Large windows / separability.** `reduce_window` over a (k,k) box is ~O(k²)
  per output — XLA does **not** auto-separate min/max windows into two 1D
  passes. Big SEs (size 15+, or 3D balls) scale with window *volume*. **Case:**
  size 5/7/15 and a **3D** flat box (volumetric morphology on MRI), to show the
  window-volume scaling and confirm the 3D path.
- **`open` / `close` are unmeasured.** Two-pass morphology compounds the slow
  branch when an explicit SE is used. **Case:** composed `open`/`close` rows
  (they inherit whichever branch their `erode`/`dilate` hit).
- **Precision.** min/max are precision-robust, so fp16/bf16 sub-sweeps are
  realistic and the fp64 oracle already supplies truth. **Case:** an fp16 row.

### Win 4 — spectral eigensolver (`laplacian_eigenmap` / `diffusion_embedding`)

**What shipped.** Four selectable paths: dense `eigh` (full spectrum,
differentiable, but cuSolver-wedge-prone); plain `lobpcg` (matrix-free,
differentiable via implicit-VJP, the only reliable GPU path, works on dense /
ELL / SectionedELL); `shift_invert` (dense-only, inner-CG accelerated, ~1e-3
accuracy); and a `polynomial` preconditioner on the lobpcg path (dense-only,
matvec-only spectral filter, ~1e-3 accuracy). `lobpcg_tol` now defaults to 1e-7
(early-stop). The 12×-off-cupy plain-lobpcg gap narrows to ~2× (shift-invert) /
~1.5–2× (polynomial) at n=1024. `safe_eigh` adaptively latches to CPU on a
cuSolver failure.

**Where it can be gamed / what's uncovered — this is the most accuracy-sensitive
win:**

- **Half the speedup is lower accuracy — it is only real if gated.** The
  `lobpcg_tol=1e-7` early-stop and the ~1e-3 preconditioned paths are *faster
  because they converge less far* than cupy `eigsh` (which runs to ~1e-10). A
  2× at 1e-3 is not comparable to a baseline at 1e-10. **Case:** gate eigenvalue
  rel-err ≤ 1e-5 (or subspace principal angle ≤ 1e-4) and print accuracy beside
  time; add rows at matched accuracy so the comparison is apples-to-apples. This
  is the highest-value guard in the whole suite.
- **The preconditioned wins are dense-only.** `shift_invert` and `polynomial`
  raise `ValueError` on ELL / SectionedELL. The *entire point* of lobpcg is
  sparse scale (n ~ 1M); the headline ~1.5–2× numbers do **not** apply to the
  sparse path. A bench reporting them as "the eigensolver" win, on dense
  n=1024, misses the regime that motivates the op. **Case:** ELL and
  SectionedELL rows on the *plain* lobpcg path at n ≫ dense-eigh feasibility
  (4k/8k/100k), kept distinct from the dense preconditioned rows.
- **Convergence is spectrum-dependent — bench a realistic spectrum.** Iterative
  solvers converge fast on a well-separated spectrum and slowly on a clustered /
  near-degenerate one. A random/expander graph has a gap; a real connectome has
  **community structure → tightly clustered low eigenvalues → small spectral
  gap**, which stresses both convergence and the implicit-VJP `eps_clamp`. A
  bench on an easy graph over-reports. **Case:** a community-structured /
  near-degenerate-spectrum graph (e.g. a planted-partition / SBM Laplacian).
- **The differentiability cost is the whole reason lobpcg is the default — and
  it's unmeasured.** cupy/scipy `eigsh` have **no gradient**; nitrix pays for
  the implicit-VJP backward. Comparing differentiable-nitrix to
  non-differentiable-eigsh on *forward time only* is apples-to-oranges in
  nitrix's favour on capability but against it on raw speed. **Case:** a
  value-and-grad timing row (correctness of the grad stays in nitrix's own
  tests — see the x64-grad fix); note in the report that the baselines provide
  no gradient at all.
- **k and seed dependence.** The win shrinks as k (number of components) grows
  (large k favours dense eigh) and varies with the random initial subspace.
  **Case:** sweep k (8/32/100); fix the seed and report variance across a few
  seeds.
- **cuSolver-wedge artifact (see cross-cutting #7).** On the broken stack the
  `eigh` baseline reports `gpu_solver_unavailable`; don't let that inflate the
  lobpcg win. **Case:** record stack health; re-bench on a healthy stack where
  dense eigh actually runs (and may *win* at small n).
- **x64 path.** The grad dtype bug only bit under `jax_enable_x64` (now fixed:
  the lobpcg initial subspace matches the operator dtype). If the suite ever
  runs x64, include a row so a regression there is caught.

---

## Part 3 — Consolidated case index

| op | current case covers (easy branch) | gameable / uncovered hard branch | proposed case(s) |
|---|---|---|---|
| `distance_transform` | euclidean default, small size, loose atol | O(n²)/axis size cliff; no `sampling=` (anisotropic); batched-as-3D leak; chamfer divergence | 256³ scaling; anisotropic vs scipy `sampling=`; batch-of-2D vs per-image loop; tight-atol euclidean (1e-4); `chebyshev`/`city_block` vs `distance_transform_cdt` |
| `sosfilt` | pinned `backend='scan'` | GPU-default FFT unmeasured; sharp-filter fallback; order/pole-radius scaling; `impulse_atol` fidelity; fp32 | no-kwarg `auto` per platform; high-Q notch + pole-radius 0.999 (assert fallback); order 2/4/8; obs≥32768 + short-signal; fp32-vs-fp64; multichannel |
| `sosfiltfilt` | default, order-4, timing-only | zero-phase forward-backward fidelity; assoc rel-to-tol ~30× scan | order-8 long-series *fidelity* guard vs scipy |
| `erode` / `dilate` | flat box `size=3`, 2D | **custom-shaped flat SE (disk) → slow semiring**; non-flat SE; border-mode parity (B13); large/3D windows | **disk SE vs `grey_erosion(footprint=disk)`**; non-flat `structure=`; border oracle-match; size 5/7/15; 3D box; fp16 |
| `open` / `close` | unmeasured | compounded slow branch under explicit SE | composed rows over the above |
| `laplacian_eigenmap` / `diffusion_embedding` | lobpcg/eigh, n≤2048, well-separated, dense, forward-only | **accuracy not pinned**; preconditioned wins dense-only; clustered spectrum; grad cost; k/seed; cuSolver artifact | accuracy-gated rows (rel-err ≤1e-5); ELL/SectionedELL at n=4k–100k; SBM/clustered-spectrum input; value+grad timing; sweep k; healthy-cuSolver re-bench |
| `degree_vector` / `laplacian` | dense only | ELL / SectionedELL scatter paths | format-variant rows |

---

## Part 4 — Priority (if you land only a few)

These convert a currently **gameable-green** case into a real gate — land first:

1. **Pin accuracy on the eigensolver** (cross-cutting #5 + Win 4). Without it the
   whole spectral-embedding comparison is uncalibrated. Highest leverage.
2. **Verify the harness blocks on async outputs** (cross-cutting #1). If it
   doesn't, *every* JAX number in the suite is dispatch latency.
3. **Disk-SE morphology row** (Win 3). The default footprint users pick is on
   the slow branch; the box-only case hides it.
4. **High-Q-notch IIR row + no-kwarg default** (Win 2). Measures the realistic
   filter and the engine users actually get on GPU.
5. **Tight-atol euclidean EDT + 256³ scaling** (Win 1). Closes the loose-atol
   crutch and exposes the O(n²) crossover.

A discipline to adopt: author each hard-path case *alongside* the optimisation
PR it guards (ships-with-a-hard-case, mirroring the mandate's §7-D
ships-with-a-case SLA), so the fast path can never land without its bar.

## Cross-references

- `../../nitrix-perf-bench/COVERAGE_MANDATE.md` §1.1 (four-axis gap), §2.4
  (per-op targets), §2.5 (precision), §5 (separation of concerns), §7 (SLA).
- Shipped optimisations: [`iir-filter-gpu-backend`](iir-filter-gpu-backend.md)
  (B12), [`spectral-embedding-gpu-solver`](spectral-embedding-gpu-solver.md)
  (B14), and the EDT / morphology perf work on the `perf/*` branches.
- Related open contracts: [`boundary-mode-parity`](boundary-mode-parity.md)
  (B13), [`interpolation-backend-cpu-gpu-gap`](interpolation-backend-cpu-gpu-gap.md)
  (B15), [`median-percentile-cpu-sort-cliff`](median-percentile-cpu-sort-cliff.md)
  (B17), `../design/perf-audit-2025-05.md` (the EDT metric-mismatch record).
- [`internal-backlog.md`](internal-backlog.md) — ledger index (B18 pointer).
