# B18. perf-bench case-hardening index — untested "hard" dispatch branches

> **Status (2026-06-05): open (benchmark-integrity index).** A
> recommendation *to* `nitrix-perf-bench`, authored nitrix-side and kept at
> arm's length on purpose: the optimiser (nitrix) proposes the hard-path cases,
> the suite maintainer disposes (COVERAGE_MANDATE §5 — the suite owns its cases;
> nitrix never gains a perf-bench dep). Provenance: surfaced formulating the
> GPU-deficit optimisation plan; the optimisations introduce or exploit
> fast-path *branches*, and the current cases exercise only one branch each.

## Why this exists

Several planned optimisations add or target a **dispatch branch** — a fast path
for the common case (flat structuring element, Euclidean metric, GPU default
backend, dense format). The current perf-bench case for each op exercises
**only that one branch**. That makes the benchmark *gameable*: an optimiser can
turn a case green by handling the easy path while the hard path stays slow — or,
worse for the no-oracle cases, *wrong*. This index lists, per op, the branch the
current case covers, the untested hard branch the optimisation creates or
exposes, and the new case(s) that close the hole so the suite measures what it
appears to measure.

It maps onto the mandate's own four-axis gap (COVERAGE_MANDATE §1.1:
breadth / platform / precision / reference-quality) and adds a fifth concern the
audit did not name explicitly: **intra-op branch coverage** — when one op has
multiple algorithmic paths, the case must pin the one users actually hit (the
*default*), not a convenient pinned variant.

## Index

| op | current case covers | untested hard branch (gameable) | proposed new case(s) |
|---|---|---|---|
| `distance_transform` | defaults, binary mask, **atol=1.0 voxel** | (a) chamfer `metric=` path now diverges from the fast default; (b) the loose atol was a chamfer crutch — it now hides an *exact-EDT* regression | tight-atol euclidean case (atol≈1e-4); `metric='chebyshev'`/`'city_block'` cases vs `scipy.ndimage.distance_transform_cdt`; anisotropic spacing; **256³** (the scale where the JAX-engine vs Pallas/cupy gap and the brute-force compile-cliff actually bite) |
| `sosfilt` | **pinned `backend='scan'`** headline | the new **GPU default** (no kwarg) is never measured; high order + long series stress the parallel recurrence's conditioning | a **no-kwarg default** row (measures what users get, per platform); order-8; obs≥32768; fp32-vs-fp64 accumulation |
| `sosfiltfilt` | default (scan-only), order-4 | post-change the zero-phase parallel path; the assoc rel_to_tol is already ~30× the scan's (0.004 vs 0.0001) | order-8 + long-series **fidelity** guard on the parallel formulation |
| `erode` / `dilate` | **flat box `size=3`, 2D** | the **non-flat (grayscale) SE** path (semiring) that a `reduce_window` fast path bypasses; larger/3D windows | non-flat `structure=` vs scipy `grey_*(structure=)`; size=5/7; 3D; fp16 |
| `open` / `close` | **unmeasured** | two-pass morphology compounds a slow erode/dilate | coverage-tier cases (compose the above) |
| `median_filter` | `size=3`, **no oracle (perf-only)** | *most gameable in the suite* — a fast-but-wrong median ships green; size>3 and border semantics | add an **interior-matched fidelity oracle**; size=5; 3D; pin the NaN-shrink border (B13) |
| `intensity_normalize` | global percentile (axis=None) | per-axis reduction; the CPU sort cliff (B17) | per-axis/per-slice case; keep the CPU platform row as the B17 signal |
| `laplacian_eigenmap` / `diffusion_embedding` | lobpcg/eigh, **n≤2048**, well-separated spectrum, dense | scaling past dense-eigh feasibility; **degenerate/clustered spectrum** (stresses the implicit-VJP `eps_clamp` + preconditioner); sparse formats; the gradient path | n=4096/8192; a near-degenerate-spectrum input; ELL/SectionedELL variants; a value+grad timing case (correctness stays in nitrix tests) |
| `degree_vector` / `laplacian` | **dense only** | the ELL / **SectionedELL** paths (Python-loop scatter; different perf) | ELL + SectionedELL variants |
| `spatial_transform` | **in-bounds-clipped** deformation only | out-of-bounds + each boundary `mode=` (B15) | OOB deformation per `mode`, oracle-matched per mode |
| `sphere_grid_unpad_2d` | **isolated slice** (vs a zero-copy view → misleading 26×) | the realistic **fused** cost (~0) | a `pad → valid-kernel → unpad` composed case so the number reflects real usage |

## Cross-cutting hardening (the mandate's open axes)

- **Precision (§2.5; today: 100% f32).** Add fp16/bf16 sub-sweeps where reduced
  precision is realistic and the fp64 oracle already supplies truth: morphology
  (min/max, precision-robust), EDT, the kernel/distance ops. Exposes the
  perf↔fidelity tradeoff that f32-only coverage hides.
- **Default-vs-pinned.** Wherever an op has an algorithm/backend **default**
  (IIR backend, `distance_transform` metric, morphology flat-vs-any SE), the
  case must measure the *default* path. A pinned-variant headline lets a
  default-only change show "no movement" and a default-only regression hide.
- **Format-variant branches.** Graph ops fan out dense / ELL / SectionedELL;
  only dense is measured. Same for any op whose ELL path is a distinct kernel.
- **Boundary parity (B13).** Morphology / median / interpolation each carry a
  boundary contract that is largely unpinned; a fast path that quietly changes
  the border is a silent regression today.

## Trigger

Author each new case *alongside* the corresponding optimisation PR, so the
hard-path bar lands in the same change that creates the fast path (a
*ships-with-a-hard-case* discipline, mirroring the mandate's §7-D
ships-with-a-case SLA). The euclidean-EDT tolerance tightening and the
`median_filter` oracle are the two highest-value (they convert a currently
**gameable-green** case into a real gate) and should land first.

## Cross-references

- `../../nitrix-perf-bench/COVERAGE_MANDATE.md` §1.1 (four-axis gap), §2.4
  (per-op targets), §2.5 (precision), §5 (separation of concerns), §7 (SLA).
- [`iir-filter-gpu-backend`](iir-filter-gpu-backend.md) (B12),
  [`spectral-embedding-gpu-solver`](spectral-embedding-gpu-solver.md) (B14),
  [`interpolation-backend-cpu-gpu-gap`](interpolation-backend-cpu-gpu-gap.md)
  (B15), [`median-percentile-cpu-sort-cliff`](median-percentile-cpu-sort-cliff.md)
  (B17), [`boundary-mode-parity`](boundary-mode-parity.md) (B13),
  `../design/perf-audit-2025-05.md` (the EDT metric-mismatch record).
- [`internal-backlog.md`](internal-backlog.md) — ledger index (add B18 pointer).
