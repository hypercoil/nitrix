# B21. Explicit-SE morphology (disk/ball footprint) pays the im2col path

> **Status (2026-06-07): open — perf characterisation from `nitrix-perf-bench`.**
> The *measured* confirmation of the gap [`perf-bench-case-hardening.md`](
> perf-bench-case-hardening.md) (B18, Win 3) predicted in the abstract:
> "Custom-shaped flat SEs (disk, cross, ring) get no speedup … the single most
> important morphology gap." Authored perf-bench-side (COVERAGE_MANDATE §5);
> nitrix disposes.

## TL;DR

`erode`/`dilate`/`open`/`close` lower a **flat box** (`structuring_element=
None`) to a fused `lax.reduce_window` — the fast path, which **beats cupy
1.2–3.0×** on the L4. But **any explicit structuring element** routes through
`semiring_conv` (im2col patches + a tropical matmul), and the disk/ball
footprint — the *default* footprint in skimage (`disk(r)`/`ball(r)`) and the
common scipy choice — is exactly the explicit-SE case. On that branch nitrix is
**1.5–11× slower than cupy** and materialises a **100–350× larger** HBM
high-water mark (the im2col patch tensor), scaling with window *volume*.

The fidelity is exact (the comparison is warranted: same op, same border —
scipy `mode='constant', cval=±inf` matches nitrix's SAME+identity by
construction; see the perf-bench cases). This is a **pure dispatch / kernel**
cost, not a correctness issue.

## Measured (L4, fp32 unless noted; fidelity ✓ exact at every point)

`dilate` (erode/open/close mirror it; composed ops compound the box win and the
disk loss):

| structuring element | path | nitrix steady | cupy steady | nitrix vs cupy | nitrix peak HBM |
|---|---|---|---|---|---|
| box size 3, 256² | reduce_window | 95.5 µs | 125 µs | **0.76× (nitrix wins)** | 0.79 MB |
| box size 3, 64³ | reduce_window | 100 µs | 178 µs | **0.56× (nitrix wins)** | 3.15 MB |
| disk r=3, 256² | semiring (im2col) | 368 µs | 301 µs | 1.22× slower | **93 MB** |
| disk r=7, 256² | semiring (im2col) | 1.58 ms | 309 µs | **5.1× slower** | **193 MB** |
| ball r=2, 64³ | semiring (im2col) | 3.07 ms | 324 µs | **9.5× slower** | **336 MB** |

(`erode` ball r=2 64³: **11× slower**, 336 MB. `open`/`close` ball r=2 64³:
~11× slower, 336 MB — the two passes peak at one im2col tensor. cupy holds
0.26–1.05 MB across all of them.)

The headline number **flips with the footprint**: a box-only bench reports
"nitrix morphology beats cupy ~1.3×"; the disk/ball footprint users actually
pick is 5–11× the other way, with a memory blow-up that will OOM 3-D volumetric
pipelines (336 MB for a single 64³ ball op; a 256³ ball would be far worse).

## Root cause

`dilate`/`erode` take the fused path **only** when `structuring_element is
None` (`_mm.py`: the `_windowed_reduce` branch). Every explicit SE — *including
a flat footprint that is morphologically identical to a masked box* — falls to
`_conv_wrap → semiring_conv`, whose im2col stage gathers a
`(*spatial, prod(kernel))` patch tensor before the tropical matmul. That tensor
is the 100–350× HBM term, and XLA does not separate the (k,k) / (k,k,k) window
into 1-D passes, so cost scales with window *volume*.

A flat footprint (disk/ball/cross/ring: participating cells contribute 0, the
rest are masked) needs none of that: it is a windowed min/max over the footprint
— the *same* primitive the box fast path already uses, plus a footprint mask.

## Suggested directions (nitrix disposes)

1. **Generalise the fast path to any *flat* footprint.** Detect a flat SE
   (finite entries all equal, `-inf`/`+inf` elsewhere) and lower it to
   `reduce_window` over the bounding box with the off-footprint cells set to the
   algebra identity — a masked windowed reduce, no im2col. This covers
   disk/ball/cross/ring (the bulk of real footprints) and inherits the fast
   path's `jit(grad)` cleanliness (B19) and tiny HBM.
2. **Separable flat boxes already on the fast path** are fine; for **non-flat
   (grayscale) additive** SEs that are rank-1 separable, decompose into per-axis
   passes. Otherwise the im2col path is the fallback.
3. **At minimum, bound / document the im2col memory.** A 64³ ball at 336 MB is a
   silent OOM risk for volumetric morphology; document that explicit SEs are
   memory-heavy and prefer the flat-box path until (1) lands — mirroring the
   ships-with-an-honest-contract discipline.

## Perf-bench follow-through (already landed)

The hardened `erode`/`dilate`/`open`/`close` cases now span **both** SE shapes
as param points (box fast path + disk/ball semiring path), so this gap is
measured, not hidden by a box-only bench. The op's *representative* point stays
the flat box (where nitrix legitimately wins, so it remains a strong GPU ref);
the disk/ball rows carry the regression as visible non-representative perf rows.
A perf win on the box branch can no longer paper over the disk/ball branch.

## Cross-references

- [`perf-bench-case-hardening.md`](perf-bench-case-hardening.md) (B18, Win 3 —
  predicted this; the disk-SE row is item 3 of its checklist).
- [`boundary-mode-parity.md`](boundary-mode-parity.md) (B13 — the border the
  perf-bench oracle pins so the comparison is the same op).
- [`morphology-reduce-window-jitgrad.md`](resolved/morphology-reduce-window-jitgrad.md)
  (B19 — the flat-box fast path whose reach (1) would extend).
- perf-bench: `reports/PERF_{DILATE,ERODE,OPEN,CLOSE}.md`,
  `src/nperf/cases/{dilate,erode,opening,closing,_morphology}.py`.
- [`internal-backlog.md`](internal-backlog.md) — ledger index (B21 pointer).
