# `volreg` CPU path: batched-all-frames kernel is 3–6× slower than mcflirt/3dvolreg AND uses O(T·N) memory (~5 GB) — needs a streaming/chunked CPU kernel — `nitrix.register`

> **Status (2026-06-30): OPEN.** Surfaced by **nitrix-perf-bench** regenerating
> the registration suite (nitrix `b09f40c`, L4 host). The GPU path wins but
> scales poorly; the **CPU** path is both slower and far heavier than the
> 20-year-old C standards it should beat. File under the **Registration suite**
> family. *(Re-filed after an instance rollback lost the original commit.)*

**What.** `nitrix.register.volreg` on `jax-cpu` is **3–6× slower** than the
community motion-correction standards (FSL `mcflirt`, AFNI `3dvolreg`) at
realistic frame counts, and its host RSS grows **~O(T·N)** to multiple GB,
whereas `mcflirt`/`3dvolreg` register **frame-by-frame** at roughly constant,
low memory.

Speed (steady min, 48³ unless noted):

| size, T | nitrix CPU | nitrix GPU | mcflirt | 3dvolreg | nitrix-CPU slowdown |
|---|---|---|---|---|---|
| 48³ T=100 | 17.5 s | 0.16 s | 8.1 s | 4.0 s | 2.2× / 4.4× |
| 48³ T=200 | 37.3 s | 0.34 s | 11.9 s | 7.3 s | 3.1× / 5.1× |
| 48³ T=500 | 112.5 s | 0.89 s | 29.1 s | 17.9 s | 3.9× / 6.3× |
| 64³ T=100 | 45.3 s | 0.42 s | 14.3 s | 9.1 s | 3.2× / 5.0× |
| 80³ T=100 | 98.6 s | 1.05 s | 26.9 s | 16.9 s | 3.7× / 5.8× |

Memory (nitrix-CPU `host_rss`, grows with T·N):

| point | host_rss |
|---|---|
| 32³ T=8 | 639 MB |
| 48³ T=200 | 2530 MB |
| 64³ T=100 | 2859 MB |
| 80³ T=100 | 4691 MB |
| 48³ T=500 | **5078 MB** |

`mcflirt`/`3dvolreg` stream one frame at a time, so their footprint is ~constant
(hundreds of MB) regardless of T. (The 5 GB figure is not theoretical — it
helped drive a 15 GB shared bench host into the OOM killer.)

**Why it matters.**

1. **CPU is the common deployment.** A user motion-correcting a long fMRI run on
   a machine without a GPU is *worse off* with nitrix than with a 20-year-old C
   tool — on **both** axes (slower **and** heavier). For a marquee op that is a
   poor result.
2. **The memory growth is a hazard.** O(T·N) RSS reaching multiple GB at ordinary
   sizes (80³, T=100 → 4.7 GB; T=500 → 5 GB) risks OOM on constrained or shared
   hosts — exactly where a streaming tool is safe.
3. **The GPU path wins today, but is the weakest-scaling registration — measure
   it on any change.** nitrix volreg on GPU is faster than mcflirt/3dvolreg
   (T=500: 0.89 s vs 29 s / 18 s), so the batched design pays off where the
   hardware parallelism amortizes the materialization. **But the GPU win erodes
   with voxel count** — it scales ~super-linearly in N while the community tools
   scale sub-2×-per-2×-voxels, so the multiplicative advantage shrinks as the
   volume grows (T=100):

   | size (voxels) | nitrix GPU | vs mcflirt | vs 3dvolreg |
   |---|---|---|---|
   | 48³ (0.11 M) | 0.157 s | 51× | 25× |
   | 64³ (0.26 M) | 0.423 s | 34× | 22× |
   | 80³ (0.51 M) | 1.048 s | **26×** | **16×** |

   nitrix-GPU time grows ~2.5× per ~2× voxels (super-linear); mcflirt/3dvolreg
   grow ~1.8× — so extrapolated to brain-scale N the advantage keeps shrinking.
   And even at its best (16–51×) the GPU win is **modest for this suite**: the
   other recipes clear ~100× (e.g. rigid GPU is ~240× over ANTs-CPU). So the GPU
   kernel has real headroom too. **Any volreg kernel redesign must re-measure GPU
   as well as CPU** — GPU is not exempt; it merely fails more gracefully.

**Likely root cause.** The recipe appears to batch **all T frames** (plus
per-frame intermediates — displacement / Jacobian / warped) into one XLA program
(a `vmap`/`scan` over frames materialising O(T·N) buffers). On GPU the massive
parallelism turns that into the throughput win. On CPU, XLA cannot parallelise
across frames the way the GPU does, so it degenerates into a slow, memory-heavy
serial pass — and loses to the cache-friendly, frame-at-a-time C kernels.

**Proposed approach (raise — this looks like a new CPU kernel, not a tweak).**
A **streaming / chunked** CPU path that processes frames in bounded-size groups
(or one-at-a-time via a `scan` that does **not** retain all per-frame
intermediates), bounding memory to O(N) (or O(chunk·N)) and matching the C
tools' frame-by-frame efficiency. The GPU path (the win) stays batched.
Alternatively / in the interim: **position volreg as GPU-first** — document that
the CPU path is a correctness fallback (not for production-scale T) and emit a
one-time warning when T·N on CPU is large.

**Composition / blast radius.** CPU-path kernel only; the GPU path and the
output (realigned series + transforms) are unchanged. A chunked path is a
performance/memory refactor, behaviour-preserving within tolerance.

**Provenance.** perf-bench, single L4 host, nitrix `b09f40c`; nitrix-CPU rows
measured under the runner's subprocess `host_rss` high-water mark; the 5 GB
T=500 figure was also observed directly as a live 5.07 GB worker RSS.

**Related.**
[`register-default-fixed-iterations-regression.md`](register-default-fixed-iterations-regression.md)
(same suite; convergence-default parity). perf-bench note: volreg keeps
`mode='fixed'` (early-exit gives the batched path no benefit — measured), which
is orthogonal to this O(T·N) materialisation issue.
