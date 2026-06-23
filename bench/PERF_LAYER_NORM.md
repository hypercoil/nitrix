# Fused LayerNorm — perf decision benchmark (2026-06-23, L4)

**Question.** P3 gates the fused norm kernel on an empirical perf signal
(suite §7.3): norms have no activation cliff, so the only possible win is
memory bandwidth — which XLA's own elementwise+reduction fusion may already
capture. Should we build a fused Pallas LayerNorm?

**Method.** Benchmark the *stock* `jax.experimental.pallas.ops.gpu.layer_norm`
(the exact kernel the plan says to fork — a faithful proxy for our fork)
against the nitrix XLA reference (`reference_layer_norm`), forward and
forward+backward, across realistic transformer shapes `(B, S, H)`. Median of
3×100 jitted iters after warmup, on the L4 (`bench/perf_layer_norm.py`).
`x` = XLA-time / Pallas-time (so `>1` means the fused kernel is faster).

## float32

| shape (B,S,H)   | fwd ×  | fwd+bwd × | peak mem (xla/pal) |
|---|---|---|---|
| (8,1024,1024)   | 1.01   | **0.61**  | 67 MB / 67 MB |
| (8,1024,2048)   | 0.97   | 0.86      | 134 / 134 |
| (8,1024,4096)   | 0.99   | 0.95      | 268 / 268 |
| (8,1024,8192)   | 0.99   | 0.95      | 537 / 537 |
| (32,512,1024)   | 0.99   | **0.78**  | 134 / 134 |
| (4,4096,2048)   | 0.99   | 0.93      | 268 / 268 |

## bfloat16

| shape (B,S,H)   | fwd ×    | fwd+bwd × |
|---|---|---|
| (8,1024,2048)   | **1.41** | **0.30**  |
| (8,1024,8192)   | **1.90** | **0.53**  |
| (32,512,1024)   | **1.42** | **0.25**  |

## Findings

1. **No memory win, ever** — identical peak in every case. As predicted: a norm
   has no large intermediate to avoid (cf. attention's `(s,t)` or the scan's
   `(l,d,n)`), so there is nothing for fusion to spare.
2. **float32: no compute win** — forward is at parity (0.97–1.01×); forward+
   backward is *slower* (0.61–0.95×). XLA's fusion already saturates the
   bandwidth, and the fused backward (two kernels: `dx` + the `dw/db` reduction)
   loses to XLA's autodiff, worst at small `H` / many tokens.
3. **bfloat16: forward wins, training loses** — the fused forward is 1.4–1.9×
   faster (bf16 is more bandwidth-bound, so the single-pass fp32-accumulating
   kernel helps), **but** forward+backward is 2–4× *slower* — the fused
   backward dominates and erases the forward gain.

## Decision: **do not implement the fused LayerNorm kernel.**

The norm op carries a `custom_vjp` and is used in training (forward **and**
backward), where the fused path is a regression in both dtypes (the fused
backward is the bottleneck), with no memory benefit anywhere. XLA is the right
backend. The shipped XLA reference + dispatch + `out_scale` hook is the final P3
norm surface; `_kernels/cuda/norm.py` stays a loud-fallback stub. GroupNorm /
InstanceNorm only add more reduction work, so the same conclusion holds a
fortiori.

**Re-evaluation trigger (perf suite, not now):** the *one* real win is the
**bf16 forward** (1.4–1.9×) — a forward-only fused norm could help **inference**
(no backward) on bf16 deployments. If a profiler ever shows bf16 norm bandwidth
on an inference-only critical path, revisit a forward-only `backend='pallas-cuda'`
variant then — gated on that specific signal. The training case is settled: XLA.
