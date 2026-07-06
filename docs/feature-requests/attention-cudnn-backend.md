# attention-cudnn-backend — cuDNN flash path for `nn.attention`

> **Status (2026-07-01).** Not present. Consumer-pipeline substrate — reported
> against `nitrix main@bbb22c0` (as vendored under ilex sibling tree). Related:
> [`attention-kernels`](resolved/attention-kernels.md) (Pallas custom kernel),
> [`mixed-precision-strategy`](mixed-precision-strategy.md) (D2 bf16
> polymorphism), [`nn-forward-kernels-suite`](nn-forward-kernels-suite.md)
> (Phase-2 fast-path ledger).

**What.** A third backend for
`nitrix.nn.attention.scaled_dot_product_attention` that dispatches to
`jax.nn.dot_product_attention(implementation='cudnn')` (JAX ≥ 0.4.30) for the
common dense-attention case, sitting between the pure-JAX reference and the
Pallas custom kernel. Public surface: `backend='cudnn'`, extending the current
`{'jax', 'pallas-cuda', 'auto'}` enum.

**Why.** Under `backend='auto'` on Ampere, we currently resolve to
`'pallas-cuda'` and then fall back to `'jax'` (the pure-JAX reference) whenever
the Pallas kernel's `_check_tileable` rejects the shape — chiefly
`s % _BLOCK != 0` (`_BLOCK=32`). For the 3DINO / ilex `ThreeDINOViT` (12-block
ViT-S / 24-block ViT-L, 7³ patches + 1 CLS = **344 tokens**), `344 % 32 = 24`,
so **every attention call in every transformer forward hits the JAX fallback**.

The cuDNN backend (via `jax.nn.dot_product_attention(implementation='cudnn')`)
has a much wider acceptance envelope:

- **No sequence-length divisibility constraint** — accepts arbitrary `s`, `t`.
- **Native fp16 / bf16 / fp8** — pairs directly with the D2 half of
  [`mixed-precision-strategy`](mixed-precision-strategy.md) (currently
  attention forwards in bf16 have no fused option).
- Ships as `cudnn_9` frontend inside `libcudnn9` (already pulled in as an
  NVIDIA torch wheel transitively; standalone JAX-cuda ships its own).
- **Standardises on cuDNN Flash Attention v2**, i.e. online-softmax streaming
  with a fp32 accumulator — same numerical profile as the Pallas custom
  kernel target.

**Empirically measured (2026-07-01, A100-PCIe-40GB, JAX 0.10.2, fp32, S=344,
H=6, D=64, B=4, 48 fwd calls):**

- Nitrix jax-fallback (current path): **3.2 ms**
- Nitrix pallas-cuda (if S padded to 352): **2.6 ms**
- `jax.nn.dot_product_attention(implementation='xla')`: **~equiv to jax-fallback**
- `jax.nn.dot_product_attention(implementation='cudnn')`: **fp32 rejected**
  (`"Q must be fp16/bf16/fp8_e4m3fn/fp8_e5m2"`). Available only after the D2
  bf16-polymorphism ships downstream — but the dispatch point should exist so
  the win is one caller-side dtype flip away.

**Layout.** The cuDNN backend does not need the `bias` capability the Pallas
custom kernel targets — Swin's additive rel-pos-bias attention has no cuDNN
fast path. This is deliberately a **narrow** third backend for the *dense
q/k/v + optional causal + optional bool mask* case that covers ~90% of ilex
attention traffic (BrainIAC, Cortex-MAE, 3DINO, MedicalNet-VLPre — all dense).
The Swin / SegVol / SAM cases stay on the Pallas roadmap.

**Proposed API.** Extend the existing `resolve_backend` in
`nitrix._internal.backend`:

```python
Backend = Literal['jax', 'pallas-cuda', 'cudnn', 'auto']
```

`resolve_backend('auto', ...)` policy (proposal):

1. If host is Ampere+ NVIDIA and dtype is fp16/bf16/fp8 → `'cudnn'`.
2. Else if host is Ampere+ and shape/dtype tileable by Pallas kernel →
   `'pallas-cuda'`.
3. Else → `'jax'`.

I.e. cuDNN wins on the fp16/bf16 dense path (its native regime), Pallas wins
on the fp32 dense path where it can, and JAX-reference is the always-safe
fallback. `'auto'` today jumps to `'pallas-cuda'` and this widens it.

**Implementation shape.**

- `nitrix/nn/attention/__init__.py`: extend `Backend` enum, extend
  `scaled_dot_product_attention` dispatch (`resolved == 'cudnn'` branch).
- `nitrix/nn/attention/_cudnn.py` (new): thin wrapper that lays q/k/v out in
  the shape `jax.nn.dot_product_attention` expects (`(..., s, h, d)` —
  transpose from our `(..., h, s, d)`), forwards the call, transposes back.
  Reject at load-time if the JAX version is too old (`jax.nn` doesn't have
  `dot_product_attention`).
- `resolve_backend` update — add the fp16/bf16 → cuDNN rule.
- Rejection sentinel: `CudnnNotAvailable` (bad JAX, no cuDNN wheel, unsupported
  dtype) → `NitrixBackendFallback` warn + drop to the next backend in the
  auto-chain (`pallas-cuda` → `jax`).

**Tests.**

- Numerical parity: `pallas-cuda` ≈ `cudnn` ≈ `jax` within
  `tests/tolerance.toml` — with a per-dtype row (`float16`, `bfloat16`),
  looser than the fp32 row (~1e-3 absolute is the cuDNN norm).
- Shape sweep: golden `s` values that fail `_check_tileable` (344, 197, 111)
  round-trip via `'cudnn'` where `'pallas-cuda'` raises.
- Auto-dispatch: `resolve_backend('auto', dtype=bf16, host=ampere)`
  returns `'cudnn'`; fp32 dense goes to `'pallas-cuda'`.

**Acceptance.**

- `ilex.models.threedino.ThreeDINOViT` in bf16 mode has attention on the
  cuDNN fast path with the same forward-parity envelope as the current fp32
  path.
- `scaling-train` sees the JAX-fallback warning go away on 3DINO's shape
  (344 tokens).
- Downstream `PrecisionPolicy(compute=bf16)` runs get the expected 2–4×
  attention-forward speedup without touching Pallas.

**Non-goals.**

- Not a replacement for the Pallas custom kernel work — Swin's rel-pos-bias
  attention (`bias` load-bearing) stays on the Pallas roadmap and cuDNN
  cannot serve it.
- Not a mixed-precision framework — bf16 correctness lives with
  [`mixed-precision-strategy`](mixed-precision-strategy.md) D2 (dtype
  polymorphism, fp32-acc floor, `PrecisionPolicy` in ilex).
- No autodiff work needed — `jax.nn.dot_product_attention` has upstream VJP.

**Trigger note (for `internal-backlog` cross-index).** Bumped by the 3DINO
scratch-training compile / step-time investigation (2026-07-01): every one
of our 48 attention calls per step hits the JAX fallback purely because
`344 % 32 != 0`. Would either want the tileability requirement relaxed
(a `_BLOCK ∈ {8, 16, 32}` search) or this cuDNN backend to close the loop.
