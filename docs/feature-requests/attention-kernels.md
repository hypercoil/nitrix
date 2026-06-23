# attention-kernels — scaled dot-product / flash attention

> **Status (2026-06-23), against `nitrix main@6449cfa`.** Not present. Part of
> the [`nn-forward-block-kernels.md`](nn-forward-block-kernels.md) bundle —
> **P0, ENABLING**. Read that ledger first for the shared framing (new family,
> parity two-tier, dispatch fit, boundary).

**What.** A single scaled-dot-product-attention primitive
`nitrix.attention.scaled_dot_product_attention`, with a `jax` reference
(exactly today's `einsum + softmax`) and a `pallas-cuda` flash-attention fast
path, subsuming the four hand-rolled reimplementations in ilex nimox:

- `ilex/nimox/architectures/vit3d.py` — dense self-attention
  (`einsum('hsd,htd->hst') * scale; softmax; einsum('hst,htd->hsd')`).
- `ilex/nimox/architectures/swin_vit.py` — windowed attention with an
  **additive relative-position bias** (`attn = q @ kᵀ + bias; softmax`),
  re-exported by `swin_unetr.py` and used by `unest.py`.
- `ilex/nimox/architectures/sam3d.py` — image/prompt cross-attention.
- `ilex/nimox/architectures/segvol.py` — dense **and** causal-masked
  (`einsum + additive causal mask + softmax`).

All four are single-implementation, FP32, no backend dispatch, no fused path.

**Why.** Attention is the dominant FLOP and the dominant memory term for every
transformer model in the zoo (BrainIAC, Neurostorm, Swift, Cortex-MAE, 3DINO,
SAM-Med3D, SegVol, the Swin-UNETR / UNesT seg backbones). Materialising the
`(h, s, s)` score matrix is the activation-memory cliff on 3-D / 4-D volumes
where `s` is large. A flash-attention kernel (online-softmax streaming, never
materialising the full score matrix) is the single highest-leverage
hardware-aware op in the cycle, and is the literal headline of the ilex
mandate ("hardware-aware modules such as flash attention").

**When it bites.** Now — every transformer forward, at both inference and
training. The memory term gates the max volume / sequence length that fits on
an Ampere card; the kernel is what makes 4-D Swin attention (Neurostorm /
Swift) tractable at full resolution.

**Proposed API.**

```python
def scaled_dot_product_attention(
    q: Float[Array, '... h s d'],
    k: Float[Array, '... h t d'],
    v: Float[Array, '... h t d_v'],
    *,
    bias: Float[Array, '... h s t'] | None = None,   # additive (Swin rel-pos)
    mask: Bool[Array, '... h s t'] | None = None,     # True = keep
    scale: float | None = None,                       # default 1/sqrt(d)
    causal: bool = False,
    backend: Backend = 'auto',
) -> Float[Array, '... h s d_v']:
    ...
```

- **Layout-agnostic** (`'... h s d'`): the module owns heads / windowing /
  patch-partition; the kernel sees already-windowed `q/k/v`. Swin passes its
  relative-position table as `bias`; SegVol passes `causal=True`; SAM passes
  distinct `s != t` for cross-attention.
- **`bias` is the load-bearing capability gap.** JAX's stock Pallas attention
  (`jax.experimental.pallas.ops.*`) supports masking/causal but **arbitrary
  additive bias inside the streamed softmax** (Swin's rel-pos) needs the bias
  tile threaded through the online-softmax accumulation. The fast kernel must
  carry `bias` and `mask` per K-block. This is the part that cannot be
  delegated wholesale to the stock kernel.

**Implementation shape (house pattern).**

- `nitrix/attention/__init__.py` — public `scaled_dot_product_attention`,
  `resolve_backend` dispatch.
- `nitrix/attention/_reference.py` — `einsum + (bias) + (mask/causal) +
  softmax + einsum`. Reproduces the current nimox math **bit-for-bit** (same
  fp32 accumulation, same softmax axis) so ilex Tier-1 parity is untouched.
- `nitrix/_kernels/cuda/attention.py` — flash-attention via online softmax;
  start by wrapping `jax.experimental.pallas.ops` where it fits, extend with a
  bias/mask-carrying block loop for the Swin/SegVol cases. Returns `None` on
  tiling failure → reference fallback + `NitrixBackendFallback`.
- Custom VJP (flash backward, recomputing the score tiles) with a
  finite-difference gradient check per algebra of `(bias, mask, causal)`.

**Tests.**

- Golden corpus: `tests/golden/attention_{dense,windowed_bias,causal,
  cross}_float32.npy` from the reference, one per code path.
- Backend parity: `pallas-cuda ≈ jax` within `tests/tolerance.toml`
  (`attention` row).
- Hypothesis: softmax row-sum = 1 under mask; `causal=True` ≡
  lower-triangular `mask`; permutation equivariance over `t`.
- Gradient: finite-difference VJP check for each of `{bias, mask, causal}`.

**Acceptance.** ilex swaps its four reimplementations for one call to this
op; the parity sweep (pinned to `backend='jax'`) stays green; nitrix's
golden corpus certifies the `pallas-cuda` path; nimox's attention module
reduces to *window/partition + this kernel + un-window*.
