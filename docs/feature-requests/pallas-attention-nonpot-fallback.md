# Pallas attention: automatic fallback on non-power-of-2 token counts

> **Home:** `nn.attention` (Pallas-Triton backend, ``_kernels/cuda``).
> **Severity:** ENABLING for DINO/MoCo/simCLR multi-crop pipelines
> (any recipe with heterogeneous view sizes producing non-POT token
> counts downstream).

## The problem

The Pallas-Triton attention kernel raises when any operand shape is
not a power of 2:

```
ValueError: The Pallas Triton lowering currently requires that all
operations have array arguments and results whose size is a power of 2.
Encountered an array of shape (28, 64)
```

(from ``src/jax/_src/pallas/triton/lowering.py:_check_tensor_size``.)

This surfaces cleanly on the standard backend-resolver fallback path
for *some* shapes -- e.g. global crops in DINO3d at 112³/16³ = 344
tokens fall back silently with a ``NitrixBackendFallback`` warning:

```
NitrixBackendFallback: scaled_dot_product_attention: falling back to
backend='jax' from 'pallas-cuda': no fused attention kernel available
for this shape/host
```

but for the DINO **local-crop** shape (48³/16³ = 27 patches + 1 CLS =
28 tokens), the fallback is not triggered -- the lowering proceeds and
then errors, killing the training run rather than dispatching to the
JAX path.

## Why it matters

DINO multi-crop is the canonical SSL recipe and mixes global + local
crops of different spatial sizes. The upstream 3DINO recipe:

- Globals at 112³ or 96³ (7³ or 6³ patches → 343 or 216 tokens + CLS)
- Locals at 48³ (3³ = 27 patches + CLS = 28 tokens)

None of those token counts is a power of 2. Global-only pipelines
work today (fallback engages); local-crop pipelines crash.

Workaround: force the entire graph off Pallas with
``NITRIX_BACKEND=jax``. This works (we shipped stage-17k this way,
2026-07-06), but it also sacrifices the fused kernel for the *global*
crops that would otherwise fall back per-shape.

## Proposal (sketch)

Extend the backend resolver's shape check so that the Pallas backend
declines to accept any (num_tokens, head_dim) shape that isn't
Pallas-lowerable, and the fallback engages before the lowering. The
same ``_check_tensor_size`` predicate the lowering uses can be lifted
into the resolver's ``supports(shape)`` hook. Cheap; localised to the
resolver.

Alternative (better ergonomics): pad token counts up to the next power
of 2 with masked padding tokens on the fused-kernel path, transparent
to the caller. Higher effort but the correct long-term fix for any
transformer with variable/awkward token counts.

## Trigger

Any ilex/nimox training recipe with multi-crop (DINO, MoCo, simCLR)
where the local-crop shape produces non-POT token counts. Also any
downstream ViT variant using inputs that don't happen to give POT
token grids (e.g. 96³/16³ = 6³ = 216 patches; 224³/16³ = 14³ = 2744
patches; both non-POT).

## Cross-references

- Existing FR context: [`ilex-training-substrate.md`](ilex-training-substrate.md).
- Related FR (also blocks the 3DINO augmentation stack):
  [`gaussian-smooth-traced-sigma.md`](gaussian-smooth-traced-sigma.md).
- Error source: JAX ``pallas/triton/lowering.py:_check_tensor_size``.
- Resolver hook: ``src/nitrix/_internal/backend.py``.

## Non-goals

- **Not** a request to remove the Pallas kernel — keep the fused path
  for the shapes it supports.
- **Not** a request for a Triton kernel that handles non-POT natively
  (that's a separate perf goal; padding is the pragmatic bridge).
