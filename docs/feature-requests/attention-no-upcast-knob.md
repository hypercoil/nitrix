# attention-no-upcast-knob — caller-controlled SDPA accumulation precision

> **Status (2026-06-24), against `nitrix main@85fc6ac`.** Not present.
> Follow-up to [`attention-kernels.md`](attention-kernels.md) (the SDPA
> primitive, now landed in `nitrix.nn.attention`). Raised by the **nimox**
> dispatch-seam migration
> (`nimox/docs/feature-requests/dispatch-seam-attention-ssm.md`): 8 of 9
> attention/SSM cores migrated cleanly onto `scaled_dot_product_attention`
> / `selective_scan`; **one core could not** and is the actual consumer
> below. Home: `nitrix.nn.attention`.

**What.** A caller-facing knob on
`nitrix.nn.attention.scaled_dot_product_attention` to **opt out of the
mandatory ≥float32 score/softmax accumulation**, so the logits, softmax,
and value-aggregation run in the input dtype instead.

Today the reference hard-codes the accumulation type
(`src/nitrix/nn/attention/_reference.py:108-109`):

```python
out_dtype = jnp.result_type(q, k, v)
acc_dtype = jnp.promote_types(out_dtype, jnp.float32)   # always >= fp32
```

and threads `acc_dtype` through both einsums (`preferred_element_type`),
the `bias` add, the mask fill, and the softmax, casting back to `out_dtype`
only at the end (`:137`). For fp16/bf16 inputs this **always upcasts** to
fp32 with no way to decline. There is no public parameter to change it.

**Why / the actual consumer.** `nimox/architectures/diffusion_unet.py`
`_Attention` (the `brain_ldm` latent-diffusion U-Net's `attn1` self- and
`attn2` cross-attention) carries an `upcast_attention: bool` field that
reproduces the upstream Generative/MONAI stability hack **verbatim**:

```python
if self.upcast_attention:            # deep levels only
    q = q.astype(jnp.float32)
    k = k.astype(jnp.float32)
attn = einsum('hqd,hkd->hqk', q, k) * scale   # native dtype unless upcast
attn = softmax(attn, -1)                       # native dtype unless upcast
attn = attn.astype(v.dtype)                    # cast back before...
out  = einsum('hqk,hkd->hqd', attn, v)         # ...the value matmul (native)
```

So the upstream defines **two** precision regimes per block:

| regime | q·kᵀ + softmax | attn·v |
|---|---|---|
| `upcast_attention=False` (shallow) | **native** (fp16/bf16) | native |
| `upcast_attention=True` (deep) | fp32 | **native** |

The nitrix kernel can express *neither* exactly for a reduced-precision
model: it forces fp32 on **all three** stages. This is why `_Attention` was
the one core left on its hand-rolled `einsum + softmax` in the dispatch-seam
landing — substituting it would silently change the model's deliberate
mixed-precision numerics and break checkpoint-faithful parity. Every other
nimox attention core (ViT3D, Swin, SAM-3D ×2, SegVol ×2, UNesT, Lifespan)
migrated bit-exact because they run fp32, where the forced upcast is a no-op.

**When it bites.** Only when `brain_ldm` runs at reduced precision (its
`upcast_attention=False` blocks in fp16/bf16). If brain_ldm is fp32-only in
ilex inference, the forced upcast is a no-op and `_Attention` can migrate
**without this knob** (the cheaper resolution — worth confirming first with
the ilex/nimox owner). The knob is the fix for the genuinely-mixed-precision
case; it is latent until then, but the consumer surface (`upcast_attention`)
already exists and is load-bearing for upstream weight fidelity.

**Proposed API.** Minimal, additive, default-preserving:

```python
def scaled_dot_product_attention(
    q, k, v, *, bias=None, mask=None, scale=None, causal=False,
    qk_norm=False,
    acc_dtype: DTypeLike | None = None,   # NEW. None -> promote(out_dtype, fp32)
    backend: Backend = 'auto',
): ...
```

- `acc_dtype=None` (default) → **byte-identical to today** (`promote_types(
  out_dtype, float32)`). The golden corpus, ilex Tier-1 parity, and every
  current caller are untouched.
- `acc_dtype=q.dtype` (or any explicit dtype) → accumulate the scores,
  softmax, and value matmul in that dtype (no forced upcast). This is what
  lets `_Attention(upcast_attention=False)` migrate bit-exact.

A boolean `upcast: bool = True` is a friendlier spelling of the common case
(`upcast=False` ≡ `acc_dtype=result_type(q,k,v)`); `acc_dtype` is preferred
because it is strictly more general and composes with the existing
`out_dtype` cast. (Either is fine; the dtype form avoids a second knob if a
caller ever wants fp64 scores on fp32 inputs.)

## Hazards & viability assessment

**Viability — high for the reference, N/A for the fused path.**

- *Reference (`_reference.py`):* trivial. Replace the hard-coded `acc_dtype`
  with `acc_dtype if acc_dtype is not None else promote_types(out_dtype,
  float32)` and leave the rest. ~3-line change, fully autodiff-native, no new
  control flow.
- *Fused `pallas-cuda` path:* **cannot honor `acc_dtype < fp32`.** Online
  (streaming) softmax keeps the running max + denominator in fp32 *by
  construction* — that fp32 register accumulation is the whole reason flash
  attention is numerically stable without materialising the score matrix.
  So a non-fp32 `acc_dtype` is a **jax-reference-only** capability.

**Hazards.**

1. **Fused-path contradiction (must be enforced).** `acc_dtype` forcing
   sub-fp32 accumulation is incompatible with `backend='pallas-cuda'`. Mirror
   the existing `selective_scan` precedent (`ssm/__init__.py:146-155`, where
   an explicit `driver` + `backend='pallas-cuda'` raises `NitrixBackendError`):
   an explicit sub-fp32 `acc_dtype` + `backend='pallas-cuda'` should **raise**,
   and `backend='auto'` should **resolve to `jax`** (not silently run fp32 in
   the kernel and violate the request). Net: opting out of upcast forgoes the
   flash speedup — acceptable, since the consumer chose low precision for a
   small-`s` shallow block where the memory cliff isn't the constraint, and
   parity dominates.
2. **Overflow / NaN footgun.** fp16 `exp()` overflows for logits ≳ 11; the
   forced upcast is precisely the contract that prevents this. `acc_dtype`
   below fp32 hands the overflow risk to the caller. Make it opt-in only and
   document loudly ("you own overflow/NaN; the default is fp32-safe"). The
   consumer accepts this because it mirrors a published model that is stable
   at that precision for its shallow levels.
3. **Asymmetry gap — a single dtype does not fully reproduce
   `upcast_attention=True`.** That regime wants **fp32 scores/softmax but
   native `attn·v`**. A single `acc_dtype` ties all three stages together, so
   `upcast=True` blocks would migrate to *strictly higher* precision (fp32
   `attn·v`), not bit-exact. Whether ilex Tier-1 parity tolerates that is the
   open question. **Recommendation:** ship the single `acc_dtype` first — it
   makes the load-bearing `upcast=False` regime exact and the `upcast=True`
   regime no-worse-than-reference — and only add split `score_acc_dtype` /
   `out_acc_dtype` controls if the `upcast=True` blocks actually fail parity.
   Don't pay for the granularity speculatively.
4. **Parity-contract safety.** The default path must stay byte-identical so
   the golden corpus and ilex's `backend='jax'` pin are unaffected; only the
   explicit opt-in diverges. Low risk if `acc_dtype=None` maps to exactly the
   current expression.
5. **Gradient precision (training only).** Sub-fp32 accumulation gives noisy
   fp16 grads. Moot for brain_ldm *inference*; note it for any future training
   consumer.

**Net.** A small, well-contained, default-preserving reference-only knob with
one hard enforcement (sub-fp32 ⇒ not `pallas-cuda`). The single real design
call is single-dtype vs split score/out dtypes (hazard 3), which is driven by
whether the consumer needs `upcast_attention=True` *bit-exact* or just
*≥-precise*.

## Tests

- Default unchanged: `acc_dtype=None` output is bit-identical to the current
  reference on the existing golden corpus (no regen).
- No-upcast path: with fp16/bf16 `q,k,v` and `acc_dtype=q.dtype`, the output
  matches a hand-rolled native-dtype `einsum + softmax + einsum` (the consumer's
  `upcast_attention=False` body) bit-for-bit.
- Dispatch guard: `acc_dtype=float16` + `backend='pallas-cuda'` raises
  `NitrixBackendError`; `backend='auto'` resolves to `jax`.
- Overflow doc-test: a large-logit fp16 case is asserted to NaN under the
  opt-in (documents the footgun) and to stay finite under the default.

## Acceptance

nimox's `diffusion_unet._Attention` drops its hand-rolled core and calls
`scaled_dot_product_attention(..., acc_dtype=<native when not upcast>)`,
reproducing the published brain_ldm mixed-precision numerics; the default
fp32 path and ilex Tier-1 parity are untouched; the fused path stays fp32-only
by contract. (If brain_ldm is confirmed fp32-only in ilex, this FR is
**WONTFIX** — the core migrates without a knob.)
