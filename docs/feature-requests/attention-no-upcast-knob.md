# attention-no-upcast-knob — caller-controlled SDPA accumulation precision

> **Status (2026-06-24), against `nitrix main@85fc6ac`.** Not present.
> Follow-up to [`attention-kernels.md`](resolved/attention-kernels.md) (the SDPA
> primitive, now landed in `nitrix.nn.attention`). Raised by the **nimox**
> dispatch-seam migration
> (`nimox/docs/feature-requests/dispatch-seam-attention-ssm.md`): 8 of 9
> attention/SSM cores migrated cleanly onto `scaled_dot_product_attention`
> / `selective_scan`; **one core could not** and is the actual consumer
> below. Home: `nitrix.nn.attention`.

## nitrix determination (2026-06-24): OPEN / latent — not WONTFIX

An earlier read leaned WONTFIX on the grounds that brain_ldm runs fp32-only in
ilex today (verified: `ilex/models/brain_ldm/parity.py` builds its oracle at
`float32`; no `float16`/`bfloat16` anywhere in nimox or ilex models). That was
**too strong** — the gating question is not "is the *port* fp32 today" but "is
the *upstream model* designed for reduced precision," and on checking the
upstream the answer is **yes**.

**Upstream MONAI is fp16-intended; the upcast is an intentional capability, not
a port artifact.** brain_ldm ports MONAI's `DiffusionModelUNet`. In the MONAI
source (`monai/networks/nets/diffusion_model_unet.py:97,110`) `upcast_attention`
maps to `attention_dtype = torch.float if upcast_attention else None`
(docstring: *"upcast attention operations to full precision"*), and the block
applies it **asymmetrically** — `monai/networks/blocks/selfattention.py:174-176`
casts **only `q` and `k`** to fp32; **`v` is never cast**, so the value matmul
stays native:

```python
if self.attention_dtype is not None:   # fp32 when upcast_attention=True
    q = q.to(self.attention_dtype)
    k = k.to(self.attention_dtype)
    # v left native -> scores/softmax in fp32, value matmul in native dtype
```

This is the verbatim Stable-Diffusion / diffusers stability hack, whose *whole
reason to exist* is **fp16 inference**: force scores/softmax to fp32 to dodge
fp16-`exp()` overflow while keeping the value matmul cheap in native precision.
At fp32 it is a no-op — a model author only sets `upcast_attention=True` when
the model is meant to run in fp16. brain_ldm's config sets it `True`
(`nimox/architectures/diffusion_unet.py:54,1470`), which is upstream's explicit
signal that **fp16 is an intended deployment mode**. So ilex's current fp32-only
run is a **port-stage parity simplification, not a permanent property** — the
nimox `_Attention.upcast_attention` field is a faithful port of a real upstream
capability, and the nitrix SDPA's inability to express it is a **genuine latent
capability gap**, inert only until ilex flips brain_ldm to fp16. For a faithful
port that is a *when*, not an *if*.

**The FR's proposed `acc_dtype` is the wrong primitive.** The upstream upcast is
**asymmetric** (q/k → fp32, value matmul → native). A single `acc_dtype` ties
all three stages together, so it **cannot reproduce `upcast_attention=True`** —
it would force the value matmul to fp32 too, breaking bit-exact upstream parity.
(The FR's own "hazard 3" half-saw this but proposed deferring it; against the
upstream it is not deferrable, because `upcast=True` is the *default* for the
deep blocks — `nimox/architectures/diffusion_unet.py:54,1470`.) The faithful
knob is a **boolean that upcasts q/k only** — e.g. `qk_upcast: bool = False` —
mapping 1:1 to both the upstream `attention_dtype`-on-qk-only semantics and the
consumer surface `upcast_attention: bool`, leaving the value matmul in the input
dtype. That is cleaner than `acc_dtype` *and* the only spelling that matches
upstream. Define it as "q/k accumulate in fp32; scores/softmax in fp32; value
matmul + output in the input dtype."

**One nuance to pin before any fp16 parity claim.** MONAI's *manual* path leaves
`att_mat` in fp32 entering the value einsum (no cast-back in
`selfattention.py`), whereas **diffusers** — and the nimox port
(`diffusion_unet.py:591-593`, `attn = attn.astype(v.dtype)`) — cast the
attention probabilities **back to `v.dtype`** before the value matmul. So
"value matmul native vs fp32-promoted" differs by which upstream is treated as
canonical. The nimox port follows the diffusers cast-back semantics, so a nitrix
`qk_upcast` defined as *"q/k fp32, everything downstream native"* matches what
the port actually implements; the ilex/nimox owner should confirm brain_ldm is
locked to diffusers-semantics (it appears to be) before relying on bit-exact
fp16 parity.

**Recommendation.** Keep OPEN (latent). Do **not** build speculatively today
(nothing runs fp16 yet, so it would be untested public surface), but when ilex
schedules fp16 brain_ldm inference, implement **`qk_upcast: bool`** (q/k → fp32,
value matmul + output native), reference-path only, mirroring the
`selective_scan` `backend='pallas-cuda'` raise for the sub-fp32 ⇒ not-fused
contract — **not** a single `acc_dtype`. This FR is also the first concrete
instance of the broader question tracked in
[`mixed-precision-strategy`](mixed-precision-strategy.md) (when, where, and how
reduced precision should enter nitrix at all).

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
`scaled_dot_product_attention(..., qk_upcast=self.upcast_attention)`,
reproducing the published brain_ldm mixed-precision numerics (q/k → fp32,
value matmul + output native); the default path and ilex Tier-1 parity are
untouched; the fused path stays fp32-only by contract.

> **Status superseded (2026-06-24): OPEN / latent, not WONTFIX** — see the
> *nitrix determination* section above. The upstream MONAI model is
> fp16-intended (`upcast_attention` is an intentional reduced-precision
> capability), so this is a genuine latent gap rather than WONTFIX; the
> faithful knob is `qk_upcast: bool` (asymmetric q/k upcast), **not** the
> single `acc_dtype` proposed in the body. Build when ilex schedules fp16
> brain_ldm inference, not before.
