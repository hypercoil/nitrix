# Mixed-precision (fp16/bf16) strategy for nitrix

> **Status (2026-06-24): design research, no code.** Synthesised from a
> three-part audit of `nitrix main@85fc6ac` (precision handling, op suitability,
> consumer demand). Companion to [`reproducible-dispatch`](reproducible-dispatch.md)
> (the `driver` axis) — mixed precision is a *fourth*, orthogonal axis and this
> doc argues it must stay outside the `driver`/divergent-op system. The first
> concrete instance is [`attention-no-upcast-knob`](attention-no-upcast-knob.md)
> (brain_ldm `qk_upcast`).

## TL;DR — the blast radius is genuinely narrow

nitrix is a **fp32/fp64-first correctness-and-reproducibility substrate**, not a
mixed-precision library, and it should not become one. Reduced precision is a
perf/memory optimisation that trades numerical fidelity — and for the scientific
core (linalg, stats, register, metrics, signal, geometry, bias, semiring) that
trade is **actively harmful**, not merely unhelpful. The audit confirms the only
place reduced precision both *makes sense* and *is wanted* is the thin
neural-network forward seam: `nn.attention`, `nn.ssm`, `nn.norm` (~3 op
families, ~5 files). Everything else gets no value and real risk.

So the strategy is small on purpose: **codify a policy, make the NN-forward seam
dtype-polymorphic with a hard fp32-accumulation floor, and add explicit
parity knobs only when a consumer actually schedules one.** No general
mixed-precision framework, no dtype axis on the scientific core.

## Two demands, opposite shapes

The single most important finding is that "mixed precision" is really **two
different requests** with **opposite implementation shapes**. Conflating them is
the main way this goes wrong.

### D1 — faithful reproduction of an upstream's *deliberate* mixed-precision numerics (inference)

A consumer is porting a published model whose checkpoint was trained/released
with a *specific, intentional* precision profile, and the port must be
bit-faithful to it.

- **Sole instance today:** `brain_ldm` (MONAI `DiffusionModelUNet`), whose
  `upcast_attention=True` (`ilex/models/brain_ldm/model.py:126-139`,
  `nimox/architectures/diffusion_unet.py:1470`) is the Stable-Diffusion stability
  hack: q/k → fp32, **value matmul stays native**. This is an *asymmetric*
  regime, and it wants *less* precision than nitrix's safe default in the value
  matmul. See [`attention-no-upcast-knob`](attention-no-upcast-knob.md).
- **Shape of the fix:** explicit, **op-specific, semantically-named** knobs that
  let the caller *match* the upstream — e.g. `qk_upcast: bool`. Reference-path
  only (a fused flash kernel keeps fp32 online accumulators by construction, so
  it cannot honour a sub-fp32 stage — mirror the `selective_scan` driver +
  `backend='pallas-cuda'` raise). These are **bespoke and demand-gated**, never a
  general framework: each upstream's hack is its own knob.

### D2 — dtype-polymorphism for memory-bound training (bf16)

A consumer wants to run an otherwise-fp32 model in bf16 to fit activations in
memory / use tensor cores. They don't want a *different* algorithm — they want
the *same* op to run correctly when handed bf16.

- **Instances:** `3DINO ViT-L` and `SwiFT`, both flagged **activation-bound** in
  `ilex/docs/design/training-engine.md:394-434` as `PrecisionPolicy(compute=bf16)`
  candidates, deferred pending the `mpx` library (Equinox-native MP-for-JAX).
- **Shape of the fix:** **no new knobs.** nitrix's NN-forward ops must simply be
  *dtype-polymorphic* — accept bf16, **accumulate reductions/matmuls in ≥fp32
  internally**, return bf16 — and be **certified within a published per-dtype
  tolerance** in the golden corpus. The outer `mpx`/`PrecisionPolicy` layer
  (in ilex, **not** nitrix) casts at the seam and owns loss-scaling. nitrix's
  obligation here is a *compatibility + accumulation + test* guarantee, not API
  surface.

| | D1 (inference parity) | D2 (training bf16) |
|---|---|---|
| Who needs it | brain_ldm | 3DINO, SwiFT |
| What nitrix adds | explicit `qk_upcast`-style knob | dtype-polymorphism, fp32-acc floor |
| New public API? | yes (one per upstream hack) | no |
| Accumulation | caller may go *below* fp32 (opt-out) | always ≥fp32 (invariant) |
| Where dispatch lives | the knob, in nitrix | mpx/PrecisionPolicy, in ilex |
| Demand | latent (fp32 today) | latent (training suite not built) |

Both are **latent** — production reduced-precision usage in ilex/nimox today is
**zero** (every model runs fp32; all parity fixtures are fp32). That is exactly
why the strategy leads with policy + invariants and defers the surface.

## Suitability map

### Bucket A — genuine candidates (the entire value)

| Op | File | Why it's safe | fp32-acc today? |
|---|---|---|---|
| `nn.attention` | `nn/attention/_reference.py:52-137` | GEMM-bound; online-softmax pattern | **yes** (`promote_types(out,f32)` :109; kernels upcast carry :135-137) |
| `nn.ssm.selective_scan` | `nn/ssm/_reference.py:45-150` | first-order linear recurrence, stable | **no** — state follows `result_type` (:125), **not** forced fp32 |
| `nn.norm` (layer/group/instance) | `nn/norm/_reference.py:37-92` | elementwise + small-axis reduction | **unverified** — mean/var reduction dtype not pinned |

**Concrete gap this surfaces:** the fp32-accumulation floor is currently only
enforced in **attention** (and the CUDA kernels). **SSM and norm do not force
it.** A bf16 selective-scan would accumulate the recurrence in bf16 (unstable
over long sequences); a bf16 norm would reduce mean/var in bf16 (lossy over
large spatial extent). Making the floor *uniform across Bucket A* is the single
most valuable concrete deliverable (P1 below) and serves D2 with no API change.

### Bucket B — reduced-precision-hostile (no dtype surface, ever)

Each is fp32/fp64-load-bearing; reduced precision is a correctness bug, not a
speed/quality trade.

- **linalg** (`_smalllinalg.py:48-50` *explicitly expects x64*; Cholesky/eigh,
  pivot floor `~1e2·finfo(dtype).eps`): normal-equation conditioning is
  **squared** (`cond(XᵀXᵀ)=cond(X)²`); at fp16 eps≈4.9e-4 a design with cond≳45
  loses all precision.
- **stats** (GLM/GLMM/LME/REML IRLS, permutation/TFCE): IRLS solves
  normal equations iteratively (B-above squared conditioning); exp-family links
  overflow in fp16; convergence gating breaks when tol≈eps.
- **register** (rigid/affine GN-LM & BFGS, SVF exp-map integration,
  `field_smooth`, MI): iterative normal-equation solves; SVF dynamic-range
  overflow; MI `log(p)` underflow; the GPU joint-histogram scatter
  nondeterminism (below) is *worse* in fp16.
- **metrics** (`information.py` joint histogram + MI/LNCC/CR gradients):
  `log(bin/total)` underflow; variance denominators underflow.
- **signal** (`_iir.py` cascades): feedback recurrence over 400+ fMRI samples →
  catastrophic accumulation; coefficients designed in `np.float64`.
- **geometry** (`_interpolate.py` cubic B-spline prefilter): recursive pole
  filter, unstable in fp16; mesh differential ops underflow on fine meshes.
- **bias** (`n4`): log-domain iteration → `log(fp16 tiny)` degenerate.
- **semiring** (LOG algebra): log-sum-exp underflows in fp16.

**Cross-cutting hazard — finfo guards.** Pervasive `finfo(dtype).eps`/`.tiny`
floors (linalg pivot `_smalllinalg.py:74-90`; register RMS `_force.py:133-167`,
SVF step-cap `_svf.py:467-472`; smoothing) are *dtype-adaptive in code* but the
surrounding *algorithms* assume the guard rarely fires. At fp16 (eps≈4.9e-4,
tiny≈6e-8) they misfire — either constantly (everything reads "zero") or the
intended `1e-12`-scale thresholds become unrepresentable. The guard machinery
silently changes character below fp32.

### Bucket C — neutral (fp16-safe but pointless)

Layout/indexing (`coords`, `grid`, `algebra`), morphology (integer/elementwise),
augment (elementwise noise), separable smoothing, sparse topology. These tolerate
fp16 but gain nothing; no action.

## Hazard catalog (consolidated)

1. **Squared conditioning in normal equations** — linalg/stats/register. Fatal
   below fp32 for any non-trivially-conditioned design.
2. **Log-domain underflow** — semiring LOG, metrics MI, bias N4, glmm AGQ
   logsumexp. `log(fp16 tiny)≈-16.6` → denormals, NaN gradients.
3. **finfo-guard misfire** — the calibrated `eps`/`tiny` floors change behaviour
   at fp16 (see above). The most *insidious* hazard: no error, just wrong guards.
4. **GPU atomic-scatter nondeterminism, amplified** — `metrics.joint_histogram`
   scatter path (already the driver of affine-MI multistart / E2 one-hot). fp16
   atomic adds are noisier *and* the reordering error is larger relative to fp16
   eps. `reproducible()` + sub-fp32 scatter is a contradiction.
5. **Recurrence/feedback accumulation** — signal IIR, geometry prefilter, and
   (today, unguarded) `nn.ssm`. Long recurrences must keep a ≥fp32 accumulator.
6. **exp overflow** — softmax (logit≳11), SSM `exp(ΔA)`, GLM exp-links. nitrix's
   forced-fp32 softmax is *precisely* this guard; opting out hands overflow to
   the caller (document loudly).
7. **Catastrophic cancellation** — variance/covariance/LNCC/CR denominators.
8. **Divergent-op tolerance-matrix explosion** — adding dtype as a 6th axis turns
   the 5-op × variant × dtype contract unmaintainable. Keep dtype *out* of the
   `driver` system (see principles).
9. **Gradient precision (training)** — bf16 grads are noisy; the fp32-acc floor
   plus loss-scaling (mpx's job) is required. nitrix should accumulate gradients
   in ≥fp32 (semiring backward already pins `preferred_element_type=primal`;
   verify it never drops below fp32).

## Design principles

1. **fp32/fp64-first; reduced precision is admitted only at the NN-forward
   seam.** The scientific core is fp32/fp64-only. This is a *tenet*, not a
   default — Bucket B ops take no dtype surface.
2. **The fp32-accumulation invariant.** Wherever reduced precision is admitted,
   *I/O dtype may be fp16/bf16 but reductions/matmuls/recurrences accumulate in
   ≥fp32*, output cast back at the end. This is the load-bearing safety
   guarantee. It already holds in attention; P1 makes it uniform.
3. **dtype is its own axis, orthogonal to `driver`/`backend`/`method`.** The
   `driver` axis chooses among numerically-divergent *algorithms of the same
   math at the same precision*; dtype is the *precision of the data*. Conflating
   them is a category error and explodes the tolerance matrix (hazard 8). Mixed
   precision is governed by (a) the input array dtype and (b) explicit
   per-op precision knobs — never by a divergent-op registry entry.
4. **Sub-fp32 accumulation ⇒ reference-path only.** Fused flash/scan kernels keep
   fp32 online accumulators by construction; an explicit sub-fp32 request +
   `backend='pallas-cuda'` must **raise**, and `backend='auto'` resolve to
   `jax` — exactly the `selective_scan` precedent (`ssm/__init__.py:148-154`).
5. **`reproducible()` forbids determinism-breaking reduced precision.** Under
   `nitrix.reproducible()`, sub-fp32 accumulation on any path whose determinism
   it guarantees (notably the histogram scatter) is rejected or promoted — the
   determinism contract outranks the precision request.
6. **Loud boundary into Bucket B.** Consistent with the loud-fallback tenet, a
   sub-fp32 array entering a Bucket-B op should not silently produce garbage:
   either promote-to-fp32 with a one-time `NitrixPrecisionFallback` note
   (default) or raise under `reproducible()`. (Decision open — see below.)

## Phasing

**P0 — Policy + SPEC tenet (doc-only, do first).** Fold principles 1–6 into a
SPEC tenet ("§2 tenet 11: precision policy") and a short §3 dispatch note that
dtype is an axis distinct from `driver`. Mirrors how reproducible-dispatch
rolled out (principle before code). Cheapest, highest-leverage step; prevents
ad-hoc dtype kwargs creeping onto Bucket-B ops.

**P1 — Uniform fp32-accumulation floor + bf16 golden corpus for Bucket A
(highest-value concrete work; unblocks D2 with no new API).**
- Enforce the fp32-acc invariant in `nn.ssm` (currently `result_type` at
  `_reference.py:125` — add an explicit `≥fp32` floor on the recurrence state and
  the chunk accumulators) and `nn.norm` (pin mean/var reduction to ≥fp32).
- Add fp16 + bf16 cases to the golden corpus for all three Bucket-A families,
  certified within published per-dtype tolerances (extend the existing
  fp32/fp64 corpus). This is the artifact that lets ilex's mpx layer trust
  "hand nitrix bf16, get bf16 back within tolerance."
- No public signature change. Serves 3DINO/SwiFT bf16 *training* directly.

**P2 — Explicit parity knobs for D1 (demand-gated, one per upstream hack).**
- Implement `qk_upcast: bool` on `scaled_dot_product_attention` (reference-only,
  pallas raise per principle 4) — the brain_ldm case. Build **when ilex
  schedules fp16 brain_ldm inference**, not before (untested public surface
  otherwise). Spec'd in [`attention-no-upcast-knob`](attention-no-upcast-knob.md).
- Any future asymmetric-upstream port gets its own bespoke knob the same way.
  Resist generalising prematurely — there is exactly one instance today.

**P3 — Bucket-B guardrails (defensive, lower priority).** Add the loud boundary
of principle 6 to the genuinely-hostile entry points (linalg solvers, stats
fitting, register recipes, N4). Implementation: a small `_require_fp32(...)`
helper at the public boundary. Decision needed on promote-vs-raise (below).
Deferrable until something actually feeds fp16 into the core by accident.

## What we explicitly will NOT do

- No dtype/`acc_dtype` parameter on any Bucket-B op.
- No dtype dimension in the divergent-op registry / `driver` axis.
- No global "mixed-precision mode" switch (that lives in ilex's PrecisionPolicy).
- No speculative knobs ahead of a scheduled consumer (P2 is demand-gated).
- nitrix does not own loss-scaling, autocast, or skip-on-nonfinite — that is the
  `mpx`/PrecisionPolicy layer's job.

## Open decisions for the owner

1. **P3 promote-vs-raise:** when a sub-fp32 array hits a Bucket-B op, promote to
   fp32 with a loud one-time note (forgiving, hides a perf cost), or raise
   (strict, surfaces the mistake)? Recommendation: **promote + loud note by
   default, raise under `reproducible()`** — matches the loud-fallback tenet
   without breaking a casual fp16 pipeline.
2. **P1 tolerances:** adopt fp16/bf16 golden-corpus tolerances now, or wait for
   the first real bf16 training run to calibrate against? Recommendation: seed
   from the existing per-dtype divergent-op budgets (e.g. attention bf16 ≈ a few
   ×1e-2) and tighten once 3DINO/SwiFT produce real envelopes.
3. **P0 timing:** fold the tenet into SPEC now (alongside this doc), or hold
   until P1 lands? Recommendation: **now** — the policy is the point; it stops
   dtype kwargs leaking onto the core regardless of when P1/P2 ship.

## Cross-references

- [`attention-no-upcast-knob`](attention-no-upcast-knob.md) — the first concrete
  instance (D1: brain_ldm `qk_upcast`).
- [`reproducible-dispatch`](reproducible-dispatch.md) — the `driver` axis; dtype
  is the orthogonal fourth axis argued here.
- `ilex/docs/design/training-engine.md:394-434` — the `PrecisionPolicy` /
  `mpx` deferral and the 3DINO/SwiFT bf16 candidates (D2).
- `nimox/docs/feature-requests/dispatch-seam-attention-ssm.md:148-163` — why
  `diffusion_unet._Attention` was the one core held off the nitrix SDPA.
- `src/nitrix/_internal/_divergent_ops.py` — the 5 divergent ops; this doc argues
  dtype must **not** join them.
