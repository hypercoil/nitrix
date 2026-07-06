# Neural-network forward-block kernel suite — implementation plan (`nitrix.nn`)

> **Status (2026-06-23): sequenced implementation plan.** Turns the four
> forward-block feature requests collected in
> [`nn-forward-block-kernels.md`](nn-forward-block-kernels.md) (the context
> ledger / duplicate guard) into a concrete, phased build. Reads against
> `nitrix main@6449cfa`. Numerics-only scope; learnable modules stay in
> `nimox`, container/IO in `thrux`. **Wall-clock parity benchmarking is
> delegated to the sibling perf suite** (see §3).

Atomised requests this plan implements (do not duplicate — append drivers to
those docs): [`attention-kernels.md`](resolved/attention-kernels.md) (P0),
[`selective-scan.md`](resolved/selective-scan.md) (P1),
[`fused-norm-kernels.md`](fused-norm-kernels.md) (P3). P2 (affine / spherical /
field-regulariser nimox-extraction blockers) is cross-referenced only — no new
code here.

---

## 0. Locked decisions (2026-06-23)

1. **One namespace, not three.** All forward-block kernels live under a single
   top-level package **`nitrix.nn`** with one submodule per op family —
   `nitrix.nn.attention`, `nitrix.nn.ssm`, `nitrix.nn.norm`. This supersedes
   the per-FR `Home` hints (`nitrix.attention`, `nitrix.ssm`) which would
   pollute the top level as the family grows. Rationale in §2.
2. **nitrix owns correctness + gross memory; the perf suite owns wall-clock
   parity.** §3.
3. **Golden corpus + tolerance matrix is materialised in Phase 0.** This stands
   up the long-mandated but never-built `tests/golden/` + `tests/tolerance.toml`
   convention (SPEC §2 tenet 8; `IMPLEMENTATION_PLAN.md §2.2-4`). §6.
4. **Two-tier parity contract is the load-bearing constraint.** Each op ships a
   bit-faithful JAX *reference* (the oracle ilex pins its forward-parity to) and
   a *fused* Pallas path certified `pallas-cuda ≈ jax` only inside nitrix. §4.
5. **Functional, not modular.** `nitrix.nn` holds pure `(Array, …) -> Array`
   kernels despite the `nn` name; the `eqx.Module`s that hold parameters stay in
   nimox. Stated explicitly because the name invites the opposite assumption.

---

## 1. Why this doc exists

ilex's nimox layer hand-rolls the transformer and state-space forward blocks
four-plus times each — dense / windowed-bias / causal / cross attention as
distinct `einsum + softmax`, and a `lax.scan` Mamba recurrence — all
single-implementation, FP32, **zero backend dispatch**. These are the dominant
FLOP and the dominant activation-memory term for every model in the zoo, and
the literal headline of the ilex hardening cycle ("hardware-aware modules such
as flash attention"). nitrix already owns the dispatch machinery
(`resolve_backend` → `pallas-cuda | jax`, loud `NitrixBackendFallback`, the
streaming-kernel pattern); these kernels are the missing payload.

The inner contractions are exactly what the nitrix boundary admits — pure
array→array — and the one place a hardware-aware (Pallas/Triton) implementation
pays off most.

---

## 2. Namespace decision — `nitrix.nn`

The FR ledger proposed three new top-level packages (`nitrix.attention`,
`nitrix.ssm`, `nitrix.nn`). Three top-level names for what is conceptually one
family (and an open-ended one — the zoo will want fused MLP/SwiGLU, RoPE,
KV-cache decode, MoE routing, … next) pollutes the flat top level that today
reads as coherent domains (`geometry`, `metrics`, `semiring`, `register`, …).

**Decision: a single `nitrix.nn` domain package, op families as submodules.**

```
nitrix.nn.attention.scaled_dot_product_attention
nitrix.nn.ssm.selective_scan
nitrix.nn.norm.{layer_norm, group_norm, instance_norm}
```

The headline ops are re-exported at `nitrix.nn` for ergonomics
(`from nitrix.nn import scaled_dot_product_attention`). This mirrors the
existing `stats` package, which already nests substantial subpackages
(`stats.glmm`, `stats.lme`, `stats.inference`) under one domain name. Private
Pallas kernels stay in the flat `_kernels/cuda/` dir (house convention) as
`attention.py`, `selective_scan.py`, `norm.py`.

Naming note: these are *functional* kernels, not Equinox-style modules. The
`nn` name is the consumer-facing mnemonic (these are the building blocks of NN
forward passes); the docstring states the functional contract up front so the
name does not imply a module surface.

---

## 3. Scope boundary — nitrix vs the perf suite

**nitrix (this suite) owns *correctness and the memory contract*:**

- Golden-corpus reproducibility (§6) and `pallas-cuda ≈ jax` backend parity
  within a pinned tolerance matrix.
- Finite-difference-checked custom VJPs (forward *and* backward correctness),
  including the learnable-bias gradient (§7.1).
- Hypothesis property tests (softmax invariants, scan algebra, norm statistics).
- **Gross memory efficiency** — the structural guarantee that the fused path
  does *not* materialise the cliff term (the `(s, s)` score matrix for
  attention; the full `(l, d_state)` state trajectory for the scan). This is a
  correctness-adjacent *shape/peak-allocation* assertion, exercised with a
  `bench/mem_streaming_kernel.py`-style probe, not a wall-clock claim.
- Loud fallback on untileable shapes / unsupported hardware.

**The sibling perf suite owns *wall-clock parity at scale*** — i.e. does the
fused kernel actually *near* heavy torch / triton / cuda (FlashAttention-2,
`mamba-ssm`, apex/`triton` norms) reference throughput on real model shapes.
That work lives in `bench/` (precedents: `bench/PERF_*.md`,
`bench/MEM_STREAMING_KERNEL.md`) and is governed by
[`perf-wins-must-certify-at-scale.md`](perf-wins-must-certify-at-scale.md) and
[`perf-bench-case-hardening.md`](perf-bench-case-hardening.md). This plan ships
the *kernels and their correctness budget*; it deliberately does **not** gate
on a wall-clock number, and it must not silently claim a perf win it has not
certified (loud-fallback discipline + the perf suite's at-scale certification
are the two halves).

Concretely: a phase here is "done" when the golden + parity + VJP + gross-mem
tests are green and the loud fallback works. "Reaches FA-2 throughput on a
Swin-4D shape" is a perf-suite ticket, filed against `bench/` when the kernel
lands.

---

## 4. The contract every op follows (house pattern, verified against live code)

Each op is three files plus tests, matching `semiring/` and `register/_force`:

| Layer | File | Responsibility |
|---|---|---|
| Public | `nn/<family>/__init__.py` | `fn(…, *, backend='auto')`; `resolve_backend`; layout adapter; `custom_vjp` wiring |
| Reference | `nn/<family>/_reference.py` | pure-JAX oracle == current nimox math, **bit-faithful**; autodiff-native (no hand VJP) |
| Kernel | `_kernels/cuda/<op>.py` | Pallas/Triton fused path; `PallasNotTileable` → dispatcher returns `None` → loud JAX fallback |

Dispatch + differentiability mechanics, reused verbatim:

- `nitrix/_internal/backend.py`: `Backend = Literal['auto','pallas-cuda','jax']`,
  `resolve_backend()` (keyword → `NITRIX_BACKEND` → Ampere-autodetect via
  `_HAS_AMPERE_NVIDIA`), `fallback()` emitting `NitrixBackendFallback` once per
  `(fn, shape, dtype, backend)`.
- `default_backend_is_gpu()` for **platform-dependent algorithm** selection
  (distinct from kernel-backend selection) — the precedent for choosing
  sequential `lax.scan` (CPU) vs parallel `lax.associative_scan` (GPU) in the
  *reference* path. Already used by `signal/_iir.py` and `signal/interpolate.py`.
- `custom_vjp` on the **fused** path only; the reference path is autodiff-native
  (mirrors stock `mha`'s `backward_pass_impl='xla'` vs `'triton'`). The public
  op registers the VJP so `jax.grad`/`vjp`/`jacrev` flow through either backend.

The two-tier split means **no ilex parity fixture ever runs the
non-deterministic fused path**: ilex pins its forward-parity oracle
(`atol 1e-6 / rtol 1e-5`) to `backend='jax'`, and nitrix's golden corpus owns
the fast-path budget.

---

## 5. JAX / Pallas source reuse map

Investigated `jax/experimental/pallas/ops/` in the pinned venv (jax 0.10):

| Stock kernel | Reuse for | What it gives / what is missing |
|---|---|---|
| `gpu/attention.py` (Triton `mha`) | **P0 fork base** | online-softmax flash fwd + full Triton bwd (`dq/dk/dv`, `delta = Σ o·do` preprocess, score-tile recompute), `causal` + `segment_ids`. **Missing: arbitrary additive `bias` (Swin rel-pos), general bool `mask`, and `d_bias` in bwd** — the part that cannot be delegated. |
| `gpu/layer_norm.py`, `gpu/rms_norm.py` | **P3 fork base** | fused single-pass fwd + recompute bwd. GroupNorm/InstanceNorm need a reshape-to-groups reference of our own. |
| `gpu/softmax.py` | P0 reference cross-check | standalone softmax tiling reference. |
| `gpu/attention_mgpu.py`, `gpu/decode_attention.py`, `tpu/{flash,splash}_attention` | **deferred** | Mosaic-GPU (Hopper/Blackwell), single-query decode, TPU. Future fast-paths; cross-ref [`mosaic-hopper-registration-kernels.md`](mosaic-hopper-registration-kernels.md). Ampere/Triton is the baseline (matches `_HAS_AMPERE_NVIDIA`). |
| — (none) | **P1 selective-scan** | **No stock SSM kernel exists.** Clean-room. JAX-side substrate is `lax.associative_scan` with the linear-recurrence combinator; fused kernel is Mamba-style chunked block-parallel scan. |

Conceptual tie-in worth recording: stock attention's `(running_max, sum_exp)`
online-softmax carry is the **same pytree-state streaming pattern as nitrix's
`LOG` semiring monoid** (SPEC §4.1 / `docs/design/streaming-kernel.md`).
Attention is structurally a LOG-semiring reduction with a value-weighted
readout — good framing for reviewers and for the companion design doc.

Two adaptation points for the stock attention fork:

- **Layout.** Stock is `(batch, seq, heads, head_dim)`; the nimox/FR contract is
  layout-agnostic `'... h s d'`. The dispatcher owns the transpose + flatten of
  leading dims to a single batch axis (and the inverse on output).
- **Shape constraints.** `head_dim` is padded to `next_power_of_2`; `seq` must
  divide `block_q`/`block_k`. Arbitrary token counts → pad-to-multiple-with-mask
  (clean: we need `mask` anyway) or `PallasNotTileable` → loud fallback.

---

## 6. Golden-corpus + tolerance harness (Phase 0 deliverable)

This suite finally materialises the convention the spec has mandated since v0.1
(`IMPLEMENTATION_PLAN.md §2.2-4`, SPEC §2 tenet 8) but which the recent force
kernels skipped (they use live-reference parity only). Attention is the first
NN kernel and the right place to set it up properly.

```
tests/
  golden/                       # checked-in reference arrays (.npy), small
    attention_dense_float32.npy
    attention_windowed_bias_float32.npy
    attention_causal_float32.npy
    attention_cross_float32.npy
    selective_scan_float32.npy
    layer_norm_float32.npy
    group_norm_float32.npy
    instance_norm_float32.npy
  tolerance.toml                # (op, dtype[, backend]) -> {atol, rtol}
  _golden.py                    # load_golden(name); tol(op, dtype, backend)
tools/
  regen_golden.py               # deterministic regen from the reference path
```

- **`tolerance.toml`** is a table: `[<op>.<dtype>]` rows with `atol`/`rtol`, and
  optional `[<op>.<dtype>.<backend>]` overrides (the fused path gets a looser
  row than the reference-vs-golden row). Pinned per release; a change is a
  CHANGELOG entry (SPEC §2 tenet 6).
- **`regen_golden.py`** generates each array from the *reference* path with a
  fixed seed and `jax_enable_x64` policy, so goldens are reproducible and
  reviewable. Never regenerated from the fused path.
- **`_golden.py`** is the only place tests touch the files: `load_golden(name)`
  and `tol(op, dtype, backend)`.
- The existing **live-reference parity** tests (recompute the reference at test
  time, assert scale-relative tolerance — the `test_lncc_force_kernel.py`
  pattern) are kept *in addition*: goldens catch cross-release drift in the
  reference; live parity catches `pallas ≠ reference` regressions. Different
  failure modes, both wanted.

Retrofitting the older force kernels onto goldens is **out of scope** (logged as
a one-line backlog item, not done here).

---

## 7. Per-op design + plan

### 7.1 `nitrix.nn.attention.scaled_dot_product_attention` — P0, ENABLING

**Surface** (immutable kwargs, jaxtyping, layout-agnostic):

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

Subsumes the four nimox reimplementations: dense (`vit3d`), windowed +
relative-position `bias` (`swin_vit`/`swin_unetr`/`unest`), `causal` (`segvol`),
cross-attention `s != t` (`sam3d`).

**Reference** (`nn/attention/_reference.py`) — bit-faithful to nimox:
`scale * einsum('...hsd,...htd->...hst')` `+ bias` (if given) → apply `mask` /
`causal` via `where(·, logits, -inf)` → `softmax` along `t` (fp32 accumulation,
same axis) → `einsum('...hst,...htd->...hsd')`. Autodiff-native; this is the
oracle ilex Tier-1 swaps onto with no drift.

**Fused kernel** (`_kernels/cuda/attention.py`) — fork stock Triton `mha`,
extending exactly the three things it lacks:

1. **Additive `bias` tile** threaded per K-block via an extra `BlockSpec`, added
   to `qk` *before* the running-max update so the online softmax stays correct.
2. **Explicit boolean `mask` tile** generalising the `segment_mask`/`causal`
   `where(mask, qk, DEFAULT_MASK_VALUE)` path (covers SAM padding, `s != t`).
3. **`d_bias` accumulation in the backward** — the subtle correctness item. The
   Swin rel-pos table is *learnable*, so the VJP must return
   `d_bias = Σ ds` (reduced over the batch/head axes `bias` is broadcast on) in
   addition to `dq/dk/dv`. Stock `mha` returns only `dq/dk/dv`; this is a
   genuine extension, FD-checked per algebra of `(bias, mask, causal)`.

Layout adapter + `next_power_of_2(head_dim)` pad + pad-seq-to-block-with-mask in
the dispatcher; `PallasNotTileable` → `None` → loud fallback. `custom_vjp`
routes `pallas-cuda` to the fused backward and `jax` to XLA autodiff.

**Tests:** golden corpus (4 paths) + `tolerance.toml[attention]`; backend
parity `pallas ≈ jax`; Hypothesis (softmax row-sum = 1 under mask; `causal` ≡
lower-triangular `mask`; permutation-equivariance over `t`); FD-VJP for each of
`{bias, mask, causal}` **including `d_bias`**; the dispatch / pad / loud-fallback
battery (from `test_lncc_force_kernel.py`); gross-mem probe asserting the
`(s, t)` score matrix is never materialised on the fused path.

> **Status (2026-06-23): P0b SHIPPED (fused forward + fused backward).**
> Validated on an L4.
>
> *Forward* — fused Pallas/Triton online-softmax flash (additive-bias +
> boolean-mask tiles + causal, base-2 streamed): `pallas ≈ jax` ≤ 4e-4 (under
> the 2e-3 pinned tolerance) for dense / bias / causal / mask / cross; gross-mem
> probe = **0 temp bytes** vs the reference's full `(s, t)` score tensor.
>
> *Backward* — fully-fused, two kernels (no `q_blocks == k_blocks` constraint):
> one accumulates `dk`/`dv` and writes the learnable-bias gradient `d_bias`
> tile-by-tile (in-kernel, reduced to the bias shape by the wrapper), the other
> accumulates `dq`; both **recompute** the score tiles in-SRAM from the
> forward's `lse` residual (no stored `(s, t)`). Grad-parity vs autodiff
> reference ≤ 9e-4 (`dq/dk/dv/d_bias`) across all paths; backward gross-mem
> probe confirms no `(s, t)` temp for `dq/dk/dv`.
>
> Scope gate (→ loud fallback): Ampere+, float32, power-of-two head dim ≥ 16
> with `d_v == d`, seq divisible by the block. Unified block = 32 (fwd + bwd
> tile identically and fit the bias/mask + accumulator tiles inside the SM
> shared-memory budget); x64-robust (pinned carry/divisor dtypes). **Deferred
> (perf only):** tile-ladder tuning, pad-to-multiple-with-mask for arbitrary
> seq / non-pow2 head dim, and the at-scale wall-clock-vs-FA2 certification —
> the last is the sibling perf suite's job (`bench/`).

> **QK-norm (curse-of-depth / logit-growth control) — SHIPPED (2026-06-23,
> zero-regression).** Opt-in `qk_norm: bool = False` on
> `scaled_dot_product_attention`: RMS-normalise `q`/`k` over the head dim before
> the dot (Gemma2 / ViT-22B convention, `eps=1e-6`, no learnable scale).
> **Built to leave the fused kernel + its `custom_vjp` byte-for-byte unchanged:**
> the RMS pre-op (`qk_rms_norm`) is applied *outside* the fused core on the
> pallas path (and inside the reference on the jax path), so autodiff composes
> the norm's VJP with the unchanged fused forward/backward. `qk_norm=False` is
> byte-identical to the prior path — verified as an explicit regression guard on
> both backends. Validated on L4: reference `qk_norm` == manual RMS+attention
> (exact); fused vs reference ≤ 3e-4 (dense/bias/causal); grad ≤ 3.6e-4; golden
> case `attention_qk_norm` added. **Follow-up (perf only):** the *in-kernel*
> tile-fusion of the RMS (normalise `q`/`k` tiles before the dot, avoiding the
> `q̂`/`k̂` materialisation) is a pure-bandwidth optimisation for the perf suite;
> the `(s, t)` flash memory contract is unaffected. A learnable per-dim QK scale,
> if ever needed, reuses the `d_bias` reduction pattern.

### 7.2 `nitrix.nn.ssm.selective_scan` — P1, ENABLING

**Surface:**

```python
def selective_scan(
    x: Float[Array, '... l d'],
    delta: Float[Array, '... l d'],      # per-step, per-channel Δ (post-softplus)
    A: Float[Array, 'd n'],              # diagonal-plus state matrix
    B: Float[Array, '... l n'],          # selective input projection
    C: Float[Array, '... l n'],          # selective output projection
    D: Float[Array, 'd'] | None = None,  # skip / residual
    *,
    backend: Backend = 'auto',
) -> Float[Array, '... l d']:
    ...
```

S6/Mamba selective form (input-dependent `B, C, Δ`). Discretisation
`Δ·A → exp(Δ·A)` is inside the op; surrounding `softplus`/`silu` stays in nimox.

**Reference** (`nn/ssm/_reference.py`) — **two** oracles:

- the sequential `lax.scan` recurrence (`h_t = exp(Δ_t A) ⊙ h_{t-1} + (Δ_t B_t)
  ⊙ x_t; y_t = C_t·h_t (+ D⊙x_t)`) — the bit-exact slow oracle ilex Tier-1 swaps
  onto;
- a `lax.associative_scan` form with the linear-recurrence combinator
  `(a₁,b₁) ∘ (a₂,b₂) = (a₁a₂, a₂b₁ + b₂)`.

The reference *itself* picks `scan` (CPU) vs `associative_scan` (GPU) via
`default_backend_is_gpu()` — the `signal/_iir.py` precedent — so the JAX backend
already gets the `O(log L)`-depth parallel speedup on GPU before any Pallas
kernel exists.

**Fused kernel** (`_kernels/cuda/selective_scan.py`) — clean-room Mamba-style
**chunked block-parallel scan**: sequential combine *within* a chunk in
SRAM/registers, parallel carry *across* chunks; the per-step state never
round-trips HBM. `PallasNotTileable` on bad `(l, d, n)` tiling → loud fallback.

**VJP:** custom backward following the **`numerics/ode.py` adjoint precedent**
(recompute states forward, accumulate grads in reverse) + `jax.checkpoint` for
`O(state)` memory — the SSM analogue of the velocity-field integrator's
recompute-backward, different combinator. The FR explicitly asks this
relationship be recorded. Second oracle for the VJP = autodiff through the
`associative_scan` reference.

**Tests:** golden `selective_scan_float32.npy`; backend parity; Hypothesis
(`scan` ≡ `associative_scan`; `D`-skip linearity; degenerate `A → 0` reduces to
a cumulative input map); FD-VJP vs the `associative_scan` autodiff oracle;
gross-mem probe (no full `(l, d_state)` trajectory stored on the fused path).

> **Status (2026-06-23): P1a SHIPPED (reference + dispatcher).** The pure-JAX
> reference (`nn/ssm/_reference.py`) implements the discretised S6 recurrence
> with both a sequential `lax.scan` oracle (`method='sequential'`) and a
> parallel `lax.associative_scan` (`method='associative'`), `method='auto'`
> flipping by platform via `default_backend_is_gpu()` — so the **GPU already
> gets the `O(log L)` parallel-scan speedup** through the reference. Public
> `nn.ssm.selective_scan` dispatches; autodiff-native (no hand VJP).
> Validated: golden corpus; reference == independent numpy naive oracle (≤1e-10);
> `sequential ≡ associative` (≤1e-10, fp64); `D`-skip linearity; `A→0` ≡ cumsum
> map; autodiff FD-check; byte-identical `backend='jax'`; jit; loud fallback on
> `pallas-cuda` (P1a stub) — 14 tests. Cataloged in the op-matrix; combined nn
> suite 47 passed / 3 skipped; ruff + mypy clean.

> **Status (2026-06-23): P1b forward SHIPPED.** Fused Pallas forward kernel
> (`_kernels/cuda/selective_scan.py`). **Triton-Pallas constraint discovered:**
> it lowers only `cumsum` — not `cumprod` / `associative_scan` / `slice` (no
> element indexing of register tiles) / `flip`. The recurrence is therefore a
> **chunked cumsum closed form** — `logP_t = A·cumsum(Δ)_t`,
> `h_t = exp(logP_t)·(cumsum(Δ·B·x·exp(−logP_t)) + h_start)`, chunked (16) so
> `exp(±logP)` stays in fp32 range, carrying the `(n,)` state across chunks via
> whole-tile sums. Grid over `(batch, channel)`; the `(l, d, n)` trajectory
> never hits HBM (gross-mem probe: fused temp ≪ reference's `(l,d,n)`).
> `custom_vjp` forward = fused kernel; **backward = fully-fused recompute-adjoint**
> (reverse chunked cumsum `a_t = dy_t C_t + dA_{t+1}a_{t+1}` via
> `rev_cumsum(z)=sum(z)−cumsum(z)+z`; `h_{t-1}` recovered as
> `(h_t−Δ B x)/exp(Δ A)`; `dx`/`dΔ` direct per-channel writes; grid-shared
> `dB`/`dC` via `plgpu.atomic_add` with zero-init `input_output_aliases`;
> `dA`/`dD` as per-program partials reduced in JAX) — so the `(l,d,n)` trajectory
> hits HBM in **neither** pass (the training-memory win). Validated on L4
> (realistic Mamba: small Δ, `A=−(1..n)`): fwd `pallas≈jax` ≤ 1.7e-7; grad parity
> vs autodiff reference ~1e-7 for `dx/dΔ/dA/dB/dC/dD` (with-D / no-D / no-batch);
> fwd + bwd gross-mem probes confirm no `(l,d,n)` temp (bwd 11 KB vs ref 135 KB);
> dispatch + loud fallback (non-pow2 `n`, non-divisible / non-pow2-chunk length,
> float64). **fp32 dynamic-range limit** (documented, both passes): within-chunk
> `|A·cumsum(Δ)| < ~80` — fine for the small-Δ Mamba regime; extreme ranges fall
> back to `backend='jax'`. **P1 COMPLETE** (reference + fused forward + fused
> backward). Remaining suite-wide: P3 fused norms (gated on a perf signal);
> at-scale wall-clock-vs-`mamba-ssm` is the perf suite's job.

> **Status (2026-06-23): pure-XLA memory-sparing `method='chunked'` SHIPPED.**
> `reference_selective_scan(method='chunked', chunk_size=)` — a Pallas-free
> analogue of the fused kernel's memory behaviour for rigs without Triton:
> chunked `lax.scan` (carry = `(d,n)` state) + within-chunk `associative_scan`,
> `n` collapsed into `y` in-body so the `(l,d,n)` trajectory is never
> materialised, `jax.checkpoint` on the body for the backward. Two wins over
> naive `associative_scan`: (1) memory — temp stays ~flat in `l` vs `O(l·d·n)`
> (L4, d=n=16: fwd `l=4096` = 60 KB vs 2.6 MB ≈ 44×; bwd 1.7 MB vs 11.6 MB ≈
> 6.6×; crossover by `l≈256`); (2) numerics — XLA-stable, so **none of the fused
> kernel's fp32 within-chunk range limit** (correct on aggressive `A`/`Δ`). It
> is a memory↔parallel-depth trade (more sequential chunk-steps), so it wins
> where memory matters and can be marginally slower in the small/latency-bound
> regime → **size-based dispatch into the public `jax` path is deferred to the
> perf suite**; the public `selective_scan` is unchanged for now. Validated
> `chunked == sequential == associative` ~1e-16 (fwd + grad), ±D / ±batch.

### 7.3 `nitrix.nn.norm.{layer_norm, group_norm, instance_norm}` — P3, CONVENIENCE (deferred)

**Surface:**

```python
def layer_norm(x: Float[Array, '... c'], weight=None, bias=None, *,
               eps: float = 1e-5, out_scale: float = 1.0,
               backend: Backend = 'auto') -> ...
# group_norm(x, num_groups, weight, bias, *, eps, out_scale, backend)
# instance_norm(x, weight, bias, *, eps, out_scale, backend)
```

**`out_scale` — the "curse of depth" hook.** A constant multiplier on the norm
output, `out = out_scale · (norm(x)·w + b)`. This covers the constant
depth-scaling family — **LayerNorm Scaling** (`out_scale = 1/√l`, Sun et al.),
residual `1/√(2N)`, DeepNorm-α, ReZero/SkipInit — at **zero marginal cost**:
it folds into the affine params (`out = norm(x)·(out_scale·w) + (out_scale·b)`),
so the fused kernel is unchanged when `out_scale = 1.0` and the per-layer
constant rides on the norm's single pass; the backward just scales the
cotangent. Per-layer-varying `out_scale` adds no specialisation cost (it folds
into `w`/`b`, which already vary per layer). Default `1.0` keeps the golden
corpus untouched. *Not* covered (orthogonal, leave to XLA's elementwise fusion):
per-channel **LayerScale** `γ` on the residual-branch epilogue — learnable, so a
custom epilogue would need a `dγ` reduction, exactly the `d_bias` / `dB` reduce
pattern already built; only worth a kernel if profiled. Sandwich-LN / nGPT are
just more norm ops, each fused independently.

**Reference** (`nn/norm/_reference.py`): pure-JAX `lax.rsqrt`-based stats
reproducing the equinox `LayerNorm`/`GroupNorm` and nimox `instance_norm` math
(we cannot import equinox — SPEC §6.2); group/instance add a reshape-to-groups
reduction.  `out_scale` applied uniformly (forward + backward).

**Fused kernel** (`_kernels/cuda/norm.py`): fork stock `layer_norm.py` /
`rms_norm.py` (single-pass Welford or two-pass in SRAM, fused affine, recompute
bwd). **Gated**: only built/promoted when the perf suite shows norm bandwidth on
a model's critical path *after* attention/scan are fused. Never blocks a model;
the XLA path is correct and ships today.

Cross-ref [`lp-normalize.md`](lp-normalize.md) (the *reference* instance-norm
stats home in `numerics.normalize`) — that owns the plain reference op; this
owns the fused-kernel variant. Do not duplicate.

> **Status (2026-06-23): P3 reference + API + `out_scale` SHIPPED; fused kernel
> stays perf-gated.** `nitrix.nn.norm.{layer_norm, group_norm, instance_norm}` —
> pure-JAX references (rsqrt / biased variance; LN over the trailing axis,
> GN/IN channels-first n-D, `instance == group(C)`) + three-level dispatch, each
> with the curse-of-depth `out_scale` hook (`out = out_scale·(x̂·w + b)`, folds
> LayerNorm-Scaling / DeepNorm / ReZero into the affine for free). The
> references are the **shippable correct path today**; `_kernels/cuda/norm.py`
> stubs raise `PallasNotTileable` → loud fallback. **The fused single-pass
> kernel is deliberately *not* built yet**: unlike attention / selective-scan
> there is **no activation cliff** to remove — the win is pure bandwidth, which
> XLA's own elementwise+reduction fusion already largely captures, so a custom
> Pallas norm must be **certified to beat XLA by the perf suite before it ships**
> (the suite's "perf wins must certify at scale" rule). The stock LN fork
> (fwd + dx + dw/db) is well-trodden and ready as a drop-in when a profiler flags
> norm bandwidth. Tests (16): golden (LN/GN/IN), numpy-oracle parity, `out_scale`
> linearity, `instance == group(C)`, normalised statistics, autodiff FD,
> byte-identical `backend='jax'`, jit, loud fallback. Op-matrix: 3 entries. Full
> nn suite 77 passed / 4 skipped.

> **Decision (2026-06-23): fused LN kernel MEASURED — not worth implementing.**
> `bench/perf_layer_norm.py` benchmarks the *stock* Pallas LayerNorm (the kernel
> we'd fork) vs the XLA reference on the L4 (`bench/PERF_LAYER_NORM.md`).
> **Verdict:** **no memory win ever** (identical peak — no cliff to remove);
> **fp32** forward at parity (0.97–1.01×) and forward+backward *slower*
> (0.61–0.95×); **bf16** forward 1.4–1.9× faster **but** forward+backward 2–4×
> *slower* (the fused backward dominates). Since the norm op carries a
> `custom_vjp` and is used in training (fwd+bwd), the fused path is a regression
> with no memory upside → **the XLA reference is the final P3 norm surface; the
> kernel stays an unbuilt loud-fallback stub.** The one real win — bf16
> *forward-only* — is a perf-suite re-eval trigger for an inference-only
> deployment, not a now item. This is the gating working as designed: measure,
> then don't build.

### 7.4 P2 — nimox-extraction blockers (cross-reference only, no code here)

affine-matrix-algebra / spherical-parameterisation / field-regularisers are
already filed under [`ilex-training-substrate.md`](ilex-training-substrate.md)
and the registration ledger. They appear in the NN ledger only so nimox can
delete its interim vendored copies when promoted to a standalone repo.
**Action:** append the nimox-extraction driver to those existing docs; build
nothing new under `nitrix.nn`.

---

## 8. Phasing & sequencing

Capability-oriented, each phase gated on correctness (not wall-clock). Phase N
exit = golden + parity + VJP + gross-mem green and loud fallback verified.

| Phase | Capability | Unblocks | Exit gate |
|---|---|---|---|
| **0** | Scaffold + golden harness | all of the below | `nitrix.nn` importable; `tests/golden/` + `tolerance.toml` + `_golden.py` + `regen_golden.py` land with a first trivial golden; CI exercises the JAX floor |
| **1** | Attention **reference** (P0a) | ilex swaps its 4 reimpls onto the JAX reference *now*; Tier-1 parity green | reference + 4-path golden + Hypothesis + autodiff VJP; `backend='jax'` only |
| **2** | Attention **fused kernel** (P0b) | flash path on Ampere | forked Triton + bias/mask tiles + `d_bias` bwd; layout/pad/fallback; `pallas ≈ jax` parity + FD-VJP (incl `d_bias`) on L4; gross-mem probe |
| **3** | Selective-scan **reference + GPU parallel path** (P1a) | nimox `_mamba.py` swaps onto reference; GPU gets `associative_scan` speedup free | two oracles + golden + Hypothesis + autodiff VJP |
| **4** | Selective-scan **fused kernel** (P1b) | flash-analogue on Ampere | chunked block-parallel scan + adjoint-recompute VJP; parity + FD-VJP; gross-mem probe |
| **5** | Fused norms (P3) — **gated** | bandwidth follow-up | only on a perf-suite profiler signal; fork norm kernels; reference unchanged |
| — | P2 doc updates | nimox standalone promotion | drivers appended to existing affine/spherical/field-regulariser FRs |

**Critical path & rationale.** Phase 1 ships first and standalone: it unblocks
ilex immediately (the reference is bit-faithful, so the swap is a no-op for
Tier-1 parity) and de-risks the kernel by pinning the oracle + golden corpus
before any Triton is written. Phases 2 and 3 are independent and can interleave
(attention kernel is a fork; scan reference is the `associative_scan` idiom);
Phase 4 depends on 3's oracle. Phase 5 is deferred behind a perf signal by
design. Estimated effort: P0 ≈ 1.5–2 wks (≈3–4 days reference + harness, rest
kernel), P1 ≈ 1.5 wks, P3 ≈ 3–4 days when triggered.

Per `IMPLEMENTATION_PLAN.md §2.3`, a JAX-only phase (1, 3) is independently
shippable if the kernel phase (2, 4) slips — the fallback floor makes the
reference a complete deliverable.

---

## 9. File-layout manifest

```
src/nitrix/nn/
  __init__.py                 # re-export scaled_dot_product_attention,
                              #   selective_scan, layer_norm, group_norm,
                              #   instance_norm; package docstring states the
                              #   FUNCTIONAL (non-module) contract
  attention/
    __init__.py               # public sdpa + resolve_backend + layout adapter
                              #   + custom_vjp wiring
    _reference.py             # einsum + bias + mask/causal + softmax oracle
  ssm/
    __init__.py               # public selective_scan + dispatch + custom_vjp
    _reference.py             # lax.scan oracle + associative_scan (platform flip)
  norm/
    __init__.py               # layer_norm / group_norm / instance_norm + dispatch
    _reference.py             # rsqrt-based references (+ reshape-to-groups)
src/nitrix/_kernels/cuda/
  attention.py                # forked Triton mha + bias/mask tiles + d_bias bwd
  selective_scan.py           # clean-room chunked block-parallel scan + adjoint
  norm.py                     # forked fused layer/group/instance norm (Phase 5)
tests/
  golden/…                    # §6
  tolerance.toml              # §6
  _golden.py                  # §6
  test_nn_attention.py        # reference: golden + Hypothesis + autodiff VJP
  test_nn_attention_kernel.py # fused: parity + FD-VJP(+d_bias) + dispatch/fallback + gross-mem
  test_nn_ssm.py
  test_nn_ssm_kernel.py
  test_nn_norm.py
  test_nn_norm_kernel.py      # (Phase 5)
tools/
  regen_golden.py             # §6
docs/design/
  nn-forward-kernels.md       # companion design doc (spun out during P0):
                              #   online-softmax bias threading, chunked-scan
                              #   tiling, d_bias derivation, adjoint VJP
```

`src/nitrix/__init__.py` top-level docstring gains an `nn` entry.

---

## 10. Testing matrix (what "correct" means here)

| Test class | attention | selective_scan | norms |
|---|---|---|---|
| Golden corpus (reference == checked-in `.npy`) | 4 paths | 1 | 3 |
| Backend parity (`pallas ≈ jax`, `tolerance.toml`) | ✓ | ✓ | ✓ |
| Hypothesis properties | row-sum=1; causal≡tril; perm-equiv over t | scan≡assoc; D-linearity; A→0 | zero-mean/unit-var stats; affine recovery |
| FD-VJP | `{bias, mask, causal}` incl **d_bias** | vs assoc-scan autodiff | vs reference autodiff |
| Dispatch / pad / loud fallback | ✓ | ✓ | ✓ |
| Gross-memory probe | no `(s,t)` materialised | no `(l,d_state)` stored | single-pass |
| dtype coverage | fp32 (fp64 where it certifies) | fp32 | fp32 |

**Out of scope (→ perf suite):** wall-clock vs FlashAttention-2 / `mamba-ssm` /
triton-norm references at model scale; tuned block-size sweeps; multi-GPU. Filed
against `bench/` when each kernel lands.

---

## 11. Non-negotiables compliance (`IMPLEMENTATION_PLAN.md §2.2`)

1. **Dependency contract** — jax + jaxtyping + numpy only; no equinox (norm
   reference reimplements the math). ✓
2. **Pure-functional surface** — `nitrix.nn` is functions; modules stay in
   nimox. ✓
3. **JAX fallback floor** — every kernel falls back to its reference, CI-exercised
   on CPU. ✓
4. **Golden corpus** — materialised in Phase 0 for every shipped op×dtype. ✓
5. **Cross-release stability** — `tolerance.toml` pinned; changes are CHANGELOG
   entries. ✓
6. **Loud fallbacks** — `PallasNotTileable` → `NitrixBackendFallback`; no silent
   perf regression. ✓

Style: jaxtyping signatures, single-quote / 79-col ruff, mypy-strict
(`disallow_untyped_defs`), immutable frozen tile-config dataclasses
(`BlockSizes`-style), `Protocol` for the scan combinator (echoing
`Semigroup`/`Monoid`).

---

## 12. Risks & open items

- **Novel kernel math is narrow but real.** The only genuinely new derivations
  are (a) additive bias inside the online softmax + its `d_bias` gradient, and
  (b) the chunked-scan adjoint. Everything else is a fork or a known idiom.
  Budget FD-VJP coverage there; spin the derivations into
  `docs/design/nn-forward-kernels.md`.
- **Arbitrary seq / head_dim vs Triton tiling** — solved by
  pad-to-multiple-with-mask + `next_power_of_2` head pad + loud fallback; never
  silent.
- **GPU validation env** — L4 via `uv pip install jax[cuda12]`, run via
  `./.venv/bin/python` (see `nitrix-gpu-dev-env`); CPU exercises reference +
  Pallas `interpret=True` parity only. cuSOLVER is irrelevant here (no dense
  factorisations).
- **Mosaic-GPU / TPU paths deferred** — Ampere/Triton baseline first; Hopper
  `attention_mgpu` and TPU `splash_attention` are follow-ups, cross-referenced
  to `mosaic-hopper-registration-kernels.md`.
- **Open (minor):** whether `selective_scan` submodule is named `ssm` or `scan`
  — `ssm` chosen (family name, room for S4/S5 later); whether to re-export the
  three norm fns at `nitrix.nn` top level or keep them under `nitrix.nn.norm`
  only — leaning re-export the headline two (attention, scan) and keep norms
  namespaced, revisit at GA.

---

## 13. Cross-references

- Context ledger / duplicate guard:
  [`nn-forward-block-kernels.md`](nn-forward-block-kernels.md).
- Atomised FRs: [`attention-kernels.md`](resolved/attention-kernels.md),
  [`selective-scan.md`](resolved/selective-scan.md),
  [`fused-norm-kernels.md`](fused-norm-kernels.md).
- House pattern design docs: `docs/design/backend-selection.md`,
  `docs/design/streaming-kernel.md`, `docs/design/backward-kernels.md`,
  `docs/design/testing-strategy.md`, `docs/design/ell-on-triton.md`.
- Precedents in code: `semiring/matmul.py` + `_kernels/cuda/semiring_matmul.py`
  (dispatch + streaming kernel); `register/_force.py` +
  `_kernels/cuda/{demons_force,lncc_force}.py` (recent kernel + size-gate +
  loud fallback); `signal/_iir.py` (scan vs associative_scan platform flip);
  `numerics/ode.py` (recompute-forward adjoint VJP).
- Perf delegation: `bench/` (`PERF_*.md`, `MEM_STREAMING_KERNEL.md`),
  [`perf-wins-must-certify-at-scale.md`](perf-wins-must-certify-at-scale.md),
  [`perf-bench-case-hardening.md`](perf-bench-case-hardening.md).
</content>
</invoke>
