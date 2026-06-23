# neural-network forward-block kernels — context & ledger

> **Status (2026-06-23), against `nitrix main@6449cfa`.** None of the four
> items below exist in `src/nitrix` (verified: no `attention`, `ssm`,
> `selective_scan`, or fused-norm kernel; `ell_row_softmax` in `semiring/`
> is unrelated). This is a **new consumer family**: the first
> *neural-network forward-block* kernel tier in nitrix. The existing two
> consumer families — [`ilex-pipeline-substrate.md`](ilex-pipeline-substrate.md)
> (vendored-model pre/post-processing → thrux) and
> [`ilex-training-substrate.md`](ilex-training-substrate.md) (augment / loss /
> model numerics) — already filed and largely shipped every loss, augment,
> geometry, and stats duplication in the model zoo. Neither audit considered
> the **transformer and state-space forward blocks**, which is the gap this
> ledger collects.

## Why this is a new family, not an addition to an existing doc

`ilex-training-substrate.md` explicitly fenced "all `eqx.Module`s stay in
nimox; only their *extractable pure kernels* move." It then enumerated the
extractable kernels it found (affine, PCA, KL/NLL, Dice, NT-Xent, …) but did
**not** look at the attention or selective-scan math inside the ViT / Swin /
SAM / SegVol / UNesT / Mamba modules. Those inner contractions are exactly
the kind of primitive the boundary admits:

> A primitive belongs in nitrix iff it is a pure `(Array, …) -> Array`
> expressible in `jax` + `jaxtyping` + `numpy` only, **and** is reusable
> substrate rather than model glue, container/IO orchestration, or a
> trainable module. — `ilex-training-substrate.md`

`scaled_dot_product_attention(q, k, v, …) -> out` and
`selective_scan(u, Δ, A, B, C, …) -> y` are pure array→array contractions.
The `eqx.Module`s that *hold* `W_qkv` / the SSM projections stay in nimox; the
**inner kernel** is the nitrix atom — and the one place a hardware-aware
(Pallas/Triton) implementation pays off most.

## The driver

ilex's nimox layer hand-rolls these blocks four-plus times each, all as
single-implementation reference `einsum + softmax` / `lax.scan`, with **zero**
backend dispatch:

- attention: `ilex/nimox/architectures/{vit3d.py, swin_vit.py, sam3d.py,
  segvol.py, swin_unetr.py, unest.py}` — four distinct `einsum('…')` +
  `jax.nn.softmax` reimplementations (dense, windowed-with-relative-bias,
  causal-masked, cross-attention).
- selective scan: `ilex/nimox/modules/_mamba.py` — a reference `lax.scan`
  recurrence, consumed by `neurostorm` and `swift`.

The prompt for the ilex hardening cycle calls for "transformers and state
space models supporting hardware-aware modules such as flash attention,"
dispatching on backend + config. nitrix already owns the dispatch machinery
(`resolve_backend` → `pallas-cuda` | `jax`, golden-corpus parity, loud
`NitrixBackendFallback`); these kernels are the missing payload.

## Parity contract (the load-bearing constraint)

ilex gates every ported model with a **forward-parity oracle**
(`ilex/docs/design/parity-oracle.md`): the ilex forward must match the
upstream framework forward within `atol 1e-6 / rtol 1e-5`. A fused
flash-attention / selective-scan kernel is *mathematically* equivalent to the
reference but **not bit-exact**, so ilex pins its oracle to the **`jax`
reference backend** and relies on **nitrix's own golden-corpus parity**
(`pallas-cuda ≈ jax` within nitrix's pinned tolerance) to certify the fast
path. Concretely, every item below MUST ship:

1. a `jax` reference impl that reproduces the *current nimox math exactly*
   (so ilex Tier-1 parity is unchanged when it swaps to the nitrix reference);
2. a `pallas-cuda` fused kernel with a golden-corpus + tolerance-matrix test
   against the reference (nitrix Tier-2);
3. a custom VJP, finite-difference checked per the house pattern.

This two-tier split means **no ilex parity fixture ever runs the
non-deterministic fused path**, and nitrix owns the fast-path correctness
budget where it belongs.

## Dispatch & style fit

Each kernel follows the shipped `semiring_matmul` template verbatim:
`fn(…, *, backend='auto') -> Array` with `resolve_backend(backend)` →
`_kernels/cuda/*` (returns `None` on tiling failure → reference fallback) or
`*/_reference.py`. jaxtyping signatures, single-quote / 79-col ruff,
mypy-strict, golden corpus under `tests/golden/`, tolerance in
`tests/tolerance.toml`. New top-level packages `nitrix.attention` and
`nitrix.ssm` (flat-domain style, matching `geometry` / `metrics` / `semiring`).

## Ledger (priority-marked)

Priority: **P0** = blocks the largest model set + headline of the ilex cycle;
**P1** = blocks a named model family; **P2** = unblocks the nimox extraction
(already filed elsewhere — cross-reference, do not duplicate); **P3** =
perf-only follow-up, reference path already adequate.

| Pri | Item | Doc | Severity | Home | Status |
|---|---|---|---|---|---|
| **P0** | Scaled dot-product / flash attention (dense + windowed-bias + causal + cross) | [attention-kernels](attention-kernels.md) | ENABLING | `nitrix.attention` | proposed |
| **P1** | Selective state-space scan (Mamba/S6) | [selective-scan](selective-scan.md) | ENABLING | `nitrix.ssm` | proposed |
| **P2** | Affine param↔matrix algebra (unblocks nimox affine vendor) | [affine-matrix-algebra](affine-matrix-algebra.md) *(existing)* | ENABLING | `geometry.transform` | filed 2026-06-08 |
| **P2** | Spherical parameterisation (JOSA) | [spherical-parameterisation](spherical-parameterisation.md) *(existing)* | ENABLING | `geometry.sphere` | filed |
| **P2** | Jacobian-determinant / field regularisers (JOSA jacobian) | [field-regularisers](field-regularisers.md) *(existing)* | ENABLING | `register.regulariser` | partly shipped |
| **P3** | Fused LayerNorm / GroupNorm / InstanceNorm (forward+backward) | [fused-norm-kernels](fused-norm-kernels.md) | CONVENIENCE (perf) | `nitrix.nn` | proposed |

**P2 note.** The affine / spherical / jacobian gaps are already filed under
`ilex-training-substrate.md` and `layering-roadmap.md §6`; they are listed
here only because closing them lets nimox **delete its interim vendored
copies** (`nimox/modules/affine.py`, the JOSA `_spherical.py` / `_njf.py`)
when it is promoted to a standalone repo (ilex axis ii). No new spec is
needed; this ledger just records them as nimox-extraction blockers so the
sequencing is visible. New drivers should be appended to those existing docs.

## Out of scope (the boundary, restated)

- The `eqx.Module`s that hold the params (`MultiHeadAttention`, the Mamba
  block, `PatchEmbed`, relative-position-bias tables) stay in **nimox**; only
  the inner pure kernel moves. nitrix never sees a learnable module.
- General convolution stays on XLA/cuDNN (`lax.conv_general_dilated`); it is
  already optimal and is not requested here. `HyperConv3DFromDense` stays a
  nimox module over `lax.conv`.
- Windowing / patch-partition / unfold for Swin stays in the nimox module;
  nitrix's attention kernel takes already-windowed `q/k/v` plus an additive
  bias. The kernel is layout-agnostic (`"... h s d"`), not Swin-aware.
- No container / IO awareness (that is thrux); no loss/scalarisation namespace
  (that is `nimox.loss`); no FFI / native-suite kernels (that is niffi — and
  per the ilex-cycle steer, flash attention is **nitrix-via-Pallas, never
  niffi**).
