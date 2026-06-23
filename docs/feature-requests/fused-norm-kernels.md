# fused-norm-kernels — fused LayerNorm / GroupNorm / InstanceNorm

> **Status (2026-06-23), against `nitrix main@6449cfa`.** Not present. Part of
> the [`nn-forward-block-kernels.md`](nn-forward-block-kernels.md) bundle —
> **P3, CONVENIENCE (perf-only)**. Lowest priority: the reference path is
> already adequate, so this is a follow-up, not a blocker.

**What.** Fused forward+backward Pallas kernels for the three normalisations
the zoo uses on the hot path: `layer_norm`, `group_norm`, `instance_norm`
(n-D, channels-first), behind the standard `backend='auto'` dispatch.

Reference homes today: `equinox.nn.{LayerNorm,GroupNorm}` (XLA) and
`ilex/nimox/modules/instance_norm.py` (`lax.rsqrt`-based). The *reference*
numerics here are fine — this item is **purely the fused kernel**.

**Why (and why only P3).** LayerNorm/GroupNorm are memory-bandwidth-bound:
the XLA lowering reads the activation tensor several times (mean, variance,
normalise, affine). A fused kernel (single pass, Welford or two-pass in SRAM)
roughly halves the bandwidth and fuses the affine. The win is real but small
relative to attention/scan, and it never blocks a model — the XLA path is
correct and ships today. So this is parked behind P0/P1 and only promoted when
a profiler shows norm bandwidth on a model's critical path.

This overlaps the already-filed [`lp-normalize`](lp-normalize.md) (which
covers the *reference* instance-norm stats, `numerics.normalize`); this doc is
strictly the **fused-kernel perf follow-up**, distinct from that reference
primitive. Cross-reference, do not duplicate: if only the reference op is
wanted, that lives in `lp-normalize`; this doc exists for when the fused kernel
is justified.

**When it bites.** Deep transformer/UNet stacks at large spatial extent, once
attention/scan are fused and norm bandwidth becomes the next bottleneck. Not
before.

**Proposed API.**

```python
def layer_norm(
    x: Float[Array, '... c'],
    weight: Float[Array, 'c'] | None = None,
    bias: Float[Array, 'c'] | None = None,
    *,
    eps: float = 1e-5,
    backend: Backend = 'auto',
) -> Float[Array, '... c']:
    ...
# group_norm(x, num_groups, weight, bias, *, eps, backend); channels-first n-D.
# instance_norm(x, weight, bias, *, eps, backend); per-sample, per-channel.
```

Home: a small new `nitrix.nn` package (or fold into `numerics.normalize` if
preferred — the impl plan should pick one; `nitrix.nn` keeps the fused-kernel
forward blocks together with `attention` / `ssm`).

**Implementation + tests.** Standard pattern: `*/_reference.py` reproduces the
equinox/`rsqrt` math; `_kernels/cuda/` holds the fused single-pass kernel with
recompute-in-backward VJP; golden corpus + tolerance row + finite-difference
gradient check. `None` on tiling failure → reference fallback.

**Acceptance.** Opt-in perf: models that adopt `backend='auto'` get the fused
path on Ampere+; correctness is unchanged (Tier-1 parity stays on `jax`). No
model is required to adopt it.
