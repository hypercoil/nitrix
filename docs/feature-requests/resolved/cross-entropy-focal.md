# Cross-entropy family + focal loss — `nitrix.metrics`

> **Status (2026-06-08): SHIPPED.** `metrics/classification.py` adds
> `bce_with_logits`, `cross_entropy_with_logits`, `focal_loss`. The stable
> BCE core (`max(x,0) − x·t + log1p(exp(−|x|))`) is a shared private helper
> that `focal_loss` reuses (dedup of the three ilex copies). All are
> `Reduction`-aware costs, pure (GPU-safe), and differentiable.
> Loss-numeric item from the 2026-06-08 ilex audit
> ([`ilex-training-substrate.md`](../ilex-training-substrate.md)).

**What.** The cross-entropy primitives, all pure and numerically careful:

1. **`bce_with_logits`** — stable binary CE from logits
   `max(x,0) − x·t + log1p(exp(−|x|))` (avoids `log(sigmoid)` overflow).
   Duplicated at `segmentation.py:234`, `classification.py:25`, and inlined
   inside `sigmoid_focal_loss`.
2. **`cross_entropy_with_logits`** — categorical CE with integer targets via
   `log_softmax` + `take_along_axis`. `segmentation.py:108`.
3. **`focal_loss`** — Lin et al. focal: stable BCE × `(1−p_t)^gamma` ×
   `alpha_t`. `segmentation.py:139` (`sigmoid_focal_loss`); `focal` builds
   directly on `bce_with_logits`.

**Drivers.** Segmentation ports (focal: `sam_med3d`, `segvol`, RetinaNet-
style heads); `brain_qc_net` / `brainiac` classification (`bce_with_logits`);
categorical CE across the SynthSeg-family label heads. The dedup motive:
three identical stable-BCE inlines today.

**API sketch.**

```python
def bce_with_logits(logits, targets, *, reduction='mean') -> Array: ...
def cross_entropy_with_logits(logits, target_idx, *, axis=1,
                              reduction='mean') -> Array: ...
def focal_loss(logits, targets, *, gamma=2.0, alpha=0.25,
               reduction='mean') -> Array: ...
```

**Pure / XLA note.** `jnp` + `jax.nn.{log_softmax}`; the stable-BCE form is
the whole point (no `inf`/`nan` under jit). `focal`'s `alpha >= 0` test is a
static-float branch (trace-safe).

**Triviality bar.** `cross_entropy_with_logits` alone is thin (`log_softmax`
+ gather); it earns its place by completing the CE surface alongside the
non-trivial stable-BCE and focal, which are the load-bearing atoms.

**Home.** `nitrix.metrics` (e.g. `metrics.cross_entropy` with binary /
categorical / focal members), following the `metrics` "return the quantity,
caller reduces" convention.

## Cross-references

- [`ilex-training-substrate.md`](../ilex-training-substrate.md) — survey context.
- [`dice-loss.md`](dice-loss.md) — pairs with CE in compound seg losses.
