# Generative bias field — simulated INU — `nitrix.augment`

> **Status (2026-06-08): not started — ENABLING.** Training-substrate item
> from the 2026-06-08 ilex audit
> ([`ilex-training-substrate.md`](ilex-training-substrate.md)). The
> *forward simulation* counterpart to the corrective `nitrix.bias` family —
> a distinct, complementary primitive.

**What.** Generate a smooth low-frequency multiplicative intensity
non-uniformity (INU / bias) field with mean ≈ 1: draw a small
low-resolution Gaussian field (`std ~ U(0, max_std)`), upsample it linearly
to the full spatial shape, and exponentiate. `max_std == 0` ⇒ all-ones
(no-op). This is the lab2im bias-field augmentation; it *simulates* INU to
make a network robust to it — the inverse problem to N4/B-spline correction.

**Boundary nuance (important).** `nitrix.bias` is *corrective* (N4,
B-spline, histogram). This is *generative*. Keep them as distinct surfaces
so the names do not collide; a generative `simulate_bias_field` sitting
beside the corrective `bias_field_correction` is intentional.

**Driver.** `ilex/train/augment/lab2im.py:116` (`random_bias_field`),
applied in `labels_to_image` (`:326`) and via `spec._apply_bias_field`
(`spec.py:71`). The user-emphasised "simulated INU" augmentation.

**API sketch.**

```python
def simulate_bias_field(
    spatial_shape: tuple[int, ...],
    key: Array,
    *,
    max_std: float = 0.5,
    grid_fraction: float = 0.04,   # low-res grid = round(shape * fraction)
) -> Float[Array, '*spatial']:
    """exp(upsample(N(0, std^2) on a coarse grid)); multiplicative, mean ~ 1."""
```

**Note / possible richer variant.** The shipped lab2im form is a stationary
Gaussian-on-coarse-grid field. A *local-intensity-histogram-conditioned* INU
(the Tier-2 "careful thought" augmentation) is a candidate richer kernel if
it lands upstream; track it here rather than opening a parallel doc.

**XLA note.** `jax.random.normal` + `jax.image.resize(method='linear')` +
`exp`; the coarse-grid shape is computed from the static `spatial_shape`, so
jit-clean. Could optionally reuse the existing B-spline upsampling in
`nitrix.bias._bspline` for a smoother field.

**Home.** `nitrix.augment` (generative), cross-linked from `nitrix.bias`.

## Cross-references

- [`ilex-training-substrate.md`](ilex-training-substrate.md) — survey context.
- [`lab2im-gmm-synthesis.md`](lab2im-gmm-synthesis.md) — the GMM render this
  multiplies.
- `src/nitrix/bias/` — the *corrective* bias family this complements
  (`n4`, `bspline_approximate`, `histogram_match`).
