# Soft / binary Dice — `nitrix.metrics.dice`

> **Status (2026-06-08): SHIPPED.** `nitrix.metrics.dice` (soft
> Sørensen–Dice coefficient) added in `metrics/overlap.py`, plus
> `nitrix.metrics.jaccard` (soft Jaccard / IoU) as the complementary
> overlap metric (`jaccard = dice / (2 - dice)`). Both return the
> *coefficient* (loss = `1 - metric`), operate on soft masks
> (activation-free → pure / GPU-safe), and take `axis` (per-region) +
> Laplace `smooth` (num+denom, so empty-vs-empty = 1) + `reduction`.
> Loss-numeric item from the 2026-06-08 ilex audit
> ([`ilex-training-substrate.md`](ilex-training-substrate.md)). The single
> major segmentation-overlap metric absent from `nitrix.metrics` (which had
> `ssd`/`ncc`/`lncc` + `mutual_information`/`correlation_ratio`).

**What.** Soft Dice overlap loss `1 − 2·Σ(p·t) / (Σp + Σt + eps)`, reducing
over spatial axes per (sample, class). Two input forms collapse to one
primitive:

- **multi-class on probabilities** — `(B, C, *spatial)` probs vs one-hot
  target → `(B, C)`. `ilex/nimox/loss/functional/segmentation.py:49`
  (`soft_dice_loss`).
- **binary on logits** — `sigmoid(logit)` front-end + Laplace `smooth`
  → `(B,)`. `segmentation.py:201` (`binary_dice_loss`).

These are the same math; ship one primitive with an `on='probs'|'logits'`
(or pre-activation) switch and a multi-class/binary axis convention, not two
public symbols.

**Drivers.** Effectively every segmentation port — `synthseg`,
`wmh_synthseg`, `supersynth`, `brats_segresnet`, `fastsurfer_cnn`,
`pglands_seg`, `fsm_seg`, `sam_med3d`, `segvol`, `wholebrain_unest` — plus
the nimox loss library's own `segmentation` family.

**API sketch.**

```python
def dice(
    pred: Float[Array, 'B C *spatial'],
    target: Float[Array, 'B C *spatial'],
    *,
    eps: float = 1e-7,
    reduction: Literal['none', 'mean', 'sum'] = 'mean',
) -> Float[Array, '...']:
    """Soft Dice overlap (returns the *coefficient*; loss = 1 - dice)."""
```

Ship `dice` (the coefficient, sibling of `ncc`/`lncc`) and let the loss be
`1 - dice`, matching the existing `metrics` convention (metrics return the
similarity, callers form the loss). Consider a `weight=` per-class option
(generalised/Tversky is a natural extension — flag, don't build yet).

**Pure / XLA note.** `jnp` reductions only; eps-stabilised; jit-clean.

**Home.** `nitrix.metrics.dice`.

## Cross-references

- [`ilex-training-substrate.md`](ilex-training-substrate.md) — survey context.
- [`cross-entropy-focal.md`](cross-entropy-focal.md) — the CE family that
  pairs with Dice in compound seg losses.
- `src/nitrix/metrics/intensity.py` — `ssd`/`ncc`/`lncc`, the existing
  similarity metrics this joins.
