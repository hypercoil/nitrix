# GMM label→image synthesis (lab2im) — `nitrix.augment`

> **Status (2026-06-08): SHIPPED.** `augment/synthesis.py` adds
> `gmm_label_to_image(label_map, means, stds, key, *, nonneg)` — the
> per-label Gaussian-mixture render `means[label] + stds[label]·N(0,1)`.
> Takes **explicit** per-label stats (deterministic render; randomised
> contrast is a 2-line caller draw of `means`/`stds`, and explicit stats
> also serve fixed intensities), gathering over a static `n_labels` table.
> Training-substrate item from the 2026-06-08 ilex audit
> ([`ilex-training-substrate.md`](../ilex-training-substrate.md)).

**What.** Render an MR-like intensity volume from an integer anatomical
label map by drawing a per-label Gaussian `(mean, std)` and sampling
`image = mean[label] + std[label] · N(0, 1)`. This is the BBillot `lab2im`
generative augmentation behind the whole synth\* family
(SynthSeg/SynthStrip/SynthSR/SynthMorph) and now ilex FM pretraining
(brainiac / 3DINO). Pure gather + keyed normal draw.

**Driver.** `ilex/train/augment/lab2im.py:66` (`sample_gmm_intensities`),
consumed by `labels_to_image` (`:254`) and the `Lab2imSource` toggle in the
FM scaling-law study. The deformation half (`deform_label_map`,
`sample_svf_displacement`, `sample_affine_matrix`) is covered by
[`geometric-augmentation-ops`](geometric-augmentation-ops.md) +
[`affine-matrix-algebra`](affine-matrix-algebra.md); this doc is the
intensity render.

**API sketch.**

```python
def gmm_label_to_image(
    label_map: Int[Array, '*spatial'],
    means: Float[Array, 'n_labels'],
    stds: Float[Array, 'n_labels'],
    key: Array,
    *,
    nonneg: bool = True,
) -> Float[Array, '*spatial']:
    """means[label] + stds[label] * N(0,1); optional clamp at 0."""
```

Keep the **deterministic render** (explicit per-label `means`/`stds` +
residual-noise `key`) as the nitrix atom; the `U(mean_range)` /
`U(std_range)` prior draw is a thin keyed wrapper (a one-liner upstream, or
an optional `from_ranges=` convenience in nitrix). `n_labels` is the static
gather-table size.

**XLA note.** Pure `jnp` gather + `jax.random.normal`; jit-clean. Channel
axis: nitrix convention is channels-last / channel-free `*spatial`; the ilex
`[None]` channels-first lift is a caller concern.

**Home.** `nitrix.augment` (new generative-synthesis surface). Pairs with
[`generative-bias-field`](generative-bias-field.md) as the lab2im substrate.

## Cross-references

- [`ilex-training-substrate.md`](../ilex-training-substrate.md) — survey context
  + the boundary.
- [`generative-bias-field.md`](generative-bias-field.md),
  [`intensity-augmentation-ops.md`](intensity-augmentation-ops.md) — the rest
  of the lab2im intensity chain.
- [`geometric-augmentation-ops.md`](geometric-augmentation-ops.md) — the
  label-map deformation half of the chain.
