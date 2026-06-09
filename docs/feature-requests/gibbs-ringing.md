# Gibbs (truncation) ringing artefact — `nitrix.augment.intensity`

> **Status (2026-06-09): SHIPPED.** `augment/intensity.py` adds
> `gibbs_ringing(x, alpha, *, axes=None)` — injects Gibbs ringing by hard
> high-frequency k-space truncation (`fftn` → zero frequencies outside a
> normalised radius `1 − alpha` → `ifftn` → real). `alpha=0` is the
> identity; N-D; deterministic in `alpha` (caller draws it for a random
> augmentation); linear in `x` so differentiable; channels-last via `axes`.
> Requested 2026-06-09 during the Phase-2 augmentation review.

**What.** Gibbs ringing is the oscillation near sharp edges produced when a
signal's spectrum is sharply truncated — the artefact of a finitely-sampled
(band-limited) MR acquisition. It is a standard appearance augmentation
(MONAI `RandGibbsNoise`). The model is direct and physical: truncate
high-frequency k-space and transform back; the **sharp cutoff is what
produces the ringing** (a smooth taper would merely blur).

**Why nitrix.** Pure `(Array, …) -> Array` k-space numerics (FFT + a radial
mask), no container/IO. Sits with the other image-only intensity
perturbations (`gamma_contrast`, noise) in `augment/intensity.py`.

**API.**

```python
def gibbs_ringing(
    x: Float[Array, '...'],
    alpha: float,                 # truncation strength in [0, 1]; 0 = identity
    *,
    axes: Optional[Sequence[int]] = None,   # transform axes (spatial)
) -> Float[Array, '...']:
    ...
```

**Implementation notes.** Centered transform (`fftshift(fftn)`); the mask is
`radius <= (1 - alpha) * radius_max` over a normalised radial frequency
grid, so `alpha=0` keeps the whole spectrum (exact identity) and `alpha=1`
keeps only DC. The forward/inverse FFT pair composes to the identity, so the
`alpha=0` round-trip is exact. Output is the real part (the input is real and
the truncation is conjugate-symmetric). The random variant (draw
`alpha ~ U(range)`) is a one-line caller wrapper.

**Home.** `nitrix.augment.intensity`.

## Cross-references

- [`ilex-training-substrate.md`](ilex-training-substrate.md) — survey context
  + the augmentation tier.
- [`intensity-augmentation-ops.md`](intensity-augmentation-ops.md) — the
  sibling intensity perturbations (gamma, histogram-shift, noise).
