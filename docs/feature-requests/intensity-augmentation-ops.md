# Intensity-augmentation ops — `nitrix.augment.intensity`

> **Status (2026-06-08): not started — CONVENIENCE.** Training-substrate
> item from the 2026-06-08 ilex audit
> ([`ilex-training-substrate.md`](ilex-training-substrate.md)). Four
> image-only intensity perturbations, all pure keyed `(Array,…)->Array`.

**What.** The image-only ("perturbation"-role) intensity augmentations from
the FM pretraining recipe (3DINO §5), each a pure deterministic transform
(the `U(range)` draw being a thin keyed wrapper):

1. **`gamma_contrast`** — per-volume min/max-bracket normalise, raise to a
   `gamma`, rescale back (`normed**gamma * span + min`). `gamma<1` raises
   contrast. `ilex/train/augment/intensity.py:71` (`random_contrast`).
2. **`histogram_shift`** — piecewise-linear intensity remap through
   `n_control_points` equally-spaced reference levels, each perturbed by a
   random offset (endpoints pinned, `cummax` to enforce monotonicity),
   applied via `jnp.interp` (MONAI `RandHistogramShift`).
   `intensity.py:111` (`random_histogram_shift`).
3. **`gaussian_noise`** — additive i.i.d. `N(0, sigma^2)`, `sigma ~ U`.
   `intensity.py:38` (`random_gaussian_noise`).
4. **`rician_noise`** — MR magnitude noise `sqrt((x+n_r)^2 + n_i^2)`,
   `n_r,n_i ~ N(0,sigma^2)`. `ilex/train/augment/lab2im.py:138`
   (`random_rician_noise`). At `sigma=0` this is `|x|`.

**Drivers.** FM pretraining (brainiac / 3DINO via
`augment/compose.py` perturbation pipelines); `rician_noise` also feeds the
lab2im chain (`labels_to_image:328`).

**API sketch** (deterministic core; keyed sampler is the thin wrapper):

```python
def gamma_contrast(x, gamma, *, bracket='minmax') -> Array: ...
def histogram_shift(x, refs, shifted) -> Array:          # remap through a table
    ...
def gaussian_noise(x, key, *, sigma) -> Array: ...
def rician_noise(x, key, *, sigma) -> Array: ...
```

**Pure / XLA note.** All four are `jnp`-only + `jax.random`; jit-clean.
`histogram_shift`'s remap is `jnp.interp` — it can reuse
`nitrix.signal.interpolate`'s linear path; the control-point construction +
monotone-cummax is the new bit. `histogram_shift` validates
`n_control_points >= 2` on a static int (trace-safe).

**Relation to existing nitrix.** Distinct from `nitrix.bias.histogram_match`
(Nyúl–Udupa landmark *matching* between volumes) and
`nitrix.bias.sharpen_histogram` — those match/sharpen toward a reference;
this randomly *perturbs* the transfer curve. Distinct from
`nitrix.numerics.normalize` (no gamma / no noise there). Noise generators
are the first additive-noise primitives in nitrix.

**Home.** `nitrix.augment.intensity` (gamma, histogram_shift); the noise
generators could equally sit in `nitrix.signal.noise` — see the namespace
open question in the ledger.

## Cross-references

- [`ilex-training-substrate.md`](ilex-training-substrate.md) — survey context.
- [`intensity-normalize-variants.md`](intensity-normalize-variants.md) — the
  deterministic percentile/zscore normalisers (the inference-side tail);
  this doc is the *augmentation* (training-side) perturbations. See its note
  on the `nonzero`-masked percentile variant still wanted by FM
  `percentile_normalize` (`intensity.py:166`).
- `src/nitrix/signal/interpolate.py` — the linear interp `histogram_shift`
  reuses.
