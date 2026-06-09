# Contrastive / self-supervised losses — `nitrix.metrics` / `nitrix.stats`

> **Status (2026-06-09): SHIPPED.** `metrics/contrastive.py` adds `nt_xent`
> (InfoNCE, finite `−2/τ` diagonal self-mask, adjacent-pair positives),
> `dino_cross_entropy` (teacher centering+sharpening, `stop_gradient`),
> `ibot_cross_entropy` (per-token, safe masked-mean over masked positions),
> and `koleo` (NN-via-max-cosine entropy regulariser). All reuse
> `numerics.l2_normalize`, are `reduction`-aware, and differentiable; the
> teacher `center` is an argument (its EMA stays upstream). Loss-numeric
> item from the 2026-06-08 ilex audit
> ([`ilex-training-substrate.md`](ilex-training-substrate.md)).

**What.** Four pure-tensor SSL objectives:

1. **`nt_xent`** (SimCLR InfoNCE) — L2-normalise, scaled cosine-sim matrix
   `z@zᵀ/τ`, **finite** diagonal self-mask (`−2/τ·I`, deliberately not
   `−inf` to dodge `inf−inf` under jit), `log_softmax`, gather the paired
   positive. `ilex/nimox/loss/functional/contrastive.py:32`.
2. **`dino_cross_entropy`** — teacher centering+sharpening
   `softmax((teacher−center)/τ_t)` (stop-grad) × student `log_softmax(/τ_s)`
   CE. `dino.py:55` (`dino_loss`).
3. **`ibot_cross_entropy`** — same centering/sharpening CE per patch token,
   masked-mean over masked positions (denominator clipped at 1).
   `dino.py:108` (`ibot_loss`).
4. **`koleo`** — Kozachenko–Leonenko differential-entropy / feature-spread
   regulariser: nearest-neighbour via max cosine sim (self-masked),
   `−log(d_NN + eps)`. `dino.py:166` (`koleo_loss`).

**Drivers.** `ilex/train/pipelines/{dino3d,simclr3d}.py` (3DINO, SimCLR,
BrainIAC FM pretraining). The pipelines themselves (heads, EMA-teacher, step
builders) stay in nimox; these are the loss kernels they call.

**API sketch.**

```python
def nt_xent(z, key=None, *, temperature=0.5) -> Array: ...   # z: (2N, d) views
def dino_cross_entropy(student_logits, teacher_logits, center, *,
                       student_temp, teacher_temp) -> Array: ...
def ibot_cross_entropy(student_logits, teacher_logits, center, mask, *,
                       student_temp, teacher_temp) -> Array: ...
def koleo(z, *, eps=1e-8) -> Array: ...
```

**Pure / XLA note.** `jnp` + `jax.nn.{log_softmax}` + `lax.stop_gradient`;
all jit-clean. The finite-mask trick in `nt_xent` and the safe masked-mean in
`ibot` are the reusable numerical care worth capturing once. The DINO
**center EMA update** (`update_center`, `dino.py:213`) is training-state
bookkeeping → stays upstream (or a generic `numerics.ema`; see
[`lp-normalize.md`](lp-normalize.md) note).

**Home.** `nitrix.metrics` for `nt_xent` / `dino_cross_entropy` /
`ibot_cross_entropy`; `koleo` fits `nitrix.stats` (entropy estimators) or
`metrics` — it is a kNN-via-inner-product entropy estimator, reusable beyond
DINO.

## Cross-references

- [`ilex-training-substrate.md`](ilex-training-substrate.md) — survey context.
- [`gaussian-kl-nll.md`](gaussian-kl-nll.md) — the other distributional
  losses headed for `nitrix.stats`.
