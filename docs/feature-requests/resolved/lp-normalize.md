# Lp / unit normalize + instance-norm statistics — `nitrix.numerics.normalize`

> **Status (2026-07-06): SHIPPED.** `numerics.l2_normalize` / `lp_normalize`
> (torch `F.normalize` clamp-eps semantics) + `numerics.instance_norm`
> (statistics-only, rank-agnostic `axes=`), in `numerics/normalize.py` and
> tested in `test_numerics.py` (6 tests). Model-numeric item from
> the 2026-06-08 ilex audit
> ([`ilex-training-substrate.md`](../ilex-training-substrate.md)). Two small,
> recurrently re-implemented reductions absent from `numerics.normalize`
> (which has `zscore`/`robust_zscore`/`psc`/`intensity`/`percentile_rescale`/
> `demean` — but no plain unit-vector normalize and no instance-norm stat).

**What.**

1. **`l2_normalize` / `unit_normalize`** — `x / max(‖x‖_p, eps)` along an
   axis (torch `F.normalize` eps semantics). `ilex/models/krakencoder/
   _krakencoder.py:59` (`_l2_normalise`); also implicit in `nt_xent` /
   `koleo` (see [`contrastive-ssl-losses.md`](contrastive-ssl-losses.md)),
   which L2-normalise before the cosine-sim matrix.
2. **`instance_norm`** — per-sample/per-channel zero-mean/unit-(biased)var
   over a configurable set of spatial axes, `(x−μ)·rsqrt(var+eps)` (the
   *statistics*, not the trainable affine). `ilex/nimox/modules/
   instance_norm.py:51` (`instance_norm3d`, hard-coded rank-3) — make it
   rank-agnostic via an `axes=` argument.

The `InstanceNorm3d` `eqx.Module` (the learnable `γ/β` wrapper) stays in
nimox and calls this kernel.

**Drivers.** `krakencoder` (`_l2_normalise`); every UNETR/SegResNet/nnU-Net-
style decoder (`unetr_blocks.py`, `instance_norm3d`); the SSL losses
(L2-normalise of embeddings).

**API sketch.**

```python
def l2_normalize(x, *, axis=-1, eps=1e-12) -> Array: ...
def lp_normalize(x, *, p=2.0, axis=-1, eps=1e-12) -> Array: ...
def instance_norm(x, *, axes, eps=1e-5) -> Array: ...   # statistics only
```

**Pure / XLA note.** `jnp` reductions; jit-clean and rank-agnostic. These are
small, but the dedup + reuse (SSL losses, krakencoder, every decoder block)
justifies one home.

**Borderline siblings (note, do not build yet).** `cosine_momentum_schedule`
(`ema.py:116`) and the pytree-level `ema_update` (`ema.py:63`) are
training-loop utilities — keep upstream unless a generic `numerics.ema` /
`numerics.schedules` is wanted (the DINO center EMA would also use it).

**Home.** `nitrix.numerics.normalize`.

## Cross-references

- [`ilex-training-substrate.md`](../ilex-training-substrate.md) — survey context.
- [`contrastive-ssl-losses.md`](contrastive-ssl-losses.md) — L2-normalize
  consumer.
- `src/nitrix/numerics/normalize.py` — the normalisation family this joins.
