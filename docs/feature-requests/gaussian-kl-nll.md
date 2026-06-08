# Diagonal-Gaussian KL / NLL — `nitrix.stats`

> **Status (2026-06-08): not started — CONVENIENCE.** Loss-numeric item from
> the 2026-06-08 ilex audit
> ([`ilex-training-substrate.md`](ilex-training-substrate.md)). Closed-form,
> log-variance-parameterised Gaussian objectives; the careful
> `exp(log_var)` form is what makes them worth a shared primitive.

**What.** Two analytic diagonal-Gaussian quantities:

1. **`kl_diagonal_gaussian`** — KL of `N(μ, diag e^{logvar})` from `N(0, I)`:
   `0.5·(μ² + e^{logvar} − 1 − logvar)`, per dim. `ilex/nimox/loss/
   functional/vae.py:77` (`kl_unit_gaussian`).
2. **`gaussian_nll`** — diagonal-Gaussian negative log-likelihood:
   `0.5·(log 2π + logvar + (x−μ)²·e^{−logvar})`. `vae.py:113`.

**Drivers.** `brain_ldm_vae`, `brain_ldm`, the `trajectory` shared-private
VAE (`pipelines/shared_private.py`), and any future variational head. The VAE
*modules* and the reparameterisation/decode glue stay in nimox; these are the
distributional kernels.

**API sketch.**

```python
def kl_diagonal_gaussian(mean, log_var, *, reduction='sum') -> Array: ...
def gaussian_nll(x, mean, log_var, *, reduction='mean') -> Array: ...
```

**Pure / XLA note.** `jnp` only; the `log_var` parameterisation avoids
`log(var)` and keeps the variance positive without a clamp; jit-clean.

**Triviality bar.** Above it: both are multi-term closed forms with a
deliberate numerically-stable parameterisation (unlike `mse`/`l1`, which the
audit leaves upstream as one-liners). They belong with `nitrix.stats`
(`covariance`, `fourier`) as the analytic-statistics surface.

**Home.** `nitrix.stats` (e.g. `stats.gaussian` with `kl_diagonal_gaussian` /
`gaussian_nll`). Optionally a general `kl_gaussian(mean0, logvar0, mean1,
logvar1)` with the unit-Gaussian case as the default.

## Cross-references

- [`ilex-training-substrate.md`](ilex-training-substrate.md) — survey context.
- [`contrastive-ssl-losses.md`](contrastive-ssl-losses.md) — the other
  distributional losses (`koleo` entropy estimator).
- `src/nitrix/stats/` — `covariance` / `fourier`, the analytic-stats surface
  this joins.
