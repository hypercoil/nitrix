# FR (doc): instantaneous_phase / instantaneous_frequency — narrowband caveat

**Status:** RESOLVED (2026-07-06) · **Type:** documentation · **Severity:** low
· **Source:** nitrix-perf-bench signal cases. Fixed in `signal/fourier.py`: a
narrowband **Notes** caveat added to both `instantaneous_phase` and
`instantaneous_frequency`.

## Summary

`stats.fourier.instantaneous_phase` (`unwrap(angle(analytic_signal))`) and
`instantaneous_frequency` (`diff(unwrap(angle))`) are only well-posed on a
**narrowband** signal — one with a single, well-defined instantaneous frequency
well below Nyquist and an envelope that stays away from zero (an analytic /
AM-FM signal, a chirp, a filtered band). The docstrings should say so. On
**broadband** input (e.g. white noise) the instantaneous frequency is
intrinsically ill-defined, and the result is unstable in a way that is **not an
fp32 bug and not fixable by reformulating the algorithm**.

## Why (the two failure modes on broadband input)

1. **Nyquist (±π) wrap ambiguity.** The per-sample phase increment `Δφ` can land
   at ±π (frequency at Nyquist). There `e^{+iπ} = e^{−iπ}`, so `+0.5` and `−0.5`
   cycles are the *same* rotation — the wrap direction is genuinely undefined.
   fp32 and fp64 round to opposite sides, differing by a full cycle at that
   sample. Broadband noise has continuous energy up to Nyquist, so some
   increments always sit there.
2. **Low-envelope phase noise.** Where `|analytic_signal| ≈ 0`, the phase is
   undefined and a tiny perturbation swings it across the whole circle.

## Evidence (L4, instantaneous_frequency vs the fp64 oracle, t=16384)

`rel_to_tol` (gate rtol=1e-3/atol=1e-4), worst over the batch:

| input | rel_to_tol | near-Nyquist fraction |
|-------|-----------|-----------------------|
| white noise                 | **1666×** | 1.2e-2  |
| band-limited noise (σ=4)    | **1666×** | 7.9e-5  |
| narrowband chirp (≪ Nyq)    | **0.23×** | 0       |

The worst sample has `Δφ = −π` exactly (oracle `+0.5` cycle vs fp32 `−0.5`,
`|err| = 1.0` cycle). Note band-limited *noise* still fails — even a handful of
near-Nyquist / low-envelope samples is enough — only a genuinely narrowband
signal (the chirp) is clean.

**The conjugate-product reformulation does NOT fix it (verified).** Computing the
phase increment as `angle(z[n+1]·conj(z[n]))` (the textbook unwrap-free
estimator) gives the *identical* 1666× error — it has the same near-±π rounding
ambiguity. So this is not a stabilizable implementation; it is the nature of
instantaneous frequency on broadband signals.

## Suggested action (documentation)

Add to both docstrings a short "Use on narrowband signals" note:

- These estimate a *single* instantaneous frequency; apply them to a narrowband
  / analytic signal (band-pass first), not to broadband or noisy input.
- On broadband input the instantaneous frequency is ill-defined near Nyquist
  (±π wrap ambiguity) and where the envelope nears zero; results there are
  unstable and not bit-reproducible across precisions — by nature, not by
  implementation.
- Optionally mention band-passing (or EMD/IMF decomposition) as the standard
  pre-step.

(The perf-bench cases now feed these ops a narrowband chirp, the realistic
regime, where fp32 matches the fp64 oracle.)
