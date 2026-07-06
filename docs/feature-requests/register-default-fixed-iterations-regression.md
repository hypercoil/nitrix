# Registration default `mode='fixed'` silently ~2× regresses out-of-box single-pair rigid/affine — `nitrix.register`

> **Status (2026-06-29): OPEN.** Surfaced by **nitrix-perf-bench** while
> regenerating the registration suite across the register refactor
> (`a5b7e80` → `b09f40c`): the rigid/affine cases looked ~2× slower on CPU. Root
> cause is a **default convergence-policy flip**, *not* a per-iteration
> regression. File under the **Registration suite** family.
>
> *Note:* an earlier doc
> ([`registration-early-stopping-while-loop.md`](resolved/registration-early-stopping-while-loop.md))
> shipped `mode='early_exit'` as an opt-in and reasoned about which recipes it
> helps. I understand several of that doc's per-recipe conclusions (including
> "no benefit for SyN") to have since been **refuted**, so this FR does **not**
> rely on them — every claim below is from a fresh perf-bench A/B on the two
> pins. The default-policy question should likewise be re-decided from current
> measurement, not that doc.

**What.** For the single-pair matrix recipes (`rigid_register` /
`affine_register`), the *default* optimiser policy changed from early-exiting at
convergence to running the full fixed iteration budget, so a user who keeps the
defaults now pays ~2× for the same result. Matched-policy A/B on one L4 host
(`levels=3, iterations=30`, 48³, CPU, x64), perf-bench harness:

| nitrix | spec (defaults) | rigid L3×30 | affine L3×30 |
|---|---|---|---|
| `a5b7e80` (pre-refactor) | default `convergence='auto'` (early-exits) | **95.8 ms** | **126.5 ms** |
| `b09f40c` (post-refactor) | default `mode='fixed'` | **182.6 ms** | **246.9 ms** |
| `b09f40c` | explicit `mode='early_exit'` | **93.6 ms** | **126.8 ms** |

`b09f40c` with `mode='early_exit'` reproduces the *old default* to within noise
(93.6 vs 95.8 ms; 126.8 vs 126.5 ms); the new default `mode='fixed'` is the
outlier (~2× on rigid, ~1.9× on affine). **There is no per-iteration regression**
— at a matched convergence policy the two pins are identical; the entire delta is
the default now running ~2× more iterations. Recovery is unaffected: early-exit
stops *at* convergence, and perf-bench's rigid/affine recovery pins pass
unchanged with it. The likely cause is the convergence-API refactor that split
`convergence='auto'` into `mode` + `convergence`, landing `mode='fixed'` as the
spec default (`_core.py` `RegistrationSpec.mode: ConvergenceMode = 'fixed'`).

**Why it matters.**

1. **Most users keep the defaults.** `rigid_register(m, f,
   spec=RegistrationSpec(levels=3, iterations=30))` (or the bare spec default)
   silently costs ~2× for **identical** output — second-order GN/LM on these
   recipes converges well inside the budget, so the extra iterations are spin a
   defaults-trusting user neither asked for nor benefits from.
2. **It breaks like-for-like benchmarking / real-world comparison.** The
   community references early-stop internally — ANTs runs its preset with a
   `[…,1e-7,8]` windowed convergence criterion; dipy's affine MI runs an L-BFGS
   that converges before the `level_iters` cap. With `mode='fixed'`, nitrix runs
   its **full** budget while the refs early-stop, so nitrix does ~2× the
   iterations of the tools it is measured against — a self-inflicted handicap.
3. **It is silent.** Same call, same recovered transform, 2× the wall-clock, no
   warning.

**Proposed approach (raised for your call — not prescribing).** Reconsider the
*default* for the recipes that demonstrably converge early. I have clean
evidence only for **single-pair rigid/affine**, where defaulting `mode` to
`'early_exit'` restores the prior (faster) out-of-box behaviour with identical
recovery. Whether the early-exit default should extend further (e.g. SyN — which
the superseded doc wrongly dismissed) is an open question to settle from current
measurement, not from prior claims. If `fixed` must stay the global default for
out-of-box reproducibility / reverse-differentiability, the lighter alternative
is a one-time hint on the fast-converging matrix path pointing at
`mode='early_exit'`, plus a prominent `RegistrationSpec.mode` docstring note that
the default can double work on these recipes.

**Composition / blast radius.** Default-only change; no signature or
`RegistrationResult` change for the hint variant, only the `mode` default for the
flip variant. Recovery is identical within tolerance (early-exit stops at
convergence). `mode='early_exit'`'s `while_loop` is not reverse-differentiable —
so any default flip must exclude the differentiable path (the implicit layer
covers that), and any `vmap`-batched caller should be checked on current
measurement before flipping (I measured perf-bench's batched `volreg` and saw
**no** early-exit benefit there — the vmap `while_loop` runs to the max per-frame
trip count — so I kept it on `fixed`; that is a measurement, not an assumption).

**Meanwhile, perf-bench side:** `rigid_register` / `affine_register` cases now
pin `mode='early_exit'` (the fair, like-for-like path vs the early-stopping
refs); `volreg` stays `fixed` per the measurement above.

**Related.**
[`registration-early-stopping-while-loop.md`](resolved/registration-early-stopping-while-loop.md)
(shipped the `early_exit` opt-in this uses — its per-recipe conclusions
superseded);
[`register-affine-small-grid-divergence.md`](resolved/register-affine-small-grid-divergence.md)
(same recipe family).
