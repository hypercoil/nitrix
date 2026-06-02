# nitrix-perf-bench feedback

Documentation / definition drift and consumer-facing gaps surfaced while
building benchmark cases in `nitrix-perf-bench` (the perf migration of the op
matrix; see DESIGN there). Each entry cites file:line and the measurement that
surfaced it so the fix is mechanical. Perf *numbers* live in the perf-bench
`COVERAGE_DEFICIT` report; this file is for **correctness-of-documentation**
findings only.

## Open

### 2026-06-02 — Lomb-Scargle: stale normalisation claim + signal/numerics doc drift

Surfaced while building a perf-bench case for `nitrix.signal.lomb_scargle_periodogram`
(nitrix-perf-bench commits forthcoming). Verified against `scipy 1.17.1` and a
from-scratch fp64 Scargle-1982 reference on this repo's checkout.

**A. `lomb_scargle_periodogram` normalisation docstring is wrong (the high-value one).**
`src/nitrix/signal/lomb_scargle.py:154` claims *"Normalisation matches
`scipy.signal.lombscargle(normalize=True)`."* Measured, it does **not**:

- nitrix returns the **classic Scargle-normalised** periodogram `P_raw / var`
  (where `var` is the masked-sample variance) — matches my from-scratch
  Scargle-1982 `raw/var` and `scipy.signal.lombscargle(..., normalize=False)/var`
  to **~1e-9 in fp64 at every length 256–8192** (it is algorithmically exact).
- `scipy.signal.lombscargle(..., normalize=True)` on 1.17.1 returns
  `2·P_raw / (N·var)` — i.e. it differs from nitrix's output by a constant
  factor of **N/2** (measured exactly: 95.0 for N=190).

So anyone trusting the docstring and comparing nitrix to scipy `normalize=True`
sees a flat N/2 discrepancy and concludes nitrix is "wrong" when it is correct.
**Fix:** change the docstring to *"matches `scipy.signal.lombscargle(normalize=False)`
divided by the observed-sample variance (the classic Scargle 1982
normalisation)."* (scipy's `normalize` convention has drifted across versions,
so naming a specific scipy flag is fragile — describe the math.)

**B. Module docstring says "Cholesky / triangular solve"; the code uses `eigh` + truncated pseudo-inverse.**
The module "Memory regime" docstring (`lomb_scargle.py:43–49`) describes the
shared-mask interpolation path as *"compute the Gram matrix `G = Bᵀ diag(mask) B`
and its **Cholesky L** once, then solve … as a single batched **triangular
solve**"* and budgets *"Shared Gram / Cholesky"*. But the implementation
`_lomb_scargle_solve_shared_mask` (lines 246–261, and its own docstring at
229–233) factors `G` via **`safe_eigh`** (symmetric eigendecomposition) and
applies a **threshold-truncated pseudo-inverse** (`rcond·max(eigval)`) — no
Cholesky, no triangular solve. The module docstring is stale relative to the
rank-deficiency-robust eigh path the code actually ships. **Fix:** update the
module "Memory regime" prose to describe the eigh / pseudo-inverse factorisation.

**C. `lomb_scargle_interpolate` silently runs its eigh on CPU on cuSolver-broken stacks (undocumented).**
`_lomb_scargle_solve_shared_mask` calls `safe_eigh` (`linalg/_solver.py:147`),
which `device_put`s the Gram to `eigh_device()` and runs `jnp.linalg.eigh`
there. On this L4 / driver-580 / CUDA-12 stack `eigh_device()` probes to
**`cpu:0`** (dense cuSolver `eigh` fails at d≥256 — `gpusolverDnCreate failed`;
confirmed at K=499). Because `device_put` to CPU is an XLA placement hint (it is
**not** traced away under `jit`), `lomb_scargle_interpolate` invoked on GPU data
runs its eigendecomposition **on the host** with GPU→CPU→GPU round-trips — i.e.
it is *not* a GPU-resident op on affected stacks, and its K×K Gram (K up to
~`2·n_freq+1` ≈ 499 at fMRI `n_obs`=500) is exactly in the broken range. This is
correct behaviour (it is what `safe_eigh` is *for*), but it is a real
portability/perf caveat that the `lomb_scargle_interpolate` docstring does not
mention. **Fix:** add a Notes line: on stacks where dense cuSolver `eigh` is
unavailable, the Gram solve is routed to CPU (correct results, with host
transfer cost); GPU-residency of the solve is not guaranteed. (Contrast: the
matrix-function ops `symlog`/`symsqrt`/`sympower` *consume* a raw eigh that XLA
lowers off cuSolver, so they stay GPU-resident — a useful pattern to note.)

**D. `tsconv` is documented as "convolution" but implements cross-correlation.**
`src/nitrix/signal/tsconv.py:45` — *"1-D convolution along the trailing axis"* —
wraps `jax.lax.conv_general_dilated` (line 67), which does **not** flip the
kernel (cross-correlation). Verified: an impulse `[0,0,1,0,0,0,0]` ⊛ `[1,2,3]`
returns `[0,0,3,2,1,0,0]` (the kernel reversed about the centre = correlation,
not convolution). This is the standard deep-learning convention (`torch.nn.Conv1d`
is also cross-correlation), so it is fine for ML users — but in a module named
`signal` a DSP user expects a *flipped*-kernel convolution. **Fix (low priority,
clarity only):** one line in the docstring — *"cross-correlation convention
(kernel not flipped), as in conv layers; reverse the kernel for a true
convolution."*

## Resolved

_(none yet; reference the resolving nitrix commit on fix.)_
