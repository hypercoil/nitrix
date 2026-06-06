# signal + numerics: time series, interpolation, tensor utilities

> **TL;DR.**  ``nitrix.signal`` ships windowing, polynomial detrend,
> 1-D temporal convolution, **two interpolation primitives** (linear
> via associative-scan, Lomb-Scargle via joint-GLM with the
> boundary-discontinuity bug from the legacy implementation
> fixed and verified by regression test), the Lomb-Scargle
> periodogram for spectral analysis of irregular samples, and
> **two filter families** (zero-phase frequency-domain band/notch/
> low/high-pass, and a genuine recursive Butterworth IIR designed
> from scratch and validated to machine precision against
> ``scipy.signal``).  ``nitrix.numerics`` ships shape / layout
> utilities (re-exporting the externally-useful subset of
> ``_internal.util``) plus intensity-normalisation primitives.

## `signal.linear_interpolate`: associative-scan reformulation

The legacy ``hypercoil.functional.interpolate.linear_interpolate``
used three sequential ``lax.scan`` passes: ``number_consecutive``,
``delta_lookahead``, and ``_interpolate_from_deltas``.  Each scan
is ``O(n)`` parallel depth on GPU.

The new implementation uses **two parallel associative scans**:

1. ``left_idx = associative_scan(jnp.maximum, where(mask, idx, -1))``
   -- ``O(log n)`` depth.  Gives the nearest observed frame at-or-
   before each position.
2. ``right_idx = associative_scan(jnp.minimum, where(mask, idx, n),
   reverse=True)`` -- ditto, at-or-after.

Then a per-frame linear interpolation between left and right.  On
Ampere at typical fMRI sizes (n_obs ~ 500), this is ~3-5× faster
than the sequential scan.

Verified by ``test_linear_interpolate_associative_scan_used``
which greps the compiled HLO for ``while`` and asserts its
absence -- the sequential ``lax.scan`` lowers to ``while_loop``;
the parallel ``associative_scan`` lowers to a tree-reduce pattern
without while.

Edge handling is **edge-replicate** (leading / trailing missing
frames take the value of the nearest observed frame), matching
the rest of nitrix's boundary conventions.

## `signal.lomb_scargle_*`: the joint-GLM rewrite

This was the marquee debug of the sprint.  The legacy
"Lomb-Scargle interpolation" attempt produced **visible boundary
discontinuities** at observed / censored transitions -- the
classic failure mode of independent-per-frequency Lomb-Scargle
when used as an interpolant.

### Why per-frequency LS fails as an interpolant

Scargle's 1982 formulation orthogonalises the sin / cos pair at
each trial frequency via a frequency-dependent time offset
``tau_omega``, then estimates ``(A_omega, B_omega)`` independently
per frequency.  This gives the right *periodogram* (unbiased
spectral power estimate); but the implied reconstruction
``sum_omega A cos(omega (t - tau)) + B sin(omega (t - tau))``
does **not** pass through the observed samples exactly.  At an
observed-to-censored boundary, ``recon[obs] ≠ data[obs]``,
``recon[censored]`` is smooth, and the spliced output
``where(mask, data, recon)`` has a jump of size
``data[obs] - recon[obs]``.

In fMRI motion-censoring, these jumps inject high-frequency
noise into the time series at every censored-frame boundary --
exactly the artefact the interpolation is supposed to avoid.

### The fix: joint GLM regression

Instead of independent per-frequency LS, solve a **single joint
masked least-squares**:

    min || M (B β - y) ||²

where ``B = [DC | cos(omega_1 t) | sin(omega_1 t) | ...]`` is the
``(n_obs, K)`` basis, ``M = diag(mask)``, and ``y`` is the data.
The joint fit guarantees ``B[obs] @ β = y[obs]`` (modulo rank
deficiency, handled by pseudoinverse with eigenvalue threshold).
The spliced output is exact at observed points, smooth between
them, and has **no boundary jump**.

This is what Power et al. 2014 actually describe (under the
Lomb-Scargle name).  Implementation matters; the legacy code
followed Scargle 1982 form when it should have followed the
joint-regression form.

### The memory concern: V × K² blowup

A naive ``vmap`` over channels for the GLM solve would compute a
per-channel Gram matrix ``B^T diag(mask) B`` of shape ``(V, K, K)``
-- 1 TB at V = 1M voxels, T = 500, K = 499.  OOM at any realistic
fMRI scale.

**The fix**: when the mask is shared across channels (the
canonical fMRI case: one motion-censoring mask per scan, applied
to all voxels), the Gram matrix and its eigendecomposition are
**also shared**.  Compute ``G`` and the pseudoinverse via ``eigh``
once; solve all the right-hand sides as a single batched matmul.

Memory regime at V = 1M, T = 500, K = 499:
- Shared basis: 1 MB.
- Shared Gram + eigvecs: 2 MB.
- Per-channel rhs / β / recon: 6 GB.
- Data / output: 4 GB.
- Total: ~10 GB.  Fits on a 24 GB GPU with headroom.

The HLO is audited by ``test_lomb_scargle_no_VK2_intermediate_in_hlo``
which greps the compiled program for any 3-D tensor with the
leading dim equal to ``V`` -- regression-safety against future
refactors that might inadvertently introduce a per-channel
Gram.

Per-channel masks (where each channel has its own censoring
pattern) are explicitly **rejected** at the API; the user is
directed toward either flattening to a shared mask (the common
case) or vmapping manually (if they actually have the HBM).

### Pseudoinverse instead of Cholesky

The natural Cholesky solve fails at any nontrivial censoring
rate because the masked basis is rank-deficient: ``rank(B_w) ≤
n_valid``, but ``B`` has ``K = 2 n_freq + 1`` columns.  At
default ``oversampling = 4`` and ``high_factor = 1``, ``K`` can
exceed ``n_valid`` whenever the censoring rate is above the
``censoring_budget`` parameter (default 0.4 = up to 40%
censoring tolerated).

The pseudoinverse via ``safe_eigh`` + eigenvalue truncation
gracefully handles rank deficiency without arbitrary ridge
parameters.  The ``rcond`` parameter (default ``1e-6``) sets
the relative truncation threshold; eigenvalues below
``rcond * max(eigval)`` are zeroed.

### Tests

The regression covers:

- ``test_lomb_scargle_recovers_observed_exactly`` -- forces
  ``recon[obs] - data[obs] < 1e-10``.  The load-bearing property.
- ``test_lomb_scargle_no_boundary_discontinuity`` -- per-frame
  diffs at observed/censored transitions match the clean signal's
  diffs to ``1e-2``.  The regression for the legacy failure mode.
- ``test_lomb_scargle_recovers_low_freq_sinusoid`` -- pure
  sinusoid below trial-grid Nyquist reconstructed to ``1e-2``.
- ``test_lomb_scargle_heavy_censoring_with_budget`` -- 50%
  censoring with ``censoring_budget=0.6`` is finite (pseudoinverse
  handles the rank deficiency).
- ``test_lomb_scargle_random_recon_finite`` -- random data with
  no spectral structure produces finite output.
- ``test_lomb_scargle_rejects_per_channel_mask`` -- API safety
  for memory.
- ``test_lomb_scargle_no_VK2_intermediate_in_hlo`` -- memory
  regression via HLO grep.
- ``test_lomb_scargle_memory_footprint_bound`` -- max tensor at
  V=10k below the shared-Gram budget.
- ``test_lomb_scargle_periodogram_peaks_at_signal_frequency`` --
  spectral-analysis sanity check.

## `signal.filter.polynomial_detrend`

Lean implementation: build a polynomial basis
``[1, t, t^2, ..., t^d]`` over rescaled time ``t ∈ [-1, 1]``
(rescaling for stability of the Vandermonde-like matrix at higher
degrees), then call ``linalg.residualise`` with it.  The rescale
matters for ``degree > 5``; at the typical fMRI degrees 1-3 it's
defensive but not load-bearing.

## `signal.filter`: frequency-domain and recursive-IIR filters

Two filtering families requested by `bitsjax` (`bits.signal.bandpass`,
`bits.signal.iir_filter`); both are SPEC §4.3 primitives.  Neither ports
the rejected hypercoil `nn/freqfilter.py` / `nn/iirfilter.py`.

**Frequency-domain (`bandpass` / `bandstop` / `lowpass` / `highpass`).**
A real magnitude weight on the rfft grid, applied via
`stats.fourier.product_filter` (one rfft + multiply + irfft).  Because a
real mask is a circular convolution with its inverse DFT, this is a
**zero-phase, frequency-sampled FIR** -- explicitly *not* a recursive
filter.  Designs: `'maxflat'` (the Butterworth *magnitude* shape
`1/sqrt(1+(f/fc)^{2n})`, named so it is not mistaken for a real
Butterworth), `'ideal'` (brick wall), `'cosine'` (raised-cosine edges).
`bandstop` is the notch (soft-union of low-/high-pass), whose canonical use
is removing the respiratory peak from head-motion estimates.  Per-channel,
differentiable, JIT-friendly.

**Recursive Butterworth (`iir_filter` / `butterworth_sos` / `sosfilt` /
`sosfiltfilt`).**  A genuine IIR: analog poles → frequency transform →
bilinear → second-order sections, designed **from scratch in NumPy** (no
runtime scipy, SPEC §5.2) and validated to machine precision against
`scipy.signal.butter`/`sosfilt`/`sosfiltfilt`.  The design is a static host
constant; only the *application* is traced.  Three application engines,
selected by `backend='auto'` per platform:

- `backend='fft'` (**GPU default**) -- the FFT-convolution engine.  An IIR
  filter is LTI, so its zero-state output is *exactly* convolution with the
  impulse response (a host constant that decays geometrically for a stable
  filter); `sosfiltfilt`'s scipy-exact `zi` edges are recovered by adding the
  cascade's zero-input response `x[0] * g` over the first `n_taps` samples.
  This is `O(T log T)`, fully parallel, and **beats cupy on the L4** (sosfilt
  0.95 ms vs 1.78 ms, sosfiltfilt 1.77 ms vs 9.4 ms at ch=1024/obs=4096).  The
  impulse response is truncated at `impulse_atol` (default `1e-12`, ~exact, the
  truncation error is geometrically bounded -- see the theory note below);
  a filter too sharp to decay within `2**15` taps falls back to a recurrence.
- `backend='scan'` (**CPU default**) -- sequential `lax.scan`, one fused loop,
  low memory; exact and fastest on the CPU, where the recursion is not
  latency-bound.
- `backend='associative'` -- the `lax.associative_scan` parallel-prefix
  linear-recurrence (transposed-DF2 state, `O(log T)` depth); the GPU
  recurrence used as the FFT engine's too-sharp-filter fallback and available
  explicitly.

`sosfiltfilt` is zero-phase forward-backward with scipy-exact steady-state
initial conditions (`lfilter_zi`-equivalent) and odd padding.  All engines
match `scipy.signal` to round-off and are reverse-mode differentiable through
the signal.

### Recursive-filter theory, briefly

An IIR (infinite-impulse-response) filter is *recursive*: the output
depends on past **outputs**, ``y[n] = Σ b_k x[n-k] - Σ a_k y[n-k]``, so its
transfer function ``H(z) = B(z)/A(z)`` is rational.  The roots of ``A`` are
the **poles** (the feedback / resonance; they must lie inside the unit
circle for stability), the roots of ``B`` the **zeros**.  The impulse
response never fully dies (hence "infinite") -- the opposite of the FIR
frequency-domain filters above, which have finite support and only touch
past *inputs*.  The payoff is steep roll-off from very few coefficients;
the cost is a non-linear, frequency-dependent phase.

**Why an IIR filter still has a parallel FFT path.** Recursive *application*
is one algorithm, not the definition: a fixed-coefficient IIR is LTI, so its
output is *exactly* ``y = h * x`` (convolution with the impulse response ``h``)
-- the recursion is just the ``O(N)`` way to evaluate that convolution when
``h`` is infinite.  Because the poles lie inside the unit circle, ``|h[n]| ≤ C
rⁿ`` with ``r = max|pole| < 1``: ``h`` decays geometrically, so truncating it
to ``L`` taps incurs an error bounded by ``‖x‖_∞ · C rᴸ/(1-r)`` -- push ``L``
until that tail is below the rounding floor and the FFT convolution is as exact
as the recursion (the ``backend='fft'`` engine).  The only knob is ``L``, which
grows like ``1/(1-r)`` for razor-sharp filters (`impulse_atol` trades it off;
the engine falls back to the recurrence when it would exceed the tap cap).  The
*exact* parallel alternative -- block scan with boundary reconciliation -- has
no truncation but is fp32-ill-conditioned (the ``M^C`` block transition) for
sharp filters, which is why the (truncation-controlled) FFT path is preferred.

**Butterworth** is the all-pole prototype with a *maximally-flat* passband:
``|H(jω)|² = 1 / (1 + (ω/ω_c)^{2N})``, whose ``N`` analog poles sit equally
spaced on a circle in the left half ``s``-plane (monotonic magnitude, no
ripple, ``6N`` dB/octave roll-off).  Design proceeds analog → digital:

1. **Prototype** (`_buttap`): the unit-circle poles of the order-``N``
   analog low-pass.
2. **Frequency transform** (`_lp2lp/_lp2hp/_lp2bp/_lp2bs`): move/reshape the
   prototype to the requested band.  Band-pass / band-stop map each pole to
   two and add zeros, doubling the order to ``2N``.
3. **Bilinear transform** (`_bilinear`): the conformal map
   ``s = 2 f_s (z-1)/(z+1)`` from the analog plane to the unit disk, which
   preserves stability.  It warps frequency by a ``tan``, so the cut-offs
   are **pre-warped** (``warped = 2 f_s tan(π W_n / f_s)``) to land exactly
   where requested after warping.
4. **Second-order sections** (`_zpk2sos`): factor the order-``2N`` rational
   into a cascade of biquads.  This is not cosmetic -- a single high-order
   ``B(z)/A(z)`` has coefficients spanning many orders of magnitude and
   tightly clustered poles, so direct-form evaluation loses catastrophic
   precision; cascaded biquads keep every factor well-scaled (the reason
   scipy defaults to SOS).  The *pairing* of poles and zeros into sections
   affects only conditioning, not the transfer function; we use a simple
   conjugate pairing (adequate at these orders) and verify via the
   pairing-invariant frequency response.

Application is the transposed-Direct-Form-II recurrence (two delay
registers per biquad, cascaded).  **Zero phase** matters because a causal
IIR imposes a frequency-dependent group delay that smears event timing;
`filtfilt` runs the filter forward then on the time-reversed signal, whose
phase responses are exact negatives and cancel -- leaving zero phase and a
*squared* magnitude (so attenuation doubles and the effective ``-3`` dB
point shifts, a documented consequence).  **Initial conditions** matter
because a filter started from a silent state "rings up" with a startup
transient; `_sos_zi` sets the delay registers to the steady state for the
signal's edge value so there is no transient (scipy `lfilter_zi`).

### Matching scipy exactly: the process

`scipy.signal` is the reference but a **test-only** dependency (SPEC §5.2),
so the runtime is pure NumPy + JAX and scipy is the oracle in `test_iir.py`.
We aligned by **decomposing the pipeline** and validating each stage on its
own -- debugging a blind end-to-end `filtfilt` mismatch is otherwise
hopeless:

1. **Design** -- compare the **frequency response** ``|H(e^{jω})|`` of our
   SOS to scipy's, *not* the coefficients: valid pairings differ but the
   transfer function is identical.  Matched to ``~1e-15`` first, before any
   filtering code existed.
2. **Forward apply** -- our `sosfilt` (zero state) vs `scipy.sosfilt`: same
   recurrence, **exact** (``0.0``).
3. **Zero-phase** -- `sosfiltfilt` vs `scipy.sosfiltfilt`, exact only after
   three hard-won subtleties surfaced (each with a localising symptom):

   - **`zi` formula.** The first attempt used the DC steady state
     (``y = dc_gain · x``); fine for low-pass but wrong for band-pass /
     band-stop, whose DC gain is ``≈ 0`` -> ``zi ≈ 0`` -> transients leak in
     (symptom: band-pass interior off by ``~1e-2`` while low-pass was fine).
     Fix: scipy's companion-matrix solve ``(I - Aᵀ) zi = B``, valid at any
     DC gain.
   - **Odd padding.** The right-edge odd extension was double-reversed
     (scipy's slice already reverses via ``step=-1``).  Symptom: ``~0.5``
     error concentrated at the edges *while forward+zi was exact* -- which
     is what localised the bug to the pad/flip, not the recurrence.
   - **`padlen`.** scipy shrinks ``ntaps = 2·n_sections + 1`` by the number
     of first-order sections (``min`` count of zero ``b2`` / ``a2``), so
     odd-order filters pad less.  Symptom: *only* odd-order `filtfilt`
     mismatched; even orders were already exact.

   With all three fixed, `sosfiltfilt` matches `scipy.sosfiltfilt` to
   machine precision across every band type and order tested.

## `signal.tsconv`

Thin batched wrapper around ``lax.conv_general_dilated`` for the
``(..., C, obs)`` layout.  No new functionality vs the legacy
``tsconv`` -- dropped the ``basisconv2d`` / ``polyconv2d`` /
``basischan`` / ``polychan`` variants (basis-function HRF
modelling utilities, scientific-question-specific; live elsewhere
when they live again).

## `numerics.tensor_ops`: the externally-useful subset of `_internal.util`

``_internal.util`` accumulated 30+ utility functions during
the legacy port (axis normalisation, fold / unfold, broadcasting,
masking, complex decomposition, vmap-over-outer).  This module
promotes the externally-useful subset (``orient_and_conform``,
``fold_axis``, ``unfold_axes``, ``complex_decompose``,
``conform_mask``, etc.) to a public namespace under a cleaner
name (``numerics.tensor_ops``).

No re-implementation -- just re-exports with a curated
``__all__``.  The internal-only utilities (``_dim_or_none``,
``_compose``, ``_seq_pad``, ``_conform_bform_weight``, etc.)
stay in ``_internal.util`` as implementation detail.

## `numerics.normalize`

Four canonical normalisations:

- ``zscore_normalize`` -- ``(x - mean) / std``, optionally
  per-observation weighted.
- ``psc_normalize`` -- ``100 * (x - mean) / mean``, the fMRI BOLD
  percent-signal-change convention.
- ``robust_zscore_normalize`` -- median + MAD instead of mean +
  std; resistant to outliers.  Tested via the outlier-injection
  regression: a 100× outlier collapses the regular z-score's
  interior std to ~``0.01`` but leaves the robust z-score's
  interior at ``~1``.
- ``intensity_normalize`` -- percentile-clip then rescale to
  ``[0, 1]``; the synthstrip / SynthSeg pre-training
  convention.

All four support per-axis ``axis`` argument and (where applicable)
weighted variants.

## Cross-references

- ``src/nitrix/signal/{window, filter, _iir, tsconv, interpolate,
  lomb_scargle}.py`` (``_iir`` re-exported through ``filter``).
- ``src/nitrix/numerics/{tensor_ops, normalize}.py``.
- ``tests/test_signal.py``, ``tests/test_signal_filter.py`` (frequency-
  domain), ``tests/test_iir.py`` (recursive Butterworth, scipy parity),
  ``tests/test_signal_interpolate.py``, ``tests/test_numerics.py``.
- [`linalg.md`](linalg.md) -- the ``residualise`` consumer used by
  ``polynomial_detrend``.
- ``stats.fourier.product_filter`` -- the rfft-multiply engine the
  frequency-domain filters apply their magnitude weight through.
- ``bitsjax`` feature requests N1 (band-pass) and N2 (IIR), now satisfied.
- Power, J. D. et al. (2014). NeuroImage 84, 320-341 -- the
  Lomb-Scargle interpolation protocol.
- Scargle, J. D. (1982). Astrophys. J. 263, 835-853 -- the
  periodogram formulation.
- Press, W. H. & Rybicki, G. B. (1989). Astrophys. J. 338, 277-280 --
  the fast Lomb-Scargle algorithm we don't use (independent-per-
  frequency is wrong for interpolation; we use joint GLM instead).
