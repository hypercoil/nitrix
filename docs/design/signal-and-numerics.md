# signal + numerics: time series, interpolation, tensor utilities

> **TL;DR.**  ``nitrix.signal`` ships windowing, polynomial detrend,
> 1-D temporal convolution, **two interpolation primitives** (linear
> via associative-scan, Lomb-Scargle via joint-GLM with the
> boundary-discontinuity bug from the legacy implementation
> fixed and verified by regression test), and the Lomb-Scargle
> periodogram for spectral analysis of irregular samples.
> ``nitrix.numerics`` ships shape / layout utilities (re-exporting
> the externally-useful subset of ``_internal.util``) plus
> intensity-normalisation primitives.

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

- ``src/nitrix/signal/{window, filter, tsconv, interpolate,
  lomb_scargle}.py``.
- ``src/nitrix/numerics/{tensor_ops, normalize}.py``.
- ``tests/test_signal.py``, ``tests/test_signal_interpolate.py``,
  ``tests/test_numerics.py``.
- [`linalg.md`](linalg.md) -- the ``residualise`` consumer used by
  ``polynomial_detrend``.
- Power, J. D. et al. (2014). NeuroImage 84, 320-341 -- the
  Lomb-Scargle interpolation protocol.
- Scargle, J. D. (1982). Astrophys. J. 263, 835-853 -- the
  periodogram formulation.
- Press, W. H. & Rybicki, G. B. (1989). Astrophys. J. 338, 277-280 --
  the fast Lomb-Scargle algorithm we don't use (independent-per-
  frequency is wrong for interpolation; we use joint GLM instead).
