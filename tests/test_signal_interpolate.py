# -*- coding: utf-8 -*-
"""Tests for ``nitrix.signal.{linear_interpolate, lomb_scargle_*}``.

Two failure modes the implementation must explicitly defend against:

1. **Visible boundary discontinuities** in Lomb-Scargle output --
   the classic failure mode of the independent-per-frequency LS
   form when used as an interpolant.  The joint-GLM solver here
   should make ``recon[obs] == data[obs]`` exactly (up to ridge
   / pseudoinverse threshold), so the spliced output has no jump
   at observed / censored transitions.
2. **Memory blow-up at fMRI scale** -- the per-channel Gram path
   would require ``V * K^2`` HBM (1 TB at V=1M, T=500).  The
   shared-mask fast path uses a single ``(K, K)`` Gram for all
   channels; we verify the compiled HLO has no ``(V, K, K)``
   intermediate.

Plus the standard correctness checks: low-freq sinusoid exact
reconstruction, leading / trailing edge handling for linear, and
the per-channel-mask rejection for Lomb-Scargle.
"""
from __future__ import annotations

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update('jax_enable_x64', True)

from nitrix.signal import (
    linear_interpolate,
    lomb_scargle_interpolate,
    lomb_scargle_periodogram,
)


# ---------------------------------------------------------------------------
# linear_interpolate
# ---------------------------------------------------------------------------


def test_linear_interpolate_basic():
    '''Single-step interpolation on a 5-frame ramp.'''
    data = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mask = jnp.array([True, False, True, False, True])
    out = linear_interpolate(data, mask)
    np.testing.assert_array_equal(out, data)


def test_linear_interpolate_leading_missing_edge_replicate():
    data = jnp.array([1.0, 2.0, 3.0])
    mask = jnp.array([False, True, True])
    out = linear_interpolate(data, mask)
    np.testing.assert_array_equal(out, jnp.array([2.0, 2.0, 3.0]))


def test_linear_interpolate_trailing_missing_edge_replicate():
    data = jnp.array([1.0, 2.0, 3.0])
    mask = jnp.array([True, True, False])
    out = linear_interpolate(data, mask)
    np.testing.assert_array_equal(out, jnp.array([1.0, 2.0, 2.0]))


def test_linear_interpolate_all_observed_is_identity():
    data = jnp.asarray(np.random.default_rng(0).standard_normal(50))
    mask = jnp.ones(50, dtype=bool)
    out = linear_interpolate(data, mask)
    np.testing.assert_array_equal(out, data)


def test_linear_interpolate_batched():
    data = jnp.arange(50, dtype=jnp.float32).reshape(5, 10)
    mask = jnp.tile(jnp.array([True, False, True, True, False, True, True, True, False, True]), (5, 1))
    out = linear_interpolate(data, mask)
    assert out.shape == data.shape
    # Observed values preserved.
    np.testing.assert_array_equal(out[mask], data[mask])


def test_linear_interpolate_associative_scan_used():
    '''Smoke check: the compiled HLO uses parallel scan rather
    than the sequential while_loop pattern.

    associative_scan compiles to a tree-reduction pattern (with
    a ``reduce``/``scatter`` shape); the sequential ``lax.scan``
    has a ``while_loop`` op.  We assert the absence of the latter.
    '''
    data = jnp.arange(100, dtype=jnp.float32)
    mask = jnp.asarray(np.random.default_rng(0).random(100) > 0.3)
    f = jax.jit(linear_interpolate)
    hlo = f.lower(data, mask).compile().as_text()
    assert 'while' not in hlo, 'linear_interpolate compiled to while_loop; expected associative_scan'


def test_linear_interpolate_differentiable():
    data = jnp.asarray(np.random.default_rng(0).standard_normal(20))
    mask = jnp.array([True, False, True, True, False] * 4)
    def loss(d):
        return jnp.sum(linear_interpolate(d, mask) ** 2)
    g = jax.grad(loss)(data)
    assert g.shape == data.shape
    assert bool(jnp.all(jnp.isfinite(g)))


# ---------------------------------------------------------------------------
# lomb_scargle_interpolate: the boundary-discontinuity failure mode
# ---------------------------------------------------------------------------


def _make_sinusoid_test(
    T: int = 500,
    dt: float = 2.0,
    freq: float = 0.01,
    censoring_rate: float = 0.2,
    seed: int = 0,
):
    '''Synthesise a low-frequency sinusoid with random censoring.'''
    t = jnp.arange(T, dtype=jnp.float64) * dt
    clean = jnp.sin(2 * jnp.pi * freq * t)
    rng = np.random.default_rng(seed)
    mask = jnp.asarray(rng.random(T) > censoring_rate)
    corrupt = jnp.where(mask, clean, 99.0)
    return clean, mask, corrupt


def test_lomb_scargle_recovers_observed_exactly():
    '''Joint-GLM Lomb-Scargle: ``recon[obs] == data[obs]`` up to
    pseudoinverse threshold.  This is the property that makes the
    spliced output discontinuity-free.
    '''
    clean, mask, corrupt = _make_sinusoid_test()
    recon = lomb_scargle_interpolate(corrupt, mask, dt=2.0)
    obs_err = float(jnp.abs(recon[mask] - clean[mask]).max())
    assert obs_err < 1e-10, (
        f'Lomb-Scargle should be exact at observed; max err = {obs_err}'
    )


def test_lomb_scargle_no_boundary_discontinuity():
    '''At observed / censored transitions, the reconstruction's
    per-frame difference should match the true signal's per-frame
    difference -- no jump beyond the local signal slope.

    This is the regression test for the legacy LS-interpolation
    failure mode: with the independent-per-frequency form, you'd
    see jumps of order ~ ``signal_amplitude * (1 - LS_fit_ratio)``
    at boundaries.  With the joint-GLM form, jumps should be at
    the pseudoinverse precision floor.
    '''
    clean, mask, corrupt = _make_sinusoid_test()
    recon = lomb_scargle_interpolate(corrupt, mask, dt=2.0)
    clean_diff = jnp.diff(clean)
    recon_diff = jnp.diff(recon)
    trans = jnp.diff(mask.astype(jnp.int32)) != 0
    boundary_anomaly = float(
        jnp.abs(recon_diff[trans] - clean_diff[trans]).max()
    )
    # ~1e-3 acceptable; 1e-1 would indicate a real discontinuity.
    assert boundary_anomaly < 1e-2, (
        f'Lomb-Scargle boundary jump anomaly = {boundary_anomaly}, '
        'too large; indicates a real discontinuity at observed/'
        'censored transitions (the legacy failure mode).'
    )


def test_lomb_scargle_recovers_low_freq_sinusoid():
    '''A pure sinusoid below the trial-grid Nyquist should be
    reconstructed to near-zero error at censored positions too.
    '''
    clean, mask, corrupt = _make_sinusoid_test(freq=0.01)
    recon = lomb_scargle_interpolate(corrupt, mask, dt=2.0)
    censored_err = float(jnp.abs(recon[~mask] - clean[~mask]).max())
    assert censored_err < 1e-2, (
        f'Censored-frame err = {censored_err}; expected < 1e-2 for a '
        'pure 0.01Hz sinusoid.'
    )


def test_lomb_scargle_random_recon_finite():
    '''On random data (no spectral structure), the recon should
    still be finite -- the pseudoinverse handles rank-deficient
    Gram gracefully.
    '''
    rng = np.random.default_rng(0)
    T = 500
    data = jnp.asarray(rng.standard_normal((4, T)))
    mask = jnp.asarray(rng.random(T) > 0.3)
    out = lomb_scargle_interpolate(data, mask, dt=2.0)
    assert bool(jnp.all(jnp.isfinite(out)))


def test_lomb_scargle_heavy_censoring_with_budget():
    '''At 50% censoring, the default budget (40%) would leave
    insufficient frames; pass ``censoring_budget=0.6`` to expand
    the budget.  Result should still be finite (pseudoinverse
    handles any rank deficiency).
    '''
    clean, mask, corrupt = _make_sinusoid_test(censoring_rate=0.5)
    out = lomb_scargle_interpolate(
        corrupt, mask, dt=2.0, censoring_budget=0.6,
    )
    assert bool(jnp.all(jnp.isfinite(out)))


def test_lomb_scargle_rejects_per_channel_mask():
    '''Per-channel masks would require ``V * K^2`` HBM at fMRI
    scale; we reject them at the API to force the user toward
    the shared-mask path.
    '''
    data = jnp.zeros((4, 100))
    mask = jnp.ones((4, 100), dtype=bool)
    with pytest.raises(ValueError, match='per-channel mask'):
        lomb_scargle_interpolate(data, mask)


def test_lomb_scargle_accepts_broadcast_mask():
    '''1-D mask of length ``n_obs`` should broadcast across all
    channels (the canonical fMRI motion-censoring API).
    '''
    data = jnp.asarray(np.random.default_rng(0).standard_normal((4, 100)))
    mask_1d = jnp.asarray(
        np.random.default_rng(1).random(100) > 0.2
    )
    out = lomb_scargle_interpolate(data, mask_1d)
    assert out.shape == data.shape


# ---------------------------------------------------------------------------
# Lomb-Scargle: memory verification (no V*K^2 blowup)
# ---------------------------------------------------------------------------


def test_lomb_scargle_no_VK2_intermediate_in_hlo():
    '''Memory regression: the compiled HLO must NOT contain a
    tensor with shape ``(V, K, K)`` -- the per-channel Gram path
    that would OOM at fMRI scale.

    We grep the HLO for tensors with the leading-dim of the
    data; if any (V, K, K) shape appears, the shared-Gram path
    has regressed.
    '''
    T = 200
    V = 4096
    K_approx = 1 + 2 * ((int(T * 0.6) - 1) // 2)  # default censoring_budget=0.4
    data = jnp.zeros((V, T), dtype=jnp.float32)
    mask = jnp.asarray(np.random.default_rng(0).random(T) > 0.2)
    f = jax.jit(lambda d, m: lomb_scargle_interpolate(d, m, dt=2.0))
    hlo = f.lower(data, mask).compile().as_text()
    # Look for any 3+-dim tensor with V as the leading dim.
    shapes = re.findall(r'f(?:32|64)\[([0-9,]+)\]', hlo)
    bad_shapes = []
    for s in shapes:
        dims = tuple(int(x) for x in s.split(',') if x)
        if len(dims) >= 3 and dims[0] == V:
            bad_shapes.append(dims)
    assert not bad_shapes, (
        f'shared-Gram path regressed: per-channel 3-D tensors found: '
        f'{bad_shapes[:5]}'
    )


def test_lomb_scargle_memory_footprint_bound():
    '''At V=10k, T=200, fp32, the max single tensor should be
    much smaller than ``V * K^2``.  Sanity check that the
    compiled program respects the memory analysis in the docstring.
    '''
    T = 200
    V = 10_000
    data = jnp.zeros((V, T), dtype=jnp.float32)
    mask = jnp.asarray(np.random.default_rng(0).random(T) > 0.2)
    f = jax.jit(lambda d, m: lomb_scargle_interpolate(d, m, dt=2.0))
    hlo = f.lower(data, mask).compile().as_text()
    shapes = re.findall(r'f(?:32|64)\[([0-9,]+)\]', hlo)
    max_size = 0
    for s in shapes:
        dims = tuple(int(x) for x in s.split(',') if x)
        sz = 1
        for d in dims:
            sz *= d
        max_size = max(max_size, sz)
    # Worst-case "per-channel Gram" would be V * K^2 ~ 10000 * 100^2 = 1e8.
    # Shared-Gram path should be at most V * T = 2e6 (data buffer).
    assert max_size < 5_000_000, (
        f'max tensor size {max_size} exceeds shared-Gram budget; '
        'per-channel Gram intermediate may have crept back in.'
    )


# ---------------------------------------------------------------------------
# lomb_scargle_periodogram: spectral-analysis primitive (not interpolation)
# ---------------------------------------------------------------------------


def test_lomb_scargle_periodogram_peaks_at_signal_frequency():
    '''Inject a clean sinusoid at f0 = 0.05 Hz; the periodogram
    should peak near that frequency.
    '''
    T, dt = 500, 2.0
    f0 = 0.05
    t = jnp.arange(T) * dt
    clean = jnp.sin(2 * jnp.pi * f0 * t)
    mask = jnp.ones(T, dtype=bool)
    freqs, power = lomb_scargle_periodogram(clean, mask, dt=dt)
    peak_idx = int(jnp.argmax(power))
    peak_freq = float(freqs[peak_idx])
    np.testing.assert_allclose(peak_freq, f0, atol=0.01)


def test_lomb_scargle_periodogram_matches_scipy_scargle_normalisation():
    '''Regression pinning the *corrected* normalisation docstring.

    nitrix returns the classic Scargle 1982 ``P_raw / var`` -- i.e.
    ``scipy.signal.lombscargle(..., normalize=False)`` divided by
    the observed-sample (population) variance.  It is **not**
    scipy's ``normalize=True``, which (1.17.x) returns
    ``2 * P_raw / (N * var)`` and so is off by a constant ``N / 2``.
    Both relations are asserted so the docstring's specific factor
    cannot drift silently.
    '''
    sps = pytest.importorskip('scipy.signal')
    T, dt = 256, 2.0
    rng = np.random.default_rng(0)
    t = np.arange(T) * dt
    y = (
        np.sin(2 * np.pi * 0.03 * t)
        + 0.5 * np.sin(2 * np.pi * 0.11 * t)
        + 0.1 * rng.standard_normal(T)
    )
    data = jnp.asarray(y, dtype=jnp.float64)
    mask = jnp.ones(T, dtype=bool)

    freqs, power = lomb_scargle_periodogram(data, mask, dt=dt)
    # Reuse exactly the angular grid nitrix chose so the comparison
    # is element-wise, not interpolated.
    omega = np.asarray(freqs, dtype=np.float64) * 2.0 * np.pi
    var = float(np.var(y))  # population (ddof=0), matches nitrix var_y
    # Centre manually rather than via the deprecated `precenter`
    # kwarg (scipy >= 1.17 deprecates it; this is the substitution
    # scipy itself recommends), so the test is stable across versions.
    yc = y - y.mean()

    p_raw = sps.lombscargle(t, yc, omega, normalize=False)
    np.testing.assert_allclose(
        np.asarray(power), p_raw / var, rtol=1e-6, atol=1e-9,
    )

    p_norm = sps.lombscargle(t, yc, omega, normalize=True)
    np.testing.assert_allclose(
        np.asarray(power), (T / 2.0) * p_norm, rtol=1e-6, atol=1e-9,
    )
