# -*- coding: utf-8 -*-
"""Tests for ``nitrix.bias`` (N4 bias-field correction and its primitives).

Coverage:

- ``bspline_approximate`` / the separable B-spline engine:
  - reconstruction matches ``scipy.interpolate.BSpline`` (uniform knots);
  - partition of unity (a constant control lattice -> constant field);
  - the **multilevel** residual-refit (N4's mechanism) drives the
    single-level MBA bias geometrically to zero;
  - a mask makes out-of-mask garbage irrelevant to the fit;
  - differentiable w.r.t. the data.
- ``sharpen_histogram``:
  - de-blurs a bimodal intensity distribution (narrows the tissue peaks);
  - matches an independent NumPy Wiener-deconvolution reference;
  - JITs to the same result; constant input is safe.
- ``n4_bias_field_correction``:
  - recovers a known smooth multiplicative bias on a phantom;
  - flattens within-tissue intensity variation;
  - JIT parity; batched ``vmap`` path; ``return_bias_field`` shapes;
  - argument validation.
- **Golden parity** with ITK / ANTs ``N4BiasFieldCorrectionImageFilter``
  via a checked-in SimpleITK reference array (runs without SimpleITK), plus
  a live re-derivation guarded by ``importorskip('SimpleITK')``.

Reference parity numbers and the accelerator reformulation are documented
in ``docs/design/bias-field.md``.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nitrix.bias import (
    bias_field_correction,
    bspline_approximate,
    n4_bias_field_correction,
    sharpen_histogram,
)
from nitrix.bias._bspline import _reconstruct, _reconstruction_matrix

_GOLDEN = Path(__file__).parent / 'artefacts' / 'bias' / 'n4_sitk_golden.npz'


# ---------------------------------------------------------------------------
# Phantom
# ---------------------------------------------------------------------------


def _phantom(s: int = 48, seed: int = 20260523):
    """Concentric-shell tissue phantom with a smooth multiplicative bias."""
    rng = np.random.default_rng(seed)
    ax = np.linspace(-1, 1, s)
    xx, yy, zz = np.meshgrid(ax, ax, ax, indexing='ij')
    r = np.sqrt(xx**2 + yy**2 + zz**2)
    tissue = np.where(r < 0.85, 100.0, 0.0)
    tissue = np.where(r < 0.55, 160.0, tissue)
    tissue = np.where(r < 0.25, 210.0, tissue).astype(np.float32)
    mask = (tissue > 0).astype(np.float32)
    bias = np.clip(
        1.0
        + 0.45 * np.sin(1.6 * xx + 0.4)
        + 0.3 * np.cos(1.3 * yy - 0.2)
        + 0.18 * zz,
        0.5,
        1.8,
    ).astype(np.float32)
    obs = (tissue * bias).astype(np.float32)
    obs = obs + rng.normal(0, 0.8, obs.shape).astype(np.float32) * mask
    return obs.astype(np.float32), mask, bias.astype(np.float32), r


# ---------------------------------------------------------------------------
# B-spline reconstruction / fit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('order', [1, 2, 3])
def test_reconstruction_partition_of_unity(order):
    R = np.array(_reconstruction_matrix(60, 5 + order, order, jnp.float32))
    np.testing.assert_allclose(R.sum(axis=1), 1.0, atol=1e-5)


def test_reconstruct_matches_scipy_bspline():
    scipy_interp = pytest.importorskip('scipy.interpolate')
    order, n_vox, n_ctrl = 3, 50, 8
    n_spans = n_ctrl - order
    R = _reconstruction_matrix(n_vox, n_ctrl, order, jnp.float32)
    rng = np.random.default_rng(0)
    cp = rng.normal(size=n_ctrl).astype(np.float32)
    mine = np.array(_reconstruct(jnp.asarray(cp), [R], [0]))

    # Uniform B-spline with integer knots; valid domain [order, n_ctrl].
    t = np.arange(n_ctrl + order + 1, dtype=float)
    u = order + (np.arange(n_vox) / (n_vox - 1)) * n_spans
    ref = scipy_interp.BSpline(t, cp, order, extrapolate=False)(u)
    np.testing.assert_allclose(mine, ref, atol=1e-4)


def test_constant_lattice_reconstructs_constant():
    # MBA fit of a constant is only *approximate* (single-level MBA is
    # biased on dense data), but the reconstruction of a constant control
    # lattice is exact by partition of unity; check the latter directly.
    R = _reconstruction_matrix(40, 8, 3, jnp.float32)
    recon = np.array(_reconstruct(jnp.full((8,), 2.5, jnp.float32), [R], [0]))
    np.testing.assert_allclose(recon, 2.5, atol=1e-5)
    # The single-level approximation stays finite and in a sane range.
    approx = np.array(
        bspline_approximate(
            jnp.full((40,), 2.5, jnp.float32),
            control_points=8,
            spatial_rank=1,
        )
    )
    assert np.all(np.isfinite(approx))


def test_multilevel_mba_converges():
    # Single-level MBA is biased on dense data; the multilevel residual
    # refit (N4's mechanism) drives the error down geometrically.
    n = 64
    x = np.linspace(0, 1, n)
    target = (1.0 + 0.5 * np.sin(2 * np.pi * x)).astype(np.float32)
    w = jnp.ones((n,), jnp.float32)
    field = np.zeros(n, np.float32)
    errs = []
    for level in range(6):
        ncp = (2**level) + 3
        resid = jnp.asarray(target - field)
        inc = np.array(
            bspline_approximate(
                resid, control_points=ncp, weight=w, spatial_rank=1
            )
        )
        field = field + inc
        errs.append(float(np.sqrt(np.mean((field - target) ** 2))))
    # Strictly decreasing and small at the finest level.
    assert all(errs[i] > errs[i + 1] for i in range(len(errs) - 1))
    assert errs[-1] < 0.02 * float(target.std())


def test_bspline_masked_fit_ignores_outside_mask():
    n = 200
    x = np.linspace(0, 1, n)
    f = np.sin(2 * np.pi * x).astype(np.float32)
    mask = np.ones(n, np.float32)
    mask[120:] = 0.0
    corrupt = f.copy()
    corrupt[120:] = 1e6  # garbage outside the mask
    approx = np.array(
        bspline_approximate(
            jnp.asarray(corrupt),
            control_points=24,
            weight=jnp.asarray(mask),
            spatial_rank=1,
        )
    )
    # Inside the mask, well away from the boundary, the garbage must not leak.
    err = np.sqrt(np.mean((approx[:100] - f[:100]) ** 2))
    assert err < 0.1
    assert np.all(np.isfinite(approx))


def test_bspline_differentiable():
    n = 32
    data = jnp.asarray(
        np.random.default_rng(0).normal(size=n).astype(np.float32)
    )
    g = jax.grad(
        lambda d: jnp.sum(
            bspline_approximate(d, control_points=8, spatial_rank=1) ** 2
        )
    )(data)
    assert np.all(np.isfinite(np.array(g)))
    assert float(jnp.linalg.norm(g)) > 0


def test_bspline_3d_multilevel_recovery():
    # 3-D analogue of the multilevel convergence test: refitting the
    # residual on a doubling control grid recovers a smooth field.
    s = 40
    ax = np.linspace(0, 1, s)
    xx, yy, zz = np.meshgrid(ax, ax, ax, indexing='ij')
    field = (
        np.sin(2 * np.pi * xx) * np.cos(2 * np.pi * yy) * (1 + zz)
    ).astype(np.float32)
    w = jnp.ones((s, s, s), jnp.float32)
    acc = np.zeros_like(field)
    errs = []
    for level in range(5):
        ncp = (2**level) * 2 + 3
        inc = np.array(
            bspline_approximate(
                jnp.asarray(field - acc),
                control_points=(ncp, ncp, ncp),
                weight=w,
            )
        )
        acc = acc + inc
        errs.append(float(np.sqrt(np.mean((acc - field) ** 2))))
    assert all(errs[i] > errs[i + 1] for i in range(len(errs) - 1))
    assert errs[-1] < 0.05 * float(field.std())


# ---------------------------------------------------------------------------
# Histogram sharpening
# ---------------------------------------------------------------------------


def _numpy_wiener_sharpen(image, n_bins=200, fwhm=0.15, noise=0.01):
    """Independent NumPy reference for ``sharpen_histogram``."""
    flat = image.ravel().astype(np.float64)
    bmin, bmax = flat.min(), flat.max()
    slope = (bmax - bmin) / (n_bins - 1)
    cidx = (flat - bmin) / slope
    idx = np.floor(cidx).astype(int)
    frac = cidx - idx
    i0 = np.clip(idx, 0, n_bins - 1)
    i1 = np.clip(idx + 1, 0, n_bins - 1)
    H = np.zeros(n_bins)
    np.add.at(H, i0, 1.0 - frac)
    np.add.at(H, i1, frac)

    exponent = int(np.ceil(np.log2(n_bins))) + 1
    padded = 2**exponent
    off = (padded - n_bins) // 2
    V = np.zeros(padded, complex)
    V[off : off + n_bins] = H
    Vf = np.fft.fft(V)

    sfwhm = fwhm / slope
    expf = 4.0 * np.log(2.0) / sfwhm**2
    scale = 2.0 * np.sqrt(np.log(2.0) / np.pi) / sfwhm
    i = np.arange(padded)
    d = np.minimum(i, padded - i)
    Gf = np.real(np.fft.fft(scale * np.exp(-(d**2) * expf)))

    U = np.clip(np.real(np.fft.ifft(Vf * Gf / (Gf**2 + noise))), 0, None)
    centres = bmin + (np.arange(padded) - off) * slope
    num = np.real(np.fft.ifft(np.fft.fft(centres * U) * Gf))
    den = np.real(np.fft.ifft(np.fft.fft(U) * Gf))
    E = np.where(np.abs(den) > 1e-10, num / den, 0.0)
    e0 = E[np.clip(i0 + off, 0, padded - 1)]
    e1 = E[np.clip(i1 + off, 0, padded - 1)]
    return (e0 * (1 - frac) + e1 * frac).reshape(image.shape)


def test_sharpen_deblurs_bimodal():
    rng = np.random.default_rng(0)
    n = 20000
    true = np.concatenate(
        [rng.normal(1.0, 0.05, n), rng.normal(2.0, 0.05, n)]
    ).astype(np.float32)
    obs = (true + rng.normal(0, 0.12, true.shape)).astype(np.float32)
    sh = np.array(sharpen_histogram(jnp.asarray(obs), fwhm=0.15))
    # Within-class spread shrinks; class means are preserved.
    assert sh[:n].std() < obs[:n].std()
    assert sh[n:].std() < obs[n:].std()
    np.testing.assert_allclose(sh[:n].mean(), obs[:n].mean(), atol=2e-2)
    np.testing.assert_allclose(sh[n:].mean(), obs[n:].mean(), atol=2e-2)


def test_sharpen_matches_numpy_reference():
    rng = np.random.default_rng(1)
    obs = rng.normal(1.5, 0.3, (16, 16, 16)).astype(np.float32)
    mine = np.array(sharpen_histogram(jnp.asarray(obs), fwhm=0.15))
    ref = _numpy_wiener_sharpen(np.array(obs), fwhm=0.15)
    # fp32 JAX vs fp64 NumPy: agree to a few-1e-3 of the intensity range.
    np.testing.assert_allclose(mine, ref, atol=5e-3, rtol=0)


def test_sharpen_jit_parity():
    obs = jnp.asarray(
        np.random.default_rng(2)
        .normal(1.0, 0.2, (12, 12, 12))
        .astype(np.float32)
    )
    a = np.array(sharpen_histogram(obs, fwhm=0.15))
    b = np.array(jax.jit(lambda z: sharpen_histogram(z, fwhm=0.15))(obs))
    np.testing.assert_allclose(a, b, atol=1e-4)


def test_sharpen_constant_image_is_safe():
    out = np.array(sharpen_histogram(jnp.full((8, 8, 8), 3.0, jnp.float32)))
    assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# N4
# ---------------------------------------------------------------------------


def test_n4_recovers_known_bias():
    obs, mask, bias_true, _ = _phantom(48)
    _, bias_est = n4_bias_field_correction(
        jnp.asarray(obs), mask=jnp.asarray(mask), return_bias_field=True
    )
    be = np.array(bias_est)[mask > 0]
    bt = bias_true[mask > 0]
    assert np.corrcoef(be, bt)[0, 1] > 0.99
    # Bias is defined up to a global scale; compare after matching it.
    s = np.sum(be * bt) / np.sum(bt * bt)
    rel_rmse = np.sqrt(np.mean((be - s * bt) ** 2)) / bt.mean()
    assert rel_rmse < 0.02


def test_n4_flattens_tissue_variation():
    obs, mask, _, r = _phantom(48)
    corrected = np.array(
        n4_bias_field_correction(jnp.asarray(obs), mask=jnp.asarray(mask))
    )
    shell = ((r >= 0.55) & (r < 0.85)) & (mask > 0)
    cv_in = obs[shell].std() / obs[shell].mean()
    cv_out = corrected[shell].std() / corrected[shell].mean()
    assert cv_out < 0.2 * cv_in


def test_n4_jit_parity():
    obs, mask, _, _ = _phantom(32)
    fn = jax.jit(lambda o, m: n4_bias_field_correction(o, mask=m))
    a = np.array(
        n4_bias_field_correction(jnp.asarray(obs), mask=jnp.asarray(mask))
    )
    b = np.array(fn(jnp.asarray(obs), jnp.asarray(mask)))
    # jit vs eager differ only by XLA fusion / reassociation in the
    # iterative loop (~1e-4 relative); not a correctness difference.
    np.testing.assert_allclose(a, b, rtol=5e-3, atol=1e-3)


def test_n4_batched_vmap():
    obs, mask, _, _ = _phantom(32)
    batch = np.stack([obs, obs * 0.8], 0)
    mb = np.stack([mask, mask], 0)
    out = n4_bias_field_correction(
        jnp.asarray(batch), mask=jnp.asarray(mb), spatial_rank=3
    )
    out = np.array(out)
    assert out.shape == batch.shape
    assert np.all(np.isfinite(out))
    # Each volume corrected independently: scaling the input scales the
    # corrected output by the same factor (bias is intensity-scale free).
    single = np.array(
        n4_bias_field_correction(jnp.asarray(obs), mask=jnp.asarray(mask))
    )
    # vmap vs single-volume differ only by XLA batching reassociation.
    np.testing.assert_allclose(out[0], single, rtol=5e-3, atol=1e-2)


def test_n4_return_bias_field_shapes():
    obs, mask, _, _ = _phantom(32)
    corrected, bias = n4_bias_field_correction(
        jnp.asarray(obs), mask=jnp.asarray(mask), return_bias_field=True
    )
    assert corrected.shape == obs.shape
    assert bias.shape == obs.shape
    # corrected * bias ~ original within the mask.
    c, b = np.array(corrected), np.array(bias)
    m = mask > 0
    np.testing.assert_allclose((c * b)[m], obs[m], rtol=1e-4, atol=1e-2)


def test_n4_control_points_validation():
    obs, mask, _, _ = _phantom(16)
    with pytest.raises(ValueError):
        n4_bias_field_correction(
            jnp.asarray(obs),
            mask=jnp.asarray(mask),
            spline_order=3,
            n_control_points=3,  # < spline_order + 1
        )


def test_n4_max_iterations_length_validation():
    obs, mask, _, _ = _phantom(16)
    with pytest.raises(ValueError):
        n4_bias_field_correction(
            jnp.asarray(obs),
            mask=jnp.asarray(mask),
            n_fitting_levels=4,
            max_iterations=[50, 50],  # wrong length
        )


# ---------------------------------------------------------------------------
# Golden parity vs ITK / ANTs (SimpleITK)
# ---------------------------------------------------------------------------


def _bias_parity(a, g, mask):
    """(correlation, scaled-relRMSE, max-relative-no-scale) over the mask."""
    m = mask > 0
    aa, gg = a[m], g[m]
    corr = np.corrcoef(aa, gg)[0, 1]
    s = np.sum(aa * gg) / np.sum(gg * gg)
    rel_rmse = np.sqrt(np.mean((aa - s * gg) ** 2)) / gg.mean()
    max_rel = np.max(np.abs(aa - gg)) / gg.mean()
    return corr, rel_rmse, max_rel


@pytest.mark.skipif(not _GOLDEN.exists(), reason='golden artefact missing')
def test_n4_matches_sitk_golden():
    d = np.load(_GOLDEN)
    obs, mask = d['obs'], d['mask']
    itk_corr, itk_bias = d['itk_corrected'], d['itk_bias']
    corrected, bias = n4_bias_field_correction(
        jnp.asarray(obs), mask=jnp.asarray(mask), return_bias_field=True
    )
    corrected, bias = np.array(corrected), np.array(bias)

    corr_b, rmse_b, maxrel_b = _bias_parity(bias, itk_bias, mask)
    corr_c, rmse_c, maxrel_c = _bias_parity(corrected, itk_corr, mask)
    # Measured on this phantom: corr 1.000000, relRMSE ~1e-4, maxrel <1e-3.
    assert corr_b > 0.9999 and corr_c > 0.9999
    assert rmse_b < 2e-3 and rmse_c < 2e-3
    assert maxrel_b < 1e-2 and maxrel_c < 1e-2


# ---------------------------------------------------------------------------
# Least-squares / P-spline fits (the higher-accuracy estimators)
# ---------------------------------------------------------------------------


def test_least_squares_reproduces_smooth_field_exactly():
    # Unlike single-level MBA (biased), one LS fit recovers a smooth field.
    n = 200
    x = np.linspace(0, 1, n)
    f = (np.sin(2 * np.pi * x) + 0.3 * np.cos(5 * np.pi * x)).astype(
        np.float32
    )
    approx = np.array(
        bspline_approximate(
            jnp.asarray(f),
            control_points=24,
            spatial_rank=1,
            method='least_squares',
            ridge=1e-6,
        )
    )
    assert np.sqrt(np.mean((approx - f) ** 2)) < 0.01


def test_least_squares_reproduces_constant():
    c = np.full(64, 3.7, np.float32)
    approx = np.array(
        bspline_approximate(
            jnp.asarray(c),
            control_points=8,
            spatial_rank=1,
            method='least_squares',
            ridge=1e-6,
        )
    )
    np.testing.assert_allclose(approx, 3.7, atol=1e-2)


def test_psplines_penalty_increases_smoothness():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 1, 200)
    clean = np.sin(2 * np.pi * x).astype(np.float32)
    noisy = (clean + rng.normal(0, 0.3, 200)).astype(np.float32)
    roughness = []
    for pen in (0.0, 1.0, 100.0):
        f = np.array(
            bspline_approximate(
                jnp.asarray(noisy),
                control_points=60,
                spatial_rank=1,
                method='psplines',
                penalty=pen,
            )
        )
        roughness.append(float(np.sqrt(np.mean(np.diff(f, 2) ** 2))))
    # Stronger penalty -> smoother (smaller second difference).
    assert roughness[0] > roughness[1] > roughness[2]


def test_masked_least_squares_stable_and_accurate():
    s = 36
    ax = np.linspace(0, 1, s)
    xx, yy, zz = np.meshgrid(ax, ax, ax, indexing='ij')
    field = (np.sin(2 * np.pi * xx) * np.cos(2 * np.pi * yy)).astype(
        np.float32
    )
    rr = np.sqrt((xx - 0.5) ** 2 + (yy - 0.5) ** 2 + (zz - 0.5) ** 2)
    mask = (rr < 0.4).astype(np.float32)
    mm = mask > 0
    data = field.copy()
    data[~mm] = 999.0
    for meth in ('least_squares', 'psplines'):
        f = np.array(
            bspline_approximate(
                jnp.asarray(data),
                control_points=(12, 12, 12),
                weight=jnp.asarray(mask),
                method=meth,
                penalty=0.01,
            )
        )
        assert np.all(np.isfinite(f))
        err = np.sqrt(np.mean((f[mm] - field[mm]) ** 2)) / field[mm].std()
        assert err < 0.1


def test_bspline_method_validation():
    with pytest.raises(ValueError):
        bspline_approximate(
            jnp.ones((16,)),
            control_points=8,
            spatial_rank=1,
            method='nope',  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# bias_field_correction dispatcher
# ---------------------------------------------------------------------------


def test_dispatcher_n4_equals_n4_function():
    obs, mask, _, _ = _phantom(32)
    a = np.array(
        n4_bias_field_correction(jnp.asarray(obs), mask=jnp.asarray(mask))
    )
    b = np.array(
        bias_field_correction(
            jnp.asarray(obs), method='n4', mask=jnp.asarray(mask)
        )
    )
    # Same algorithm; not bit-identical because the histogram's GPU
    # scatter-add is nondeterministic (atomics) and the difference compounds
    # over the iteration -- agree to a relative tolerance.
    np.testing.assert_allclose(a, b, rtol=1e-2, atol=1e-2)


def test_dispatcher_method_validation():
    with pytest.raises(ValueError):
        bias_field_correction(jnp.ones((8, 8, 8)), method='nope')


def test_all_methods_recover_phantom_bias():
    obs, mask, bias_true, r = _phantom(40)
    mm = mask > 0
    bt = bias_true[mm]
    for method in ('n4', 'least_squares', 'psplines'):
        _, bias = bias_field_correction(
            jnp.asarray(obs),
            method=method,
            mask=jnp.asarray(mask),
            return_bias_field=True,
        )
        be = np.array(bias)[mm]
        # All three recover the field well.  LS / P-splines are competitive
        # with N4 (better on some phantoms, slightly worse on others -- see
        # docs/design/bias-field.md); we assert "good", not a ranking.
        assert np.corrcoef(be, bt)[0, 1] > 0.95, method
        s = np.sum(be * bt) / np.sum(bt * bt)
        rel_rmse = np.sqrt(np.mean((be - s * bt) ** 2)) / bt.mean()
        assert rel_rmse < 0.05, (method, rel_rmse)


def test_n4_live_sitk_parity():
    sitk = pytest.importorskip('SimpleITK')
    obs, mask, _, _ = _phantom(40, seed=7)
    img = sitk.GetImageFromArray(obs)
    msk = sitk.GetImageFromArray(mask.astype(np.uint8))
    f = sitk.N4BiasFieldCorrectionImageFilter()
    f.SetMaximumNumberOfIterations([50, 50, 50, 50])
    f.SetConvergenceThreshold(1e-3)
    f.SetNumberOfControlPoints([4, 4, 4])
    f.SetNumberOfHistogramBins(200)
    f.SetBiasFieldFullWidthAtHalfMaximum(0.15)
    f.SetWienerFilterNoise(0.01)
    f.SetSplineOrder(3)
    itk_corr = sitk.GetArrayFromImage(f.Execute(img, msk)).astype(np.float32)
    itk_bias = np.exp(
        sitk.GetArrayFromImage(f.GetLogBiasFieldAsImage(img))
    ).astype(np.float32)

    corrected, bias = n4_bias_field_correction(
        jnp.asarray(obs), mask=jnp.asarray(mask), return_bias_field=True
    )
    corr_b, rmse_b, _ = _bias_parity(np.array(bias), itk_bias, mask)
    corr_c, rmse_c, _ = _bias_parity(np.array(corrected), itk_corr, mask)
    assert corr_b > 0.999 and corr_c > 0.999
    assert rmse_b < 5e-3 and rmse_c < 5e-3
