# -*- coding: utf-8 -*-
"""Tests for ``nitrix.bias.histogram_match``.

Coverage:

- Identity property -- matching an image to itself recovers it (modulo
  bin discretisation).
- Mask semantics -- ``threshold_at_mean=True`` is the ITK default;
  passing an explicit ``source_weight`` / ``reference_weight`` overrides
  the mean-threshold; ``threshold_at_mean=False`` uses every voxel.
- Apply contract -- outer landmarks are the actual image extrema, so
  the output is always inside the reference's intensity range; no
  extrapolation regime to choose.
- Statistical match -- after matching, the source's quantile profile
  approximates the reference's at the chosen landmark positions.
- Validation -- bad ``n_match_points`` / ``n_histogram_levels`` /
  mismatched threshold-vs-weight policies are caught early.
- JIT friendliness -- the op compiles cleanly under ``jax.jit``.
- Live SimpleITK parity -- agreement with
  ``HistogramMatchingImageFilter`` to a tolerance proportional to the
  intensity range (the only published reference for this algorithm).

The SimpleITK parity test is gated by ``pytest.importorskip`` so the
suite still runs in environments without SimpleITK.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nitrix.bias import (
    histogram_match,
    histogram_match_apply,
    histogram_match_fit,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _synth_pair(seed: int = 0, n: int = 32):
    rng = np.random.default_rng(seed)
    src = rng.uniform(0.0, 1.0, (n, n, n)).astype(np.float32)
    ref = rng.normal(0.5, 0.15, (n + 8, n + 8, n + 8)).astype(np.float32)
    return src, ref


def _quantiles(x: np.ndarray, q: np.ndarray, mask=None) -> np.ndarray:
    if mask is not None:
        x = x[mask > 0]
    return np.quantile(x, q)


# ---------------------------------------------------------------------------
# Algorithmic properties
# ---------------------------------------------------------------------------


def test_identity_when_source_is_reference():
    # Matching an image to itself recovers it (modulo bin discretisation).
    # threshold_at_mean=False so every voxel contributes -- with the default
    # ``True``, below-mean voxels would extrapolate and pick up sub-bin slope
    # noise that this property is not about.
    src, _ = _synth_pair()
    out = histogram_match(
        jnp.asarray(src),
        jnp.asarray(src),
        threshold_at_mean=False,
    )
    np.testing.assert_allclose(np.asarray(out), src, atol=1e-5)


def test_quantile_profile_matches_reference_within_landmarks():
    # After matching, the source's quantile profile at the match points
    # should land near the reference's quantile profile at the same points.
    src, ref = _synth_pair(seed=1, n=48)
    out = np.asarray(
        histogram_match(
            jnp.asarray(src),
            jnp.asarray(ref),
            threshold_at_mean=False,
            n_match_points=7,
        )
    )
    # Sample interior quantiles (skip the extremes -- those are the bin_min /
    # bin_max corners, which the algorithm preserves only via extrapolation).
    qs = np.linspace(0.2, 0.8, 5)
    out_q = _quantiles(out, qs)
    ref_q = _quantiles(ref, qs)
    # A 2% absolute tolerance on the reference range is comfortable for the
    # finite landmark set + bin discretisation; the algorithm's contract is
    # an approximate match, not bit-equality of CDFs.
    rng = float(ref.max() - ref.min())
    np.testing.assert_allclose(out_q, ref_q, atol=0.02 * rng)


def test_threshold_at_mean_uses_above_mean_only():
    # With threshold_at_mean=True (default), below-source-mean voxels do not
    # contribute to the source histogram -- so the source landmark range
    # starts at roughly the source mean and the apply linearly extrapolates
    # below.  Verify that toggling the flag changes the landmark range.
    rng = np.random.default_rng(3)
    src = rng.uniform(0.0, 1.0, (32, 32, 32)).astype(np.float32)
    ref = rng.uniform(0.0, 1.0, (32, 32, 32)).astype(np.float32)

    out_default = np.asarray(
        histogram_match(jnp.asarray(src), jnp.asarray(ref))
    )
    out_all = np.asarray(
        histogram_match(
            jnp.asarray(src),
            jnp.asarray(ref),
            threshold_at_mean=False,
        )
    )
    # The two should not be identical (different contributing populations
    # -> different landmarks).
    assert not np.allclose(out_default, out_all)


def test_explicit_weight_overrides_threshold_at_mean():
    # When a weight is supplied explicitly, threshold_at_mean is ignored for
    # that image -- so weight=ones is equivalent to threshold_at_mean=False
    # on that side.
    rng = np.random.default_rng(4)
    src = rng.uniform(0.0, 1.0, (16, 16, 16)).astype(np.float32)
    ref = rng.uniform(0.0, 1.0, (16, 16, 16)).astype(np.float32)
    ones = jnp.ones_like(jnp.asarray(src))

    out_explicit = np.asarray(
        histogram_match(
            jnp.asarray(src),
            jnp.asarray(ref),
            source_weight=ones,
            reference_weight=ones,
            threshold_at_mean=True,  # ignored on both sides now
        )
    )
    out_flagged_off = np.asarray(
        histogram_match(
            jnp.asarray(src),
            jnp.asarray(ref),
            threshold_at_mean=False,
        )
    )
    np.testing.assert_allclose(out_explicit, out_flagged_off, atol=1e-5)


def test_output_bounded_by_reference_extrema():
    # Outer landmarks are always the image extrema, so the apply step is
    # interpolation-only -- every output value lies inside the reference's
    # intensity range, regardless of source range or threshold policy.
    rng = np.random.default_rng(5)
    src = rng.uniform(-1.0, 2.0, (24, 24, 24)).astype(np.float32)
    ref = rng.uniform(0.0, 1.0, (24, 24, 24)).astype(np.float32)
    for thresh in (True, False):
        out = np.asarray(
            histogram_match(
                jnp.asarray(src),
                jnp.asarray(ref),
                threshold_at_mean=thresh,
            )
        )
        assert out.min() >= float(ref.min()) - 1e-6, (thresh, out.min())
        assert out.max() <= float(ref.max()) + 1e-6, (thresh, out.max())


def test_endpoints_map_to_reference_extrema():
    # The actual src.min() and src.max() voxels must map exactly to
    # ref.min() and ref.max() respectively -- the corner landmarks are the
    # full extrema.
    rng = np.random.default_rng(8)
    src = rng.uniform(0.0, 1.0, (16, 16, 16)).astype(np.float32)
    ref = rng.uniform(2.0, 5.0, (16, 16, 16)).astype(np.float32)
    out = np.asarray(histogram_match(jnp.asarray(src), jnp.asarray(ref)))
    src_argmin = int(np.argmin(src))
    src_argmax = int(np.argmax(src))
    assert abs(float(out.flat[src_argmin]) - float(ref.min())) < 1e-5
    assert abs(float(out.flat[src_argmax]) - float(ref.max())) < 1e-5


# ---------------------------------------------------------------------------
# fit / apply seam (§6.5)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'kwargs',
    [
        {},  # ITK default: threshold_at_mean=True, mean landmark inserted
        {'threshold_at_mean': False},
        {'n_match_points': 5, 'n_histogram_levels': 256},
    ],
)
def test_convenience_equals_fit_then_apply_byteexact(kwargs):
    # The convenience IS apply(source, fit(reference)) -- same dtype on both
    # sides (the norm), so the composition is byte-identical, not merely
    # close.  This is the §6.5 byte-faithful-by-construction invariant.
    src, ref = _synth_pair(seed=2, n=24)
    src_j, ref_j = jnp.asarray(src), jnp.asarray(ref)
    via_convenience = np.asarray(histogram_match(src_j, ref_j, **kwargs))
    landmarks = histogram_match_fit(
        ref_j,
        reference_weight=None,
        **{k: v for k, v in kwargs.items() if k != 'source_weight'},
    )
    via_split = np.asarray(histogram_match_apply(src_j, landmarks, **kwargs))
    np.testing.assert_array_equal(via_split, via_convenience)


def test_fit_returns_resolved_landmark_vector():
    # threshold_at_mean=True with no weight inserts the mean landmark
    # (extrema + mean + n match points = n + 3); False / explicit weight
    # gives the simpler extrema + n match points = n + 2.
    _, ref = _synth_pair(seed=7, n=20)
    ref_j = jnp.asarray(ref)
    with_mean = histogram_match_fit(ref_j, n_match_points=7)
    without_mean = histogram_match_fit(
        ref_j, n_match_points=7, threshold_at_mean=False
    )
    assert with_mean.shape == (7 + 3,)
    assert without_mean.shape == (7 + 2,)
    # Landmarks are non-decreasing (quantile-ordered intensities).
    assert np.all(np.diff(np.asarray(with_mean)) >= -1e-5)


def test_apply_to_many_sources_reuses_one_fit():
    # Fit the reference once, apply to several sources -- each result equals
    # the single-pair convenience (the whole point of the split).
    _, ref = _synth_pair(seed=9, n=24)
    ref_j = jnp.asarray(ref)
    landmarks = histogram_match_fit(ref_j)
    for s in range(3):
        src = (
            np.random.default_rng(100 + s)
            .uniform(0.0, 1.0, (24, 24, 24))
            .astype(np.float32)
        )
        src_j = jnp.asarray(src)
        np.testing.assert_array_equal(
            np.asarray(histogram_match_apply(src_j, landmarks)),
            np.asarray(histogram_match(src_j, ref_j)),
        )


def test_apply_rejects_landmark_length_mismatch():
    # Fit with the mean landmark (length n+3) but apply with
    # threshold_at_mean=False (source length n+2) -> malformed interp,
    # caught at the apply boundary.
    src, ref = _synth_pair(seed=10, n=16)
    landmarks = histogram_match_fit(jnp.asarray(ref))  # has the mean landmark
    with pytest.raises(ValueError):
        histogram_match_apply(
            jnp.asarray(src), landmarks, threshold_at_mean=False
        )


def test_fit_apply_jit_compatible():
    src, ref = _synth_pair(seed=12, n=20)
    landmarks = jax.jit(histogram_match_fit)(jnp.asarray(ref))
    out = jax.jit(histogram_match_apply)(jnp.asarray(src), landmarks)
    assert out.shape == src.shape
    assert jnp.isfinite(out).all()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_validation_rejects_bad_kwargs():
    src = jnp.ones((8, 8), dtype=jnp.float32)
    ref = jnp.ones((8, 8), dtype=jnp.float32)
    ones = jnp.ones_like(src)
    with pytest.raises(ValueError):
        histogram_match(src, ref, n_match_points=0)
    with pytest.raises(ValueError):
        histogram_match(src, ref, n_histogram_levels=1)
    # Mixed weight / threshold policies would yield landmark arrays of
    # different lengths (one side has the mean inserted, the other does
    # not) -- the apply would be malformed; we raise early.
    with pytest.raises(ValueError):
        histogram_match(
            src,
            ref,
            source_weight=ones,
            threshold_at_mean=True,
        )


# ---------------------------------------------------------------------------
# JIT friendliness
# ---------------------------------------------------------------------------


def test_jit_compatible():
    src, ref = _synth_pair(seed=6, n=24)
    f = jax.jit(histogram_match)
    out = f(jnp.asarray(src), jnp.asarray(ref))
    assert out.shape == src.shape
    assert jnp.isfinite(out).all()


# ---------------------------------------------------------------------------
# Live SimpleITK parity
# ---------------------------------------------------------------------------


def test_live_sitk_parity():
    sitk = pytest.importorskip('SimpleITK')
    src, ref = _synth_pair(seed=11, n=40)

    img_src = sitk.GetImageFromArray(src)
    img_ref = sitk.GetImageFromArray(ref)
    f = sitk.HistogramMatchingImageFilter()
    f.SetNumberOfHistogramLevels(1024)
    f.SetNumberOfMatchPoints(7)
    f.ThresholdAtMeanIntensityOn()
    itk_out = sitk.GetArrayFromImage(f.Execute(img_src, img_ref)).astype(
        np.float32
    )

    jax_out = np.asarray(
        histogram_match(
            jnp.asarray(src),
            jnp.asarray(ref),
            n_match_points=7,
            n_histogram_levels=1024,
            threshold_at_mean=True,
        )
    )

    # The intensity range of the reference is the natural normaliser for
    # absolute-error comparison.  Now that the algorithm exactly matches
    # ITK's landmark structure (extrema + optional mean + quantile match
    # points), residual diffs come from sub-bin landmark interpolation
    # only -- well under 1e-3 of the reference range, with the mean diff
    # in the 1e-5 ballpark.
    ref_range = float(ref.max() - ref.min())
    diff = np.abs(jax_out - itk_out)
    assert diff.max() < 1e-3 * ref_range, (
        f'max diff {diff.max():.4g} exceeds {1e-3 * ref_range:.4g} '
        f'(ref range {ref_range:.4g})'
    )
    assert diff.mean() < 5e-5 * ref_range, (
        f'mean diff {diff.mean():.4g} exceeds {5e-5 * ref_range:.4g}'
    )
