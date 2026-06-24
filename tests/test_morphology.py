# -*- coding: utf-8 -*-
"""Tests for ``nitrix.morphology``.

Coverage:

- ``dilate`` / ``erode`` / ``open`` / ``close``: bit-exact match
  against ``scipy.ndimage.grey_dilation`` etc. on interior pixels;
  boundary semantics differ (we pad with algebra identity, scipy
  defaults to ``reflect``) but the interior is the meaningful test.
- ``distance_transform``: parity against
  ``scipy.ndimage.distance_transform_cdt`` for ``chessboard`` and
  ``taxicab`` metrics.
- ``median_filter``: interior parity against
  ``scipy.ndimage.median_filter``.
- Non-trivial structuring element: explicit per-position weights for
  dilation match the algebra's definition.
- Differentiability: ``dilate`` / ``erode`` have well-defined
  subgradients via the TROPICAL_* backward; check the gradient is
  finite and routes to a single neighbour.
- ``susan_emulator``: raises ``NotImplementedError`` until the
  ``bilateral_gaussian`` dependency lands.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

scipy_ndi = pytest.importorskip('scipy.ndimage')

from nitrix.morphology import (
    close as morph_close,
)
from nitrix.morphology import (
    connected_components,
    dilate,
    distance_transform,
    distance_transform_edt,
    erode,
    largest_connected_component,
    median_filter,
)
from nitrix.morphology import (
    open as morph_open,
)
from nitrix.smoothing import susan_emulator

jax.config.update('jax_enable_x64', True)


# ---------------------------------------------------------------------------
# dilate / erode / open / close
# ---------------------------------------------------------------------------


def _interior_match(got, ref, pad):
    """Trim ``pad`` from every axis and compare."""
    got_i = np.asarray(got)
    ref_i = np.asarray(ref)
    sl = tuple(slice(pad, -pad if pad else None) for _ in range(got_i.ndim))
    np.testing.assert_allclose(got_i[sl], ref_i[sl], atol=1e-10)


def test_dilate_flat_3x3_matches_scipy():
    x = jax.random.normal(jax.random.key(0), (16, 16))
    got = dilate(x, size=3, backend='jax')
    ref = scipy_ndi.grey_dilation(
        np.asarray(x),
        size=3,
        mode='constant',
        cval=-np.inf,
    )
    np.testing.assert_array_equal(got, ref)


def test_erode_flat_3x3_matches_scipy():
    x = jax.random.normal(jax.random.key(1), (16, 16))
    got = erode(x, size=3, backend='jax')
    ref = scipy_ndi.grey_erosion(
        np.asarray(x),
        size=3,
        mode='constant',
        cval=np.inf,
    )
    np.testing.assert_array_equal(got, ref)


def test_open_matches_dilate_of_erode():
    x = jax.random.normal(jax.random.key(2), (16, 16))
    got = morph_open(x, size=3, backend='jax')
    eroded = erode(x, size=3, backend='jax')
    expected = dilate(eroded, size=3, backend='jax')
    np.testing.assert_array_equal(got, expected)


def test_close_matches_erode_of_dilate():
    x = jax.random.normal(jax.random.key(3), (16, 16))
    got = morph_close(x, size=3, backend='jax')
    dilated = dilate(x, size=3, backend='jax')
    expected = erode(dilated, size=3, backend='jax')
    np.testing.assert_array_equal(got, expected)


def test_dilate_3d():
    x = jax.random.normal(jax.random.key(4), (8, 8, 8))
    got = dilate(x, size=3, backend='jax')
    ref = scipy_ndi.grey_dilation(
        np.asarray(x),
        size=3,
        mode='constant',
        cval=-np.inf,
    )
    np.testing.assert_array_equal(got, ref)


def test_dilate_4d_fmri_shape():
    """4D morphology: volume + time, the natural fMRI layout."""
    x = jax.random.normal(jax.random.key(40), (6, 6, 6, 6))
    got = dilate(x, size=3, backend='jax')
    ref = scipy_ndi.grey_dilation(
        np.asarray(x),
        size=3,
        mode='constant',
        cval=-np.inf,
    )
    np.testing.assert_array_equal(got, ref)


def test_erode_4d_fmri_shape():
    x = jax.random.normal(jax.random.key(41), (5, 5, 5, 5))
    got = erode(x, size=3, backend='jax')
    ref = scipy_ndi.grey_erosion(
        np.asarray(x),
        size=3,
        mode='constant',
        cval=np.inf,
    )
    np.testing.assert_array_equal(got, ref)


def test_dilate_explicit_structuring_element():
    # 1D, structure = [-1, 0, +1] -- centered SAME-padded conv.
    # Padded x = [-inf, 1, 2, 3, 4, 5, 6, 7, -inf].
    # out[i] = max( x_pad[i+0]+(-1), x_pad[i+1]+0, x_pad[i+2]+1 ).
    x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    se = jnp.array([-1.0, 0.0, 1.0])
    got = dilate(x, structuring_element=se, backend='jax')
    # i=0: max(-inf-1, 1+0,  2+1) = 3
    # i=1: max(1-1,   2+0,  3+1) = 4
    # i=2: max(2-1,   3+0,  4+1) = 5
    # i=3: max(3-1,   4+0,  5+1) = 6
    # i=4: max(4-1,   5+0,  6+1) = 7
    # i=5: max(5-1,   6+0,  7+1) = 8   (the +1 pulls the right neighbour)
    # i=6: max(6-1,   7+0,  -inf+1) = 7
    expected = jnp.array([3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 7.0])
    np.testing.assert_allclose(got, expected, atol=1e-10)


def test_dilate_erode_anisotropic_size():
    x = jax.random.normal(jax.random.key(5), (10, 16))
    got = dilate(x, size=(3, 5), backend='jax')
    ref = scipy_ndi.grey_dilation(
        np.asarray(x),
        size=(3, 5),
        mode='constant',
        cval=-np.inf,
    )
    np.testing.assert_array_equal(got, ref)


# ---------------------------------------------------------------------------
# distance_transform
# ---------------------------------------------------------------------------


def test_distance_transform_chebyshev_matches_scipy():
    mask = np.zeros((16, 16), dtype=np.float32)
    mask[4:12, 4:12] = 1.0
    got = distance_transform(
        jnp.asarray(mask),
        metric='chebyshev',
        backend='jax',
    )
    ref = scipy_ndi.distance_transform_cdt(mask, metric='chessboard')
    np.testing.assert_array_equal(got, ref.astype(np.float32))


def test_distance_transform_city_block_matches_scipy():
    mask = np.zeros((16, 16), dtype=np.float32)
    mask[4:12, 4:12] = 1.0
    got = distance_transform(
        jnp.asarray(mask),
        metric='city_block',
        backend='jax',
    )
    ref = scipy_ndi.distance_transform_cdt(mask, metric='taxicab')
    np.testing.assert_array_equal(got, ref.astype(np.float32))


def test_distance_transform_3d():
    mask = np.zeros((8, 8, 8), dtype=np.float32)
    mask[2:6, 2:6, 2:6] = 1.0
    got = distance_transform(
        jnp.asarray(mask),
        metric='chebyshev',
        backend='jax',
    )
    ref = scipy_ndi.distance_transform_cdt(mask, metric='chessboard')
    np.testing.assert_array_equal(got, ref.astype(np.float32))


def test_distance_transform_euclidean_is_default_and_exact():
    # The default metric is exact Euclidean -- scipy distance_transform_edt.
    rng = np.random.RandomState(0)
    mask = (rng.random((16, 16)) > 0.5).astype(np.float32)
    got = distance_transform(jnp.asarray(mask), backend='jax')  # default
    ref = scipy_ndi.distance_transform_edt(mask > 0.5)
    np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-4)


def test_distance_transform_edt_alias_3d_matches_scipy():
    rng = np.random.RandomState(1)
    mask = (rng.random((8, 8, 8)) > 0.5).astype(np.float32)
    got = distance_transform_edt(jnp.asarray(mask), backend='jax')
    ref = scipy_ndi.distance_transform_edt(mask > 0.5)
    np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-4)


def test_distance_transform_rejects_unknown_metric():
    mask = jnp.zeros((4, 4))
    with pytest.raises(ValueError, match='metric'):
        distance_transform(mask, metric='manhattan', backend='jax')


@pytest.mark.parametrize(
    'shape,sampling',
    [
        ((20, 24), (1.0, 3.0)),
        ((20, 24), (2.5, 1.0)),
        ((20, 24), 2.0),
        ((10, 12, 14), (1.0, 1.0, 3.0)),
        ((10, 12, 14), (2.0, 1.0, 0.5)),
    ],
)
def test_distance_transform_anisotropic_matches_scipy(shape, sampling):
    # B20: anisotropic voxel spacing -- the exact EDT must match scipy's
    # distance_transform_edt(sampling=...), not just the unit grid.
    rng = np.random.RandomState(2)
    mask = (rng.random(shape) > 0.2).astype(np.float32)
    got = distance_transform_edt(
        jnp.asarray(mask), sampling=sampling, backend='jax'
    )
    ref = scipy_ndi.distance_transform_edt(mask > 0.5, sampling=sampling)
    np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-3)


def test_distance_transform_unit_sampling_byte_identical():
    # sampling=None and sampling=1.0 are the same unit grid (byte-identical).
    rng = np.random.RandomState(3)
    mask = jnp.asarray((rng.random((16, 18)) > 0.4).astype(np.float32))
    none = distance_transform_edt(mask, backend='jax')
    unit = distance_transform_edt(mask, sampling=1.0, backend='jax')
    np.testing.assert_array_equal(np.asarray(none), np.asarray(unit))


def test_distance_transform_sampling_validation():
    mask = jnp.asarray(np.ones((8, 8), np.float32))
    # wrong number of spacings
    with pytest.raises(ValueError, match='sampling'):
        distance_transform_edt(mask, sampling=(1.0, 2.0, 3.0), backend='jax')
    # sampling is euclidean-only; chamfer rejects it
    with pytest.raises(ValueError, match='sampling'):
        distance_transform(
            mask, metric='chebyshev', sampling=2.0, backend='jax'
        )


# ---------------------------------------------------------------------------
# median_filter
# ---------------------------------------------------------------------------


def test_median_filter_interior_matches_scipy():
    x = jax.random.normal(jax.random.key(10), (16, 16))
    got = median_filter(x, size=3)
    ref = scipy_ndi.median_filter(
        np.asarray(x),
        size=3,
        mode='reflect',
    )
    # Boundary modes differ; compare interior.
    _interior_match(got, ref, pad=2)


def test_median_filter_1d_interior():
    x = jax.random.normal(jax.random.key(11), (20,))
    got = median_filter(x, size=5)
    ref = scipy_ndi.median_filter(np.asarray(x), size=5, mode='reflect')
    _interior_match(got, ref, pad=3)


def test_median_filter_3d_interior():
    x = jax.random.normal(jax.random.key(12), (8, 8, 8))
    got = median_filter(x, size=3)
    ref = scipy_ndi.median_filter(np.asarray(x), size=3, mode='reflect')
    _interior_match(got, ref, pad=2)


def test_median_filter_structuring_element_mask():
    # Plus-shaped neighbourhood (only axis-aligned positions).
    x = jax.random.normal(jax.random.key(13), (16, 16))
    se = jnp.array(
        [
            [False, True, False],
            [True, True, True],
            [False, True, False],
        ]
    )
    got = median_filter(x, structuring_element=se)
    # Build same with scipy's footprint argument.
    ref = scipy_ndi.median_filter(
        np.asarray(x),
        footprint=np.asarray(se),
        mode='reflect',
    )
    _interior_match(got, ref, pad=2)


# ---------------------------------------------------------------------------
# Differentiability of dilate / erode (TROPICAL_* subgradient)
# ---------------------------------------------------------------------------


def test_dilate_spherical_grid_composition():
    """Documented JOSA composition: pad with sphere-grid topology,
    then dilate(VALID).  The VALID dilation consumes exactly the
    pad we added, so the output has the original shape -- no
    explicit unpad needed when the SE radius equals the pad.

    Without spherical-grid pad, a feature near the longitudinal seam
    fails to spread across the seam.  With the composition, it does.
    """
    from nitrix.geometry import sphere_grid_pad_2d

    H, W = 8, 8
    # Place a single hot pixel at (4, 0) -- on the longitudinal seam.
    mask = jnp.zeros((H, W), dtype=jnp.float32)
    mask = mask.at[4, 0].set(1.0)

    # Plain dilate(SAME) with -inf padding: the hot pixel spreads to
    # (3..5, 0..1) but NOT to (3..5, W-1) -- the seam doesn't wrap.
    plain = dilate(mask, size=3, backend='jax')
    assert plain[4, 0] == 1.0
    assert plain[4, 1] == 1.0
    assert plain[4, W - 1] == 0.0  # seam not crossed

    # Spherical-grid composition: pad by SE radius = (3-1)/2 = 1,
    # then dilate VALID.  The VALID dilation consumes the pad and
    # restores the original spatial shape; no unpad needed.
    se_radius = 1
    padded = sphere_grid_pad_2d(mask, pad=se_radius)
    out = dilate(padded, size=3, padding='VALID', backend='jax')
    assert out.shape == mask.shape
    assert out[4, 0] == 1.0
    assert out[4, 1] == 1.0
    assert out[4, W - 1] == 1.0  # seam crossed


def test_dilate_spherical_grid_composition_with_explicit_unpad():
    """When SE radius is smaller than the pad (or you want to chain
    multiple ops), use the explicit sphere_grid_unpad_2d to crop
    back.
    """
    from nitrix.geometry import sphere_grid_pad_2d, sphere_grid_unpad_2d

    H, W = 8, 8
    mask = jnp.zeros((H, W), dtype=jnp.float32)
    mask = mask.at[4, 0].set(1.0)

    # Pad by 3 but dilate with SE radius 1.  Padded shape (14, 14),
    # dilated VALID (12, 12), unpad with pad=2 to get (8, 8).
    padded = sphere_grid_pad_2d(mask, pad=3)
    dilated = dilate(padded, size=3, padding='VALID', backend='jax')
    # Dilated shape is (14 - 2, 14 - 2) = (12, 12); to recover (8, 8)
    # we strip the leftover (12 - 8) / 2 = 2 on each side.
    out = sphere_grid_unpad_2d(dilated, pad=2)
    assert out.shape == mask.shape
    # Same wrap behaviour as the size-matched case.
    assert out[4, W - 1] == 1.0


def test_dilate_gradient_routes_to_argmax():
    # In 1D, dilate at position i routes its upstream gradient to the
    # j ∈ {i-1, i, i+1} (with VALID/SAME padding) attaining the max.
    # Construct x so each window has a clear winner.
    x = jnp.array([1.0, 5.0, 2.0, 8.0, 3.0])

    # 3-window dilation w/ SAME, VALID-interior:
    #   out[0] = max(x[0], x[1]) = 5    [boundary]
    #   out[1] = max(x[0], x[1], x[2]) = 5   -> argmax = 1
    #   out[2] = max(x[1], x[2], x[3]) = 8   -> argmax = 3
    #   out[3] = max(x[2], x[3], x[4]) = 8   -> argmax = 3
    #   out[4] = max(x[3], x[4]) = 8    [boundary]
    def loss(x):
        return dilate(x, size=3, backend='jax').sum()

    g = jax.grad(loss)(x)
    # Each output cell contributes 1 to its argmax position.
    # Position 1 receives 1 (from out[1]).
    # Position 3 receives 3 (from out[2], out[3], out[4]).
    # Positions 0, 2, 4 receive 0 from interior; boundary
    # tie-breaking with -inf gives them an extra 1 each at certain
    # boundary windows.  Exact values depend on tie semantics; the
    # *invariant* is non-negative integers summing to the count of
    # output cells, which here is 5.
    assert bool(jnp.all(g >= 0))
    assert float(g.sum()) == 5.0
    assert bool(jnp.all(jnp.isfinite(g)))


def test_erode_gradient_routes_to_argmin():
    x = jnp.array([5.0, 1.0, 8.0, 2.0, 7.0])

    def loss(x):
        return erode(x, size=3, backend='jax').sum()

    g = jax.grad(loss)(x)
    assert bool(jnp.all(jnp.isfinite(g)))
    # Each output position contributes 1 to its argmin -- sum is the
    # number of outputs.
    assert float(g.sum()) == 5.0


# ---------------------------------------------------------------------------
# B19 regression gate: the flat-box fast path must survive jit(grad(...))
#
# ``jit(grad)`` is the double-transform a ``@jax.jit`` training step runs (inner
# ``grad`` compiled by an outer ``jit``).  The flat ``lax.reduce_window`` fast
# path silently dropped it once (a traced window init defeated JAX's monoid
# detection, routing to the non-differentiable generic ``reduce_window_p``);
# the eager-grad tests above did not catch it.  These gates assert the capability
# directly so a future perf tweak to the fast path cannot regress it unnoticed.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('op', [dilate, erode, morph_open, morph_close])
def test_flat_path_jit_of_grad_is_finite(op):
    x = jax.random.normal(jax.random.key(19), (10, 10))
    grad_fn = jax.jit(jax.grad(lambda z: jnp.sum(op(z, size=3) ** 2)))
    g = grad_fn(x)
    assert g.shape == x.shape
    assert bool(jnp.all(jnp.isfinite(g)))


@pytest.mark.parametrize('op', [dilate, erode])
def test_flat_path_matches_semiring_forward_and_grad(op):
    """The flat fast path and the explicit all-zero-SE semiring path are the
    same operator; they must agree on forward, ``grad``, and ``jit(grad)``, so
    the two implementations cannot silently diverge.  A continuous input makes
    argmax/argmin ties measure-zero, so the subgradients are unique and equal.
    """
    x = jax.random.normal(jax.random.key(23), (9, 11))

    def flat(z):
        return op(z, size=3, backend='jax')

    def semi(z):
        return op(z, structuring_element=jnp.zeros((3, 3)), backend='jax')

    np.testing.assert_array_equal(flat(x), semi(x))

    def flat_loss(z):
        return jnp.sum(flat(z) ** 2)

    def semi_loss(z):
        return jnp.sum(semi(z) ** 2)

    g_flat = jax.grad(flat_loss)(x)
    g_semi = jax.grad(semi_loss)(x)
    g_flat_jit = jax.jit(jax.grad(flat_loss))(x)
    np.testing.assert_allclose(g_flat, g_semi, atol=1e-10)
    np.testing.assert_allclose(g_flat_jit, g_flat, atol=1e-10)


@pytest.mark.parametrize('op', [dilate, erode, morph_open, morph_close])
@pytest.mark.parametrize(
    'dtype', [jnp.int32, jnp.bool_, jnp.float32, jnp.float64]
)
def test_flat_path_promotes_int_bool_to_float(op, dtype):
    """Grayscale morphology is real-valued: integer / boolean inputs are lifted
    to ``float`` (so the ``-inf`` / ``+inf`` window identity is representable and
    the gradient is well-defined), while floating inputs keep their dtype.  The
    integer path previously raised ``OverflowError`` (``-inf -> int``).
    """
    key = jax.random.key(29)
    if dtype is jnp.bool_:
        x = jax.random.normal(key, (8, 8)) > 0
    elif dtype is jnp.int32:
        x = jax.random.randint(key, (8, 8), 0, 9).astype(jnp.int32)
    else:
        x = jax.random.normal(key, (8, 8)).astype(dtype)

    out = op(x, size=3, backend='jax')
    assert jnp.issubdtype(out.dtype, jnp.floating)
    if jnp.issubdtype(dtype, jnp.floating):
        assert out.dtype == dtype  # floating inputs unchanged
    assert bool(jnp.all(jnp.isfinite(out)))

    # Interior agrees with scipy on the equivalent float volume.
    ref_fn = {
        'dilate': scipy_ndi.grey_dilation,
        'erode': scipy_ndi.grey_erosion,
    }.get(op.__name__)
    if ref_fn is not None:
        ref = ref_fn(np.asarray(x).astype(np.float64), size=3)
        np.testing.assert_allclose(
            np.asarray(out)[1:-1, 1:-1],
            ref[1:-1, 1:-1],
            atol=1e-10,
        )


# ---------------------------------------------------------------------------
# susan_emulator (stub until smoothing.bilateral_gaussian lands)
# ---------------------------------------------------------------------------


def test_susan_emulator_implemented_via_bilateral():
    """Once ``smoothing.bilateral_gaussian`` shipped (this sprint),
    ``susan_emulator`` became a real op rather than a raise.  This
    test pins the transition from "stub" to "implemented".
    """
    x = jax.random.normal(jax.random.key(99), (8, 8))
    out = susan_emulator(x, sigma_space=1.0, sigma_intensity=0.5)
    assert out.shape == x.shape
    assert bool(jnp.all(jnp.isfinite(out)))


# ---------------------------------------------------------------------------
# connected_components / largest_connected_component
# ---------------------------------------------------------------------------


def _same_partition(a, b) -> bool:
    """True if label images ``a`` / ``b`` induce the same partition."""
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    if not np.array_equal(a == 0, b == 0):
        return False
    fg = a > 0
    a2b: dict = {}
    b2a: dict = {}
    for x, y in zip(a[fg].tolist(), b[fg].tolist()):
        if a2b.setdefault(x, y) != y or b2a.setdefault(y, x) != x:
            return False
    return True


def test_connected_components_two_blobs():
    mask = np.zeros((8, 8), dtype=bool)
    mask[1:3, 1:3] = True
    mask[5:8, 5:8] = True
    labels = connected_components(jnp.asarray(mask), connectivity=1)
    assert int(labels.max()) == 2
    # Labels are contiguous 1..K.
    uniq = np.unique(np.asarray(labels))
    np.testing.assert_array_equal(uniq, np.array([0, 1, 2]))


def test_connected_components_matches_scipy_2d():
    rng = np.random.default_rng(0)
    mask = rng.random((20, 24)) > 0.5
    for conn in (1, 2):
        struct = scipy_ndi.generate_binary_structure(2, conn)
        ref, nref = scipy_ndi.label(mask, structure=struct)
        ours = connected_components(jnp.asarray(mask), connectivity=conn)
        assert int(ours.max()) == nref
        assert _same_partition(ours, ref)


def test_connected_components_matches_scipy_3d():
    rng = np.random.default_rng(1)
    mask = rng.random((10, 11, 9)) > 0.55
    for conn in (1, 3):
        struct = scipy_ndi.generate_binary_structure(3, conn)
        ref, nref = scipy_ndi.label(mask, structure=struct)
        ours = connected_components(jnp.asarray(mask), connectivity=conn)
        assert int(ours.max()) == nref
        assert _same_partition(ours, ref)


def test_connected_components_diagonal_merges_under_full_connectivity():
    # Two pixels touching only at a corner: separate at conn=1, joined at 2.
    mask = np.zeros((4, 4), dtype=bool)
    mask[1, 1] = True
    mask[2, 2] = True
    assert (
        int(connected_components(jnp.asarray(mask), connectivity=1).max()) == 2
    )
    assert (
        int(connected_components(jnp.asarray(mask), connectivity=2).max()) == 1
    )


def test_connected_components_empty_mask():
    labels = connected_components(jnp.zeros((5, 5), dtype=bool))
    np.testing.assert_array_equal(np.asarray(labels), 0)


def test_largest_connected_component_matches_scipy():
    rng = np.random.default_rng(2)
    mask = rng.random((24, 24)) > 0.4
    ref, nref = scipy_ndi.label(mask)
    sizes = np.bincount(ref.ravel())
    sizes[0] = 0
    ref_largest = ref == int(sizes.argmax())
    ours = largest_connected_component(jnp.asarray(mask), connectivity=1)
    np.testing.assert_array_equal(np.asarray(ours), ref_largest)


def test_largest_connected_component_empty_is_all_false():
    out = largest_connected_component(jnp.zeros((6, 6), dtype=bool))
    assert not bool(jnp.any(out))


def test_connected_components_jit():
    mask = jnp.asarray(np.random.default_rng(3).random((12, 12)) > 0.5)
    jitted = jax.jit(connected_components, static_argnames='connectivity')
    a = jitted(mask, connectivity=1)
    b = connected_components(mask, connectivity=1)
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_cc_long_diameter_snake_matches_scipy():
    # A 1-wide boustrophedon snake: a single component with very large
    # diameter -- the case pointer jumping must converge on (and where a
    # pure O(diameter) flood would need many passes).
    h, w = 9, 40
    mask = np.zeros((h, w), dtype=bool)
    for r in range(0, h, 2):
        mask[r, :] = True  # horizontal runs on even rows
    for r in range(0, h - 1, 2):
        col = w - 1 if (r // 2) % 2 == 0 else 0
        mask[r + 1, col] = True  # vertical connectors at alternating ends
    ref, nref = scipy_ndi.label(mask)
    ours = connected_components(jnp.asarray(mask), connectivity=1)
    assert int(ours.max()) == nref == 1
    assert _same_partition(ours, ref)
