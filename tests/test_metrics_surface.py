# -*- coding: utf-8 -*-
"""Tests for ``nitrix.metrics.surface``: ``hausdorff95`` / ``surface_dice``.

Parity oracle: a scipy reimplementation of **MONAI**'s documented algorithm
(``monai.metrics.compute_hausdorff_distance(percentile=95)`` /
``compute_surface_dice``) -- connectivity-1 surface ``binary_erosion(mask) ^
mask``, exact Euclidean ``distance_transform_edt`` of the complement, HD95 =
``max`` over directions of the 95th-percentile directed distance, NSD =
``(|S_p<=tau| + |S_t<=tau|) / (|S_p| + |S_t|)``.  scipy *is* what MONAI calls
under the hood, so this oracle is MONAI-exact (separately confirmed against a
live MONAI run during development).  Test masks keep a background margin so the
volume-border erosion convention is not exercised (it matches scipy
``border_value=0`` regardless).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update('jax_enable_x64', True)

from nitrix.metrics import hausdorff95, surface_dice  # noqa: E402
from nitrix.metrics.surface import _surface  # noqa: E402

ndi = pytest.importorskip('scipy.ndimage')


# --- MONAI-equivalent scipy oracle --------------------------------------


def _surf(m):
    return m ^ ndi.binary_erosion(m)


def _hd95(p, g, spacing=None):
    ep, eg = _surf(p), _surf(g)
    if not ep.any() or not eg.any():
        return np.inf
    d_pg = ndi.distance_transform_edt(~eg, sampling=spacing)[ep]
    d_gp = ndi.distance_transform_edt(~ep, sampling=spacing)[eg]
    return max(np.percentile(d_pg, 95), np.percentile(d_gp, 95))


def _nsd(p, g, tol, spacing=None):
    ep, eg = _surf(p), _surf(g)
    complete = ep.sum() + eg.sum()
    if complete == 0:
        return np.nan
    d_pg = (
        ndi.distance_transform_edt(~eg, sampling=spacing)
        if eg.any()
        else np.full(p.shape, np.inf)
    )
    d_gp = (
        ndi.distance_transform_edt(~ep, sampling=spacing)
        if ep.any()
        else np.full(p.shape, np.inf)
    )
    return ((d_pg[ep] <= tol).sum() + (d_gp[eg] <= tol).sum()) / complete


def _ball(shape, c, r):
    idx = np.indices(shape).astype(float)
    d2 = sum((idx[i] - c[i]) ** 2 for i in range(len(shape)))
    return d2 <= r * r


def _close(a, b, tol=1e-6):
    a, b = float(a), float(b)
    if np.isinf(a) or np.isinf(b):
        return np.isinf(a) and np.isinf(b) and (a > 0) == (b > 0)
    if np.isnan(a) or np.isnan(b):
        return np.isnan(a) and np.isnan(b)
    return abs(a - b) <= tol


# --- surface extraction --------------------------------------------------


@pytest.mark.parametrize('shape', [(24, 24), (16, 18, 20)])
def test_surface_extraction_matches_scipy(shape):
    c = tuple(s // 2 for s in shape)
    mask = _ball(shape, c, min(shape) // 3)
    got = np.asarray(_surface(jnp.asarray(mask), backend='jax'))
    assert np.array_equal(got, _surf(mask))


# --- hausdorff95 ---------------------------------------------------------


@pytest.mark.parametrize('spacing', [None, (1.0, 1.0, 3.0), 2.0])
def test_hausdorff95_matches_monai_oracle_3d(spacing):
    p = _ball((40, 40, 40), (20, 20, 20), 10)
    g = _ball((40, 40, 40), (22, 21, 20), 10)
    got = hausdorff95(jnp.asarray(p), jnp.asarray(g), spacing=spacing)
    assert _close(got, _hd95(p, g, spacing))


def test_hausdorff95_2d_and_symmetry():
    p = _ball((50, 50), (25, 25), 12)
    g = _ball((50, 50), (25, 29), 12)
    assert _close(hausdorff95(jnp.asarray(p), jnp.asarray(g)), _hd95(p, g))
    # symmetric in its arguments (max over both directions)
    assert _close(
        hausdorff95(jnp.asarray(p), jnp.asarray(g)),
        hausdorff95(jnp.asarray(g), jnp.asarray(p)),
    )


# --- surface_dice --------------------------------------------------------


@pytest.mark.parametrize('tol', [0.5, 1.0, 2.0, 5.0])
def test_surface_dice_matches_monai_oracle_3d(tol):
    p = _ball((36, 36, 36), (18, 18, 18), 9)
    g = _ball((36, 36, 36), (20, 18, 18), 9)
    got = surface_dice(jnp.asarray(p), jnp.asarray(g), tolerance=tol)
    assert _close(got, _nsd(p, g, tol))


def test_surface_dice_anisotropic():
    p = _ball((30, 30, 30), (15, 15, 15), 8)
    g = _ball((30, 30, 30), (15, 17, 15), 8)
    sp = (1.0, 1.0, 2.5)
    got = surface_dice(
        jnp.asarray(p), jnp.asarray(g), tolerance=2.0, spacing=sp
    )
    assert _close(got, _nsd(p, g, 2.0, sp))


# --- degenerate / identity ----------------------------------------------


def test_identical_masks():
    m = _ball((30, 30, 30), (15, 15, 15), 8)
    assert float(hausdorff95(jnp.asarray(m), jnp.asarray(m))) == 0.0
    assert (
        float(surface_dice(jnp.asarray(m), jnp.asarray(m), tolerance=0.5))
        == 1.0
    )


def test_degenerate_empty():
    m = _ball((20, 20, 20), (10, 10, 10), 5)
    empty = np.zeros((20, 20, 20), bool)
    # one empty -> HD95 inf, NSD 0
    assert np.isinf(float(hausdorff95(jnp.asarray(m), jnp.asarray(empty))))
    assert (
        float(surface_dice(jnp.asarray(m), jnp.asarray(empty), tolerance=1.0))
        == 0.0
    )
    # both empty -> HD95 inf, NSD nan
    assert np.isinf(float(hausdorff95(jnp.asarray(empty), jnp.asarray(empty))))
    assert np.isnan(
        float(
            surface_dice(jnp.asarray(empty), jnp.asarray(empty), tolerance=1.0)
        )
    )


# --- jit -----------------------------------------------------------------


def test_jit_clean():
    p = jnp.asarray(_ball((28, 28, 28), (14, 14, 14), 7))
    g = jnp.asarray(_ball((28, 28, 28), (16, 14, 14), 7))
    hd_j = jax.jit(lambda a, b: hausdorff95(a, b, spacing=(1.0, 1.0, 2.0)))
    nsd_j = jax.jit(lambda a, b: surface_dice(a, b, tolerance=2.0))
    assert _close(hd_j(p, g), hausdorff95(p, g, spacing=(1.0, 1.0, 2.0)))
    assert _close(nsd_j(p, g), surface_dice(p, g, tolerance=2.0))
