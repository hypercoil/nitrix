# -*- coding: utf-8 -*-
"""Tests for the spin-test spatial null.

Covers the ``geometry`` generators (uniform-SO(3) ``random_rotation``;
``spin_surrogates`` rotate-and-reassign) and the ``stats.inference.spin_test``
(observed Pearson correlation, rotated null, add-one permutation p-value).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update('jax_enable_x64', True)

from nitrix.geometry import (  # noqa: E402
    parcel_centroids,
    random_rotation,
    spin_surrogates,
)
from nitrix.stats.inference import spin_test  # noqa: E402


def _sphere(v=200, seed=2):
    coords = jax.random.normal(jax.random.key(seed), (v, 3))
    return coords / jnp.linalg.norm(coords, axis=-1, keepdims=True)


def test_random_rotation_is_proper_orthogonal():
    R = random_rotation(jax.random.key(0), 200)
    ident = jnp.einsum('nij,nkj->nik', R, R)
    assert float(jnp.max(jnp.abs(ident - jnp.eye(3)))) < 1e-10
    np.testing.assert_allclose(jnp.linalg.det(R), 1.0, atol=1e-9)


def test_random_rotation_is_uniform_on_the_group():
    """The Haar mean rotation is the zero matrix (uniform, not Euler-biased)."""
    mean = random_rotation(jax.random.key(1), 40000).mean(0)
    assert float(jnp.max(jnp.abs(mean))) < 0.05


def test_random_rotation_single_shape():
    assert random_rotation(jax.random.key(0)).shape == (3, 3)


def test_spin_surrogates_identity_is_noop():
    coords = _sphere()
    x = jax.random.normal(jax.random.key(3), (coords.shape[0],))
    surr = spin_surrogates(coords, x, jnp.eye(3)[None])
    np.testing.assert_allclose(surr[0], x, atol=1e-10)


def test_spin_surrogates_are_a_resampling_of_x():
    coords = _sphere()
    x = jax.random.normal(jax.random.key(3), (coords.shape[0],))
    surr = spin_surrogates(coords, x, random_rotation(jax.random.key(0), 10))
    assert surr.shape == (10, coords.shape[0])
    assert np.all(np.isin(np.asarray(surr).round(9), np.asarray(x).round(9)))


def test_spin_test_self_correspondence_is_significant():
    """A map vs itself: r == 1, and spun surrogates decorrelate, so p is tiny."""
    coords = _sphere()
    x = jax.random.normal(jax.random.key(3), (coords.shape[0],))
    res = spin_test(x, x, coords, key=jax.random.key(0), n_spin=500)
    assert float(res.statistic) == 1.0 or np.isclose(float(res.statistic), 1.0)
    assert float(res.pvalue) < 0.05
    assert res.null_distribution.shape == (500,)


def test_spin_test_pvalue_in_valid_range_and_stat_is_pearson():
    coords = _sphere()
    x = jax.random.normal(jax.random.key(3), (coords.shape[0],))
    y = jax.random.normal(jax.random.key(7), (coords.shape[0],))
    res = spin_test(x, y, coords, key=jax.random.key(0), n_spin=500)
    manual_r = np.corrcoef(np.asarray(x), np.asarray(y))[0, 1]
    np.testing.assert_allclose(float(res.statistic), manual_r, atol=1e-10)
    assert 1.0 / 501.0 <= float(res.pvalue) <= 1.0


def test_spin_test_matches_manual_composition():
    """spin_test's null is exactly pearson over the spin surrogates."""
    coords = _sphere()
    x = jax.random.normal(jax.random.key(3), (coords.shape[0],))
    y = jax.random.normal(jax.random.key(7), (coords.shape[0],))
    key = jax.random.key(5)
    res = spin_test(x, y, coords, key=key, n_spin=100)
    xs = spin_surrogates(coords, x, random_rotation(key, 100))
    xm = xs - xs.mean(-1, keepdims=True)
    ym = y - y.mean()
    null = (xm * ym).sum(-1) / jnp.sqrt((xm * xm).sum(-1) * (ym * ym).sum())
    np.testing.assert_allclose(res.null_distribution, null, atol=1e-12)


def test_spin_test_jit():
    coords = _sphere()
    x = jax.random.normal(jax.random.key(3), (coords.shape[0],))
    y = jax.random.normal(jax.random.key(7), (coords.shape[0],))
    p = jax.jit(
        lambda x, y, c: (
            spin_test(x, y, c, key=jax.random.key(0), n_spin=200).pvalue
        )
    )(x, y, coords)
    assert bool(jnp.isfinite(p))


# --- refinements: per-hemisphere rotation + medial-wall (NaN) handling -------


def _bihemi(nlh=60, nrh=60):
    def unit(n, seed):
        c = np.random.default_rng(seed).standard_normal((n, 3))
        return c / np.linalg.norm(c, axis=1, keepdims=True)

    coords = jnp.asarray(np.vstack([unit(nlh, 1), unit(nrh, 2)]))
    hemi = jnp.asarray(np.array([0] * nlh + [1] * nrh))
    return coords, hemi


def test_spin_per_hemisphere_reassigns_within_hemisphere():
    """With ``hemisphere``, each vertex takes a value from its own hemisphere."""
    coords, hemi = _bihemi()
    x = jnp.asarray(np.random.default_rng(0).standard_normal(120))
    surr = spin_surrogates(
        coords, x, random_rotation(jax.random.key(0), 8), hemisphere=hemi
    )
    lh = set(np.asarray(x)[:60].round(9))
    rh = set(np.asarray(x)[60:].round(9))
    for k in range(8):
        assert set(np.asarray(surr[k])[:60].round(9)) <= lh
        assert set(np.asarray(surr[k])[60:].round(9)) <= rh


def test_spin_test_medial_wall_excludes_nan_pairwise():
    """Medial-wall (NaN) vertices drop from the observed and the null."""
    coords, hemi = _bihemi()
    rng = np.random.default_rng(0)
    x = rng.standard_normal(120)
    y = rng.standard_normal(120)
    mw = np.zeros(120, bool)
    mw[[5, 10, 65, 70]] = True
    x[mw] = np.nan
    y[mw] = np.nan
    res = spin_test(
        jnp.asarray(x),
        jnp.asarray(y),
        coords,
        key=jax.random.key(1),
        n_spin=200,
        hemisphere=hemi,
    )
    manual = np.corrcoef(x[~mw], y[~mw])[0, 1]
    np.testing.assert_allclose(float(res.statistic), manual, atol=1e-9)
    assert bool(jnp.all(jnp.isfinite(res.null_distribution)))
    assert 1.0 / 201.0 <= float(res.pvalue) <= 1.0


def test_spin_test_medial_wall_self_is_significant():
    coords, hemi = _bihemi()
    x = np.random.default_rng(0).standard_normal(120)
    x[[5, 10, 65, 70]] = np.nan
    res = spin_test(
        jnp.asarray(x),
        jnp.asarray(x),
        coords,
        key=jax.random.key(1),
        n_spin=300,
        hemisphere=hemi,
    )
    assert np.isclose(float(res.statistic), 1.0)
    assert float(res.pvalue) < 0.05


# --- Vasa / Hungarian bijective assignment ----------------------------------


def test_spin_bijective_is_exact_permutation():
    """assignment='bijective' -> each surrogate is a permutation of x."""
    coords = _sphere(v=80)
    x = jax.random.normal(jax.random.key(3), (80,))
    surr = spin_surrogates(
        coords,
        x,
        random_rotation(jax.random.key(0), 6),
        assignment='bijective',
    )
    x_sorted = np.sort(np.asarray(x))
    for k in range(6):
        assert len(np.unique(np.asarray(surr[k]))) == 80  # all distinct
        np.testing.assert_allclose(
            np.sort(np.asarray(surr[k])), x_sorted, atol=1e-12
        )


def test_spin_bijective_within_hemisphere_permutation():
    coords, hemi = _bihemi(nlh=40, nrh=40)
    x = jnp.asarray(np.random.default_rng(0).standard_normal(80))
    surr = spin_surrogates(
        coords,
        x,
        random_rotation(jax.random.key(0), 6),
        hemisphere=hemi,
        assignment='bijective',
    )
    lh = np.sort(np.asarray(x)[:40])
    rh = np.sort(np.asarray(x)[40:])
    for k in range(6):
        np.testing.assert_allclose(
            np.sort(np.asarray(surr[k])[:40]), lh, atol=1e-12
        )
        np.testing.assert_allclose(
            np.sort(np.asarray(surr[k])[40:]), rh, atol=1e-12
        )


def test_spin_test_bijective_self_is_significant():
    coords = _sphere(v=80)
    x = jax.random.normal(jax.random.key(3), (80,))
    res = spin_test(
        x, x, coords, key=jax.random.key(1), n_spin=200, assignment='bijective'
    )
    assert np.isclose(float(res.statistic), 1.0)
    assert float(res.pvalue) < 0.05


def test_spin_surrogates_invalid_assignment_raises():
    coords = _sphere(v=40)
    x = jax.random.normal(jax.random.key(3), (40,))
    with pytest.raises(ValueError, match='assignment'):
        spin_surrogates(coords, x, jnp.eye(3)[None], assignment='bogus')


# --- parcel-level spin ------------------------------------------------------


def _parcellation(n_vert=600, n_parcels=30, seed=0):
    rng = np.random.default_rng(seed)
    coords = rng.standard_normal((n_vert, 3))
    coords = coords / np.linalg.norm(coords, axis=1, keepdims=True)
    parc = np.repeat(np.arange(n_parcels), n_vert // n_parcels)
    return jnp.asarray(coords), jnp.asarray(parc), n_parcels


def test_parcel_centroids():
    coords, parc, p = _parcellation()
    cent = parcel_centroids(coords, parc, n_parcels=p)
    assert cent.shape == (p, 3)
    np.testing.assert_allclose(
        np.linalg.norm(np.asarray(cent), axis=1), 1.0, atol=1e-10
    )
    # centroid == unit-normalised mean of the parcel's vertices
    m = np.asarray(coords)[:20].mean(0)
    m = m / np.linalg.norm(m)
    np.testing.assert_allclose(np.asarray(cent[0]), m, atol=1e-10)
    # n_parcels=None infers P eagerly
    assert parcel_centroids(coords, parc).shape == (p, 3)


def test_parcel_level_spin_test():
    coords, parc, p = _parcellation()
    cent = parcel_centroids(coords, parc, n_parcels=p)
    px = jnp.asarray(np.random.default_rng(1).standard_normal(p))
    res = spin_test(px, px, cent, key=jax.random.key(1), n_spin=500)
    assert np.isclose(float(res.statistic), 1.0)
    assert float(res.pvalue) < 0.05
    assert res.null_distribution.shape == (500,)
