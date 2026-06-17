# -*- coding: utf-8 -*-
"""Tests for ``nitrix.stats.inference`` (permutation / TFCE / randomise).

TFCE and the cluster maps are anchored against ``scipy.ndimage``; the FDR /
Bonferroni helpers against ``statsmodels``; and the ``randomise`` driver against
``scipy`` (one- and two-sample t) plus the structural guarantees that make a
permutation test valid (identity-first floor ``p >= 1/n_perm``, FWE control
under the null, signal recovery, Freedman-Lane observed == GLM).
"""

from __future__ import annotations

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update('jax_enable_x64', True)

from nitrix.stats.glm import glm_fit, t_contrast
from nitrix.stats.inference import (
    bonferroni,
    cluster_size_map,
    fdr_bh,
    permutation_test,
    permutations,
    sign_flips,
    tfce,
)
from nitrix.stats.inference.cluster import (
    cluster_mass_map,
    supra_threshold_clusters,
)


def _has(mod: str) -> bool:
    try:
        __import__(mod)
        return True
    except ImportError:
        return False


needs_scipy = pytest.mark.skipif(not _has('scipy'), reason='scipy missing')
needs_sm = pytest.mark.skipif(not _has('statsmodels'), reason='statsmodels missing')


# ---------------------------------------------------------------------------
# Permutation operators
# ---------------------------------------------------------------------------


def test_sign_flips_properties():
    sf = sign_flips(10, 200, jax.random.PRNGKey(0))
    assert bool(jnp.all(sf[0] == 1.0))  # identity first
    assert bool(jnp.all(jnp.abs(sf) == 1.0))
    blocks = jnp.asarray([0, 0, 0, 1, 1, 2, 2, 2, 2, 2])
    sfb = sign_flips(10, 50, jax.random.PRNGKey(1), blocks=blocks)
    for p in range(50):
        for b in range(3):
            vals = np.asarray(sfb[p])[np.asarray(blocks) == b]
            assert len(set(vals.tolist())) == 1  # whole block shares sign


def test_permutations_properties():
    pm = permutations(10, 200, jax.random.PRNGKey(0))
    assert bool(jnp.all(pm[0] == jnp.arange(10)))  # identity first
    for p in range(200):
        assert sorted(np.asarray(pm[p]).tolist()) == list(range(10))
    assert not bool(jnp.all(pm[1] == jnp.arange(10)))  # non-trivial
    blocks = jnp.asarray([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
    pmb = permutations(10, 100, jax.random.PRNGKey(1), blocks=blocks)
    for p in range(100):
        assert bool(jnp.all(blocks[pmb[p]] == blocks))  # within-block


# ---------------------------------------------------------------------------
# TFCE / cluster maps vs scipy.ndimage
# ---------------------------------------------------------------------------


def _ref_tfce(pos, E, H, n_steps, structure):
    from scipy import ndimage

    mx = pos.max()
    if mx <= 0:
        return np.zeros_like(pos)
    dh = mx / n_steps
    out = np.zeros_like(pos)
    for i in range(1, n_steps + 1):
        h = i * dh
        lab, k = ndimage.label(pos > h, structure=structure)
        if k == 0:
            continue
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        out += (sizes[lab].astype(float) ** E) * (h**H) * dh
    return out


@needs_scipy
def test_tfce_matches_scipy_reference():
    from scipy import ndimage

    rng = np.random.default_rng(0)
    for ndim, shape in [(2, (20, 25)), (3, (12, 10, 11))]:
        s = rng.standard_normal(shape) * 1.5
        struct = ndimage.generate_binary_structure(ndim, 1)
        ref = _ref_tfce(np.clip(s, 0, None), 0.5, 2.0, 100, struct) + _ref_tfce(
            np.clip(-s, 0, None), 0.5, 2.0, 100, struct
        )
        got = np.asarray(
            tfce(jnp.asarray(s), n_steps=100, connectivity=1, two_sided=True)
        )
        np.testing.assert_allclose(got, ref, atol=1e-5)


@needs_scipy
def test_cluster_maps_match_scipy():
    from scipy import ndimage

    rng = np.random.default_rng(1)
    s = jnp.asarray(rng.standard_normal((24, 24)))
    thr = 0.5
    labels = supra_threshold_clusters(s, thr, connectivity=1)
    size = np.asarray(cluster_size_map(labels))
    mass = np.asarray(cluster_mass_map(labels, s, thr))

    ref_lab, _ = ndimage.label(np.asarray(s) > thr)
    sizes = np.bincount(ref_lab.ravel())
    sizes[0] = 0
    np.testing.assert_array_equal(size, sizes[ref_lab].astype(float))
    excess = np.clip(np.asarray(s) - thr, 0, None)
    masses = np.bincount(ref_lab.ravel(), weights=excess.ravel())
    masses[0] = 0
    np.testing.assert_allclose(mass, masses[ref_lab], atol=1e-6)


# ---------------------------------------------------------------------------
# Multiple-comparison corrections vs statsmodels
# ---------------------------------------------------------------------------


@needs_sm
def test_fdr_and_bonferroni_match_statsmodels():
    from statsmodels.stats.multitest import multipletests

    rng = np.random.default_rng(2)
    p = np.clip(rng.beta(0.4, 4.0, 200), 1e-6, 1.0)
    rej, padj = fdr_bh(jnp.asarray(p), alpha=0.05)
    rej_sm, padj_sm, _, _ = multipletests(p, alpha=0.05, method='fdr_bh')
    np.testing.assert_allclose(np.asarray(padj), padj_sm, atol=1e-10)
    np.testing.assert_array_equal(np.asarray(rej), rej_sm)

    rejb, padjb = bonferroni(jnp.asarray(p), alpha=0.05)
    _, padjb_sm, _, _ = multipletests(p, alpha=0.05, method='bonferroni')
    np.testing.assert_allclose(np.asarray(padjb), padjb_sm, atol=1e-10)


# ---------------------------------------------------------------------------
# randomise driver
# ---------------------------------------------------------------------------


@needs_scipy
def test_randomise_one_sample_observed_matches_scipy():
    from scipy import stats

    rng = np.random.default_rng(0)
    H, W, N = 12, 12, 20
    data = rng.standard_normal((H, W, N))
    data[4:8, 4:8, :] += 1.0
    res = permutation_test(
        jnp.asarray(data),
        jnp.ones((N, 1)),
        jnp.asarray([1.0]),
        key=jax.random.PRNGKey(0),
        n_perm=200,
        exchange='sign',
    )
    t_ref = stats.ttest_1samp(data, 0.0, axis=2).statistic
    np.testing.assert_allclose(np.asarray(res.stat), t_ref, atol=1e-10)


def test_randomise_fwe_floor_and_identity():
    """p_fwe >= 1/n_perm everywhere, and the observed enhanced max equals the
    first (identity) entry of the null distribution."""
    rng = np.random.default_rng(0)
    H, W, N, P = 12, 12, 20, 200
    data = rng.standard_normal((H, W, N))
    data[4:8, 4:8, :] += 1.0
    res = permutation_test(
        jnp.asarray(data),
        jnp.ones((N, 1)),
        jnp.asarray([1.0]),
        key=jax.random.PRNGKey(0),
        n_perm=P,
    )
    assert float(res.p_fwe.min()) >= 1.0 / P - 1e-9
    assert float(res.p_uncorrected.min()) >= 1.0 / P - 1e-9
    np.testing.assert_allclose(
        float(res.null_max[0]), float(jnp.max(res.enhanced)), atol=1e-9
    )


def test_randomise_detects_signal_and_controls_null():
    rng = np.random.default_rng(1)
    H, W, N = 14, 14, 24
    sig = rng.standard_normal((H, W, N))
    sig[5:9, 5:9, :] += 1.0
    r_sig = permutation_test(
        jnp.asarray(sig), jnp.ones((N, 1)), jnp.asarray([1.0]),
        key=jax.random.PRNGKey(0), n_perm=300,
    )
    assert float((r_sig.p_fwe[5:9, 5:9] < 0.05).mean()) > 0.5

    noise = rng.standard_normal((H, W, N))
    r_noise = permutation_test(
        jnp.asarray(noise), jnp.ones((N, 1)), jnp.asarray([1.0]),
        key=jax.random.PRNGKey(1), n_perm=300,
    )
    # family-wise: under the null at most ~alpha of *experiments* have any
    # detection; here a single map should rarely light up many voxels.
    assert float((r_noise.p_fwe < 0.05).mean()) < 0.05


@needs_scipy
def test_randomise_two_sample_permutation():
    """Two-sample test via label permutation; observed matches scipy ttest_ind."""
    from scipy import stats

    rng = np.random.default_rng(3)
    H, W, n1, n2 = 10, 10, 12, 12
    N = n1 + n2
    g = np.r_[np.zeros(n1), np.ones(n2)]
    X = np.column_stack([np.ones(N), g])  # intercept + group
    data = rng.standard_normal((H, W, N))
    data[3:6, 3:6, n1:] += 1.2  # group-2 effect
    res = permutation_test(
        jnp.asarray(data),
        jnp.asarray(X),
        jnp.asarray([0.0, 1.0]),  # group contrast
        key=jax.random.PRNGKey(0),
        n_perm=200,
        exchange='perm',
    )
    t_ref = stats.ttest_ind(
        data[..., n1:], data[..., :n1], axis=2, equal_var=True
    ).statistic
    np.testing.assert_allclose(np.asarray(res.stat), t_ref, atol=1e-9)


def test_randomise_freedman_lane_observed_matches_glm():
    """With nuisance, the observed statistic equals the parametric GLM t on the
    full design (the unpermuted Freedman-Lane fit)."""
    rng = np.random.default_rng(4)
    H, W, N = 8, 8, 30
    age = rng.standard_normal(N)
    grp = rng.standard_normal(N)
    X = np.column_stack([np.ones(N), grp, age])  # effect=grp, nuisance=age+intercept
    data = rng.standard_normal((H, W, N))
    res = permutation_test(
        jnp.asarray(data),
        jnp.asarray(X),
        jnp.asarray([0.0, 1.0, 0.0]),
        key=jax.random.PRNGKey(0),
        n_perm=50,
        exchange='perm',
        nuisance=jnp.asarray(np.column_stack([np.ones(N), age])),
    )
    glm = glm_fit(jnp.asarray(data.reshape(H * W, N)), jnp.asarray(X))
    _, _, t_glm, _ = t_contrast(glm, jnp.asarray([0.0, 1.0, 0.0]))
    np.testing.assert_allclose(
        np.asarray(res.stat).reshape(-1), np.asarray(t_glm), atol=1e-9
    )


def test_randomise_voxel_enhancement_and_cusolver_free():
    rng = np.random.default_rng(5)
    H, W, N = 10, 10, 16
    data = jnp.asarray(rng.standard_normal((H, W, N)))
    f = jax.jit(
        lambda d: permutation_test(
            d, jnp.ones((N, 1)), jnp.asarray([1.0]),
            key=jax.random.PRNGKey(0), n_perm=64, enhancement='voxel',
        ).p_fwe
    )
    hlo = f.lower(data).compile().as_text()
    targets = set(re.findall(r'custom_call_target="([^"]+)"', hlo))
    bad = [
        c for c in targets
        if any(t in c.lower() for t in ('cusolver', 'syevd', 'potrf', 'getrf', 'cholesky', 'eigh'))
    ]
    assert not bad, bad
