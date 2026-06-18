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

from nitrix.stats.glm import f_contrast, glm_fit, t_contrast
from nitrix.stats.inference import (
    bonferroni,
    cluster_size_map,
    fdr_bh,
    gpd_pvalue,
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
needs_sm = pytest.mark.skipif(
    not _has('statsmodels'), reason='statsmodels missing'
)


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
        ref = _ref_tfce(
            np.clip(s, 0, None), 0.5, 2.0, 100, struct
        ) + _ref_tfce(np.clip(-s, 0, None), 0.5, 2.0, 100, struct)
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


@needs_scipy
def test_randomise_uncorrected_p_uses_raw_statistic():
    """The uncorrected p-map is built from the raw statistic (FSL convention),
    not the TFCE-enhanced value -- so it is ~monotone in the parametric
    voxelwise p, which the enhanced (extent-mixed) map would not be."""
    from scipy import stats

    rng = np.random.default_rng(7)
    H, W, N = 14, 14, 24
    data = rng.standard_normal((H, W, N))
    data[5:9, 5:9, :] += 1.0
    res = permutation_test(
        jnp.asarray(data),
        jnp.ones((N, 1)),
        jnp.asarray([1.0]),
        key=jax.random.PRNGKey(0),
        n_perm=300,
        two_sided=True,
    )
    p_param = stats.ttest_1samp(data, 0.0, axis=2).pvalue
    rho = stats.spearmanr(
        np.asarray(res.p_uncorrected).ravel(), p_param.ravel()
    ).correlation
    assert rho > 0.95


def test_randomise_excludes_constant_voxel():
    """A constant (zero-variance) in-mask voxel must not produce an SE-floor
    spurious statistic: it is folded out of the mask (p=1, stat=0) and does not
    inflate the max-statistic null."""
    rng = np.random.default_rng(8)
    H, W, N = 12, 12, 20
    data = rng.standard_normal((H, W, N))
    data[0, 0, :] = 3.0  # constant nonzero voxel
    res = permutation_test(
        jnp.asarray(data),
        jnp.ones((N, 1)),
        jnp.asarray([1.0]),
        key=jax.random.PRNGKey(0),
        n_perm=200,
    )
    assert abs(float(res.stat[0, 0])) < 1e-9  # zeroed, not +/-inf
    assert float(res.p_fwe[0, 0]) == 1.0  # excluded
    assert bool(jnp.isfinite(res.enhanced).all())
    # the artifact must not dominate the null: observed max is from real signal,
    # not the constant voxel.
    assert float(jnp.max(res.enhanced)) < 1e6


def test_randomise_detects_signal_and_controls_null():
    rng = np.random.default_rng(1)
    H, W, N = 14, 14, 24
    sig = rng.standard_normal((H, W, N))
    sig[5:9, 5:9, :] += 1.0
    r_sig = permutation_test(
        jnp.asarray(sig),
        jnp.ones((N, 1)),
        jnp.asarray([1.0]),
        key=jax.random.PRNGKey(0),
        n_perm=300,
    )
    assert float((r_sig.p_fwe[5:9, 5:9] < 0.05).mean()) > 0.5

    noise = rng.standard_normal((H, W, N))
    r_noise = permutation_test(
        jnp.asarray(noise),
        jnp.ones((N, 1)),
        jnp.asarray([1.0]),
        key=jax.random.PRNGKey(1),
        n_perm=300,
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
    X = np.column_stack(
        [np.ones(N), grp, age]
    )  # effect=grp, nuisance=age+intercept
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


@pytest.mark.parametrize('mode', ['cluster_extent', 'cluster_mass'])
def test_randomise_cluster_enhancement(mode):
    """Cluster-extent / cluster-mass enhancement: valid FWE (floor + signal +
    null control), and the cluster-forming threshold is required."""
    rng = np.random.default_rng(0)
    H, W, N = 16, 16, 24
    data = rng.standard_normal((H, W, N))
    data[5:10, 5:10, :] += 1.0
    res = permutation_test(
        jnp.asarray(data),
        jnp.ones((N, 1)),
        jnp.asarray([1.0]),
        key=jax.random.PRNGKey(1),
        n_perm=300,
        enhancement=mode,
        cluster_thresh=2.0,
    )
    assert float(res.p_fwe.min()) >= 1.0 / 300 - 1e-9
    np.testing.assert_allclose(
        float(res.null_max[0]), float(jnp.max(res.enhanced)), atol=1e-9
    )
    assert float((res.p_fwe[5:10, 5:10] < 0.05).mean()) > 0.5

    noise = rng.standard_normal((H, W, N))
    r0 = permutation_test(
        jnp.asarray(noise),
        jnp.ones((N, 1)),
        jnp.asarray([1.0]),
        key=jax.random.PRNGKey(2),
        n_perm=300,
        enhancement=mode,
        cluster_thresh=2.0,
    )
    assert float((r0.p_fwe < 0.05).mean()) < 0.05

    # the forming threshold is mandatory for cluster modes
    with pytest.raises(ValueError):
        permutation_test(
            jnp.asarray(data),
            jnp.ones((N, 1)),
            jnp.asarray([1.0]),
            key=jax.random.PRNGKey(0),
            n_perm=10,
            enhancement=mode,
        )


def test_randomise_voxel_enhancement_and_cusolver_free():
    rng = np.random.default_rng(5)
    H, W, N = 10, 10, 16
    data = jnp.asarray(rng.standard_normal((H, W, N)))
    f = jax.jit(
        lambda d: (
            permutation_test(
                d,
                jnp.ones((N, 1)),
                jnp.asarray([1.0]),
                key=jax.random.PRNGKey(0),
                n_perm=64,
                enhancement='voxel',
            ).p_fwe
        )
    )
    hlo = f.lower(data).compile().as_text()
    targets = set(re.findall(r'custom_call_target="([^"]+)"', hlo))
    bad = [
        c
        for c in targets
        if any(
            t in c.lower()
            for t in (
                'cusolver',
                'syevd',
                'potrf',
                'getrf',
                'cholesky',
                'eigh',
            )
        )
    ]
    assert not bad, bad


# ---------------------------------------------------------------------------
# F-contrast (§3.2)
# ---------------------------------------------------------------------------


def _two_effect_design(N, rng):
    """Intercept + two regressors design (p=3)."""
    x1 = rng.standard_normal(N)
    x2 = rng.standard_normal(N)
    return np.column_stack([np.ones(N), x1, x2])


def test_randomise_f_observed_matches_glm_f_contrast():
    """The observed (identity) F-map equals glm.f_contrast exactly -- same
    quadratic form, dof, and dispersion convention."""
    rng = np.random.default_rng(11)
    H, W, N = 8, 8, 30
    X = _two_effect_design(N, rng)
    data = rng.standard_normal((H, W, N))
    data[2:6, 2:6, :] += 1.5 * X[:, 1]  # an x1 effect in a patch
    C = jnp.asarray([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])  # joint x1, x2

    res = permutation_test(
        jnp.asarray(data),
        jnp.asarray(X),
        C,
        key=jax.random.PRNGKey(0),
        n_perm=20,
        exchange='perm',
        enhancement='voxel',
    )
    result = glm_fit(jnp.asarray(data.reshape(H * W, N)), jnp.asarray(X))
    f_glm, _, df1, df2 = f_contrast(result, C)
    assert df1 == 2.0 and df2 == float(N - 3)
    np.testing.assert_allclose(
        np.asarray(res.stat).reshape(H * W), np.asarray(f_glm), atol=1e-8
    )
    # F is non-negative; the enhanced (one-sided voxel) map equals the F itself.
    assert float(jnp.min(res.stat)) >= 0.0
    np.testing.assert_allclose(
        np.asarray(res.enhanced), np.asarray(res.stat), atol=1e-10
    )


def test_randomise_f_contrast_fwe_floor_and_signal():
    """F-contrast randomise is a valid FWE test (identity floor + signal
    recovery + null control)."""
    rng = np.random.default_rng(12)
    H, W, N, P = 12, 12, 28, 200
    X = _two_effect_design(N, rng)
    data = rng.standard_normal((H, W, N))
    data[4:8, 4:8, :] += 1.2 * X[:, 1] - 1.0 * X[:, 2]
    C = jnp.asarray([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    res = permutation_test(
        jnp.asarray(data),
        jnp.asarray(X),
        C,
        key=jax.random.PRNGKey(3),
        n_perm=P,
        exchange='perm',
        enhancement='cluster_extent',
        cluster_thresh=6.0,
    )
    assert float(res.p_fwe.min()) >= 1.0 / P - 1e-9
    np.testing.assert_allclose(
        float(res.null_max[0]), float(jnp.max(res.enhanced)), atol=1e-9
    )
    assert float((res.p_fwe[4:8, 4:8] < 0.05).mean()) > 0.5

    noise = rng.standard_normal((H, W, N))
    r0 = permutation_test(
        jnp.asarray(noise),
        jnp.asarray(X),
        C,
        key=jax.random.PRNGKey(4),
        n_perm=P,
        exchange='perm',
        enhancement='cluster_extent',
        cluster_thresh=6.0,
    )
    assert float((r0.p_fwe < 0.05).mean()) < 0.05


# ---------------------------------------------------------------------------
# GPD tail-accelerated p-values (§3.3)
# ---------------------------------------------------------------------------


def test_gpd_pvalue_recovers_exponential_tail():
    """For an Exp(1) null (a GPD with shape xi=0), the GPD survival estimate
    tracks exp(-T) into the tail -- and resolves below the empirical 1/n floor
    (the whole point).  The exceedances above the threshold are themselves
    Exp(1) (memorylessness), so the moment fit recovers xi~0, sigma~1 and the
    estimate is exact in expectation; finite-sample MoM noise widens as the
    query extrapolates past the sample max (~8.5 here)."""
    rng = np.random.default_rng(20)
    null = jnp.asarray(rng.exponential(1.0, size=2000))
    # Moderate tail (near / just past the sample max): accurate to a factor ~3.
    for T in (6.0, 7.0):
        p = float(gpd_pvalue(T, null, n_exceedances=500))
        assert 0.0 < p < 1.0
        assert abs(np.log(p) - (-T)) < np.log(3.0)
    # Deep extrapolation: the right order of magnitude (factor ~5) and -- the
    # point of the method -- a smooth value strictly below the empirical 1/n
    # floor where the count estimate would be 0.
    p_deep = float(gpd_pvalue(10.0, null, n_exceedances=500))
    assert abs(np.log(p_deep) - (-10.0)) < np.log(5.0)
    assert 0.0 < p_deep < 1.0 / 2000


def test_gpd_pvalue_body_is_empirical():
    """Below the tail threshold the GPD p-value is the empirical fraction."""
    rng = np.random.default_rng(21)
    null = jnp.asarray(rng.standard_normal(1000))
    T = 0.0  # near the median -> well inside the body
    p = float(gpd_pvalue(T, null, n_exceedances=250))
    emp = float(jnp.mean(null >= T))
    np.testing.assert_allclose(p, emp, atol=1e-12)


def test_randomise_gpd_resolves_below_floor():
    """pvalue_method='gpd' yields a smooth FWE map that drops below 1/n_perm at
    the peak, where the empirical map is floored; the uncorrected map and the
    null distribution are unchanged."""
    rng = np.random.default_rng(22)
    H, W, N, P = 14, 14, 24, 200
    data = rng.standard_normal((H, W, N))
    data[5:9, 5:9, :] += 1.5
    common = dict(
        design=jnp.ones((N, 1)),
        contrast=jnp.asarray([1.0]),
        key=jax.random.PRNGKey(0),
        n_perm=P,
        enhancement='voxel',
    )
    emp = permutation_test(jnp.asarray(data), **common)
    gpd = permutation_test(
        jnp.asarray(data),
        **common,
        pvalue_method='gpd',
        gpd_n_exceedances=100,
    )
    # null distribution + observed are identical (only the p-map mapping differs)
    np.testing.assert_allclose(
        np.asarray(emp.null_max), np.asarray(gpd.null_max), atol=1e-10
    )
    np.testing.assert_allclose(
        np.asarray(emp.enhanced), np.asarray(gpd.enhanced), atol=1e-10
    )
    # empirical is floored at 1/P at the peak; GPD resolves strictly below it.
    assert float(emp.p_fwe.min()) >= 1.0 / P - 1e-9
    assert 0.0 < float(gpd.p_fwe.min()) < 1.0 / P
    # still a valid p-map
    assert float(gpd.p_fwe.min()) >= 0.0 and float(gpd.p_fwe.max()) <= 1.0
    # uncorrected map untouched by the FWE method
    np.testing.assert_allclose(
        np.asarray(emp.p_uncorrected),
        np.asarray(gpd.p_uncorrected),
        atol=1e-12,
    )
