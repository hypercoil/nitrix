# -*- coding: utf-8 -*-
"""Tests for the cr / gp / mrf smooth bases (v3 §3.2).

- ``cr`` (cubic regression spline): the design reproduces ``scipy``'s **natural**
  ``CubicSpline`` interpolation at the knots, a linear function carries zero
  curvature penalty, and a GAM recovers a smooth signal.
- ``gp`` (Gaussian-process / kriging smooth): a Matern-3/2 kernel smooth recovers
  a smooth signal and round-trips through the pytree (its kernel range).
- ``mrf`` (Markov random field): the penalty *is* the combinatorial graph
  Laplacian of the region adjacency, and a GAM recovers region effects, shrinking
  neighbouring regions together.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update('jax_enable_x64', True)

from nitrix.graph import laplacian
from nitrix.stats import cr_basis, gp_basis, mrf_smooth
from nitrix.stats.basis import spline_design
from nitrix.stats.gam import gam_fit, smooth_partial_effect


def _gam_fit_curve(basis, x, truth, y):
    res = gam_fit(jnp.asarray(y[None, :]), [basis])
    eff, se = smooth_partial_effect(res, 0, basis, jnp.asarray(x))
    fit = float(res.coef[0, 0]) + np.asarray(eff[0])
    return res, fit, np.asarray(se)


# ---------------------------------------------------------------------------
# cr -- cubic regression spline
# ---------------------------------------------------------------------------


def test_cr_matches_scipy_natural_cubic_spline():
    scipy_interp = pytest.importorskip('scipy.interpolate')
    rng = np.random.default_rng(0)
    x = np.sort(rng.uniform(0.0, 1.0, 60))
    b = cr_basis(jnp.asarray(x), 9, center=False)
    knots = np.asarray(b.knots)
    beta = rng.standard_normal(9)
    cs = scipy_interp.CubicSpline(knots, beta, bc_type='natural')
    got = np.asarray(b.design) @ beta
    np.testing.assert_allclose(got, cs(x), atol=1e-9)


def test_cr_linear_has_zero_curvature_penalty():
    x = jnp.asarray(np.linspace(0, 1, 40))
    b = cr_basis(x, 8, center=False)
    knots = np.asarray(b.knots)
    beta = 2.0 + 3.0 * knots  # linear in the knot locations
    assert abs(float(beta @ np.asarray(b.penalty) @ beta)) < 1e-8


def test_cr_recovers_smooth_via_gam():
    rng = np.random.default_rng(3)
    x = np.sort(rng.uniform(0.0, 1.0, 300))
    truth = np.sin(2 * np.pi * x)
    y = truth + rng.standard_normal(300) * 0.2
    b = cr_basis(jnp.asarray(x), 15)
    res, fit, se = _gam_fit_curve(b, x, truth, y)
    interior = (x > 0.05) & (x < 0.95)
    assert float(np.sqrt(np.mean((fit[interior] - truth[interior]) ** 2))) < 0.06
    assert 2.0 < float(res.edf[0, 0]) < float(b.dim)
    assert (se > 0).all()


def test_cr_reeval_matches_construction_design():
    x = jnp.asarray(np.linspace(0.0, 1.0, 50))
    b = cr_basis(x, 10)
    np.testing.assert_allclose(
        np.asarray(spline_design(b, x)), np.asarray(b.design), atol=1e-10
    )


# ---------------------------------------------------------------------------
# gp -- Gaussian-process smooth
# ---------------------------------------------------------------------------


def test_gp_recovers_smooth_via_gam():
    rng = np.random.default_rng(4)
    x = np.sort(rng.uniform(0.0, 1.0, 300))
    truth = np.sin(2 * np.pi * x)
    y = truth + rng.standard_normal(300) * 0.2
    b = gp_basis(jnp.asarray(x), 15)
    res, fit, se = _gam_fit_curve(b, x, truth, y)
    interior = (x > 0.05) & (x < 0.95)
    assert float(np.sqrt(np.mean((fit[interior] - truth[interior]) ** 2))) < 0.08
    assert (se > 0).all()


def test_gp_pytree_roundtrip_preserves_kernel():
    x = jnp.asarray(np.linspace(0.0, 1.0, 40))
    b = gp_basis(x, 8, rho=0.3)
    leaves, treedef = jax.tree_util.tree_flatten(b)
    b2 = jax.tree_util.tree_unflatten(treedef, leaves)
    assert b2.kind == 'gp' and abs(b2.kernel_param - 0.3) < 1e-12
    np.testing.assert_allclose(
        np.asarray(spline_design(b2, x)), np.asarray(b.design), atol=1e-10
    )


# ---------------------------------------------------------------------------
# mrf -- Markov random field
# ---------------------------------------------------------------------------


def _chain_adjacency(r):
    a = np.zeros((r, r))
    for i in range(r - 1):
        a[i, i + 1] = a[i + 1, i] = 1.0
    return a


def test_mrf_penalty_is_graph_laplacian():
    r = 6
    a = _chain_adjacency(r)
    labels = np.arange(r)  # one obs per region
    b = mrf_smooth(jnp.asarray(labels), jnp.asarray(a), center=False)
    np.testing.assert_allclose(
        np.asarray(b.penalty),
        np.asarray(laplacian(jnp.asarray(a), normalisation='combinatorial')),
        atol=1e-10,
    )
    # quadratic form = sum of squared neighbour differences
    beta = np.array([0.0, 1.0, 0.0, 2.0, 0.0, 1.0])
    expect = sum((beta[i] - beta[i + 1]) ** 2 for i in range(r - 1))
    assert abs(float(beta @ np.asarray(b.penalty) @ beta) - expect) < 1e-9


def test_mrf_recovers_region_effects_via_gam():
    rng = np.random.default_rng(5)
    r, n = 8, 400
    a = _chain_adjacency(r)
    labels = rng.integers(0, r, n)
    region_eff = np.linspace(-1.0, 1.0, r)
    y = region_eff[labels] + rng.standard_normal(n) * 0.2
    b = mrf_smooth(jnp.asarray(labels), jnp.asarray(a))
    res = gam_fit(jnp.asarray(y[None, :]), [b])
    eff, _ = smooth_partial_effect(res, 0, b, jnp.arange(r))
    fitted = float(res.coef[0, 0]) + np.asarray(eff[0])
    # recovered region means track the truth (centred)
    assert np.corrcoef(fitted, region_eff)[0, 1] > 0.97


def test_mrf_smooths_toward_neighbours():
    """An unobserved interior region is interpolated from its neighbours (the
    Laplacian penalty shrinks adjacent regions together)."""
    rng = np.random.default_rng(6)
    r, n = 7, 400
    a = _chain_adjacency(r)
    region_eff = np.linspace(-1.5, 1.5, r)
    labels = rng.integers(0, r, n)
    labels = labels[labels != 3]  # leave region 3 unobserved
    y = region_eff[labels] + rng.standard_normal(labels.shape[0]) * 0.2
    b = mrf_smooth(jnp.asarray(labels), jnp.asarray(a))
    res = gam_fit(jnp.asarray(y[None, :]), [b])
    eff, _ = smooth_partial_effect(res, 0, b, jnp.arange(r))
    fitted = float(res.coef[0, 0]) + np.asarray(eff[0])
    # region 3's estimate lies between its observed neighbours 2 and 4
    lo, hi = sorted([fitted[2], fitted[4]])
    assert lo - 0.1 <= fitted[3] <= hi + 0.1


def test_mrf_rejects_nonsquare_adjacency():
    with pytest.raises(ValueError, match='square'):
        mrf_smooth(jnp.asarray([0, 1, 2]), jnp.ones((3, 4)))
