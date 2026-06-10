# -*- coding: utf-8 -*-
"""V3a: transform algebra -- Lie-group Fréchet mean + geodesic interpolation.

The barycentre substrate for groupwise / template construction and motion
summary: the Fréchet (Karcher) mean of homogeneous transforms (rigid SE(n) ⊂
affine), the SVF mean, and geodesic interpolation -- plus the general
``matrix_log`` that the affine mean warranted.

The mean / geodesic use ``matrix_log`` (hence ``safe_inv``), so they are
forward / eager ops on the wedged-cuSolver dev box (jit- and grad-clean only on
a healthy GPU); these tests exercise the forward pass.
"""

from __future__ import annotations

import jax

jax.config.update('jax_enable_x64', True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from nitrix.geometry import (  # noqa: E402
    affine_exp,
    rigid_exp,
    transform_geodesic,
    transform_mean,
    velocity_mean,
)
from nitrix.linalg import matrix_exp, matrix_log  # noqa: E402


def _np(x):
    return np.asarray(x)


def _se3_algebra(omega, trans):
    """A rigid (se(3)) algebra matrix: skew rotation block + translation."""
    ox, oy, oz = omega
    skew = np.array([[0, -oz, oy], [oz, 0, -ox], [-oy, ox, 0]])
    d = np.zeros((4, 4))
    d[:3, :3] = skew
    d[:3, 3] = trans
    return jnp.asarray(d)


def test_matrix_log_inverts_matrix_exp():
    rng = np.random.RandomState(0)
    # affine-algebra generator (last row zero): log(exp(X)) == X
    x = jnp.asarray(rng.standard_normal((4, 4)) * 0.2).at[3, :].set(0.0)
    assert np.allclose(_np(matrix_log(matrix_exp(x))), _np(x), atol=1e-10)
    # rigid with a large translation: exp(log(M)) == M
    m = rigid_exp(jnp.asarray([0.2, -0.1, 0.15, 8.0, -6.0, 7.0]), ndim=3)
    assert np.allclose(_np(matrix_exp(matrix_log(m))), _np(m), atol=1e-10)


def test_transform_mean_of_identical():
    t = rigid_exp(jnp.asarray([0.1, -0.05, 0.08, 3.0, -2.0, 1.5]), ndim=3)
    assert np.allclose(_np(transform_mean(jnp.stack([t, t, t]))), _np(t), atol=1e-6)


def test_transform_mean_rigid_symmetric_recovers_centre():
    centre = rigid_exp(jnp.asarray([0.12, 0.05, -0.1, 2.0, -3.0, 4.0]), ndim=3)
    # symmetric in the true matrix chart (matrix_exp of an se(3) algebra elt)
    d = _se3_algebra([0.04, -0.03, 0.05], [1.0, 0.5, -0.8])
    stack = jnp.stack([centre @ matrix_exp(d), centre @ matrix_exp(-d)])
    mean = transform_mean(stack)
    assert np.allclose(_np(mean), _np(centre), atol=1e-5)
    # the mean of rigids is rigid (orthogonal block, det +1)
    r = _np(mean)[:3, :3]
    assert np.allclose(r @ r.T, np.eye(3), atol=1e-6)


def test_transform_mean_pure_translations_is_arithmetic():
    ts = np.array([[1.0, 2.0, -3.0], [4.0, -1.0, 0.5], [-2.0, 3.0, 1.0]])
    stack = jnp.stack(
        [rigid_exp(jnp.asarray([0, 0, 0, *t]), ndim=3) for t in ts]
    )
    mean = transform_mean(stack)
    assert np.allclose(_np(mean)[:3, :3], np.eye(3), atol=1e-6)
    assert np.allclose(_np(mean)[:3, 3], ts.mean(0), atol=1e-5)


def test_transform_mean_affine_symmetric_recovers_centre():
    rng = np.random.RandomState(1)
    centre = affine_exp(
        jnp.asarray([*(rng.standard_normal(9) * 0.15), 2.0, -1.0, 1.5]), ndim=3
    )
    d = jnp.asarray(rng.standard_normal((4, 4)) * 0.1).at[3, :].set(0.0)
    stack = jnp.stack([centre @ matrix_exp(d), centre @ matrix_exp(-d)])
    assert np.allclose(_np(transform_mean(stack)), _np(centre), atol=1e-4)


def test_transform_geodesic_endpoints_and_halfway():
    t = rigid_exp(jnp.asarray([0.3, -0.2, 0.25, 5.0, -4.0, 3.0]), ndim=3)
    eye = jnp.eye(4)
    assert np.allclose(_np(transform_geodesic(t, 0.0)), _np(eye), atol=1e-9)
    assert np.allclose(_np(transform_geodesic(t, 1.0)), _np(t), atol=1e-8)
    half = transform_geodesic(t, 0.5)
    assert np.allclose(_np(half @ half), _np(t), atol=1e-8)


def test_velocity_mean():
    rng = np.random.RandomState(2)
    v = jnp.asarray(rng.standard_normal((5, 16, 16, 2)))
    assert np.allclose(_np(velocity_mean(v)), _np(v).mean(0), atol=1e-10)
    w = jnp.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
    expect = np.tensordot(_np(w) / _np(w).sum(), _np(v), axes=(0, 0))
    assert np.allclose(_np(velocity_mean(v, weights=w)), expect, atol=1e-10)
