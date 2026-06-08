# -*- coding: utf-8 -*-
"""Tests for the R0 registration substrate.

Coverage:

- **linalg.solve / cho_solve**: agreement with ``numpy.linalg.solve`` on
  an SPD system; matrix and batched right-hand sides; the ``l2`` ridge;
  reverse-mode differentiability (finite-difference parity) -- the
  cuSolver-robust fallback path is exercised implicitly (this box's L4
  has a dead cuSolver handle pool).
- **geometry.spatial_gradient**: exact recovery of the slope of an
  affine intensity ramp (interior) for ``central`` / ``sobel`` /
  ``scharr``; per-axis ``spacing``; output shape; differentiability.
- **geometry.gaussian_pyramid / downsample / upsample**: level count and
  shapes; constant-preservation; round-trip shape.
- **metrics**: ``ssd`` (zero iff identical), ``ncc`` (+1 / -1 / ~0),
  ``lncc`` (~1 identical, lower when mis-registered), ``joint_histogram``
  (normalised; diagonal for identical), ``mutual_information`` (identical
  > independent; NMI range), ``correlation_ratio`` (~1 under a functional
  relationship, ~0 under independence).  Finite-difference gradient
  checks for SSD / LNCC / MI / CR -- the G-R0 gate.
"""

from __future__ import annotations

import jax

jax.config.update('jax_enable_x64', True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from nitrix.geometry import (  # noqa: E402
    downsample,
    gaussian_pyramid,
    spatial_gradient,
    upsample,
)
from nitrix.linalg import cho_solve, solve  # noqa: E402
from nitrix.metrics import (  # noqa: E402
    correlation_ratio,
    joint_histogram,
    lncc,
    mutual_information,
    ncc,
    ssd,
)


def _fd_grad(f, x, *, eps=1e-6, n_dirs=4, seed=0):
    """Directional finite-difference check of ``jax.grad(f)`` at ``x``."""
    g = np.asarray(jax.grad(f)(x))
    rng = np.random.RandomState(seed)
    x_np = np.asarray(x)
    for _ in range(n_dirs):
        d = rng.standard_normal(x_np.shape)
        d /= np.linalg.norm(d)
        fp = float(f(jnp.asarray(x_np + eps * d)))
        fm = float(f(jnp.asarray(x_np - eps * d)))
        fd = (fp - fm) / (2 * eps)
        ana = float(np.sum(g * d))
        assert np.isclose(fd, ana, rtol=2e-4, atol=2e-5), (fd, ana)


# ---------------------------------------------------------------------------
# linalg.solve / cho_solve
# ---------------------------------------------------------------------------


def test_solve_and_cho_solve_match_numpy():
    a_np = np.array([[4.0, 1.0, 0.0], [1.0, 3.0, 0.5], [0.0, 0.5, 2.0]])
    b_np = np.array([1.0, 2.0, -1.0])
    a = jnp.asarray(a_np)
    b = jnp.asarray(b_np)
    ref = np.linalg.solve(a_np, b_np)
    assert np.allclose(np.asarray(solve(a, b)), ref, atol=1e-10)
    assert np.allclose(np.asarray(cho_solve(a, b)), ref, atol=1e-10)


def test_cho_solve_matrix_rhs_and_ridge():
    a_np = np.array([[4.0, 1.0], [1.0, 3.0]])
    b_np = np.array([[1.0, 0.0], [0.0, 1.0]])  # solve for the inverse
    a = jnp.asarray(a_np)
    b = jnp.asarray(b_np)
    assert np.allclose(np.asarray(cho_solve(a, b)), np.linalg.inv(a_np), atol=1e-10)
    # Ridge: (A + l2 I) x = b.
    l2 = 0.5
    ref = np.linalg.solve(a_np + l2 * np.eye(2), b_np)
    assert np.allclose(np.asarray(cho_solve(a, b, l2=l2)), ref, atol=1e-10)


def test_cho_solve_batched():
    a0 = np.array([[4.0, 1.0], [1.0, 3.0]])
    a1 = np.array([[2.0, 0.0], [0.0, 5.0]])
    b0 = np.array([1.0, 2.0])
    b1 = np.array([-1.0, 1.0])
    a = jnp.asarray(np.stack([a0, a1]))
    b = jnp.asarray(np.stack([b0, b1]))
    out = np.asarray(cho_solve(a, b))
    assert out.shape == (2, 2)
    assert np.allclose(out[0], np.linalg.solve(a0, b0), atol=1e-10)
    assert np.allclose(out[1], np.linalg.solve(a1, b1), atol=1e-10)


def test_cho_solve_differentiable():
    def f(p):
        m = jnp.array([[p, 0.5], [0.5, 2.0]])
        return cho_solve(m, jnp.array([1.0, 1.0])).sum()

    _fd_grad(f, jnp.asarray(3.0))


# ---------------------------------------------------------------------------
# geometry.spatial_gradient
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('scheme', ['central', 'sobel', 'scharr'])
def test_spatial_gradient_recovers_affine_slope(scheme):
    a, b = 0.7, -1.3
    ii, jj = np.meshgrid(np.arange(9), np.arange(11), indexing='ij')
    img = jnp.asarray(a * ii + b * jj)
    g = spatial_gradient(img, scheme=scheme)
    assert g.shape == (9, 11, 2)
    # Interior is exact for an affine ramp (boundary uses the
    # half-slope voxelmorph convention, excluded here).
    gi = np.asarray(g)[1:-1, 1:-1]
    assert np.allclose(gi[..., 0], a, atol=1e-9)
    assert np.allclose(gi[..., 1], b, atol=1e-9)


def test_spatial_gradient_spacing_scales():
    a = 2.0
    x = np.arange(20, dtype=float)
    img = jnp.asarray(a * x)
    g1 = spatial_gradient(img, spacing=1.0)
    g2 = spatial_gradient(img, spacing=2.0)
    # doubling the spacing halves the estimated derivative.
    assert np.allclose(np.asarray(g1)[1:-1, 0], a, atol=1e-9)
    assert np.allclose(np.asarray(g2)[1:-1, 0], a / 2.0, atol=1e-9)


def test_spatial_gradient_differentiable():
    rng = np.random.RandomState(0)
    img = jnp.asarray(rng.rand(8, 8))

    def f(m):
        return (spatial_gradient(m) ** 2).sum()

    _fd_grad(f, img)


# ---------------------------------------------------------------------------
# geometry pyramid
# ---------------------------------------------------------------------------


def test_gaussian_pyramid_shapes_and_levels():
    img = jnp.asarray(np.random.RandomState(0).rand(16, 16, 1))
    pyr = gaussian_pyramid(img, levels=3)
    assert len(pyr) == 3
    assert pyr[0].shape == (16, 16, 1)  # finest first
    assert pyr[1].shape == (8, 8, 1)
    assert pyr[2].shape == (4, 4, 1)


def test_downsample_preserves_constant():
    img = jnp.full((16, 16, 2), 3.5)
    out = downsample(img, factor=2)
    assert out.shape == (8, 8, 2)
    assert np.allclose(np.asarray(out), 3.5, atol=1e-5)


def test_upsample_shape():
    img = jnp.asarray(np.random.RandomState(1).rand(4, 4, 1))
    out = upsample(img, (16, 16))
    assert out.shape == (16, 16, 1)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def test_ssd_zero_iff_identical():
    rng = np.random.RandomState(0)
    x = jnp.asarray(rng.rand(10, 10))
    assert float(ssd(x, x)) == pytest.approx(0.0, abs=1e-12)
    assert float(ssd(x, x + 0.1)) > 0.0


def test_ncc_bounds():
    rng = np.random.RandomState(0)
    x = jnp.asarray(rng.rand(20, 20))
    assert float(ncc(x, x)) == pytest.approx(1.0, abs=1e-6)
    assert float(ncc(x, 2.0 * x + 5.0)) == pytest.approx(1.0, abs=1e-6)
    assert float(ncc(x, -x)) == pytest.approx(-1.0, abs=1e-6)
    y = jnp.asarray(np.random.RandomState(99).rand(20, 20))
    assert abs(float(ncc(x, y))) < 0.3


def test_lncc_identical_and_misregistered():
    rng = np.random.RandomState(0)
    base = rng.rand(24, 24)
    x = jnp.asarray(base)
    assert float(lncc(x, x, radius=3)) == pytest.approx(1.0, abs=1e-3)
    shifted = jnp.asarray(np.roll(base, 5, axis=0))
    assert float(lncc(x, shifted, radius=3)) < float(lncc(x, x, radius=3))


def test_joint_histogram_normalised_and_diagonal():
    rng = np.random.RandomState(0)
    x = jnp.asarray(rng.rand(40, 40))
    p = joint_histogram(x, x, bins=16, range_moving=(0.0, 1.0), range_fixed=(0.0, 1.0))
    assert float(p.sum()) == pytest.approx(1.0, abs=1e-6)
    p_np = np.asarray(p)
    # Identical images -> for each voxel lower_m == lower_f and the soft
    # weights coincide, so *all* mass lands in the tridiagonal band
    # |i - j| <= 1.  The exact diagonal carries E[(1-f)^2 + f^2] = 2/3
    # of it under linear (bilinear) Parzen binning.
    band = np.where(np.abs(np.subtract.outer(np.arange(16), np.arange(16))) <= 1, p_np, 0.0)
    assert float(band.sum()) == pytest.approx(1.0, abs=1e-6)
    assert np.trace(p_np) == pytest.approx(2.0 / 3.0, abs=0.05)


def test_mutual_information_identical_beats_independent():
    rng = np.random.RandomState(0)
    x = jnp.asarray(rng.rand(48, 48))
    y = jnp.asarray(np.random.RandomState(1).rand(48, 48))
    rng_kw = dict(bins=24, range_moving=(0.0, 1.0), range_fixed=(0.0, 1.0))
    mi_same = float(mutual_information(x, x, **rng_kw))
    mi_indep = float(mutual_information(x, y, **rng_kw))
    assert mi_same > mi_indep
    nmi = float(mutual_information(x, x, normalized=True, **rng_kw))
    assert 1.0 <= nmi <= 2.0 + 1e-6


def test_correlation_ratio_functional_vs_independent():
    rng = np.random.RandomState(0)
    x = jnp.asarray(rng.rand(64, 64))
    # y is a deterministic (non-affine) function of x -> high CR.
    y = jnp.sin(3.0 * x)
    cr_func = float(
        correlation_ratio(y, x, bins=32, range_fixed=(0.0, 1.0))
    )
    z = jnp.asarray(np.random.RandomState(7).rand(64, 64))
    cr_indep = float(
        correlation_ratio(z, x, bins=32, range_fixed=(0.0, 1.0))
    )
    assert cr_func > 0.9
    assert cr_indep < 0.2


@pytest.mark.parametrize('name', ['ssd', 'lncc', 'mi', 'cr'])
def test_metric_gradients_finite_difference(name):
    """G-R0 gate: analytic metric gradients match finite differences."""
    rng = np.random.RandomState(0)
    x = jnp.asarray(rng.rand(14, 14))
    y = jnp.asarray(np.random.RandomState(1).rand(14, 14))
    if name == 'ssd':
        f = lambda m: ssd(m, y)
    elif name == 'lncc':
        f = lambda m: lncc(m, y, radius=2)
    elif name == 'mi':
        f = lambda m: mutual_information(
            m, y, bins=14, range_moving=(0.0, 1.0), range_fixed=(0.0, 1.0)
        )
    else:
        f = lambda m: correlation_ratio(m, y, bins=14, range_fixed=(0.0, 1.0))
    _fd_grad(f, x, eps=1e-5)
