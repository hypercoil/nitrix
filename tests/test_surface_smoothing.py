# -*- coding: utf-8 -*-
"""Geodesic surface smoothing (geometry-suite P1.2 / GS-12).

Backward-Euler heat diffusion ``(M + tL) x = M x_0`` via ``linalg.krylov.cg``.
Anchored on the exact invariants (area-weighted mass conservation, constant
preservation, ``fwhm=0`` identity), monotone variance reduction with FWHM, and
a real fsaverage smoke test.  This is the nitrix-native geodesic smoother
(documented divergence from ``wb_command -metric-smoothing``, not parity).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from _real_meshes import fsaverage_white

from nitrix.geometry import (
    SurfaceSmoothOperator,
    surface_smooth,
    surface_smooth_apply,
    surface_smooth_operator,
)
from nitrix.sparse import Mesh, icosphere, vertex_areas


def _noise(n: int, seed: int = 0) -> jax.Array:
    return jax.random.normal(jax.random.PRNGKey(seed), (n,))


def test_mass_is_conserved() -> None:
    # Heat diffusion conserves the area-weighted integral sum_i A_i x_i.
    mesh = icosphere(3)
    area = vertex_areas(mesh)
    x0 = _noise(mesh.n_vertices)
    x = surface_smooth(mesh, x0, fwhm=0.3)
    assert np.allclose(
        float(jnp.sum(area * x)), float(jnp.sum(area * x0)), rtol=1e-4
    )


def test_constant_is_preserved() -> None:
    mesh = icosphere(3)
    const = jnp.full((mesh.n_vertices,), 2.5)
    out = surface_smooth(mesh, const, fwhm=0.5)
    assert np.allclose(np.asarray(out), 2.5, atol=1e-4)


def test_zero_fwhm_is_identity() -> None:
    mesh = icosphere(3)
    x0 = _noise(mesh.n_vertices, seed=1)
    out = surface_smooth(mesh, x0, fwhm=0.0)
    assert np.allclose(np.asarray(out), np.asarray(x0), atol=1e-5)


def test_larger_fwhm_smooths_more() -> None:
    # Monotone: a larger kernel removes more high-frequency variance.
    mesh = icosphere(4)
    x0 = _noise(mesh.n_vertices, seed=2)
    v0 = float(jnp.var(x0))
    variances = [
        float(jnp.var(surface_smooth(mesh, x0, fwhm=fw)))
        for fw in (0.05, 0.1, 0.2, 0.4)
    ]
    assert variances[0] < v0  # any smoothing reduces variance
    assert all(b <= a + 1e-7 for a, b in zip(variances, variances[1:]))


def test_batched_matches_per_field() -> None:
    # Smoothing a stack of fields == smoothing each separately.
    mesh = icosphere(3)
    fields = jnp.stack([_noise(mesh.n_vertices, s) for s in (3, 4)])  # (2, n)
    both = surface_smooth(mesh, fields, fwhm=0.2)
    sep = jnp.stack(
        [surface_smooth(mesh, fields[i], fwhm=0.2) for i in (0, 1)]
    )
    assert np.allclose(np.asarray(both), np.asarray(sep), atol=1e-4)


def test_smoothing_differentiable() -> None:
    mesh = icosphere(2)
    x0 = _noise(mesh.n_vertices, seed=5)

    def loss(v: jax.Array) -> jax.Array:
        return jnp.sum(surface_smooth(mesh, v, fwhm=0.3) ** 2)

    g = jax.grad(loss)(x0)
    assert g.shape == x0.shape
    assert np.all(np.isfinite(np.asarray(g)))


def test_real_mesh_smoothing() -> None:
    # Smooth FS curv on the real white surface: finite, mass-conserving,
    # variance-reducing (mm-scale fp32, folded geometry).
    v, f, overlays = fsaverage_white()
    if 'curv' not in overlays:
        import pytest

        pytest.skip('no curv overlay')
    mesh = Mesh(jnp.asarray(v), jnp.asarray(f))
    area = vertex_areas(mesh)
    curv = jnp.asarray(overlays['curv'])
    out = surface_smooth(mesh, curv, fwhm=5.0)  # 5 mm FWHM
    assert np.all(np.isfinite(np.asarray(out)))
    assert float(jnp.var(out)) < float(jnp.var(curv))
    assert np.allclose(
        float(jnp.sum(area * out)), float(jnp.sum(area * curv)), rtol=1e-3
    )


# --------------------------------------------------------------------------- #
# ROI / medial-wall masking (audit AI-B1)
# --------------------------------------------------------------------------- #


def test_roi_does_not_bleed_across_boundary() -> None:
    # ROI (upper hemisphere) constant 1.0; the off-ROI (medial-wall stand-in)
    # carries different fillings. The Neumann boundary makes the ROI output
    # INDEPENDENT of the off-ROI values (the tolerance-robust no-bleed proof),
    # and off-ROI vertices are returned unchanged.
    mesh = icosphere(3)
    z = np.asarray(mesh.vertices[:, 2])
    roi = jnp.asarray(z >= 0.0)
    roi_np = np.asarray(roi)
    v1 = jnp.asarray(np.where(roi_np, 1.0, 10.0).astype(np.float32))
    v2 = jnp.asarray(np.where(roi_np, 1.0, -50.0).astype(np.float32))
    o1 = np.asarray(surface_smooth(mesh, v1, fwhm=3.0, roi=roi))
    o2 = np.asarray(surface_smooth(mesh, v2, fwhm=3.0, roi=roi))
    assert np.allclose(o1[roi_np], o2[roi_np], atol=1e-5)  # no bleed
    assert np.allclose(o1[roi_np], 1.0, atol=1e-2)  # constant preserved
    assert np.allclose(o1[~roi_np], 10.0, atol=1e-3)  # off-ROI unchanged


def test_roi_precision_independent_of_offroi_magnitude() -> None:
    # A real failure mode: the medial wall can carry large out-of-distribution
    # values (unnormalised intensity, sentinel fills). Because CG shares one
    # global tolerance, a huge off-ROI value must NOT starve the ROI of
    # precision -- the ROI result must match the reference (small off-ROI fill).
    mesh = icosphere(3)
    z = np.asarray(mesh.vertices[:, 2])
    roi = jnp.asarray(z >= 0.0)
    roi_np = np.asarray(roi)
    rng = np.random.default_rng(2)
    roi_field = rng.standard_normal(mesh.n_vertices).astype(np.float32)
    small = jnp.asarray(np.where(roi_np, roi_field, 0.1).astype(np.float32))
    huge = jnp.asarray(np.where(roi_np, roi_field, 1e8).astype(np.float32))
    o_small = np.asarray(surface_smooth(mesh, small, fwhm=4.0, roi=roi))
    o_huge = np.asarray(surface_smooth(mesh, huge, fwhm=4.0, roi=roi))
    assert np.allclose(o_small[roi_np], o_huge[roi_np], atol=1e-5)


def test_roi_is_nan_safe_off_region() -> None:
    # NaN-filled masked regions are common; they must not poison the ROI solve.
    mesh = icosphere(3)
    z = np.asarray(mesh.vertices[:, 2])
    roi = jnp.asarray(z >= 0.0)
    roi_np = np.asarray(roi)
    vals = jnp.asarray(np.where(roi_np, 1.0, np.nan).astype(np.float32))
    out = np.asarray(surface_smooth(mesh, vals, fwhm=4.0, roi=roi))
    assert np.all(np.isfinite(out[roi_np]))  # ROI clean despite off-ROI NaN
    assert np.allclose(out[roi_np], 1.0, atol=1e-3)
    assert np.all(np.isnan(out[~roi_np]))  # masked region stays NaN


def test_roi_preserves_constants_and_integral_within_roi() -> None:
    mesh = icosphere(3)
    z = np.asarray(mesh.vertices[:, 2])
    roi = jnp.asarray(z >= -0.2)
    roi_np = np.asarray(roi)
    area = np.asarray(vertex_areas(mesh))
    rng = np.random.default_rng(0)
    vals = jnp.asarray(rng.standard_normal(mesh.n_vertices).astype(np.float32))
    out = np.asarray(surface_smooth(mesh, vals, fwhm=4.0, roi=roi))
    # Constant in -> constant out within ROI.
    cout = np.asarray(
        surface_smooth(mesh, jnp.ones(mesh.n_vertices), fwhm=4.0, roi=roi)
    )
    assert np.allclose(cout[roi_np], 1.0, atol=1e-4)
    # Area-weighted integral conserved over the ROI (Neumann boundary).
    lhs = float(np.sum(area[roi_np] * out[roi_np]))
    rhs = float(np.sum(area[roi_np] * np.asarray(vals)[roi_np]))
    assert np.isclose(lhs, rhs, rtol=1e-3)


def test_surface_smooth_convenience_equals_fit_apply() -> None:
    # The single-call convenience is defined as apply(values, operator(mesh)),
    # so the two paths are byte-identical.
    mesh = icosphere(3)
    v = _noise(mesh.n_vertices)
    direct = surface_smooth(mesh, v, fwhm=6.0)
    op = surface_smooth_operator(mesh)
    composed = surface_smooth_apply(v, op, fwhm=6.0)
    np.testing.assert_array_equal(np.asarray(direct), np.asarray(composed))


def test_surface_smooth_apply_is_jittable() -> None:
    # The apply half runs under jit with the operator held as a pytree arg,
    # even though the mesh-building convenience is not jittable through mesh.
    mesh = icosphere(3)
    v = _noise(mesh.n_vertices)
    op = surface_smooth_operator(mesh)

    def apply(vals: jax.Array, operator: SurfaceSmoothOperator) -> jax.Array:
        return surface_smooth_apply(vals, operator, fwhm=6.0)

    eager = np.asarray(apply(v, op))
    jitted = np.asarray(jax.jit(apply)(v, op))
    np.testing.assert_allclose(jitted, eager, atol=1e-5)


def test_surface_smooth_operator_reused_across_fwhm() -> None:
    # Building the operator once and applying at several fwhm matches building
    # it fresh for each fwhm (the amortisation is exact).
    mesh = icosphere(3)
    v = _noise(mesh.n_vertices)
    op = surface_smooth_operator(mesh)
    for fwhm in (2.0, 5.0):
        reused = np.asarray(surface_smooth_apply(v, op, fwhm=fwhm))
        fresh = np.asarray(surface_smooth(mesh, v, fwhm=fwhm))
        np.testing.assert_array_equal(reused, fresh)


def test_surface_smooth_roi_via_seam() -> None:
    # The ROI mask flows through the operator state and reproduces the
    # convenience path exactly.
    mesh = icosphere(3)
    z = mesh.vertices[:, 2]
    roi = jnp.asarray(z >= 0.0)
    roi_np = np.asarray(roi)
    v = jnp.asarray(np.where(roi_np, 1.0, 10.0).astype(np.float32))
    op = surface_smooth_operator(mesh, roi=roi)
    assert op.roi is not None
    seam = np.asarray(surface_smooth_apply(v, op, fwhm=3.0))
    conv = np.asarray(surface_smooth(mesh, v, fwhm=3.0, roi=roi))
    np.testing.assert_array_equal(seam, conv)
