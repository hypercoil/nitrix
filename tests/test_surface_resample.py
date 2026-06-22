# -*- coding: utf-8 -*-
"""Surface resampling between registered spheres (geometry-suite P4.1 / 12.15).

``surface_resample`` is the ``wb_command -metric-resample`` analogue: carry a
per-vertex field between two tessellations of the same registered sphere via a
host-built barycentric ``ELL`` applied through the differentiable apply-seam.

Two methods, two guarantees (the Workbench dichotomy):

- ``'barycentric'`` -- row-stochastic, **constants preserved exactly**.
- ``'adap_bary_area'`` -- **area-weighted integral conserved exactly** on
  down-sampling (constants only approximate; the documented divergence).

Anchored on the analytic icosphere (identity, integral conservation, constant
behaviour) and exercised on real fsaverage spheres at two resolutions.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _real_meshes import fsaverage_surface

from nitrix.geometry import surface_resample
from nitrix.sparse import ELL, apply_operator, icosphere, vertex_areas


def test_identity_when_source_equals_target() -> None:
    m = icosphere(3)
    vals = jnp.asarray(m.vertices[:, 2])
    for method in ('barycentric', 'adap_bary_area'):
        op, out = surface_resample(m, vals, m, method=method)
        assert isinstance(op, ELL)
        assert np.allclose(np.asarray(out), np.asarray(vals), atol=1e-5)


def test_barycentric_preserves_constants_exactly() -> None:
    src = icosphere(4)
    tgt = icosphere(2)
    const = jnp.full((src.n_vertices,), 2.5)
    _, out = surface_resample(src, const, tgt, method='barycentric')
    assert np.allclose(np.asarray(out), 2.5, atol=1e-5)


def test_adap_bary_area_conserves_integral_on_downsample() -> None:
    # Down-sample a non-constant field; the area-weighted integral is invariant.
    src = icosphere(4)  # 2562 vertices
    tgt = icosphere(2)  # 162 vertices
    vals = jnp.asarray(src.vertices[:, 0] ** 2 - 0.3 * src.vertices[:, 1])
    sa = np.asarray(vertex_areas(src))
    ta = np.asarray(vertex_areas(tgt))
    _, out = surface_resample(src, vals, tgt, method='adap_bary_area')
    lhs = float(np.sum(ta * np.asarray(out)))
    rhs = float(np.sum(sa * np.asarray(vals)))
    assert abs(lhs - rhs) <= 1e-4 * abs(rhs)


def test_barycentric_roundtrip_bounded() -> None:
    # Down then up a smooth field (the z-harmonic); barycentric interpolation
    # error is bounded by the coarse-level resolution.
    fine = icosphere(4)
    coarse = icosphere(2)
    z = jnp.asarray(fine.vertices[:, 2])
    _, down = surface_resample(fine, z, coarse, method='barycentric')
    _, back = surface_resample(coarse, down, fine, method='barycentric')
    err = np.abs(np.asarray(back) - np.asarray(z))
    assert err.max() < 0.1
    assert err.std() < 0.03


def test_operator_reuse_matches_apply() -> None:
    # The returned operator reproduces the resampled field via the apply-seam.
    src = icosphere(3)
    tgt = icosphere(2)
    vals = jnp.asarray(src.vertices[:, 0])
    op, out = surface_resample(src, vals, tgt, method='barycentric')
    reapplied = apply_operator(op, vals[:, None])[:, 0]
    assert np.allclose(np.asarray(out), np.asarray(reapplied), atol=1e-6)
    assert op.indices.shape[0] == tgt.n_vertices


def test_multichannel_field() -> None:
    src = icosphere(3)
    tgt = icosphere(2)
    vals = jnp.asarray(src.vertices)  # (n, 3) -- treat xyz as 3 channels
    op, out = surface_resample(src, vals, tgt, method='barycentric')
    assert out.shape == (tgt.n_vertices, 3)
    # Each channel resamples like the scalar path.
    _, out0 = surface_resample(src, vals[:, 0], tgt, method='barycentric')
    assert np.allclose(np.asarray(out[:, 0]), np.asarray(out0), atol=1e-6)


def test_bad_method_raises() -> None:
    m = icosphere(2)
    with pytest.raises(ValueError, match="'adap_bary_area' or 'barycentric'"):
        surface_resample(m, jnp.ones(m.n_vertices), m, method='nope')


def test_differentiable_through_apply() -> None:
    src = icosphere(3)
    tgt = icosphere(2)
    op, _ = surface_resample(
        src, jnp.zeros(src.n_vertices), tgt, method='barycentric'
    )

    def loss(vals: jax.Array) -> jax.Array:
        return jnp.sum(apply_operator(op, vals[:, None])[:, 0] ** 2)

    g = jax.grad(loss)(jnp.asarray(src.vertices[:, 2]))
    assert g.shape == (src.n_vertices,)
    assert np.all(np.isfinite(np.asarray(g)))


# --------------------------------------------------------------------------- #
# Real fsaverage spheres (two resolutions, same registered frame)
# --------------------------------------------------------------------------- #


def _fsaverage_sphere(mesh: str):
    from nitrix.sparse import Mesh

    v, f = fsaverage_surface('sphere', 'left', mesh)
    return Mesh(jnp.asarray(v), jnp.asarray(f))


def test_real_fsaverage_roundtrip_bounded() -> None:
    # fsaverage5 (10242) <-> fsaverage4 (2562): a smooth field round-trips with
    # bounded error through the barycentric resampler.
    fs5 = _fsaverage_sphere('fsaverage5')
    fs4 = _fsaverage_sphere('fsaverage4')
    field = jnp.asarray(fs5.vertices[:, 2] / jnp.linalg.norm(fs5.vertices[0]))
    _, down = surface_resample(fs5, field, fs4, method='barycentric')
    _, back = surface_resample(fs4, down, fs5, method='barycentric')
    err = np.abs(np.asarray(back) - np.asarray(field))
    assert np.all(np.isfinite(err))
    assert err.std() < 0.05


def test_real_fsaverage_adap_conserves_integral() -> None:
    # Down-sampling a real cross-resolution pair conserves the area-weighted
    # integral exactly.
    fs5 = _fsaverage_sphere('fsaverage5')
    fs4 = _fsaverage_sphere('fsaverage4')
    rng = np.random.default_rng(0)
    vals = jnp.asarray(rng.standard_normal(fs5.n_vertices).astype(np.float32))
    sa = np.asarray(vertex_areas(fs5))
    ta = np.asarray(vertex_areas(fs4))
    _, out = surface_resample(fs5, vals, fs4, method='adap_bary_area')
    lhs = float(np.sum(ta * np.asarray(out)))
    rhs = float(np.sum(sa * np.asarray(vals)))
    assert abs(lhs - rhs) <= 1e-3 * (abs(rhs) + 1.0)
