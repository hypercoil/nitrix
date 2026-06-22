# -*- coding: utf-8 -*-
"""Ribbon-constrained volume->surface mapping (geometry-suite P4.2 / GS-14).

``ribbon_map`` integrates a volume along each white->pial cortical column.
Anchored on analytic volumes: a constant volume maps to a constant surface
field (both weightings preserve constants), and a volume that ramps linearly
*along the column* reduces to its mid-thickness value (both weightings are
symmetric about the mid-thickness).  Exercised differentiably and on a real
white/pial pair.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _real_meshes import fsaverage_surface

from nitrix.geometry import inflate_surface, ribbon_map
from nitrix.sparse import Mesh, icosphere


def _offset_pair(radius: float = 8.0, gap: float = 2.0):
    # White / pial as two concentric icospheres centred in a volume, in vertex
    # correspondence (shared topology, radially offset).
    m = icosphere(3)
    centre = jnp.array([16.0, 16.0, 16.0])
    white = Mesh(m.vertices * radius + centre, m.faces)
    pial = Mesh(m.vertices * (radius + gap) + centre, m.faces)
    return white, pial


def test_constant_volume_maps_to_constant() -> None:
    white, pial = _offset_pair()
    vol = jnp.full((32, 32, 32), 3.5)
    for weighting in ('pv', 'gaussian'):
        out = np.asarray(ribbon_map(vol, white, pial, weighting=weighting))
        assert out.shape == (white.n_vertices,)
        assert np.allclose(out, 3.5, atol=1e-4)


def test_linear_ramp_reduces_to_midthickness() -> None:
    # A volume that is a linear function of position: f(x) = a . x + b.  Along
    # each column the value is linear in t, so the symmetric column reduction
    # equals the value at the mid-thickness point (white + pial) / 2.
    g = jnp.arange(32, dtype=jnp.float32)
    vol = (
        2.0 * g[:, None, None]
        - 1.0 * g[None, :, None]
        + 0.5 * g[None, None, :]
    )
    white, pial = _offset_pair()
    mid = 0.5 * (white.vertices + pial.vertices)
    mx, my, mz = mid[:, 0], mid[:, 1], mid[:, 2]
    expected = np.asarray(2.0 * mx - 1.0 * my + 0.5 * mz)
    for weighting in ('pv', 'gaussian'):
        out = np.asarray(
            ribbon_map(vol, white, pial, n_samples=12, weighting=weighting)
        )
        assert np.allclose(out, expected, atol=2e-3)


def test_channels_last_volume() -> None:
    white, pial = _offset_pair()
    vol = jnp.stack(
        [jnp.full((32, 32, 32), 1.0), jnp.full((32, 32, 32), -2.0)], axis=-1
    )
    out = np.asarray(ribbon_map(vol, white, pial))
    assert out.shape == (white.n_vertices, 2)
    assert np.allclose(out[:, 0], 1.0, atol=1e-4)
    assert np.allclose(out[:, 1], -2.0, atol=1e-4)


def test_mismatched_counts_raises() -> None:
    white = icosphere(2)
    pial = icosphere(3)
    with pytest.raises(ValueError, match='vertex correspondence'):
        ribbon_map(jnp.zeros((8, 8, 8)), white, pial)


def test_bad_weighting_raises() -> None:
    white, pial = _offset_pair()
    with pytest.raises(ValueError, match="'pv' or 'gaussian'"):
        ribbon_map(jnp.zeros((32, 32, 32)), white, pial, weighting='nope')


def test_differentiable_wrt_volume_and_surfaces() -> None:
    white, pial = _offset_pair()

    def loss(vol: jax.Array, wv: jax.Array) -> jax.Array:
        return jnp.sum(ribbon_map(vol, Mesh(wv, white.faces), pial) ** 2)

    vol = jax.random.normal(jax.random.PRNGKey(0), (32, 32, 32))
    gv, gw = jax.grad(loss, argnums=(0, 1))(vol, white.vertices)
    assert np.all(np.isfinite(np.asarray(gv)))
    assert np.all(np.isfinite(np.asarray(gw)))
    assert gw.shape == white.vertices.shape


def test_real_white_pial_smoke() -> None:
    # A real white/pial pair (shared fsaverage5 topology -> correspondence):
    # a synthetic "T1w/T2w" volume sampled through the ribbon is finite and
    # spatially varying (a myelin-map smoke test).
    vw, fw = fsaverage_surface('white')
    vp, _ = fsaverage_surface('pial')
    # Shift into a positive index frame and build a smooth intensity volume.
    allv = np.concatenate([vw, vp], axis=0)
    lo = allv.min(0) - 4.0
    span = (allv.max(0) - lo) + 4.0
    shape = np.ceil(span).astype(int)
    gx, gy, gz = np.meshgrid(
        np.arange(shape[0]),
        np.arange(shape[1]),
        np.arange(shape[2]),
        indexing='ij',
    )
    vol = jnp.asarray(
        np.sin(0.05 * gx) + np.cos(0.04 * gy) + 0.03 * gz, dtype=jnp.float32
    )
    white = Mesh(jnp.asarray(vw) - jnp.asarray(lo), jnp.asarray(fw))
    pial = Mesh(jnp.asarray(vp) - jnp.asarray(lo), jnp.asarray(fw))
    out = np.asarray(ribbon_map(vol, white, pial, weighting='gaussian'))
    assert out.shape == (white.n_vertices,)
    assert np.all(np.isfinite(out))
    assert out.std() > 1e-3  # genuinely varies across the cortex


def test_inflate_then_ribbon_compose() -> None:
    # ribbon_map composes with the rest of the suite: inflate a sphere, build a
    # pial by a radial offset, map a constant -> constant (sanity of the seam).
    m = icosphere(3)
    centre = jnp.array([16.0, 16.0, 16.0])
    white_v = m.vertices * 8.0 + centre
    inflated, _ = inflate_surface(Mesh(white_v, m.faces), n_iterations=5)
    pial_v = inflated + 1.5 * (inflated - centre) / jnp.linalg.norm(
        inflated - centre, axis=1, keepdims=True
    )
    out = np.asarray(
        ribbon_map(
            jnp.full((40, 40, 40), 2.0),
            Mesh(inflated, m.faces),
            Mesh(pial_v, m.faces),
        )
    )
    assert np.allclose(out, 2.0, atol=1e-3)
