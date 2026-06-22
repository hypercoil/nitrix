# -*- coding: utf-8 -*-
"""Cortical thickness (geometry-suite P2.5 / GS-9).

Correspondence thickness (||pial - white||, differentiable) and symmetric
thickness (Fischl & Dale, host-side nearest-surface).  Anchored on concentric
spheres (constant thickness R2 - R1) and correlated with FreeSurfer
``?h.thickness`` on the real white/pial surfaces.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _real_meshes import fsaverage_surface, fsaverage_white

from nitrix.geometry import cortical_thickness
from nitrix.sparse import Mesh, icosphere


def _concentric(r1: float, r2: float, n_sub: int = 3) -> tuple[Mesh, Mesh]:
    m = icosphere(n_sub)
    return Mesh(m.vertices * r1, m.faces), Mesh(m.vertices * r2, m.faces)


def test_correspondence_constant_on_concentric_spheres() -> None:
    white, pial = _concentric(8.0, 11.5)
    t = np.asarray(cortical_thickness(white, pial, method='correspondence'))
    assert np.allclose(t, 3.5, atol=1e-4)


def test_symmetric_on_concentric_spheres() -> None:
    white, pial = _concentric(8.0, 11.5)
    t = np.asarray(cortical_thickness(white, pial, method='symmetric'))
    assert np.allclose(t.mean(), 3.5, atol=0.05)
    assert t.std() < 0.05


def test_correspondence_differentiable() -> None:
    white, pial = _concentric(8.0, 11.5)

    def loss(vp: jax.Array) -> jax.Array:
        return jnp.sum(
            cortical_thickness(
                white, Mesh(vp, pial.faces), method='correspondence'
            )
        )

    g = jax.grad(loss)(pial.vertices)
    assert g.shape == pial.vertices.shape
    assert np.all(np.isfinite(np.asarray(g)))


def test_mismatched_vertex_counts_raises() -> None:
    with pytest.raises(ValueError, match='vertex correspondence'):
        cortical_thickness(icosphere(1), icosphere(2))


def test_bad_method_raises() -> None:
    white, pial = _concentric(8.0, 11.5)
    with pytest.raises(ValueError, match='symmetric.*correspondence'):
        cortical_thickness(white, pial, method='nearest')


def test_real_thickness_correlates_with_freesurfer() -> None:
    # fsaverage white/pial share topology -> correspondence thickness; compare
    # to FreeSurfer's ?h.thickness (class/correlation, not bit-parity).
    vw, fw = fsaverage_surface('white')
    vp, _ = fsaverage_surface('pial')
    _, _, overlays = fsaverage_white()
    if 'thick' not in overlays:
        pytest.skip('no thickness overlay')
    white = Mesh(jnp.asarray(vw), jnp.asarray(fw))
    pial = Mesh(jnp.asarray(vp), jnp.asarray(fw))
    t = np.asarray(cortical_thickness(white, pial, method='correspondence'))
    assert np.all(t >= 0.0) and np.all(np.isfinite(t))
    corr = float(np.corrcoef(t, overlays['thick'])[0, 1])
    assert corr > 0.8
