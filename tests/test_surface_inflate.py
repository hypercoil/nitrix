# -*- coding: utf-8 -*-
"""Surface inflation + sulcal depth (geometry-suite P3.1 / GS-1).

The `mris_inflate` analogue: unfold a folded genus-0 surface (spring smoothing +
metric-restoring force) and emit sulcal depth as the integrated normal
displacement.  Anchored on roundness increasing under inflation, an
already-round sphere staying put, and -- the real-data oracle -- sulc
correlating with FreeSurfer ``?h.sulc``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _real_meshes import fsaverage_white

from nitrix.geometry import inflate_surface
from nitrix.sparse import Mesh, icosphere


def _radius_cv(v: jax.Array) -> float:
    """Coefficient of variation of vertex radius from the centroid (roundness:
    lower = rounder)."""
    a = np.asarray(v)
    r = np.linalg.norm(a - a.mean(0), axis=1)
    return float(r.std() / r.mean())


def test_inflation_unfolds_real_white_surface() -> None:
    v, f, _ = fsaverage_white()
    white = Mesh(jnp.asarray(v), jnp.asarray(f))
    inflated, sulc = inflate_surface(
        white, n_iterations=100, spring_weight=0.8, metric_weight=0.3, step=0.6
    )
    assert np.all(np.isfinite(np.asarray(inflated)))
    assert np.all(np.isfinite(np.asarray(sulc)))
    # The cortex gets rounder (unfolds).
    assert _radius_cv(inflated) < _radius_cv(white.vertices)
    # Topology preserved.
    assert np.array_equal(
        np.asarray(inflated.shape), np.asarray(white.vertices.shape)
    )


def test_sulc_correlates_with_freesurfer() -> None:
    v, f, overlays = fsaverage_white()
    if 'sulc' not in overlays:
        pytest.skip('no sulc overlay')
    white = Mesh(jnp.asarray(v), jnp.asarray(f))
    _, sulc = inflate_surface(
        white, n_iterations=100, spring_weight=0.8, metric_weight=0.3, step=0.6
    )
    corr = float(np.corrcoef(np.asarray(sulc), overlays['sulc'])[0, 1])
    assert (
        corr > 0.85
    )  # positive: nitrix sulc matches FS sign (sulci positive)


def test_noisy_sphere_is_smoothed() -> None:
    # Synthetic (networkless): radial bumps on an icosphere are smoothed out.
    m = icosphere(3)
    rng = np.asarray(
        jax.random.normal(jax.random.PRNGKey(0), (m.n_vertices, 1))
    )
    bumpy = Mesh(
        jnp.asarray(np.asarray(m.vertices) * (1.0 + 0.25 * rng)), m.faces
    )
    inflated, _ = inflate_surface(bumpy, n_iterations=80, spring_weight=0.8)
    assert _radius_cv(inflated) < _radius_cv(bumpy.vertices)


def test_round_sphere_stays_round() -> None:
    inflated, _ = inflate_surface(icosphere(3), n_iterations=50)
    assert _radius_cv(inflated) < 0.02  # already round -> stays round


def test_inflation_differentiable_and_jittable() -> None:
    mesh = icosphere(2)
    jitted = jax.jit(
        lambda v: inflate_surface(Mesh(v, mesh.faces), n_iterations=10)
    )
    inflated, sulc = jitted(mesh.vertices)
    assert inflated.shape == mesh.vertices.shape and sulc.shape == (
        mesh.n_vertices,
    )

    def loss(v: jax.Array) -> jax.Array:
        inf, s = inflate_surface(Mesh(v, mesh.faces), n_iterations=10)
        return jnp.sum(inf**2) + jnp.sum(s**2)

    g = jax.grad(loss)(mesh.vertices)
    assert np.all(np.isfinite(np.asarray(g)))
