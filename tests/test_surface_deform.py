# -*- coding: utf-8 -*-
"""SDF normal-march surface deformation (geometry-suite P2.4 / GS-10).

The geometry-light mover: advance a genus-0 mesh's vertices along their normals
onto a target SDF zero-set, preserving vertex correspondence and genus-0 (no
re-tessellation).  Validated on a sphere R1 -> R2 march (converges, stays
spherical, topology + correspondence preserved), with jit and gradient checks.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from nitrix.geometry import (
    deform_to_sdf,
    euler_characteristic,
    sample_at_points,
)
from nitrix.sparse import Mesh, icosphere


def _sphere_sdf(n: int, radius: float, centre: float) -> jax.Array:
    ax = np.arange(n)
    x, y, z = np.meshgrid(ax, ax, ax, indexing='ij')
    return jnp.asarray(
        np.sqrt((x - centre) ** 2 + (y - centre) ** 2 + (z - centre) ** 2)
        - radius,
        dtype=jnp.float32,
    )


def _start_sphere(radius: float, centre: float, n_sub: int = 3) -> Mesh:
    m = icosphere(n_sub)
    return Mesh(m.vertices * radius + centre, m.faces)


def test_march_converges_to_target_sphere() -> None:
    centre, r2 = 18.0, 12.0
    sdf = _sphere_sdf(36, r2, centre)
    start = _start_sphere(7.0, centre)
    out = deform_to_sdf(
        start, sdf, n_iterations=120, step=0.4, smooth_weight=0.05
    )
    radii = np.linalg.norm(np.asarray(out.vertices) - centre, axis=1)
    assert abs(radii.mean() - r2) / r2 < 0.02  # reached the target radius
    assert radii.std() < 0.1  # stayed spherical


def test_topology_and_correspondence_preserved() -> None:
    centre = 18.0
    sdf = _sphere_sdf(36, 12.0, centre)
    start = _start_sphere(7.0, centre)
    out = deform_to_sdf(
        start, sdf, n_iterations=80, step=0.3, smooth_weight=0.05
    )
    # faces unchanged (vertex correspondence) and genus-0 inherited.
    assert np.array_equal(np.asarray(out.faces), np.asarray(start.faces))
    assert euler_characteristic(out) == 2


def test_vertices_land_near_zero_level() -> None:
    centre = 18.0
    sdf = _sphere_sdf(36, 12.0, centre)
    start = _start_sphere(7.0, centre)
    out = deform_to_sdf(
        start, sdf, n_iterations=120, step=0.4, smooth_weight=0.05
    )
    residual = np.asarray(sample_at_points(sdf, out.vertices, mode='nearest'))
    assert np.abs(residual).mean() < 0.2  # vertices sit ~on the surface


def test_already_on_surface_is_near_fixed() -> None:
    centre, r = 18.0, 12.0
    sdf = _sphere_sdf(36, r, centre)
    on = _start_sphere(r, centre)
    out = deform_to_sdf(on, sdf, n_iterations=40, step=0.3, smooth_weight=0.02)
    move = np.linalg.norm(
        np.asarray(out.vertices) - np.asarray(on.vertices), axis=1
    )
    assert move.max() < 0.5  # barely moves (already at the zero level)


def test_jittable_and_differentiable() -> None:
    centre = 18.0
    sdf = _sphere_sdf(32, 11.0, centre)
    start = _start_sphere(7.0, centre, n_sub=2)
    jitted = jax.jit(
        lambda v: (
            deform_to_sdf(Mesh(v, start.faces), sdf, n_iterations=20).vertices
        )
    )
    assert jitted(start.vertices).shape == start.vertices.shape

    def loss(v: jax.Array) -> jax.Array:
        return jnp.sum(
            deform_to_sdf(Mesh(v, start.faces), sdf, n_iterations=10).vertices
            ** 2
        )

    g = jax.grad(loss)(start.vertices)
    assert np.all(np.isfinite(np.asarray(g)))


def test_deform_mode_kwarg_is_forwarded() -> None:
    # mode= is exposed (audit AI-A4) and forwarded to sample_at_points; both
    # boundary modes run and the SDF gradient still drives a Linear sample.
    m = icosphere(2)
    centre = jnp.array([8.0, 8.0, 8.0])
    mesh = Mesh(m.vertices * 4.0 + centre, m.faces)
    # radial SDF of a sphere of radius 5 centred in a 16^3 grid.
    gx, gy, gz = jnp.meshgrid(
        jnp.arange(16.0), jnp.arange(16.0), jnp.arange(16.0), indexing='ij'
    )
    sdf = jnp.sqrt((gx - 8) ** 2 + (gy - 8) ** 2 + (gz - 8) ** 2) - 5.0
    out_n = deform_to_sdf(mesh, sdf, n_iterations=10, mode='nearest')
    out_c = deform_to_sdf(mesh, sdf, n_iterations=10, mode='constant')
    assert np.all(np.isfinite(np.asarray(out_n.vertices)))
    assert np.all(np.isfinite(np.asarray(out_c.vertices)))
