# -*- coding: utf-8 -*-
"""Mesh topology invariants -- the genus-0 defect gate (geometry-suite P2.1).

Euler characteristic / genus on closed (sphere, torus) and open (disk) meshes,
plus the real fsaverage genus-0 white surface.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from _real_meshes import fsaverage_white

from nitrix.geometry import euler_characteristic, genus
from nitrix.sparse import Mesh, icosphere


def _torus(
    nu: int = 12, nv: int = 8, big: float = 3.0, small: float = 1.0
) -> Mesh:
    """A closed triangulated torus (genus 1) with periodic wrap."""
    us = np.linspace(0, 2 * np.pi, nu, endpoint=False)
    vs = np.linspace(0, 2 * np.pi, nv, endpoint=False)
    verts = []
    for u in us:
        for v in vs:
            verts.append(
                [
                    (big + small * np.cos(v)) * np.cos(u),
                    (big + small * np.cos(v)) * np.sin(u),
                    small * np.sin(v),
                ]
            )
    verts = np.asarray(verts, dtype=np.float32)

    def vid(i: int, j: int) -> int:
        return (i % nu) * nv + (j % nv)

    faces = []
    for i in range(nu):
        for j in range(nv):
            a, b, c, d = (
                vid(i, j),
                vid(i + 1, j),
                vid(i + 1, j + 1),
                vid(i, j + 1),
            )
            faces.append([a, b, c])
            faces.append([a, c, d])
    return Mesh(jnp.asarray(verts), jnp.asarray(np.asarray(faces, np.int32)))


def _disk() -> Mesh:
    # A single triangle: V=3, E=3, F=1 -> chi = 1 (open surface).
    return Mesh(
        jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        jnp.array([[0, 1, 2]]),
    )


@pytest.mark.parametrize('n_sub', [0, 1, 3])
def test_icosphere_is_genus0(n_sub: int) -> None:
    mesh = icosphere(n_sub)
    assert euler_characteristic(mesh) == 2
    assert genus(mesh) == 0


def test_torus_is_genus1() -> None:
    torus = _torus()
    assert euler_characteristic(torus) == 0
    assert genus(torus) == 1


def test_open_disk_euler_is_one_and_genus_raises() -> None:
    disk = _disk()
    assert euler_characteristic(disk) == 1
    with pytest.raises(ValueError, match='not a closed orientable'):
        genus(disk)


def test_real_white_surface_is_genus0() -> None:
    v, f, _ = fsaverage_white()
    mesh = Mesh(jnp.asarray(v), jnp.asarray(f))
    assert euler_characteristic(mesh) == 2
    assert genus(mesh) == 0
