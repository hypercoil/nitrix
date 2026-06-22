# -*- coding: utf-8 -*-
"""Spectral (recon-surf) spherical embedding (geometry-suite P3.2 / GS-2b).

One-shot spherical map from the first three non-constant Laplace-Beltrami
eigenfunctions (the FastSurfer / recon-surf method).  Validated bijective on the
icosphere and -- within the recon-surf quality tolerance -- on a real inflated
cortical surface.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from _real_meshes import fsaverage_white

from nitrix.geometry import (
    inflate_surface,
    is_bijective_sphere_map,
    spectral_sphere_embedding,
)
from nitrix.sparse import Mesh, icosphere


def test_icosphere_spectral_is_bijective() -> None:
    m = icosphere(3)
    emb = spectral_sphere_embedding(m)
    assert is_bijective_sphere_map(emb, m.faces, flip_area_tol=8e-4)


def test_output_is_on_the_requested_sphere() -> None:
    m = icosphere(2)
    emb = np.asarray(spectral_sphere_embedding(m, radius=7.0))
    assert np.allclose(np.linalg.norm(emb, axis=1), 7.0, atol=1e-4)


def test_real_inflated_surface_spectral_is_bijective() -> None:
    # The realistic pipeline: white -> inflate -> spectral spherical map.
    v, f, _ = fsaverage_white()
    white = Mesh(jnp.asarray(v), jnp.asarray(f))
    inflated, _ = inflate_surface(
        white, n_iterations=150, spring_weight=0.8, metric_weight=0.3, step=0.6
    )
    emb = spectral_sphere_embedding(Mesh(inflated, white.faces))
    # recon-surf-grade: bijective within the recon-surf flipped-area tolerance.
    assert is_bijective_sphere_map(emb, white.faces, flip_area_tol=8e-4)
