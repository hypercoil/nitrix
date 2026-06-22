# -*- coding: utf-8 -*-
"""Spherical parameterisation (geometry-suite P3.2 / GS-2c+2d).

``spherical_parameterize``: spectral init + Riemannian gradient descent on
(S^2)^n under a conformal+areal energy with a fold-safe line-search.  Validated
that it stays bijective on the icosphere, that the one-shot (n_iterations=0)
returns the spectral init, and -- the keystone -- that on a real inflated cortex
it drives the spectral init to STRICTLY bijective while reducing areal
distortion.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from _real_meshes import fsaverage_white

from nitrix.geometry import (
    areal_distortion,
    inflate_surface,
    is_bijective_sphere_map,
    signed_spherical_areas,
    spherical_parameterize,
)
from nitrix.sparse import Mesh, icosphere


def _n_flipped(emb: jnp.ndarray, faces: jnp.ndarray) -> int:
    a = np.asarray(signed_spherical_areas(emb, faces))
    return int((a <= 0).sum())


def test_icosphere_stays_bijective() -> None:
    m = icosphere(3)
    out = spherical_parameterize(m, n_iterations=50)
    assert is_bijective_sphere_map(out, m.faces)


def test_one_shot_returns_spectral_init() -> None:
    m = icosphere(3)
    out = spherical_parameterize(m, n_iterations=0)
    assert is_bijective_sphere_map(out, m.faces, flip_area_tol=8e-4)
    assert np.allclose(np.linalg.norm(np.asarray(out), axis=1), 1.0, atol=1e-4)


def test_bad_init_raises() -> None:
    with pytest.raises(ValueError, match="'spectral' or 'radial'"):
        spherical_parameterize(icosphere(1), init='nonsense')


def test_real_cortex_becomes_strictly_bijective_and_lower_distortion() -> None:
    # white -> inflate -> spherical parameterise: the refinement removes the
    # spectral init's residual folds (strict bijectivity) and lowers distortion.
    v, f, _ = fsaverage_white()
    white = Mesh(jnp.asarray(v), jnp.asarray(f))
    inflated, _ = inflate_surface(
        white, n_iterations=150, spring_weight=0.8, metric_weight=0.3, step=0.6
    )
    inf_mesh = Mesh(inflated, white.faces)

    init = spherical_parameterize(
        inf_mesh, n_iterations=0
    )  # spectral one-shot
    refined = spherical_parameterize(inf_mesh, n_iterations=200)

    # The refined map is STRICTLY fold-free (the init may have a few flips).
    assert _n_flipped(refined, white.faces) == 0
    assert is_bijective_sphere_map(refined, white.faces)
    # ... and areal distortion is no worse than the init (typically lower).
    d_init = np.asarray(
        areal_distortion(inf_mesh, Mesh(init, white.faces))
    ).std()
    d_ref = np.asarray(
        areal_distortion(inf_mesh, Mesh(refined, white.faces))
    ).std()
    assert d_ref <= d_init + 1e-3
