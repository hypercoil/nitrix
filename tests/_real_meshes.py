# -*- coding: utf-8 -*-
"""Real FreeSurfer / fs_LR mesh fixtures for the geometry suite (test-only IO).

The IO lives **here, never in nitrix**: ``nilearn`` / ``templateflow`` /
``nibabel`` are test-only dependencies (SPEC §5.2 + §8 golden corpus); the
library imports only ``jax`` / ``jaxtyping`` / ``numpy``.  Each loader reads
files and hands nitrix **plain arrays** -- the "consumer reads files -> nitrix
gets arrays" contract that ``icosphere_hierarchy_from_levels`` was designed
around.  Exercising a real surface through the suite is therefore also a live
test of that array-handoff seam.

Loaders ``importorskip`` their dependency and ``skip`` (not fail) when the data
download is unavailable, so the core suite still runs networkless.  Results are
cached per process.

Notes on the fsaverage overlays
-------------------------------
``fetch_surf_fsaverage('fsaverage5')`` ships ``area`` / ``curv`` / ``sulc`` /
``thick`` overlays, but the ``area`` overlay is **not** the geometric vertex
area of the ``white`` tessellation it returns (its sum is ~0.75x and it
correlates only ~0.83) -- a resampling artefact of the downsampled fsaverage5
surface.  So ``area`` is *not* used as a tight oracle; the area primitives are
validated against robust geometric invariants (partition, positivity) instead.
``curv`` / ``sulc`` are valid per-vertex feature oracles for the curvature /
inflation primitives (class / correlation, not bit-parity).
"""

from __future__ import annotations

import functools
from typing import Any, Dict, Tuple

import numpy as np
import pytest
from numpy.typing import NDArray


@functools.lru_cache(maxsize=None)
def fsaverage_white(
    mesh: str = 'fsaverage5',
    hemi: str = 'left',
) -> Tuple[NDArray[Any], NDArray[Any], Dict[str, NDArray[Any]]]:
    """Return ``(vertices, faces, overlays)`` for a real FreeSurfer white surface.

    ``overlays`` maps ``'curv'`` / ``'sulc'`` / ``'thick'`` / ``'area'`` to
    per-vertex arrays (those present in the fetch).  Skips (not fails) when
    nilearn or the download is unavailable.
    """
    datasets = pytest.importorskip('nilearn.datasets')
    surface = pytest.importorskip('nilearn.surface')
    try:
        fs = datasets.fetch_surf_fsaverage(mesh)
        coords, faces = surface.load_surf_mesh(fs[f'white_{hemi}'])
        overlays: Dict[str, NDArray[Any]] = {}
        for key in ('curv', 'sulc', 'thick', 'area'):
            full = f'{key}_{hemi}'
            if full in fs:
                overlays[key] = np.asarray(
                    surface.load_surf_data(fs[full]), dtype=np.float32
                )
    except Exception as exc:  # network / data unavailable -> skip, don't fail
        pytest.skip(f'real mesh {mesh}/{hemi} unavailable: {exc}')
    return (
        np.asarray(coords, dtype=np.float32),
        np.asarray(faces, dtype=np.int32),
        overlays,
    )


@functools.lru_cache(maxsize=None)
def fsaverage_surface(
    surf: str = 'white',
    hemi: str = 'left',
    mesh: str = 'fsaverage5',
) -> Tuple[NDArray[Any], NDArray[Any]]:
    """Return ``(vertices, faces)`` for any fsaverage surface (``white`` /
    ``pial`` / ``infl`` / ``sphere``).

    All surfaces of a given (mesh, hemi) **share topology** (same ``faces`` and
    vertex correspondence), so e.g. ``white`` -> ``sphere`` is a real surface
    warp for distortion tests.  Skips when unavailable.
    """
    datasets = pytest.importorskip('nilearn.datasets')
    surface = pytest.importorskip('nilearn.surface')
    try:
        fs = datasets.fetch_surf_fsaverage(mesh)
        coords, faces = surface.load_surf_mesh(fs[f'{surf}_{hemi}'])
    except Exception as exc:  # network / data unavailable -> skip, don't fail
        pytest.skip(f'real surface {surf}/{hemi} unavailable: {exc}')
    return (
        np.asarray(coords, dtype=np.float32),
        np.asarray(faces, dtype=np.int32),
    )
