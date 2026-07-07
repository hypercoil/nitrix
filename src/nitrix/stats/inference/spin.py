# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Spin-test spatial null for the correspondence of two surface maps.

Testing whether two brain maps are spatially related by an ordinary
correlation p-value is anticonservative: both maps carry strong spatial
autocorrelation, so even unrelated maps correlate. The **spin test**
(Alexander-Bloch 2018 / Vazquez-Rodriguez 2019) builds a null that preserves
each map's spatial autocorrelation by randomly rotating one map's spherical
projection and reassigning vertices to their nearest rotated neighbour
(:func:`nitrix.geometry.spin_surrogates`), then compares the observed statistic
against the rotated null (:func:`spatial_null_test`). ``neuromaps.nulls`` is
the reference.
"""

from __future__ import annotations

from typing import Optional

from jaxtyping import Array, Float, Int

from ...geometry.sphere import random_rotation, spin_surrogates
from ._spatial_null import SpatialNullResult, spatial_null_test

# The spatial-null result is shared across generators (spin / Moran / ...).
SpinTestResult = SpatialNullResult

__all__ = ['SpinTestResult', 'spin_test']


def spin_test(
    x: Float[Array, 'V'],
    y: Float[Array, 'V'],
    coords: Float[Array, 'V 3'],
    *,
    key: Array,
    n_spin: int = 1000,
    two_sided: bool = True,
    hemisphere: Optional[Int[Array, 'V']] = None,
) -> SpatialNullResult:
    r"""Spin-test spatial null for the correlation between two surface maps.

    Rotates ``x``'s spherical projection ``n_spin`` times
    (:func:`~nitrix.geometry.random_rotation` +
    :func:`~nitrix.geometry.spin_surrogates`) and tests the observed
    ``corr(x, y)`` against the rotated null (:func:`spatial_null_test`). The
    rotation is applied to ``x`` only (the standard convention).

    **Medial wall:** set medial-wall entries of ``x`` (and ``y``) to ``NaN`` --
    a rotated vertex that pulls from the medial wall carries ``NaN``, and the
    statistic drops non-finite entries pairwise, so the null is computed over
    each spin's valid support (the Alexander-Bloch handling).

    Parameters
    ----------
    x, y : Float[Array, 'V']
        Two per-vertex maps over the same ``V`` vertices; medial-wall entries
        may be ``NaN``.
    coords : Float[Array, 'V 3']
        Vertex coordinates on the sphere (any radius; normalised internally).
    key : Array
        A :func:`jax.random.key` for the rotations.
    n_spin : int, optional
        Number of spin rotations (null size). Default ``1000``.
    two_sided : bool, optional
        Two-sided (``|null| >= |observed|``) if ``True`` (default).
    hemisphere : Int[Array, 'V'], optional
        Per-vertex hemisphere label (``0`` left / ``1`` right). When given, the
        hemispheres are spun independently with the Alexander-Bloch mirror
        reflection and reassigned within-hemisphere (see
        :func:`~nitrix.geometry.spin_surrogates`). ``None`` (default) spins the
        whole surface jointly.

    Returns
    -------
    SpatialNullResult
        ``(statistic, pvalue, null_distribution)``.

    Notes
    -----
    ``spin_surrogates`` forms a dense per-rotation vertex similarity, so peak
    memory is :math:`O(V^2)`; see its notes for the large-mesh caveat.
    """
    rotations = random_rotation(key, n_spin)
    x_spun = spin_surrogates(coords, x, rotations, hemisphere=hemisphere)
    return spatial_null_test(x, y, x_spun, two_sided=two_sided)
