# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Moran spectral-randomization spatial null for two brain maps.

The spectral counterpart of the spin test: the surrogate maps
(:func:`nitrix.graph.moran_surrogates`) preserve the Moran's-``I`` spectrum
(spatial autocorrelation) of ``x`` by sign-flipping its Moran-eigenvector
coefficients, so it needs **no coordinates** and applies to any symmetric
adjacency / weight matrix (surface *or* volume). ``neuromaps.nulls`` /
BrainSpace's ``MoranRandomization`` are the reference.
"""

from __future__ import annotations

from jaxtyping import Array, Float, Num

from ...graph import moran_surrogates
from ._spatial_null import SpatialNullResult, spatial_null_test

__all__ = ['moran_test']


def moran_test(
    x: Float[Array, 'n'],
    y: Float[Array, 'n'],
    adjacency: Num[Array, 'n n'],
    *,
    key: Array,
    n_surrogates: int = 1000,
    two_sided: bool = True,
) -> SpatialNullResult:
    r"""Moran spectral-randomization spatial null for ``corr(x, y)``.

    Generates ``n_surrogates`` spatial-autocorrelation-matched surrogates of
    ``x`` (:func:`~nitrix.graph.moran_surrogates`) and tests the observed
    correlation against them (:func:`spatial_null_test`).

    Parameters
    ----------
    x, y : Float[Array, 'n']
        Two maps over the ``n`` graph nodes.
    adjacency : Num[Array, 'n n']
        Symmetric non-negative weight / adjacency matrix defining the spatial
        relationships.
    key : Array
        A :func:`jax.random.key` for the surrogate sign flips.
    n_surrogates : int, optional
        Number of surrogate maps (null size). Default ``1000``.
    two_sided : bool, optional
        Two-sided (``|null| >= |observed|``) if ``True`` (default).

    Returns
    -------
    SpatialNullResult
        ``(statistic, pvalue, null_distribution)``.
    """
    surrogates = moran_surrogates(adjacency, x, n_surrogates, key)
    return spatial_null_test(x, y, surrogates, two_sided=two_sided)
