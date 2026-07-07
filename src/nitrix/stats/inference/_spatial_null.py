# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Shared spatial-null test seam: the surrogate -> p-value half.

A spatial null model = a **surrogate generator** (method-specific: spin,
Moran, variogram) + a **test** (shared). This module owns the test: given
surrogate maps of ``x`` that preserve its spatial autocorrelation, it compares
the observed correlation with ``y`` against the surrogate null. The generators
live with their numerical kind (:func:`nitrix.geometry.spin_surrogates`,
:func:`nitrix.graph.moran_surrogates`) and the per-method wrappers
(:func:`spin_test`, :func:`moran_test`) are thin: generate, then call
:func:`spatial_null_test`.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jaxtyping import Array, Float

__all__ = ['SpatialNullResult', 'spatial_null_test']


class SpatialNullResult(NamedTuple):
    """Result of a spatial-null test.

    Attributes
    ----------
    statistic : Float[Array, '']
        The observed Pearson correlation between the two maps.
    pvalue : Float[Array, '']
        The add-one permutation p-value: the fraction of surrogates whose
        (absolute, if ``two_sided``) correlation with ``y`` is at least the
        observed, computed as ``(1 + n_extreme) / (n_surrogates + 1)`` so it is
        never zero.
    null_distribution : Float[Array, 'n_surrogates']
        The surrogate correlations.
    """

    statistic: Float[Array, '']
    pvalue: Float[Array, '']
    null_distribution: Float[Array, 'n_surrogates']


def _pearson(
    a: Float[Array, '... V'], b: Float[Array, '... V']
) -> Float[Array, '...']:
    """Pearson correlation over the trailing axis."""
    a = a - a.mean(axis=-1, keepdims=True)
    b = b - b.mean(axis=-1, keepdims=True)
    num = (a * b).sum(axis=-1)
    den = jnp.sqrt((a * a).sum(axis=-1) * (b * b).sum(axis=-1))
    return num / den


def spatial_null_test(
    x: Float[Array, 'V'],
    y: Float[Array, 'V'],
    surrogates: Float[Array, 'n_surrogates V'],
    *,
    two_sided: bool = True,
) -> SpatialNullResult:
    r"""Spatial-null p-value from precomputed surrogate maps.

    The generator-agnostic half of a spatial null. Compares the observed
    Pearson correlation ``corr(x, y)`` against the surrogate null
    ``corr(surrogate, y)``.

    Parameters
    ----------
    x, y : Float[Array, 'V']
        The two maps over ``V`` locations.
    surrogates : Float[Array, 'n_surrogates V']
        Spatial-autocorrelation-preserving surrogate maps of ``x`` (e.g. from
        :func:`nitrix.geometry.spin_surrogates` or
        :func:`nitrix.graph.moran_surrogates`).
    two_sided : bool, optional
        If ``True`` (default), compare ``|null| >= |observed|``; otherwise the
        signed upper tail ``null >= observed``.

    Returns
    -------
    SpatialNullResult
        ``(statistic, pvalue, null_distribution)``.
    """
    observed = _pearson(x, y)
    null = _pearson(surrogates, y[None, :])
    n = surrogates.shape[0]
    if two_sided:
        extreme = jnp.abs(null) >= jnp.abs(observed)
    else:
        extreme = null >= observed
    pvalue = (1.0 + extreme.sum()) / (n + 1.0)
    return SpatialNullResult(
        statistic=observed, pvalue=pvalue, null_distribution=null
    )
