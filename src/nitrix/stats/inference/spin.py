# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Spin-test spatial null for the correspondence of two surface maps.

Testing whether two brain maps are spatially related by an ordinary
correlation p-value is anticonservative: both maps carry strong spatial
autocorrelation, so even unrelated maps correlate. The **spin test**
(Alexander-Bloch 2018 / Vazquez-Rodriguez 2019) builds a null distribution
that preserves each map's spatial autocorrelation by randomly rotating one
map's spherical projection and reassigning vertices to their nearest rotated
neighbour (:func:`nitrix.geometry.spin_surrogates`), then compares the observed
statistic against the rotated null.

The surrogate *generation* is a geometric primitive (in ``geometry.sphere``);
this module owns the *test* -- the observed statistic, the rotated null
distribution, and the exact permutation p-value. ``neuromaps.nulls`` is the
reference.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jaxtyping import Array, Float

from ...geometry.sphere import random_rotation, spin_surrogates

__all__ = ['SpinTestResult', 'spin_test']


class SpinTestResult(NamedTuple):
    """Result of a :func:`spin_test`.

    Attributes
    ----------
    statistic : Float[Array, '']
        The observed Pearson correlation between the two maps.
    pvalue : Float[Array, '']
        The spin-test p-value: the fraction of rotated surrogates whose
        (absolute, if ``two_sided``) correlation with ``y`` is at least the
        observed, with the ``(1 + ...)/(n_spin + 1)`` add-one correction so it
        is never zero.
    null_distribution : Float[Array, 'n_spin']
        The surrogate correlations, one per rotation.
    """

    statistic: Float[Array, '']
    pvalue: Float[Array, '']
    null_distribution: Float[Array, 'n_spin']


def _pearson(
    a: Float[Array, '... V'], b: Float[Array, '... V']
) -> Float[Array, '...']:
    """Pearson correlation over the trailing (vertex) axis."""
    a = a - a.mean(axis=-1, keepdims=True)
    b = b - b.mean(axis=-1, keepdims=True)
    num = (a * b).sum(axis=-1)
    den = jnp.sqrt((a * a).sum(axis=-1) * (b * b).sum(axis=-1))
    return num / den


def spin_test(
    x: Float[Array, 'V'],
    y: Float[Array, 'V'],
    coords: Float[Array, 'V 3'],
    *,
    key: Array,
    n_spin: int = 1000,
    two_sided: bool = True,
) -> SpinTestResult:
    r"""Spin-test spatial null for the correlation between two surface maps.

    Rotates ``x``'s spherical projection ``n_spin`` times
    (:func:`~nitrix.geometry.random_rotation` +
    :func:`~nitrix.geometry.spin_surrogates`), correlates each rotated
    surrogate with ``y``, and returns the observed correlation with its
    spin-test p-value. The rotation is applied to ``x`` only, which is the
    standard one-sided-generation convention.

    Parameters
    ----------
    x, y : Float[Array, 'V']
        Two per-vertex maps over the same ``V`` vertices.
    coords : Float[Array, 'V 3']
        Vertex coordinates on the sphere (any radius; normalised internally).
    key : Array
        A :func:`jax.random.key` for the rotations.
    n_spin : int, optional
        Number of spin rotations (null size). Default ``1000``.
    two_sided : bool, optional
        If ``True`` (default), compare ``|null| >= |observed|``; otherwise the
        signed upper tail ``null >= observed``.

    Returns
    -------
    SpinTestResult
        ``(statistic, pvalue, null_distribution)``.

    Notes
    -----
    ``spin_surrogates`` forms a dense per-rotation vertex similarity, so peak
    memory is :math:`O(V^2)`; see its notes for the large-mesh caveat.
    """
    rotations = random_rotation(key, n_spin)
    x_spun = spin_surrogates(coords, x, rotations)  # (n_spin, V)
    observed = _pearson(x, y)
    null = _pearson(x_spun, y[None, :])  # (n_spin,)
    if two_sided:
        extreme = jnp.abs(null) >= jnp.abs(observed)
    else:
        extreme = null >= observed
    pvalue = (1.0 + extreme.sum()) / (n_spin + 1.0)
    return SpinTestResult(
        statistic=observed, pvalue=pvalue, null_distribution=null
    )
