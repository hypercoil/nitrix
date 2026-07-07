# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
r"""BrainSMASH: variogram-matched generative spatial null (Burt 2020).

The most general parameterized null: given a map ``x`` and a **distance matrix**
``D`` (geodesic or Euclidean; surface *or* volume), it generates surrogate maps
that reproduce ``x``'s empirical **variogram** -- the full distance-dependence
of its spatial autocorrelation -- rather than relying on a spherical rotation
(spin) or a graph spectrum (Moran). Each surrogate is a random permutation of
``x`` smoothed at several bandwidths, linearly recombined so its variogram
matches the target's (non-negative least squares), then rank-matched back to
``x``'s value distribution.

Reference: Burt JB et al. (2020), *Generative modeling of brain maps with
spatial autocorrelation*, NeuroImage. https://doi.org/10.1016/j.neuroimage.2020.117038
(the ``brainsmash`` package).
"""

from __future__ import annotations

from typing import Literal, Sequence, cast

import jax
import jax.lax as lax
import jax.numpy as jnp
from jaxtyping import Array, Float, Num

from ._spatial_null import SpatialNullResult, spatial_null_test

__all__ = ['brainsmash_surrogates', 'brainsmash_test', 'variogram']

Kernel = Literal['exp', 'gaussian', 'uniform']


def variogram(
    x: Float[Array, 'n'],
    distance: Num[Array, 'n n'],
    *,
    n_bins: int = 25,
) -> Float[Array, 'n_bins']:
    r"""Empirical (semi-)variogram of a map, binned by pairwise distance.

    :math:`\gamma(d) = \tfrac{1}{2}\,\mathbb{E}\big[(x_i - x_j)^2 \mid
    D_{ij} \approx d\big]`, over the unordered pairs, in ``n_bins`` equal-width
    distance bins. The rising variogram encodes the spatial-autocorrelation
    structure (low :math:`\gamma` at short range = smooth). :math:`O(n^2)`.

    Parameters
    ----------
    x : Float[Array, 'n']
        The map.
    distance : Num[Array, 'n n']
        Pairwise distance matrix (symmetric, zero diagonal).
    n_bins : int, optional
        Number of distance bins. Default ``25``.

    Returns
    -------
    Float[Array, 'n_bins']
        The mean semivariance in each distance bin.
    """
    n = x.shape[0]
    diff2 = 0.5 * (x[:, None] - x[None, :]) ** 2
    mask = jnp.triu(jnp.ones((n, n), dtype=bool), k=1)
    dmax = jnp.where(mask, distance, 0.0).max()
    edges = jnp.linspace(0.0, dmax, n_bins + 1)

    def bin_mean(b: Array) -> Float[Array, '']:
        lo = edges[b]
        hi = edges[b + 1]
        last = b == (n_bins - 1)
        in_bin = mask & (distance >= lo) & ((distance < hi) | last)
        total = jnp.where(in_bin, diff2, 0.0).sum()
        count = in_bin.sum()
        return total / jnp.maximum(count, 1.0)

    return cast(Float[Array, 'n_bins'], lax.map(bin_mean, jnp.arange(n_bins)))


def _kernel_smooth(
    x: Float[Array, 'n'],
    distance: Num[Array, 'n n'],
    bandwidth: Float[Array, ''],
    kernel: str,
) -> Float[Array, 'n']:
    """Distance-weighted smoothing of ``x`` at a given bandwidth."""
    if kernel == 'exp':
        weight = jnp.exp(-distance / bandwidth)
    elif kernel == 'gaussian':
        weight = jnp.exp(-0.5 * (distance / bandwidth) ** 2)
    elif kernel == 'uniform':
        weight = (distance <= bandwidth).astype(x.dtype)
    else:
        raise ValueError(
            f"kernel must be 'exp'/'gaussian'/'uniform'; got {kernel!r}."
        )
    weight = weight / weight.sum(axis=-1, keepdims=True)
    return weight @ x


def _nnls(
    design: Float[Array, 'm k'],
    target: Float[Array, 'm'],
    iters: int = 300,
) -> Float[Array, 'k']:
    """Non-negative least squares ``min_{b>=0} ||design b - target||`` by
    projected gradient descent (cuSolver-free; the ``design`` is tiny)."""
    gram = design.T @ design
    dty = design.T @ target
    step = 1.0 / (jnp.trace(gram) + 1e-12)
    beta0 = jnp.zeros(design.shape[1], dtype=target.dtype)

    def body(
        beta: Float[Array, 'k'], _: None
    ) -> tuple[Float[Array, 'k'], None]:
        grad = gram @ beta - dty
        return jnp.maximum(beta - step * grad, 0.0), None

    beta, _ = lax.scan(body, beta0, None, length=iters)
    return beta


def brainsmash_surrogates(
    x: Float[Array, 'n'],
    distance: Num[Array, 'n n'],
    n_surrogates: int,
    key: Array,
    *,
    deltas: Sequence[float] = (0.1, 0.2, 0.3, 0.4, 0.5),
    n_bins: int = 25,
    kernel: Kernel = 'exp',
    resample: bool = True,
) -> Float[Array, 'n_surrogates n']:
    r"""BrainSMASH variogram-matched surrogate maps (Burt 2020).

    Each surrogate: permute ``x``, smooth the permutation at each bandwidth
    ``delta * max(distance)``, solve a non-negative least squares so the
    combined map's variogram matches ``x``'s, then (``resample=True``, default)
    rank-match to ``x``'s value distribution.

    Parameters
    ----------
    x : Float[Array, 'n']
        The map to generate surrogates of.
    distance : Num[Array, 'n n']
        Pairwise distance matrix (geodesic or Euclidean); surface or volume.
    n_surrogates : int
        Number of surrogate maps.
    key : Array
        A :func:`jax.random.key`.
    deltas : sequence of float, optional
        Smoothing bandwidths as fractions of the maximum distance. Default
        ``(0.1, ..., 0.5)``.
    n_bins : int, optional
        Variogram distance bins. Default ``25``.
    kernel : {'exp', 'gaussian', 'uniform'}, optional
        Smoothing kernel. Default ``'exp'``.
    resample : bool, optional
        If ``True`` (default), rank-match each surrogate to ``x``'s exact value
        distribution (the BrainSMASH ``resample`` variant); otherwise rescale
        to ``x``'s mean and standard deviation.

    Returns
    -------
    Float[Array, 'n_surrogates n']
        One variogram-matched surrogate per row.

    Notes
    -----
    :math:`O(n^2)` per bandwidth (dense distance matrix); apt at parcel or
    moderate-mesh resolution. The full-mesh 'sampled' subset variant is a
    follow-up.
    """
    target = variogram(x, distance, n_bins=n_bins)
    dmax = distance.max()
    x_sorted = jnp.sort(x)

    def one(subkey: Array) -> Float[Array, 'n']:
        permuted = jax.random.permutation(subkey, x)
        smoothed = jnp.stack(
            [
                _kernel_smooth(permuted, distance, d * dmax, kernel)
                for d in deltas
            ]
        )  # (k, n)
        design = jnp.stack(
            [variogram(s, distance, n_bins=n_bins) for s in smoothed],
            axis=1,
        )  # (n_bins, k)
        beta = _nnls(design, target)
        combined = (jnp.sqrt(beta)[:, None] * smoothed).sum(0)  # (n,)
        if resample:
            ranks = jnp.argsort(jnp.argsort(combined))
            return x_sorted[ranks]
        centred = (combined - combined.mean()) / (combined.std() + 1e-12)
        return x.mean() + x.std() * centred

    return jax.vmap(one)(jax.random.split(key, n_surrogates))


def brainsmash_test(
    x: Float[Array, 'n'],
    y: Float[Array, 'n'],
    distance: Num[Array, 'n n'],
    *,
    key: Array,
    n_surrogates: int = 1000,
    deltas: Sequence[float] = (0.1, 0.2, 0.3, 0.4, 0.5),
    n_bins: int = 25,
    kernel: Kernel = 'exp',
    resample: bool = True,
    two_sided: bool = True,
) -> SpatialNullResult:
    r"""BrainSMASH variogram-matched spatial null for ``corr(x, y)``.

    Generates ``n_surrogates`` variogram-matched surrogates of ``x``
    (:func:`brainsmash_surrogates`) and tests the observed correlation against
    them (:func:`spatial_null_test`). See :func:`brainsmash_surrogates` for the
    generation arguments.

    Returns
    -------
    SpatialNullResult
        ``(statistic, pvalue, null_distribution)``.
    """
    surrogates = brainsmash_surrogates(
        x,
        distance,
        n_surrogates,
        key,
        deltas=deltas,
        n_bins=n_bins,
        kernel=kernel,
        resample=resample,
    )
    return spatial_null_test(x, y, surrogates, two_sided=two_sided)
