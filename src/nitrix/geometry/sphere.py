# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
2-sphere primitives.

Coordinate conventions
----------------------

We use *Cartesian* normal-vector format ``(x, y, z)`` as the
canonical representation: every coordinate on the sphere is a
unit 3-vector (or an ``r``-scaled 3-vector for a radius-``r``
sphere).  This is the format consumed by ``spherical_geodesic_distance``,
``spherical_conv``, and the icosphere generation utilities.

For lat/long inputs (e.g. data digitised from a globe), convert
once via ``latlong_to_cartesian``.  All downstream operations assume
the Cartesian representation.

``spherical_conv`` re-backs the legacy O(N²) all-pairs convolution
on ``semiring_ell_matmul`` over a k-NN adjacency for O(N · k) cost
-- the marquee Phase 3 task per SPEC §6.1 ``3.2`` ("validates the
§3.1 design bet end-to-end").
"""
from __future__ import annotations

from typing import Optional, Union

import jax
import jax.lax as lax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from ..semiring import REAL, semiring_ell_matmul


__all__ = [
    'latlong_to_cartesian',
    'cartesian_to_latlong',
    'spherical_geodesic_distance',
    'spherical_conv',
]


# ---------------------------------------------------------------------------
# Coordinate conversions
# ---------------------------------------------------------------------------


def latlong_to_cartesian(
    latlong: Float[Array, '... 2'],
    r: float = 1.0,
) -> Float[Array, '... 3']:
    '''Latitude/longitude (radians) -> Cartesian ``(x, y, z)`` on a sphere.

    Latitude is the angle from the equator (``-pi/2`` at south pole,
    ``+pi/2`` at north pole); longitude is the angle around the
    equator from the prime meridian.

    Parameters
    ----------
    latlong
        Stacked ``(latitude, longitude)`` in radians, ``(..., 2)``.
    r
        Radius of the sphere.

    Returns
    -------
    Cartesian coordinates, ``(..., 3)``.
    '''
    lat = latlong[..., 0]
    lon = latlong[..., 1]
    cos_lat = jnp.cos(lat)
    return jnp.stack(
        (r * cos_lat * jnp.cos(lon),
         r * cos_lat * jnp.sin(lon),
         r * jnp.sin(lat)),
        axis=-1,
    )


def cartesian_to_latlong(
    xyz: Float[Array, '... 3'],
) -> Float[Array, '... 2']:
    '''Cartesian ``(x, y, z)`` -> ``(latitude, longitude)`` in radians.

    Inverse of ``latlong_to_cartesian``.  ``xyz`` need not be a unit
    vector; the returned angles are scale-invariant.

    Parameters
    ----------
    xyz
        Cartesian coordinates, ``(..., 3)``.

    Returns
    -------
    ``(latitude, longitude)``, ``(..., 2)``.
    '''
    x = xyz[..., 0]
    y = xyz[..., 1]
    z = xyz[..., 2]
    lat = jnp.arctan2(z, jnp.sqrt(x ** 2 + y ** 2))
    lon = jnp.arctan2(y, x)
    return jnp.stack((lat, lon), axis=-1)


# ---------------------------------------------------------------------------
# Geodesic distance
# ---------------------------------------------------------------------------


def _geodesic_pair(X, Y, r):
    '''Per-pair geodesic on already-aligned points.

    ``X`` and ``Y`` must have matching shapes ``(..., 3)``; output is
    ``(...,)``.  Used by both the all-pairs entry point and the
    inline distance computation inside ``spherical_conv``.
    '''
    cross = jnp.cross(X, Y, axis=-1)
    num = jnp.sqrt((cross ** 2).sum(-1))
    denom = (X * Y).sum(-1)
    angle = jnp.arctan2(num, denom)
    # ``arctan2`` returns in ``(-pi, pi]``; for antipodal points we
    # want the non-negative angle, so wrap negative values into
    # ``[0, pi]``.  Geodesic distance is then ``r * angle``.
    angle = jnp.where(angle < 0, angle + jnp.pi, angle)
    return r * angle


def spherical_geodesic_distance(
    X: Float[Array, '... n 3'],
    Y: Optional[Float[Array, '... m 3']] = None,
    r: float = 1.0,
) -> Float[Array, '... n m']:
    '''All-pairs great-circle distance between Cartesian points on a sphere.

    Parameters
    ----------
    X
        First set of points on the sphere, Cartesian, ``(..., n, 3)``.
    Y
        Second set; if ``None``, computes pairwise distance within
        ``X``.  Shape ``(..., m, 3)``.
    r
        Sphere radius.  The returned distance is ``r * angle``.

    Returns
    -------
    Distance matrix ``(..., n, m)`` in the same units as ``r``.
    '''
    if Y is None:
        Y = X
    if X.shape[-1] != 3 or Y.shape[-1] != 3:
        raise ValueError(
            'spherical_geodesic_distance: both inputs must have '
            f'shape (..., 3); got X.shape={X.shape}, Y.shape={Y.shape}.'
        )
    X_b = X[..., :, None, :]
    Y_b = Y[..., None, :, :]
    X_b, Y_b = jnp.broadcast_arrays(X_b, Y_b)
    return _geodesic_pair(X_b, Y_b, r)


# ---------------------------------------------------------------------------
# Spherical convolution (re-backed on the semiring substrate)
# ---------------------------------------------------------------------------


def _spherical_knn_indices(
    coor: Float[Array, 'n 3'],
    k: int,
    r: float,
) -> Int[Array, 'n k']:
    '''Top-k nearest neighbours by spherical geodesic distance.

    Materialises the ``(n, n)`` distance matrix; quadratic memory.
    Practical for ``n <= 10k``.  Larger meshes should pre-compute the
    adjacency (via a hierarchical tree or by the icosphere's natural
    k-ring) and pass it as ``neighbourhood=indices``.
    '''
    d = spherical_geodesic_distance(coor, coor, r=r)
    _, indices = lax.top_k(-d, k)
    return indices


def spherical_conv(
    data: Float[Array, '... n c'],
    coor: Float[Array, 'n 3'],
    *,
    sigma: float,
    neighbourhood: Union[int, Int[Array, 'n k']],
    r: float = 1.0,
    truncate: Optional[float] = None,
) -> Float[Array, '... n c']:
    '''Convolve data on a 2-sphere with an isotropic Gaussian kernel.

    Specialises onto ``semiring_ell_matmul``: build a per-point k-NN
    adjacency by spherical geodesic distance, weight neighbours by a
    Gaussian over the geodesic distance, normalise, reduce.  Cost is
    ``O(n * k * c)`` per call once the adjacency is in hand --
    *not* the legacy ``O(n^2 * c)`` of the all-pairs implementation.

    Parameters
    ----------
    data
        Per-point feature vectors, ``(..., n, c)``.  Leading batch
        dims are passed through.
    coor
        Per-point Cartesian coordinates on the sphere, ``(n, 3)``.
        Currently single-batch (no leading dims); ``vmap`` for
        per-subject coordinate sets.
    sigma
        Standard deviation of the Gaussian kernel, in the same units
        as ``r`` (radians × ``r``).
    neighbourhood
        Either an ``int`` ``k`` (compute k-NN by spherical geodesic
        on the fly; O(n²) memory) or an explicit ``(n, k)`` index
        array.  For large ``n`` (>~10k) provide the adjacency
        explicitly, e.g. from the icosphere's natural k-ring.
    r
        Sphere radius.
    truncate
        Optional hard distance cutoff: neighbours beyond
        ``truncate`` get weight zero.  ``None`` (default) means
        Gaussian weighting only.  Useful when the k-NN includes
        far-away points that should not contribute.

    Returns
    -------
    Smoothed data, ``(..., n, c)``.

    Notes
    -----
    The weights are normalised so each row sums to 1.  Constant data
    is therefore preserved.

    For ``data`` with multiple channels, every channel is smoothed
    by the same spatial kernel (the standard "depthwise" semantics).
    Per-channel sigma is not currently supported via this function;
    call multiple times if needed.
    '''
    if data.shape[-2] != coor.shape[0]:
        raise ValueError(
            f'spherical_conv: data.shape[-2]={data.shape[-2]} must '
            f'equal coor.shape[0]={coor.shape[0]}.'
        )
    n = coor.shape[0]

    # Resolve adjacency.
    if isinstance(neighbourhood, int):
        indices = _spherical_knn_indices(coor, neighbourhood, r=r)
    else:
        indices = jnp.asarray(neighbourhood, dtype=jnp.int32)
        if indices.shape[0] != n:
            raise ValueError(
                f'neighbourhood.shape[0]={indices.shape[0]} must '
                f'equal n={n}.'
            )

    # Geodesic distance from each point to each of its neighbours.
    X = coor[:, None, :]
    Y = coor[indices]
    dist = _geodesic_pair(X, Y, r)                        # (n, k)
    weights = jnp.exp(-0.5 * (dist / sigma) ** 2)         # (n, k)
    if truncate is not None:
        weights = jnp.where(dist > truncate, 0.0, weights)
    Z = weights.sum(axis=-1, keepdims=True)
    weights = weights / jnp.maximum(Z, jnp.finfo(weights.dtype).tiny)

    # Reduce via semiring_ell_matmul: out[..., i, c] = sum_p
    # weights[i, p] * data[..., indices[i, p], c].  ``data`` carries
    # leading batch dims; ``weights`` and ``indices`` do not, so we
    # broadcast at the call site by tiling weights / indices.
    batch_dims = data.shape[:-2]
    if batch_dims:
        # Tile the per-row weights/indices to match leading batch dims.
        weights_b = jnp.broadcast_to(
            weights, batch_dims + weights.shape,
        )
        indices_b = jnp.broadcast_to(
            indices, batch_dims + indices.shape,
        )
        return semiring_ell_matmul(
            weights_b, indices_b, data,
            semiring=REAL, n_cols=n, backend='jax',
        )
    return semiring_ell_matmul(
        weights, indices, data,
        semiring=REAL, n_cols=n, backend='jax',
    )
