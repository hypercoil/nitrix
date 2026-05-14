# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Bilateral Gaussian smoothing via the semiring substrate.

Per SPEC_UPDATE §3.3 the marquee edge-preserving capability,
delivered at GA regardless of permutohedral risk.  For each output
position, gather the feature-space neighbourhood, weight by a
Gaussian over the per-feature distance, normalise, and reduce via
``semiring_ell_matmul`` with the REAL semiring.

Practical use case: multi-modal neuroimaging smoothing where
``features`` contains spatial coordinates concatenated with one or
two intensity / modality channels.  ``d_f ≤ 5`` and neighbourhoods
up to ~7³ voxels are well-supported.

The relationship to ``gaussian``: in the limit
``sigma_features[intensity] -> infinity``, the intensity-distance
contribution to the weight goes to zero and the bilateral reduces
to a Gaussian over the spatial coordinates.  This is the standard
"bilateral degrades gracefully to Gaussian when there's no edge to
preserve" property; we verify it in the test suite.
"""
from __future__ import annotations

from typing import Optional, Union

import jax
import jax.lax as lax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from .._internal.backend import Backend
from ..semiring import REAL, semiring_ell_matmul


__all__ = ['bilateral_gaussian', 'brute_force_knn']


def brute_force_knn(
    features: Float[Array, 'n d_f'],
    k: int,
    *,
    sigma_features: Optional[Float[Array, 'd_f']] = None,
) -> Int[Array, 'n k']:
    '''Brute-force k-nearest-neighbour search in (optionally rescaled) feature space.

    Materialises the ``(n, n)`` distance matrix; quadratic in memory.
    Practical for ``n ≲ 10k``.  For larger ``n``, the caller should
    pre-compute the adjacency with a spatial-index data structure
    (KD-tree, grid hashing, etc.) and pass it as
    ``neighbourhood=indices`` to ``bilateral_gaussian``.

    Parameters
    ----------
    features
        Per-point feature vectors, ``(n, d_f)``.
    k
        Number of neighbours per query point.  The point itself is
        included in its own ``k`` (so ``k=1`` returns self-only).
    sigma_features
        Optional per-feature ``sigma`` for rescaling before the
        distance.  ``None`` means use raw distances.  When supplied,
        the search ranks by ``sum_d ((diff[d] / sigma[d])**2)``, the
        same metric ``bilateral_gaussian`` uses for its weights.

    Returns
    -------
    Indices of the ``k`` nearest neighbours for each query point,
    shape ``(n, k)``.  Sorted by ascending distance.
    '''
    if sigma_features is not None:
        rescaled = features / sigma_features
    else:
        rescaled = features
    # Pairwise squared distances: (n, n).
    diff = rescaled[:, None, :] - rescaled[None, :, :]
    d2 = (diff ** 2).sum(axis=-1)
    # Top-k smallest = top-k largest of negated distances.
    _, indices = lax.top_k(-d2, k)
    return indices


def bilateral_gaussian(
    values: Float[Array, 'n d_v'],
    features: Float[Array, 'n d_f'],
    *,
    sigma_features: Float[Array, 'd_f'],
    neighbourhood: Union[int, Int[Array, 'n k_max']],
    backend: Backend = 'auto',
) -> Float[Array, 'n d_v']:
    '''Direct N-body bilateral Gaussian smoothing.

    For each output position ``i``, the result is the
    feature-distance-weighted average of ``values`` over the
    neighbourhood of ``i``::

        out[i, :] = (1 / Z[i]) * sum_p w[i, p] * values[indices[i, p], :]
        w[i, p]   = exp(-0.5 * sum_d ((features[i, d] - features[indices[i, p], d]) / sigma_features[d])**2)
        Z[i]      = sum_p w[i, p]

    Parameters
    ----------
    values
        Per-point values to smooth, ``(n, d_v)``.  Can be any
        signal: image intensity, multi-channel features, etc.
    features
        Per-point feature vectors, ``(n, d_f)``.  Typically a
        concatenation of spatial coordinates and intensity (or
        other modality channels).  The bilateral metric is over
        these features, not over the values being smoothed.
    sigma_features
        Per-feature standard deviation, ``(d_f,)``.  Controls how
        quickly the weight falls off in each feature dimension --
        large ``sigma`` means "this feature contributes little to
        edge detection".
    neighbourhood
        Either an ``int`` ``k`` (do brute-force k-NN in feature
        space; quadratic memory in ``n``) or an explicit
        ``(n, k_max)`` index array of pre-computed neighbours.
        Use the latter for ``n > ~10k`` or for non-Euclidean
        neighbourhood structures.
    backend
        ``"auto"``, ``"pallas-cuda"``, or ``"jax"``.  Passed to
        the underlying ``semiring_ell_matmul``; currently routes
        to JAX (the ELL Pallas kernel is the open G0 item).

    Returns
    -------
    Smoothed values, ``(n, d_v)``.

    Notes
    -----
    The implementation is a direct N-body reduction: the inner sum
    has ``k_max`` terms, and we explicitly evaluate the Gaussian
    weight per term.  This is the "no clever algorithmic trick" path
    and is correct by construction.  Cost is ``O(n * k_max * (d_f +
    d_v))`` per call.

    For large spatial extents or large ``d_f``, the
    ``permutohedral_lattice`` (Phase 4 stretch goal) achieves linear
    time via hashing.  The bilateral path here is the marquee
    capability promised at first GA regardless of whether
    permutohedral lands.
    '''
    n, d_v = values.shape
    n2, d_f = features.shape
    if n != n2:
        raise ValueError(
            f'values has n={n}, features has n={n2}; must match.'
        )
    sigma_features = jnp.asarray(sigma_features)
    if sigma_features.shape != (d_f,):
        raise ValueError(
            f'sigma_features.shape={sigma_features.shape} must equal '
            f'(d_f,)=({d_f},).'
        )

    # Resolve the neighbourhood adjacency.
    if isinstance(neighbourhood, int):
        indices = brute_force_knn(
            features, neighbourhood, sigma_features=sigma_features,
        )
    else:
        indices = jnp.asarray(neighbourhood, dtype=jnp.int32)
        if indices.shape[0] != n:
            raise ValueError(
                f'neighbourhood.shape[0]={indices.shape[0]} must '
                f'equal n={n}.'
            )

    # Gather neighbour features and compute Gaussian weights.
    # features[indices]: (n, k_max, d_f)
    feat_n = features[indices]
    # Per-feature normalised distance, then squared distance summed.
    rescaled_diff = (features[:, None, :] - feat_n) / sigma_features
    d2 = (rescaled_diff ** 2).sum(axis=-1)              # (n, k_max)
    weights = jnp.exp(-0.5 * d2)                        # (n, k_max)

    # Normalise weights so each row sums to 1.  This is the standard
    # bilateral normalisation; without it the smoothed values would
    # drift toward zero as the weights generally don't sum to 1.
    Z = weights.sum(axis=-1, keepdims=True)
    weights = weights / jnp.maximum(Z, jnp.finfo(weights.dtype).tiny)

    # Reduce via semiring_ell_matmul: out[i, j] = sum_p w[i, p] *
    # values[indices[i, p], j].
    return semiring_ell_matmul(
        weights, indices, values,
        semiring=REAL, n_cols=n, backend=backend,
    )
