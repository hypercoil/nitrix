# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Bounded bilateral smoothing via the semiring substrate.

A true high-dimensional bilateral filter over a **bounded**
neighbourhood.  For each output position, gather the feature-space
neighbourhood, weight by a Gaussian over a metric on feature space,
normalise, and reduce via :func:`~nitrix.semiring.semiring_ell_matmul`
with the REAL semiring:

.. math::

    \\mathrm{out}_{i,:} &= \\frac{1}{Z_i} \\sum_p w_{i,p}\\,
        \\mathrm{values}_{\\mathrm{neighbours}_{i,p},\\,:} \\\\
    w_{i,p} &= \\exp\\!\\left(-\\tfrac{1}{2}\\,
        (f_i - f_j)^{\\top} M (f_i - f_j)\\right)
        \\quad [\\,j = \\text{neighbour } p\\,] \\\\
    Z_i &= \\sum_p w_{i,p}

The whole filter is one gather plus one weighted reduction: statically
shaped, ``jit`` / ``vmap`` / ``grad`` clean, and GPU-native.  The
weight is a smooth :math:`\\exp` of a quadratic form over a *fixed*
bounded neighbourhood, so the gradient is smooth everywhere (no sort,
no scatter, no simplex-identity branch).  This is a bounded-support
filter for moderate feature dimensionalities and, via a low-rank
metric, well beyond.

Three independent choices parameterise a call:

- **Values vs. features are decoupled.** ``values`` (e.g. a BOLD time
  series, :math:`d_v` frames) and ``features`` (the multimodal
  signature set, :math:`d_f` channels) are separate arguments with
  separate column counts.  The common "filter the features by
  themselves" case is just ``values is features`` at the call site.
- **The metric** :math:`M = L L^{\\top}` is supplied as a
  :class:`~nitrix.smoothing.metric.FeatureMetric`: diagonal per-channel
  bandwidths, a low-rank projection of correlated channels, or a full
  anisotropic factor.  In the limit of a large bandwidth on a channel,
  that channel drops out and the filter degrades gracefully toward a
  spatial-only Gaussian.
- **The neighbourhood** is bounded: an ``int`` ``k`` (brute-force k-NN
  in feature space), an explicit ``(n, k_max)`` index array, or an
  :class:`~nitrix.sparse.ell.ELL` adjacency (grid box, mesh k-ring,
  geodesic ball).  Padded / ragged neighbourhoods carry a validity
  ``mask`` so padding contributes nothing -- which also removes the
  double-counting a naive padded gather would incur at boundaries and
  at low-degree mesh vertices.

``n_iters > 1`` grows the effective support cheaply: the affinity graph
(features, neighbours, weights) is fixed across iterations and only the
values diffuse, so the normalised weights are built once and re-applied
``n_iters`` times -- ``n_iters`` mean-field updates of a bounded dense
CRF at :math:`O(n_{\\mathrm{iters}} \\cdot n \\cdot k_{\\max} \\cdot d_v)`
for the diffusion on top of a single
:math:`O(n \\cdot k_{\\max} \\cdot k)` weight build.

References
----------
.. [1] Krähenbühl, P., & Koltun, V. (2011). Efficient inference in
   fully connected CRFs with Gaussian edge potentials. *Advances in
   Neural Information Processing Systems*, 24.
   https://arxiv.org/abs/1210.5644
"""

from __future__ import annotations

from typing import Optional, Union

import jax.lax as lax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from .._internal.backend import Backend
from ..semiring import REAL, semiring_ell_matmul
from ..sparse.ell import ELL
from .metric import FeatureMetric

__all__ = ['bilateral_gaussian', 'brute_force_knn']


def brute_force_knn(
    features: Float[Array, 'n d_f'],
    k: int,
    *,
    metric: Optional[FeatureMetric] = None,
) -> Int[Array, 'n k']:
    """Brute-force k-nearest-neighbour search in (optionally projected) feature space.

    Materialises the :math:`(n, n)` distance matrix; quadratic in
    memory.  Practical for :math:`n \\lesssim 10\\mathrm{k}`.  For larger
    :math:`n`, the caller should pre-compute the adjacency with a
    spatial-index data structure (KD-tree, grid hashing, mesh k-ring,
    etc.) and pass it as ``neighbourhood=`` to
    :func:`bilateral_gaussian`.

    Parameters
    ----------
    features
        Per-point feature vectors, ``(n, d_f)``.
    k
        Number of neighbours per query point.  The point itself is
        included in its own ``k`` (so ``k=1`` returns self-only).
    metric
        Optional :class:`~nitrix.smoothing.metric.FeatureMetric`.
        ``None`` ranks by raw Euclidean distance; when supplied, the
        search ranks by ``sum(metric.project(diff)**2, -1)`` -- the same
        quadratic form :func:`bilateral_gaussian` uses for its weights,
        so the k-NN adjacency matches the weight metric.

    Returns
    -------
    Int[Array, 'n k']
        Indices of the ``k`` nearest neighbours for each query point,
        shape ``(n, k)``, sorted by ascending distance.
    """
    # Pairwise feature differences: (n, n, d_f).
    diff = features[:, None, :] - features[None, :, :]
    if metric is not None:
        proj = metric.project(diff)
    else:
        proj = diff
    d2 = (proj**2).sum(axis=-1)
    # Top-k smallest = top-k largest of negated distances.
    _, indices = lax.top_k(-d2, k)
    return indices


def bilateral_gaussian(
    values: Float[Array, 'n d_v'],
    features: Float[Array, 'n d_f'],
    *,
    metric: FeatureMetric,
    neighbourhood: Union[int, Int[Array, 'n k_max'], ELL],
    mask: Optional[Bool[Array, 'n k_max']] = None,
    n_iters: int = 1,
    backend: Backend = 'auto',
) -> Float[Array, 'n d_v']:
    """Bounded bilateral smoothing over a feature-space metric.

    For each output position ``i``, the result is the
    metric-distance-weighted average of ``values`` over the bounded
    neighbourhood of ``i`` (see the module docstring for the
    formulation).

    Parameters
    ----------
    values
        Per-point values to smooth, ``(n, d_v)``.  Any signal: image
        intensity, a BOLD time series, multi-channel features,
        segmentation logits, etc.  Independent of ``features``.
    features
        Per-point feature vectors, ``(n, d_f)``.  Typically spatial
        coordinates concatenated with intensity / modality / functional
        channels.  The bilateral metric is over these, not over the
        values being smoothed.
    metric
        A :class:`~nitrix.smoothing.metric.FeatureMetric` defining
        :math:`M = L L^{\\top}`.  The weight falls off as
        :math:`\\exp\\!\\left(-\\tfrac{1}{2}\\sum_c
        (\\mathrm{metric.project}(f_i - f_j))_c^2\\right)`.
    neighbourhood
        The bounded neighbour source, one of:

        - ``int`` ``k`` -- brute-force k-NN in feature space (quadratic
          memory in ``n``); the metric is applied to the ranking.
        - ``(n, k_max)`` index array -- explicit pre-computed neighbours.
        - :class:`~nitrix.sparse.ell.ELL` -- an adjacency operator (grid
          box, mesh k-ring, geodesic ball).  Its ``indices`` are used;
          when ``mask`` is not given, the validity mask is derived from
          its padding (``values != identity``).
    mask
        Optional boolean validity mask ``(n, k_max)``: ``True`` for real
        neighbours, ``False`` for padding.  Masked positions contribute
        nothing to the weighted average.  Defaults to "all valid" for
        the ``int`` and explicit-index paths, and to the ELL's padding
        validity for the ``ELL`` path.  Supplying it explicitly always
        overrides the derived mask.
    n_iters
        Number of bilateral passes.  The affinity (features, neighbours,
        weights) is held fixed; only the values diffuse, so the weights
        are computed once and re-applied.  ``n_iters`` applications of a
        radius-``r`` filter give ~radius-``(r * n_iters)`` support at
        ``O(n_iters)`` cost.  Default ``1``.
    backend
        ``"auto"``, ``"pallas-cuda"``, or ``"jax"``.  Passed to the
        underlying :func:`~nitrix.semiring.semiring_ell_matmul`.

    Returns
    -------
    Float[Array, 'n d_v']
        Smoothed values, ``(n, d_v)``.

    Notes
    -----
    The implementation is a direct N-body reduction: the inner sum has
    ``k_max`` terms, each an explicit Gaussian weight over the metric.
    Correct by construction; cost
    :math:`O(n \\cdot k_{\\max} \\cdot (k + d_v))` per pass, where
    :math:`k` is the metric's projected dimension (:math:`\\leq d_f`).
    """
    if n_iters < 1:
        raise ValueError(f'n_iters must be >= 1; got {n_iters}.')

    n, d_v = values.shape
    n2, d_f = features.shape
    if n != n2:
        raise ValueError(f'values has n={n}, features has n={n2}; must match.')

    # Resolve the neighbourhood to (indices, mask).  An explicit ``mask``
    # argument always wins; otherwise an ELL contributes its validity and
    # the int / array paths default to "all valid".
    derived_mask: Optional[Bool[Array, 'n k_max']] = None
    if isinstance(neighbourhood, int):
        indices = brute_force_knn(features, neighbourhood, metric=metric)
    elif isinstance(neighbourhood, ELL):
        if neighbourhood.n_cols != n:
            raise ValueError(
                f'ELL neighbourhood has n_cols={neighbourhood.n_cols}; '
                f'must equal n={n} (a self-adjacency over the points).'
            )
        indices = jnp.asarray(neighbourhood.indices, dtype=jnp.int32)
        if neighbourhood.identity is not None:
            id_val = jnp.asarray(
                neighbourhood.identity,
                dtype=neighbourhood.values.dtype,
            )
            derived_mask = neighbourhood.values != id_val
    else:
        indices = jnp.asarray(neighbourhood, dtype=jnp.int32)
    if indices.shape[0] != n:
        raise ValueError(
            f'neighbourhood has n={indices.shape[0]} rows; must equal n={n}.'
        )

    if mask is None:
        mask = derived_mask
    elif tuple(mask.shape) != tuple(indices.shape):
        raise ValueError(
            f'mask.shape={tuple(mask.shape)} must equal the neighbour '
            f'shape {tuple(indices.shape)}.'
        )

    # Gather neighbour features and weight by the metric quadratic form.
    feat_n = features[indices]  # (n, k_max, d_f)
    diff = features[:, None, :] - feat_n  # (n, k_max, d_f)
    proj = metric.project(diff)  # (n, k_max, k)
    q = (proj**2).sum(axis=-1)  # (n, k_max)
    weights = jnp.exp(-0.5 * q)  # (n, k_max)
    if mask is not None:
        weights = weights * mask.astype(weights.dtype)  # null padding

    # Normalise so each row sums to 1 (standard bilateral normalisation);
    # a fully-masked / isolated row sums to 0 and yields a 0 output via
    # the guarded division.
    Z = weights.sum(axis=-1, keepdims=True)
    weights = weights / jnp.maximum(Z, jnp.finfo(weights.dtype).tiny)

    # Diffuse the values.  Affinity is fixed across iterations, so the
    # weights above are reused: out_{t+1} = W out_t.
    out = values
    for _ in range(n_iters):
        out = semiring_ell_matmul(
            weights,
            indices,
            out,
            semiring=REAL,
            n_cols=n,
            backend=backend,
        )
    return out
