# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Feature-space metrics for the bounded bilateral filter.

``bilateral_gaussian`` weights neighbours by a Gaussian over a metric
on feature space::

    w_ij = exp(-1/2 * (f_i - f_j)^T M (f_i - f_j))

where ``M >= 0`` is supplied **factored** as ``M = L L^T`` with
``L`` of shape ``(d_f, k)``, ``k <= d_f``.  The kernel never needs
``M`` itself: every weight depends on the feature difference only
through the projection ``z = L^T (f_i - f_j)`` and the squared norm
``q = z^T z = (f_i - f_j)^T M (f_i - f_j)``.  This module supplies that
projection as a small ADT.

Three tiers, all feeding the identical kernel (only the shape / source
of the factor changes):

- ``DiagonalMetric(sigma)`` -- one bandwidth per feature; ``M`` is
  ``diag(1 / sigma**2)``.  The cheap, interpretable default; recovers
  the classic per-channel bilateral.
- ``FactorMetric(factor)`` -- a general factor ``L`` of shape
  ``(d_f, k)``.  ``k < d_f`` gives a **low-rank** metric that projects
  correlated feature channels into a ``k``-dimensional discriminative
  subspace before the norm (cost ``O(k)`` per neighbour rather than
  ``O(d_f)``); ``k == d_f`` gives a general anisotropic metric.

Constructors compose the tiers without leaving the substrate:
``block_diagonal_metric`` assembles a block-diagonal ``L`` (independent
per-modality bandwidths -- e.g. a tissue-intensity block weighted
separately from a functional-correlation block), and ``metric_from_spd``
factors an explicit SPD ``M`` via Cholesky.

Both metric types are registered JAX pytrees whose array field is the
single leaf, so a metric flows through ``jit`` / ``vmap`` and is
differentiable end-to-end: the factor ``L`` is a learnable parameter if
the consumer wants one.  Data-driven *fitting* of ``L`` (population PCA,
supervised metric learning) is deliberately out of scope -- that is a
modelling concern for a consumer, built from ``nitrix.stats.covariance``
and ``nitrix.linalg``; this module supplies only the pure mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence, Tuple, runtime_checkable

import jax
import jax.numpy as jnp
from jax.scipy.linalg import block_diag
from jaxtyping import Array, Float

__all__ = [
    'FeatureMetric',
    'DiagonalMetric',
    'FactorMetric',
    'block_diagonal_metric',
    'metric_from_spd',
]


# A ``FeatureMetric`` is anything exposing ``project``: it maps a stack
# of feature differences ``(..., d_f)`` to a projection ``(..., k)``
# whose squared L2 norm is the Mahalanobis quadratic form
# ``(f_i - f_j)^T M (f_i - f_j)``.  We rely on structural conformance
# rather than inheritance so the concrete records below stay plain
# frozen dataclasses (clean pytree registration, no metaclass / MRO
# interaction with ``Protocol``).
@runtime_checkable
class FeatureMetric(Protocol):
    """Projection of feature differences into a metric subspace.

    The only operation the bilateral kernel requires of a metric:
    given feature differences ``deltas`` of shape ``(..., d_f)``,
    return ``project(deltas)`` of shape ``(..., k)`` such that
    ``sum(project(deltas)**2, axis=-1)`` equals the quadratic form
    ``deltas^T M deltas`` for the metric ``M``.
    """

    def project(
        self,
        deltas: Float[Array, '... d_f'],
    ) -> Float[Array, '... k']: ...


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class DiagonalMetric:
    """Per-feature isotropic-in-each-axis metric ``M = diag(1 / sigma**2)``.

    The classic bilateral metric: each feature channel ``d`` is divided
    by its own bandwidth ``sigma[d]`` before the squared distance is
    accumulated.  A large ``sigma[d]`` means "channel ``d`` contributes
    little to the weight" (the filter is near-blind to that feature);
    in the limit ``sigma[d] -> inf`` the channel drops out entirely.

    Attributes
    ----------
    sigma
        Per-feature bandwidth, shape ``(d_f,)``.  Strictly positive.
    """

    sigma: Float[Array, 'd_f']

    def project(
        self,
        deltas: Float[Array, '... d_f'],
    ) -> Float[Array, '... d_f']:
        return deltas / self.sigma

    def tree_flatten(
        self,
    ) -> Tuple[Tuple[Float[Array, 'd_f']], None]:
        return (self.sigma,), None

    @classmethod
    def tree_unflatten(
        cls,
        _aux: None,
        children: Tuple[Any, ...],
    ) -> 'DiagonalMetric':
        return cls(*children)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class FactorMetric:
    """Factored metric ``M = L L^T`` with factor ``L`` of shape ``(d_f, k)``.

    ``project(deltas) = deltas @ L`` maps each ``(..., d_f)`` feature
    difference to a ``(..., k)`` projection, and the bilateral weight
    uses ``q = sum((deltas @ L)**2, -1) = deltas^T (L L^T) deltas``.

    - ``k == d_f`` -- a general anisotropic metric (cross-channel
      coupling, e.g. ``L = chol(M)`` from :func:`metric_from_spd`).
    - ``k < d_f`` -- a **low-rank** metric: correlated channels are
      projected into a ``k``-dimensional subspace before the norm, so
      the per-neighbour weight costs ``O(k)`` rather than ``O(d_f)``.

    Attributes
    ----------
    factor
        The factor ``L``, shape ``(d_f, k)`` with ``k <= d_f``.
    """

    factor: Float[Array, 'd_f k']

    def project(
        self,
        deltas: Float[Array, '... d_f'],
    ) -> Float[Array, '... k']:
        return deltas @ self.factor

    def tree_flatten(
        self,
    ) -> Tuple[Tuple[Float[Array, 'd_f k']], None]:
        return (self.factor,), None

    @classmethod
    def tree_unflatten(
        cls,
        _aux: None,
        children: Tuple[Any, ...],
    ) -> 'FactorMetric':
        return cls(*children)


def block_diagonal_metric(
    blocks: Sequence[Float[Array, 'd_b k_b']],
) -> FactorMetric:
    """Assemble a block-diagonal factor from per-modality factor blocks.

    Each modality (e.g. a tissue-intensity channel set, a
    network-correlation channel set) contributes its own factor block
    ``L_b`` of shape ``(d_b, k_b)``; the assembled factor is the
    block-diagonal ``L = blkdiag(L_0, L_1, ...)`` of shape
    ``(sum d_b, sum k_b)``.  Because the blocks share no rows or
    columns, the resulting metric weights each modality's similarity
    independently -- no cross-modality coupling -- which is the
    recommended default when feature channels group into distinct
    modalities.

    Parameters
    ----------
    blocks
        Per-modality factor blocks, each a 2-D array ``(d_b, k_b)``.
        A diagonal block is just ``jnp.diag(1 / sigma_b)``; a dense
        anisotropic block is ``chol(M_b)`` (see :func:`metric_from_spd`).

    Returns
    -------
    ``FactorMetric`` whose ``factor`` is the block-diagonal assembly.

    Raises
    ------
    ValueError
        If ``blocks`` is empty or any block is not 2-D.
    """
    if len(blocks) == 0:
        raise ValueError('block_diagonal_metric: blocks must be non-empty.')
    arrays = [jnp.asarray(b) for b in blocks]
    for i, b in enumerate(arrays):
        if b.ndim != 2:
            raise ValueError(
                f'block_diagonal_metric: block {i} must be 2-D '
                f'(d_b, k_b); got shape {b.shape}.'
            )
    return FactorMetric(factor=block_diag(*arrays))


def metric_from_spd(
    spd: Float[Array, 'd_f d_f'],
) -> FactorMetric:
    """Factor an explicit SPD metric ``M`` into ``FactorMetric(chol(M))``.

    Use when the metric is naturally expressed as a full SPD matrix
    (a learned or estimated precision over feature space) rather than
    as a factor.  Returns the lower-triangular Cholesky factor ``L``
    with ``M = L L^T``, so ``project(d) = d @ L`` reproduces
    ``d^T M d`` exactly.

    Parameters
    ----------
    spd
        Symmetric positive-definite metric ``M``, shape ``(d_f, d_f)``.

    Returns
    -------
    ``FactorMetric`` with ``factor = cholesky(M)`` (shape ``(d_f, d_f)``).

    Raises
    ------
    ValueError
        If ``spd`` is not a square 2-D array.

    Notes
    -----
    Numerical conditioning of ``M`` is the caller's responsibility;
    ``nitrix.linalg`` provides reconditioning helpers if a near-singular
    metric must be regularised before factoring.
    """
    spd = jnp.asarray(spd)
    if spd.ndim != 2 or spd.shape[0] != spd.shape[1]:
        raise ValueError(
            f'metric_from_spd: spd must be square 2-D (d_f, d_f); got '
            f'shape {spd.shape}.'
        )
    return FactorMetric(factor=jnp.linalg.cholesky(spd))
