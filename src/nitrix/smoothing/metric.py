# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Feature-space metrics for the bounded bilateral filter.

:func:`bilateral_gaussian` weights neighbours by a Gaussian over a metric
on feature space,

.. math::

    w_{ij} = \\exp\\left(
        -\\tfrac{1}{2}\\,(f_i - f_j)^{\\top} M (f_i - f_j)
    \\right)

where :math:`M \\succeq 0` is supplied **factored** as :math:`M = L L^{\\top}`
with :math:`L` of shape :math:`(d_f, k)`, :math:`k \\le d_f`.  The kernel
never needs :math:`M` itself: every weight depends on the feature
difference only through the projection :math:`z = L^{\\top} (f_i - f_j)` and
the squared norm :math:`q = z^{\\top} z = (f_i - f_j)^{\\top} M (f_i - f_j)`.
This module supplies that projection as a small algebraic data type.

Three tiers, all feeding the identical kernel (only the shape and source
of the factor changes):

- :class:`DiagonalMetric` -- one bandwidth per feature; :math:`M` is
  :math:`\\operatorname{diag}(1 / \\sigma^2)`.  The cheap, interpretable
  default; recovers the classic per-channel bilateral.
- :class:`FactorMetric` -- a general factor :math:`L` of shape
  :math:`(d_f, k)`.  :math:`k < d_f` gives a **low-rank** metric that
  projects correlated feature channels into a :math:`k`-dimensional
  discriminative subspace before the norm (cost :math:`O(k)` per neighbour
  rather than :math:`O(d_f)`); :math:`k = d_f` gives a general anisotropic
  metric.

Constructors compose the tiers without leaving the substrate:
:func:`block_diagonal_metric` assembles a block-diagonal :math:`L`
(independent per-modality bandwidths -- e.g. a tissue-intensity block
weighted separately from a functional-correlation block), and
:func:`metric_from_spd` factors an explicit SPD :math:`M` via Cholesky.

Both metric types are registered JAX pytrees whose array field is the
single leaf, so a metric flows through ``jit`` / ``vmap`` and is
differentiable end-to-end: the factor :math:`L` is a learnable parameter
if the consumer wants one.  Data-driven *fitting* of :math:`L` (population
PCA, supervised metric learning) is deliberately out of scope -- that is a
modelling concern for a consumer, built from :mod:`nitrix.stats` covariance
tools and :mod:`nitrix.linalg`; this module supplies only the pure
mechanism.
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
    given feature differences ``deltas`` of shape :math:`(\\dots, d_f)`,
    return :meth:`project` of shape :math:`(\\dots, k)` such that
    ``sum(project(deltas)**2, axis=-1)`` equals the quadratic form
    :math:`\\text{deltas}^{\\top} M\\,\\text{deltas}` for the metric
    :math:`M`.
    """

    def project(
        self,
        deltas: Float[Array, '... d_f'],
    ) -> Float[Array, '... k']:
        """Project feature differences into the metric subspace.

        Parameters
        ----------
        deltas
            Stack of feature differences, shape :math:`(\\dots, d_f)`,
            where the trailing axis indexes the :math:`d_f` feature
            channels.

        Returns
        -------
        Float[Array, '... k']
            The projection, shape :math:`(\\dots, k)`, whose squared L2
            norm over the trailing axis equals the Mahalanobis quadratic
            form :math:`\\text{deltas}^{\\top} M\\,\\text{deltas}`.
        """
        ...


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class DiagonalMetric:
    """Per-feature axis-aligned metric :math:`M = \\operatorname{diag}(1 / \\sigma^2)`.

    The classic bilateral metric: each feature channel :math:`d` is
    divided by its own bandwidth :math:`\\sigma_d` before the squared
    distance is accumulated.  A large :math:`\\sigma_d` means channel
    :math:`d` contributes little to the weight (the filter is near-blind
    to that feature); in the limit :math:`\\sigma_d \\to \\infty` the
    channel drops out entirely.

    Attributes
    ----------
    sigma
        Per-feature bandwidth, shape :math:`(d_f,)`.  Strictly positive.
    """

    sigma: Float[Array, 'd_f']

    def project(
        self,
        deltas: Float[Array, '... d_f'],
    ) -> Float[Array, '... d_f']:
        """Divide each feature difference by its per-channel bandwidth.

        Parameters
        ----------
        deltas
            Stack of feature differences, shape :math:`(\\dots, d_f)`,
            with the trailing axis indexing the :math:`d_f` feature
            channels.

        Returns
        -------
        Float[Array, '... d_f']
            The elementwise ratio ``deltas / sigma``, shape
            :math:`(\\dots, d_f)`.  Its squared L2 norm over the trailing
            axis is :math:`\\sum_d (\\text{deltas}_d / \\sigma_d)^2`, the
            Mahalanobis quadratic form for
            :math:`M = \\operatorname{diag}(1 / \\sigma^2)`.
        """
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
    """Factored metric :math:`M = L L^{\\top}` with factor :math:`L` of shape :math:`(d_f, k)`.

    :meth:`project` computes ``deltas @ L``, mapping each
    :math:`(\\dots, d_f)` feature difference to a :math:`(\\dots, k)`
    projection, and the bilateral weight uses
    :math:`q = \\sum (\\text{deltas} \\cdot L)^2 = \\text{deltas}^{\\top} (L L^{\\top}) \\,\\text{deltas}`.

    - :math:`k = d_f` -- a general anisotropic metric (cross-channel
      coupling, e.g. :math:`L = \\operatorname{chol}(M)` from
      :func:`metric_from_spd`).
    - :math:`k < d_f` -- a **low-rank** metric: correlated channels are
      projected into a :math:`k`-dimensional subspace before the norm, so
      the per-neighbour weight costs :math:`O(k)` rather than
      :math:`O(d_f)`.

    Attributes
    ----------
    factor
        The factor :math:`L`, shape :math:`(d_f, k)` with
        :math:`k \\le d_f`.
    """

    factor: Float[Array, 'd_f k']

    def project(
        self,
        deltas: Float[Array, '... d_f'],
    ) -> Float[Array, '... k']:
        """Project feature differences through the factor :math:`L`.

        Parameters
        ----------
        deltas
            Stack of feature differences, shape :math:`(\\dots, d_f)`,
            with the trailing axis indexing the :math:`d_f` feature
            channels.

        Returns
        -------
        Float[Array, '... k']
            The projection ``deltas @ factor``, shape
            :math:`(\\dots, k)`.  Its squared L2 norm over the trailing
            axis equals the Mahalanobis quadratic form
            :math:`\\text{deltas}^{\\top} (L L^{\\top}) \\,\\text{deltas}`.
        """
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
    :math:`L_b` of shape :math:`(d_b, k_b)`; the assembled factor is the
    block-diagonal :math:`L = \\operatorname{blkdiag}(L_0, L_1, \\dots)`
    of shape :math:`(\\sum_b d_b, \\sum_b k_b)`.  Because the blocks share
    no rows or columns, the resulting metric weights each modality's
    similarity independently -- no cross-modality coupling -- which is the
    recommended default when feature channels group into distinct
    modalities.

    Parameters
    ----------
    blocks
        Per-modality factor blocks, each a 2-D array of shape
        :math:`(d_b, k_b)`.  A diagonal block is just
        ``jnp.diag(1 / sigma_b)``; a dense anisotropic block is
        :math:`\\operatorname{chol}(M_b)` (see :func:`metric_from_spd`).

    Returns
    -------
    FactorMetric
        A :class:`FactorMetric` whose ``factor`` is the block-diagonal
        assembly, shape :math:`(\\sum_b d_b, \\sum_b k_b)`.

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
    """Factor an explicit SPD metric :math:`M` into a :class:`FactorMetric`.

    Use when the metric is naturally expressed as a full SPD matrix
    (a learned or estimated precision over feature space) rather than
    as a factor.  Returns a :class:`FactorMetric` holding the
    lower-triangular Cholesky factor :math:`L` with
    :math:`M = L L^{\\top}`, so :meth:`FactorMetric.project` on a
    difference :math:`d` yields ``d @ L`` and reproduces
    :math:`d^{\\top} M d` exactly.

    Parameters
    ----------
    spd
        Symmetric positive-definite metric :math:`M`, shape
        :math:`(d_f, d_f)`.

    Returns
    -------
    FactorMetric
        A :class:`FactorMetric` with ``factor = cholesky(spd)``, shape
        :math:`(d_f, d_f)`.

    Raises
    ------
    ValueError
        If ``spd`` is not a square 2-D array.

    Notes
    -----
    Numerical conditioning of :math:`M` is the caller's responsibility;
    :mod:`nitrix.linalg` provides reconditioning helpers if a
    near-singular metric must be regularised before factoring.
    """
    spd = jnp.asarray(spd)
    if spd.ndim != 2 or spd.shape[0] != spd.shape[1]:
        raise ValueError(
            f'metric_from_spd: spd must be square 2-D (d_f, d_f); got '
            f'shape {spd.shape}.'
        )
    return FactorMetric(factor=jnp.linalg.cholesky(spd))
