# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Transform models: the chart a matrix-transform recipe optimises over.

A :class:`TransformModel` is the bridge between the flat Lie-parameter
vector the Gauss-Newton / Levenberg-Marquardt optimiser moves and the
homogeneous transform matrix that :func:`affine_grid` and
:func:`spatial_transform` apply -- plus the two pieces of
parameter-layout knowledge the coarse-to-fine driver needs (the
parameter count and the inter-level translation rescale). Holding that
layout on the model (rather than the driver slicing the trailing
translation block) is what lets a new parametrisation -- a 7-DOF
similarity, a constrained rigid, the rigid block a boundary-based
recipe optimises -- drop in as one frozen record without touching the
driver.

This mirrors the interpolator precedent: a structural
:class:`~typing.Protocol` with immutable, frozen dataclass implementers
(:class:`Rigid` and :class:`Affine`), hashable so they ride ``jit``
static arguments on the registration specification.

The dense-velocity-field family (diffeomorphic / symmetric normalisation)
is *not* a :class:`TransformModel` -- its parameter is a field, not a
small vector, and it lowers onto the stationary-velocity-field driver,
not this matrix-transform one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import jax.numpy as jnp
from jaxtyping import Array, Float

from ..geometry import affine_exp, rigid_exp

__all__ = [
    'TransformModel',
    'Rigid',
    'Affine',
]


class TransformModel(Protocol):
    """Chart between Lie parameters and a homogeneous transform.

    An implementer exposes the exponential map (parameters -> matrix),
    the parameter count for a given spatial rank, and the inter-level
    translation rescale the coarse-to-fine driver applies when it
    warm-starts a finer pyramid level.
    """

    def n_params(self, ndim: int) -> int:
        """Length of the parameter vector for an ``ndim``-D transform.

        Parameters
        ----------
        ndim : int
            Spatial rank of the transform (2 or 3).

        Returns
        -------
        int
            Number of scalar entries in the flat parameter vector the
            optimiser moves for this parametrisation at rank ``ndim``.
        """
        ...

    def exp(
        self,
        params: Float[Array, '... p'],
        *,
        ndim: int,
    ) -> Float[Array, '... d1 d1']:
        """Exponential map from parameters to a homogeneous matrix.

        Parameters
        ----------
        params : Float[Array, '... p']
            Flat Lie-parameter vector (with optional leading batch
            axes) whose length matches :meth:`n_params` for ``ndim``.
        ndim : int
            Spatial rank of the transform (2 or 3).

        Returns
        -------
        Float[Array, '... d1 d1']
            Homogeneous :math:`(\\text{ndim} + 1, \\text{ndim} + 1)`
            transform matrix (``d1`` being ``ndim + 1``), broadcasting
            over any leading batch axes of ``params``.
        """
        ...

    def rescale_to_grid(
        self,
        params: Float[Array, ' p'],
        ratio: Float[Array, ' d'],
    ) -> Float[Array, ' p']:
        """Rescale the parameters from one voxel grid to a finer one.

        Used by the coarse-to-fine driver when it warm-starts a finer
        pyramid level: the returned parameters reproduce the same
        physical transform on the finer grid, since translations are
        expressed in voxel units and so scale with the grid.

        Parameters
        ----------
        params : Float[Array, ' p']
            Parameter vector fitted on the coarser grid.
        ratio : Float[Array, ' d']
            Per-axis ``finer_shape / coarser_shape`` scaling factor.

        Returns
        -------
        Float[Array, ' p']
            Parameter vector expressed on the finer grid.
        """
        ...


def _rescale_translation_last(
    params: Float[Array, ' p'],
    ratio: Float[Array, ' d'],
) -> Float[Array, ' p']:
    """Scale the trailing translation parameters by ``ratio``.

    Implements the grid rescale for the layout shared by :class:`Rigid`
    and :class:`Affine`, where the rotation or linear block comes first
    and the translation last: the final ``len(ratio)`` entries are
    multiplied by ``ratio`` and the leading block is left unchanged.

    Parameters
    ----------
    params : Float[Array, ' p']
        Parameter vector with the translation as its trailing block.
    ratio : Float[Array, ' d']
        Per-axis ``finer_shape / coarser_shape`` scaling factor; its
        length is the number of trailing (translation) entries scaled.

    Returns
    -------
    Float[Array, ' p']
        Parameter vector with the translation block rescaled.
    """
    n = ratio.shape[0]
    return jnp.concatenate([params[:-n], params[-n:] * ratio])


@dataclass(frozen=True)
class Rigid:
    """Rigid (rotation + translation) transform on ``SE(2)`` / ``SE(3)``.

    The parameter vector is an axis-angle rotation followed by a
    translation: in 2-D :math:`(\\theta, t_x, t_y)` (3 entries) and in
    3-D :math:`(\\omega_x, \\omega_y, \\omega_z, t_x, t_y, t_z)`
    (6 entries). The exponential map is closed-form (Rodrigues' formula)
    and GPU-native.
    """

    def n_params(self, ndim: int) -> int:
        """Parameter count: one rotation angle plus ``ndim``
        translations in 2-D, or three axis-angle rotations plus
        ``ndim`` translations in 3-D."""
        return (1 if ndim == 2 else 3) + ndim

    def exp(
        self,
        params: Float[Array, '... p'],
        *,
        ndim: int,
    ) -> Float[Array, '... d1 d1']:
        """Closed-form rigid exponential map via :func:`rigid_exp`,
        returning the homogeneous
        :math:`(\\text{ndim} + 1, \\text{ndim} + 1)` transform."""
        return rigid_exp(params, ndim=ndim)

    def rescale_to_grid(
        self,
        params: Float[Array, ' p'],
        ratio: Float[Array, ' d'],
    ) -> Float[Array, ' p']:
        """Rescale to a finer grid by scaling the trailing translation
        block by ``ratio``; the rotation parameters are grid-invariant
        and left unchanged."""
        return _rescale_translation_last(params, ratio)


@dataclass(frozen=True)
class Affine:
    """Affine (general linear + translation) transform.

    The parameter vector holds ``ndim**2`` row-major linear-generator
    entries followed by ``ndim`` translation entries (12 in 3-D, 6 in
    2-D). The linear block is the matrix exponential (:func:`matrix_exp`)
    of the generator, guaranteeing an invertible, orientation-preserving
    map.
    """

    def n_params(self, ndim: int) -> int:
        """Parameter count: ``ndim**2`` linear-generator entries plus
        ``ndim`` translations."""
        return ndim * ndim + ndim

    def exp(
        self,
        params: Float[Array, '... p'],
        *,
        ndim: int,
    ) -> Float[Array, '... d1 d1']:
        """Affine exponential map via :func:`affine_exp`, returning the
        homogeneous :math:`(\\text{ndim} + 1, \\text{ndim} + 1)`
        transform whose linear block is the exponential of the
        generator."""
        return affine_exp(params, ndim=ndim)

    def rescale_to_grid(
        self,
        params: Float[Array, ' p'],
        ratio: Float[Array, ' d'],
    ) -> Float[Array, ' p']:
        """Rescale to a finer grid by scaling the trailing translation
        block by ``ratio``; the linear-generator block is
        grid-invariant and left unchanged."""
        return _rescale_translation_last(params, ratio)
