# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Transform models: the chart a matrix-transform recipe optimises over.

A ``TransformModel`` is the bridge between the flat Lie-parameter vector
the Gauss-Newton / Levenberg-Marquardt optimiser moves and the
homogeneous transform matrix ``affine_grid`` + ``spatial_transform``
apply -- plus the two pieces of parameter-layout knowledge the
coarse-to-fine driver needs (``n_params`` and the inter-level translation
rescale).  Holding that layout on the model (rather than the driver
slicing ``params[-ndim:]``) is what lets a new parametrisation -- a 7-DOF
similarity, a constrained rigid, the rigid block a BBR recipe optimises
-- drop in as one frozen record without touching the driver.

This mirrors the ``geometry._interpolate.Interpolator`` precedent: a
structural ``Protocol`` with immutable ``@dataclass(frozen=True)``
implementers (``Rigid`` / ``Affine``), hashable so they ride ``jit``
static arguments on the ``RegistrationSpec``.

The dense-velocity-field family (diffeomorphic / SyN) is *not* a
``TransformModel`` -- its parameter is a field, not a small vector, and it
lowers onto the SVF driver, not this matrix-transform one.
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
        """Length of the parameter vector for a ``ndim``-D transform."""
        ...

    def exp(
        self,
        params: Float[Array, '... p'],
        *,
        ndim: int,
    ) -> Float[Array, '... d1 d1']:
        """Homogeneous ``(ndim + 1, ndim + 1)`` matrix of ``params``."""
        ...

    def rescale_to_grid(
        self,
        params: Float[Array, ' p'],
        ratio: Float[Array, ' d'],
    ) -> Float[Array, ' p']:
        """Rescale the parameters from one voxel grid to a finer one.

        ``ratio`` is the per-axis ``finer_shape / coarser_shape``; the
        returned parameters reproduce the same physical transform on the
        finer grid (translations are in voxel units, so they scale).
        """
        ...


def _rescale_translation_last(
    params: Float[Array, ' p'],
    ratio: Float[Array, ' d'],
) -> Float[Array, ' p']:
    """Scale the trailing ``len(ratio)`` (translation) parameters by
    ``ratio`` -- the layout shared by ``Rigid`` and ``Affine`` (rotation
    / linear block first, translation last)."""
    n = ratio.shape[0]
    return jnp.concatenate([params[:-n], params[-n:] * ratio])


@dataclass(frozen=True)
class Rigid:
    """Rigid (rotation + translation) transform on ``SE(2)`` / ``SE(3)``.

    Parameters: 2-D ``(theta, t_x, t_y)`` (3); 3-D ``(omega_x, omega_y,
    omega_z, t_x, t_y, t_z)`` (6) -- axis-angle rotation then
    translation.  Closed-form exponential (Rodrigues), GPU-native.
    """

    def n_params(self, ndim: int) -> int:
        return (1 if ndim == 2 else 3) + ndim

    def exp(
        self,
        params: Float[Array, '... p'],
        *,
        ndim: int,
    ) -> Float[Array, '... d1 d1']:
        return rigid_exp(params, ndim=ndim)

    def rescale_to_grid(
        self,
        params: Float[Array, ' p'],
        ratio: Float[Array, ' d'],
    ) -> Float[Array, ' p']:
        return _rescale_translation_last(params, ratio)


@dataclass(frozen=True)
class Affine:
    """Affine (general linear + translation) transform.

    Parameters: ``ndim**2`` row-major linear-generator entries then
    ``ndim`` translation (12 in 3-D, 6 in 2-D).  The linear block is
    ``matrix_exp`` of the generator, guaranteeing an invertible,
    orientation-preserving map.
    """

    def n_params(self, ndim: int) -> int:
        return ndim * ndim + ndim

    def exp(
        self,
        params: Float[Array, '... p'],
        *,
        ndim: int,
    ) -> Float[Array, '... d1 d1']:
        return affine_exp(params, ndim=ndim)

    def rescale_to_grid(
        self,
        params: Float[Array, ' p'],
        ratio: Float[Array, ' d'],
    ) -> Float[Array, ' p']:
        return _rescale_translation_last(params, ratio)
