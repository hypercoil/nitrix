# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Pairwise rigid / affine registration recipes.

Pure functions ``(moving, fixed) -> RegistrationResult`` -- the ``reml_fit``
precedent (a ``NamedTuple`` of arrays, no PyTree module, no atlas / I/O).
They compose the R0/R1 substrate (pyramid + metric + transform
parametrisation + matrix-free optimiser); the orchestration lives here so
``entense`` can wrap it (or re-implement) without re-deriving it.

The representative algorithm is intensity-based Gauss-Newton /
Levenberg-Marquardt on SE(3)/affine (the ``3dvolreg`` / AIR lineage):
coarse-to-fine, second-order, differentiable.  The metric (SSD by
default; LNCC / MI / CR for intensity-robust or cross-modal cases) and
the schedule are set on the ``RegistrationSpec``.
"""

from __future__ import annotations

from jaxtyping import Array, Float

from ._core import (
    RegistrationResult,
    RegistrationSpec,
    multi_resolution_register,
)
from ._model import Affine, Rigid

__all__ = ['rigid_register', 'affine_register']


def _spatial_ndim(moving: Array, fixed: Array) -> int:
    ndim = moving.ndim
    if ndim not in (2, 3):
        raise ValueError(
            f'registration supports 2-D / 3-D single-channel images; '
            f'got shape {moving.shape}.'
        )
    return ndim


def rigid_register(
    moving: Float[Array, '*spatial'],
    fixed: Float[Array, '*spatial'],
    *,
    spec: RegistrationSpec = RegistrationSpec(),
) -> RegistrationResult:
    """Estimate the rigid transform aligning ``moving`` to ``fixed``.

    Optimises the 6-DOF (3-D) / 3-DOF (2-D) rigid Lie parameters
    (``geometry.rigid_exp``) coarse-to-fine so that ``moving`` resampled
    by the result matches ``fixed`` under ``spec.metric``.

    Parameters
    ----------
    moving, fixed
        Single-channel images of identical shape (2-D or 3-D).
    spec
        ``RegistrationSpec`` (pyramid depth, iterations, metric, ...).

    Returns
    -------
    ``RegistrationResult`` (``matrix``, ``params``, ``warped``,
    ``cost_history``).  ``matrix`` maps ``fixed`` coordinates to
    ``moving`` coordinates; ``warped`` is ``moving`` on the ``fixed``
    grid.
    """
    ndim = _spatial_ndim(moving, fixed)
    return multi_resolution_register(
        moving,
        fixed,
        model=Rigid(),
        ndim=ndim,
        spec=spec,
    )


def affine_register(
    moving: Float[Array, '*spatial'],
    fixed: Float[Array, '*spatial'],
    *,
    spec: RegistrationSpec = RegistrationSpec(),
) -> RegistrationResult:
    """Estimate the affine transform aligning ``moving`` to ``fixed``.

    Optimises the 12-DOF (3-D) / 6-DOF (2-D) affine Lie parameters
    (``geometry.affine_exp`` -- linear block via ``matrix_exp``,
    guaranteeing an invertible map) coarse-to-fine.  For a robust result
    on a large initial misalignment, run ``rigid_register`` first and
    pass its parameters (extended with a zero linear-generator block) as
    a warm start, or compose the two transforms.

    Parameters / returns as ``rigid_register``.
    """
    ndim = _spatial_ndim(moving, fixed)
    return multi_resolution_register(
        moving,
        fixed,
        model=Affine(),
        ndim=ndim,
        spec=spec,
    )
