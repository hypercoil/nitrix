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

import warnings
from dataclasses import replace

from jaxtyping import Array, Float

from ._core import (
    RegistrationResult,
    RegistrationSpec,
    multi_resolution_register,
)
from ._inverse_compositional import ic_affine_register, ic_rigid_register
from ._model import Affine, Rigid, TransformModel
from ._space import CoordinateSpace, IndexSpace

__all__ = ['rigid_register', 'affine_register']

# Coarsest pyramid axis (voxels) below which the matrix Hessian -- especially the
# 12-DOF affine one -- is too few-voxel to estimate reliably, so the constant-
# template step converges to a wrong (anti-correlated) minimum
# (`register-affine-small-grid-divergence`: a 24³/28³ image at the default 3
# levels drops its coarsest level to ≤14³ and diverges; single-level recovers).
# Below this, the affine pyramid is shortened so the coarsest level stays reliable
# -- a targeted fix that needs no damping, so the well-conditioned path is
# untouched.  Affine-only: rigid (6-DOF) is robust at small coarse grids.
_MIN_COARSE_AXIS = 16


class AffinePyramidDepthWarning(UserWarning):
    """The affine pyramid was shortened so its coarsest level stays reliable."""


def _cap_levels(
    spec: RegistrationSpec, shape: tuple[int, ...]
) -> RegistrationSpec:
    """Shorten the (affine) pyramid so its coarsest level keeps ``>=
    _MIN_COARSE_AXIS`` voxels per axis; **loud** (a warning, per the loud-fallback
    tenet) and a no-op when the requested depth already satisfies it.  Honours an
    explicit per-level ``iterations`` tuple by keeping its finest entries."""
    smallest = float(min(shape))
    effective = 1
    while (
        effective < spec.levels
        and smallest / (spec.pyramid_factor**effective) >= _MIN_COARSE_AXIS
    ):
        effective += 1
    if effective >= spec.levels:
        return spec
    warnings.warn(
        f'affine_register: shortening the pyramid from {spec.levels} to '
        f'{effective} level(s) -- a {int(smallest)}-voxel image at '
        f'{spec.levels} levels drops the coarsest grid below '
        f'{_MIN_COARSE_AXIS} vox/axis, where the affine Hessian is unreliable '
        f'and the fit diverges (register-affine-small-grid-divergence).  Pass a '
        f'smaller spec.levels to silence this.',
        category=AffinePyramidDepthWarning,
        stacklevel=3,
    )
    iterations = spec.iterations
    if isinstance(iterations, tuple):
        iterations = iterations[-effective:]  # keep the finest levels' counts
    return replace(spec, levels=effective, iterations=iterations)


def _spatial_ndim(moving: Array, fixed: Array) -> int:
    ndim = moving.ndim
    if ndim not in (2, 3):
        raise ValueError(
            f'registration supports 2-D / 3-D single-channel images; '
            f'got shape {moving.shape}.'
        )
    return ndim


def _use_inverse_compositional(
    method: str,
    space: CoordinateSpace,
    spec: RegistrationSpec,
    model: TransformModel,
) -> bool:
    """Resolve the ``method`` argument against the IC fast-path preconditions.

    The inverse-compositional kernel (constant-template Hessian, ~4-7x the
    forward GN/LM throughput) applies to a **rigid or affine** least-squares
    (SSD) registration in **IndexSpace** (the template is linearised in voxel
    coordinates).  ``"auto"`` takes it when those hold and falls back to the
    forward path otherwise (the parity oracle); ``"inverse_compositional"``
    forces it (and validates); ``"forward"`` always takes the forward path.
    """
    supported = (
        isinstance(space, IndexSpace)
        and spec.metric.is_least_squares
        and spec.optimizer in ('auto', 'lm', 'gn')
        and isinstance(model, (Rigid, Affine))
    )
    if method == 'auto':
        return supported
    if method == 'inverse_compositional':
        if not supported:
            raise ValueError(
                'method="inverse_compositional" requires IndexSpace + a '
                'least-squares (SSD) metric + a Rigid/Affine model.'
            )
        return True
    if method == 'forward':
        return False
    raise ValueError(
        f'method must be "auto", "forward", or "inverse_compositional"; '
        f'got {method!r}.'
    )


def rigid_register(
    moving: Float[Array, '*mspatial'],
    fixed: Float[Array, '*fspatial'],
    *,
    spec: RegistrationSpec = RegistrationSpec(),
    space: CoordinateSpace = IndexSpace(),
    method: str = 'auto',
) -> RegistrationResult:
    """Estimate the rigid transform aligning ``moving`` to ``fixed``.

    Optimises the 6-DOF (3-D) / 3-DOF (2-D) rigid Lie parameters
    (``geometry.rigid_exp``) coarse-to-fine so that ``moving`` resampled
    by the result matches ``fixed`` under ``spec.metric``.

    Parameters
    ----------
    moving, fixed
        Single-channel images (2-D or 3-D).  Shapes need not match (the
        warp is built on the ``fixed`` grid); the default ``IndexSpace``
        additionally assumes a shared voxel grid.
    spec
        ``RegistrationSpec`` (pyramid depth, iterations, metric, ...).
    space
        Coordinate space to optimise in (``_space``): ``IndexSpace()``
        (default; voxel-space, shared-grid, on-device) or
        ``WorldSpace(fixed_affine=..., moving_affine=...)`` (physical
        space -- correct under anisotropic voxels and different grids).
    method
        Solver: ``"auto"`` (default; the inverse-compositional fast path --
        ~4-7x the forward throughput -- when its preconditions hold:
        ``IndexSpace`` + a least-squares / SSD metric; the forward
        Gauss-Newton / LM path otherwise), ``"inverse_compositional"`` (force
        it; validates), or ``"forward"``.  The forward path is the parity
        oracle the fast path is tested against.

    Returns
    -------
    ``RegistrationResult`` (``matrix``, ``params``, ``warped``,
    ``cost_history``).  ``matrix`` maps ``fixed`` to ``moving`` (index
    coordinates in ``IndexSpace``, world coordinates in ``WorldSpace``);
    ``warped`` is ``moving`` on the ``fixed`` grid.
    """
    ndim = _spatial_ndim(moving, fixed)
    model = Rigid()
    if _use_inverse_compositional(method, space, spec, model):
        return ic_rigid_register(moving, fixed, ndim=ndim, spec=spec)
    return multi_resolution_register(
        moving,
        fixed,
        model=model,
        ndim=ndim,
        spec=spec,
        space=space,
    )


def affine_register(
    moving: Float[Array, '*mspatial'],
    fixed: Float[Array, '*fspatial'],
    *,
    spec: RegistrationSpec = RegistrationSpec(),
    space: CoordinateSpace = IndexSpace(),
    method: str = 'auto',
) -> RegistrationResult:
    """Estimate the affine transform aligning ``moving`` to ``fixed``.

    Optimises the 12-DOF (3-D) / 6-DOF (2-D) affine Lie parameters
    (``geometry.affine_exp`` -- linear block via ``matrix_exp``,
    guaranteeing an invertible map) coarse-to-fine.  For a robust result
    on a large initial misalignment, run ``rigid_register`` first and
    pass its parameters (extended with a zero linear-generator block) as
    a warm start, or compose the two transforms.

    Parameters / returns as ``rigid_register`` (including the ``space`` and
    ``method`` arguments; the inverse-compositional fast path -- where affine's
    large parameter count makes the forward ``jacfwd`` costliest -- engages
    under ``method="auto"`` for ``IndexSpace`` + an SSD metric).
    """
    ndim = _spatial_ndim(moving, fixed)
    model = Affine()
    spec = _cap_levels(spec, fixed.shape)
    if _use_inverse_compositional(method, space, spec, model):
        return ic_affine_register(moving, fixed, ndim=ndim, spec=spec)
    return multi_resolution_register(
        moving,
        fixed,
        model=model,
        ndim=ndim,
        spec=spec,
        space=space,
    )
