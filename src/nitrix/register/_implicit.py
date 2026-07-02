# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Differentiable registration via the implicit-function theorem.

The differentiable-layer counterpart of the coarse-to-fine recipes: instead
of unrolling the optimisation (whose reverse pass would hold every iterate),
the transform's optimum is differentiated *at the optimum* through
:func:`~nitrix.linalg.implicit_least_squares` /
:func:`~nitrix.linalg.implicit_minimize`. This costs :math:`O(1)` memory in
the iteration count and is exact at a converged stationary point. The result
is a registration that is differentiable with respect to the input images:
``jax.grad`` through the returned :attr:`~nitrix.register.RegistrationResult.matrix`
/ :attr:`~nitrix.register.RegistrationResult.warped` flows to ``moving`` /
``fixed``.

Three entry points share a single coarse-to-fine driver with an implicit
per-level solver, so the centre-baked-in transform convention and the warp
are owned by reuse -- the same coordinate-space sampler and warp application
the forward recipes use, never a re-derived copy:

- :func:`register_implicit` -- the single-level core: one implicit solve at
  full resolution. The natural refinement / amortised-prediction-correction
  primitive (no coarse-to-fine basin search; differentiate the converged
  transform).
- :func:`rigid_register_implicit` / :func:`affine_register_implicit` -- the
  coarse-to-fine recipes (each level solved implicitly), mirroring the
  forward :func:`~nitrix.register.rigid_register` /
  :func:`~nitrix.register.affine_register` signatures, for the classical
  multi-resolution case.

A least-squares metric (:class:`~nitrix.register.SSD`) routes to
:func:`~nitrix.linalg.implicit_least_squares` (the Gauss-Newton Hessian); any
other (:class:`~nitrix.register.LNCC` / :class:`~nitrix.register.MI` /
:class:`~nitrix.register.CorrelationRatio`) routes to
:func:`~nitrix.linalg.implicit_minimize` (the exact Hessian, with a BFGS
forward solve) -- so a differentiable LNCC / MI registration layer falls out
of the same path. Because the implicit backward pass yields
:math:`\\mathrm{d}\\theta^{*}/\\mathrm{d}\\theta_{\\mathrm{init}} = 0`, a
multi-level run's coarse levels act as a gradient-stopped initialiser and the
finest level carries the gradient that is exact under the implicit-function
theorem.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Optional

from jaxtyping import Array, Float

from ._core import (
    RegistrationResult,
    RegistrationSpec,
    _implicit_level_solve,
    multi_resolution_register,
)
from ._metric import SSD, Metric
from ._model import Affine, Rigid, TransformModel
from ._space import CoordinateSpace, IndexSpace

__all__ = [
    'register_implicit',
    'rigid_register_implicit',
    'affine_register_implicit',
]


def _spatial_ndim(moving: Array, fixed: Array) -> int:
    """Validate the images and return their spatial rank.

    Check that ``moving`` and ``fixed`` are single-channel 2-D or 3-D images
    sharing the same rank, and return that rank.

    Parameters
    ----------
    moving
        Moving image; must be 2-D or 3-D.
    fixed
        Fixed image; must match the rank of ``moving``.

    Returns
    -------
    int
        The common spatial rank (``2`` or ``3``).

    Raises
    ------
    ValueError
        If either image is not 2-D / 3-D or the two ranks differ.
    """
    ndim = moving.ndim
    if ndim not in (2, 3) or fixed.ndim != ndim:
        raise ValueError(
            f'registration supports 2-D / 3-D single-channel images; got '
            f'moving {moving.shape}, fixed {fixed.shape}.'
        )
    return ndim


def register_implicit(
    moving: Float[Array, '*mspatial'],
    fixed: Float[Array, '*fspatial'],
    *,
    model: TransformModel = Affine(),
    metric: Metric = SSD(),
    space: CoordinateSpace = IndexSpace(),
    n_iters: int = 50,
    init_params: Optional[Float[Array, ' p']] = None,
) -> RegistrationResult:
    """Single-level registration differentiable with respect to the images.

    Solve one transform optimisation at full resolution and differentiate it
    *at the optimum* (the implicit-function theorem), so ``jax.grad`` through
    the returned :attr:`~nitrix.register.RegistrationResult.matrix` /
    :attr:`~nitrix.register.RegistrationResult.warped` flows to ``moving`` /
    ``fixed`` with :math:`O(1)` memory in the iteration count -- the
    differentiable-layer alternative to unrolling a recipe. The result is the
    same self-contained :class:`~nitrix.register.RegistrationResult` the
    forward recipes return: the centre is baked into the transform matrix by
    the shared coordinate-space machinery, so
    :func:`~nitrix.register.apply_transform` reproduces the warp and two
    results compose.

    With no coarse-to-fine basin search this is the refinement primitive: seed
    ``init_params`` from a prior estimate (e.g. an amortised prediction) and
    differentiate the converged correction. For a multi-resolution capture
    range, use :func:`rigid_register_implicit` /
    :func:`affine_register_implicit`.

    Parameters
    ----------
    moving, fixed
        Single-channel images (2-D or 3-D), sharing a grid for the default
        :class:`~nitrix.register.IndexSpace`. A :class:`~nitrix.register.WorldSpace`
        handles anisotropy and differing grids; both bake the centre into the
        transform matrix identically.
    model
        Transform chart (:class:`~nitrix.register.Affine` by default, or
        :class:`~nitrix.register.Rigid`).
    metric
        Similarity metric. :class:`~nitrix.register.SSD` (default) routes to
        the Gauss-Newton :func:`~nitrix.linalg.implicit_least_squares`;
        :class:`~nitrix.register.LNCC` / :class:`~nitrix.register.MI` /
        :class:`~nitrix.register.CorrelationRatio` route to the general-metric
        :func:`~nitrix.linalg.implicit_minimize` (a differentiable LNCC / MI
        layer). MI differentiates only through its soft-histogram cost.
    space
        Coordinate space (:class:`~nitrix.register.IndexSpace` by default, or
        :class:`~nitrix.register.WorldSpace`).
    n_iters
        Number of forward solve iterations (Levenberg-Marquardt for
        :class:`~nitrix.register.SSD`; BFGS otherwise). Run to convergence --
        the implicit gradient is exact only at a stationary point.
    init_params
        Starting about-origin Lie parameters in the ``model``'s layout, of
        shape ``(p,)``; ``None`` uses zeros. The optimum, not the starting
        point, carries the gradient
        (:math:`\\mathrm{d}\\theta^{*}/\\mathrm{d}\\theta_{\\mathrm{init}} = 0`).

    Returns
    -------
    RegistrationResult
        The registration record, carrying the transform matrix, the estimated
        parameters, the warped moving image, and the cost history. Here
        ``cost_history`` holds the ``[initial, final]`` cost of the implicit
        solve.
    """
    ndim = _spatial_ndim(moving, fixed)
    spec = RegistrationSpec(
        levels=1, iterations=n_iters, metric=metric, mode='fixed'
    )
    return multi_resolution_register(
        moving,
        fixed,
        model=model,
        ndim=ndim,
        spec=spec,
        space=space,
        init_params=init_params,
        solve_level=_implicit_level_solve,
    )


def rigid_register_implicit(
    moving: Float[Array, '*mspatial'],
    fixed: Float[Array, '*fspatial'],
    *,
    spec: RegistrationSpec = RegistrationSpec(),
    space: CoordinateSpace = IndexSpace(),
    init_params: Optional[Float[Array, ' p']] = None,
) -> RegistrationResult:
    """Coarse-to-fine rigid registration, each level solved implicitly.

    The multi-resolution sibling of :func:`register_implicit` for the 6-DOF
    (3-D) / 3-DOF (2-D) rigid chart: the shared coarse-to-fine driver runs the
    implicit per-level solve at every pyramid level (``spec.levels`` /
    ``spec.iterations`` / ``spec.metric`` as in
    :func:`~nitrix.register.rigid_register`). The coarse levels supply the
    basin (gradient-stopped, since
    :math:`\\mathrm{d}\\theta^{*}/\\mathrm{d}\\theta_{\\mathrm{init}} = 0`);
    the finest level carries the gradient with respect to the images that is
    exact under the implicit-function theorem. ``spec.mode`` is forced to
    ``'fixed'`` (the implicit solve is its own fixed-iteration forward, so
    early-exit is inapplicable).

    Parameters
    ----------
    moving, fixed
        Single-channel images (2-D or 3-D), as in :func:`register_implicit`.
    spec
        Multi-resolution schedule (:class:`~nitrix.register.RegistrationSpec`):
        pyramid levels, per-level iterations, and the similarity metric.
        ``spec.mode`` is overridden to ``'fixed'``.
    space
        Coordinate space (:class:`~nitrix.register.IndexSpace` by default, or
        :class:`~nitrix.register.WorldSpace`).
    init_params
        Starting about-origin Lie parameters in the :class:`~nitrix.register.Rigid`
        layout, of shape ``(p,)``; ``None`` uses zeros.

    Returns
    -------
    RegistrationResult
        The registration record, with the model fixed to
        :class:`~nitrix.register.Rigid`. Here ``cost_history`` is the
        per-level ``[initial, final]`` implicit-solve costs, concatenated.
    """
    ndim = _spatial_ndim(moving, fixed)
    spec = replace(spec, mode='fixed')
    return multi_resolution_register(
        moving,
        fixed,
        model=Rigid(),
        ndim=ndim,
        spec=spec,
        space=space,
        init_params=init_params,
        solve_level=_implicit_level_solve,
    )


def affine_register_implicit(
    moving: Float[Array, '*mspatial'],
    fixed: Float[Array, '*fspatial'],
    *,
    spec: RegistrationSpec = RegistrationSpec(),
    space: CoordinateSpace = IndexSpace(),
    init_params: Optional[Float[Array, ' p']] = None,
) -> RegistrationResult:
    """Coarse-to-fine affine registration, each level solved implicitly.

    The 12-DOF (3-D) / 6-DOF (2-D) affine sibling of
    :func:`rigid_register_implicit` (model fixed to
    :class:`~nitrix.register.Affine`). As with the forward
    :func:`~nitrix.register.affine_register`, the affine basin is narrow --
    seed ``init_params`` (e.g. from a rigid stage or an amortised prediction),
    or use enough levels for the coarse stages to find the basin (the gradient
    comes from the finest level regardless).

    Parameters
    ----------
    moving, fixed
        Single-channel images (2-D or 3-D), as in :func:`register_implicit`.
    spec
        Multi-resolution schedule (:class:`~nitrix.register.RegistrationSpec`):
        pyramid levels, per-level iterations, and the similarity metric.
        ``spec.mode`` is overridden to ``'fixed'``.
    space
        Coordinate space (:class:`~nitrix.register.IndexSpace` by default, or
        :class:`~nitrix.register.WorldSpace`).
    init_params
        Starting about-origin Lie parameters in the :class:`~nitrix.register.Affine`
        layout, of shape ``(p,)``; ``None`` uses zeros.

    Returns
    -------
    RegistrationResult
        The registration record, with the model fixed to
        :class:`~nitrix.register.Affine`. Here ``cost_history`` is the
        per-level ``[initial, final]`` implicit-solve costs, concatenated.
    """
    ndim = _spatial_ndim(moving, fixed)
    spec = replace(spec, mode='fixed')
    return multi_resolution_register(
        moving,
        fixed,
        model=Affine(),
        ndim=ndim,
        spec=spec,
        space=space,
        init_params=init_params,
        solve_level=_implicit_level_solve,
    )
