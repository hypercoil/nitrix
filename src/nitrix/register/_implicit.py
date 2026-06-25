# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Differentiable registration via the implicit-function theorem.

The differentiable-layer counterpart of the coarse-to-fine recipes: instead
of unrolling the optimise (whose reverse pass holds every iterate), the
transform's optimum is differentiated **at the optimum** through
``linalg.implicit_least_squares`` / ``implicit_minimize`` -- O(1) memory in
the iteration count and exact at a converged stationary point.  The result is
a registration that is differentiable w.r.t. the input images: ``jax.grad``
through ``.matrix`` / ``.warped`` flows to ``moving`` / ``fixed``.

Three entry points, layered over the shared driver
(``_core.multi_resolution_register`` with the implicit per-level solver
``_implicit_level_solve``), so the centre-baked-in ``.matrix`` convention and
the warp are **owned by reuse** -- the same ``_space`` sampler / ``_warp`` /
``result_transform`` the forward recipes use, never a re-derived copy:

- :func:`register_implicit` -- the **single-level** core: one implicit solve at
  full resolution.  The natural refinement / amortised-prediction-correction
  primitive (no coarse-to-fine basin search; differentiate the converged
  transform).
- :func:`rigid_register_implicit` / :func:`affine_register_implicit` -- the
  **coarse-to-fine** recipes (each level solved implicitly), mirroring the
  forward ``rigid_register`` / ``affine_register`` signatures, for the
  classical multi-resolution case.

A least-squares metric (``SSD``) routes to ``implicit_least_squares`` (the
Gauss-Newton Hessian); any other (``LNCC`` / ``MI`` / ``CorrelationRatio``) to
``implicit_minimize`` (the exact Hessian, BFGS forward) -- so a differentiable
LNCC / MI registration layer falls out of the same path.  Because the implicit
backward yields ``dtheta*/dinit = 0``, a multi-level run's coarse levels act as
a gradient-stopped initialiser and the finest level carries the IFT-exact
gradient.
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
    """Validate the images are 2-D / 3-D single-channel and return the rank."""
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
    """Single-level registration differentiable w.r.t. the images (implicit).

    Solves one transform optimise at full resolution and differentiates it **at
    the optimum** (the implicit-function theorem), so ``jax.grad`` through the
    returned ``.matrix`` / ``.warped`` flows to ``moving`` / ``fixed`` with O(1)
    memory in the iteration count -- the differentiable-layer alternative to
    unrolling a recipe.  The result is the same self-contained ``RegistrationResult``
    the forward recipes return (the centre is baked into ``.matrix`` by the
    shared ``_space`` machinery, so ``apply_transform`` reproduces the warp and
    two results compose).

    With no coarse-to-fine basin search this is the **refinement** primitive:
    seed ``init_params`` from a prior estimate (e.g. an amortised prediction) and
    differentiate the converged correction.  For a multi-resolution capture
    range, use :func:`rigid_register_implicit` / :func:`affine_register_implicit`.

    Parameters
    ----------
    moving, fixed
        Single-channel images (2-D or 3-D), sharing a grid for the default
        ``IndexSpace`` (a ``WorldSpace`` handles anisotropy / differing grids;
        both bake the centre into ``.matrix`` identically).
    model
        Transform chart (``Affine()`` default, or ``Rigid()``).
    metric
        Similarity metric.  ``SSD()`` (default) routes to the Gauss-Newton
        ``implicit_least_squares``; ``LNCC`` / ``MI`` / ``CorrelationRatio`` to
        the general-metric ``implicit_minimize`` (a differentiable LNCC / MI
        layer).  MI differentiates only through its soft-histogram cost.
    space
        Coordinate space (``IndexSpace()`` default / ``WorldSpace(...)``).
    n_iters
        Forward solve iterations (LM for ``SSD``; BFGS otherwise).  Run to
        convergence -- the implicit gradient is exact only at a stationary point.
    init_params
        Starting **about-origin** Lie parameters (the ``model``'s layout);
        ``None`` -> zeros.  ``dtheta*/dinit = 0`` (the optimum, not the start,
        carries the gradient).

    Returns
    -------
    ``RegistrationResult`` (``matrix``, ``params``, ``warped``, ``cost_history``).
    ``cost_history`` is the ``[initial, final]`` cost of the implicit solve.
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
    ``spec.iterations`` / ``spec.metric`` as in ``rigid_register``).  The
    coarse levels supply the basin (gradient-stopped, since ``dtheta*/dinit =
    0``); the finest level carries the IFT-exact gradient w.r.t. the images.
    ``spec.mode`` is forced to ``'fixed'`` (the implicit solve is its own
    fixed-iteration forward; early-exit is inapplicable).

    Parameters / returns as :func:`register_implicit`, with the schedule on
    ``spec`` and the model fixed to ``Rigid``.  ``cost_history`` is the
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
    :func:`rigid_register_implicit` (model fixed to ``Affine``).  As with the
    forward ``affine_register``, the affine basin is narrow -- seed
    ``init_params`` (e.g. from a rigid stage or an amortised prediction) or use
    enough levels for the coarse stages to find the basin (the gradient comes
    from the finest level regardless).

    Parameters / returns as :func:`rigid_register_implicit`, model ``Affine``.
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
