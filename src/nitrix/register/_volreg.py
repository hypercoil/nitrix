# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Batched volume registration -- motion realignment of a 4-D series.

``volreg`` rigidly registers every frame of a ``(T, *spatial)`` series to
a common reference, returning the per-frame transform stack and the
realigned series (the ``3dvolreg`` / ``mcflirt`` task).  The reference is
registered against **once**: its Gaussian pyramid (and, in a future
inverse-compositional path, its steepest-descent images + Hessian) are
computed a single time and the per-frame core (:func:`._core.register_core`)
is ``vmap``-ed over the series -- the shared reference work is hoisted out
of the batch, and the whole series compiles once.

The per-frame solve is the same matrix-free Gauss-Newton / Levenberg-
Marquardt path the single-pair recipes use (so it ``vmap``-s cleanly over
the frame axis); ``volreg`` therefore requires a least-squares metric
(``SSD``, the default).  The ``space`` argument carries the per-image
voxel->world geometry, so motion that is rigid in **physical** space is
recovered correctly on anisotropic grids (``WorldSpace``) -- the same
foundation as the single-pair recipes (``_space``).

A two-pass schedule (``passes=2``) re-references to the mean of the
first-pass realignment and warm-starts from the first-pass parameters --
the standard robustness step when the initial reference is the (blurred)
mean of an unaligned series.
"""

from __future__ import annotations

from typing import NamedTuple, Optional, Union

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ..geometry import gaussian_pyramid, rigid_exp
from ._converge import resolve_convergence_mode
from ._core import RegistrationSpec, register_core
from ._inverse_compositional import (
    ReferenceLevel,
    ic_reference,
    ic_register_core,
)
from ._model import Rigid
from ._space import CoordinateSpace, IndexSpace

__all__ = ['volreg', 'VolregResult']


class VolregResult(NamedTuple):
    """Output of :func:`volreg`.

    Attributes
    ----------
    matrices
        Per-frame homogeneous transforms, ``(T, ndim + 1, ndim + 1)`` --
        index-space (``IndexSpace``) or world-space (``WorldSpace``), each
        mapping the reference grid to that frame.
    params
        Per-frame rigid Lie parameters, ``(T, p)``.
    realigned
        The series resampled onto the reference grid, ``(T, *spatial)``.
    reference
        The reference image the series was aligned to, ``(*spatial,)``
        (the final-pass reference for a multi-pass run).
    cost_history
        Per-frame concatenated optimiser cost traces, ``(T, h)``.
    """

    matrices: Float[Array, 't d1 d1']
    params: Float[Array, 't p']
    realigned: Float[Array, 't *spatial']
    reference: Float[Array, '*spatial']
    cost_history: Float[Array, 't h']


def _resolve_reference(
    series: Float[Array, 't *spatial'],
    reference: Union[int, str, Float[Array, '*spatial']],
) -> Array:
    """Reference image from a frame index, ``"mean"``, or an explicit array."""
    if isinstance(reference, str):
        if reference != 'mean':
            raise ValueError(
                f'reference string must be "mean"; got {reference!r}.'
            )
        return jnp.mean(series, axis=0)
    if isinstance(reference, int):
        return series[reference]
    ref = jnp.asarray(reference)
    if ref.shape != series.shape[1:]:
        raise ValueError(
            f'reference image shape {ref.shape} does not match the frame '
            f'shape {series.shape[1:]}.'
        )
    return ref


def volreg(
    series: Float[Array, 't *spatial'],
    *,
    reference: Union[int, str, Float[Array, '*spatial']] = 'mean',
    passes: int = 1,
    spec: RegistrationSpec = RegistrationSpec(),
    space: CoordinateSpace = IndexSpace(),
    method: str = 'auto',
) -> VolregResult:
    """Rigid motion realignment of a 4-D (or 3-D) series.

    Registers every frame of ``series`` to a common reference and returns
    the per-frame rigid transforms and the realigned series.

    Parameters
    ----------
    series
        ``(T, *spatial)`` -- ``T`` frames of a 2-D or 3-D single-channel
        image sharing one voxel grid.
    reference
        ``"mean"`` (default; the per-voxel mean over frames), a frame
        index (``int``), or an explicit reference image ``(*spatial,)``.
    passes
        Number of realignment passes.  ``2`` re-references to the mean of
        the first-pass realignment and warm-starts from its parameters
        (robust when the initial reference is a blurred mean).
    spec
        ``RegistrationSpec`` for the per-frame registration.  The metric
        must be least-squares (``SSD``, the default) -- the batched LM
        path does not use the scalar BFGS metrics.  **Schedule:** motion
        realignment is a *small*-displacement problem and the second-order
        per-frame solver converges in a handful of steps, so a **tight**
        ``iterations`` (e.g. ``(4, 2, 1)`` over a 3-level pyramid) realigns as
        well as a loose one for a fraction of the cost -- the default 30/level
        over-iterates motion correction by ~25x (measured: a 20-frame MNI-2mm
        series realigns identically at ``(4,2,1)`` for ~1/8 the time).
        ``spec.mode``:
        ``'fixed'`` (default) runs the fixed ``scan`` (reproducible,
        reverse-differentiable); ``'early_exit'`` opts into a **batch-aggregate
        early-exit** (inverse-compositional path only) -- the per-frame
        ``while_loop`` runs under ``vmap`` until *all* frames have converged, so
        the cohort adapts to the actual motion (a looser ``convergence.threshold``
        ~1e-3 gives ~2-3x at identical realignment; not reverse-differentiable).
        Rejected on the forward (Gauss-Newton) path, which threads only the scan.
    space
        Coordinate space (``IndexSpace`` default, or ``WorldSpace`` for
        physically-rigid motion on anisotropic grids).
    method
        Per-frame solver: ``"auto"`` (default; the inverse-compositional
        constant-template Hessian in ``IndexSpace`` -- the reference's
        steepest-descent + Hessian are built **once** for the whole
        series; the forward Gauss-Newton path otherwise),
        ``"inverse_compositional"`` (force IC; ``IndexSpace`` only), or
        ``"forward"``.

    Returns
    -------
    ``VolregResult`` (``matrices``, ``params``, ``realigned``,
    ``reference``, ``cost_history``).
    """
    if series.ndim not in (3, 4):
        raise ValueError(
            f'series must be (T, *spatial) with 2-D or 3-D frames; got '
            f'shape {series.shape}.'
        )
    if passes < 1:
        raise ValueError(f'passes must be >= 1; got {passes}.')
    if not spec.metric.is_least_squares:
        raise ValueError(
            'volreg requires a least-squares metric (e.g. SSD); the scalar '
            'BFGS metrics do not support the batched LM path.'
        )
    is_index = isinstance(space, IndexSpace)
    if method == 'auto':
        use_ic = is_index
    elif method == 'inverse_compositional':
        if not is_index:
            raise ValueError(
                'inverse_compositional requires IndexSpace (the template '
                'linearisation is in voxel coordinates).'
            )
        use_ic = True
    elif method == 'forward':
        use_ic = False
    else:
        raise ValueError(
            f'method must be "auto", "forward", or '
            f'"inverse_compositional"; got {method!r}.'
        )
    # Batch-aggregate early-exit (opt-in, the inverse-compositional path).  A
    # ``lax.while_loop`` *is* vmap-compatible: ``jax.vmap`` runs it until **all**
    # lanes' cond is false, freezing each frame once it converges -- so
    # ``mode='early_exit'`` makes the cohort stop at the slowest frame's
    # convergence rather than a fixed cap, adapting to the actual motion.  The
    # B2 gate below validates eligibility once (forward GN threads only the
    # scan); ``ic_register_core`` re-derives the concrete Convergence from
    # ``spec.mode`` per frame.  The forward branch passes no ``convergence`` to
    # ``register_core`` by construction, so it always runs the fixed scan.
    resolve_convergence_mode(
        spec.mode,
        spec.convergence,
        supports_early_exit=use_ic,
        path='the volreg forward (Gauss-Newton) path',
    )

    ndim = series.ndim - 1
    n_frames = series.shape[0]
    frame_shape = series.shape[1:]
    dtype = series.dtype
    model = Rigid()
    sampler = space.sampler(
        ndim=ndim,
        full_fixed_shape=frame_shape,
        full_moving_shape=frame_shape,
        dtype=dtype,
    )

    ref = _resolve_reference(series, reference)
    init = jnp.zeros((n_frames, model.n_params(ndim)), dtype=dtype)
    matrix = params = realigned = costs = None
    for pass_idx in range(passes):
        pyr_f = gaussian_pyramid(
            ref[..., None],
            levels=spec.levels,
            factor=spec.pyramid_factor,
            sigma=spec.pyramid_sigma,
        )

        # The inverse-compositional reference steepest-descent + Hessian are
        # built once for the whole series; only the per-frame update is
        # vmap-ed.  ``None`` selects the forward Gauss-Newton path.
        ref_levels = ic_reference(pyr_f, ndim) if use_ic else None

        def per_frame(
            moving: Array,
            init_p: Array,
            ref_levels: Optional[list[ReferenceLevel]] = ref_levels,
            pyr_f: tuple[Array, ...] = pyr_f,
        ) -> tuple[Array, Array, Array, Array]:
            if ref_levels is not None:
                res = ic_register_core(
                    moving,
                    ref_levels,
                    ndim=ndim,
                    spec=spec,
                    init_matrix=rigid_exp(init_p, ndim=ndim),
                )
            else:
                res = register_core(
                    moving,
                    pyr_f,
                    model=model,
                    ndim=ndim,
                    spec=spec,
                    space=space,
                    sampler=sampler,
                    init_params=init_p,
                )
            return res.matrix, res.params, res.warped, res.cost_history

        matrix, params, realigned, costs = jax.vmap(per_frame, in_axes=(0, 0))(
            series, init
        )
        if pass_idx < passes - 1:
            ref = jnp.mean(realigned, axis=0)
            init = params

    assert (
        matrix is not None
        and params is not None
        and realigned is not None
        and costs is not None
    )
    return VolregResult(
        matrices=matrix,
        params=params,
        realigned=realigned,
        reference=ref,
        cost_history=costs,
    )
