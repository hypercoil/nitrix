# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
nitrix.register -- pairwise registration recipes.

Pure-functional registrators that compose the substrate (geometry
transforms + pyramid, image-similarity metrics, the matrix-free
nonlinear-least-squares optimiser) into end-to-end alignment.  Outputs
are ``NamedTuple``s of arrays (the ``reml_fit`` precedent) -- no PyTree
modules, no atlas data structures, no I/O; ``entense`` wraps these.

- ``rigid_register`` -- 6-DOF (3-D) / 3-DOF (2-D) Gauss-Newton / LM
  intensity registration on SE(2)/SE(3) (the 3dvolreg / AIR lineage).
- ``affine_register`` -- 12-DOF (3-D) / 6-DOF (2-D), linear block via
  ``matrix_exp``.
- ``volreg`` -- batched rigid motion realignment of a ``(T, *spatial)``
  series to a common reference (the ``3dvolreg`` / ``mcflirt`` task);
  ``vmap``-ed over frames with the reference work hoisted out of the
  batch.  Returns ``VolregResult``.
- ``bbr_register`` -- volumetric boundary-based registration (Greve-Fischl):
  rigid alignment to a tissue boundary (points + normals), maximising the
  cross-boundary contrast.  Returns ``BBRResult``.
- ``Objective`` -- the objective ADT (``θ ↦ cost``) the optimiser
  minimises; ``MetricObjective`` (an image pair) and ``BoundaryObjective``
  (BBR) are its implementers.
- ``RegistrationSpec`` -- static config (pyramid, iterations, metric,
  interpolation); ``RegistrationResult`` -- the output record.
- ``Metric`` -- the similarity objective ADT (``SSD`` / ``LNCC`` / ``MI``
  / ``CorrelationRatio``), each carrying its own hyper-parameters.
- ``TransformModel`` -- the chart a matrix-transform recipe optimises
  over (``Rigid`` / ``Affine``): the ``exp`` map plus the parameter-layout
  knowledge the coarse-to-fine driver needs.
- ``CoordinateSpace`` -- the space a recipe optimises in (``IndexSpace``,
  the default voxel-space shared-grid fast path; ``WorldSpace``, the
  physical-space path correct under anisotropic voxels / different grids).
- ``diffeomorphic_demons_register`` -- log-domain diffeomorphic Demons
  (stationary velocity field; ESM force; fluid+diffusion Gaussian
  regularisation; scaling-and-squaring exp), with ``DemonsSpec`` /
  ``DiffeomorphicResult``.
- ``greedy_syn_register`` -- greedy symmetric diffeomorphic registration
  (SyN-style): symmetric forward/inverse velocity fields driven to a
  midpoint by the analytic LNCC force (``metrics.lncc_grad``); robust to
  smooth intensity inhomogeneity.  ``SyNSpec`` / ``SyNResult``.
"""

from ._bbr import BBRResult, BBRSpec, BoundaryObjective, bbr_cost, bbr_register
from ._core import RegistrationResult, RegistrationSpec
from ._force import DemonsForce, Force, LNCCForce, MetricForce
from ._metric import LNCC, MI, SSD, CorrelationRatio, Metric
from ._model import Affine, Rigid, TransformModel
from ._objective import MetricObjective, Objective
from ._space import CoordinateSpace, IndexSpace, WorldSpace
from ._syn import SyNResult, SyNSpec, greedy_syn_register
from ._volreg import VolregResult, volreg
from .diffeomorphic import (
    DemonsSpec,
    DiffeomorphicResult,
    diffeomorphic_demons_register,
)
from .recipes import affine_register, rigid_register
from .regulariser import (
    bending_energy,
    gradient_smoothness,
    jacobian_folding_penalty,
)

__all__ = [
    'rigid_register',
    'affine_register',
    'volreg',
    'VolregResult',
    'bbr_register',
    'bbr_cost',
    'BBRSpec',
    'BBRResult',
    'BoundaryObjective',
    'Objective',
    'MetricObjective',
    'Force',
    'LNCCForce',
    'DemonsForce',
    'MetricForce',
    'RegistrationSpec',
    'RegistrationResult',
    'Metric',
    'SSD',
    'LNCC',
    'MI',
    'CorrelationRatio',
    'TransformModel',
    'Rigid',
    'Affine',
    'CoordinateSpace',
    'IndexSpace',
    'WorldSpace',
    'diffeomorphic_demons_register',
    'DemonsSpec',
    'DiffeomorphicResult',
    'greedy_syn_register',
    'SyNSpec',
    'SyNResult',
    'gradient_smoothness',
    'bending_energy',
    'jacobian_folding_penalty',
]
