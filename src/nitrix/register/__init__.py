# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
r"""
nitrix.register -- pairwise alignment recipes (spatial + functional).

Pure-functional registrators that compose the substrate (geometry
transforms + pyramid, image-similarity metrics, the matrix-free
nonlinear-least-squares optimiser) into end-to-end alignment.  Outputs
are ``NamedTuple``s of arrays (the :func:`reml_fit` precedent) -- no PyTree
modules, no atlas data structures, no I/O; ``entense`` wraps these.

Two alignment families: **spatial** registration (rigid/affine/volreg/BBR,
log-Demons/SyN -- aligning *images* in voxel/world space) and **functional**
alignment (:func:`functional_align` -- aligning *representations* in feature
space, the hyperalignment task; ProMises is its first method).

- :func:`rigid_register` -- 6-DOF (3-D) / 3-DOF (2-D) Gauss-Newton / LM
  intensity registration on SE(2)/SE(3) (the 3dvolreg / AIR lineage).
- :func:`affine_register` -- 12-DOF (3-D) / 6-DOF (2-D), linear block via
  :func:`matrix_exp`.
- ``volreg`` -- batched rigid motion realignment of a ``(T, *spatial)``
  series to a common reference (the ``3dvolreg`` / ``mcflirt`` task);
  ``vmap``-ed over frames with the reference work hoisted out of the
  batch.  Returns ``VolregResult``.
- ``bbr_register`` -- volumetric boundary-based registration (Greve-Fischl):
  rigid alignment to a tissue boundary (points + normals), maximising the
  cross-boundary contrast.  Returns ``BBRResult``.
- ``Objective`` -- the objective ADT (:math:`\theta \mapsto \mathrm{cost}`)
  the optimiser minimises; ``MetricObjective`` (an image pair) and
  ``BoundaryObjective`` (BBR) are its implementers.
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

Scope & validation
------------------
**ANTs-*style*, synthetic-recovery-validated** -- the algorithms are the
ANTs / FSL / AFNI lineage (3dvolreg / mcflirt rigid, AIR affine, log-Demons,
greedy SyN, BBR), and the test suite validates *recovery of a known synthetic
warp* (high NCC / MI, positive Jacobian) plus the closed-form-vs-autodiff
force parity oracles.  It is **not** a drop-in ``antsRegistration``: the
following ANTs / fMRIPrep features are **not yet implemented** --

- intensity **winsorization** (``--winsorize-image-intensities``) and
  **histogram matching** (``--use-histogram-matching``);
- **multi-metric summation** within a stage (``-m MI -m CC``; a ``SumForce``);
- **restrict-deformation** (per-axis deformation masking);
- a closed-form **correlation-ratio** fast-force -- CR drives the diffeomorphic
  recipes only via the generic autodiff ``MetricForce`` escape hatch (the
  closed-form **(Mattes) MI** force, ``MIForce``, *is* shipped -- the fast
  cross-modal path; CR's ``cr_grad`` is the same machinery, built when a
  consumer asks);
- **early-exit** (windowed cost-slope convergence): every recipe carries the
  orthogonal ``mode`` (``'fixed'`` default / ``'early_exit'``) + ``convergence``
  (threshold / window) spec fields (B2); ``'early_exit'`` runs the
  ``lax.while_loop`` where the path supports it (the matrix inverse-compositional
  recipes recommend it).

**Real-data and ANTs-reference parity** (comparing nitrix transforms / warps
to an ``antsRegistration`` reference on real volumes, and the iso-accuracy
wall-clock comparison) are **delegated to the nitrix-perf-bench agent**, which
owns the cross-tool harness; they are not asserted in this repo.
"""

from ._bbr import (
    BBRResult,
    BBRSearch,
    BBRSpec,
    BoundaryObjective,
    bbr_cost,
    bbr_register,
)
from ._converge import Convergence, ConvergenceMode
from ._core import (
    RegistrationResult,
    RegistrationSpec,
)
from ._force import (
    DemonsForce,
    Force,
    LNCCForce,
    MetricForce,
    MIForce,
    SumForce,
)
from ._functional import (
    AlignmentMethod,
    FunctionalAlignment,
    ProMises,
    functional_align,
    functional_align_apply,
    functional_align_fit,
)
from ._implicit import (
    affine_register_implicit,
    register_implicit,
    rigid_register_implicit,
)
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
from .recipes import (
    PipelineResult,
    affine_register,
    apply_transform,
    rigid_register,
    syn_pipeline,
)
from .regulariser import (
    bending_energy,
    gradient_smoothness,
    jacobian_folding_penalty,
)

__all__ = [
    'rigid_register',
    'affine_register',
    'register_implicit',
    'rigid_register_implicit',
    'affine_register_implicit',
    'apply_transform',
    'syn_pipeline',
    'PipelineResult',
    'volreg',
    'VolregResult',
    'bbr_register',
    'bbr_cost',
    'BBRSpec',
    'BBRSearch',
    'BBRResult',
    'BoundaryObjective',
    'Objective',
    'MetricObjective',
    'Force',
    'LNCCForce',
    'DemonsForce',
    'MIForce',
    'MetricForce',
    'SumForce',
    'RegistrationSpec',
    'RegistrationResult',
    'Convergence',
    'ConvergenceMode',
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
    # functional alignment (representation-space)
    'functional_align',
    'functional_align_fit',
    'functional_align_apply',
    'FunctionalAlignment',
    'AlignmentMethod',
    'ProMises',
]
