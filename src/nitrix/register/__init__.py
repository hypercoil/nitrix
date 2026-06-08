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
- ``RegistrationSpec`` -- static config (pyramid, iterations, metric,
  interpolation); ``RegistrationResult`` -- the output record.
- ``diffeomorphic_demons_register`` -- log-domain diffeomorphic Demons
  (stationary velocity field; ESM force; fluid+diffusion Gaussian
  regularisation; scaling-and-squaring exp), with ``DemonsSpec`` /
  ``DiffeomorphicResult``.
"""

from ._core import RegistrationResult, RegistrationSpec
from .diffeomorphic import (
    DemonsSpec,
    DiffeomorphicResult,
    diffeomorphic_demons_register,
)
from .recipes import affine_register, rigid_register

__all__ = [
    'rigid_register',
    'affine_register',
    'RegistrationSpec',
    'RegistrationResult',
    'diffeomorphic_demons_register',
    'DemonsSpec',
    'DiffeomorphicResult',
]
