# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
# isort: skip_file
"""
nitrix.linalg -- linear-algebra primitives.

Four submodules:

- ``matrix``   -- symmetric / vec triangular conversions, diagonal
  utilities, Toeplitz, eigenspace reconditioning.
- ``residual`` -- residualisation (confound regression) numerics.
- ``kernel``   -- kernel functions (RBF, polynomial, ...) over
  point clouds; specialises onto ``semiring_ell_matmul`` for
  ``O(n * k)`` scaling at scale.
- ``spd``      -- symmetric positive-definite manifold ops
  (matrix exp/log/sqrt/power) with the SPEC §4.1 stability fix.

Re-exports the most commonly imported public symbols.  See the
submodule docstrings for the full surface.

This subpackage replaces the legacy ``nitrix.functional.matrix``
and ``nitrix.functional.residual`` modules; the green-field
rewrite leverages the substrate (``semiring_matmul``,
``semiring_ell_matmul``) where applicable and drops the
backward-compatible naming the legacy code was constrained by.
"""

from .kernel import (
    cosine_kernel,
    gaussian_kernel,
    linear_distance,
    linear_kernel,
    parameterised_norm,
    polynomial_kernel,
    rbf_kernel,
    sigmoid_kernel,
)
from .matrix import (
    delete_diagonal,
    fill_diagonal,
    recondition_eigenspaces,
    squareform,
    sym2vec,
    symmetric,
    toeplitz,
    toeplitz_2d,
    vec2sym,
)
from .decompose import randomized_svd
from .krylov import cg
from .matrix_function import matrix_exp, matrix_log
from .optimize import (
    OptimizeResult,
    gauss_newton,
    implicit_least_squares,
    implicit_minimize,
    levenberg_marquardt,
)
from .residual import partial_residualise, residualise
from .solve import cho_solve, solve
from .spd import (
    cone_project_spd,
    mean_euclidean,
    mean_log_euclidean,
    symexp,
    symlog,
    symmap,
    sympower,
    symsqrt,
    tangent_project_spd,
)

__all__ = [
    # matrix
    'delete_diagonal',
    'fill_diagonal',
    'recondition_eigenspaces',
    'squareform',
    'sym2vec',
    'symmetric',
    'toeplitz',
    'toeplitz_2d',
    'vec2sym',
    # residual
    'residualise',
    'partial_residualise',
    # solve
    'solve',
    'cho_solve',
    'cg',
    # matrix functions
    'matrix_exp',
    'matrix_log',
    # decomposition
    'randomized_svd',
    # optimisation (nonlinear least squares)
    'gauss_newton',
    'levenberg_marquardt',
    'implicit_least_squares',
    'implicit_minimize',
    'OptimizeResult',
    # kernel
    'cosine_kernel',
    'gaussian_kernel',
    'linear_distance',
    'linear_kernel',
    'parameterised_norm',
    'polynomial_kernel',
    'rbf_kernel',
    'sigmoid_kernel',
    # spd
    'cone_project_spd',
    'mean_euclidean',
    'mean_log_euclidean',
    'symexp',
    'symlog',
    'symmap',
    'sympower',
    'symsqrt',
    'tangent_project_spd',
]
