# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
# isort: skip_file
"""
nitrix -- pure-JAX numerical substrate for neuroimaging research and
differentiable programming.

Top-level subpackages (full surface in each ``__init__.py``):

- ``linalg``   -- matrix utilities, residualisation, kernels, SPD ops.
- ``stats``    -- mass-univariate modelling: GLM / GAM(M) / Gaussian-process
                  (HSGP) and hierarchical-GP regression, linear & generalised
                  mixed models (LME / GLMM, REML / FLAME), beta / ordinal /
                  location-scale GLMs, regularised covariance / connectivity,
                  PCA, and permutation / TFCE cluster inference.
- ``signal``   -- windowing, interpolation (linear + Lomb-Scargle),
                  polynomial detrend, time-series convolution.
- ``numerics`` -- shape / layout utilities, intensity normalisation.
- ``geometry`` -- grid + sphere + sphere_grid + coords; voxelmorph-
                  style deformable registration primitives.
- ``graph``    -- Laplacian variants, community detection,
                  connectopy (eigh + lobpcg with implicit VJP).
- ``morphology`` -- dilate, erode, open, close, distance_transform,
                    median_filter, susan_emulator.
- ``smoothing`` -- gaussian, bilateral_gaussian.
- ``metrics``  -- differentiable image-similarity metrics (ssd, ncc,
                  lncc, mutual_information, correlation_ratio) for the
                  registration recipes and segmentation / QA losses.
- ``register`` -- pairwise registration recipes (rigid_register,
                  affine_register) composing the geometry / metrics /
                  optimise substrate; NamedTuple outputs.
- ``bias``     -- N4 (Tustison) bias-field correction, plus the
                  B-spline scattered-data approximator and N3/N4
                  histogram-sharpening primitives it is built from.
- ``semiring`` -- the differentiable streaming-kernel substrate
                  (REAL / LOG / TROPICAL_* / EUCLIDEAN / BOOLEAN).
- ``sparse``   -- ELL and sectioned-ELL sparse formats.
- ``augment``  -- pure-numeric augmentation kernels: intensity tone
                  curves + noise, spatial flip / crop / resized-crop,
                  random affine + diffeomorphic fields, and label-to-
                  image (GMM) + bias-field synthesis.
- ``nn``       -- functional neural-network forward-block kernels
                  (scaled_dot_product_attention; selective_scan and fused
                  norms planned) behind the pallas-cuda / jax dispatch.
- ``transport`` -- entropic optimal transport (Sinkhorn / Wasserstein /
                  barycentric map), composed over the ``LOG`` semiring.

Library-wide reproducibility (the ``driver`` axis -- see
``docs/feature-requests/reproducible-dispatch.md``): where an op has more than
one numerically-divergent implementation, the default is hardware-aware; wrap
work in ``with nitrix.reproducible():`` (or set ``NITRIX_REPRODUCIBLE=1``) to
force the canonical variant everywhere for cross-platform / cross-run stability,
and call ``nitrix.divergent_ops()`` to enumerate the sites and their tolerances.
"""

from ._internal.config import (
    DivergentOp,
    divergent_ops,
    reproducible,
    reproducible_enabled,
    set_reproducible,
)

# Eagerly populate the divergent-op registry (the central contract manifest) so
# divergent_ops() is complete at ``import nitrix`` -- without importing each
# subpackage that owns a site.  Import for side effect only.
from ._internal import _divergent_ops as _divergent_ops  # noqa: F401

__all__ = [
    'DivergentOp',
    'divergent_ops',
    'reproducible',
    'reproducible_enabled',
    'set_reproducible',
]
