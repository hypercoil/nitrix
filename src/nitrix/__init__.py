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
"""
