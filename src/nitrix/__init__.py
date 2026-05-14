# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
# isort: skip_file
"""
nitrix -- pure-JAX numerical substrate for neuroimaging research and
differentiable programming.

Top-level subpackages (full surface in each ``__init__.py``):

- ``linalg``   -- matrix utilities, residualisation, kernels, SPD ops.
- ``stats``    -- (paired / partial / conditional) covariance and
                  correlation; spectral / analytic-signal utilities.
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
- ``semiring`` -- the differentiable streaming-kernel substrate
                  (REAL / LOG / TROPICAL_* / EUCLIDEAN / BOOLEAN).
- ``sparse``   -- ELL and sectioned-ELL sparse formats.
"""
