# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
# isort: skip_file
"""
nitrix.augment -- pure-numeric augmentation primitives.

The deterministic numerical kernels behind data augmentation: intensity
tone curves, noise models, label-to-image synthesis, simulated bias
fields, and the spatial (flip / crop / deformation) transforms.  These
are pure ``(Array, ...[, key]) -> Array`` functions; the augmentation
*policy* (which ops, parameter ranges, RNG scheduling, container / world
-space bookkeeping) belongs to the consumer layers, not here.

Submodules:

- ``intensity`` -- gamma contrast, random histogram shift, Gibbs
  (truncation) ringing, Gaussian / Rician noise.
- ``geometric`` -- random flip / crop / resized-crop, random affine
  matrix, random diffeomorphic (SVF) displacement.
- ``synthesis`` -- label-map-to-image Gaussian-mixture render and
  simulated (forward) multiplicative bias / INU fields.
- ``simulate`` -- synthetic connectivity / time-series generators with
  known ground-truth structure (band-limited sources, covariance
  colouring [the whitening inverse], sparse mixing, the mixture forward
  model, low-rank-block connectomes, Markov state sequences).
"""

from .intensity import (
    gamma_contrast,
    gaussian_noise,
    gibbs_ringing,
    random_histogram_shift,
    rician_noise,
)
from .geometric import (
    random_affine_matrix,
    random_crop,
    random_flip,
    random_resized_crop,
    random_svf_displacement,
)
from .synthesis import (
    gmm_label_to_image,
    simulate_bias_field,
)
from .simulate import (
    band_limited_signals,
    color_signals,
    lowrank_block_connectome,
    markov_state_sequence,
    mix_signals,
    sparse_mixture_matrix,
)

__all__ = [
    # intensity
    'gamma_contrast',
    'random_histogram_shift',
    'gibbs_ringing',
    'gaussian_noise',
    'rician_noise',
    # geometric
    'random_flip',
    'random_crop',
    'random_resized_crop',
    'random_affine_matrix',
    'random_svf_displacement',
    # synthesis
    'gmm_label_to_image',
    'simulate_bias_field',
    # simulate (synthetic connectivity / time series)
    'band_limited_signals',
    'color_signals',
    'sparse_mixture_matrix',
    'mix_signals',
    'lowrank_block_connectome',
    'markov_state_sequence',
]
