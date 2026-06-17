# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
# isort: skip_file
"""
nitrix.stats.inference -- permutation / cluster inference.

The on-device substrate for FSL ``randomise``-style nonparametric inference on
the GLM (the kernel ``niffi`` wraps; the CLI / containers / design parsing stay
in the consumer):

- ``permutation`` -- keyed exchangeability operators (``sign_flips`` /
  ``permutations``) with exchangeability blocks.
- ``tfce``        -- threshold-free cluster enhancement.
- ``cluster``     -- cluster size / mass maps.
- ``randomise``   -- the permutation driver: GLM refit (Freedman-Lane) +
  enhancement per permutation -> FWE / uncorrected p-maps + the null max
  distribution.
- ``multiple_comparisons`` -- FDR / Bonferroni / cluster p-value companions.
"""

from .cluster import cluster_mass_map, cluster_size_map
from .multiple_comparisons import bonferroni, fdr_bh
from .permutation import permutations, sign_flips
from .randomise import PermResult, permutation_test
from .tfce import tfce

__all__ = [
    'sign_flips',
    'permutations',
    'tfce',
    'cluster_size_map',
    'cluster_mass_map',
    'permutation_test',
    'PermResult',
    'fdr_bh',
    'bonferroni',
]
