# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
# isort: skip_file
"""
nitrix.stats.inference -- permutation / cluster inference.

The on-device substrate for FSL ``randomise``-style nonparametric inference on
the GLM (the kernel ``niffi`` wraps; the CLI / containers / design parsing stay
in the consumer):

- ``permutation`` -- keyed exchangeability operators (:func:`sign_flips` /
  :func:`permutations`) with exchangeability blocks.
- ``tfce``        -- threshold-free cluster enhancement (:func:`tfce`).
- ``cluster``     -- cluster size / mass maps.
- ``randomise``   -- the permutation driver: GLM refit (Freedman-Lane) +
  enhancement per permutation -> FWE / uncorrected p-maps + the null max
  distribution.
- ``multiple_comparisons`` -- FDR / Bonferroni / cluster p-value companions.

Permutation-only, by design
---------------------------

This module is **deliberately nonparametric**: cluster / TFCE significance comes
from the permutation null, not from a parametric random-field-theory (RFT)
Gaussian-smoothness model.  Eklund, Nichols & Knutsson (2016, PNAS) showed the
parametric cluster-extent tests (the SPM / FSL-FLAME RFT path) carry badly
inflated false-positive rates under realistic spatial autocorrelation, whereas
the permutation test holds its nominal level.  So there is **no ACF / FWHM
smoothness estimator** here (no ``3dFWHMx`` / ``smoothest`` analogue) and none is
planned -- the exchangeability-based null is the suite's single, defensible
cluster-inference route.  A consumer that needs a parametric fallback for a
tiny-``N`` design (where permutation is degenerate) supplies it itself.

References
----------
Eklund A, Nichols TE, Knutsson H (2016). Cluster failure: why fMRI inferences
for spatial extent have inflated false-positive rates. *Proceedings of the
National Academy of Sciences*, 113(28), 7900-7905.
https://doi.org/10.1073/pnas.1602413113
"""

from .cluster import cluster_mass_map, cluster_size_map
from .multiple_comparisons import (
    bonferroni,
    conjunction,
    conjunction_pvalue,
    fdr,
    fdr_bh,
    fdr_by,
    fdr_storey,
    storey_pi0,
)
from .permutation import permutations, sign_flips
from ._spatial_null import SpatialNullResult, spatial_null_test
from .brainsmash import (
    brainsmash_surrogates,
    brainsmash_surrogates_sampled,
    brainsmash_test,
    brainsmash_test_sampled,
    variogram,
)
from .moran import moran_test
from .randomise import PermResult, gpd_pvalue, permutation_test
from .spin import SpinTestResult, spin_test
from .tfce import tfce

__all__ = [
    'sign_flips',
    'permutations',
    'tfce',
    'cluster_size_map',
    'cluster_mass_map',
    'permutation_test',
    'PermResult',
    'gpd_pvalue',
    'spin_test',
    'SpinTestResult',
    'moran_test',
    'brainsmash_test',
    'brainsmash_test_sampled',
    'brainsmash_surrogates',
    'brainsmash_surrogates_sampled',
    'variogram',
    'spatial_null_test',
    'SpatialNullResult',
    'fdr',
    'fdr_bh',
    'fdr_by',
    'fdr_storey',
    'storey_pi0',
    'bonferroni',
    'conjunction',
    'conjunction_pvalue',
]
