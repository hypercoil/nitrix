# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
# isort: skip_file
"""
nitrix.metrics -- differentiable image-similarity metrics.

The similarity reductions that drive the registration recipes
(``nitrix.register``) and double as segmentation / QA losses.  Two
families:

- ``intensity`` -- ``ssd``, ``ncc`` (global), ``lncc`` (local /
  windowed).  Within-modality; ``lncc`` is robust to smooth intensity
  inhomogeneity and is the diffeomorphic-recipe default.
- ``information`` -- ``joint_histogram``, ``mutual_information`` (and
  normalised MI), ``correlation_ratio``.  Cross-modal; assume only a
  functional (MI: arbitrary; CR: deterministic) intensity relationship.
- ``overlap`` -- ``dice`` and ``jaccard`` (IoU), the soft
  region-overlap coefficients (segmentation similarities / losses on
  probabilistic masks).
- ``classification`` -- ``bce_with_logits``,
  ``cross_entropy_with_logits``, ``focal_loss``: the cross-entropy
  family (numerically stable from-logits forms) for supervised
  classification / segmentation.
- ``contrastive`` -- ``nt_xent`` (InfoNCE), ``dino_cross_entropy`` /
  ``ibot_cross_entropy`` (self-distillation), ``koleo`` (feature-spread
  entropy regulariser): self-supervised representation losses.

Substrate-composition note (SPEC_UPDATE_v0.3 §0 invariant): ``lncc``'s
local sums are a separable box filter (the ``_internal.separable``
engine shared with ``geometry.spatial_gradient``); ``ncc`` is the
``stats.corr`` shape; MI / CR are soft (Parzen) histogram scatter-adds.
No new kernel.  All metrics are differentiable w.r.t. their image
arguments so they sit inside a registration loss.
"""

from .intensity import lncc, ncc, ssd
from .information import (
    correlation_ratio,
    joint_histogram,
    mutual_information,
)
from .overlap import dice, jaccard
from .classification import (
    bce_with_logits,
    cross_entropy_with_logits,
    focal_loss,
)
from .contrastive import (
    dino_cross_entropy,
    ibot_cross_entropy,
    koleo,
    nt_xent,
)

__all__ = [
    # intensity
    'ssd',
    'ncc',
    'lncc',
    # information
    'joint_histogram',
    'mutual_information',
    'correlation_ratio',
    # overlap
    'dice',
    'jaccard',
    # classification
    'bce_with_logits',
    'cross_entropy_with_logits',
    'focal_loss',
    # contrastive / self-supervised
    'nt_xent',
    'dino_cross_entropy',
    'ibot_cross_entropy',
    'koleo',
]
