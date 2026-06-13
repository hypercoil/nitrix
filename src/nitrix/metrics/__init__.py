# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
# isort: skip_file
"""
nitrix.metrics -- differentiable comparison kernels.

The *score kernels* that compare arrays and return a numeric tensor with
genuine numerical content: similarity, overlap, classification, and
contrastive objectives.  Per ``SPEC_UPDATE_v0.5 §1`` these are the nitrix
half of the score-kernel / scalarisation boundary -- they return the
unreduced (or flat-reduced) score; turning a score into a *loss* (sign,
scalarisation, multi-term weighting) is nimox's ``scalarise`` / ``scheme``.
"Loss" is therefore a nimox concept; ``metrics`` ships the kernels (a
metric becomes a loss via ``1 - metric`` or its cross-entropy + scalarise).

Five families:

- ``intensity`` -- ``ssd``, ``ncc`` (global), ``lncc`` (local /
  windowed).  Within-modality; ``lncc`` is robust to smooth intensity
  inhomogeneity and is the diffeomorphic-recipe default.
- ``information`` -- ``joint_histogram``, ``mutual_information`` (and
  normalised MI), ``mi_grad`` (the closed-form Mattes ``∂MI/∂moving``),
  ``correlation_ratio``.  Cross-modal; assume only a functional (MI:
  arbitrary; CR: deterministic) intensity relationship.
- ``overlap`` -- ``dice`` and ``jaccard`` (IoU), the soft
  region-overlap coefficients on probabilistic masks.
- ``classification`` -- ``bce_with_logits``,
  ``cross_entropy_with_logits``, ``focal_loss``: the cross-entropy
  family (numerically stable from-logits forms) for supervised
  classification / segmentation.
- ``contrastive`` -- ``info_nce`` (InfoNCE / NT-Xent),
  ``dino_cross_entropy`` / ``ibot_cross_entropy`` (self-distillation),
  ``koleo`` (feature-spread entropy regulariser): self-supervised
  representation kernels.

Substrate-composition note (SPEC_UPDATE_v0.3 §0 invariant): ``lncc``'s
local sums are a separable box filter (the ``_internal.separable``
engine shared with ``geometry.spatial_gradient``); ``ncc`` is the
``stats.corr`` shape; MI / CR are soft (Parzen) histogram scatter-adds;
all share the one ``_internal.reductions`` leaf.  No new kernel.  All are
differentiable w.r.t. their array arguments.
"""

from .intensity import lncc, lncc_grad, ncc, ssd
from .information import (
    correlation_ratio,
    joint_histogram,
    mi_grad,
    mutual_information,
)
from .preprocess import match_histogram, winsorize
from .overlap import dice, jaccard
from .classification import (
    bce_with_logits,
    cross_entropy_with_logits,
    focal_loss,
)
from .contrastive import (
    dino_cross_entropy,
    ibot_cross_entropy,
    info_nce,
    koleo,
)

__all__ = [
    # intensity
    'ssd',
    'ncc',
    'lncc',
    'lncc_grad',
    # information
    'joint_histogram',
    'mutual_information',
    'mi_grad',
    'correlation_ratio',
    # preprocessing (fMRIPrep front-end)
    'winsorize',
    'match_histogram',
    # overlap
    'dice',
    'jaccard',
    # classification
    'bce_with_logits',
    'cross_entropy_with_logits',
    'focal_loss',
    # contrastive / self-supervised
    'info_nce',
    'dino_cross_entropy',
    'ibot_cross_entropy',
    'koleo',
]
