# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
# isort: skip_file
"""
Differentiable comparison kernels.

The *score kernels* that compare arrays and return a numeric tensor with
genuine numerical content: similarity, overlap, classification, and
contrastive objectives.  These return the unreduced (or flat-reduced)
score; turning a score into a *loss* (sign, scalarisation, multi-term
weighting) is handled downstream.  A metric becomes a loss via, for
example, ``1 - metric`` or its cross-entropy composed with a scalarisation.

Six families:

- intensity -- :func:`ssd`, :func:`ncc` (global), :func:`lncc` (local /
  windowed).  Within-modality; :func:`lncc` is robust to smooth intensity
  inhomogeneity and is the diffeomorphic-recipe default.
- information -- :func:`joint_histogram`, :func:`mutual_information` (and
  normalised MI), :func:`mi_grad` (the closed-form Mattes gradient
  :math:`\\partial\\,\\mathrm{MI} / \\partial\\,\\text{moving}`),
  :func:`correlation_ratio`.  Cross-modal; these assume only a functional
  intensity relationship (arbitrary for MI; deterministic for the
  correlation ratio).
- overlap -- :func:`dice` and :func:`jaccard` (IoU), the soft
  region-overlap coefficients on probabilistic masks.
- classification -- the cross-entropy family (:func:`bce_with_logits`,
  :func:`cross_entropy_with_logits`, :func:`focal_loss`: numerically stable
  from-logits losses) plus the evaluation metrics :func:`roc_auc` (rank /
  Mann-Whitney AUROC), :func:`confusion_matrix`, and :func:`topk_accuracy`
  (reporting; non-differentiable).
- contrastive -- :func:`info_nce` (InfoNCE / NT-Xent),
  :func:`dino_cross_entropy` / :func:`ibot_cross_entropy`
  (self-distillation), :func:`koleo` (feature-spread entropy regulariser):
  self-supervised representation kernels.
- surface -- :func:`hausdorff95` and :func:`surface_dice`: boundary-distance
  reporting metrics over the erosion + Euclidean distance transform
  substrate, parity pinned to MONAI (reporting; non-differentiable).

Notes
-----
The families share a common numerical substrate rather than introducing
new kernels.  The local sums of :func:`lncc` are a separable box filter
(the same engine that backs the spatial gradient); :func:`ncc` follows the
correlation shape; mutual information and the correlation ratio are soft
(Parzen) histogram scatter-adds; the surface metrics reuse the erosion and
Euclidean distance transform operations.  All share a single reduction
leaf.  The comparison kernels are differentiable with respect to their
array arguments; the classification evaluation metrics and the surface
metrics produce hard (reporting) outputs and are not differentiable.
"""

from .intensity import lncc, lncc_grad, lncc_grad_center, ncc, ssd
from .information import (
    correlation_ratio,
    joint_histogram,
    mi_grad,
    mutual_information,
    nmi_grad,
)
from .preprocess import match_histogram, winsorize
from .overlap import dice, jaccard
from .surface import hausdorff95, surface_dice
from .classification import (
    bce_with_logits,
    confusion_matrix,
    cross_entropy_with_logits,
    focal_loss,
    roc_auc,
    topk_accuracy,
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
    'lncc_grad_center',
    # information
    'joint_histogram',
    'mutual_information',
    'mi_grad',
    'nmi_grad',
    'correlation_ratio',
    # preprocessing (fMRIPrep front-end)
    'winsorize',
    'match_histogram',
    # overlap
    'dice',
    'jaccard',
    # surface distance
    'hausdorff95',
    'surface_dice',
    # classification
    'bce_with_logits',
    'cross_entropy_with_logits',
    'focal_loss',
    'roc_auc',
    'confusion_matrix',
    'topk_accuracy',
    # contrastive / self-supervised
    'info_nce',
    'dino_cross_entropy',
    'ibot_cross_entropy',
    'koleo',
]
