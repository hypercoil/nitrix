# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Tensor shape / layout utilities.

The "low-level building blocks" that downstream consumers compose:

- **Axis normalisation**: ``standard_axis_number``,
  ``negative_axis_number`` -- turn negative or out-of-range axis
  indices into canonical positive indices.
- **Reshape helpers**: ``fold_axis`` (one axis -> two),
  ``unfold_axes`` (multiple -> one), and the
  ``fold_and_promote`` / ``demote_and_unfold`` composites.
- **Broadcasting helpers**: ``orient_and_conform`` (pad with
  singleton axes so a vector broadcasts against a reference),
  ``broadcast_ignoring`` (broadcast all axes except a named
  one), ``promote_to_rank``.
- **Complex utilities**: ``complex_decompose`` (split into
  amplitude / phase), ``complex_recompose``, ``amplitude_apply``
  (apply a real-valued function to a complex tensor via its
  amplitude).
- **Masking helpers**: ``conform_mask``, ``apply_mask``,
  ``mask_tensor``.

Most of these existed as internal utilities in
``nitrix._internal.util``; this module promotes the
externally-useful subset to a public namespace.  The internal
``_internal.util`` continues to be the implementation detail
(used by other nitrix subpackages); this module re-exports the
public surface with cleaner docstrings.
"""

from __future__ import annotations

from .._internal.util import (
    amplitude_apply,
    apply_mask,
    atleast_4d,
    axis_complement,
    broadcast_ignoring,
    complex_decompose,
    complex_recompose,
    conform_mask,
    demote_and_unfold,
    demote_axis,
    extend_to_max_size,
    extend_to_size,
    fold_and_promote,
    fold_axis,
    mask_tensor,
    masker,
    negative_axis_number,
    negative_axis_number_strict,
    orient_and_conform,
    promote_axis,
    promote_to_rank,
    standard_axis_number,
    standard_axis_number_strict,
    unfold_axes,
    vmap_over_outer,
)

__all__ = [
    # Axis normalisation
    'standard_axis_number',
    'standard_axis_number_strict',
    'negative_axis_number',
    'negative_axis_number_strict',
    'axis_complement',
    # Reshape / fold
    'fold_axis',
    'unfold_axes',
    'fold_and_promote',
    'demote_and_unfold',
    'promote_axis',
    'demote_axis',
    # Broadcasting / rank
    'orient_and_conform',
    'broadcast_ignoring',
    'promote_to_rank',
    'atleast_4d',
    'extend_to_size',
    'extend_to_max_size',
    # Complex
    'complex_decompose',
    'complex_recompose',
    'amplitude_apply',
    # Masking
    'conform_mask',
    'apply_mask',
    'mask_tensor',
    'masker',
    # vmap helpers
    'vmap_over_outer',
]
