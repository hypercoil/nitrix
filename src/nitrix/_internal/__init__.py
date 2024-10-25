# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
# isort: skip_file
from .docutil import (
    NestedDocParse,
    tensor_dimensions,
)
from .util import (
    Tensor,
    _conform_bform_weight,
    _dim_or_none,
    _compose,
    _seq_pad,
    atleast_4d,
    apply_vmap_over_outer,
    vmap_over_outer,
    broadcast_ignoring,
    orient_and_conform,
    promote_axis,
    demote_axis,
    fold_axis,
    unfold_axes,
    axis_complement,
    standard_axis_number,
    negative_axis_number,
    fold_and_promote,
    demote_and_unfold,
    promote_to_rank,
    extend_to_size,
    extend_to_max_size,
    argsort,
)
