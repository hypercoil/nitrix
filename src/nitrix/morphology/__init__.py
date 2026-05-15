# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
# isort: skip_file
"""
nitrix.morphology -- mathematical morphology built atop the semiring substrate.

Per SPEC §3.4 / SPEC_UPDATE §3.4 the user-facing API takes
single-channel arrays ``(..., *spatial)`` -- no explicit channel
dim.  Users with multi-channel inputs should ``jax.vmap`` over the
channel axis.

Semiring-backed (specialisations of ``semiring_conv``):

- ``dilate`` -- ``TROPICAL_MAX_PLUS`` conv.
- ``erode``  -- ``TROPICAL_MIN_PLUS`` conv (with the structuring
  element reflected per the standard mathematical-morphology
  convention).
- ``open``   -- erode then dilate.
- ``close``  -- dilate then erode.
- ``distance_transform`` -- iterative ``TROPICAL_MIN_PLUS`` conv
  with a fixed-point Chebyshev / city-block / approximate-Euclidean
  structuring element.

Gather-backed:

- ``median_filter`` -- gather → ``jnp.median``.  Explicitly *not* a
  semiring op (state size is unbounded in the K loop); see
  SPEC_UPDATE §3.4.

Convenience:

- ``susan_emulator`` -- composes ``bilateral_gaussian`` + ``median_filter``
  (raises ``NotImplementedError`` until ``smoothing.bilateral_gaussian``
  lands -- this is the documented deferral).
"""
from ._mm import (
    close,
    dilate,
    distance_transform,
    erode,
    open,
)
from ._median import median_filter
from .pooling import max_pool_with_indices_nd, max_unpool_nd

__all__ = [
    'dilate',
    'erode',
    'open',
    'close',
    'distance_transform',
    'median_filter',
    'max_pool_with_indices_nd',
    'max_unpool_nd',
]

# ``susan_emulator`` lives in ``nitrix.smoothing`` per SPEC_UPDATE §3.3
# (it composes ``bilateral_gaussian`` + ``median_filter`` and is a
# smoothing op that *uses* ``median_filter`` as a sub-step).  We do
# *not* re-export it here -- doing so creates a circular import via
# ``smoothing.susan`` which imports ``morphology.median_filter``.
# Users import it from ``nitrix.smoothing`` directly.
