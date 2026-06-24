# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
# isort: skip_file
"""
nitrix.morphology -- mathematical morphology built atop the semiring substrate.

Per SPEC §4.3 / SPEC §4.3 the user-facing API takes
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
- ``distance_transform`` -- **exact Euclidean** DT by default (separable,
  each axis a ``TROPICAL_MIN_PLUS`` matmul against the squared-distance
  matrix -- reuses the semiring Pallas-CUDA kernel), with an opt-in chamfer
  engine (``metric="chebyshev"`` / ``"city_block"`` / custom structuring
  element) on the iterative ``TROPICAL_MIN_PLUS`` conv substrate.
- ``distance_transform_edt`` -- the exact-Euclidean path as a scipy-named
  alias.

Gather-backed:

- ``median_filter`` -- gather → ``jnp.median``.  Explicitly *not* a
  semiring op (state size is unbounded in the K loop); see
  SPEC §4.3.

Labelling:

- ``connected_components`` / ``largest_connected_component`` -- N-D
  connected-components labelling by jit-able fixed-point label
  propagation (the recurring mask clean-up / largest-region step).

Convenience:

- ``susan_emulator`` -- composes ``bilateral_gaussian`` + ``median_filter``
  (raises ``NotImplementedError`` until ``smoothing.bilateral_gaussian``
  lands -- this is the documented deferral).
"""

from ._mm import (
    close,
    dilate,
    distance_transform,
    distance_transform_edt,
    erode,
    open,
)
from ._median import median_filter
from ._label import connected_components, largest_connected_component
from .pooling import max_pool_with_indices_nd, max_unpool_nd

__all__ = [
    'dilate',
    'erode',
    'open',
    'close',
    'distance_transform',
    'distance_transform_edt',
    'median_filter',
    'connected_components',
    'largest_connected_component',
    'max_pool_with_indices_nd',
    'max_unpool_nd',
]

# ``susan_emulator`` lives in ``nitrix.smoothing`` per SPEC §4.4
# (it composes ``bilateral_gaussian`` + ``median_filter`` and is a
# smoothing op that *uses* ``median_filter`` as a sub-step).  We do
# *not* re-export it here -- doing so creates a circular import via
# ``smoothing.susan`` which imports ``morphology.median_filter``.
# Users import it from ``nitrix.smoothing`` directly.
