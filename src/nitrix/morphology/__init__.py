# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
# isort: skip_file
"""
Mathematical morphology built atop the semiring substrate.

The user-facing API takes single-channel arrays ``(..., *spatial)`` with
no explicit channel dimension. Callers with multi-channel inputs should
``jax.vmap`` over the channel axis.

Semiring-backed (specialisations of :func:`semiring_conv`):

- :func:`dilate` -- a :data:`TROPICAL_MAX_PLUS` convolution.
- :func:`erode` -- a :data:`TROPICAL_MIN_PLUS` convolution (with the
  structuring element reflected per the standard mathematical-morphology
  convention).
- :func:`open` -- erode then dilate.
- :func:`close` -- dilate then erode.
- :func:`distance_transform` -- an exact Euclidean distance transform by
  default (separable, each axis a :data:`TROPICAL_MIN_PLUS` matmul against
  the squared-distance matrix, reusing the semiring Pallas-CUDA kernel),
  with an opt-in chamfer engine (``metric="chebyshev"`` /
  ``"city_block"`` / a custom structuring element) on the iterative
  :data:`TROPICAL_MIN_PLUS` convolution substrate.
- :func:`distance_transform_edt` -- the exact-Euclidean path exposed under
  the scipy-compatible name.

Gather-backed:

- :func:`median_filter` -- a gather followed by ``jnp.median``. Explicitly
  not a semiring op, because the state size is unbounded in the kernel
  loop.

Labelling:

- :func:`connected_components` / :func:`largest_connected_component` --
  N-dimensional connected-components labelling by a jit-able fixed-point
  label-propagation scheme (the recurring mask clean-up and
  largest-region step).

Convenience:

- ``susan_emulator`` -- composes ``bilateral_gaussian`` and
  :func:`median_filter`. It raises ``NotImplementedError`` until
  ``smoothing.bilateral_gaussian`` lands; this is a deliberate deferral.
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
