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

__all__ = [
    'dilate',
    'erode',
    'open',
    'close',
    'distance_transform',
]

# ``_median`` and ``_susan`` are wired in by their own modules below as
# they land; keeping the imports staged avoids partial-module import
# errors when we work in increments.
try:
    from ._median import median_filter
    __all__.append('median_filter')
except ImportError:
    pass

try:
    from ._susan import susan_emulator
    __all__.append('susan_emulator')
except ImportError:
    pass
