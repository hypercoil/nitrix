# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
SUSAN emulator convenience wrapper.

Per SPEC §4.4, ``susan_emulator`` composes
``smoothing.bilateral_gaussian`` (for the brightness-similarity
weighting half) with ``morphology.median_filter`` (for the impulse-
noise half FSL's SUSAN handles via local median).

``bilateral_gaussian`` is a Phase 4 ``smoothing`` capability not yet
implemented; until it lands, ``susan_emulator`` raises a clear
``NotImplementedError`` pointing the user at ``median_filter`` (for
the impulse-noise half on its own) or at ``smoothing.gaussian`` (for
the simple-Gaussian half on its own).  This matches the SPEC's
"reserve the namespace; raise with a pointer until the dependency
lands" pattern.
"""

from __future__ import annotations

from typing import Optional

from jaxtyping import Array, Float

__all__ = ['susan_emulator']


def susan_emulator(
    image: Float[Array, '... *spatial'],
    *,
    sigma_space: float,
    sigma_intensity: float,
    use_median: bool = True,
    bthresh: Optional[float] = None,
) -> Float[Array, '... *spatial']:
    """SUSAN-style edge-preserving smoothing.

    Composes a bilateral (edge-preserving) Gaussian with an optional
    local-median fallback to reproduce FSL SUSAN's behaviour.

    Currently a stub: ``smoothing.bilateral_gaussian`` is not yet
    implemented.  Raises ``NotImplementedError`` with a pointer at
    the alternatives.

    Notes (for when the dependency lands)
    -------------------------------------
    - The brightness-similarity weighting that FSL SUSAN does
      explicitly is recovered by feeding intensity into the
      bilateral filter's feature space.
    - The median fallback for impulse noise is recovered by
      composing with ``median_filter``.
    - The auto-flat-kernel-at-small-extents behaviour from FSL
      SUSAN is *not* replicated; this is the documented behavioural
      delta.
    """
    raise NotImplementedError(
        'susan_emulator depends on smoothing.bilateral_gaussian, '
        'which has not yet shipped.  For the impulse-noise half, '
        'call morphology.median_filter directly; for the '
        'simple-Gaussian half, smoothing.gaussian (once Phase 4 '
        'lands).'
    )
