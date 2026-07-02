# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
SUSAN emulator convenience wrapper.

:func:`susan_emulator` composes :func:`bilateral_gaussian` (for the
brightness-similarity weighting half) with :func:`median_filter` (for
the impulse-noise half that FSL's SUSAN handles via a local median).

In this stub the composition is not yet wired up, so
:func:`susan_emulator` raises a clear :class:`NotImplementedError`
pointing the user at :func:`median_filter` (for the impulse-noise half
on its own) or at :func:`gaussian` (for the simple-Gaussian half on its
own).
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
    local-median fallback to reproduce the behaviour of FSL's SUSAN.
    The brightness-similarity weighting is contributed by the bilateral
    Gaussian, while the local-median step suppresses impulse noise.

    Currently a stub: the composition is not yet wired up, so the call
    raises :class:`NotImplementedError` with a pointer at the
    alternatives (:func:`median_filter` for the impulse-noise half on
    its own, or :func:`gaussian` for the simple-Gaussian half).

    Parameters
    ----------
    image : Float[Array, '... *spatial']
        Input image, with one or more leading batch/channel axes
        followed by the spatial axes to be smoothed.
    sigma_space : float
        Standard deviation of the spatial (geometric) Gaussian weight,
        in voxel units.  Larger values pool over a wider neighbourhood.
    sigma_intensity : float
        Standard deviation of the brightness-similarity (range) weight.
        Controls how strongly intensity differences down-weight a
        neighbour, giving the filter its edge-preserving character.
    use_median : bool, default=True
        If ``True``, apply the local-median fallback that handles
        impulse noise; if ``False``, use the bilateral Gaussian alone.
    bthresh : float or None, optional
        Optional bilateral weight threshold, accepted for API compatibility
        with FSL-SUSAN.  Advisory only: it clips the per-edge weight at the
        threshold rather than acting as a hard cutoff.

    Returns
    -------
    Float[Array, '... *spatial']
        The edge-preserving smoothed image, with the same shape and
        dtype as ``image``.

    Notes
    -----
    - The brightness-similarity weighting that FSL SUSAN does
      explicitly is recovered by feeding intensity into the bilateral
      filter's feature space.
    - The median fallback for impulse noise is recovered by composing
      with :func:`median_filter`.
    - The auto-flat-kernel-at-small-extents behaviour from FSL SUSAN is
      *not* replicated; this is the documented behavioural delta.
    """
    raise NotImplementedError(
        'susan_emulator depends on smoothing.bilateral_gaussian, '
        'which has not yet shipped.  For the impulse-noise half, '
        'call morphology.median_filter directly; for the '
        'simple-Gaussian half, smoothing.gaussian (once Phase 4 '
        'lands).'
    )
