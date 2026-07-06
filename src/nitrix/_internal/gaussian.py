# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
The one separable-Gaussian profile.

A single definition of the 1-D Gaussian functional form
``exp(-0.5 * ((x - center) / sigma) ** 2)``, shared by the separable
Gaussian builders that would otherwise each inline it: the convolution
kernel in ``smoothing.gaussian`` (sum-normalised) and the patch window in
``numerics.spatial.gaussian_window`` (peak-normalised).  Callers supply the
coordinates and apply whatever normalisation they need; this owns only the
profile.

Not used by ``bias._sharpen``: that builds a *frequency-domain*, amplitude-
scaled Gaussian (the N3/N4 deconvolution kernel), a different
parameterisation that this profile would obscure rather than unify.
"""

from __future__ import annotations

from typing import Union

import jax.numpy as jnp
from jaxtyping import Array, Float


def gaussian_profile_1d(
    coords: Float[Array, '...'],
    sigma: Union[float, Float[Array, '...']],
    *,
    center: Union[float, Float[Array, '...']] = 0.0,
) -> Float[Array, '...']:
    """Evaluate the unnormalised 1-D Gaussian profile at given coordinates.

    Computes the bare Gaussian functional form
    :math:`\\exp\\!\\left(-\\tfrac{1}{2}\\left(\\frac{x - c}{\\sigma}\\right)^2\\right)`
    at each coordinate :math:`x`, with peak value 1 attained where
    ``coords`` equals ``center``. No normalisation is applied; callers
    supply whatever normalisation (sum- or peak-normalisation) they
    require. The result carries the dtype of ``coords``.

    Parameters
    ----------
    coords : Float[Array, '...']
        Coordinates at which to evaluate the profile, of arbitrary shape.
        The output dtype follows this array.
    sigma : float
        Standard deviation :math:`\\sigma` of the Gaussian, controlling its
        width.
    center : float or Float[Array, '...'], optional
        Location :math:`c` of the peak, broadcast against ``coords``.
        Defaults to ``0.0``.

    Returns
    -------
    Float[Array, '...']
        The Gaussian profile evaluated at ``coords``, of the same shape as
        the broadcast of ``coords`` and ``center``, with peak value 1 where
        ``coords`` equals ``center``.
    """
    return jnp.exp(-0.5 * ((coords - center) / sigma) ** 2)
