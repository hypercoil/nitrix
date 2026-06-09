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
    sigma: float,
    *,
    center: Union[float, Float[Array, '...']] = 0.0,
) -> Float[Array, '...']:
    """Unnormalised Gaussian ``exp(-0.5 * ((coords - center) / sigma) ** 2)``.

    Peak 1 at ``coords == center``.  ``coords`` carries the dtype.
    """
    return jnp.exp(-0.5 * ((coords - center) / sigma) ** 2)
