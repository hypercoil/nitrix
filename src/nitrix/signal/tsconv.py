# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
1-D time-series convolution along the trailing axis.

A thin batched wrapper around ``lax.conv_general_dilated`` for
1-D temporal convolution with the typical fMRI / time-series
layout ``(..., channels, observations)``.

What the legacy ``hypercoil.functional.tsconv`` had that we drop:

- ``tsconv2d`` -- "2D conv where one axis is time" via separate
  ``lax.conv_general_dilated`` calls.  Users wanting that should
  call ``lax.conv_general_dilated`` directly with the explicit
  dimension-numbers spec; wrapping it doesn't save code.
- ``basisconv2d``, ``polyconv2d``, ``basischan``, ``polychan`` --
  scientific-question-specific (basis-function / polynomial-channel
  construction for HRF modelling).  Live elsewhere when they live
  again (probably ``hypercoil``-side or a downstream consumer's
  utility module).
"""
from __future__ import annotations

from typing import Literal

import jax
import jax.lax as lax
import jax.numpy as jnp
from jaxtyping import Array, Float, Num


__all__ = ['tsconv']


Padding = Literal['SAME', 'VALID']


def tsconv(
    X: Num[Array, '... C obs'],
    weight: Float[Array, 'C_out C K'],
    *,
    padding: Padding = 'SAME',
) -> Num[Array, '... C_out obs_out']:
    '''1-D convolution along the trailing axis.

    Parameters
    ----------
    X
        Input time series, ``(..., C, obs)`` -- the trailing two
        axes are channels and observations.  Leading axes are
        broadcast / batched.
    weight
        Convolution kernel, ``(C_out, C, K)`` -- standard
        channels-out / channels-in / kernel-size layout.
    padding
        ``"SAME"`` (default): output has the same observation
        length as the input.  ``"VALID"``: output length is
        ``obs - K + 1``.

    Returns
    -------
    Convolved time series, ``(..., C_out, obs_out)``.

    Notes
    -----
    Wraps ``jax.lax.conv_general_dilated`` with the standard
    NCW (batch, channel, time) dim spec.  Batches over any leading
    axes by reshaping into a single batch dim and unwrapping.
    For multi-channel or multi-subject fMRI ``(B, C, T)`` is the
    natural input shape.

    This is the **cross-correlation** convention (the kernel is
    **not** flipped), as in deep-learning conv layers
    (``torch.nn.Conv1d`` is likewise cross-correlation).  A DSP
    "convolution" flips the kernel about its centre; reverse
    ``weight`` along its last (``K``) axis -- ``weight[..., ::-1]``
    -- if you need a true (flipped) convolution.
    '''
    if X.ndim < 2:
        raise ValueError(
            f'tsconv: X must have at least 2 dims (C, obs); got '
            f'{X.shape}.'
        )
    if weight.ndim != 3:
        raise ValueError(
            f'tsconv: weight must have 3 dims (C_out, C_in, K); '
            f'got {weight.shape}.'
        )
    C, obs = X.shape[-2], X.shape[-1]
    C_out, C_in, K = weight.shape
    if C != C_in:
        raise ValueError(
            f'tsconv: X channel dim {C} != weight in-channel dim {C_in}.'
        )
    batch_shape = X.shape[:-2]
    # Reshape to (batch, C, obs) for conv_general_dilated.
    X_b = X.reshape((-1, C, obs)) if batch_shape else X[None, ...]
    out = lax.conv_general_dilated(
        X_b, weight,
        window_strides=(1,),
        padding=padding,
        dimension_numbers=('NCT', 'OIT', 'NCT'),
    )
    # Unwrap leading batch.
    obs_out = out.shape[-1]
    if batch_shape:
        return out.reshape(batch_shape + (C_out, obs_out))
    return out[0]
