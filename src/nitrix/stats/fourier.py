# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Spectral / analytic-signal utilities.

The bread-and-butter primitives for working with bandpass /
analytic-signal representations of time series:

- ``product_filter`` / ``product_filtfilt`` -- frequency-domain
  convolution via ``rfft`` multiplication.  Forward and zero-
  phase forward-backward variants.
- ``analytic_signal`` -- compute the complex analytic signal
  ``x + i * Hilbert(x)`` whose real part is ``x`` and whose
  imaginary part is the Hilbert transform of ``x``.
- ``hilbert_transform`` -- the imaginary part of the analytic
  signal.
- ``envelope`` -- magnitude of the analytic signal.
- ``instantaneous_phase`` / ``instantaneous_frequency`` --
  unwrapped phase and its time derivative.
- ``env_inst`` -- compute envelope + freq + phase from one
  ``analytic_signal`` call.

All functions are reverse-mode differentiable and JIT-friendly.
The Hilbert mask construction is vectorised; FFT shape comes
from ``axis`` and (optionally) explicit ``n``.

What changed from ``nitrix.functional.fourier``:

- ``form_docstring`` machinery removed; plain docstrings.
- Hilbert mask built from a single ``jnp.where`` chain rather
  than four scatters.  Compiles to equivalent HLO but reads
  cleaner.
- ``analytic_signal`` raises ``TypeError`` (not ``ValueError``)
  for complex input -- matches the type-error intent.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import jax.numpy as jnp
from jaxtyping import Array, Complex, Float, Num


__all__ = [
    'product_filter',
    'product_filtfilt',
    'analytic_signal',
    'hilbert_transform',
    'envelope',
    'instantaneous_phase',
    'instantaneous_frequency',
    'env_inst',
]


def product_filter(
    X: Num[Array, '... obs'],
    weight: Num[Array, '... obs_freq'],
    **fft_params,
) -> Num[Array, '... obs']:
    '''Frequency-domain convolution via ``rfft`` multiplication.

    Equivalent to a circular convolution along the trailing axis.

    Parameters
    ----------
    X
        Multi-channel signal, trailing axis is observations.
    weight
        Frequency-domain weights, trailing axis is the rfft
        frequency axis (length ``obs // 2 + 1``).
    **fft_params
        Forwarded to ``jnp.fft.rfft`` / ``irfft``.

    Returns
    -------
    Filtered signal, same shape as ``X``.
    '''
    n = X.shape[-1]
    Xf = jnp.fft.rfft(X, n=n, **fft_params)
    return jnp.fft.irfft(weight * Xf, n=n, **fft_params)


def product_filtfilt(
    X: Num[Array, '... obs'],
    weight: Num[Array, '... obs_freq'],
    **fft_params,
) -> Num[Array, '... obs']:
    '''Zero-phase forward-backward frequency-domain filter.

    Filter forward, reverse, filter again, reverse back.  Net
    effect is zero phase delay even when ``weight`` is complex;
    amplitude response is quadratic in the filter weight.  For
    real ``weight``, ``product_filter`` already has zero phase.
    '''
    X_filt = product_filter(X, weight, **fft_params)
    return jnp.flip(
        product_filter(jnp.flip(X_filt, axis=-1), weight, **fft_params),
        axis=-1,
    )


def _hilbert_mask(n: int, dtype) -> Float[Array, 'n']:
    '''Hilbert mask for the analytic signal at FFT length ``n``.

    Standard recipe (Marple 1999):

    - DC (``k = 0``): 1.
    - Positive frequencies (``0 < k < n / 2``): 2.
    - Nyquist (``k = n / 2``, even ``n`` only): 1.
    - Negative frequencies (``k > n / 2``): 0.
    '''
    k = jnp.arange(n)
    if n % 2 == 0:
        return jnp.where(
            k == 0, 1.0,
            jnp.where(
                k < n // 2, 2.0,
                jnp.where(k == n // 2, 1.0, 0.0),
            ),
        ).astype(dtype)
    return jnp.where(
        k == 0, 1.0,
        jnp.where(k < (n + 1) // 2, 2.0, 0.0),
    ).astype(dtype)


def _reshape_for_axis(h: Array, axis: int, reference: Array) -> Array:
    '''Reshape a 1-D mask so it broadcasts against ``reference`` along ``axis``.'''
    shape = [1] * reference.ndim
    shape[axis % reference.ndim] = -1
    return h.reshape(shape)


def analytic_signal(
    X: Float[Array, '...'],
    *,
    axis: int = -1,
    n: Optional[int] = None,
) -> Complex[Array, '...']:
    '''Complex analytic signal of a real time series.

    Satisfies ``Re(X_a) = X``, ``Im(X_a) = Hilbert(X)``;
    ``|X_a|`` is the envelope and ``angle(X_a)`` is the
    instantaneous phase.

    Parameters
    ----------
    X
        Real-valued signal.  Complex input raises ``TypeError``.
    axis
        Axis along which the FFT is taken.  Default ``-1``.
    n
        FFT length.  ``None`` (default) uses ``X.shape[axis]``;
        larger ``n`` zero-pads.

    Returns
    -------
    The complex analytic signal with the same shape as ``X``
    (or with the ``axis`` resized to ``n``).
    '''
    if jnp.iscomplexobj(X):
        raise TypeError(
            'analytic_signal: input must be strictly real; '
            f'got dtype={X.dtype}.'
        )
    Xf = jnp.fft.fft(X, n=n, axis=axis)
    n_eff = n if n is not None else X.shape[axis]
    h = _hilbert_mask(n_eff, dtype=Xf.real.dtype)
    h = _reshape_for_axis(h, axis, Xf)
    return jnp.fft.ifft(Xf * h, axis=axis)


def hilbert_transform(
    X: Float[Array, '...'],
    *,
    axis: int = -1,
    n: Optional[int] = None,
) -> Float[Array, '...']:
    '''Hilbert transform of a real signal.

    Equivalent to ``analytic_signal(X).imag``.
    '''
    return analytic_signal(X, axis=axis, n=n).imag


def envelope(
    X: Float[Array, '...'],
    *,
    axis: int = -1,
    n: Optional[int] = None,
) -> Float[Array, '...']:
    '''Magnitude of the analytic signal: ``|X_a|``.

    If you also need ``instantaneous_phase`` or
    ``instantaneous_frequency``, use ``env_inst`` to compute all
    three from a single analytic-signal call.
    '''
    return jnp.abs(analytic_signal(X, axis=axis, n=n))


def instantaneous_phase(
    X: Float[Array, '...'],
    *,
    axis: int = -1,
    n: Optional[int] = None,
    period: float = 2 * math.pi,
) -> Float[Array, '...']:
    '''Unwrapped instantaneous phase: ``unwrap(angle(X_a))``.'''
    return jnp.unwrap(
        jnp.angle(analytic_signal(X, axis=axis, n=n)),
        axis=axis,
        period=period,
    )


def instantaneous_frequency(
    X: Float[Array, '...'],
    *,
    axis: int = -1,
    n: Optional[int] = None,
    fs: float = 1.0,
    period: float = 2 * math.pi,
) -> Float[Array, '...']:
    '''Instantaneous frequency: ``fs * diff(unwrap(angle)) / period``.

    The output axis is one shorter than the input axis (discrete
    derivative).
    '''
    phase = instantaneous_phase(X, axis=axis, n=n, period=period)
    return fs * jnp.diff(phase, axis=axis) / period


def env_inst(
    X: Float[Array, '...'],
    *,
    axis: int = -1,
    n: Optional[int] = None,
    fs: float = 1.0,
    period: float = 2 * math.pi,
) -> Tuple[Float[Array, '...'], Float[Array, '...'], Float[Array, '...']]:
    '''Envelope + instantaneous frequency + phase from one ``analytic_signal`` call.

    More efficient than three separate calls when you need all
    three quantities -- they share the analytic-signal
    computation.

    Returns
    -------
    ``(envelope, instantaneous_frequency, instantaneous_phase)``.
    The frequency axis is one shorter than the phase axis.
    '''
    Xa = analytic_signal(X, axis=axis, n=n)
    env = jnp.abs(Xa)
    phase = jnp.unwrap(jnp.angle(Xa), axis=axis, period=period)
    freq = fs * jnp.diff(phase, axis=axis) / period
    return env, freq, phase
