# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Spectral / analytic-signal utilities.

The bread-and-butter primitives for working with bandpass /
analytic-signal representations of time series:

- :func:`product_filter` / :func:`product_filtfilt` --
  frequency-domain convolution via ``rfft`` multiplication.
  Forward and zero-phase forward-backward variants.
- :func:`analytic_signal` -- compute the complex analytic signal
  :math:`x + i\\,\\operatorname{Hilbert}(x)` whose real part is
  :math:`x` and whose imaginary part is the Hilbert transform of
  :math:`x`.
- :func:`hilbert_transform` -- the imaginary part of the analytic
  signal.
- :func:`envelope` -- magnitude of the analytic signal.
- :func:`instantaneous_phase` / :func:`instantaneous_frequency`
  -- unwrapped phase and its time derivative.
- :func:`env_inst` -- compute envelope, frequency, and phase from
  one :func:`analytic_signal` call.

All functions are reverse-mode differentiable and JIT-friendly.
The Hilbert mask construction is vectorised; the FFT shape comes
from ``axis`` and (optionally) an explicit ``n``.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple

import jax.numpy as jnp
from jax.typing import DTypeLike
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
    **fft_params: Any,
) -> Num[Array, '... obs']:
    """Frequency-domain convolution via ``rfft`` multiplication.

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
    """
    n = X.shape[-1]
    Xf = jnp.fft.rfft(X, n=n, **fft_params)
    return jnp.fft.irfft(weight * Xf, n=n, **fft_params)


def product_filtfilt(
    X: Num[Array, '... obs'],
    weight: Num[Array, '... obs_freq'],
    **fft_params: Any,
) -> Num[Array, '... obs']:
    """Zero-phase forward-backward frequency-domain filter.

    Filter forward, reverse the signal, filter again, then reverse
    back.  The net effect is zero phase delay even when ``weight``
    is complex; the amplitude response is the square of the filter
    weight's amplitude response.  For real ``weight``,
    :func:`product_filter` already has zero phase.

    Parameters
    ----------
    X : Num[Array, '... obs']
        Multi-channel signal; the trailing axis indexes
        observations.
    weight : Num[Array, '... obs_freq']
        Frequency-domain weights; the trailing axis is the rfft
        frequency axis (length ``obs // 2 + 1``).
    **fft_params
        Forwarded to ``jnp.fft.rfft`` / ``jnp.fft.irfft`` via
        :func:`product_filter`.

    Returns
    -------
    Num[Array, '... obs']
        Zero-phase filtered signal, same shape as ``X``.
    """
    X_filt = product_filter(X, weight, **fft_params)
    return jnp.flip(
        product_filter(jnp.flip(X_filt, axis=-1), weight, **fft_params),
        axis=-1,
    )


def _hilbert_mask(n: int, dtype: DTypeLike) -> Float[Array, 'n']:
    """Hilbert mask for the analytic signal at FFT length ``n``.

    The multiplier applied to each frequency bin of the DFT before
    the inverse transform, following the standard recipe:

    - DC (:math:`k = 0`): 1.
    - Positive frequencies (:math:`0 < k < n / 2`): 2.
    - Nyquist (:math:`k = n / 2`, even ``n`` only): 1.
    - Negative frequencies (:math:`k > n / 2`): 0.

    Parameters
    ----------
    n : int
        FFT length; determines the number of frequency bins and
        whether a Nyquist bin exists (even ``n`` only).
    dtype : DTypeLike
        Floating dtype the returned mask is cast to.

    Returns
    -------
    Float[Array, 'n']
        The per-bin multiplier, length ``n``.

    References
    ----------
    Marple, S.L. (1999). Computing the discrete-time analytic
    signal via FFT. *IEEE Transactions on Signal Processing*,
    47(9), 2600-2603. https://doi.org/10.1109/78.782222
    """
    k = jnp.arange(n)
    if n % 2 == 0:
        return jnp.where(
            k == 0,
            1.0,
            jnp.where(
                k < n // 2,
                2.0,
                jnp.where(k == n // 2, 1.0, 0.0),
            ),
        ).astype(dtype)
    return jnp.where(
        k == 0,
        1.0,
        jnp.where(k < (n + 1) // 2, 2.0, 0.0),
    ).astype(dtype)


def _reshape_for_axis(h: Array, axis: int, reference: Array) -> Array:
    """Reshape a 1-D mask so it broadcasts against ``reference`` along ``axis``.

    Parameters
    ----------
    h : Array
        One-dimensional mask to be reshaped.
    axis : int
        Axis of ``reference`` along which ``h`` should vary; all
        other axes become length-one broadcast dimensions.
    reference : Array
        Array whose rank the reshaped mask must match.

    Returns
    -------
    Array
        ``h`` reshaped to ``reference.ndim`` dimensions, with its
        length placed on ``axis`` and ones elsewhere.
    """
    shape = [1] * reference.ndim
    shape[axis % reference.ndim] = -1
    return h.reshape(shape)


def analytic_signal(
    X: Float[Array, '...'],
    *,
    axis: int = -1,
    n: Optional[int] = None,
) -> Complex[Array, '...']:
    """Complex analytic signal of a real time series.

    The analytic signal :math:`X_a` satisfies
    :math:`\\operatorname{Re}(X_a) = X` and
    :math:`\\operatorname{Im}(X_a) = \\operatorname{Hilbert}(X)`;
    its magnitude :math:`|X_a|` is the envelope and its argument
    :math:`\\arg(X_a)` is the instantaneous phase.

    Parameters
    ----------
    X : Float[Array, '...']
        Real-valued signal.  Complex input raises a ``TypeError``.
    axis : int, optional
        Axis along which the FFT is taken.  Default ``-1``.
    n : int, optional
        FFT length.  ``None`` (default) uses ``X.shape[axis]``;
        a larger ``n`` zero-pads.

    Returns
    -------
    The complex analytic signal with the same shape as ``X``
    (or with the ``axis`` resized to ``n``).
    """
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
    """Hilbert transform of a real signal.

    Equivalent to the imaginary part of the analytic signal,
    ``analytic_signal(X).imag``.

    Parameters
    ----------
    X : Float[Array, '...']
        Real-valued signal.
    axis : int, optional
        Axis along which the transform is taken.  Default ``-1``.
    n : int, optional
        FFT length.  ``None`` (default) uses ``X.shape[axis]``;
        a larger ``n`` zero-pads.

    Returns
    -------
    Float[Array, '...']
        Hilbert transform of ``X``, same shape as ``X`` (or with
        the ``axis`` resized to ``n``).

    See Also
    --------
    analytic_signal
    """
    return analytic_signal(X, axis=axis, n=n).imag


def envelope(
    X: Float[Array, '...'],
    *,
    axis: int = -1,
    n: Optional[int] = None,
) -> Float[Array, '...']:
    """Magnitude of the analytic signal, :math:`|X_a|`.

    If you also need :func:`instantaneous_phase` or
    :func:`instantaneous_frequency`, use :func:`env_inst` to
    compute all three from a single analytic-signal call.

    Parameters
    ----------
    X : Float[Array, '...']
        Real-valued signal.
    axis : int, optional
        Axis along which the analytic signal is taken.  Default
        ``-1``.
    n : int, optional
        FFT length.  ``None`` (default) uses ``X.shape[axis]``;
        a larger ``n`` zero-pads.

    Returns
    -------
    Float[Array, '...']
        Envelope of ``X``, same shape as ``X`` (or with the
        ``axis`` resized to ``n``).
    """
    return jnp.abs(analytic_signal(X, axis=axis, n=n))


def instantaneous_phase(
    X: Float[Array, '...'],
    *,
    axis: int = -1,
    n: Optional[int] = None,
    period: float = 2 * math.pi,
) -> Float[Array, '...']:
    """Unwrapped instantaneous phase of the analytic signal.

    Computed as the unwrapped argument of the analytic signal,
    :math:`\\operatorname{unwrap}(\\arg(X_a))`.

    Parameters
    ----------
    X : Float[Array, '...']
        Real-valued signal.
    axis : int, optional
        Axis along which the phase is taken and unwrapped.
        Default ``-1``.
    n : int, optional
        FFT length.  ``None`` (default) uses ``X.shape[axis]``;
        a larger ``n`` zero-pads.
    period : float, optional
        Size of the range over which the phase wraps, passed to
        the unwrapping step.  Default :math:`2\\pi`.

    Returns
    -------
    Float[Array, '...']
        Unwrapped phase of ``X``, same shape as ``X`` (or with the
        ``axis`` resized to ``n``).
    """
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
    """Instantaneous frequency of the analytic signal.

    The discrete time derivative of the unwrapped instantaneous
    phase, scaled to physical units:
    :math:`f_s \\, \\operatorname{diff}(\\operatorname{unwrap}(
    \\arg(X_a))) / \\mathrm{period}`.  The output ``axis`` is one
    element shorter than the input (a first difference).

    Parameters
    ----------
    X : Float[Array, '...']
        Real-valued signal.
    axis : int, optional
        Axis along which the frequency is computed.  Default
        ``-1``.
    n : int, optional
        FFT length.  ``None`` (default) uses ``X.shape[axis]``;
        a larger ``n`` zero-pads.
    fs : float, optional
        Sampling frequency, used to scale the phase difference
        into physical frequency units.  Default ``1.0``.
    period : float, optional
        Size of the phase-wrapping range, both for unwrapping and
        for normalising the phase difference.  Default
        :math:`2\\pi`.

    Returns
    -------
    Float[Array, '...']
        Instantaneous frequency of ``X``.  Same shape as ``X``
        except the ``axis`` is one element shorter.
    """
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
    """Envelope, instantaneous frequency, and phase from one call.

    Computes all three quantities from a single
    :func:`analytic_signal` evaluation.  More efficient than three
    separate calls when you need all three, since they share the
    underlying analytic-signal computation.

    Parameters
    ----------
    X : Float[Array, '...']
        Real-valued signal.
    axis : int, optional
        Axis along which the quantities are computed.  Default
        ``-1``.
    n : int, optional
        FFT length.  ``None`` (default) uses ``X.shape[axis]``;
        a larger ``n`` zero-pads.
    fs : float, optional
        Sampling frequency, used to scale the instantaneous
        frequency into physical units.  Default ``1.0``.
    period : float, optional
        Size of the phase-wrapping range, both for unwrapping and
        for normalising the frequency.  Default :math:`2\\pi`.

    Returns
    -------
    envelope : Float[Array, '...']
        Magnitude of the analytic signal, same shape as ``X`` (or
        with the ``axis`` resized to ``n``).
    frequency : Float[Array, '...']
        Instantaneous frequency; same shape as ``envelope`` except
        the ``axis`` is one element shorter.
    phase : Float[Array, '...']
        Unwrapped instantaneous phase, same shape as ``envelope``.
    """
    Xa = analytic_signal(X, axis=axis, n=n)
    env = jnp.abs(Xa)
    phase = jnp.unwrap(jnp.angle(Xa), axis=axis, period=period)
    freq = fs * jnp.diff(phase, axis=axis) / period
    return env, freq, phase
