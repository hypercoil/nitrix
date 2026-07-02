# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Linear filtering: polynomial detrend and frequency-domain band-pass.

**Polynomial detrending** (:func:`polynomial_detrend`): fit a polynomial of
degree :math:`d` to a time series and subtract the fit.  Used for
low-frequency drift removal in fMRI BOLD signals.  Built on a rescaled
Vandermonde basis fed to :func:`residualise`.

**Frequency-domain filtering** (:func:`bandpass` / :func:`bandstop` /
:func:`lowpass` / :func:`highpass`): a zero-phase filter applied as a real
magnitude weight over the rfft frequency grid, via
:func:`product_filter` (the low-level FFT-multiply engine).
:func:`bandstop` is the notch case -- it *rejects* the ``(lo, hi)`` band and
passes outside it; the canonical use is removing a respiratory peak from
motion-estimate timeseries.  Three response designs:

- ``'maxflat'`` (default) -- the maximally-flat (Butterworth) *magnitude*
  shape :math:`1 / \\sqrt{1 + (f / f_c)^{2 n}}` (-3 dB at :math:`f_c`),
  applied as a zero-phase weight.  Because a real frequency-domain mask is a
  circular convolution with its inverse DFT, this is a frequency-sampled
  **FIR** filter that merely *borrows the Butterworth magnitude curve* -- it
  has no phase response and no poles.  It is **not** a recursive
  Butterworth; the genuine IIR Butterworth (with poles and real or
  ``filtfilt`` phase) is :func:`iir_filter` (separate primitive).
- ``'ideal'`` -- a brick-wall pass mask (sharpest, but rings in time).
- ``'cosine'`` -- raised-cosine transition bands (compact, tunable width).

All operate on a trailing observation/time axis (overridable via ``axis``),
parameterised by a sampling frequency ``fs`` (pass ``fs = 1 / TR``).  They
filter every channel independently with the same weight, so they are
per-unit invariant.  Differentiable through the signal and JIT-friendly
(the weight is data-independent; ``fs`` / cut-offs / ``order`` are static).

**Recursive IIR** filtering -- the genuine Butterworth (poles, bilinear
transform, second-order sections) -- is :func:`iir_filter` /
:func:`butterworth_sos` / :func:`sosfilt` / :func:`sosfiltfilt`, re-exported
here.  Unlike the ``ftype='maxflat'`` magnitude window above, these are real
recursive filters with the Butterworth phase (or zero-phase via
``filtfilt``), validated to machine precision against ``scipy.signal``.
"""

from __future__ import annotations

from typing import Callable, Optional

import jax.numpy as jnp
from jax.typing import DTypeLike
from jaxtyping import Array, Float, Num

from ..linalg.residual import residualise
from ._iir import butterworth_sos, iir_filter, sosfilt, sosfiltfilt
from .fourier import product_filter

__all__ = [
    'polynomial_detrend',
    'bandpass',
    'bandstop',
    'lowpass',
    'highpass',
    'iir_filter',
    'butterworth_sos',
    'sosfilt',
    'sosfiltfilt',
]


def _polynomial_basis(
    n_obs: int,
    degree: int,
    dtype: DTypeLike,
) -> Float[Array, 'degree+1 obs']:
    """Build a polynomial design basis along rescaled time.

    Rows are the integer powers :math:`[1, t, t^2, \\ldots, t^d]` evaluated
    on time rescaled to :math:`t \\in [-1, 1]`.  Rescaling is essential for
    numerical stability when ``degree`` is moderately large; integer powers
    of :math:`[0, n_{obs})` blow up fast and ill-condition the regression
    matrix.

    Parameters
    ----------
    n_obs
        Number of time samples (columns of the returned basis).  Fewer than
        two samples is degenerate and falls back to a matrix of ones.
    degree
        Highest polynomial power :math:`d`; the basis has ``degree + 1``
        rows (powers ``0`` through ``degree``).
    dtype
        Floating dtype of the returned basis.

    Returns
    -------
    Float[Array, 'degree+1 obs']
        Polynomial basis with one row per power and one column per time
        sample.
    """
    if n_obs < 2:
        # Edge case: too few samples for any polynomial.
        return jnp.ones((degree + 1, n_obs), dtype=dtype)
    t = jnp.linspace(-1.0, 1.0, n_obs, dtype=dtype)
    return jnp.stack([t**k for k in range(degree + 1)], axis=0)


def polynomial_detrend(
    X: Num[Array, '... obs'],
    *,
    degree: int = 1,
    rowvar: bool = True,
) -> Num[Array, '... obs']:
    """Subtract a polynomial fit of the named ``degree`` from each
    observation channel.

    Equivalent to :func:`residualise` against a rescaled Vandermonde basis
    with rows :math:`[1, t, t^2, \\ldots, t^{degree}]`, which regresses the
    polynomial trend out of every channel and returns the residual.

    Parameters
    ----------
    X
        Time series, observation axis is last (``rowvar=True``,
        default).
    degree
        Polynomial degree.  ``0`` -- demean only.  ``1`` --
        linear detrend.  ``2`` -- quadratic.  ``3`` -- cubic.
    rowvar
        ``True`` (default): observation axis is the *last* axis.

    Returns
    -------
    Detrended time series, same shape as ``X``.

    Notes
    -----
    Differentiable through ``X``; ``degree`` is static.
    """
    if degree < 0:
        raise ValueError(f'degree must be >= 0; got {degree}.')
    # Determine n_obs from X (last axis if rowvar, else penultimate).
    n_obs = X.shape[-1] if rowvar else X.shape[-2]
    basis = _polynomial_basis(n_obs, degree, X.dtype)
    # Broadcast basis to share leading dims with X.
    while basis.ndim < X.ndim:
        basis = basis[None, ...]
    if not rowvar:
        basis = basis.swapaxes(-1, -2)
    return residualise(X, basis, rowvar=rowvar, method='cholesky')


# ---------------------------------------------------------------------------
# Frequency-domain filtering
# ---------------------------------------------------------------------------

_FilterType = str  # 'maxflat' | 'ideal' | 'cosine'


def _cosine_edge(
    f: Array, cutoff: float, width: float, pass_below: bool
) -> Array:
    """Raised-cosine edge centred on ``cutoff`` with transition ``width``.

    The gain transitions smoothly across a band of width ``width`` centred
    on ``cutoff``, following a half-cosine taper, and is clamped to the
    plateau values outside that band.

    Parameters
    ----------
    f
        Frequency grid (cycles/sample) at which to evaluate the edge.
    cutoff
        Centre of the transition band, in cycles/sample.
    width
        Width of the raised-cosine transition band, in cycles/sample.
    pass_below
        If ``True``, unit gain below the band rolling to zero above (the
        low-pass edge); if ``False``, zero gain below rolling to unity above
        (the high-pass edge).

    Returns
    -------
    Array
        Magnitude weights in ``[0, 1]``, matching the shape of ``f``.
    """
    lo_e = cutoff - 0.5 * width
    hi_e = cutoff + 0.5 * width
    taper = 0.5 * (1.0 + jnp.cos(jnp.pi * (f - lo_e) / width))  # 1@lo_e 0@hi_e
    if pass_below:
        return jnp.where(f <= lo_e, 1.0, jnp.where(f >= hi_e, 0.0, taper))
    return jnp.where(f <= lo_e, 0.0, jnp.where(f >= hi_e, 1.0, 1.0 - taper))


def _lowpass_mag(
    f: Array, cutoff: float, ftype: _FilterType, order: int, width: float
) -> Array:
    """Magnitude response passing frequencies *below* ``cutoff``.

    Selects one of the three magnitude-window designs and evaluates it on
    the frequency grid.

    Parameters
    ----------
    f
        Frequency grid (cycles/sample) at which to evaluate the response.
    cutoff
        Pass-band edge, in cycles/sample.
    ftype
        Window design: ``'maxflat'`` (Butterworth magnitude shape),
        ``'ideal'`` (brick wall), or ``'cosine'`` (raised-cosine edge).
    order
        Roll-off exponent of the ``'maxflat'`` design; ignored otherwise.
    width
        Transition-band width for the ``'cosine'`` design, in cycles/sample;
        ignored otherwise.

    Returns
    -------
    Array
        Magnitude weights in ``[0, 1]``, matching the shape of ``f``.
    """
    if ftype == 'maxflat':
        # Butterworth *magnitude* shape (-3 dB at cutoff), applied as a
        # zero-phase weight -- a frequency-sampled FIR, not a recursive IIR.
        return 1.0 / jnp.sqrt(1.0 + (f / cutoff) ** (2 * order))
    if ftype == 'ideal':
        return (f <= cutoff).astype(f.dtype)
    if ftype == 'cosine':
        return _cosine_edge(f, cutoff, width, pass_below=True)
    raise ValueError(
        f"ftype={ftype!r}; expected 'maxflat', 'ideal', or 'cosine'."
    )


def _highpass_mag(
    f: Array, cutoff: float, ftype: _FilterType, order: int, width: float
) -> Array:
    """Magnitude response passing frequencies *above* ``cutoff``.

    Selects one of the three magnitude-window designs and evaluates it on
    the frequency grid.  The DC bin is always rejected.

    Parameters
    ----------
    f
        Frequency grid (cycles/sample) at which to evaluate the response.
    cutoff
        Pass-band edge, in cycles/sample.
    ftype
        Window design: ``'maxflat'`` (Butterworth magnitude shape),
        ``'ideal'`` (brick wall), or ``'cosine'`` (raised-cosine edge).
    order
        Roll-off exponent of the ``'maxflat'`` design; ignored otherwise.
    width
        Transition-band width for the ``'cosine'`` design, in cycles/sample;
        ignored otherwise.

    Returns
    -------
    Array
        Magnitude weights in ``[0, 1]``, matching the shape of ``f``.
    """
    if ftype == 'maxflat':
        # Guard the DC bin (f = 0) before dividing; it is always rejected.
        f_safe = jnp.where(f > 0, f, 1.0)
        resp = 1.0 / jnp.sqrt(1.0 + (cutoff / f_safe) ** (2 * order))
        return jnp.where(f > 0, resp, 0.0)
    if ftype == 'ideal':
        return (f >= cutoff).astype(f.dtype)
    if ftype == 'cosine':
        return _cosine_edge(f, cutoff, width, pass_below=False)
    raise ValueError(
        f"ftype={ftype!r}; expected 'maxflat', 'ideal', or 'cosine'."
    )


def _apply_frequency_filter(
    X: Num[Array, '... obs'],
    weight_fn: Callable[[Array], Array],
    *,
    axis: int,
    padding: int,
) -> Num[Array, '... obs']:
    """Move ``axis`` to the end, build the rfft weight, filter, restore.

    Moves the filtered axis to the trailing position, optionally reflect-pads
    it, builds the real magnitude weight over the rfft frequency grid,
    multiplies in the frequency domain via :func:`product_filter`, then crops
    the padding and restores the original axis order.

    Parameters
    ----------
    X
        Signal to filter; the axis given by ``axis`` is the observation/time
        axis.
    weight_fn
        Callable mapping the rfft frequency grid (cycles/sample) to a real
        magnitude weight of the same shape.
    axis
        Axis of ``X`` to filter.
    padding
        Number of samples to reflect-pad on each end of the filtered axis
        before filtering, then crop afterwards; reduces circular-convolution
        wrap-around at the edges.  Must be non-negative.

    Returns
    -------
    Num[Array, '... obs']
        Filtered signal, same shape as ``X``.
    """
    x = jnp.moveaxis(jnp.asarray(X), axis, -1)
    n = x.shape[-1]
    if padding < 0:
        raise ValueError(f'padding must be >= 0; got {padding}.')
    if padding:
        pad = [(0, 0)] * x.ndim
        pad[-1] = (padding, padding)
        x = jnp.pad(x, pad, mode='reflect')
    n_fft = x.shape[-1]
    freq = jnp.fft.rfftfreq(n_fft, d=1.0).astype(jnp.float32)
    # rfftfreq with d=1 gives cycles/sample; the caller's weight_fn closes
    # over the physical cut-offs already expressed in cycles/sample (= f/fs).
    weight = weight_fn(freq).astype(
        x.real.dtype if jnp.iscomplexobj(x) else x.dtype
    )
    y = product_filter(x, weight)
    if padding:
        y = y[..., padding : padding + n]
    return jnp.moveaxis(y, -1, axis)


def _normalise_cutoff(cutoff: float, fs: float, name: str) -> float:
    """Convert a physical cut-off (Hz) to normalised cycles/sample.

    Validates that the cut-off lies strictly between DC and the Nyquist
    frequency, then divides by the sampling frequency.

    Parameters
    ----------
    cutoff
        Physical cut-off frequency, in Hz.
    fs
        Sampling frequency, in Hz.
    name
        Argument name, used only to make the validation error message
        specific.

    Returns
    -------
    float
        The cut-off in cycles/sample (``cutoff / fs``).
    """
    nyq = 0.5 * fs
    if not 0.0 < cutoff < nyq:
        raise ValueError(
            f'{name}={cutoff} must lie in (0, fs/2) = (0, {nyq}); a cut-off '
            'at or beyond Nyquist is undefined.'
        )
    return cutoff / fs


def bandpass(
    X: Num[Array, '... obs'],
    *,
    fs: float = 1.0,
    lo: float,
    hi: float,
    ftype: _FilterType = 'maxflat',
    order: int = 2,
    transition: Optional[float] = None,
    axis: int = -1,
    padding: int = 0,
) -> Num[Array, '... obs']:
    """Zero-phase frequency-domain band-pass: keep the ``(lo, hi)`` band.

    Parameters
    ----------
    X
        Signal; the filtered axis is last by default (``axis``).
    fs
        Sampling frequency in Hz.  Pass ``fs = 1 / TR`` for fMRI.
    lo, hi
        Pass-band edges in Hz (``0 < lo < hi < fs / 2``).
    ftype
        Magnitude-window design: ``'maxflat'`` (default; the Butterworth
        magnitude *shape*, applied zero-phase -- a frequency-sampled FIR,
        not a recursive IIR Butterworth -- see :func:`iir_filter` for that),
        ``'ideal'`` (brick wall), or ``'cosine'`` (raised-cosine edges).
    order
        Steepness of the ``'maxflat'`` roll-off (the exponent of the
        Butterworth magnitude shape).  Ignored for the other designs.
        Default ``2``.
    transition
        Raised-cosine transition width in Hz (``'cosine'`` only); defaults
        to 10 % of Nyquist.
    axis
        Axis to filter.  Default ``-1`` (trailing time/observation axis).
    padding
        Reflect-pad this many samples on each end before filtering, then
        crop -- reduces the circular-convolution wrap-around at the edges.
        Default ``0`` (pure FFT).

    Returns
    -------
    Filtered signal, same shape as ``X``.  Differentiable through ``X``;
    ``fs`` / ``lo`` / ``hi`` / ``order`` are static.
    """
    lo_n = _normalise_cutoff(lo, fs, 'lo')
    hi_n = _normalise_cutoff(hi, fs, 'hi')
    if lo_n >= hi_n:
        raise ValueError(f'require lo < hi; got lo={lo}, hi={hi}.')
    tw = (transition / fs) if transition is not None else 0.05

    def weight_fn(f: Array) -> Array:
        return _highpass_mag(f, lo_n, ftype, order, tw) * _lowpass_mag(
            f, hi_n, ftype, order, tw
        )

    return _apply_frequency_filter(X, weight_fn, axis=axis, padding=padding)


def bandstop(
    X: Num[Array, '... obs'],
    *,
    fs: float = 1.0,
    lo: float,
    hi: float,
    ftype: _FilterType = 'maxflat',
    order: int = 2,
    transition: Optional[float] = None,
    axis: int = -1,
    padding: int = 0,
) -> Num[Array, '... obs']:
    """Zero-phase frequency-domain band-stop (notch): reject ``(lo, hi)``.

    The complement of :func:`bandpass`: passes frequencies below ``lo`` and
    above ``hi``, attenuating the band between.  Canonical use is notching
    a respiratory peak (~0.2-0.4 Hz) out of head-motion estimates before
    they are used as nuisance regressors.

    Parameters
    ----------
    X
        Signal; the filtered axis is last by default (``axis``).
    fs
        Sampling frequency in Hz.  Pass ``fs = 1 / TR`` for fMRI.
    lo, hi
        Edges of the *rejected* band in Hz (``0 < lo < hi < fs / 2``).
    ftype
        Magnitude-window design: ``'maxflat'`` (default; the Butterworth
        magnitude *shape*, applied zero-phase -- a frequency-sampled FIR,
        not a recursive IIR Butterworth -- see :func:`iir_filter` for that),
        ``'ideal'`` (brick wall), or ``'cosine'`` (raised-cosine edges).
    order
        Steepness of the ``'maxflat'`` roll-off (the exponent of the
        Butterworth magnitude shape).  Ignored for the other designs.
        Default ``2``.
    transition
        Raised-cosine transition width in Hz (``'cosine'`` only); defaults
        to 10 % of Nyquist.
    axis
        Axis to filter.  Default ``-1`` (trailing time/observation axis).
    padding
        Reflect-pad this many samples on each end before filtering, then
        crop -- reduces the circular-convolution wrap-around at the edges.
        Default ``0`` (pure FFT).

    Returns
    -------
    Num[Array, '... obs']
        Filtered signal, same shape as ``X``.  Differentiable through ``X``;
        ``fs`` / ``lo`` / ``hi`` / ``order`` are static.
    """
    lo_n = _normalise_cutoff(lo, fs, 'lo')
    hi_n = _normalise_cutoff(hi, fs, 'hi')
    if lo_n >= hi_n:
        raise ValueError(f'require lo < hi; got lo={lo}, hi={hi}.')
    tw = (transition / fs) if transition is not None else 0.05

    def weight_fn(f: Array) -> Array:
        # Soft union of "pass below lo" and "pass above hi".
        below = _lowpass_mag(f, lo_n, ftype, order, tw)
        above = _highpass_mag(f, hi_n, ftype, order, tw)
        return below + above - below * above

    return _apply_frequency_filter(X, weight_fn, axis=axis, padding=padding)


def lowpass(
    X: Num[Array, '... obs'],
    *,
    fs: float = 1.0,
    cutoff: float,
    ftype: _FilterType = 'maxflat',
    order: int = 2,
    transition: Optional[float] = None,
    axis: int = -1,
    padding: int = 0,
) -> Num[Array, '... obs']:
    """Zero-phase frequency-domain low-pass: keep frequencies below ``cutoff``.

    Passes frequencies below ``cutoff`` and attenuates those above, applied
    zero-phase as a real magnitude weight over the rfft grid.

    Parameters
    ----------
    X
        Signal; the filtered axis is last by default (``axis``).
    fs
        Sampling frequency in Hz.  Pass ``fs = 1 / TR`` for fMRI.
    cutoff
        Pass-band edge in Hz (``0 < cutoff < fs / 2``).
    ftype
        Magnitude-window design: ``'maxflat'`` (default; the Butterworth
        magnitude *shape*, applied zero-phase -- a frequency-sampled FIR,
        not a recursive IIR Butterworth -- see :func:`iir_filter` for that),
        ``'ideal'`` (brick wall), or ``'cosine'`` (raised-cosine edges).
    order
        Steepness of the ``'maxflat'`` roll-off (the exponent of the
        Butterworth magnitude shape).  Ignored for the other designs.
        Default ``2``.
    transition
        Raised-cosine transition width in Hz (``'cosine'`` only); defaults
        to 10 % of Nyquist.
    axis
        Axis to filter.  Default ``-1`` (trailing time/observation axis).
    padding
        Reflect-pad this many samples on each end before filtering, then
        crop -- reduces the circular-convolution wrap-around at the edges.
        Default ``0`` (pure FFT).

    Returns
    -------
    Num[Array, '... obs']
        Filtered signal, same shape as ``X``.  Differentiable through ``X``;
        ``fs`` / ``cutoff`` / ``order`` are static.
    """
    c_n = _normalise_cutoff(cutoff, fs, 'cutoff')
    tw = (transition / fs) if transition is not None else 0.05

    def weight_fn(f: Array) -> Array:
        return _lowpass_mag(f, c_n, ftype, order, tw)

    return _apply_frequency_filter(X, weight_fn, axis=axis, padding=padding)


def highpass(
    X: Num[Array, '... obs'],
    *,
    fs: float = 1.0,
    cutoff: float,
    ftype: _FilterType = 'maxflat',
    order: int = 2,
    transition: Optional[float] = None,
    axis: int = -1,
    padding: int = 0,
) -> Num[Array, '... obs']:
    """Zero-phase frequency-domain high-pass: keep frequencies above ``cutoff``.

    Passes frequencies above ``cutoff`` and attenuates those below (the DC
    bin is always rejected), applied zero-phase as a real magnitude weight
    over the rfft grid.

    Parameters
    ----------
    X
        Signal; the filtered axis is last by default (``axis``).
    fs
        Sampling frequency in Hz.  Pass ``fs = 1 / TR`` for fMRI.
    cutoff
        Pass-band edge in Hz (``0 < cutoff < fs / 2``).
    ftype
        Magnitude-window design: ``'maxflat'`` (default; the Butterworth
        magnitude *shape*, applied zero-phase -- a frequency-sampled FIR,
        not a recursive IIR Butterworth -- see :func:`iir_filter` for that),
        ``'ideal'`` (brick wall), or ``'cosine'`` (raised-cosine edges).
    order
        Steepness of the ``'maxflat'`` roll-off (the exponent of the
        Butterworth magnitude shape).  Ignored for the other designs.
        Default ``2``.
    transition
        Raised-cosine transition width in Hz (``'cosine'`` only); defaults
        to 10 % of Nyquist.
    axis
        Axis to filter.  Default ``-1`` (trailing time/observation axis).
    padding
        Reflect-pad this many samples on each end before filtering, then
        crop -- reduces the circular-convolution wrap-around at the edges.
        Default ``0`` (pure FFT).

    Returns
    -------
    Num[Array, '... obs']
        Filtered signal, same shape as ``X``.  Differentiable through ``X``;
        ``fs`` / ``cutoff`` / ``order`` are static.
    """
    c_n = _normalise_cutoff(cutoff, fs, 'cutoff')
    tw = (transition / fs) if transition is not None else 0.05

    def weight_fn(f: Array) -> Array:
        return _highpass_mag(f, c_n, ftype, order, tw)

    return _apply_frequency_filter(X, weight_fn, axis=axis, padding=padding)
