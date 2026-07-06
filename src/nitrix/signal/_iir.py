# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Recursive IIR (Butterworth) filtering: design and application.

Unlike the zero-phase frequency-domain filters in ``filter`` (which are
FIR magnitude windows), this is a *genuine recursive Butterworth*: poles on
the analog prototype circle, frequency-transformed and mapped to the
digital domain by the bilinear transform, realised as a cascade of
second-order sections (biquads).

The design is implemented from scratch in NumPy -- ``scipy`` is a test-only
dependency, so :func:`butterworth_sos` reproduces the mathematics of
``scipy.signal.butter(output='sos')`` (validated to machine precision in
the test suite) without importing it.  Cut-offs and order are static
(constant-folded into the trace); the design is not differentiated.

Application is the recurrence ``y[n] = b0 x[n] + b1 x[n-1] + b2 x[n-2]
- a1 y[n-1] - a2 y[n-2]`` per biquad.  The ``driver`` axis selects the
engine; ``driver='auto'`` (the default) resolves by platform to the FFT
engine on GPU and the ``scan`` recurrence on CPU:

- ``driver='fft'`` -- truncate the biquad impulse response and convolve via
  the FFT; the GPU-default path and the headline speed-up on long series.
- ``driver='scan'`` (the canonical variant) -- sequential ``lax.scan`` over
  time, vectorised across all channels; compiles to a single fused loop, low
  memory; the CPU default.  Best for the fMRI regime (modest ``T``, many
  voxels).
- ``driver='associative'`` -- the biquad's linear recurrence composed via
  ``lax.associative_scan`` (the parallel-prefix pattern also used by
  ``signal.linear_interpolate``); ``O(log T)`` depth, for latency-bound
  long single series.  Forward filtering only.

:func:`sosfiltfilt` is the zero-phase forward-backward filter, with
scipy-exact steady-state initial conditions (``lfilter_zi``-equivalent) and
odd padding; it matches ``scipy.signal.sosfiltfilt`` to machine precision.

Everything is reverse-mode differentiable through the signal.
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple

import jax.lax as lax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Num

from .._internal.backend import default_backend_is_gpu
from .._internal.config import resolve_driver

# Hard cap on the impulse-response length the FFT engine will truncate to.
# A filter that has not decayed below ``impulse_atol`` within this many taps
# is too sharp (poles hugging the unit circle) for a cheap FFT convolution;
# the engine falls back to the recurrence instead.
_IIR_FFT_MAX_TAPS = 1 << 15

__all__ = ['butterworth_sos', 'sosfilt', 'sosfiltfilt', 'iir_filter']

_BType = str  # 'lowpass' | 'highpass' | 'bandpass' | 'bandstop'
_Driver = str  # 'auto' | 'fft' | 'scan' | 'associative'

# The 'signal.iir' divergent-op contract is registered centrally in
# nitrix._internal._divergent_ops (so divergent_ops() is complete at import);
# this module only resolves against it.


def _resolve_iir_driver(driver: _Driver) -> _Driver:
    """Resolve the IIR ``driver`` to a concrete engine.

    ``'auto'`` resolves to the platform-fast engine (``'fft'`` on GPU -- the
    recursion is latency-bound there and the parallel FFT convolution beats
    cupy on the L4; sequential ``'scan'`` on CPU).  Reproducibility mode forces
    the canonical ``'scan'``.  Explicit ``'fft'`` / ``'scan'`` /
    ``'associative'`` pass through (validated against the registered set).

    Parameters
    ----------
    driver
        Requested driver: ``'auto'``, ``'fft'``, ``'scan'`` or
        ``'associative'``.

    Returns
    -------
    str
        The concrete engine name (``'fft'``, ``'scan'`` or ``'associative'``)
        that the requested driver resolves to on the current platform.
    """
    return resolve_driver(
        driver,
        op='signal.iir',
        fast=lambda: 'fft' if default_backend_is_gpu() else 'scan',
    )


# ---------------------------------------------------------------------------
# Butterworth design (pure NumPy; static cut-offs).  Mirrors the zpk maths of
# scipy.signal.butter + lp2{lp,hp,bp,bs} + bilinear + zpk2sos.
# ---------------------------------------------------------------------------


def _buttap(order: int) -> Tuple[np.ndarray, np.ndarray, float]:
    """Analog Butterworth low-pass prototype (zeros, poles, gain).

    Parameters
    ----------
    order
        Butterworth filter order (number of prototype poles).

    Returns
    -------
    zeros : numpy.ndarray
        Empty complex array -- the analog prototype has no finite zeros.
    poles : numpy.ndarray
        The ``order`` complex prototype poles, evenly spaced on the left half
        of the unit circle.
    gain : float
        The prototype gain, always ``1.0``.
    """
    m = np.arange(-order + 1, order, 2)
    poles = -np.exp(1j * np.pi * m / (2 * order))
    return np.array([], dtype=complex), poles, 1.0


def _lp2lp(
    z: np.ndarray, p: np.ndarray, k: float, wo: float
) -> Tuple[np.ndarray, np.ndarray, float]:
    deg = len(p) - len(z)
    return wo * z, wo * p, k * wo**deg


def _lp2hp(
    z: np.ndarray, p: np.ndarray, k: float, wo: float
) -> Tuple[np.ndarray, np.ndarray, float]:
    deg = len(p) - len(z)
    z_hp = wo / z if len(z) else np.array([], complex)
    p_hp = wo / p
    z_hp = np.append(z_hp, np.zeros(deg))
    prod_z = np.prod(-z) if len(z) else 1.0
    k_hp = float(k * np.real(prod_z / np.prod(-p)))
    return z_hp, p_hp, k_hp


def _lp2bp(
    z: np.ndarray, p: np.ndarray, k: float, wo: float, bw: float
) -> Tuple[np.ndarray, np.ndarray, float]:
    deg = len(p) - len(z)
    z_lp, p_lp = z * bw / 2, p * bw / 2
    z_bp = np.concatenate(
        [z_lp + np.sqrt(z_lp**2 - wo**2), z_lp - np.sqrt(z_lp**2 - wo**2)]
    )
    p_bp = np.concatenate(
        [p_lp + np.sqrt(p_lp**2 - wo**2), p_lp - np.sqrt(p_lp**2 - wo**2)]
    )
    z_bp = np.append(z_bp, np.zeros(deg))
    return z_bp, p_bp, k * bw**deg


def _lp2bs(
    z: np.ndarray, p: np.ndarray, k: float, wo: float, bw: float
) -> Tuple[np.ndarray, np.ndarray, float]:
    deg = len(p) - len(z)
    z_hp = (bw / 2) / z if len(z) else np.array([], complex)
    p_hp = (bw / 2) / p
    z_bs = np.concatenate(
        [z_hp + np.sqrt(z_hp**2 - wo**2), z_hp - np.sqrt(z_hp**2 - wo**2)]
    )
    p_bs = np.concatenate(
        [p_hp + np.sqrt(p_hp**2 - wo**2), p_hp - np.sqrt(p_hp**2 - wo**2)]
    )
    z_bs = np.append(
        z_bs,
        np.concatenate([np.full(deg, 1j * wo), np.full(deg, -1j * wo)]),
    )
    prod_z = np.prod(-z) if len(z) else 1.0
    k_bs = float(k * np.real(prod_z / np.prod(-p)))
    return z_bs, p_bs, k_bs


def _bilinear(
    z: np.ndarray, p: np.ndarray, k: float, fs: float
) -> Tuple[np.ndarray, np.ndarray, float]:
    deg = len(p) - len(z)
    fs2 = 2.0 * fs
    z_z = (fs2 + z) / (fs2 - z)
    p_z = (fs2 + p) / (fs2 - p)
    z_z = np.append(z_z, -np.ones(deg))
    prod_z = np.prod(fs2 - z) if len(z) else 1.0
    k_z = float(k * np.real(prod_z / np.prod(fs2 - p)))
    return z_z, p_z, k_z


def _group_conjugates(
    roots: np.ndarray, tol: float = 1e-6
) -> list[np.ndarray]:
    """Group roots into conjugate pairs and real pairs (each at most two roots).

    Complex roots are matched with their conjugates; remaining real roots are
    paired two at a time.  Each group holds at most two roots, ready to form a
    real second-order section.

    Parameters
    ----------
    roots
        The complex roots (poles or zeros) to group.
    tol
        Absolute tolerance for deciding whether a root is real (small
        imaginary part) and for matching a root to its conjugate.

    Returns
    -------
    list of numpy.ndarray
        Groups of roots, each of length at most two: first the
        conjugate pairs, then the real pairs.
    """
    roots = list(roots)
    used = [False] * len(roots)
    cpairs, reals = [], []
    for i, r in enumerate(roots):
        if used[i]:
            continue
        used[i] = True
        if abs(r.imag) < tol:
            reals.append(complex(r.real))
        else:
            for j in range(len(roots)):
                if not used[j] and abs(roots[j] - np.conj(r)) < tol:
                    used[j] = True
                    break
            cpairs.append(np.array([r, np.conj(r)]))
    real_pairs = [np.array(reals[i : i + 2]) for i in range(0, len(reals), 2)]
    return cpairs + real_pairs


def _section_poly(group: np.ndarray) -> np.ndarray:
    """Real length-3 polynomial coefficients from a root group of size at most 2.

    Expands the group into a monic polynomial, takes the real part (conjugate
    pairs and real pairs give real coefficients) and right-pads with zeros to a
    fixed length of three.

    Parameters
    ----------
    group
        A root group of length at most two (from :func:`_group_conjugates`).

    Returns
    -------
    numpy.ndarray
        The three real polynomial coefficients ``[c0, c1, c2]`` (highest degree
        first), zero-padded on the right when the group has fewer than two
        roots.
    """
    coeffs = np.real(np.poly(group))
    return np.concatenate([coeffs, np.zeros(3 - len(coeffs))])


def _zpk2sos(z: np.ndarray, p: np.ndarray, k: float) -> np.ndarray:
    """Group conjugate poles/zeros into biquad second-order sections.

    Any valid conjugate grouping yields the same overall transfer function;
    for the modest orders here this simple pairing is well-conditioned (and
    parity is checked against scipy via the *frequency response*, which is
    grouping-invariant).

    Parameters
    ----------
    z
        Complex zeros of the digital filter.
    p
        Complex poles of the digital filter.
    k
        Overall filter gain, applied to the numerator of the first section.

    Returns
    -------
    numpy.ndarray
        The ``(n_sections, 6)`` second-order-section array, each row laid out
        as ``[b0, b1, b2, a0, a1, a2]``.
    """
    pg = _group_conjugates(p)
    zg = _group_conjugates(z)
    n = max(len(pg), len(zg))
    pg = pg + [np.array([])] * (n - len(pg))
    zg = zg + [np.array([])] * (n - len(zg))
    sos = np.zeros((n, 6))
    for i in range(n):
        sos[i, :3] = _section_poly(zg[i])
        sos[i, 3:] = _section_poly(pg[i])
    sos[0, :3] *= k
    return sos


def _normalise_band(
    btype: _BType, fs: float, lo: Optional[float], hi: Optional[float]
) -> np.ndarray:
    """Validate band edges and return them normalised to (0, 1) over Nyquist.

    Checks that the required cut-offs are supplied for the requested band type
    and that they lie strictly within ``(0, fs/2)`` (with ``lo < hi`` for band
    types), then divides each by the Nyquist frequency.

    Parameters
    ----------
    btype
        Band type: ``'lowpass'``, ``'highpass'``, ``'bandpass'`` or
        ``'bandstop'``.
    fs
        Sampling frequency in Hz; the Nyquist frequency is ``fs / 2``.
    lo, hi
        Cut-offs in Hz.  ``'lowpass'`` uses ``hi``; ``'highpass'`` uses ``lo``;
        band types use both.  Unused edges may be ``None``.

    Returns
    -------
    numpy.ndarray
        The Nyquist-normalised cut-off(s) in ``(0, 1)``: a length-one array for
        ``'lowpass'`` / ``'highpass'`` and a length-two ``[lo, hi]`` array for
        the band types.
    """
    nyq = 0.5 * fs

    def norm(c: float, name: str) -> float:
        if not 0.0 < c < nyq:
            raise ValueError(f'{name}={c} must lie in (0, fs/2) = (0, {nyq}).')
        return c / nyq

    if btype in ('bandpass', 'bandstop'):
        if lo is None or hi is None:
            raise ValueError(f"btype={btype!r} needs both 'lo' and 'hi'.")
        if lo >= hi:
            raise ValueError(f'require lo < hi; got lo={lo}, hi={hi}.')
        return np.array([norm(lo, 'lo'), norm(hi, 'hi')])
    if btype == 'lowpass':
        if hi is None:
            raise ValueError("btype='lowpass' needs 'hi' (the cut-off).")
        return np.array([norm(hi, 'hi')])
    if btype == 'highpass':
        if lo is None:
            raise ValueError("btype='highpass' needs 'lo' (the cut-off).")
        return np.array([norm(lo, 'lo')])
    raise ValueError(
        f"btype={btype!r}; expected 'lowpass', 'highpass', 'bandpass', "
        "or 'bandstop'."
    )


def butterworth_sos(
    *,
    order: int,
    fs: float = 1.0,
    btype: _BType,
    lo: Optional[float] = None,
    hi: Optional[float] = None,
) -> np.ndarray:
    """Design a digital Butterworth filter as second-order sections.

    Returns the ``(n_sections, 6)`` SOS array with ``[b0, b1, b2, a0, a1, a2]``
    per row -- the same layout as ``scipy.signal.butter(output='sos')`` (and
    matching its transfer function to machine precision), implemented in
    pure NumPy.  ``order``, ``fs`` and the cut-offs are static, so the result
    is a host (NumPy) constant -- the *design* is not traced; only the
    *application* (:func:`sosfilt` / :func:`sosfiltfilt`) runs in JAX.

    Parameters
    ----------
    order
        Butterworth order.  ``'bandpass'`` / ``'bandstop'`` produce
        ``order`` sections (``2 * order`` poles); ``'lowpass'`` /
        ``'highpass'`` produce ``ceil(order / 2)``.
    fs
        Sampling frequency in Hz (``1 / TR``).
    btype
        ``'lowpass'`` / ``'highpass'`` / ``'bandpass'`` / ``'bandstop'``.
    lo, hi
        Cut-offs in Hz (``0 < f < fs/2``).  ``'lowpass'`` uses ``hi``;
        ``'highpass'`` uses ``lo``; band types use both (``lo < hi``).

    Returns
    -------
    numpy.ndarray
        The ``(n_sections, 6)`` second-order-section coefficient array, each
        row laid out as ``[b0, b1, b2, a0, a1, a2]``, as a host NumPy constant.
    """
    wn = _normalise_band(btype, fs, lo, hi)
    # Bilinear pre-warp at the scipy fs = 2 convention.
    fs_d = 2.0
    warped = 2 * fs_d * np.tan(np.pi * wn / fs_d)
    z, p, k = _buttap(order)
    if btype == 'lowpass':
        z, p, k = _lp2lp(z, p, k, warped[0])
    elif btype == 'highpass':
        z, p, k = _lp2hp(z, p, k, warped[0])
    else:
        bw = warped[1] - warped[0]
        wo = np.sqrt(warped[0] * warped[1])
        transform = _lp2bp if btype == 'bandpass' else _lp2bs
        z, p, k = transform(z, p, k, wo, bw)
    z, p, k = _bilinear(z, p, k, fs_d)
    return _zpk2sos(z, p, k)


# ---------------------------------------------------------------------------
# Steady-state initial conditions (scipy lfilter_zi-equivalent, per section)
# ---------------------------------------------------------------------------


def _sos_zi(sos: np.ndarray) -> np.ndarray:
    """Per-section steady-state delay state, scaled along the cascade.

    For each biquad, the transposed-DF2 state that makes a step input
    produce no startup transient: :math:`z_i = (I - A^{\\top})^{-1} B` with
    :math:`A` the companion matrix.  Scaled by the running DC gain so the
    cascade is consistent (scipy ``sosfilt_zi``).

    Parameters
    ----------
    sos
        The ``(n_sections, 6)`` second-order-section coefficient array.

    Returns
    -------
    numpy.ndarray
        The ``(n_sections, 2)`` per-section steady-state delay state for a
        unit-amplitude step input, scaled by the running DC gain along the
        cascade.
    """
    zi = np.empty((sos.shape[0], 2))
    scale = 1.0
    for i, s in enumerate(sos):
        b = s[:3] / s[3]
        a = s[3:] / s[3]
        i_minus_at = np.array([[1 + a[1], -1.0], [a[2], 1.0]])
        rhs = np.array([b[1] - a[1] * b[0], b[2] - a[2] * b[0]])
        zi[i] = scale * np.linalg.solve(i_minus_at, rhs)
        scale *= b.sum() / a.sum()
    return zi


# ---------------------------------------------------------------------------
# Impulse / zero-input responses (host) -- the FFT-convolution engine
# ---------------------------------------------------------------------------


def _sos_responses(
    sos: np.ndarray,
    n: int,
    zi: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Host (NumPy, fp64) impulse and zero-input responses of the cascade.

    Runs the transposed-DF2 cascade recurrence -- the same one
    :func:`_sosfilt_scan` realises -- for ``n`` samples:

    - ``h`` -- the cascade **impulse response** (unit impulse, zero state).  A
      fixed-coefficient IIR filter is LTI, so its zero-state output is exactly
      ``conv(x, h)``; ``h`` decays geometrically (poles inside the unit
      circle), so an ``n``-tap truncation has a geometrically bounded error.
    - ``g`` -- the cascade **zero-input response** to the per-section initial
      state ``zi`` (zero input).  That response is linear in the ``x[0]``
      scaling applied to ``zi``, so :func:`sosfilt` with ``zi`` is
      ``conv(x, h) + x[0] * g``.

    Both are host constants (``sos`` / ``zi`` are static) and fold into the
    trace; the device side only runs the FFT convolution.

    Parameters
    ----------
    sos
        The ``(n_sections, 6)`` second-order-section coefficient array.
    n
        Number of samples to run the recurrence for (the response length).
    zi
        Optional ``(n_sections, 2)`` per-section initial delay state driving
        the zero-input response ``g``.  When ``None`` the zero-input response
        is identically zero.

    Returns
    -------
    h : numpy.ndarray
        Length-``n`` fp64 impulse response of the cascade (unit impulse, zero
        initial state).
    g : numpy.ndarray
        Length-``n`` fp64 zero-input response of the cascade to the initial
        state ``zi`` (zero input).
    """
    b0, b1, b2 = (
        sos[:, 0] / sos[:, 3],
        sos[:, 1] / sos[:, 3],
        sos[:, 2] / sos[:, 3],
    )
    a1, a2 = sos[:, 4] / sos[:, 3], sos[:, 5] / sos[:, 3]
    ns = sos.shape[0]
    h = np.empty(n, dtype=np.float64)
    g = np.empty(n, dtype=np.float64)
    sh = np.zeros((ns, 2))
    sg = (
        np.zeros((ns, 2))
        if zi is None
        else np.asarray(zi, dtype=np.float64).copy()
    )
    for t in range(n):
        yh = 1.0 if t == 0 else 0.0
        yg = 0.0
        for i in range(ns):
            oh = b0[i] * yh + sh[i, 0]
            sh[i, 0] = b1[i] * yh - a1[i] * oh + sh[i, 1]
            sh[i, 1] = b2[i] * yh - a2[i] * oh
            yh = oh
            og = b0[i] * yg + sg[i, 0]
            sg[i, 0] = b1[i] * yg - a1[i] * og + sg[i, 1]
            sg[i, 1] = b2[i] * yg - a2[i] * og
            yg = og
        h[t], g[t] = yh, yg
    return h, g


def _sos_impulse_taps(
    sos: np.ndarray,
    atol: float,
    cap: int = _IIR_FFT_MAX_TAPS,
) -> Optional[int]:
    """Find the smallest tap count after which the impulse response has decayed.

    Doubles a trial length (starting near 512 samples) until the impulse
    response ``h`` stays below ``atol`` beyond some tap, returning that tap
    count.  Returns ``None`` if the response has not decayed within ``cap``
    taps -- a filter too sharp (poles hugging the unit circle) for a cheap FFT
    convolution.

    Parameters
    ----------
    sos
        The ``(n_sections, 6)`` second-order-section coefficient array.
    atol
        Absolute tolerance; the impulse response is deemed to have decayed once
        every later tap satisfies ``|h| <= atol``.
    cap
        Hard upper bound on the tap count to search (default
        ``_IIR_FFT_MAX_TAPS``).

    Returns
    -------
    int or None
        The smallest number of taps after which ``|h| <= atol``, or ``None`` if
        the response has not decayed within ``cap`` taps.
    """
    n = min(512, cap)
    while True:
        h, _ = _sos_responses(sos, n)
        nz = np.nonzero(np.abs(h) > atol)[0]
        length = int(nz[-1]) + 1 if len(nz) else 1
        if length < n:
            return length
        if n >= cap:
            return None
        n = min(n * 2, cap)


def _expand_freq(arr: Array, ndim: int) -> Array:
    """Reshape a length-F frequency vector to broadcast over ``(F, *channels)``.

    Parameters
    ----------
    arr
        A one-dimensional length-F frequency-domain vector.
    ndim
        Target number of dimensions (the rank of the array it must broadcast
        against).

    Returns
    -------
    Array
        ``arr`` reshaped to ``(F, 1, ..., 1)`` with ``ndim`` axes, so it
        broadcasts along the leading frequency axis over trailing channel axes.
    """
    return arr.reshape((-1,) + (1,) * (ndim - 1))


def _sosfilt_fft(
    sos: np.ndarray,
    x: Array,
    zi: Optional[np.ndarray],
    n_taps: int,
) -> Array:
    """Apply the cascade as an FFT convolution with its impulse response.

    ``x`` is time-major ``(T, *channels)``.  The zero-state output is
    ``conv(x, h)`` (linear convolution via the FFT, truncated to ``n_taps``);
    with ``zi`` the steady-state transient ``x[0] * g`` is added back over the
    first ``n_taps`` samples so the edges stay scipy-exact.  ``O(T log T)``,
    no recurrence, fully parallel.

    Parameters
    ----------
    sos
        The ``(n_sections, 6)`` second-order-section coefficient array.
    x
        Time-major signal ``(T, *channels)`` to filter along the leading axis.
    zi
        Optional ``(n_sections, 2)`` per-section steady-state delay state; when
        given, its transient ``x[0] * g`` is added over the first ``n_taps``
        samples.  ``None`` for a zero-state (pure convolution) filter.
    n_taps
        Truncation length of the impulse response used as the convolution
        kernel.

    Returns
    -------
    Array
        The filtered signal, time-major ``(T, *channels)`` with the same dtype
        as ``x``.
    """
    h, g = _sos_responses(sos, n_taps, zi=zi)
    n_t = x.shape[0]
    nfft = 1
    while nfft < n_t + n_taps - 1:
        nfft *= 2
    h_freq = jnp.fft.rfft(jnp.asarray(h, x.dtype), n=nfft)
    x_freq = jnp.fft.rfft(x, n=nfft, axis=0)
    y = jnp.fft.irfft(x_freq * _expand_freq(h_freq, x.ndim), n=nfft, axis=0)
    y = y[:n_t]
    if zi is not None:
        # The zero-input transient spans ``n_taps`` samples, but a short series
        # (or, in ``sosfiltfilt``, a sharp filter whose impulse outlasts the
        # padded segment) may be shorter; there are no output samples past
        # ``n_t`` to carry it, so add it over the first ``min(n_taps, n_t)``.
        m = min(n_taps, n_t)
        g_seq = jnp.asarray(g[:m], x.dtype).reshape((m,) + (1,) * (x.ndim - 1))
        y = y.at[:m].add(g_seq * x[:1])
    return y


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------


def _sosfilt_scan(
    sos: np.ndarray,
    x: Array,
    zi: Optional[np.ndarray],
) -> Array:
    """Cascade of biquads via sequential ``lax.scan`` over time (axis 0).

    ``x`` is time-major ``(T, *channels)``.  ``zi`` (per-section delay
    state for unit input) is scaled by ``x[0]`` when provided.

    Parameters
    ----------
    sos
        The ``(n_sections, 6)`` second-order-section coefficient array.
    x
        Time-major signal ``(T, *channels)`` filtered along the leading axis.
    zi
        Optional ``(n_sections, 2)`` per-section delay state for a unit input,
        scaled by ``x[0]``.  ``None`` starts each section from zero state.

    Returns
    -------
    Array
        The filtered signal, time-major ``(T, *channels)``.
    """
    y = x
    x0 = x[0]
    for i, s in enumerate(sos):
        b0, b1, b2 = (float(v) / float(s[3]) for v in s[:3])
        a1, a2 = float(s[4]) / float(s[3]), float(s[5]) / float(s[3])
        if zi is None:
            init = (jnp.zeros(y.shape[1:]), jnp.zeros(y.shape[1:]))
        else:
            init = (float(zi[i, 0]) * x0, float(zi[i, 1]) * x0)

        def step(
            state: Tuple[Array, Array],
            xn: Array,
            b0: float = b0,
            b1: float = b1,
            b2: float = b2,
            a1: float = a1,
            a2: float = a2,
        ) -> Tuple[Tuple[Array, Array], Array]:
            z1, z2 = state
            yn = b0 * xn + z1
            return (b1 * xn - a1 * yn + z2, b2 * xn - a2 * yn), yn

        _, y = lax.scan(step, init, y)
    return y


def _sosfilt_associative(
    sos: np.ndarray,
    x: Array,
    zi: Optional[np.ndarray] = None,
) -> Array:
    """Cascade of biquads via parallel ``lax.associative_scan``.

    The transposed-DF2 state ``s = (z1, z2)`` -- the *same* state
    :func:`_sosfilt_scan` carries -- evolves as a first-order linear recurrence
    ``s[n+1] = M s[n] + c u[n]`` with constant ``M = [[-a1, 1], [-a2, 0]]`` and
    ``c = [b1 - a1 b0, b2 - a2 b0]``, and the output is
    ``y[n] = s[n][0] + b0 u[n]``.  A constant-coefficient linear recurrence
    composes associatively, so the prefix scan resolves the whole cascade in
    ``O(log T)`` depth.

    ``zi`` (per-section steady-state delay, scaled by ``x[0]``) sets the
    initial state ``s[0]``; its homogeneous response :math:`M^n s_0` is read
    off the same scan's cumulative transition ``a_cum``.  With ``zi=None`` the
    initial state is zero (the homogeneous term vanishes and ``a_cum`` is not
    materialised).  This is the engine that makes :func:`sosfilt` /
    :func:`sosfiltfilt` competitive on the GPU; it matches :func:`_sosfilt_scan`
    (and ``scipy.signal``) to round-off.

    Parameters
    ----------
    sos
        The ``(n_sections, 6)`` second-order-section coefficient array.
    x
        Time-major signal ``(T, *channels)`` filtered along the leading axis.
    zi
        Optional ``(n_sections, 2)`` per-section steady-state delay state,
        scaled by ``x[0]`` to set the initial state.  ``None`` starts each
        section from zero state.

    Returns
    -------
    Array
        The filtered signal, time-major ``(T, *channels)``.
    """
    y = x
    n_t = x.shape[0]
    x0 = x[0]
    for i, s in enumerate(sos):
        b0, b1, b2 = (float(v) / float(s[3]) for v in s[:3])
        a1, a2 = float(s[4]) / float(s[3]), float(s[5]) / float(s[3])
        u = y
        m_mat = jnp.array([[-a1, 1.0], [-a2, 0.0]], dtype=x.dtype)
        c_vec = jnp.array([b1 - a1 * b0, b2 - a2 * b0], dtype=x.dtype)
        a_seq = jnp.broadcast_to(m_mat, (n_t,) + u.shape[1:] + (2, 2))
        b_seq = c_vec * u[..., None]

        def combine(
            left: Tuple[Array, Array], right: Tuple[Array, Array]
        ) -> Tuple[Array, Array]:
            a_l, b_l = left
            a_r, b_r = right
            a = jnp.einsum('...ij,...jk->...ik', a_r, a_l)
            b = jnp.einsum('...ij,...j->...i', a_r, b_l) + b_r
            return a, b

        if zi is None:
            _, b_cum = lax.associative_scan(combine, (a_seq, b_seq), axis=0)
            s0_row = jnp.zeros((1,) + u.shape[1:] + (2,), x.dtype)
            s = jnp.concatenate([s0_row, b_cum[:-1]], axis=0)
        else:
            a_cum, b_cum = lax.associative_scan(
                combine,
                (a_seq, b_seq),
                axis=0,
            )
            s0 = jnp.stack(
                [float(zi[i, 0]) * x0, float(zi[i, 1]) * x0],
                axis=-1,
            )
            homog = jnp.einsum('n...ij,...j->n...i', a_cum[:-1], s0)
            s = jnp.concatenate([s0[None], homog + b_cum[:-1]], axis=0)
        y = s[..., 0] + b0 * u
    return y


def _sosfilt_apply(
    sos: np.ndarray,
    x: Array,
    *,
    zi: Optional[np.ndarray],
    driver: _Driver,
    impulse_atol: float,
) -> Array:
    """Apply the cascade with the resolved engine (the ``driver`` axis).

    ``'fft'`` truncates the impulse response at ``impulse_atol`` and convolves;
    if the filter has not decayed within :data:`_IIR_FFT_MAX_TAPS` it falls
    back to a recurrence (``'associative'`` on GPU, ``'scan'`` on CPU) with a
    warning.

    Parameters
    ----------
    sos
        The ``(n_sections, 6)`` second-order-section coefficient array.
    x
        Time-major signal ``(T, *channels)`` filtered along the leading axis.
    zi
        Optional ``(n_sections, 2)`` per-section steady-state delay state, or
        ``None`` for a zero-state filter.
    driver
        Requested engine (``'auto'``, ``'fft'``, ``'scan'`` or
        ``'associative'``), resolved to a concrete engine per platform.
    impulse_atol
        FFT engine only: absolute tolerance for truncating the impulse
        response.

    Returns
    -------
    Array
        The filtered signal, time-major ``(T, *channels)``.
    """
    driver = _resolve_iir_driver(driver)
    if driver == 'fft':
        n_taps = _sos_impulse_taps(sos, impulse_atol)
        if n_taps is not None:
            return _sosfilt_fft(sos, x, zi, n_taps)
        driver = 'associative' if default_backend_is_gpu() else 'scan'
        warnings.warn(
            f'sosfilt: impulse response has not decayed below '
            f'impulse_atol={impulse_atol:g} within {_IIR_FFT_MAX_TAPS} taps '
            f'(filter too sharp for the FFT engine); falling back to '
            f'driver={driver!r}.',
            stacklevel=3,
        )
    if driver == 'scan':
        return _sosfilt_scan(sos, x, zi=zi)
    if driver == 'associative':
        return _sosfilt_associative(sos, x, zi=zi)
    raise ValueError(
        f"driver={driver!r}; expected 'auto', 'fft', 'scan' or 'associative'."
    )


def sosfilt(
    X: Num[Array, '... obs'],
    sos: np.ndarray,
    *,
    axis: int = -1,
    driver: _Driver = 'auto',
    impulse_atol: float = 1e-12,
) -> Num[Array, '... obs']:
    """Apply a causal IIR filter (forward only) given its SOS cascade.

    Parameters
    ----------
    X
        Signal; filtered along ``axis`` (default trailing).
    sos
        ``(n_sections, 6)`` second-order sections (e.g. from
        :func:`butterworth_sos`).  Static host coefficients (NumPy); they are
        baked into the engine, not traced.
    axis
        Axis of ``X`` along which to filter (default ``-1``, the trailing
        axis).
    driver
        Numerical variant (the ``driver`` axis).  ``'auto'`` (default) picks the
        engine by platform: ``'fft'`` on GPU, ``'scan'`` on CPU.  The recursion
        is latency-bound on the GPU, so the parallel FFT-convolution engine (an
        IIR filter is LTI, so its output is exactly convolution with the impulse
        response) wins there -- it beats cupy on the L4; on the CPU the
        sequential ``lax.scan`` is fast and exact, so it is kept.  Force
        ``'fft'``, ``'scan'`` or the parallel-prefix ``'associative'`` to
        override; ``nitrix.reproducible()`` forces the canonical ``'scan'``.
        These variants reassociate the arithmetic differently, so they agree
        only to a documented tolerance (``nitrix.divergent_ops()``), not bit-
        for-bit -- hence the per-call / reproducibility override.
    impulse_atol
        FFT engine only: truncate the impulse response where it has decayed
        below this absolute tolerance (default ``1e-12``, effectively exact).
        Larger values give shorter kernels / smaller FFTs at the cost of a
        (geometrically bounded) edge error; a filter too sharp to decay within
        ``2**15`` taps falls back to a recurrence with a warning.

    Returns
    -------
    Num[Array, '... obs']
        The causally filtered signal, same shape and dtype as ``X``.
        Differentiable through ``X``.
    """
    sos_np = np.asarray(sos)
    x = jnp.moveaxis(jnp.asarray(X), axis, 0)
    y = _sosfilt_apply(
        sos_np,
        x,
        zi=None,
        driver=driver,
        impulse_atol=impulse_atol,
    )
    return jnp.moveaxis(y, 0, axis)


def sosfiltfilt(
    X: Num[Array, '... obs'],
    sos: np.ndarray,
    *,
    axis: int = -1,
    padtype: str = 'odd',
    padlen: Optional[int] = None,
    driver: _Driver = 'auto',
    impulse_atol: float = 1e-12,
) -> Num[Array, '... obs']:
    """Zero-phase forward-backward IIR filter (scipy ``sosfiltfilt``-exact).

    Filters forward then backward with steady-state initial conditions and
    odd padding, cancelling phase and squaring the magnitude response.
    Matches ``scipy.signal.sosfiltfilt`` to machine precision.

    Both passes thread the steady-state ``zi`` initial conditions through the
    resolved engine (``driver='auto'`` -> ``'fft'`` on GPU, ``'scan'`` on CPU;
    see :func:`sosfilt`).  The FFT engine adds the ``zi`` transient
    ``x[0] * g`` (the cascade's zero-input response) over the first ``n_taps``
    samples, so the edges stay scipy-exact -- the zero-phase path is no longer
    scan-only.  ``impulse_atol`` controls the FFT truncation (see
    :func:`sosfilt`).

    Parameters
    ----------
    X
        Signal; filtered along ``axis`` (default trailing).
    sos
        ``(n_sections, 6)`` second-order sections (e.g. from
        :func:`butterworth_sos`).  Static host coefficients (NumPy).
    axis
        Axis of ``X`` along which to filter (default ``-1``, the trailing
        axis).
    padtype
        Edge-padding scheme applied before the forward-backward passes.  Only
        ``'odd'`` padding is supported.
    padlen
        Number of samples reflected at each edge.  ``None`` (default) uses the
        scipy convention (``3 * (2 * n_sections + 1 - n_trivial)``, where
        ``n_trivial`` counts first-order sections); must be less than the
        signal length along ``axis``.
    driver
        Numerical variant for both passes (see :func:`sosfilt`): ``'auto'``
        (default), or a forced ``'fft'`` / ``'scan'`` / ``'associative'``.
    impulse_atol
        FFT-engine impulse-response truncation tolerance (see :func:`sosfilt`).

    Returns
    -------
    Num[Array, '... obs']
        The zero-phase filtered signal, same shape and dtype as ``X``.
        Differentiable through ``X``.
    """
    sos_np = np.asarray(sos)
    n_sections = sos_np.shape[0]
    x = jnp.moveaxis(jnp.asarray(X), axis, 0)  # time-major
    n = x.shape[0]
    if padlen is None:
        # scipy convention: ntaps = 2*n_sections + 1, reduced by the number
        # of first-order sections (those with a zero b2 or a2), so odd-order
        # filters pad less.  Matching it makes the edges scipy-exact.
        n_trivial = int(
            min((sos_np[:, 2] == 0).sum(), (sos_np[:, 5] == 0).sum())
        )
        padlen = 3 * (2 * n_sections + 1 - n_trivial)
    if padlen >= n:
        raise ValueError(
            f'padlen={padlen} must be < signal length {n}; pass a smaller '
            'padlen for short signals.'
        )

    if padtype == 'odd' and padlen > 0:
        left = 2 * x[:1] - x[padlen:0:-1]
        right = 2 * x[-1:] - x[-2 : -padlen - 2 : -1]
        xp = jnp.concatenate([left, x, right], axis=0)
    elif padlen == 0:
        xp = x
    else:
        raise ValueError(f"padtype={padtype!r}; only 'odd' is supported.")

    zi = _sos_zi(sos_np)
    y = _sosfilt_apply(
        sos_np,
        xp,
        zi=zi,
        driver=driver,
        impulse_atol=impulse_atol,
    )
    y = _sosfilt_apply(
        sos_np,
        jnp.flip(y, axis=0),
        zi=zi,
        driver=driver,
        impulse_atol=impulse_atol,
    )
    y = jnp.flip(y, axis=0)
    if padlen > 0:
        y = y[padlen:-padlen]
    return jnp.moveaxis(y, 0, axis)


def iir_filter(
    X: Num[Array, '... obs'],
    *,
    fs: float = 1.0,
    btype: _BType,
    lo: Optional[float] = None,
    hi: Optional[float] = None,
    order: int = 2,
    zero_phase: bool = True,
    driver: _Driver = 'auto',
    impulse_atol: float = 1e-12,
    axis: int = -1,
) -> Num[Array, '... obs']:
    """Recursive Butterworth IIR filter (design + apply).

    Convenience wrapper: designs a Butterworth ``sos``
    (:func:`butterworth_sos`) and applies it.

    Parameters
    ----------
    X
        Signal; filtered along ``axis`` (default trailing time axis).
    fs
        Sampling frequency in Hz (``1 / TR``).
    btype
        ``'lowpass'`` / ``'highpass'`` / ``'bandpass'`` / ``'bandstop'``.
    lo, hi
        Cut-offs in Hz.  ``'lowpass'`` uses ``hi``; ``'highpass'`` uses
        ``lo``; band types use both (``lo < hi``).
    order
        Butterworth order.
    zero_phase
        ``True`` (default) -- forward-backward :func:`sosfiltfilt` (zero phase,
        squared magnitude, the fMRI default).  ``False`` -- single causal
        forward pass (:func:`sosfilt`); preserves causality but imposes the
        Butterworth phase delay.
    driver
        Numerical variant for both the causal (``zero_phase=False``) and
        zero-phase (``zero_phase=True``) paths: ``'auto'`` (default; ``'fft'``
        on GPU, ``'scan'`` on CPU -- see :func:`sosfilt`) or a forced ``'fft'``
        / ``'scan'`` / ``'associative'``.
    impulse_atol
        FFT-engine impulse-response truncation tolerance (see :func:`sosfilt`).
    axis
        Axis of ``X`` along which to filter (default ``-1``, the trailing time
        axis).

    Returns
    -------
    Filtered signal, same shape as ``X``.  Differentiable through ``X``;
    ``fs`` / cut-offs / ``order`` are static.
    """
    sos = butterworth_sos(order=order, fs=fs, btype=btype, lo=lo, hi=hi)
    if zero_phase:
        return sosfiltfilt(
            X,
            sos,
            axis=axis,
            driver=driver,
            impulse_atol=impulse_atol,
        )
    return sosfilt(
        X,
        sos,
        axis=axis,
        driver=driver,
        impulse_atol=impulse_atol,
    )
