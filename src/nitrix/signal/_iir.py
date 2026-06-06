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
dependency (SPEC §5.2), so ``butterworth_sos`` reproduces the math of
``scipy.signal.butter(output='sos')`` (validated to machine precision in
``tests/test_iir.py``) without importing it.  Cut-offs / order are static
(constant-folded into the trace); the design is not differentiated.

Application is the recurrence ``y[n] = b0 x[n] + b1 x[n-1] + b2 x[n-2]
- a1 y[n-1] - a2 y[n-2]`` per biquad, with two engines:

- ``backend='scan'`` (default) -- sequential ``lax.scan`` over time,
  vectorised across all channels; compiles to a single fused loop, low
  memory.  Best for the fMRI regime (modest ``T``, many voxels).
- ``backend='associative'`` -- the biquad's linear recurrence composed via
  ``lax.associative_scan`` (the parallel-prefix pattern also used by
  ``signal.linear_interpolate``); ``O(log T)`` depth, for latency-bound
  long single series.  Forward filtering only.

``sosfiltfilt`` is the zero-phase forward-backward filter, with scipy-exact
steady-state initial conditions (``lfilter_zi``-equivalent) and odd
padding; it matches ``scipy.signal.sosfiltfilt`` to machine precision.

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

# Hard cap on the impulse-response length the FFT engine will truncate to.
# A filter that has not decayed below ``impulse_atol`` within this many taps
# is too sharp (poles hugging the unit circle) for a cheap FFT convolution;
# the engine falls back to the recurrence instead.
_IIR_FFT_MAX_TAPS = 1 << 15

__all__ = ['butterworth_sos', 'sosfilt', 'sosfiltfilt', 'iir_filter']

_BType = str  # 'lowpass' | 'highpass' | 'bandpass' | 'bandstop'
_Backend = str  # 'auto' | 'fft' | 'scan' | 'associative'


def _resolve_iir_backend(backend: _Backend) -> _Backend:
    '''Resolve ``'auto'`` to the engine that wins on this platform.

    On GPU the recursion is latency-bound, so ``'auto'`` selects the parallel
    ``'fft'`` convolution engine (which beats cupy on the L4 and falls back to
    a recurrence for filters too sharp to truncate cheaply).  On CPU the
    sequential ``'scan'`` recurrence is fast and exact, so ``'auto'`` keeps it.
    Explicit ``'fft'`` / ``'scan'`` / ``'associative'`` pass through.
    '''
    if backend == 'auto':
        return 'fft' if default_backend_is_gpu() else 'scan'
    return backend


# ---------------------------------------------------------------------------
# Butterworth design (pure NumPy; static cut-offs).  Mirrors the zpk maths of
# scipy.signal.butter + lp2{lp,hp,bp,bs} + bilinear + zpk2sos.
# ---------------------------------------------------------------------------


def _buttap(order: int) -> Tuple[np.ndarray, np.ndarray, float]:
    '''Analog Butterworth low-pass prototype (zeros, poles, gain).'''
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
    '''Group roots into conjugate pairs / real pairs (each <= 2 roots).'''
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
    real_pairs = [
        np.array(reals[i:i + 2]) for i in range(0, len(reals), 2)
    ]
    return cpairs + real_pairs


def _section_poly(group: np.ndarray) -> np.ndarray:
    '''Real length-3 polynomial coefficients from a <=2 root group.'''
    coeffs = np.real(np.poly(group))
    return np.concatenate([coeffs, np.zeros(3 - len(coeffs))])


def _zpk2sos(z: np.ndarray, p: np.ndarray, k: float) -> np.ndarray:
    '''Group conjugate poles/zeros into biquad second-order sections.

    Any valid conjugate grouping yields the same overall transfer function;
    for the modest orders here this simple pairing is well-conditioned (and
    parity is checked against scipy via the *frequency response*, which is
    grouping-invariant).
    '''
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
    '''Validate edges and return them normalised to (0, 1) over Nyquist.'''
    nyq = 0.5 * fs

    def norm(c: float, name: str) -> float:
        if not 0.0 < c < nyq:
            raise ValueError(
                f'{name}={c} must lie in (0, fs/2) = (0, {nyq}).'
            )
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
    '''Design a digital Butterworth filter as second-order sections.

    Returns the ``(n_sections, 6)`` SOS array ``[b0, b1, b2, a0, a1, a2]``
    per row -- the same layout as ``scipy.signal.butter(output='sos')`` (and
    matching its transfer function to machine precision), implemented in
    pure NumPy.  ``order``, ``fs`` and the cut-offs are static, so the result
    is a host (NumPy) constant -- the *design* is not traced; only the
    *application* (``sosfilt`` / ``sosfiltfilt``) runs in JAX.

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
    '''
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
    '''Per-section steady-state delay state, scaled along the cascade.

    For each biquad, the transposed-DF2 state that makes a step input
    produce no startup transient: ``zi = (I - A^T)^{-1} B`` with
    ``A`` the companion matrix.  Scaled by the running DC gain so the
    cascade is consistent (scipy ``sosfilt_zi``).
    '''
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
    sos: np.ndarray, n: int, zi: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    '''Host (NumPy, fp64) impulse and zero-input responses of the cascade.

    Runs the transposed-DF2 cascade recurrence -- the same one
    ``_sosfilt_scan`` realises -- for ``n`` samples:

    - ``h`` -- the cascade **impulse response** (unit impulse, zero state).  A
      fixed-coefficient IIR filter is LTI, so its zero-state output is exactly
      ``conv(x, h)``; ``h`` decays geometrically (poles inside the unit
      circle), so an ``n``-tap truncation has a geometrically bounded error.
    - ``g`` -- the cascade **zero-input response** to the per-section initial
      state ``zi`` (zero input).  That response is linear in the ``x[0]``
      scaling applied to ``zi``, so ``sosfilt`` with ``zi`` is
      ``conv(x, h) + x[0] * g``.

    Both are host constants (``sos`` / ``zi`` are static) and fold into the
    trace; the device side only runs the FFT convolution.
    '''
    b0, b1, b2 = sos[:, 0] / sos[:, 3], sos[:, 1] / sos[:, 3], sos[:, 2] / sos[:, 3]
    a1, a2 = sos[:, 4] / sos[:, 3], sos[:, 5] / sos[:, 3]
    ns = sos.shape[0]
    h = np.empty(n, dtype=np.float64)
    g = np.empty(n, dtype=np.float64)
    sh = np.zeros((ns, 2))
    sg = np.zeros((ns, 2)) if zi is None else np.asarray(zi, dtype=np.float64).copy()
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
    sos: np.ndarray, atol: float, cap: int = _IIR_FFT_MAX_TAPS,
) -> Optional[int]:
    '''Smallest tap count after which ``|h| <= atol`` (or ``None`` if the
    impulse response has not decayed within ``cap`` -- a too-sharp filter).'''
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
    '''Reshape a length-F frequency vector to broadcast over ``(F, *channels)``.'''
    return arr.reshape((-1,) + (1,) * (ndim - 1))


def _sosfilt_fft(
    sos: np.ndarray, x: Array, zi: Optional[np.ndarray], n_taps: int,
) -> Array:
    '''Apply the cascade as an FFT convolution with its impulse response.

    ``x`` is time-major ``(T, *channels)``.  The zero-state output is
    ``conv(x, h)`` (linear convolution via the FFT, truncated to ``n_taps``);
    with ``zi`` the steady-state transient ``x[0] * g`` is added back over the
    first ``n_taps`` samples so the edges stay scipy-exact.  ``O(T log T)``,
    no recurrence, fully parallel.
    '''
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
        g_seq = jnp.asarray(g, x.dtype).reshape((n_taps,) + (1,) * (x.ndim - 1))
        y = y.at[:n_taps].add(g_seq * x[:1])
    return y


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------


def _sosfilt_scan(
    sos: np.ndarray,
    x: Array,
    zi: Optional[np.ndarray],
) -> Array:
    '''Cascade of biquads via sequential ``lax.scan`` over time (axis 0).

    ``x`` is time-major ``(T, *channels)``.  ``zi`` (per-section delay
    state for unit input) is scaled by ``x[0]`` when provided.
    '''
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
            b0: float = b0, b1: float = b1, b2: float = b2,
            a1: float = a1, a2: float = a2,
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
    '''Cascade of biquads via parallel ``lax.associative_scan``.

    The transposed-DF2 state ``s = (z1, z2)`` -- the *same* state
    ``_sosfilt_scan`` carries -- evolves as a first-order linear recurrence
    ``s[n+1] = M s[n] + c u[n]`` with constant ``M = [[-a1, 1], [-a2, 0]]`` and
    ``c = [b1 - a1 b0, b2 - a2 b0]``, and the output is
    ``y[n] = s[n][0] + b0 u[n]``.  A constant-coefficient linear recurrence
    composes associatively, so the prefix scan resolves the whole cascade in
    ``O(log T)`` depth.

    ``zi`` (per-section steady-state delay, scaled by ``x[0]``) sets the
    initial state ``s[0]``; its homogeneous response ``M^n s[0]`` is read off
    the same scan's cumulative transition ``a_cum``.  With ``zi=None`` the
    initial state is zero (the homogeneous term vanishes and ``a_cum`` is not
    materialised).  This is the engine that makes ``sosfilt`` /
    ``sosfiltfilt`` competitive on the GPU; it matches ``_sosfilt_scan`` (and
    ``scipy.signal``) to round-off.
    '''
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
                combine, (a_seq, b_seq), axis=0,
            )
            s0 = jnp.stack(
                [float(zi[i, 0]) * x0, float(zi[i, 1]) * x0], axis=-1,
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
    backend: _Backend,
    impulse_atol: float,
) -> Array:
    '''Apply the cascade with the platform-resolved engine.

    ``'fft'`` truncates the impulse response at ``impulse_atol`` and convolves;
    if the filter has not decayed within ``_IIR_FFT_MAX_TAPS`` it falls back to
    a recurrence (``associative`` on GPU, ``scan`` on CPU) with a warning.
    '''
    backend = _resolve_iir_backend(backend)
    if backend == 'fft':
        n_taps = _sos_impulse_taps(sos, impulse_atol)
        if n_taps is not None:
            return _sosfilt_fft(sos, x, zi, n_taps)
        backend = 'associative' if default_backend_is_gpu() else 'scan'
        warnings.warn(
            f'sosfilt: impulse response has not decayed below '
            f'impulse_atol={impulse_atol:g} within {_IIR_FFT_MAX_TAPS} taps '
            f'(filter too sharp for the FFT engine); falling back to '
            f'backend={backend!r}.',
            stacklevel=3,
        )
    if backend == 'scan':
        return _sosfilt_scan(sos, x, zi=zi)
    if backend == 'associative':
        return _sosfilt_associative(sos, x, zi=zi)
    raise ValueError(
        f"backend={backend!r}; expected 'auto', 'fft', 'scan' or "
        "'associative'."
    )


def sosfilt(
    X: Num[Array, '... obs'],
    sos: np.ndarray,
    *,
    axis: int = -1,
    backend: _Backend = 'auto',
    impulse_atol: float = 1e-12,
) -> Num[Array, '... obs']:
    '''Apply a causal IIR filter (forward only) given its SOS cascade.

    Parameters
    ----------
    X
        Signal; filtered along ``axis`` (default trailing).
    sos
        ``(n_sections, 6)`` second-order sections (e.g. from
        ``butterworth_sos``).  Static host coefficients (NumPy); they are
        baked into the engine, not traced.
    backend
        ``'auto'`` (default) picks the engine by platform: ``'fft'`` on GPU,
        ``'scan'`` on CPU.  The recursion is latency-bound on the GPU, so the
        parallel FFT-convolution engine (an IIR filter is LTI, so its output
        is exactly convolution with the impulse response) wins there -- it
        beats cupy on the L4; on the CPU the sequential ``lax.scan`` is fast
        and exact, so it is kept.  Force ``'fft'``, ``'scan'`` or the
        parallel-prefix ``'associative'`` to override.
    impulse_atol
        FFT engine only: truncate the impulse response where it has decayed
        below this absolute tolerance (default ``1e-12``, effectively exact).
        Larger values give shorter kernels / smaller FFTs at the cost of a
        (geometrically bounded) edge error; a filter too sharp to decay within
        ``2**15`` taps falls back to a recurrence with a warning.
    '''
    sos_np = np.asarray(sos)
    x = jnp.moveaxis(jnp.asarray(X), axis, 0)
    y = _sosfilt_apply(
        sos_np, x, zi=None, backend=backend, impulse_atol=impulse_atol,
    )
    return jnp.moveaxis(y, 0, axis)


def sosfiltfilt(
    X: Num[Array, '... obs'],
    sos: np.ndarray,
    *,
    axis: int = -1,
    padtype: str = 'odd',
    padlen: Optional[int] = None,
    backend: _Backend = 'auto',
    impulse_atol: float = 1e-12,
) -> Num[Array, '... obs']:
    '''Zero-phase forward-backward IIR filter (scipy ``sosfiltfilt``-exact).

    Filters forward then backward with steady-state initial conditions and
    odd padding, cancelling phase and squaring the magnitude response.
    Matches ``scipy.signal.sosfiltfilt`` to machine precision.

    Both passes thread the steady-state ``zi`` initial conditions through the
    platform-resolved engine (``backend='auto'`` -> ``'fft'`` on GPU, ``'scan'``
    on CPU; see ``sosfilt``).  The FFT engine adds the ``zi`` transient
    ``x[0] * g`` (the cascade's zero-input response) over the first ``n_taps``
    samples, so the edges stay scipy-exact -- the zero-phase path is no longer
    scan-only.  ``impulse_atol`` controls the FFT truncation (see ``sosfilt``).
    '''
    sos_np = np.asarray(sos)
    n_sections = sos_np.shape[0]
    x = jnp.moveaxis(jnp.asarray(X), axis, 0)  # time-major
    n = x.shape[0]
    if padlen is None:
        # scipy convention: ntaps = 2*n_sections + 1, reduced by the number
        # of first-order sections (those with a zero b2 or a2), so odd-order
        # filters pad less.  Matching it makes the edges scipy-exact.
        n_trivial = int(min((sos_np[:, 2] == 0).sum(),
                            (sos_np[:, 5] == 0).sum()))
        padlen = 3 * (2 * n_sections + 1 - n_trivial)
    if padlen >= n:
        raise ValueError(
            f'padlen={padlen} must be < signal length {n}; pass a smaller '
            'padlen for short signals.'
        )

    if padtype == 'odd' and padlen > 0:
        left = 2 * x[:1] - x[padlen:0:-1]
        right = 2 * x[-1:] - x[-2:-padlen - 2:-1]
        xp = jnp.concatenate([left, x, right], axis=0)
    elif padlen == 0:
        xp = x
    else:
        raise ValueError(f"padtype={padtype!r}; only 'odd' is supported.")

    zi = _sos_zi(sos_np)
    y = _sosfilt_apply(
        sos_np, xp, zi=zi, backend=backend, impulse_atol=impulse_atol,
    )
    y = _sosfilt_apply(
        sos_np, jnp.flip(y, axis=0), zi=zi, backend=backend,
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
    backend: _Backend = 'auto',
    impulse_atol: float = 1e-12,
    axis: int = -1,
) -> Num[Array, '... obs']:
    '''Recursive Butterworth IIR filter (design + apply).

    Convenience wrapper: designs a Butterworth ``sos`` (``butterworth_sos``)
    and applies it.

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
        ``True`` (default) -- forward-backward ``sosfiltfilt`` (zero phase,
        squared magnitude, the fMRI default).  ``False`` -- single causal
        forward pass (``sosfilt``); preserves causality but imposes the
        Butterworth phase delay.
    backend
        Engine for both the causal (``zero_phase=False``) and zero-phase
        (``zero_phase=True``) paths: ``'auto'`` (default; ``'fft'`` on GPU,
        ``'scan'`` on CPU -- see ``sosfilt``) or a forced ``'fft'`` /
        ``'scan'`` / ``'associative'``.
    impulse_atol
        FFT-engine impulse-response truncation tolerance (see ``sosfilt``).

    Returns
    -------
    Filtered signal, same shape as ``X``.  Differentiable through ``X``;
    ``fs`` / cut-offs / ``order`` are static.
    '''
    sos = butterworth_sos(order=order, fs=fs, btype=btype, lo=lo, hi=hi)
    if zero_phase:
        return sosfiltfilt(
            X, sos, axis=axis, backend=backend, impulse_atol=impulse_atol,
        )
    return sosfilt(
        X, sos, axis=axis, backend=backend, impulse_atol=impulse_atol,
    )
