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

from typing import Optional, Tuple

import jax.lax as lax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Num

from .._internal.backend import default_backend_is_gpu

__all__ = ['butterworth_sos', 'sosfilt', 'sosfiltfilt', 'iir_filter']

_BType = str  # 'lowpass' | 'highpass' | 'bandpass' | 'bandstop'
_Backend = str  # 'auto' | 'scan' | 'associative'


def _resolve_iir_backend(backend: _Backend) -> _Backend:
    '''Resolve ``'auto'`` to the recurrence engine that wins on this platform.

    The sequential ``scan`` wins on CPU; the parallel-prefix ``associative``
    wins on GPU (measured 8.3x faster on the L4 at ch=1024/obs=4096, while
    losing ~9x on CPU).  Explicit ``'scan'`` / ``'associative'`` pass through.
    '''
    if backend == 'auto':
        return 'associative' if default_backend_is_gpu() else 'scan'
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


def _sosfilt_associative(sos: np.ndarray, x: Array) -> Array:
    '''Cascade of biquads via parallel ``lax.associative_scan`` (zero state).

    Each biquad's recursive (AR) part is the linear recurrence
    ``w[n] = u[n] - a1 w[n-1] - a2 w[n-2]``; the state ``[w[n], w[n-1]]``
    evolves as ``s[n] = A s[n-1] + [u[n], 0]`` with constant ``A``, which
    composes associatively, so the prefix scan resolves it in ``O(log T)``.
    The numerator is then a 3-tap FIR on ``w``.
    '''
    y = x
    n_t = x.shape[0]
    for s in sos:
        b0, b1, b2 = (float(v) / float(s[3]) for v in s[:3])
        a1, a2 = float(s[4]) / float(s[3]), float(s[5]) / float(s[3])
        u = y
        a_mat = jnp.array([[-a1, -a2], [1.0, 0.0]], dtype=x.dtype)
        a_seq = jnp.broadcast_to(a_mat, (n_t,) + u.shape[1:] + (2, 2))
        b_seq = jnp.stack([u, jnp.zeros_like(u)], axis=-1)

        def combine(
            left: Tuple[Array, Array], right: Tuple[Array, Array]
        ) -> Tuple[Array, Array]:
            a_l, b_l = left
            a_r, b_r = right
            a = jnp.einsum('...ij,...jk->...ik', a_r, a_l)
            b = jnp.einsum('...ij,...j->...i', a_r, b_l) + b_r
            return a, b

        _, b_cum = lax.associative_scan(combine, (a_seq, b_seq), axis=0)
        w = b_cum[..., 0]
        w1 = jnp.concatenate([jnp.zeros((1,) + w.shape[1:]), w[:-1]], axis=0)
        w2 = jnp.concatenate([jnp.zeros((2,) + w.shape[1:]), w[:-2]], axis=0)
        y = b0 * w + b1 * w1 + b2 * w2
    return y


def sosfilt(
    X: Num[Array, '... obs'],
    sos: np.ndarray,
    *,
    axis: int = -1,
    backend: _Backend = 'auto',
) -> Num[Array, '... obs']:
    '''Apply a causal IIR filter (forward only) given its SOS cascade.

    Parameters
    ----------
    X
        Signal; filtered along ``axis`` (default trailing).
    sos
        ``(n_sections, 6)`` second-order sections (e.g. from
        ``butterworth_sos``).  Static host coefficients (NumPy); they are
        baked into the recurrence, not traced.
    backend
        ``'auto'`` (default) picks the recurrence engine by platform --
        ``'scan'`` on CPU, ``'associative'`` on GPU -- because the optimal
        choice flips with the device (the sequential ``lax.scan`` is
        low-overhead on CPU; the parallel-prefix ``associative_scan`` is
        ``O(log T)`` depth and fills the GPU).  Pass ``'scan'`` /
        ``'associative'`` to force a specific engine.
    '''
    sos_np = np.asarray(sos)
    x = jnp.moveaxis(jnp.asarray(X), axis, 0)
    backend = _resolve_iir_backend(backend)
    if backend == 'scan':
        y = _sosfilt_scan(sos_np, x, zi=None)
    elif backend == 'associative':
        y = _sosfilt_associative(sos_np, x)
    else:
        raise ValueError(
            f"backend={backend!r}; expected 'auto', 'scan' or 'associative'."
        )
    return jnp.moveaxis(y, 0, axis)


def sosfiltfilt(
    X: Num[Array, '... obs'],
    sos: np.ndarray,
    *,
    axis: int = -1,
    padtype: str = 'odd',
    padlen: Optional[int] = None,
) -> Num[Array, '... obs']:
    '''Zero-phase forward-backward IIR filter (scipy ``sosfiltfilt``-exact).

    Filters forward then backward with steady-state initial conditions and
    odd padding, cancelling phase and squaring the magnitude response.
    Matches ``scipy.signal.sosfiltfilt`` to machine precision.  Uses the
    ``'scan'`` recurrence internally (the initial-condition handling that
    makes it exact is cleanest there).
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
    y = _sosfilt_scan(sos_np, xp, zi=zi)
    y = _sosfilt_scan(sos_np, jnp.flip(y, axis=0), zi=zi)
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
        Recurrence engine for the *causal* (``zero_phase=False``) path:
        ``'auto'`` (default; ``scan`` on CPU, ``associative`` on GPU) or a
        forced ``'scan'`` / ``'associative'``.  The zero-phase path is
        always ``'scan'`` (exact initial conditions).

    Returns
    -------
    Filtered signal, same shape as ``X``.  Differentiable through ``X``;
    ``fs`` / cut-offs / ``order`` are static.
    '''
    sos = butterworth_sos(order=order, fs=fs, btype=btype, lo=lo, hi=hi)
    if zero_phase:
        return sosfiltfilt(X, sos, axis=axis)
    return sosfilt(X, sos, axis=axis, backend=backend)
