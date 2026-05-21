# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Lomb-Scargle spectral methods for irregularly-sampled time series.

Two distinct primitives that share the underlying sin / cos basis
but solve different problems:

- ``lomb_scargle_periodogram`` -- canonical Press-Rybicki / Scargle
  1982 periodogram for **spectral analysis** of irregularly-sampled
  data.  Each (cosine, sine) amplitude pair is estimated
  independently per trial frequency after the
  ``tau``-orthogonalisation that makes the per-frequency
  least-squares closed-form.  Fast; the *reconstruction* implied
  by summing the per-frequency components does **not** pass through
  the observed samples exactly -- if used for interpolation,
  produces **visible boundary discontinuities** at observed /
  censored transitions.  Wrong primitive for interpolation.

- ``lomb_scargle_interpolate`` -- **interpolation** primitive for
  fMRI motion-censoring (Power 2014 protocol).  Solves a **joint**
  least-squares regression of the observed samples on the entire
  sin / cos basis (plus DC):
  ``min ||M (B beta - y)||^2`` where ``M`` is the observed-sample
  mask, ``B`` is the basis, ``y`` is the data, ``beta`` are the
  spectral coefficients.  Joint fit guarantees
  ``B[obs] @ beta == y[obs]`` (modulo rank deficiency handled
  by a Tikhonov ridge), so the spliced output has **no boundary
  discontinuity**.

Memory regime
-------------

The dominant cost is the basis matrix ``B`` of shape
``(n_obs, K)`` where ``K = 1 + 2 * n_freq``.  At fMRI typical
``n_obs = 500`` and ``K = 499``, ``B`` is ~1 MB at fp32 -- shared
across channels.

For **shared-mask** input (mask is broadcast-compatible with data
along the leading dims; the typical fMRI case where one
motion-censor mask applies to all voxels), we compute the Gram
matrix ``G = B^T diag(mask) B`` of shape ``(K, K)`` and its
Cholesky ``L`` **once**, then solve the right-hand sides for all
channels as a single batched triangular solve.  Total memory at
V=1M voxels, T=500, K=499:

- Shared basis: ``T * K * 4 = 1 MB``.
- Shared Gram / Cholesky: ``2 * K^2 * 4 = 2 MB``.
- Per-channel rhs / beta / recon: ``3 * V * K * 4 = 6 GB``.
- Output / data: ``2 * V * T * 4 = 4 GB``.

Total ~10 GB -- fits on an A100 80GB or H100 with room.  For
**per-channel-mask** input (mask shape matches data shape; each
channel has its own censoring pattern), we fall back to a
``vmap`` over channels which creates a per-channel Gram of size
``(V, K, K)`` -- 1 TB at V=1M.  In that case the function raises
a clear error pointing at the shared-mask path.

References
----------
Lomb, N. R. (1976).  Astrophys. Space Sci. 39, 447-462.
Scargle, J. D. (1982).  Astrophys. J. 263, 835-853.
Press, W. H. & Rybicki, G. B. (1989).  Astrophys. J. 338, 277-280.
Power, J. D. et al. (2014).  NeuroImage 84, 320-341.
"""
from __future__ import annotations

from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp
from jax.typing import DTypeLike
from jaxtyping import Array, Bool, Float, Num

from ..linalg._solver import safe_eigh


__all__ = ['lomb_scargle_interpolate', 'lomb_scargle_periodogram']


# ---------------------------------------------------------------------------
# Frequency grid
# ---------------------------------------------------------------------------


def _trial_frequencies(
    n_obs: int,
    dt: float,
    oversampling: float,
    high_factor: float,
    dtype: DTypeLike,
    censoring_budget: float = 0.4,
) -> Float[Array, 'n_freq']:
    '''Choose trial angular frequencies (Press-Rybicki convention).

    Min: ``2 pi df`` with ``df = 1 / (T * oversampling)``.
    Max: ``2 pi f_max`` with ``f_max = high_factor * Nyquist``.
    Count: bounded so ``2 * n_freq + 1 <= n_obs * (1 - censoring_budget)``
    -- i.e., we assume up to ``censoring_budget * n_obs`` frames may
    be censored and leave the basis with enough column slack to
    stay rank-sufficient.  Default ``censoring_budget = 0.4`` (40%
    censoring tolerated without rank deficiency); raise the budget
    or pass a smaller ``oversampling`` if you expect heavier
    censoring.
    '''
    T = n_obs * dt
    df = 1.0 / (T * oversampling)
    f_max = high_factor / (2.0 * dt)
    n_freq_grid = max(int(f_max / df), 1)
    # Static cap: ensure 2 * n_freq + 1 <= n_obs * (1 - budget),
    # leaving slack for runtime censoring.
    margin = max(int(n_obs * (1.0 - censoring_budget)), 3)
    n_freq_cap = max((margin - 1) // 2, 1)
    n_freq = min(n_freq_grid, n_freq_cap)
    return 2.0 * jnp.pi * jnp.arange(1, n_freq + 1, dtype=dtype) * df


def _lomb_scargle_basis(
    n_obs: int,
    dt: float,
    omega: Float[Array, 'n_freq'],
    dtype: DTypeLike,
) -> Float[Array, 'n_obs n_basis']:
    '''Joint regression basis ``[DC | cos | sin]``.

    Returns a ``(n_obs, 1 + 2 * n_freq)`` matrix; rows index time,
    columns index basis component.  Shared across channels.
    '''
    t = jnp.arange(n_obs, dtype=dtype) * dt
    arg = omega[None, :] * t[:, None]
    return jnp.concatenate(
        [jnp.ones((n_obs, 1), dtype=dtype), jnp.cos(arg), jnp.sin(arg)],
        axis=-1,
    )


# ---------------------------------------------------------------------------
# Periodogram (Press-Rybicki / Scargle 1982 form, spectral-analysis only)
# ---------------------------------------------------------------------------


def lomb_scargle_periodogram(
    data: Float[Array, '... obs'],
    mask: Bool[Array, '... obs'],
    *,
    dt: float = 1.0,
    oversampling: float = 4.0,
    high_factor: float = 1.0,
) -> Tuple[Float[Array, 'n_freq'], Float[Array, '... n_freq']]:
    '''Scargle 1982 normalised Lomb-Scargle periodogram.

    Per-frequency power computed from the observed (non-masked)
    samples.  Normalisation matches
    ``scipy.signal.lombscargle(normalize=True)``.

    For **interpolation** use ``lomb_scargle_interpolate`` -- the
    implied reconstruction from per-frequency LS coefficients
    does not pass through observed samples exactly.

    Parameters
    ----------
    data, mask
        Observation tensor (last axis = time) and boolean
        validity mask.
    dt
        Sampling interval.  Default ``1.0``.
    oversampling, high_factor
        Frequency-grid parameters.

    Returns
    -------
    ``(freqs, power)`` -- cyclic frequencies and per-channel
    normalised power.
    '''
    n_obs = data.shape[-1]
    # Periodogram doesn't have a Gram-rank concern (per-frequency
    # univariates), so use a near-zero censoring budget to expose
    # the full Press-Rybicki frequency grid up to Nyquist.
    omega = _trial_frequencies(
        n_obs, dt, oversampling, high_factor, data.dtype,
        censoring_budget=0.0,
    )
    freqs = omega / (2.0 * jnp.pi)

    def core(
        d: Float[Array, 'n_obs'], m: Bool[Array, 'n_obs']
    ) -> Float[Array, 'n_freq']:
        t = jnp.arange(n_obs, dtype=d.dtype) * dt
        mf = m.astype(d.dtype)
        n_valid = jnp.sum(mf)
        y_mean = jnp.sum(mf * d) / n_valid
        y_centred = (d - y_mean) * mf
        two_w = 2.0 * omega[:, None]
        t_b = t[None, :]
        num = jnp.sum(mf * jnp.sin(two_w * t_b), axis=-1)
        den = jnp.sum(mf * jnp.cos(two_w * t_b), axis=-1)
        tau = jnp.atan2(num, den) / (2.0 * omega)
        arg = omega[:, None] * (t[None, :] - tau[:, None])
        c = jnp.cos(arg) * mf[None, :]
        s = jnp.sin(arg) * mf[None, :]
        cy = jnp.sum(y_centred[None, :] * c, axis=-1)
        sy = jnp.sum(y_centred[None, :] * s, axis=-1)
        c_norm = jnp.sum(c * c, axis=-1)
        s_norm = jnp.sum(s * s, axis=-1)
        var_y = jnp.sum(mf * (d - y_mean) ** 2) / n_valid
        return 0.5 * (cy ** 2 / c_norm + sy ** 2 / s_norm) / var_y

    fn: Callable[..., Any] = core
    for _ in range(data.ndim - 1):
        fn = jax.vmap(fn, in_axes=(0, 0))
    return freqs, fn(data, mask)


# ---------------------------------------------------------------------------
# Interpolation: joint GLM regression with shared-Gram fast path
# ---------------------------------------------------------------------------


def _lomb_scargle_solve_shared_mask(
    data: Float[Array, 'n_chan n_obs'],
    mask: Bool[Array, 'n_obs'],
    basis: Float[Array, 'n_obs K'],
    rcond: float,
) -> Float[Array, 'n_chan n_obs']:
    '''Joint LS interpolation with a shared (channel-invariant) mask.

    Robust solver: factor the Gram via ``eigh`` (symmetric
    eigendecomposition) and apply a thresholded pseudo-inverse.
    Truncating eigenvalues below ``rcond * max(eigval)`` handles
    rank-deficient Gram matrices (which arise whenever
    ``2 * n_freq + 1 > n_valid``) without arbitrary ridge
    parameters.

    Memory:

    - ``B_w``: ``(n_obs, K)``  (1 MB at fMRI scale).
    - ``G``, eigvecs ``V``: ``(K, K)`` each (1 MB each).
    - ``rhs``: ``(n_chan, K)``.
    - ``recon``: ``(n_chan, n_obs)``.

    No ``(n_chan, K, K)`` intermediate -- the Gram and its
    eigendecomposition are shared across all channels and
    computed once.
    '''
    mf = mask.astype(data.dtype)
    B_w = basis * mf[:, None]               # (n_obs, K)
    G = B_w.T @ basis                       # (K, K), symmetric PSD
    # Threshold-truncated pseudo-inverse via eigh.
    ev, V = safe_eigh(G)                    # ev ascending
    # Truncate eigenvalues below the relative threshold.  The
    # max eigval is at the end; ``rcond * max`` is the floor.
    thresh = rcond * ev[-1]
    ev_inv = jnp.where(ev > thresh, 1.0 / ev, 0.0)
    rhs = data @ B_w                        # (n_chan, K)
    # betas = V @ diag(ev_inv) @ V.T @ rhs.T
    z = V.T @ rhs.swapaxes(-1, -2)          # (K, n_chan)
    z = ev_inv[:, None] * z
    betas = V @ z                           # (K, n_chan)
    recon = (basis @ betas).swapaxes(-1, -2)
    return jnp.where(mask, data, recon)


def lomb_scargle_interpolate(
    data: Num[Array, '... obs'],
    mask: Bool[Array, '... obs'],
    *,
    dt: float = 1.0,
    oversampling: float = 4.0,
    high_factor: float = 1.0,
    rcond: float = 1e-6,
    censoring_budget: float = 0.4,
) -> Num[Array, '... obs']:
    '''Fill censored time-series frames via joint Lomb-Scargle GLM
    spectral reconstruction.

    Solves a masked least-squares regression of the observed
    samples on a ``[DC, cos(omega_k t), sin(omega_k t)]`` basis --
    **joint** over all trial frequencies, not the independent
    per-frequency form of ``lomb_scargle_periodogram``.  The
    joint fit ensures the reconstruction passes through the
    observed samples (modulo a small Tikhonov ridge), so the
    spliced output has **no boundary discontinuity** between
    observed and interpolated frames.

    This is the **Power 2014** (NeuroImage 84:320-341) protocol
    for motion-censored fMRI.  Recommended over
    ``linear_interpolate`` because:

    1. It preserves the spectral content -- linear interpolation
       injects low-frequency artefacts at bridges across long
       censored runs.
    2. The reconstruction is smooth at observed boundaries.

    Memory: shared-mask path uses ``(K, K)`` Gram / Cholesky
    intermediates regardless of leading-dim count, so V = 1M
    voxels fits in ~10 GB at fMRI shapes.  See the module
    docstring for the breakdown.

    Parameters
    ----------
    data
        Time series, observation axis is last.  Leading axes
        index channels.
    mask
        Boolean validity mask.  Must be **broadcast-compatible**
        with ``data`` (a single ``(obs,)`` mask broadcasts over
        all leading dims, which is the canonical fMRI case --
        one motion-censoring mask per scan, applied across all
        voxels).  Same-shape masks (per-channel censoring) are
        rejected -- they require a separate per-channel solve
        whose memory scales as ``V * K^2``, OOM at fMRI scale.
        Use ``linear_interpolate`` for per-channel masks, or
        explicitly ``jax.vmap`` this function over the channel
        axis if you really need it (and have the HBM).
    dt
        Sampling interval (fMRI TR in seconds).  Default ``1.0``.
    oversampling
        Frequency-grid oversampling factor.  Default ``4``
        (Press-Rybicki convention).
    high_factor
        Multiplier of Nyquist for the highest trial frequency.
        Default ``1.0``.
    rcond
        Relative threshold for eigenvalue truncation in the
        pseudo-inverse of the Gram matrix.  Eigenvalues below
        ``rcond * max(eigval)`` are zeroed in the solve.  Default
        ``1e-6``; tighten (smaller value) to keep more
        near-zero modes (risky at fp32), loosen to be more
        regularised.
    censoring_budget
        Static estimate of the maximum fraction of frames that
        may be censored.  Used at trace time to choose the
        trial-frequency count so the basis stays rank-sufficient
        under the worst-case censoring; lower values (e.g. ``0.1``)
        give more trial frequencies and better spectral
        resolution at the cost of robustness to heavy censoring.
        Default ``0.4`` -- accommodates up to 40% censored
        frames without truncation kicking in.

    Returns
    -------
    Filled time series, same shape as ``data``.

    Raises
    ------
    ValueError
        If ``mask`` is not broadcast-compatible with ``data`` (or
        is fully same-shape, which we deliberately reject for
        memory safety).

    Notes
    -----
    Differentiable through ``data``; ``mask`` is static.

    References
    ----------
    Power, J. D. et al. (2014).  NeuroImage 84, 320-341.
    '''
    n_obs = data.shape[-1]

    # Require mask to be either 1-D (n_obs,) or broadcast-compatible
    # with the trailing axis.  Reject same-shape masks for the memory
    # reason documented above.
    if mask.shape == data.shape and data.ndim > 1:
        raise ValueError(
            'lomb_scargle_interpolate: per-channel masks (mask shape '
            'matches data shape) would require a per-channel Gram '
            'matrix and are rejected for memory safety -- at fMRI '
            'scale this OOMs.  Pass a single 1-D mask of shape '
            f'({n_obs},) that applies to all channels (the typical '
            'fMRI motion-censoring case), or vmap this function '
            'manually over the channel axis if you really need it.'
        )
    if mask.ndim == data.ndim:
        # Allow when mask has explicit singletons in all leading dims
        # (i.e., it broadcasts as if 1-D).  Squeeze for the solver path.
        for ax in range(data.ndim - 1):
            if mask.shape[ax] != 1:
                raise ValueError(
                    f'lomb_scargle_interpolate: mask shape {mask.shape} '
                    f'must broadcast against data shape {data.shape} as '
                    f'a shared mask (singleton in all leading dims).'
                )
        mask = mask.reshape((n_obs,))
    elif mask.ndim == 1:
        if mask.shape[0] != n_obs:
            raise ValueError(
                f'lomb_scargle_interpolate: 1-D mask length '
                f'{mask.shape[0]} must match data n_obs={n_obs}.'
            )
    else:
        raise ValueError(
            f'lomb_scargle_interpolate: mask must be 1-D of length '
            f'n_obs or fully-singleton broadcast-compatible; got '
            f'shape {mask.shape} vs data shape {data.shape}.'
        )

    omega = _trial_frequencies(
        n_obs, dt, oversampling, high_factor, data.dtype,
        censoring_budget=censoring_budget,
    )
    basis = _lomb_scargle_basis(n_obs, dt, omega, data.dtype)

    # Flatten leading dims into a single channel batch axis.
    leading_shape = data.shape[:-1]
    data_2d = data.reshape((-1, n_obs)) if leading_shape else data[None, :]
    out_2d = _lomb_scargle_solve_shared_mask(
        data_2d, mask, basis, rcond,
    )
    if leading_shape:
        return out_2d.reshape(data.shape)
    return out_2d[0]
