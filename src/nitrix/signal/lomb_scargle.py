# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Lomb-Scargle spectral methods for irregularly-sampled time series.

Two distinct primitives that share the underlying sine / cosine
basis but solve different problems:

- :func:`lomb_scargle_periodogram` -- canonical Press-Rybicki /
  Scargle periodogram for **spectral analysis** of
  irregularly-sampled data.  Each (cosine, sine) amplitude pair is
  estimated independently per trial frequency after the
  :math:`\\tau`-orthogonalisation that makes the per-frequency
  least-squares closed-form.  Fast; the *reconstruction* implied
  by summing the per-frequency components does **not** pass through
  the observed samples exactly -- if used for interpolation, it
  produces **visible boundary discontinuities** at observed /
  censored transitions.  This is the wrong primitive for
  interpolation.

- :func:`lomb_scargle_interpolate` -- **interpolation** primitive
  for fMRI motion-censoring (the Power 2014 protocol).  Solves a
  **joint** least-squares regression of the observed samples on the
  entire sine / cosine basis (plus DC),
  :math:`\\min \\lVert M (B \\beta - y) \\rVert^2`, where :math:`M`
  is the observed-sample mask, :math:`B` is the basis, :math:`y` is
  the data, and :math:`\\beta` are the spectral coefficients.  The
  joint fit guarantees :math:`B[\\mathrm{obs}] \\, \\beta =
  y[\\mathrm{obs}]` (modulo rank deficiency handled by a Tikhonov
  ridge), so the spliced output has **no boundary discontinuity**.

Memory regime
-------------

The dominant cost is the basis matrix :math:`B` of shape
``(n_obs, K)`` where :math:`K = 1 + 2 \\, n_{\\mathrm{freq}}`.  At a
typical fMRI ``n_obs = 500`` and ``K = 499``, :math:`B` is ~1 MB at
fp32 -- shared across channels.

For **shared-mask** input (the mask is broadcast-compatible with
the data along the leading dims; the typical fMRI case where one
motion-censor mask applies to all voxels), we compute the Gram
matrix :math:`G = B^{\\top} \\operatorname{diag}(M) B` of shape
``(K, K)`` and factor it **once** via a symmetric
eigendecomposition (``safe_eigh``), then apply a threshold-
truncated pseudo-inverse (eigenvalues below ``rcond * max`` are
dropped) to solve the right-hand sides for all channels at once.
The eigendecomposition-plus-pseudo-inverse path -- rather than a
Cholesky factor plus triangular solve -- is deliberate: the masked
Gram is rank-deficient whenever
:math:`2 \\, n_{\\mathrm{freq}} + 1 > n_{\\mathrm{valid}}`, and a
singular Gram has no Cholesky factor; the truncated pseudo-inverse
absorbs the rank deficiency without an arbitrary ridge.  Total
memory at :math:`V = 10^6` voxels, :math:`T = 500`, :math:`K = 499`:

- Shared basis: :math:`T \\, K \\times 4 = 1` MB.
- Shared Gram / eigenvectors: :math:`2 K^2 \\times 4 = 2` MB.
- Per-channel right-hand side / :math:`\\beta` / reconstruction:
  :math:`3 V K \\times 4 = 6` GB.
- Output / data: :math:`2 V T \\times 4 = 4` GB.

Total ~10 GB -- fits on an A100 80 GB or H100 with room.  For
**per-channel-mask** input (the mask shape matches the data shape;
each channel has its own censoring pattern), a per-channel solve
would create a per-channel Gram of size ``(V, K, K)`` -- 1 TB at
:math:`V = 10^6`.  In that case the function raises a clear error
pointing at the shared-mask path.

References
----------
.. [1] Lomb, N. R. (1976).  Least-squares frequency analysis of
   unequally spaced data.  Astrophysics and Space Science, 39,
   447-462.  https://doi.org/10.1007/BF00648343
.. [2] Scargle, J. D. (1982).  Studies in astronomical time series
   analysis. II.  Statistical aspects of spectral analysis of
   unevenly spaced data.  The Astrophysical Journal, 263, 835-853.
   https://doi.org/10.1086/160554
.. [3] Press, W. H. & Rybicki, G. B. (1989).  Fast algorithm for
   spectral analysis of unevenly sampled data.  The Astrophysical
   Journal, 338, 277-280.  https://doi.org/10.1086/167197
.. [4] Power, J. D., Mitra, A., Laumann, T. O., Snyder, A. Z.,
   Schlaggar, B. L., & Petersen, S. E. (2014).  Methods to detect,
   characterize, and remove motion artifact in resting state fMRI.
   NeuroImage, 84, 320-341.
   https://doi.org/10.1016/j.neuroimage.2013.08.048
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
    """Choose trial angular frequencies (Press-Rybicki convention).

    The grid runs from a minimum angular frequency
    :math:`2 \\pi \\, df`, with :math:`df = 1 / (T \\cdot
    \\mathrm{oversampling})`, up to a maximum :math:`2 \\pi \\,
    f_{\\max}`, with :math:`f_{\\max} = \\mathrm{high\\_factor}
    \\cdot f_{\\mathrm{Nyquist}}`.  The frequency count is bounded so
    that :math:`2 \\, n_{\\mathrm{freq}} + 1 \\le n_{\\mathrm{obs}}
    (1 - \\mathrm{censoring\\_budget})` -- i.e. up to
    :math:`\\mathrm{censoring\\_budget} \\cdot n_{\\mathrm{obs}}`
    frames are assumed censorable, leaving the basis with enough
    column slack to stay rank-sufficient.

    Parameters
    ----------
    n_obs
        Number of time samples.
    dt
        Sampling interval, so the record length is
        :math:`T = n_{\\mathrm{obs}} \\cdot dt`.
    oversampling
        Frequency-grid oversampling factor; larger values give a
        finer frequency spacing :math:`df`.
    high_factor
        Multiplier of the Nyquist frequency setting the highest
        trial frequency.
    dtype
        Floating dtype of the returned array.
    censoring_budget
        Assumed maximum fraction of frames that may be censored,
        used to cap the frequency count so the basis stays
        rank-sufficient under worst-case censoring.  Default
        ``0.4`` (40% censoring tolerated without rank deficiency);
        raise the budget or pass a smaller ``oversampling`` if you
        expect heavier censoring.

    Returns
    -------
    Float[Array, 'n_freq']
        The trial angular frequencies :math:`\\omega`, in ascending
        order.
    """
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
    """Build the joint regression basis :math:`[\\mathrm{DC} \\mid
    \\cos \\mid \\sin]`.

    Stacks a constant (DC) column, the cosine columns
    :math:`\\cos(\\omega_k t)`, and the sine columns
    :math:`\\sin(\\omega_k t)` for every trial frequency, evaluated
    on the regular time grid :math:`t = dt \\cdot [0, 1, \\dots,
    n_{\\mathrm{obs}} - 1]`.  Shared across channels.

    Parameters
    ----------
    n_obs
        Number of time samples (rows).
    dt
        Sampling interval defining the time grid.
    omega
        Trial angular frequencies, shape ``(n_freq,)``.
    dtype
        Floating dtype of the returned matrix.

    Returns
    -------
    Float[Array, 'n_obs n_basis']
        Basis matrix of shape ``(n_obs, 1 + 2 * n_freq)``; rows
        index time, columns index basis component (DC first, then
        cosines, then sines).
    """
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
    """Normalised Lomb-Scargle periodogram (Scargle convention).

    Computes per-frequency power from the observed (non-masked)
    samples, normalised by the observed-sample variance -- the
    classic Scargle convention :math:`P = P_{\\mathrm{raw}} /
    \\operatorname{var}`.  Equals
    ``scipy.signal.lombscargle(..., normalize=False)`` divided by
    the observed-sample variance (population variance,
    ``ddof=0``).

    For **interpolation** use :func:`lomb_scargle_interpolate` --
    the reconstruction implied by these per-frequency least-squares
    coefficients does not pass through the observed samples exactly.

    Parameters
    ----------
    data
        Observation tensor of shape ``(..., obs)``; the last axis
        is time and leading axes index channels.
    mask
        Boolean validity mask of the same shape as ``data``; ``True``
        marks observed (non-censored) samples.
    dt
        Sampling interval.  Default ``1.0``.
    oversampling
        Frequency-grid oversampling factor.  Default ``4.0``
        (Press-Rybicki convention).
    high_factor
        Multiplier of the Nyquist frequency for the highest trial
        frequency.  Default ``1.0``.

    Returns
    -------
    freqs : Float[Array, 'n_freq']
        Cyclic (not angular) trial frequencies, in ascending order.
    power : Float[Array, '... n_freq']
        Per-channel normalised power at each trial frequency, with
        the same leading (channel) axes as ``data``.

    Notes
    -----
    This is **not** ``scipy.signal.lombscargle(normalize=True)``.
    scipy's ``normalize=True`` (as of 1.17.1) returns
    :math:`2 P_{\\mathrm{raw}} / (N \\operatorname{var})`, so it
    differs from this output by a constant factor of :math:`N / 2`,
    where :math:`N` is the number of observed samples.  scipy's
    ``normalize`` convention has drifted across versions, so the
    expression above -- not a scipy flag name -- is the stable
    definition to compare against.
    """
    n_obs = data.shape[-1]
    # Periodogram doesn't have a Gram-rank concern (per-frequency
    # univariates), so use a near-zero censoring budget to expose
    # the full Press-Rybicki frequency grid up to Nyquist.
    omega = _trial_frequencies(
        n_obs,
        dt,
        oversampling,
        high_factor,
        data.dtype,
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
        return 0.5 * (cy**2 / c_norm + sy**2 / s_norm) / var_y

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
    """Joint least-squares interpolation with a shared mask.

    Solves the masked joint regression for a single
    channel-invariant mask via a robust factorisation: form the
    Gram matrix, factor it with a symmetric eigendecomposition, and
    apply a thresholded pseudo-inverse.  Truncating eigenvalues
    below :math:`\\mathrm{rcond} \\cdot \\max(\\mathrm{eigval})`
    handles rank-deficient Gram matrices (which arise whenever
    :math:`2 \\, n_{\\mathrm{freq}} + 1 > n_{\\mathrm{valid}}`)
    without arbitrary ridge parameters.  Because the mask is shared,
    the Gram and its eigendecomposition are computed **once** and
    reused across every channel; there is no ``(n_chan, K, K)``
    intermediate.  The mask-weighted basis :math:`B_w`, Gram
    :math:`G`, and eigenvectors are all ``(K, K)`` or smaller
    (~1 MB at fMRI scale), while the right-hand sides and
    reconstruction scale only linearly in the channel count.

    Parameters
    ----------
    data
        Time series of shape ``(n_chan, n_obs)``; rows index
        channels, columns index time.
    mask
        Shared boolean validity mask of shape ``(n_obs,)``; ``True``
        marks observed samples that the reconstruction is pinned to.
    basis
        Joint regression basis of shape ``(n_obs, K)`` with
        ``K = 1 + 2 * n_freq``, as returned by
        :func:`_lomb_scargle_basis`.
    rcond
        Relative eigenvalue-truncation threshold for the
        pseudo-inverse; eigenvalues below ``rcond * max(eigval)``
        are dropped from the solve.

    Returns
    -------
    Float[Array, 'n_chan n_obs']
        Filled time series of shape ``(n_chan, n_obs)``: observed
        frames are the original data, censored frames are the joint
        spectral reconstruction.
    """
    mf = mask.astype(data.dtype)
    B_w = basis * mf[:, None]  # (n_obs, K)
    G = B_w.T @ basis  # (K, K), symmetric PSD
    # Threshold-truncated pseudo-inverse via eigh.
    ev, V = safe_eigh(G)  # ev ascending
    # Truncate eigenvalues below the relative threshold.  The
    # max eigval is at the end; ``rcond * max`` is the floor.
    thresh = rcond * ev[-1]
    ev_inv = jnp.where(ev > thresh, 1.0 / ev, 0.0)
    rhs = data @ B_w  # (n_chan, K)
    # betas = V @ diag(ev_inv) @ V.T @ rhs.T
    z = V.T @ rhs.swapaxes(-1, -2)  # (K, n_chan)
    z = ev_inv[:, None] * z
    betas = V @ z  # (K, n_chan)
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
    """Fill censored time-series frames by joint Lomb-Scargle
    spectral reconstruction.

    Solves a masked least-squares regression of the observed
    samples on a :math:`[\\mathrm{DC},\\ \\cos(\\omega_k t),\\
    \\sin(\\omega_k t)]` basis -- **joint** over all trial
    frequencies, not the independent per-frequency form of
    :func:`lomb_scargle_periodogram`.  The joint fit ensures the
    reconstruction passes through the observed samples (modulo a
    small Tikhonov ridge), so the spliced output has **no boundary
    discontinuity** between observed and interpolated frames.

    This is the Power 2014 protocol for motion-censored fMRI.
    Recommended over :func:`linear_interpolate` because:

    1. It preserves the spectral content -- linear interpolation
       injects low-frequency artefacts at bridges across long
       censored runs.
    2. The reconstruction is smooth at observed boundaries.

    Memory: shared-mask path uses ``(K, K)`` Gram / eigenvector
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
        whose memory scales as :math:`V K^2`, which runs out of
        memory at fMRI scale.  Use :func:`linear_interpolate` for
        per-channel masks, or explicitly ``jax.vmap`` this function
        over the channel
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
    Num[Array, '... obs']
        Filled time series of the same shape as ``data``: observed
        frames are returned unchanged, censored frames are replaced
        by the joint spectral reconstruction.

    Raises
    ------
    ValueError
        If ``mask`` is not broadcast-compatible with ``data`` (or
        is fully same-shape, which we deliberately reject for
        memory safety).

    Notes
    -----
    Differentiable through ``data``; ``mask`` is static.

    **Intended use: a spectral bridge, not durable imputation.**
    The Power 2014 procedure exists to produce a
    *spectrally-consistent* fill so that downstream autoregressive /
    IIR temporal filters (band-pass, high-pass) run across the gaps
    without ringing or spectral leakage; the censored frames are
    typically dropped again after filtering.  Treat the filled
    values as a transient means to that end, **not** as durable
    per-frame imputations for direct analysis.  Only two things are
    well-defined: the observed-frame splice-through
    (``recon[obs] == data[obs]`` exactly) and the band-limited
    spectral content.  The reconstruction *at censored frames* is
    the regularised solution of an ill-conditioned masked Gram
    (condition number :math:`\\sim 10^{32}`); which near-null-space
    modes survive the
    ``rcond``-truncated pseudo-inverse differs between fp32 and
    fp64, so the censored-frame values are **not bit-reproducible
    across precisions** (two valid regularised solutions can differ
    by O(1) on O(1) signals at the worst frame).  This sensitivity
    lives in high-frequency directions the irregular observed grid
    cannot pin down -- benign for the intended low/band-pass use,
    but a correctness trap if the fills are used durably.  If you
    need durable per-frame values, compute in fp64 and treat the
    high-frequency content as undetermined.

    **Device placement.**  The shared-Gram factorisation uses
    ``safe_eigh`` (``nitrix.linalg._solver``), which routes the
    eigendecomposition to a device where dense cuSolver ``eigh``
    works.  On stacks where GPU ``eigh`` is broken (some
    CUDA/driver combinations -- ``gpusolverDnCreate failed``), the
    Gram solve is placed on the **host (CPU)** with GPU->CPU->GPU
    transfers: results are correct but the solve is not
    GPU-resident, and the K x K Gram (K up to ``2*n_freq+1`` ~ 499
    at fMRI ``n_obs``) is exactly in the affected size range.
    Contrast the matrix-function ops (``symlog`` / ``symsqrt`` /
    ``sympower``), which consume a raw ``eigh`` that XLA lowers off
    cuSolver and so stay GPU-resident.

    References
    ----------
    .. [1] Power, J. D., Mitra, A., Laumann, T. O., Snyder, A. Z.,
       Schlaggar, B. L., & Petersen, S. E. (2014).  Methods to
       detect, characterize, and remove motion artifact in resting
       state fMRI.  NeuroImage, 84, 320-341.
       https://doi.org/10.1016/j.neuroimage.2013.08.048
    """
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
        n_obs,
        dt,
        oversampling,
        high_factor,
        data.dtype,
        censoring_budget=censoring_budget,
    )
    basis = _lomb_scargle_basis(n_obs, dt, omega, data.dtype)

    # Flatten leading dims into a single channel batch axis.
    leading_shape = data.shape[:-1]
    data_2d = data.reshape((-1, n_obs)) if leading_shape else data[None, :]
    out_2d = _lomb_scargle_solve_shared_mask(
        data_2d,
        mask,
        basis,
        rcond,
    )
    if leading_shape:
        return out_2d.reshape(data.shape)
    return out_2d[0]
