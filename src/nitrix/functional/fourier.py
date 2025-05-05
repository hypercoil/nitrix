# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Signal transformations in the frequency domain.
"""

import math
from typing import Optional, Tuple

import jax.numpy as jnp

from .._internal import (
    DocTemplateFormat,
    Tensor,
    form_docstring,
    orient_and_conform,
    tensor_dimensions,
)


@form_docstring
def document_frequency_filter() -> DocTemplateFormat:
    freqfilter_param_spec = r"""
    Parameters
    ----------
    X : ($N$, $*$, $obs$) tensor
        The (potentially multivariate) 1D signal to be filtered. The final
        axis should correspond to the time domain of each signal or its
        analogue.
    weight : ($*$, $\left\lfloor \frac{obs}{2} \right\rfloor + 1$) tensor
        The filter gain at each frequency bin in the spectrum, ordered low to
        high along the last axis. Dimensions before the last can be used to
        apply different filters to different variables in the input signal
        according to tensor broadcasting rules.
    **params
        Any additional parameters provided will be passed to ``jnp.fft.rfft``
        and ``jnp.fft.irfft``."""
    freqfilter_return_spec = r"""
    Returns
    -------
    ($N$, $*$, $2 \left\lfloor \frac{obs}{2} \right\rfloor$) tensor
        Original 1D signal filtered via multiplication in the frequency
        domain."""
    desc_obs = (
        'Number of observations per data channel. For example, if `X` '
        'is a time series tensor, then $obs$ could be the number of time '
        'points.'
    )
    freqfilter_dim_spec = """
    | Dim | Description | Notes |
    |-----|-------------|-------|
    | $N$ | Batch size  | Optional |
    |$obs$| {desc_obs} ||
    |$*$  | Any number of intervening dimensions ||
    """.format(desc_obs=desc_obs)
    freqfilter_dim_spec = tensor_dimensions(freqfilter_dim_spec)
    return DocTemplateFormat(
        freqfilter_dim_spec=freqfilter_dim_spec,
        freqfilter_param_spec=freqfilter_param_spec,
        freqfilter_return_spec=freqfilter_return_spec,
    )


@form_docstring
def document_analytic_signal() -> DocTemplateFormat:
    analytic_signal_base_spec = """
    X : ($*$, $obs$, $?$) tensor
        Input tensor; the original real-valued signal.
    axis : int
        Axis along which the transform is applied.
    n : int
        Number of frequency components; dimension of the Fourier transform.
        This defaults to the size of the input along the transform axis."""
    analytic_signal_sampling_freq = """
    fs : float
        Sampling frequency."""
    analytic_signal_period = """
    period : float
        Range over which the signal wraps. (See ``jax.numpy.unwrap``.)"""
    analytic_signal_return_spec = r"""
    Returns
    -------
    ($*$, {ax_shape}, $?$) tensor
        Output tensor containing the {return_signal}.
    """
    ax_shape = r'$2 \left\lfloor \frac{{n}}{{2}} \right\rfloor$'
    ax_shape_inst_freq = r'$2 \left\lfloor \frac{{n}}{{2}} \right\rfloor - 1$'
    hilbert_transform_return_spec = analytic_signal_return_spec.format(
        return_signal='Hilbert transform',
        ax_shape=ax_shape,
    )
    inst_freq_return_spec = analytic_signal_return_spec.format(
        return_signal='instantaneous frequency',
        ax_shape=ax_shape_inst_freq,
    )
    inst_phase_return_spec = analytic_signal_return_spec.format(
        return_signal='instantaneous phase',
        ax_shape=ax_shape,
    )
    envelope_return_spec = analytic_signal_return_spec.format(
        return_signal='envelope',
        ax_shape=ax_shape,
    )
    analytic_signal_return_spec = analytic_signal_return_spec.format(
        return_signal='analytic signal',
        ax_shape=ax_shape,
    )
    desc_obs = (
        'Number of observations per input signal. For example, if `X` '
        'is a time series tensor, then $obs$ could be the number of time '
        'points.'
    )
    desc_n = (
        'Number of frequency components in the Fourier transform. Can be '
        'configured explicitly using the `n` parameter.'
    )
    analytic_signal_dim_spec = """
    | Dim | Description | Notes |
    |-----|-------------|-------|
    |$obs$| {desc_obs} ||
    | $n$ | {desc_n} | Equal to $obs$ unless otherwise specified |
    | $*$ | Any number of preceding dimensions ||
    | $?$ | Any number of trailing dimensions ||
    """.format(desc_obs=desc_obs, desc_n=desc_n)
    analytic_signal_dim_spec = tensor_dimensions(analytic_signal_dim_spec)
    analytic_signal_see_also = """
    See also
    --------
    [analytic_signal](nitrix.functional.fourier.analytic_signal.html)
        Analytic signal
    [hilbert_transform](nitrix.functional.fourier.hilbert_transform.html)
        Hilbert transform
    [envelope](nitrix.functional.fourier.envelope.html)
        Envelope of a signal
    [instantaneous_phase](nitrix.functional.fourier.instantaneous_phase.html)
        Instantaneous phase of a signal
    [instantaneous_frequency](nitrix.functional.fourier.instantaneous_frequency.html)
        Instantaneous frequency of a signal
    [env_inst](nitrix.functional.fourier.env_inst.html)
        Envelope, instantaneous phase, and instantaneous frequency of a signal"""
    return DocTemplateFormat(
        analytic_signal_dim_spec=analytic_signal_dim_spec,
        analytic_signal_base_spec=analytic_signal_base_spec,
        analytic_signal_sampling_freq=analytic_signal_sampling_freq,
        analytic_signal_period=analytic_signal_period,
        analytic_signal_return_spec=analytic_signal_return_spec,
        hilbert_transform_return_spec=hilbert_transform_return_spec,
        inst_freq_return_spec=inst_freq_return_spec,
        inst_phase_return_spec=inst_phase_return_spec,
        envelope_return_spec=envelope_return_spec,
        analytic_signal_see_also=analytic_signal_see_also,
    )


@document_frequency_filter
def product_filter(
    X: Tensor,
    weight: Tensor,
    **params,
) -> Tensor:
    """
    Convolve a multivariate signal via multiplication in the frequency domain.

    :::{{.callout-note}}
    For a filter that is guaranteed to be zero-phase even when the weight
    tensor is not strictly real-valued, use
    [`product_filtfilt`](nitrix.functional.fourier.product_filtfilt.qmd).
    :::
    \
    {freqfilter_dim_spec}
    \
    {freqfilter_param_spec}
    \
    {freqfilter_return_spec}
    """
    n = X.shape[-1]
    Xf = jnp.fft.rfft(X, n=n, **params)
    Xf_filt = weight * Xf
    return jnp.fft.irfft(Xf_filt, n=n, **params)


@document_frequency_filter
def product_filtfilt(
    X: Tensor,
    weight: Tensor,
    **params,
) -> Tensor:
    """
    Perform zero-phase digital filtering of a signal via multiplication in the
    frequency domain.

    This function operates by first filtering the data and then filtering a
    time-reversed copy of the filtered data again. Note that the effect on the
    amplitude is quadratic in the filter weight.

    :::{{.callout-note}}
    If the ``weight`` argument is strictly real, then the filter has no
    phase delay component and it could make sense to simply use
    [`product_filter`](nitrix.functional.fourier.product_filter.qmd) depending
    on the context.
    :::
    \
    {freqfilter_dim_spec}
    \
    {freqfilter_param_spec}
    \
    {freqfilter_return_spec}
    """
    X_filt = product_filter(X, weight, **params)
    out = product_filter(jnp.flip(X_filt, -1), weight, **params)
    return jnp.flip(out, -1)


@document_analytic_signal
def analytic_signal(
    X: Tensor,
    axis: int = -1,
    n: Optional[int] = None,
) -> Tensor:
    """
    Compute the analytic signal.

    The analytic signal is a helical representation of the input signal in
    the complex plane. Its real and imaginary parts are related by the
    Hilbert transform.  Its properties can be used to quickly derive measures
    such as a signal envelope and instantaneous measures of frequency and
    phase.
    \
    {analytic_signal_dim_spec}

    Parameters
    ----------\
    {analytic_signal_base_spec}
    \
    {analytic_signal_return_spec}
    \
    {analytic_signal_see_also}
    """
    if jnp.iscomplexobj(X):
        raise ValueError('Input for analytic signal must be strictly real')

    Xf = jnp.fft.fft(X, n=n, axis=axis)
    n = n or X.shape[axis]
    h = jnp.zeros(n)

    # TODO: don't like this assignment implementation or the conditionals
    if n % 2 == 0:
        h = h.at[0].set(1)
        h = h.at[n // 2].set(1)
        h = h.at[1 : (n // 2)].set(2)
    else:
        h = h.at[0].set(1)
        h = h.at[1 : ((n + 1) // 2)].set(2)

    h = orient_and_conform(h, axis=axis, reference=Xf)
    return jnp.fft.ifft(Xf * h, axis=axis)


@document_analytic_signal
def hilbert_transform(
    X: Tensor,
    axis: int = -1,
    n: Optional[int] = None,
) -> Tensor:
    """
    Hilbert transform of an input signal.
    \
    {analytic_signal_dim_spec}

    Parameters
    ----------\
    {analytic_signal_base_spec}
    \
    {hilbert_transform_return_spec}
    \
    {analytic_signal_see_also}
    """
    return analytic_signal(X=X, axis=axis, n=n).imag


@document_analytic_signal
def envelope(
    X: Tensor,
    axis: int = -1,
    n: Optional[int] = None,
) -> Tensor:
    """
    Envelope of a signal, computed via the analytic signal.

    :::{{.callout-note}}
    If you require the instantaneous phase or frequency in addition to the
    envelope, [env_inst](nitrix.functional.fourier.env_inst.qmd) will be more
    efficient.
    :::
    \
    {analytic_signal_dim_spec}

    Parameters
    ----------\
    {analytic_signal_base_spec}
    \
    {envelope_return_spec}
    \
    {analytic_signal_see_also}
    """
    return jnp.abs(analytic_signal(X=X, axis=axis, n=n))


@document_analytic_signal
def instantaneous_phase(
    X: Tensor,
    axis: int = -1,
    n: Optional[int] = None,
    period: float = (2 * math.pi),
) -> Tensor:
    """
    Instantaneous phase of a signal, computed via the analytic signal.

    :::{{.callout-note}}
    If you require the envelope or instantaneous frequency in addition to the
    instantaneous phase, [env_inst](nitrix.functional.fourier.env_inst.qmd)
    will be more efficient.
    :::
    \
    {analytic_signal_dim_spec}

    Parameters
    ----------\
    {analytic_signal_base_spec}\
    {analytic_signal_period}\
    \
    {inst_phase_return_spec}
    \
    {analytic_signal_see_also}
    """
    return jnp.unwrap(
        jnp.angle(analytic_signal(X=X, axis=axis, n=n)),
        axis=axis,
        period=period,
    )


@document_analytic_signal
def instantaneous_frequency(
    X: Tensor,
    axis: int = -1,
    n: Optional[int] = None,
    fs: float = 1,
    period: float = (2 * math.pi),
) -> Tensor:
    """
    Instantaneous frequency of a signal, computed via the analytic signal.

    :::{{.callout-note}}
    If you require the envelope or instantaneous phase in addition to the
    instantaneous frequency,
    [env_inst](nitrix.functional.fourier.env_inst.qmd) will be more efficient.
    :::
    \
    {analytic_signal_dim_spec}

    Parameters
    ----------\
    {analytic_signal_base_spec}\
    {analytic_signal_period}\
    {analytic_signal_sampling_frequency}
    \
    {inst_freq_return_spec}
    \
    {analytic_signal_see_also}
    """
    inst_phase = instantaneous_phase(X=X, axis=axis, n=n, period=period)
    return fs * jnp.diff(inst_phase, axis=axis) / period


@document_analytic_signal
def env_inst(
    X: Tensor,
    axis: int = -1,
    n: Optional[int] = None,
    fs: float = 1,
    period: float = (2 * math.pi),
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Compute the analytic signal, and then decompose it into the envelope and
    instantaneous phase and frequency.

    :::{{.callout-note}}
    If you only require two out of the three outputs, the XLA compiler should
    optimise the computation to avoid unnecessary work if the function is
    JIT-compiled. If you require only a single output, it is more efficient to
    call the corresponding function directly.
    :::
    \
    {analytic_signal_dim_spec}

    Parameters
    ----------\
    {analytic_signal_base_spec}\
    {analytic_signal_period}\
    {analytic_signal_sampling_frequency}

    Returns
    -------
    ($*$, $2 \\left\\lfloor \frac{{n}}{{2}} \right\rfloor$, $?$) tensor
        Envelope of the analytic signal.
    ($*$, $2 \\left\\lfloor \frac{{n}}{{2}} \right\rfloor$, $?$) tensor
        Instantaneous frequency of the analytic signal.
    ($*$, $2 \\left\\lfloor \frac{{n}}{{2}} \right\rfloor$, $?$) tensor
        Instantaneous phase of the analytic signal.
    \
    {analytic_signal_see_also}
    """
    Xa = analytic_signal(X=X, axis=axis, n=n)
    env = jnp.abs(Xa)
    inst_phase = jnp.unwrap(jnp.angle(Xa), axis=axis, period=period)
    inst_freq = fs * jnp.diff(inst_phase, axis=axis) / period
    return env, inst_freq, inst_phase
