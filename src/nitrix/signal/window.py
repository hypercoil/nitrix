# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Windowed-sample extraction.

Sample fixed-size windows from a tensor along a specified axis,
with optional overlap.  Used for time-series augmentation (random
crops, multi-window bootstrap), per-window covariance estimation,
and sliding-window analysis.

The single public symbol is ``sample_windows``; behaviour is
parameterised by two flags:

- ``allow_overlap``: ``True`` -- sample starts uniformly from
  ``[0, n - window_size]`` *without replacement* (so window
  count is exact).  ``False`` (default) -- starts are
  constrained to produce non-overlapping windows; the
  non-overlap is enforced by distributing the "unused" indices
  among ``num_windows + 1`` gaps via a multinomial.
- ``create_new_axis``: ``True`` -- stack windows along a freshly
  inserted axis (``multiplying_axis``).  ``False`` (default) --
  concatenate along an existing axis.

What changed from ``nitrix.functional.window``:

- The four-function factory (``sample_windows(allow_overlap=...,
  create_new_axis=...)`` returning a callable) collapsed into a
  single function with two flags.  Caller writes
  ``sample_windows(tensor, ...)`` directly, not
  ``sample_windows(...)(tensor, ...)``.
- Internal ``_select_fn`` / ``_slice_fn`` dispatch removed in
  favour of a single body that branches on the flags.
- The numpyro-era ``Multinomial`` distribution dropped in favour
  of ``jax.random.multinomial`` (already done in the prior
  numpyro-strip pass; this rewrite preserves it).
"""
from __future__ import annotations

from typing import Any, Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array, Num


__all__ = ['sample_windows']


def _select_overlapping_starts(
    input_size: int,
    window_size: int,
    num_windows: int,
    key: jax.Array,
) -> Array:
    '''Pick ``num_windows`` distinct starts uniformly from the valid range.'''
    return jax.random.choice(
        key,
        a=(input_size - window_size + 1),
        shape=(num_windows,),
        replace=False,
    )


def _select_nonoverlapping_starts(
    input_size: int,
    window_size: int,
    num_windows: int,
    key: jax.Array,
) -> Array:
    '''Choose non-overlapping window starts.

    Distribute the ``unused = input_size - num_windows * window_size``
    free slots across the ``num_windows + 1`` gaps via a uniform
    multinomial draw; the cumulative gap sizes plus window-offsets
    give the starts.
    '''
    unused_size = input_size - window_size * num_windows
    if unused_size < 0:
        raise ValueError(
            f'sample_windows: cannot fit {num_windows} non-overlapping '
            f'windows of size {window_size} into input of size '
            f'{input_size} (would need at least '
            f'{num_windows * window_size}).'
        )
    probs = jnp.full(num_windows + 1, 1.0 / (num_windows + 1))
    intervals = jax.random.multinomial(key, unused_size, probs).astype(jnp.int32)
    starts = jnp.arange(num_windows + 1) * window_size + jnp.cumsum(intervals)
    return starts[:-1]


def sample_windows(
    tensor: Num[Array, '...'],
    *,
    window_size: int,
    num_windows: int = 1,
    allow_overlap: bool = False,
    create_new_axis: bool = False,
    windowing_axis: int = -1,
    multiplying_axis: int = 0,
    key: jax.Array,
) -> Num[Array, '...']:
    '''Sample fixed-size windows from a tensor along one axis.

    Parameters
    ----------
    tensor
        Input from which to draw windows.
    window_size
        Length of each window along ``windowing_axis``.
    num_windows
        Number of windows to sample.  Default ``1``.
    allow_overlap
        ``True`` -- windows may overlap (random distinct starts).
        ``False`` (default) -- windows are guaranteed non-
        overlapping; the spacing is randomised via a multinomial
        over the unused indices.
    create_new_axis
        ``True`` -- stack windows along a freshly-inserted axis
        (``multiplying_axis`` is interpreted in the output rank).
        ``False`` (default) -- concatenate windows along an
        existing axis (``multiplying_axis`` indexes the input
        rank).
    windowing_axis
        Axis along which to sample windows.  Default ``-1``.
    multiplying_axis
        Axis along which to concatenate / insert the windows.
        Default ``0``.
    key
        PRNG key for the random window starts.  Required (no
        implicit ``key=0`` default -- caller is responsible for
        randomness).

    Returns
    -------
    Tensor of windows.  Shape depends on ``create_new_axis``:

    - ``False``: ``windowing_axis`` shrinks to ``window_size``;
      ``multiplying_axis`` grows by ``num_windows``.
    - ``True``: ``windowing_axis`` shrinks to ``window_size``;
      a new axis of length ``num_windows`` is inserted at
      ``multiplying_axis`` in the output rank.

    Notes
    -----
    The window starts depend on ``key`` but the slicing itself is
    deterministic; gradients flow through the sliced values but
    not through the random key.  For reproducibility, pin
    ``key``.
    '''
    tensor = jnp.asarray(tensor)
    windowing_axis = windowing_axis % tensor.ndim
    input_size = tensor.shape[windowing_axis]

    if allow_overlap:
        starts = _select_overlapping_starts(
            input_size, window_size, num_windows, key,
        )
    else:
        starts = _select_nonoverlapping_starts(
            input_size, window_size, num_windows, key,
        )

    sizes = tuple(
        window_size if i == windowing_axis else s
        for i, s in enumerate(tensor.shape)
    )
    base_slc: list[Any] = [0] * tensor.ndim

    def slice_one(start_val: jax.Array) -> Array:
        slc = base_slc.copy()
        slc[windowing_axis] = start_val
        w = jax.lax.dynamic_slice(tensor, tuple(slc), sizes)
        if create_new_axis:
            w = jnp.expand_dims(w, axis=multiplying_axis)
        return w

    windows = tuple(slice_one(starts[i]) for i in range(num_windows))
    return jnp.concatenate(windows, axis=multiplying_axis)
