# -*- coding: utf-8 -*-
"""Shared timing utilities for nitrix benchmark scripts.

The single load-bearing rule: **first calls don't count**.  Both JAX
and Pallas trace + compile on first use for a new ``(shape, dtype,
algebra)`` signature; including that in the measured budget would make
the JAX path look fast (XLA caches aggressively) and the Pallas path
look broken (Triton compilation is slow on cold cache).

``bench_call`` drops a fixed number of warm-up runs and then reports
the median of the remaining timed runs.  Wall-time is measured with
``time.perf_counter`` around ``block_until_ready`` on the result.

For the same reason, every bench script must ``jit``-compile the
callable it wraps -- otherwise the timed loop re-traces every call
and the warm-up is moot.  ``timed_jit`` is the convenience helper.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import jax
import numpy as np


@dataclass(frozen=True)
class BenchSample:
    """A single timed measurement.

    ``compile_s`` is the wall-time of the first call (compile +
    execute).  ``warm_s`` is the median of the post-warm-up runs --
    the number we actually report.  ``samples_s`` lists every timed
    sample in the order taken so callers can compute their own
    statistics if they want.
    """

    compile_s: float
    warm_s: float
    samples_s: tuple[float, ...]


def bench_call(
    fn: Callable[..., Any],
    *args,
    warmup: int = 3,
    repeats: int = 10,
) -> BenchSample:
    """Time ``fn(*args)`` with warm-up.

    Parameters
    ----------
    fn
        Callable; expected to be ``jax.jit``-wrapped already (see
        ``timed_jit``).
    *args
        Arguments to pass on each call.  These should be jax arrays
        already on-device so the timing isn't polluted by host->device
        transfer.
    warmup
        Number of un-timed warm-up runs.  Must be >= 1 so that the
        first-call compile cost is excluded.
    repeats
        Number of timed runs after warm-up; the *median* of these is
        reported as ``warm_s``.

    Returns
    -------
    A ``BenchSample`` recording both the first-call cost (compile +
    execute) and the steady-state median.
    """
    if warmup < 1:
        raise ValueError('warmup must be >= 1 to exclude compile time')
    # First call -- compile + execute.
    t0 = time.perf_counter()
    out = fn(*args)
    if hasattr(out, 'block_until_ready'):
        out.block_until_ready()
    compile_s = time.perf_counter() - t0
    # Remaining warm-up calls (steady-state, but discarded).
    for _ in range(warmup - 1):
        out = fn(*args)
        if hasattr(out, 'block_until_ready'):
            out.block_until_ready()
    # Timed runs.
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn(*args)
        if hasattr(out, 'block_until_ready'):
            out.block_until_ready()
        samples.append(time.perf_counter() - t0)
    return BenchSample(
        compile_s=compile_s,
        warm_s=float(np.median(samples)),
        samples_s=tuple(samples),
    )


def timed_jit(fn: Callable[..., Any], **jit_kwargs) -> Callable[..., Any]:
    """Wrap ``fn`` in ``jax.jit`` to get steady-state timing semantics.

    Use::

        f = timed_jit(lambda A, B: semiring_matmul(A, B, semiring=LOG, backend='jax'))
        sample = bench_call(f, A, B)
    """
    return jax.jit(fn, **jit_kwargs)


def host_summary() -> dict[str, str]:
    """Snapshot of the host configuration for the bench report."""
    import platform

    try:
        d = jax.devices()[0]
        device = f'{d.device_kind} ({d.platform})'
    except Exception:  # pragma: no cover
        device = 'unknown'
    return {
        'device': device,
        'platform': platform.platform(),
        'jax_version': jax.__version__,
    }


def format_us(seconds: float) -> str:
    if seconds < 1e-3:
        return f'{seconds * 1e6:.1f} µs'
    if seconds < 1.0:
        return f'{seconds * 1e3:.2f} ms'
    return f'{seconds:.3f} s'
