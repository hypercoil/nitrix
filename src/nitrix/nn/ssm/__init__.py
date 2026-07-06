# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
# isort: skip_file
"""
Selective state-space scan (Mamba / S6).

This subpackage exposes :func:`selective_scan`, the discretised selective
recurrence at the heart of the Mamba / S6 state-space model, with three-level
backend selection.  The ``'jax'`` reference is the bit-faithful oracle (and
auto-selects the parallel associative scan on GPU); ``'pallas-cuda'`` is the
fused chunked-scan fast path, certified equivalent to the ``'jax'`` reference
only within the nitrix tolerance.  On Ampere+ NVIDIA hardware the fused
chunked-scan kernel runs; a shape it cannot tile -- non-float32, or a
non-power-of-two state dimension / chunk count -- declines and falls back to
the reference, emitting a loud ``NitrixBackendFallback`` warning (the GPU
parallel-scan speedup still applies via the reference).
"""

from __future__ import annotations

from typing import Optional

from jaxtyping import Array, Float

from ..._internal.backend import (
    Backend,
    NitrixBackendError,
    fallback,
    resolve_backend,
)
from ._reference import Driver, reference_selective_scan

__all__ = [
    'selective_scan',
    'reference_selective_scan',
]


def _validate(
    x: Float[Array, '... l d'],
    delta: Float[Array, '... l d'],
    A: Float[Array, 'd n'],
    B: Float[Array, '... l n'],
    C: Float[Array, '... l n'],
    D: Optional[Float[Array, 'd']],
) -> None:
    if x.shape != delta.shape:
        raise ValueError(
            f'selective_scan: x and delta must match; got x.shape={x.shape}, '
            f'delta.shape={delta.shape}.'
        )
    if x.ndim < 2:
        raise ValueError(
            f'selective_scan: x must be at least (l, d); got ndim {x.ndim}.'
        )
    d = x.shape[-1]
    length = x.shape[-2]
    if A.ndim != 2 or A.shape[0] != d:
        raise ValueError(
            f'selective_scan: A must be (d, n) with d={d}; got A.shape={A.shape}.'
        )
    n = A.shape[1]
    if B.shape[-1] != n or C.shape[-1] != n:
        raise ValueError(
            f'selective_scan: B/C last dim must be n={n}; got B.shape={B.shape}, '
            f'C.shape={C.shape}.'
        )
    if B.shape[-2] != length or C.shape[-2] != length:
        raise ValueError(
            f'selective_scan: B/C sequence dim must be l={length}; got '
            f'B.shape={B.shape}, C.shape={C.shape}.'
        )
    if D is not None and D.shape != (d,):
        raise ValueError(
            f'selective_scan: D must be (d,) with d={d}; got D.shape={D.shape}.'
        )


def _selective_scan_pallas(
    x: Float[Array, '... l d'],
    delta: Float[Array, '... l d'],
    A: Float[Array, 'd n'],
    B: Float[Array, '... l n'],
    C: Float[Array, '... l n'],
    D: Optional[Float[Array, 'd']],
) -> Optional[Float[Array, '... l d']]:
    """Dispatch to the fused Pallas selective-scan kernel.

    Attempts to import and run the CUDA Pallas kernel on the given inputs,
    returning ``None`` when the kernel is unavailable (import failure) or
    rejects the shape / host tiling, so the caller can fall back to the
    reference path.

    Parameters
    ----------
    x, delta
        Input sequence and per-step / per-channel :math:`\\Delta`, both
        ``(..., l, d)``.
    A
        State matrix in diagonal-plus form, ``(d, n)``.
    B, C
        Selective input / output projections, ``(..., l, n)``.
    D
        Optional per-channel skip / residual, ``(d,)`` or ``None``.

    Returns
    -------
    Optional[Float[Array, '... l d']]
        The output sequence ``(..., l, d)`` if the fused kernel ran, or
        ``None`` if the kernel could not be imported or the shape / host was
        not tileable.
    """
    try:
        from ..._kernels.cuda.selective_scan import (
            PallasNotTileable,
            selective_scan_pallas,
        )
    except Exception:
        return None
    try:
        return selective_scan_pallas(x, delta, A, B, C, D)
    except PallasNotTileable:
        return None


def selective_scan(
    x: Float[Array, '... l d'],
    delta: Float[Array, '... l d'],
    A: Float[Array, 'd n'],
    B: Float[Array, '... l n'],
    C: Float[Array, '... l n'],
    D: Optional[Float[Array, 'd']] = None,
    *,
    driver: Driver = 'auto',
    backend: Backend = 'auto',
) -> Float[Array, '... l d']:
    """Mamba / S6 selective state-space scan with backend dispatch.

    Computes the discretised selective recurrence (see
    :func:`reference_selective_scan` for the math).  ``backend`` selects the
    execution engine: ``'jax'`` (the reference; parallel associative scan on
    GPU), ``'pallas-cuda'`` (the fused chunked-scan fast path, which currently
    falls back to the reference with a loud warning), or ``'auto'``.

    Parameters
    ----------
    x, delta, A, B, C, D
        As in :func:`reference_selective_scan`.
    driver
        The numerical variant of the **reference** path (the ``driver`` axis):
        ``'auto'`` / ``'sequential'`` / ``'associative'`` / ``'chunked'`` (see
        :func:`reference_selective_scan`).  A non-``'auto'`` ``driver`` selects
        the reference path (the fused kernel has no driver variants), so it is a
        contradiction to combine it with an explicit ``backend='pallas-cuda'``.
        ``nitrix.reproducible()`` forces the canonical ``'sequential'`` *and*
        the reference engine (see ``backend``).
    backend
        ``'auto'`` / ``'pallas-cuda'`` / ``'jax'`` execution engine.  Under
        ``nitrix.reproducible()``, ``'auto'`` resolves to ``'jax'`` (the
        reference), since the fused kernel is certified only to a tolerance.

    Returns
    -------
    Output sequence ``(..., l, d)``.

    Notes
    -----
    The reference path is autodiff-native (``jax.grad`` flows through the
    scan).  The fused path will register a recompute-adjoint ``custom_vjp``
    (the state-space analogue of the recompute-forward adjoint used in
    :mod:`nitrix.numerics`).
    """
    _validate(x, delta, A, B, C, D)
    if driver != 'auto':
        # An explicit numerical variant selects the (jax) reference path.
        if backend == 'pallas-cuda':
            raise NitrixBackendError(
                f'selective_scan: driver={driver!r} selects a reference '
                "variant and does not apply to backend='pallas-cuda' (the "
                "fused kernel has no driver variants). Use backend='jax' or "
                "backend='auto'."
            )
        return reference_selective_scan(x, delta, A, B, C, D, driver=driver)
    resolved = resolve_backend(backend)
    if resolved == 'pallas-cuda':
        out = _selective_scan_pallas(x, delta, A, B, C, D)
        if out is not None:
            return out
        fallback(
            function='selective_scan',
            requested='pallas-cuda',
            resolved='jax',
            reason=(
                'the fused selective-scan kernel does not support this '
                'shape/host (needs Ampere+, float32, and a power-of-two state '
                'dimension / chunk count)'
            ),
            shapes=(tuple(x.shape), tuple(A.shape)),
            dtype=x.dtype,
        )
    return reference_selective_scan(x, delta, A, B, C, D, driver='auto')
