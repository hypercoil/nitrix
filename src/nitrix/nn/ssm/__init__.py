# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
# isort: skip_file
"""
nitrix.nn.ssm -- selective state-space scan (Mamba / S6).

Public ``selective_scan`` with three-level backend selection.  The ``jax``
reference is the bit-faithful oracle (and auto-selects the parallel
``associative_scan`` on GPU); ``pallas-cuda`` is the fused chunked-scan fast
path (suite P1b), certified ``pallas-cuda ≈ jax`` only inside nitrix.  Until
the fused kernel lands, a ``pallas-cuda`` request falls back to the reference
with a loud ``NitrixBackendFallback`` (the GPU parallel-scan speedup still
applies via the reference).
"""

from __future__ import annotations

from typing import Optional

from jaxtyping import Array, Float

from ..._internal.backend import Backend, fallback, resolve_backend
from ._reference import reference_selective_scan

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
    """Pallas dispatch; ``None`` if the kernel rejects the shape / host."""
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
    backend: Backend = 'auto',
) -> Float[Array, '... l d']:
    """Mamba / S6 selective state-space scan with backend dispatch.

    Computes the discretised selective recurrence (see
    :func:`reference_selective_scan` for the math).  ``backend`` selects the
    implementation: ``'jax'`` (the reference; parallel ``associative_scan`` on
    GPU), ``'pallas-cuda'`` (the fused chunked-scan fast path, suite P1b;
    currently falls back to the reference with a loud warning), or ``'auto'``.

    Parameters
    ----------
    x, delta, A, B, C, D
        As in :func:`reference_selective_scan`.
    backend
        ``'auto'`` / ``'pallas-cuda'`` / ``'jax'``.

    Returns
    -------
    Output sequence ``(..., l, d)``.

    Differentiability
    -----------------
    The reference path is autodiff-native (``jax.grad`` flows through the scan).
    The fused path will register a recompute-adjoint ``custom_vjp`` in P1b
    (the SSM analogue of the ``numerics.ode`` recompute-forward adjoint).
    """
    _validate(x, delta, A, B, C, D)
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
                'no fused selective-scan kernel available for this shape/host '
                '(the fused path lands in suite P1b)'
            ),
            shapes=(tuple(x.shape), tuple(A.shape)),
            dtype=x.dtype,
        )
    return reference_selective_scan(x, delta, A, B, C, D)
