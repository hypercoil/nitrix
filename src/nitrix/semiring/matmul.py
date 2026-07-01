# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Public :func:`semiring_matmul` with three-level backend selection.

The public entry point accepts a left operand of shape ``(..., m, k)``
and a right operand of shape ``(..., k, n)``, an optional ``semiring``
algebra, and a ``backend`` selector, returning the semiring-generalised
product of shape ``(..., m, n)``.

Leading batch dims are broadcast-compatible.  The 2-D core is dispatched
to either :func:`reference_semiring_matmul` (pure JAX) or the Pallas /
Triton kernel ``semiring_matmul_pallas``.  If the Pallas path cannot
tile a given shape and algebra combination, the call falls back to JAX
and emits exactly one :class:`NitrixBackendFallback` warning per
``(function, shape, dtype, backend)`` per process; see
:mod:`nitrix._internal.backend` for the environment-variable knobs.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Callable, Optional, Tuple, TypeVar

import jax
import jax.numpy as jnp
from jaxtyping import Array, Num

from .._internal.backend import (
    Backend,
    ResolvedBackend,
    fallback,
    resolve_backend,
)
from ._reference import reference_semiring_matmul
from ._types import Semiring
from .algebras import REAL

_T = TypeVar('_T')

__all__ = [
    'semiring_matmul',
    'reference_semiring_matmul',
]


def _check_2d_compat(
    A_shape: tuple[int, ...], B_shape: tuple[int, ...], name: str
) -> tuple[int, int, int]:
    if len(A_shape) < 2 or len(B_shape) < 2:
        raise ValueError(
            f'{name}: A and B must have at least 2 dimensions; got '
            f'A.shape={A_shape}, B.shape={B_shape}.'
        )
    m, k = A_shape[-2], A_shape[-1]
    k2, n = B_shape[-2], B_shape[-1]
    if k != k2:
        raise ValueError(
            f'{name}: contraction dim mismatch -- A has k={k}, B has '
            f'k={k2} (A.shape={A_shape}, B.shape={B_shape}).'
        )
    return int(m), int(k), int(n)


def _broadcast_batch(
    A: Array, B: Array
) -> Tuple[Array, Array, Tuple[int, ...]]:
    """Broadcast the leading batch dims of ``A`` and ``B`` together.

    The trailing two dims of each operand are left unchanged; only the
    leading batch dims are broadcast to a common shape.  ``jnp.broadcast_to``
    is used so that broadcast copies are avoided where possible.

    Parameters
    ----------
    A
        Left operand of shape ``(*A_batch, m, k)``.
    B
        Right operand of shape ``(*B_batch, k, n)``.

    Returns
    -------
    A_b : Array
        ``A`` broadcast to shape ``(*out_batch, m, k)``.
    B_b : Array
        ``B`` broadcast to shape ``(*out_batch, k, n)``.
    out_batch : tuple of int
        The common broadcast batch shape.
    """
    A_batch = A.shape[:-2]
    B_batch = B.shape[:-2]
    out_batch = jnp.broadcast_shapes(A_batch, B_batch)
    A_b = jnp.broadcast_to(A, out_batch + A.shape[-2:])
    B_b = jnp.broadcast_to(B, out_batch + B.shape[-2:])
    return A_b, B_b, out_batch


def _run_jax_core(A2: Array, B2: Array, *, semiring: Semiring[Any]) -> Array:
    """Compute the 2-D pure-JAX reference product for a single batch slice.

    This is the reference-backend core, expected to be ``vmap``-ed over
    the leading batch dims by the caller.

    Parameters
    ----------
    A2
        Left operand of shape ``(m, k)``.
    B2
        Right operand of shape ``(k, n)``.
    semiring
        Algebra defining the combine and reduction operators.

    Returns
    -------
    Array
        The semiring product of shape ``(m, n)``.
    """
    return reference_semiring_matmul(A2, B2, semiring=semiring)


def _vmap_over_batch(
    fn: Callable[..., _T], n_batch_dims: int
) -> Callable[..., _T]:
    out = fn
    for _ in range(n_batch_dims):
        out = jax.vmap(out, in_axes=(0, 0))
    return out


def _semiring_matmul_jax(
    A: Array, B: Array, *, semiring: Semiring[Any]
) -> Array:
    Ab, Bb, batch = _broadcast_batch(A, B)
    core = partial(_run_jax_core, semiring=semiring)
    return _vmap_over_batch(core, len(batch))(Ab, Bb)


def _semiring_matmul_pallas(
    A: Array, B: Array, *, semiring: Semiring[Any]
) -> Optional[Array]:
    """Dispatch to the Pallas kernel, or return ``None`` if it declines.

    The kernel is imported on call so that a JAX install with a broken
    Pallas can still use the JAX fallback.  Only ``PallasNotTileable``
    is caught: a real kernel failure (an unexpected lowering error, a
    CUDA error) is not silently swallowed.  Leading batch dims are
    broadcast and the kernel is ``vmap``-ed over them.

    Parameters
    ----------
    A
        Left operand of shape ``(*A_batch, m, k)``.
    B
        Right operand of shape ``(*B_batch, k, n)``.
    semiring
        Algebra defining the combine and reduction operators.

    Returns
    -------
    Array or None
        The semiring product of shape ``(*out_batch, m, n)`` if the
        Pallas kernel is available and can tile the requested shape;
        otherwise ``None``.
    """
    try:
        from .._kernels.cuda.semiring_matmul import (
            PallasNotTileable,
            semiring_matmul_pallas,
        )
    except Exception:
        return None
    Ab, Bb, batch = _broadcast_batch(A, B)
    try:
        core = partial(semiring_matmul_pallas, semiring=semiring)
        return _vmap_over_batch(core, len(batch))(Ab, Bb)
    except PallasNotTileable:
        return None


def _forward_only_2d(
    A: Array,
    B: Array,
    *,
    semiring: Semiring[Any],
    backend: Backend,
) -> Array:
    """Run the backend-dispatching 2-D forward pass without a VJP wrapper.

    Called both from the user-facing :func:`semiring_matmul` (when the
    algebra is forward-only) and from the ``jax.custom_vjp`` forward
    function.  Operates on a single 2-D batch slice; batching is handled
    upstream via ``jax.vmap``.  When the resolved backend is
    ``'pallas-cuda'`` but the kernel is unavailable or cannot tile the
    shape, the call falls back to the pure-JAX reference and emits a
    fallback warning.

    Parameters
    ----------
    A
        Left operand of shape ``(m, k)``.
    B
        Right operand of shape ``(k, n)``.
    semiring
        Algebra defining the combine and reduction operators.
    backend
        Backend selector (``'auto'``, ``'pallas-cuda'``, or ``'jax'``),
        resolved before dispatch.

    Returns
    -------
    Array
        The semiring product of shape ``(m, n)``.
    """
    resolved: ResolvedBackend = resolve_backend(backend)
    if resolved == 'pallas-cuda':
        try:
            from .._kernels.cuda.semiring_matmul import (
                PallasNotTileable,
                semiring_matmul_pallas,
            )
        except Exception:
            out = None
        else:
            try:
                out = semiring_matmul_pallas(A, B, semiring=semiring)
            except PallasNotTileable:
                out = None
        if out is None:
            resolved = fallback(
                function='semiring_matmul',
                requested='pallas-cuda',
                resolved='jax',
                reason=(
                    f'algebra={semiring.name!r}: Pallas Triton cannot '
                    'tile the requested (M, K, N) for the chosen block '
                    'sizes, or the kernel is unavailable on this host'
                ),
                shapes=(tuple(A.shape), tuple(B.shape)),
                dtype=A.dtype,
            )
        else:
            return out
    return reference_semiring_matmul(A, B, semiring=semiring)


@partial(jax.custom_vjp, nondiff_argnums=(2, 3))
def _diff_matmul_2d(
    A: Array,
    B: Array,
    semiring: Semiring[Any],
    backend: Backend,
) -> Array:
    """Compute the differentiable 2-D core of :func:`semiring_matmul`.

    The forward pass delegates to :func:`_forward_only_2d`; the backward
    is routed to the per-algebra ``semiring.matmul_vjp`` rule.
    ``semiring`` and ``backend`` are non-differentiable (carried through
    but not differentiated), and their hashable identity is part of
    JAX's trace cache key.  Batching is composed via ``jax.vmap``
    upstream, with which ``jax.custom_vjp`` is compatible.

    Parameters
    ----------
    A
        Left operand of shape ``(m, k)``.
    B
        Right operand of shape ``(k, n)``.
    semiring
        Algebra defining the combine and reduction operators, and (for
        differentiable algebras) the backward rule.  Non-differentiable.
    backend
        Backend selector (``'auto'``, ``'pallas-cuda'``, or ``'jax'``).
        Non-differentiable.

    Returns
    -------
    Array
        The semiring product of shape ``(m, n)``.
    """
    return _forward_only_2d(A, B, semiring=semiring, backend=backend)


def _diff_matmul_2d_fwd(
    A: Array, B: Array, semiring: Semiring[Any], backend: Backend
) -> Tuple[Array, Tuple[Array, Array, Array]]:
    # NB: ``custom_vjp`` ``fwd`` mirrors the primal's full signature.
    # The nondiff args are *not* prepended here -- only ``bwd`` sees
    # them at the front of its argument list.
    out = _forward_only_2d(A, B, semiring=semiring, backend=backend)
    # Stash all of (A, B, out) so per-algebra backwards (LOG, TROPICAL_*,
    # EUCLIDEAN) that need the forward output have it.  REAL's backward
    # ignores ``out``; the residual cost is one (m, n) array we already
    # had to compute.
    return out, (A, B, out)


def _diff_matmul_2d_bwd(
    semiring: Semiring[Any],
    backend: Backend,
    residuals: Tuple[Array, Array, Array],
    g_out: Array,
) -> Tuple[Array, Array]:
    rule = semiring.matmul_vjp
    if rule is None:
        raise TypeError(
            f'semiring={semiring.name!r} is forward-only; supply '
            '``matmul_vjp`` to enable gradients (or wrap the call in '
            'your own ``jax.custom_vjp``).'
        )
    return rule(residuals, g_out)


_diff_matmul_2d.defvjp(_diff_matmul_2d_fwd, _diff_matmul_2d_bwd)


def semiring_matmul(
    A: Num[Array, '... m k'],
    B: Num[Array, '... k n'],
    *,
    semiring: Semiring[Any] = REAL,
    backend: Backend = 'auto',
) -> Num[Array, '... m n']:
    r"""Semiring-generalised matrix multiplication.

    Computes the semiring product
    :math:`C_{\ldots ij} = \bigoplus_k \left( A_{\ldots ik} \otimes
    B_{\ldots kj} \right)` under the supplied ``semiring``, where
    :math:`\oplus` and :math:`\otimes` are the algebra's reduction and
    combine operators.  Leading batch dims are broadcast together and
    the 2-D core is dispatched to the selected backend.

    Parameters
    ----------
    A
        Left operand of shape ``(..., m, k)``; the trailing two dims are
        ``(m, k)``.
    B
        Right operand of shape ``(..., k, n)``; the trailing two dims
        are ``(k, n)``.  The leading batch dims of ``A`` and ``B`` are
        broadcast-compatible.
    semiring
        Algebra (a :class:`Semiring` or :class:`StrictSemiring`)
        defining the reduction and combine operators. Defaults to
        :data:`REAL`.
    backend
        Backend selector: ``'auto'``, ``'pallas-cuda'``, or ``'jax'``.
        See :mod:`nitrix._internal.backend` for the resolution rules
        and environment-variable overrides.

    Returns
    -------
    Array
        The semiring product of shape ``(*broadcast_batch, m, n)``,
        where ``broadcast_batch`` is the broadcast of the operands'
        leading batch dims.

    Notes
    -----
    For :data:`REAL` on real-valued inputs, the result equals ``A @ B``;
    downstream consumers wanting tensor-core throughput on the real
    semiring should call ``jnp.matmul`` directly. This routine's
    advantage is the *other* algebras and the streaming kernel for the
    log, tropical, and euclidean semirings.

    Built-in algebras carry a hand-derived ``jax.custom_vjp`` rule.
    When ``semiring.matmul_vjp`` is not ``None`` the call routes through
    that wrapper, so ``jax.grad``, ``jax.vjp``, and ``jax.jacrev``
    return finite-difference-checked gradients without further setup.
    Algebras whose ``matmul_vjp`` is ``None`` (user-defined,
    forward-only) raise on backward with a clear message.
    """
    _check_2d_compat(A.shape, B.shape, name='semiring_matmul')

    Ab, Bb, batch = _broadcast_batch(A, B)
    core: Callable[[Array, Array], Array]
    if semiring.matmul_vjp is None:

        def core(A_: Array, B_: Array) -> Array:
            return _forward_only_2d(
                A_,
                B_,
                semiring=semiring,
                backend=backend,
            )
    else:
        # custom_vjp wrapper; vmap below composes with the registered
        # forward / backward.  semiring / backend are passed positionally
        # because ``nondiff_argnums`` is positional-only.
        def core(A_: Array, B_: Array) -> Array:
            return _diff_matmul_2d(A_, B_, semiring, backend)

    return _vmap_over_batch(core, len(batch))(Ab, Bb)
