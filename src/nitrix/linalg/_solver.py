# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Robust-device wrappers for cuSolver-backed linear-algebra ops.

The motivating reality: certain CUDA / JAX combinations have a broken
cuSolver handle (an ABI mismatch between the cuSolver library and the GPU
driver, manifesting as ``gpusolverDnCreate(&handle) failed``).  On this
repo's dev L4 the *entire* dense-solver handle pool is dead -- ``eigh`` /
``qr`` / ``cholesky`` / ``solve`` / ``inv`` / ``svd`` all fail; only
``triangular_solve`` (cuBLAS ``trsm``), conv, and the elementwise surface
work.

This module routes the affected ops to a device where they work (CPU on
such stacks) and moves the result back, **adaptively**: each op probes a
2x2 GPU call once (cached); ``safe_*`` additionally *latches* to CPU if a
real call fails at runtime (the import-time probe can pass on a stack whose
handle wedges later or only at larger sizes -- B14 #2).

Used by ``nitrix.linalg.{spd,_eigsolve}``, ``nitrix.bias``,
``nitrix.register``, ``nitrix.stats.pca``, ``nitrix.geometry.affine``.

The probe / cache / latch / device-pick / try-force-retry-move-back
boilerplate is identical for every op, so it is factored once:
``_probed_device_pair`` builds the ``(device, latch)`` pair and
``_run_safe`` runs an op on the chosen device with the adaptive fallback.
Adding a ``safe_{op}`` when the next op breaks is then a one-liner -- build
its device pair, wrap its compute closure in ``_run_safe``.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, List, Optional, Tuple, cast

import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Float

__all__ = [
    'safe_eigh',
    'safe_inv',
    'safe_cholesky',
    'safe_cho_solve',
    'safe_solve',
    'eigh_device',
    'inv_device',
    'cholesky_device',
    'solver_device',
    'device_of_concrete',
    'source_device',
]


def _cpu_device() -> jax.Device:
    cpu_devs = jax.devices('cpu')
    return cpu_devs[0] if cpu_devs else jax.devices()[0]


def _is_cusolver_failure(exc: BaseException) -> bool:
    """Whether ``exc`` is the broken-cuSolver-handle failure (vs a real error)."""
    msg = str(exc).lower()
    return any(
        tok in msg
        for tok in ('cusolver', 'gpusolverdn', 'solver_handle', 'jclapack')
    )


def _probed_device_pair(
    probe_op: Callable[[Array], Any],
    *,
    doc: str,
) -> Tuple[Callable[[], jax.Device], Callable[[], None]]:
    """Build a ``(device, latch)`` pair for a cuSolver-backed op.

    ``device()`` returns ``jax.devices()[0]`` if a cached 2x2 GPU probe of
    ``probe_op`` succeeds, else the first CPU device; once ``latch()`` is
    called (after an observed runtime failure) it returns CPU permanently.
    The latch is a per-op process-global, so both eager and (subsequent)
    traced callers become GPU-safe after the first failure.
    """
    latched: List[bool] = [False]

    @functools.lru_cache(maxsize=1)
    def _probe() -> jax.Device:
        try:
            jax.block_until_ready(probe_op(jnp.eye(2, dtype=jnp.float32)))
            return jax.devices()[0]
        except Exception:
            return _cpu_device()

    def device() -> jax.Device:
        return _cpu_device() if latched[0] else _probe()

    def latch() -> None:
        latched[0] = True
        _probe.cache_clear()

    device.__doc__ = doc
    return device, latch


eigh_device, _latch_eigh = _probed_device_pair(
    jnp.linalg.eigh,
    doc='Pick a device where dense ``eigh`` works (CPU on broken stacks).',
)
inv_device, _latch_inv = _probed_device_pair(
    jnp.linalg.inv,
    doc='Pick a device where dense ``inv`` / ``solve`` (getrf) works.',
)
cholesky_device, _latch_cholesky = _probed_device_pair(
    jnp.linalg.cholesky,
    doc='Pick a device where dense ``cholesky`` (potrf) works.',
)


def solver_device() -> jax.Device:
    """Device for matrix-free iterative solvers (LOBPCG, etc.).

    They call cuSolver-backed QR / Cholesky internally, which share the
    handle pool with ``eigh``, so this follows ``eigh_device()``'s verdict.
    """
    return eigh_device()


def device_of_concrete(arr: Any) -> Optional[jax.Device]:
    """Return the device of a concrete array, or ``None`` for tracers.

    ``arr.devices()`` raises ``ConcretizationTypeError`` inside a JAX
    trace; we treat tracers as "no fixed device" so the caller can let JAX
    abstract evaluation handle dispatch.
    """
    if not hasattr(arr, 'devices'):
        return None
    try:
        devs = arr.devices()
    except jax.errors.ConcretizationTypeError:
        return None
    return next(iter(devs), None)


def source_device(tree: Any) -> Optional[jax.Device]:
    """The "originating" device for a tree of arrays (first found, or None)."""
    leaves = jax.tree_util.tree_leaves(tree)
    devs = set()
    for leaf in leaves:
        if not hasattr(leaf, 'devices'):
            continue
        try:
            devs.update(leaf.devices())
        except jax.errors.ConcretizationTypeError:
            continue
        except Exception:
            continue
    return next(iter(devs), None) if devs else None


def _run_safe(
    compute: Callable[[jax.Device], Any],
    source: Optional[jax.Device],
    device_fn: Callable[[], jax.Device],
    latch_fn: Callable[[], None],
) -> Any:
    """Run ``compute(target)`` on a working device, with adaptive fallback.

    ``compute`` takes the target device and returns the result (it is
    responsible for ``device_put``-ing its operands there).  For a concrete
    input on a GPU the call is forced so a late cuSolver-handle failure
    surfaces here; on such a failure the op's device is latched to CPU and
    the call retried there.  A genuine numerical error is re-raised.  When
    the input was concrete and on another device, the result is moved back.
    Under trace (``source is None``) a runtime failure is uncatchable, but
    the latched verdict from any prior eager failure routes the
    ``device_put`` to CPU -- so grad / jit become safe after the first
    observed failure.
    """
    target = device_fn()
    try:
        out = compute(target)
        if source is not None and target.platform != 'cpu':
            jax.block_until_ready(out)
    except Exception as exc:
        if target.platform == 'cpu' or not _is_cusolver_failure(exc):
            raise
        latch_fn()
        target = _cpu_device()
        out = compute(target)
    if source is not None and source != target:
        out = jax.tree_util.tree_map(lambda x: jax.device_put(x, source), out)
    return out


def safe_eigh(
    A: Float[Array, '... n n'],
) -> Tuple[Float[Array, '... n'], Float[Array, '... n n']]:
    """``jnp.linalg.eigh`` with the cuSolver-robust, adaptive device pick.

    ``A`` is taken as symmetric (not symmetrised here).  Returns
    ``(eigenvalues, eigenvectors)``; see ``_run_safe`` for the fallback.
    """
    return cast(
        Tuple[Float[Array, '... n'], Float[Array, '... n n']],
        _run_safe(
            lambda dev: jnp.linalg.eigh(jax.device_put(A, dev)),
            device_of_concrete(A),
            eigh_device,
            _latch_eigh,
        ),
    )


def safe_inv(
    A: Float[Array, '... n n'],
) -> Float[Array, '... n n']:
    """``jnp.linalg.inv`` with the cuSolver-robust, adaptive device pick."""
    return cast(
        Float[Array, '... n n'],
        _run_safe(
            lambda dev: jnp.linalg.inv(jax.device_put(A, dev)),
            device_of_concrete(A),
            inv_device,
            _latch_inv,
        ),
    )


def safe_cholesky(
    A: Float[Array, '... n n'],
) -> Float[Array, '... n n']:
    """``jnp.linalg.cholesky`` with the cuSolver-robust, adaptive device pick.

    ``A`` must be symmetric positive-definite (not symmetrised /
    regularised here).
    """
    return cast(
        Float[Array, '... n n'],
        _run_safe(
            lambda dev: jnp.linalg.cholesky(jax.device_put(A, dev)),
            device_of_concrete(A),
            cholesky_device,
            _latch_cholesky,
        ),
    )


def _cho_solve_on(a: Array, b: Array, l2: float) -> Array:
    """SPD solve ``(a + l2 I) x = b`` via Cholesky, co-located on a/b's device.

    Only the factorisation hits cuSolver; ``triangular_solve`` (cuBLAS
    ``trsm``) is unaffected, so the whole solve runs on one device to avoid
    a cross-device factor / RHS.
    """
    n = a.shape[-1]
    mat = a if l2 == 0.0 else a + l2 * jnp.eye(n, dtype=a.dtype)
    chol = jnp.linalg.cholesky(mat)
    is_vector = b.ndim == a.ndim - 1
    rhs = b[..., None] if is_vector else b
    z = lax.linalg.triangular_solve(
        chol, rhs, left_side=True, lower=True, transpose_a=False
    )
    x = lax.linalg.triangular_solve(
        chol, z, left_side=True, lower=True, transpose_a=True
    )
    return x[..., 0] if is_vector else x


def safe_cho_solve(
    a: Float[Array, '... n n'],
    b: Float[Array, '...'],
    *,
    l2: float = 0.0,
) -> Float[Array, '...']:
    """SPD Cholesky solve ``(a + l2 I) x = b``, cuSolver-robust + adaptive."""
    return cast(
        Float[Array, '...'],
        _run_safe(
            lambda dev: _cho_solve_on(
                jax.device_put(a, dev), jax.device_put(b, dev), l2
            ),
            source_device((a, b)),
            cholesky_device,
            _latch_cholesky,
        ),
    )


def safe_solve(
    a: Float[Array, '... n n'],
    b: Float[Array, '...'],
) -> Float[Array, '...']:
    """``jnp.linalg.solve`` with the cuSolver-robust, adaptive device pick."""
    return cast(
        Float[Array, '...'],
        _run_safe(
            lambda dev: jnp.linalg.solve(
                jax.device_put(a, dev), jax.device_put(b, dev)
            ),
            source_device((a, b)),
            inv_device,
            _latch_inv,
        ),
    )
