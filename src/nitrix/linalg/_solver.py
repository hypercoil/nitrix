# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Robust-device wrappers for cuSolver-backed linear-algebra ops.

The motivating reality: certain CUDA / JAX combinations have a
broken cuSolver handle (an ABI mismatch between the cuSolver
library and the GPU driver, manifesting as
``gpusolverDnCreate(&handle) failed``).  Concretely on the
nitrix test runner: GPU ``eigh`` / ``qr`` are broken; GPU
``svd`` / ``lstsq`` / ``cholesky`` / ``solve`` / ``solve_triangular``
work.

This module provides ``safe_eigh`` (and the device-pick helpers
it uses) for the routines that must fall back to CPU on the
affected stacks.  We probe once at first use and cache the verdict;
``safe_eigh`` additionally *latches* to CPU if a real eigh is
observed to fail on the GPU at call time (B14 #2 -- the import-time
probe can pass on a stack that wedges later or only at larger sizes).

Used by:

- ``nitrix.linalg.spd`` -- ``symmap`` / ``symlog`` / ``symsqrt``
  and friends.
- ``nitrix.linalg._eigsolve`` -- the extremal-eigensolver
  dispatcher: ``eigh`` for the dense ``laplacian_eigenmap`` /
  ``diffusion_embedding`` path (through ``graph.connectopy``), and
  the iterative solvers' internal QR / Cholesky.

The eigh-vs-other-solvers asymmetry is intentional: on a healthy
stack we don't pay any overhead for the wrapper; on a broken
stack we route eigh-shaped ops to CPU automatically.  Other
solvers (svd, lstsq) are called via their plain JAX surfaces
because they aren't affected on the runners we test against.

If a future GPU stack breaks one of those instead, the right
move is to add a ``safe_{op}`` helper here -- the per-op probe
+ cache pattern is the same.
"""

from __future__ import annotations

import functools
from typing import Any, Optional, Tuple, cast

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

__all__ = [
    'safe_eigh',
    'safe_inv',
    'eigh_device',
    'inv_device',
    'solver_device',
    'device_of_concrete',
    'source_device',
]


# Adaptive override: set once a *real* eigh has failed on the GPU at call time
# (the import-time probe can pass on a stack whose cuSolver handle wedges later
# or only at larger sizes -- B14 issue 2).  Once latched, every subsequent
# ``eigh_device()`` verdict is CPU, so both eager and traced callers become
# GPU-safe after the first observed failure.
_eigh_cpu_latched = False


def _cpu_device() -> jax.Device:
    cpu_devs = jax.devices('cpu')
    return cpu_devs[0] if cpu_devs else jax.devices()[0]


def _is_cusolver_failure(exc: BaseException) -> bool:
    '''Whether ``exc`` is the broken-cuSolver-handle failure (vs a real error).'''
    msg = str(exc).lower()
    return any(
        tok in msg
        for tok in ('cusolver', 'gpusolverdn', 'solver_handle', 'jclapack')
    )


def _latch_eigh_cpu() -> None:
    '''Latch the eigh device to CPU after an observed cuSolver failure.'''
    global _eigh_cpu_latched
    _eigh_cpu_latched = True
    _probe_eigh_device.cache_clear()


@functools.lru_cache(maxsize=1)
def _probe_eigh_device() -> jax.Device:
    '''One-shot 2x2 GPU eigh probe (cheap; the happy-path device pick).'''
    try:
        probe = jnp.eye(2, dtype=jnp.float32)
        out = jnp.linalg.eigh(probe)
        jax.block_until_ready(out)
        return jax.devices()[0]
    except Exception:
        return _cpu_device()


def eigh_device() -> jax.Device:
    '''Pick a device where dense ``eigh`` works.

    Returns ``jax.devices()[0]`` if a 2x2 GPU eigh probe succeeds; otherwise
    (or once a real eigh has been observed to fail on the GPU at call time --
    see ``safe_eigh``) the first CPU device.  The probe is cached; the
    call-time latch makes the verdict *adaptive* so a stack whose cuSolver
    handle is healthy at import but wedges later still ends up on CPU.

    Use ``solver_device()`` if you want the same verdict applied to a
    different solver (e.g. LOBPCG / QR / Cholesky, which share the cuSolver
    handle pool on broken stacks).
    '''
    if _eigh_cpu_latched:
        return _cpu_device()
    return _probe_eigh_device()


def solver_device() -> jax.Device:
    '''Pick the device for matrix-free iterative solvers (LOBPCG, etc.).

    Internally calls cuSolver-backed QR / Cholesky, so it shares
    ``eigh_device()``'s verdict.
    '''
    return eigh_device()


@functools.lru_cache(maxsize=1)
def inv_device() -> jax.Device:
    '''Pick a device where dense ``inv`` / ``solve`` works.

    Probes a 2x2 GPU ``inv`` (cuSolver ``getrf`` / ``getri``).  On the
    runners documented above only ``eigh`` / ``qr`` are broken, but some
    stacks (and this repo's current CI box) also break ``getrf``; the
    per-op probe routes ``inv`` to CPU exactly when it must.  Cached so the
    probe runs at most once per process.
    '''
    try:
        probe = jnp.eye(2, dtype=jnp.float32)
        out = jnp.linalg.inv(probe)
        jax.block_until_ready(out)
        return jax.devices()[0]
    except Exception:
        cpu_devs = jax.devices('cpu')
        return cpu_devs[0] if cpu_devs else jax.devices()[0]


def device_of_concrete(arr: Any) -> Optional[jax.Device]:
    '''Return the device of a concrete array, or ``None`` for tracers.

    ``arr.devices()`` raises ``ConcretizationTypeError`` inside a
    JAX trace; we treat tracers as "no fixed device" so the caller
    can let JAX abstract evaluation handle dispatch.
    '''
    if not hasattr(arr, 'devices'):
        return None
    try:
        devs = arr.devices()
    except jax.errors.ConcretizationTypeError:
        return None
    return next(iter(devs), None)


def source_device(tree: Any) -> Optional[jax.Device]:
    '''The "originating" device for a tree of arrays.

    If all leaves share a device, return it.  If multiple, return
    the first found.  ``None`` if no concrete-array leaves.
    '''
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


def safe_eigh(
    A: Float[Array, '... n n'],
) -> Tuple[Float[Array, '... n'], Float[Array, '... n n']]:
    '''``jnp.linalg.eigh`` with the cuSolver-robust, adaptive device pick.

    Pins the eigh to a device where it works (per ``eigh_device()``), routing
    it there even under trace (so it works under ``jax.grad`` where the input
    is a tracer with no concrete device).  When the input *is* concrete and on
    a different device, results are moved back so the caller doesn't see a
    surprise CPU array.

    **Adaptive fallback (B14 #2).**  The import-time probe can pass on a stack
    whose cuSolver handle is healthy at 2x2 but wedges later or at larger
    sizes.  For a concrete (eager) input on the GPU we therefore force
    execution and, if a cuSolver-handle failure surfaces, latch the device to
    CPU and retry there -- so ``solver='eigh'`` / ``'auto'`` become GPU-safe
    after the first observed failure, for both eager and (subsequent) traced
    callers.  A genuine numerical error is re-raised unchanged.

    Parameters
    ----------
    A
        Symmetric matrix batch.  Caller is responsible for
        symmetry; we do not symmetrise.

    Returns
    -------
    ``(eigenvalues, eigenvectors)`` per ``jnp.linalg.eigh``.
    '''
    target = eigh_device()
    source = device_of_concrete(A)
    A_dev = jax.device_put(A, target)
    try:
        eigvals, eigvecs = jnp.linalg.eigh(A_dev)
        # Concrete input on a GPU: force execution so a cuSolver-handle
        # failure surfaces here and can be retried on CPU.  Under trace
        # (source is None) a runtime failure is uncatchable; the latched
        # verdict from any prior eager failure still routes us to CPU.
        if source is not None and target.platform != 'cpu':
            jax.block_until_ready((eigvals, eigvecs))
    except Exception as exc:
        if target.platform == 'cpu' or not _is_cusolver_failure(exc):
            raise
        _latch_eigh_cpu()
        target = _cpu_device()
        eigvals, eigvecs = jnp.linalg.eigh(jax.device_put(A, target))
    if source is not None and source != target:
        eigvals = jax.device_put(eigvals, source)
        eigvecs = jax.device_put(eigvecs, source)
    return eigvals, eigvecs


def safe_inv(
    A: Float[Array, '... n n'],
) -> Float[Array, '... n n']:
    '''``jnp.linalg.inv`` with the cuSolver-robust device pick.

    Mirrors ``safe_eigh``: pins the inverse to a device where ``getrf`` /
    ``getri`` works (CPU on the affected stacks), then moves the result
    back to the caller's device when the input was concrete and elsewhere.
    Used for the small, regularised (SPD) control-lattice Gram inverse in
    ``nitrix.bias`` -- computed once per fitting level, so the CPU round
    trip (if any) is negligible against the GPU matmuls it feeds.
    '''
    target = inv_device()
    source = device_of_concrete(A)
    A_dev = jax.device_put(A, target)
    out = jnp.linalg.inv(A_dev)
    if source is not None and source != target:
        out = jax.device_put(out, source)
    return cast(Float[Array, '... n n'], out)
