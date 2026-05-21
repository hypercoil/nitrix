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
affected stacks.  We probe once at first use; subsequent calls
read the cached verdict.

Used by:

- ``nitrix.linalg.spd`` -- ``symmap`` / ``symlog`` / ``symsqrt``
  and friends.
- ``nitrix.graph.connectopy`` -- ``laplacian_eigenmap`` /
  ``diffusion_embedding`` dense path.
- ``nitrix.graph._lobpcg_diff`` -- LOBPCG's internal QR /
  Cholesky path.

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
from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


__all__ = [
    'safe_eigh',
    'eigh_device',
    'solver_device',
    'device_of_concrete',
    'source_device',
]


@functools.lru_cache(maxsize=1)
def eigh_device() -> jax.Device:
    '''Pick a device where dense ``eigh`` works.

    Returns ``jax.devices()[0]`` if a 2x2 GPU eigh probe succeeds;
    otherwise the first CPU device.  Cached so the probe runs at
    most once per process.

    Use ``solver_device()`` if you want the same verdict applied
    to a different solver (e.g. LOBPCG / QR / Cholesky, which
    share the cuSolver handle pool on broken stacks).
    '''
    try:
        probe = jnp.eye(2, dtype=jnp.float32)
        out = jnp.linalg.eigh(probe)
        jax.block_until_ready(out)
        return jax.devices()[0]
    except Exception:
        cpu_devs = jax.devices('cpu')
        return cpu_devs[0] if cpu_devs else jax.devices()[0]


def solver_device() -> jax.Device:
    '''Pick the device for matrix-free iterative solvers (LOBPCG, etc.).

    Internally calls cuSolver-backed QR / Cholesky, so it shares
    ``eigh_device()``'s verdict.
    '''
    return eigh_device()


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
    '''``jnp.linalg.eigh`` with the cuSolver-robust device pick.

    Probes once at module-import time; subsequent calls use the
    cached verdict.  Always routes the eigh call itself to the
    safe device (so it works under ``jax.grad`` where the input
    is a tracer with no concrete device).  When the input *is*
    concrete and lives on a different device, results are moved
    back so the caller doesn't see a surprise CPU array.

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
    # Always pin the eigh call to the safe device, even under
    # trace -- the JIT-time dispatcher otherwise picks the GPU
    # path which is broken on the affected stacks.  device_put is
    # cheap-or-free at JIT time (it's a placement hint), so the
    # cost is bounded.
    A_dev = jax.device_put(A, target)
    eigvals, eigvecs = jnp.linalg.eigh(A_dev)
    if source is not None and source != target:
        eigvals = jax.device_put(eigvals, source)
        eigvecs = jax.device_put(eigvecs, source)
    return eigvals, eigvecs
