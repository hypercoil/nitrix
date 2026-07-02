# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Backend selection and fallback observability.

Implements a three-level resolution of the execution backend:

    explicit ``backend=`` keyword > ``NITRIX_BACKEND`` env var > auto-detect

Auto-detect resolves to ``pallas-cuda`` only when an NVIDIA GPU of Ampere
generation (compute capability ``sm_80``) or newer is present; otherwise
to ``jax``.

The :class:`NitrixBackendFallback` warning category is emitted at most once
per ``(function, shape-signature, dtype, backend)`` tuple per process.
Set ``NITRIX_SILENCE_FALLBACK=1`` to suppress; set
``NITRIX_STRICT_BACKEND=1`` to convert fallback to error.
"""

from __future__ import annotations

import os
import threading
import warnings
from typing import Any, Literal, Optional

import jax

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

Backend = Literal['auto', 'pallas-cuda', 'jax']
ResolvedBackend = Literal['pallas-cuda', 'jax']

_VALID_BACKENDS: tuple[Backend, ...] = ('auto', 'pallas-cuda', 'jax')


class NitrixBackendFallback(UserWarning):
    """Emitted when a backend resolves to a fallback path.

    Fallbacks are made loud on the principle that a silent performance
    regression is a bug. Whenever Pallas Triton cannot tile a given
    shape × algebra combination, the resolved backend changes to ``jax``
    and this warning is raised exactly once per ``(function,
    shape-signature, dtype, backend)`` tuple per process.

    To silence: ``NITRIX_SILENCE_FALLBACK=1``.
    To escalate to error: ``NITRIX_STRICT_BACKEND=1``.
    """


class NitrixBackendError(RuntimeError):
    """Raised when a backend cannot be honoured on this host.

    This occurs either when ``NITRIX_STRICT_BACKEND=1`` and a fallback
    would otherwise occur, or when a user-requested backend is
    unsupported on the current hardware.
    """


# ---------------------------------------------------------------------------
# Compute-capability probe (once at import time)
# ---------------------------------------------------------------------------


_AMPERE_MIN_MAJOR = 8  # sm_80 = A100; Ampere consumer parts (sm_86) also pass


def _has_ampere_or_newer_nvidia() -> bool:
    """Detect whether the default JAX backend exposes an Ampere+ NVIDIA GPU.

    Uses ``jax.devices('gpu')`` plus per-device ``compute_capability``
    attribute (recent JAX builds expose this as a ``(major, minor)``
    tuple or as a dotted-string).  If the capability is not exposed,
    we conservatively trust ``device_kind`` for the Ampere+ family
    names.  Pre-Ampere NVIDIA falls back to JAX with a one-time warning.

    Returns
    -------
    bool
        ``True`` if every visible GPU is of Ampere generation
        (compute capability major version 8) or newer, ``False`` if
        no GPU is visible or any visible GPU is pre-Ampere or of an
        undetermined capability.
    """
    try:
        gpus = jax.devices('gpu')
    except RuntimeError:
        return False
    if not gpus:
        return False
    for d in gpus:
        cc = getattr(d, 'compute_capability', None)
        if cc is None:
            kind = (getattr(d, 'device_kind', '') or '').lower()
            # Best-effort match: the names we know to be Ampere+.
            ampere_plus_names = (
                'a100',
                'a10',
                'a30',
                'a40',  # Ampere data-centre
                'rtx 30',
                'rtx 40',
                'rtx 50',  # Ampere/Lovelace/Blackwell consumer
                'l4',
                'l40',  # Lovelace data-centre
                'h100',
                'h200',
                'h800',  # Hopper
                'b100',
                'b200',  # Blackwell
            )
            if any(n in kind for n in ampere_plus_names):
                return True
            return False
        if isinstance(cc, str):
            try:
                major = int(cc.split('.')[0])
            except (ValueError, AttributeError):
                return False
        else:
            try:
                major = int(cc[0])
            except (TypeError, IndexError, ValueError):
                return False
        if major < _AMPERE_MIN_MAJOR:
            return False
    return True


_HAS_AMPERE_NVIDIA: bool = _has_ampere_or_newer_nvidia()


# ---------------------------------------------------------------------------
# Environment-variable knobs
# ---------------------------------------------------------------------------


def _env_flag(name: str) -> bool:
    return os.environ.get(name, '0').lower() in ('1', 'true', 'yes', 'on')


def silence_fallback() -> bool:
    return _env_flag('NITRIX_SILENCE_FALLBACK')


def strict_backend() -> bool:
    return _env_flag('NITRIX_STRICT_BACKEND')


def env_backend() -> Optional[Backend]:
    value = os.environ.get('NITRIX_BACKEND')
    if value is None:
        return None
    value = value.strip().lower()
    if value not in _VALID_BACKENDS:
        raise NitrixBackendError(
            f'NITRIX_BACKEND={value!r} not in {_VALID_BACKENDS!r}'
        )
    return value


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def auto_backend() -> ResolvedBackend:
    """Resolve ``auto`` to a concrete backend based on host hardware.

    Under reproducibility mode (``nitrix.reproducible()`` /
    ``NITRIX_REPRODUCIBLE=1``) the **reference** engine (``jax``) is preferred,
    so a fused-kernel op (selective_scan, attention, ...) runs its canonical
    reference path and reproduces across platforms -- the ``pallas-cuda``
    kernels are certified to the ``[op.dtype.pallas-cuda]`` tolerance, not
    bit-for-bit, which is a real cross-platform divergence.  This is the
    backend-axis half of the reproducible-dispatch principle; an explicit
    ``backend=`` keyword or ``NITRIX_BACKEND`` still overrides (explicit > mode),
    exactly as an explicit ``driver=`` overrides the numerical-variant axis.

    Returns
    -------
    {'pallas-cuda', 'jax'}
        ``'pallas-cuda'`` when an Ampere+ NVIDIA GPU is present and
        reproducibility mode is off; ``'jax'`` otherwise (including
        whenever reproducibility mode is enabled).
    """
    # Local import to avoid any import-order coupling (config imports no
    # backend symbols, so this direction is acyclic).
    from .config import reproducible_enabled

    if reproducible_enabled():
        return 'jax'
    if _HAS_AMPERE_NVIDIA:
        return 'pallas-cuda'
    return 'jax'


def default_backend_is_gpu() -> bool:
    r"""Whether JAX's default backend is a GPU.

    The static (trace-time) signal for *platform-dependent algorithm*
    selection -- distinct from :func:`auto_backend`, which picks a *kernel*
    backend from host capability.  The motivating case is the recursive IIR
    filters: the sequential ``lax.scan`` recurrence wins on the CPU (low
    overhead, no parallelism to exploit) while the parallel-prefix
    ``associative_scan`` wins on the GPU (:math:`O(\log T)` depth fills the
    device), so the optimal engine flips with the deployment target.
    ``jax.default_backend()`` is concrete at trace time, so branching on it
    is ``jit``-safe.

    Returns
    -------
    bool
        ``True`` if JAX's default backend platform is ``'gpu'``,
        ``False`` otherwise (including when no default backend can be
        determined).
    """
    try:
        return jax.default_backend() == 'gpu'
    except RuntimeError:
        return False


def resolve_backend(backend: Backend) -> ResolvedBackend:
    """Three-level resolution: keyword > env var > auto.

    Parameters
    ----------
    backend
        The user-supplied ``backend=`` argument; ``'auto'`` to defer to
        env / autodetect.

    Returns
    -------
    Either ``'pallas-cuda'`` or ``'jax'``.

    Raises
    ------
    NitrixBackendError
        If ``backend='pallas-cuda'`` was requested but no Ampere+ NVIDIA
        GPU is visible.  This is independent of
        ``NITRIX_STRICT_BACKEND``: an explicit ``pallas-cuda`` request on
        unsupported hardware is always an error, since the alternative
        (silently downgrading) would mask a deployment problem.
    """
    if backend not in _VALID_BACKENDS:
        raise NitrixBackendError(
            f'backend={backend!r} not in {_VALID_BACKENDS!r}'
        )
    if backend == 'auto':
        env = env_backend()
        if env is not None and env != 'auto':
            backend = env
        else:
            return auto_backend()
    if backend == 'pallas-cuda':
        if not _HAS_AMPERE_NVIDIA:
            raise NitrixBackendError(
                "backend='pallas-cuda' requested, but no Ampere or newer "
                'NVIDIA GPU is visible to JAX. Use backend="jax" or '
                'backend="auto", or run on supported hardware.'
            )
        return 'pallas-cuda'
    return 'jax'


# ---------------------------------------------------------------------------
# Fallback warning machinery
# ---------------------------------------------------------------------------


_warned_keys: set[tuple[str, str, str, str]] = set()
_warned_lock = threading.Lock()


def _shape_signature(*shapes: Any) -> str:
    """Stringify shapes for the deduplication key.

    Parameters
    ----------
    *shapes
        Variadic shape descriptors. Each may be ``None``, a tuple or
        list of dimension sizes, or any other object; tuples and lists
        are rendered as their dimensions joined by ``'x'``, and other
        values via :func:`str`.

    Returns
    -------
    str
        A single string encoding every argument, with per-shape
        descriptors joined by ``';'``.
    """
    parts = []
    for s in shapes:
        if s is None:
            parts.append('None')
        elif isinstance(s, (tuple, list)):
            parts.append('x'.join(str(d) for d in s))
        else:
            parts.append(str(s))
    return ';'.join(parts)


def fallback(
    *,
    function: str,
    requested: ResolvedBackend,
    resolved: ResolvedBackend,
    reason: str,
    shapes: tuple[Any, ...] = (),
    dtype: Any = None,
) -> ResolvedBackend:
    """Emit a fallback warning if appropriate and return the resolved backend.

    Parameters
    ----------
    function
        Public function name (e.g. ``'semiring_matmul'``).
    requested
        The backend that was attempted.
    resolved
        The backend that will actually run.
    reason
        Short, user-facing explanation.
    shapes
        Tuple of input shape tuples (or scalar descriptors) for the dedupe key.
    dtype
        The relevant dtype (or ``None``) for the dedupe key.

    Returns
    -------
    {'pallas-cuda', 'jax'}
        The ``resolved`` backend, returned unchanged so the caller can
        proceed on the backend that will actually run.

    Raises
    ------
    NitrixBackendError
        If ``NITRIX_STRICT_BACKEND=1`` and ``requested`` differs from
        ``resolved`` (a fallback is escalated to an error).
    """
    if requested == resolved:
        return resolved
    if strict_backend():
        raise NitrixBackendError(
            f'{function}: requested backend={requested!r} but would fall '
            f'back to {resolved!r} ({reason}). NITRIX_STRICT_BACKEND=1 '
            'converted this to an error.'
        )
    if silence_fallback():
        return resolved
    key = (
        function,
        _shape_signature(*shapes),
        str(dtype),
        requested,
    )
    with _warned_lock:
        if key in _warned_keys:
            return resolved
        _warned_keys.add(key)
    warnings.warn(
        f'{function}: falling back to backend={resolved!r} from '
        f'{requested!r}: {reason}. '
        'Set NITRIX_SILENCE_FALLBACK=1 to suppress this warning, or '
        'NITRIX_STRICT_BACKEND=1 to convert it to an error.',
        category=NitrixBackendFallback,
        stacklevel=3,
    )
    return resolved


def reset_fallback_state() -> None:
    """Forget all previously-issued fallback warnings.

    Useful in tests, and as the public way to re-trigger a warning after
    silencing during library bring-up.
    """
    with _warned_lock:
        _warned_keys.clear()


__all__ = [
    'Backend',
    'ResolvedBackend',
    'NitrixBackendFallback',
    'NitrixBackendError',
    'auto_backend',
    'default_backend_is_gpu',
    'env_backend',
    'fallback',
    'resolve_backend',
    'reset_fallback_state',
    'silence_fallback',
    'strict_backend',
]
