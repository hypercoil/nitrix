# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Reproducible-dispatch substrate (the ``driver`` axis).

nitrix has two orthogonal dispatch axes.  ``_internal.backend`` governs the
*execution-engine* axis (``backend=`` -- ``pallas-cuda`` vs ``jax``).  This
module governs the *numerical-variant* axis (``driver=``): where one operation
has more than one numerically-**divergent** implementation of the same math
(e.g. a sequential ``lax.scan`` vs a parallel ``lax.associative_scan``, which
reassociate floating-point differently, or a deterministic one-hot histogram vs
a non-associative atomic scatter-add), this is where the choice is resolved.

See ``docs/feature-requests/reproducible-dispatch.md`` for the design principle.
The contract, in brief:

- The default is **hardware-aware** (fastest correct variant per platform).
- A per-call ``driver=`` keyword overrides it (the scipy / torch convention;
  distinct from ``backend=``).
- A single library-level **reproducibility mode** -- ``NITRIX_REPRODUCIBLE=1``
  (env) or ``with nitrix.reproducible():`` (scoped) -- forces the **canonical**
  variant at *every* such site, trading peak performance for cross-platform /
  cross-run stability up to each site's documented tolerance.
- Every divergent site is a **registered contract**
  (:func:`register_divergent_op`); :func:`divergent_ops` lists them and the
  completeness guard fails CI if a site is unregistered.

Resolution precedence (mirrors :func:`resolve_backend`)::

    driver= (per-call, not 'auto')  >  reproducible mode  >  hardware-auto

The reproducibility flag is read at **trace** time (a ``contextvars.ContextVar``,
like ``jax.config`` flags and :func:`default_backend_is_gpu`): a function traced
outside ``reproducible()`` and called inside (from the ``jit`` cache) keeps its
traced variant.  Set the mode before first trace -- the deployment env var is the
robust path.
"""

from __future__ import annotations

import contextlib
import contextvars
import os
from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Iterator, Mapping

__all__ = [
    'DivergentOp',
    'divergent_ops',
    'register_divergent_op',
    'reproducible',
    'reproducible_enabled',
    'resolve_driver',
    'set_reproducible',
]


# ---------------------------------------------------------------------------
# Reproducibility mode (the intent switch)
# ---------------------------------------------------------------------------


def _env_flag(name: str) -> bool:
    return os.environ.get(name, '0').strip().lower() in (
        '1',
        'true',
        'yes',
        'on',
    )


# Trace-time-readable; contextvars scope correctly across threads and async.
# The base value is seeded once from the environment at import (matching the
# ``NITRIX_BACKEND`` convention); ``reproducible()`` / ``set_reproducible``
# override it per-context / globally.
_REPRODUCIBLE: contextvars.ContextVar[bool] = contextvars.ContextVar(
    'nitrix_reproducible',
    default=_env_flag('NITRIX_REPRODUCIBLE'),
)


def reproducible_enabled() -> bool:
    """Report whether reproducibility mode is currently active.

    This is the trace-time signal that divergent dispatch sites consult:
    it reads the context-scoped reproducibility flag as seen at the point of
    the call, so a function that is traced inside a :func:`reproducible` block
    observes ``True``.

    Returns
    -------
    bool
        ``True`` if reproducibility mode is active in the current context,
        ``False`` otherwise.
    """
    return _REPRODUCIBLE.get()


def set_reproducible(enabled: bool = True) -> None:
    """Set reproducibility mode for the current context (un-scoped).

    Prefer :func:`reproducible` (the context manager) where the scope is
    known; this is the imperative form for notebook / session-level use.

    Parameters
    ----------
    enabled
        Whether to activate reproducibility mode. When ``True`` (the default),
        subsequent divergent dispatch sites resolve to their canonical
        variant; when ``False``, they resolve to the hardware-optimal pick.
    """
    _REPRODUCIBLE.set(bool(enabled))


@contextlib.contextmanager
def reproducible(enabled: bool = True) -> Iterator[None]:
    """Force the canonical variant at every divergent dispatch site in scope.

    Within the ``with`` block, every :func:`resolve_driver` call whose own
    ``driver=`` is left at ``'auto'`` resolves to the site's **canonical**
    variant rather than the hardware-optimal one -- so results are stable
    across platforms (and deterministic where the default is not) up to each
    site's documented tolerance.  ``with reproducible(False):`` carves a
    fast region out of an otherwise-reproducible deployment.

    Must wrap the *trace* (see the module docstring's ``jit`` note).

    Parameters
    ----------
    enabled
        Whether the block runs with reproducibility mode active. ``True``
        (the default) forces canonical variants within the block; ``False``
        forces the hardware-optimal picks, carving a fast region out of an
        otherwise-reproducible deployment.

    Yields
    ------
    None
        The context manager yields no value; it exists for its scoped effect
        on the reproducibility flag, which is restored to its previous state
        on exit.
    """
    token = _REPRODUCIBLE.set(bool(enabled))
    try:
        yield
    finally:
        _REPRODUCIBLE.reset(token)


# ---------------------------------------------------------------------------
# The divergent-op registry (the contract)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DivergentOp:
    """A registered numerically-divergent dispatch site (a contract).

    Attributes
    ----------
    op
        Dotted public name, e.g. ``'nn.ssm.selective_scan'``.
    canonical
        The reference variant reproducibility mode forces -- the one the
        golden corpus is pinned to (chosen for faithfulness, not always the
        CPU default).
    fast
        Descriptive map of the hardware-auto pick, e.g.
        ``{'gpu': 'associative', 'cpu': 'sequential'}``.  This is for
        introspection only; the *runtime* fast pick (which may also depend on
        shape / parameters) is the ``fast`` callable each site passes to
        :func:`resolve_driver`.
    driver_values
        The accepted ``driver=`` values (always including ``'auto'``).
    tolerance
        Per-dtype cross-variant budget (``variant ~= canonical``), mirroring
        ``tests/tolerance.toml``; e.g. ``{'float32': 2e-3}``.
    summary
        One-line human description of the divergence.
    """

    op: str
    canonical: str
    fast: Mapping[str, str]
    driver_values: tuple[str, ...]
    tolerance: Mapping[str, float]
    summary: str = ''


_REGISTRY: dict[str, DivergentOp] = {}


def register_divergent_op(
    op: str,
    *,
    canonical: str,
    fast: Mapping[str, str],
    driver_values: tuple[str, ...],
    tolerance: Mapping[str, float],
    summary: str = '',
) -> DivergentOp:
    """Register a divergent dispatch site; validated and idempotent-by-name.

    ``'auto'`` is added to ``driver_values`` if absent.  ``canonical`` and
    every ``fast`` value must be in ``driver_values`` (a registration-time
    invariant, so :func:`resolve_driver` never returns an unlisted variant).
    Registering the same ``op`` again overwrites the previous entry.

    Parameters
    ----------
    op
        Dotted public name of the site, e.g. ``'nn.ssm.selective_scan'``.
        Used as the registry key.
    canonical
        The reference variant that reproducibility mode forces. Must be a
        concrete value present in ``driver_values`` (never ``'auto'``).
    fast
        Descriptive map from platform label (e.g. ``'gpu'``, ``'cpu'``) to
        the variant the hardware-auto default picks there. Every value must
        be a concrete member of ``driver_values`` (never ``'auto'``). This
        is a description for introspection, not the runtime selector.
    driver_values
        The accepted ``driver=`` values for this site. ``'auto'`` is prepended
        automatically if not already present.
    tolerance
        Per-dtype cross-variant budget mapping a dtype name to a numerical
        tolerance, e.g. ``{'float32': 2e-3}``.
    summary
        One-line human-readable description of the divergence. Defaults to an
        empty string.

    Returns
    -------
    DivergentOp
        The frozen registry entry that was created and stored, with ``'auto'``
        normalised into ``driver_values`` and the mappings made read-only.

    Raises
    ------
    ValueError
        If ``canonical`` is ``'auto'`` or absent from ``driver_values``, or if
        any ``fast`` value is ``'auto'`` or absent from ``driver_values``.
    """
    values = tuple(driver_values)
    if 'auto' not in values:
        values = ('auto',) + values
    if canonical == 'auto' or canonical not in values:
        raise ValueError(
            f'register_divergent_op({op!r}): canonical={canonical!r} must be '
            f'a concrete value in {values!r}.'
        )
    for platform, value in fast.items():
        if value not in values or value == 'auto':
            raise ValueError(
                f'register_divergent_op({op!r}): fast[{platform!r}]={value!r} '
                f'must be a concrete value in {values!r}.'
            )
    entry = DivergentOp(
        op=op,
        canonical=canonical,
        fast=MappingProxyType(dict(fast)),
        driver_values=values,
        tolerance=MappingProxyType(dict(tolerance)),
        summary=summary,
    )
    _REGISTRY[op] = entry
    return entry


def divergent_ops() -> tuple[DivergentOp, ...]:
    """Every registered numerically-divergent dispatch site, sorted by name.

    The discoverability surface: rather than emit a runtime warning on the
    (correct) default path, nitrix lets a parity-sensitive consumer enumerate
    exactly which operations diverge across platforms and within what
    tolerance, and pin them via ``driver=`` or :func:`reproducible`.

    Returns
    -------
    tuple of DivergentOp
        Every registered dispatch site, ordered by dotted op name.
    """
    return tuple(_REGISTRY[name] for name in sorted(_REGISTRY))


# ---------------------------------------------------------------------------
# The resolver (the single seam every divergent site calls)
# ---------------------------------------------------------------------------


def resolve_driver(
    driver: str,
    *,
    op: str,
    fast: Callable[[], str],
) -> str:
    """Resolve a ``driver=`` request to a concrete variant for ``op``.

    Precedence: an explicit ``driver`` (anything but ``'auto'``) wins; else
    the **canonical** variant under reproducibility mode; else ``fast()`` --
    the site's hardware-/shape-aware pick, evaluated lazily (so its trace-time
    probe runs only when actually needed).

    Parameters
    ----------
    driver
        The per-call request; ``'auto'`` (or ``None``) defers to mode / fast.
    op
        The registered op name (must be registered, else a programming error).
    fast
        Zero-arg callable returning the hardware-optimal variant; called only
        when ``driver='auto'`` and reproducibility mode is off.

    Returns
    -------
    A concrete variant string from ``op``'s registered ``driver_values``.

    Raises
    ------
    KeyError
        If ``op`` is not registered (an internal contract violation -- every
        divergent site must register; enforced by the P4 completeness guard).
    ValueError
        If ``driver`` is neither ``'auto'`` nor a registered value.
    """
    entry = _REGISTRY.get(op)
    if entry is None:
        raise KeyError(
            f'resolve_driver: op {op!r} is not a registered divergent op; '
            'call register_divergent_op at import.'
        )
    resolved = 'auto' if driver is None else driver
    if resolved != 'auto':
        if resolved not in entry.driver_values:
            raise ValueError(
                f'{op}: driver={driver!r} not in {entry.driver_values!r}.'
            )
        return resolved
    if reproducible_enabled():
        return entry.canonical
    return fast()
