# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""P4 completeness guard for the reproducible-dispatch principle.

The structural backstop that keeps the principle alive past the branch that
introduced it (sibling to ``test_op_matrix_completeness``).  It statically scans
``src/nitrix`` and enforces, by AST:

1. **op-string consistency** -- the set of ``op=`` strings passed to
   ``resolve_driver`` equals the registered set (``nitrix.divergent_ops()``):
   no call site resolves an unregistered op, and no registration is an orphan.

2. **no ungoverned platform flip** -- every call site of
   ``default_backend_is_gpu()`` lives in a module on the governed allowlist
   below.  A *new* module branching on the platform must either route the
   choice through ``resolve_driver`` (and register the op + add a P3 runner) or,
   if it is genuinely not a numerically-divergent algorithm choice, be added
   here with a justification.  This is what makes "silent perf-divergent
   dispatch" fail CI rather than ship.

Scope/limitation: the platform-flip check is file-level (it catches a new
subsystem introducing a silent flip, the main regression vector); a second bare
flip added *within* an already-governed module is a reviewer concern, not caught
here.  See ``docs/feature-requests/reproducible-dispatch.md``.
"""

from __future__ import annotations

import ast
import pathlib

from nitrix import divergent_ops

_SRC = pathlib.Path(__file__).resolve().parents[1] / 'src'
_NITRIX = _SRC / 'nitrix'

# Modules permitted to call ``default_backend_is_gpu()`` -- each routes the
# platform choice into a ``resolve_driver(fast=...)`` callable (governed), with
# the one documented exception noted.
_PLATFORM_FLIP_ALLOWLIST = {
    'nitrix/nn/ssm/_reference.py',  # selective_scan fast= callable
    'nitrix/signal/_iir.py',  # iir fast= callable + the FFT-too-sharp fallback
    'nitrix/geometry/_interpolate.py',  # cubic B-spline prefilter fast=
    'nitrix/register/_svf.py',  # _smooth_fast (field regulariser) fast=
    'nitrix/metrics/information.py',  # joint-histogram fast=
}


def _scan():
    """AST-scan ``src/nitrix`` for the two governed call kinds."""
    resolve_ops: set[str] = set()
    flip_files: set[str] = set()
    for path in _NITRIX.rglob('*.py'):
        rel = str(path.relative_to(_SRC))
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, 'id', None) or getattr(
                node.func, 'attr', None
            )
            if name == 'resolve_driver':
                for kw in node.keywords:
                    if kw.arg == 'op' and isinstance(kw.value, ast.Constant):
                        resolve_ops.add(kw.value.value)
            elif name == 'default_backend_is_gpu':
                flip_files.add(rel)
    return resolve_ops, flip_files


def test_resolve_driver_ops_match_registry():
    """Every resolved op is registered; every registration is used (no orphan)."""
    resolve_ops, _ = _scan()
    registered = {o.op for o in divergent_ops()}
    unregistered = resolve_ops - registered
    orphaned = registered - resolve_ops
    assert not unregistered, (
        f'resolve_driver(op=...) for unregistered op(s) {sorted(unregistered)}; '
        'register them in nitrix._internal._divergent_ops.'
    )
    assert not orphaned, (
        f'registered divergent op(s) {sorted(orphaned)} are never resolved; '
        'wire a resolve_driver call site or drop the registration.'
    )


def test_no_ungoverned_platform_flip():
    """No module flips on the platform outside the governed allowlist."""
    _, flip_files = _scan()
    ungoverned = flip_files - _PLATFORM_FLIP_ALLOWLIST
    assert not ungoverned, (
        f'default_backend_is_gpu() called in ungoverned module(s) '
        f'{sorted(ungoverned)}. Route the choice through resolve_driver (and '
        'register the divergent op + add a P3 contract runner), or -- if this '
        'is not a numerically-divergent algorithm choice -- add the module to '
        '_PLATFORM_FLIP_ALLOWLIST in this test with a justification.'
    )


def test_allowlist_entries_still_call_the_probe():
    """Keep the allowlist honest: every entry actually exists and still flips
    (a stale entry that no longer calls the probe should be pruned)."""
    _, flip_files = _scan()
    stale = {
        m
        for m in _PLATFORM_FLIP_ALLOWLIST
        if (_SRC / m).exists() and m not in flip_files
    }
    assert not stale, (
        f'_PLATFORM_FLIP_ALLOWLIST entries no longer call '
        f'default_backend_is_gpu(): {sorted(stale)} -- prune them.'
    )
