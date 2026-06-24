# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Tests for the reproducible-dispatch substrate (the ``driver`` axis).

P1 substrate only: the reproducibility mode (context manager + flag), the
divergent-op registry + introspection, and the ``resolve_driver`` resolver and
its precedence / validation.  The 5 real sites are wired in P2; here we exercise
the machinery against a registered *test* op so the substrate is verified in
isolation.
"""

from __future__ import annotations

import jax

jax.config.update('jax_enable_x64', True)

import pytest  # noqa: E402

from nitrix import (  # noqa: E402
    divergent_ops,
    reproducible,
    reproducible_enabled,
    set_reproducible,
)
from nitrix._internal.config import (  # noqa: E402
    _REGISTRY,
    register_divergent_op,
    resolve_driver,
)

_TEST_OP = 'test.dummy_scan'


@pytest.fixture
def dummy_op():
    """Register a divergent test op; remove it afterwards (clean registry)."""
    entry = register_divergent_op(
        _TEST_OP,
        canonical='sequential',
        fast={'gpu': 'associative', 'cpu': 'sequential'},
        driver_values=('sequential', 'associative'),
        tolerance={'float32': 1e-3, 'float64': 1e-11},
        summary='test-only dummy scan',
    )
    try:
        yield entry
    finally:
        _REGISTRY.pop(_TEST_OP, None)


@pytest.fixture(autouse=True)
def _reset_mode():
    """Ensure each test starts and ends with reproducibility mode off."""
    set_reproducible(False)
    yield
    set_reproducible(False)


# --- reproducibility mode (the intent switch) ----------------------------


def test_default_mode_is_off():
    assert reproducible_enabled() is False


def test_context_manager_scopes_and_restores():
    assert not reproducible_enabled()
    with reproducible():
        assert reproducible_enabled()
    assert not reproducible_enabled()


def test_context_manager_nests_and_can_disable():
    with reproducible():
        assert reproducible_enabled()
        with reproducible(False):  # carve a fast region out
            assert not reproducible_enabled()
        assert reproducible_enabled()
    assert not reproducible_enabled()


def test_set_reproducible_imperative():
    set_reproducible(True)
    assert reproducible_enabled()
    set_reproducible(False)
    assert not reproducible_enabled()


# --- resolver precedence: explicit > reproducible > fast -----------------


def test_explicit_driver_wins_over_everything(dummy_op):
    fast = lambda: 'associative'  # noqa: E731
    # explicit beats the fast default
    assert resolve_driver('sequential', op=_TEST_OP, fast=fast) == 'sequential'
    # explicit beats reproducible mode too (explicit is explicit)
    with reproducible():
        assert (
            resolve_driver('associative', op=_TEST_OP, fast=fast)
            == 'associative'
        )


def test_reproducible_forces_canonical(dummy_op):
    fast = lambda: 'associative'  # noqa: E731
    assert resolve_driver('auto', op=_TEST_OP, fast=fast) == 'associative'
    with reproducible():
        assert resolve_driver('auto', op=_TEST_OP, fast=fast) == 'sequential'


def test_auto_defers_to_fast_when_mode_off(dummy_op):
    assert resolve_driver('auto', op=_TEST_OP, fast=lambda: 'associative') == (
        'associative'
    )
    assert resolve_driver('auto', op=_TEST_OP, fast=lambda: 'sequential') == (
        'sequential'
    )


def test_none_is_treated_as_auto(dummy_op):
    assert resolve_driver(None, op=_TEST_OP, fast=lambda: 'associative') == (
        'associative'
    )


def test_fast_is_lazy_and_skipped_when_not_needed(dummy_op):
    def boom():
        raise AssertionError('fast() must not be called for explicit/repro')

    # explicit driver: fast not consulted
    assert resolve_driver('sequential', op=_TEST_OP, fast=boom) == 'sequential'
    # reproducible mode: fast not consulted
    with reproducible():
        assert resolve_driver('auto', op=_TEST_OP, fast=boom) == 'sequential'


# --- resolver validation -------------------------------------------------


def test_unknown_driver_raises_listing_valid(dummy_op):
    with pytest.raises(ValueError, match='driver='):
        resolve_driver('bogus', op=_TEST_OP, fast=lambda: 'associative')


def test_unregistered_op_raises(dummy_op):
    with pytest.raises(KeyError, match='not a registered'):
        resolve_driver('auto', op='test.nope', fast=lambda: 'associative')


# --- registry validation + introspection ---------------------------------


def test_registration_validates_canonical_in_values():
    with pytest.raises(ValueError, match='canonical'):
        register_divergent_op(
            'test.bad_canonical',
            canonical='nope',
            fast={'gpu': 'a'},
            driver_values=('a', 'b'),
            tolerance={},
        )
    _REGISTRY.pop('test.bad_canonical', None)


def test_registration_validates_fast_in_values():
    with pytest.raises(ValueError, match='fast'):
        register_divergent_op(
            'test.bad_fast',
            canonical='a',
            fast={'gpu': 'nope'},
            driver_values=('a', 'b'),
            tolerance={},
        )
    _REGISTRY.pop('test.bad_fast', None)


def test_registration_injects_auto_and_freezes(dummy_op):
    assert 'auto' in dummy_op.driver_values
    # tolerance / fast are read-only views
    with pytest.raises(TypeError):
        dummy_op.tolerance['float32'] = 9.0  # type: ignore[index]


def test_divergent_ops_lists_registered_sorted(dummy_op):
    ops = divergent_ops()
    names = [o.op for o in ops]
    assert _TEST_OP in names
    assert names == sorted(names)
    entry = next(o for o in ops if o.op == _TEST_OP)
    assert entry.canonical == 'sequential'
    assert entry.tolerance['float32'] == 1e-3


# --- jit-safety: the resolved variant is baked at trace time -------------


def test_resolver_is_jit_safe(dummy_op):
    import jax.numpy as jnp

    def f(x, driver):
        choice = resolve_driver(
            driver, op=_TEST_OP, fast=lambda: 'associative'
        )
        # branch on the trace-time-resolved string (jit-safe: it is concrete)
        if choice == 'sequential':
            return jnp.sum(x)
        return jnp.sum(x) * 2.0

    x = jnp.arange(4.0)
    seq = jax.jit(lambda x: f(x, 'sequential'))(x)
    assoc = jax.jit(lambda x: f(x, 'associative'))(x)
    assert float(seq) == float(jnp.sum(x))
    assert float(assoc) == float(jnp.sum(x) * 2.0)
    # reproducible mode around the trace flips the auto branch to canonical
    with reproducible():
        repro = jax.jit(lambda x: f(x, 'auto'))(x)
    assert float(repro) == float(jnp.sum(x))
