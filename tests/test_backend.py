# -*- coding: utf-8 -*-
"""Tests for ``nitrix._internal.backend``: resolution and fallback observability."""

import warnings

import jax.numpy as jnp
import pytest

from nitrix._internal.backend import (
    NitrixBackendError,
    NitrixBackendFallback,
    env_backend,
    fallback,
    reset_fallback_state,
    resolve_backend,
)


def test_explicit_jax_resolves_to_jax():
    assert resolve_backend('jax') == 'jax'


def test_explicit_auto_resolves_to_concrete():
    out = resolve_backend('auto')
    assert out in ('pallas-cuda', 'jax')


def test_invalid_backend_raises():
    with pytest.raises(NitrixBackendError):
        resolve_backend('weird-backend')  # type: ignore[arg-type]


def test_fallback_emits_warning_once(monkeypatch):
    reset_fallback_state()
    monkeypatch.delenv('NITRIX_SILENCE_FALLBACK', raising=False)
    monkeypatch.delenv('NITRIX_STRICT_BACKEND', raising=False)

    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter('always')
        out = fallback(
            function='test_op',
            requested='pallas-cuda',
            resolved='jax',
            reason='not tileable',
            shapes=((4, 8), (8, 16)),
            dtype=jnp.float32,
        )
        assert out == 'jax'
        n_first = sum(1 for w in ws if w.category is NitrixBackendFallback)

        out = fallback(
            function='test_op',
            requested='pallas-cuda',
            resolved='jax',
            reason='not tileable',
            shapes=((4, 8), (8, 16)),
            dtype=jnp.float32,
        )
        n_second = sum(1 for w in ws if w.category is NitrixBackendFallback)

    assert n_first == 1
    assert n_second == 1, 'dedupe should keep a single warning'


def test_fallback_dedupes_per_signature():
    reset_fallback_state()
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter('always')
        fallback(
            function='op',
            requested='pallas-cuda',
            resolved='jax',
            reason='x',
            shapes=((8, 8),),
            dtype=jnp.float32,
        )
        fallback(
            function='op',
            requested='pallas-cuda',
            resolved='jax',
            reason='x',
            shapes=((16, 16),),
            dtype=jnp.float32,
        )
        n = sum(1 for w in ws if w.category is NitrixBackendFallback)
    # Different shapes -> two distinct keys -> two warnings.
    assert n == 2


def test_fallback_silenced_by_env(monkeypatch):
    reset_fallback_state()
    monkeypatch.setenv('NITRIX_SILENCE_FALLBACK', '1')
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter('always')
        fallback(
            function='op',
            requested='pallas-cuda',
            resolved='jax',
            reason='silenced',
            shapes=((1,),),
            dtype=jnp.float32,
        )
        n = sum(1 for w in ws if w.category is NitrixBackendFallback)
    assert n == 0


def test_strict_backend_converts_to_error(monkeypatch):
    reset_fallback_state()
    monkeypatch.setenv('NITRIX_STRICT_BACKEND', '1')
    with pytest.raises(NitrixBackendError):
        fallback(
            function='op',
            requested='pallas-cuda',
            resolved='jax',
            reason='strict',
            shapes=((1,),),
            dtype=jnp.float32,
        )


def test_env_backend_validation(monkeypatch):
    monkeypatch.setenv('NITRIX_BACKEND', 'bogus')
    with pytest.raises(NitrixBackendError):
        env_backend()


def test_env_backend_round_trip(monkeypatch):
    monkeypatch.setenv('NITRIX_BACKEND', 'jax')
    assert env_backend() == 'jax'
    monkeypatch.setenv('NITRIX_BACKEND', 'pallas-cuda')
    assert env_backend() == 'pallas-cuda'
