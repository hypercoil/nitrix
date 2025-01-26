# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Utility functions for tests.
"""
import pytest
import jax


def cfg_variants_test(base_fn: callable, jit_params = None):
    if jit_params is None:
        jit_params = {}
    def test_variants(test: callable):
        return pytest.mark.parametrize(
            'fn', [base_fn, jax.jit(base_fn, **jit_params)]
        )(test)
    return test_variants
