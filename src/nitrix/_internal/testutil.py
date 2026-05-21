# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Utility functions for tests.
"""

from typing import Any, Callable, Dict, Optional

import jax
import pytest


def cfg_variants_test(
    base_fn: Callable[..., Any],
    jit_params: Optional[Dict[str, Any]] = None,
) -> Callable[[Callable[..., Any]], Any]:
    if jit_params is None:
        jit_params = {}

    def test_variants(test: Callable[..., Any]) -> Any:
        return pytest.mark.parametrize(
            'fn', [base_fn, jax.jit(base_fn, **jit_params)]
        )(test)

    return test_variants
