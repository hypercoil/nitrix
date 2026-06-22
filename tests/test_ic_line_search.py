# -*- coding: utf-8 -*-
"""F1: opt-in cost-decrease (backtracking) guard on the inverse-compositional step.

The IC Gauss-Newton step is trust-region-clamped but has no cost-decrease check,
so a step from the *constant* template Hessian can still increase the SSD on a
hard case (no monotonicity guarantee).  ``RegistrationSpec(ic_line_search=True)``
backtracks along the clamped direction and accepts the largest fraction that
decreases the cost (else leaves the iterate unmoved), making the per-level cost
**monotone non-increasing** -- at identical recovery.  It is off by default (the
candidate warps ~3x the IC step on GPU); the default path is unchanged.
"""

from __future__ import annotations

import jax

jax.config.update('jax_enable_x64', True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from nitrix.geometry import (  # noqa: E402
    affine_exp,
    affine_grid,
    spatial_transform,
)
from nitrix.metrics import ncc  # noqa: E402
from nitrix.register import (  # noqa: E402
    RegistrationSpec,
    affine_register,
)

_LEVELS, _ITERS = 3, 40


def _blobs(n=64, seed=0):
    rng = np.random.RandomState(seed)
    g = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
    img = np.zeros((n, n))
    for _ in range(6):
        c = rng.uniform(0.25, 0.75, 2) * n
        s = rng.uniform(0.08, 0.14) * n
        img += rng.uniform(0.4, 1.0) * np.exp(
            -sum((gi - ci) ** 2 for gi, ci in zip(g, c)) / (2 * s * s)
        )
    return jnp.asarray(img / img.max())


def _pair(params, n=64, seed=0):
    fixed = _blobs(n, seed)
    c = (jnp.asarray((n, n), dtype=fixed.dtype) - 1.0) / 2.0
    grid = affine_grid(
        affine_exp(jnp.asarray(params), ndim=2), (n, n), center=c
    )
    moving = spatial_transform(fixed[..., None], grid, mode='nearest')[..., 0]
    return moving, fixed


def _max_within_level_increase(cost_history):
    ch = np.asarray(cost_history).reshape(_LEVELS, _ITERS)
    return max(float(np.diff(ch[lvl]).max()) for lvl in range(_LEVELS))


# the stiff affine is where the constant-template Hessian can propose an ascent
# step; the easy one already (almost) decreases every step.
_CASES = [
    pytest.param([0.04, -0.08, 0.06, -0.05, 3.0, -2.0], id='easy'),
    pytest.param([0.18, -0.22, 0.20, -0.16, 6.0, -5.0], id='hard'),
]


@pytest.mark.parametrize('params', _CASES)
def test_ic_line_search_is_monotone(params):
    moving, fixed = _pair(params)
    res = affine_register(
        moving,
        fixed,
        spec=RegistrationSpec(
            levels=_LEVELS, iterations=_ITERS, ic_line_search=True
        ),
        method='inverse_compositional',
    )
    # strictly monotone non-increasing within each pyramid level (the guarantee).
    assert _max_within_level_increase(res.cost_history) <= 1e-9


@pytest.mark.parametrize('params', _CASES)
def test_ic_line_search_recovers_like_default(params):
    moving, fixed = _pair(params)
    spec = RegistrationSpec(levels=_LEVELS, iterations=_ITERS)
    default = affine_register(
        moving, fixed, spec=spec, method='inverse_compositional'
    )
    guarded = affine_register(
        moving,
        fixed,
        spec=RegistrationSpec(
            levels=_LEVELS, iterations=_ITERS, ic_line_search=True
        ),
        method='inverse_compositional',
    )
    # same optimum (no recovery regression from the guard).
    assert float(ncc(guarded.warped, fixed)) > 0.99
    assert np.allclose(
        np.asarray(guarded.warped), np.asarray(default.warped), atol=2e-2
    )


def test_default_is_unguarded_fast_path():
    # The default (flag off) does NOT pay the guard: it need not be monotone
    # (the very non-monotonicity F1 fixes when enabled), confirming the guard is
    # genuinely opt-in and the fast path is untouched.
    moving, fixed = _pair([0.18, -0.22, 0.20, -0.16, 6.0, -5.0])
    res = affine_register(
        moving,
        fixed,
        spec=RegistrationSpec(levels=_LEVELS, iterations=_ITERS),
        method='inverse_compositional',
    )
    assert float(ncc(res.warped, fixed)) > 0.99  # still recovers
    # the unguarded step is allowed to (and here does) transiently increase.
    assert _max_within_level_increase(res.cost_history) > 1e-9
