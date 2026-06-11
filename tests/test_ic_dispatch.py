# -*- coding: utf-8 -*-
"""V4a: the inverse-compositional fast path under rigid_register.

``rigid_register(method="auto")`` takes the inverse-compositional kernel (the
constant-template Hessian, ~4-7x the forward GN/LM throughput) when its
preconditions hold (IndexSpace + a least-squares / SSD metric), and the forward
path otherwise.  The forward path is the parity oracle: the two must recover the
same transform.
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
    rigid_exp,
    spatial_transform,
)
from nitrix.metrics import ncc  # noqa: E402
from nitrix.register import (  # noqa: E402
    LNCC,
    RegistrationSpec,
    WorldSpace,
    affine_register,
    rigid_register,
)


def _blobs(n=64, seed=0):
    grids = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
    rng = np.random.RandomState(seed)
    img = np.zeros((n, n), dtype='float64')
    for _ in range(5):
        c = rng.uniform(0.3, 0.7, 2) * n
        s = rng.uniform(0.1, 0.16) * n
        img += rng.uniform(0.4, 1.0) * np.exp(
            -sum((g - ci) ** 2 for g, ci in zip(grids, c)) / (2 * s * s)
        )
    return jnp.asarray(img)


def _rigid_pair(n=64):
    fixed = _blobs(n)
    center = (jnp.asarray((n, n), dtype=fixed.dtype) - 1.0) / 2.0
    grid = affine_grid(
        rigid_exp(jnp.asarray([0.15, 4.0, -3.0]), ndim=2),
        (n, n),
        center=center,
    )
    moving = spatial_transform(fixed[..., None], grid, mode='nearest')[..., 0]
    return moving, fixed


def test_ic_matches_forward():
    moving, fixed = _rigid_pair(64)
    spec = RegistrationSpec(levels=3, iterations=20)
    ic = rigid_register(
        moving, fixed, spec=spec, method='inverse_compositional'
    )
    fwd = rigid_register(moving, fixed, spec=spec, method='forward')
    assert float(ncc(ic.warped, fixed)) > 0.99
    assert float(ncc(fwd.warped, fixed)) > 0.99
    # the parity oracle: both recover the same alignment
    assert np.allclose(
        np.asarray(ic.warped), np.asarray(fwd.warped), atol=2e-2
    )
    assert np.allclose(
        np.asarray(ic.matrix), np.asarray(fwd.matrix), atol=5e-2
    )


def _affine_pair(n=64):
    fixed = _blobs(n)
    center = (jnp.asarray((n, n), dtype=fixed.dtype) - 1.0) / 2.0
    # modest rotation + scale + shear + translation (4 linear + 2 translation)
    params = jnp.asarray([0.04, -0.08, 0.06, -0.05, 3.0, -2.0])
    grid = affine_grid(affine_exp(params, ndim=2), (n, n), center=center)
    moving = spatial_transform(fixed[..., None], grid, mode='nearest')[..., 0]
    return moving, fixed


def test_affine_ic_matches_forward():
    moving, fixed = _affine_pair(64)
    spec = RegistrationSpec(levels=3, iterations=25)
    ic = affine_register(
        moving, fixed, spec=spec, method='inverse_compositional'
    )
    fwd = affine_register(moving, fixed, spec=spec, method='forward')
    assert float(ncc(ic.warped, fixed)) > 0.99
    assert float(ncc(fwd.warped, fixed)) > 0.99
    assert np.allclose(
        np.asarray(ic.warped), np.asarray(fwd.warped), atol=3e-2
    )


def test_affine_auto_uses_ic():
    moving, fixed = _affine_pair(64)
    spec = RegistrationSpec(levels=3, iterations=25)
    auto = affine_register(moving, fixed, spec=spec, method='auto')
    ic = affine_register(
        moving, fixed, spec=spec, method='inverse_compositional'
    )
    assert np.allclose(
        np.asarray(auto.warped), np.asarray(ic.warped), atol=1e-9
    )


def test_auto_uses_ic_for_index_ssd():
    moving, fixed = _rigid_pair(64)
    spec = RegistrationSpec(levels=3, iterations=20)
    auto = rigid_register(moving, fixed, spec=spec, method='auto')
    ic = rigid_register(
        moving, fixed, spec=spec, method='inverse_compositional'
    )
    # auto picks IC for the default IndexSpace + SSD, so results are identical
    assert np.allclose(
        np.asarray(auto.warped), np.asarray(ic.warped), atol=1e-9
    )


def test_ic_preconditions_validated():
    moving, fixed = _rigid_pair(32)
    # WorldSpace is not an IC frame
    with pytest.raises(ValueError):
        rigid_register(
            moving, fixed, space=WorldSpace(), method='inverse_compositional'
        )
    # LNCC is not a least-squares metric
    with pytest.raises(ValueError):
        rigid_register(
            moving,
            fixed,
            spec=RegistrationSpec(metric=LNCC(radius=2)),
            method='inverse_compositional',
        )
    # unknown method
    with pytest.raises(ValueError):
        rigid_register(moving, fixed, method='nope')


def test_per_level_iteration_schedule_ic():
    # coarse-to-fine schedule (front-load the cheap coarse levels); recovers,
    # and the per-level counts are honoured (IC cost_history = sum of schedule).
    moving, fixed = _rigid_pair(64)
    res = rigid_register(
        moving, fixed, spec=RegistrationSpec(levels=3, iterations=(40, 20, 10))
    )
    assert float(ncc(res.warped, fixed)) > 0.99
    assert res.cost_history.shape[0] == 70


def test_per_level_iteration_schedule_forward():
    moving, fixed = _rigid_pair(64)
    res = rigid_register(
        moving,
        fixed,
        spec=RegistrationSpec(levels=3, iterations=(40, 20, 10)),
        method='forward',
    )
    assert float(ncc(res.warped, fixed)) > 0.99
    # LM records an extra initial cost per level -> sum(schedule) + levels
    assert res.cost_history.shape[0] == 73


def test_iterations_schedule_length_validation():
    moving, fixed = _rigid_pair(32)
    with pytest.raises(ValueError):
        rigid_register(
            moving, fixed, spec=RegistrationSpec(levels=3, iterations=(40, 20))
        )


def test_auto_falls_back_to_forward_off_preconditions():
    # WorldSpace + auto -> forward path (no IC); still recovers.
    moving, fixed = _rigid_pair(64)
    res = rigid_register(
        moving,
        fixed,
        spec=RegistrationSpec(levels=3, iterations=20),
        space=WorldSpace(),
        method='auto',
    )
    assert float(ncc(res.warped, fixed)) > 0.99
