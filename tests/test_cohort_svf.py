# -*- coding: utf-8 -*-
"""D4: cohort SVF registration via ``jax.vmap`` over the recipe.

A diffeomorphic recipe is a pure ``(moving, fixed) -> result`` function, so a
**cohort** (many moving images to one reference) is just ``jax.vmap`` of the
recipe -- and the batch-aggregate early-exit comes for free: a ``vmap``-ed
``lax.while_loop`` runs to the **all-lanes** exit (the slowest subject sets the
trip count), exactly the ``volreg`` pattern.  No dedicated cohort driver is
needed.  Gates: vmap'd Demons / SyN recover every subject's planted warp under
both ``mode='fixed'`` (scan) and ``mode='early_exit'`` (the cohort while_loop),
staying diffeomorphic.
"""

from __future__ import annotations

import jax

jax.config.update('jax_enable_x64', True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from nitrix.geometry import (  # noqa: E402
    identity_grid,
    integrate_velocity_field,
    spatial_transform,
)
from nitrix.metrics import ncc  # noqa: E402
from nitrix.register import (  # noqa: E402
    Convergence,
    DemonsSpec,
    SyNSpec,
    diffeomorphic_demons_register,
    greedy_syn_register,
)
from nitrix.smoothing import gaussian  # noqa: E402


def _blobs(n=64):
    yy, xx = np.mgrid[0:n, 0:n].astype('float64')
    img = np.zeros((n, n))
    for cy, cx, s, a in [
        (0.31, 0.38, 0.13, 1.0),
        (0.62, 0.69, 0.16, 0.7),
        (0.72, 0.31, 0.11, 0.6),
        (0.44, 0.75, 0.14, 0.5),
    ]:
        img += a * np.exp(
            -((xx - cx * n) ** 2 + (yy - cy * n) ** 2) / (2 * (s * n) ** 2)
        )
    return jnp.asarray(img / img.max())


def _smooth_velocity(shape, sigma, scale, seed):
    rng = np.random.RandomState(seed)
    v = rng.standard_normal(shape + (2,))
    v = np.moveaxis(v, -1, 0)
    v = np.asarray(gaussian(jnp.asarray(v), sigma=sigma, spatial_rank=2))
    return jnp.asarray(scale * np.moveaxis(v, 0, -1))


def _warp(image, v):
    grid = identity_grid(image.shape, dtype=image.dtype)
    s = integrate_velocity_field(v)
    return spatial_transform(image[..., None], grid + s, mode='nearest')[
        ..., 0
    ]


def _cohort(n=64, k=3):
    fixed = _blobs(n)
    # k subjects, each a distinct planted smooth warp of the shared reference.
    moving = jnp.stack(
        [
            _warp(fixed, _smooth_velocity((n, n), 8.0, 22.0, s))
            for s in range(k)
        ]
    )
    return moving, fixed


@pytest.mark.parametrize(
    'spec',
    [
        pytest.param(DemonsSpec(levels=2, iterations=30), id='demons-fixed'),
        pytest.param(
            SyNSpec(levels=2, iterations=30, step=0.5), id='syn-fixed'
        ),
    ],
)
def test_vmap_cohort_recovers_each_subject(spec):
    moving, fixed = _cohort()
    recipe = (
        diffeomorphic_demons_register
        if isinstance(spec, DemonsSpec)
        else greedy_syn_register
    )
    warped = jax.vmap(lambda m: recipe(m, fixed, spec=spec).warped)(moving)
    assert warped.shape == moving.shape
    for i in range(moving.shape[0]):
        assert float(ncc(warped[i], fixed)) > 0.99


def test_vmap_cohort_early_exit_batch_aggregate():
    # The cohort while_loop runs to the all-lanes exit (volreg pattern); each
    # subject still recovers, and the result is diffeomorphic per subject.
    moving, fixed = _cohort()
    spec = DemonsSpec(
        levels=2,
        iterations=40,
        mode='early_exit',
        convergence=Convergence(threshold=1e-3, window=6),
    )

    def one(m):
        res = diffeomorphic_demons_register(m, fixed, spec=spec)
        return res.warped, res.jacobian_det

    warped, jdet = jax.vmap(one)(moving)
    for i in range(moving.shape[0]):
        assert float(ncc(warped[i], fixed)) > 0.99
        assert float(jdet[i].min()) > 0.0  # diffeomorphic per subject


def test_vmap_cohort_jits():
    # The cohort map compiles + runs under jit (the production path).
    moving, fixed = _cohort(n=48, k=2)
    spec = DemonsSpec(levels=2, iterations=20)
    fn = jax.jit(
        lambda ms: jax.vmap(
            lambda m: diffeomorphic_demons_register(m, fixed, spec=spec).warped
        )(ms)
    )
    warped = fn(moving)
    assert bool(jnp.all(jnp.isfinite(warped)))
