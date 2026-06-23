# -*- coding: utf-8 -*-
"""E2: deterministic joint-histogram (one-hot matmul) at affine sizes.

The joint-histogram ``.at[].add`` scatter is a non-associative atomic add whose
float result depends on the GPU run -- the source of the affine-MI
non-determinism that ``affine_register(restarts=k)`` otherwise papers over.
Below ``_ONEHOT_HIST_MAX_VOXELS`` the histogram is built as a one-hot matmul
(``ohₘᵀ @ ohf``), a deterministic reduction.  Gates: the one-hot path equals the
scatter path (parity); affine-MI on the GPU is run-to-run bit-identical without
``restarts`` (the determinism contract, gated to the L4).
"""

from __future__ import annotations

import jax

jax.config.update('jax_enable_x64', True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import nitrix.metrics.information as _info  # noqa: E402
from nitrix.geometry import (  # noqa: E402
    affine_exp,
    affine_grid,
    spatial_transform,
)
from nitrix.metrics import joint_histogram, mutual_information  # noqa: E402
from nitrix.register import MI, RegistrationSpec, affine_register  # noqa: E402

_ON_GPU = jax.default_backend() == 'gpu'
_RM = (0.0, 1.0)
_gpu_only = pytest.mark.skipif(not _ON_GPU, reason='requires a GPU (L4)')


def _blobs(n, seed=0):
    rng = np.random.RandomState(seed)
    g = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
    img = np.zeros((n, n))
    for _ in range(7):
        c = rng.uniform(0.25, 0.75, 2) * n
        s = rng.uniform(0.08, 0.14) * n
        img += rng.uniform(0.4, 1.0) * np.exp(
            -sum((gi - ci) ** 2 for gi, ci in zip(g, c)) / (2 * s * s)
        )
    return jnp.asarray(img / img.max())


# ---------------------------------------------------------------------------
# Parity: the one-hot path equals the scatter path (and the analytic histogram)
# ---------------------------------------------------------------------------


def test_onehot_matches_scatter(monkeypatch):
    rng = np.random.RandomState(0)
    a = jnp.asarray(rng.rand(48, 48))
    b = jnp.asarray(rng.rand(48, 48))
    # force the one-hot branch regardless of backend (it is GPU-gated in prod).
    monkeypatch.setattr(_info, 'default_backend_is_gpu', lambda: True)
    onehot = np.asarray(  # N = 2304 << gate -> one-hot
        joint_histogram(a, b, bins=24, range_moving=_RM, range_fixed=_RM)
    )
    monkeypatch.setattr(_info, '_ONEHOT_HIST_MAX_VOXELS', 0)  # force scatter
    scatter = np.asarray(
        joint_histogram(a, b, bins=24, range_moving=_RM, range_fixed=_RM)
    )
    assert np.allclose(onehot, scatter, atol=1e-12)
    assert np.allclose(onehot.sum(), 1.0)


def test_onehot_mi_matches_scatter_mi(monkeypatch):
    rng = np.random.RandomState(1)
    a = jnp.asarray(rng.rand(40, 40))
    b = jnp.asarray(rng.rand(40, 40) ** 2)
    monkeypatch.setattr(_info, 'default_backend_is_gpu', lambda: True)
    onehot = float(
        mutual_information(a, b, bins=32, range_moving=_RM, range_fixed=_RM)
    )
    monkeypatch.setattr(_info, '_ONEHOT_HIST_MAX_VOXELS', 0)
    scatter = float(
        mutual_information(a, b, bins=32, range_moving=_RM, range_fixed=_RM)
    )
    assert np.allclose(onehot, scatter, atol=1e-12)


def test_onehot_mask_matches_scatter_mask(monkeypatch):
    # the deterministic path honours the A3 mask too (gated scatter).
    rng = np.random.RandomState(2)
    a = jnp.asarray(rng.rand(40, 40))
    b = jnp.asarray(rng.rand(40, 40))
    m = np.ones((40, 40))
    m[:14, :] = 0.0
    m = jnp.asarray(m)
    monkeypatch.setattr(_info, 'default_backend_is_gpu', lambda: True)
    onehot = np.asarray(
        joint_histogram(
            a, b, bins=24, range_moving=_RM, range_fixed=_RM, mask=m
        )
    )
    monkeypatch.setattr(_info, '_ONEHOT_HIST_MAX_VOXELS', 0)
    scatter = np.asarray(
        joint_histogram(
            a, b, bins=24, range_moving=_RM, range_fixed=_RM, mask=m
        )
    )
    assert np.allclose(onehot, scatter, atol=1e-12)


# ---------------------------------------------------------------------------
# Determinism contract (GPU): the scatter drifts run-to-run, the one-hot does not
# ---------------------------------------------------------------------------


@_gpu_only
def test_joint_histogram_deterministic_on_gpu():
    rng = np.random.RandomState(0)
    n = 120_000  # under the gate -> one-hot (GPU)
    a = jnp.asarray(rng.rand(n))
    b = jnp.asarray(rng.rand(n))
    jh = jax.jit(
        lambda x, y: joint_histogram(
            x, y, bins=32, range_moving=_RM, range_fixed=_RM
        )
    )
    runs = [np.asarray(jh(a, b)) for _ in range(8)]
    assert all(
        np.array_equal(runs[0], r) for r in runs
    )  # one-hot deterministic


@_gpu_only
def test_affine_mi_deterministic_on_gpu_without_restarts():
    n = 96  # every pyramid level << the one-hot gate
    fixed = _blobs(n)
    c = (jnp.asarray((n, n), dtype=fixed.dtype) - 1.0) / 2.0
    moving = spatial_transform(
        (fixed**2)[..., None],
        affine_grid(
            affine_exp(
                jnp.asarray([0.05, -0.04, 0.03, -0.03, 2.5, -2.0]), ndim=2
            ),
            (n, n),
            center=c,
        ),
        mode='nearest',
    )[..., 0]
    spec = RegistrationSpec(
        levels=3, iterations=(60, 40, 20), metric=MI(bins=32)
    )
    r1 = affine_register(moving, fixed, spec=spec, restarts=1)
    r2 = affine_register(moving, fixed, spec=spec, restarts=1)
    assert np.array_equal(np.asarray(r1.params), np.asarray(r2.params))
    assert np.array_equal(np.asarray(r1.warped), np.asarray(r2.warped))
