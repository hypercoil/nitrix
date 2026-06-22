# -*- coding: utf-8 -*-
"""A3: masked / weighted **cost** (distinct from the SVF force mask).

A fixed-grid ``mask`` restricts the similarity *cost* to a region: out-of-mask
voxels are ignored.  For a spatial-mean metric (SSD / LNCC) it weights the
reduction; for a histogram metric (MI / CR) it gates the joint-histogram scatter
(an out-of-mask voxel leaves the distribution -- masking the reduction alone
would not).  Gates: a hard mask matches an explicit crop on the kernels; the
matrix recipes thread it (an all-ones mask is a no-op; masking a corrupted
region restores recovery); the inverse-compositional path rejects a mask.

This is orthogonal to ``test_masks.py`` (the diffeomorphic *force* mask, which
gates the velocity update, not the cost).
"""

from __future__ import annotations

import jax

jax.config.update('jax_enable_x64', True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from nitrix.geometry import (  # noqa: E402
    affine_grid,
    rigid_exp,
    spatial_transform,
)
from nitrix.metrics import (  # noqa: E402
    correlation_ratio,
    lncc,
    mutual_information,
    ssd,
)
from nitrix.register import (  # noqa: E402
    MI,
    SSD,
    CorrelationRatio,
    RegistrationSpec,
    rigid_register,
)

_RM = (0.0, 1.0)  # pinned ranges so masked-vs-crop share bin edges
_RF = (0.0, 1.0)


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


def _rigid_pair(n=64):
    fixed = _blobs(n)
    c = (jnp.asarray((n, n), dtype=fixed.dtype) - 1.0) / 2.0
    grid = affine_grid(
        rigid_exp(jnp.asarray([0.12, 3.0, -2.0]), ndim=2), (n, n), center=c
    )
    moving = spatial_transform(fixed[..., None], grid, mode='nearest')[..., 0]
    return moving, fixed


# ---------------------------------------------------------------------------
# Kernel: a hard mask matches an explicit crop (scatter-gating for MI / CR)
# ---------------------------------------------------------------------------


def _half_mask(n=24):
    m = np.ones((n, n))
    m[: n // 3, :] = 0.0
    return jnp.asarray(m)


def test_ssd_mask_equals_crop():
    rng = np.random.RandomState(1)
    a = jnp.asarray(rng.rand(24, 24))
    b = jnp.asarray(rng.rand(24, 24))
    m = _half_mask(24)
    keep = m.astype(bool).reshape(-1)
    masked = float(ssd(a, b, mask=m))
    crop = float(ssd(a.reshape(-1)[keep], b.reshape(-1)[keep]))
    assert np.allclose(masked, crop, atol=1e-12)


def test_mi_mask_gates_scatter_equals_crop():
    rng = np.random.RandomState(2)
    a = jnp.asarray(rng.rand(24, 24))
    b = jnp.asarray(rng.rand(24, 24))
    m = _half_mask(24)
    keep = m.astype(bool).reshape(-1)
    masked = float(
        mutual_information(a, b, mask=m, range_moving=_RM, range_fixed=_RF)
    )
    crop = float(
        mutual_information(
            a.reshape(-1)[keep],
            b.reshape(-1)[keep],
            range_moving=_RM,
            range_fixed=_RF,
        )
    )
    assert np.allclose(masked, crop, atol=1e-9)


def test_cr_mask_gates_scatter_equals_crop():
    rng = np.random.RandomState(3)
    a = jnp.asarray(rng.rand(24, 24))
    b = jnp.asarray(rng.rand(24, 24))
    m = _half_mask(24)
    keep = m.astype(bool).reshape(-1)
    masked = float(correlation_ratio(a, b, mask=m, range_fixed=_RF))
    crop = float(
        correlation_ratio(
            a.reshape(-1)[keep], b.reshape(-1)[keep], range_fixed=_RF
        )
    )
    assert np.allclose(masked, crop, atol=1e-9)


def test_ones_mask_is_noop_on_kernels():
    rng = np.random.RandomState(4)
    a = jnp.asarray(rng.rand(20, 20))
    b = jnp.asarray(rng.rand(20, 20))
    ones = jnp.ones((20, 20))
    assert np.allclose(ssd(a, b), ssd(a, b, mask=ones))
    assert np.allclose(lncc(a, b, radius=3), lncc(a, b, radius=3, mask=ones))
    assert np.allclose(
        mutual_information(a, b, range_moving=_RM, range_fixed=_RF),
        mutual_information(a, b, mask=ones, range_moving=_RM, range_fixed=_RF),
    )
    assert np.allclose(
        correlation_ratio(a, b, range_fixed=_RF),
        correlation_ratio(a, b, mask=ones, range_fixed=_RF),
    )


# ---------------------------------------------------------------------------
# Metric ADT: cost / residual thread the mask
# ---------------------------------------------------------------------------


def test_metric_cost_threads_mask():
    rng = np.random.RandomState(5)
    a = jnp.asarray(rng.rand(20, 20))
    b = jnp.asarray(rng.rand(20, 20))
    m = _half_mask(20)
    assert np.allclose(SSD().cost(a, b, mask=m), ssd(a, b, mask=m))
    assert np.allclose(
        MI(bins=32, range_moving=_RM, range_fixed=_RF).cost(a, b, mask=m),
        -mutual_information(
            a, b, bins=32, mask=m, range_moving=_RM, range_fixed=_RF
        ),
    )
    assert np.allclose(
        CorrelationRatio(bins=32, range_fixed=_RF).cost(a, b, mask=m),
        1.0 - correlation_ratio(a, b, bins=32, mask=m, range_fixed=_RF),
    )


def test_ssd_residual_is_sqrt_mask_weighted():
    rng = np.random.RandomState(6)
    a = jnp.asarray(rng.rand(8, 8))
    b = jnp.asarray(rng.rand(8, 8))
    m = jnp.asarray(np.random.RandomState(7).rand(8, 8))
    r = SSD().residual(a, b, mask=m)
    expect = ((a - b) * jnp.sqrt(m)).ravel()
    assert np.allclose(np.asarray(r), np.asarray(expect))
    # half the masked sum-of-squares is the masked SSD-sum (the GN cost).
    assert np.allclose(
        0.5 * float(jnp.sum(r * r)), 0.5 * float(jnp.sum(m * (a - b) ** 2))
    )


# ---------------------------------------------------------------------------
# Recipes: all-ones no-op, masked recovery, IC rejection, shape guard
# ---------------------------------------------------------------------------


def test_rigid_ones_mask_equals_no_mask():
    moving, fixed = _rigid_pair(64)
    spec = RegistrationSpec(levels=3, iterations=40)
    ref = rigid_register(moving, fixed, spec=spec, method='forward')
    ones = rigid_register(
        moving, fixed, spec=spec, method='forward', mask=jnp.ones((64, 64))
    )
    assert np.allclose(
        np.asarray(ref.params), np.asarray(ones.params), atol=1e-6
    )


def test_masked_cost_recovers_through_corruption():
    # A region of the FIXED image is replaced with unrelated texture; masking it
    # out restores the recovery that the corruption otherwise pulls off-target.
    moving, fixed = _rigid_pair(64)
    corrupt = np.asarray(fixed).copy()
    corrupt[:20, :] = np.asarray(_blobs(64, seed=99))[:20, :]
    corrupt = jnp.asarray(corrupt)
    mask = np.ones((64, 64))
    mask[:20, :] = 0.0
    mask = jnp.asarray(mask)
    spec = RegistrationSpec(levels=3, iterations=40)

    unmasked = rigid_register(moving, corrupt, spec=spec)
    masked = rigid_register(moving, corrupt, spec=spec, mask=mask)

    from nitrix.metrics import ncc

    # score recovery on the clean interior (rows 24+) against the true fixed.
    ncc_unmasked = float(ncc(unmasked.warped[24:], fixed[24:]))
    ncc_masked = float(ncc(masked.warped[24:], fixed[24:]))
    assert ncc_masked > ncc_unmasked
    assert ncc_masked > 0.999


def test_ic_path_rejects_mask():
    moving, fixed = _rigid_pair(48)
    mask = jnp.ones((48, 48))
    with pytest.raises(ValueError, match='cannot honour a cost mask'):
        rigid_register(
            moving,
            fixed,
            spec=RegistrationSpec(levels=2, iterations=10),
            method='inverse_compositional',
            mask=mask,
        )


def test_mask_shape_validation():
    moving, fixed = _rigid_pair(48)
    with pytest.raises(ValueError, match='must match the fixed grid'):
        rigid_register(
            moving,
            fixed,
            spec=RegistrationSpec(levels=2, iterations=10),
            method='forward',
            mask=jnp.ones((40, 40)),
        )
