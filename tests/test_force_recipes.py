# -*- coding: utf-8 -*-
"""V1b: the dense-field recipes are metric-configurable via ``force``.

The keystone payoff at the recipe level: the diffeomorphic recipes accept a
``force`` (the generic ``MetricForce`` escape hatch or a per-level schedule),
so a single driver does mono-modal LNCC SyN, cross-modal MI deformable
registration, and a fast-coarse / high-signal-fine pyramid -- the
composability the welded recipes could not.
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
from nitrix.metrics import mutual_information, ncc  # noqa: E402
from nitrix.register import (  # noqa: E402
    LNCC,
    MI,
    DemonsForce,
    DemonsSpec,
    LNCCForce,
    MetricForce,
    MIForce,
    SumForce,
    SyNSpec,
    diffeomorphic_demons_register,
    greedy_syn_register,
)
from nitrix.smoothing import gaussian  # noqa: E402


def _blobs(n=48, seed=0):
    yy, xx = np.mgrid[0:n, 0:n].astype('float64')
    img = np.zeros((n, n), dtype='float64')
    for cy, cx, s, a in [
        (0.31, 0.38, 0.13, 1.0),
        (0.62, 0.69, 0.16, 0.7),
        (0.72, 0.31, 0.11, 0.6),
        (0.44, 0.75, 0.14, 0.5),
    ]:
        img += a * np.exp(
            -((xx - cx * n) ** 2 + (yy - cy * n) ** 2) / (2 * (s * n) ** 2)
        )
    return jnp.asarray(img)


def _smooth_velocity(shape, sigma, scale, seed):
    rng = np.random.RandomState(seed)
    v = rng.standard_normal(shape + (2,))
    v = np.moveaxis(v, -1, 0)
    v = np.asarray(gaussian(jnp.asarray(v), sigma=sigma, spatial_rank=2))
    v = np.moveaxis(v, 0, -1)
    return jnp.asarray(scale * v)


def _warp(image, v):
    grid = identity_grid(image.shape, dtype=image.dtype)
    s = integrate_velocity_field(v)
    return spatial_transform(image[..., None], grid + s, mode='nearest')[..., 0]


def test_syn_metricforce_lncc_matches_default():
    # The autodiff escape hatch drives the full recipe to the *same* recovery
    # as the closed-form LNCCForce default -- the voxel-count rescale makes
    # MetricForce(LNCC) numerically identical to the closed form, so the
    # warped results match closely.
    fixed = _blobs(64)
    moving = _warp(fixed, _smooth_velocity((64, 64), 8.0, 45.0, 0))
    spec = SyNSpec(levels=3, iterations=60, radius=2, step=0.5)
    default = greedy_syn_register(moving, fixed, spec=spec)
    generic = greedy_syn_register(
        moving, fixed, spec=spec, force=MetricForce(LNCC(radius=2))
    )
    assert float(ncc(default.warped, fixed)) > 0.99
    assert float(ncc(generic.warped, fixed)) > 0.99
    assert np.allclose(
        np.asarray(default.warped), np.asarray(generic.warped), atol=1e-4
    )
    assert float(generic.jacobian_det.min()) > 0.0


def test_syn_multimodal_mi_improves():
    # The multimodal payoff: a cross-modal pair (a monotone-nonlinear intensity
    # remap of the deformed fixed) driven by MetricForce(MI) -- no hand-written
    # MI force -- improves the MI it optimises and stays diffeomorphic.
    fixed = _blobs(64)
    deformed = _warp(fixed, _smooth_velocity((64, 64), 9.0, 22.0, 0))
    moving = jnp.sqrt(deformed - deformed.min() + 0.05)  # "different modality"
    spec = SyNSpec(levels=3, iterations=60, radius=3, step=0.5)
    res = greedy_syn_register(
        moving, fixed, spec=spec, force=MetricForce(MI(bins=24))
    )
    mi0 = float(mutual_information(moving, fixed, bins=24))
    mi1 = float(mutual_information(res.warped, fixed, bins=24))
    assert mi1 > mi0
    assert float(res.jacobian_det.min()) > 0.0


def test_syn_miforce_mi_recovers():
    # The closed-form Mattes MI fast path (MIForce) drives cross-modal SyN to a
    # real MI gain and stays diffeomorphic -- the fMRIPrep cross-modal deformable
    # path with the autodiff histogram tape removed (1a).
    fixed = _blobs(64)
    deformed = _warp(fixed, _smooth_velocity((64, 64), 9.0, 22.0, 0))
    moving = jnp.sqrt(deformed - deformed.min() + 0.05)  # "different modality"
    spec = SyNSpec(levels=3, iterations=60, radius=3, step=0.5)
    res = greedy_syn_register(moving, fixed, spec=spec, force=MIForce(bins=24))
    mi0 = float(mutual_information(moving, fixed, bins=24))
    mi1 = float(mutual_information(res.warped, fixed, bins=24))
    assert mi1 > mi0
    assert float(res.jacobian_det.min()) > 0.0


def test_demons_miforce_mi_recovers():
    # MIForce on the *unclamped* Demons driver: the RMS magnitude control (the
    # 0c reconciliation baked into MIForce) gives a tuned step, so the fast
    # closed-form path recovers like the MetricForce(MI) escape hatch.
    fixed = _blobs(64)
    deformed = _warp(fixed, _smooth_velocity((64, 64), 9.0, 22.0, 0))
    moving = jnp.sqrt(deformed - deformed.min() + 0.05)
    spec = DemonsSpec(levels=3, iterations=80)
    res = diffeomorphic_demons_register(
        moving, fixed, spec=spec, force=MIForce(bins=24)
    )
    mi0 = float(mutual_information(moving, fixed, bins=24))
    mi1 = float(mutual_information(res.warped, fixed, bins=24))
    assert mi1 > mi0 + 0.2
    assert float(res.jacobian_det.min()) > 0.0


def test_demons_metricforce_mi_recovers():
    # B2: MI drives the *unclamped* Demons driver (step=None) to a real
    # recovery.  The RMS-controlled MetricForce magnitude replaces the arbitrary
    # `*size` constant that no one had tuned, so the cross-modal deformable run
    # both improves its MI substantially and stays diffeomorphic -- the
    # convergence test the escape hatch lacked.
    fixed = _blobs(64)
    deformed = _warp(fixed, _smooth_velocity((64, 64), 9.0, 22.0, 0))
    moving = jnp.sqrt(deformed - deformed.min() + 0.05)  # "different modality"
    spec = DemonsSpec(levels=3, iterations=80)
    res = diffeomorphic_demons_register(
        moving, fixed, spec=spec, force=MetricForce(MI(bins=24))
    )
    mi0 = float(mutual_information(moving, fixed, bins=24))
    mi1 = float(mutual_information(res.warped, fixed, bins=24))
    assert mi1 > mi0 + 0.2  # a substantial improvement, not merely nonzero
    assert float(res.jacobian_det.min()) > 0.0
    assert bool(jnp.all(jnp.isfinite(res.warped)))


def test_syn_sumforce_multi_metric_recovers():
    # A5: a multi-metric SumForce (two LNCC windows summed) drives the recipe end
    # to end with no driver change -- the composability payoff.
    fixed = _blobs(64)
    moving = _warp(fixed, _smooth_velocity((64, 64), 8.0, 45.0, 0))
    spec = SyNSpec(levels=3, iterations=60, step=0.5)
    force = SumForce(((0.5, LNCCForce(4)), (0.5, LNCCForce(2))))
    res = greedy_syn_register(moving, fixed, spec=spec, force=force)
    assert float(ncc(res.warped, fixed)) > 0.99
    assert float(res.jacobian_det.min()) > 0.0


def test_syn_per_level_schedule():
    # Per-level schedule (coarse-to-fine): a large LNCC window at the coarse
    # levels for capture range, a small one at the finest for detail.
    fixed = _blobs(64)
    moving = _warp(fixed, _smooth_velocity((64, 64), 8.0, 45.0, 0))
    spec = SyNSpec(levels=3, iterations=60, step=0.5)
    schedule = [LNCCForce(radius=4), LNCCForce(radius=3), LNCCForce(radius=2)]
    res = greedy_syn_register(moving, fixed, spec=spec, force=schedule)
    assert float(ncc(res.warped, fixed)) > 0.99
    assert float(res.jacobian_det.min()) > 0.0


def test_syn_anisotropic_physical_window_recovers():
    # The physical-window path runs end-to-end and recovers with spacing set
    # (the LNCC window + sigmas are anisotropy-corrected).
    fixed = _blobs(64)
    moving = _warp(fixed, _smooth_velocity((64, 64), 8.0, 30.0, 0))
    spec = SyNSpec(
        levels=3, iterations=60, radius=3, step=0.5, spacing=(2.0, 1.0)
    )
    res = greedy_syn_register(moving, fixed, spec=spec)
    assert float(ncc(res.warped, fixed)) > float(ncc(moving, fixed))
    assert float(ncc(res.warped, fixed)) > 0.99
    assert float(res.jacobian_det.min()) > 0.0


def test_force_schedule_length_validation():
    fixed, moving = _blobs(32), _blobs(32, seed=1)
    spec = SyNSpec(levels=3, iterations=5)
    with pytest.raises(ValueError):
        greedy_syn_register(
            moving, fixed, spec=spec, force=[LNCCForce(2), DemonsForce(0.4)]
        )
