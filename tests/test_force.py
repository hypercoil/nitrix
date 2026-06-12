# -*- coding: utf-8 -*-
"""V1: the dense-field ``Force`` protocol and its parity oracle.

The closed-form forces (``LNCCForce``, ``DemonsForce``) are Tier-1 fast paths;
``MetricForce(metric)`` is the generic autodiff escape hatch.  The design
contract is that the escape hatch agrees in *direction* with the closed form it
shadows (the magnitude is fixed downstream by the driver's step-normalisation),
which is the parity oracle that licenses the fast path -- and that any
``Metric`` (incl. cross-modal MI / CR) yields a usable force.
"""

from __future__ import annotations

import jax

jax.config.update('jax_enable_x64', True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from nitrix.geometry import spatial_gradient  # noqa: E402
from nitrix.metrics import (
    lncc,  # noqa: E402
    mi_grad,  # noqa: E402
)
from nitrix.register import (  # noqa: E402
    LNCC,
    MI,
    SSD,
    CorrelationRatio,
    DemonsForce,
    Force,
    LNCCForce,
    MetricForce,
    MIForce,
)
from nitrix.register._svf import pin_force_ranges  # noqa: E402


def _blobs(n=48, seed=0):
    yy, xx = np.mgrid[0:n, 0:n].astype('float64')
    rng = np.random.RandomState(seed)
    img = np.zeros((n, n), dtype='float64')
    for _ in range(5):
        cy, cx = rng.uniform(0.25, 0.75, 2) * n
        s = rng.uniform(0.1, 0.18) * n
        img += rng.uniform(0.4, 1.0) * np.exp(
            -((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s * s)
        )
    return jnp.asarray(img)


def _cosine(u: jax.Array, w: jax.Array) -> float:
    u, w = u.ravel(), w.ravel()
    return float(u @ w / (jnp.linalg.norm(u) * jnp.linalg.norm(w)))


def test_force_protocol_membership():
    assert isinstance(LNCCForce(2), Force)
    assert isinstance(DemonsForce(0.4), Force)
    assert isinstance(MetricForce(MI()), Force)


def test_metricforce_lncc_matches_closed_form():
    # The parity oracle: with the voxel-count rescale that undoes the metric's
    # mean reduction, the autodiff escape hatch MetricForce(LNCC) is
    # *numerically identical* to the closed-form LNCCForce -- magnitude, not
    # only direction.
    warped, fixed = _blobs(48, 0), _blobs(48, 1)
    u_closed = LNCCForce(radius=3).bind(fixed, ndim=2).update(warped)
    u_generic = MetricForce(LNCC(radius=3)).bind(fixed, ndim=2).update(warped)
    assert _cosine(u_closed, u_generic) > 1 - 1e-12
    assert np.allclose(np.asarray(u_closed), np.asarray(u_generic), atol=1e-9)


def test_metricforce_ssd_matches_thirion_direction():
    # MetricForce(SSD) is the definitional gradient force -- the autodiff of
    # the SSD cost equals the analytic Thirion direction (F-warped)*grad.
    # (This is distinct from the ESM DemonsForce, which symmetrises the
    # gradient to 1/2(grad_F + grad_warped) and reweights by denom -- a
    # genuinely specialised force, which is why it is hand-written Tier-1.)
    warped, fixed = _blobs(48, 0), _blobs(48, 1)
    u_generic = MetricForce(SSD()).bind(fixed, ndim=2).update(warped)
    thirion = (fixed - warped)[..., None] * spatial_gradient(warped)
    assert _cosine(thirion, u_generic) > 1 - 1e-9


def test_bound_force_cost_matches_metric():
    warped, fixed = _blobs(48, 0), _blobs(48, 1)
    assert np.allclose(
        float(LNCCForce(3).bind(fixed, ndim=2).cost(warped)),
        float(1.0 - lncc(warped, fixed, radius=3)),
    )
    # MetricForce.cost is exactly the metric cost.
    for metric in (SSD(), LNCC(radius=3), MI(bins=24), CorrelationRatio(bins=24)):
        assert np.allclose(
            float(MetricForce(metric).bind(fixed, ndim=2).cost(warped)),
            float(metric.cost(warped, fixed)),
        )


def test_lncc_force_physical_window_is_anisotropic_in_voxels():
    # On an anisotropic grid the LNCC window is made physically isotropic: the
    # per-axis voxel radius scales by 1/rel_spacing, so radius*spacing (the mm
    # extent) is constant across axes.
    fixed = _blobs(48)
    rel = (2.0, 0.5)  # axis 0 twice as coarse, axis 1 twice as fine
    bound = LNCCForce(radius=4).bind(fixed, ndim=2, rel_spacing=rel)
    radii = bound._radii()
    assert radii == (2, 8)  # 4/2, 4/0.5
    # physical extents (radius * rel_spacing) match across axes
    assert np.allclose([r * s for r, s in zip(radii, rel)], 4.0)
    # isotropic leaves the plain voxel radius untouched
    assert LNCCForce(radius=4).bind(fixed, ndim=2)._radii() == 4


def test_metricforce_cross_modal_yields_finite_force():
    # The escape-hatch payoff: a cross-modal metric (MI) produces a usable,
    # finite force field with no hand-written gradient.
    warped, fixed = _blobs(48, 0), _blobs(48, 2)
    u = MetricForce(MI(bins=24)).bind(fixed, ndim=2).update(warped)
    assert u.shape == (48, 48, 2)
    assert bool(jnp.all(jnp.isfinite(u)))
    assert float(jnp.abs(u).max()) > 0.0


class _ScaledMetric:
    """A user metric that does *not* declare ``is_spatial_mean`` (so it takes
    the ``getattr`` default) and scales the cost by an arbitrary constant."""

    def __init__(self, scale: float, bins: int):
        self.scale = scale
        self.bins = bins

    def cost(self, warped, fixed):
        return self.scale * MI(bins=self.bins).cost(warped, fixed)


def _rms(u: jax.Array) -> float:
    return float(jnp.sqrt(jnp.mean(jnp.sum(u**2, axis=-1))))


def test_metricforce_non_spatial_mean_is_rms_controlled():
    # B2: a non-spatial-mean metric (MI) is normalised to the target per-voxel
    # RMS `magnitude` (not the arbitrary *size constant) -- the controlled,
    # metric-scale-invariant step the unclamped Demons driver needs.  The RMS
    # hits the target exactly and scales linearly with it.
    warped, fixed = _blobs(48, 0), _blobs(48, 2)
    u3 = MetricForce(MI(bins=24), magnitude=0.3).bind(fixed, ndim=2).update(warped)
    u6 = MetricForce(MI(bins=24), magnitude=0.6).bind(fixed, ndim=2).update(warped)
    assert np.isclose(_rms(u3), 0.3, rtol=1e-6)
    assert np.isclose(_rms(u6), 0.6, rtol=1e-6)
    assert np.allclose(np.asarray(u6), 2.0 * np.asarray(u3), rtol=1e-9, atol=1e-12)


def test_miforce_is_a_force():
    assert isinstance(MIForce(), Force)


def test_miforce_rms_controlled_and_direction_is_mi_grad():
    # MIForce is RMS-magnitude-controlled (like MetricForce for a histogram
    # metric -- the raw (1/N) MI gradient is an arbitrary scale), and its
    # direction is exactly the closed-form mi_grad·∇warped (RMS normalisation is
    # direction-preserving).
    warped, fixed = _blobs(48, 0), _blobs(48, 2)
    rm = (float(warped.min()), float(warped.max()))
    rf = (float(fixed.min()), float(fixed.max()))
    u = (
        MIForce(bins=24, range_moving=rm, range_fixed=rf, magnitude=0.4)
        .bind(fixed, ndim=2)
        .update(warped)
    )
    assert np.isclose(_rms(u), 0.4, rtol=1e-6)
    raw = (
        mi_grad(warped, fixed, bins=24, range_moving=rm, range_fixed=rf)[..., None]
        * spatial_gradient(warped)
    )
    assert _cosine(u, raw) > 1 - 1e-9


def test_miforce_direction_matches_metricforce_mi():
    # The §3 parity oracle: the closed-form fast path agrees in direction with
    # the autodiff escape hatch it replaces (cos shy of 1 only by the documented
    # empty-bin divergence), at the same data-derived range.
    warped, fixed = _blobs(48, 0), _blobs(48, 2)
    u_fast = MIForce(bins=24).bind(fixed, ndim=2).update(warped)
    u_auto = MetricForce(MI(bins=24)).bind(fixed, ndim=2).update(warped)
    assert _cosine(u_fast, u_auto) > 0.99


def test_pin_force_ranges_resolves_only_unpinned_miforce():
    moving, fixed = _blobs(32, 0), _blobs(32, 1)
    pinned = pin_force_ranges(MIForce(bins=16), moving, fixed)
    assert isinstance(pinned, MIForce)
    assert pinned.range_moving == (float(moving.min()), float(moving.max()))
    assert pinned.range_fixed == (float(fixed.min()), float(fixed.max()))
    # already-pinned and non-MIForce forces pass through untouched (same object)
    already = MIForce(bins=16, range_moving=(0.0, 2.0), range_fixed=(0.0, 3.0))
    assert pin_force_ranges(already, moving, fixed) is already
    lncc_force = LNCCForce(2)
    assert pin_force_ranges(lncc_force, moving, fixed) is lncc_force


def test_metricforce_non_spatial_mean_is_metric_scale_invariant():
    # The point of the RMS normalisation: an arbitrary constant on the cost
    # (and an *undeclared* is_spatial_mean -> the safe normalised default) gives
    # an identical force, where the old `*size` rescale would have differed by
    # that constant.  Doubles as the "arbitrary user metric" coverage.
    warped, fixed = _blobs(48, 0), _blobs(48, 2)
    u = MetricForce(MI(bins=24), magnitude=0.3).bind(fixed, ndim=2).update(warped)
    u_scaled = (
        MetricForce(_ScaledMetric(7.0, 24), magnitude=0.3)
        .bind(fixed, ndim=2)
        .update(warped)
    )
    assert np.allclose(np.asarray(u_scaled), np.asarray(u), rtol=1e-6, atol=1e-9)
