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
from nitrix.metrics import lncc  # noqa: E402
from nitrix.register import (  # noqa: E402
    LNCC,
    MI,
    SSD,
    CorrelationRatio,
    DemonsForce,
    Force,
    LNCCForce,
    MetricForce,
)


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


def test_metricforce_lncc_matches_closed_form_direction():
    # The autodiff escape hatch MetricForce(LNCC) and the closed-form
    # LNCCForce differ only by a scalar (sum- vs mean-reduction), so their
    # update fields are exactly parallel -- the parity oracle.
    warped, fixed = _blobs(48, 0), _blobs(48, 1)
    u_closed = LNCCForce(radius=3).bind(fixed, ndim=2).update(warped)
    u_generic = MetricForce(LNCC(radius=3)).bind(fixed, ndim=2).update(warped)
    assert _cosine(u_closed, u_generic) > 1 - 1e-9


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


def test_metricforce_cross_modal_yields_finite_force():
    # The escape-hatch payoff: a cross-modal metric (MI) produces a usable,
    # finite force field with no hand-written gradient.
    warped, fixed = _blobs(48, 0), _blobs(48, 2)
    u = MetricForce(MI(bins=24)).bind(fixed, ndim=2).update(warped)
    assert u.shape == (48, 48, 2)
    assert bool(jnp.all(jnp.isfinite(u)))
    assert float(jnp.abs(u).max()) > 0.0
