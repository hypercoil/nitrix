# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""P3 contract: every divergent variant agrees with its canonical within budget.

For each registered divergent op (``nitrix.divergent_ops()``), run *every*
``driver`` variant on representative inputs and assert it matches the op's
**canonical** variant within the op's **registered** per-dtype tolerance -- read
directly from the registry, the single source of truth (no ``tolerance.toml``
mirror).  This is what turns "numerically divergent but bounded" into a tested
contract: a regression that widens any variant's divergence fails CI.

The variants are forced explicitly (``driver=...``), so the comparison is
variant-vs-canonical on this host -- exactly the reassociation / approximation /
determinism divergence the tolerance bounds (a true CPU-vs-GPU run reduces to the
same algorithm pair).  ``register.field_smooth`` is the one looser contract: its
two variants use different *boundary conditions* (FIR reflect vs Young-van Vliet
edge) and the recursive path is a 3rd-order pole *approximation* (~1% interior),
so it is tested on a boundary-decaying field and carries a ~3e-2 budget by
design -- not the ~ULP reassociation of the other four.
"""

from __future__ import annotations

import jax

jax.config.update('jax_enable_x64', True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from nitrix import divergent_ops  # noqa: E402

_DTYPES = {'float32': np.float32, 'float64': np.float64}


def _relerr(a, b, *, crop: int = 0) -> float:
    a, b = np.asarray(a), np.asarray(b)
    if crop:
        sl = (slice(crop, -crop),) * a.ndim
        a, b = a[sl], b[sl]
    return float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-30))


# --- per-op runners: {driver_value: result} on representative inputs ------


def _run_iir(npdt):
    from nitrix.signal import butterworth_sos, sosfilt

    x = jnp.asarray(np.random.RandomState(0).standard_normal(256).astype(npdt))
    sos = butterworth_sos(order=4, fs=1.0, btype='lowpass', hi=0.1)
    return {
        v: sosfilt(x, sos, driver=v) for v in ('fft', 'scan', 'associative')
    }


def _run_ssm(npdt):
    from nitrix.nn.ssm import reference_selective_scan as ssm

    rng = np.random.RandomState(1)
    bn, ln, d, n = 2, 64, 4, 3
    x = jnp.asarray(rng.standard_normal((bn, ln, d)).astype(npdt))
    delta = jax.nn.softplus(
        jnp.asarray(rng.standard_normal((bn, ln, d)).astype(npdt))
    )
    a = -jnp.asarray(np.abs(rng.standard_normal((d, n))).astype(npdt))
    b = jnp.asarray(rng.standard_normal((bn, ln, n)).astype(npdt))
    c = jnp.asarray(rng.standard_normal((bn, ln, n)).astype(npdt))
    dd = jnp.asarray(rng.standard_normal(d).astype(npdt))
    return {
        v: ssm(x, delta, a, b, c, dd, driver=v)
        for v in ('sequential', 'associative', 'chunked')
    }


def _run_cubic(npdt):
    from nitrix.geometry import CubicBSpline, resample

    img = jnp.asarray(
        np.random.RandomState(2).standard_normal((20, 20, 1)).astype(npdt)
    )
    return {
        v: resample(
            img, (27, 27), method=CubicBSpline(driver=v), mode='mirror'
        )
        for v in ('sequential', 'associative')
    }


def _run_gaussian(npdt):
    # A centred blob that decays to ~0 inside the boundary (the regulariser
    # regime), so the FIR/recursive *boundary-condition* mismatch does not
    # dominate -- the contract then bounds the YvV approximation (~1%).
    from nitrix.smoothing import gaussian

    yy, xx = np.mgrid[0:48, 0:48]
    blob = np.exp(-(((yy - 24) ** 2 + (xx - 24) ** 2) / (2 * 8.0**2)))
    fld = jnp.asarray(blob.astype(npdt))
    return {
        v: gaussian(fld, sigma=1.5, driver=v) for v in ('fir', 'recursive')
    }


def _run_hist(npdt):
    from nitrix.metrics.information import _joint_hist_from_softbins as hist

    rng = np.random.RandomState(4)
    n, bins = 5000, 16
    lm = jnp.asarray(rng.randint(0, bins - 1, n))
    fm = jnp.asarray(rng.rand(n).astype(npdt))
    lf = jnp.asarray(rng.randint(0, bins - 1, n))
    ff = jnp.asarray(rng.rand(n).astype(npdt))
    dt = jnp.dtype(npdt)
    return {
        v: hist(lm, fm, lf, ff, bins, dt, driver=v)
        for v in ('onehot', 'scatter')
    }


_RUNNERS = {
    'signal.iir': _run_iir,
    'nn.ssm.selective_scan': _run_ssm,
    'geometry.cubic_bspline_prefilter': _run_cubic,
    'register.field_smooth': _run_gaussian,
    'metrics.joint_histogram': _run_hist,
}


def _entry(op_name):
    return next(o for o in divergent_ops() if o.op == op_name)


# --- the contract --------------------------------------------------------


def test_every_registered_op_has_a_contract_runner():
    """Keeps P3 in sync with the registry: a new divergent op fails here until
    its variant-vs-canonical runner is added."""
    registered = {o.op for o in divergent_ops()}
    assert registered == set(_RUNNERS), (
        f'registry {registered} != runners {set(_RUNNERS)}'
    )


@pytest.mark.parametrize('op_name', sorted(_RUNNERS))
def test_variants_agree_with_canonical_within_registered_tolerance(op_name):
    entry = _entry(op_name)
    runner = _RUNNERS[op_name]
    for dtype_name, tol in entry.tolerance.items():
        results = runner(_DTYPES[dtype_name])
        assert entry.canonical in results, (
            f'{op_name}: runner missing canonical {entry.canonical!r}'
        )
        canon = results[entry.canonical]
        # Every concrete driver value must be exercised by the runner.
        for drv in entry.driver_values:
            if drv == 'auto':
                continue
            assert drv in results, f'{op_name}: runner missing variant {drv!r}'
            if drv == entry.canonical:
                continue
            err = _relerr(results[drv], canon)
            assert err <= tol, (
                f'{op_name}: driver={drv!r} vs canonical {entry.canonical!r} '
                f'@ {dtype_name}: relerr {err:.2e} > registered tol {tol:.2e}'
            )


def test_canonical_is_self_consistent():
    """The canonical variant reproduces itself exactly (sanity on the harness)."""
    for op_name, runner in _RUNNERS.items():
        entry = _entry(op_name)
        results = runner(
            np.float64 if 'float64' in entry.tolerance else np.float32
        )
        assert (
            _relerr(results[entry.canonical], results[entry.canonical]) == 0.0
        )
