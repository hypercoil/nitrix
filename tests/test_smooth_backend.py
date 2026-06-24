# -*- coding: utf-8 -*-
"""Backend-aware Gaussian regulariser (``_svf._smooth_method``).

The fluid/diffusion smooths dominate the CPU per-iteration cost; the O(N)
recursive Gaussian is cheaper there, while the shifted-slice FIR path wins on
GPU.  Gate: GPU always selects FIR (byte-identical to the prior behaviour); CPU
selects recursive only when every per-axis sigma clears the YvV floor (>= 0.5),
else FIR; and the recipe still recovers a planted warp on the recursive path.
"""

from __future__ import annotations

import jax

jax.config.update('jax_enable_x64', True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import nitrix.register._svf as svf  # noqa: E402
from nitrix.metrics import ncc  # noqa: E402
from nitrix.register import (  # noqa: E402
    DemonsSpec,
    diffeomorphic_demons_register,
)
from nitrix.smoothing import gaussian  # noqa: E402


def test_smooth_method_gpu_always_fir(monkeypatch):
    monkeypatch.setattr(svf, 'default_backend_is_gpu', lambda: True)
    assert svf._smooth_method(1.0) == 'fir'
    assert svf._smooth_method(1.5) == 'fir'
    assert svf._smooth_method((1.5, 0.3)) == 'fir'  # GPU keeps FIR regardless


def test_smooth_method_cpu_recursive_with_floor(monkeypatch):
    monkeypatch.setattr(svf, 'default_backend_is_gpu', lambda: False)
    assert svf._smooth_method(1.0) == 'recursive'
    assert svf._smooth_method(1.5) == 'recursive'
    assert svf._smooth_method((1.0, 1.5)) == 'recursive'
    # below the YvV validity floor -> fall back to FIR
    assert svf._smooth_method(0.3) == 'fir'
    assert svf._smooth_method((1.0, 0.4)) == 'fir'  # one axis below floor


def test_smooth_vector_gpu_byte_identical_to_fir(monkeypatch):
    # On the GPU arm the engine is FIR, so the regulariser is bit-for-bit the
    # prior (FIR) behaviour -- the no-regression witness for the GPU path.
    monkeypatch.setattr(svf, 'default_backend_is_gpu', lambda: True)
    rng = np.random.RandomState(0)
    fld = jnp.asarray(rng.standard_normal((32, 30, 2)))
    out = svf._smooth_vector(fld, 1.0, 2)
    moved = jnp.moveaxis(fld, -1, 0)
    ref = jnp.moveaxis(
        gaussian(moved, sigma=1.0, spatial_rank=2, driver='fir'), 0, -1
    )
    assert np.array_equal(np.asarray(out), np.asarray(ref))


def _blobs(n=64):
    yy, xx = np.mgrid[0:n, 0:n].astype('float64')

    def blob(cy, cx, s, a):
        return a * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s * s))

    return jnp.asarray(
        blob(0.31 * n, 0.38 * n, 0.13 * n, 1.0)
        + blob(0.62 * n, 0.69 * n, 0.16 * n, 0.7)
        + blob(0.72 * n, 0.31 * n, 0.11 * n, 0.6)
    )


def test_demons_recovers_on_recursive_arm(monkeypatch):
    # Force the CPU (recursive) arm regardless of the test backend and confirm
    # the planted warp is still recovered and the deformation diffeomorphic.
    monkeypatch.setattr(svf, 'default_backend_is_gpu', lambda: False)
    from nitrix.geometry import (
        identity_grid,
        integrate_velocity_field,
        spatial_transform,
    )

    fixed = _blobs(64)
    rng = np.random.RandomState(3)
    v = rng.standard_normal((64, 64, 2))
    v = np.moveaxis(v, -1, 0)
    v = np.asarray(gaussian(jnp.asarray(v), sigma=8.0, spatial_rank=2))
    v = jnp.asarray(30.0 * np.moveaxis(v, 0, -1))
    grid = identity_grid(fixed.shape, dtype=fixed.dtype)
    moving = spatial_transform(
        fixed[..., None], grid + integrate_velocity_field(v), mode='nearest'
    )[..., 0]

    init = float(ncc(moving, fixed))
    res = diffeomorphic_demons_register(
        moving, fixed, spec=DemonsSpec(levels=3, iterations=60)
    )
    assert float(ncc(res.warped, fixed)) > 0.98
    assert float(ncc(res.warped, fixed)) > init + 0.02
    assert float(res.jacobian_det.min()) > 0.0
