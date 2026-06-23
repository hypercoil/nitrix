# -*- coding: utf-8 -*-
"""Opt-in batch-aggregate early-exit for volreg (the inverse-compositional path).

``jax.vmap`` runs a ``lax.while_loop`` until **all** lanes' cond is false, so an
explicit ``Convergence`` makes the per-frame IC early-exit work *under the vmap*:
the cohort stops at the slowest frame's convergence (adapting to the motion)
instead of a fixed cap.  It is **opt-in** -- ``'auto'`` / ``None`` keep the fixed
``lax.scan`` (reproducible AND reverse-differentiable; the vmap'd while_loop has
no reverse rule).  Gates: early-exit realigns like the fixed scan; default is the
scan; the forward/BFGS path rejects it.
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
from nitrix.register import (  # noqa: E402
    Convergence,
    RegistrationSpec,
    volreg,
)


def _blobs(n=48):
    yy, xx = np.mgrid[0:n, 0:n].astype('float64')

    def blob(cy, cx, s, a):
        return a * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s * s))

    return jnp.asarray(
        blob(0.30 * n, 0.38 * n, 0.12 * n, 1.0)
        + blob(0.62 * n, 0.69 * n, 0.15 * n, 0.7)
        + blob(0.74 * n, 0.29 * n, 0.10 * n, 0.6)
    )


def _series(n=48, T=8, seed=0):
    base = _blobs(n)
    c = (jnp.asarray(base.shape, dtype=base.dtype) - 1.0) / 2.0
    rng = np.random.RandomState(seed)
    frames = []
    for _ in range(T):
        p = jnp.asarray(
            np.concatenate(
                [rng.uniform(-0.12, 0.12, 1), rng.uniform(-2.0, 2.0, 2)]
            ),
            dtype=base.dtype,
        )
        g = affine_grid(rigid_exp(p, ndim=2), base.shape, center=c)
        frames.append(
            spatial_transform(base[..., None], g, mode='constant')[..., 0]
        )
    return jnp.stack(frames)


def _interframe_var(arr):
    return float(np.std(np.asarray(arr), axis=0).mean())


def test_volreg_batch_early_exit_realigns_like_fixed():
    series = _series()
    base = RegistrationSpec(levels=2, iterations=(40, 20))
    fixed = volreg(
        series,
        reference=0,
        spec=base.__class__(**{**base.__dict__, 'mode': 'fixed'}),
    )
    ee = volreg(
        series,
        reference=0,
        spec=base.__class__(
            **{
                **base.__dict__,
                'mode': 'early_exit',
                'convergence': Convergence(1e-3, 6),
            }
        ),
    )
    init_var = _interframe_var(series)
    # both realign (inter-frame variance drops sharply); early-exit matches fixed.
    assert _interframe_var(fixed.realigned) < 0.4 * init_var
    assert _interframe_var(ee.realigned) < 0.4 * init_var
    assert (
        abs(_interframe_var(ee.realigned) - _interframe_var(fixed.realigned))
        < 0.1 * init_var
    )


def test_volreg_default_is_fixed_scan():
    # mode='fixed' (the default) runs the reproducible fixed scan.
    series = _series()
    spec = RegistrationSpec(levels=2, iterations=(40, 20))  # mode='fixed'
    a = volreg(series, reference=0, spec=spec)
    b = volreg(
        series,
        reference=0,
        spec=spec.__class__(**{**spec.__dict__, 'mode': 'fixed'}),
    )
    assert np.allclose(
        np.asarray(a.realigned), np.asarray(b.realigned), atol=1e-5
    )


def test_volreg_forward_path_rejects_convergence():
    # the forward path (forced via method='forward') cannot early-exit -- only the
    # inverse-compositional path honours the vmap'd while_loop.
    series = _series()
    spec = RegistrationSpec(levels=2, iterations=(20, 10), mode='early_exit')
    with pytest.raises(ValueError, match='volreg forward'):
        volreg(series, reference=0, spec=spec, method='forward')
