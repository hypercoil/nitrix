# -*- coding: utf-8 -*-
"""``gp_fit`` (Gaussian HSGP) is jit / vmap-able with the covariate closed over.

The nimox-estimators GP-fit traceability request: the rho-search epilogue is now
JAX-native (``_parabolic_argmin_jax`` + a traced ``rho_hat``), so a vmap-fit over
datasets sharing one covariate grid returns per-dataset ``GPResult``s that
fp-match a Python loop of eager fits.
"""

from __future__ import annotations

import jax

jax.config.update('jax_enable_x64', True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from nitrix.stats import GPResult, gp_fit  # noqa: E402


def _draw(rng, x, rho=0.3, noise=0.1):
    k = np.exp(-0.5 * (x[:, None] - x[None, :]) ** 2 / rho**2) + 1e-6 * np.eye(
        len(x)
    )
    f = rng.multivariate_normal(np.zeros(len(x)), k)
    return f + rng.standard_normal(len(x)) * noise


def _fits():
    rng = np.random.default_rng(0)
    n, b = 60, 4
    x = np.sort(rng.uniform(-1.0, 1.0, n))
    y_stack = np.stack([_draw(rng, x)[None, :] for _ in range(b)])  # (B, 1, N)
    return jnp.asarray(x), jnp.asarray(y_stack)


def test_gp_fit_jit_with_closed_over_covariate():
    x, y_stack = _fits()
    y0 = y_stack[0]
    fn = jax.jit(lambda Y: gp_fit(Y, x, rank=15, n_rho=16))
    res = fn(y0)
    assert isinstance(res, GPResult)
    assert res.coef.shape == (1, 1 + 15)
    assert bool(jnp.all(jnp.isfinite(res.coef)))
    assert bool(jnp.all(jnp.isfinite(res.theta)))


def test_gp_fit_vmap_matches_loop_of_eager():
    x, y_stack = _fits()

    def one(Y):
        return gp_fit(Y, x, rank=15, n_rho=16)

    # vmap over the B datasets sharing the covariate x.
    batched = jax.vmap(one)(y_stack)
    assert batched.coef.shape == (y_stack.shape[0], 1, 1 + 15)

    # fp-faithful to a Python loop of independent eager fits.
    for i in range(y_stack.shape[0]):
        ref = one(y_stack[i])
        np.testing.assert_allclose(
            np.asarray(batched.coef[i]), np.asarray(ref.coef), atol=1e-5
        )
        np.testing.assert_allclose(
            np.asarray(batched.theta[i]), np.asarray(ref.theta), atol=1e-5
        )
