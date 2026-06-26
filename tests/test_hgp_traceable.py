# -*- coding: utf-8 -*-
"""``hgp_fit`` (Gaussian HSGP) is jit / vmap-able with the covariate + grouping
closed over.

The follow-on to the ``gp_fit`` traceability lift: ``hgp_fit`` shares the
identical Gaussian-HSGP rho-search and now reuses the JAX-native
``_parabolic_argmin_jax`` (a traced ``rho_hat``), so a vmap-fit over datasets
sharing one covariate grid and grouping factor returns per-dataset
``HGPResult``s that fp-match a Python loop of eager fits.
"""

from __future__ import annotations

import jax

jax.config.update('jax_enable_x64', True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from nitrix.stats import HGPResult, hgp_fit  # noqa: E402


def _draw(rng, x, rho=0.3, noise=0.1):
    k = np.exp(-0.5 * (x[:, None] - x[None, :]) ** 2 / rho**2) + 1e-6 * np.eye(
        len(x)
    )
    f = rng.multivariate_normal(np.zeros(len(x)), k)
    return f + rng.standard_normal(len(x)) * noise


def _data():
    rng = np.random.default_rng(0)
    n, b, ell = 60, 4, 3
    x = np.sort(rng.uniform(-1.0, 1.0, n))
    group = (np.arange(n) % ell).astype(np.int32)  # 3 interleaved groups
    y_stack = np.stack([_draw(rng, x)[None, :] for _ in range(b)])  # (B, 1, N)
    return jnp.asarray(x), jnp.asarray(group), jnp.asarray(y_stack), ell


def test_hgp_fit_jit_with_closed_over_covariate_and_group():
    x, group, y_stack, ell = _data()
    y0 = y_stack[0]
    fn = jax.jit(
        lambda Y: hgp_fit(Y, x, group, rank=8, n_rho=12, n_levels=ell)
    )
    res = fn(y0)
    assert isinstance(res, HGPResult)
    assert res.coef.shape[0] == 1
    assert bool(jnp.all(jnp.isfinite(res.coef)))
    assert bool(jnp.all(jnp.isfinite(res.theta)))


def test_hgp_fit_vmap_matches_loop_of_eager():
    x, group, y_stack, ell = _data()

    def one(Y):
        return hgp_fit(Y, x, group, rank=8, n_rho=12, n_levels=ell)

    # vmap over the B datasets sharing the covariate x and grouping factor.
    batched = jax.vmap(one)(y_stack)
    assert batched.coef.shape[0] == y_stack.shape[0]

    # fp-faithful to a Python loop of independent eager fits.
    for i in range(y_stack.shape[0]):
        ref = one(y_stack[i])
        np.testing.assert_allclose(
            np.asarray(batched.coef[i]), np.asarray(ref.coef), atol=1e-5
        )
        np.testing.assert_allclose(
            np.asarray(batched.theta[i]), np.asarray(ref.theta), atol=1e-5
        )
