# -*- coding: utf-8 -*-
"""Tests for the spectral graph wavelet transform (graph.wavelet).

The Chebyshev filter is checked against an exact dense eigendecomposition of the
Laplacian (the ground-truth spectral filter), the band-pass property is checked
by annihilating the Laplacian's zero-eigenvalue mode, and the transform is shown
eigh-free (jit-clean) and differentiable.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update('jax_enable_x64', True)

from nitrix.graph import (  # noqa: E402
    graph_wavelet_transform,
    laplacian,
    mexican_hat_kernel,
)


def _graph(n=30, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.standard_normal((n, n))
    adj = np.abs(p @ p.T)
    np.fill_diagonal(adj, 0.0)
    return jnp.asarray(adj), rng


def _dense_reference(adj, signal, t, kernel, normalisation):
    ld = np.asarray(laplacian(adj, normalisation=normalisation))
    ev, V = np.linalg.eigh(ld)
    g = np.asarray(kernel(jnp.asarray(t * ev)))
    return V @ np.diag(g) @ V.T @ np.asarray(signal)


def test_gwt_matches_dense_eigendecomposition():
    adj, rng = _graph()
    f = jnp.asarray(rng.standard_normal(30))
    scales = jnp.asarray([1.0, 3.0])
    W = graph_wavelet_transform(adj, f, scales, order=40)
    assert W.shape == (2, 30)
    for i, t in enumerate([1.0, 3.0]):
        ref = _dense_reference(adj, f, t, mexican_hat_kernel, 'symmetric')
        np.testing.assert_allclose(np.asarray(W[i]), ref, atol=1e-9)


def test_gwt_custom_kernel_matches_dense():
    adj, rng = _graph(seed=2)
    f = jnp.asarray(rng.standard_normal(30))

    def heat(x):  # low-pass diffusion kernel
        return jnp.exp(-x)

    W = graph_wavelet_transform(adj, f, jnp.asarray([2.0]), kernel=heat, order=40)
    ref = _dense_reference(adj, f, 2.0, heat, 'symmetric')
    np.testing.assert_allclose(np.asarray(W[0]), ref, atol=1e-9)


def test_gwt_annihilates_zero_mode():
    # The symmetric Laplacian's zero-eigenvalue eigenvector is D^{1/2} 1 (its
    # spectrum lies in [0, 2], well within Chebyshev resolution); a band-pass
    # kernel with g(0)=0 must annihilate it.
    adj, _ = _graph(seed=1)
    v0 = jnp.sqrt(jnp.sum(adj, axis=-1))  # L_sym v0 = 0
    W = graph_wavelet_transform(
        adj, v0, jnp.asarray([1.0, 4.0]), normalisation='symmetric', order=40
    )
    assert float(jnp.max(jnp.abs(W))) / float(jnp.linalg.norm(v0)) < 1e-8


def test_gwt_explicit_lmax_matches_estimated():
    adj, rng = _graph(seed=3)
    f = jnp.asarray(rng.standard_normal(30))
    scales = jnp.asarray([1.5])
    est = graph_wavelet_transform(adj, f, scales, order=40)
    # symmetric Laplacian spectrum is bounded by 2; an explicit bound agrees.
    fixed = graph_wavelet_transform(adj, f, scales, order=40, lmax=2.0)
    np.testing.assert_allclose(np.asarray(est), np.asarray(fixed), atol=1e-6)


def test_gwt_batches_over_signals():
    adj, rng = _graph(seed=4)
    F = jnp.asarray(rng.standard_normal((3, 30)))
    scales = jnp.asarray([1.0, 2.0])
    W = graph_wavelet_transform(adj, F, scales, order=30)
    assert W.shape == (2, 3, 30)
    for i in range(3):
        single = graph_wavelet_transform(adj, F[i], scales, order=30)
        np.testing.assert_allclose(
            np.asarray(W[:, i]), np.asarray(single), atol=1e-9
        )


def test_gwt_jit_is_eigh_free():
    adj, rng = _graph(seed=5)
    f = jnp.asarray(rng.standard_normal(30))
    scales = jnp.asarray([1.0, 3.0])
    W = jax.jit(lambda A, f, s: graph_wavelet_transform(A, f, s, order=30))(
        adj, f, scales
    )
    assert bool(jnp.all(jnp.isfinite(W)))


def test_gwt_grad_finite():
    adj, rng = _graph(seed=6)
    f = jnp.asarray(rng.standard_normal(30))
    scales = jnp.asarray([1.0, 3.0])
    g = jax.grad(
        lambda f: jnp.sum(graph_wavelet_transform(adj, f, scales, order=30) ** 2)
    )(f)
    assert bool(jnp.all(jnp.isfinite(g)))


def test_mexican_hat_kernel_shape():
    x = jnp.asarray([0.0, 1.0, 5.0])
    g = mexican_hat_kernel(x)
    np.testing.assert_allclose(float(g[0]), 0.0, atol=1e-12)  # g(0) = 0
    np.testing.assert_allclose(float(g[1]), np.exp(-1.0), atol=1e-12)
    assert float(g[1]) > float(g[2])  # decays past the x=1 peak
