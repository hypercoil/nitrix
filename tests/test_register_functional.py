# -*- coding: utf-8 -*-
"""Tests for ``nitrix.register`` functional alignment (Procrustes / ProMises).

Functional alignment is a *family* (an ``AlignmentMethod`` ADT); ``ProMises`` is
its first method.  Coverage:

- Plain Procrustes (``ProMises(prior=None)``, the default) recovers a planted
  rotation and matches the ``scipy`` Procrustes mapping.
- The SPEC 6.5 seam: ``functional_align`` == ``apply(source, fit(...))``
  byte-for-byte; ``apply`` co-transports arbitrary feature-space data.
- The matrix-vMF ``prior`` pulls the map toward the prior orientation (the
  ProMises MAP); ``ProMises`` is a registered pytree (jit / grad / grad-wrt-prior
  clean); structural conformance to the ``AlignmentMethod`` protocol.
- ``allow_reflection`` (O vs SO).

Run on the CPU correctness floor (the eigh solve is cuSOLVER-free but, like
``decompose`` / ``pca``, ``jit``-able only on a healthy-``eigh`` backend).
"""

from __future__ import annotations

import jax
import numpy as np

jax.config.update('jax_enable_x64', True)

import jax.numpy as jnp  # noqa: E402
from scipy.linalg import (
    orthogonal_procrustes as scipy_procrustes,  # noqa: E402
)

from nitrix.register import (  # noqa: E402
    AlignmentMethod,
    FunctionalAlignment,
    ProMises,
    functional_align,
    functional_align_apply,
    functional_align_fit,
)


def _orthogonal(p: int, *, seed: int, proper: bool = False) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.standard_normal((p, p)))
    q = q * np.sign(np.diag(r))
    if proper and np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def test_default_recovers_rotation_and_matches_scipy() -> None:
    rng = np.random.default_rng(0)
    q = _orthogonal(5, seed=1, proper=True)
    x = rng.standard_normal((60, 5))
    m = x @ q
    fit = functional_align_fit(jnp.asarray(x), jnp.asarray(m))
    assert isinstance(fit, FunctionalAlignment)
    np.testing.assert_allclose(np.asarray(fit.matrix), q, atol=1e-9)
    # Matches the scipy Procrustes mapping (source -> reference).
    r_scipy, _ = scipy_procrustes(x, m)
    np.testing.assert_allclose(np.asarray(fit.matrix), r_scipy, atol=1e-9)


def test_single_call_is_fit_then_apply() -> None:
    rng = np.random.default_rng(2)
    x = jnp.asarray(rng.standard_normal((40, 4)))
    m = jnp.asarray(rng.standard_normal((40, 4)))
    one = np.asarray(functional_align(x, m))
    split = np.asarray(functional_align_apply(x, functional_align_fit(x, m)))
    np.testing.assert_array_equal(one, split)  # byte-faithful (SPEC 6.5)


def test_apply_co_transports_other_data() -> None:
    rng = np.random.default_rng(3)
    x = jnp.asarray(rng.standard_normal((30, 4)))
    m = jnp.asarray(rng.standard_normal((30, 4)))
    fit = functional_align_fit(x, m)
    aux = jnp.asarray(rng.standard_normal((7, 4)))  # other data, same features
    np.testing.assert_allclose(
        np.asarray(functional_align_apply(aux, fit)),
        np.asarray(aux) @ np.asarray(fit.matrix),
        atol=1e-12,
    )


def test_prior_pulls_toward_orientation() -> None:
    rng = np.random.default_rng(4)
    q = _orthogonal(5, seed=5, proper=True)
    x = rng.standard_normal((50, 5))
    m = x @ q
    eye = np.eye(5)
    r0 = np.asarray(
        functional_align_fit(jnp.asarray(x), jnp.asarray(m)).matrix
    )
    r1 = np.asarray(
        functional_align_fit(
            jnp.asarray(x),
            jnp.asarray(m),
            method=ProMises(prior=jnp.asarray(eye), prior_weight=400.0),
        ).matrix
    )
    assert np.linalg.norm(r1 - eye) < np.linalg.norm(r0 - eye)
    np.testing.assert_allclose(r1.T @ r1, eye, atol=1e-9)


def test_promises_is_pytree_and_conforms() -> None:
    assert isinstance(ProMises(), AlignmentMethod)
    f = jnp.asarray(3.0 * np.eye(4))
    method = ProMises(prior=f, prior_weight=2.0, allow_reflection=False)
    leaves, treedef = jax.tree_util.tree_flatten(method)
    assert len(leaves) == 1  # the prior array; the rest is static aux
    back = jax.tree_util.tree_unflatten(treedef, leaves)
    assert back.prior_weight == 2.0 and back.allow_reflection is False
    np.testing.assert_array_equal(np.asarray(back.prior), np.asarray(f))


def test_allow_reflection_method_flag() -> None:
    rng = np.random.default_rng(6)
    x = rng.standard_normal((40, 4))
    ref = np.eye(4)
    ref[0, 0] = -1.0
    m = x @ ref
    r_o = np.asarray(
        functional_align_fit(jnp.asarray(x), jnp.asarray(m)).matrix
    )
    r_so = np.asarray(
        functional_align_fit(
            jnp.asarray(x),
            jnp.asarray(m),
            method=ProMises(allow_reflection=False),
        ).matrix
    )
    assert np.linalg.det(r_o) < 0  # O(p) recovers the reflection
    np.testing.assert_allclose(np.linalg.det(r_so), 1.0, atol=1e-9)  # SO(p)


def test_jit_grad_through_method_and_prior() -> None:
    rng = np.random.default_rng(7)
    x = jnp.asarray(rng.standard_normal((50, 5)))
    m = jnp.asarray(rng.standard_normal((50, 5)))
    eager = np.asarray(functional_align(x, m))
    jitted = np.asarray(jax.jit(functional_align)(x, m))
    np.testing.assert_allclose(jitted, eager, atol=1e-10)

    g = jax.grad(
        lambda d: functional_align(
            d, m, psi=1e-3, key=jax.random.PRNGKey(0)
        ).sum()
    )(x)
    assert bool(jnp.all(jnp.isfinite(g)))

    f = jnp.asarray(2.0 * np.eye(5))
    gp = jax.grad(
        lambda prior: functional_align(
            x, m, method=ProMises(prior=prior)
        ).sum()
    )(f)
    assert bool(jnp.all(jnp.isfinite(gp)))
