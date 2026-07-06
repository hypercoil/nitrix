# -*- coding: utf-8 -*-
"""Tests for ``nitrix.register`` functional alignment (Procrustes / ProMises).

Functional alignment is a *family* (an ``AlignmentMethod`` ADT); ``ProMises``
(dense) and ``EfficientProMises`` (subspace) are its first methods.  Coverage:

- Plain Procrustes (``ProMises(prior=None)``, the default) recovers a planted
  rotation and matches the ``scipy`` Procrustes mapping.
- The SPEC 6.5 seam: ``functional_align`` == ``apply(source, fit(...))``
  byte-for-byte; ``apply`` co-transports arbitrary feature-space data.
- The matrix-vMF ``prior`` pulls the map toward the prior orientation (the
  ProMises MAP); ``ProMises`` is a registered pytree (jit / grad / grad-wrt-prior
  clean); structural conformance to the ``AlignmentMethod`` protocol.
- ``allow_reflection`` (O vs SO).
- ``EfficientProMises`` (the whole-brain subspace method, Andreella & Finos
  2022, Theorem 3 / Lemma 5): the reduced solve is genuinely orthogonal; the
  fitted bases are semi-orthogonal; the implicit map is never materialised;
  when the reduction is lossless it equals dense ``ProMises`` exactly; the
  Lemma-5 prior projection; ``n_components`` truncation; jit / vmap / grad.

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
    DenseAlignment,
    EfficientProMises,
    FunctionalAlignment,
    ProMises,
    SubspaceAlignment,
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


# --------------------------------------------------------------------------- #
# Efficient ProMises (the whole-brain subspace method).                       #
# --------------------------------------------------------------------------- #


def _shared_subspace_problem(
    n: int, p: int, r: int, *, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A source/reference pair sharing an ``r``-dim feature subspace.

    ``reference`` is ``source`` rotated by a planted orthogonal ``Q`` *within*
    that subspace, so the alignment is exactly recoverable and the reduction to
    ``r`` components is lossless.
    """
    rng = np.random.default_rng(seed)
    basis = np.linalg.qr(rng.standard_normal((p, r)))[0]  # (p, r) dict
    latent = rng.standard_normal((n, r))
    rot = _orthogonal(r, seed=seed + 1, proper=True)  # in-subspace rotation
    source = latent @ basis.T
    reference = latent @ rot @ basis.T
    return source, reference, basis


def test_efficient_fit_is_subspace_alignment_and_conforms() -> None:
    rng = np.random.default_rng(10)
    x = jnp.asarray(rng.standard_normal((8, 60)))
    m = jnp.asarray(rng.standard_normal((8, 60)))
    fit = functional_align_fit(x, m, method=EfficientProMises())
    assert isinstance(fit, SubspaceAlignment)
    assert isinstance(fit, FunctionalAlignment)  # structural conformance
    assert isinstance(EfficientProMises(), AlignmentMethod)
    # Dense fit is the DenseAlignment sibling.
    assert isinstance(
        functional_align_fit(x, m, method=ProMises()), DenseAlignment
    )


def test_efficient_never_materialises_the_pp_map() -> None:
    # The (p, p) map is never formed: state is two (p, l) bases + an (l, l)
    # reduced rotation, with l <= n << p.
    n, p = 10, 400
    rng = np.random.default_rng(11)
    x = jnp.asarray(rng.standard_normal((n, p)))
    m = jnp.asarray(rng.standard_normal((n, p)))
    fit = functional_align_fit(x, m, method=EfficientProMises())
    assert fit.source_basis.shape == (p, n)
    assert fit.reference_basis.shape == (p, n)
    assert fit.reduced.shape == (n, n)
    # No leaf is (p, p).
    for leaf in jax.tree_util.tree_leaves(fit):
        assert leaf.shape != (p, p)


def test_efficient_bases_and_reduced_are_orthogonal() -> None:
    n, p = 12, 200
    rng = np.random.default_rng(12)
    x = jnp.asarray(rng.standard_normal((n, p)))
    m = jnp.asarray(rng.standard_normal((n, p)))
    fit = functional_align_fit(x, m, method=EfficientProMises())
    eye_l = np.eye(n)
    qx = np.asarray(fit.source_basis)
    qm = np.asarray(fit.reference_basis)
    rstar = np.asarray(fit.reduced)
    np.testing.assert_allclose(qx.T @ qx, eye_l, atol=1e-10)  # semi-orthogonal
    np.testing.assert_allclose(qm.T @ qm, eye_l, atol=1e-10)
    np.testing.assert_allclose(
        rstar.T @ rstar, eye_l, atol=1e-10
    )  # orthogonal


def test_efficient_lossless_equals_dense() -> None:
    # p >> n and the full row rank is kept (n_components=None) => the reduction
    # is lossless, so EfficientProMises and ProMises agree on the aligned data.
    n, p = 10, 150
    rng = np.random.default_rng(13)
    x = jnp.asarray(rng.standard_normal((n, p)))
    m = jnp.asarray(rng.standard_normal((n, p)))
    dense = np.asarray(functional_align(x, m, method=ProMises()))
    eff = np.asarray(functional_align(x, m, method=EfficientProMises()))
    np.testing.assert_allclose(eff, dense, atol=1e-6)


def test_efficient_reproduces_the_dense_map_oracle() -> None:
    # The subspace map lifts back through the *reference* basis, so it
    # reproduces the dense ProMises MAP -- ``X @ scipy.orthogonal_procrustes``,
    # an independent oracle -- not the legacy source-basis reconstruction.  In
    # the p >> n rank-deficient regime the reduced (n x n) solve is better
    # conditioned than the dense (p x p) polar, so this is a tight match.
    n, p = 10, 160
    rng = np.random.default_rng(20)
    x = rng.standard_normal((n, p))
    m = rng.standard_normal((n, p))
    r_scipy, _ = scipy_procrustes(x, m)  # the dense MAP, U V^T of X^T M
    aligned_map = x @ r_scipy
    eff = np.asarray(
        functional_align(
            jnp.asarray(x), jnp.asarray(m), method=EfficientProMises()
        )
    )
    np.testing.assert_allclose(eff, aligned_map, atol=1e-7)


def test_efficient_recovers_planted_subspace_rotation() -> None:
    n, p, r = 12, 300, 6
    source, reference, _ = _shared_subspace_problem(n, p, r, seed=14)
    aligned = np.asarray(
        functional_align(
            jnp.asarray(source),
            jnp.asarray(reference),
            method=EfficientProMises(),
        )
    )
    # Exact recovery: the aligned source matches the reference to machine eps.
    np.testing.assert_allclose(aligned, reference, atol=1e-9)


def test_efficient_co_transports_within_source_span() -> None:
    # Auxiliary data living in the source's row space transports identically
    # under the dense and the (lossless) subspace maps.
    n, p = 10, 150
    rng = np.random.default_rng(15)
    x = jnp.asarray(rng.standard_normal((n, p)))
    m = jnp.asarray(rng.standard_normal((n, p)))
    fit_d = functional_align_fit(x, m, method=ProMises())
    fit_e = functional_align_fit(x, m, method=EfficientProMises())
    aux = jnp.asarray(rng.standard_normal((5, n))) @ x  # rows in source span
    np.testing.assert_allclose(
        np.asarray(functional_align_apply(aux, fit_e)),
        np.asarray(functional_align_apply(aux, fit_d)),
        atol=1e-6,
    )


def test_efficient_prior_is_lemma5_projection() -> None:
    # The (p, p) prior enters the reduced solve only as F* = Q_X^T F Q_M
    # (Lemma 5), scaled by the concentration.  Verify the reduced rotation is
    # exactly the reduced Procrustes with that projected prior.
    from nitrix.linalg import orthogonal_procrustes

    n, p = 8, 120
    rng = np.random.default_rng(16)
    x = jnp.asarray(rng.standard_normal((n, p)))
    m = jnp.asarray(rng.standard_normal((n, p)))
    f = jnp.asarray(rng.standard_normal((p, p)))
    k = 3.0
    fit = functional_align_fit(
        x, m, method=EfficientProMises(prior=f, prior_weight=k)
    )
    # The bases do not depend on the prior; recover them from a plain fit.
    plain = functional_align_fit(x, m, method=EfficientProMises())
    qx, qm = plain.source_basis, plain.reference_basis
    f_star = qx.T @ (k * f) @ qm  # Lemma 5
    expected = orthogonal_procrustes(x @ qx, m @ qm, prior=f_star)
    np.testing.assert_allclose(
        np.asarray(fit.reduced), np.asarray(expected), atol=1e-10
    )
    np.testing.assert_array_equal(np.asarray(fit.source_basis), np.asarray(qx))


def test_efficient_n_components_truncation() -> None:
    n, p, keep = 12, 200, 5
    rng = np.random.default_rng(17)
    x = jnp.asarray(rng.standard_normal((n, p)))
    m = jnp.asarray(rng.standard_normal((n, p)))
    fit = functional_align_fit(
        x, m, method=EfficientProMises(n_components=keep)
    )
    assert fit.source_basis.shape == (p, keep)
    assert fit.reduced.shape == (keep, keep)
    r = np.asarray(fit.reduced)
    np.testing.assert_allclose(r.T @ r, np.eye(keep), atol=1e-10)


def test_efficient_allow_reflection_constrains_reduced() -> None:
    n, p = 10, 120
    rng = np.random.default_rng(18)
    x = jnp.asarray(rng.standard_normal((n, p)))
    m = jnp.asarray(rng.standard_normal((n, p)))
    r_so = np.asarray(
        functional_align_fit(
            x, m, method=EfficientProMises(allow_reflection=False)
        ).reduced
    )
    np.testing.assert_allclose(np.linalg.det(r_so), 1.0, atol=1e-9)


def test_efficient_is_pytree_jit_vmap_grad() -> None:
    n, p = 10, 100
    rng = np.random.default_rng(19)
    x = jnp.asarray(rng.standard_normal((n, p)))
    m = jnp.asarray(rng.standard_normal((n, p)))

    # EfficientProMises is a registered pytree (the prior is the only child).
    method = EfficientProMises(prior=jnp.asarray(np.eye(p)), n_components=6)
    leaves, treedef = jax.tree_util.tree_flatten(method)
    assert len(leaves) == 1
    back = jax.tree_util.tree_unflatten(treedef, leaves)
    assert back.n_components == 6 and back.prior_weight == 1.0

    eager = np.asarray(functional_align(x, m, method=EfficientProMises()))
    jitted = np.asarray(
        jax.jit(
            lambda a, b: functional_align(a, b, method=EfficientProMises())
        )(x, m)
    )
    np.testing.assert_allclose(jitted, eager, atol=1e-10)

    xb = jnp.asarray(rng.standard_normal((3, n, p)))
    mb = jnp.asarray(rng.standard_normal((3, n, p)))
    batched = jax.vmap(
        lambda a, b: functional_align(a, b, method=EfficientProMises())
    )(xb, mb)
    looped = jnp.stack(
        [
            functional_align(xb[i], mb[i], method=EfficientProMises())
            for i in range(3)
        ]
    )
    np.testing.assert_allclose(
        np.asarray(batched), np.asarray(looped), atol=1e-10
    )

    g = jax.grad(
        lambda a: functional_align(a, m, method=EfficientProMises()).sum()
    )(x)
    assert bool(jnp.all(jnp.isfinite(g)))


# ---------------------------------------------------------------------------
# CoordinateKernelPrior: whole-brain-tractable spatial ProMises prior
# ---------------------------------------------------------------------------


def _fa_data(n=20, p=60, d=3, seed=0):
    rng = np.random.default_rng(seed)
    coords = jnp.asarray(rng.standard_normal((p, d)))
    x = jnp.asarray(rng.standard_normal((n, p)))
    m = jnp.asarray(rng.standard_normal((n, p)))
    return coords, x, m


def test_coordinate_kernel_prior_rff_converges_to_exact_projection():
    """F* via random Fourier features approaches the exact Q_X^T K Q_M as the
    feature count grows (error ~ r^{-1/2}); the (p, p) kernel is never formed."""
    from nitrix.linalg import rbf_kernel
    from nitrix.register import CoordinateKernelPrior
    from nitrix.register._functional import _right_basis

    coords, x, m = _fa_data()
    qx, qm = _right_basis(x, None), _right_basis(m, None)
    ell = 1.5
    k_exact = rbf_kernel(coords, coords, gamma=1.0 / (2 * ell**2))
    f_exact = qx.T @ k_exact @ qm

    def rel_err(r):
        ckp = CoordinateKernelPrior(coords, ell, jax.random.PRNGKey(0), r)
        f = ckp.reduced(qx, qm)
        assert f.shape == (qx.shape[1], qm.shape[1])  # (l, l), never (p, p)
        return float(jnp.linalg.norm(f - f_exact) / jnp.linalg.norm(f_exact))

    e_small, e_large = rel_err(256), rel_err(8192)
    assert e_large < e_small  # more features -> closer
    assert e_large < 0.05  # accurate at high feature count


def test_efficient_promises_spatial_prior_matches_dense_prior_path():
    """The whole-brain spatial_prior path equals projecting the dense (p, p)
    kernel prior, once the RFF approximation is accurate."""
    from nitrix.linalg import rbf_kernel
    from nitrix.register import CoordinateKernelPrior, EfficientProMises

    coords, x, m = _fa_data()
    ell = 1.5
    k_dense = rbf_kernel(coords, coords, gamma=1.0 / (2 * ell**2))
    ckp = CoordinateKernelPrior(coords, ell, jax.random.PRNGKey(1), 16384)

    fit_spatial = EfficientProMises(spatial_prior=ckp, prior_weight=2.0).fit(
        x, m
    )
    fit_dense = EfficientProMises(prior=k_dense, prior_weight=2.0).fit(x, m)
    np.testing.assert_allclose(
        fit_spatial.reduced, fit_dense.reduced, atol=5e-3
    )


def test_spatial_prior_influences_the_alignment():
    """A spatial prior actually biases the map (differs from prior=None)."""
    from nitrix.register import CoordinateKernelPrior, EfficientProMises

    coords, x, m = _fa_data(seed=2)
    ckp = CoordinateKernelPrior(coords, 1.0, jax.random.PRNGKey(2), 1024)
    with_prior = EfficientProMises(spatial_prior=ckp, prior_weight=5.0).fit(
        x, m
    )
    without = EfficientProMises().fit(x, m)
    assert float(jnp.max(jnp.abs(with_prior.reduced - without.reduced))) > 1e-3


def test_spatial_prior_jit_and_pytree():
    from nitrix.register import CoordinateKernelPrior, EfficientProMises

    coords, x, m = _fa_data(seed=3)

    def run(c, key):
        ckp = CoordinateKernelPrior(c, 1.2, key, 512)
        return EfficientProMises(spatial_prior=ckp).fit(x, m).reduced

    out = jax.jit(run)(coords, jax.random.PRNGKey(0))
    assert bool(jnp.all(jnp.isfinite(out)))
    # pytree round-trip carries coords/key as children, lengthscale/n as aux
    ckp = CoordinateKernelPrior(coords, 1.2, jax.random.PRNGKey(0), 512)
    method = EfficientProMises(spatial_prior=ckp, prior_weight=3.0)
    leaves, treedef = jax.tree_util.tree_flatten(method)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert rebuilt.prior_weight == 3.0
    assert rebuilt.spatial_prior.n_features == 512
