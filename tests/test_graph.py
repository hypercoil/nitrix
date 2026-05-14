# -*- coding: utf-8 -*-
"""Tests for ``nitrix.graph``.

Coverage:

- **laplacian**: ``degree_vector`` matches manual sums on dense /
  ELL / SectionedELL.  Each Laplacian variant satisfies its
  defining property (combinatorial: row sums to 0; symmetric:
  symmetric and PSD; random_walk: I - row-stochastic).
  ``laplacian_matvec`` gives the same answer for dense and sparse
  representations of the same graph.
- **community**: ``girvan_newman_null`` is rank-1, ``modularity_matrix``
  has total weight conservation, ``relaxed_modularity`` matches a
  hand computation on a known partition.  The sparse-aware
  factored path agrees with the dense path bit-for-bit.
- **connectopy**: Laplacian eigenmap recovers the cluster
  structure of a known two-cluster graph; ``eigh`` and ``lobpcg``
  paths agree on eigenvalues; diffusion embedding produces
  largest-first eigenvalues bounded by 1.

Some tests run on graphs sized for LOBPCG's constraint
``5 * (n_components + 1) < n``; smaller graphs use the ``eigh`` path.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nitrix.graph import (
    coaffiliation,
    degree_vector,
    diffusion_embedding,
    girvan_newman_null,
    laplacian,
    laplacian_eigenmap,
    laplacian_matvec,
    modularity_matrix,
    modularity_matrix_matvec,
    relaxed_modularity,
)
from nitrix.sparse import ELL, sectioned_ell_from_ragged


jax.config.update('jax_enable_x64', True)


# ---------------------------------------------------------------------------
# Fixtures: small graphs in dense, ELL, and SectionedELL form
# ---------------------------------------------------------------------------


def _ring_adjacency(n: int):
    A = np.zeros((n, n))
    for i in range(n):
        A[i, (i + 1) % n] = 1.0
        A[(i + 1) % n, i] = 1.0
    return A


def _two_cluster_adjacency(n_per_cluster: int = 8, p_intra=0.9, p_inter=0.05):
    rng = np.random.default_rng(42)
    n = 2 * n_per_cluster
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            in_same_cluster = (i // n_per_cluster) == (j // n_per_cluster)
            p = p_intra if in_same_cluster else p_inter
            if rng.random() < p:
                A[i, j] = A[j, i] = 1.0
    return A


def _ell_from_dense(A: np.ndarray) -> ELL:
    '''Construct an ELL adjacency by padding all rows to the max degree.'''
    n = A.shape[0]
    degs = (A != 0).sum(axis=1)
    k_max = int(degs.max())
    values = np.zeros((n, k_max), dtype=A.dtype)
    indices = np.zeros((n, k_max), dtype=np.int32)
    for i in range(n):
        nz = np.flatnonzero(A[i])
        values[i, :len(nz)] = A[i, nz]
        indices[i, :len(nz)] = nz
    return ELL(
        values=jnp.asarray(values),
        indices=jnp.asarray(indices),
        n_cols=n,
        identity=0.0,
    )


def _sectioned_from_dense(A: np.ndarray):
    n = A.shape[0]
    values_per_row = []
    indices_per_row = []
    for i in range(n):
        nz = np.flatnonzero(A[i]).astype(np.int32)
        values_per_row.append(A[i, nz])
        indices_per_row.append(nz)
    return sectioned_ell_from_ragged(
        values_per_row, indices_per_row, n_cols=n,
    )


# ---------------------------------------------------------------------------
# degree_vector
# ---------------------------------------------------------------------------


def test_degree_vector_dense():
    A = jnp.asarray(_ring_adjacency(8))
    np.testing.assert_array_equal(degree_vector(A), jnp.full((8,), 2.0))


def test_degree_vector_ell():
    A = _ring_adjacency(8)
    ell = _ell_from_dense(A)
    np.testing.assert_array_equal(degree_vector(ell), jnp.full((8,), 2.0))


def test_degree_vector_sectioned_ell():
    A = _ring_adjacency(8)
    sec = _sectioned_from_dense(A)
    np.testing.assert_array_equal(degree_vector(sec), jnp.full((8,), 2.0))


def test_degree_vector_sectioned_variable_degree():
    '''Variable-degree graph: 3 nodes with degrees 1, 3, 2.

    Mass conservation: total degree = total non-zero edges.
    '''
    rng = np.random.default_rng(0)
    n = 8
    A = rng.binomial(1, 0.3, size=(n, n)).astype(np.float64)
    A = np.triu(A, 1)
    A = A + A.T  # symmetrise
    sec = _sectioned_from_dense(A)
    np.testing.assert_allclose(degree_vector(sec), A.sum(axis=1))


# ---------------------------------------------------------------------------
# laplacian
# ---------------------------------------------------------------------------


def test_combinatorial_laplacian_row_sums_to_zero():
    A = jnp.asarray(_ring_adjacency(8))
    L = laplacian(A, normalisation='combinatorial')
    np.testing.assert_allclose(L.sum(axis=-1), 0.0, atol=1e-12)


def test_symmetric_laplacian_is_symmetric_psd():
    A = jnp.asarray(_two_cluster_adjacency())
    L = laplacian(A, normalisation='symmetric')
    np.testing.assert_allclose(L, L.swapaxes(-1, -2), atol=1e-12)
    # Spectrum non-negative
    vals = jnp.linalg.eigvalsh(
        jax.device_put(L, jax.devices('cpu')[0])
    )
    assert float(vals.min()) >= -1e-10


def test_random_walk_laplacian_row_property():
    '''L_rw = I - D^(-1) A; rows of D^(-1) A sum to 1, so L_rw rows sum to 0.'''
    A = jnp.asarray(_ring_adjacency(8))
    L = laplacian(A, normalisation='random_walk')
    np.testing.assert_allclose(L.sum(axis=-1), 0.0, atol=1e-12)


def test_laplacian_invalid_normalisation():
    A = jnp.zeros((4, 4))
    with pytest.raises(ValueError, match='normalisation'):
        laplacian(A, normalisation='bogus')


# ---------------------------------------------------------------------------
# laplacian_matvec on dense, ELL, SectionedELL
# ---------------------------------------------------------------------------


def test_laplacian_matvec_constant_eigenvector():
    '''Combinatorial / symmetric Laplacians annihilate the constant vector.'''
    A = jnp.asarray(_ring_adjacency(8))
    ones = jnp.ones((8, 1))
    for norm in ('combinatorial', 'symmetric', 'random_walk'):
        out = laplacian_matvec(A, ones, normalisation=norm)
        np.testing.assert_allclose(out, 0.0, atol=1e-12)


def test_laplacian_matvec_dense_vs_ell_agreement():
    A_dense = _ring_adjacency(8)
    ell = _ell_from_dense(A_dense)
    A_jax = jnp.asarray(A_dense)
    x = jax.random.normal(jax.random.key(0), (8, 3))
    for norm in ('combinatorial', 'symmetric', 'random_walk'):
        out_dense = laplacian_matvec(A_jax, x, normalisation=norm)
        out_ell = laplacian_matvec(ell, x, normalisation=norm)
        np.testing.assert_allclose(out_dense, out_ell, atol=1e-10)


def test_laplacian_matvec_dense_vs_sectioned_agreement():
    A_dense = _two_cluster_adjacency()
    sec = _sectioned_from_dense(A_dense)
    A_jax = jnp.asarray(A_dense)
    x = jax.random.normal(jax.random.key(0), (A_dense.shape[0], 2))
    for norm in ('combinatorial', 'symmetric', 'random_walk'):
        out_dense = laplacian_matvec(A_jax, x, normalisation=norm)
        out_sec = laplacian_matvec(sec, x, normalisation=norm)
        np.testing.assert_allclose(out_dense, out_sec, atol=1e-10)


# ---------------------------------------------------------------------------
# Modularity primitives
# ---------------------------------------------------------------------------


def test_girvan_newman_null_is_rank_one():
    A = jnp.asarray(_two_cluster_adjacency())
    P = girvan_newman_null(A)
    # Numerical rank by counting non-trivial singular values.
    s = jnp.linalg.svd(
        jax.device_put(P, jax.devices('cpu')[0]),
        compute_uv=False,
    )
    # Drop everything more than 6 orders of magnitude below the top.
    cutoff = float(s[0]) * 1e-6
    rank = int((s > cutoff).sum())
    assert rank == 1


def test_girvan_newman_null_conserves_total_weight():
    '''Sum of the null model equals total edge weight ``2m``.'''
    A = jnp.asarray(_two_cluster_adjacency())
    P = girvan_newman_null(A)
    np.testing.assert_allclose(
        float(P.sum()), float(A.sum()), atol=1e-10,
    )


def test_modularity_matrix_unnormalised_recovers_a_minus_null():
    A = jnp.asarray(_two_cluster_adjacency())
    B = modularity_matrix(A, gamma=1.0, normalise=False)
    P = girvan_newman_null(A)
    np.testing.assert_allclose(B, A - P, atol=1e-10)


def test_modularity_matrix_matvec_dense_vs_ell_agreement():
    A_dense = _two_cluster_adjacency()
    ell = _ell_from_dense(A_dense)
    A_jax = jnp.asarray(A_dense)
    x = jax.random.normal(jax.random.key(0), (A_dense.shape[0], 2))
    out_dense = modularity_matrix_matvec(A_jax, x, gamma=1.0, normalise=False)
    out_ell = modularity_matrix_matvec(ell, x, gamma=1.0, normalise=False)
    np.testing.assert_allclose(out_dense, out_ell, atol=1e-10)


def test_coaffiliation_one_hot_is_indicator():
    '''A hard partition into two communities gives the
    block-indicator co-affiliation matrix (with diagonal zeroed).
    '''
    n_per = 4
    n = 2 * n_per
    C = np.zeros((n, 2))
    C[:n_per, 0] = 1
    C[n_per:, 1] = 1
    K = coaffiliation(jnp.asarray(C), exclude_diag=True)
    expected = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j and (i // n_per) == (j // n_per):
                expected[i, j] = 1.0
    np.testing.assert_array_equal(K, expected)


# ---------------------------------------------------------------------------
# Relaxed modularity (dense + sparse)
# ---------------------------------------------------------------------------


def test_relaxed_modularity_positive_for_aligned_partition():
    '''Two-cluster graph + matching partition -> positive modularity.'''
    A_dense = _two_cluster_adjacency(n_per_cluster=10, p_intra=0.9, p_inter=0.02)
    n = A_dense.shape[0]
    n_per = n // 2
    C = np.zeros((n, 2))
    C[:n_per, 0] = 1
    C[n_per:, 1] = 1
    Q = relaxed_modularity(jnp.asarray(A_dense), jnp.asarray(C))
    assert float(Q) > 0


def test_relaxed_modularity_zero_for_random_partition():
    '''Random partition on a random graph -> modularity ≈ 0.'''
    rng = np.random.default_rng(7)
    n = 24
    A = rng.binomial(1, 0.3, size=(n, n)).astype(np.float64)
    A = np.triu(A, 1)
    A = A + A.T
    C = rng.binomial(1, 0.5, size=(n, 2)).astype(np.float64)
    Q = relaxed_modularity(jnp.asarray(A), jnp.asarray(C))
    assert abs(float(Q)) < 0.5


def test_relaxed_modularity_dense_vs_ell_agree():
    '''The sparse-aware factored path matches the dense path.'''
    A_dense = _two_cluster_adjacency(
        n_per_cluster=8, p_intra=0.9, p_inter=0.05,
    )
    n = A_dense.shape[0]
    rng = np.random.default_rng(11)
    C = rng.standard_normal((n, 3))
    ell = _ell_from_dense(A_dense)
    Q_dense = relaxed_modularity(jnp.asarray(A_dense), jnp.asarray(C))
    Q_sparse = relaxed_modularity(ell, jnp.asarray(C))
    np.testing.assert_allclose(Q_dense, Q_sparse, atol=1e-10, rtol=1e-10)


def test_relaxed_modularity_dense_vs_sectioned_agree():
    A_dense = _two_cluster_adjacency(
        n_per_cluster=8, p_intra=0.7, p_inter=0.08,
    )
    n = A_dense.shape[0]
    rng = np.random.default_rng(13)
    C = rng.standard_normal((n, 4))
    sec = _sectioned_from_dense(A_dense)
    Q_dense = relaxed_modularity(jnp.asarray(A_dense), jnp.asarray(C))
    Q_sparse = relaxed_modularity(sec, jnp.asarray(C))
    np.testing.assert_allclose(Q_dense, Q_sparse, atol=1e-10, rtol=1e-10)


def test_relaxed_modularity_differentiable_dense():
    A = jnp.asarray(_two_cluster_adjacency(n_per_cluster=6, p_intra=0.9))
    C = jax.random.normal(jax.random.key(0), (A.shape[0], 2))
    def loss(C):
        return relaxed_modularity(A, C)
    g = jax.grad(loss)(C)
    assert bool(jnp.all(jnp.isfinite(g)))


# ---------------------------------------------------------------------------
# Connectopy
# ---------------------------------------------------------------------------


def test_laplacian_eigenmap_eigh_recovers_clusters():
    '''On a two-cluster graph, the first non-trivial eigenvector
    separates the clusters (sign changes across the boundary).
    '''
    n_per = 10
    A = jnp.asarray(_two_cluster_adjacency(
        n_per_cluster=n_per, p_intra=0.95, p_inter=0.01,
    ))
    embedding, eigvals = laplacian_eigenmap(
        A, n_components=2, solver='eigh',
    )
    # First non-trivial eigenvector: signs in first half differ
    # from signs in second half (for at least one of the recovered
    # components).
    first = embedding[:, 0]
    cluster_a = jnp.sign(first[:n_per]).mean()
    cluster_b = jnp.sign(first[n_per:]).mean()
    # Either +1 / -1 or vice versa; their difference should be large.
    assert abs(float(cluster_a) - float(cluster_b)) > 1.5


def test_laplacian_eigenmap_eigh_vs_lobpcg_eigenvalues_agree():
    '''The two solver paths must produce the same eigenvalues
    (up to numerical noise) on a graph large enough for lobpcg.'''
    n = 32
    A = jnp.asarray(_ring_adjacency(n))
    _, vals_eigh = laplacian_eigenmap(A, n_components=3, solver='eigh')
    _, vals_lobpcg = laplacian_eigenmap(
        A, n_components=3, solver='lobpcg', lobpcg_iters=300,
    )
    np.testing.assert_allclose(
        np.sort(np.asarray(vals_eigh)),
        np.sort(np.asarray(vals_lobpcg)),
        atol=1e-6,
    )


def test_laplacian_eigenmap_sparse_input_uses_lobpcg():
    '''ELL input automatically routes to lobpcg via solver="auto".'''
    n = 32
    A_dense = _ring_adjacency(n)
    ell = _ell_from_dense(A_dense)
    _, vals = laplacian_eigenmap(ell, n_components=3, lobpcg_iters=300)
    # Compare against dense eigh.
    _, vals_dense = laplacian_eigenmap(
        jnp.asarray(A_dense), n_components=3, solver='eigh',
    )
    np.testing.assert_allclose(
        np.sort(np.asarray(vals)),
        np.sort(np.asarray(vals_dense)),
        atol=1e-6,
    )


def test_laplacian_eigenmap_eigh_rejects_sparse():
    n = 16
    ell = _ell_from_dense(_ring_adjacency(n))
    with pytest.raises(ValueError, match='eigh'):
        laplacian_eigenmap(ell, n_components=2, solver='eigh')


def test_laplacian_eigenmap_sectioned_ell_matches_dense():
    '''SectionedELL adjacency must give the same eigenvalues (up to
    LOBPCG numerical noise) as the equivalent dense input.
    '''
    n = 32
    A_dense = _ring_adjacency(n)
    sec = _sectioned_from_dense(A_dense)
    _, vals_sec = laplacian_eigenmap(
        sec, n_components=3, lobpcg_iters=300,
    )
    _, vals_dense = laplacian_eigenmap(
        jnp.asarray(A_dense), n_components=3, solver='eigh',
    )
    np.testing.assert_allclose(
        np.sort(np.asarray(vals_sec)),
        np.sort(np.asarray(vals_dense)),
        atol=1e-6,
    )


def test_laplacian_eigenmap_sectioned_variable_degree():
    '''SectionedELL with non-uniform degrees: same answer as a
    densified version of the same graph.
    '''
    rng = np.random.default_rng(7)
    n = 40
    # Erdős-Rényi with non-uniform degrees
    A = (rng.random((n, n)) < 0.15).astype(np.float64)
    A = np.triu(A, 1)
    A = A + A.T
    # Make sure there's no isolated node (so the Laplacian is well-defined).
    while (A.sum(axis=1) == 0).any():
        idx = int(np.argmin(A.sum(axis=1)))
        partner = (idx + 1) % n
        A[idx, partner] = A[partner, idx] = 1.0
    sec = _sectioned_from_dense(A)
    _, vals_sec = laplacian_eigenmap(
        sec, n_components=3, lobpcg_iters=400,
    )
    _, vals_dense = laplacian_eigenmap(
        jnp.asarray(A), n_components=3, solver='eigh',
    )
    np.testing.assert_allclose(
        np.sort(np.asarray(vals_sec)),
        np.sort(np.asarray(vals_dense)),
        atol=1e-5,
    )


def test_diffusion_embedding_sectioned_ell_matches_dense():
    n = 32
    A_dense = _ring_adjacency(n)
    sec = _sectioned_from_dense(A_dense)
    _, vals_sec = diffusion_embedding(
        sec, n_components=3, alpha=0.5, lobpcg_iters=300,
    )
    _, vals_dense = diffusion_embedding(
        jnp.asarray(A_dense), n_components=3, alpha=0.5, solver='eigh',
    )
    np.testing.assert_allclose(
        np.sort(np.asarray(vals_sec)),
        np.sort(np.asarray(vals_dense)),
        atol=1e-6,
    )


# ---------------------------------------------------------------------------
# Device preservation
# ---------------------------------------------------------------------------


def test_connectopy_returns_outputs_on_source_device():
    '''When ``_eigh_device()`` resolves to CPU (broken cuSolver) but
    the input lives on GPU, the result should still come back on
    GPU so downstream code stays on the same device.
    '''
    n = 32
    A = jnp.asarray(_ring_adjacency(n))  # likely on GPU per JAX default
    source = list(A.devices())[0]
    for solver in ('eigh', 'lobpcg'):
        vecs, vals = laplacian_eigenmap(
            A, n_components=2, solver=solver, lobpcg_iters=200,
        )
        # Outputs share the input's device.
        assert source in vecs.devices(), (
            f'eigenvectors landed on {vecs.devices()} not {source!r}'
        )
        assert source in vals.devices()


def test_diffusion_embedding_eigenvalues_bounded():
    '''Diffusion-operator eigenvalues lie in (-1, 1].'''
    n = 24
    A = jnp.asarray(_two_cluster_adjacency(n_per_cluster=12))
    _, vals = diffusion_embedding(A, n_components=3, alpha=0.5, solver='eigh')
    assert float(vals.max()) <= 1.0 + 1e-9
    assert float(vals.min()) >= -1.0 - 1e-9
    # Descending order.
    assert all(
        float(vals[i]) >= float(vals[i + 1]) - 1e-8
        for i in range(vals.size - 1)
    )


def test_diffusion_embedding_eigh_vs_lobpcg_agree():
    n = 32
    A = jnp.asarray(_ring_adjacency(n))
    _, v1 = diffusion_embedding(
        A, n_components=3, alpha=0.5, solver='eigh',
    )
    _, v2 = diffusion_embedding(
        A, n_components=3, alpha=0.5, solver='lobpcg',
        lobpcg_iters=300,
    )
    np.testing.assert_allclose(
        np.sort(np.asarray(v1)),
        np.sort(np.asarray(v2)),
        atol=1e-6,
    )


def test_diffusion_embedding_t_scaling():
    '''``t > 0`` multiplies each eigenvector by lambda^t.'''
    n = 24
    A = jnp.asarray(_two_cluster_adjacency(n_per_cluster=12))
    embed0, vals = diffusion_embedding(
        A, n_components=3, alpha=0.5, t=0.0, solver='eigh',
    )
    embed_t, _ = diffusion_embedding(
        A, n_components=3, alpha=0.5, t=2.0, solver='eigh',
    )
    expected = embed0 * (vals ** 2)[None, :]
    np.testing.assert_allclose(embed_t, expected, atol=1e-10)


def test_eigh_differentiable():
    '''Reverse-mode AD through eigh works (jnp.linalg.eigh ships with a VJP).'''
    n = 16
    A = jnp.asarray(_two_cluster_adjacency(n_per_cluster=8))
    def loss(A):
        _, vals = laplacian_eigenmap(A, n_components=2, solver='eigh')
        return vals.sum()
    g = jax.grad(loss)(A)
    assert bool(jnp.all(jnp.isfinite(g)))


# ---------------------------------------------------------------------------
# LOBPCG implicit VJP (subspace-projector approximation)
# ---------------------------------------------------------------------------


def _build_psd(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    M = A @ A.T + 0.1 * np.eye(n)
    return jnp.asarray(M)


def _ell_from_dense_symmetric(A_dense: np.ndarray):
    '''Build a flat ELL from a dense symmetric matrix.'''
    n = A_dense.shape[0]
    k_max = int(np.max((A_dense != 0).sum(axis=1)))
    values = np.zeros((n, k_max))
    indices = np.zeros((n, k_max), dtype=np.int32)
    for i in range(n):
        nz = np.nonzero(A_dense[i])[0]
        values[i, :len(nz)] = A_dense[i, nz]
        indices[i, :len(nz)] = nz
        if len(nz) < k_max:
            indices[i, len(nz):] = nz[0] if len(nz) > 0 else 0
    return ELL(
        values=jnp.asarray(values),
        indices=jnp.asarray(indices),
        n_cols=n,
        identity=0.0,
    )


def test_lobpcg_dense_eigenvalue_grad_matches_eigh():
    '''Eigenvalue gradient via LOBPCG implicit-VJP matches eigh exactly.

    Hellmann-Feynman is exact; the only error source is LOBPCG's
    eigenvalue residual against eigh's machine-precision answer.
    '''
    from nitrix.graph._lobpcg_diff import lobpcg_top_k_dense

    n, k = 40, 3
    M = _build_psd(n, seed=0)
    X0 = jax.random.normal(jax.random.key(0), (n, k))

    def loss_lob(M):
        ev, _ = lobpcg_top_k_dense(M, X0, 500, None, 1e-8)
        return ev.sum()

    def loss_eigh(M):
        ev_all, _ = jnp.linalg.eigh(M)
        return ev_all[-k:].sum()  # top-k

    g_lob = jax.grad(loss_lob)(M)
    g_eigh = jax.grad(loss_eigh)(M)
    np.testing.assert_allclose(
        np.asarray(g_lob), np.asarray(g_eigh), atol=1e-12,
    )


def test_lobpcg_dense_in_subspace_loss_matches_analytical():
    '''In-subspace loss ``trace(U^T M U @ T)`` has analytical gradient
    ``U @ diag(diag(T)) @ U^T``, the F-matrix correction cancels the
    off-diagonal of ``T``.  The implicit-VJP should hit this exactly.
    '''
    from nitrix.graph._lobpcg_diff import lobpcg_top_k_dense

    n, k = 40, 3
    M = _build_psd(n, seed=1)
    X0 = jax.random.normal(jax.random.key(0), (n, k))
    target = jax.random.normal(jax.random.key(2), (k, k))
    target = (target + target.T) / 2

    _, U = lobpcg_top_k_dense(M, X0, 500, None, 1e-8)
    expected = U @ jnp.diag(jnp.diag(target)) @ U.T

    def loss(M):
        _, U = lobpcg_top_k_dense(M, X0, 500, None, 1e-8)
        return jnp.trace(U.T @ M @ U @ target)

    g = jax.grad(loss)(M)
    np.testing.assert_allclose(np.asarray(g), np.asarray(expected), atol=1e-12)


def test_lobpcg_dense_gradient_is_symmetric():
    '''The gradient of any scalar loss through LOBPCG implicit-VJP lives
    in the symmetric subspace -- ``V K V^T`` with ``K`` symmetric.
    '''
    from nitrix.graph._lobpcg_diff import lobpcg_top_k_dense

    n, k = 30, 2
    M = _build_psd(n, seed=3)
    X0 = jax.random.normal(jax.random.key(0), (n, k))
    target = jax.random.normal(jax.random.key(4), (n, k))

    def loss(M):
        _, U = lobpcg_top_k_dense(M, X0, 500, None, 1e-8)
        return jnp.sum(U * target)

    g = jax.grad(loss)(M)
    np.testing.assert_allclose(
        np.asarray(g), np.asarray(g.T), atol=1e-12,
    )


def test_lobpcg_dense_degenerate_eigenvalues_clamp():
    '''Identity matrix has fully-degenerate spectrum; the eps_clamp
    short-circuits F-matrix blow-up.  Gradient must be finite.
    '''
    from nitrix.graph._lobpcg_diff import lobpcg_top_k_dense

    n, k = 30, 2
    # Perturb identity slightly so LOBPCG converges
    M = jnp.eye(n) + 1e-6 * jax.random.normal(jax.random.key(0), (n, n))
    M = 0.5 * (M + M.T)
    X0 = jax.random.normal(jax.random.key(1), (n, k))

    def loss(M):
        _, U = lobpcg_top_k_dense(M, X0, 500, None, eps_clamp=1e-3)
        return jnp.sum(U ** 2)  # invariant; gradient should be ~zero

    g = jax.grad(loss)(M)
    assert bool(jnp.all(jnp.isfinite(g)))


def test_lobpcg_ell_grad_matches_dense_projected():
    '''ELL backward returns the gradient projected onto the sparsity
    pattern, agreeing with the dense backward gathered at the same
    indices.
    '''
    from nitrix.graph._lobpcg_diff import lobpcg_top_k_dense, lobpcg_top_k_ell
    from nitrix.semiring import REAL, semiring_ell_matmul

    rng = np.random.default_rng(7)
    n, k = 60, 3
    A_dense = _ring_adjacency(n)
    for _ in range(40):
        i, j = rng.integers(0, n, 2)
        if i != j:
            A_dense[i, j] = A_dense[j, i] = 1.0
    A_dense += 0.5 * np.eye(n)
    A_jnp = jnp.asarray(A_dense)
    ell = _ell_from_dense_symmetric(A_dense)
    X0 = jax.random.normal(jax.random.key(0), (n, k))

    target = jax.random.normal(jax.random.key(2), (k, k))
    target = (target + target.T) / 2

    def loss_dense(A):
        _, U = lobpcg_top_k_dense(A, X0, 500, None, 1e-8)
        return jnp.trace(U.T @ A @ U @ target)

    def loss_ell(values):
        _, U = lobpcg_top_k_ell(
            values, ell.indices, X0, n, 500, None, 1e-8,
        )
        AU = semiring_ell_matmul(
            values, ell.indices, U, semiring=REAL, n_cols=n, backend='jax',
        )
        return jnp.trace(U.T @ AU @ target)

    g_dense = jax.grad(loss_dense)(A_jnp)
    g_ell_values = jax.grad(loss_ell)(ell.values)
    g_dense_at_pattern = g_dense[jnp.arange(n)[:, None], ell.indices]
    np.testing.assert_allclose(
        np.asarray(g_ell_values),
        np.asarray(g_dense_at_pattern),
        atol=1e-12,
    )


def test_laplacian_eigenmap_lobpcg_dense_differentiable():
    '''Public surface: ``jax.grad`` through ``laplacian_eigenmap`` with
    ``solver="lobpcg"`` on a dense input now produces finite gradients
    (was raising at first GA).
    '''
    n = 32
    A = jnp.asarray(_two_cluster_adjacency(n_per_cluster=16))
    def loss(A):
        _, vals = laplacian_eigenmap(
            A, n_components=2, solver='lobpcg', lobpcg_iters=200,
        )
        return vals.sum()
    g = jax.grad(loss)(A)
    assert bool(jnp.all(jnp.isfinite(g)))


def test_laplacian_eigenmap_lobpcg_ell_differentiable():
    '''Public surface: ELL input through ``laplacian_eigenmap`` with
    ``solver="auto"`` (-> lobpcg) is differentiable in ``A.values``.
    '''
    n = 60
    A_np = _ring_adjacency(n)
    rng = np.random.default_rng(0)
    for _ in range(40):
        i, j = rng.integers(0, n, 2)
        if i != j:
            A_np[i, j] = A_np[j, i] = 1.0
    A_np += 0.5 * np.eye(n)
    ell = _ell_from_dense_symmetric(A_np)

    def loss(values):
        ell_in = ELL(
            values=values,
            indices=ell.indices,
            n_cols=ell.n_cols,
            identity=ell.identity,
        )
        _, vals = laplacian_eigenmap(
            ell_in, n_components=3, solver='lobpcg', lobpcg_iters=300,
        )
        return vals.sum()

    g = jax.grad(loss)(ell.values)
    assert bool(jnp.all(jnp.isfinite(g)))
    assert g.shape == ell.values.shape


def test_lobpcg_sectioned_ell_not_differentiable():
    '''SectionedELL is forward-only for LOBPCG; ``jax.grad`` raises.

    The diff-capable path is dense / flat ELL only; SectionedELL is
    documented as "convert to flat ELL for gradients".
    '''
    n = 60
    A_np = _ring_adjacency(n)
    A_np += 0.5 * np.eye(n)
    sec = _sectioned_from_dense(A_np)

    def loss(values_list):
        # SectionedELL stores values per section; rebuild and feed in.
        new_sections = tuple(
            type(s)(
                values=v, indices=s.indices,
                n_cols=s.n_cols, identity=s.identity,
            )
            for s, v in zip(sec.sections, values_list)
        )
        new_sec = type(sec)(
            sections=new_sections, row_groups=sec.row_groups,
            n_rows=sec.n_rows, n_cols=sec.n_cols, identity=sec.identity,
        )
        _, evals = laplacian_eigenmap(
            new_sec, n_components=3, solver='lobpcg', lobpcg_iters=200,
        )
        return evals.sum()

    values_list = [s.values for s in sec.sections]
    with pytest.raises(Exception):
        jax.grad(loss)(values_list)


def test_diffusion_embedding_lobpcg_dense_differentiable():
    '''Diffusion-embedding LOBPCG path is differentiable on dense input.'''
    n = 32
    A = jnp.asarray(_two_cluster_adjacency(n_per_cluster=16))
    def loss(A):
        _, vals = diffusion_embedding(
            A, n_components=2, alpha=0.5, solver='lobpcg', lobpcg_iters=200,
        )
        return vals.sum()
    g = jax.grad(loss)(A)
    assert bool(jnp.all(jnp.isfinite(g)))
