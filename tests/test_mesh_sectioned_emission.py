# -*- coding: utf-8 -*-
"""SectionedELL operator emission for irregular-valence meshes (P0.3 / D3).

``mesh_cotangent_laplacian`` / ``mesh_k_ring_adjacency`` gain a ``format=``
kwarg.  Default ``'ell'`` is byte-identical to before; ``'sectioned'`` emits a
degree-bucketed operator that, applied through ``sparse.apply_operator``,
produces the *same* result as the flat ``ELL``.  ``'auto'`` sections only when
the degree spread is wide -- the near-uniform icosphere / fsaverage families
stay flat; a high-valence fan sections.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nitrix.sparse import (
    ELL,
    Mesh,
    SectionedELL,
    apply_operator,
    icosphere,
    mesh_cotangent_laplacian,
    mesh_k_ring_adjacency,
)


def _fan_mesh(n_ring: int = 20) -> Mesh:
    """A triangle fan: one high-valence centre + a low-valence ring."""
    angles = np.linspace(0.0, 2 * np.pi, n_ring, endpoint=False)
    ring = np.stack([np.cos(angles), np.sin(angles), np.zeros(n_ring)], axis=1)
    verts = np.concatenate([[[0.0, 0.0, 0.0]], ring], axis=0)
    faces = np.array([[0, 1 + i, 1 + (i + 1) % n_ring] for i in range(n_ring)])
    return Mesh(jnp.asarray(verts, jnp.float32), jnp.asarray(faces, jnp.int32))


def test_default_format_is_flat_ell() -> None:
    assert isinstance(mesh_cotangent_laplacian(icosphere(1)), ELL)
    assert isinstance(mesh_k_ring_adjacency(icosphere(1)), ELL)


def test_cotangent_sectioned_matches_ell_through_seam() -> None:
    mesh = icosphere(2)
    x = jax.random.normal(jax.random.PRNGKey(0), (mesh.n_vertices, 3))
    ell = mesh_cotangent_laplacian(mesh, format='ell')
    sec = mesh_cotangent_laplacian(mesh, format='sectioned')
    assert isinstance(sec, SectionedELL)
    assert np.allclose(
        np.asarray(apply_operator(ell, x)),
        np.asarray(apply_operator(sec, x)),
        atol=1e-5,
    )


def test_kring_sectioned_matches_ell_through_seam() -> None:
    mesh = icosphere(2)
    x = jax.random.normal(jax.random.PRNGKey(1), (mesh.n_vertices, 2))
    ell = mesh_k_ring_adjacency(mesh, k=1, format='ell')
    sec = mesh_k_ring_adjacency(mesh, k=1, format='sectioned')
    assert isinstance(sec, SectionedELL)
    assert np.allclose(
        np.asarray(apply_operator(ell, x)),
        np.asarray(apply_operator(sec, x)),
        atol=1e-5,
    )


def test_auto_keeps_uniform_mesh_flat() -> None:
    # Icosphere valence is 5-6 (ratio ~ 1) -> 'auto' stays on flat ELL.
    assert isinstance(
        mesh_cotangent_laplacian(icosphere(2), format='auto'), ELL
    )
    assert isinstance(mesh_k_ring_adjacency(icosphere(2), format='auto'), ELL)


def test_auto_sections_high_valence_fan() -> None:
    # Centre valence 20 vs ring valence ~3 -> wide spread -> 'auto' sections.
    fan = _fan_mesh(20)
    assert isinstance(mesh_k_ring_adjacency(fan, format='auto'), SectionedELL)
    assert isinstance(
        mesh_cotangent_laplacian(fan, format='auto'), SectionedELL
    )


def test_fan_sectioned_matches_ell() -> None:
    # Parity on the irregular mesh that motivates sectioning.
    fan = _fan_mesh(16)
    x = jax.random.normal(jax.random.PRNGKey(2), (fan.n_vertices, 1))
    ell = mesh_cotangent_laplacian(fan, format='ell')
    sec = mesh_cotangent_laplacian(fan, format='sectioned')
    assert np.allclose(
        np.asarray(apply_operator(ell, x)),
        np.asarray(apply_operator(sec, x)),
        atol=1e-5,
    )


def test_bad_format_raises() -> None:
    with pytest.raises(ValueError, match="'ell', 'sectioned', or 'auto'"):
        mesh_cotangent_laplacian(icosphere(0), format='bogus')


# --------------------------------------------------------------------------- #
# Vectorised cotangent assembly invariants (Tier C / audit AI-C1)
# --------------------------------------------------------------------------- #


def test_vectorised_cotangent_invariants() -> None:
    # The vectorised flat-ELL assembly must preserve the defining cotangent
    # Laplacian invariants: rows sum to zero (constants in the kernel) and the
    # operator is symmetric (w_ij == w_ji).
    mesh = icosphere(3)
    lap = mesh_cotangent_laplacian(mesh)
    assert isinstance(lap, ELL)
    n = mesh.n_vertices
    dense = np.asarray(apply_operator(lap, jnp.eye(n)[..., None])[..., 0]).T
    # The diagonal-first layout keeps column 0 == the vertex index.
    assert np.array_equal(np.asarray(lap.indices)[:, 0], np.arange(n))
    assert np.allclose(dense.sum(axis=1), 0.0, atol=1e-4)  # L @ 1 == 0
    assert np.allclose(dense, dense.T, atol=1e-5)  # symmetric


def test_vectorised_cotangent_real_mesh_rowsum_zero() -> None:
    # On a real irregular-valence cortical surface (where the old Python-dict
    # assembly was the bottleneck) the vectorised path still gives L @ 1 == 0.
    from _real_meshes import fsaverage_white

    v, f, _ = fsaverage_white()
    lap = mesh_cotangent_laplacian(Mesh(jnp.asarray(v), jnp.asarray(f)))
    ones = jnp.ones((v.shape[0], 1))
    row_sums = np.asarray(apply_operator(lap, ones)[..., 0])
    assert np.allclose(row_sums, 0.0, atol=1e-3)
