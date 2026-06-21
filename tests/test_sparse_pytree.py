# -*- coding: utf-8 -*-
"""Pytree registration of the ``nitrix.sparse`` array-bearing containers.

Covers B22 (``register-sparse-dataclasses-as-pytrees``) / geometry-suite P0.1:
``ELL`` / ``SectionedELL`` / ``Mesh`` are registered JAX pytrees so they cross
``jit`` / ``vmap`` / ``grad`` boundaries as first-class operands;
``IcosphereHierarchy`` is deliberately *not* registered.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from nitrix.semiring import REAL, semiring_ell_matmul
from nitrix.sparse import (
    ELL,
    IcosphereHierarchy,
    Mesh,
    icosphere,
    icosphere_hierarchy,
    sectioned_ell_from_ragged,
    sectioned_semiring_ell_matmul,
)


def _small_ell() -> ELL:
    return ELL(
        values=jnp.array([[1.0, 2.0], [3.0, 0.0]]),
        indices=jnp.array([[0, 1], [1, 0]]),
        n_cols=2,
        identity=0.0,
    )


# --------------------------------------------------------------------------- #
# ELL
# --------------------------------------------------------------------------- #


def test_ell_flatten_leaves_are_the_arrays() -> None:
    ell = _small_ell()
    leaves = jax.tree_util.tree_leaves(ell)
    # The trap (tree_leaves(ell) == [ell]) is gone: leaves are the arrays.
    assert len(leaves) == 2
    assert any(
        np.array_equal(np.asarray(x), np.asarray(ell.values)) for x in leaves
    )


def test_ell_roundtrip() -> None:
    ell = _small_ell()
    leaves, treedef = jax.tree_util.tree_flatten(ell)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(rebuilt, ELL)
    assert np.array_equal(np.asarray(rebuilt.values), np.asarray(ell.values))
    assert np.array_equal(np.asarray(rebuilt.indices), np.asarray(ell.indices))
    assert rebuilt.n_cols == ell.n_cols
    assert rebuilt.identity == ell.identity


def test_ell_tree_map_touches_arrays_not_object() -> None:
    ell = _small_ell()
    doubled = jax.tree_util.tree_map(lambda x: x * 2, ell)
    assert isinstance(doubled, ELL)
    assert np.allclose(np.asarray(doubled.values), 2 * np.asarray(ell.values))
    # aux (n_cols / identity) is preserved unchanged.
    assert doubled.n_cols == ell.n_cols and doubled.identity == ell.identity


def test_ell_as_jit_argument() -> None:
    ell = _small_ell()
    b = jnp.array([[1.0], [10.0]])

    def f(e: ELL, b: jax.Array) -> jax.Array:
        return semiring_ell_matmul(
            e.values,
            e.indices,
            b,
            semiring=REAL,
            n_cols=e.n_cols,
            backend='jax',
        )

    eager = f(ell, b)
    jitted = jax.jit(f)(ell, b)
    assert np.allclose(np.asarray(eager), np.asarray(jitted))


def test_ell_vmap_over_stack() -> None:
    ells = [
        ELL(
            values=jnp.array([[1.0, 2.0], [3.0, 0.0]]) * k,
            indices=jnp.array([[0, 1], [1, 0]]),
            n_cols=2,
            identity=0.0,
        )
        for k in (1.0, 2.0, 3.0)
    ]
    stacked = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *ells)
    out = jax.vmap(lambda e: e.values.sum())(stacked)
    expected = jnp.array([float(e.values.sum()) for e in ells])
    assert np.allclose(np.asarray(out), np.asarray(expected))


def test_ell_grad_structure_values_populated_indices_float0() -> None:
    ell = _small_ell()
    b = jnp.array([[1.0], [10.0]])

    def loss(e: ELL) -> jax.Array:
        y = semiring_ell_matmul(
            e.values,
            e.indices,
            b,
            semiring=REAL,
            n_cols=e.n_cols,
            backend='jax',
        )
        return jnp.sum(y**2)

    # Differentiating the whole container needs allow_int=True (indices is an
    # integer leaf); the contract is values-populated, indices-float0.
    g = jax.grad(loss, allow_int=True)(ell)
    assert isinstance(g, ELL)
    assert g.values.shape == ell.values.shape
    assert jnp.issubdtype(g.values.dtype, jnp.floating)
    assert np.any(np.asarray(g.values) != 0.0)
    assert g.indices.dtype == jax.dtypes.float0


def test_ell_grad_wrt_values_directly() -> None:
    # The common pattern: grad w.r.t. the float values, ELL rebuilt in-loss.
    ell = _small_ell()
    b = jnp.array([[1.0], [10.0]])

    def loss(values: jax.Array) -> jax.Array:
        y = semiring_ell_matmul(
            values,
            ell.indices,
            b,
            semiring=REAL,
            n_cols=ell.n_cols,
            backend='jax',
        )
        return jnp.sum(y**2)

    g = jax.grad(loss)(ell.values)
    assert g.shape == ell.values.shape
    assert np.any(np.asarray(g) != 0.0)


# --------------------------------------------------------------------------- #
# Mesh
# --------------------------------------------------------------------------- #


def test_mesh_roundtrip_and_jit() -> None:
    mesh = icosphere(1)
    leaves, treedef = jax.tree_util.tree_flatten(mesh)
    assert len(leaves) == 2
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(rebuilt, Mesh)
    assert np.array_equal(np.asarray(rebuilt.faces), np.asarray(mesh.faces))

    centroid = jax.jit(lambda m: m.vertices.mean(axis=0))(mesh)
    assert np.allclose(np.asarray(centroid), np.asarray(mesh.vertices.mean(0)))


def test_mesh_grad_through_vertices() -> None:
    mesh = icosphere(1)

    def loss(m: Mesh) -> jax.Array:
        return jnp.sum(m.vertices**2)

    # Whole-Mesh grad needs allow_int=True (faces is an integer leaf).
    g = jax.grad(loss, allow_int=True)(mesh)
    assert isinstance(g, Mesh)
    assert np.allclose(np.asarray(g.vertices), 2 * np.asarray(mesh.vertices))
    assert g.faces.dtype == jax.dtypes.float0

    # The surface-optimiser pattern: grad w.r.t. vertices, Mesh rebuilt inside.
    gv = jax.grad(lambda v: jnp.sum(Mesh(v, mesh.faces).vertices ** 2))(
        mesh.vertices
    )
    assert np.allclose(np.asarray(gv), 2 * np.asarray(mesh.vertices))


# --------------------------------------------------------------------------- #
# SectionedELL
# --------------------------------------------------------------------------- #


def _small_sectioned():
    # Rows of degree 1 and 3 -> two buckets.
    values = [jnp.array([1.0]), jnp.array([2.0, 3.0, 4.0])]
    indices = [jnp.array([0]), jnp.array([0, 1, 2])]
    return sectioned_ell_from_ragged(values, indices, n_cols=3)


def test_sectioned_roundtrip() -> None:
    sec = _small_sectioned()
    leaves, treedef = jax.tree_util.tree_flatten(sec)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert rebuilt.n_rows == sec.n_rows
    assert rebuilt.n_cols == sec.n_cols
    assert rebuilt.n_buckets == sec.n_buckets


def test_sectioned_as_jit_argument_matches_eager() -> None:
    sec = _small_sectioned()
    b = jnp.eye(3)
    eager = sectioned_semiring_ell_matmul(sec, b, semiring=REAL)
    jitted = jax.jit(
        lambda s, b: sectioned_semiring_ell_matmul(s, b, semiring=REAL)
    )(sec, b)
    assert np.allclose(np.asarray(eager), np.asarray(jitted))


def test_sectioned_closed_over_constant_still_works() -> None:
    sec = _small_sectioned()
    b = jnp.eye(3)
    eager = sectioned_semiring_ell_matmul(sec, b, semiring=REAL)
    g = jax.jit(lambda b: sectioned_semiring_ell_matmul(sec, b, semiring=REAL))
    assert np.allclose(np.asarray(eager), np.asarray(g(b)))


# --------------------------------------------------------------------------- #
# IcosphereHierarchy: deliberately NOT a pytree
# --------------------------------------------------------------------------- #


def test_icosphere_hierarchy_is_not_a_pytree() -> None:
    hier = icosphere_hierarchy(1)
    # An un-registered container is an opaque single leaf.
    leaves = jax.tree_util.tree_leaves(hier)
    assert leaves == [hier]
    assert isinstance(hier, IcosphereHierarchy)
