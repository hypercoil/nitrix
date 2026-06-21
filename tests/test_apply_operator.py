# -*- coding: utf-8 -*-
"""The format-agnostic ``sparse.apply_operator`` seam (geometry-suite P0.4).

Same logical operator, two storage formats (flat ``ELL`` / bucketed
``SectionedELL``), one call site -> identical results.  This is the seam the
surface-algorithm layer (curvature, distortion, smoothing) builds on so it
never branches on storage format.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nitrix.semiring import REAL, semiring_ell_matmul
from nitrix.sparse import (
    ELL,
    apply_operator,
    sectioned_ell_from_ragged,
    sectioned_semiring_ell_matmul,
)

# A 3x3 logical sparse matrix with varying row degree (1, 3, 2):
#   row 0: [0]->1
#   row 1: [0]->2 [1]->3 [2]->4
#   row 2: [1]->5 [2]->6
_DENSE = np.array(
    [[1.0, 0.0, 0.0], [2.0, 3.0, 4.0], [0.0, 5.0, 6.0]], dtype=np.float32
)


def _as_ell() -> ELL:
    values = jnp.array([[1.0, 0.0, 0.0], [2.0, 3.0, 4.0], [5.0, 6.0, 0.0]])
    indices = jnp.array([[0, 0, 0], [0, 1, 2], [1, 2, 0]])
    return ELL(values=values, indices=indices, n_cols=3, identity=0.0)


def _as_sectioned():
    values = [
        jnp.array([1.0]),
        jnp.array([2.0, 3.0, 4.0]),
        jnp.array([5.0, 6.0]),
    ]
    indices = [jnp.array([0]), jnp.array([0, 1, 2]), jnp.array([1, 2])]
    return sectioned_ell_from_ragged(values, indices, n_cols=3)


def test_apply_ell_matches_direct_matmul() -> None:
    ell = _as_ell()
    x = jnp.arange(6.0).reshape(3, 2)
    out = apply_operator(ell, x, semiring=REAL)
    ref = semiring_ell_matmul(
        ell.values, ell.indices, x, semiring=REAL, n_cols=3, backend='jax'
    )
    assert np.allclose(np.asarray(out), np.asarray(ref))


def test_apply_sectioned_matches_direct() -> None:
    sec = _as_sectioned()
    x = jnp.arange(6.0).reshape(3, 2)
    out = apply_operator(sec, x, semiring=REAL)
    ref = sectioned_semiring_ell_matmul(sec, x, semiring=REAL)
    assert np.allclose(np.asarray(out), np.asarray(ref))


def test_ell_and_sectioned_agree_through_seam() -> None:
    # The load-bearing parity: identical logical operator, both formats.
    x = jnp.eye(3)  # M @ I == dense M
    out_ell = apply_operator(_as_ell(), x, semiring=REAL)
    out_sec = apply_operator(_as_sectioned(), x, semiring=REAL)
    assert np.allclose(np.asarray(out_ell), _DENSE)
    assert np.allclose(np.asarray(out_sec), _DENSE)
    assert np.allclose(np.asarray(out_ell), np.asarray(out_sec))


def test_apply_default_semiring_is_real() -> None:
    ell = _as_ell()
    x = jnp.eye(3)
    assert np.allclose(
        np.asarray(apply_operator(ell, x)),
        np.asarray(apply_operator(ell, x, semiring=REAL)),
    )


def test_apply_batched_leading_axes() -> None:
    ell = _as_ell()
    x = jnp.stack([jnp.eye(3), 2.0 * jnp.eye(3)])  # (2, 3, 3)
    out = apply_operator(ell, x, semiring=REAL)
    assert out.shape == (2, 3, 3)
    assert np.allclose(np.asarray(out[0]), _DENSE)
    assert np.allclose(np.asarray(out[1]), 2.0 * _DENSE)


def test_apply_requires_2d_core() -> None:
    ell = _as_ell()
    with pytest.raises(ValueError, match='at least 2-D|n, d'):
        apply_operator(ell, jnp.ones((3,)), semiring=REAL)


def test_apply_rejects_unknown_operator() -> None:
    with pytest.raises(TypeError, match='ELL or SectionedELL'):
        apply_operator(object(), jnp.eye(3), semiring=REAL)  # type: ignore[arg-type]


def test_apply_is_differentiable() -> None:
    ell = _as_ell()

    def loss(x: jax.Array) -> jax.Array:
        return jnp.sum(apply_operator(ell, x, semiring=REAL) ** 2)

    g = jax.grad(loss)(jnp.eye(3))
    assert g.shape == (3, 3)
    assert np.any(np.asarray(g) != 0.0)
