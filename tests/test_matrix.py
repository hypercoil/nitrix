# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for specialised matrix operations
"""
import pytest
import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import toeplitz as toeplitz_ref
from nitrix.functional import (
    toeplitz,
    symmetric,
    recondition_eigenspaces,
    delete_diagonal,
    fill_diagonal,
    sym2vec,
    vec2sym,
    squareform
)
from hypercoil.engine import vmap_over_outer


TOL = 5e-3
def approx(out, ref):
    return np.allclose(out, ref, atol=TOL)


@pytest.fixture(scope='module')
def c():
    return np.random.rand(3)

@pytest.fixture(scope='module')
def r(c):
    r = np.random.rand(3)
    r[0] = c[0]
    return r

@pytest.fixture(scope='module')
def C():
    return np.random.rand(3, 3)

@pytest.fixture(scope='module')
def R(C):
    R = np.random.rand(3, 4)
    R[:, 0] = C[:, 0]
    return R

@pytest.fixture(scope='module')
def B():
    return np.random.rand(20, 10, 10)

@pytest.fixture(scope='module')
def BLR():
    BLR = np.random.rand(20, 10, 2)
    return BLR @ BLR.swapaxes(-1, -2)


def test_symmetric(B):
    out = symmetric(B)
    ref = out.swapaxes(-1, -2)
    assert approx(out, ref)


def test_toeplitz(c, r):
    out = toeplitz(c, r)
    ref = toeplitz_ref(c, r)
    assert approx(out, ref)


def test_toeplitz_stack(C, R):
    C0 = np.random.rand(3, 10)
    R0 = np.random.rand(2, 3, 8)
    out = toeplitz(C0, R0)
    assert out.shape == (2, 3, 10, 8)

    out = jax.jit(toeplitz)(C0, R0)
    assert out.shape == (2, 3, 10, 8)

    rr = R0[0, 1]
    cc = C0[1]
    ref = toeplitz(cc, rr)
    assert approx(out[0, 1], ref)

    out = toeplitz(C, R)
    ref = np.stack(
        [toeplitz_ref(c, r) for c, r in zip(C, R)],
        axis=0,
    )
    assert approx(out, ref)


def test_toeplitz_extend(C, R):
    shape = (10, 8)
    out = toeplitz(C, R, shape=shape)
    assert out.shape == (3, 10, 8)
    Cx, Rx = (np.zeros((C.shape[0], shape[0])),
                np.zeros((R.shape[0], shape[1])))
    Cx[:, :C.shape[-1]] = C
    Rx[:, :R.shape[-1]] = R
    ref = np.stack([toeplitz_ref(c, r) for c, r in zip(Cx, Rx)])
    assert approx(out, ref)


def test_toeplitz_fill(C, R):
    f = 2
    shape = (8, 8)
    out = toeplitz(C, R, shape=shape, fill_value=f)
    assert out.shape == (3, 8, 8)
    #out = toeplitz(self.C, self.R, shape=shape, fill_value=self.f)
    Cx, Rx = (
        np.zeros((C.shape[0], shape[0])) + f,
        np.zeros((R.shape[0], shape[1])) + f
    )
    Cx[:, :C.shape[-1]] = C
    Rx[:, :R.shape[-1]] = R
    ref = np.stack([toeplitz_ref(c, r) for c, r in zip(Cx, Rx)])
    assert approx(out, ref)


def test_recondition():
    key = jax.random.PRNGKey(np.random.randint(0, 2**32))
    V = jnp.ones((7, 3))
    d = d = jax.grad(
        lambda X: jnp.linalg.svd(X, full_matrices=False)[0].sum())
    assert np.all(np.isnan(d(V @ V.T)))

    arg = recondition_eigenspaces(V @ V.T, key=key, psi=1e-3, xi=1e-3)
    assert np.logical_not(np.any(np.isnan(d(arg))))


def test_sym2vec_correct():
    from scipy.spatial.distance import squareform
    K = symmetric(np.random.rand(3, 4, 5, 5))
    out = sym2vec(K)

    ref = np.stack([
        np.stack([
            squareform(j * (1 - np.eye(j.shape[0])))
            for j in k
        ]) for k in K
    ])
    assert np.allclose(out, ref)


def test_fill_diag():
    d = 6
    key = jax.random.PRNGKey(np.random.randint(0, 2**32))
    A = jax.random.uniform(key=key, shape=(2, 2, 2, d, d))
    A_fd = fill_diagonal(A)
    A_dd = delete_diagonal(A)
    assert np.allclose(A_fd, A_dd)
    A_fd = fill_diagonal(A, 5)
    assert np.allclose(A_fd, A_dd + 5 * np.eye(d))

    grad = jax.grad(lambda A: fill_diagonal(3 * A, 4).sum())(A)
    grad_ref = 3. * ~np.eye(d, dtype=bool)
    assert np.allclose(grad, grad_ref)

    d2 = 3
    A = jax.random.uniform(key=key, shape=(3, 1, 4, d, d2))
    A_fd = fill_diagonal(A, offset=-1, fill=float('nan'))
    ref = jnp.diagflat(jnp.ones(d, dtype=bool), k=-1)
    ref = ref[:d, :d2]
    assert np.all(np.isnan(A_fd.sum((0, 1, 2))) == ref)


def test_sym2vec_inversion():
    K = symmetric(np.random.rand(3, 4, 5, 5))
    out = vec2sym(sym2vec(K, offset=0), offset=0)
    assert np.allclose(out, K)


def test_squareform_equivalence():
    K = symmetric(np.random.rand(3, 4, 5, 5))
    out = squareform(K)
    ref = sym2vec(K)
    assert np.allclose(out, ref)

    out = squareform(out)
    ref = vec2sym(ref)
    assert np.allclose(out, ref)
