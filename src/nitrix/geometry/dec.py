# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
r"""
Discrete exterior calculus on triangle meshes.

The discrete-exterior-calculus (DEC) operator stack that the cotangent Laplacian
is one instance of. On a triangle mesh a scalar field is a *0-form* (on
vertices), a tangential field is a *1-form* (on oriented edges), and a density is
a *2-form* (on faces); the discrete exterior derivatives and Hodge stars connect
them:

- :func:`mesh_gradient` -- the exterior derivative :math:`d_0` (vertex 0-form
  :math:`\to` edge 1-form): the signed vertex--edge incidence matrix.
- :func:`mesh_curl` -- the exterior derivative :math:`d_1` (edge 1-form
  :math:`\to` face 2-form): the signed edge--face incidence matrix. By
  construction :math:`d_1 d_0 = 0` (the discrete :math:`d^2 = 0`).
- :func:`mesh_star_k` -- the diagonal Hodge stars :math:`\star_0` (dual vertex
  areas), :math:`\star_1` (cotangent edge weights -- the same weights
  :func:`~nitrix.sparse.mesh_cotangent_laplacian` assembles, so
  :math:`d_0^\top \star_1 d_0` **is** that Laplacian) and :math:`\star_2`
  (reciprocal face areas).
- :func:`mesh_divergence` -- the codifferential on 1-forms,
  :math:`\star_0^{-1} d_0^\top \star_1` (edge 1-form :math:`\to` vertex 0-form).
- :func:`hodge_decompose` -- the discrete Helmholtz--Hodge decomposition of an
  edge 1-form into a curl-free (exact), divergence-free (coexact) and harmonic
  part, by two matrix-free Poisson solves.

Every operator is returned as an :class:`~nitrix.sparse.ELL`, so it applies
through :func:`~nitrix.sparse.apply_operator`; the topology and orientation are
built host-side (a combinatorial function of the connectivity only), like
:func:`~nitrix.sparse.mesh_cotangent_laplacian`. The Hodge stars split on
differentiability: :math:`\star_0` and :math:`\star_2` (:func:`mesh_star_k`
``k=0``/``k=2``) are pure-JAX functions of the vertex positions and *are*
differentiable with respect to them, whereas :math:`\star_1` (``k=1``, and hence
:func:`mesh_divergence` and the :math:`\star_1` inner product of
:func:`hodge_decompose`) is assembled host-side from the cotangent weights and is
*not* differentiable with respect to the vertex coordinates.
:func:`hodge_decompose` is differentiable with respect to its input 1-form.

References
----------
Desbrun M, Hirani AN, Leok M, Marsden JE (2005). Discrete exterior calculus.
https://arxiv.org/abs/math/0508341
"""

from __future__ import annotations

from typing import Any, NamedTuple, Tuple

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float
from numpy.typing import NDArray

from ..linalg import cg
from ..sparse import Mesh
from ..sparse.ell import ELL
from ..sparse.mesh import _cotangent_weights, face_areas

__all__ = [
    'HodgeDecomposition',
    'HodgeOperator',
    'hodge_apply',
    'hodge_decompose',
    'hodge_operator',
    'mesh_curl',
    'mesh_divergence',
    'mesh_gradient',
    'mesh_star_k',
]


def _edge_topology(
    faces_np: NDArray[Any],
) -> Tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
    """Unique oriented edges and per-face edge indices / signs.

    Returns ``edges`` ``(E, 2)`` (each ``[i, j]`` with ``i < j``, the positive
    orientation :math:`i \\to j`), ``face_edge`` ``(F, 3)`` the edge index of each
    face's three directed boundary edges ``(a\\to b, b\\to c, c\\to a)``, and
    ``face_sign`` ``(F, 3)`` ``+1`` when that directed edge matches the edge's
    positive orientation, else ``-1``.
    """
    n_faces = faces_np.shape[0]
    directed = np.concatenate(
        [faces_np[:, [0, 1]], faces_np[:, [1, 2]], faces_np[:, [2, 0]]], axis=0
    )  # (3F, 2), in face-major order per column block
    undirected = np.sort(directed, axis=1)
    edges, inv = np.unique(undirected, axis=0, return_inverse=True)
    sign = np.where(directed[:, 0] < directed[:, 1], 1.0, -1.0)
    # The concatenation is [all a->b | all b->c | all c->a]; reshape to (F, 3).
    face_edge = inv.reshape(3, n_faces).T.astype(np.int64)
    face_sign = sign.reshape(3, n_faces).T
    return edges, face_edge, face_sign


def _star1_values(
    verts_np: NDArray[Any],
    faces_np: NDArray[Any],
    face_edge: NDArray[Any],
    n_edges: int,
) -> NDArray[Any]:
    """Cotangent Hodge-star :math:`\\star_1` weights, one per edge.

    The combined cotangent weight assembled exactly as
    :func:`~nitrix.sparse.mesh_cotangent_laplacian`, so
    :math:`d_0^\\top \\operatorname{diag}(\\star_1) d_0` equals that Laplacian.
    ``cot[:, k]`` (opposite local vertex ``k``) contributes to the edge of the
    other two vertices, which is directed edge ``(k + 1) mod 3`` of the face.
    """
    cot = _cotangent_weights(verts_np, faces_np)  # (F, 3)
    star1 = np.zeros(n_edges, dtype=np.float64)
    for k in range(3):
        np.add.at(star1, face_edge[:, (k + 1) % 3], cot[:, k])
    return star1


def _diagonal_ell(values: Array, n: int) -> ELL:
    """A diagonal ``(n, n)`` operator as an ELL (one entry per row)."""
    return ELL(
        values=values[:, None],
        indices=jnp.arange(n, dtype=jnp.int32)[:, None],
        n_cols=n,
        identity=0.0,
    )


def mesh_gradient(mesh: Mesh) -> ELL:
    r"""Discrete gradient :math:`d_0` (vertex 0-form :math:`\to` edge 1-form).

    The signed vertex--edge incidence operator: row ``e`` (edge :math:`i \to j`,
    :math:`i < j`) has :math:`-1` in column :math:`i` and :math:`+1` in column
    :math:`j`, so :math:`(d_0 f)_e = f_j - f_i` is the difference of the scalar
    along the edge.

    Parameters
    ----------
    mesh
        Triangle mesh.

    Returns
    -------
    ELL
        The :math:`(E, V)` operator (``E`` edges, ``V`` vertices).
    """
    faces_np = np.asarray(mesh.faces)
    edges, _, _ = _edge_topology(faces_np)
    dtype = mesh.vertices.dtype
    values = jnp.broadcast_to(
        jnp.asarray([-1.0, 1.0], dtype=dtype), edges.shape
    )
    return ELL(
        values=values,
        indices=jnp.asarray(edges.astype(np.int32)),
        n_cols=mesh.n_vertices,
        identity=0.0,
    )


def mesh_curl(mesh: Mesh) -> ELL:
    r"""Discrete curl :math:`d_1` (edge 1-form :math:`\to` face 2-form).

    The signed edge--face incidence operator: row ``f`` sums the 1-form around
    the face's oriented boundary, :math:`(d_1 \omega)_f = \sum_k s_{fk}\,
    \omega_{e_{fk}}` with :math:`s_{fk} = \pm 1` per boundary edge. Satisfies the
    discrete :math:`d^2 = 0`: :math:`d_1 d_0 = 0` exactly.

    Parameters
    ----------
    mesh
        Triangle mesh.

    Returns
    -------
    ELL
        The :math:`(F, E)` operator (``F`` faces, ``E`` edges).
    """
    faces_np = np.asarray(mesh.faces)
    edges, face_edge, face_sign = _edge_topology(faces_np)
    dtype = mesh.vertices.dtype
    return ELL(
        values=jnp.asarray(face_sign, dtype=dtype),
        indices=jnp.asarray(face_edge.astype(np.int32)),
        n_cols=edges.shape[0],
        identity=0.0,
    )


def mesh_star_k(mesh: Mesh, k: int) -> ELL:
    r"""Diagonal Hodge star :math:`\star_k` for ``k`` in ``{0, 1, 2}``.

    Returns the diagonal DEC Hodge star mapping primal ``k``-forms to dual
    ``(2-k)``-forms:

    - ``k = 0`` -- :math:`\star_0`, the dual vertex (barycentric) areas
      :math:`(V, V)`.
    - ``k = 1`` -- :math:`\star_1`, the cotangent edge weights
      :math:`(\cot\alpha + \cot\beta)/2` :math:`(E, E)` (the
      :func:`~nitrix.sparse.mesh_cotangent_laplacian` weights).
    - ``k = 2`` -- :math:`\star_2`, the reciprocal face areas :math:`(F, F)`.

    Parameters
    ----------
    mesh
        Triangle mesh.
    k : int
        The form degree ``0``, ``1`` or ``2``.

    Returns
    -------
    ELL
        The diagonal Hodge-star operator.

    Notes
    -----
    The ``k=0`` (:math:`\star_0`) and ``k=2`` (:math:`\star_2`) branches are pure
    JAX functions of the vertex positions -- jittable through ``mesh`` and
    differentiable with respect to the vertex coordinates. The ``k=1``
    (:math:`\star_1`) branch assembles the cotangent weights host-side: its
    sparsity is fixed by the (concrete) connectivity and its values are not
    differentiable with respect to the vertex coordinates, so that branch is not
    jittable through the ``mesh`` argument.
    """
    dtype = mesh.vertices.dtype
    if k == 0:
        star0 = _barycentric_vertex_areas(mesh)
        return _diagonal_ell(star0, mesh.n_vertices)
    if k == 2:
        star2 = 1.0 / face_areas(mesh)
        return _diagonal_ell(star2, mesh.n_faces)
    if k == 1:
        verts_np = np.asarray(mesh.vertices, dtype=np.float64)
        faces_np = np.asarray(mesh.faces)
        edges, face_edge, _ = _edge_topology(faces_np)
        star1 = _star1_values(verts_np, faces_np, face_edge, edges.shape[0])
        return _diagonal_ell(
            jnp.asarray(star1.astype(np.float64)).astype(dtype), edges.shape[0]
        )
    raise ValueError(f'mesh_star_k: k={k} must be 0, 1 or 2.')


def _barycentric_vertex_areas(mesh: Mesh) -> Float[Array, ' n_vertices']:
    """Barycentric dual vertex areas: each face donates area/3 to each vertex."""
    areas = face_areas(mesh) / 3.0
    faces = mesh.faces
    out = jnp.zeros((mesh.n_vertices,), dtype=areas.dtype)
    for c in range(3):
        out = out.at[faces[:, c]].add(areas)
    return out


def mesh_divergence(mesh: Mesh) -> ELL:
    r"""Discrete divergence :math:`\star_0^{-1} d_0^\top \star_1`.

    The codifferential on 1-forms (edge 1-form :math:`\to` vertex 0-form): the
    area-normalised, cotangent-weighted signed edge sum incident to each vertex.
    The composition with :func:`mesh_gradient` is the (mass-normalised) cotangent
    Laplacian.

    Parameters
    ----------
    mesh
        Triangle mesh.

    Returns
    -------
    ELL
        The :math:`(V, E)` operator.
    """
    verts_np = np.asarray(mesh.vertices, dtype=np.float64)
    faces_np = np.asarray(mesh.faces)
    edges, face_edge, _ = _edge_topology(faces_np)
    n_edges = edges.shape[0]
    n = mesh.n_vertices
    star1 = _star1_values(verts_np, faces_np, face_edge, n_edges)
    star0 = np.asarray(_barycentric_vertex_areas(mesh), dtype=np.float64)

    # Each edge e = (i, j) contributes to vertex i (-star1[e]) and j (+star1[e]).
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([np.arange(n_edges), np.arange(n_edges)])
    vals = np.concatenate([-star1, star1]) / star0[rows]

    order = np.argsort(rows, kind='stable')
    rows_s, cols_s, vals_s = rows[order], cols[order], vals[order]
    deg = np.bincount(rows_s, minlength=n)
    k_max = int(deg.max(initial=1))
    start = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(deg, out=start[1:])
    slot = np.arange(rows_s.shape[0], dtype=np.int64) - start[rows_s]
    indices = np.zeros((n, k_max), dtype=np.int32)
    values = np.zeros((n, k_max), dtype=np.float64)
    indices[rows_s, slot] = cols_s
    values[rows_s, slot] = vals_s
    return ELL(
        values=jnp.asarray(values).astype(mesh.vertices.dtype),
        indices=jnp.asarray(indices),
        n_cols=n_edges,
        identity=0.0,
    )


class HodgeDecomposition(NamedTuple):
    """A Helmholtz--Hodge decomposition of an edge 1-form.

    The three parts sum to the input, are mutually orthogonal in the
    :math:`\\star_1` inner product, and are respectively curl-free,
    divergence-free, and harmonic.

    Attributes
    ----------
    exact : Float[Array, ' n_edges']
        The curl-free (gradient) part :math:`d_0 \\alpha`.
    coexact : Float[Array, ' n_edges']
        The divergence-free (curl) part.
    harmonic : Float[Array, ' n_edges']
        The harmonic remainder (zero on a genus-0 surface).
    """

    exact: Float[Array, ' n_edges']
    coexact: Float[Array, ' n_edges']
    harmonic: Float[Array, ' n_edges']


class HodgeOperator(NamedTuple):
    """Prebuilt DEC operator state for :func:`hodge_apply`.

    The mesh-derived operators the Helmholtz--Hodge solve needs -- the discrete
    gradient :math:`d_0`, the discrete curl :math:`d_1` and the cotangent Hodge
    star :math:`\\star_1` -- built once by :func:`hodge_operator` so the two
    Poisson solves run under :func:`jax.jit` without re-tracing the host-side
    topology construction.  It is the ``fit`` state of the decomposition's
    fit/apply pair.

    Attributes
    ----------
    gradient : ELL
        The discrete gradient :math:`d_0` (:func:`mesh_gradient`); its edge
        endpoints supply :math:`d_0` / :math:`d_0^\\top` and its ``n_cols``
        carries the (static) vertex count.
    curl : ELL
        The discrete curl :math:`d_1` (:func:`mesh_curl`); its per-face edge
        indices and signs supply :math:`d_1` / :math:`d_1^\\top`.
    star1 : Float[Array, ' n_edges']
        The cotangent Hodge-star :math:`\\star_1` weights (one per edge).
    """

    gradient: ELL
    curl: ELL
    star1: Float[Array, ' n_edges']


def hodge_operator(mesh: Mesh) -> HodgeOperator:
    """Build the DEC operators for :func:`hodge_apply` (the host-side step).

    The construction half of the Helmholtz--Hodge fit/apply pair: assembles the
    incidence operators and the cotangent star once (host-side -- a combinatorial
    function of the connectivity plus the cotangent metric) so that repeated
    :func:`hodge_apply` calls on the same mesh reuse them.

    Parameters
    ----------
    mesh
        Triangle mesh (concrete -- the operators are built host-side).

    Returns
    -------
    HodgeOperator
        The reusable ``(gradient, curl, star1)`` state.
    """
    d0 = mesh_gradient(mesh)
    d1 = mesh_curl(mesh)
    verts_np = np.asarray(mesh.vertices, dtype=np.float64)
    faces_np = np.asarray(mesh.faces)
    edges, face_edge, _ = _edge_topology(faces_np)
    star1 = jnp.asarray(
        _star1_values(verts_np, faces_np, face_edge, edges.shape[0])
    ).astype(mesh.vertices.dtype)
    return HodgeOperator(gradient=d0, curl=d1, star1=star1)


def hodge_apply(
    omega: Float[Array, ' n_edges'],
    operator: HodgeOperator,
    *,
    tol: float = 1e-9,
    ridge: float = 1e-8,
) -> HodgeDecomposition:
    r"""Apply a prebuilt Hodge operator to decompose an edge 1-form.

    The apply half of the decomposition's fit/apply pair; see
    :func:`hodge_decompose` for the mathematics.  As ``operator`` is a
    plain-array pytree, this call is fully jittable and differentiable with
    respect to ``omega``.

    Parameters
    ----------
    omega : Float[Array, ' n_edges']
        The input 1-form on the edges (in the :func:`mesh_gradient` edge order).
    operator : HodgeOperator
        The state from :func:`hodge_operator`.
    tol : float, optional
        Conjugate-gradient tolerance. Default ``1e-9``.
    ridge : float, optional
        Small Tikhonov ridge stabilising the singular (constant-null-space)
        Poisson systems. Default ``1e-8``.

    Returns
    -------
    HodgeDecomposition
        The ``(exact, coexact, harmonic)`` parts, each a ``(n_edges,)`` 1-form.
    """
    n_v = operator.gradient.n_cols
    ei = operator.gradient.indices[:, 0]
    ej = operator.gradient.indices[:, 1]
    fe = operator.curl.indices  # (F, 3)
    fs = operator.curl.values  # (F, 3)
    star1 = operator.star1
    n_edges = star1.shape[0]

    def d0(alpha: Array) -> Array:  # (V,) -> (E,)
        return alpha[ej] - alpha[ei]

    def d0t(w: Array) -> Array:  # (E,) -> (V,)
        return jnp.zeros((n_v,), w.dtype).at[ej].add(w).at[ei].add(-w)

    def d1(w: Array) -> Array:  # (E,) -> (F,)
        return jnp.sum(fs * w[fe], axis=-1)

    def d1t(u: Array) -> Array:  # (F,) -> (E,)
        contrib = (fs * u[:, None]).reshape(-1)
        return jnp.zeros((n_edges,), u.dtype).at[fe.reshape(-1)].add(contrib)

    def laplace0(alpha: Array) -> Array:
        return d0t(star1 * d0(alpha))

    def laplace2(u: Array) -> Array:
        return d1((1.0 / star1) * d1t(u))

    alpha = cg(laplace0, d0t(star1 * omega), tol=tol, l2=ridge)
    exact = d0(alpha)

    u = cg(laplace2, d1(omega), tol=tol, l2=ridge)
    coexact = (1.0 / star1) * d1t(u)

    harmonic = omega - exact - coexact
    return HodgeDecomposition(exact=exact, coexact=coexact, harmonic=harmonic)


def hodge_decompose(
    omega: Float[Array, ' n_edges'],
    mesh: Mesh,
    *,
    tol: float = 1e-9,
    ridge: float = 1e-8,
) -> HodgeDecomposition:
    r"""Helmholtz--Hodge decomposition of an edge 1-form.

    Splits a tangential field ``omega`` (a 1-form on the mesh edges) into

    .. math::

        \omega = \underbrace{d_0 \alpha}_{\text{exact (curl-free)}}
        + \underbrace{\star_1^{-1} d_1^\top u}_{\text{coexact (div-free)}}
        + \underbrace{h}_{\text{harmonic}},

    by two matrix-free Poisson solves (:func:`~nitrix.linalg.cg`): the exact-part
    potential solves :math:`d_0^\top \star_1 d_0\, \alpha = d_0^\top \star_1
    \omega` (the cotangent Laplacian) and the coexact-part potential solves
    :math:`d_1 \star_1^{-1} d_1^\top u = d_1 \omega`. Both right-hand sides are
    orthogonal to the constant null-space, so the solves are consistent; a small
    ``ridge`` stabilises them. The three parts are mutually orthogonal in the
    :math:`\star_1` inner product; on a genus-0 surface the harmonic part is zero.

    This is the eager single-call convenience, defined as
    ``hodge_apply(omega, hodge_operator(mesh))``.  The operators are built
    host-side, so ``hodge_decompose`` as a whole is **not** jittable through
    ``mesh``.  To place the solves inside a jitted region -- or to decompose many
    1-forms on one mesh -- build the operator once with :func:`hodge_operator`
    (host) and call the jittable :func:`hodge_apply`.

    Parameters
    ----------
    omega : Float[Array, ' n_edges']
        The input 1-form on the edges (in the :func:`mesh_gradient` edge order).
    mesh
        Triangle mesh.
    tol : float, optional
        Conjugate-gradient tolerance. Default ``1e-9``.
    ridge : float, optional
        Small Tikhonov ridge stabilising the singular (constant-null-space)
        Poisson systems. Default ``1e-8``.

    Returns
    -------
    HodgeDecomposition
        The ``(exact, coexact, harmonic)`` parts, each a ``(n_edges,)`` 1-form.
    """
    return hodge_apply(omega, hodge_operator(mesh), tol=tol, ridge=ridge)
