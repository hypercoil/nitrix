# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Surface algorithms on triangle meshes.

The *algorithm* layer of the geometry suite: differential-geometry quantities
(curvature, ...) that compose the *operator / measure* layer in
``nitrix.sparse.mesh`` -- the cotangent Laplace-Beltrami stiffness, the
vertex-area mass, and per-vertex normals -- through the format-agnostic
``sparse.apply_operator`` seam.  ``geometry`` depends on ``sparse``, never the
reverse.

Everything here is pure JAX and differentiable w.r.t. ``mesh.vertices``.
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from ..sparse import Mesh, compute_vertex_normals, vertex_areas

__all__ = [
    'mean_curvature',
    'gaussian_curvature',
    'principal_curvatures',
    'areal_distortion',
    'strain_distortion',
]


def _cotangent_apply(
    vertices: Float[Array, 'n_vertices 3'],
    faces: Int[Array, 'n_faces 3'],
    field: Float[Array, 'n_vertices d'],
) -> Float[Array, 'n_vertices d']:
    """Apply the cotangent stiffness ``L`` to ``field``, JAX-native.

    Computes ``(L f)[i] = sum_j w_ij (f_i - f_j)`` with
    ``w_ij = (cot a_ij + cot b_ij) / 2`` by per-face assembly entirely in JAX,
    so it is **differentiable w.r.t. ``vertices``** (the cotangent weights are
    a function of geometry).  This is the differentiate-through-geometry
    companion to the host-side ``sparse.mesh.mesh_cotangent_laplacian`` (which
    bakes fixed weights into an ELL -- correct for applying a *fixed* operator
    to many fields, e.g. smoothing, but not differentiable w.r.t. vertices).
    The two agree numerically.
    """
    f = faces
    a, b, c = vertices[f[:, 0]], vertices[f[:, 1]], vertices[f[:, 2]]
    fa, fb, fc = field[f[:, 0]], field[f[:, 1]], field[f[:, 2]]

    def _half_cot(
        u: Float[Array, 'n_faces 3'], w: Float[Array, 'n_faces 3']
    ) -> Float[Array, 'n_faces 1']:
        cross = jnp.sqrt(jnp.sum(jnp.cross(u, w) ** 2, axis=-1))
        return (0.5 * jnp.sum(u * w, axis=-1) / jnp.maximum(cross, 1e-12))[
            :, None
        ]

    cot_a = _half_cot(b - a, c - a)  # angle at a -> edge (b, c)
    cot_b = _half_cot(c - b, a - b)  # angle at b -> edge (c, a)
    cot_c = _half_cot(a - c, b - c)  # angle at c -> edge (a, b)

    out = jnp.zeros_like(field)
    out = out.at[f[:, 1]].add(cot_a * (fb - fc))
    out = out.at[f[:, 2]].add(cot_a * (fc - fb))
    out = out.at[f[:, 2]].add(cot_b * (fc - fa))
    out = out.at[f[:, 0]].add(cot_b * (fa - fc))
    out = out.at[f[:, 0]].add(cot_c * (fa - fb))
    out = out.at[f[:, 1]].add(cot_c * (fb - fa))
    return out


def mean_curvature(
    mesh: Mesh,
    *,
    area_scheme: str = 'voronoi',
) -> Float[Array, 'n_vertices']:
    """Per-vertex mean curvature ``H`` via the cotangent / mass operator.

    The discrete mean-curvature *normal* (Meyer et al. 2003) is
    ``K(v_i) = (1 / 2 A_i) sum_j (cot a_ij + cot b_ij)(v_i - v_j)``, which is
    exactly ``(M^{-1} L v)[i]`` for the shipped cotangent stiffness ``L``
    (``mesh_cotangent_laplacian``) and lumped mass ``M`` (``vertex_areas``),
    and equals ``2 H n``.  So the mean-curvature vector is
    ``H_vec = 1/2 M^{-1} L v`` and ``H = sign(H_vec . n) ||H_vec||``.

    Sign convention: **positive where the surface is convex w.r.t. the outward
    vertex normal** (a sphere, gyral crowns), negative in concave regions
    (sulcal fundi).  This is the *opposite* sign to FreeSurfer ``?h.curv``
    (which is positive in sulci); flip the sign to compare.

    Parameters
    ----------
    mesh
        Triangle mesh.
    area_scheme
        Vertex-area scheme for the mass ``M`` (``'voronoi'`` default;
        ``'barycentric'``).

    Returns
    -------
    ``(n_vertices,)`` mean curvature.  Pure JAX; differentiable w.r.t.
    ``mesh.vertices``.
    """
    lv = _cotangent_apply(mesh.vertices, mesh.faces, mesh.vertices)  # L v
    area = vertex_areas(mesh, scheme=area_scheme)
    h_vec = 0.5 * lv / jnp.maximum(area, 1e-12)[:, None]
    normals = compute_vertex_normals(mesh.vertices, mesh.faces)
    magnitude = jnp.sqrt(jnp.sum(h_vec**2, axis=-1))
    sign = jnp.sign(jnp.sum(h_vec * normals, axis=-1))
    return sign * magnitude


def gaussian_curvature(
    mesh: Mesh,
    *,
    area_scheme: str = 'voronoi',
) -> Float[Array, 'n_vertices']:
    """Per-vertex Gaussian curvature ``K`` via the angle-defect formula.

    ``K(v_i) = (2 pi - sum_j theta_ij) / A_i`` where ``theta_ij`` are the
    interior triangle angles incident at ``v_i`` and ``A_i`` the vertex area.
    The numerator (the angle defect) is the *integrated* Gaussian curvature,
    so ``sum_i K_i A_i = 2 pi chi`` exactly (discrete Gauss-Bonnet) -- ``4 pi``
    for a genus-0 surface.

    Assumes a **closed** mesh (the ``2 pi`` term); for a surface with boundary
    the boundary vertices would need the ``pi`` defect instead.  The geometry
    suite's targets (icosphere, genus-0 cortical surfaces) are closed.

    Parameters
    ----------
    mesh
        Triangle mesh (closed).
    area_scheme
        Vertex-area scheme for ``A_i`` (``'voronoi'`` default).

    Returns
    -------
    ``(n_vertices,)`` Gaussian curvature.  Pure JAX; differentiable.
    """
    v = mesh.vertices
    f = mesh.faces
    n = mesh.n_vertices
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]

    def _angle(
        u: Float[Array, 'n_faces 3'], w: Float[Array, 'n_faces 3']
    ) -> Float[Array, 'n_faces']:
        un = jnp.sqrt(jnp.sum(u**2, axis=-1))
        wn = jnp.sqrt(jnp.sum(w**2, axis=-1))
        cosang = jnp.sum(u * w, axis=-1) / jnp.maximum(un * wn, 1e-12)
        return jnp.arccos(jnp.clip(cosang, -1.0, 1.0))

    ang_a = _angle(b - a, c - a)
    ang_b = _angle(a - b, c - b)
    ang_c = _angle(a - c, b - c)
    angle_sum = jnp.zeros((n,), dtype=v.dtype)
    angle_sum = angle_sum.at[f[:, 0]].add(ang_a)
    angle_sum = angle_sum.at[f[:, 1]].add(ang_b)
    angle_sum = angle_sum.at[f[:, 2]].add(ang_c)

    area = vertex_areas(mesh, scheme=area_scheme)
    return (2.0 * jnp.pi - angle_sum) / jnp.maximum(area, 1e-12)


def principal_curvatures(
    mesh: Mesh,
    *,
    area_scheme: str = 'voronoi',
) -> Float[Array, 'n_vertices 2']:
    """Per-vertex principal curvatures ``(kappa_1, kappa_2)``, ``kappa_1 >= kappa_2``.

    From the mean and Gaussian curvatures via
    ``kappa = H +/- sqrt(max(H^2 - K, 0))`` (the discriminant is clamped at
    zero, where it can dip slightly negative from discretisation).  Inherits
    ``mean_curvature``'s sign convention.

    Parameters
    ----------
    mesh
        Triangle mesh (closed).
    area_scheme
        Vertex-area scheme (``'voronoi'`` default).

    Returns
    -------
    ``(n_vertices, 2)`` -- column 0 is ``kappa_1`` (larger), column 1 is
    ``kappa_2``.  Pure JAX; differentiable.
    """
    h = mean_curvature(mesh, area_scheme=area_scheme)
    k = gaussian_curvature(mesh, area_scheme=area_scheme)
    disc = jnp.sqrt(jnp.maximum(h**2 - k, 0.0))
    return jnp.stack([h + disc, h - disc], axis=-1)


def areal_distortion(
    source: Mesh,
    warped: Mesh,
    *,
    area_scheme: str = 'voronoi',
) -> Float[Array, 'n_vertices']:
    """Per-vertex areal distortion ``log2(A_warped / A_source)``.

    The surface analogue of ``jacobian_det_displacement`` and the MSM areal
    regulariser / ``?h.jacobian`` QA readout: positive where the warp expands
    area, negative where it contracts, zero for an isometry.  ``source`` and
    ``warped`` **must share topology** (identical ``faces`` and corresponding
    vertex indexing -- e.g. ``warped`` is ``source`` after a spherical /
    inflation warp).

    Parameters
    ----------
    source, warped
        Triangle meshes with the same topology, before / after the warp.
    area_scheme
        Vertex-area scheme (``'voronoi'`` default).

    Returns
    -------
    ``(n_vertices,)`` log2 area ratio.  Pure JAX; differentiable w.r.t. both
    meshes' vertices.

    Raises
    ------
    ValueError
        If the two meshes' ``faces`` shapes differ (a topology mismatch).
    """
    if source.faces.shape != warped.faces.shape:
        raise ValueError(
            'areal_distortion: source and warped must share topology; '
            f'faces shapes {source.faces.shape} != {warped.faces.shape}.'
        )
    a_src = vertex_areas(source, scheme=area_scheme)
    a_warp = vertex_areas(warped, scheme=area_scheme)
    return jnp.log2(jnp.maximum(a_warp, 1e-12) / jnp.maximum(a_src, 1e-12))


def strain_distortion(
    source: Mesh,
    warped: Mesh,
) -> Float[Array, 'n_faces 2']:
    """Per-face principal stretches ``(lambda_1, lambda_2)``, ``lambda_1 >= lambda_2``.

    The eigenvalues of the right Cauchy-Green tensor ``C = G_s^{-1} G_w`` are
    the squared principal stretches of the per-triangle deformation from
    ``source`` to ``warped``, where ``G_s`` / ``G_w`` are the 2x2 first
    fundamental forms (edge Gram matrices) of the source / warped triangle.
    Computed via the 2x2 eigenvalue closed form (no ``eigh``, so it bypasses
    the cuSolver wedge): ``lambda^2 = (tr C +/- sqrt(tr C^2 - 4 det C)) / 2``.

    An isometry gives ``(1, 1)``; a uniform scale ``s`` gives ``(s, s)``; an
    anisotropic scale ``diag(a, b)`` gives ``(max(a,b), min(a,b))``.
    ``source`` and ``warped`` **must share topology**.

    Parameters
    ----------
    source, warped
        Triangle meshes with the same topology, before / after the warp.

    Returns
    -------
    ``(n_faces, 2)`` -- column 0 is ``lambda_1`` (larger stretch).  Pure JAX;
    differentiable.

    Raises
    ------
    ValueError
        If the two meshes' ``faces`` shapes differ (a topology mismatch).
    """
    if source.faces.shape != warped.faces.shape:
        raise ValueError(
            'strain_distortion: source and warped must share topology; '
            f'faces shapes {source.faces.shape} != {warped.faces.shape}.'
        )
    f = source.faces
    vs, vw = source.vertices, warped.vertices
    s1, s2 = vs[f[:, 1]] - vs[f[:, 0]], vs[f[:, 2]] - vs[f[:, 0]]
    t1, t2 = vw[f[:, 1]] - vw[f[:, 0]], vw[f[:, 2]] - vw[f[:, 0]]
    g11 = jnp.sum(s1 * s1, axis=-1)
    g12 = jnp.sum(s1 * s2, axis=-1)
    g22 = jnp.sum(s2 * s2, axis=-1)
    w11 = jnp.sum(t1 * t1, axis=-1)
    w12 = jnp.sum(t1 * t2, axis=-1)
    w22 = jnp.sum(t2 * t2, axis=-1)
    det_s = jnp.maximum(g11 * g22 - g12**2, 1e-12)
    # tr(G_s^{-1} G_w) and det(G_s^{-1} G_w), C symmetrisable -> real eigenvalues.
    tr_c = (g22 * w11 - 2.0 * g12 * w12 + g11 * w22) / det_s
    det_c = (w11 * w22 - w12**2) / det_s
    # Floor the discriminant strictly above zero: it keeps the gradient finite
    # at the eigenvalue crossing lambda_1 == lambda_2 (where ``sqrt`` is
    # otherwise non-differentiable), at the cost of a ~1e-6 split of genuinely
    # equal stretches -- far below the float32 cancellation noise (~5e-4) that
    # near-degenerate stretches already carry.  Distinct stretches are
    # well-conditioned and unaffected.
    disc = jnp.sqrt(jnp.maximum(tr_c**2 - 4.0 * det_c, 1e-12))
    lam1 = jnp.sqrt(jnp.maximum((tr_c + disc) / 2.0, 0.0))
    lam2 = jnp.sqrt(jnp.maximum((tr_c - disc) / 2.0, 0.0))
    return jnp.stack([lam1, lam2], axis=-1)
