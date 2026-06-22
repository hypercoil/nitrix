# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
2-sphere primitives.

Coordinate conventions
----------------------

We use *Cartesian* normal-vector format ``(x, y, z)`` as the
canonical representation: every coordinate on the sphere is a
unit 3-vector (or an ``r``-scaled 3-vector for a radius-``r``
sphere).  This is the format consumed by ``spherical_geodesic_distance``,
``spherical_conv``, and the icosphere generation utilities.

For lat/long inputs (e.g. data digitised from a globe), convert
once via ``latlong_to_cartesian``.  All downstream operations assume
the Cartesian representation.

``spherical_conv`` re-backs the legacy O(N²) all-pairs convolution
on ``semiring_ell_matmul`` over a k-NN adjacency for O(N · k) cost
-- the marquee Phase 3 task per SPEC §6.1 ``3.2`` ("validates the
§3.1 design bet end-to-end").
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union

import jax.lax as lax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int

from ..semiring import REAL, semiring_ell_matmul

if TYPE_CHECKING:
    from ..sparse import Mesh

__all__ = [
    'latlong_to_cartesian',
    'cartesian_to_latlong',
    'spherical_geodesic_distance',
    'spherical_conv',
    'signed_spherical_areas',
    'is_bijective_sphere_map',
    'spectral_sphere_embedding',
]


# ---------------------------------------------------------------------------
# Coordinate conversions
# ---------------------------------------------------------------------------


def latlong_to_cartesian(
    latlong: Float[Array, '... 2'],
    r: float = 1.0,
) -> Float[Array, '... 3']:
    """Latitude/longitude (radians) -> Cartesian ``(x, y, z)`` on a sphere.

    Latitude is the angle from the equator (``-pi/2`` at south pole,
    ``+pi/2`` at north pole); longitude is the angle around the
    equator from the prime meridian.

    Parameters
    ----------
    latlong
        Stacked ``(latitude, longitude)`` in radians, ``(..., 2)``.
    r
        Radius of the sphere.

    Returns
    -------
    Cartesian coordinates, ``(..., 3)``.
    """
    lat = latlong[..., 0]
    lon = latlong[..., 1]
    cos_lat = jnp.cos(lat)
    return jnp.stack(
        (
            r * cos_lat * jnp.cos(lon),
            r * cos_lat * jnp.sin(lon),
            r * jnp.sin(lat),
        ),
        axis=-1,
    )


def cartesian_to_latlong(
    xyz: Float[Array, '... 3'],
) -> Float[Array, '... 2']:
    """Cartesian ``(x, y, z)`` -> ``(latitude, longitude)`` in radians.

    Inverse of ``latlong_to_cartesian``.  ``xyz`` need not be a unit
    vector; the returned angles are scale-invariant.

    Parameters
    ----------
    xyz
        Cartesian coordinates, ``(..., 3)``.

    Returns
    -------
    ``(latitude, longitude)``, ``(..., 2)``.
    """
    x = xyz[..., 0]
    y = xyz[..., 1]
    z = xyz[..., 2]
    lat = jnp.arctan2(z, jnp.sqrt(x**2 + y**2))
    lon = jnp.arctan2(y, x)
    return jnp.stack((lat, lon), axis=-1)


# ---------------------------------------------------------------------------
# Geodesic distance
# ---------------------------------------------------------------------------


def _geodesic_pair(
    X: Float[Array, '... 3'],
    Y: Float[Array, '... 3'],
    r: float,
) -> Float[Array, '...']:
    """Per-pair geodesic on already-aligned points.

    ``X`` and ``Y`` must have matching shapes ``(..., 3)``; output is
    ``(...,)``.  Used by both the all-pairs entry point and the
    inline distance computation inside ``spherical_conv``.
    """
    cross = jnp.cross(X, Y, axis=-1)
    num = jnp.sqrt((cross**2).sum(-1))
    denom = (X * Y).sum(-1)
    angle = jnp.arctan2(num, denom)
    # ``arctan2`` returns in ``(-pi, pi]``; for antipodal points we
    # want the non-negative angle, so wrap negative values into
    # ``[0, pi]``.  Geodesic distance is then ``r * angle``.
    angle = jnp.where(angle < 0, angle + jnp.pi, angle)
    return r * angle


def spherical_geodesic_distance(
    X: Float[Array, '... n 3'],
    Y: Optional[Float[Array, '... m 3']] = None,
    r: float = 1.0,
) -> Float[Array, '... n m']:
    """All-pairs great-circle distance between Cartesian points on a sphere.

    Parameters
    ----------
    X
        First set of points on the sphere, Cartesian, ``(..., n, 3)``.
    Y
        Second set; if ``None``, computes pairwise distance within
        ``X``.  Shape ``(..., m, 3)``.
    r
        Sphere radius.  The returned distance is ``r * angle``.

    Returns
    -------
    Distance matrix ``(..., n, m)`` in the same units as ``r``.
    """
    if Y is None:
        Y = X
    if X.shape[-1] != 3 or Y.shape[-1] != 3:
        raise ValueError(
            'spherical_geodesic_distance: both inputs must have '
            f'shape (..., 3); got X.shape={X.shape}, Y.shape={Y.shape}.'
        )
    X_b = X[..., :, None, :]
    Y_b = Y[..., None, :, :]
    X_b, Y_b = jnp.broadcast_arrays(X_b, Y_b)
    return _geodesic_pair(X_b, Y_b, r)


# ---------------------------------------------------------------------------
# Spherical convolution (re-backed on the semiring substrate)
# ---------------------------------------------------------------------------


def _spherical_knn_indices(
    coor: Float[Array, 'n 3'],
    k: int,
    r: float,
) -> Int[Array, 'n k']:
    """Top-k nearest neighbours by spherical geodesic distance.

    Materialises the ``(n, n)`` distance matrix; quadratic memory.
    Practical for ``n <= 10k``.  Larger meshes should pre-compute the
    adjacency (via a hierarchical tree or by the icosphere's natural
    k-ring) and pass it as ``neighbourhood=indices``.
    """
    d = spherical_geodesic_distance(coor, coor, r=r)
    _, indices = lax.top_k(-d, k)
    return indices


def spherical_conv(
    data: Float[Array, '... n c'],
    coor: Float[Array, 'n 3'],
    *,
    sigma: float,
    neighbourhood: Union[int, Int[Array, 'n k']],
    r: float = 1.0,
    truncate: Optional[float] = None,
) -> Float[Array, '... n c']:
    """Convolve data on a 2-sphere with an isotropic Gaussian kernel.

    Specialises onto ``semiring_ell_matmul``: build a per-point k-NN
    adjacency by spherical geodesic distance, weight neighbours by a
    Gaussian over the geodesic distance, normalise, reduce.  Cost is
    ``O(n * k * c)`` per call once the adjacency is in hand --
    *not* the legacy ``O(n^2 * c)`` of the all-pairs implementation.

    Parameters
    ----------
    data
        Per-point feature vectors, ``(..., n, c)``.  Leading batch
        dims are passed through.
    coor
        Per-point Cartesian coordinates on the sphere, ``(n, 3)``.
        Currently single-batch (no leading dims); ``vmap`` for
        per-subject coordinate sets.
    sigma
        Standard deviation of the Gaussian kernel, in the same units
        as ``r`` (radians × ``r``).
    neighbourhood
        Either an ``int`` ``k`` (compute k-NN by spherical geodesic
        on the fly; O(n²) memory) or an explicit ``(n, k)`` index
        array.  For large ``n`` (>~10k) provide the adjacency
        explicitly, e.g. from the icosphere's natural k-ring.
    r
        Sphere radius.
    truncate
        Optional hard distance cutoff: neighbours beyond
        ``truncate`` get weight zero.  ``None`` (default) means
        Gaussian weighting only.  Useful when the k-NN includes
        far-away points that should not contribute.

    Returns
    -------
    Smoothed data, ``(..., n, c)``.

    Notes
    -----
    The weights are normalised so each row sums to 1.  Constant data
    is therefore preserved.

    For ``data`` with multiple channels, every channel is smoothed
    by the same spatial kernel (the standard "depthwise" semantics).
    Per-channel sigma is not currently supported via this function;
    call multiple times if needed.
    """
    if data.shape[-2] != coor.shape[0]:
        raise ValueError(
            f'spherical_conv: data.shape[-2]={data.shape[-2]} must '
            f'equal coor.shape[0]={coor.shape[0]}.'
        )
    n = coor.shape[0]

    # Resolve adjacency.
    if isinstance(neighbourhood, int):
        indices = _spherical_knn_indices(coor, neighbourhood, r=r)
    else:
        indices = jnp.asarray(neighbourhood, dtype=jnp.int32)
        if indices.shape[0] != n:
            raise ValueError(
                f'neighbourhood.shape[0]={indices.shape[0]} must equal n={n}.'
            )

    # Geodesic distance from each point to each of its neighbours.
    X = coor[:, None, :]
    Y = coor[indices]
    dist = _geodesic_pair(X, Y, r)  # (n, k)
    weights = jnp.exp(-0.5 * (dist / sigma) ** 2)  # (n, k)
    if truncate is not None:
        weights = jnp.where(dist > truncate, 0.0, weights)
    Z = weights.sum(axis=-1, keepdims=True)
    weights = weights / jnp.maximum(Z, jnp.finfo(weights.dtype).tiny)

    # Reduce via semiring_ell_matmul: out[..., i, c] = sum_p
    # weights[i, p] * data[..., indices[i, p], c].  ``data`` carries
    # leading batch dims; ``weights`` and ``indices`` do not, so we
    # broadcast at the call site by tiling weights / indices.
    batch_dims = data.shape[:-2]
    if batch_dims:
        # Tile the per-row weights/indices to match leading batch dims.
        weights_b = jnp.broadcast_to(
            weights,
            batch_dims + weights.shape,
        )
        indices_b = jnp.broadcast_to(
            indices,
            batch_dims + indices.shape,
        )
        return semiring_ell_matmul(
            weights_b,
            indices_b,
            data,
            semiring=REAL,
            n_cols=n,
            backend='jax',
        )
    return semiring_ell_matmul(
        weights,
        indices,
        data,
        semiring=REAL,
        n_cols=n,
        backend='jax',
    )


# ---------------------------------------------------------------------------
# Spherical-map bijectivity (GS-2a)
# ---------------------------------------------------------------------------


def signed_spherical_areas(
    vertices: Float[Array, 'n_vertices 3'],
    faces: Int[Array, 'n_faces 3'],
) -> Float[Array, 'n_faces']:
    """Signed solid angle (steradians) per triangle, subtended at the origin.

    For triangle vertices on a sphere centred at the origin this is the signed
    spherical-triangle area (the unit-sphere area; **radius-independent** -- the
    solid angle does not depend on the sphere radius).  The **sign** is the
    triangle's orientation (the triple product ``a . (b x c)``): a flipped /
    folded triangle is negative.  For a bijective degree-1 cover the values sum
    to ``+/- 4 pi`` and every triangle shares the dominant sign.  Computed by the
    Van Oosterom-Strackee formula; pure JAX, differentiable.

    Parameters
    ----------
    vertices
        ``(n_vertices, 3)`` positions (assumed on a sphere about the origin).
    faces
        ``(n_faces, 3)`` triangle vertex indices.

    Returns
    -------
    ``(n_faces,)`` signed solid angles.
    """
    a = vertices[faces[:, 0]]
    b = vertices[faces[:, 1]]
    c = vertices[faces[:, 2]]
    la = jnp.sqrt(jnp.sum(a * a, axis=-1))
    lb = jnp.sqrt(jnp.sum(b * b, axis=-1))
    lc = jnp.sqrt(jnp.sum(c * c, axis=-1))
    num = jnp.sum(a * jnp.cross(b, c), axis=-1)
    den = (
        la * lb * lc
        + jnp.sum(a * b, axis=-1) * lc
        + jnp.sum(b * c, axis=-1) * la
        + jnp.sum(c * a, axis=-1) * lb
    )
    return 2.0 * jnp.arctan2(num, den)


def is_bijective_sphere_map(
    vertices: Float[Array, 'n_vertices 3'],
    faces: Int[Array, 'n_faces 3'],
    *,
    flip_area_tol: float = 0.0,
    total_tol: float = 1e-2,
) -> bool:
    """Whether (vertices-on-sphere, faces) is a (near-)bijective spherical map.

    ``True`` iff the signed solid angles (``signed_spherical_areas``) form a
    **degree-1 cover** -- their sum is within ``total_tol`` of ``+/- 4 pi`` --
    **and** the fraction of solid angle in triangles flipped against the
    dominant orientation is ``<= flip_area_tol``.  ``flip_area_tol = 0`` is the
    strict fold-free test; a small tolerance (e.g. ``8e-4``) mirrors the
    FastSurfer/recon-surf quality gate, which tolerates a tiny flipped fraction.

    Host-side boolean (a decision, not a traced value) -- used to gate the
    spherical-parameterisation init fallback chain.

    Parameters
    ----------
    vertices, faces
        The candidate spherical map.
    flip_area_tol
        Maximum tolerated flipped solid-angle fraction (``0`` = strict).
    total_tol
        Relative tolerance on ``|sum| == 4 pi`` (the cover test).

    Returns
    -------
    ``bool``.
    """
    areas = np.asarray(signed_spherical_areas(vertices, faces))
    total = float(areas.sum())
    abs_total = float(np.abs(areas).sum())
    if abs_total <= 0.0:
        return False
    four_pi = 4.0 * np.pi
    cover_ok = abs(abs(total) - four_pi) <= total_tol * four_pi
    dominant = 1.0 if total >= 0.0 else -1.0
    flipped = float(np.abs(areas[np.sign(areas) != dominant]).sum())
    flip_ok = (flipped / abs_total) <= flip_area_tol
    return bool(cover_ok and flip_ok)


# ---------------------------------------------------------------------------
# Spectral spherical embedding (GS-2b -- the recon-surf one-shot)
# ---------------------------------------------------------------------------


def spectral_sphere_embedding(
    mesh: 'Mesh',
    *,
    radius: float = 1.0,
    n_eig: int = 10,
    eig_iters: int = 50,
) -> Float[Array, 'n_vertices 3']:
    """One-shot spherical map from the Laplace-Beltrami eigenfunctions.

    The FastSurfer / recon-surf method (verified against
    ``Deep-MI/FastSurfer recon_surf/spherically_project.py``): solve the
    **generalised** LBO eigenproblem ``L phi = lambda M phi`` -- cotangent
    stiffness ``L`` (``mesh_cotangent_laplacian``), lumped mass ``M``
    (``vertex_areas``) -- for the **first three non-constant eigenfunctions**
    (smallest eigenvalues), and scale each vertex's ``(phi_1, phi_2, phi_3)`` to
    ``radius``.  On the round sphere these eigenfunctions are the degree-1
    harmonics ``x, y, z``, so a near-spherical input maps to the sphere in one
    eigensolve, no iteration.

    The diagonal mass makes ``L tilde = M^{-1/2} L M^{-1/2}`` symmetric.  We form
    the affinity ``A = I - L tilde / c`` (``c`` a Gershgorin bound on the
    spectrum), whose *largest* eigenpairs are ``L tilde``'s *smallest*, and find
    them with **shift-invert** LOBPCG (``eigsolve_top_k`` + a tight negative
    ``sigma``) -- the cuSolver-free path.  Shift-invert (not a plain reflection)
    is essential: the smallest LBO eigenvalues are tightly clustered near 0, and
    shift-invert is what amplifies their gaps enough to resolve the constant +
    first-three eigenfunctions; ``n_eig`` over-samples to aid convergence.

    **Not guaranteed bijective** -- it is the recon-surf-grade one-shot map
    (a tiny flipped fraction may remain).  Gate with ``is_bijective_sphere_map``
    and fall back to a guaranteed-bijective init, or refine with the iterative
    optimiser, for a strictly fold-free result.  The eigenfunction
    **sign/ordering gauge** is *not* resolved here (nitrix is
    orientation-agnostic; recon-surf's anatomical sign/swap alignment is a
    consumer step).

    Host-side cotangent construction + JAX eigensolve; not differentiable w.r.t.
    ``mesh.vertices`` (the cotangent weights are built host-side) -- this is an
    init / preprocessing artefact.

    Parameters
    ----------
    mesh
        Genus-0 triangle mesh (ideally inflated / near-spherical).
    radius
        Output sphere radius.
    n_eig
        Number of eigenpairs to request (>= 4); over-sampling beyond the
        constant + 3 used aids LOBPCG convergence on the clustered low spectrum.
    eig_iters
        LOBPCG outer / inner-CG iteration budget for the shift-invert solve.

    Returns
    -------
    ``(n_vertices, 3)`` vertices on the sphere of the given radius.
    """
    from ..linalg._eigsolve import SolverSpec, eigsolve_top_k
    from ..sparse import ELL, mesh_cotangent_laplacian, vertex_areas

    lap = mesh_cotangent_laplacian(mesh)
    inv_sqrt = 1.0 / jnp.sqrt(jnp.maximum(vertex_areas(mesh), 1e-12))
    # L_tilde = D^{-1/2} L D^{-1/2} (symmetric): scale each entry by the
    # inverse-sqrt mass of its row and column vertex.
    lt_vals = lap.values * inv_sqrt[:, None] * inv_sqrt[lap.indices]
    # Affinity A = I - L_tilde / c (spectrum in [0, 1], A's largest = L_tilde's
    # smallest); the diagonal is column 0 in the cotangent ELL layout.
    c = jnp.max(
        jnp.sum(jnp.abs(lt_vals), axis=1)
    )  # Gershgorin lambda_max bound
    aff_vals = (-lt_vals / c).at[:, 0].add(1.0)
    affinity = ELL(
        values=aff_vals, indices=lap.indices, n_cols=lap.n_cols, identity=0.0
    )
    pair = eigsolve_top_k(
        affinity,
        n_eig,
        spec=SolverSpec.shift_invert(
            sigma=-0.01, outer_iters=eig_iters, cg_iters=eig_iters
        ),
    )
    lam = c * (1.0 - pair.values)  # recovered L_tilde eigenvalues
    order = jnp.argsort(lam)
    psi = pair.vectors[:, order]  # ascending by eigenvalue
    phi = (
        inv_sqrt[:, None] * psi
    )  # generalised eigenvectors (LBO eigenfunctions)
    emb = phi[:, 1:4]  # skip the constant eigenfunction, take the next three
    norm = jnp.maximum(
        jnp.sqrt(jnp.sum(emb * emb, axis=-1, keepdims=True)), 1e-12
    )
    return radius * emb / norm
