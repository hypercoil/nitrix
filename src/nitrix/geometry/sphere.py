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

from typing import TYPE_CHECKING, Any, Literal, Optional, Union, cast

import jax
import jax.lax as lax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int
from numpy.typing import NDArray

from ..semiring import REAL, semiring_ell_matmul

if TYPE_CHECKING:
    from ..sparse import ELL, Mesh

__all__ = [
    'latlong_to_cartesian',
    'cartesian_to_latlong',
    'spherical_geodesic_distance',
    'spherical_conv',
    'signed_spherical_areas',
    'is_bijective_sphere_map',
    'spectral_sphere_embedding',
    'spherical_parameterize',
    'surface_resample',
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
    # ``num >= 0`` so ``arctan2(num, denom)`` is already the unsigned great-circle
    # angle in ``[0, pi]`` (it handles the antipodal case via the sign of
    # ``denom``); geodesic distance is then ``r * angle``.
    angle = jnp.arctan2(num, denom)
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

    Tiled row-by-row with ``lax.map`` (Tier C / audit AI-C8): peak memory is
    ``O(n)`` (one query's distances at a time), not the ``O(n^2)`` full distance
    matrix -- so the int-``k`` path degrades gracefully instead of OOMing on a
    large mesh.  Still ``O(n^2)`` *compute*; for very large meshes pre-compute
    the adjacency (e.g. the icosphere's natural k-ring) and pass it as
    ``neighbourhood=indices``.  Results are identical to the dense path.
    """

    def _knn_row(x: Float[Array, '3']) -> Int[Array, 'k']:
        d_row = _geodesic_pair(x[None, :], coor, r)  # (n,)
        _, idx = lax.top_k(-d_row, k)
        return idx

    return cast(Int[Array, 'n k'], lax.map(_knn_row, coor))


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
    # NB: this is the Van Oosterom-Strackee solid angle with the apex at the
    # origin; ``isosurface._solid_angle`` is the same formula for an arbitrary
    # apex point (the numpy/host copy used by the winding-number SDF sign). Keep
    # the two in sync (a shared apex-parameterised core is a tracked follow-up).
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
    laplacian: Optional['ELL'] = None,
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
    init / preprocessing artefact.  Pass ``laplacian=`` to reuse an
    already-assembled cotangent operator (``spherical_parameterize`` does this so
    the operator is built once per call, not twice).

    Note on a wedged dense-solver stack: the shift-invert LOBPCG core is
    matrix-free / cuSolver-free, but its orthonormalisation shares the dense
    solver-handle pool, so on a box whose pool is dead the eigensolve (and hence
    this embedding) executes on the solver-device fallback (CPU), incurring a
    host round-trip -- correct, just not GPU-resident there.

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

    if laplacian is None:
        laplacian = mesh_cotangent_laplacian(mesh)  # default 'ell' -> flat ELL
    lap = laplacian
    assert isinstance(lap, ELL)  # narrow the ELL | SectionedELL return type
    # The affinity I - L_tilde/c adds the identity to the diagonal entry, which
    # the flat cotangent ELL stores at column 0; enforce that layout rather than
    # assume it (see mesh_cotangent_laplacian's diagonal-first contract).
    n = mesh.n_vertices
    assert bool(jnp.all(lap.indices[:, 0] == jnp.arange(n))), (
        'spectral_sphere_embedding expects the cotangent ELL diagonal at '
        'column 0 (the mesh_cotangent_laplacian diagonal-first layout).'
    )
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


# ---------------------------------------------------------------------------
# Spherical parameterisation (GS-2c / GS-2d)
# ---------------------------------------------------------------------------


def spherical_parameterize(
    mesh: 'Mesh',
    *,
    init: Literal['spectral', 'radial'] = 'spectral',
    n_iterations: int = 200,
    conformal_weight: float = 1.0,
    area_weight: float = 1.0,
    step: float = 0.05,
    radius: float = 1.0,
) -> Float[Array, 'n_vertices 3']:
    """Map an inflated genus-0 surface to the sphere, minimising distortion.

    The `mris_sphere` analogue.  Starts from a fast (near-)bijective init and,
    if ``n_iterations > 0``, refines it by **Riemannian gradient descent on
    (S^2)^n** under a conformal + areal distortion energy, with a **fold-safe
    backtracking line-search** (a step is accepted only if it lowers the energy
    and does not increase the number of flipped triangles) and a
    centroid/Mobius normalisation each step (which, with the area term, blocks
    the conformal collapse-to-a-point).  Bijectivity is thus driven to strict
    (no flipped triangles) and held.

    ``n_iterations = 0`` returns the init alone -- with ``init='spectral'`` that
    is the recon-surf-grade one-shot map (``spectral_sphere_embedding``).

    Execution class: **host-orchestrated**, not a jittable kernel.  It builds the
    cotangent operator host-side (``mesh_cotangent_laplacian``, NumPy) once and,
    for ``init='spectral'``, runs the host eigensolve; the per-iteration refine
    *body* (energy, tangent-projected gradient, fold-safe line-search) is pure
    JAX and runs on device, but the function as a whole is **not** jittable nor
    differentiable w.r.t. ``mesh.vertices`` (the operator weights are NumPy
    constants).  Call it to *produce* a spherical map; do not wrap it in
    ``jax.jit`` / ``jax.grad``.

    Parameters
    ----------
    mesh
        Inflated genus-0 mesh (e.g. ``inflate_surface`` output).
    init
        ``'spectral'`` (default, the recon-surf LBO embedding) or ``'radial'``
        (centroid projection -- only for near-convex inputs).
    n_iterations
        Refinement iterations (``0`` -> init only).
    conformal_weight, area_weight
        Energy weights (Dirichlet/conformal core + areal log-ratio term).
    step
        Initial line-search step.
    radius
        Output sphere radius.

    Returns
    -------
    ``(n_vertices, 3)`` vertices on the sphere of the given radius (same
    ``faces`` -- correspondence preserved).
    """
    from ..sparse import (
        ELL,
        apply_operator,
        face_areas,
        mesh_cotangent_laplacian,
    )

    faces = mesh.faces
    # Build the cotangent operator once and reuse it for both the spectral init
    # and the refinement energy (the host-side assembly is the costly step).
    lap = mesh_cotangent_laplacian(mesh)
    assert isinstance(lap, ELL)
    if init == 'spectral':
        phi = spectral_sphere_embedding(mesh, radius=radius, laplacian=lap)
    elif init == 'radial':
        d = mesh.vertices - jnp.mean(mesh.vertices, axis=0)
        nrm = jnp.maximum(jnp.linalg.norm(d, axis=1, keepdims=True), 1e-12)
        phi = radius * d / nrm
    else:
        raise ValueError(
            f"spherical_parameterize: init must be 'spectral' or 'radial'; "
            f'got {init!r}.'
        )
    # Normalise to positive overall orientation (consistent outward winding) by
    # mirroring one axis if needed, so a bijective map has all signed areas > 0
    # (the spectral embedding's eigenfunction-sign gauge is otherwise arbitrary).
    # jit-safe: select the sign via jnp.where rather than a host branch, so the
    # 'radial' init + refinement compose under jit/grad without a device sync.
    sign = jnp.where(jnp.sum(signed_spherical_areas(phi, faces)) < 0.0, -1.0, 1.0)
    phi = phi.at[:, 0].multiply(sign)
    if n_iterations <= 0:
        return phi

    a0 = face_areas(mesh)
    target = 4.0 * jnp.pi * a0 / jnp.sum(a0)  # solid-angle targets, sum 4 pi
    r2 = radius * radius

    def _energy(p: Array) -> Array:
        e_conf = 0.5 * jnp.sum(p * apply_operator(lap, p))
        areas = signed_spherical_areas(p, faces)
        s = areas / target  # relative area (target s == 1)
        # Area-ratio barrier s + 1/s (min at s == 1, -> inf as s -> 0+), plus a
        # hinge that pushes small / negative (flipped) areas back up -- the
        # latter keeps a *gradient* on folds (a plain max-clamp would not).
        s_safe = jnp.maximum(s, 1e-3)
        e_area = jnp.sum(s_safe + 1.0 / s_safe - 2.0)
        e_flip = jnp.sum(jnp.maximum(1e-3 - s, 0.0))
        return conformal_weight * e_conf + area_weight * e_area + 1e3 * e_flip

    grad_energy = jax.grad(_energy)

    def _retract(p: Array, grad_t: Array, alpha: float) -> Array:
        q = p - alpha * grad_t
        q = (
            radius
            * q
            / jnp.maximum(jnp.linalg.norm(q, axis=1, keepdims=True), 1e-12)
        )
        q = q - jnp.mean(q, axis=0)  # centroid (Mobius) normalisation
        return (
            radius
            * q
            / jnp.maximum(jnp.linalg.norm(q, axis=1, keepdims=True), 1e-12)
        )

    def _body(_: int, p: Array) -> Array:
        e0 = _energy(p)
        n_flip0 = jnp.sum(signed_spherical_areas(p, faces) <= 0.0)
        g = grad_energy(p)
        grad_t = g - jnp.sum(g * p, axis=-1, keepdims=True) / r2 * p  # tangent
        # Normalise so ``alpha`` bounds the largest per-vertex move: the stiff
        # fold barrier otherwise makes the raw gradient huge at flipped
        # triangles and any step overshoots.
        gmax = jnp.max(jnp.linalg.norm(grad_t, axis=-1)) + 1e-12
        grad_t = grad_t / gmax

        def _cond(c: Any) -> Array:
            alpha, accepted, _ = c
            return jnp.logical_and(jnp.logical_not(accepted), alpha > 1e-5)

        def _try(c: Any) -> Any:
            alpha, _, _ = c
            cand = _retract(p, grad_t, alpha)
            n_flip = jnp.sum(signed_spherical_areas(cand, faces) <= 0.0)
            ok = jnp.logical_and(_energy(cand) < e0, n_flip <= n_flip0)
            return (
                jnp.where(ok, alpha, alpha * 0.5),
                ok,
                jnp.where(ok, cand, p),
            )

        _, _, out = jax.lax.while_loop(_cond, _try, (step, False, p))
        return out

    return cast(
        Float[Array, 'n_vertices 3'],
        jax.lax.fori_loop(0, n_iterations, _body, phi),
    )


# ---------------------------------------------------------------------------
# Surface resampling between registered spheres (ADAP_BARY_AREA / barycentric)
# ---------------------------------------------------------------------------


def _spherical_barycentric(
    verts: NDArray[Any],
    faces: NDArray[Any],
    query: NDArray[Any],
    *,
    chunk: int = 256,
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Host-side spherical point-in-triangle search + barycentric weights.

    For each ``query`` point (unit vector) find the ``verts``/``faces`` triangle
    that (radially) contains it and return its three source-vertex indices and
    the planar barycentric weights of the query's radial projection onto that
    triangle's plane.  Clean-room (no ``scipy.spatial``): the containing
    triangle maximises the minimum signed great-circle-edge distance (robust to
    fp; a query on a shared edge / vertex ties to an incident triangle, which
    interpolates consistently).  ``O(n_query * n_faces)``, chunked over queries.

    Returns ``(idx (Q, 3) int32, weights (Q, 3) float32)`` -- weights are
    clipped non-negative and renormalised to a partition of unity (so a query
    fractionally outside every triangle projects to the nearest edge / vertex
    and constants are still preserved).
    """
    a = verts[faces[:, 0]]
    b = verts[faces[:, 1]]
    c = verts[faces[:, 2]]
    n_ab = np.cross(a, b)
    n_bc = np.cross(b, c)
    n_ca = np.cross(c, a)
    q_n = query.shape[0]
    best_face = np.empty(q_n, dtype=np.int64)
    for start in range(0, q_n, chunk):
        q = query[start : start + chunk]
        score = np.minimum(
            np.minimum(q @ n_ab.T, q @ n_bc.T), q @ n_ca.T
        )  # (q, F)
        best_face[start : start + chunk] = np.argmax(score, axis=1)
    fa = faces[best_face]  # (Q, 3) source vertex indices
    av = verts[fa[:, 0]]
    bv = verts[fa[:, 1]]
    cv = verts[fa[:, 2]]
    # Radial projection of the query onto the triangle plane.
    m = np.cross(bv - av, cv - av)
    denom = np.sum(query * m, axis=1)
    denom = np.where(np.abs(denom) < 1e-20, 1e-20, denom)
    p = (np.sum(av * m, axis=1) / denom)[:, None] * query
    # Planar barycentric of p in (av, bv, cv).
    v0 = bv - av
    v1 = cv - av
    v2 = p - av
    d00 = np.sum(v0 * v0, axis=1)
    d01 = np.sum(v0 * v1, axis=1)
    d11 = np.sum(v1 * v1, axis=1)
    d20 = np.sum(v2 * v0, axis=1)
    d21 = np.sum(v2 * v1, axis=1)
    det = d00 * d11 - d01 * d01
    det = np.where(np.abs(det) < 1e-20, 1e-20, det)
    wb = (d11 * d20 - d01 * d21) / det
    wc = (d00 * d21 - d01 * d20) / det
    wa = 1.0 - wb - wc
    w = np.stack([wa, wb, wc], axis=1)
    w = np.clip(w, 0.0, None)
    w = w / np.maximum(w.sum(axis=1, keepdims=True), 1e-20)
    return fa.astype(np.int32), w.astype(np.float32)


def _ell_from_triples(
    rows: NDArray[Any],
    cols: NDArray[Any],
    vals: NDArray[Any],
    n_rows: int,
    n_cols: int,
) -> 'ELL':
    """Pack coordinate ``(row, col, value)`` triples into a padded ``ELL``.

    Vectorised (Tier C / audit AI-C2): sort the triples by row and scatter each
    into its row's next free slot -- no Python per-row loop.  Caller must have
    aggregated any duplicate ``(row, col)`` pairs; padding slots keep index 0 /
    value 0 (zero weight -> no contribution).
    """
    from ..sparse import ELL

    rows = rows.astype(np.int64)
    deg = np.bincount(rows, minlength=n_rows)
    k_max = max(int(deg.max(initial=0)), 1)
    order = np.argsort(rows, kind='stable')
    rs = rows[order]
    start = np.zeros(n_rows + 1, dtype=np.int64)
    np.cumsum(deg, out=start[1:])
    slot = np.arange(rs.shape[0], dtype=np.int64) - start[rs]
    idx = np.zeros((n_rows, k_max), dtype=np.int32)
    val = np.zeros((n_rows, k_max), dtype=np.float32)
    idx[rs, slot] = cols[order].astype(np.int32)
    val[rs, slot] = vals[order].astype(np.float32)
    return ELL(
        values=jnp.asarray(val),
        indices=jnp.asarray(idx),
        n_cols=n_cols,
        identity=0.0,
    )


def surface_resample(
    source_sphere: 'Mesh',
    source_vals: Float[Array, 'n_source ...'],
    target_sphere: 'Mesh',
    *,
    method: Literal['adap_bary_area', 'barycentric'] = 'adap_bary_area',
    source_area: Optional[Float[Array, 'n_source']] = None,
    target_area: Optional[Float[Array, 'n_target']] = None,
    semiring: Any = None,
) -> tuple['ELL', Array]:
    """Resample a per-vertex field between two registered spherical meshes.

    The ``wb_command -metric-resample`` analogue and the fsaverage<->fs_LR (or
    cross-resolution) bridge: ``source_sphere`` and ``target_sphere`` are two
    tessellations of the **same** registered sphere (radius-independent -- both
    are normalised to unit vectors for the search), and ``source_vals`` is
    carried to the target tessellation.  Returns ``(operator, resampled)`` --
    the resampling ``ELL`` (host-side construction) **and** the field after
    applying it through the differentiable apply-seam; reuse the operator for
    further fields of the same (source, target) pair.

    Two methods, matching the two Workbench modes:

    - ``method='barycentric'`` -- forward gather: each target vertex takes the
      barycentric interpolation of the source triangle it falls in.  The
      operator is **row-stochastic**, so constants are preserved exactly; best
      for smooth features and up-sampling.
    - ``method='adap_bary_area'`` (default) -- *adaptive barycentric, area
      weighted*: each source vertex scatters its (anatomical) vertex area into
      the target triangle it falls in, and each target row is normalised by the
      target vertex area.  This **conserves the area-weighted integral exactly**
      (``sum_t A_target[t] * out[t] == sum_s A_source[s] * in[s]``) whenever
      every target vertex receives source mass (the down-sampling / matched
      regime); target vertices that receive none (up-sampling holes) fall back
      to the barycentric gather, where conservation becomes approximate.  This
      is the faithful Workbench behaviour and the documented divergence: pick
      ``'barycentric'`` when exact constant-preservation matters more than the
      integral.

    The ``*_area`` arrays are the **anatomical** vertex areas to weight by
    (the Workbench ``-area-metrics``); default to the spheres' own vertex areas
    (``vertex_areas``), which conserve the *sphere*-area-weighted integral.  The
    conserved inner product is always w.r.t. the **supplied** ``source_area`` /
    ``target_area`` (i.e. ``sum_t target_area[t]*out[t] == sum_s
    source_area[s]*in[s]``): if you override them, measure conservation against
    the same arrays, not the sphere ``vertex_areas``.

    Host-side construction (the spherical search emits a plain ``ELL``, the
    HOST-CTOR class); the *application* is pure-JAX and differentiable w.r.t.
    ``source_vals``.  The search is ``O(n_query * n_faces)`` (a uniform-bucket
    broad-phase is a profile-gated follow-up).

    Parameters
    ----------
    source_sphere, target_sphere
        Registered spherical meshes (any radius; normalised internally).
    source_vals
        Per-source-vertex field, ``(n_source,)`` or ``(n_source, c)``.
    method
        ``'adap_bary_area'`` (default) or ``'barycentric'``.
    source_area, target_area
        Optional anatomical vertex areas (``adap_bary_area`` only); default to
        the spheres' ``vertex_areas``.
    semiring
        Optional semiring for the apply (default ``REAL``).

    Returns
    -------
    ``(operator, resampled)`` -- the resampling ``ELL`` ``(n_target, n_source)``
    and the resampled field (``(n_target,)`` or ``(n_target, c)``).

    Raises
    ------
    ValueError
        If ``method`` is unknown.
    """
    from ..sparse import ELL, apply_operator, vertex_areas

    src_v = np.asarray(source_sphere.vertices, dtype=np.float64)
    src_v = src_v / np.maximum(
        np.linalg.norm(src_v, axis=1, keepdims=True), 1e-20
    )
    tgt_v = np.asarray(target_sphere.vertices, dtype=np.float64)
    tgt_v = tgt_v / np.maximum(
        np.linalg.norm(tgt_v, axis=1, keepdims=True), 1e-20
    )
    src_f = np.asarray(source_sphere.faces)
    tgt_f = np.asarray(target_sphere.faces)
    n_src = src_v.shape[0]
    n_tgt = tgt_v.shape[0]

    if method == 'barycentric':
        idx, w = _spherical_barycentric(src_v, src_f, tgt_v)
        op: ELL = ELL(
            values=jnp.asarray(w),
            indices=jnp.asarray(idx),
            n_cols=n_src,
            identity=0.0,
        )
    elif method == 'adap_bary_area':
        sa = (
            np.asarray(vertex_areas(source_sphere), dtype=np.float64)
            if source_area is None
            else np.asarray(source_area, dtype=np.float64)
        )
        ta = (
            np.asarray(vertex_areas(target_sphere), dtype=np.float64)
            if target_area is None
            else np.asarray(target_area, dtype=np.float64)
        )
        # Forward gather (for up-sampling holes) + reverse scatter (the
        # area-conserving core).
        fwd_idx, fwd_w = _spherical_barycentric(src_v, src_f, tgt_v)
        rev_idx, rev_w = _spherical_barycentric(tgt_v, tgt_f, src_v)
        # Vectorised reverse scatter: each source s drops its (anatomical) area
        # into the 3 corners of the target triangle it falls in.
        rows = rev_idx.reshape(-1).astype(np.int64)  # target vertex
        srcs = np.repeat(np.arange(n_src), 3).astype(np.int64)
        wts = (rev_w * sa[:, None]).reshape(-1).astype(np.float64)
        nz = wts != 0.0
        rows, srcs, wts = rows[nz], srcs[nz], wts[nz]
        if rows.size:  # aggregate duplicate (target, source) pairs
            key = rows * np.int64(n_src) + srcs
            ukey, inv = np.unique(key, return_inverse=True)
            wagg = np.zeros(ukey.shape[0], dtype=np.float64)
            np.add.at(wagg, inv, wts)
            t_agg = (ukey // np.int64(n_src)).astype(np.int64)
            s_agg = (ukey % np.int64(n_src)).astype(np.int64)
        else:
            t_agg = np.zeros(0, dtype=np.int64)
            s_agg = np.zeros(0, dtype=np.int64)
            wagg = np.zeros(0, dtype=np.float64)
        val_agg = wagg / ta[t_agg]  # row-normalise by target vertex area
        # Up-sampling holes (targets that received no source) fall back to the
        # forward barycentric gather.
        has = np.zeros(n_tgt, dtype=bool)
        has[t_agg] = True
        empt = np.where(~has)[0]
        fb_rows = np.repeat(empt, 3).astype(np.int64)
        fb_cols = fwd_idx[empt].reshape(-1).astype(np.int64)
        fb_vals = fwd_w[empt].reshape(-1).astype(np.float64)
        op = _ell_from_triples(
            np.concatenate([t_agg, fb_rows]),
            np.concatenate([s_agg, fb_cols]),
            np.concatenate([val_agg, fb_vals]),
            n_tgt,
            n_src,
        )
    else:
        raise ValueError(
            f"surface_resample: method must be 'adap_bary_area' or "
            f"'barycentric'; got {method!r}."
        )

    vals_arr = jnp.asarray(source_vals)
    squeeze = vals_arr.ndim == 1
    x = vals_arr[:, None] if squeeze else vals_arr
    resampled = apply_operator(op, x, semiring=semiring)
    if squeeze:
        resampled = resampled[:, 0]
    return op, resampled
